import cv2
import polars as pl
from tqdm import tqdm
from pathlib import Path
from typing import List, Iterator
from src.entities.models import ImageData
from src.config.constants import REQUIRED_COLUMNS


class Loader:
    """
    A class to load and process image data from a specified directory and CSV file.

    Attributes:
        image_dir (Path): The directory containing the images.
        labels_df (pl.DataFrame): A Polars DataFrame containing the image labels.
    """

    def __init__(self, image_dir: Path, labels_path: Path) -> None:
        """
        Initializes the DataLoader with image directory and labels file path.

        Args:
            image_dir (Path): The path to the directory containing images.
            labels_path (Path): The path to the CSV file with image labels.
        """
        self.image_dir = image_dir
        self.labels_df = self._load_labels_from_csv(labels_path)

    def _load_labels_from_csv(self, path: Path) -> pl.DataFrame:
        """
        Loads and validates the CSV file, selecting only the required columns.

        Args:
            path (Path): The path to the CSV file.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the selected label data.
        """
        labels = pl.read_csv(path)
        return labels.select(REQUIRED_COLUMNS)

    def load_data(self) -> List[ImageData]:
        """
        Loads all images and their corresponding labels into a list of ImageData objects.

        Returns:
            List[ImageData]: A list of ImageData objects.
        """
        return list(self._load_images_with_labels())

    def _load_images_with_labels(self) -> Iterator[ImageData]:
        """
        A generator that yields ImageData objects for each row in the labels DataFrame.

        Yields:
            Iterator[ImageData]: An iterator of ImageData objects.
        """
        for row in tqdm(
            self.labels_df.iter_rows(named=True),
            total=self.labels_df.height,
            desc="Loading Images",
        ):
            image_filename = f"{row['Image name']}.jpg"
            image_path = self.image_dir / image_filename

            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            yield ImageData(
                img_path=image_path,
                img_data=image_rgb,
                retinopathy_grade=row["Retinopathy grade"],
                macular_edema_risk=row["Risk of macular edema "],
            )
