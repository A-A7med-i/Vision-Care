import cv2
import numpy as np
from tqdm import tqdm
from typing import Iterator, List
from src.config.constants import IMAGE_RESIZE
from src.entities.models import ProcessedImageData, ImageData


class Preprocessing:
    """
    Utility class for preprocessing retinal fundus images.
    Includes cropping, resizing, CLAHE enhancement, and Ben Graham normalization.
    """

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def circle_crop(image: np.ndarray, sigmaX: int = 10) -> np.ndarray:
        """
        Apply a circular crop around the center of the image.

        Args:
            image (np.ndarray): Input image array in BGR format.
            sigmaX (int, optional): Standard deviation for Gaussian blur (unused here).
                                    Defaults to 10.

        Returns:
            np.ndarray: Circularly cropped image.
        """

        height, width, _ = image.shape
        center_x, center_y = width // 2, height // 2
        radius = min(center_x, center_y)

        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 1, thickness=-1)

        cropped_img = cv2.bitwise_and(image, image, mask=mask)
        return cropped_img

    @staticmethod
    def process_image(image: np.ndarray) -> np.ndarray:
        """
        Perform preprocessing on a single retinal image.
        Steps:
            - Circular crop
            - Resize to (512, 512)
            - Extract green channel
            - Apply CLAHE enhancement
            - Apply Ben Graham normalization
            - Stack green, CLAHE, and Ben Graham channels

        Args:
            image (np.ndarray): Input image array in BGR format.

        Returns:
            np.ndarray: Preprocessed image with shape (512, 512, 3).
        """
        image = Preprocessing.circle_crop(image)
        resized_img = (
            cv2.resize(image, IMAGE_RESIZE, interpolation=cv2.INTER_AREA) / 255.0
        )

        green_channel = resized_img[:, :, 1]
        clahe_img = (
            Preprocessing.clahe.apply((green_channel * 255).astype(np.uint8)) / 255.0
        )

        blurred_img = cv2.GaussianBlur(green_channel, (0, 0), 30)
        ben_graham = cv2.addWeighted(green_channel, 4, blurred_img, -4, 128 / 255.0)
        ben_graham = cv2.normalize(ben_graham, None, 0, 255, cv2.NORM_MINMAX)

        processed_img = np.stack(
            [green_channel, clahe_img, ben_graham], axis=-1
        ).astype(np.float32)
        return processed_img

    @staticmethod
    def process_dataset(data: Iterator["ImageData"]) -> List[ProcessedImageData]:
        """
        Apply preprocessing to an entire dataset of images.

        Args:
            data (Iterator[ImageData]): Iterator of ImageData objects.

        Returns:
            List[ProcessedImageData]: List of preprocessed image data with labels.
        """
        processed_list: List[ProcessedImageData] = []

        for item in tqdm(data, desc="Preprocessing Images"):
            processed_img = Preprocessing.process_image(item.img_data)
            processed_list.append(
                ProcessedImageData(
                    img_path=item.img_path,
                    img_data=processed_img,
                    retinopathy_grade=item.retinopathy_grade,
                    macular_edema_risk=item.macular_edema_risk,
                )
            )

        return processed_list
