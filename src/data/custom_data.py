import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset
from src.utils.helper import build_transforms


class EyeData(Dataset):
    """
    PyTorch Dataset for retinal fundus images with associated labels.
    Supports optional training augmentations.

    Attributes:
        data (List): List of data objects containing image and labels.
        transforms (albumentations.core.composition.Compose):
            Albumentations augmentation pipeline.
    """

    def __init__(self, data: List, is_train: bool) -> None:
        """
        Initialize the dataset.

        Args:
            data (List): List of image data objects,
                each containing (img_data, retinopathy_grade, macular_edema_risk).
            is_training (bool, optional): Whether to apply training augmentations.
                Defaults to True.
        """
        self.data = data
        self.transforms = build_transforms(is_train)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Image tensor of shape (C, H, W).
                - Retinopathy grade tensor (long).
                - Macular edema risk tensor (long).
        """
        item = self.data[index]

        image: np.ndarray = item.img_data
        retinopathy_grade: int = item.retinopathy_grade
        macular_edema_risk: int = item.macular_edema_risk

        # Apply augmentations
        augmented = self.transforms(image=image)
        image_augmented = augmented["image"]

        # Convert to torch tensor (C, H, W)
        image_tensor = torch.from_numpy(image_augmented).permute(2, 0, 1)

        # Convert labels to torch tensors
        retinopathy_tensor = torch.tensor(retinopathy_grade, dtype=torch.long)
        edema_risk_tensor = torch.tensor(macular_edema_risk, dtype=torch.long)

        return image_tensor, retinopathy_tensor, edema_risk_tensor
