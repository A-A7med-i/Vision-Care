import albumentations as A
from albumentations.core.composition import Compose


def build_transforms(is_training: bool = True) -> Compose:
    """
    Build image augmentation pipeline for retinal fundus images using Albumentations.

    Args:
        is_training (bool, optional):
            If True, returns a set of augmentations for training.
            If False, returns an empty Compose (no augmentation).
            Defaults to True.

    Returns:
        Compose: Albumentations Compose object containing the augmentation pipeline.
    """
    if is_training:
        augmentations = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-45, 45), p=0.7
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=0.5, p=1.0),
                ],
                p=0.8,
            ),
        ]
        return A.Compose(augmentations)
    else:
        return A.Compose([])
