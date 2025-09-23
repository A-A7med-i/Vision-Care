import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ImageData:
    """A data class to hold image and label information."""

    img_path: Path
    img_data: np.ndarray
    retinopathy_grade: int
    macular_edema_risk: int


@dataclass
class ProcessedImageData:
    """
    Data class representing a preprocessed retinal image and its associated labels.
    """

    img_path: Path
    img_data: np.ndarray
    retinopathy_grade: int
    macular_edema_risk: int
