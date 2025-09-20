from pathlib import Path
from typing import List
import torch

# Paths
SCRIPT_DIR = Path(__file__).resolve().parents[2]
BASE_PATH = SCRIPT_DIR / "data" / "B. Disease Grading"
IMAGE_PATHS = BASE_PATH / "1. Original Images"
TRAIN_IMAGES = IMAGE_PATHS / "a. Training Set"
TEST_IMAGES = IMAGE_PATHS / "b. Testing Set"
CSV_FILES = BASE_PATH / "2. Groundtruths"
LABEL_TRAIN = CSV_FILES / "a. IDRiD_Disease Grading_Training Labels.csv"
LABEL_TEST = CSV_FILES / "b. IDRiD_Disease Grading_Testing Labels.csv"
CHECKPOINT_PATH = SCRIPT_DIR / "checkpoint"
HISTORY_PLOT_PATH = SCRIPT_DIR / "history.png"


# Data directory
REQUIRED_COLUMNS: List[str] = [
    "Image name",
    "Retinopathy grade",
    "Risk of macular edema ",
]

# Preprocessing directory
IMAGE_RESIZE = (512, 512)

# Model Directory
DR_CLASSES = 5
DME_CLASSES = 3
PRUNE_AMOUNT = 0.2

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
CONFIG = {
    "model_params": {"hidden_units": 1024, "dropout_p": 0.2},
    "data_params": {"augment_and_combine": True},
    "optimizer_params": {
        "lr_backbone": 1e-5,
        "wd_backbone": 1e-6,
        "lr_shared": 1e-4,
        "wd_shared": 1e-5,
        "lr_head_dr": 1e-4,
        "wd_head_dr": 1e-4,
        "lr_head_dme": 1e-4,
        "wd_head_dme": 1e-4,
    },
    "training_params": {
        "epochs": 50,
        "batch_size": 16,
        "unfreeze_backbone": False,
        "unfreeze_step": 5,
    },
}
