<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1I-eeaXJo_zEF53ajzdyCNJlng3PWqb-G"
       alt="VisionCare Logo"
       height="140"
       width="550"
       style="border-radius: 12px;"/>
</div>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-380/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0-red.svg" alt="PyTorch"/>
  </a>
  <a href="#contributing">
    <img src="https://img.shields.io/badge/contributions-welcome-orange.svg" alt="Contributions welcome"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  </a>
</p>

---

## Overview

**VisionCare** is a multi-task deep learning project for automated grading of **Diabetic Retinopathy (DR)** and **Diabetic Macular Edema (DME)** from retinal fundus images.

> The goal: Assist ophthalmologists and healthcare professionals in **early detection** and **accurate classification** of disease severity, improving patient outcomes and reducing vision loss.

Detailed documentation and methods are available in the [`docs`](docs) folder.

---

## Objectives & Motivation

Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME) are **leading causes of vision loss worldwide**, especially in diabetic patients. Manual grading is:

- **Time-consuming**
- **Subjective**
- Requires **expert ophthalmologists**

VisionCare aims to:

- Provide an **automated, reliable, and scalable** solution for DR and DME grading
- Deliver **fast, consistent, objective assessments**
- Advance AI in **medical imaging research**

---

## Features

- **Multi-task classification**:
  - DR: 5 severity levels
  - DME: 3 severity levels
- **Custom preprocessing & augmentation pipeline**
- **Modular PyTorch codebase**
- **Cosine Annealing LR Scheduler**
- **Gradual Layer Unfreezing strategy**
- **Model checkpointing and reproducibility**

---

## Dataset

Using the **[IDRiD Dataset](https://idrid.grand-challenge.org/)**:

| Task | Labels |
|------|--------|
| DR   | 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative |
| DME  | 0: No DME, 1: Non-clinically significant, 2: Clinically significant |

The dataset is organized as follows:

- `data/B. Disease Grading/1. Original Images/a. Training Set/` — Training images
- `data/B. Disease Grading/1. Original Images/b. Testing Set/` — Testing images
- `data/B. Disease Grading/2. Groundtruths/` — Ground truth CSV files (with both DR and DME severity labels)

## Installation

1. **Clone the repository:**

 ```bash
 git clone https://github.com/A-A7med-i/Vision-Care.git
 cd VisionCare
 ```

2. **Create a virtual environment (recommended):**

```bash
python3 -m venv vision-care
source vision-care/bin/activate
```

3. **Install dependencies:**

 ```bash
 pip install -r requirements.txt
 ```

## Usage

### Training

You can run the training process using either the main pipeline script or the dedicated training pipeline:

- **Main pipeline (recommended for full workflow):**

    ```bash
    python -m src.pipeline.main
    ```

- **Direct training pipeline:**

    ```bash
    python -m src.pipeline.training_pipeline
    ```

## Project Structure

The project is organized as follows:

```
VisionCare/
│
├── src/                         # Main source code
│   ├── config/                  # Configuration files and constants
│   │   └── constants.py         # Project-wide constant values
│   ├── data/                    # Data loading and custom dataset classes
│   │   ├── custom_data.py       # Custom PyTorch Dataset definitions
│   │   └── data_loader.py       # DataLoader utilities
│   ├── entities/                # Data models and entity definitions
│   │   └── models.py            # Data model classes
│   ├── loss/                    # Custom loss functions
│   │   └── multitask_loss.py    # Multi-task loss for DR & DME
│   ├── models/                  # Model architectures
│   │   └── eyenet.py            # Main neural network architecture
│   ├── pipeline/                # Training and main pipeline scripts
│   │   ├── main.py              # Entry point for running the pipeline
│   │   └── training_pipeline.py # Training workflow
│   ├── preprocessing/           # Image preprocessing utilities
│   │   └── fundus_preprocessing.py # Fundus image preprocessing steps
│   ├── training/                # Training utilities
│   │   └── trainer.py           # Training loop and logic
│   └── utils/                   # Helper functions
│       └── helper.py            # Utility functions
│
├── data/                        # Dataset (images and ground truth CSVs)
│   └── B. Disease Grading/      # IDRiD dataset structure
│       ├── 1. Original Images/  # Raw images (train/test)
│       └── 2. Groundtruths/     # CSV files with DR & DME labels
│
├── checkpoint/                  # Model checkpoints (saved weights)
│   └── best_model.pth           # Example: best model weights
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Project setup script
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
└── LICENSE                      # License file
```

Each folder and file is organized for clarity and modularity, making it easy to extend or modify the project.

## Acknowledgments

- [IDRiD Dataset](https://idrid.grand-challenge.org/)
- PyTorch, NumPy, OpenCV, and other open-source libraries
- Inspired by research on multi-task learning for medical imaging

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
