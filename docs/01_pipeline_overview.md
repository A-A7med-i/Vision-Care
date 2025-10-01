# VisionCare Pipeline Documentation

![Pipeline Architecture](https://drive.google.com/uc?export=view&id=1t5RUJOb06-Fgtp7flGF6RnRpBRgM5YoS)

---

## Overview of the Pipeline

The VisionCare project is designed as a modular, extensible deep learning pipeline for multi-task classification of Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME) from retinal fundus images. The pipeline is built with PyTorch and follows best practices for reproducibility, scalability, and clarity.

### Main Pipeline Stages

1. **Data Loading & Preprocessing**
    - Loads images and ground-truth labels (DR & DME) from the IDRiD dataset.
    - Applies fundus-specific preprocessing and data augmentation using OpenCV and Albumentations.
    - Custom PyTorch Dataset and DataLoader classes handle batching and shuffling.

2. **Model Architecture**
    - Utilizes a custom EyeNet model based on EfficientNetV2 (from torchvision).
    - Supports multi-task heads for simultaneous DR and DME classification.
    - Includes options for model pruning and gradual layer unfreezing.

3. **Loss Function**
    - Implements a multi-task loss combining DR and DME objectives.
    - Supports weighted loss and advanced optimization strategies.

4. **Training Loop**
    - Handles training, validation, and checkpointing.
    - Uses Cosine Annealing LR Scheduler.
    - Tracks metrics (accuracy, F1-score) and saves the best model.

5. **Evaluation & Inference**
    - Evaluates model performance on test data.
    - Provides scripts for inference on new images.

---

## Next Sections

- [Data Preprocessing](02_preprocessing.md)
- [Model Architecture](03_model_architecture.md)
- [Training & Loss](04_training_loss.md)
- [Pipeline Results](05_pipeline_results.md)

*Continue to the next sections for deeper technical details on each pipeline component.*
