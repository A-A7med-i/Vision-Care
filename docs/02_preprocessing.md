# Data Preprocessing Details

Fundus images in their raw form often contain black borders, uneven illumination, and noise, making them unsuitable for direct use in training. The preprocessing pipeline standardizes and enhances the retinal images to improve model robustness and accuracy.

## Main Steps

### 1. Circular Cropping

- A circular mask is applied around the center of the retina to remove black corners and irrelevant background.
- This focuses the model only on the retinal region of interest (ROI).

**Equation: Cropping with Circular Mask**

```
I_crop(x, y) = I(x, y) × M(x, y)
```

Where:

- `I_crop(x, y)` = cropped image at position (x, y)
- `I(x, y)` = original image at position (x, y)
- `M(x, y)` = binary circular mask at position (x, y)

### 2. Resizing

- All images are resized to a fixed resolution of **512 × 512** pixels.
- This ensures consistency in input size for the convolutional backbone.

### 3. Channel Selection

- The **green channel** of the image is extracted.
- Medical literature shows that the green channel provides the highest contrast for retinal vessels and lesions.

### 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)

- Enhances local contrast while avoiding over-amplification of noise.
- Useful for highlighting microaneurysms and other fine structures.

### 5. Ben Graham Normalization

- A technique introduced in diabetic retinopathy research for illumination correction.
- A blurred version of the image is subtracted from the original to normalize brightness variations.

**Equation: Green Channel Normalization**

```
I_norm = α × I_green - β × G(I_green, σ) + C
```

Where:

- `I_norm` = normalized image
- `α` = scaling factor for original green channel
- `β` = scaling factor for blurred version
- `I_green` = green channel of the image
- `G(I_green, σ)` = Gaussian blur of the green channel with standard deviation σ
- `C` = constant offset

### 6. Channel Stacking

The final preprocessed image is a 3-channel tensor:

- **Channel 1**: Green channel
- **Channel 2**: CLAHE-enhanced image
- **Channel 3**: Ben Graham normalized image

This creates a richer representation for the CNN compared to a single-channel input.

---

## Why This Matters

- **Improved Visibility**: Enhances small lesions (microaneurysms, exudates).
- **Reduced Noise**: Removes irrelevant black regions and normalizes illumination.
- **Consistent Input**: Ensures all samples share the same scale and format.
- **Domain Knowledge Integration**: Uses ophthalmology-inspired transformations instead of generic preprocessing.

---

## Example Code

```python
from src.preprocessing.fundus_preprocessing import Preprocessing

# Preprocess a single image
processed = Preprocessing.process_image(image)

# Preprocess a dataset
processed_dataset = Preprocessing.process_dataset(dataset)
```

---

## Data Augmentation

While preprocessing ensures images are standardized and enhanced, **data augmentation** artificially increases the dataset size and diversity.
This helps the model generalize better, reduces overfitting, and makes it robust against variations commonly seen in retinal imaging (e.g., rotation, brightness, distortion).

---

### Main Augmentations

#### 1. Geometric Transformations

- **Horizontal & Vertical Flips**: Simulate mirrored fundus images.
- **Random Rotations / Transpose**: Introduce invariance to orientation.
- **Affine Transformations**: Random scaling, translation, and rotation.

#### 2. Photometric Transformations

- **Brightness & Contrast Adjustment**: Simulates changes in imaging conditions.
- **Hue, Saturation, Value Shift**: Models variations in camera or illumination settings.

#### 3. Elastic & Optical Distortions

- **Elastic Transform**: Warps the image to mimic biological variability.
- **Grid Distortion / Optical Distortion**: Introduces controlled deformations to improve robustness.

---

### Why This Matters

- **Improves Generalization**: Prevents the model from memorizing training data.
- **Robust to Real-World Variations**: Handles differences in fundus cameras and acquisition settings.
- **Increases Effective Dataset Size**: Especially useful when annotated medical data is limited.

---

### Example Code

```python
from src.preprocessing.augmentations import build_transforms

# Training augmentations
train_transforms = build_transforms(is_training=True)

# Validation (no augmentation)
val_transforms = build_transforms(is_training=False)
```
