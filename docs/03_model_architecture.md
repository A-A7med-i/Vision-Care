## Model Architecture

![Model Architecture](https://drive.google.com/uc?export=view&id=1DJm9p6lsFsW93CqFdiUR4DrjdluTWQpc)

The **EyeNet** model is a custom multi-task neural network designed to jointly classify:

- **Diabetic Retinopathy (DR) severity** (5 classes)
- **Diabetic Macular Edema (DME) risk** (3 classes)

This approach leverages shared feature learning while maintaining task-specific heads, improving generalization and efficiency.

---

## Components of the Architecture

### 1. Backbone: EfficientNetV2-S

- Pretrained on ImageNet using `torchvision.models`
- Final classifier layer removed (`nn.Identity()`)
- Provides strong low-level and mid-level feature extraction

### 2. Structured Pruning

- 20% of filters in the first convolutional layer are pruned using **L2-norm structured pruning**
- Reduces redundancy and computational cost

**Pruning Formula:**

```
W' = {w ∈ W : ||w||₂ ≥ τ}
```

Where:

- `W'` = pruned weight set
- `W` = original weight set
- `||w||₂` = L2-norm of weight w
- `τ` = pruning threshold

Mathematically, pruning removes weights with the smallest L2-norm across filters.

### 3. Shared Fully Connected Layer

Acts as a bottleneck feature aggregator and consists of:

- `Linear(in_features → hidden_units)`
- `BatchNorm1d`
- `SiLU` activation
- `Dropout`

This shared representation encourages **knowledge transfer** between DR and DME tasks.

### 4. Task-Specific Heads

- **DR Head**: Predicts severity level across 5 classes
- **DME Head**: Predicts risk level across 3 classes
- Each head is an MLP with intermediate nonlinearity and dropout for regularization

---

## Forward Pass

The forward pass follows these steps:

1. **Input Processing**: Input image `(B, C, H, W)` is passed through EfficientNetV2-S backbone
2. **Feature Extraction**: Extracted features → Shared Fully Connected Layer
3. **Task Branching**: Features are branched into:
   - **DR Classifier Head** → Output shape `(B, 5)`
   - **DME Classifier Head** → Output shape `(B, 3)`

**Mathematical Representation:**

```
Input: X ∈ R^(B×C×H×W)
Backbone Features: F = EfficientNetV2(X) ∈ R^(B×d)
Shared Features: S = FC_shared(F) ∈ R^(B×h)
DR Output: Y_DR = Head_DR(S) ∈ R^(B×5)
DME Output: Y_DME = Head_DME(S) ∈ R^(B×3)
```

Where:

- `B` = batch size
- `C` = number of channels (3)
- `H, W` = height and width (512×512)
- `d` = backbone output dimension
- `h` = hidden units dimension

---

## Why This Design?

- **Multi-task learning** → exploits correlations between DR and DME
- **EfficientNetV2 backbone** → lightweight yet high-performing
- **Pruning** → reduces computational cost without sacrificing accuracy
- **Shared representation** → improves generalization across related retinal diseases

---

## Example Code

```python
import torch
from src.models.eyenet import EyeNet

# Create model instance
model = EyeNet(hidden_units=1024, dropout_p=0.3)

# Forward pass with dummy input
x = torch.randn(8, 3, 512, 512)  # batch of 8 fundus images
out_dr, out_dme = model(x)

print(out_dr.shape)   # torch.Size([8, 5])
print(out_dme.shape)  # torch.Size([8, 3])
```

---

## Model Summary

| Component | Input Shape | Output Shape | Parameters |
|-----------|-------------|--------------|------------|
| EfficientNetV2-S | (B, 3, 512, 512) | (B, d) | ~20M |
| Shared FC Layer | (B, d) | (B, 1024) | Variable |
| DR Head | (B, 1024) | (B, 5) | ~1M |
| DME Head | (B, 1024) | (B, 3) | ~1M |

**Total Parameters**: ~22M (after pruning: ~18M)
