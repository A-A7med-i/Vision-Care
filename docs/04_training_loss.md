# Training Optimization

This section documents the training strategies and optimization techniques used in the **VisionCare** project to improve the robustness, efficiency, and generalization of the model.

---

## 1. Multi-Task Learning with Uncertainty Weighting

### Overview

Implemented a **multi-task loss function** following Kendall et al. (2018) that optimizes two tasks simultaneously:

- **Diabetic Retinopathy (DR) classification** (5 classes)
- **Diabetic Macular Edema (DME) classification** (3 classes)

### Mathematical Formulation

Instead of manually tuning task weights, the loss function **learns task-specific uncertainty parameters**.

**Multi-Task Loss Equation:**

```
L_total = (1/2σ²_DR) × L_DR + (1/2σ²_DME) × L_DME + log(σ_DR) + log(σ_DME)
```

Where:

- `L_DR` = Cross-entropy loss for DR classification
- `L_DME` = Cross-entropy loss for DME classification
- `σ²_DR` = Learned uncertainty parameter for DR task
- `σ²_DME` = Learned uncertainty parameter for DME task
- `log(σ_DR)`, `log(σ_DME)` = Regularization terms to prevent σ → 0

### Benefits

- **Adaptive balancing** between tasks
- Gives harder tasks (e.g., DR) more weight when needed
- Eliminates manual hyperparameter tuning for task weights

---

## 2. Gradual Unfreezing Strategy

### Implementation

```python
# Pseudo-code for gradual unfreezing
if epoch % unfreeze_step == 0 and epoch > 0:
    unfreeze_next_layer(model.backbone)
```

### Strategy Details

1. **Initial Phase**: Backbone (EfficientNetV2-S) is frozen
2. **Progressive Unfreezing**: Layers are unfrozen every few epochs (`unfreeze_step`)
3. **Final Phase**: All layers are trainable

### Benefits

- **Prevents catastrophic forgetting** of pre-trained features
- **Stabilizes early training** by allowing classifier head to adapt first
- **Inspired by transfer learning best practices** (Howard & Ruder, 2018)

### Unfreezing Schedule

| Epoch Range | Frozen Layers | Trainable Layers |
|-------------|---------------|------------------|
| 0-5 | Entire Backbone | Classifier Heads Only |
| 6-10 | Bottom 75% | Top 25% + Heads |
| 11-15 | Bottom 50% | Top 50% + Heads |
| 16+ | None | All Layers |

---

## 3. Learning Rate Scheduling

### Cosine Annealing Scheduler

Uses **CosineAnnealingLR** for smooth learning rate decay.

**Mathematical Formula:**

```
LR(t) = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2
```

Where:

- `η_max` = Initial learning rate
- `η_min` = Minimum learning rate (eta_min)
- `t` = Current epoch
- `T` = Total number of epochs

### Advantages

- **Smooth decay** without sharp drops
- **Better convergence** compared to step decay
- **Improved generalization** through gradual learning rate reduction

### Learning Rate Visualization

```
η_max ┌─╲
      │  ╲
      │   ╲
      │    ╲
      │     ╲
η_min └──────╲────────── epochs
      0      T/2      T
```

---

## 4. Mixed Precision Training

### Implementation

Utilizes **`torch.cuda.amp`** for automatic mixed precision:

```python
# Pseudo-code for mixed precision
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Technical Details

- **Forward/backward passes**: Partly in **FP16** for efficiency
- **Gradient scaling**: Maintains numerical stability
- **Automatic conversion**: Between FP16 and FP32 as needed

### Benefits

- **50% reduction in GPU memory usage**
- **1.5-2x faster training** on modern GPUs
- **No accuracy loss** with proper implementation

---

## 5. Checkpointing & Model Selection

### Comprehensive State Saving

At each epoch, the following state is saved:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'epoch': epoch,
    'best_accuracy': best_accuracy
}
```

### Model Selection Criteria

- **Primary Metric**: Average test accuracy across DR & DME tasks
- **Secondary Metrics**: Individual task F1-scores
- **Selection Strategy**: Keep model with highest average performance

### Benefits

- **Full reproducibility** of training state
- **Overfitting prevention** through best model selection
- **Training resumption** capability

---

## 6. Comprehensive Metrics Tracking

### Tracked Metrics

For both **DR** and **DME** tasks:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Overall performance |
| **Macro F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Balanced evaluation |
| **Per-Class Accuracy** | Individual class performance | Class-specific insights |

### Visualization Features

- **Real-time plotting** of training/validation curves
- **Loss convergence** visualization
- **Accuracy trends** for both tasks
- **F1-score evolution** over epochs

---

## 7. Advanced Training Loop Features

### Technical Features

- **Device-aware execution** (automatic CPU/GPU detection)
- **Memory-efficient batching**
- **Gradient clipping** for training stability
- **Early stopping** based on validation metrics

### Per-Epoch Logging

Detailed logging includes:

- Train/Test Loss (total, DR, DME)
- Accuracy (DR & DME)
- F1-score (DR & DME)
- Current learning rate
- GPU memory usage
- Training time per epoch

---

## Summary of Optimization Techniques

| ✅ Technique | Impact | Benefit |
|-------------|---------|---------|
| **Multi-task Uncertainty Weighting** | High | Automatic task balancing |
| **Gradual Unfreezing** | High | Prevents catastrophic forgetting |
| **Cosine Annealing** | Medium | Smooth convergence |
| **Mixed Precision** | High | 2x faster training, 50% less memory |
| **Smart Checkpointing** | Medium | Reproducibility & overfitting prevention |
| **Comprehensive Metrics** | Medium | Better model understanding |
