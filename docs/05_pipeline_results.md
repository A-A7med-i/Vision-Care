# Pipeline Results

![Training Curves](https://drive.google.com/uc?export=view&id=1jluE3rHFWVNnpw5Omv882lkzyTL4S2ti)

---

## Training Summary

The model was trained for **50 epochs** using the **IDRiD dataset** with multi-task objectives:

- **Diabetic Retinopathy (DR) classification**
- **Diabetic Macular Edema (DME) classification**

The training pipeline used **Cosine Annealing LR Scheduler**, **multi-task loss balancing**, and **strong data augmentation** to improve generalization.

---

## Key Observations

- **Early Improvement**
  Rapid accuracy gains were observed in the first 10 epochs, especially for DME.

- **DME > DR Performance**
  The DME classifier consistently outperformed the DR classifier (≈ 75–80% vs. 50–55% test accuracy).

- **F1-Score Trends**
  - DME F1 steadily improved, reaching ~66–70% in later epochs.
  - DR F1 remained more challenging, plateauing around ~40%.

- **Generalization Gap**
  Training accuracy kept improving, while test accuracy for DR stabilized early → suggests DR classification remains harder due to **data imbalance and task complexity**.

---

## Final Metrics (Epoch 50)

| Metric         | **DR**   | **DME** |
|----------------|----------|---------|
| **Train Acc**  | 67.57%   | 80.60%  |
| **Test Acc**   | 49.87%   | 74.49%  |
| **Train F1**   | 58.28%   | 65.97%  |
| **Test F1**    | 37.40%   | 61.87%  |

---

## Key Takeaways

- **Multi-task setup works well** → DME benefits from DR supervision, but DR remains more challenging.
- **Data Augmentation helps** → Training curves suggest improved generalization vs. baseline.
- **Future Improvements**:
  - Class balancing or **focal loss** to handle DR difficulty.
  - Semi-supervised or **self-supervised learning** for better feature extraction.
  - Larger datasets could reduce overfitting on DR.
