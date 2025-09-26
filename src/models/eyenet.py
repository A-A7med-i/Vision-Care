import torch
import torch.nn as nn
from typing import Tuple
from src.config.constants import *
import torch.nn.utils.prune as prune
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class EyeNet(nn.Module):
    """
    Multi-task neural network for diabetic retinopathy grading and
    diabetic macular edema risk classification.

    Architecture:
        - EfficientNetV2-S backbone (pretrained).
        - Shared fully connected layer.
        - Two classification heads:
            (1) Retinopathy grading (multi-class).
            (2) Macular edema risk classification (multi-class).

    Args:
        hidden_units (int): Number of hidden units in the shared fully connected layer.
        dropout_p (float): Dropout probability.
        num_classes_dr (int, optional): Number of classes for retinopathy grading. Defaults to 5.
        num_classes_dme (int, optional): Number of classes for DME risk. Defaults to 3.
        prune_amount (float, optional): Amount of structured pruning to apply to the first conv layer. Defaults to 0.2.
    """

    def __init__(
        self,
        hidden_units: int,
        dropout_p: float,
        num_classes_dr: int = DR_CLASSES,
        num_classes_dme: int = DME_CLASSES,
        prune_amount: float = PRUNE_AMOUNT,
    ) -> None:
        super().__init__()

        # Load pretrained EfficientNetV2-S backbone
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()  # remove final classifier
        self.backbone = backbone

        # Apply pruning to the first convolutional layer
        first_conv = self.backbone.features[0][0]
        prune.ln_structured(first_conv, name="weight", amount=prune_amount, n=2, dim=0)

        # Shared feature extractor (MLP head)
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.SiLU(),
            nn.Dropout(dropout_p),
        )

        # Retinopathy grading classifier
        self.dr_classifier_head = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_units // 2, num_classes_dr),
        )

        # Macular edema risk classifier
        self.dme_classifier_head = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_units // 2, num_classes_dme),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Retinopathy grading predictions (B, num_classes_dr).
                - Macular edema risk predictions (B, num_classes_dme).
        """
        features = self.backbone(x)
        shared_features = self.shared_fc(features)

        out_dr = self.dr_classifier_head(shared_features)
        out_dme = self.dme_classifier_head(shared_features)

        return out_dr, out_dme
