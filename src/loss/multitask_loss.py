import torch
import torch.nn as nn
from torch import Tensor


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with uncertainty weighting.

    This loss function balances two tasks (DR and DME classification)
    by learning task-specific uncertainty parameters as described in
    Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to
    Weigh Losses for Scene Geometry and Semantics".

    Args:
        None
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_var_dr = nn.Parameter(torch.zeros(1))
        self.log_var_dme = nn.Parameter(torch.zeros(1))
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        preds_dr: Tensor,
        targets_dr: Tensor,
        preds_dme: Tensor,
        targets_dme: Tensor,
    ) -> Tensor:
        """
        Compute the multi-task loss for DR and DME classification.

        Args:
            preds_dr (Tensor): Predictions for DR classification (logits).
            targets_dr (Tensor): Ground truth labels for DR classification.
            preds_dme (Tensor): Predictions for DME classification (logits).
            targets_dme (Tensor): Ground truth labels for DME classification.

        Returns:
            Tensor: Weighted multi-task loss combining DR and DME losses.
        """
        loss_dr = self.cross_entropy(preds_dr, targets_dr)
        loss_dme = self.cross_entropy(preds_dme, targets_dme)

        weighted_loss = (
            torch.exp(-self.log_var_dr) * loss_dr
            + self.log_var_dr
            + torch.exp(-self.log_var_dme) * loss_dme
            + self.log_var_dme
        )

        return weighted_loss
