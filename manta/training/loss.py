"""Loss functions for imbalanced exoplanet transit classification.

This module provides focal BCE variants for MANTA, emphasizing hard examples
such as shallow/grazing transits that are easily suppressed by class imbalance.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

EPS_LOG: float = 1.0e-6


class FocalBCELoss(nn.Module):
    """Binary focal loss with optional positive-class weighting.

    Physics Justification
    ---------------------
    Rare low-SNR transit cases are scientifically valuable and difficult.
    Focal loss upweights those hard examples so the model does not collapse to
    majority easy negatives.

    Parameters
    ----------
    gamma : float, optional
        Focusing parameter.
    alpha : float, optional
        Positive-class balancing factor.
    pos_weight : float | None, optional
        Additional multiplicative weight for positive examples.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, pos_weight: float | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute focal BCE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted probabilities in [0, 1], shape ``(batch, 1)`` or ``(batch,)``.
        target : torch.Tensor
            Binary labels with matching shape.

        Returns
        -------
        torch.Tensor
            Scalar focal BCE loss.
        """
        pred = pred.float().view(-1)
        target = target.float().view(-1)

        pred = torch.clamp(pred, min=EPS_LOG, max=1.0 - EPS_LOG)
        bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))

        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_factor = (1.0 - p_t) ** self.gamma

        loss = alpha_t * focal_factor * bce
        if self.pos_weight is not None:
            class_weight = torch.where(target > 0.5, torch.tensor(self.pos_weight, device=target.device), torch.tensor(1.0, device=target.device))
            loss = loss * class_weight

        return loss.mean()
