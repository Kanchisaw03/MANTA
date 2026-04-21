"""Physics-constrained output head for bounded transit probability predictions.

This module implements the third physics-derived constraint: physically valid
flux/probability outputs must remain in [0, 1], with numerical safeguards for
stable log-loss optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

OUTPUT_MIN: float = 1.0e-6
OUTPUT_MAX: float = 1.0 - 1.0e-6
DEFAULT_CALIBRATION_BINS: int = 10


@dataclass(slots=True)
class CalibrationCurve:
    """Reliability diagram payload."""

    bin_centers: np.ndarray
    mean_predicted: np.ndarray
    empirical_frequency: np.ndarray
    bin_counts: np.ndarray


class PhysicsConstrainedOutput(nn.Module):
    """Temperature-scaled, hard-clamped sigmoid output layer.

    Physics Justification
    ---------------------
    Constraint 3: physically meaningful output should be bounded, and training
    should avoid exact zeros/ones to keep log-likelihood finite.

    Parameters
    ----------
    in_features : int
        Input feature dimension.

    Examples
    --------
    >>> head = PhysicsConstrainedOutput(in_features=64)
    >>> x = torch.randn(8, 64)
    >>> p = head(x)
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self._cached_true: list[np.ndarray] = []
        self._cached_pred: list[np.ndarray] = []

    def forward(self, x: Tensor) -> Tensor:
        """Compute bounded probability outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor with shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Probability tensor with shape ``(batch, 1)`` and values in ``[1e-6, 1-1e-6]``.
        """
        # Constraint 3: learned temperature calibrates confidence while preserving probability semantics.
        temperature = torch.exp(self.log_temperature).clamp(min=0.05, max=20.0)
        logits = self.linear(x) / temperature
        # Constraint 3: sigmoid maps logits into probability domain.
        probs = torch.sigmoid(logits)
        # Constraint 3: hard clamp enforces strict physical/numerical bounds for stable BCE/Focal losses.
        return torch.clamp(probs, min=OUTPUT_MIN, max=OUTPUT_MAX)

    @torch.no_grad()
    def update_calibration_cache(self, y_true: Tensor, y_pred: Tensor) -> None:
        """Append predictions to internal cache for reliability analysis."""
        self._cached_true.append(y_true.detach().flatten().cpu().numpy())
        self._cached_pred.append(y_pred.detach().flatten().cpu().numpy())

    def get_calibration_curve(
        self,
        y_true: np.ndarray | None = None,
        y_pred: np.ndarray | None = None,
        n_bins: int = DEFAULT_CALIBRATION_BINS,
    ) -> dict[str, np.ndarray]:
        """Compute reliability-diagram data.

        Parameters
        ----------
        y_true : numpy.ndarray | None, optional
            Ground-truth binary labels.
        y_pred : numpy.ndarray | None, optional
            Predicted probabilities.
        n_bins : int, optional
            Number of calibration bins.

        Returns
        -------
        dict[str, numpy.ndarray]
            Dictionary containing bin centers, predicted means, empirical means,
            and sample counts.

        Raises
        ------
        ValueError
            If no prediction data is available.
        """
        if y_true is None or y_pred is None:
            if not self._cached_true or not self._cached_pred:
                raise ValueError("No calibration data available; provide y_true/y_pred or cache predictions first")
            y_true = np.concatenate(self._cached_true)
            y_pred = np.concatenate(self._cached_pred)

        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])

        pred_mean = np.zeros(n_bins, dtype=np.float64)
        true_mean = np.zeros(n_bins, dtype=np.float64)
        counts = np.zeros(n_bins, dtype=np.float64)

        indices = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)
        for i in range(n_bins):
            mask = indices == i
            counts[i] = float(mask.sum())
            if counts[i] > 0:
                pred_mean[i] = float(y_pred[mask].mean())
                true_mean[i] = float(y_true[mask].mean())
            else:
                pred_mean[i] = centers[i]
                true_mean[i] = np.nan

        return {
            "bin_centers": centers,
            "mean_predicted": pred_mean,
            "empirical_frequency": true_mean,
            "bin_counts": counts,
        }
