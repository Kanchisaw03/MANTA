"""Elliptic-integral-inspired activation functions for MANTA.

This module implements the second physics-derived architectural constraint:
transit geometry includes smooth non-monotonic elliptic-integral behavior, so
the activation should avoid piecewise-linear ReLU dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

EPS_BESSEL_INPUT: float = 1.0e-6
DEFAULT_CURVE_POINTS: int = 512
DEFAULT_CURVE_MIN: float = -8.0
DEFAULT_CURVE_MAX: float = 8.0


@dataclass(slots=True)
class ActivationCurve:
    """Container for exported activation-curve samples."""

    x: np.ndarray
    y: np.ndarray


def _scaled_bessel_k0(x: Tensor) -> Tensor:
    """Compute scaled modified Bessel K0 robustly across torch versions."""
    if hasattr(torch.special, "scaled_modified_bessel_k0"):
        return torch.special.scaled_modified_bessel_k0(x)
    # Fallback: K0e(x) alias in older versions.
    if hasattr(torch.special, "modified_bessel_k0"):
        return torch.exp(x) * torch.special.modified_bessel_k0(x)
    raise RuntimeError("PyTorch special Bessel functions are unavailable in this environment")


class EllipticMish(nn.Module):
    """Smooth non-monotonic activation for transit-geometry representation.

    Physics Justification
    ---------------------
    Mandel-Agol ingress/egress terms involve elliptic integrals with smooth,
    non-monotonic curvature near grazing incidence. This module starts from Mish
    and adds a learnable Bessel-K0 modulation to emulate that behavior.

    Parameters
    ----------
    alpha_init : float, optional
        Initial weight for the elliptic modulation term.

    Examples
    --------
    >>> act = EllipticMish(alpha_init=0.1)
    >>> x = torch.linspace(-3.0, 3.0, 32)
    >>> y = act(x)
    """

    def __init__(self, alpha_init: float = 0.1) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """Apply EllipticMish activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated tensor with smooth non-monotonic response.
        """
        # Constraint 2: Mish core provides smooth, non-piecewise response matching elliptic transit geometry.
        mish_term = x * torch.tanh(F.softplus(x))
        # Constraint 2: Bessel modulation approximates near-singular curvature around grazing-geometry regimes.
        bessel_input = x.abs().clamp(min=EPS_BESSEL_INPUT)
        # Constraint 2: Scaled K0 maintains numerical stability while preserving elliptic-like behavior.
        bessel_term = _scaled_bessel_k0(bessel_input)
        # Constraint 2: Learnable alpha lets training tune how strongly elliptic behavior contributes.
        return mish_term + self.alpha * bessel_term

    @torch.no_grad()
    def get_activation_curve(
        self,
        xmin: float = DEFAULT_CURVE_MIN,
        xmax: float = DEFAULT_CURVE_MAX,
        num_points: int = DEFAULT_CURVE_POINTS,
        device: torch.device | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the activation curve for figure generation.

        Parameters
        ----------
        xmin : float, optional
            Minimum x-value.
        xmax : float, optional
            Maximum x-value.
        num_points : int, optional
            Number of sample points.
        device : torch.device | None, optional
            Device used for curve evaluation.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Sampled x and f(x) arrays.
        """
        if num_points < 2:
            raise ValueError("num_points must be >= 2")
        eval_device = device or self.alpha.device
        x = torch.linspace(xmin, xmax, num_points, device=eval_device)
        y = self.forward(x)
        return x.detach().cpu().numpy(), y.detach().cpu().numpy()
