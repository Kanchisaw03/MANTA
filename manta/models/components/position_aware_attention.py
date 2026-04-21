"""Position-aware attention modulated by limb-darkening geometry.

This module implements MANTA's first physics-derived constraint: transit signal
strength depends on projected stellar-disk position because limb darkening is
radially varying, so attention should not be translation invariant.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

EPS_ATTENTION: float = 1.0e-8
DEFAULT_IMPACT_PARAMETER: float = 0.5


class PositionAwareAttention(nn.Module):
    """Multi-head attention with limb-darkening bias.

    Physics Justification
    ---------------------
    Constraint 1: the quadratic limb-darkening law
    ``I(r)=1-u1(1-mu)-u2(1-mu)^2`` implies center and limb samples carry
    different transit information. This layer maps sequence phase to projected
    position along a transit chord and uses impact parameter (when available)
    to compute a physically motivated attention bias.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    stellar_radius_bins : int
        Number of radial bins used to discretize disk geometry.
    dropout : float, optional
        Attention dropout probability.

    Raises
    ------
    ValueError
        If ``d_model`` is not divisible by ``n_heads``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        stellar_radius_bins: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.stellar_radius_bins = int(stellar_radius_bins)

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model)

        self._last_attention: Tensor | None = None

    def _infer_limb_darkening_coefficients(self, stellar_params: Tensor) -> tuple[Tensor, Tensor]:
        """Infer or extract quadratic limb-darkening coefficients (u1, u2)."""
        if stellar_params.ndim != 2:
            raise ValueError("stellar_params must have shape (batch, features)")

        if stellar_params.shape[1] >= 7:
            u1 = stellar_params[:, 5]
            u2 = stellar_params[:, 6]
            return torch.clamp(u1, 0.0, 1.0), torch.clamp(u2, 0.0, 1.0)

        if stellar_params.shape[1] < 3:
            # Constraint 1: fall back to solar-like coefficients when stellar metadata is sparse.
            batch = stellar_params.shape[0]
            return (
                torch.full((batch,), 0.4, device=stellar_params.device, dtype=stellar_params.dtype),
                torch.full((batch,), 0.25, device=stellar_params.device, dtype=stellar_params.dtype),
            )

        teff = stellar_params[:, 0]
        log_g = stellar_params[:, 1]
        feh = stellar_params[:, 2]

        # Constraint 1: approximate mapping from stellar properties to limb-darkening profile.
        u1 = 0.45 - 3.0e-5 * (teff - 5777.0) + 0.05 * (log_g - 4.4) - 0.03 * feh
        # Constraint 1: second-order coefficient captures curvature near stellar limb.
        u2 = 0.25 + 2.0e-5 * (teff - 5777.0) - 0.03 * (log_g - 4.4) + 0.02 * feh
        return torch.clamp(u1, 0.0, 1.0), torch.clamp(u2, 0.0, 1.0)

    def _extract_impact_parameter(self, stellar_params: Tensor) -> Tensor:
        """Extract impact parameter ``b`` from stellar parameters if available."""
        batch = stellar_params.shape[0]
        if stellar_params.shape[1] >= 5:
            b = stellar_params[:, 4]
            return torch.clamp(b, 0.0, 0.95)
        return torch.full((batch,), DEFAULT_IMPACT_PARAMETER, device=stellar_params.device, dtype=stellar_params.dtype)

    def _compute_limb_darkening_bias(
        self,
        u1: Tensor,
        u2: Tensor,
        positions: Tensor,
        impact_parameter: Tensor,
    ) -> Tensor:
        """Compute radial intensity profile from quadratic limb-darkening law.

        Parameters
        ----------
        u1 : torch.Tensor
            First limb-darkening coefficient with shape ``(batch,)``.
        u2 : torch.Tensor
            Second limb-darkening coefficient with shape ``(batch,)``.
        positions : torch.Tensor
            Normalized phase positions in ``[-1, 1]`` with shape ``(time,)``.
        impact_parameter : torch.Tensor
            Transit impact parameter ``b`` with shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Limb-darkening intensity profile with shape ``(batch, time)``.
        """
        s = positions.abs().clamp(0.0, 1.0)
        b = impact_parameter.unsqueeze(1)
        # Constraint 1: sequence phase is mapped to a transit chord, then to projected disk radius r.
        r = torch.sqrt(torch.clamp((b * b) + (1.0 - b * b) * (s.unsqueeze(0) ** 2), min=0.0, max=0.999999))
        # Constraint 1: mu = cos(theta) term from projected stellar-disk geometry.
        mu = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))
        one_minus_mu = 1.0 - mu

        # Constraint 1: quadratic law directly encodes center-to-limb intensity variation.
        intensity = 1.0 - u1.unsqueeze(1) * one_minus_mu - u2.unsqueeze(1) * (one_minus_mu ** 2)

        # Constraint 1: normalize per star to produce stable additive attention bias.
        intensity = intensity / torch.clamp(intensity.mean(dim=1, keepdim=True), min=EPS_ATTENTION)
        return intensity

    def forward(self, x: Tensor, stellar_params: Tensor) -> Tensor:
        """Apply position-aware attention to sequence features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, time, d_model)``.
        stellar_params : torch.Tensor
            Stellar parameters used to derive or provide ``u1`` and ``u2``.

        Returns
        -------
        torch.Tensor
            Attended features with shape ``(batch, time, d_model)``.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, time, d_model)")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"x last dimension must be d_model={self.d_model}")

        batch, time_steps, _ = x.shape

        # Constraint 1: project features to query/key/value spaces for geometry-aware routing.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, time_steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time_steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time_steps, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        # Constraint 1: base content attention is computed before adding physical radial bias.
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        positions = torch.linspace(-1.0, 1.0, time_steps, device=x.device, dtype=x.dtype)
        u1, u2 = self._infer_limb_darkening_coefficients(stellar_params)
        impact_parameter = self._extract_impact_parameter(stellar_params)
        # Constraint 1: compute chord-aware brightness profile from limb-darkening coefficients and impact parameter.
        intensity = self._compute_limb_darkening_bias(
            u1=u1,
            u2=u2,
            positions=positions,
            impact_parameter=impact_parameter,
        )

        # Constraint 1: pairwise additive bias favors attention among physically brighter disk positions.
        pair_bias = torch.log(
            torch.clamp(intensity.unsqueeze(2) * intensity.unsqueeze(1), min=EPS_ATTENTION)
        )

        # Constraint 1: add limb-darkening bias to all heads to break translation invariance.
        attn_logits = attn_logits + pair_bias.unsqueeze(1)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attended = torch.matmul(attn_weights, v)

        attended = attended.transpose(1, 2).contiguous().view(batch, time_steps, self.d_model)
        output = self.out_proj(attended)
        output = self.norm(x + output)

        self._last_attention = attn_weights.detach()
        return output

    def get_last_attention(self) -> Tensor | None:
        """Return most recent attention maps for visualization."""
        return self._last_attention
