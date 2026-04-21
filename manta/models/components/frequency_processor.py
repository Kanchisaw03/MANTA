"""Parallel frequency-branch processor for stellar variability separation.

This module implements the fifth physics-derived constraint by processing
frequency-separated signals in specialized neural branches matched to physical
timescales (granulation, asteroseismology, and starspot rotation).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .elliptic_activation import EllipticMish


class GranulationBranch(nn.Module):
    """Wide receptive-field branch for slow granulation evolution.

    Physics Justification
    ---------------------
    Constraint 5: granulation varies on relatively slow hour-scale structure,
    motivating dilated convolutions with broad context windows.
    """

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        # Constraint 5: Dilated kernels increase temporal context for low-frequency granulation envelopes.
        self.net = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=15, padding=7, dilation=1),
            EllipticMish(alpha_init=0.05),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=14, dilation=2),
            EllipticMish(alpha_init=0.05),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AsteroseismologyBranch(nn.Module):
    """Narrow receptive-field branch for higher-frequency oscillations.

    Physics Justification
    ---------------------
    Constraint 5: minute-to-hour oscillatory components require fine temporal
    localization, so this branch uses smaller kernels without dilation.
    """

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        # Constraint 5: Small kernels preserve high-frequency detail in p-mode-like structures.
        self.net = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=5, padding=2),
            EllipticMish(alpha_init=0.05),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            EllipticMish(alpha_init=0.05),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StarspotBranch(nn.Module):
    """Very wide-context branch for rotational starspot modulation.

    Physics Justification
    ---------------------
    Constraint 5: starspot signals evolve on rotation timescales, so this branch
    combines strided downsampling and self-attention for long-context modeling.
    """

    def __init__(self, out_channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.out_channels = out_channels
        # Constraint 5: Strided convolution compresses long rotational context efficiently.
        self.pre = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=21, padding=10, stride=2),
            EllipticMish(alpha_init=0.05),
            nn.Conv1d(out_channels, out_channels, kernel_size=11, padding=5, stride=2),
            EllipticMish(alpha_init=0.05),
        )
        # Constraint 5: Attention models long-period dependencies in rotational modulation.
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x_reduced = self.pre(x)
        seq = x_reduced.transpose(1, 2)
        attended, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm(seq + attended)
        return seq.transpose(1, 2)


class ParallelFrequencyProcessor(nn.Module):
    """Process three decomposed frequency bands in parallel specialized branches.

    Parameters
    ----------
    granulation_channels : int
        Output channel count for granulation branch.
    astero_channels : int
        Output channel count for asteroseismology branch.
    starspot_channels : int
        Output channel count for starspot branch.

    Notes
    -----
    The expected input shape is ``(batch, 3, time)``, ordered as:
    ``[granulation, asteroseismology, starspot]``.
    """

    def __init__(
        self,
        granulation_channels: int,
        astero_channels: int,
        starspot_channels: int,
    ) -> None:
        super().__init__()
        self.granulation_branch = GranulationBranch(out_channels=granulation_channels)
        self.asteroseismology_branch = AsteroseismologyBranch(out_channels=astero_channels)
        self.starspot_branch = StarspotBranch(out_channels=starspot_channels)

        total_channels = granulation_channels + astero_channels + starspot_channels
        # Constraint 5: 1x1 fusion keeps branch-specific physics separated before learned recombination.
        self.fusion = nn.Sequential(
            nn.Conv1d(total_channels, total_channels, kernel_size=1),
            EllipticMish(alpha_init=0.05),
            nn.GroupNorm(1, total_channels),
        )
        self.output_channels = total_channels
        self._last_feature_maps: dict[str, Tensor] = {}

    def forward(self, freq_bands: Tensor) -> Tensor:
        """Forward pass through parallel frequency branches.

        Parameters
        ----------
        freq_bands : torch.Tensor
            Tensor of shape ``(batch, 3, time)``.

        Returns
        -------
        torch.Tensor
            Fused feature tensor of shape ``(batch, channels, time)``.
        """
        if freq_bands.ndim != 3 or freq_bands.shape[1] != 3:
            raise ValueError("freq_bands must have shape (batch, 3, time)")

        gran = freq_bands[:, 0:1, :]
        astero = freq_bands[:, 1:2, :]
        star = freq_bands[:, 2:3, :]

        # Constraint 5: Each branch processes only its matched physical frequency band.
        gran_feat = self.granulation_branch(gran)
        astero_feat = self.asteroseismology_branch(astero)
        star_feat = self.starspot_branch(star)

        target_length = freq_bands.shape[-1]
        if star_feat.shape[-1] != target_length:
            # Constraint 5: Upsampling preserves rotational-context features while restoring common timeline.
            star_feat = F.interpolate(star_feat, size=target_length, mode="linear", align_corners=False)

        # Constraint 5: Concatenation keeps physically distinct representations explicit before fusion.
        fused_input = torch.cat([gran_feat, astero_feat, star_feat], dim=1)
        fused = self.fusion(fused_input)

        self._last_feature_maps = {
            "granulation": gran_feat.detach(),
            "asteroseismology": astero_feat.detach(),
            "starspot": star_feat.detach(),
            "fused": fused.detach(),
        }
        return fused

    def get_feature_maps(self) -> dict[str, Tensor]:
        """Return intermediate branch activations for visualization."""
        return self._last_feature_maps
