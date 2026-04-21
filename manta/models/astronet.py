"""AstroNet baseline model (Shallue & Vanderburg 2018) in PyTorch.

This module provides a faithful two-tower 1D-CNN baseline for direct comparison
against MANTA in ablations and headline benchmark experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class AstroNetPaperConfig:
    """Hyperparameters matching Shallue and Vanderburg (2018)."""

    global_input_length: int = 2001
    local_input_length: int = 201
    conv_kernel_size: int = 5
    global_pool_kernel_size: int = 5
    global_pool_stride: int = 2
    local_pool_kernel_size: int = 7
    local_pool_stride: int = 2
    dropout: float = 0.4
    fc_units: int = 512


class _ConvReluPool(nn.Module):
    """Helper block: Conv1d -> ReLU -> optional MaxPool1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_kernel: int | None,
        pool_stride: int | None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()
        if pool_kernel is None:
            self.pool = nn.Identity()
        else:
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.act(self.conv(x)))


class AstroNet(nn.Module):
    """AstroNet binary transit classifier.

    References
    ----------
    Shallue, C. J., and Vanderburg, A. (2018),
    "Identifying Exoplanets with Deep Learning: A Five-Planet Resonant Chain
    Around Kepler-80 and an Eighth Planet Around Kepler-90".

    Parameters
    ----------
    config : AstroNetPaperConfig | None, optional
        Model hyperparameter set. If ``None``, paper defaults are used.
    """

    def __init__(self, config: AstroNetPaperConfig | None = None) -> None:
        super().__init__()
        self.config = config or AstroNetPaperConfig()
        c = self.config

        # Global-view tower (paper: five convolutional blocks).
        self.global_tower = nn.Sequential(
            _ConvReluPool(1, 16, c.conv_kernel_size, c.global_pool_kernel_size, c.global_pool_stride),
            _ConvReluPool(16, 16, c.conv_kernel_size, c.global_pool_kernel_size, c.global_pool_stride),
            _ConvReluPool(16, 32, c.conv_kernel_size, c.global_pool_kernel_size, c.global_pool_stride),
            _ConvReluPool(32, 32, c.conv_kernel_size, c.global_pool_kernel_size, c.global_pool_stride),
            _ConvReluPool(32, 64, c.conv_kernel_size, None, None),
            nn.AdaptiveAvgPool1d(3),
            nn.Flatten(),
        )

        # Local-view tower (paper: five convolutional blocks with stronger pooling).
        self.local_tower = nn.Sequential(
            _ConvReluPool(1, 16, c.conv_kernel_size, c.local_pool_kernel_size, c.local_pool_stride),
            _ConvReluPool(16, 16, c.conv_kernel_size, c.local_pool_kernel_size, c.local_pool_stride),
            _ConvReluPool(16, 32, c.conv_kernel_size, None, None),
            _ConvReluPool(32, 32, c.conv_kernel_size, None, None),
            _ConvReluPool(32, 64, c.conv_kernel_size, None, None),
            nn.AdaptiveAvgPool1d(2),
            nn.Flatten(),
        )

        merged_features = (64 * 3) + (64 * 2)
        self.classifier = nn.Sequential(
            nn.Linear(merged_features, c.fc_units),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.fc_units, c.fc_units),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.fc_units, c.fc_units),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.fc_units, c.fc_units),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.fc_units, 1),
            nn.Sigmoid(),
        )

    @classmethod
    def from_paper_config(cls) -> "AstroNet":
        """Instantiate AstroNet with exact paper hyperparameters."""
        return cls(config=AstroNetPaperConfig())

    def forward(self, batch: dict[str, Tensor] | Tensor, local_view: Tensor | None = None) -> Tensor:
        """Run forward inference.

        Parameters
        ----------
        batch : dict[str, torch.Tensor] | torch.Tensor
            Either a batch dictionary containing ``global_view`` and ``local_view``
            or a global-view tensor if ``local_view`` is provided separately.
        local_view : torch.Tensor | None, optional
            Local-view tensor used when ``batch`` is a tensor.

        Returns
        -------
        torch.Tensor
            Transit probability tensor with shape ``(batch, 1)``.
        """
        if isinstance(batch, dict):
            global_view = batch["global_view"]
            local = batch["local_view"]
        else:
            if local_view is None:
                raise ValueError("local_view must be provided when batch is a tensor")
            global_view = batch
            local = local_view

        g_feat = self.global_tower(global_view)
        l_feat = self.local_tower(local)
        merged = torch.cat([g_feat, l_feat], dim=1)
        return self.classifier(merged)
