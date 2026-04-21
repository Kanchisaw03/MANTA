"""Symmetry-constrained convolutional encoder for transit morphology.

This module implements the fourth physics-derived constraint: for near-circular
orbits, ingress and egress are time-reversed counterparts, so convolutional
filters should be palindromic by construction.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .elliptic_activation import EllipticMish


class PalindromicConv1d(nn.Module):
    """1D convolution with enforced palindromic kernel weights.

    Physics Justification
    ---------------------
    Constraint 4 enforces ingress/egress time symmetry by constructing each
    kernel such that ``W == reverse(W)`` exactly.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Odd kernel width.
    bias : bool, optional
        Whether to include additive bias.

    Raises
    ------
    ValueError
        If ``kernel_size`` is not odd.

    Examples
    --------
    >>> layer = PalindromicConv1d(1, 8, kernel_size=7)
    >>> x = torch.randn(2, 1, 201)
    >>> y = layer(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("PalindromicConv1d requires an odd kernel_size")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2

        half_size = (self.kernel_size + 1) // 2
        self.half_kernel = nn.Parameter(torch.empty(out_channels, in_channels, half_size))
        nn.init.kaiming_uniform_(self.half_kernel, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def _build_full_kernel(self) -> Tensor:
        """Reconstruct full symmetric kernel from half-kernel parameters."""
        # Constraint 4: The right side is the mirror of left side to enforce ingress/egress equivalence.
        mirrored = torch.flip(self.half_kernel[..., :-1], dims=(-1,))
        # Constraint 4: Concatenation guarantees exact palindromic symmetry in every forward pass.
        full_kernel = torch.cat([self.half_kernel, mirrored], dim=-1)
        return full_kernel

    def forward(self, x: Tensor) -> Tensor:
        """Apply palindromic convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Output tensor after symmetric convolution.
        """
        # Constraint 4: Kernel is reconstructed from tied parameters before convolution.
        kernel = self._build_full_kernel()
        # Constraint 4: Same-padding preserves temporal alignment around transit center.
        return F.conv1d(x, kernel, bias=self.bias, stride=1, padding=self.padding)


class SymmetricTransitEncoder(nn.Module):
    """Transit-shape encoder using stacked palindromic convolutions.

    Physics Justification
    ---------------------
    Constraint 4 is implemented through repeated symmetric filters, while
    Constraint 2 is co-implemented via EllipticMish nonlinearities to preserve
    smooth ingress/egress geometry.

    Parameters
    ----------
    in_channels : int, optional
        Input channel count.
    base_channels : int, optional
        Base feature width for encoder stack.
    kernel_sizes : Iterable[int], optional
        Odd kernel sizes per layer.

    Examples
    --------
    >>> encoder = SymmetricTransitEncoder(in_channels=1, base_channels=16)
    >>> x = torch.randn(4, 1, 201)
    >>> features = encoder(x)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        kernel_sizes: Iterable[int] = (9, 7, 5),
    ) -> None:
        super().__init__()
        kernels = tuple(int(k) for k in kernel_sizes)
        if any(k % 2 == 0 for k in kernels):
            raise ValueError("All kernel sizes must be odd for symmetry preservation")

        channel_plan = [in_channels, base_channels, base_channels * 2, base_channels * 2]
        layers: list[nn.Module] = []
        for i, kernel_size in enumerate(kernels):
            c_in = channel_plan[i]
            c_out = channel_plan[i + 1]
            # Constraint 4: Palindromic filters encode ingress/egress time reversibility directly in weights.
            layers.append(PalindromicConv1d(c_in, c_out, kernel_size=kernel_size, bias=True))
            # Constraint 2: Smooth non-monotonic activation tracks elliptic transit geometry better than ReLU.
            layers.append(EllipticMish(alpha_init=0.1))
            layers.append(nn.GroupNorm(num_groups=1, num_channels=c_out))

        self.encoder = nn.Sequential(*layers)
        self.output_channels = channel_plan[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Encode local transit morphology.

        Parameters
        ----------
        x : torch.Tensor
            Local-view tensor with shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Encoded transit features with preserved temporal dimension.
        """
        # Constraint 4: Entire stack preserves symmetry-aware feature extraction at every depth.
        return self.encoder(x)

    def get_symmetric_kernels(self) -> list[Tensor]:
        """Return reconstructed full kernels for visualization and testing."""
        kernels: list[Tensor] = []
        for module in self.modules():
            if isinstance(module, PalindromicConv1d):
                kernels.append(module._build_full_kernel().detach().cpu())
        return kernels
