"""Learning-rate scheduling utilities for MANTA training.

This module implements the physics warmup schedule motivated by the need to
stabilize symmetry-constrained kernels before aggressive optimization.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PhysicsWarmupScheduler(_LRScheduler):
    """Warmup + cosine decay learning-rate schedule.

    Physics Justification
    ---------------------
    Palindromic kernels (constraint 4) can collapse early under large updates.
    A short warmup allows symmetric filters to stabilize before full-rate
    optimization, then cosine annealing improves late-stage convergence.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_epochs : int
        Number of warmup epochs.
    total_epochs : int
        Total training epochs.
    min_lr : float, optional
        Final cosine floor learning rate.
    last_epoch : int, optional
        Last epoch index.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1.0e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.total_epochs = int(max(1, total_epochs))
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """Compute per-parameter-group learning rates."""
        epoch = self.last_epoch + 1
        lrs: list[float] = []

        for base_lr in self.base_lrs:
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                warmup_scale = epoch / float(self.warmup_epochs)
                lr = max(self.min_lr, base_lr * warmup_scale)
            else:
                progress_numerator = max(0, epoch - self.warmup_epochs)
                progress_denominator = max(1, self.total_epochs - self.warmup_epochs)
                progress = min(1.0, progress_numerator / progress_denominator)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            lrs.append(float(lr))

        return lrs
