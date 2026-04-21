"""Training loop implementation for MANTA and baseline models.

This trainer provides production-grade experiment control with checkpointing,
metrics, early stopping, and Kaggle-session-aware emergency persistence.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any

import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from manta.evaluation.metrics import compute_all_metrics
from manta.utils.config import MANTAConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_SESSION_MINUTES: float = 9.0 * 60.0


class MANTATrainer:
    """Train, validate, and checkpoint MANTA experiments.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    scheduler : Any
        Learning-rate scheduler or ``None``.
    loss_fn : torch.nn.Module
        Loss function.
    device : str | torch.device
        Compute device.
    config : MANTAConfig
        Experiment configuration.
    checkpoint_dir : str | Path
        Checkpoint output directory.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        loss_fn: nn.Module,
        device: str | torch.device,
        config: MANTAConfig,
        checkpoint_dir: str | Path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.config = config

        self.model.to(self.device)

        self._is_kaggle = bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()
        base_dir = Path("/kaggle/working") if self._is_kaggle else Path(checkpoint_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = base_dir

        self._start_time = monotonic()
        self._best_auc = -np.inf
        self._best_epoch = -1
        self.history: list[dict[str, float]] = []

        LOGGER.info("Trainer initialized on device=%s, checkpoint_dir=%s", self.device, self.checkpoint_dir)

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor batch entries to trainer device."""
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def train_epoch(self, dataloader: DataLoader[Any]) -> dict[str, float]:
        """Run one training epoch.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Training loader.

        Returns
        -------
        dict[str, float]
            Training metrics including loss and classification statistics.
        """
        self.model.train()
        total_loss = 0.0
        all_pred: list[np.ndarray] = []
        all_true: list[np.ndarray] = []

        LOGGER.info("Training epoch started with %d batches", len(dataloader))
        for batch_idx, raw_batch in enumerate(dataloader):
            batch = self._move_batch_to_device(raw_batch)
            target = batch["label"].float().view(-1, 1)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(batch)
            loss = self.loss_fn(pred, target)
            loss.backward()

            grad_clip = float(self.config.training.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            self.optimizer.step()

            total_loss += float(loss.item())
            all_pred.append(pred.detach().flatten().cpu().numpy())
            all_true.append(target.detach().flatten().cpu().numpy())

            if batch_idx % 50 == 0:
                LOGGER.debug("Train batch %d/%d loss=%.6f", batch_idx + 1, len(dataloader), float(loss.item()))

        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.float64)
        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.float64)

        metrics = compute_all_metrics(
            y_true=y_true,
            y_pred_proba=y_pred,
            threshold=self.config.evaluation.threshold,
        )
        metrics["loss"] = float(total_loss / max(1, len(dataloader)))
        LOGGER.info("Training epoch completed: loss=%.6f auc=%.4f", metrics["loss"], metrics["auc_roc"])
        return metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader[Any]) -> dict[str, float]:
        """Run model validation and return metrics.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Validation loader.

        Returns
        -------
        dict[str, float]
            Validation metrics including AUC-ROC, precision, recall, F1, and AP.
        """
        self.model.eval()
        total_loss = 0.0
        all_pred: list[np.ndarray] = []
        all_true: list[np.ndarray] = []

        LOGGER.info("Validation started with %d batches", len(dataloader))
        for raw_batch in dataloader:
            batch = self._move_batch_to_device(raw_batch)
            target = batch["label"].float().view(-1, 1)
            pred = self.model(batch)
            loss = self.loss_fn(pred, target)

            total_loss += float(loss.item())
            all_pred.append(pred.detach().flatten().cpu().numpy())
            all_true.append(target.detach().flatten().cpu().numpy())

        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.float64)
        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.float64)

        metrics = compute_all_metrics(
            y_true=y_true,
            y_pred_proba=y_pred,
            threshold=self.config.evaluation.threshold,
        )
        metrics["loss"] = float(total_loss / max(1, len(dataloader)))
        LOGGER.info(
            "Validation completed: loss=%.6f auc=%.4f precision=%.4f recall=%.4f f1=%.4f ap=%.4f",
            metrics["loss"],
            metrics["auc_roc"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["average_precision"],
        )
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict[str, float], tag: str = "latest") -> Path:
        """Save full training state.

        Parameters
        ----------
        epoch : int
            Epoch index.
        metrics : dict[str, float]
            Metrics dictionary.
        tag : str, optional
            Checkpoint tag suffix.

        Returns
        -------
        pathlib.Path
            Saved checkpoint path.
        """
        path = self.checkpoint_dir / f"checkpoint_{tag}_epoch_{epoch:03d}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None and hasattr(self.scheduler, "state_dict") else None,
            "metrics": metrics,
            "config": asdict(self.config),
        }
        torch.save(state, path)

        metrics_path = self.checkpoint_dir / f"metrics_{tag}_epoch_{epoch:03d}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        LOGGER.info("Checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load full training state from checkpoint.

        Parameters
        ----------
        path : str | Path
            Checkpoint path.

        Returns
        -------
        dict[str, Any]
            Loaded checkpoint dictionary.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])

        scheduler_state = payload.get("scheduler_state_dict")
        if scheduler_state is not None and self.scheduler is not None and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(scheduler_state)

        LOGGER.info("Loaded checkpoint from %s (epoch=%s)", checkpoint_path, payload.get("epoch"))
        return payload

    def _ensure_scheduler(self) -> None:
        """Instantiate default scheduler if none is provided."""
        if self.scheduler is None:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                min_lr=self.config.training.min_lr,
            )

    def _estimate_remaining_session_minutes(self) -> float:
        """Estimate remaining session minutes for Kaggle-aware checkpointing."""
        max_minutes = float(os.environ.get("MANTA_SESSION_MAX_MINUTES", DEFAULT_MAX_SESSION_MINUTES))
        elapsed = (monotonic() - self._start_time) / 60.0
        return max(0.0, max_minutes - elapsed)

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        n_epochs: int,
    ) -> list[dict[str, float]]:
        """Train model with early stopping and scheduler updates.

        Parameters
        ----------
        train_loader : DataLoader
            Training loader.
        val_loader : DataLoader
            Validation loader.
        n_epochs : int
            Number of epochs to run.

        Returns
        -------
        list[dict[str, float]]
            Epoch-wise metrics history.
        """
        self._ensure_scheduler()

        patience = int(self.config.training.early_stopping_patience)
        patience_counter = 0

        LOGGER.info("Training fit loop started for %d epochs", n_epochs)
        for epoch in range(1, n_epochs + 1):
            LOGGER.info("Epoch %d/%d started", epoch, n_epochs)

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            epoch_metrics = {
                "epoch": float(epoch),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            }
            self.history.append(epoch_metrics)

            val_auc = float(val_metrics.get("auc_roc", 0.0))
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_auc)
            elif self.scheduler is not None and hasattr(self.scheduler, "step"):
                self.scheduler.step()

            self.save_checkpoint(epoch=epoch, metrics=epoch_metrics, tag="latest")

            if val_auc > self._best_auc:
                self._best_auc = val_auc
                self._best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch=epoch, metrics=epoch_metrics, tag="best")
            else:
                patience_counter += 1

            remaining = self._estimate_remaining_session_minutes()
            if remaining < float(self.config.training.emergency_checkpoint_minutes_left):
                LOGGER.warning(
                    "Estimated remaining session time %.1f minutes below threshold; saving emergency checkpoint",
                    remaining,
                )
                self.save_checkpoint(epoch=epoch, metrics=epoch_metrics, tag="emergency")

            if patience_counter >= patience:
                LOGGER.info("Early stopping triggered at epoch %d (best epoch=%d)", epoch, self._best_epoch)
                break

            LOGGER.info(
                "Epoch %d complete: val_auc=%.4f best_auc=%.4f patience=%d/%d",
                epoch,
                val_auc,
                self._best_auc,
                patience_counter,
                patience,
            )

        LOGGER.info("Training completed. Best epoch=%d best_auc=%.4f", self._best_epoch, self._best_auc)
        return self.history
