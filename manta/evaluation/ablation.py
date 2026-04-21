"""Systematic ablation runner for MANTA architecture components.

This module automates controlled removals of physics-derived constraints to
quantify each component's independent contribution.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import logging

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from manta.models.astronet import AstroNet
from manta.models.manta import MANTA
from manta.training.loss import FocalBCELoss
from manta.training.trainer import MANTATrainer
from manta.utils.config import MANTAConfig
from manta.utils.reproducibility import set_all_seeds

LOGGER = logging.getLogger(__name__)

ABlATION_VARIANTS: tuple[str, ...] = (
    "full_manta",
    "minus_position_attention",
    "minus_elliptic_activation",
    "minus_symmetric_encoder",
    "minus_frequency_separation",
    "minus_physics_output",
    "astronet_baseline",
)


class AblationStudy:
    """Run and summarize MANTA ablation experiments.

    Parameters
    ----------
    config : MANTAConfig
        Base experiment configuration.
    device : str
        Compute device string.
    output_dir : str | Path
        Directory where ablation checkpoints and CSV outputs are saved.
    """

    def __init__(self, config: MANTAConfig, device: str, output_dir: str | Path = "outputs/ablation") -> None:
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_variant_model(self, variant: str) -> nn.Module:
        """Construct one ablation-model variant."""
        if variant == "astronet_baseline":
            return AstroNet.from_paper_config()

        model = MANTA(self.config)

        if variant == "full_manta":
            return model

        if variant == "minus_position_attention":
            model.global_attention = nn.Identity()  # type: ignore[assignment]
            model.local_attention = nn.Identity()  # type: ignore[assignment]
            return model

        if variant == "minus_elliptic_activation":
            self._replace_activation(model, target_class_name="EllipticMish", replacement=nn.ReLU())
            return model

        if variant == "minus_symmetric_encoder":
            model.local_encoder = nn.Sequential(
                nn.Conv1d(1, model.local_encoder.output_channels, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(model.local_encoder.output_channels, model.local_encoder.output_channels, kernel_size=5, padding=2),
                nn.ReLU(),
            )  # type: ignore[assignment]
            return model

        if variant == "minus_frequency_separation":
            from manta.models.components.elliptic_activation import EllipticMish

            class _SingleStream(nn.Module):
                def __init__(self, out_channels: int) -> None:
                    super().__init__()
                    self.output_channels = out_channels
                    self.net = nn.Sequential(
                        nn.Conv1d(1, out_channels, kernel_size=9, padding=4),
                        EllipticMish(alpha_init=0.1),
                        nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
                        EllipticMish(alpha_init=0.1),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.net(x[:, 0:1, :])

            model.freq_processor = _SingleStream(model.freq_processor.output_channels)  # type: ignore[assignment]
            return model

        if variant == "minus_physics_output":
            in_features = model.output_head.linear.in_features
            model.output_head = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())  # type: ignore[assignment]
            return model

        raise ValueError(f"Unknown ablation variant: {variant}")

    def _replace_activation(self, module: nn.Module, target_class_name: str, replacement: nn.Module) -> None:
        """Recursively replace activation modules by class name."""
        for name, child in module.named_children():
            if child.__class__.__name__ == target_class_name:
                setattr(module, name, replacement)
            else:
                self._replace_activation(child, target_class_name, replacement)

    def _evaluate_loader(self, model: nn.Module, loader: DataLoader[Any]) -> dict[str, float]:
        """Compute evaluation metrics for a loader."""
        from manta.evaluation.metrics import compute_all_metrics

        model.eval()
        preds: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch_device = {
                    k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                out = model(batch_device)
                preds.append(out.detach().cpu().flatten())
                labels.append(batch_device["label"].detach().cpu().flatten())

        y_pred = torch.cat(preds).numpy() if preds else torch.empty(0).numpy()
        y_true = torch.cat(labels).numpy() if labels else torch.empty(0).numpy()
        return compute_all_metrics(y_true=y_true, y_pred_proba=y_pred, threshold=self.config.evaluation.threshold)

    def run_all(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        test_loader: DataLoader[Any],
        n_seeds: int,
    ) -> pd.DataFrame:
        """Train all ablation variants across multiple seeds.

        Parameters
        ----------
        train_loader : DataLoader
            Training loader.
        val_loader : DataLoader
            Validation loader.
        test_loader : DataLoader
            Test loader.
        n_seeds : int
            Number of random seeds.

        Returns
        -------
        pandas.DataFrame
            Aggregated metrics for each variant/seed run.
        """
        results: list[dict[str, float | str | int]] = []

        for variant in ABlATION_VARIANTS:
            for seed_offset in range(n_seeds):
                seed = int(self.config.seed + seed_offset)
                set_all_seeds(seed)

                LOGGER.info("Ablation run started: variant=%s seed=%d", variant, seed)
                model = self._build_variant_model(variant)
                optimizer = AdamW(
                    model.parameters(),
                    lr=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                )
                scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
                loss_fn = FocalBCELoss(
                    gamma=self.config.training.focal_gamma,
                    alpha=self.config.training.focal_alpha,
                )

                run_dir = self.output_dir / variant / f"seed_{seed}"
                trainer = MANTATrainer(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    device=self.device,
                    config=self.config,
                    checkpoint_dir=run_dir,
                )
                trainer.fit(train_loader=train_loader, val_loader=val_loader, n_epochs=self.config.training.max_epochs)

                # Load best checkpoint if available.
                best_ckpt_candidates = sorted(run_dir.glob("checkpoint_best_epoch_*.pt"))
                if best_ckpt_candidates:
                    trainer.load_checkpoint(best_ckpt_candidates[-1])

                test_metrics = self._evaluate_loader(model=trainer.model, loader=test_loader)
                record: dict[str, float | str | int] = {
                    "variant": variant,
                    "seed": seed,
                    **{k: float(v) for k, v in test_metrics.items()},
                }
                results.append(record)

                LOGGER.info(
                    "Ablation run finished: variant=%s seed=%d auc=%.4f",
                    variant,
                    seed,
                    record.get("auc_roc", 0.0),
                )

        df = pd.DataFrame(results)
        output_csv = self.output_dir / "ablation_results.csv"
        df.to_csv(output_csv, index=False)
        LOGGER.info("Saved ablation results to %s", output_csv)
        return df

    def generate_ablation_table(self, results_df: pd.DataFrame) -> str:
        """Generate LaTeX ablation table string for manuscript inclusion."""
        summary = (
            results_df.groupby("variant")
            .agg(
                auc_roc_mean=("auc_roc", "mean"),
                auc_roc_std=("auc_roc", "std"),
                f1_mean=("f1", "mean"),
                ap_mean=("average_precision", "mean"),
            )
            .reset_index()
        )

        ordered = [v for v in ABlATION_VARIANTS if v in set(summary["variant"])]
        summary["variant"] = pd.Categorical(summary["variant"], categories=ordered, ordered=True)
        summary = summary.sort_values("variant")

        lines = [
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Variant & AUC-ROC & F1 & AP \\\\",
            "\\midrule",
        ]

        full_auc = float(summary.loc[summary["variant"] == "full_manta", "auc_roc_mean"].iloc[0]) if "full_manta" in set(summary["variant"]) else np.nan

        for row in summary.itertuples(index=False):
            delta = row.auc_roc_mean - full_auc if np.isfinite(full_auc) else np.nan
            lines.append(
                f"{row.variant.replace('_', ' ')} & "
                f"{row.auc_roc_mean:.4f} $\\pm$ {0.0 if np.isnan(row.auc_roc_std) else row.auc_roc_std:.4f} "
                f"($\\Delta$={delta:+.4f}) & {row.f1_mean:.4f} & {row.ap_mean:.4f} \\\\"  # noqa: E501
            )

        lines.extend(["\\bottomrule", "\\end{tabular}"])
        return "\n".join(lines)
