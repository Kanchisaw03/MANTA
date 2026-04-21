"""Train AstroNet baseline for MANTA comparison."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from manta.data.dataset import KeplerTransitDataset, split_dataset
from manta.data.downloader import download_kepler_tce_catalog
from manta.models.astronet import AstroNet
from manta.training.loss import FocalBCELoss
from manta.training.trainer import MANTATrainer
from manta.utils.config import load_config
from manta.utils.reproducibility import get_run_hash, set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AstroNet baseline")
    parser.add_argument("--config", type=str, default="configs/astronet_baseline.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory override")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed

    set_all_seeds(config.seed)
    run_hash = get_run_hash(config=config, seed=config.seed)

    tce_df = download_kepler_tce_catalog(cache_dir=config.data.cache_dir, url=config.data.tce_catalog_url)

    dataset = KeplerTransitDataset(
        tce_catalog=tce_df,
        cache_dir=config.data.cache_dir,
        preprocessing_config={
            "nan_strategy": config.data.nan_strategy,
            "normalization_method": config.data.normalization_method,
            "sigma_clip_threshold": config.data.sigma_clip_threshold,
            "global_view_bins": config.data.global_view_bins,
            "local_view_bins": config.data.local_view_bins,
            "kepler_cadence_days": config.data.kepler_cadence_days,
            "diagnostics_dir": config.data.diagnostics_dir,
        },
        augmentation_config={"use_augmentation": False},
    )

    train_ds, val_ds, _ = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=config.seed)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)

    model = AstroNet.from_paper_config()
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    loss_fn = FocalBCELoss(gamma=config.training.focal_gamma, alpha=config.training.focal_alpha)

    checkpoint_dir = args.checkpoint_dir or config.training.checkpoint_dir
    trainer = MANTATrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        device=config.device if torch.cuda.is_available() else "cpu",
        config=config,
        checkpoint_dir=checkpoint_dir,
    )

    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, n_epochs=config.training.max_epochs)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"astronet_metrics_{run_hash}.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logging.info("Training complete. Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
