"""Run all MANTA ablation variants and export summary tables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from manta.data.dataset import KeplerTransitDataset, split_dataset
from manta.data.downloader import download_kepler_tce_catalog
from manta.evaluation.ablation import AblationStudy
from manta.utils.config import load_config
from manta.utils.reproducibility import set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MANTA ablation variants")
    parser.add_argument("--config", type=str, default="configs/manta_default.yaml", help="Path to YAML config")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--output-dir", type=str, default="outputs/ablation", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    config = load_config(args.config)
    set_all_seeds(config.seed)

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
    )

    train_ds, val_ds, test_ds = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=config.seed)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)

    study = AblationStudy(
        config=config,
        device=config.device if torch.cuda.is_available() else "cpu",
        output_dir=args.output_dir,
    )

    results = study.run_all(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_seeds=args.n_seeds,
    )

    latex_table = study.generate_ablation_table(results)
    table_path = Path(args.output_dir) / "ablation_table.tex"
    table_path.write_text(latex_table, encoding="utf-8")
    logging.info("Ablation complete. Results: %s | Table: %s", Path(args.output_dir) / "ablation_results.csv", table_path)


if __name__ == "__main__":
    main()
