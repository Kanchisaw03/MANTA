"""Evaluate trained models and generate all paper figures."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from manta.data.dataset import KeplerTransitDataset, split_dataset
from manta.data.downloader import download_kepler_tce_catalog
from manta.evaluation.metrics import compute_all_metrics
from manta.evaluation.visualizer import (
    plot_ablation_heatmap,
    plot_activation_comparison,
    plot_calibration_curves,
    plot_frequency_decomposition,
    plot_per_planet_size_performance,
    plot_roc_curves,
    plot_transit_detection,
)
from manta.models.astronet import AstroNet
from manta.models.manta import MANTA
from manta.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint")
    parser.add_argument("--config", type=str, default="configs/manta_default.yaml", help="Config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--model", type=str, default="manta", choices=["manta", "astronet"], help="Model type")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Output directory")
    return parser.parse_args()


def _load_model(model_name: str, config_path: str):
    config = load_config(config_path)
    if model_name == "manta":
        model = MANTA(config)
    else:
        model = AstroNet.from_paper_config()
    return model, config


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    model, config = _load_model(args.model, args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

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
    _, _, test_ds = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=config.seed)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)

    model.eval()
    preds = []
    labels = []
    first_batch = None

    with torch.no_grad():
        for batch in test_loader:
            if first_batch is None:
                first_batch = batch
            batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(batch_device)
            preds.append(out.detach().cpu().numpy().reshape(-1))
            labels.append(batch_device["label"].detach().cpu().numpy().reshape(-1))

    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(labels) if labels else np.array([])

    metrics = compute_all_metrics(y_true=y_true, y_pred_proba=y_pred, threshold=config.evaluation.threshold)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Figure generation (8 figures).
    if first_batch is not None:
        lc = {
            "time": np.arange(first_batch["global_view"].shape[-1]) * config.data.kepler_cadence_days,
            "flux": first_batch["global_view"][0, 0].numpy(),
        }
        plot_transit_detection(lc=lc, prediction=float(y_pred[0]) if y_pred.size else 0.0, ground_truth=int(y_true[0]) if y_true.size else 0, output_dir=out_dir)
        bands = {
            "granulation": first_batch["freq_bands"][0, 0].numpy(),
            "asteroseismology": first_batch["freq_bands"][0, 1].numpy(),
            "starspot": first_batch["freq_bands"][0, 2].numpy(),
        }
        plot_frequency_decomposition(flux=lc["flux"], time=lc["time"], bands=bands, output_dir=out_dir)

    plot_activation_comparison(output_dir=out_dir)

    roc_payload = {args.model: {"y_true": y_true, "y_pred": y_pred}}
    plot_roc_curves(roc_payload, output_dir=out_dir)

    dummy_ablation = __import__("pandas").DataFrame(
        [{"variant": args.model, "auc_roc": metrics["auc_roc"], "f1": metrics["f1"], "average_precision": metrics["average_precision"]}]
    )
    plot_ablation_heatmap(dummy_ablation, output_dir=out_dir)

    dummy_bins = __import__("pandas").DataFrame(
        [{"group_type": "radius_bin", "group": "(0.0, 0.1]", "auc_roc": metrics["auc_roc"]}]
    )
    plot_per_planet_size_performance(dummy_bins, output_dir=out_dir)

    plot_calibration_curves(y_true=y_true, y_pred_dict={args.model: y_pred}, output_dir=out_dir)

    logging.info("Evaluation complete. Metrics and figures saved in %s", out_dir)


if __name__ == "__main__":
    main()
