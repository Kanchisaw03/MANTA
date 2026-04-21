"""Download Kepler DR25 catalog and light curves for MANTA.

This script performs resilient metadata download and parallel light-curve
caching for downstream preprocessing and training.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from manta.data.downloader import batch_download, download_kepler_tce_catalog
from manta.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kepler DR25 TCE catalog and light curves")
    parser.add_argument("--config", type=str, default="configs/manta_default.yaml", help="Path to YAML config")
    parser.add_argument("--force-refresh", action="store_true", help="Force re-download of TCE catalog")
    parser.add_argument("--max-workers", type=int, default=None, help="Override download worker count")
    parser.add_argument("--cache_dir", type=str, default=None, help="Override cache directory")
    parser.add_argument("--n_stars", type=int, default=None, help="Limit to N unique stars")
    parser.add_argument("--quarters", type=int, nargs="*", default=None, help="Limit to specific quarter numbers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible star subsampling")
    return parser.parse_args()


def _infer_column(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"None of columns {candidates} found in DataFrame")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    config = load_config(args.config)
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting DR25 TCE catalog download")
    tce_df = download_kepler_tce_catalog(
        cache_dir=cache_dir,
        url=config.data.tce_catalog_url,
        force_refresh=args.force_refresh,
        retries=config.data.download_retries,
        timeout_seconds=config.data.timeout_seconds,
    )
    logging.info("Catalog rows: %d", len(tce_df))

    # Optional subset controls for quick Colab runs.
    kepid_col = _infer_column(tce_df, ("kepid", "kepler_id"))
    quarter_col = _infer_column(tce_df, ("quarter", "koi_quarter"))

    filtered_df = tce_df.copy()
    if args.quarters:
        quarter_set = {int(q) for q in args.quarters}
        filtered_df = filtered_df[filtered_df[quarter_col].isin(quarter_set)].copy()
        logging.info("Filtered to quarters %s: %d rows", sorted(quarter_set), len(filtered_df))

    if args.n_stars is not None:
        if args.n_stars <= 0:
            raise ValueError("--n_stars must be positive")

        unique_stars = filtered_df[kepid_col].dropna().astype(int).unique()
        if unique_stars.size == 0:
            raise ValueError("No stars available after applying filters")

        rng = np.random.default_rng(args.seed)
        n_pick = min(args.n_stars, unique_stars.size)
        chosen_stars = set(int(s) for s in rng.choice(unique_stars, size=n_pick, replace=False).tolist())
        filtered_df = filtered_df[filtered_df[kepid_col].astype(int).isin(chosen_stars)].copy()
        logging.info("Filtered to %d unique stars (%d rows)", n_pick, len(filtered_df))

    if filtered_df.empty:
        raise ValueError("No rows left to download after filtering by stars/quarters")

    workers = args.max_workers if args.max_workers is not None else config.data.max_download_workers
    summary = batch_download(tce_df=filtered_df, cache_dir=cache_dir / "raw", max_workers=workers)

    logging.info(
        "Download complete: downloaded=%d skipped=%d failed=%d",
        summary.downloaded,
        summary.skipped,
        summary.failed,
    )

    if summary.failures:
        failure_file = cache_dir / "download_failures.log"
        failure_file.write_text("\n".join(summary.failures), encoding="utf-8")
        logging.warning("Failure log written to %s", failure_file)


if __name__ == "__main__":
    main()
