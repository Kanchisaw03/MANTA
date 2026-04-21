"""Kaggle session-aware MANTA runner with auto-resume support."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from scripts.train_manta import main as train_manta_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle session manager for MANTA")
    parser.add_argument("--config", type=str, default="configs/manta_default.yaml", help="Config path")
    parser.add_argument("--checkpoint-dir", type=str, default="/kaggle/working/checkpoints", help="Checkpoint directory")
    parser.add_argument("--session-max-minutes", type=float, default=9.0 * 60.0, help="Expected Kaggle session budget")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    on_kaggle = bool(os.environ.get("KAGGLE_URL_BASE")) or Path("/kaggle").exists()
    if not on_kaggle:
        logging.warning("Kaggle environment not detected; running with local paths")

    os.environ["MANTA_SESSION_MAX_MINUTES"] = str(args.session_max_minutes)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest = sorted(checkpoint_dir.glob("checkpoint_latest_epoch_*.pt"))
    if latest:
        logging.info("Found previous checkpoints. Training script will continue and overwrite latest snapshots.")
    else:
        logging.info("No prior checkpoints detected; starting fresh training session.")

    # Delegate actual training orchestration to train_manta script.
    train_manta_main()


if __name__ == "__main__":
    main()
