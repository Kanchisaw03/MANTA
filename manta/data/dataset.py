"""PyTorch dataset and split utilities for MANTA transit classification.

This module constructs train-ready samples by combining preprocessing,
frequency decomposition, stellar metadata extraction, and label handling while
preserving star-level split integrity to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, Subset

from .downloader import DataUnavailableError, download_lightcurve
from .frequency_decomposer import FrequencyDecomposer
from .preprocessor import normalize_flux, phase_fold, remove_nans, remove_outliers

LOGGER = logging.getLogger(__name__)

DEFAULT_GLOBAL_BINS: int = 2001
DEFAULT_LOCAL_BINS: int = 201
DEFAULT_STELLAR_T_EFF: float = 5777.0
DEFAULT_STELLAR_LOG_G: float = 4.4
DEFAULT_STELLAR_FEH: float = 0.0
DEFAULT_ROT_PERIOD_DAYS: float = 20.0
DEFAULT_IMPACT_PARAMETER: float = 0.5


@dataclass(slots=True)
class _SampleRecord:
    """Internal index record for one Kepler ID and quarter."""

    kepler_id: int
    quarter: int
    label: int
    period_days: float
    t0_days: float
    duration_hours: float
    teff: float
    log_g: float
    feh: float
    impact_parameter: float


class KeplerTransitDataset(Dataset):
    """Dataset returning MANTA-ready feature dictionaries.

    Parameters
    ----------
    tce_catalog : pandas.DataFrame
        DR25 TCE table with star IDs, quarter metadata, and labels.
    cache_dir : str | Path
        Cache root containing downloaded and preprocessed artifacts.
    preprocessing_config : dict[str, Any]
        Preprocessing configuration values.
    augmentation_config : dict[str, Any]
        Augmentation settings (currently parsed for compatibility).

    Notes
    -----
    Each sample dictionary has keys:
    ``global_view``, ``local_view``, ``freq_bands``, ``stellar_params``,
    ``label``, and ``kepler_id``.
    """

    def __init__(
        self,
        tce_catalog: pd.DataFrame,
        cache_dir: str | Path,
        preprocessing_config: dict[str, Any] | None = None,
        augmentation_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.tce_catalog = tce_catalog.copy()
        self.cache_dir = Path(cache_dir)
        self.raw_cache_dir = self.cache_dir / "raw"
        self.preprocessed_dir = self.cache_dir / "preprocessed"
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessing_config = preprocessing_config or {}
        self.augmentation_config = augmentation_config or {}

        self.global_bins = int(self.preprocessing_config.get("global_view_bins", DEFAULT_GLOBAL_BINS))
        self.local_bins = int(self.preprocessing_config.get("local_view_bins", DEFAULT_LOCAL_BINS))
        self.nan_strategy = str(self.preprocessing_config.get("nan_strategy", "hybrid"))
        self.norm_method = str(self.preprocessing_config.get("normalization_method", "spline"))
        self.sigma_threshold = float(self.preprocessing_config.get("sigma_clip_threshold", 5.0))
        self.cadence_days = float(self.preprocessing_config.get("kepler_cadence_days", 29.4244 / (60.0 * 24.0)))

        self._decomposer = FrequencyDecomposer(
            diagnostics_dir=self.preprocessing_config.get("diagnostics_dir", "outputs/diagnostics")
        )
        self._index: list[_SampleRecord] = self._build_index(self.tce_catalog)
        LOGGER.info("Dataset index constructed with %d samples", len(self._index))

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Return one model-ready sample.

        Returns
        -------
        dict[str, torch.Tensor | str]
            Dictionary with tensor views, frequency bands, stellar features, label,
            and Kepler ID string.

        Raises
        ------
        DataUnavailableError
            If the required raw light curve cannot be downloaded.
        """
        record = self._index[idx]
        npz_path = self._sample_npz_path(record.kepler_id, record.quarter)

        if npz_path.exists():
            sample = np.load(npz_path)
            global_view = sample["global_view"].astype(np.float32)
            local_view = sample["local_view"].astype(np.float32)
            freq_bands = sample["freq_bands"].astype(np.float32)
            prot_days = float(sample["prot_days"]) if "prot_days" in sample else DEFAULT_ROT_PERIOD_DAYS
        else:
            generated = self._generate_and_cache_sample(record)
            global_view = generated["global_view"]
            local_view = generated["local_view"]
            freq_bands = generated["freq_bands"]
            prot_days = float(generated["prot_days"])

        stellar_params = np.array(
            [record.teff, record.log_g, record.feh, prot_days, record.impact_parameter],
            dtype=np.float32,
        )

        return {
            "global_view": torch.from_numpy(global_view).unsqueeze(0),
            "local_view": torch.from_numpy(local_view).unsqueeze(0),
            "freq_bands": torch.from_numpy(freq_bands),
            "stellar_params": torch.from_numpy(stellar_params),
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "kepler_id": str(record.kepler_id),
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights.

        Returns
        -------
        torch.Tensor
            Tensor ``[w_negative, w_positive]`` suitable for imbalance handling.
        """
        labels = np.array([r.label for r in self._index], dtype=np.int64)
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        counts = np.where(counts == 0.0, 1.0, counts)
        weights = 1.0 / counts
        weights = weights / np.sum(weights) * 2.0
        return torch.tensor(weights, dtype=torch.float32)

    def _sample_npz_path(self, kepler_id: int, quarter: int) -> Path:
        """Build path for cached preprocessed sample arrays."""
        return self.preprocessed_dir / f"kic_{kepler_id}_q{quarter}.npz"

    def _build_index(self, df: pd.DataFrame) -> list[_SampleRecord]:
        """Build dataset index from a TCE DataFrame."""
        kepid_col = _infer_column(df, ("kepid", "kepler_id"))
        quarter_col = _infer_column(df, ("quarter", "koi_quarter"), default_value=1)
        label_col = _infer_label_column(df)

        period_col = _infer_column(df, ("tce_period", "period", "period_days"), default_value=np.nan)
        t0_col = _infer_column(df, ("tce_time0bk", "t0", "t0_days"), default_value=np.nan)
        duration_col = _infer_column(df, ("tce_duration", "duration_hours", "duration"), default_value=np.nan)
        impact_col = _infer_column(df, ("tce_impact", "impact_parameter", "impact", "koi_impact"), default_value=np.nan)

        teff_col = _infer_column(df, ("teff", "kic_teff"), default_value=np.nan)
        logg_col = _infer_column(df, ("logg", "log_g", "kic_logg"), default_value=np.nan)
        feh_col = _infer_column(df, ("feh", "metallicity", "kic_feh"), default_value=np.nan)

        index: list[_SampleRecord] = []
        for row in df.itertuples(index=False):
            values = row._asdict()
            kepler_id = int(values[kepid_col])
            quarter = int(values[quarter_col])
            label = int(values[label_col])
            period_days = float(values[period_col]) if np.isfinite(values[period_col]) else 10.0
            t0_days = float(values[t0_col]) if np.isfinite(values[t0_col]) else 0.0
            duration_hours = float(values[duration_col]) if np.isfinite(values[duration_col]) else 5.0
            teff = float(values[teff_col]) if np.isfinite(values[teff_col]) else DEFAULT_STELLAR_T_EFF
            log_g = float(values[logg_col]) if np.isfinite(values[logg_col]) else DEFAULT_STELLAR_LOG_G
            feh = float(values[feh_col]) if np.isfinite(values[feh_col]) else DEFAULT_STELLAR_FEH
            impact_parameter = float(values[impact_col]) if np.isfinite(values[impact_col]) else DEFAULT_IMPACT_PARAMETER
            impact_parameter = float(np.clip(impact_parameter, 0.0, 0.95))

            index.append(
                _SampleRecord(
                    kepler_id=kepler_id,
                    quarter=quarter,
                    label=label,
                    period_days=period_days,
                    t0_days=t0_days,
                    duration_hours=duration_hours,
                    teff=teff,
                    log_g=log_g,
                    feh=feh,
                    impact_parameter=impact_parameter,
                )
            )
        return index

    def _generate_and_cache_sample(self, record: _SampleRecord) -> dict[str, NDArray[np.float32] | float]:
        """Generate one sample from raw light curve and persist it as NPZ."""
        LOGGER.debug("Generating sample for KIC %d Q%d", record.kepler_id, record.quarter)
        lightcurve = download_lightcurve(record.kepler_id, record.quarter, self.raw_cache_dir)

        time = np.asarray(getattr(getattr(lightcurve, "time", lightcurve["time"]), "value", lightcurve["time"]), dtype=np.float64)
        flux = np.asarray(getattr(getattr(lightcurve, "flux", lightcurve["flux"]), "value", lightcurve["flux"]), dtype=np.float64)

        clean_flux, clean_time, _ = remove_nans(flux=flux, time=time, strategy=self.nan_strategy)
        normalized_flux, _ = normalize_flux(clean_flux, method=self.norm_method)
        clipped_flux, keep_mask, _ = remove_outliers(normalized_flux, sigma_threshold=self.sigma_threshold)
        final_time = clean_time[keep_mask]

        if record.t0_days == 0.0:
            t0_days = float(final_time[np.argmin(clipped_flux)])
        else:
            t0_days = record.t0_days

        # Constraint 5: Frequency decomposition must be done on the raw cleaned timeline,
        # before phase-folding, to preserve true stellar temporal frequency content.
        cadence_days = float(np.nanmedian(np.diff(final_time))) if final_time.size > 1 else self.cadence_days
        if not np.isfinite(cadence_days) or cadence_days <= 0.0:
            cadence_days = self.cadence_days

        raw_bands = self._decomposer.decompose(
            flux=clipped_flux,
            time=final_time,
            cadence_days=cadence_days,
        )

        folded = phase_fold(
            time=final_time,
            flux=clipped_flux,
            period=record.period_days,
            t0=t0_days,
            duration_hours=max(record.duration_hours, 0.5),
            global_bins=self.global_bins,
            local_bins=self.local_bins,
        )

        global_view = folded["global_view"]
        local_view = folded["local_view"]

        gran_folded = phase_fold(
            time=final_time,
            flux=raw_bands.granulation,
            period=record.period_days,
            t0=t0_days,
            duration_hours=max(record.duration_hours, 0.5),
            global_bins=self.global_bins,
            local_bins=self.local_bins,
        )
        astero_folded = phase_fold(
            time=final_time,
            flux=raw_bands.asteroseismology,
            period=record.period_days,
            t0=t0_days,
            duration_hours=max(record.duration_hours, 0.5),
            global_bins=self.global_bins,
            local_bins=self.local_bins,
        )
        starspot_folded = phase_fold(
            time=final_time,
            flux=raw_bands.starspot,
            period=record.period_days,
            t0=t0_days,
            duration_hours=max(record.duration_hours, 0.5),
            global_bins=self.global_bins,
            local_bins=self.local_bins,
        )

        freq_bands = np.stack(
            [
                gran_folded["global_view"],
                astero_folded["global_view"],
                starspot_folded["global_view"],
            ],
            axis=0,
        ).astype(np.float32)

        npz_path = self._sample_npz_path(record.kepler_id, record.quarter)
        np.savez_compressed(
            npz_path,
            global_view=global_view.astype(np.float32),
            local_view=local_view.astype(np.float32),
            freq_bands=freq_bands,
            prot_days=float(raw_bands.metadata["rotation_period_days"]),
        )

        return {
            "global_view": global_view.astype(np.float32),
            "local_view": local_view.astype(np.float32),
            "freq_bands": freq_bands,
            "prot_days": float(raw_bands.metadata["rotation_period_days"]),
        }


def split_dataset(
    dataset: KeplerTransitDataset,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[Subset[KeplerTransitDataset], Subset[KeplerTransitDataset], Subset[KeplerTransitDataset]]:
    """Create star-level stratified train/val/test splits.

    Parameters
    ----------
    dataset : KeplerTransitDataset
        Dataset instance to split.
    train_ratio : float
        Fraction of unique stars assigned to train split.
    val_ratio : float
        Fraction of unique stars assigned to validation split.
    seed : int
        RNG seed for deterministic partitioning.

    Returns
    -------
    tuple[Subset, Subset, Subset]
        Train, validation, and test subsets.

    Notes
    -----
    Splits are grouped by Kepler ID so no star appears across multiple splits.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = np.random.default_rng(seed)

    by_star: dict[int, list[int]] = {}
    star_label: dict[int, int] = {}
    for idx, record in enumerate(dataset._index):
        by_star.setdefault(record.kepler_id, []).append(idx)
        star_label[record.kepler_id] = max(star_label.get(record.kepler_id, 0), record.label)

    stars_pos = np.array([s for s, label in star_label.items() if label == 1], dtype=np.int64)
    stars_neg = np.array([s for s, label in star_label.items() if label == 0], dtype=np.int64)

    rng.shuffle(stars_pos)
    rng.shuffle(stars_neg)

    def _split_stars(stars: NDArray[np.int64]) -> tuple[set[int], set[int], set[int]]:
        n = stars.size
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train = set(int(v) for v in stars[:n_train])
        val = set(int(v) for v in stars[n_train : n_train + n_val])
        test = set(int(v) for v in stars[n_train + n_val :])
        return train, val, test

    train_pos, val_pos, test_pos = _split_stars(stars_pos)
    train_neg, val_neg, test_neg = _split_stars(stars_neg)

    train_stars = train_pos | train_neg
    val_stars = val_pos | val_neg
    test_stars = test_pos | test_neg

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for star, indices in by_star.items():
        if star in train_stars:
            train_indices.extend(indices)
        elif star in val_stars:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return (
        Subset(dataset, sorted(train_indices)),
        Subset(dataset, sorted(val_indices)),
        Subset(dataset, sorted(test_indices)),
    )


def _infer_label_column(df: pd.DataFrame) -> str:
    """Resolve binary label column from common Kepler catalog conventions."""
    if "label" in df.columns:
        return "label"
    if "av_training_set" in df.columns:
        values = df["av_training_set"].astype(str).str.upper()
        df.loc[:, "label"] = (values == "PC").astype(int)
        return "label"
    raise KeyError("Could not infer label column; expected 'label' or 'av_training_set'")


def _infer_column(df: pd.DataFrame, candidates: tuple[str, ...], default_value: Any | None = None) -> str:
    """Return first matching column from aliases, optionally creating fallback."""
    lower = {name.lower(): name for name in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if default_value is not None:
        new_name = candidates[0]
        df.loc[:, new_name] = default_value
        return new_name

    raise KeyError(f"Missing required columns: {candidates}")
