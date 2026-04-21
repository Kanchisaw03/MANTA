"""Tests for MANTA data pipeline modules."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manta.data.augmenter import TransitInjector
from manta.data.downloader import DataUnavailableError, batch_download
from manta.data.frequency_decomposer import FrequencyDecomposer
from manta.data.preprocessor import phase_fold, remove_nans


@pytest.fixture
def synthetic_lightcurve() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    cadence_days = 29.4244 / (60.0 * 24.0)
    time = np.arange(2048, dtype=np.float64) * cadence_days
    flux = 1.0 + 5.0e-4 * rng.normal(size=time.size)
    return {"time": time, "flux": flux}


def test_remove_nans_interpolate_small_gap(synthetic_lightcurve: dict[str, np.ndarray]) -> None:
    flux = synthetic_lightcurve["flux"].copy()
    time = synthetic_lightcurve["time"].copy()
    flux[100:105] = np.nan

    clean_flux, clean_time, mask = remove_nans(flux=flux, time=time, strategy="interpolate")

    assert clean_flux.size == flux.size
    assert clean_time.size == time.size
    assert mask.sum() == flux.size
    assert np.all(np.isfinite(clean_flux))


def test_remove_nans_mask_large_gap(synthetic_lightcurve: dict[str, np.ndarray]) -> None:
    flux = synthetic_lightcurve["flux"].copy()
    time = synthetic_lightcurve["time"].copy()
    flux[200:225] = np.nan

    clean_flux, clean_time, mask = remove_nans(flux=flux, time=time, strategy="hybrid")

    assert clean_flux.size < flux.size
    assert clean_time.size == clean_flux.size
    assert mask.sum() == clean_flux.size


def test_phase_fold_centering() -> None:
    period = 5.0
    t0 = 2.0
    duration_hours = 4.0
    time = np.linspace(0.0, 100.0, 5000)
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    flux = 1.0 - 0.01 * np.exp(-0.5 * (phase / 0.02) ** 2)

    folded = phase_fold(time=time, flux=flux, period=period, t0=t0, duration_hours=duration_hours)
    min_idx = int(np.argmin(folded["local_view"]))
    assert abs(min_idx - (len(folded["local_view"]) // 2)) <= 3


def test_frequency_decomposition_reconstruction_error(synthetic_lightcurve: dict[str, np.ndarray]) -> None:
    time = synthetic_lightcurve["time"]
    flux = synthetic_lightcurve["flux"]

    decomposer = FrequencyDecomposer(diagnostics_dir="outputs/test_diagnostics")
    bands = decomposer.decompose(flux=flux, time=time, cadence_days=np.median(np.diff(time)))
    reconstructed = decomposer.reconstruct(bands)

    error = np.max(np.abs(reconstructed - flux))
    assert error < 1.0e-6


def test_frequency_decomposition_band_selectivity() -> None:
    cadence_days = 29.4244 / (60.0 * 24.0)
    time = np.arange(4096, dtype=np.float64) * cadence_days

    starspot_component = 0.010 * np.sin(2.0 * np.pi * 0.25 * time)
    granulation_component = 0.004 * np.sin(2.0 * np.pi * 2.4 * time)
    astero_component = 0.003 * np.sin(2.0 * np.pi * 9.0 * time)
    flux = 1.0 + starspot_component + granulation_component + astero_component

    decomposer = FrequencyDecomposer(diagnostics_dir="outputs/test_diagnostics")
    bands = decomposer.decompose(flux=flux, time=time, cadence_days=cadence_days)

    gran_corr = np.corrcoef(bands.granulation - np.mean(bands.granulation), granulation_component)[0, 1]
    astero_corr = np.corrcoef(bands.asteroseismology, astero_component)[0, 1]
    starspot_corr = np.corrcoef(bands.starspot, starspot_component)[0, 1]

    assert abs(gran_corr) > 0.55
    assert abs(astero_corr) > 0.75
    assert abs(starspot_corr) > 0.75


def test_batch_download_skips_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = pd.DataFrame({"kepid": [1001, 1002], "quarter": [1, 2]})

    calls: list[tuple[int, int]] = []

    def fake_download(kepid: int, quarter: int, cache_dir: Path):
        calls.append((kepid, quarter))
        path = Path(cache_dir) / f"kic_{kepid}_q{quarter}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"time": np.array([0.0, 1.0]), "flux": np.array([1.0, 1.0])}, handle)
        return {"ok": True}

    monkeypatch.setattr("manta.data.downloader.download_lightcurve", fake_download)

    summary_first = batch_download(df, cache_dir=tmp_path, max_workers=2)
    summary_second = batch_download(df, cache_dir=tmp_path, max_workers=2)

    assert summary_first.downloaded == 2
    assert summary_second.skipped == 2
    assert len(calls) == 2


def test_data_unavailable_error_message() -> None:
    err = DataUnavailableError(kepler_id=12345, quarter=7, details="missing")
    assert "kepler_id=12345" in str(err)
    assert "quarter=7" in str(err)


def test_transit_injection_parameter_recovery(synthetic_lightcurve: dict[str, np.ndarray]) -> None:
    injector = TransitInjector(seed=11)
    params = {
        "rp_rs": 0.08,
        "impact_parameter": 0.3,
        "period_days": 12.0,
        "t0_days": float(np.median(synthetic_lightcurve["time"])),
        "u1": 0.4,
        "u2": 0.25,
    }

    try:
        injected_flux, resolved = injector.inject_transit(
            flux=synthetic_lightcurve["flux"],
            time=synthetic_lightcurve["time"],
            params=params,
        )
    except RuntimeError:
        pytest.skip("batman-package not installed in test environment")

    assert injected_flux.shape == synthetic_lightcurve["flux"].shape
    assert abs(resolved["rp_rs"] - params["rp_rs"]) < 1.0e-8
    assert abs(resolved["period_days"] - params["period_days"]) < 1.0e-8
