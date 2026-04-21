"""Synthetic transit augmentation for MANTA training.

This module injects physically valid Mandel-Agol transits using batman to enrich
training coverage of shallow and grazing events that are underrepresented in raw
Kepler candidate labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

DEFAULT_CADENCE_DAYS: float = 29.4244 / (60.0 * 24.0)
DEFAULT_SERIES_DURATION_DAYS: float = 27.0

RP_RS_MIN: float = 0.01
RP_RS_MAX: float = 0.2
IMPACT_MIN: float = 0.0
IMPACT_MAX: float = 0.9
PERIOD_MIN_DAYS: float = 1.0
PERIOD_MAX_DAYS: float = 200.0


def _import_batman() -> Any:
    """Import batman lazily to keep test environments lightweight."""
    try:
        import batman  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "batman-package is required for transit injection. Install with: pip install batman-package"
        ) from exc
    return batman


@dataclass(slots=True)
class TransitInjector:
    """Inject physically valid transits and generate augmentation datasets.

    Parameters
    ----------
    seed : int | None, optional
        Seed controlling parameter draws and noise realizations.
    """

    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def inject_transit(
        self,
        flux: NDArray[np.float64],
        time: NDArray[np.float64],
        params: dict[str, float],
    ) -> tuple[NDArray[np.float64], dict[str, float]]:
        """Inject a batman transit into an existing flux sequence.

        Parameters
        ----------
        flux : numpy.ndarray
            Baseline flux series.
        time : numpy.ndarray
            Time values in days.
        params : dict[str, float]
            Transit parameters. Required keys: ``rp_rs``, ``impact_parameter``,
            ``period_days``. Optional keys include ``t0_days``, ``u1``, ``u2``,
            ``eccentricity``, ``omega_deg``, and ``a_rs``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, float]]
            Flux with injected transit and resolved ground-truth parameter dict.

        Raises
        ------
        ValueError
            If any parameter is outside physically valid bounds.
        """
        resolved = self._resolve_and_validate_params(params=params, time=time)
        batman = _import_batman()

        tparams = batman.TransitParams()
        tparams.t0 = resolved["t0_days"]
        tparams.per = resolved["period_days"]
        tparams.rp = resolved["rp_rs"]
        tparams.a = resolved["a_rs"]
        tparams.inc = resolved["inclination_deg"]
        tparams.ecc = resolved["eccentricity"]
        tparams.w = resolved["omega_deg"]
        tparams.u = [resolved["u1"], resolved["u2"]]
        tparams.limb_dark = "quadratic"

        model = batman.TransitModel(tparams, time)
        model_flux = np.asarray(model.light_curve(tparams), dtype=np.float64)

        injected_flux = np.asarray(flux, dtype=np.float64) * model_flux
        return injected_flux, resolved

    def generate_synthetic_dataset(
        self,
        n_samples: int,
        stellar_params_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate balanced synthetic positive/negative examples.

        Parameters
        ----------
        n_samples : int
            Total number of examples.
        stellar_params_df : pandas.DataFrame
            Stellar metadata used to sample physically plausible limb-darkening
            coefficients and temperatures.

        Returns
        -------
        pandas.DataFrame
            Dataset with ``time``, ``flux``, ``label``, and ``params`` columns.
        """
        if n_samples <= 1:
            raise ValueError("n_samples must be > 1 for a balanced dataset")

        n_positive = n_samples // 2
        n_negative = n_samples - n_positive
        n_cadences = int(DEFAULT_SERIES_DURATION_DAYS / DEFAULT_CADENCE_DAYS)
        time = np.arange(n_cadences, dtype=np.float64) * DEFAULT_CADENCE_DAYS

        rows: list[dict[str, Any]] = []

        # Positive class: inject transits into realistic noise backgrounds.
        for _ in range(n_positive):
            noise_std = float(self._rng.uniform(2.0e-4, 2.0e-3))
            base_flux = 1.0 + self._rng.normal(0.0, noise_std, size=n_cadences)
            stellar = self._sample_stellar_params(stellar_params_df)
            params = self._random_transit_params(time=time, stellar=stellar)
            injected_flux, resolved = self.inject_transit(base_flux, time, params)
            rows.append({"time": time.copy(), "flux": injected_flux, "label": 1, "params": resolved})

        # Negative class: pure stellar/noise patterns without transit injections.
        for _ in range(n_negative):
            noise_std = float(self._rng.uniform(2.0e-4, 2.0e-3))
            red_noise = np.cumsum(self._rng.normal(0.0, noise_std * 0.02, size=n_cadences))
            white_noise = self._rng.normal(0.0, noise_std, size=n_cadences)
            flux = 1.0 + red_noise + white_noise
            rows.append({"time": time.copy(), "flux": flux, "label": 0, "params": {}})

        dataset = pd.DataFrame(rows)
        return dataset

    def augment_existing(self, lc: dict[str, Any], n_augmentations: int) -> list[dict[str, Any]]:
        """Augment a confirmed transit with stochastic and physical perturbations.

        Parameters
        ----------
        lc : dict[str, Any]
            Light-curve dictionary containing ``time``, ``flux``, and optional
            transit keys such as ``period_days``, ``t0_days``, and ``rp_rs``.
        n_augmentations : int
            Number of augmented examples to generate.

        Returns
        -------
        list[dict[str, Any]]
            Augmented light-curve dictionaries.
        """
        if n_augmentations < 1:
            return []

        time = np.asarray(lc["time"], dtype=np.float64)
        flux = np.asarray(lc["flux"], dtype=np.float64)
        augmentations: list[dict[str, Any]] = []

        for _ in range(n_augmentations):
            phase_shift = int(self._rng.integers(0, max(1, flux.size)))
            shifted = np.roll(flux, phase_shift)
            noise_sigma = float(np.nanstd(flux - np.nanmedian(flux)) * 0.15 + 1.0e-5)
            noisy = shifted + self._rng.normal(0.0, noise_sigma, size=shifted.size)

            params = {
                "rp_rs": float(lc.get("rp_rs", self._rng.uniform(RP_RS_MIN, RP_RS_MAX))),
                "impact_parameter": float(
                    np.clip(lc.get("impact_parameter", self._rng.uniform(IMPACT_MIN, IMPACT_MAX)), IMPACT_MIN, IMPACT_MAX)
                ),
                "period_days": float(
                    np.clip(lc.get("period_days", self._rng.uniform(PERIOD_MIN_DAYS, PERIOD_MAX_DAYS)), PERIOD_MIN_DAYS, PERIOD_MAX_DAYS)
                ),
                "t0_days": float(lc.get("t0_days", np.nanmedian(time))),
                "u1": float(np.clip(lc.get("u1", 0.4) + self._rng.normal(0.0, 0.03), 0.0, 1.0)),
                "u2": float(np.clip(lc.get("u2", 0.25) + self._rng.normal(0.0, 0.03), 0.0, 1.0)),
            }

            try:
                augmented_flux, resolved = self.inject_transit(noisy, time, params)
            except Exception:
                # If batman is unavailable or parameterization fails, retain stochastic augmentation.
                augmented_flux = noisy
                resolved = params

            augmentations.append({
                "time": time.copy(),
                "flux": augmented_flux,
                "label": int(lc.get("label", 1)),
                "params": resolved,
            })

        return augmentations

    def _sample_stellar_params(self, stellar_params_df: pd.DataFrame) -> dict[str, float]:
        """Sample one stellar row and infer limb-darkening defaults."""
        if stellar_params_df.empty:
            return {"u1": 0.4, "u2": 0.25, "teff": 5777.0, "logg": 4.4, "feh": 0.0}

        row = stellar_params_df.sample(n=1, random_state=int(self._rng.integers(0, 10_000_000))).iloc[0]
        u1 = float(row.get("u1", row.get("ld_u1", 0.4)))
        u2 = float(row.get("u2", row.get("ld_u2", 0.25)))
        return {
            "u1": float(np.clip(u1, 0.0, 1.0)),
            "u2": float(np.clip(u2, 0.0, 1.0)),
            "teff": float(row.get("teff", 5777.0)),
            "logg": float(row.get("log_g", row.get("logg", 4.4))),
            "feh": float(row.get("feh", row.get("metallicity", 0.0))),
        }

    def _random_transit_params(self, time: NDArray[np.float64], stellar: dict[str, float]) -> dict[str, float]:
        """Draw physically plausible transit parameters for simulation."""
        rp_rs = float(self._rng.uniform(RP_RS_MIN, RP_RS_MAX))
        impact = float(self._rng.uniform(IMPACT_MIN, IMPACT_MAX))
        period = float(np.exp(self._rng.uniform(np.log(PERIOD_MIN_DAYS), np.log(PERIOD_MAX_DAYS))))
        t0_days = float(self._rng.uniform(float(time[0]), float(time[-1])))

        a_rs = float(4.21 * period ** (2.0 / 3.0))
        inclination = np.degrees(np.arccos(np.clip(impact / max(a_rs, 1.0e-6), 0.0, 1.0)))

        return {
            "rp_rs": rp_rs,
            "impact_parameter": impact,
            "period_days": period,
            "t0_days": t0_days,
            "u1": stellar["u1"],
            "u2": stellar["u2"],
            "a_rs": a_rs,
            "inclination_deg": float(inclination),
            "eccentricity": 0.0,
            "omega_deg": 90.0,
        }

    def _resolve_and_validate_params(
        self,
        params: dict[str, float],
        time: NDArray[np.float64],
    ) -> dict[str, float]:
        """Fill optional transit parameters and enforce physical bounds."""
        rp_rs = float(params["rp_rs"])
        impact = float(params["impact_parameter"])
        period = float(params["period_days"])

        if not (RP_RS_MIN <= rp_rs <= RP_RS_MAX):
            raise ValueError(f"rp_rs must be in [{RP_RS_MIN}, {RP_RS_MAX}]")
        if not (IMPACT_MIN <= impact <= IMPACT_MAX):
            raise ValueError(f"impact_parameter must be in [{IMPACT_MIN}, {IMPACT_MAX}]")
        if not (PERIOD_MIN_DAYS <= period <= PERIOD_MAX_DAYS):
            raise ValueError(f"period_days must be in [{PERIOD_MIN_DAYS}, {PERIOD_MAX_DAYS}]")

        t0_days = float(params.get("t0_days", np.nanmedian(time)))
        u1 = float(np.clip(params.get("u1", 0.4), 0.0, 1.0))
        u2 = float(np.clip(params.get("u2", 0.25), 0.0, 1.0))
        eccentricity = float(np.clip(params.get("eccentricity", 0.0), 0.0, 0.99))
        omega_deg = float(params.get("omega_deg", 90.0))

        a_rs = float(params.get("a_rs", 4.21 * period ** (2.0 / 3.0)))
        if a_rs <= 0.0:
            raise ValueError("a_rs must be positive")

        if "inclination_deg" in params:
            inclination = float(params["inclination_deg"])
        else:
            cos_i = np.clip(impact / a_rs, 0.0, 1.0)
            inclination = float(np.degrees(np.arccos(cos_i)))

        return {
            "rp_rs": rp_rs,
            "impact_parameter": impact,
            "period_days": period,
            "t0_days": t0_days,
            "u1": u1,
            "u2": u2,
            "a_rs": a_rs,
            "inclination_deg": inclination,
            "eccentricity": eccentricity,
            "omega_deg": omega_deg,
        }
