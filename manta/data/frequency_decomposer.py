"""Frequency-domain decomposition for stellar variability separation.

This module operationalizes MANTA's fifth physics-derived constraint by
explicitly decomposing flux into granulation, asteroseismology, and starspot
bands before neural processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import lombscargle

LOGGER = logging.getLogger(__name__)

# Physical band boundaries.
HOURS_PER_DAY: float = 24.0
MINUTES_PER_DAY: float = 24.0 * 60.0
F_GRAN_CYCLES_PER_DAY: float = 1.0 / (8.0 / HOURS_PER_DAY)  # 1 / 8 hours -> 3 c/d.
F_ASTERO_RAW_CYCLES_PER_DAY: float = 1.0 / (5.0 / MINUTES_PER_DAY)  # 1 / 5 minutes -> 288 c/d.
ROTATION_PERIOD_MIN_DAYS: float = 0.5
ROTATION_PERIOD_MAX_DAYS: float = 60.0
ROTATION_HALF_WIDTH_FRACTION: float = 0.15
ROTATION_MIN_HALF_WIDTH_CPD: float = 0.02

DEFAULT_DIAGNOSTIC_DIR: str = "outputs/diagnostics"


@dataclass(slots=True)
class DecompositionResult:
    """Container for decomposed frequency bands and metadata."""

    granulation: NDArray[np.float64]
    asteroseismology: NDArray[np.float64]
    starspot: NDArray[np.float64]
    metadata: dict[str, float | NDArray[np.float64] | NDArray[np.bool_]]


class FrequencyDecomposer:
    """Split a light curve into physics-motivated frequency bands.

    The three outputs align with MANTA's parallel frequency branches:

    1. Granulation band: low frequency baseline up to ``f_gran``.
    2. Asteroseismology band: from ``f_gran`` to Nyquist-limited ``f_astero``.
    3. Starspot band: narrow band centered on stellar rotation frequency.

    Methods include rotation-period estimation and reconstruction validation.
    """

    def __init__(self, diagnostics_dir: str | Path = DEFAULT_DIAGNOSTIC_DIR) -> None:
        self.diagnostics_dir = Path(diagnostics_dir)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def estimate_rotation_period(
        self,
        flux: NDArray[np.float64],
        time: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Estimate stellar rotation period via Lomb-Scargle periodogram.

        Parameters
        ----------
        flux : numpy.ndarray
            Flux array.
        time : numpy.ndarray
            Time array in days.

        Returns
        -------
        tuple[float, float]
            Estimated rotation period in days and confidence metric.
        """
        flux = np.asarray(flux, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)

        if flux.size != time.size or flux.size < 16:
            raise ValueError("Flux and time must have equal length >= 16 for period estimation")

        centered = flux - np.nanmean(flux)
        freq_grid = np.linspace(
            1.0 / ROTATION_PERIOD_MAX_DAYS,
            1.0 / ROTATION_PERIOD_MIN_DAYS,
            5000,
            dtype=np.float64,
        )
        angular_freq = 2.0 * np.pi * freq_grid
        power = lombscargle(time, centered, angular_freq, normalize=True)

        peak_idx = int(np.argmax(power))
        best_freq = float(freq_grid[peak_idx])
        best_period = 1.0 / max(best_freq, 1.0e-8)

        baseline = float(np.nanmedian(power) + 1.0e-12)
        confidence = float(power[peak_idx] / baseline)
        return best_period, confidence

    def decompose(
        self,
        flux: NDArray[np.float64],
        time: NDArray[np.float64],
        cadence_days: float,
        rotation_period_days: float | None = None,
    ) -> DecompositionResult:
        """Decompose a flux sequence into granulation, astero, and starspot bands.

        Parameters
        ----------
        flux : numpy.ndarray
            Flux values.
        time : numpy.ndarray
            Time values in days.
        cadence_days : float
            Sampling cadence in days.
        rotation_period_days : float | None, optional
            Known stellar rotation period. If ``None``, estimate from data.

        Returns
        -------
        DecompositionResult
            Frequency-band arrays and decomposition metadata.
        """
        flux = np.asarray(flux, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)

        if flux.size != time.size:
            raise ValueError("flux and time must have equal lengths")
        if cadence_days <= 0.0:
            raise ValueError("cadence_days must be positive")

        n = flux.size
        mean_flux = float(np.nanmean(flux))
        centered = flux - mean_flux

        fft_values = rfft(centered)
        frequencies = rfftfreq(n, cadence_days)
        nyquist = 1.0 / (2.0 * cadence_days)
        f_astero = min(F_ASTERO_RAW_CYCLES_PER_DAY, nyquist)

        if rotation_period_days is None:
            rotation_period_days, confidence = self.estimate_rotation_period(flux=flux, time=time)
        else:
            confidence = np.nan

        rotation_freq = 1.0 / max(rotation_period_days, 1.0e-6)
        rotation_half_width = max(ROTATION_MIN_HALF_WIDTH_CPD, ROTATION_HALF_WIDTH_FRACTION * rotation_freq)

        # Starspot isolates a narrow rotation-centered band (constraint 5).
        starspot_mask = (
            np.abs(frequencies - rotation_freq) <= rotation_half_width
        ) & (frequencies <= F_GRAN_CYCLES_PER_DAY)

        # Granulation captures low-frequency background excluding the starspot band.
        granulation_mask = (frequencies >= 0.0) & (frequencies <= F_GRAN_CYCLES_PER_DAY) & (~starspot_mask)

        # Asteroseismology captures higher frequencies up to Nyquist-limited boundary.
        astero_mask = (frequencies > F_GRAN_CYCLES_PER_DAY) & (frequencies <= f_astero)

        gran_fft = fft_values * granulation_mask
        astero_fft = fft_values * astero_mask
        starspot_fft = fft_values * starspot_mask

        granulation = irfft(gran_fft, n=n) + mean_flux
        asteroseismology = irfft(astero_fft, n=n)
        starspot = irfft(starspot_fft, n=n)

        metadata: dict[str, float | NDArray[np.float64] | NDArray[np.bool_]] = {
            "f_gran_cpd": float(F_GRAN_CYCLES_PER_DAY),
            "f_astero_cpd": float(f_astero),
            "rotation_period_days": float(rotation_period_days),
            "rotation_confidence": float(confidence),
            "rotation_frequency_cpd": float(rotation_freq),
            "nyquist_cpd": float(nyquist),
            "frequencies": frequencies,
            "granulation_mask": granulation_mask,
            "asteroseismology_mask": astero_mask,
            "starspot_mask": starspot_mask,
        }

        return DecompositionResult(
            granulation=granulation,
            asteroseismology=asteroseismology,
            starspot=starspot,
            metadata=metadata,
        )

    def reconstruct(
        self,
        bands: DecompositionResult | dict[str, NDArray[np.float64]] | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Reconstruct flux by summing decomposed bands.

        Parameters
        ----------
        bands : DecompositionResult | dict | tuple
            Decomposed frequency bands.

        Returns
        -------
        numpy.ndarray
            Reconstructed flux estimate.
        """
        if isinstance(bands, DecompositionResult):
            return bands.granulation + bands.asteroseismology + bands.starspot
        if isinstance(bands, dict):
            return bands["granulation"] + bands["asteroseismology"] + bands["starspot"]
        granulation, asteroseismology, starspot = bands
        return granulation + asteroseismology + starspot

    def plot_decomposition(
        self,
        flux: NDArray[np.float64],
        time: NDArray[np.float64],
        bands: DecompositionResult,
        filename_stem: str = "frequency_decomposition",
    ) -> Path:
        """Generate stacked decomposition plot and save it to diagnostics.

        Parameters
        ----------
        flux : numpy.ndarray
            Original flux.
        time : numpy.ndarray
            Time in days.
        bands : DecompositionResult
            Output from :meth:`decompose`.
        filename_stem : str, optional
            Output file name stem.

        Returns
        -------
        pathlib.Path
            Path to saved PNG figure.
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("MANTA Frequency Decomposition", fontsize=14)

        axes[0].plot(time, flux, color="#1f77b4", linewidth=0.8)
        axes[0].set_ylabel("Flux")
        axes[0].set_title("Original")

        axes[1].plot(time, bands.granulation, color="#2ca02c", linewidth=0.8)
        axes[1].set_ylabel("Flux")
        axes[1].set_title("Granulation Band")

        axes[2].plot(time, bands.asteroseismology, color="#ff7f0e", linewidth=0.8)
        axes[2].set_ylabel("Flux")
        axes[2].set_title("Asteroseismology Band")

        axes[3].plot(time, bands.starspot, color="#d62728", linewidth=0.8)
        axes[3].set_ylabel("Flux")
        axes[3].set_title("Starspot Band")
        axes[3].set_xlabel("Time [days]")

        fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        output_path = self.diagnostics_dir / f"{filename_stem}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        LOGGER.info("Saved frequency decomposition plot to %s", output_path)
        return output_path
