"""Kepler light-curve preprocessing for transit-focused learning.

The preprocessing steps in this module preserve physical transit morphology while
removing artifacts that would otherwise dominate model learning. Gap handling is
explicitly tied to expected transit durations so short gaps can be interpolated
without distorting ingress/egress geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.ndimage import percentile_filter

LOGGER = logging.getLogger(__name__)

# Transit durations are often a few hours; with Kepler 30-minute cadence,
# 10 cadences is roughly 5 hours and typically remains below a local-shape scale.
DEFAULT_SMALL_GAP_CADENCES: int = 10
DEFAULT_SPLINE_WINDOW_CADENCES: int = 97
DEFAULT_CLIP_ITERATIONS: int = 5
DEFAULT_GLOBAL_BINS: int = 2001
DEFAULT_LOCAL_BINS: int = 201


@dataclass(slots=True)
class PreprocessingPipeline:
    """Composable end-to-end preprocessing pipeline.

    Parameters
    ----------
    nan_strategy : str
        Strategy passed to :func:`remove_nans`.
    normalization_method : str
        Method passed to :func:`normalize_flux`.
    sigma_threshold : float
        Iterative sigma-clipping threshold.
    global_bins : int
        Number of bins in full-orbit phase-folded view.
    local_bins : int
        Number of bins in transit-centered local view.
    """

    nan_strategy: str = "hybrid"
    normalization_method: str = "spline"
    sigma_threshold: float = 5.0
    global_bins: int = DEFAULT_GLOBAL_BINS
    local_bins: int = DEFAULT_LOCAL_BINS

    def fit_transform(self, lightcurve: Any) -> dict[str, NDArray[np.float64] | NDArray[np.bool_] | float]:
        """Run the full preprocessing chain and return intermediates.

        Parameters
        ----------
        lightcurve : Any
            Either a lightkurve-like object with ``time`` and ``flux`` attributes
            or a mapping with ``time`` and ``flux`` arrays.

        Returns
        -------
        dict[str, numpy.ndarray | float]
            Dictionary containing raw arrays, cleaned arrays, masks, normalization,
            and final phase-folded local/global representations.

        Raises
        ------
        ValueError
            If required arrays are missing or contain insufficient samples.
        """
        time, flux = _extract_time_flux(lightcurve)
        if time.size < 16:
            raise ValueError("Light curve has fewer than 16 cadences; preprocessing aborted")

        LOGGER.info("Preprocessing started for light curve with %d cadences", time.size)
        cleaned_flux, cleaned_time, validity_mask = remove_nans(flux=flux, time=time, strategy=self.nan_strategy)
        normalized_flux, normalization_curve = normalize_flux(cleaned_flux, method=self.normalization_method)
        clipped_flux, keep_mask, outlier_mask = remove_outliers(
            normalized_flux,
            sigma_threshold=self.sigma_threshold,
        )
        final_time = cleaned_time[keep_mask]

        # Estimate period/t0 heuristically if unavailable: this keeps pipeline usable
        # for exploratory notebooks where explicit ephemerides may not be supplied.
        inferred_period = _infer_period_from_span(final_time)
        inferred_t0 = float(final_time[np.argmin(clipped_flux)])
        inferred_duration_hours = _infer_duration_hours(final_time, clipped_flux)

        folded = phase_fold(
            time=final_time,
            flux=clipped_flux,
            period=inferred_period,
            t0=inferred_t0,
            duration_hours=inferred_duration_hours,
            global_bins=self.global_bins,
            local_bins=self.local_bins,
        )

        result: dict[str, NDArray[np.float64] | NDArray[np.bool_] | float] = {
            "raw_time": time,
            "raw_flux": flux,
            "validity_mask": validity_mask,
            "clean_time": cleaned_time,
            "clean_flux": cleaned_flux,
            "normalization_curve": normalization_curve,
            "normalized_flux": normalized_flux,
            "keep_mask": keep_mask,
            "outlier_mask": outlier_mask,
            "final_time": final_time,
            "final_flux": clipped_flux,
            "period_days": inferred_period,
            "t0_days": inferred_t0,
            "duration_hours": inferred_duration_hours,
            "global_view": folded["global_view"],
            "local_view": folded["local_view"],
            "global_phase": folded["global_phase"],
            "global_flux": folded["global_flux"],
        }
        LOGGER.info(
            "Preprocessing complete: final cadences=%d period=%.4f days duration=%.2f h",
            clipped_flux.size,
            inferred_period,
            inferred_duration_hours,
        )
        return result


def _extract_time_flux(lightcurve: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract time and flux arrays from lightkurve-like objects or mappings."""
    if isinstance(lightcurve, dict):
        if "time" not in lightcurve or "flux" not in lightcurve:
            raise ValueError("Mapping lightcurve must include 'time' and 'flux' keys")
        return np.asarray(lightcurve["time"], dtype=np.float64), np.asarray(lightcurve["flux"], dtype=np.float64)

    if not hasattr(lightcurve, "time") or not hasattr(lightcurve, "flux"):
        raise ValueError("Lightcurve object must expose time and flux attributes")

    time_raw = getattr(lightcurve, "time")
    flux_raw = getattr(lightcurve, "flux")
    time = np.asarray(getattr(time_raw, "value", time_raw), dtype=np.float64)
    flux = np.asarray(getattr(flux_raw, "value", flux_raw), dtype=np.float64)
    return time, flux


def _find_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Find contiguous True runs in a boolean mask."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, mask.size))
    return runs


def _interpolate_gap(
    values: NDArray[np.float64],
    start: int,
    end: int,
    support: int = 20,
) -> NDArray[np.float64]:
    """Interpolate a contiguous missing interval with a local cubic spline."""
    left = max(0, start - support)
    right = min(values.size, end + support)
    x = np.arange(values.size, dtype=np.float64)

    support_idx = np.r_[left:start, end:right]
    support_idx = support_idx[np.isfinite(values[support_idx])]
    if support_idx.size < 4:
        # Fallback to linear interpolation with nearest finite values.
        finite_idx = np.where(np.isfinite(values))[0]
        if finite_idx.size < 2:
            raise ValueError("Cannot interpolate gap: insufficient finite support points")
        return np.interp(x[start:end], finite_idx, values[finite_idx])

    spline = CubicSpline(x[support_idx], values[support_idx], extrapolate=True)
    return spline(x[start:end])


def remove_nans(
    flux: NDArray[np.float64],
    time: NDArray[np.float64],
    strategy: str = "hybrid",
    max_gap_cadences: int = DEFAULT_SMALL_GAP_CADENCES,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Remove or interpolate NaNs with transit-scale-aware gap logic.

    Parameters
    ----------
    flux : numpy.ndarray
        Flux series.
    time : numpy.ndarray
        Time series in days.
    strategy : str, optional
        One of ``interpolate``, ``mask``, ``hybrid``.
    max_gap_cadences : int, optional
        Maximum contiguous NaN run length to interpolate.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Cleaned flux, cleaned time, and boolean validity mask on original arrays.

    Notes
    -----
    The hybrid strategy interpolates short gaps (<10 cadences, about 5 hours) and
    masks larger gaps because long interpolation can erase ingress/egress physics.
    """
    if strategy not in {"interpolate", "mask", "hybrid"}:
        raise ValueError("strategy must be one of: interpolate, mask, hybrid")
    if flux.shape != time.shape:
        raise ValueError("flux and time must have the same shape")

    finite_time_mask = np.isfinite(time)
    working_flux = np.asarray(flux, dtype=np.float64).copy()
    valid_mask = finite_time_mask.copy()

    nan_mask = ~np.isfinite(working_flux)
    if strategy == "mask":
        valid_mask &= ~nan_mask
        return working_flux[valid_mask], time[valid_mask], valid_mask

    runs = _find_runs(nan_mask)
    for start, end in runs:
        length = end - start
        is_small_gap = length <= max_gap_cadences
        if is_small_gap and strategy in {"interpolate", "hybrid"}:
            try:
                interpolated = _interpolate_gap(working_flux, start=start, end=end)
                working_flux[start:end] = interpolated
            except ValueError:
                valid_mask[start:end] = False
        else:
            valid_mask[start:end] = False

    valid_mask &= np.isfinite(working_flux)
    cleaned_flux = working_flux[valid_mask]
    cleaned_time = time[valid_mask]
    return cleaned_flux, cleaned_time, valid_mask


def normalize_flux(
    flux: NDArray[np.float64],
    method: str = "spline",
    spline_window_cadences: int = DEFAULT_SPLINE_WINDOW_CADENCES,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize flux while preserving transit dips for detection.

    Parameters
    ----------
    flux : numpy.ndarray
        Input flux values.
    method : str, optional
        ``spline`` or ``median``.
    spline_window_cadences : int, optional
        Window size for robust baseline extraction.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Normalized flux and estimated normalization curve.
    """
    if method not in {"spline", "median"}:
        raise ValueError("method must be one of: spline, median")

    flux = np.asarray(flux, dtype=np.float64)
    if flux.size < 8:
        baseline = np.full_like(flux, np.nanmedian(flux))
        return flux / np.clip(baseline, 1.0e-12, None), baseline

    if method == "median":
        baseline = np.full_like(flux, np.nanmedian(flux))
        normalized = flux / np.clip(baseline, 1.0e-12, None)
        return normalized, baseline

    # Robust high-percentile baseline reduces contamination by downward transit dips.
    baseline_seed = percentile_filter(
        flux,
        percentile=85,
        size=spline_window_cadences,
        mode="nearest",
    )
    x = np.arange(flux.size, dtype=np.float64)

    try:
        sample_step = max(1, spline_window_cadences // 4)
        knot_x = x[::sample_step]
        knot_y = baseline_seed[::sample_step]
        spline = UnivariateSpline(knot_x, knot_y, k=3, s=max(1.0, 0.5 * knot_x.size))
        baseline = spline(x)
    except Exception:
        baseline = baseline_seed

    baseline = np.clip(baseline, 1.0e-6, None)
    normalized = flux / baseline
    return normalized, baseline


def remove_outliers(
    flux: NDArray[np.float64],
    sigma_threshold: float = 5.0,
    thruster_mask: NDArray[np.bool_] | None = None,
    max_iterations: int = DEFAULT_CLIP_ITERATIONS,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Iteratively sigma-clip outliers while preserving thruster-adjacent cadences.

    Parameters
    ----------
    flux : numpy.ndarray
        Flux values to clip.
    sigma_threshold : float, optional
        Sigma threshold per iteration.
    thruster_mask : numpy.ndarray | None, optional
        Boolean mask where ``True`` marks known thruster-firing cadences that
        should be flagged but retained for downstream auditability.
    max_iterations : int, optional
        Maximum clipping iterations.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Clipped flux, keep mask, and outlier mask on original indices.
    """
    flux = np.asarray(flux, dtype=np.float64)
    keep_mask = np.ones(flux.size, dtype=bool)
    outlier_mask = np.zeros(flux.size, dtype=bool)
    protect_mask = np.zeros(flux.size, dtype=bool) if thruster_mask is None else np.asarray(thruster_mask, dtype=bool)

    for _ in range(max_iterations):
        active = flux[keep_mask]
        median = np.nanmedian(active)
        mad = np.nanmedian(np.abs(active - median))
        robust_sigma = 1.4826 * mad
        if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
            break

        residual = np.abs(flux - median)
        candidate = residual > sigma_threshold * robust_sigma
        outlier_mask |= candidate

        removable = candidate & (~protect_mask)
        if not np.any(removable & keep_mask):
            break
        keep_mask &= ~removable

    return flux[keep_mask], keep_mask, outlier_mask


def _bin_flux(phase: NDArray[np.float64], flux: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    """Median-bin phase-folded flux onto a fixed grid."""
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(n_bins, np.nan, dtype=np.float64)

    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    for i in range(n_bins):
        mask = idx == i
        if np.any(mask):
            values[i] = np.nanmedian(flux[mask])

    missing = ~np.isfinite(values)
    if np.any(missing):
        finite = ~missing
        if np.any(finite):
            values[missing] = np.interp(centers[missing], centers[finite], values[finite])
        else:
            values[:] = 1.0
    return values


def phase_fold(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
    global_bins: int = DEFAULT_GLOBAL_BINS,
    local_bins: int = DEFAULT_LOCAL_BINS,
) -> dict[str, NDArray[np.float64]]:
    """Create global and local phase-folded views following AstroNet conventions.

    Parameters
    ----------
    time : numpy.ndarray
        Time in days.
    flux : numpy.ndarray
        Normalized flux.
    period : float
        Orbital period in days.
    t0 : float
        Transit center time in days.
    duration_hours : float
        Transit duration in hours.
    global_bins : int, optional
        Number of bins for full-orbit view.
    local_bins : int, optional
        Number of bins for transit-centered view.

    Returns
    -------
    dict[str, numpy.ndarray]
        Global/local views and sorted folded arrays.
    """
    if period <= 0.0:
        raise ValueError("period must be positive")
    if duration_hours <= 0.0:
        raise ValueError("duration_hours must be positive")

    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux[order]

    global_view = _bin_flux(phase_sorted, flux_sorted, n_bins=global_bins)

    duration_days = duration_hours / 24.0
    # Local window total width is 2x duration, centered at transit.
    half_width_phase = duration_days / period
    local_mask = np.abs(phase_sorted) <= half_width_phase
    if np.any(local_mask):
        local_phase = phase_sorted[local_mask] / max(half_width_phase, 1.0e-12) * 0.5
        local_flux = flux_sorted[local_mask]
        local_view = _bin_flux(local_phase, local_flux, n_bins=local_bins)
    else:
        local_view = np.ones(local_bins, dtype=np.float64)

    return {
        "global_view": global_view,
        "local_view": local_view,
        "global_phase": phase_sorted,
        "global_flux": flux_sorted,
    }


def _infer_period_from_span(time: NDArray[np.float64]) -> float:
    """Infer a conservative placeholder period from time coverage."""
    span = float(np.max(time) - np.min(time))
    if span <= 0.0:
        return 1.0
    return max(1.0, min(30.0, span / 8.0))


def _infer_duration_hours(time: NDArray[np.float64], flux: NDArray[np.float64]) -> float:
    """Estimate an approximate transit duration for local-window extraction."""
    cadence_days = np.nanmedian(np.diff(time)) if time.size > 1 else (30.0 / (24.0 * 60.0))
    cadence_hours = max(cadence_days * 24.0, 0.25)

    depth_threshold = np.nanmedian(flux) - 0.5 * np.nanstd(flux)
    in_transit = flux < depth_threshold
    longest = 0
    current = 0
    for flag in in_transit:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    if longest == 0:
        return max(2.0, 4.0 * cadence_hours)
    return max(1.0, float(longest) * cadence_hours)
