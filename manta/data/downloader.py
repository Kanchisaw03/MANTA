"""Data acquisition utilities for Kepler TCE metadata and light curves.

This module provides resilient download/caching primitives for the MANTA data
pipeline. Robust retrieval is critical because physically grounded preprocessing
and model constraints are only meaningful when upstream data fidelity is high.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

DEFAULT_TCE_CACHE_FILENAME: str = "kepler_dr25_tce_catalog.csv"
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 0.5
SUPPORTED_FLUX_PRIORITY: tuple[str, ...] = ("pdcsap_flux", "sap_flux", "flux")


class DataUnavailableError(RuntimeError):
    """Exception raised when a requested Kepler light curve is unavailable.

    Parameters
    ----------
    kepler_id : int
        Kepler target identifier.
    quarter : int
        Kepler quarter identifier.
    details : str
        Additional context on why data retrieval failed.
    """

    def __init__(self, kepler_id: int, quarter: int, details: str) -> None:
        message = f"Data unavailable for kepler_id={kepler_id}, quarter={quarter}: {details}"
        super().__init__(message)
        self.kepler_id = kepler_id
        self.quarter = quarter
        self.details = details


@dataclass(slots=True)
class DownloadSummary:
    """Summary metrics for a batch download call."""

    downloaded: int
    skipped: int
    failed: int
    failures: list[str]


def _requests_session(retries: int, timeout_seconds: float) -> requests.Session:
    """Build a requests session with retry and timeout-aware transport.

    Parameters
    ----------
    retries : int
        Number of transient retry attempts.
    timeout_seconds : float
        Request timeout in seconds.

    Returns
    -------
    requests.Session
        Configured requests session.
    """
    retry_config = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status_forcelist=(429, 500, 502, 503, 504),
        backoff_factor=DEFAULT_BACKOFF_FACTOR,
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_config)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.request = _wrap_session_request_with_timeout(session.request, timeout_seconds)
    return session


def _wrap_session_request_with_timeout(func: Any, timeout_seconds: float) -> Any:
    """Wrap requests session call to include default timeout."""

    def wrapper(method: str, url: str, **kwargs: Any) -> requests.Response:
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout_seconds
        return func(method, url, **kwargs)

    return wrapper


def download_kepler_tce_catalog(
    cache_dir: str | Path,
    url: str = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+*+from+q1_q17_dr25_tce&format=csv"
    ),
    force_refresh: bool = False,
    retries: int = DEFAULT_RETRIES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Download and cache the Kepler DR25 TCE catalog.

    Parameters
    ----------
    cache_dir : str | Path
        Directory where the catalog CSV is cached.
    url : str, optional
        NASA Exoplanet Archive endpoint for DR25 TCE records.
    force_refresh : bool, optional
        If ``True``, always re-download even if cache exists.
    retries : int, optional
        HTTP retry attempts for transient failures.
    timeout_seconds : float, optional
        HTTP request timeout.

    Returns
    -------
    pandas.DataFrame
        DR25 TCE table.

    Raises
    ------
    RuntimeError
        If catalog download fails after retries.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    catalog_path = cache_path / DEFAULT_TCE_CACHE_FILENAME

    if catalog_path.exists() and not force_refresh:
        LOGGER.info("Loading cached TCE catalog from %s", catalog_path)
        return pd.read_csv(catalog_path)

    LOGGER.info("Downloading Kepler DR25 TCE catalog from NASA Exoplanet Archive")
    session = _requests_session(retries=retries, timeout_seconds=timeout_seconds)
    response = session.get(url)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download TCE catalog. HTTP status={response.status_code}, url={url}"
        )

    catalog_path.write_text(response.text, encoding="utf-8")
    df = pd.read_csv(catalog_path)
    LOGGER.info("Downloaded %d TCE rows to %s", len(df), catalog_path)
    return df


def _import_lightkurve() -> Any:
    """Import lightkurve lazily to keep test/runtime dependencies optional."""
    try:
        import lightkurve as lk  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in environment-specific runs
        raise RuntimeError(
            "lightkurve is required for download_lightcurve but is not available. "
            "Install with: pip install lightkurve"
        ) from exc
    return lk


def _select_flux(lightcurve: Any) -> Any:
    """Select best-available flux column from a LightCurve object.

    Preference order follows Kepler practice: PDCSAP first, then SAP.
    """
    for attr in SUPPORTED_FLUX_PRIORITY:
        if hasattr(lightcurve, attr):
            flux = getattr(lightcurve, attr)
            if flux is not None:
                try:
                    candidate = lightcurve.copy()
                    candidate.flux = flux
                    return candidate
                except Exception:
                    continue
    return lightcurve


def _to_numpy_detached(values: Any, dtype: Any) -> np.ndarray:
    """Convert Quantity-like or masked values to detached NumPy arrays."""
    base = getattr(values, "value", values)
    arr = np.asarray(base)
    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr, np.nan)
    return np.array(arr, dtype=dtype, copy=True)


def _serialize_lightcurve(lightcurve: Any, kepler_id: int, quarter: int) -> dict[str, Any]:
    """Create a pickle-safe, file-detached light-curve payload.

    Lightkurve objects may keep references to FITS internals that are not
    pickle-safe in all environments. Persisting plain arrays avoids this.
    """
    time_values = getattr(lightcurve, "time", None)
    flux_values = getattr(lightcurve, "flux", None)

    if time_values is None or flux_values is None:
        try:
            time_values = lightcurve["time"]
            flux_values = lightcurve["flux"]
        except Exception as exc:  # pragma: no cover - defensive
            raise DataUnavailableError(
                kepler_id=kepler_id,
                quarter=quarter,
                details=f"light curve object missing time/flux fields: {exc}",
            ) from exc

    time = _to_numpy_detached(time_values, np.float64).reshape(-1)
    flux = _to_numpy_detached(flux_values, np.float64).reshape(-1)

    if time.size == 0 or flux.size == 0:
        raise DataUnavailableError(
            kepler_id=kepler_id,
            quarter=quarter,
            details="downloaded light curve has empty time/flux arrays",
        )

    if time.size != flux.size:
        n = min(time.size, flux.size)
        time = time[:n]
        flux = flux[:n]

    return {
        "time": time,
        "flux": flux,
        "meta": {"kepler_id": int(kepler_id), "quarter": int(quarter)},
    }


def _is_valid_cached_payload(payload: Any) -> bool:
    """Check whether a cached payload is usable for downstream processing."""
    if isinstance(payload, dict) and "time" in payload and "flux" in payload:
        try:
            t = np.asarray(payload["time"]).reshape(-1)
            f = np.asarray(payload["flux"]).reshape(-1)
        except Exception:
            return False
        return t.size > 0 and f.size > 0 and t.size == f.size

    # Backward compatibility for older caches storing raw lightkurve objects.
    return hasattr(payload, "time") and hasattr(payload, "flux")


def _cache_lightcurve_path(cache_dir: str | Path, kepler_id: int, quarter: int) -> Path:
    """Build cache path for a specific star-quarter light curve."""
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"kic_{kepler_id}_q{quarter}.pkl"


def download_lightcurve(kepler_id: int, quarter: int, cache_dir: str | Path) -> Any:
    """Download one Kepler light curve quarter with caching.

    Parameters
    ----------
    kepler_id : int
        Kepler target identifier.
    quarter : int
        Quarter number.
    cache_dir : str | Path
        Cache directory for serialized light curves.

    Returns
    -------
    Any
        ``lightkurve.LightCurve`` object.

    Raises
    ------
    DataUnavailableError
        If data cannot be fetched for the target-quarter pair.
    """
    cache_path = _cache_lightcurve_path(cache_dir=cache_dir, kepler_id=kepler_id, quarter=quarter)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if _is_valid_cached_payload(cached):
                LOGGER.debug("Loaded cached light curve for KIC %d Q%d", kepler_id, quarter)
                return cached

            LOGGER.warning(
                "Invalid cached payload for KIC %d Q%d at %s; re-downloading",
                kepler_id,
                quarter,
                cache_path,
            )
        except Exception as exc:
            LOGGER.warning(
                "Failed reading cached light curve for KIC %d Q%d at %s (%s); re-downloading",
                kepler_id,
                quarter,
                cache_path,
                exc,
            )

        try:
            cache_path.unlink()
        except OSError:
            pass

    lk = _import_lightkurve()

    query = f"KIC {kepler_id}"
    search = lk.search_lightcurve(query, mission="Kepler", quarter=quarter, cadence="long")
    if len(search) == 0:
        raise DataUnavailableError(
            kepler_id=kepler_id,
            quarter=quarter,
            details="no matching long-cadence light curve in archive",
        )

    lc = None
    last_download_exc: Exception | None = None
    for _ in range(3):
        try:
            lc = search.download(quality_bitmask="default")
            if lc is not None:
                break
        except Exception as exc:
            last_download_exc = exc

    if lc is None:
        details = "search result returned no downloadable product"
        if last_download_exc is not None:
            details = f"download failed: {last_download_exc}"
        raise DataUnavailableError(kepler_id=kepler_id, quarter=quarter, details=details)

    selected = _select_flux(lc)
    payload = _serialize_lightcurve(selected, kepler_id=kepler_id, quarter=quarter)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.debug("Cached light curve for KIC %d Q%d at %s", kepler_id, quarter, cache_path)
    return payload


def _infer_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Infer a required column name from candidate aliases."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise KeyError(f"None of columns {candidates} found in DataFrame")


def batch_download(
    tce_df: pd.DataFrame,
    cache_dir: str | Path,
    max_workers: int = 8,
) -> DownloadSummary:
    """Download many light curves in parallel with progress reporting.

    Parameters
    ----------
    tce_df : pandas.DataFrame
        Catalog DataFrame with Kepler IDs and quarter metadata.
    cache_dir : str | Path
        Cache directory for downloaded light curves.
    max_workers : int, optional
        ThreadPool worker count.

    Returns
    -------
    DownloadSummary
        Download, skip, and failure counts plus failure messages.
    """
    kepid_col = _infer_column(tce_df, ("kepid", "kepler_id"))
    quarter_col = _infer_column(tce_df, ("quarter", "koi_quarter"))

    unique_pairs = (
        tce_df[[kepid_col, quarter_col]]
        .dropna()
        .drop_duplicates()
        .astype({kepid_col: int, quarter_col: int})
        .itertuples(index=False, name=None)
    )

    pairs = list(unique_pairs)
    skipped = 0
    failures: list[str] = []
    downloaded = 0

    LOGGER.info("Starting batch download for %d star-quarter pairs", len(pairs))

    def _job(kepid: int, quarter: int) -> tuple[int, int, str | None]:
        cache_path = _cache_lightcurve_path(cache_dir=cache_dir, kepler_id=kepid, quarter=quarter)
        if cache_path.exists():
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                if _is_valid_cached_payload(cached):
                    return kepid, quarter, "skipped"

                LOGGER.warning(
                    "Invalid cached payload for KIC %d Q%d at %s; re-downloading",
                    kepid,
                    quarter,
                    cache_path,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Failed to read cached payload for KIC %d Q%d at %s (%s); re-downloading",
                    kepid,
                    quarter,
                    cache_path,
                    exc,
                )

            try:
                cache_path.unlink()
            except OSError:
                pass
        try:
            # Positional call keeps compatibility with monkeypatched test doubles.
            download_lightcurve(kepid, quarter, cache_dir)
        except DataUnavailableError as exc:
            return kepid, quarter, str(exc)
        except Exception as exc:  # pragma: no cover - defensive guard
            return kepid, quarter, f"unexpected error: {exc}"
        return kepid, quarter, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_job, kepid, quarter) for kepid, quarter in pairs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading light curves"):
            kepid, quarter, status = future.result()
            if status == "skipped":
                skipped += 1
                continue
            if status is None:
                downloaded += 1
                continue
            failures.append(f"KIC {kepid} Q{quarter}: {status}")
            LOGGER.warning("Download failed for KIC %d Q%d: %s", kepid, quarter, status)

    summary = DownloadSummary(
        downloaded=downloaded,
        skipped=skipped,
        failed=len(failures),
        failures=failures,
    )
    LOGGER.info(
        "Batch download complete: downloaded=%d skipped=%d failed=%d",
        summary.downloaded,
        summary.skipped,
        summary.failed,
    )
    return summary
