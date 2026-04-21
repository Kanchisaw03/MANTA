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
        with cache_path.open("rb") as handle:
            LOGGER.debug("Loaded cached light curve for KIC %d Q%d", kepler_id, quarter)
            return pickle.load(handle)

    lk = _import_lightkurve()

    query = f"KIC {kepler_id}"
    search = lk.search_lightcurve(query, mission="Kepler", quarter=quarter, cadence="long")
    if len(search) == 0:
        raise DataUnavailableError(
            kepler_id=kepler_id,
            quarter=quarter,
            details="no matching long-cadence light curve in archive",
        )

    try:
        lc = search.download(quality_bitmask="default")
    except Exception as exc:
        raise DataUnavailableError(
            kepler_id=kepler_id,
            quarter=quarter,
            details=f"download failed: {exc}",
        ) from exc

    if lc is None:
        raise DataUnavailableError(
            kepler_id=kepler_id,
            quarter=quarter,
            details="search result returned no downloadable product",
        )

    selected = _select_flux(lc)
    with cache_path.open("wb") as handle:
        pickle.dump(selected, handle)
    LOGGER.debug("Cached light curve for KIC %d Q%d at %s", kepler_id, quarter, cache_path)
    return selected


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
            return kepid, quarter, "skipped"
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
