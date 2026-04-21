# MANTA Implementation Plan — Part 2: Complete Data Pipeline

## 3. Data Pipeline (Complete)

### 3.1 Dataset Selection & Labels

**Source**: Kepler Q1-Q17 DR24 TCE table (matching Shallue & Vanderburg 2018 for fair comparison).

**Label scheme** (from `av_training_set` column):
- **PC** (Planet Candidate) → label = 1 (positive class)
- **AFP** (Astrophysical False Positive) → label = 0
- **NTP** (Non-Transiting Phenomenon) → label = 0
- **UNK** (Unknown) → **excluded** from training/evaluation

**Dataset statistics**:
- Total TCEs in DR24: ~34,032
- After removing rogues and UNK: ~15,000 usable TCEs
- PC: ~3,500 | AFP+NTP: ~11,500
- **Class imbalance ratio**: approximately 1:3.3 (planets : non-planets)

### 3.2 TCE Catalog Download & Parsing

```python
# data/tce_catalog.py

import pandas as pd
import os

TCE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nph-nstedAPI"
    "?table=q1_q17_dr24_tce"
    "&select=rowid,kepid,tce_plnt_num,tce_period,tce_time0bk,"
    "tce_duration,av_training_set"
    "&format=csv"
)

def download_tce_catalog(save_path: str) -> pd.DataFrame:
    """Download DR24 TCE table from NASA Exoplanet Archive."""
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    
    df = pd.read_csv(TCE_URL, comment='#')
    
    # Filter out UNK labels
    df = df[df['av_training_set'].isin(['PC', 'AFP', 'NTP'])].copy()
    
    # Create binary label
    df['label'] = (df['av_training_set'] == 'PC').astype(int)
    
    # Convert duration from hours to days for consistency
    df['tce_duration_days'] = df['tce_duration'] / 24.0
    
    df.to_csv(save_path, index=False)
    return df
```

### 3.3 Light Curve Download via lightkurve

```python
# data/download.py

import lightkurve as lk
import numpy as np
import os
import pickle
from tqdm import tqdm

def download_kepler_lightcurve(kepid: int, cache_dir: str) -> dict:
    """
    Download all quarters of Kepler long-cadence data for a single target.
    
    Returns dict with keys:
        'time': list of numpy arrays (one per quarter)
        'flux': list of numpy arrays (PDCSAP flux, one per quarter)
        'flux_err': list of numpy arrays
    """
    cache_file = os.path.join(cache_dir, f"kic_{kepid}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Search for all Kepler long-cadence data
    search_result = lk.search_lightcurve(
        f'KIC {kepid}',
        mission='Kepler',
        cadence='long'  # 29.4-minute cadence
    )
    
    if len(search_result) == 0:
        return None
    
    # Download all quarters
    lc_collection = search_result.download_all(quality_bitmask='default')
    
    all_time = []
    all_flux = []
    all_flux_err = []
    
    for lc in lc_collection:
        # Use PDCSAP flux (Pre-search Data Conditioning)
        # This removes instrumental systematics while preserving astrophysical signals
        time = lc.time.value.astype(np.float64)
        flux = lc.flux.value.astype(np.float64)
        flux_err = lc.flux_err.value.astype(np.float64)
        
        all_time.append(time)
        all_flux.append(flux)
        all_flux_err.append(flux_err)
    
    result = {
        'time': all_time,
        'flux': all_flux,
        'flux_err': all_flux_err,
        'kepid': kepid
    }
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result


def download_all_targets(tce_df, cache_dir: str):
    """Download light curves for all unique Kepler IDs in the TCE table."""
    unique_kepids = tce_df['kepid'].unique()
    print(f"Downloading {len(unique_kepids)} unique Kepler targets...")
    
    failed = []
    for kepid in tqdm(unique_kepids):
        try:
            result = download_kepler_lightcurve(kepid, cache_dir)
            if result is None:
                failed.append(kepid)
        except Exception as e:
            print(f"Failed KIC {kepid}: {e}")
            failed.append(kepid)
    
    print(f"Downloaded {len(unique_kepids) - len(failed)}/{len(unique_kepids)} targets")
    if failed:
        pd.Series(failed).to_csv(
            os.path.join(cache_dir, 'failed_downloads.csv'), index=False
        )
```

### 3.4 NaN Handling Strategy (Kepler-Specific)

Kepler data has specific NaN/gap patterns:
1. **Quarterly gaps**: ~1 day gaps between quarters (spacecraft rotation)
2. **Safe mode events**: Multi-day gaps from anomalies
3. **Cosmic ray hits**: Isolated NaN values
4. **Saturation**: Bright star overflow

```python
# data/preprocess.py

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter

def handle_nans(time: np.ndarray, flux: np.ndarray, 
                flux_err: np.ndarray) -> tuple:
    """
    Handle NaN values in Kepler light curves.
    
    Strategy:
    1. Remove points where flux is NaN (these are flagged bad data)
    2. Remove points with flux_err = 0 (indicates missing error estimates)
    3. Remove 5-sigma outliers (cosmic ray hits, instrumental glitches)
    4. Do NOT interpolate across gaps > 0.5 days (preserves data integrity)
    """
    # Step 1: Remove NaN flux values
    valid = np.isfinite(flux) & np.isfinite(time)
    if flux_err is not None:
        valid &= np.isfinite(flux_err) & (flux_err > 0)
    
    time = time[valid]
    flux = flux[valid]
    flux_err = flux_err[valid] if flux_err is not None else None
    
    # Step 2: Remove 5-sigma outliers using rolling median
    if len(flux) > 20:
        median_flux = median_filter(flux, size=13)
        residuals = flux - median_flux
        sigma = np.std(residuals)
        inlier = np.abs(residuals) < 5 * sigma
        
        time = time[inlier]
        flux = flux[inlier]
        flux_err = flux_err[inlier] if flux_err is not None else None
    
    return time, flux, flux_err


def fit_normalization_spline(time: np.ndarray, flux: np.ndarray,
                              period: float, duration: float, 
                              t0: float) -> np.ndarray:
    """
    Fit a spline to remove low-frequency stellar variability.
    
    Follows Shallue & Vanderburg 2018:
    1. Mask out transit windows (1.5x transit duration centered on each transit)
    2. Fit a cubic spline with breakpoints every 1.5 days
    3. Divide flux by spline to flatten the baseline
    
    This preserves the transit signal while removing starspot modulation.
    """
    # Calculate transit mid-times
    num_transits = int((time[-1] - time[0]) / period) + 2
    transit_times = t0 + np.arange(-1, num_transits + 1) * period
    
    # Mask transit windows (1.5x duration on each side)
    mask_half_width = 1.5 * duration  # in days
    transit_mask = np.ones(len(time), dtype=bool)
    for tt in transit_times:
        transit_mask &= np.abs(time - tt) > mask_half_width
    
    if np.sum(transit_mask) < 10:
        # Not enough out-of-transit points, return median normalization
        return np.full_like(flux, np.median(flux))
    
    # Fit cubic spline to out-of-transit points
    # Breakpoint spacing: 1.5 days (captures starspot evolution)
    time_range = time[-1] - time[0]
    n_knots = max(2, int(time_range / 1.5))
    
    try:
        spline = UnivariateSpline(
            time[transit_mask], flux[transit_mask],
            k=3,  # cubic
            s=len(time[transit_mask])  # smoothing factor
        )
        spline_flux = spline(time)
    except Exception:
        spline_flux = np.full_like(flux, np.median(flux))
    
    # Safety: prevent division by values close to zero
    spline_flux = np.clip(spline_flux, np.median(flux) * 0.5, None)
    
    return spline_flux


def normalize_lightcurve(time: np.ndarray, flux: np.ndarray,
                          period: float, duration: float,
                          t0: float) -> np.ndarray:
    """Normalize flux by dividing by fitted spline."""
    spline = fit_normalization_spline(time, flux, period, duration, t0)
    return flux / spline
```

### 3.5 Frequency Decomposition (MANTA-Specific)

This is a **novel preprocessing step** not present in AstroNet. We separate the light curve into distinct frequency bands based on stellar physics.

```python
# data/frequency_decomposition.py

import numpy as np
from scipy.fft import rfft, irfft, rfftfreq

# Frequency band definitions based on stellar physics literature
# All frequencies in cycles/day
FREQUENCY_BANDS = {
    'rotation': {
        'description': 'Starspot/magnetic activity modulation',
        'f_low': 0.0,           # DC
        'f_high': 0.5,          # ~2 day period lower bound
        'physics': 'P_rot ~ 2-40 days for solar-type stars'
    },
    'granulation': {
        'description': 'Convective granulation background',
        'f_low': 0.5,           # ~2 day period
        'f_high': 24.0,         # ~1 hour period  
        'physics': 'Granulation timescale ~ hours, Harvey model'
    },
    'transit': {
        'description': 'Transit signal band (broadband)',
        'f_low': 0.1,           # Long-period planets
        'f_high': 48.0,         # Short transits (~30 min)
        'physics': 'Transit durations 0.5-13 hours'
    },
    'oscillation': {
        'description': 'p-mode / asteroseismic oscillations',
        'f_low': 24.0,          # ~1 hour period
        'f_high': None,         # Up to Nyquist (~24.47 c/d for long cadence)
        'physics': 'nu_max ~ 3000 μHz for Sun, but aliased in long cadence'
    }
}

# Kepler long cadence: 29.4244 minutes = 0.020433 days
KEPLER_LONG_CADENCE_DAYS = 29.4244 / (60 * 24)
KEPLER_NYQUIST_CPD = 1.0 / (2 * KEPLER_LONG_CADENCE_DAYS)  # ~24.47 c/d


def decompose_frequency_bands(time: np.ndarray, flux: np.ndarray,
                                bands: dict = None) -> dict:
    """
    Decompose a light curve into frequency bands using FFT bandpass filtering.
    
    Args:
        time: Time array (days, must be uniformly sampled or interpolated)
        flux: Normalized flux array
        bands: Dict of frequency band definitions
    
    Returns:
        Dict mapping band name to filtered flux array
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    
    n = len(flux)
    
    # Compute sampling rate from time array
    dt = np.median(np.diff(time))  # days
    
    # Compute FFT
    flux_fft = rfft(flux - np.mean(flux))  # remove DC before FFT
    freqs = rfftfreq(n, d=dt)  # frequencies in cycles/day
    
    result = {}
    for band_name, band_def in bands.items():
        f_low = band_def['f_low']
        f_high = band_def['f_high']
        if f_high is None:
            f_high = freqs[-1]  # Nyquist
        
        # Create bandpass mask with smooth (Tukey) edges to reduce ringing
        mask = np.zeros(len(freqs))
        band_width = f_high - f_low
        transition = 0.05 * band_width  # 5% transition band
        
        for i, f in enumerate(freqs):
            if f_low + transition <= f <= f_high - transition:
                mask[i] = 1.0
            elif f_low <= f < f_low + transition and transition > 0:
                mask[i] = 0.5 * (1 - np.cos(np.pi * (f - f_low) / transition))
            elif f_high - transition < f <= f_high and transition > 0:
                mask[i] = 0.5 * (1 + np.cos(np.pi * (f - f_high + transition) / transition))
        
        # Apply bandpass and inverse FFT
        filtered_fft = flux_fft * mask
        filtered_flux = irfft(filtered_fft, n=n)
        
        result[band_name] = filtered_flux
    
    # Add back DC component to the rotation band (lowest frequency)
    result['rotation'] += np.mean(flux)
    
    return result


def interpolate_to_uniform(time: np.ndarray, flux: np.ndarray,
                            cadence: float = KEPLER_LONG_CADENCE_DAYS
                            ) -> tuple:
    """
    Interpolate light curve to uniform time sampling for FFT.
    
    Gaps > 2 cadences are filled with Gaussian noise matching local statistics.
    This is ONLY for frequency decomposition input — the original non-uniform
    data is used for all other processing.
    """
    t_uniform = np.arange(time[0], time[-1], cadence)
    flux_uniform = np.interp(t_uniform, time, flux)
    
    # Identify gap regions (where interpolation spans > 2 cadences)
    gap_threshold = 2 * cadence
    for i in range(len(time) - 1):
        if time[i + 1] - time[i] > gap_threshold:
            # Fill gap region with noise
            gap_mask = (t_uniform > time[i]) & (t_uniform < time[i + 1])
            local_std = np.std(flux[max(0, i-50):i+50])
            flux_uniform[gap_mask] = 1.0 + np.random.normal(0, local_std, 
                                                              np.sum(gap_mask))
    
    return t_uniform, flux_uniform
```

### 3.6 Phase Folding & Binning

```python
# data/phase_fold.py

import numpy as np

def phase_fold(time: np.ndarray, flux: np.ndarray,
               period: float, t0: float) -> tuple:
    """
    Phase-fold the light curve at the given period and epoch.
    
    Returns:
        phase: Array of phases in [-0.5, 0.5], centered on transit
        folded_flux: Corresponding flux values
    """
    phase = ((time - t0) % period) / period
    # Center on transit: shift so transit is at phase = 0
    phase[phase > 0.5] -= 1.0
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def bin_phase_folded(phase: np.ndarray, flux: np.ndarray,
                      num_bins: int) -> np.ndarray:
    """
    Bin phase-folded light curve into uniform phase bins.
    
    Uses median binning (robust to outliers).
    Empty bins filled with linear interpolation from neighbors.
    
    Args:
        phase: Sorted phase array in [-0.5, 0.5]
        flux: Corresponding flux values
        num_bins: Number of output bins
    
    Returns:
        binned_flux: Array of shape (num_bins,)
    """
    bin_edges = np.linspace(-0.5, 0.5, num_bins + 1)
    binned_flux = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    bin_indices = np.digitize(phase, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    for i in range(num_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            binned_flux[i] = np.median(flux[mask])
            bin_counts[i] = np.sum(mask)
        else:
            binned_flux[i] = np.nan
    
    # Fill empty bins with linear interpolation
    nans = np.isnan(binned_flux)
    if np.any(nans) and not np.all(nans):
        x = np.arange(num_bins)
        binned_flux[nans] = np.interp(x[nans], x[~nans], binned_flux[~nans])
    elif np.all(nans):
        binned_flux[:] = 1.0  # Default to flat if no data
    
    return binned_flux


def generate_views(time: np.ndarray, flux: np.ndarray,
                    period: float, t0: float, duration: float,
                    global_bins: int = 2001,
                    local_bins: int = 201) -> dict:
    """
    Generate global and local views of a TCE (matching AstroNet format).
    
    Global view: Entire orbital period, 2001 bins
    Local view: ±2x transit duration, 201 bins, higher resolution
    
    Returns:
        dict with 'global_view' and 'local_view' numpy arrays
    """
    phase, folded_flux = phase_fold(time, flux, period, t0)
    
    # Global view: full phase range
    global_view = bin_phase_folded(phase, folded_flux, global_bins)
    
    # Local view: zoom in on transit
    # Width: 4x duration (2x on each side), as fraction of period
    local_width = min(4.0 * duration / period, 0.5)
    local_mask = np.abs(phase) <= local_width / 2
    
    if np.sum(local_mask) > 0:
        local_phase = phase[local_mask]
        local_flux = folded_flux[local_mask]
        # Rescale to [-0.5, 0.5] for binning
        local_phase_rescaled = local_phase / local_width
        local_view = bin_phase_folded(local_phase_rescaled, local_flux, local_bins)
    else:
        local_view = np.ones(local_bins)
    
    return {
        'global_view': global_view.astype(np.float32),
        'local_view': local_view.astype(np.float32)
    }
```

### 3.7 Synthetic Data Augmentation

```python
# data/augmentation.py

import numpy as np
import batman

def generate_synthetic_transit(time: np.ndarray,
                                rp_rs: float = None,
                                period: float = None,
                                a_rs: float = None,
                                inc: float = None,
                                u: list = None) -> np.ndarray:
    """
    Generate a synthetic transit light curve using batman (Mandel-Agol model).
    
    If parameters are None, draw from physically motivated distributions:
    - rp_rs (Rp/R*): LogUniform(0.005, 0.2) — Earth-size to hot Jupiter
    - period: LogUniform(0.5, 400) days
    - a_rs (a/R*): Derived from period via Kepler's third law
    - inc: Uniform(85, 90) degrees (must transit)
    - u: Quadratic LD coefficients from Claret 2011 tables
    """
    if rp_rs is None:
        rp_rs = np.exp(np.random.uniform(np.log(0.005), np.log(0.2)))
    if period is None:
        period = np.exp(np.random.uniform(np.log(0.5), np.log(400)))
    if a_rs is None:
        # Kepler's 3rd law approximation for Sun-like star:
        # a/R* ≈ 4.2 * (P/days)^(2/3)
        a_rs = 4.2 * period**(2/3)
    if inc is None:
        # Must be high enough to transit: cos(i) < (1 + rp_rs) / a_rs
        max_cos_i = (1 + rp_rs) / a_rs
        cos_i = np.random.uniform(0, min(max_cos_i, 1.0))
        inc = np.degrees(np.arccos(cos_i))
    if u is None:
        # Typical quadratic LD coefficients for Sun-like star (Kepler band)
        u = [0.4, 0.26]
    
    params = batman.TransitParams()
    params.t0 = np.mean(time)
    params.per = period
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = u
    params.limb_dark = "quadratic"
    
    m = batman.TransitModel(params, time)
    flux = m.light_curve(params)
    
    return flux


def inject_transit(time: np.ndarray, flux: np.ndarray,
                    snr_target: float = None) -> tuple:
    """
    Inject a synthetic transit into a real non-transiting light curve.
    
    This creates realistic positive examples by:
    1. Taking a real stellar light curve (with real noise properties)
    2. Multiplying by a synthetic transit model
    3. Optionally scaling to achieve a target SNR
    
    Returns:
        augmented_flux: Original flux × transit model
        transit_params: Dict of injected transit parameters
    """
    # Generate random transit parameters
    rp_rs = np.exp(np.random.uniform(np.log(0.005), np.log(0.15)))
    period = np.exp(np.random.uniform(np.log(0.5), np.log(200)))
    a_rs = 4.2 * period**(2/3)
    max_cos_i = min((1 + rp_rs) / a_rs, 1.0)
    inc = np.degrees(np.arccos(np.random.uniform(0, max_cos_i)))
    
    transit_flux = generate_synthetic_transit(
        time, rp_rs=rp_rs, period=period, a_rs=a_rs, inc=inc
    )
    
    # Multiplicative injection
    augmented_flux = flux * transit_flux
    
    # Calculate approximate transit depth and duration for view generation
    depth = rp_rs**2
    # Duration approximation (circular orbit):
    # T ≈ (P/π) * arcsin(√((1+rp_rs)² - b²) / (a_rs * sin(i)))
    b = a_rs * np.cos(np.radians(inc))
    if (1 + rp_rs)**2 - b**2 > 0:
        duration_days = (period / np.pi) * np.arcsin(
            np.sqrt((1 + rp_rs)**2 - b**2) / (a_rs * np.sin(np.radians(inc)))
        )
    else:
        duration_days = 0.1  # fallback
    
    params = {
        'rp_rs': rp_rs, 'period': period, 'a_rs': a_rs,
        'inc': inc, 'depth': depth, 'duration_days': duration_days,
        't0': np.mean(time)
    }
    
    return augmented_flux, params
```

### 3.8 Train/Val/Test Split Strategy

```python
# data/splits.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def create_splits(tce_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Create train/val/test splits matching AstroNet methodology.
    
    Key constraints:
    1. Split by Kepler ID (kepid), NOT by TCE
       - All TCEs from the same star must be in the same split
       - Prevents data leakage (same star in train and test)
    2. Stratified by label to maintain class balance
    3. Ratio: 80% train / 10% val / 10% test
    
    Returns DataFrame with added 'split' column.
    """
    rng = np.random.RandomState(seed)
    
    # Get unique stars with their majority label for stratification
    star_labels = tce_df.groupby('kepid')['label'].max().reset_index()
    star_labels.columns = ['kepid', 'has_planet']
    
    kepids = star_labels['kepid'].values
    labels = star_labels['has_planet'].values
    n_stars = len(kepids)
    
    # Shuffle
    shuffle_idx = rng.permutation(n_stars)
    kepids = kepids[shuffle_idx]
    labels = labels[shuffle_idx]
    
    # Stratified split
    n_test = int(0.1 * n_stars)
    n_val = int(0.1 * n_stars)
    
    # Separate positive and negative stars for stratification
    pos_stars = kepids[labels == 1]
    neg_stars = kepids[labels == 0]
    
    rng.shuffle(pos_stars)
    rng.shuffle(neg_stars)
    
    # Allocate proportionally
    n_pos_test = max(1, int(0.1 * len(pos_stars)))
    n_pos_val = max(1, int(0.1 * len(pos_stars)))
    n_neg_test = max(1, int(0.1 * len(neg_stars)))
    n_neg_val = max(1, int(0.1 * len(neg_stars)))
    
    test_stars = set(pos_stars[:n_pos_test]) | set(neg_stars[:n_neg_test])
    val_stars = set(pos_stars[n_pos_test:n_pos_test+n_pos_val]) | \
                set(neg_stars[n_neg_test:n_neg_test+n_neg_val])
    train_stars = set(kepids) - test_stars - val_stars
    
    # Assign splits to TCE DataFrame
    def assign_split(kepid):
        if kepid in test_stars:
            return 'test'
        elif kepid in val_stars:
            return 'val'
        else:
            return 'train'
    
    tce_df = tce_df.copy()
    tce_df['split'] = tce_df['kepid'].apply(assign_split)
    
    # Print statistics
    for split in ['train', 'val', 'test']:
        subset = tce_df[tce_df['split'] == split]
        n_pos = subset['label'].sum()
        n_neg = len(subset) - n_pos
        print(f"{split}: {len(subset)} TCEs ({n_pos} PC, {n_neg} non-PC), "
              f"{subset['kepid'].nunique()} unique stars")
    
    return tce_df
```

### 3.9 PyTorch Dataset & DataLoader

```python
# data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
import pickle

class KeplerTCEDataset(Dataset):
    """
    PyTorch Dataset for Kepler TCE classification.
    
    Each item returns:
        global_view: Tensor of shape (1, 2001) — global phase-folded view
        local_view:  Tensor of shape (1, 201)  — local phase-folded view
        freq_bands:  Tensor of shape (3, 2001) — 3 frequency band decompositions
                     (rotation, granulation, transit-band) of global view
        label:       Tensor scalar (0 or 1)
    """
    
    def __init__(self, tce_df, processed_dir: str, 
                 use_freq_decomp: bool = True,
                 augment: bool = False):
        """
        Args:
            tce_df: DataFrame with columns [kepid, tce_plnt_num, label, ...]
            processed_dir: Directory containing preprocessed .npz files
            use_freq_decomp: Whether to load frequency decomposition (MANTA only)
            augment: Whether to apply data augmentation (training only)
        """
        self.tce_df = tce_df.reset_index(drop=True)
        self.processed_dir = processed_dir
        self.use_freq_decomp = use_freq_decomp
        self.augment = augment
    
    def __len__(self):
        return len(self.tce_df)
    
    def __getitem__(self, idx):
        row = self.tce_df.iloc[idx]
        kepid = row['kepid']
        plnt_num = row['tce_plnt_num']
        label = row['label']
        
        # Load preprocessed data
        filename = f"kic_{kepid}_plnt_{plnt_num}.npz"
        filepath = os.path.join(self.processed_dir, filename)
        data = np.load(filepath)
        
        global_view = data['global_view'].astype(np.float32)  # (2001,)
        local_view = data['local_view'].astype(np.float32)    # (201,)
        
        # Data augmentation for training
        if self.augment:
            # Gaussian noise injection (SNR-aware)
            noise_level = np.random.uniform(0, 0.001)
            global_view += np.random.normal(0, noise_level, global_view.shape)
            local_view += np.random.normal(0, noise_level, local_view.shape)
            
            # Random time-reversal (valid for circular orbits)
            if np.random.random() < 0.5:
                global_view = global_view[::-1].copy()
                local_view = local_view[::-1].copy()
        
        # Add channel dimension: (2001,) -> (1, 2001)
        global_view = torch.tensor(global_view).unsqueeze(0)
        local_view = torch.tensor(local_view).unsqueeze(0)
        
        result = {
            'global_view': global_view,
            'local_view': local_view,
            'label': torch.tensor(label, dtype=torch.float32)
        }
        
        # Frequency decomposition (MANTA only)
        if self.use_freq_decomp and 'freq_rotation' in data:
            freq_bands = np.stack([
                data['freq_rotation'],
                data['freq_granulation'],
                data['freq_transit']
            ], axis=0).astype(np.float32)  # (3, 2001)
            result['freq_bands'] = torch.tensor(freq_bands)
        
        return result


def create_dataloaders(tce_df, processed_dir: str,
                        batch_size: int = 64,
                        use_freq_decomp: bool = True,
                        num_workers: int = 4) -> dict:
    """
    Create train/val/test DataLoaders with class-balanced sampling.
    
    Uses WeightedRandomSampler for training to handle class imbalance.
    Batch size 64 chosen for T4 GPU (16GB VRAM): 
    - MANTA model ~5M params → ~20MB
    - Batch of 64 × (2001 + 201 + 3×2001) floats → ~2MB
    - Activations/gradients → ~4GB
    - Total: well within 16GB
    """
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        split_df = tce_df[tce_df['split'] == split]
        dataset = KeplerTCEDataset(
            split_df, processed_dir,
            use_freq_decomp=use_freq_decomp,
            augment=(split == 'train')
        )
        
        if split == 'train':
            # Weighted random sampling for class imbalance
            labels = split_df['label'].values
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True
            )
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True, drop_last=True
            )
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    
    return loaders
```

### 3.10 Normalization Strategy

All views are normalized to have the out-of-transit baseline at approximately 1.0:

1. **Per-segment normalization**: Each Kepler quarter is divided by its median flux before stitching
2. **Spline normalization**: Low-frequency variability removed by dividing by fitted spline
3. **Global/local view normalization**: After phase-folding, subtract 1.0 and divide by the standard deviation of the out-of-transit region, so the baseline is ~0 and noise is ~1
4. **No batch normalization in transit-sensitive layers**: BatchNorm would destroy the physical flux scale that Constraint 3 (bounded output) relies on. We use LayerNorm instead.
