"""Typed configuration system for MANTA.

This module defines strongly-typed experiment configuration dataclasses and
YAML load/save helpers with validation to enforce physically meaningful and
reproducible training setups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Physical and mission constants used as default configuration values.
KEPLER_LONG_CADENCE_MINUTES: float = 29.4244
DEFAULT_GRANULATION_HOURS: float = 8.0
DEFAULT_ASTERO_MINUTES: float = 5.0
DEFAULT_MIN_TRANSIT_PERIOD_DAYS: float = 1.0
DEFAULT_MAX_TRANSIT_PERIOD_DAYS: float = 200.0


@dataclass(slots=True)
class DataConfig:
    """Configuration for data download and preprocessing.

    Parameters
    ----------
    tce_catalog_url : str
        NASA Exoplanet Archive API URL for Kepler DR25 TCE catalog.
    cache_dir : str
        Local directory where raw and cached files are stored.
    processed_dir : str
        Directory where preprocessed arrays are written.
    diagnostics_dir : str
        Directory for diagnostic plots and intermediate visualizations.
    max_download_workers : int
        Thread count for parallel light-curve download.
    download_retries : int
        Number of retry attempts per HTTP request.
    timeout_seconds : float
        HTTP timeout for remote queries.
    nan_strategy : str
        NaN handling strategy: interpolate, mask, or hybrid.
    normalization_method : str
        Flux normalization method: spline or median.
    sigma_clip_threshold : float
        Sigma threshold for iterative outlier clipping.
    global_view_bins : int
        Number of bins for global phase-folded view.
    local_view_bins : int
        Number of bins for local phase-folded view.
    kepler_cadence_days : float
        Kepler long-cadence sampling in days.
    """

    tce_catalog_url: str = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+*+from+q1_q17_dr25_tce&format=csv"
    )
    cache_dir: str = "data/cache"
    processed_dir: str = "data/processed"
    diagnostics_dir: str = "outputs/diagnostics"
    max_download_workers: int = 8
    download_retries: int = 3
    timeout_seconds: float = 30.0
    nan_strategy: str = "hybrid"
    normalization_method: str = "spline"
    sigma_clip_threshold: float = 5.0
    global_view_bins: int = 2001
    local_view_bins: int = 201
    kepler_cadence_days: float = KEPLER_LONG_CADENCE_MINUTES / (60.0 * 24.0)


@dataclass(slots=True)
class AugmentationConfig:
    """Configuration for synthetic transit injection and augmentation."""

    use_augmentation: bool = True
    n_augmentations_per_positive: int = 2
    rp_rs_min: float = 0.01
    rp_rs_max: float = 0.2
    impact_parameter_min: float = 0.0
    impact_parameter_max: float = 0.9
    period_min_days: float = DEFAULT_MIN_TRANSIT_PERIOD_DAYS
    period_max_days: float = DEFAULT_MAX_TRANSIT_PERIOD_DAYS


@dataclass(slots=True)
class ModelConfig:
    """Configuration for MANTA architecture modules.

    Notes
    -----
    Defaults follow the physics-derived architecture plan: position-aware
    attention, symmetric encoder, parallel frequency branches, and
    constrained output head.
    """

    d_model: int = 128
    n_heads: int = 4
    stellar_radius_bins: int = 64
    granulation_channels: int = 32
    astero_channels: int = 32
    starspot_channels: int = 32
    symmetric_base_channels: int = 32
    symmetric_kernel_sizes: tuple[int, int, int] = (9, 7, 5)
    local_input_length: int = 201
    global_input_length: int = 2001


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for optimization, scheduling, and checkpointing."""

    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    max_epochs: int = 80
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    min_lr: float = 1.0e-6
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    checkpoint_dir: str = "checkpoints"
    emergency_checkpoint_minutes_left: float = 30.0


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for evaluation thresholds and uncertainty estimates."""

    threshold: float = 0.5
    bootstrap_samples: int = 1000
    bootstrap_ci: float = 0.95


@dataclass(slots=True)
class MANTAConfig:
    """Top-level typed configuration object for MANTA experiments.

    Parameters
    ----------
    seed : int
        Random seed used across Python, NumPy, and Torch.
    device : str
        Preferred Torch device (cpu, cuda, mps).
    data : DataConfig
        Data download and preprocessing settings.
    augmentation : AugmentationConfig
        Transit injection and data augmentation settings.
    model : ModelConfig
        MANTA architecture hyperparameters.
    training : TrainingConfig
        Training-loop and optimizer settings.
    evaluation : EvaluationConfig
        Validation and reporting settings.
    output_dir : str
        Root output directory for artifacts.
    """

    seed: int = 42
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "outputs"


def _validate_config(config: MANTAConfig) -> None:
    """Validate a loaded configuration.

    Raises
    ------
    ValueError
        If any field violates physical or implementation constraints.
    """
    if config.model.d_model % config.model.n_heads != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads")
    if config.data.nan_strategy not in {"interpolate", "mask", "hybrid"}:
        raise ValueError("data.nan_strategy must be one of: interpolate, mask, hybrid")
    if config.data.normalization_method not in {"spline", "median"}:
        raise ValueError("data.normalization_method must be one of: spline, median")
    if config.training.warmup_epochs < 0:
        raise ValueError("training.warmup_epochs must be non-negative")
    if not (0.0 < config.evaluation.threshold < 1.0):
        raise ValueError("evaluation.threshold must be in (0, 1)")
    if not (0.0 < config.augmentation.rp_rs_min < config.augmentation.rp_rs_max <= 0.2):
        raise ValueError("augmentation Rp/Rs range must satisfy 0 < min < max <= 0.2")
    if not (0.0 <= config.augmentation.impact_parameter_min < config.augmentation.impact_parameter_max <= 0.9):
        raise ValueError("augmentation impact parameter range must satisfy 0 <= min < max <= 0.9")


def _build_dataclass(data: dict[str, Any], cls: type[Any]) -> Any:
    """Instantiate a dataclass from partial dictionary values."""
    kwargs: dict[str, Any] = {}
    for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
        if field_name in data:
            kwargs[field_name] = data[field_name]
    return cls(**kwargs)


def load_config(path: str | Path) -> MANTAConfig:
    """Load configuration from YAML file and validate values.

    Parameters
    ----------
    path : str | Path
        Path to YAML configuration file.

    Returns
    -------
    MANTAConfig
        Parsed and validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If YAML parsing fails or validation errors are found.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    data_cfg = _build_dataclass(payload.get("data", {}), DataConfig)
    aug_cfg = _build_dataclass(payload.get("augmentation", {}), AugmentationConfig)
    model_cfg = _build_dataclass(payload.get("model", {}), ModelConfig)
    train_cfg = _build_dataclass(payload.get("training", {}), TrainingConfig)
    eval_cfg = _build_dataclass(payload.get("evaluation", {}), EvaluationConfig)

    top_level = {
        "seed": payload.get("seed", 42),
        "device": payload.get("device", "cuda"),
        "output_dir": payload.get("output_dir", "outputs"),
        "data": data_cfg,
        "augmentation": aug_cfg,
        "model": model_cfg,
        "training": train_cfg,
        "evaluation": eval_cfg,
    }

    config = MANTAConfig(**top_level)
    _validate_config(config)
    return config


def save_config(config: MANTAConfig, path: str | Path) -> None:
    """Save configuration to YAML with explanatory comments.

    Parameters
    ----------
    config : MANTAConfig
        Configuration object to save.
    path : str | Path
        Output YAML path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# MANTA configuration file",
        "# This file stores all experiment parameters required for reproducibility.",
        "",
        f"seed: {config.seed}  # Global random seed",
        f"device: {config.device}  # Preferred compute device",
        f"output_dir: {config.output_dir}  # Root output directory",
        "",
        "data:",
        "  # NASA Exoplanet Archive source for Kepler DR25 TCE records",
        f"  tce_catalog_url: {config.data.tce_catalog_url}",
        f"  cache_dir: {config.data.cache_dir}  # Raw/cached data path",
        f"  processed_dir: {config.data.processed_dir}  # Preprocessed arrays path",
        f"  diagnostics_dir: {config.data.diagnostics_dir}  # Diagnostic plots path",
        f"  max_download_workers: {config.data.max_download_workers}",
        f"  download_retries: {config.data.download_retries}",
        f"  timeout_seconds: {config.data.timeout_seconds}",
        f"  nan_strategy: {config.data.nan_strategy}",
        f"  normalization_method: {config.data.normalization_method}",
        f"  sigma_clip_threshold: {config.data.sigma_clip_threshold}",
        f"  global_view_bins: {config.data.global_view_bins}",
        f"  local_view_bins: {config.data.local_view_bins}",
        f"  kepler_cadence_days: {config.data.kepler_cadence_days}",
        "",
        "augmentation:",
        f"  use_augmentation: {str(config.augmentation.use_augmentation).lower()}",
        f"  n_augmentations_per_positive: {config.augmentation.n_augmentations_per_positive}",
        f"  rp_rs_min: {config.augmentation.rp_rs_min}",
        f"  rp_rs_max: {config.augmentation.rp_rs_max}",
        f"  impact_parameter_min: {config.augmentation.impact_parameter_min}",
        f"  impact_parameter_max: {config.augmentation.impact_parameter_max}",
        f"  period_min_days: {config.augmentation.period_min_days}",
        f"  period_max_days: {config.augmentation.period_max_days}",
        "",
        "model:",
        f"  d_model: {config.model.d_model}",
        f"  n_heads: {config.model.n_heads}",
        f"  stellar_radius_bins: {config.model.stellar_radius_bins}",
        f"  granulation_channels: {config.model.granulation_channels}",
        f"  astero_channels: {config.model.astero_channels}",
        f"  starspot_channels: {config.model.starspot_channels}",
        f"  symmetric_base_channels: {config.model.symmetric_base_channels}",
        f"  symmetric_kernel_sizes: [{', '.join(str(v) for v in config.model.symmetric_kernel_sizes)}]",
        f"  local_input_length: {config.model.local_input_length}",
        f"  global_input_length: {config.model.global_input_length}",
        "",
        "training:",
        f"  batch_size: {config.training.batch_size}",
        f"  num_workers: {config.training.num_workers}",
        f"  learning_rate: {config.training.learning_rate}",
        f"  weight_decay: {config.training.weight_decay}",
        f"  max_epochs: {config.training.max_epochs}",
        f"  grad_clip_norm: {config.training.grad_clip_norm}",
        f"  warmup_epochs: {config.training.warmup_epochs}",
        f"  early_stopping_patience: {config.training.early_stopping_patience}",
        f"  min_lr: {config.training.min_lr}",
        f"  focal_gamma: {config.training.focal_gamma}",
        f"  focal_alpha: {config.training.focal_alpha}",
        f"  checkpoint_dir: {config.training.checkpoint_dir}",
        f"  emergency_checkpoint_minutes_left: {config.training.emergency_checkpoint_minutes_left}",
        "",
        "evaluation:",
        f"  threshold: {config.evaluation.threshold}",
        f"  bootstrap_samples: {config.evaluation.bootstrap_samples}",
        f"  bootstrap_ci: {config.evaluation.bootstrap_ci}",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


def config_to_dict(config: MANTAConfig) -> dict[str, Any]:
    """Convert the dataclass configuration to nested dictionaries."""
    return asdict(config)
