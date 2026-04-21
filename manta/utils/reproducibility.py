"""Reproducibility utilities for deterministic MANTA experiments.

This module centralizes random seed handling and run fingerprint creation so
experiments can be reproduced across machines and accelerator backends.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np
import torch

_ENV_SEED_KEY: str = "MANTA_SEED"
_ENV_CONFIG_KEY: str = "MANTA_CONFIG_JSON"
_DEFAULT_SEED: int = 42


def _to_serializable_config(config: Any) -> dict[str, Any]:
    """Convert a config object into a JSON-serializable dictionary.

    Parameters
    ----------
    config : Any
        Mapping-like object, dataclass instance, or object with ``__dict__``.

    Returns
    -------
    dict[str, Any]
        Dictionary representation suitable for stable hashing.
    """
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {"value": str(config)}


def set_all_seeds(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds for deterministic behavior.

    Parameters
    ----------
    seed : int
        Seed value used by pseudo-random number generators.

    Notes
    -----
    This function also sets deterministic CuDNN flags to improve run-to-run
    reproducibility, at the cost of potential runtime slowdown.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic kernels where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ[_ENV_SEED_KEY] = str(seed)


def get_run_hash(config: Any | None = None, seed: int | None = None) -> str:
    """Compute a stable hash for experiment tracking.

    Parameters
    ----------
    config : Any | None, optional
        Configuration object. If ``None``, environment-backed defaults are used.
    seed : int | None, optional
        Explicit seed override. If ``None``, the function uses the ``MANTA_SEED``
        environment variable or a fallback constant.

    Returns
    -------
    str
        SHA-256 hash truncated to 16 hexadecimal characters.
    """
    resolved_seed: int = seed if seed is not None else int(os.environ.get(_ENV_SEED_KEY, _DEFAULT_SEED))

    if config is None:
        raw_json: str = os.environ.get(_ENV_CONFIG_KEY, "{}")
        try:
            config_payload = json.loads(raw_json)
        except json.JSONDecodeError:
            config_payload = {"raw": raw_json}
    else:
        config_payload = _to_serializable_config(config)

    payload = {
        "seed": resolved_seed,
        "config": config_payload,
    }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:16]
