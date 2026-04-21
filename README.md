# MANTA: Mandel-Agol Neural Transit Architecture

MANTA is a physics-derived deep-learning system for exoplanet transit detection.
Unlike standard architectures tuned by trial-and-error, MANTA encodes transit
physics directly into its neural components.

## Why This Matters (Plain-Language)

When a planet passes in front of a star, the star does not dim uniformly. The
stellar edge is darker than the center (limb darkening), transit edges are
smooth due to geometry, and stellar noise appears in distinct time-frequency
bands. MANTA builds those facts into the architecture itself.

## Physics-Derived Constraints Implemented

1. Limb darkening -> position-aware attention.
2. Elliptic ingress/egress geometry -> smooth non-monotonic activation.
3. Physical flux/probability bounds -> hard-constrained output.
4. Ingress/egress reversibility -> palindromic symmetric convolution.
5. Stellar variability bands -> parallel frequency-specialized branches.

## Repository Layout

- manta/: Python package
- configs/: YAML experiment configs
- scripts/: CLI entry points
- tests/: pytest suite
- notebooks/: research/analysis notebooks
- paper/: LaTeX paper skeleton
- docs/: planning documents

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quickstart

```bash
python scripts/download_data.py --config configs/manta_default.yaml
python scripts/train_manta.py --config configs/manta_default.yaml
python scripts/train_astronet.py --config configs/astronet_baseline.yaml
python scripts/run_ablation.py --config configs/manta_default.yaml --n-seeds 3
python scripts/evaluate.py --config configs/manta_default.yaml --model manta --checkpoint checkpoints/checkpoint_best_epoch_001.pt
```

## Reproducibility

- Global seed control: manta/utils/reproducibility.py
- Config serialization and validation: manta/utils/config.py
- Run hash computed from config + seed for artifact tracking.

## Testing

```bash
pytest -q
```

The test suite uses synthetic data and avoids network calls.

## Citation

```bibtex
@misc{manta2026,
  title={MANTA: Mandel-Agol Neural Transit Architecture},
  author={Your Name and Collaborators},
  year={2026},
  eprint={to-be-assigned},
  archivePrefix={arXiv},
  primaryClass={astro-ph.EP}
}
```

## License

This project is released under the MIT License. See LICENSE for details.
