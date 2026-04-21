# MANTA Implementation Plan — Part 1: Overview & Repository Structure

## 1. Why MANTA Is the Right Approach: Analysis of the Core Claim

### 1.1 The Gap in Existing Literature

Every neural network for exoplanet transit detection from 2018–2026 treats architecture as an empirical hyperparameter search:

| System | Year | Architecture Choice Method | Physics Role |
|--------|------|---------------------------|--------------|
| AstroNet (Shallue & Vanderburg) | 2018 | Grid search over CNN depths/widths | None — data generation only |
| ExoMiner | 2021 | Empirical multi-branch CNN | None |
| GPFC | 2024 | CNN after GPU phase-folding | None |
| ExoSpikeNet | 2024 | Spiking NN (bio-inspired, not physics-inspired) | None |

**The fundamental problem**: These architectures have no inductive bias from the physics of transits. A CNN designed for ImageNet classification is structurally identical to one used for transit detection — the only difference is the training data. This means the network must *learn* physics from scratch via gradient descent, wasting capacity and requiring more data.

**MANTA's contribution**: We derive 5 architectural constraints directly from the Mandel-Agol (2002) transit model and stellar physics. The architecture *encodes* physics as structural axioms, not learned parameters. This is analogous to how convolutional weight sharing encodes translation equivariance — we encode transit-specific symmetries and constraints.

### 1.2 Why Each Constraint Is Physically Necessary

**Constraint 1: Position-Aware Attention (from limb darkening)**
- The stellar intensity I(r) = 1 - u₁(1 - μ) - u₂(1 - μ)² where μ = cos(θ) = √(1 - r²)
- Transit depth depends on WHERE on the stellar disk the planet crosses
- A planet crossing the limb (r → 1) produces a shallower transit than one crossing center (r → 0)
- Standard convolutions are translation-invariant: they treat every position identically
- This is physically wrong — the same planet at different impact parameters produces different signals
- **Architectural implication**: Attention weights must depend on position relative to transit center

**Constraint 2: Smooth Non-Monotonic Activations (from elliptic integral geometry)**
- The Mandel-Agol flux ratio involves complete elliptic integrals K(k), E(k), Π(n,k)
- These functions are smooth, bounded, and non-monotonic in their arguments
- ReLU is piecewise linear and monotonic — zero inductive bias for elliptic integral structure
- The ingress/egress flux change is fundamentally non-linear due to circle-circle intersection geometry
- **Architectural implication**: Transit-sensitive layers need Mish/GELU-class activations, not ReLU

**Constraint 3: Constrained Output Layer (from flux ratio physics)**
- Physical flux F(t) ∈ [0, 1] always (star cannot emit negative light or more than its total luminosity)
- Transit depth Rp²/R★² is bounded by physical planet sizes
- Using unconstrained linear output + sigmoid is a soft constraint that can be violated during training
- **Architectural implication**: Hard sigmoid or clamped output, not learned soft bounds

**Constraint 4: Symmetric Weight Constraints (from time-reversal symmetry)**
- For circular orbits, ingress and egress are physically symmetric under t → -t
- The light curve shape during ingress is the mirror of egress
- Standard CNNs learn this symmetry from data, wasting parameters
- **Architectural implication**: Weight-tied or palindromic convolution kernels in the transit encoder

**Constraint 5: Parallel Frequency Streams (from stellar variability physics)**
- Starspots: rotation period P_rot ~ 1-40 days → f ~ 0.025-1 cycles/day
- Granulation: timescale ~ hours → f ~ 1-24 cycles/day  
- p-mode oscillations: ν_max ~ 3000 μHz for solar-like → f ~ 250+ cycles/day (but below Kepler long-cadence Nyquist)
- Transit signal: duration ~ 1-13 hours, period ~ 0.5-400 days
- A single convolutional pathway convolves all these frequency bands together
- **Architectural implication**: Parallel branches with bandpass-specific processing

### 1.3 Why This Hasn't Been Done Before

1. **Community separation**: ML practitioners who build transit classifiers rarely derive architectures from the forward model. Astrophysicists who know Mandel-Agol rarely design neural networks.
2. **"Good enough" syndrome**: AstroNet gets ~96% accuracy, so the community optimizes around it rather than questioning its structural assumptions.
3. **Physics-informed ≠ Physics-derived**: Existing "physics-informed" approaches add physics to the loss function (PINNs). MANTA puts physics into the architecture itself — a fundamentally different approach.

---

## 2. Full Repository Structure

```
manta/
├── README.md                          # Project overview, installation, quickstart
├── LICENSE                            # MIT License
├── pyproject.toml                     # Project metadata, dependencies
├── requirements.txt                   # Pinned dependencies for reproducibility
├── setup.py                           # Package installation
│
├── configs/                           # All hyperparameter configurations
│   ├── __init__.py
│   ├── base.yaml                      # Shared defaults (seed, device, paths)
│   ├── data.yaml                      # Data pipeline config
│   ├── astronet_baseline.yaml         # AstroNet reproduction hyperparameters
│   ├── manta.yaml                     # MANTA full model config
│   └── ablations/                     # One yaml per ablation variant
│       ├── no_position_attention.yaml
│       ├── no_smooth_activation.yaml
│       ├── no_constrained_output.yaml
│       ├── no_symmetric_weights.yaml
│       └── no_parallel_streams.yaml
│
├── data/                              # Data pipeline
│   ├── __init__.py
│   ├── download.py                    # lightkurve-based Kepler download
│   ├── tce_catalog.py                 # TCE table download & parsing
│   ├── preprocess.py                  # NaN handling, normalization, spline fitting
│   ├── frequency_decomposition.py     # FFT-based frequency band separation
│   ├── phase_fold.py                  # Phase folding & binning (global/local views)
│   ├── augmentation.py                # batman-based synthetic transit injection
│   ├── dataset.py                     # PyTorch Dataset & DataLoader
│   └── splits.py                      # Train/val/test splitting logic
│
├── models/                            # All model architectures
│   ├── __init__.py
│   ├── astronet.py                    # AstroNet baseline (PyTorch reimplementation)
│   ├── manta.py                       # Full MANTA model (assembles all components)
│   ├── components/                    # MANTA's 5 physics-derived components
│   │   ├── __init__.py
│   │   ├── position_attention.py      # Constraint 1: Limb-darkening attention
│   │   ├── smooth_activations.py      # Constraint 2: EllipticMish activation
│   │   ├── constrained_output.py      # Constraint 3: Physics-bounded output
│   │   ├── symmetric_encoder.py       # Constraint 4: Time-reversal symmetric conv
│   │   └── parallel_streams.py        # Constraint 5: Frequency-specific branches
│   └── ablations.py                   # Factory for ablation model variants
│
├── training/                          # Training infrastructure
│   ├── __init__.py
│   ├── trainer.py                     # Main training loop with checkpointing
│   ├── losses.py                      # Loss functions (BCE, focal, physics-aware)
│   ├── scheduler.py                   # Learning rate scheduling
│   ├── checkpoint.py                  # Kaggle-aware checkpoint save/resume
│   └── metrics.py                     # AUC-ROC, precision, recall, F1, AP
│
├── evaluation/                        # Evaluation & analysis
│   ├── __init__.py
│   ├── evaluate.py                    # Run evaluation on test set
│   ├── ablation_runner.py             # Automated ablation study execution
│   ├── statistical_tests.py           # McNemar's test, bootstrap CI
│   └── analysis.py                    # Per-class analysis, failure case mining
│
├── visualization/                     # All figure generation for paper
│   ├── __init__.py
│   ├── light_curves.py                # Plot raw/processed light curves
│   ├── architecture_diagram.py        # MANTA architecture figure
│   ├── attention_maps.py              # Visualize position-aware attention
│   ├── frequency_response.py          # Filter bank frequency response plots
│   ├── results_tables.py              # Generate LaTeX tables from results
│   ├── roc_curves.py                  # ROC curve comparison figure
│   └── ablation_plots.py             # Ablation result visualization
│
├── scripts/                           # Entry-point scripts
│   ├── download_data.py               # End-to-end data download
│   ├── preprocess_data.py             # End-to-end preprocessing
│   ├── train_astronet.py              # Train AstroNet baseline
│   ├── train_manta.py                 # Train MANTA
│   ├── run_ablations.py               # Run all ablation experiments
│   ├── evaluate_all.py                # Evaluate all models, generate tables
│   └── generate_figures.py            # Generate all paper figures
│
├── notebooks/                         # Kaggle notebooks (self-contained)
│   ├── 01_data_pipeline.ipynb         # Download + preprocess on Kaggle
│   ├── 02_train_astronet.ipynb        # Train baseline on Kaggle GPU
│   ├── 03_train_manta.ipynb           # Train MANTA on Kaggle GPU
│   ├── 04_ablations.ipynb             # Run ablation studies
│   └── 05_analysis.ipynb              # Generate all results & figures
│
├── paper/                             # LaTeX paper source
│   ├── main.tex                       # Main paper file
│   ├── references.bib                 # Bibliography
│   ├── figures/                       # Generated figures (PNG/PDF)
│   └── tables/                        # Generated LaTeX tables
│
└── tests/                             # Unit tests
    ├── test_data.py                   # Test data pipeline
    ├── test_models.py                 # Test model forward passes
    ├── test_components.py             # Test each MANTA component
    ├── test_losses.py                 # Test loss functions
    └── test_metrics.py                # Test metric computations
```

### 2.1 Key Design Decisions in Repository Layout

1. **`configs/` as YAML**: All hyperparameters externalized. No magic numbers in code. Every experiment is reproducible via its config file.
2. **`components/` separation**: Each physics constraint is an independent PyTorch module. This enables clean ablation — swap a component for its vanilla counterpart.
3. **`notebooks/`**: Self-contained Kaggle notebooks that import from the package. Designed for the Kaggle session limit workflow.
4. **`paper/`**: LaTeX source lives in the repo. Figures are generated programmatically — no manual figure creation.

### 2.2 Dependencies (requirements.txt)

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
lightkurve>=2.4.0
batman-package>=2.4.9
astropy>=5.3
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0
```
