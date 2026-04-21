# MANTA Implementation Plan — Part 4: Training, Ablations, Experiments & Paper

## 6. Training Protocol

### 6.1 Loss Function

```python
# training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross-Entropy Loss.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Why Focal Loss instead of standard BCE:
    1. Kepler dataset is imbalanced (1:3.3 planet:non-planet)
    2. Standard BCE gives equal weight to easy negatives (obvious eclipsing binaries)
       and hard positives (shallow transits of small planets)
    3. Focal loss down-weights easy examples (high p_t) and focuses on hard examples
    4. γ = 2.0 works well empirically for similar imbalance ratios (Lin et al. 2017)
    
    Physics justification: The most scientifically valuable detections are small
    planets with shallow transits (low SNR). Focal loss naturally prioritizes these.
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha   # Weight for positive class
        self.gamma = gamma   # Focusing parameter
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, 1) predictions in [0, 1]
            target: (batch, 1) binary labels
        """
        pred = pred.clamp(1e-7, 1 - 1e-7)  # Numerical stability
        
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce
        return loss.mean()
```

### 6.2 Training Configuration

```yaml
# configs/manta.yaml
model:
  name: MANTA
  d_model: 128
  n_heads: 4
  stream_channels: 32
  sym_base: 16

training:
  epochs: 80
  batch_size: 64                 # Fits comfortably on T4 (16GB)
  
  optimizer: adamw
  learning_rate: 3.0e-4          # Higher than AstroNet due to LayerNorm + attention
  weight_decay: 0.01             # L2 regularization for attention weights
  betas: [0.9, 0.999]
  
  # Learning rate schedule: Cosine annealing with linear warmup
  scheduler: cosine_warmup
  warmup_epochs: 5               # Linear warmup for 5 epochs
  min_lr: 1.0e-6                 # Minimum LR at end of cosine decay
  
  # Loss
  loss: focal_bce
  focal_alpha: 0.75              # Upweight planets (minority class)
  focal_gamma: 2.0               # Focus on hard examples
  
  # Regularization
  dropout: 0.2                   # In feature extractors
  attention_dropout: 0.1         # In attention layers
  label_smoothing: 0.05          # Prevents overconfident predictions
  
  # Early stopping
  early_stopping_patience: 10
  early_stopping_metric: auc_roc
  
  # Gradient clipping (important for attention models)
  max_grad_norm: 1.0
  
  # Checkpointing for Kaggle
  checkpoint_every_n_epochs: 1
  keep_top_k_checkpoints: 3
  save_on_interrupt: true        # Save checkpoint on KeyboardInterrupt
```

### 6.3 Checkpoint Strategy for Kaggle Session Limits

```python
# training/checkpoint.py

import torch
import os
import json
import signal
import sys

class KaggleCheckpointManager:
    """
    Checkpoint manager designed for Kaggle's session limits.
    
    Kaggle constraints:
    - GPU sessions: ~9-12 hours maximum
    - Persistent storage: /kaggle/working/ (up to 20GB)
    - Session can be killed without warning
    
    Strategy:
    1. Save checkpoint every epoch to /kaggle/working/checkpoints/
    2. Save "resume state" including epoch, optimizer state, scheduler state
    3. Register SIGTERM handler to save checkpoint on session termination
    4. Auto-detect and resume from latest checkpoint on restart
    5. Keep only top-K checkpoints by validation metric to save space
    """
    
    def __init__(self, save_dir: str, keep_top_k: int = 3):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_checkpoints = []  # List of (metric_value, filepath) tuples
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._emergency_save)
    
    def save(self, model, optimizer, scheduler, epoch: int,
             metrics: dict, is_best: bool = False):
        """Save a training checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
        }
        
        filepath = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch:03d}.pt')
        torch.save(state, filepath)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(state, best_path)
        
        # Prune old checkpoints
        metric_val = metrics.get('auc_roc', 0)
        self.best_checkpoints.append((metric_val, filepath))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        while len(self.best_checkpoints) > self.keep_top_k:
            _, old_path = self.best_checkpoints.pop()
            if os.path.exists(old_path):
                os.remove(old_path)
        
        # Save resume metadata
        meta = {
            'last_epoch': epoch,
            'last_checkpoint': filepath,
            'best_metric': self.best_checkpoints[0][0] if self.best_checkpoints else 0
        }
        with open(os.path.join(self.save_dir, 'resume_meta.json'), 'w') as f:
            json.dump(meta, f)
    
    def load_latest(self, model, optimizer=None, scheduler=None):
        """Load the most recent checkpoint for resuming training."""
        meta_path = os.path.join(self.save_dir, 'resume_meta.json')
        if not os.path.exists(meta_path):
            return 0  # No checkpoint found, start from epoch 0
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        checkpoint_path = meta['last_checkpoint']
        if not os.path.exists(checkpoint_path):
            # Try best model
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pt')
        
        state = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if scheduler and state.get('scheduler_state_dict'):
            scheduler.load_state_dict(state['scheduler_state_dict'])
        
        print(f"Resumed from epoch {state['epoch']}, "
              f"metrics: {state.get('metrics', {})}")
        return state['epoch'] + 1
    
    def _emergency_save(self, signum, frame):
        """Emergency save on SIGTERM (Kaggle session kill)."""
        print(f"\nReceived signal {signum}. Emergency checkpoint save...")
        # This requires access to the model/optimizer — stored as attributes
        if hasattr(self, '_current_state'):
            state = self._current_state
            emergency_path = os.path.join(self.save_dir, 'emergency_checkpoint.pt')
            torch.save(state, emergency_path)
            print(f"Emergency checkpoint saved to {emergency_path}")
        sys.exit(0)
```

### 6.4 Main Training Loop

```python
# training/trainer.py (key excerpt)

def train_epoch(model, loader, optimizer, criterion, scheduler, 
                device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        optimizer.zero_grad()
        
        global_view = batch['global_view'].to(device)
        local_view = batch['local_view'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        freq_bands = batch.get('freq_bands')
        if freq_bands is not None:
            freq_bands = freq_bands.to(device)
        
        preds = model(global_view, local_view, freq_bands=freq_bands)
        loss = criterion(preds, labels)
        
        loss.backward()
        
        # Gradient clipping (critical for attention models)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    return total_loss / len(loader), all_preds, all_labels
```

---

## 7. Ablation Study Design

### 7.1 Ablation Matrix

Each ablation removes ONE physics-derived component and replaces it with the vanilla (AstroNet-equivalent) alternative:

| Ablation ID | Component Removed | Replacement | What it Tests |
|-------------|-------------------|-------------|---------------|
| `A1` | Position-Aware Attention | Standard self-attention (no positional bias) | Does limb-darkening encoding help? |
| `A2` | EllipticMish activation | ReLU (standard in AstroNet) | Do smooth non-monotonic activations help? |
| `A3` | Constrained output | Standard Sigmoid | Do hard physical bounds improve calibration? |
| `A4` | Symmetric encoder | Standard Conv1d (no palindromic constraint) | Does enforced time-reversal help? |
| `A5` | Parallel frequency streams | Single convolutional pathway | Does frequency separation help? |
| `A_all` | All components removed | ≈ AstroNet with attention | Is the improvement from physics or just more parameters? |

### 7.2 Ablation Implementation

```python
# models/ablations.py

from models.manta import MANTA
from models.astronet import AstroNet
import torch.nn as nn

def create_ablation_model(ablation_id: str, **kwargs) -> nn.Module:
    """Factory function for ablation model variants."""
    
    if ablation_id == 'full':
        return MANTA(**kwargs)
    
    elif ablation_id == 'A1_no_position':
        # Replace LimbDarkeningPositionalEncoding with standard sinusoidal
        model = MANTA(**kwargs)
        # Swap out the positional encoding in both attention modules
        from models.components.position_attention import PositionAwareAttention
        model.global_attention = nn.TransformerEncoderLayer(
            d_model=kwargs.get('d_model', 128),
            nhead=kwargs.get('n_heads', 4),
            batch_first=True
        )
        model.local_attention = nn.TransformerEncoderLayer(
            d_model=kwargs.get('d_model', 128),
            nhead=kwargs.get('n_heads', 4),
            batch_first=True
        )
        return model
    
    elif ablation_id == 'A2_no_smooth_act':
        # Replace EllipticMish with ReLU everywhere
        model = MANTA(**kwargs)
        _replace_activations(model, nn.ReLU())
        return model
    
    elif ablation_id == 'A3_no_constrained_output':
        # Replace PhysicsConstrainedOutput with Linear + Sigmoid
        model = MANTA(**kwargs)
        d = kwargs.get('d_model', 128) // 2
        model.output = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        return model
    
    elif ablation_id == 'A4_no_symmetric':
        # Replace PalindromicConv1d with standard Conv1d
        model = MANTA(**kwargs)
        _replace_palindromic_convs(model)
        return model
    
    elif ablation_id == 'A5_no_parallel_streams':
        # Replace ParallelFrequencyProcessor with single conv pathway
        model = MANTA(**kwargs)
        sc = kwargs.get('stream_channels', 32)
        model.freq_processor = nn.Sequential(
            nn.Conv1d(1, sc, kernel_size=7, padding='same'),
            nn.LayerNorm([sc]),
            nn.GELU(),
            nn.Conv1d(sc, sc * 2, kernel_size=5, padding='same'),
            nn.LayerNorm([sc * 2]),
            nn.GELU(),
        )
        model.freq_processor.output_channels = sc * 2
        model.global_proj = nn.Conv1d(sc * 2, kwargs.get('d_model', 128), 1)
        return model
    
    elif ablation_id == 'baseline':
        return AstroNet()
    
    else:
        raise ValueError(f"Unknown ablation: {ablation_id}")


def _replace_activations(module, new_activation):
    """Recursively replace EllipticMish with a different activation."""
    from models.components.smooth_activations import EllipticMish
    for name, child in module.named_children():
        if isinstance(child, EllipticMish):
            setattr(module, name, new_activation)
        else:
            _replace_activations(child, new_activation)


def _replace_palindromic_convs(module):
    """Replace PalindromicConv1d with standard Conv1d."""
    from models.components.symmetric_encoder import PalindromicConv1d
    for name, child in module.named_children():
        if isinstance(child, PalindromicConv1d):
            replacement = nn.Conv1d(
                child.in_channels, child.out_channels, 
                child.kernel_size, padding='same'
            )
            setattr(module, name, replacement)
        else:
            _replace_palindromic_convs(child)
```

### 7.3 How to Present Ablation Results

**Table format for paper (Table 3)**:

| Model | AUC-ROC | Precision | Recall | F1 | AP | ΔAUC vs Full |
|-------|---------|-----------|--------|----|----|-------------|
| MANTA (Full) | X.XXX ± Y | ... | ... | ... | ... | — |
| − Position Attention (A1) | ... | ... | ... | ... | ... | -Z.ZZZ |
| − Smooth Activation (A2) | ... | ... | ... | ... | ... | -Z.ZZZ |
| − Constrained Output (A3) | ... | ... | ... | ... | ... | -Z.ZZZ |
| − Symmetric Encoder (A4) | ... | ... | ... | ... | ... | -Z.ZZZ |
| − Parallel Streams (A5) | ... | ... | ... | ... | ... | -Z.ZZZ |
| − All Physics (A_all) | ... | ... | ... | ... | ... | -Z.ZZZ |
| AstroNet (Baseline) | ... | ... | ... | ... | ... | -Z.ZZZ |

**Key metric**: AUC-ROC degradation when removing each component. If removing component X drops AUC-ROC significantly (p < 0.05 by McNemar's test), that proves component X contributes independently.

---

## 8. Experiments and Results Tables

### 8.1 Main Results Table (Table 1 in paper)

| Model | Params | AUC-ROC | Precision | Recall | F1 | AP | Train Time |
|-------|--------|---------|-----------|--------|----|----|-----------|
| AstroNet (S&V 2018) | 1.1M | 0.XXX±Y | ... | ... | ... | ... | Xh |
| MANTA | 4.8M | 0.XXX±Y | ... | ... | ... | ... | Xh |

### 8.2 Statistical Testing

```python
# evaluation/statistical_tests.py

import numpy as np
from scipy import stats

def mcnemar_test(y_true, pred_a, pred_b, threshold=0.5):
    """
    McNemar's test for comparing two classifiers.
    
    Tests whether the two models make significantly different errors.
    Returns chi-squared statistic and p-value.
    """
    a_correct = (pred_a >= threshold) == y_true
    b_correct = (pred_b >= threshold) == y_true
    
    # Count discordant pairs
    b01 = np.sum(a_correct & ~b_correct)  # A right, B wrong
    b10 = np.sum(~a_correct & b_correct)  # A wrong, B right
    
    if b01 + b10 == 0:
        return 0, 1.0
    
    chi2 = (abs(b01 - b10) - 1)**2 / (b01 + b10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return chi2, p_value


def bootstrap_confidence_interval(y_true, y_pred, metric_fn,
                                    n_bootstrap=1000, ci=0.95):
    """
    Bootstrap confidence interval for any metric.
    
    Returns (mean, lower, upper) of the metric across bootstrap samples.
    """
    n = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)
    
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))
    
    return np.mean(scores), lower, upper
```

### 8.3 Number of Runs

- **5 runs** with different random seeds for each model variant
- Report mean ± std for all metrics
- Use seed values: [42, 123, 456, 789, 1024]
- Total training runs: (1 baseline + 1 MANTA + 6 ablations) × 5 seeds = **40 runs**
- Estimated time per run on T4: ~2-4 hours → ~80-160 GPU hours total
- Kaggle strategy: 4 runs per 9-hour session → ~10 sessions

---

## 9. Paper Structure

### 9.1 Section Outline

```
Title: MANTA: Physics-Derived Neural Architecture for Exoplanet Transit Detection

Abstract: ~250 words with placeholders for metrics

1. Introduction (1.5 pages)
   - Exoplanet transit detection problem
   - Current ML approaches and their limitation (empirical architecture search)
   - Our contribution: first physics-derived architecture
   - Summary of results

2. Background and Related Work (1.5 pages)
   2.1 The Mandel-Agol Transit Model
   2.2 Neural Networks for Transit Detection
       - AstroNet (Shallue & Vanderburg 2018)
       - ExoMiner (Valizadegan et al. 2021)
       - GPFC (Kunimoto et al. 2024)
       - ExoSpikeNet (2024)
   2.3 Physics-Informed Neural Networks (PINNs) — distinguish from MANTA

3. Physics-Derived Architectural Constraints (3 pages) [KEY SECTION]
   3.1 From Limb Darkening to Position-Aware Attention
   3.2 From Elliptic Integrals to Smooth Activations
   3.3 From Flux Bounds to Constrained Output
   3.4 From Time-Reversal Symmetry to Symmetric Encoding
   3.5 From Stellar Variability to Parallel Processing

4. MANTA Architecture (2 pages)
   4.1 Overall Architecture
   4.2 Implementation Details
   4.3 Comparison with AstroNet

5. Experimental Setup (1.5 pages)
   5.1 Dataset: Kepler DR24 TCEs
   5.2 Preprocessing and Data Augmentation
   5.3 Training Protocol
   5.4 Evaluation Metrics

6. Results (2 pages)
   6.1 Main Comparison with AstroNet
   6.2 Ablation Study
   6.3 Analysis of Physics-Derived Components
   
7. Discussion (1 page)
   7.1 When Does Physics-Derived Architecture Help Most?
   7.2 Limitations
   7.3 Implications for Other Scientific Domains

8. Conclusion (0.5 pages)

References

Appendix A: Mandel-Agol Derivation Details
Appendix B: Hyperparameter Sensitivity
Appendix C: Additional Results
```

### 9.2 Figures to Generate

| Figure # | Description | Script |
|----------|-------------|--------|
| Fig. 1 | Example Kepler light curve showing transit, starspots, and noise | `visualization/light_curves.py` |
| Fig. 2 | MANTA architecture diagram showing all 5 components | `visualization/architecture_diagram.py` |
| Fig. 3 | Physics-to-architecture mapping: 5 panels showing each constraint | `visualization/architecture_diagram.py` |
| Fig. 4 | Frequency band decomposition of example light curve | `visualization/frequency_response.py` |
| Fig. 5 | ROC curve comparison: MANTA vs AstroNet vs ablations | `visualization/roc_curves.py` |
| Fig. 6 | Position-aware attention heatmap on transit example | `visualization/attention_maps.py` |
| Fig. 7 | Ablation results bar chart | `visualization/ablation_plots.py` |
| Fig. 8 | EllipticMish vs ReLU activation response to transit signal | `visualization/architecture_diagram.py` |

### 9.3 Abstract Template

```
We present MANTA (Mandel-Agol Neural Transit Architecture), the first neural
network for exoplanet transit detection whose architecture is derived from the
mathematical structure of the Mandel-Agol transit model. While existing approaches
(AstroNet, ExoMiner, GPFC) treat architecture design as empirical optimization
and use transit physics only for data generation, MANTA derives five architectural
constraints directly from stellar physics: (1) position-aware attention from limb
darkening geometry, (2) smooth non-monotonic activations from elliptic integral
structure, (3) hard-bounded output from flux ratio physics, (4) symmetric weight
constraints from transit time-reversibility, and (5) parallel processing streams
from stellar variability frequency separation. On the Kepler DR24 benchmark,
MANTA achieves an AUC-ROC of [X.XXX] compared to AstroNet's [Y.YYY]
(p < [Z.ZZZ], McNemar's test). Ablation studies confirm that each physics-derived
component contributes independently to performance, with the largest gains from
[COMPONENT] (ΔAUC = [W.WWW]). Our work demonstrates that the mathematical
structure of physical models can serve as architectural axioms for scientific
machine learning, not merely as regularization terms.
```

---

## 10. Risk Register

### Risk 1: Compute Limits (Kaggle Session Timeouts)

**Probability**: HIGH | **Impact**: HIGH

**Risk**: Kaggle GPU sessions have a ~30 hr/week limit with max ~9-12 continuous hours. Training MANTA for 80 epochs on 15K samples may require multiple sessions.

**Mitigation**:
- Checkpoint every epoch with full optimizer/scheduler state (Section 6.3)
- Auto-resume from latest checkpoint on session restart
- Estimate per-epoch time in first session and adjust total epochs accordingly
- If T4 is insufficient: reduce d_model from 128→64, reduce attention heads 4→2
- Precompute all preprocessed data ONCE, save to disk. Training sessions only need to load `.npz` files
- Use `torch.compile()` (PyTorch 2.0+) for training speedups

### Risk 2: Negative Results (MANTA ≤ AstroNet)

**Probability**: MEDIUM | **Impact**: HIGH

**Risk**: MANTA may not outperform AstroNet on AUC-ROC despite physics-derived architecture.

**Mitigation**:
- **This is still a publishable result.** The paper's contribution is the *derivation method*, not necessarily beating SOTA. Frame as: "We demonstrate how to derive architectures from physics. On Kepler DR24, MANTA achieves competitive/comparable/superior performance while providing interpretable physics-aligned components."
- If performance is close, focus the paper on ablation results: which physics constraints help, which don't, and why. This is valuable scientific knowledge.
- Ensure fair comparison: same train/val/test split, same preprocessing, same evaluation metrics.
- If MANTA underperforms, investigate whether the parameter budget is the issue (MANTA has 4.8M vs AstroNet's 1.1M — but also has structural constraints that limit capacity). Try a parameter-matched ablation.

### Risk 3: Data Download / API Issues

**Probability**: MEDIUM | **Impact**: MEDIUM

**Risk**: lightkurve downloads may fail for some targets, MAST servers may be slow, Kaggle may have network restrictions.

**Mitigation**:
- Use Kaggle dataset hub: search for existing Kepler light curve datasets already uploaded to Kaggle (e.g., "kepler-exoplanet-search-results")
- Cache all downloads to persistent storage immediately
- Implement retry logic with exponential backoff in download script
- If lightkurve fails on Kaggle: pre-download on a local machine and upload the preprocessed `.npz` files as a Kaggle dataset
- Fallback: use the pre-computed TFRecord dataset from Google's AstroNet repo (convert TFRecord → numpy)

### Risk 4: Frequency Decomposition Ineffectiveness

**Probability**: MEDIUM | **Impact**: LOW

**Risk**: The FFT-based frequency decomposition (Constraint 5) may not improve performance because Kepler PDCSAP flux has already been detrended, removing some stellar variability.

**Mitigation**:
- This is explicitly tested in ablation A5
- If frequency bands don't help with PDCSAP data, try using SAP (raw) flux where stellar variability is fully present
- The parallel stream architecture still helps even without explicit frequency decomposition (different kernel sizes act as implicit bandpass filters)
- Report both results: with and without frequency decomposition

### Risk 5: Reviewer Skepticism About Physics Justifications

**Probability**: MEDIUM | **Impact**: HIGH

**Risk**: ML reviewers may dismiss the physics derivations as post-hoc justification rather than genuine architectural constraints. Astrophysics reviewers may find the ML techniques standard.

**Mitigation**:
- **Strong ablation study is key**: If removing each physics component degrades performance independently, this is empirical evidence that the constraints are genuine, not cosmetic
- Cross-list on both cs.LG and astro-ph.EP to reach both communities
- In Section 3, provide rigorous mathematical derivation from Mandel-Agol → architectural constraint, not just analogy
- Cite specific equations from Mandel & Agol 2002 for each constraint
- Provide attention visualization (Figure 6) showing that position-aware attention actually attends to physically relevant regions
- Compare against a "parameter-matched vanilla" model (same total parameters as MANTA but no physics constraints) to control for capacity differences

---

## Appendix: Kaggle Notebook Workflow

Since all compute is Kaggle-based, the workflow is:

### Session 1: Data Preparation (~2 hours)
```
notebooks/01_data_pipeline.ipynb
- Download TCE catalog
- Download light curves (subset for initial testing)
- Run preprocessing & frequency decomposition
- Save all preprocessed .npz files as Kaggle dataset output
```

### Session 2: Train AstroNet Baseline (~4 hours)
```
notebooks/02_train_astronet.ipynb
- Load preprocessed data
- Train AstroNet for 50 epochs
- Save best model checkpoint
- Evaluate on test set
```

### Session 3: Train MANTA (~6 hours)
```
notebooks/03_train_manta.ipynb
- Load preprocessed data
- Train MANTA for ~40 epochs (will need 2 sessions)
- Save checkpoint at epoch 40
```

### Session 4: Resume MANTA + Ablations (~8 hours)
```
notebooks/03_train_manta.ipynb (resume)
- Resume from epoch 40, train to 80
- Save best model

notebooks/04_ablations.ipynb
- Run ablation A1 and A2 (shorter training)
```

### Sessions 5-7: Remaining Ablations
```
notebooks/04_ablations.ipynb
- Run ablations A3, A4, A5, A_all
- 5 seeds each = multiple sessions
```

### Session 8: Analysis & Figures
```
notebooks/05_analysis.ipynb
- Load all results
- Generate all figures and tables
- Statistical tests
- Export LaTeX tables
```

---

## Quick Start Commands

```bash
# Clone and setup
cd manta
pip install -e .

# Download data
python scripts/download_data.py --output_dir data/raw/

# Preprocess
python scripts/preprocess_data.py --input_dir data/raw/ --output_dir data/processed/

# Train baseline
python scripts/train_astronet.py --config configs/astronet_baseline.yaml

# Train MANTA
python scripts/train_manta.py --config configs/manta.yaml

# Run ablations
python scripts/run_ablations.py --config configs/manta.yaml --ablations all

# Generate all results
python scripts/evaluate_all.py --output_dir results/

# Generate paper figures
python scripts/generate_figures.py --results_dir results/ --output_dir paper/figures/
```
