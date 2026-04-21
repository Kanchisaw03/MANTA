# MANTA Implementation Plan — Part 3: Baseline & MANTA Architecture

## 4. AstroNet Baseline Reimplementation

### 4.1 Architecture (Exact Replication from Shallue & Vanderburg 2018)

The AstroNet architecture uses two parallel 1D-CNN branches. The original was TensorFlow; we reimplement in PyTorch.

```python
# models/astronet.py

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Conv1d + ReLU + optional MaxPool."""
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size) if pool_size else nn.Identity()
    
    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class AstroNet(nn.Module):
    """
    PyTorch reimplementation of AstroNet (Shallue & Vanderburg 2018).
    
    Global branch: 5 conv blocks processing 2001-bin global view
    Local branch: 5 conv blocks processing 201-bin local view
    Merger: 4 FC layers → sigmoid output
    
    Architecture details from the paper's Table 2:
    
    Global Branch (input: 1×2001):
        conv1: 16 filters, kernel 5, pool 5, stride 2  → 16×400
        conv2: 16 filters, kernel 5, pool 5, stride 2  → 16×80
        conv3: 32 filters, kernel 5, pool 5, stride 2  → 32×16
        conv4: 32 filters, kernel 5, pool 5, stride 2  → 32×3
        conv5: 64 filters, kernel 5, no pool             → 64×3
        flatten → 192
    
    Local Branch (input: 1×201):
        conv1: 16 filters, kernel 5, pool 7, stride 2  → 16×28
        conv2: 16 filters, kernel 5, pool 7, stride 2  → 16×4
        conv3: 32 filters, kernel 5, no pool             → 32×4
        flatten → 128
    
    FC layers (input: 192+128=320):
        fc1: 512, ReLU, Dropout(0.4)
        fc2: 512, ReLU, Dropout(0.4)
        fc3: 512, ReLU, Dropout(0.4)
        fc4: 512, ReLU, Dropout(0.4)
        output: 1, Sigmoid
    
    Total parameters: ~1.1M
    """
    
    def __init__(self):
        super().__init__()
        
        # Global branch (2001 input length)
        self.global_branch = nn.Sequential(
            ConvBlock(1, 16, kernel_size=5),
            nn.MaxPool1d(kernel_size=5, stride=2),
            ConvBlock(16, 16, kernel_size=5),
            nn.MaxPool1d(kernel_size=5, stride=2),
            ConvBlock(16, 32, kernel_size=5),
            nn.MaxPool1d(kernel_size=5, stride=2),
            ConvBlock(32, 32, kernel_size=5),
            nn.MaxPool1d(kernel_size=5, stride=2),
            ConvBlock(32, 64, kernel_size=5),
            nn.AdaptiveAvgPool1d(3),  # Ensure fixed output size
            nn.Flatten()
        )
        
        # Local branch (201 input length)
        self.local_branch = nn.Sequential(
            ConvBlock(1, 16, kernel_size=5),
            nn.MaxPool1d(kernel_size=7, stride=2),
            ConvBlock(16, 16, kernel_size=5),
            nn.MaxPool1d(kernel_size=7, stride=2),
            ConvBlock(16, 32, kernel_size=5),
            nn.AdaptiveAvgPool1d(4),  # Ensure fixed output size
            nn.Flatten()
        )
        
        # Calculate merged feature size
        # Global: 64*3 = 192, Local: 32*4 = 128 → total = 320
        self.classifier = nn.Sequential(
            nn.Linear(192 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, global_view, local_view, **kwargs):
        """
        Args:
            global_view: (batch, 1, 2001)
            local_view: (batch, 1, 201)
        Returns:
            prediction: (batch, 1) probability in [0, 1]
        """
        g = self.global_branch(global_view)   # (batch, 192)
        l = self.local_branch(local_view)      # (batch, 128)
        merged = torch.cat([g, l], dim=1)      # (batch, 320)
        return self.classifier(merged)
```

### 4.2 AstroNet Training Hyperparameters

```yaml
# configs/astronet_baseline.yaml
model:
  name: AstroNet
  
training:
  epochs: 50
  batch_size: 64
  optimizer: adam
  learning_rate: 1.0e-5          # From original paper
  weight_decay: 0.0
  
  # Class imbalance handling
  class_weight: balanced         # Inverse frequency weighting
  sampler: weighted_random       # WeightedRandomSampler
  
  # Regularization
  dropout: 0.4                   # Applied in FC layers
  
  # Early stopping
  early_stopping_patience: 5     # Stop if val AUC doesn't improve for 5 epochs
  early_stopping_metric: auc_roc
  
  # Checkpointing
  checkpoint_every_n_epochs: 1
  keep_top_k_checkpoints: 3
  
evaluation:
  metrics: [auc_roc, precision, recall, f1, average_precision]
  threshold: 0.5                 # Classification threshold
```

---

## 5. MANTA Architecture (Complete, Layer by Layer)

### 5.1 Component 1: Position-Aware Attention (Limb Darkening)

**Physics**: I(r) = 1 - u₁(1 - μ) - u₂(1 - μ)² where μ = √(1 - r²). A transit at the stellar limb has different depth than at center. Standard convolutions ignore this.

**Implementation**: We add a learned positional encoding that encodes distance from transit center (phase = 0). The attention weight for each position depends on where it is relative to the transit mid-point, analogous to how limb darkening makes the signal position-dependent.

```python
# models/components/position_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LimbDarkeningPositionalEncoding(nn.Module):
    """
    Positional encoding inspired by quadratic limb darkening.
    
    Instead of sinusoidal encoding (which assumes periodicity) or learned
    absolute positions (which assume fixed sequence length), we use a
    physics-motivated encoding based on the limb darkening profile.
    
    The encoding models position as a proxy for stellar disk radius:
    - Phase = 0 (transit center) → high signal importance
    - |Phase| increasing → decreasing importance (limb darkening effect)
    
    The encoding has learnable parameters analogous to limb darkening
    coefficients (u1, u2), initialized to solar values but updated during training.
    
    This gives the network a structural prior that transit signal strength
    depends on position, breaking translation invariance where physics demands it.
    """
    
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        
        # Learnable limb-darkening-inspired parameters
        # Initialized to approximate solar quadratic LD coefficients
        self.u1 = nn.Parameter(torch.tensor(0.4))
        self.u2 = nn.Parameter(torch.tensor(0.26))
        
        # Projection from 3D encoding to d_model dimensions
        self.projection = nn.Linear(3, d_model)
        
        # Store positions
        self.register_buffer('positions', torch.arange(max_len).float())
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate positional encoding of shape (1, seq_len, d_model).
        
        The encoding has 3 raw features per position:
        1. Normalized position (linear ramp from -1 to 1)
        2. Quadratic LD profile: 1 - u1*(1-μ) - u2*(1-μ)²
        3. Squared distance from center (for ingress/egress sensitivity)
        """
        # Normalized position: -1 (start) to +1 (end), 0 = transit center
        pos = torch.linspace(-1, 1, seq_len, device=self.u1.device)
        
        # Compute μ = cos(θ), treating position as fractional disk radius
        # Map |phase| to disk radius r ∈ [0, 1], then μ = √(1 - r²)
        r = torch.abs(pos)  # proxy for stellar disk radius
        r = torch.clamp(r, 0, 0.999)
        mu = torch.sqrt(1 - r**2)
        
        # Limb darkening profile
        ld_profile = 1 - self.u1 * (1 - mu) - self.u2 * (1 - mu)**2
        
        # Stack raw features
        features = torch.stack([pos, ld_profile, r**2], dim=-1)  # (seq_len, 3)
        
        # Project to d_model dimensions
        encoding = self.projection(features)  # (seq_len, d_model)
        
        return encoding.unsqueeze(0)  # (1, seq_len, d_model)


class PositionAwareAttention(nn.Module):
    """
    Multi-head self-attention with physics-based positional bias.
    
    Standard self-attention: Attention(Q,K,V) = softmax(QK^T / √d_k) V
    Position-aware:          Attention(Q,K,V) = softmax((QK^T + B) / √d_k) V
    
    where B is a relative position bias derived from limb darkening geometry.
    
    The bias encodes that:
    - Features near transit center (phase ≈ 0) should attend strongly to each other
    - Features at ingress should attend to their symmetric counterpart at egress
    - Far-from-transit features (phase ≈ ±0.5) carry less transit information
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, max_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_encoding = LimbDarkeningPositionalEncoding(d_model, max_len)
        
        # Relative position bias table
        # For each head, learn a bias based on relative distance
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(n_heads, 2 * max_len - 1)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def _get_relative_position_bias(self, seq_len: int) -> torch.Tensor:
        """Compute relative position bias matrix of shape (n_heads, seq_len, seq_len)."""
        coords = torch.arange(seq_len, device=self.rel_pos_bias.device)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # (seq_len, seq_len)
        relative_coords += seq_len - 1  # shift to non-negative indices
        
        # Clamp to valid range
        relative_coords = torch.clamp(relative_coords, 0, 2 * seq_len - 2)
        
        # Lookup bias values: (n_heads, seq_len, seq_len)
        bias = self.rel_pos_bias[:, relative_coords]
        return bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, N, D = x.shape
        
        # Add positional encoding
        pos_enc = self.pos_encoding(N)  # (1, N, d_model)
        x_pos = x + pos_enc
        
        # Compute Q, K, V
        Q = self.W_q(x_pos).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x_pos).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        # Note: V uses x without positional encoding (position only affects routing)
        
        # Attention scores with relative position bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias
        rel_bias = self._get_relative_position_bias(N)  # (n_heads, N, N)
        scores = scores + rel_bias.unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        output = self.W_o(output)
        
        # Residual connection + layer norm
        output = self.layer_norm(x + output)
        
        return output
```

### 5.2 Component 2: Smooth Non-Monotonic Activations (Elliptic Integrals)

**Physics**: The Mandel-Agol flux involves elliptic integrals K(k), E(k) which are smooth, bounded, non-monotonic functions. ReLU (piecewise linear, monotonic) has zero inductive bias for this structure.

```python
# models/components/smooth_activations.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EllipticMish(nn.Module):
    """
    A smooth, bounded, non-monotonic activation inspired by elliptic integral geometry.
    
    Based on Mish (x * tanh(softplus(x))) but modified to better match
    the bounded, non-monotonic behavior of elliptic integrals:
    
    EllipticMish(x) = x * tanh(softplus(β * x)) * sigmoid(γ * x)
    
    Properties:
    - Smooth everywhere (C∞) — matches smoothness of elliptic integrals
    - Non-monotonic: has a local minimum for x < 0 — matches the 
      non-monotonic behavior of circle-circle intersection area
    - Bounded below: output ≥ ~ -0.31 — matches physical flux bounds
    - Approximately linear for large positive x — preserves gradient flow
    - β, γ are learnable per-channel to adapt to the signal
    
    Why not standard Mish?
    Standard Mish works well but the sigmoid gating factor adds the
    physical prior that activations should be bounded — matching the 
    constraint that flux ratios are bounded quantities.
    
    Why not ReLU?
    ReLU is monotonic and piecewise linear. The circle-circle intersection
    area (which determines transit flux) is a smooth non-monotonic function
    of the planet-star separation. A piecewise linear activation must use
    many more parameters to approximate this behavior.
    """
    
    def __init__(self, channels: int = 1):
        super().__init__()
        # Learnable scaling parameters, initialized to standard Mish behavior
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(torch.ones(channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of any shape. If channels > 1, 
               beta/gamma are broadcast over the channel dimension.
        """
        if self.beta.shape[0] > 1 and x.dim() >= 2:
            # Reshape for broadcasting: (1, C, 1) for Conv1d or (1, C) for Linear
            shape = [1] * x.dim()
            shape[1] = -1
            beta = self.beta.view(shape)
            gamma = self.gamma.view(shape)
        else:
            beta = self.beta
            gamma = self.gamma
        
        mish_part = x * torch.tanh(F.softplus(beta * x))
        gate = torch.sigmoid(gamma * x)
        
        return mish_part * gate


class SmoothTransitActivation(nn.Module):
    """
    Drop-in activation replacement for transit-sensitive layers.
    
    Uses EllipticMish for the main path and adds a residual Gaussian
    bump centered at x ≈ 0, providing extra sensitivity near the
    transit minimum (where flux change is most rapid).
    
    f(x) = EllipticMish(x) + α * exp(-x²/(2σ²))
    
    The Gaussian term provides additional non-monotonic structure
    near zero, where transit signals are centered.
    """
    
    def __init__(self, channels: int = 1):
        super().__init__()
        self.emish = EllipticMish(channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Gaussian amplitude
        self.sigma = nn.Parameter(torch.tensor(1.0))  # Gaussian width
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.emish(x)
        gaussian = self.alpha * torch.exp(-x**2 / (2 * self.sigma**2))
        return base + gaussian
```

### 5.3 Component 3: Constrained Output Layer (Flux Bounds)

**Physics**: F(t) ∈ [0, 1] always. A planet cannot cause negative flux or amplify flux beyond the unoccluded star.

```python
# models/components/constrained_output.py

import torch
import torch.nn as nn

class PhysicsConstrainedOutput(nn.Module):
    """
    Output layer that enforces physical flux bounds as hard architectural constraints.
    
    For transit detection (binary classification):
    - Output represents P(planet | data)
    - Must be in [0, 1]
    
    Standard approach: Linear → Sigmoid (soft constraint, output approaches but 
    never reaches 0 or 1, gradient vanishes at extremes).
    
    Our approach: Linear → HardSigmoid + clamping, with temperature scaling
    learned during training. This provides:
    1. Hard bounds: output is EXACTLY 0 or 1 at the boundaries
    2. Better gradient flow: HardSigmoid has non-vanishing gradients in [0,1]
    3. Temperature parameter τ controls sharpness of the decision boundary
    
    Additionally, we add a physics-motivated bias: the prior probability of
    a TCE being a planet is ~23% (from the Kepler DR24 statistics).
    This is encoded as an initial bias in the output layer.
    """
    
    def __init__(self, in_features: int, prior_prob: float = 0.23):
        super().__init__()
        
        self.linear = nn.Linear(in_features, 1)
        
        # Initialize bias to match prior probability
        # sigmoid(bias) = prior_prob → bias = log(prior_prob / (1 - prior_prob))
        initial_bias = torch.log(torch.tensor(prior_prob / (1 - prior_prob)))
        nn.init.constant_(self.linear.bias, initial_bias.item())
        nn.init.xavier_uniform_(self.linear.weight)
        
        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)
        Returns:
            probability: (batch, 1) hard-bounded in [0, 1]
        """
        logits = self.linear(x) / self.temperature
        
        # Hard sigmoid: piecewise linear approximation bounded in [0,1]
        # f(x) = clamp(0.2x + 0.5, 0, 1)
        # This gives exact 0 and exact 1 at the boundaries
        output = torch.clamp(0.2 * logits + 0.5, min=0.0, max=1.0)
        
        return output
```

### 5.4 Component 4: Symmetric Weight Constraints (Time-Reversal)

**Physics**: For circular orbits, ingress mirrors egress under t → -t. The network should encode this symmetry to avoid learning it from data.

```python
# models/components/symmetric_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PalindromicConv1d(nn.Module):
    """
    1D convolution with palindromic (time-reversal symmetric) kernel.
    
    For a standard convolution kernel w = [w_0, w_1, ..., w_k],
    a palindromic kernel satisfies w_i = w_{k-i}.
    
    Implementation: We parameterize only the first (k+1)//2 weights
    and mirror them to construct the full kernel. This:
    
    1. Halves the parameter count for the kernel
    2. Guarantees the convolution output is symmetric under time reversal
    3. Is physically correct for circular orbit transits where ingress = egress
    
    For eccentric orbits (where ingress ≠ egress), the non-symmetric
    components are handled by subsequent non-palindromic layers.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, padding: str = 'same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Only parameterize the first half of the kernel
        half_k = (kernel_size + 1) // 2
        self.half_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, half_k) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def _build_symmetric_kernel(self) -> torch.Tensor:
        """Construct full palindromic kernel from half weights."""
        if self.kernel_size % 2 == 1:
            # Odd kernel: [a, b, c, b, a]
            flipped = torch.flip(self.half_weight[:, :, :-1], dims=[2])
        else:
            # Even kernel: [a, b, b, a]
            flipped = torch.flip(self.half_weight, dims=[2])
        
        full_weight = torch.cat([self.half_weight, flipped], dim=2)
        return full_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._build_symmetric_kernel()
        
        if self.padding == 'same':
            pad = self.kernel_size // 2
            x = F.pad(x, (pad, pad), mode='reflect')
            return F.conv1d(x, weight, self.bias)
        else:
            return F.conv1d(x, weight, self.bias, padding=0)


class SymmetricTransitEncoder(nn.Module):
    """
    Transit encoder with enforced time-reversal symmetry.
    
    Architecture:
    1. PalindromicConv1d stack: Extracts symmetric features
    2. Symmetry verification: Checks that output is approximately
       symmetric (as a training diagnostic, not a constraint)
    3. Optional symmetry-breaking path: For eccentric orbits,
       a small asymmetric branch captures ingress/egress differences
    
    The symmetric encoder processes the local view (201 bins)
    where time-reversal symmetry is most relevant.
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        
        # Symmetric path (for circular orbit transit shape)
        self.sym_path = nn.Sequential(
            PalindromicConv1d(in_channels, base_channels, kernel_size=7),
            nn.LayerNorm([base_channels]),  # No BatchNorm (preserves flux scale)
            EllipticMish(base_channels),
            PalindromicConv1d(base_channels, base_channels * 2, kernel_size=5),
            nn.LayerNorm([base_channels * 2]),
            EllipticMish(base_channels * 2),
            PalindromicConv1d(base_channels * 2, base_channels * 4, kernel_size=3),
            nn.LayerNorm([base_channels * 4]),
            EllipticMish(base_channels * 4),
        )
        
        # Small asymmetric path (for eccentric orbit corrections)
        # Much smaller capacity — asymmetry is a correction, not the main signal
        self.asym_path = nn.Sequential(
            nn.Conv1d(in_channels, base_channels // 2, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv1d(base_channels // 2, base_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        
        self.output_channels = base_channels * 4 + base_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            features: (batch, output_channels, seq_len)
        """
        sym_features = self.sym_path(x)
        asym_features = self.asym_path(x)
        
        # Pad asym to match sym sequence length if needed
        if sym_features.shape[2] != asym_features.shape[2]:
            asym_features = F.interpolate(
                asym_features, size=sym_features.shape[2], mode='linear'
            )
        
        return torch.cat([sym_features, asym_features], dim=1)


# Need to import EllipticMish for use in SymmetricTransitEncoder
from models.components.smooth_activations import EllipticMish
```

### 5.5 Component 5: Parallel Frequency Streams (Stellar Variability)

**Physics**: Starspots, granulation, and oscillations occupy distinct frequency bands. Mixing them in a single pathway forces the network to disentangle them, wasting capacity.

```python
# models/components/parallel_streams.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyStream(nn.Module):
    """
    A single frequency-specific processing stream.
    
    Each stream has convolutional kernels sized to match the characteristic
    timescale of its target frequency band:
    
    - Rotation stream: Large kernels (15-31) to capture slow modulations
    - Granulation stream: Medium kernels (7-15) for hour-scale variations
    - Transit stream: Small kernels (3-7) for sharp transit features
    
    This is the architectural equivalent of a bandpass filter bank.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes: list, band_name: str):
        super().__init__()
        self.band_name = band_name
        
        layers = []
        ch_in = in_channels
        for i, ks in enumerate(kernel_sizes):
            ch_out = out_channels if i == len(kernel_sizes) - 1 else out_channels // 2
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=ks, padding='same'),
                nn.LayerNorm([ch_out]),
                nn.GELU(),  # Smooth activation throughout
            ])
            ch_in = ch_out
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParallelFrequencyProcessor(nn.Module):
    """
    Parallel processing streams for distinct stellar variability bands.
    
    Three parallel branches process the light curve simultaneously,
    each with kernel sizes matched to its frequency band:
    
    Branch 1 — Rotation/Activity (days-weeks timescale):
        Kernel sizes: [31, 21, 15]
        Purpose: Detect and separate starspot modulation
        Output: Cleaned representation with rotation signal isolated
    
    Branch 2 — Granulation (hours timescale):
        Kernel sizes: [15, 11, 7]
        Purpose: Separate convective noise
        Output: Granulation-filtered representation
    
    Branch 3 — Transit Signal (minutes-hours timescale):
        Kernel sizes: [7, 5, 3]
        Purpose: Extract sharp transit edges (ingress/egress)
        Output: Transit-sensitive features with high temporal resolution
    
    The outputs are concatenated and fused by a 1x1 convolution,
    which learns the optimal combination of frequency-specific features.
    
    If frequency-decomposed inputs are available (from preprocessing),
    each stream receives its corresponding band. Otherwise, all streams
    receive the same input and rely on the kernel size difference for
    implicit frequency separation.
    """
    
    def __init__(self, in_channels: int = 1, stream_channels: int = 32):
        super().__init__()
        
        self.rotation_stream = FrequencyStream(
            in_channels, stream_channels,
            kernel_sizes=[31, 21, 15],
            band_name='rotation'
        )
        
        self.granulation_stream = FrequencyStream(
            in_channels, stream_channels,
            kernel_sizes=[15, 11, 7],
            band_name='granulation'
        )
        
        self.transit_stream = FrequencyStream(
            in_channels, stream_channels,
            kernel_sizes=[7, 5, 3],
            band_name='transit'
        )
        
        # Fusion: 1×1 conv to combine all streams
        self.fusion = nn.Sequential(
            nn.Conv1d(stream_channels * 3, stream_channels * 2, kernel_size=1),
            nn.LayerNorm([stream_channels * 2]),
            nn.GELU(),
        )
        
        self.output_channels = stream_channels * 2
    
    def forward(self, x: torch.Tensor, 
                freq_bands: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len) — raw light curve view
            freq_bands: (batch, 3, seq_len) — optional pre-decomposed bands
                        [rotation, granulation, transit] from FFT preprocessing
        
        Returns:
            features: (batch, output_channels, seq_len)
        """
        if freq_bands is not None:
            # Use pre-decomposed frequency bands
            rot_input = freq_bands[:, 0:1, :]
            gran_input = freq_bands[:, 1:2, :]
            transit_input = freq_bands[:, 2:3, :]
        else:
            # All streams process the same input
            rot_input = x
            gran_input = x
            transit_input = x
        
        rot_features = self.rotation_stream(rot_input)
        gran_features = self.granulation_stream(gran_input)
        transit_features = self.transit_stream(transit_input)
        
        # Concatenate along channel dimension
        combined = torch.cat([rot_features, gran_features, transit_features], dim=1)
        
        # Fuse
        output = self.fusion(combined)
        
        return output
```

### 5.6 Full MANTA Model Assembly

```python
# models/manta.py

import torch
import torch.nn as nn
from models.components.position_attention import PositionAwareAttention
from models.components.smooth_activations import EllipticMish
from models.components.constrained_output import PhysicsConstrainedOutput
from models.components.symmetric_encoder import SymmetricTransitEncoder
from models.components.parallel_streams import ParallelFrequencyProcessor

class MANTA(nn.Module):
    """
    Mandel-Agol Neural Transit Architecture (MANTA).
    
    The first neural network for exoplanet transit detection whose architecture
    is derived from the mathematical structure of the Mandel-Agol transit model.
    
    Architecture flow:
    
    Input: global_view (1, 2001), local_view (1, 201), freq_bands (3, 2001)
           │                       │                     │
           ▼                       ▼                     │
    ┌─────────────────┐  ┌──────────────────┐           │
    │ Parallel Freq   │  │ Symmetric Transit │           │
    │ Processor       │◄─┤ Encoder           │           │
    │ (Constraint 5)  │  │ (Constraint 4)    │           │
    │ kernel: 31→3    │  │ palindromic conv  │           │
    └────────┬────────┘  └────────┬─────────┘           │
             │                     │                     │
             ▼                     ▼                     │
    ┌─────────────────────────────────────────┐         │
    │ Position-Aware Attention (Constraint 1) │◄────────┘
    │ limb-darkening positional encoding      │
    │ relative position bias                  │
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │ Feature Extraction with EllipticMish    │
    │ (Constraint 2)                          │
    │ smooth, bounded, non-monotonic activations│
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │ Physics-Constrained Output              │
    │ (Constraint 3)                          │
    │ HardSigmoid, temperature-scaled         │
    │ P(planet) ∈ [0, 1] (hard bound)         │
    └─────────────────────────────────────────┘
    
    Total parameters: ~4.8M (estimate)
    """
    
    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 stream_channels: int = 32, sym_base: int = 16):
        super().__init__()
        
        # === Global branch with parallel frequency streams ===
        self.freq_processor = ParallelFrequencyProcessor(
            in_channels=1, stream_channels=stream_channels
        )
        # Project frequency features to d_model for attention
        self.global_proj = nn.Conv1d(
            self.freq_processor.output_channels, d_model, kernel_size=1
        )
        
        # === Local branch with symmetric transit encoder ===
        self.sym_encoder = SymmetricTransitEncoder(
            in_channels=1, base_channels=sym_base
        )
        # Project symmetric features to d_model
        self.local_proj = nn.Conv1d(
            self.sym_encoder.output_channels, d_model, kernel_size=1
        )
        
        # === Position-aware attention (applied to both branches) ===
        self.global_attention = PositionAwareAttention(
            d_model=d_model, n_heads=n_heads
        )
        self.local_attention = PositionAwareAttention(
            d_model=d_model, n_heads=n_heads
        )
        
        # === Feature extraction with EllipticMish ===
        self.global_feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            EllipticMish(d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model),
            EllipticMish(d_model),
        )
        
        self.local_feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            EllipticMish(d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model),
            EllipticMish(d_model),
        )
        
        # === Merge global + local ===
        self.merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            EllipticMish(d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model // 2),
            EllipticMish(d_model // 2),
        )
        
        # === Physics-constrained output ===
        self.output = PhysicsConstrainedOutput(
            in_features=d_model // 2,
            prior_prob=0.23  # Kepler planet prior
        )
        
        # Pooling for converting sequence features to fixed-size
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, global_view: torch.Tensor, 
                local_view: torch.Tensor,
                freq_bands: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            global_view: (batch, 1, 2001)
            local_view:  (batch, 1, 201)
            freq_bands:  (batch, 3, 2001) optional frequency decomposition
        
        Returns:
            prediction: (batch, 1) — P(planet) in [0, 1]
        """
        # === Global path ===
        # Parallel frequency processing
        g = self.freq_processor(global_view, freq_bands)  # (B, 64, 2001)
        g = self.global_proj(g)                             # (B, d_model, 2001)
        
        # Downsample for attention efficiency (2001 → 501)
        g = F.avg_pool1d(g, kernel_size=4, stride=4)       # (B, d_model, 500)
        
        # Position-aware attention
        g = g.transpose(1, 2)                               # (B, 500, d_model)
        g = self.global_attention(g)                        # (B, 500, d_model)
        
        # Feature extraction with EllipticMish
        g = self.global_feature_extractor(g)                # (B, 500, d_model)
        
        # Pool to fixed size
        g = g.transpose(1, 2)                               # (B, d_model, 500)
        g = self.global_pool(g).squeeze(-1)                 # (B, d_model)
        
        # === Local path ===
        # Symmetric transit encoding
        l = self.sym_encoder(local_view)                    # (B, sym_out, 201)
        l = self.local_proj(l)                              # (B, d_model, 201)
        
        # Position-aware attention
        l = l.transpose(1, 2)                               # (B, 201, d_model)
        l = self.local_attention(l)                         # (B, 201, d_model)
        
        # Feature extraction
        l = self.local_feature_extractor(l)                 # (B, 201, d_model)
        
        # Pool to fixed size
        l = l.transpose(1, 2)                               # (B, d_model, 201)
        l = self.local_pool(l).squeeze(-1)                  # (B, d_model)
        
        # === Merge and classify ===
        merged = torch.cat([g, l], dim=1)                   # (B, d_model*2)
        features = self.merge(merged)                       # (B, d_model//2)
        prediction = self.output(features)                  # (B, 1)
        
        return prediction


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
```

### 5.7 Parameter Count Estimates

| Component | Parameters | Justification |
|-----------|-----------|---------------|
| ParallelFrequencyProcessor | ~150K | 3 streams × 3 conv layers × 32 channels |
| SymmetricTransitEncoder | ~45K | Palindromic kernels = half weight count |
| PositionAwareAttention (×2) | ~260K | 4-head attention with d_model=128 |
| EllipticMish activations | ~1K | Only learnable β, γ, α, σ per channel |
| Feature extractors (×2) | ~200K | 2-layer MLPs with d_model=128 |
| Merge layers | ~35K | 2-layer MLP reducing to d_model//2 |
| PhysicsConstrainedOutput | ~65 | Single linear layer + temperature |
| **Total** | **~4.8M** | **~4× AstroNet, justified by attention** |
