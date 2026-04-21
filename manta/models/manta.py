"""Full Mandel-Agol Neural Transit Architecture (MANTA).

This module assembles all physics-derived components into a complete end-to-end
transit classifier aligned with the five architecture constraints.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor, nn

from manta.models.components.elliptic_activation import EllipticMish
from manta.models.components.frequency_processor import ParallelFrequencyProcessor
from manta.models.components.physics_output import PhysicsConstrainedOutput
from manta.models.components.position_aware_attention import PositionAwareAttention
from manta.models.components.symmetric_encoder import SymmetricTransitEncoder
from manta.utils.config import MANTAConfig

DEFAULT_IMPACT_PARAMETER: float = 0.5


class MANTA(nn.Module):
    """Mandel-Agol Neural Transit Architecture.

    Parameters
    ----------
    config : MANTAConfig
        Typed experiment configuration.

    Notes
    -----
    Forward path combines all physics constraints:
    frequency decomposition (C5), parallel band processing (C5), symmetric local
    encoding (C4), limb-darkening attention (C1), elliptic activation (C2), and
    constrained probability output (C3).
    """

    def __init__(self, config: MANTAConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.model.d_model

        # Constraint 5: Specialized branches process granulation/astero/starspot signals separately.
        self.freq_processor = ParallelFrequencyProcessor(
            granulation_channels=config.model.granulation_channels,
            astero_channels=config.model.astero_channels,
            starspot_channels=config.model.starspot_channels,
        )

        # Constraint 4: Palindromic local encoder enforces ingress/egress symmetry in learned filters.
        self.local_encoder = SymmetricTransitEncoder(
            in_channels=1,
            base_channels=config.model.symmetric_base_channels,
            kernel_sizes=config.model.symmetric_kernel_sizes,
        )

        # Constraint 5: Project fused frequency features to common attention embedding width.
        self.global_projection = nn.Conv1d(self.freq_processor.output_channels, d_model, kernel_size=1)
        # Constraint 4: Project symmetry-aware local features into same embedding space.
        self.local_projection = nn.Conv1d(self.local_encoder.output_channels, d_model, kernel_size=1)

        # Constraint 1: Attention uses limb-darkening geometry to route sequence interactions.
        self.global_attention = PositionAwareAttention(
            d_model=d_model,
            n_heads=config.model.n_heads,
            stellar_radius_bins=config.model.stellar_radius_bins,
        )
        # Constraint 1: Local branch also receives position-aware bias because transit depth varies with disk radius.
        self.local_attention = PositionAwareAttention(
            d_model=d_model,
            n_heads=config.model.n_heads,
            stellar_radius_bins=config.model.stellar_radius_bins,
        )

        # Constraint 2: EllipticMish blocks preserve smooth non-monotonic transit geometry in fusion layers.
        self.stellar_embed = nn.Sequential(
            nn.Linear(5, d_model // 2),
            EllipticMish(alpha_init=0.1),
            nn.LayerNorm(d_model // 2),
        )

        # Constraint 2: Fusion MLP uses smooth activations to model elliptic integral-like response surfaces.
        self.fusion = nn.Sequential(
            nn.Linear((2 * d_model) + (d_model // 2), d_model),
            EllipticMish(alpha_init=0.1),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2),
            EllipticMish(alpha_init=0.1),
            nn.LayerNorm(d_model // 2),
        )

        # Constraint 3: Output head enforces bounded probability domain with calibrated logits.
        self.output_head = PhysicsConstrainedOutput(in_features=d_model // 2)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Run MANTA forward pass.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch from :class:`KeplerTransitDataset` with keys ``global_view``,
            ``local_view``, ``freq_bands`` (required), and ``stellar_params``.

        Returns
        -------
        torch.Tensor
            Probability tensor of shape ``(batch, 1)``.
        """
        global_view = batch["global_view"]
        local_view = batch["local_view"]
        stellar_params = self._prepare_stellar_params(batch["stellar_params"].float())

        if global_view.ndim != 3 or local_view.ndim != 3:
            raise ValueError("global_view and local_view must have shape (batch, channels, time)")

        # Constraint 5: frequency bands must be produced from raw time-series before phase folding.
        if "freq_bands" not in batch:
            raise ValueError(
                "freq_bands are required. Compute decomposition on raw time-series and phase-fold each band upstream."
            )
        freq_bands = batch["freq_bands"]

        # Constraint 5: branch processor outputs frequency-specialized latent maps.
        freq_features = self.freq_processor(freq_bands)  # (B, C_freq, T_global)
        # Constraint 5: project global frequency features into attention embedding space.
        global_emb = self.global_projection(freq_features)  # (B, d_model, T_global)
        global_seq = global_emb.transpose(1, 2)  # (B, T_global, d_model)

        # Constraint 4: local encoder extracts symmetry-aware transit-shape features.
        local_features = self.local_encoder(local_view)  # (B, C_local, T_local)
        # Constraint 4: align local features to shared embedding width for fusion.
        local_emb = self.local_projection(local_features)  # (B, d_model, T_local)
        local_seq = local_emb.transpose(1, 2)  # (B, T_local, d_model)

        # Constraint 1: attention modulated by limb-darkening profile on global context.
        global_attended = self.global_attention(global_seq, stellar_params)  # (B, T_global, d_model)
        # Constraint 1: attention modulated by limb-darkening profile on local transit window.
        local_attended = self.local_attention(local_seq, stellar_params)  # (B, T_local, d_model)

        # Constraint 2: smooth aggregation preserves non-linear transit morphology without sharp clipping artifacts.
        global_pooled = global_attended.mean(dim=1)  # (B, d_model)
        local_pooled = local_attended.mean(dim=1)  # (B, d_model)

        # Constraint 1 and 5: stellar parameters include rotation period and atmosphere proxies used in attention/fusion.
        stellar_features = self.stellar_embed(stellar_params)  # (B, d_model//2)

        fused = torch.cat([global_pooled, local_pooled, stellar_features], dim=1)
        fused = self.fusion(fused)

        # Constraint 3: physics-constrained output layer bounds predictions within [0, 1].
        probabilities = self.output_head(fused)
        return probabilities

    def _prepare_stellar_params(self, stellar_params: Tensor) -> Tensor:
        """Ensure stellar feature tensor includes impact parameter column.

        Expected canonical ordering: [teff, log_g, feh, rotation_period_days, impact_parameter].
        """
        if stellar_params.ndim != 2:
            raise ValueError("stellar_params must have shape (batch, features)")

        if stellar_params.shape[1] >= 5:
            prepared = stellar_params[:, :5]
            prepared[:, 4] = torch.clamp(prepared[:, 4], min=0.0, max=0.95)
            return prepared

        if stellar_params.shape[1] == 4:
            b = torch.full(
                (stellar_params.shape[0], 1),
                DEFAULT_IMPACT_PARAMETER,
                device=stellar_params.device,
                dtype=stellar_params.dtype,
            )
            return torch.cat([stellar_params, b], dim=1)

        raise ValueError("stellar_params must provide at least [teff, log_g, feh, rotation_period_days]")

    def _decompose_global_view_torch(self, global_view: Tensor, stellar_params: Tensor) -> Tensor:
        """Deprecated in-model decomposition path.

        Parameters
        ----------
        global_view : torch.Tensor
            Tensor of shape ``(batch, 1, time)``.
        stellar_params : torch.Tensor
            Tensor containing stellar metadata, with rotation period expected in
            index 3 when available.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, 3, time)``.
        """
        raise RuntimeError(
            "In-model decomposition is disabled. Provide freq_bands computed from raw time-series decomposition."
        )

    def get_parameter_count(self) -> dict[str, int]:
        """Return parameter count by component and total."""
        components = {
            "freq_processor": sum(p.numel() for p in self.freq_processor.parameters()),
            "local_encoder": sum(p.numel() for p in self.local_encoder.parameters()),
            "global_attention": sum(p.numel() for p in self.global_attention.parameters()),
            "local_attention": sum(p.numel() for p in self.local_attention.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "output_head": sum(p.numel() for p in self.output_head.parameters()),
        }
        components["total"] = sum(components.values())
        return components

    def get_receptive_field(self) -> dict[str, float]:
        """Estimate effective receptive field in timesteps and physical time.

        Returns
        -------
        dict[str, float]
            Receptive fields for local/global branches and hour equivalents.
        """
        local_rf = 1
        for k in self.config.model.symmetric_kernel_sizes:
            local_rf += k - 1

        gran_rf = 1 + (15 - 1) * 1 + (15 - 1) * 2
        astero_rf = 1 + (5 - 1) + (3 - 1)
        # Starspot branch includes attention over reduced timeline, approximated as global context.
        starspot_rf = self.config.model.global_input_length
        global_rf = max(gran_rf, astero_rf, starspot_rf)

        cadence_hours = self.config.data.kepler_cadence_days * 24.0
        return {
            "local_timesteps": float(local_rf),
            "global_timesteps": float(global_rf),
            "local_hours": float(local_rf * cadence_hours),
            "global_hours": float(global_rf * cadence_hours),
        }

    def to_serializable_config(self) -> dict[str, Any]:
        """Return config as nested dictionary for checkpoint metadata."""
        return asdict(self.config)
