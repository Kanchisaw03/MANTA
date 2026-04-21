"""Tests for MANTA model components and full forward behavior."""

from __future__ import annotations

import numpy as np
import torch

from manta.data.frequency_decomposer import FrequencyDecomposer
from manta.data.preprocessor import phase_fold
from manta.models.components.frequency_processor import ParallelFrequencyProcessor
from manta.models.components.physics_output import PhysicsConstrainedOutput
from manta.models.components.position_aware_attention import PositionAwareAttention
from manta.models.components.symmetric_encoder import PalindromicConv1d
from manta.models.manta import MANTA
from manta.utils.config import MANTAConfig


def test_palindromic_symmetry() -> None:
    layer = PalindromicConv1d(in_channels=1, out_channels=4, kernel_size=7)
    x = torch.randn(2, 1, 64)
    _ = layer(x)

    kernel = layer._build_full_kernel().detach()
    flipped = torch.flip(kernel, dims=(-1,))
    assert torch.allclose(kernel, flipped, atol=1.0e-7)


def test_output_bounds() -> None:
    head = PhysicsConstrainedOutput(in_features=16)
    x = torch.randn(32, 16)
    y = head(x)

    assert torch.all(y >= 1.0e-6)
    assert torch.all(y <= 1.0 - 1.0e-6)


def test_attention_limb_darkening() -> None:
    d_model = 32
    attn = PositionAwareAttention(d_model=d_model, n_heads=4, stellar_radius_bins=64)
    x = torch.randn(2, 128, d_model)
    stellar = torch.tensor([[5777.0, 4.4, 0.0, 20.0], [5200.0, 4.2, -0.2, 15.0]])

    _ = attn(x, stellar)
    maps = attn.get_last_attention()
    assert maps is not None

    mean_attention = maps.mean(dim=(0, 1)).detach().cpu().numpy()
    diag = np.diag(mean_attention)

    positions = np.linspace(-1.0, 1.0, diag.size)
    r = np.clip(np.abs(positions), 0.0, 0.999999)
    mu = np.sqrt(1.0 - r**2)
    intensity = 1.0 - 0.4 * (1.0 - mu) - 0.25 * (1.0 - mu) ** 2

    corr = np.corrcoef(diag, intensity)[0, 1]
    assert corr > 0.1


def test_attention_changes_with_impact_parameter() -> None:
    d_model = 32
    attn = PositionAwareAttention(d_model=d_model, n_heads=4, stellar_radius_bins=64, dropout=0.0)
    attn.eval()

    torch.manual_seed(7)
    x_single = torch.randn(1, 128, d_model)
    x = x_single.repeat(2, 1, 1)
    stellar = torch.tensor(
        [
            [5777.0, 4.4, 0.0, 20.0, 0.1],
            [5777.0, 4.4, 0.0, 20.0, 0.9],
        ],
        dtype=torch.float32,
    )

    out = attn(x, stellar)
    assert not torch.allclose(out[0], out[1], atol=1.0e-6)

    maps = attn.get_last_attention()
    assert maps is not None
    mean_abs_diff = torch.mean(torch.abs(maps[0] - maps[1])).item()
    assert mean_abs_diff > 1.0e-5


def test_parallel_frequency_branch_selectivity() -> None:
    processor = ParallelFrequencyProcessor(granulation_channels=8, astero_channels=8, starspot_channels=8)
    processor.eval()

    t = torch.linspace(0.0, 1.0, 512)
    signal = torch.sin(2.0 * torch.pi * 7.0 * t).view(1, 1, -1)
    zeros = torch.zeros_like(signal)

    gran_input = torch.cat([signal, zeros, zeros], dim=1)
    astero_input = torch.cat([zeros, signal, zeros], dim=1)

    with torch.no_grad():
        out_gran = processor(gran_input)
        out_astero = processor(astero_input)

    assert not torch.allclose(out_gran, out_astero, atol=1.0e-6)
    feature_maps = processor.get_feature_maps()
    assert set(feature_maps.keys()) == {"granulation", "asteroseismology", "starspot", "fused"}


def test_manta_realistic_transit_integration() -> None:
    config = MANTAConfig()
    model = MANTA(config)
    model.eval()

    cadence_days = config.data.kepler_cadence_days
    time = np.arange(config.model.global_input_length, dtype=np.float64) * cadence_days

    period = 12.0
    t0 = float(np.median(time))
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5

    rng = np.random.default_rng(1234)
    starspot = 0.0025 * np.sin(2.0 * np.pi * time / 14.0)
    granulation = 0.0012 * np.sin(2.0 * np.pi * 2.3 * time)
    astero = 0.0009 * np.sin(2.0 * np.pi * 9.0 * time)
    transit = 1.0 - 0.008 * np.exp(-0.5 * (phase / 0.015) ** 2)
    noise = rng.normal(0.0, 2.0e-4, size=time.size)
    flux = (1.0 + starspot + granulation + astero + noise) * transit

    decomposer = FrequencyDecomposer(diagnostics_dir="outputs/test_diagnostics")
    bands = decomposer.decompose(flux=flux, time=time, cadence_days=cadence_days)

    folded = phase_fold(time=time, flux=flux, period=period, t0=t0, duration_hours=4.0)
    gran_folded = phase_fold(time=time, flux=bands.granulation, period=period, t0=t0, duration_hours=4.0)
    astero_folded = phase_fold(time=time, flux=bands.asteroseismology, period=period, t0=t0, duration_hours=4.0)
    star_folded = phase_fold(time=time, flux=bands.starspot, period=period, t0=t0, duration_hours=4.0)

    batch = {
        "global_view": torch.tensor(folded["global_view"], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        "local_view": torch.tensor(folded["local_view"], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        "freq_bands": torch.tensor(
            np.stack(
                [
                    gran_folded["global_view"],
                    astero_folded["global_view"],
                    star_folded["global_view"],
                ],
                axis=0,
            ),
            dtype=torch.float32,
        ).unsqueeze(0),
        "stellar_params": torch.tensor([[5777.0, 4.4, 0.0, 21.0, 0.35]], dtype=torch.float32),
        "label": torch.tensor([1.0], dtype=torch.float32),
    }

    with torch.no_grad():
        pred = model(batch)

    assert pred.shape == (1, 1)
    assert torch.isfinite(pred).all()
    assert float(pred.min()) >= 1.0e-6
    assert float(pred.max()) <= 1.0 - 1.0e-6


def test_manta_forward_pass() -> None:
    config = MANTAConfig()
    model = MANTA(config)

    batch_size = 4
    batch = {
        "global_view": torch.randn(batch_size, 1, config.model.global_input_length),
        "local_view": torch.randn(batch_size, 1, config.model.local_input_length),
        "freq_bands": torch.randn(batch_size, 3, config.model.global_input_length),
        "stellar_params": torch.tensor(
            [[5777.0, 4.4, 0.0, 22.0, 0.2]] * batch_size,
            dtype=torch.float32,
        ),
        "label": torch.randint(0, 2, (batch_size,), dtype=torch.float32),
    }

    y = model(batch)
    assert y.shape == (batch_size, 1)


def test_gradient_flow() -> None:
    config = MANTAConfig()
    model = MANTA(config)

    batch_size = 4
    batch = {
        "global_view": torch.randn(batch_size, 1, config.model.global_input_length),
        "local_view": torch.randn(batch_size, 1, config.model.local_input_length),
        "freq_bands": torch.randn(batch_size, 3, config.model.global_input_length),
        "stellar_params": torch.tensor(
            [[5777.0, 4.4, 0.0, 22.0, 0.2]] * batch_size,
            dtype=torch.float32,
        ),
        "label": torch.randint(0, 2, (batch_size,), dtype=torch.float32),
    }

    out = model(batch)
    loss = out.mean()
    loss.backward()

    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing
