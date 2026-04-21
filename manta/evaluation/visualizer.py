"""Publication-quality visualization utilities for MANTA experiments.

All plotting functions save both PNG and PDF outputs at 300 DPI to support
paper-ready figure generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.metrics import roc_curve, auc

from manta.models.components.elliptic_activation import EllipticMish

DPI: int = 300
FIGURE_DIR_DEFAULT: str = "outputs/figures"
COLOR_PRIMARY: str = "#1f4e79"
COLOR_SECONDARY: str = "#ff7f0e"
COLOR_TERTIARY: str = "#2ca02c"


def _ensure_output_dir(output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_dual(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_transit_detection(lc: dict[str, Any], prediction: float, ground_truth: int, output_dir: str | Path = FIGURE_DIR_DEFAULT) -> tuple[Path, Path]:
    """Plot one light curve with model prediction and truth annotation."""
    out_dir = _ensure_output_dir(output_dir)
    time = np.asarray(lc["time"], dtype=np.float64)
    flux = np.asarray(lc["flux"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, flux, linewidth=0.8, color=COLOR_PRIMARY)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Relative Flux")
    ax.set_title(f"Transit Detection | Prediction={prediction:.3f} | Truth={ground_truth}")
    ax.grid(alpha=0.2)
    return _save_dual(fig, out_dir, "figure_1_transit_detection")


def plot_frequency_decomposition(flux: np.ndarray, time: np.ndarray, bands: Any, output_dir: str | Path = FIGURE_DIR_DEFAULT) -> tuple[Path, Path]:
    """Plot original flux and three frequency bands (Figure 2)."""
    out_dir = _ensure_output_dir(output_dir)

    if hasattr(bands, "granulation"):
        gran = bands.granulation
        astero = bands.asteroseismology
        star = bands.starspot
    else:
        gran = bands["granulation"]
        astero = bands["asteroseismology"]
        star = bands["starspot"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(time, flux, color=COLOR_PRIMARY, linewidth=0.8)
    axes[0].set_title("Original")
    axes[1].plot(time, gran, color=COLOR_TERTIARY, linewidth=0.8)
    axes[1].set_title("Granulation")
    axes[2].plot(time, astero, color=COLOR_SECONDARY, linewidth=0.8)
    axes[2].set_title("Asteroseismology")
    axes[3].plot(time, star, color="#d62728", linewidth=0.8)
    axes[3].set_title("Starspot")
    axes[3].set_xlabel("Time [days]")
    for ax in axes:
        ax.set_ylabel("Flux")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save_dual(fig, out_dir, "figure_2_frequency_decomposition")


def plot_activation_comparison(output_dir: str | Path = FIGURE_DIR_DEFAULT) -> tuple[Path, Path]:
    """Plot EllipticMish vs ReLU vs Mish activation curves (Figure 3)."""
    out_dir = _ensure_output_dir(output_dir)

    x = np.linspace(-8.0, 8.0, 800)
    x_t = __import__("torch").tensor(x, dtype=__import__("torch").float32)

    elliptic = EllipticMish(alpha_init=0.1)
    y_elliptic = elliptic(x_t).detach().numpy()
    y_relu = np.maximum(x, 0.0)
    y_mish = (x_t * __import__("torch").tanh(__import__("torch").nn.functional.softplus(x_t))).detach().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_elliptic, label="EllipticMish", color=COLOR_PRIMARY, linewidth=2.0)
    ax.plot(x, y_mish, label="Mish", color=COLOR_SECONDARY, linewidth=1.5)
    ax.plot(x, y_relu, label="ReLU", color="#555555", linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Activation Comparison")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return _save_dual(fig, out_dir, "figure_3_activation_comparison")


def plot_attention_weights(attention_maps: np.ndarray, stellar_disk: np.ndarray, output_dir: str | Path = FIGURE_DIR_DEFAULT) -> tuple[Path, Path]:
    """Overlay attention intensity over a limb-darkened stellar disk (Figure 4)."""
    out_dir = _ensure_output_dir(output_dir)

    attn = np.asarray(attention_maps, dtype=np.float64)
    if attn.ndim == 4:
        attn = attn.mean(axis=(0, 1))
    elif attn.ndim == 3:
        attn = attn.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(stellar_disk, cmap="gray", alpha=0.8)
    ax.imshow(attn, cmap="inferno", alpha=0.6)
    ax.set_title("Position-Aware Attention on Limb-Darkened Disk")
    ax.axis("off")
    return _save_dual(fig, out_dir, "figure_4_attention_weights")


def plot_roc_curves(results_dict: dict[str, dict[str, np.ndarray]], output_dir: str | Path = FIGURE_DIR_DEFAULT) -> tuple[Path, Path]:
    """Plot ROC curves for multiple models on one axis (Figure 5)."""
    out_dir = _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("tab10", n_colors=max(3, len(results_dict)))

    for idx, (name, payload) in enumerate(results_dict.items()):
        y_true = np.asarray(payload["y_true"], dtype=np.int64)
        y_prob = np.asarray(payload["y_pred"], dtype=np.float64)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=palette[idx])

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    return _save_dual(fig, out_dir, "figure_5_roc_curves")


def plot_ablation_heatmap(ablation_df: pd.DataFrame, output_dir: str | Path = FIGURE_DIR_DEFAULT) -> tuple[Path, Path]:
    """Plot ablation metric heatmap (Figure 6)."""
    out_dir = _ensure_output_dir(output_dir)

    pivot = ablation_df.pivot_table(index="variant", values=["auc_roc", "f1", "average_precision"], aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, cbar=True)
    ax.set_title("Ablation Performance Heatmap")
    return _save_dual(fig, out_dir, "figure_6_ablation_heatmap")


def plot_per_planet_size_performance(results_df: pd.DataFrame, output_dir: str | Path = FIGURE_DIR_DEFAULT) -> tuple[Path, Path]:
    """Plot performance by planet-size bins (Figure 7)."""
    out_dir = _ensure_output_dir(output_dir)

    subset = results_df[results_df["group_type"] == "radius_bin"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=subset, x="group", y="auc_roc", ax=ax, color=COLOR_PRIMARY)
    ax.set_xlabel("Rp/Rs Bin")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Performance by Planet Size")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2, axis="y")
    return _save_dual(fig, out_dir, "figure_7_planet_size_performance")


def plot_calibration_curves(y_true: np.ndarray, y_pred_dict: dict[str, np.ndarray], output_dir: str | Path = FIGURE_DIR_DEFAULT) -> tuple[Path, Path]:
    """Plot reliability curves for multiple models (Figure 8)."""
    out_dir = _ensure_output_dir(output_dir)

    y_true = np.asarray(y_true, dtype=np.int64)
    bins = np.linspace(0.0, 1.0, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.0, label="Perfect calibration")

    for name, y_pred in y_pred_dict.items():
        y_pred = np.asarray(y_pred, dtype=np.float64)
        idx = np.clip(np.digitize(y_pred, bins) - 1, 0, 9)
        emp = np.zeros(10, dtype=np.float64)
        pred_mean = np.zeros(10, dtype=np.float64)

        for i in range(10):
            mask = idx == i
            if np.any(mask):
                emp[i] = y_true[mask].mean()
                pred_mean[i] = y_pred[mask].mean()
            else:
                emp[i] = np.nan
                pred_mean[i] = centers[i]

        ax.plot(pred_mean, emp, marker="o", linewidth=1.5, label=name)

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Empirical Frequency")
    ax.set_title("Calibration Curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return _save_dual(fig, out_dir, "figure_8_calibration_curves")
