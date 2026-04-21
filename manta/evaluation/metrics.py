"""Evaluation metrics and statistical analysis utilities.

This module provides publication-ready metric aggregation, significance tests,
bootstrap uncertainty, and parameter-binned performance breakdowns.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

EPS_METRIC: float = 1.0e-12


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += float(mask.sum()) / max(n, 1) * abs(acc - conf)
    return float(ece)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute standard binary-classification and calibration metrics.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary labels.
    y_pred_proba : numpy.ndarray
        Predicted probabilities.
    threshold : float
        Classification threshold for hard-label metrics.

    Returns
    -------
    dict[str, float]
        AUC-ROC, AUC-PR, F1, precision, recall, average precision, calibration error.
    """
    y_true = np.asarray(y_true, dtype=np.int64).flatten()
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64).flatten()

    if y_true.size == 0:
        return {
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "average_precision": 0.0,
            "calibration_error": 0.0,
        }

    y_pred_label = (y_pred_proba >= threshold).astype(np.int64)

    try:
        auc_roc = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        auc_roc = 0.5

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = float(np.trapz(precision_curve, recall_curve))

    metrics = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": float(f1_score(y_true, y_pred_label, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_label, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_pred_proba)),
        "calibration_error": _expected_calibration_error(y_true, y_pred_proba),
    }
    return metrics


def mcnemar_test(model_a_errors: np.ndarray, model_b_errors: np.ndarray) -> dict[str, float]:
    """Run McNemar's test on paired model error indicators.

    Parameters
    ----------
    model_a_errors : numpy.ndarray
        Boolean/int array where 1 indicates model A is wrong.
    model_b_errors : numpy.ndarray
        Boolean/int array where 1 indicates model B is wrong.

    Returns
    -------
    dict[str, float]
        Statistic, p-value, and discordant cell counts.
    """
    a = np.asarray(model_a_errors).astype(bool)
    b = np.asarray(model_b_errors).astype(bool)
    if a.shape != b.shape:
        raise ValueError("model_a_errors and model_b_errors must have the same shape")

    n01 = np.sum((~a) & b)
    n10 = np.sum(a & (~b))
    denom = n01 + n10
    if denom == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": float(n01), "n10": float(n10)}

    chi2_stat = ((abs(n01 - n10) - 1.0) ** 2) / max(denom, EPS_METRIC)
    p_value = 1.0 - float(chi2.cdf(chi2_stat, df=1))
    return {"chi2": float(chi2_stat), "p_value": p_value, "n01": float(n01), "n10": float(n10)}


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    ci: float,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for an arbitrary metric.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth labels.
    y_pred : numpy.ndarray
        Predictions or probabilities.
    metric_fn : Callable
        Metric function accepting ``(y_true, y_pred)``.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level in (0,1), e.g. 0.95.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict[str, float]
        Mean metric and lower/upper confidence bounds.
    """
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have same length")

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    scores = np.zeros(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = float(metric_fn(y_true[idx], y_pred[idx]))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(scores, alpha))
    upper = float(np.quantile(scores, 1.0 - alpha))
    return {"mean": float(np.mean(scores)), "lower": lower, "upper": upper}


def per_class_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    planet_params: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Break down model performance by physical planet-parameter bins.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary labels.
    y_pred : numpy.ndarray
        Predicted probabilities.
    planet_params : pandas.DataFrame
        Must include columns for radius, period, and impact parameter.
    threshold : float, optional
        Hard-label threshold for precision/recall/F1.

    Returns
    -------
    pandas.DataFrame
        Binned performance table for paper reporting.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    if len(planet_params) != len(y_true):
        raise ValueError("planet_params length must match y_true")

    df = planet_params.copy()
    if "rp_rs" not in df.columns:
        if "planet_radius" in df.columns:
            df["rp_rs"] = df["planet_radius"].astype(float)
        else:
            raise KeyError("planet_params requires 'rp_rs' or 'planet_radius'")
    if "period_days" not in df.columns:
        if "period" in df.columns:
            df["period_days"] = df["period"].astype(float)
        else:
            raise KeyError("planet_params requires 'period_days' or 'period'")
    if "impact_parameter" not in df.columns:
        if "impact" in df.columns:
            df["impact_parameter"] = df["impact"].astype(float)
        else:
            raise KeyError("planet_params requires 'impact_parameter' or 'impact'")

    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["y_hat"] = (y_pred >= threshold).astype(int)

    bin_specs = {
        "radius_bin": ("rp_rs", [0.0, 0.03, 0.06, 0.1, 0.2]),
        "period_bin": ("period_days", [0.0, 10.0, 50.0, 200.0, np.inf]),
        "impact_bin": ("impact_parameter", [0.0, 0.3, 0.6, 0.9, 1.0]),
    }

    rows: list[dict[str, float | str]] = []
    for bin_name, (column, edges) in bin_specs.items():
        categories = pd.cut(df[column], bins=edges, include_lowest=True)
        grouped = df.groupby(categories)

        for group_label, group in grouped:
            if group.empty:
                continue
            g_true = group["y_true"].to_numpy(dtype=int)
            g_prob = group["y_pred"].to_numpy(dtype=float)
            g_hat = group["y_hat"].to_numpy(dtype=int)
            try:
                auc = float(roc_auc_score(g_true, g_prob)) if len(np.unique(g_true)) > 1 else np.nan
            except ValueError:
                auc = np.nan

            rows.append(
                {
                    "group_type": bin_name,
                    "group": str(group_label),
                    "count": float(len(group)),
                    "positives": float(g_true.sum()),
                    "auc_roc": auc,
                    "precision": float(precision_score(g_true, g_hat, zero_division=0)),
                    "recall": float(recall_score(g_true, g_hat, zero_division=0)),
                    "f1": float(f1_score(g_true, g_hat, zero_division=0)),
                    "average_precision": float(average_precision_score(g_true, g_prob)),
                }
            )

    return pd.DataFrame(rows)
