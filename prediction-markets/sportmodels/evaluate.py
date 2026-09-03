"""Scoring rules and calibration.

Accuracy is close to useless here. A model that says 60% and a model that says
95% can have identical accuracy while only one of them is bettable. These are
the metrics that actually distinguish them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def log_loss(p, y) -> float:
    """Mean negative log likelihood. Lower is better. The main metric."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    """Mean squared error on probabilities. Lower is better."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def accuracy(p, y, threshold: float = 0.5) -> float:
    return float(np.mean((np.asarray(p, float) > threshold) == np.asarray(y, bool)))


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual rate, bucketed. The honesty check.

    If you say 70% and it happens 55% of the time, you do not have an edge,
    you have a leak.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bucket": f"{edges[b]:.2f}-{edges[b + 1]:.2f}",
                "n": int(m.sum()),
                "predicted": float(p[m].mean()),
                "actual": float(y[m].mean()),
                "diff": float(y[m].mean() - p[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(p, y, bins: int = 10) -> float:
    """Sample-weighted mean |predicted - actual| across buckets."""
    tab = calibration_table(p, y, bins)
    if tab.empty:
        return float("nan")
    w = tab["n"] / tab["n"].sum()
    return float((w * tab["diff"].abs()).sum())


def summarize(name: str, p, y) -> dict:
    return {
        "model": name,
        "n": int(len(y)),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "accuracy": accuracy(p, y),
        "ece": expected_calibration_error(p, y),
    }


def compare(results: list[dict], baseline: str) -> pd.DataFrame:
    """Table of models with log-loss delta vs a named baseline.

    Negative `vs_baseline` means better than the baseline.
    """
    df = pd.DataFrame(results)
    base = df.loc[df["model"] == baseline, "log_loss"].iloc[0]
    df["vs_baseline"] = df["log_loss"] - base
    return df.sort_values("log_loss").reset_index(drop=True)


def season_walk_forward(seasons, first_test_season: int):
    """Yield (train_mask, test_mask) per season, training only on the past.

    This is the only backtest shape that means anything. Any split that lets
    the model see future seasons will flatter you and then lose money.
    """
    seasons = np.asarray(seasons)
    for s in sorted(np.unique(seasons[seasons >= first_test_season])):
        yield s, seasons < s, seasons == s
