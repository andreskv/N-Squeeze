"""Measurement. Score estimators on *changes*, not levels.

The lesson of the inspiration video: a satellite estimate off by
≈ 20 Mbbl is useless when the market reacts to ≈ 1.5 Mbbl weekly changes, however
good its long-run trend looks. SNR = σ(true Δ) / RMSE(estimated Δ − true Δ);
a ratio near or below 1 kills the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def change_snr(estimate: pd.Series, truth: pd.Series, *, periods: int = 1) -> float:
    """Signal-to-noise ratio of an estimator, computed on aligned ``periods``-step changes."""
    est, tru = estimate.align(truth, join="inner")
    d_est, d_tru = est.diff(periods), tru.diff(periods)
    ok = d_est.notna() & d_tru.notna()
    if ok.sum() < 3:
        raise ValueError("need at least 3 aligned changes")
    rmse = float(np.sqrt(np.mean((d_est[ok] - d_tru[ok]) ** 2)))
    sigma = float(d_tru[ok].std(ddof=1))
    return np.inf if rmse == 0 else sigma / rmse


def level_vs_change_report(estimate: pd.Series, truth: pd.Series) -> dict[str, float]:
    """Contrast the flattering level correlation with the decisive change-based SNR."""
    est, tru = estimate.align(truth, join="inner")
    return {
        "level_corr": float(est.corr(tru)),
        "change_corr": float(est.diff().corr(tru.diff())),
        "change_snr": change_snr(est, tru),
    }
