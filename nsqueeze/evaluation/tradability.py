"""Tradability. Deflated Sharpe ratio using the *true* number of trials.

Bailey, D. H. & López de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management.

All Sharpe ratios here are per-period (not annualised) and must be computed from
total returns including dividends/distributions.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, float)
    return float(r.mean() / r.std(ddof=1))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] across ``n_trials`` independent zero-skill trials (the SR₀ hurdle)."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def probabilistic_sharpe(sr: float, sr_benchmark: float, t: int, skew: float, kurt: float) -> float:
    """PSR: P(true SR > benchmark) given T observations, skewness and (non-excess) kurtosis."""
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(t - 1) / denom))


def deflated_sharpe(returns: np.ndarray, *, n_trials: int, trial_sr_variance: float) -> float:
    """DSR = PSR evaluated at the expected maximum Sharpe of ``n_trials`` trials.

    ``trial_sr_variance`` is the cross-sectional variance of the Sharpe ratios of
    *all* trials actually run — log every specification you try.
    """
    r = np.asarray(returns, float)
    sr = sharpe(r)
    sr0 = expected_max_sharpe(n_trials, trial_sr_variance)
    return probabilistic_sharpe(
        sr, sr0, len(r), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    )
