"""Proper null distributions.

Circular-shift placebos keep a signal's own autocorrelation and average exposure
while breaking its alignment with outcomes — unlike coin-flip portfolios, which
destroy both and make any persistent signal look skilful. Block bootstrap gives
intervals that respect serial dependence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaceboResult:
    observed: float
    null: np.ndarray
    p_value: float  # one-sided: share of placebo stats >= observed (with +1 correction)


def circular_shift_placebo(
    signal: np.ndarray,
    outcome: np.ndarray,
    stat: Callable[[np.ndarray, np.ndarray], float],
    *,
    n: int = 2000,
    min_offset: int = 20,
    rng: np.random.Generator | int | None = None,
) -> PlaceboResult:
    """Null of ``stat(signal, outcome)`` under random circular shifts of the signal.

    ``min_offset`` excludes shifts so small that the placebo is still nearly aligned
    (set it to at least the signal's decorrelation time).
    """
    s, y = np.asarray(signal, float), np.asarray(outcome, float)
    if s.shape != y.shape:
        raise ValueError("signal and outcome must have the same shape")
    t = len(s)
    if t <= 2 * min_offset:
        raise ValueError("series too short for the requested min_offset")
    g = np.random.default_rng(rng)
    offsets = g.integers(min_offset, t - min_offset, size=n)
    null = np.array([stat(np.roll(s, k), y) for k in offsets])
    obs = float(stat(s, y))
    p = (1 + np.sum(null >= obs)) / (n + 1)
    return PlaceboResult(obs, null, float(p))


def moving_block_indices(
    t: int, block: int, *, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Index array for one moving-block bootstrap resample of length ``t``."""
    if not 1 <= block <= t:
        raise ValueError("block length must be in [1, t]")
    g = np.random.default_rng(rng)
    n_blocks = int(np.ceil(t / block))
    starts = g.integers(0, t - block + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:t]


def block_bootstrap_ci(
    x: np.ndarray,
    stat: Callable[[np.ndarray], float],
    *,
    block: int,
    n: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float]:
    x = np.asarray(x, float)
    g = np.random.default_rng(rng)
    draws = np.array([stat(x[moving_block_indices(len(x), block, rng=g)]) for _ in range(n)])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
