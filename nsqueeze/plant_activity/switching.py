"""Switching model: when does a plant stop and restart?

Each plant-month is on or off. The decision follows the margin m(t) of the month before:

    P(on → off) = σ((c − m) / τ)        c: stop threshold
    P(off → on) = σ((m − r) / τ)        r: restart threshold, r − c: the hysteresis band

The observations come at different frequencies. Verified emissions are annual: they depend
on how many months the plant ran that year. I therefore add that count to the state,
(on/off, months on so far this year), 2 × 13 states, so that the annual figure is an ordinary
observation in December and the forward-backward pass stays exact. Dated stops are hard
constraints. The pass is vectorised over a grid of (c, r), so the likelihood surface comes
out in one go.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

K = 13  # months on so far this year: 0..12


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def forward(
    margin: np.ndarray,
    month: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    *,
    tau: float,
    annual: dict[int, float],
    year: np.ndarray,
    q: float,
    sigma: float,
    off: np.ndarray,
    p_on0: float = 0.9,
    keep: bool = False,
):
    """Log-likelihood for each (c, r) pair; with ``keep`` also the scaled forward messages.

    margin, month, year, off: one value per month (off = True forces the plant off).
    annual: year → verified emissions; q: emissions of one full month; sigma: noise as a
    share of a full year.
    """
    g = np.broadcast(c, r).shape
    on = np.zeros((*g, K))
    of = np.zeros((*g, K))
    on[..., 1] = p_on0
    of[..., 0] = 1 - p_on0
    loglik = np.zeros(g)
    msgs = []
    sd = sigma * 12 * q
    for t in range(len(margin)):
        if t > 0:
            m = margin[t - 1]
            ps, pr = _sigmoid((c - m) / tau)[..., None], _sigmoid((m - r) / tau)[..., None]
            if month[t] == 1:  # new year: the count restarts
                a_on, a_of = on.sum(-1, keepdims=True), of.sum(-1, keepdims=True)
                on = np.zeros_like(on)
                of = np.zeros_like(of)
                on[..., 1:2] = a_on * (1 - ps) + a_of * pr
                of[..., 0:1] = a_on * ps + a_of * (1 - pr)
            else:
                new_on = np.zeros_like(on)
                new_on[..., 1:] = (on * (1 - ps) + of * pr)[..., :-1]
                of = on * ps + of * (1 - pr)
                on = new_on
        if off[t]:
            on = np.zeros_like(on)
        if month[t] == 12 and np.isfinite(annual.get(year[t], np.nan)):
            ll = -0.5 * ((annual[year[t]] - q * np.arange(K)) / sd) ** 2
            loglik += ll.max()  # scale out the largest term, so the product cannot underflow
            like = np.exp(ll - ll.max())
            on, of = on * like, of * like
        z = np.maximum(on.sum(-1) + of.sum(-1), 1e-300)
        loglik += np.log(z)
        on, of = on / z[..., None], of / z[..., None]
        if keep:
            msgs.append((on.copy(), of.copy()))
    return (loglik, msgs) if keep else loglik


def p_on(margin, month, c: float, r: float, **kw) -> np.ndarray:
    """Filtered P(on) per month at one (c, r): what the data up to each month imply."""
    _, msgs = forward(margin, month, np.asarray(c), np.asarray(r), keep=True, **kw)
    return np.array([o.sum() for o, _ in msgs])


def fit_plant(
    margin: pd.Series,
    *,
    annual: dict[int, float],
    q: float,
    off: pd.Series,
    tau: float,
    sigma: float,
    grid: np.ndarray,
) -> dict:
    """Grid MLE of (c, r) with r ≥ c, and the log-likelihood surface."""
    c, r = np.meshgrid(grid, grid, indexing="ij")
    kw = dict(
        tau=tau,
        annual=annual,
        year=margin.index.year.to_numpy(),
        q=q,
        sigma=sigma,
        off=off.reindex(margin.index, fill_value=False).to_numpy(),
    )
    ll = forward(margin.to_numpy(), margin.index.month.to_numpy(), c, r, **kw)
    ll = np.where(r >= c, ll, -np.inf)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return {
        "c": grid[i],
        "r": grid[j],
        "loglik": ll[i, j],
        "surface": ll,
        "p_on": p_on(margin.to_numpy(), margin.index.month.to_numpy(), grid[i], grid[j], **kw),
    }
