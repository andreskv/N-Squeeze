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


def p_on_smoothed(margin, month, c: float, r: float, **kw) -> np.ndarray:
    """Smoothed P(on) per month at one (c, r): what all the data imply (forward-backward).

    The backward pass mirrors the forward transitions; the observations of month t + 1 (the
    forced-off constraint and the December total) weight the backward message of month t.
    """
    _, msgs = forward(margin, month, np.asarray(c), np.asarray(r), keep=True, **kw)
    off, annual, year, q, sigma = kw["off"], kw["annual"], kw["year"], kw["q"], kw["sigma"]
    tau = kw["tau"]
    n = len(margin)
    b_on, b_of = np.ones(K), np.ones(K)
    out = np.empty(n)
    for t in range(n - 1, -1, -1):
        a_on, a_of = msgs[t]
        w_on, w_of = (a_on * b_on).sum(), (a_of * b_of).sum()
        out[t] = w_on / (w_on + w_of)
        if t == 0:
            break
        e_on = np.zeros(K) if off[t] else np.ones(K)  # observations of month t
        e_of = np.ones(K)
        if month[t] == 12 and np.isfinite(annual.get(year[t], np.nan)):
            ll = -0.5 * ((annual[year[t]] - q * np.arange(K)) / (sigma * 12 * q)) ** 2
            like = np.exp(ll - ll.max())
            e_on, e_of = e_on * like, e_of * like
        v_on, v_of = e_on * b_on, e_of * b_of
        m = margin[t - 1]
        ps, pr = _sigmoid((c - m) / tau), _sigmoid((m - r) / tau)
        if month[t] == 1:  # the count restarts: every k of December leads to k = 0 or 1
            b_on = np.full(K, (1 - ps) * v_on[1] + ps * v_of[0])
            b_of = np.full(K, pr * v_on[1] + (1 - pr) * v_of[0])
        else:
            up = np.append(v_on[1:], 0.0)  # on next month: k + 1
            b_on, b_of = (1 - ps) * up + ps * v_of, pr * up + (1 - pr) * v_of
        s = max(b_on.max(), b_of.max(), 1e-300)
        b_on, b_of = b_on / s, b_of / s
    return out


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
        "p_on_smoothed": p_on_smoothed(
            margin.to_numpy(), margin.index.month.to_numpy(), grid[i], grid[j], **kw
        ),
    }


def _log_prior(c, r, z, theta):
    """log N(c; a + b z, s_c) + log N(r - c; m_band, s_band), on the (c, r) grid; r < c excluded."""
    a, b, log_sc, m_band, log_sb = theta
    sc, sb = np.exp(log_sc), np.exp(log_sb)
    lp = (
        -0.5 * ((c - a - b * z) / sc) ** 2
        - np.log(sc)
        - 0.5 * ((r - c - m_band) / sb) ** 2
        - np.log(sb)
    )
    return np.where(r >= c, lp, -np.inf)


def empirical_bayes(
    surfaces: list[np.ndarray], z: np.ndarray, grid: np.ndarray, theta0, free=(True,) * 5
) -> dict:
    """Shared prior on the thresholds, fitted by maximum marginal likelihood over all plants.

    ``surfaces``: one log-likelihood grid over (c, r) per plant (from ``fit_plant``).
    ``z``: one covariate per plant (centred), 0 where unknown. The marginal likelihood of a
    plant is the sum over the grid of likelihood × prior, so it is exact on the grid.
    ``theta0`` = (a, b, log s_c, band mean, log s_band); entries with ``free`` False stay at their
    starting value (the band is weakly identified when few restarts are observed).
    Returns the hyperparameters and, per plant, the posterior mean of c and r.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    c, r = np.meshgrid(grid, grid, indexing="ij")
    lls = [np.where(np.isfinite(s), s, -np.inf) for s in surfaces]

    scale = np.array([100.0, 100.0, 1.0, 100.0, 1.0])  # EUR/t terms optimised in units of 100

    free = np.asarray(free, bool)
    base = np.asarray(theta0, float) / scale

    def full(x):
        y = base.copy()
        y[free] = x
        return y * scale

    def neg(x):
        theta = full(x)
        tot = 0.0
        for ll, zi in zip(lls, z, strict=True):
            lp = _log_prior(c, r, zi, theta)
            tot += logsumexp(ll + lp) - logsumexp(lp)  # prior normalised on the grid
        return -tot

    x0 = base[free]
    simplex = np.vstack([x0, x0 + np.eye(len(x0))])  # first steps of 100 EUR/t, even from 0
    res = minimize(
        neg,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4, "initial_simplex": simplex},
    )
    theta = full(res.x)
    post = []
    for ll, zi in zip(lls, z, strict=True):
        lw = ll + _log_prior(c, r, zi, theta)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        post.append(((w * c).sum(), (w * r).sum()))
    return {
        "theta": theta,
        "loglik": -res.fun,
        "posterior_cr": np.array(post),
        "converged": res.success,
    }
