"""Monte Carlo validation of the inference protocol.

Each experiment puts one claim of the rigour protocol to a test where the truth is known:

E1  Measurement   An estimator can track *levels* well and still fail on *changes*.
E2  Nulls         With autocorrelated series and no true link, naive permutation tests
                  over-reject; circular-shift placebos hold their nominal size.
E3  Selection     The best of N zero-skill strategies looks significant; the deflated
                  Sharpe ratio restores the nominal false-positive rate.
E4  Few events    When a regime effect arrives as a handful of episodes, i.i.d. OLS
                  t-statistics over-reject; episode-clustered inference does not.

All experiments are seeded and vectorised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from nsqueeze.evaluation.tradability import (
    expected_max_sharpe,
    probabilistic_sharpe,
)

SEED = 20261012  # project start date; fixed so every rendered number is reproducible


def ar1(rng: np.random.Generator, n: int, t: int, phi: float, scale: float = 1.0) -> np.ndarray:
    """``n`` independent stationary AR(1) paths of length ``t`` (rows = paths)."""
    eps = rng.normal(0.0, scale, (n, t))
    x = np.empty((n, t))
    x[:, 0] = eps[:, 0] / np.sqrt(1 - phi**2)
    for i in range(1, t):
        x[:, i] = phi * x[:, i - 1] + eps[:, i]
    return x


def _rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


# ── E1 · levels flatter, changes decide ──────────────────────────────────────


def e1_levels_vs_changes(
    noise_ratios=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    *,
    n_paths: int = 500,
    t: int = 260,
    seed: int = SEED,
) -> pd.DataFrame:
    """Truth = slow seasonal inventory cycle + random walk (weekly, 5 years).

    The estimator adds white measurement noise whose s.d. is ``k`` × σ(true weekly
    change). Reports the median level correlation and median change-SNR per ``k``.
    A Cushing storage example sits near k ≈ 13 (20 Mbbl error vs 1.5 Mbbl moves).
    """
    rng = np.random.default_rng(seed)
    w = np.arange(t)
    cycle = 40.0 * np.sin(2 * np.pi * w / 52)
    walk = np.cumsum(rng.normal(0, 1.5, (n_paths, t)), axis=1)
    truth = 400 + cycle + walk
    d_truth = np.diff(truth, axis=1)
    sd_change = d_truth.std(axis=1, ddof=1, keepdims=True)
    rows = []
    for k in noise_ratios:
        est = truth + rng.normal(0, 1, truth.shape) * k * sd_change
        d_est = np.diff(est, axis=1)
        rmse = np.sqrt(((d_est - d_truth) ** 2).mean(axis=1))
        rows.append(
            {
                "noise_ratio": k,
                "level_corr": float(np.median(_rowcorr(est, truth))),
                "change_corr": float(np.median(_rowcorr(d_est, d_truth))),
                "change_snr": float(np.median(sd_change[:, 0] / rmse)),
            }
        )
    return pd.DataFrame(rows)


# ── E2 · size of permutation vs circular-shift nulls ─────────────────────────


def e2_null_size(
    phis=(0.0, 0.5, 0.8, 0.9, 0.95),
    *,
    n_mc: int = 400,
    n_null: int = 199,
    t: int = 500,
    min_offset: int = 50,
    alpha: float = 0.05,
    seed: int = SEED,
) -> pd.DataFrame:
    """Rejection rate at ``alpha`` when signal and outcome are independent AR(1) paths.

    Naive null: permute the signal (destroys its autocorrelation).
    Circular-shift null: roll the signal by a random offset (preserves it).
    Statistic: |correlation|. A valid test rejects ≈ ``alpha`` of the time.
    """
    rng = np.random.default_rng(seed)
    rows = []
    chunk = 50  # keeps the (draws × null × t) cube near 40 MB
    for phi in phis:
        rej_naive = rej_circ = 0
        for start in range(0, n_mc, chunk):
            m = min(chunk, n_mc - start)
            s, y = ar1(rng, m, t, phi), ar1(rng, m, t, phi)
            obs = np.abs(_rowcorr(s, y))[:, None]
            perm = np.argsort(rng.random((n_null, t)), axis=1)  # destroys autocorrelation
            offs = rng.integers(min_offset, t - min_offset, size=n_null)
            roll = (np.arange(t)[None, :] - offs[:, None]) % t  # preserves it
            naive = np.abs(_rowcorr(s[:, perm], y[:, None, :]))
            circ = np.abs(_rowcorr(s[:, roll], y[:, None, :]))
            p_naive = (1 + (naive >= obs).sum(1)) / (n_null + 1)
            p_circ = (1 + (circ >= obs).sum(1)) / (n_null + 1)
            rej_naive += int((p_naive <= alpha).sum())
            rej_circ += int((p_circ <= alpha).sum())
        rows.append(
            {"phi": phi, "reject_naive": rej_naive / n_mc, "reject_circular": rej_circ / n_mc}
        )
    return pd.DataFrame(rows)


# ── E3 · selection bias and the deflated Sharpe ratio ────────────────────────


def e3_selection(
    n_trials_grid=(1, 5, 20, 100, 500),
    *,
    n_mc: int = 400,
    t: int = 756,
    threshold: float = 0.95,
    true_sharpe_annual: float = 0.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Zero-skill strategies (i.i.d. N(0, 1 %) daily returns, three years each).

    For each Monte Carlo draw, ``N`` strategies are tried and the best Sharpe is kept.
    Reports how often the winner is declared significant by the probabilistic Sharpe
    ratio (PSR > ``threshold`` against zero) versus the deflated Sharpe ratio (DSR,
    PSR against the expected maximum of N trials). Under the null both should be ≈ 5 %.

    With ``true_sharpe_annual`` > 0, trial 0 has genuine skill and the other N − 1 do
    not; the "declared" rates then measure power (only a correct winner counts).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_trials_grid:
        best_sr, psr, dsr = np.empty(n_mc), np.empty(n_mc), np.empty(n_mc)
        for i in range(n_mc):  # one research project = N trials, keep the winner
            r = rng.normal(0, 0.01, (n, t))
            r[0] += true_sharpe_annual * 0.01 / np.sqrt(252)
            sr = r.mean(1) / r.std(1, ddof=1)
            k = int(sr.argmax())
            correct = true_sharpe_annual == 0 or k == 0
            w = r[k]
            skew, kurt = float(stats.skew(w)), float(stats.kurtosis(w, fisher=False))
            hurdle = expected_max_sharpe(n, float(sr.var(ddof=1))) if n > 1 else 0.0
            best_sr[i] = sr[k]
            psr[i] = probabilistic_sharpe(sr[k], 0.0, t, skew, kurt) if correct else 0.0
            dsr[i] = probabilistic_sharpe(sr[k], hurdle, t, skew, kurt) if correct else 0.0
        rows.append(
            {
                "n_trials": n,
                "median_best_sharpe_annual": float(np.median(best_sr) * np.sqrt(252)),
                "declared_psr": float((psr > threshold).mean()),
                "declared_dsr": float((dsr > threshold).mean()),
            }
        )
    return pd.DataFrame(rows)


# ── E4 · few independent events ──────────────────────────────────────────────


def e4_few_events(
    n_episodes_grid=(2, 4, 8, 16, 32),
    *,
    n_mc: int = 300,
    days_per_episode: int = 60,
    episode_sd: float = 0.004,
    alpha: float = 0.05,
    seed: int = SEED,
) -> pd.DataFrame:
    """Daily returns contain an episode-level shock unrelated to the regime label.

    Each episode carries a common daily mean drawn from N(0, ``episode_sd``) plus
    i.i.d. noise; half the episodes are labelled "stress". There is **no** regime
    effect, so a valid test of the stress dummy rejects ≈ ``alpha``. Compares i.i.d.
    OLS standard errors with standard errors clustered by episode (t-distribution
    with G − 1 degrees of freedom).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in n_episodes_grid:
        rej_iid = rej_cl = 0
        groups = np.repeat(np.arange(g), days_per_episode)
        label = (groups % 2 == 0).astype(float)
        x = sm.add_constant(label)
        for _ in range(n_mc):
            mu = rng.normal(0, episode_sd, g)[groups]
            y = mu + rng.normal(0, 0.01, g * days_per_episode)
            fit = sm.OLS(y, x).fit()
            rej_iid += fit.pvalues[1] < alpha
            cl = sm.OLS(y, x).fit(
                cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True}
            )
            t_cl = cl.params[1] / cl.bse[1]
            p_cl = 2 * stats.t.sf(abs(t_cl), df=g - 1)
            rej_cl += p_cl < alpha
        # With G = 2 the regime dummy is collinear with the cluster: the clustered
        # variance is identically zero and the test does not exist.
        rows.append(
            {
                "episodes": g,
                "reject_iid": rej_iid / n_mc,
                "reject_clustered": np.nan if g <= 2 else rej_cl / n_mc,
            }
        )
    return pd.DataFrame(rows)


def run_all(seed: int = SEED) -> dict[str, pd.DataFrame]:
    return {
        "e1": e1_levels_vs_changes(seed=seed),
        "e2": e2_null_size(seed=seed),
        "e3_null": e3_selection(seed=seed),
        "e3_power": e3_selection(true_sharpe_annual=1.0, seed=seed),
        "e4": e4_few_events(seed=seed),
    }
