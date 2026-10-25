import numpy as np
import pandas as pd

from nsqueeze.plant_activity.switching import fit_plant


def test_recovers_stop_and_restart_thresholds_from_annual_totals():
    idx = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
    t = np.arange(len(idx))
    margin = pd.Series(150 - 450 * np.exp(-(((t - 80) / 12.0) ** 2)), index=idx)  # one deep dip
    c_true, r_true = -100.0, 50.0
    on, state = [], True
    for i in range(len(idx)):  # deterministic rule on the margin of the month before
        m = margin.iloc[i - 1] if i else margin.iloc[0]
        state = (m > c_true) if state else (m > r_true)
        on.append(state)
    on = pd.Series(on, index=idx)
    q = 1000.0
    annual = {y: q * on[on.index.year == y].sum() for y in range(2016, 2026)}
    grid = np.arange(-300.0, 301.0, 25.0)
    fit = fit_plant(
        margin, annual=annual, q=q, off=pd.Series(False, index=idx), tau=5.0, sigma=0.02, grid=grid
    )
    assert abs(fit["c"] - c_true) <= 50 and abs(fit["r"] - r_true) <= 50
    assert fit["p_on"][on.to_numpy()].mean() > 0.8 and fit["p_on"][~on.to_numpy()].mean() < 0.3


def test_a_missing_annual_value_is_skipped_not_nan():
    idx = pd.date_range("2016-01-01", "2017-12-01", freq="MS")
    margin = pd.Series(100.0, index=idx)
    fit = fit_plant(
        margin,
        annual={2016: 12000.0, 2017: float("nan")},
        q=1000.0,
        off=pd.Series(False, index=idx),
        tau=5.0,
        sigma=0.1,
        grid=np.array([-100.0, 0.0]),
    )
    assert np.isfinite(fit["loglik"])


def test_smoothing_uses_later_years_and_beats_filtering():
    idx = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
    t = np.arange(len(idx))
    margin = pd.Series(150 - 450 * np.exp(-(((t - 80) / 12.0) ** 2)), index=idx)
    on, state = [], True
    for i in range(len(idx)):
        m = margin.iloc[i - 1] if i else margin.iloc[0]
        state = (m > -100.0) if state else (m > 50.0)
        on.append(state)
    on = np.array(on)
    annual = {y: 1000.0 * on[idx.year == y].sum() for y in range(2016, 2026)}
    fit = fit_plant(
        margin,
        annual=annual,
        q=1000.0,
        off=pd.Series(False, index=idx),
        tau=5.0,
        sigma=0.02,
        grid=np.arange(-300.0, 301.0, 25.0),
    )
    err_f = np.abs(fit["p_on"] - on).mean()
    err_s = np.abs(fit["p_on_smoothed"] - on).mean()
    assert err_s <= err_f and err_s < 0.1
    assert np.all((fit["p_on_smoothed"] >= 0) & (fit["p_on_smoothed"] <= 1))


def test_smoothed_matches_brute_force_enumeration():
    import itertools

    from nsqueeze.plant_activity.switching import p_on_smoothed

    idx = pd.date_range("2016-06-01", "2017-07-01", freq="MS")  # 14 months across New Year
    rng = np.random.default_rng(1)
    margin = rng.normal(0, 60, len(idx))
    month, year = idx.month.to_numpy(), idx.year.to_numpy()
    off = np.zeros(len(idx), bool)
    off[9] = True  # March 2017 forced off
    annual, q, sigma, tau, c, r, p0 = {2016: 5200.0}, 1000.0, 0.1, 30.0, -20.0, 40.0, 0.9
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    w = np.zeros(len(idx))
    total = 0.0
    for u in itertools.product((0, 1), repeat=len(idx)):
        if any(u[t] and off[t] for t in range(len(idx))):
            continue
        p = p0 if u[0] else 1 - p0
        for t in range(1, len(idx)):
            ps, pr = sig((c - margin[t - 1]) / tau), sig((margin[t - 1] - r) / tau)
            p *= (1 - ps if u[t] else ps) if u[t - 1] else (pr if u[t] else 1 - pr)
        k2016 = sum(u[t] for t in range(len(idx)) if year[t] == 2016)
        p *= np.exp(-0.5 * ((annual[2016] - q * k2016) / (sigma * 12 * q)) ** 2)
        total += p
        w += p * np.array(u)
    got = p_on_smoothed(
        margin, month, c, r, tau=tau, annual=annual, year=year, q=q, sigma=sigma, off=off, p_on0=p0
    )
    assert np.allclose(got, w / total, atol=1e-9)


def test_empirical_bayes_recovers_the_covariate_effect():
    from nsqueeze.plant_activity.switching import empirical_bayes

    idx = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
    t = np.arange(len(idx))
    margin = pd.Series(150 - 450 * np.exp(-(((t - 80) / 12.0) ** 2)), index=idx)
    grid = np.arange(-300.0, 301.0, 25.0)
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, -1.5, 1.5])
    surfaces = []
    for zi in z:  # true stop threshold rises with z: c = -100 + 60 z; restart 100 above
        c_true, on, state = -100 + 60 * zi, [], True
        for i in range(len(idx)):
            m = margin.iloc[i - 1] if i else margin.iloc[0]
            state = (m > c_true) if state else (m > c_true + 100)
            on.append(state)
        on = np.array(on)
        annual = {y: 1000.0 * on[idx.year == y].sum() for y in range(2016, 2026)}
        surfaces.append(
            fit_plant(
                margin,
                annual=annual,
                q=1000.0,
                off=pd.Series(False, index=idx),
                tau=10.0,
                sigma=0.05,
                grid=grid,
            )["surface"]
        )
    eb = empirical_bayes(surfaces, z, grid, theta0=[0.0, 0.0, np.log(200.0), 100.0, np.log(100.0)])
    assert eb["theta"][1] > 20  # the covariate effect b is found, with the right sign
