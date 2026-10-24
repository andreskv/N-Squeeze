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
