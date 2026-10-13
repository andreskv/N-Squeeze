"""Layer 1 gas balance and its HDD definition, on synthetic data."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nsqueeze import gas_balance as gb


def test_hdd_eurostat_definition():
    # Eurostat's worked example: Tm 12 °C → 6 HDD; Tm 16 °C → 0. Threshold is inclusive.
    assert gb.heating_degree_days(np.array([12.0, 16.0, 15.0]))[:2].tolist() == [6.0, 0.0]
    assert gb.heating_degree_days(np.array([15.0]))[0] == 3.0
    s = gb.heating_degree_days(pd.Series([10.0, np.nan, 20.0]))
    assert s.iloc[0] == 8.0 and np.isnan(s.iloc[1]) and s.iloc[2] == 0.0
    assert gb.eurostat_mean_temp(10.0, 0.0) == 5.0


def test_weighting_hdd_not_temperatures():
    # One country at 10 °C, one at 20 °C: average HDD is 4, HDD of the average (15 °C) is 3.
    t = pd.DataFrame({"A": [10.0], "B": [20.0]})
    w = pd.Series({"A": 1.0, "B": 1.0})
    assert gb.population_weighted(gb.heating_degree_days(t), w).iloc[0] == 4.0


def _synthetic(start="2015-10-01", end="2025-04-30", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D")
    doy = idx.dayofyear.to_numpy()
    hdd = np.clip(
        10 + 9 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 2, len(idx)), 0, None
    )
    post = idx >= pd.Timestamp("2022-03-01")
    alpha = np.where(post, 8.0, 9.0)  # TWh/d
    beta = np.where(post, 0.30, 0.35)  # TWh/d per HDD
    demand = alpha + beta * hdd + 0.2 * (idx.dayofweek >= 5) + rng.normal(0, 0.3, len(idx))
    supply = 11.0 + 0.5 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    storage = 600 + np.cumsum(supply - demand)
    storage = storage - storage.min() + 200
    return pd.DataFrame(
        {"storage": storage, "wgv": 1100.0, "supply": supply, "hdd": hdd}, index=idx
    )


def test_fit_recovers_regime_coefficients():
    f = _synthetic()
    f["dS"] = f.storage.diff()
    m = gb.BalanceModel().fit(f)
    c = m.coefficients
    assert c.beta_pre == pytest.approx(0.35, abs=0.01)
    assert c.beta_post == pytest.approx(0.30, abs=0.01)
    assert c.alpha_post - c.alpha_pre == pytest.approx(-1.0, abs=0.1)


def test_backcast_and_g2_gate():
    f = _synthetic()
    bc = gb.backcast(f, list(range(2016, 2025)), start_md=(10, 1), end_md=(4, 1))
    assert len(bc) == 9 and {"error_pp", "predicted_pct"} <= set(bc.columns)
    r = gb.g2_gate(bc, mae_limit_pp=5.0, max_limit_pp=10.0)
    assert r.passed and r.winters == 9
    assert not gb.g2_gate(bc, mae_limit_pp=0.0, max_limit_pp=0.0).passed


def test_fan_chart_ranks_weather_years():
    f = _synthetic()
    f["dS"] = f.storage.diff()
    m = gb.BalanceModel().fit(f)
    hdd = f.hdd.copy()
    hdd.loc["2017-11-01":"2018-03-31"] += 5  # a cold winter in the history
    end = m.simulate(
        600.0,
        dt.date(2024, 10, 31),
        dt.date(2025, 3, 31),
        hdd,
        gb.SupplyScenario("lng_mean", 11.0),
    )
    assert end.idxmin() == 2017  # keyed by start year: Nov 2017 – Mar 2018 is the cold one
    p = gb.breach_probability(end.to_numpy() / 1100, threshold=end.median() / 1100)
    assert 0 < p < 1


def test_entsog_supply_counts_each_flow_once():
    ic = pd.DataFrame(
        {
            "pointKey": ["P1", "P1", "L1", "X1", "U1"],
            "toOperatorKey": ["DE-A", "NL-B", "FR-C", "DE-A", "UK-Z"],
            "fromCountryKey": ["NO", "NO", "FR", "PL", "NO"],
            "toCountryKey": ["DE", "NL", "FR", "DE", "UK"],
            "fromInfrastructureTypeLabel": [
                "Transmission",
                "Transmission",
                "LNG Terminals",
                "Transmission",
                "Transmission",
            ],
        }
    )
    cls = gb.classify_points(ic)
    assert set(cls.bucket) == {"norway", "lng"}  # PL→DE is intra-EU; NO→UK is not into the EU
    flows = pd.DataFrame(
        {
            "series_id": [
                "ENTSOG.DE-A.P1.entry",
                "ENTSOG.NO-X.P1.exit",  # the Norwegian side's report of the same gas
                "ENTSOG.FR-C.L1.entry",
                "ENTSOG.DE-A.X1.entry",
            ],
            "observed_at": pd.to_datetime(["2025-01-01"] * 4),
            "value": [2e9, 2e9, 1e9, 5e9],
        }
    )
    s = gb.supply_from_entsog(flows, cls)
    assert s.loc["2025-01-01", "norway"] == pytest.approx(2.0)
    assert s.loc["2025-01-01", "lng"] == pytest.approx(1.0)
    assert s.loc["2025-01-01", "other_pipe"] == 0.0
