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


def test_population_weighted_reference():
    # AT: a cold small region (500 HDD, 1 person) and a mild big one (100 HDD, 3 people).
    # Population weighting gives 200; area weighting would not. A month with a missing
    # region is NaN, not the mean of the rest. EU needs all 27 countries.
    m = pd.to_datetime(["2020-01-01", "2020-02-01"])
    hdd = pd.DataFrame(
        {
            "series_id": [
                "ESTAT.HDD2.AT11",
                "ESTAT.HDD2.AT12",
                "ESTAT.HDD2.AT11",
                "ESTAT.HDD2.NO01",
            ],
            "observed_at": [m[0], m[0], m[1], m[0]],
            "value": [500.0, 100.0, 400.0, 900.0],
        }
    )
    pop = pd.DataFrame(
        {
            "series_id": ["ESTAT.POP2.AT11", "ESTAT.POP2.AT12"],
            "observed_at": pd.to_datetime(["2020-01-01"] * 2),
            "value": [1.0, 3.0],
        }
    )
    ref = gb.population_weighted_reference(hdd, pop)
    assert list(ref.columns) == ["AT", "EU27_2020"]  # NO is not EU-27
    assert ref.loc[m[0], "AT"] == 200.0
    assert np.isnan(ref.loc[m[1], "AT"]) and ref.EU27_2020.isna().all()


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


def test_backcast_with_power_burn_and_missing_day():
    # Demand gets a power term that varies by winter; only the power-aware model tracks it.
    f = _synthetic()
    rng = np.random.default_rng(1)
    winter_level = pd.Series(rng.uniform(0, 4, len(f.index.year.unique())), f.index.year.unique())
    f["power_burn"] = winter_level.reindex(f.index.year).to_numpy() * (f.hdd > 10)
    f["storage"] = f.storage - (0.8 * f.power_burn).cumsum()
    winters = list(range(2016, 2025))
    plain = gb.backcast(f, winters, start_md=(10, 1), end_md=(4, 1))
    with_pw = gb.backcast(f, winters, start_md=(10, 1), end_md=(4, 1), power=True)
    assert with_pw.error_pp.abs().mean() < plain.error_pp.abs().mean() / 3
    gap = f.drop(pd.Timestamp("2019-01-10"))
    assert 2019 not in set(gb.backcast(gap, winters, start_md=(10, 1), end_md=(4, 1)).winter)


def test_power_panel_fills_short_gaps_only():
    def rows(geo, days):
        return pd.DataFrame(
            {
                "series_id": f"ENTSOE.{geo}.B04",
                "observed_at": pd.DatetimeIndex(days) + pd.Timedelta(hours=10),
                "value": 1000.0,  # MW for one hour -> 1 GWh
                "meta": '{"resolution": "PT60M"}',
            }
        )

    days = pd.date_range("2020-01-01", "2020-01-20")
    a = rows("AA", days)
    b = rows("BB", days.delete([2, 3]))  # 2-day gap: filled
    b = b[~b.observed_at.dt.day.between(10, 17)]  # 8-day gap: not filled
    p = gb.power_panel(pd.concat([a, b]), panel=("AA", "BB"), max_gap_days=7)
    assert p.loc["2020-01-03"] == pytest.approx(2e-3)
    assert p.loc["2020-01-10":"2020-01-17"].isna().all()
    assert p.loc["2020-01-18"] == pytest.approx(2e-3)


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
            "fromOperatorKey": ["NO-X", "NO-X", "FR-L", "PL-Y", "NO-X"],
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
            "toInfrastructureTypeLabel": ["Transmission"] * 5,
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


def test_entsog_supply_is_net_across_the_boundary():
    ic = pd.DataFrame(
        {
            "pointKey": ["T1", "R1", "R2", "VTP-00043", "L2"],
            "fromOperatorKey": ["TR-X", "BG-A", "RS-Y", "ES-L", "ES-L"],
            "toOperatorKey": ["BG-A", "RS-Y", "HU-B", "ES-T", "ES-T"],
            "fromCountryKey": ["TR", "BG", "RS", "ES", "ES"],
            "toCountryKey": ["BG", "RS", "HU", "ES", "ES"],
            "fromInfrastructureTypeLabel": ["Transmission", "Transmission", "Transmission",
                                            "LNG Terminals", "LNG Terminals"],
            "toInfrastructureTypeLabel": ["Transmission"] * 5,
        }
    )  # fmt: skip
    cls = gb.classify_points(ic)
    assert set(zip(cls.pointKey, cls.operatorKey, cls.bucket, strict=True)) == {
        ("T1", "BG-A", "other_pipe"),
        ("R1", "BG-A", "other_pipe"),  # leaves the EU: counted by the EU-side operator
        ("R2", "HU-B", "other_pipe"),
        ("L2", "ES-T", "lng"),  # the virtual tank VTP-00043 is dropped
    }
    flows = pd.DataFrame(
        {
            "series_id": [
                "ENTSOG.BG-A.T1.entry",  # 10 in from Turkey
                "ENTSOG.BG-A.R1.exit",  # 6 out to Serbia ...
                "ENTSOG.HU-B.R2.entry",  # ... and the same 6 back in
                "ENTSOG.ES-T.VTP-00043.entry",
                "ENTSOG.ES-T.L2.entry",
            ],
            "observed_at": pd.to_datetime(["2025-01-01"] * 5),
            "value": [10e9, 6e9, 6e9, 4e9, 4e9],
        }
    )
    s = gb.supply_from_entsog(flows, cls)
    assert s.loc["2025-01-01", "other_pipe"] == pytest.approx(10.0)
    assert s.loc["2025-01-01", "lng"] == pytest.approx(4.0)


def test_norway_landing_counts_once():
    cls = pd.DataFrame({"pointKey": ["Z1"], "operatorKey": ["BE-A"], "bucket": ["norway"]})
    landings = pd.DataFrame(
        {
            "pointKey": ["D1", "D2", "E1", "E2"],
            "operatorKey": ["DE-A", "DE-B", "DE-A", "NL-C"],
            "bucket": "norway",
            "landing": ["Dornum", "Dornum", "Emden", "Emden"],
        }
    )
    cls = gb.landing_points(cls, landings)
    flows = pd.DataFrame(
        {
            "series_id": [
                "ENTSOG.BE-A.Z1.entry",
                "ENTSOG.DE-A.D1.entry",
                "ENTSOG.DE-B.D2.entry",  # same Dornum flow, reported again
                "ENTSOG.DE-A.E1.entry",
                "ENTSOG.NL-C.E2.entry",  # Emden, measured a little higher
            ],
            "observed_at": pd.to_datetime(["2025-01-01"] * 5),
            "value": [1e9, 5e9, 5e9, 3e9, 3.2e9],
        }
    )
    s = gb.supply_from_entsog(flows, cls)
    assert s.loc["2025-01-01", "norway"] == pytest.approx(1 + 5 + 3.2)


def test_eurostat_buckets_and_check():
    ids = [
        "ESTAT.GAS.IMP.G3000.NO",
        "ESTAT.GAS.IMP.G3000.US",
        "ESTAT.GAS.IMP.G3200.US",  # all US gas is LNG
        "ESTAT.GAS.IMP.G3000.DZ",
        "ESTAT.GAS.EXP.G3000.DZ",
        "ESTAT.GAS.IMP.G3000.DE",  # intra-EU: drops out
        "ESTAT.GAS.IMP.G3000.TOTAL",
    ]
    trade = pd.DataFrame(
        {
            "series_id": ids,
            "observed_at": pd.Timestamp("2025-01-01"),
            "value": [3600.0, 1800.0, 1800.0, 900.0, 180.0, 9e9, 9e9],
        }
    )
    gap = pd.DataFrame(  # a month without the EU-27 total is left out, not zero
        {"series_id": ["ESTAT.GAS.IMP.G3000.US"], "observed_at": pd.Timestamp("2025-02-01"),
         "value": [1.0]}
    )  # fmt: skip
    assert len(gb.eurostat_net_by_bucket(pd.concat([trade, gap]))) == 1
    e = gb.eurostat_net_by_bucket(trade).iloc[0]
    assert (e.norway, e.lng, e.other_pipe) == pytest.approx((1.0, 0.5, 0.2))
    idx = pd.date_range("2025-01-01", "2025-02-28")
    ours = pd.DataFrame({"norway": 1.05 / 31, "lng": 0.5 / 31, "other_pipe": 0.3 / 31}, index=idx)
    summary, _ = gb.bucket_check(ours, gb.eurostat_net_by_bucket(trade), tolerance=0.10)
    assert summary.loc["norway", "within"] == 1.0  # +5%
    assert summary.loc["other_pipe", "within"] == 0.0  # +50%
    assert summary.loc["lng", "months"] == 1  # February has no Eurostat row


def test_eurostat_nsp_and_incomplete_month():
    def rows(month, pairs):
        ids, vals = zip(*pairs, strict=True)
        return pd.DataFrame(
            {"series_id": [f"ESTAT.GAS.IMP.{i}" for i in ids], "observed_at": pd.Timestamp(month),
             "value": vals}
        )  # fmt: skip

    full = rows(
        "2024-02-01",
        [("G3000.TOTAL", 7200.0), ("G3000.NO", 3600.0), ("G3000.NSP", 3600.0),
         ("G3200.NSP", 1800.0)],
    )  # fmt: skip
    # March 2024 pattern: total published, partners cover less than 99% of it
    partial = rows("2024-03-01", [("G3000.TOTAL", 7200.0), ("G3000.NO", 3600.0)])
    e = gb.eurostat_net_by_bucket(pd.concat([full, partial]))
    assert list(e.index) == [pd.Timestamp("2024-02-01")]
    # NSP LNG counts as LNG; NSP pipeline (3600 - 1800 TJ) is in no bucket
    assert (e.iloc[0].norway, e.iloc[0].lng, e.iloc[0].other_pipe) == pytest.approx((1.0, 0.5, 0.0))


def test_archive_points_rename_and_bucket():
    table = pd.DataFrame(
        {
            "series": ["ENTSOG.EIC.OPX.21ZA", "ENTSOG.NAME.OPY.Tarifa"],
            "pointKey": ["P1", ""],
            "operatorKey": ["DE-A", ""],
            "bucket": ["", "other_pipe"],
            "landing": ["", ""],
        }
    )
    day = pd.Timestamp("2017-01-01")
    flows = pd.DataFrame(
        {
            "series_id": [
                "ENTSOG.EIC.OPX.21ZA.entry",
                "ENTSOG.DE-A.P1.entry",  # the API key already has this day: it wins
                "ENTSOG.NAME.OPY.Tarifa.entry",
                "ENTSOG.EIC.OPZ.21ZQ.entry",  # not in the table: unchanged
            ],
            "observed_at": [day] * 4,
            "value": [7e9, 2e9, 3e9, 1e9],
        }
    )
    out, extra = gb.archive_points(flows, table)
    assert dict(zip(out.series_id, out.value, strict=True)) == {
        "ENTSOG.DE-A.P1.entry": 2e9,
        "ENTSOG.ARCHIVE.A1.entry": 3e9,
        "ENTSOG.EIC.OPZ.21ZQ.entry": 1e9,
    }
    cls = pd.concat(
        [pd.DataFrame({"pointKey": ["P1"], "operatorKey": ["DE-A"], "bucket": ["norway"]}), extra]
    )
    s = gb.supply_from_entsog(out, cls)
    assert s.loc[day, "other_pipe"] == pytest.approx(3.0)
    assert s.loc[day, "norway"] == pytest.approx(2.0)


def test_layer1_inputs():
    d1, d2 = pd.Timestamp("2025-01-15"), pd.Timestamp("2025-01-16")
    agsi = pd.DataFrame(
        {
            "series_id": ["AGSI.EU.gasInStorage", "AGSI.EU.workingGasVolume", "AGSI.DE.full"],
            "observed_at": [d1] * 3,
            "value": ["800.5", "1100", "70"],  # GIE values arrive as strings
        }
    )
    st = gb.eu_storage(agsi)
    assert (st.loc[d1, "storage"], st.loc[d1, "wgv"]) == (800.5, 1100.0)

    # HDD: A at Tm 10 °C (HDD 8), B at Tm 20 °C (HDD 0); weights 1 and 3 -> 2.0.
    # 2026 has no population row: the 2025 weights carry forward.
    days = [d1, pd.Timestamp("2026-01-15")]
    temps = [("AA", "max", 12), ("AA", "min", 8), ("BB", "max", 22), ("BB", "min", 18)]
    weather = pd.DataFrame(
        [(f"OM.{g}.temperature_2m_{v}", d, t) for d in days for g, v, t in temps],
        columns=["series_id", "observed_at", "value"],
    )
    pop = pd.DataFrame(
        {
            "series_id": ["ESTAT.POP.AA", "ESTAT.POP.BB", "ESTAT.POP.EU27_2020"],
            "observed_at": pd.Timestamp("2025-01-01"),
            "value": [1.0, 3.0, 4.0],
        }
    )
    h = gb.daily_hdd(weather, pop)
    assert list(h) == pytest.approx([2.0, 2.0])
    gap = gb.daily_hdd(weather[~((weather.series_id == "OM.BB.temperature_2m_max")
                                 & (weather.observed_at == d1))], pop)  # fmt: skip
    assert np.isnan(gap.loc[d1])

    # Power: 06:00 Brussels (05:00 UTC in winter) starts the gas day.
    entsoe = pd.DataFrame(
        {
            "series_id": ["ENTSOE.DE.B04"] * 3 + ["ENTSOE.FR.B04"],
            "observed_at": pd.to_datetime(
                ["2025-01-15 04:45", "2025-01-15 05:00", "2025-01-15 05:15", "2025-01-15 05:00"]
            ),
            "value": [400.0, 400.0, 400.0, 1000.0],
            "meta": ['{"resolution": "PT15M"}'] * 3 + ['{"resolution": "PT60M"}'],
        }
    )
    pw = gb.power_generation(entsoe)
    assert pw.loc[pd.Timestamp("2025-01-14"), "power_twh_e"] == pytest.approx(100 / 1e6)
    assert pw.loc[d1, "power_twh_e"] == pytest.approx((200 + 1000) / 1e6)
    assert pw.loc[d1, "countries"] == 2

    supply = pd.DataFrame({"norway": [1.0, 1.0], "lng": [0.5, 0.5]}, index=[d1, d2])
    f = gb.layer1_frame(st, supply, h, pw.power_twh_e)
    assert list(f.index) == [d1]  # only days present in every input
    assert f.loc[d1, "supply"] == 1.5


def test_fan_chart_pools_scenarios_and_needs_every_weather_year():
    f = _synthetic()
    f["dS"] = f.storage.diff()
    m = gb.BalanceModel().fit(f)
    args = (m, 600.0, dt.date(2025, 1, 5), dt.date(2025, 4, 1), f.hdd)
    supply = {"low": 10.98, "high": 11.02}
    kw = dict(
        weather_years=[2016, 2017, 2018], draws=20, wgv_twh=1100.0, thresholds_pct=[37.8, 38.1]
    )
    p = gb.fan_chart(*args, supply, **kw)
    assert p == gb.fan_chart(*args, supply, **kw)  # fixed seed: same entry every run
    for s in ("low", "high", "pooled"):
        assert 0 < p[s]["37.8"] < p[s]["38.1"] < 1  # both inside the spread
    assert p["low"]["38.1"] > p["high"]["38.1"]  # less supply, more mass below the threshold
    assert p["pooled"]["38.1"] == pytest.approx((p["low"]["38.1"] + p["high"]["38.1"]) / 2)
    with pytest.raises(ValueError):
        gb.fan_chart(*args, supply, **{**kw, "weather_years": [2016, 2030]})
