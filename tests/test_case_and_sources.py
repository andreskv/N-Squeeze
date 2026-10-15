"""Phase 1 case-study logic and the World Bank / EEX parsers, on synthetic inputs (no network)."""

import io

import numpy as np
import openpyxl
import pandas as pd
import pytest

from nsqueeze import case_nitrogen as cn
from nsqueeze import simulation as sim
from nsqueeze import units
from nsqueeze.io.sources.eex import parse_report, to_canonical
from nsqueeze.io.sources.worldbank import parse_workbook
from nsqueeze.params import ParameterSet


@pytest.fixture(scope="module")
def P():
    return ParameterSet.load()


def _panel(ttf_eur_mwh, urea_usd, eua=80.0, fx=1.1):
    idx = pd.date_range("2021-01-01", periods=len(ttf_eur_mwh), freq="MS")
    p = pd.DataFrame(index=idx)
    p["fx_usd_per_eur"] = fx
    p["ttf_eur_mwh"] = ttf_eur_mwh
    p["ttf_usd_mmbtu"] = p.ttf_eur_mwh * fx / 3.41214
    p["eua_eur_t"] = eua
    p["urea_usd_t"] = urea_usd
    return p


def test_urea_cost_decomposition(P):
    c = cn.urea_cost(_panel([40.0], [400.0]), cn.CostInputs.from_params(P, "eu_average")).iloc[0]
    # 0.567 × 11.0 MWh × €40 ≈ €249 of ammonia gas in a tonne of urea
    assert c.nh3_gas == pytest.approx(0.567 * 10.988 * 40, rel=1e-3)
    # 0.567 × 2.0 t × €80 ≈ €91 of carbon
    assert c.nh3_carbon == pytest.approx(0.567 * 2.003 * 80, rel=1e-3)
    assert c.total == pytest.approx(c[["nh3_gas", "nh3_carbon", "conv_gas", "conv_carbon"]].sum())


def test_parity_and_bound_sign():
    p = _panel([40.0, 200.0], [400.0, 400.0])
    parity = cn.urea_parity(p, freight_usd=40, duty=0.065)
    assert parity.iloc[0] == pytest.approx((400 + 40) * 1.065 / 1.1)
    cost = pd.Series([300.0, 1500.0], index=p.index)
    b = cn.margin_bound(parity, cost)
    assert b.iloc[0] > 0 > b.iloc[1]


def test_point_calibration_uses_tier_intensity(P):
    p = _panel([60.0] * 9, [400.0] * 9)
    out = cn.point_calibration_sep2021(p, P)
    # EU average: 37.5 MMBtu/t × TTF in $/MMBtu
    assert out["eu_average"] == pytest.approx(37.49 * p.loc["2021-09-01"].ttf_usd_mmbtu, rel=1e-3)


def test_spell_starts_and_scoring(monkeypatch):
    idx = pd.date_range("2021-01-01", periods=36, freq="MS")
    s = pd.Series(False, index=idx)
    s["2021-09-01":"2022-12-01"] = True  # one long spell
    assert list(cn.spell_starts(s)) == [pd.Timestamp("2021-09-01")]
    ep = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-08-15", "2021-10-15", "2022-08-15", "2023-06-15"]),
            "direction": "curtail",
            "sample": "calibration",
        }
    )
    monkeypatch.setattr(cn, "load_episodes", lambda: ep)
    sc = cn.score_onsets(s).set_index("episode")
    assert list(sc.index) == ["2021-08", "2022-08", "2023-06"]  # Aug + Oct 2021 = one episode
    assert sc.outcome.tolist() == ["hit", "ongoing", "miss"]
    assert sc.loc["2021-08", "lead_months"] == -1  # signal one month after the documented onset


def _wb_workbook(updated="October 02, 2026") -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Monthly Prices"
    ws.append(["World Bank Commodity Price Data (The Pink Sheet)"])
    ws.append(["monthly prices in nominal US dollars"])
    ws.append([None])
    ws.append([f"Updated on {updated}"])
    names = [
        "Crude oil, average",
        "Natural gas, US",
        "Natural gas, Europe",
        "Liquefied natural gas, Japan",
        "Urea ",
        "DAP",
    ]
    unit = ["($/bbl)", "($/mmbtu)", "($/mmbtu)", "($/mmbtu)", "($/mt)", "($/mt)"]
    ws.append([None, *names])
    ws.append([None, *unit])
    ws.append(["2021M09", 70, 5.1, 22.8, 18.0, 418.8, 600])
    ws.append(["2021M10", 80, 5.5, "..", 30.0, 700.0, 700])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_pink_sheet_parser():
    frame, updated = parse_workbook(_wb_workbook())
    assert str(updated) == "2026-10-02"
    eur = frame[frame.series_id == "WB.NGAS_EUR"]
    assert len(eur) == 1 and eur.value.iloc[0] == 22.8  # ".." is missing, not zero
    assert set(frame.series_id) == {"WB.NGAS_US", "WB.NGAS_EUR", "WB.LNG_JP", "WB.UREA", "WB.DAP"}
    assert "Black Sea" in frame[frame.series_id == "WB.UREA"].meta.iloc[0]


def _eex_report() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    for _ in range(5):
        ws.append([None])
    ws.append(
        [
            None,
            "Date",
            "Time",
            "Auction Name",
            "Contract",
            "Status",
            "Auction Price €/tCO2",
            "Auction Volume tCO2",
        ]
    )
    ws.append([None, "2025-12-15", "11:00", "CAP3 EU", "T3PA", "successful", 84.6, 3_000_000])
    ws.append([None, "2025-12-15", "11:00", "DE", "T3PA", "successful", 83.6, 1_000_000])
    ws.append([None, "2025-12-12", "11:00", "Aviation", "EAA3", "successful", 80.0, 500_000])
    ws.append([None, "2025-12-11", "11:00", "PL", "T3PA", "cancelled", 10.0, 1_000_000])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_eex_parser_filters_and_volume_weights():
    a = parse_report(_eex_report())
    assert len(a) == 2  # aviation and cancelled rows dropped
    c = to_canonical(a)
    px = c[c.series_id == "EEX.EUA.auction_price"].value.iloc[0]
    assert px == pytest.approx((84.6 * 3 + 83.6 * 1) / 4)


def test_simulation_smoke():
    e2 = sim.e2_null_size(phis=(0.9,), n_mc=60, n_null=49, t=300, min_offset=40)
    assert e2.reject_naive.iloc[0] > e2.reject_circular.iloc[0]
    e1 = sim.e1_levels_vs_changes(noise_ratios=(2.0,), n_paths=50)
    assert e1.level_corr.iloc[0] > 0.9 and e1.change_snr.iloc[0] < 1
    e4 = sim.e4_few_events(n_episodes_grid=(2, 16), n_mc=40)
    assert np.isnan(e4.reject_clustered.iloc[0])


def test_fit_softness_recovers_s_from_utilisation_levels():
    months = pd.to_datetime(["2021-09-01", "2022-08-01", "2023-04-01"])
    bound = pd.Series([120.0, 40.0, 80.0], index=months)
    u = pd.Series(1 / (1 + np.exp(-(bound.to_numpy() - 100.0) / 50.0)), index=months)
    assert cn.fit_softness(bound, 100.0, u) == 50.0


def test_panel_from_extract_matches_derive(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text(
        "month,ttf_usd_mmbtu,hh_usd_mmbtu,urea_usd_t,eua_eur_t,fx_usd_per_eur\n"
        "2022-02-01,27.0,4.7,750,90,1.13\n2022-03-01,42.0,4.9,900,77,1.10\n"
    )
    p = cn.panel_from_extract(f)
    assert p.index.name == "month" and list(p.urea_origin) == ["Black Sea", "Middle East"]
    assert p.ttf_eur_mwh.iloc[0] == pytest.approx(units.usd_mmbtu_to_eur_mwh(27.0, 1.13))
