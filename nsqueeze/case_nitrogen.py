"""Phase 1 case study: EU urea cash cost against import parity.

Pipeline:

1. ``monthly_panel``     official monthly anchors from the PIT store, sealed at the
                         end of the calibration window until the pre-registration freeze.
2. ``urea_cost``         EU cash cost of a tonne of urea from gas and carbon (κ = 1),
                         excluding the unknown other cash costs O ≥ 0.
3. ``urea_parity``       landed import parity at both ends of the freight and duty bands.
4. ``margin_bound``      parity − (gas + carbon cost) is an upper bound on the true
                         cash margin, because O ≥ 0. A negative bound means the plant loses
                         cash on every tonne whatever O is: a parameter-free curtailment
                         signal.
5. ``score_onsets``      compare predicted curtailment months with documented episodes.
6. ``fit_threshold_loo`` the parametric run rule M̄ < θ′ (θ′ = θ + O, which are not
                         separately identified), fitted on one episode, scored on the other.

Urea is the product because the only free nitrogen price series is the World Bank urea FOB. The FOB
origin changes from the Black Sea to the Middle East in March 2022.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nsqueeze import units
from nsqueeze.freeze import load_episodes, sealed_after
from nsqueeze.io.pit import PITStore
from nsqueeze.params import ParameterSet
from nsqueeze.provenance import Provenance

START = pd.Timestamp("2015-04-01")  # Pink Sheet "Natural gas, Europe" = TTF from here
UREA_ORIGIN_BREAK = pd.Timestamp("2022-03-01")
TIERS = {
    "best_practice": "ammonia_tech.energy_best_practice",
    "eu_average": "ammonia_tech.energy_eu_average",
}


# 1. data


def monthly_panel(store: PITStore | None = None, *, as_of=None) -> pd.DataFrame:
    """Month-start index; prices in source units plus €-converted columns.

    Columns: ttf_usd_mmbtu, hh_usd_mmbtu, urea_usd_t, eua_eur_t, fx_usd_per_eur,
    ttf_eur_mwh, hh_eur_mwh, urea_origin. ``attrs`` records the seal and vintages.
    """
    store = store or PITStore()
    wb = store.read("pink_sheet_monthly", as_of=as_of)
    wide = wb.pivot_table(index="observed_at", columns="series_id", values="value", aggfunc="last")
    eex = store.read("eua_auctions", as_of=as_of).pivot_table(
        index="observed_at", columns="series_id", values="value", aggfunc="last"
    )
    px, vol = eex["EEX.EUA.auction_price"], eex["EEX.EUA.auction_volume"]
    eua = (px * vol).resample("MS").sum() / vol.resample("MS").sum()
    fx = store.read("ecb_fx", as_of=as_of).set_index("observed_at")["value"].resample("MS").mean()

    p = pd.DataFrame(
        {
            "ttf_usd_mmbtu": wide["WB.NGAS_EUR"],
            "hh_usd_mmbtu": wide["WB.NGAS_US"],
            "urea_usd_t": wide["WB.UREA"],
            "eua_eur_t": eua,
            "fx_usd_per_eur": fx,
        }
    )
    seal = sealed_after()
    end = seal if seal is not None else p.index.max()
    p = _derive(p.loc[START:end].dropna())
    p.attrs.update(
        sealed_after=None if seal is None else seal.date().isoformat(),
        vintage_pink_sheet=str(wb.retrieved_at.max()),
        vintage_eex=str(store.read("eua_auctions", as_of=as_of).retrieved_at.max()),
    )
    return p


def panel_from_extract(path) -> pd.DataFrame:
    """The ``monthly_panel`` frame from a CSV extract of its five source columns."""
    p = pd.read_csv(path, index_col="month", parse_dates=True)
    return _derive(
        p[["ttf_usd_mmbtu", "hh_usd_mmbtu", "urea_usd_t", "eua_eur_t", "fx_usd_per_eur"]]
    )


def _derive(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["ttf_eur_mwh"] = units.usd_mmbtu_to_eur_mwh(p.ttf_usd_mmbtu, p.fx_usd_per_eur)
    p["hh_eur_mwh"] = units.usd_mmbtu_to_eur_mwh(p.hh_usd_mmbtu, p.fx_usd_per_eur)
    p["urea_origin"] = np.where(p.index >= UREA_ORIGIN_BREAK, "Middle East", "Black Sea")
    p.index.name = "month"
    return p


# 2 cost


@dataclass(frozen=True)
class CostInputs:
    energy_gj_lhv: float
    hhv_over_lhv: float
    ef_t_per_gj: float
    nh3_per_urea: float
    conv_gas_mmbtu: float
    provenance: Provenance

    @classmethod
    def from_params(cls, P: ParameterSet, tier: str = "eu_average") -> CostInputs:
        keys = [
            TIERS[tier],
            "units.hhv_over_lhv_gas",
            "carbon.gas_emission_factor",
            "products.nh3_per_t_urea",
            "products.urea_conversion_gas",
        ]
        prov = Provenance.of(*(P[k].ref for k in keys))
        return cls(
            energy_gj_lhv=float(P[TIERS[tier]].value),
            hhv_over_lhv=float(P["units.hhv_over_lhv_gas"].value),
            ef_t_per_gj=float(P["carbon.gas_emission_factor"].value) / 1000,
            nh3_per_urea=float(P["products.nh3_per_t_urea"].value),
            conv_gas_mmbtu=float(P["products.urea_conversion_gas"].value),
            provenance=prov,
        )


def urea_cost(panel: pd.DataFrame, c: CostInputs) -> pd.DataFrame:
    """€/t urea, gas and carbon only (κ = 1). Other cash costs O are excluded on purpose."""
    g = units.gj_lhv_to_mwh_hhv(c.energy_gj_lhv, c.hhv_over_lhv)  # MWh HHV / t NH3
    e = c.energy_gj_lhv * c.ef_t_per_gj  # t CO2 / t NH3
    conv_gj_lhv = c.conv_gas_mmbtu * units.GJ_PER_MMBTU / c.hhv_over_lhv
    out = pd.DataFrame(index=panel.index)
    out["nh3_gas"] = c.nh3_per_urea * g * panel.ttf_eur_mwh
    out["nh3_carbon"] = c.nh3_per_urea * e * panel.eua_eur_t
    out["conv_gas"] = c.conv_gas_mmbtu * panel.ttf_usd_mmbtu / panel.fx_usd_per_eur
    out["conv_carbon"] = conv_gj_lhv * c.ef_t_per_gj * panel.eua_eur_t
    out["total"] = out.sum(axis=1)
    return out


# 3 parity


def urea_parity(panel: pd.DataFrame, *, freight_usd: float, duty: float) -> pd.Series:
    """Landed import parity, €/t urea: (FOB + F)(1 + τ) / FX. No CBAM before 2026."""
    return (panel.urea_usd_t + freight_usd) * (1 + duty) / panel.fx_usd_per_eur


def parity_band(panel: pd.DataFrame, P: ParameterSet) -> pd.DataFrame:
    f_lo, f_hi = P["assumptions.urea_freight_usd_t"].range
    t_lo, t_hi = P["assumptions.urea_duty_band"].range
    return pd.DataFrame(
        {
            "low": urea_parity(panel, freight_usd=f_lo, duty=t_lo),
            "high": urea_parity(panel, freight_usd=f_hi, duty=t_hi),
        }
    )


#  4 margin bound


def margin_bound(parity: pd.Series, cost: pd.Series, *, lookback: int = 1) -> pd.Series:
    """Upper bound on the cash margin, averaged over ``lookback`` months (decision lag)."""
    return (parity - cost).rolling(lookback, min_periods=1).mean()


#  5


CLUSTER_GAP_MONTHS = 2  # onsets closer than this share a cause and are one episode


def episode_onsets() -> pd.DataFrame:
    """Calibration curtailment onsets. ``cluster`` labels each independent episode.

    A new episode starts when an onset comes more than CLUSTER_GAP_MONTHS after
    the previous one. The label is the first onset month, e.g. "2022-03".
    """
    ep = load_episodes()
    ep = ep[(ep.direction == "curtail") & (ep["sample"] == "calibration")].copy()
    ep = ep.sort_values("date")
    ep["month"] = ep.date.dt.to_period("M").dt.to_timestamp()
    gap = ep.month.dt.to_period("M").astype(int).diff().fillna(CLUSTER_GAP_MONTHS + 1)
    first = ep.month.where(gap > CLUSTER_GAP_MONTHS).ffill()
    ep["cluster"] = first.dt.strftime("%Y-%m")
    return ep


def spell_starts(signal: pd.Series) -> pd.DatetimeIndex:
    """Months where ``signal`` turns True (a False → True step, or True in the first month)."""
    s = signal.astype(bool)
    return s.index[s & ~s.shift(1, fill_value=False)]


def score_onsets(signal: pd.Series, *, tolerance_months: int = 1, window: int = 6) -> pd.DataFrame:
    """One row per episode: documented onset, nearest predicted spell start, outcome.

    outcome: "hit" (a spell starts within ± tolerance), "ongoing" (the signal was
    already on at the documented onset, so nothing starts) or "miss".
    """
    starts = spell_starts(signal)
    rows = []
    for label, grp in episode_onsets().groupby("cluster"):
        doc = grp.month.min()
        near = [t for t in starts if abs(_months(t, doc)) <= window]
        pred = min(near, key=lambda t: abs(_months(t, doc))) if near else None
        lead = None if pred is None else -_months(pred, doc)
        if lead is not None and abs(lead) <= tolerance_months:
            outcome = "hit"
        elif bool(signal.get(doc, False)):
            outcome = "ongoing"
        else:
            outcome = "miss"
        rows.append(
            {
                "episode": label,
                "documented_onset": doc,
                "predicted_onset": pred,
                "lead_months": lead,
                "outcome": outcome,
                "hit": outcome == "hit",
            }
        )
    return pd.DataFrame(rows)


def _months(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (a.year - b.year) * 12 + a.month - b.month


def false_alarms(signal: pd.Series, *, quiet=("2015-04-01", "2020-12-31")) -> int:
    """Months flagged in a period with no documented economic curtailment."""
    return int(signal.loc[quiet[0] : quiet[1]].sum())


# 6 parametric threshold, leave-one-episode-out


THETA_GRID = np.arange(0, 401, 5)  # €/t urea


def _onset_loss(bound: pd.Series, theta: float, labels) -> float:
    sig = bound < theta
    sc = score_onsets(sig)
    sc = sc[sc.episode.isin(labels)]
    return sc.lead_months.fillna(12).abs().sum() + false_alarms(sig)


def fit_threshold(bound: pd.Series, labels=None, grid: np.ndarray = THETA_GRID) -> float:
    """θ′ that minimises the onset loss on ``labels`` (default: every episode).

    Ties break toward the smallest θ′ (the most conservative rule).
    """
    labels = sorted(episode_onsets().cluster.unique()) if labels is None else labels
    losses = [_onset_loss(bound, t, labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def fit_softness(
    bound: pd.Series, theta: float, utilisation: pd.Series, grid: np.ndarray | None = None
) -> float:
    """s that best fits sigmoid((M̄ − θ′)/s) to documented utilisation levels (least squares).

    ``utilisation`` is indexed by month start, values in [0, 1].
    """
    grid = np.arange(1, 401, 1.0) if grid is None else grid
    m = bound.reindex(utilisation.index).to_numpy(float) - theta
    u = utilisation.to_numpy(float)
    sse = [float(np.sum((1 / (1 + np.exp(-m / s)) - u) ** 2)) for s in grid]
    return float(grid[int(np.argmin(sse))])


def fit_threshold_loo(bound: pd.Series, grid: np.ndarray = THETA_GRID) -> pd.DataFrame:
    """Fit θ′ on all episodes but one, score the one left out, for each episode.

    The run rule flags curtailment when the margin bound falls below θ′ = θ + O.
    Loss on the training episodes = |onset error| in months + 1 per false alarm in
    2015–2020. Ties break toward the smallest θ′ (the most conservative rule).
    """
    clusters = sorted(episode_onsets().cluster.unique())
    rows = []
    for test in clusters:
        train = [c for c in clusters if c != test]
        theta = fit_threshold(bound, train, grid)
        sc = score_onsets(bound < theta)
        held = sc[sc.episode == test].iloc[0]
        rows.append(
            {
                "train": "+".join(map(str, train)),
                "test": str(test),
                "theta_prime": theta,
                "test_documented": held.documented_onset,
                "test_predicted": held.predicted_onset,
                "test_lead_months": held.lead_months,
                "test_outcome": held.outcome,
                "false_alarms_2015_20": false_alarms(bound < theta),
            }
        )
    return pd.DataFrame(rows)


def point_calibration_sep2021(panel: pd.DataFrame, P: ParameterSet) -> dict[str, float]:
    """Validation test 2: gas cost in a tonne of NH3, Sep 2021, vs CF's CEO quote ($900)."""
    m = panel.loc["2021-09-01"]
    out = {}
    for tier, key in TIERS.items():
        g_mmbtu = units.mwh_to_mmbtu(units.gj_lhv_to_mwh_hhv(float(P[key].value)))
        out[tier] = g_mmbtu * m.ttf_usd_mmbtu
    out["ttf_usd_mmbtu"] = float(m.ttf_usd_mmbtu)
    out["reference_usd_t"] = 900.0
    return out
