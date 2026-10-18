"""Cost model validation: six tests.

Each test returns a :class:TestResult.

    1 gas-intensity check     h reproduces CF's $32–33/short ton per $1/MMBtu
    2 2021 cost check         EU gas cost of a tonne of NH3, Sep 2021, vs CF's CEO quote ($900)
    3 curtailment timing test signal crosses within ± tolerance of each documented onset
    4 annual output test      ETS-implied output vs capacity × mean U_pred
    5 import response test    falls in U_pred → rising Comext ammonia imports, lag 1–3 months
    6 CBAM cost check         2026 imports × CBAM vs Copa-Cogeca's €820m (after the freeze)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nsqueeze import case_nitrogen as cn
from nsqueeze import units
from nsqueeze.evaluation.nulls import circular_shift_placebo
from nsqueeze.freeze import require_frozen, sealed_after
from nsqueeze.gas_balance import EU27_ENTSOG
from nsqueeze.params import ParameterSet

CF_SENSITIVITY_USD_PER_SHORT_TON = (32.0, 33.0)  # CF 10-K FY2023 / FY2020 (grade A)
CF_CEO_GAS_COST_SEP2021_USD_T = 900.0  # BCInsight, episodes.csv E21-05 (grade B)


@dataclass(frozen=True)
class TestResult:
    test: int
    name: str
    status: str  # "pass" | "fail" | "blocked"
    metric: str = ""
    value: float | None = None
    tolerance: str = ""
    note: str = ""


def gas_intensity_check(P: ParameterSet, *, rel_tol: float) -> TestResult:
    lo, hi = P["us_cost.h_cf_mmbtu_per_t"].range
    implied = [units.per_t_to_per_short_ton(h) for h in (lo, hi)]
    err = max(
        abs(a - b) / b for a, b in zip(implied, CF_SENSITIVITY_USD_PER_SHORT_TON, strict=True)
    )
    return TestResult(
        1,
        "gas-intensity check",
        "pass" if err <= rel_tol else "fail",
        "max relative error",
        err,
        f"≤ {rel_tol:.0%}",
    )


def cost_check_2021(
    panel: pd.DataFrame, P: ParameterSet, *, rel_tol: float, tier: str = "eu_average"
) -> TestResult:
    out = cn.point_calibration_sep2021(panel, P)
    err = out[tier] / CF_CEO_GAS_COST_SEP2021_USD_T - 1
    return TestResult(
        2,
        "2021 cost check",
        "pass" if abs(err) <= rel_tol else "fail",
        f"{tier} vs ${CF_CEO_GAS_COST_SEP2021_USD_T:.0f}/t",
        err,
        f"± {rel_tol:.0%}",
        note="Monthly TTF average stands in for the UK spot on the quote day",
    )


def curtailment_timing_test(signal: pd.Series, *, tolerance_months: int) -> TestResult:
    """Monthly data cannot resolve ±4 weeks; ±1 month is the nearest testable window."""
    sc = cn.score_onsets(signal, tolerance_months=tolerance_months)
    hits = int(sc.hit.sum())
    return TestResult(
        3,
        "curtailment timing test",
        "pass" if hits == len(sc) else "fail",
        "episode clusters hit",
        hits,
        f"all {len(sc)} within ± {tolerance_months} mo",
        note="; ".join(f"{r.episode}: lead {r.lead_months}" for r in sc.itertuples()),
    )


def annual_output_test(*_, **__) -> TestResult:
    return TestResult(
        4,
        "annual output test",
        "blocked",
        note="Needs U_pred; the softness s is not identified by the episode record",
    )


def trade_response(signal: pd.Series, imports_t: pd.Series, lags=(1, 2, 3)) -> pd.DataFrame:
    """Correlation of a curtailment signal with the change in imports ``lag`` months later.

    ``signal`` is 1 in months the model says plants should be curtailing.
    """
    d_imp = imports_t.pct_change()
    rows = []
    for k in lags:
        j = pd.concat([signal.astype(float), d_imp.shift(-k)], axis=1).dropna()
        rows.append(
            {
                "lag_months": k,
                "n": len(j),
                "corr": float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _max_lagged_corr(lags):
    def stat(sig: np.ndarray, d_imp: np.ndarray) -> float:
        return max(float(np.corrcoef(sig[:-k], d_imp[k:])[0, 1]) for k in lags)

    return stat


def import_response_test(
    signal: pd.Series,
    imports_t: pd.Series,
    *,
    alpha: float,
    min_offset_months: int,
    lags=(1, 2, 3),
    rng=0,
) -> TestResult:
    """Pass if the best lagged correlation is positive and beats a circular-shift placebo.

    The placebo takes the same maximum over lags, so picking the best lag is priced in.
    """
    seal = sealed_after()
    if seal is not None:
        signal, imports_t = signal.loc[:seal], imports_t.loc[:seal]
    j = pd.concat([signal.astype(float), imports_t.pct_change()], axis=1).dropna()
    sig, d_imp = j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy()
    res = circular_shift_placebo(
        sig, d_imp, _max_lagged_corr(lags), min_offset=min_offset_months, rng=rng
    )
    ok = res.observed > 0 and res.p_value <= alpha
    return TestResult(
        5,
        "import response test",
        "pass" if ok else "fail",
        "max corr over lags 1-3 (placebo p)",
        res.observed,
        f"> 0 and p <= {alpha}",
        note=f"placebo p = {res.p_value:.3f}, n months = {len(j)}",
    )


def cbam_cost_check(*_, **__) -> TestResult:
    require_frozen()  # 2026 import volumes are hold-out data
    raise NotImplementedError(
        "CBAM cost check: allowed now that the freeze is in place; not built yet"
    )


def comext_imports_t(store, product: str = "2814", *, as_of=None) -> pd.Series:
    """Monthly EU-27 extra-EU imports in tonnes from the ``comext`` PIT source."""
    df = store.read("comext", as_of=as_of)
    df = df[
        df.series_id.str.startswith(f"COMEXT.EU27.IMP.{product}.")
        & df.series_id.str.endswith(".QUANTITY_IN_100KG")
    ]
    partner = df.series_id.str.split(".").str[4]
    df = df[~partner.isin(EXCLUDE_PARTNERS)]
    return df.groupby("observed_at").value.sum().mul(0.1).rename(f"imports_{product}_t")


# Comext aggregates and intra-EU partners, which would double count if summed with countries.
EXCLUDE_PARTNERS = frozenset(
    {"EU27_2020", "EXT_EU27_2020", "INT_EU27_2020", "EU", "EXT_EU", "INT_EU", "WORLD", "EA"}
    | EU27_ENTSOG
)


def reconcile_monthly(daily: pd.Series, monthly: pd.Series, *, tolerance: float) -> pd.DataFrame:
    """Calendar-month mean of a daily series vs an official monthly series.

    Both in the same unit. ``monthly`` is indexed by the first day of the month. Only
    months that both series cover are compared. The series passes if every month is
    within ``tolerance`` (relative).
    """
    d = daily.dropna()
    m = d.groupby(d.index.to_period("M")).agg(["mean", "count"])
    ref = monthly.set_axis(pd.DatetimeIndex(monthly.index).to_period("M"))
    out = m.join(ref.rename("official"), how="inner")
    out["deviation"] = out["mean"] / out.official - 1
    out["within"] = out.deviation.abs() <= tolerance
    return out
