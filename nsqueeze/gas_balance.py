"""Layer 1

EU gas balance nowcast and end-of-winter fan chart.

    ΔS_t = (N_t + L_t + P_t) − (β_HDD · HDD_t + G_t + I_t) + ε_t
           Norway, LNG send-out,   weather demand, power burn,
           other pipeline          industry (← EAUI feeds back here)

Estimation works on implied consumption D_t ≡ supply_t − ΔS_t:

    D_t = α_r + β_r · HDD_t + Σ_k δ_k · weekday_k [+ γ · G_t] + ε_t,   r ∈ {pre, post 2022 break}

Calibration: back-cast end-of-winter storage for 2016–2025 must land within a pre-set error of AGSI+
actuals before any fan chart is published. The back-cast is leave-one-winter-out.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nsqueeze import units

# HDD
# HDD = 18 °C − Tm if Tm ≤ 15 °C, else 0, with Tm = (Tmax + Tmin) / 2 (Eurostat nrg_chdd metadata)
HDD_REFERENCE_C = 18.0
HDD_THRESHOLD_C = 15.0


def eurostat_mean_temp(tmax_c, tmin_c):
    """Tm as Eurostat defines it: the mid-range, not the 24-hour mean."""
    return (tmax_c + tmin_c) / 2


def heating_degree_days(
    mean_temp_c, reference_c: float = HDD_REFERENCE_C, threshold_c: float = HDD_THRESHOLD_C
):
    """Daily HDD, Eurostat definition. Series and frames keep NaN as NaN."""
    t = mean_temp_c
    if isinstance(t, pd.Series | pd.DataFrame):
        return (reference_c - t).mask(t > threshold_c, 0.0)
    t = np.asarray(t, float)
    return np.where(t <= threshold_c, reference_c - t, 0.0)


def population_weighted(by_geo: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Collapse a (date × country) frame with population weights.

    Weight country **HDD**, not temperatures: the 15 °C threshold is non-linear, so
    the HDD of an averaged temperature is not the average HDD.
    """
    w = weights.reindex(by_geo.columns).fillna(0)
    return by_geo.mul(w, axis=1).sum(axis=1) / w.sum()


def hdd_check(daily_hdd: pd.DataFrame, eurostat_monthly: pd.DataFrame) -> pd.DataFrame:
    """Compare monthly sums of my daily HDD with Eurostat's, per country.

    Both frames are (date × geo). Returns correlation and mean absolute percentage
    error per geo over the months both cover. This is how the grade-C capital proxy
    in ``config/weather_points.csv`` earns (or loses) its place.
    """
    ours = daily_hdd.resample("MS").sum(min_count=28)
    rows = []
    for g in ours.columns.intersection(eurostat_monthly.columns):
        a, b = ours[g], eurostat_monthly[g]
        j = pd.concat([a, b], axis=1, keys=["ours", "eurostat"]).dropna()
        winter = j[j.eurostat > 50]  # summer months have near-zero HDD and explode MAPE
        rows.append(
            {
                "geo": g,
                "months": len(j),
                "corr": j.ours.corr(j.eurostat),
                "mape_winter": float((winter.ours / winter.eurostat - 1).abs().mean()),
                "bias_winter": float((winter.ours - winter.eurostat).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("geo")


# Supply from ENTSOG

EU27_ENTSOG = frozenset(
    [
        "AT",
        "BE",
        "BG",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "ES",
        "FI",
        "FR",
        "GR",
        "HR",
        "HU",
        "IE",
        "IT",
        "LT",
        "LU",
        "LV",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SE",
        "SI",
        "SK",
    ]
)
BUCKETS = ("norway", "lng", "other_pipe", "production")


def classify_points(interconnections: pd.DataFrame) -> pd.DataFrame:
    """Map (pointKey, EU-side operator) to a supply bucket.

    ``interconnections`` is ``config/entsog_interconnections.csv`` (an ENTSOG API
    snapshot). Rules, in order: LNG terminal → ``lng``; production → ``production``;
    from Norway → ``norway``; from any other non-EU country (incl. UK) → ``other_pipe``.
    Only flows *into* an EU-27 system count; intra-EU points are transit.
    """
    ic = interconnections
    into_eu = ic.toCountryKey.isin(EU27_ENTSOG)
    from_infra = ic.fromInfrastructureTypeLabel.fillna("")
    bucket = np.select(
        [
            from_infra.eq("LNG Terminals"),
            from_infra.eq("Production"),
            ic.fromCountryKey.eq("NO"),
            ~ic.fromCountryKey.isin(EU27_ENTSOG) & from_infra.eq("Transmission"),
        ],
        ["lng", "production", "norway", "other_pipe"],
        default="",
    )
    out = pd.DataFrame({"pointKey": ic.pointKey, "operatorKey": ic.toOperatorKey, "bucket": bucket})
    return out[into_eu & (out.bucket != "")].drop_duplicates().reset_index(drop=True)


def supply_from_entsog(flows: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Daily supply by bucket, TWh/d, from PIT rows of the ``entsog`` source.

    Entry flows only, reported by the EU-side operator, so each physical flow is
    counted once even when both TSOs publish it.
    """
    parts = flows.series_id.str.split(".", expand=True)
    f = flows.assign(operatorKey=parts[1], pointKey=parts[2], direction=parts[3])
    f = f[f.direction == "entry"].merge(classes, on=["pointKey", "operatorKey"])
    out = f.pivot_table(index="observed_at", columns="bucket", values="value", aggfunc="sum")
    return units.kwh_to_twh(out.reindex(columns=list(BUCKETS)).fillna(0.0))


# Model


@dataclass(frozen=True)
class SupplyScenario:
    name: str  # "lng_mean" | "lng_low" | "lng_high"
    supply_twh_d: float  # total daily supply into the EU system under the scenario


def _design(index: pd.DatetimeIndex, hdd: np.ndarray, post: np.ndarray, extra=None):
    """Columns: α_pre, α_post, β_pre, β_post, weekday dummies (Mon = base), extras."""
    wd = index.dayofweek.to_numpy()
    cols = [1 - post, post, hdd * (1 - post), hdd * post]
    cols += [(wd == k).astype(float) for k in range(1, 7)]
    if extra is not None:
        cols.append(extra)
    return np.column_stack(cols)


@dataclass
class BalanceModel:
    """Daily balance identity with a structural break. Phase 2."""

    break_date: dt.date = dt.date(2022, 3, 1)
    coef_: np.ndarray | None = None
    resid_: pd.Series | None = None
    use_power_burn: bool = False
    names_: list[str] = field(default_factory=list)

    def _post(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index >= pd.Timestamp(self.break_date)).astype(float)

    def fit(self, frame: pd.DataFrame) -> BalanceModel:
        """``frame`` (daily index) needs ``dS``, ``supply`` and ``hdd``; ``power_burn`` optional.

        All flows in TWh/d. Rows with any missing input are dropped, not filled.
        """
        cols = ["dS", "supply", "hdd"] + (["power_burn"] if self.use_power_burn else [])
        f = frame[cols].dropna()
        idx = pd.DatetimeIndex(f.index)
        demand = (f.supply - f.dS).to_numpy()
        extra = f.power_burn.to_numpy() if self.use_power_burn else None
        X = _design(idx, f.hdd.to_numpy(), self._post(idx), extra)
        self.coef_, *_ = np.linalg.lstsq(X, demand, rcond=None)
        self.resid_ = pd.Series(demand - X @ self.coef_, index=idx, name="resid")
        self.names_ = ["alpha_pre", "alpha_post", "beta_pre", "beta_post"]
        self.names_ += [f"weekday_{k}" for k in range(1, 7)]
        self.names_ += ["gamma_power"] if self.use_power_burn else []
        return self

    @property
    def coefficients(self) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("fit the model first")
        return pd.Series(self.coef_, index=self.names_)

    def demand(self, index: pd.DatetimeIndex, hdd, power_burn=None) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit the model first")
        idx = pd.DatetimeIndex(index)
        extra = None
        if self.use_power_burn:
            extra = np.asarray(power_burn if power_burn is not None else np.zeros(len(idx)), float)
        return _design(idx, np.asarray(hdd, float), self._post(idx), extra) @ self.coef_

    def path(self, start_storage_twh: float, index: pd.DatetimeIndex, supply, hdd) -> pd.Series:
        """Storage at the end of each day, given supply and HDD paths (no residual)."""
        flow = np.asarray(supply, float) - self.demand(index, hdd)
        return pd.Series(start_storage_twh + np.cumsum(flow), index=index, name="storage")

    def simulate(
        self,
        start_storage_twh: float,
        start: dt.date,
        until: dt.date,
        hdd_history: pd.Series,
        scenario: SupplyScenario,
        *,
        weather_years: list[int] | None = None,
        block_days: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> pd.Series:
        """Storage on ``until`` under each historical weather year.

        Days ``start+1 … until`` are replayed with the HDD of the same calendar days in
        weather year y. With ``block_days`` set, residuals are added by a moving-block
        bootstrap from the fitted residuals (one draw per weather year).
        """
        index = pd.date_range(pd.Timestamp(start) + pd.Timedelta(days=1), until, freq="D")
        years = weather_years or sorted(
            {y for y in hdd_history.index.year if _covers(hdd_history, index, y - start.year)}
        )
        out = {}
        for y in years:
            hdd = _analog(hdd_history, index, y - start.year)
            if hdd is None:
                continue
            end = self.path(start_storage_twh, index, scenario.supply_twh_d, hdd).iloc[-1]
            if block_days:
                end += _block_draw(self.resid_.to_numpy(), len(index), block_days, rng)
            out[y] = end
        return pd.Series(out, name=f"storage_{until}").rename_axis("weather_year")


def _analog(hdd: pd.Series, index: pd.DatetimeIndex, shift_years: int) -> np.ndarray | None:
    """HDD on the same calendar days, ``shift_years`` earlier/later. 29 Feb → 28 Feb."""
    dates = [
        d.replace(year=d.year + shift_years, day=28 if (d.month, d.day) == (2, 29) else d.day)
        for d in index
    ]
    vals = hdd.reindex(pd.DatetimeIndex(dates))
    return None if vals.isna().any() else vals.to_numpy()


def _covers(hdd: pd.Series, index: pd.DatetimeIndex, shift_years: int) -> bool:
    try:
        return _analog(hdd, index, shift_years) is not None
    except ValueError:
        return False


def _block_draw(resid: np.ndarray, n: int, block: int, rng: np.random.Generator | None) -> float:
    """Sum of ``n`` residuals drawn in contiguous blocks (keeps their autocorrelation)."""
    rng = rng or np.random.default_rng()
    total, left = 0.0, n
    while left > 0:
        k = min(block, left)
        i = rng.integers(0, len(resid) - k + 1)
        total += resid[i : i + k].sum()
        left -= k
    return float(total)


def breach_probability(end_storage_share: np.ndarray, threshold: float) -> float:
    """P(storage on 1 April < threshold) from simulated end-of-winter fill shares."""
    x = np.asarray(end_storage_share, float)
    return float(np.mean(x < threshold))


# leave-one-winter-out back-cast


def winter_window(
    end_year: int, start_md: tuple[int, int], end_md: tuple[int, int]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Winter ``end_year`` runs from ``start_md`` of end_year − 1 to ``end_md`` of end_year."""
    return (
        pd.Timestamp(end_year - 1, *start_md),
        pd.Timestamp(end_year, *end_md),
    )


def backcast(
    frame: pd.DataFrame,
    winters: list[int],
    *,
    start_md: tuple[int, int],
    end_md: tuple[int, int],
    break_date: dt.date = dt.date(2022, 3, 1),
) -> pd.DataFrame:
    """Back-cast end-of-winter storage with realised supply and weather.

    ``frame`` columns: storage (TWh, end of gas day), wgv (TWh), supply (TWh/d), hdd.
    For each winter the model is fitted on every other day in ``frame`` (the winter
    itself is excluded), then the storage path is rebuilt from the observed start.
    Errors are in percentage points of working-gas volume.
    """
    f = frame.copy()
    f["dS"] = f.storage.diff()
    rows = []
    for w in winters:
        a, b = winter_window(w, start_md, end_md)
        if a not in f.index or b not in f.index:
            continue
        train = f[(f.index <= a) | (f.index > b)]
        m = BalanceModel(break_date=break_date).fit(train)
        test = f.loc[a + pd.Timedelta(days=1) : b]
        pred = m.path(f.storage.loc[a], pd.DatetimeIndex(test.index), test.supply, test.hdd)
        wgv = f.wgv.loc[b]
        rows.append(
            {
                "winter": w,
                "start": a.date(),
                "end": b.date(),
                "actual_twh": f.storage.loc[b],
                "predicted_twh": pred.iloc[-1],
                "actual_pct": 100 * f.storage.loc[b] / wgv,
                "predicted_pct": 100 * pred.iloc[-1] / wgv,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["error_pp"] = out.predicted_pct - out.actual_pct
    return out


@dataclass(frozen=True)
class G2Result:
    passed: bool
    mae_pp: float
    max_abs_pp: float
    mae_limit_pp: float
    max_limit_pp: float
    winters: int


def g2_gate(bc: pd.DataFrame, *, mae_limit_pp: float, max_limit_pp: float) -> G2Result:
    """Limits are passed in by the caller, never defaulted."""
    err = bc.error_pp.abs()
    mae, mx = float(err.mean()), float(err.max())
    return G2Result(
        mae <= mae_limit_pp and mx <= max_limit_pp, mae, mx, mae_limit_pp, max_limit_pp, len(bc)
    )
