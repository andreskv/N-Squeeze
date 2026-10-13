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
    """Map (pointKey, EU-side operator) on the EU-27 boundary to a supply bucket.

    ``interconnections`` is ``config/entsog_interconnections.csv`` (an ENTSOG API
    snapshot). A row crosses the boundary in either direction. The bucket comes from
    the side outside the EU-27 system, in this order: LNG terminal → ``lng``;
    production → ``production``; Norway → ``norway``; any other non-EU transmission
    system (incl. UK) → ``other_pipe``. Intra-EU points are transit and drop out.
    Virtual points (``VTP-``) drop out too: Spain's virtual LNG tank (TVB) restates
    the send-out that the individual terminals already report.
    """
    ic = interconnections
    into = ic.toCountryKey.isin(EU27_ENTSOG)
    out_of = ic.fromCountryKey.isin(EU27_ENTSOG) & ~ic.toCountryKey.isin(EU27_ENTSOG)
    rows = []
    for side, op, country, infra in (
        (into, ic.toOperatorKey, ic.fromCountryKey, ic.fromInfrastructureTypeLabel),
        (out_of, ic.fromOperatorKey, ic.toCountryKey, ic.toInfrastructureTypeLabel),
    ):
        infra = infra.fillna("")
        bucket = np.select(
            [
                infra.eq("LNG Terminals"),
                infra.eq("Production"),
                country.eq("NO"),
                ~country.isin(EU27_ENTSOG) & infra.eq("Transmission"),
            ],
            ["lng", "production", "norway", "other_pipe"],
            default="",
        )
        rows.append(
            pd.DataFrame({"pointKey": ic.pointKey, "operatorKey": op, "bucket": bucket})[side]
        )
    out = pd.concat(rows)
    keep = (out.bucket != "") & out.operatorKey.notna() & ~out.pointKey.str.startswith("VTP-")
    return out[keep].drop_duplicates(["pointKey", "operatorKey"]).reset_index(drop=True)


def landing_points(classes: pd.DataFrame, landings: pd.DataFrame) -> pd.DataFrame:
    """Add the series from ``config/entsog_landing_points.csv``.

    The interconnection snapshot lists only points valid now, so it misses the
    per-operator keys at Emden and Dornum and the closed Russian routes (Nord Stream
    legs at Greifswald, Kondratki). Several operators report one landing flow: the
    file names each series' landing and bucket.
    """
    extra = landings[["pointKey", "operatorKey", "bucket", "landing"]]
    rest = classes.merge(extra[["pointKey", "operatorKey"]], how="left", indicator=True)
    rest = rest[rest._merge == "left_only"].drop(columns="_merge")
    return pd.concat([rest, extra], ignore_index=True)


def archive_points(flows: pd.DataFrame, table: pd.DataFrame):
    """Give TP Archives series without an API key (``ENTSOG.EIC.…``, ``ENTSOG.NAME.…``)
    a place in the buckets, from ``config/entsog_archive_points.csv``.

    A row with a ``pointKey`` renames the series to that API key, so the normal
    classification applies. A row with a ``bucket`` (a point that has no API
    counterpart) becomes its own point. Returns (flows, extra class rows).
    A renamed series never replaces a day the API key already has.
    """
    stem, direction = (
        flows.series_id.str.rsplit(".", n=1).str[0],
        flows.series_id.str.rsplit(".", n=1).str[1],
    )
    t = table.fillna("").reset_index(drop=True)
    new = {}
    for i, r in t.iterrows():
        new[r.series] = (
            f"ENTSOG.{r.operatorKey}.{r.pointKey}" if r.pointKey else f"ENTSOG.ARCHIVE.A{i}"
        )
    renamed = stem.map(new)
    out = flows.assign(series_id=(renamed + "." + direction).fillna(flows.series_id))
    out = out.assign(_old=renamed.isna()).sort_values("_old", ascending=False)
    out = out.drop_duplicates(["series_id", "observed_at"]).drop(columns="_old").sort_index()
    m = t[t.pointKey == ""]
    extra = pd.DataFrame(
        {
            "pointKey": [f"A{i}" for i in m.index],
            "operatorKey": "ARCHIVE",
            "bucket": m.bucket.to_numpy(),
            "landing": m.landing.replace("", None).to_numpy(),
        }
    )
    return out, extra


def supply_from_entsog(flows: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Daily net supply across the EU-27 boundary by bucket, TWh/d, from ``entsog`` PIT rows.

    Net = entry − exit, as reported by the EU-side operator, so each physical flow is
    counted once even when both TSOs publish it, and gas that leaves the EU and comes
    back (TurkStream via Serbia, TAP via Italy) cancels. If ``classes`` has a
    ``landing`` column, series that share a landing point count once (the largest).
    """
    parts = flows.series_id.str.split(".", expand=True)
    f = flows.assign(operatorKey=parts[1], pointKey=parts[2], direction=parts[3])
    f = f[f.direction.isin(["entry", "exit"])].merge(classes, on=["pointKey", "operatorKey"])
    f["net"] = f.value.where(f.direction == "entry", -f.value)
    if "landing" in f:
        # Several operators report one landing flow: per point and day, net each
        # series, then keep the largest series of each landing point once.
        f["landing"] = f.landing.fillna(f.pointKey + "|" + f.operatorKey)
        per_series = f.groupby(
            ["observed_at", "bucket", "landing", "pointKey", "operatorKey"]
        ).net.sum()
        f = per_series.groupby(["observed_at", "bucket", "landing"]).max().reset_index()
    out = f.pivot_table(index="observed_at", columns="bucket", values="net", aggfunc="sum")
    return units.kwh_to_twh(out.reindex(columns=list(BUCKETS)).fillna(0.0))


# Eurostat EU-27 country codes (Greece is EL here, GR in ENTSOG)
EU27_EUROSTAT = frozenset((EU27_ENTSOG - {"GR"}) | {"EL"})


def eurostat_net_by_bucket(trade: pd.DataFrame) -> pd.DataFrame:
    """Monthly net imports into the EU-27 by bucket, TWh, from ``eurostat_gas_trade`` rows.

    Pipeline gas = natural gas (G3000) − LNG (G3200), per partner. Net = imports −
    exports. Partners inside the EU-27, ``TOTAL`` and ``NSP`` (not specified) drop
    out; regional "other" partners are outside the EU-27 and count as other pipeline.

    Eurostat does not publish the EU-27 aggregate for some months (most of 2019 and
    all of 2020). A month counts only if both the import total and Norway are
    published. Other months are left out, not read as zero.
    """
    parts = trade.series_id.str.split(".", expand=True)
    t = trade.assign(flow=parts[2], siec=parts[3], partner=parts[4])
    imp = t[(t.flow == "IMP") & (t.siec == "G3000")]
    published = set(imp[imp.partner == "TOTAL"].observed_at) & set(
        imp[imp.partner == "NO"].observed_at
    )
    t = t[t.observed_at.isin(published)]
    t = t[~t.partner.isin(EU27_EUROSTAT | {"TOTAL", "NSP"})]
    t["signed"] = t.value.where(t.flow == "IMP", -t.value)
    w = t.pivot_table(
        index="observed_at", columns=["siec", "partner"], values="signed", aggfunc="sum"
    )
    w = w.fillna(0.0)
    lng = w["G3200"].sum(axis=1)
    pipe = w["G3000"].sub(w["G3200"].reindex(columns=w["G3000"].columns, fill_value=0.0))
    out = pd.DataFrame(
        {
            "norway": pipe.get("NO", 0.0),
            "lng": lng,
            "other_pipe": pipe.drop(columns="NO", errors="ignore").sum(axis=1),
        }
    )
    return units.tj_to_twh(out)


def bucket_check(entsog_daily: pd.DataFrame, eurostat_monthly: pd.DataFrame, tolerance: float):
    """Share of months where the ENTSOG bucket is within ``tolerance`` of Eurostat.

    Months need a full set of ENTSOG days. Returns (summary per bucket, monthly table).
    """
    days = entsog_daily.resample("MS").size()
    ours = entsog_daily.resample("MS").sum()[days == days.index.days_in_month]
    j = ours.join(eurostat_monthly, lsuffix="_entsog", rsuffix="_eurostat", how="inner")
    rows = {}
    for b in eurostat_monthly.columns:
        err = j[f"{b}_entsog"] / j[f"{b}_eurostat"] - 1
        j[f"{b}_err"] = err
        rows[b] = {
            "months": int(err.notna().sum()),
            "within": float((err.abs() <= tolerance).mean()),
            "median_err": float(err.median()),
        }
    return pd.DataFrame(rows).T, j


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
