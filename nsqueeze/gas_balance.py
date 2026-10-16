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
import json
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


def population_weighted_reference(hdd_nuts2: pd.DataFrame, pop_nuts2: pd.DataFrame) -> pd.DataFrame:
    """Monthly HDD per EU-27 country and ``EU27_2020``, NUTS-2 regions weighted by population.

    Eurostat's own country and EU HDD weight regions by area. Area weighting measures the
    cold of the land, not the cold where people heat buildings. A month is NaN for a
    country if one of its regions has no HDD or no population for that year.
    """
    from nsqueeze.io.sources.eurostat import EU27_GEO

    h = hdd_nuts2.assign(geo=hdd_nuts2.series_id.str.split(".").str[2])
    h = h[h.geo.str[:2].isin(EU27_GEO)].pivot_table(
        index="observed_at", columns="geo", values="value"
    )
    p = pop_nuts2.assign(
        geo=pop_nuts2.series_id.str.split(".").str[2], year=pop_nuts2.observed_at.dt.year
    )
    p = p.pivot_table(index="year", columns="geo", values="value").reindex(columns=h.columns)
    w = p.reindex(h.index.year).set_axis(h.index)
    country = h.columns.str[:2]
    num = (h * w).T.groupby(country).sum().T
    den = w.T.groupby(country).sum().T
    complete = (h.notna() & w.notna()).T.groupby(country).all().T
    out = (num / den).where(complete)
    eu = num.sum(axis=1) / den.sum(axis=1)
    out["EU27_2020"] = eu.where(complete.all(axis=1) & (complete.shape[1] == len(EU27_GEO)))
    return out


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
    exports. Partners inside the EU-27 and ``TOTAL`` drop out; regional "other"
    partners are outside the EU-27 and count as other pipeline. ``NSP`` (not
    specified) counts as LNG for G3200, since the product already says LNG; pipeline
    NSP cannot be split between Norway and other pipeline and drops out.

    Eurostat does not publish the EU-27 aggregate for some months (most of 2019 and
    all of 2020), and in March 2024 it publishes the total without a partner split.
    A month counts only if the import total and Norway are published and the named
    partners (NSP included) sum to at least 99% of the total. Other months are left
    out, not read as zero.
    """
    parts = trade.series_id.str.split(".", expand=True)
    t = trade.assign(flow=parts[2], siec=parts[3], partner=parts[4])
    imp = t[(t.flow == "IMP") & (t.siec == "G3000")]
    total = imp[imp.partner == "TOTAL"].set_index("observed_at").value
    named = imp[imp.partner != "TOTAL"].groupby("observed_at").value.sum()
    complete = named.reindex(total.index) >= 0.99 * total
    published = set(total.index[complete]) & set(imp[imp.partner == "NO"].observed_at)
    t = t[t.observed_at.isin(published)]
    t = t[~t.partner.isin(EU27_EUROSTAT | {"TOTAL"})]
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
            "other_pipe": pipe.drop(columns=["NO", "NSP"], errors="ignore").sum(axis=1),
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


# Layer 1 inputs (PIT rows -> one daily frame)

EU_STORAGE = {"AGSI.EU.gasInStorage": "storage", "AGSI.EU.workingGasVolume": "wgv"}


def eu_storage(agsi: pd.DataFrame) -> pd.DataFrame:
    """EU gas in storage and working-gas volume, TWh per gas day, from ``agsi`` rows."""
    a = agsi[agsi.series_id.isin(EU_STORAGE)]
    w = a.assign(value=pd.to_numeric(a.value, errors="coerce")).pivot_table(
        index="observed_at", columns="series_id", values="value", aggfunc="last"
    )
    return w.rename(columns=EU_STORAGE).reindex(columns=list(EU_STORAGE.values()))


def daily_hdd(weather: pd.DataFrame, population: pd.DataFrame) -> pd.Series:
    """Population-weighted EU HDD per day from ``weather`` and ``eurostat_population`` rows.

    Country HDD first, then the weights (see ``population_weighted``). A day's weights
    are the populations on 1 January of its year, or the latest earlier year. A day
    with a missing country temperature is NaN, not a smaller average.
    """
    t = weather.pivot_table(index="observed_at", columns="series_id", values="value")
    geos = sorted({c.split(".")[1] for c in t.columns})
    tmax = t[[f"OM.{g}.temperature_2m_max" for g in geos]].set_axis(geos, axis=1)
    tmin = t[[f"OM.{g}.temperature_2m_min" for g in geos]].set_axis(geos, axis=1)
    hdd = heating_degree_days(eurostat_mean_temp(tmax, tmin))
    pop = population[population.series_id.str.match(r"ESTAT\.POP\.[A-Z]{2}$")]
    p = pop.assign(geo=pop.series_id.str[-2:], year=pop.observed_at.dt.year)
    p = p.pivot_table(index="year", columns="geo", values="value").reindex(columns=geos)
    years = range(min(p.index.min(), hdd.index.year.min()), hdd.index.year.max() + 1)
    w = p.reindex(years).ffill().reindex(hdd.index.year).set_axis(hdd.index)
    out = (hdd * w).sum(axis=1, min_count=1) / w.sum(axis=1)
    return out.where(hdd.notna().all(axis=1)).rename("hdd")


def power_generation(entsoe: pd.DataFrame) -> pd.DataFrame:
    """Gas-fired electricity per gas day, TWh_e, from ``entsoe`` rows (MW at native resolution).

    The gas day runs 06:00-06:00 Brussels time. ``countries`` counts the countries
    with data that day: coverage changes over time (see the ENTSO-E gaps), and the
    caller decides which days to use. This is electricity, not gas: the fit turns it
    into gas burn through its own coefficient, so no plant efficiency is assumed here.
    """
    res = entsoe.meta.map(
        {m: pd.Timedelta(json.loads(m)["resolution"]) for m in entsoe.meta.unique()}
    )
    mwh = entsoe.value * (res / pd.Timedelta(hours=1))
    local = pd.DatetimeIndex(entsoe.observed_at).tz_localize("UTC").tz_convert("Europe/Brussels")
    day = (local - pd.Timedelta(hours=6)).tz_localize(None).normalize()
    f = pd.DataFrame(
        {
            "day": day,
            "geo": entsoe.series_id.str.split(".").str[1].to_numpy(),
            "mwh": mwh.to_numpy(),
        }
    )
    g = f.groupby("day")
    return pd.DataFrame(
        {"power_twh_e": units.mwh_to_twh(g.mwh.sum()), "countries": g.geo.nunique()}
    ).rename_axis("observed_at")


# EU countries with ENTSO-E gas-fired generation from January 2015 (ENTSO-E codes).
# BG starts 2017, HR 2019, SE Dec 2021, CY Oct 2025, MT has no data. Fixed for G2-b.
POWER_PANEL = tuple(sorted(EU27_ENTSOG - {"BG", "HR", "SE", "CY", "MT"}))


def power_panel(
    entsoe: pd.DataFrame, panel: tuple[str, ...] = POWER_PANEL, max_gap_days: int = 7
) -> pd.Series:
    """Gas-fired electricity of a fixed set of countries, TWh_e per gas day.

    A country's missing days are filled by linear interpolation when the gap is at most
    ``max_gap_days`` long. A day with a longer gap in any panel country is NaN.
    """
    e = entsoe[entsoe.series_id.str.split(".").str[1].isin(panel)]
    by_geo = pd.concat(
        {g: power_generation(e[e.series_id.str.split(".").str[1] == g]).power_twh_e for g in panel},
        axis=1,
    )
    by_geo = by_geo.reindex(pd.date_range(by_geo.index.min(), by_geo.index.max()))
    gap = by_geo.isna()
    run_len = gap.apply(lambda s: s.groupby((s != s.shift()).cumsum()).transform("sum"))
    filled = by_geo.interpolate(limit_area="inside").where(~gap | (run_len <= max_gap_days))
    return filled.sum(axis=1).where(filled.notna().all(axis=1)).rename("power_twh_e")


def layer1_frame(
    storage: pd.DataFrame, supply: pd.DataFrame, hdd: pd.Series, power: pd.Series | None = None
) -> pd.DataFrame:
    """Daily frame for ``backcast``/``BalanceModel``: storage, wgv, supply, hdd[, power_burn].

    ``supply`` is the bucket frame from ``supply_from_entsog``; its buckets are summed.
    Only days present in every input are kept, nothing is filled.
    """
    parts = [storage[["storage", "wgv"]], supply.sum(axis=1).rename("supply"), hdd.rename("hdd")]
    if power is not None:
        parts.append(power.rename("power_burn"))
    return pd.concat(parts, axis=1, join="inner").dropna().sort_index()


def layer1_inputs(store, as_of=None, *, power: bool = False) -> pd.DataFrame:
    """Read every Layer 1 input from the PIT store (as known at ``as_of``) into one frame.

    Supply uses the boundary classes plus ``config/entsog_landing_points.csv`` and
    ``config/entsog_archive_points.csv``, as in the OI-26 checks.
    """
    from nsqueeze.config import config_dir

    cfg = config_dir()
    flows, extra = archive_points(
        store.read("entsog", as_of=as_of), pd.read_csv(cfg / "entsog_archive_points.csv", dtype=str)
    )
    classes = landing_points(
        classify_points(pd.read_csv(cfg / "entsog_interconnections.csv")),
        pd.read_csv(cfg / "entsog_landing_points.csv"),
    )
    supply = supply_from_entsog(flows, pd.concat([classes, extra], ignore_index=True))
    hdd = daily_hdd(
        store.read("weather", as_of=as_of), store.read("eurostat_population", as_of=as_of)
    )
    pw = power_panel(store.read("entsoe", as_of=as_of)) if power else None
    return layer1_frame(eu_storage(store.read("agsi", as_of=as_of)), supply, hdd, pw)


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

    def path(
        self, start_storage_twh: float, index: pd.DatetimeIndex, supply, hdd, power_burn=None
    ) -> pd.Series:
        """Storage at the end of each day, given supply, HDD [and power] paths (no residual)."""
        flow = np.asarray(supply, float) - self.demand(index, hdd, power_burn)
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


def fan_chart(
    model: BalanceModel,
    start_storage_twh: float,
    start: dt.date,
    until: dt.date,
    hdd_history: pd.Series,
    supply_twh_d: dict[str, float],
    *,
    weather_years: list[int],
    draws: int,
    wgv_twh: float,
    thresholds_pct: list[float],
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """P(storage on ``until`` < X % of working gas), per supply scenario and pooled.

    Each draw adds one block of fitted residuals as long as the whole window. Every
    scenario starts from the same seed, so scenarios differ only by supply. Pooled is
    the equal-weight mean over scenarios.
    """
    block = (pd.Timestamp(until) - pd.Timestamp(start)).days
    out = {}
    for name, s in supply_twh_d.items():
        rng = np.random.default_rng(seed)
        ends = [
            model.simulate(
                start_storage_twh,
                start,
                until,
                hdd_history,
                SupplyScenario(name, s),
                weather_years=weather_years,
                block_days=block,
                rng=rng,
            )
            for _ in range(draws)
        ]
        if any(len(e) != len(weather_years) for e in ends):
            raise ValueError("HDD history does not cover every weather year for this window")
        share = 100 * np.concatenate([e.to_numpy() for e in ends]) / wgv_twh
        out[name] = {f"{x:g}": breach_probability(share, x) for x in thresholds_pct}
    keys = [f"{x:g}" for x in thresholds_pct]
    out["pooled"] = {k: float(np.mean([out[n][k] for n in supply_twh_d])) for k in keys}
    return out


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
    power: bool = False,
) -> pd.DataFrame:
    """Back-cast end-of-winter storage with realised supply and weather [and power burn].

    ``frame`` columns: storage (TWh, end of gas day), wgv (TWh), supply (TWh/d), hdd.
    For each winter the model is fitted on every other day in ``frame`` (the winter
    itself is excluded), then the storage path is rebuilt from the observed start.
    A winter with a missing day is left out (missing). Errors are in percentage points
    of working-gas volume. ``power=True`` adds the ``power_burn`` column (G2-b).
    """
    f = frame.copy()
    f["dS"] = f.storage.diff()
    rows = []
    for w in winters:
        a, b = winter_window(w, start_md, end_md)
        if a not in f.index or b not in f.index:
            continue
        test = f.loc[a + pd.Timedelta(days=1) : b]
        if len(test) != (b - a).days:
            continue
        train = f[(f.index <= a) | (f.index > b)]
        m = BalanceModel(break_date=break_date, use_power_burn=power).fit(train)
        pw = test.power_burn if power else None
        pred = m.path(f.storage.loc[a], pd.DatetimeIndex(test.index), test.supply, test.hdd, pw)
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
