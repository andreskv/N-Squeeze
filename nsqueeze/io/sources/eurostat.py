"""Eurostat (key-free JSON-stat 2.0 APIs), all checked live on 2026-10-12.

``comext``              EU-27 imports by partner, CN 2814 (ammonia) and HS6 310210 (urea),
                        monthly tonnes and euros, DS-045409.
``eurostat_population`` population on 1 January by country (demo_pjan): the weights for the
                        population-weighted HDD.
``eurostat_hdd``        monthly heating degree days by country (nrg_chdd_m, JRC data,
                        Eurostat definition). The reference I check my daily HDD against.
``eurostat_gas_trade``  EU-27 monthly gas imports (nrg_ti_gasm) and exports (nrg_te_gasm)
                        by partner, natural gas (G3000) and LNG (G3200), TJ (GCV). The
                        reference for the ENTSOG supply buckets. Checked live on 2026-10-13.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import ClassVar

import pandas as pd

from nsqueeze.io.sources.base import Source

COMEXT_URL = "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/DS-045409"
STATS_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

EU27_GEO = tuple(
    [
        "AT",
        "BE",
        "BG",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "EL",
        "ES",
        "FI",
        "FR",
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
COMEXT_PRODUCTS = ("2814", "310210")
COMEXT_INDICATORS = {"QUANTITY_IN_100KG": "100kg", "VALUE_IN_EUROS": "EUR"}


def parse_jsonstat(body: dict) -> pd.DataFrame:
    """JSON-stat 2.0 dataset → long frame, one column per dimension plus ``value``.

    ``value`` is sparse: a dict from the row-major flat index to the observation.
    """
    ids, sizes = body["id"], body["size"]
    cats = []
    for d in ids:
        idx = body["dimension"][d]["category"]["index"]
        order = sorted(idx, key=idx.get) if isinstance(idx, dict) else list(idx)
        cats.append(order)
    values = body.get("value") or {}
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    rows = []
    for k, v in values.items():
        k = int(k)
        rows.append(
            {d: cats[i][(k // strides[i]) % sizes[i]] for i, d in enumerate(ids)} | {"value": v}
        )
    return pd.DataFrame(rows, columns=[*ids, "value"])


def _month_start(period: str) -> pd.Timestamp:
    return pd.Timestamp(period + "-01") if len(period) == 7 else pd.Timestamp(f"{period}-01-01")


class Comext(Source):
    name = "comext"
    verified = True

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        r = self.client.get(
            COMEXT_URL,
            params={
                "reporter": "EU27_2020",  # fixed composition; "EU" changes membership over time
                "product": list(COMEXT_PRODUCTS),
                "flow": "1",  # imports
                "indicators": list(COMEXT_INDICATORS),
                "freq": "M",
                "sinceTimePeriod": f"{start:%Y-%m}",
            },
        )
        r.raise_for_status()
        self.keep_raw(r.content, "json")
        body = r.json()
        df = parse_jsonstat(body)
        df = df[df.time.map(_month_start) <= pd.Timestamp(end)]
        meta = json.dumps({"updated": body.get("updated")})
        return pd.DataFrame(
            {
                "series_id": "COMEXT.EU27.IMP."
                + df["product"]
                + "."
                + df["partner"]
                + "."
                + df["indicators"],
                "observed_at": df.time.map(_month_start),
                "value": df.value,
                "unit": df.indicators.map(COMEXT_INDICATORS),
                "meta": meta,
            }
        )


class _EurostatStats(Source):
    dataset: str
    fixed: ClassVar[dict[str, str]]
    prefix: str
    unit: str
    verified = True

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        r = self.client.get(
            STATS_URL + self.dataset,
            params={
                **self.fixed,
                **self.geo_params(),
                "sinceTimePeriod": self.since(start),
            },
        )
        r.raise_for_status()
        self.keep_raw(r.content, "json")
        body = r.json()
        df = parse_jsonstat(body)
        df["observed_at"] = df.time.map(_month_start)
        df = df[df.observed_at <= pd.Timestamp(end)]
        return pd.DataFrame(
            {
                "series_id": f"{self.prefix}." + df.geo,
                "observed_at": df.observed_at,
                "value": df.value,
                "unit": self.unit,
                "meta": json.dumps({"updated": body.get("updated")}),
            }
        )

    def since(self, start: dt.date) -> str:
        return f"{start:%Y-%m}"

    def geo_params(self) -> dict:
        return {"geo": [*EU27_GEO, "EU27_2020"]}


class _Nuts2(_EurostatStats):
    """All NUTS-2 regions. The first two letters of a region code give its country."""

    def geo_params(self) -> dict:
        return {"geoLevel": "nuts2"}


class EurostatPopulation(_EurostatStats):
    name = "eurostat_population"
    dataset = "demo_pjan"
    fixed: ClassVar[dict[str, str]] = {"sex": "T", "age": "TOTAL"}
    prefix = "ESTAT.POP"
    unit = "persons"

    def since(self, start: dt.date) -> str:
        return f"{start:%Y}"


class EurostatFertiliserIndex(_EurostatStats):
    # Monthly production index for NACE C20.15 (fertilisers and nitrogen compounds), calendar
    # adjusted. Several countries keep it confidential; Greece has one ammonia plant.
    name = "eurostat_c2015"
    dataset = "sts_inpr_m"
    fixed: ClassVar[dict[str, str]] = {"nace_r2": "C2015", "s_adj": "CA", "unit": "I21"}
    prefix = "ESTAT.IPI.C2015"
    unit = "index 2021=100"


class EurostatHDD(_EurostatStats):
    name = "eurostat_hdd"
    dataset = "nrg_chdd_m"
    fixed: ClassVar[dict[str, str]] = {"indic_nrg": "HDD", "unit": "NR"}
    prefix = "ESTAT.HDD"
    unit = "degree-days"


class EurostatHDDNuts2(_Nuts2):
    # Eurostat's country HDD is the area-weighted mean of these regions. Weighted by
    # population instead, they give the reference for a population-weighted HDD.
    name = "eurostat_hdd_nuts2"
    dataset = "nrg_chddr2_m"
    fixed: ClassVar[dict[str, str]] = {"indic_nrg": "HDD", "unit": "NR"}
    prefix = "ESTAT.HDD2"
    unit = "degree-days"


class EurostatPopulationNuts2(_Nuts2):
    name = "eurostat_population_nuts2"
    dataset = "demo_r_d2jan"
    fixed: ClassVar[dict[str, str]] = {"sex": "T", "age": "TOTAL", "unit": "NR"}
    prefix = "ESTAT.POP2"
    unit = "persons"

    def since(self, start: dt.date) -> str:
        return f"{start:%Y}"


class EurostatGasTrade(Source):
    name = "eurostat_gas_trade"
    verified = True
    datasets: ClassVar[dict[str, str]] = {"IMP": "nrg_ti_gasm", "EXP": "nrg_te_gasm"}

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames = []
        for flow, dataset in self.datasets.items():
            r = self.client.get(
                STATS_URL + dataset,
                params={
                    "geo": "EU27_2020",  # fixed composition, so partner = across the boundary
                    "unit": "TJ_GCV",
                    "siec": ["G3000", "G3200"],
                    "sinceTimePeriod": f"{start:%Y-%m}",
                },
            )
            r.raise_for_status()
            self.keep_raw(r.content, "json")
            body = r.json()
            df = parse_jsonstat(body)
            df["observed_at"] = df.time.map(_month_start)
            df = df[df.observed_at <= pd.Timestamp(end)]
            frames.append(
                pd.DataFrame(
                    {
                        "series_id": f"ESTAT.GAS.{flow}." + df.siec + "." + df.partner,
                        "observed_at": df.observed_at,
                        "value": df.value,
                        "unit": "TJ_GCV",
                        "meta": json.dumps({"updated": body.get("updated")}),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)
