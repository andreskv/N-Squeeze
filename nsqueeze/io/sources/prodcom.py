"""Eurostat PRODCOM annual production (Comext dissemination, key-free), checked 2026-10-27.

``eurostat_prodcom``  total production of anhydrous ammonia (20151075) and ammonia in
                      aqueous solution (20151077) by country and year, as published in
                      kg of nitrogen. Total production is sold production plus the
                      production that the producing enterprise keeps for further
                      processing, so it is the sum of plant output over a country and a
                      year. The old PRODCOM list (DS-059359) holds the years to 2024 and
                      the list by CPA 2.2 (DS-059368) the years from 2025. A confidential
                      cell carries only its flag and stays out of the rows.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from nsqueeze.io.sources.base import Source
from nsqueeze.io.sources.eurostat import EU27_GEO, parse_jsonstat

COMEXT_API = "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/"
PRODCOM_DATASETS = ("DS-059359", "DS-059368")  # old list to 2024, list by CPA 2.2 from 2025
PRODCOM_PRODUCTS = ("20151075", "20151077")  # anhydrous ammonia, ammonia in aqueous solution
# The unit, the flag and the rounding base describe a quantity cell. They are not rows.
QUANTITY_FIELDS = ("QNTUNIT", "APQNTFLAG", "APQNTBASE")
CONFIDENTIAL = ":C"


def to_canonical(body: dict) -> pd.DataFrame:
    """One PRODCOM response → one row per product, indicator, country and year.

    I keep the published unit of each quantity and I put the flag of the value (blank,
    confidential, estimate, rounded) and the update date of the dataset in meta. A
    confidential cell is suppressed or carries only its flag, and gets no row.
    """
    df = parse_jsonstat(body)
    key = ["product", "reporter", "time"]
    aux = (
        df[df["indicators"].isin(("QNTUNIT", "APQNTFLAG"))]
        .pivot_table(index=key, columns="indicators", values="value", aggfunc="first")
        .rename(columns={"QNTUNIT": "unit", "APQNTFLAG": "flag"})
        .reindex(columns=["unit", "flag"])
    )
    rows = df[~df["indicators"].isin(QUANTITY_FIELDS)].merge(aux.reset_index(), on=key, how="left")
    rows = rows[rows["flag"] != CONFIDENTIAL]
    return pd.DataFrame(
        {
            "series_id": "ESTAT.PRODCOM."
            + rows["product"]
            + "."
            + rows["indicators"]
            + "."
            + rows["reporter"],
            "observed_at": pd.to_datetime(rows["time"] + "-01-01"),
            "value": rows["value"],
            "unit": rows["unit"],
            "meta": rows["flag"].map(
                lambda f: json.dumps(
                    {"flag": None if pd.isna(f) else f, "updated": body.get("updated")}
                )
            ),
        }
    )


class EurostatProdcom(Source):
    name = "eurostat_prodcom"
    verified = True  # fetched and parsed on 2026-10-27

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames = []
        for dataset in PRODCOM_DATASETS:  # one list ends at 2024, the other starts at 2025
            r = self.client.get(
                COMEXT_API + dataset,
                params={
                    "product": list(PRODCOM_PRODUCTS),
                    "reporter": [*EU27_GEO, "EU27_2020"],
                    "sinceTimePeriod": f"{start:%Y}",
                },
            )
            r.raise_for_status()
            self.keep_raw(r.content, "json")
            frames.append(to_canonical(r.json()))
        df = pd.concat(frames, ignore_index=True)
        return df[df["observed_at"] <= pd.Timestamp(end)].reset_index(drop=True)
