"""Daily gas use of French industrial customers on the transmission grid, by region (no key).

NaTran (formerly GRTgaz) and Teréga publish on ODRÉ, the open-data portal of the French
network operators, the daily consumption of the industrial customers connected directly to
their transmission grids, summed by region, in MWh (gross calorific value). The ammonia plants
of Le Havre and Grand-Quevilly (Normandie), Ottmarsheim (Grand-Est) and Grandpuits
(Ile-de-France) are such customers, so each regional series holds one or two plants among
other industry. The data start on 1 January 2018. Licence Ouverte v2.0 (Etalab).

series_id = ODRE.IND.<NATRAN|TEREGA>.<region code>. ``meta`` holds the region and the status.
The status label changed over time (Définitives, Meilleur statut, Définitif). Where a day has
more than one row, the final one wins.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from nsqueeze.io.sources.base import Source

EXPORT = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "conso-journa-industriel-grtgazterega/exports/json"
)
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]
OPERATORS = {"NaTran": "NATRAN", "Teréga": "TEREGA"}


def canonical(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["final"] = df.statut.str.startswith("Définiti")
    df = df.sort_values("final", kind="stable")
    df = df.drop_duplicates(["operateur", "code_region", "date"], keep="last")
    region = df.code_region.astype(int).astype(str)
    return pd.DataFrame(
        {
            "series_id": "ODRE.IND." + df.operateur.map(OPERATORS) + "." + region,
            "observed_at": pd.to_datetime(df.date.str[:10]),
            "value": df.consommation_journaliere_mwh_pcs.astype(float),
            "unit": "MWh",
            "meta": [
                json.dumps({"region": r, "status": s}, ensure_ascii=False)
                for r, s in zip(df.region, df.statut, strict=True)
            ],
        }
    ).sort_values(["series_id", "observed_at"], ignore_index=True)[COLUMNS]


class NatranIndustrial(Source):
    name = "natran_industrial"
    verified = True  # ODRÉ export checked live on 2026-11-21

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        r = self.client.get(
            EXPORT,
            params={
                "select": (
                    "date,operateur,region,code_region,statut,consommation_journaliere_mwh_pcs"
                ),
                "where": f"date >= '{start.isoformat()}' AND date <= '{end.isoformat()}'",
            },
        )
        r.raise_for_status()
        self.keep_raw(r.content, "json")
        return canonical(r.json())
