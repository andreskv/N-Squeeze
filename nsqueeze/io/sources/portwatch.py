"""IMF PortWatch daily chokepoint transits (key-free ArcGIS FeatureServer).

Checked live on 2026-10-12. Facts that matter for the regime indicator:

* History starts on **2019-01-01**, not 2015, so a regime model fitted on 2015–2025
  cannot use PortWatch for 2015–2018.
* Vessel classes are container, dry bulk, general cargo, ro-ro and tanker. There is
  **no LNG or gas-carrier split**: LNG carriers sit inside ``n_tanker``.
* Counts are AIS-based, so dark transits are missed by construction.

Rows are snapshotted weekly (``recoverable: false``): PortWatch revises its filters.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from nsqueeze.io.sources.base import Source

URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)
# Routes between the Gulf / Asia and NW Europe, plus the Black Sea exit.
CHOKEPOINTS = (
    "Strait of Hormuz",
    "Bab el-Mandeb Strait",
    "Suez Canal",
    "Cape of Good Hope",
    "Gibraltar Strait",
    "Bosporus Strait",
)
FIELDS = (
    "n_total",
    "n_tanker",
    "n_cargo",
    "n_container",
    "n_dry_bulk",
    "capacity",
    "capacity_tanker",
)
UNITS = {"capacity": "dwt", "capacity_tanker": "dwt"}
PAGE = 2000


def to_canonical(features: list[dict]) -> pd.DataFrame:
    out = []
    for f in features:
        a = f["attributes"]
        for k in FIELDS:
            if a.get(k) is None:
                continue
            out.append(
                {
                    "series_id": f"PW.{a['portid']}.{k}",
                    "observed_at": a["date"],
                    "value": a[k],
                    "unit": UNITS.get(k, "vessels"),
                    "meta": json.dumps({"portname": a["portname"]}),
                }
            )
    return pd.DataFrame(out, columns=["series_id", "observed_at", "value", "unit", "meta"])


class PortWatch(Source):
    name = "portwatch"
    verified = True

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        names = ", ".join(f"'{n}'" for n in CHOKEPOINTS)
        where = f"portname IN ({names}) AND date >= DATE '{start}' AND date <= DATE '{end}'"
        feats, offset = [], 0
        while True:
            r = self.client.get(
                URL,
                params={
                    "where": where,
                    "outFields": ",".join(("date", "portid", "portname", *FIELDS)),
                    "orderByFields": "date,portid",
                    "resultOffset": offset,
                    "resultRecordCount": PAGE,
                    "returnGeometry": "false",
                    "f": "json",
                },
            )
            r.raise_for_status()
            body = r.json()
            if "error" in body:
                raise RuntimeError(f"portwatch: {body['error']}")
            self.keep_raw(r.content, "json")
            page = body.get("features", [])
            feats.extend(page)
            if not body.get("exceededTransferLimit") and len(page) < PAGE:
                break
            offset += len(page)
        return to_canonical(feats)
