"""Ships at the berth of each coastal ammonia plant: AIS presence from Global Fishing Watch.

GFW's 4Wings report on the dataset ``public-global-presence`` gives, for a polygon, one row per
vessel and month: hours present, mean speed, vessel class, IMO, MMSI, name and flag, for all
vessel types (not only fishing). The polygons are the berth zones in ``config/berth_zones.geojson``.
Needs ``GFW_API_TOKEN`` (free, non-commercial use). GFW allows one report at a time per user.

When a plant with downstream units stops making ammonia, it can buy ammonia by ship instead;
an exporting plant that stops ships less. Which ships are gas carriers is decided later.
"""

from __future__ import annotations

import datetime as dt
import json
import time

import pandas as pd

from nsqueeze.config import config_dir
from nsqueeze.io.sources.base import Source

URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-presence:latest"
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]
META = ("imo", "mmsi", "shipName", "geartype", "vesselType", "flag", "speed")
RETRY_WAIT_S = (30, 60, 120, 240)  # 429 (a report is already running) or 524 (gateway timeout)


def zones() -> list[dict]:
    return json.loads((config_dir() / "berth_zones.geojson").read_text())["features"]


def to_canonical(rows: list[dict], plant_id: str) -> pd.DataFrame:
    """Report rows (one per vessel and month) → canonical rows ``GFWP.<plant>.<vesselId>``."""
    out = [
        (
            f"GFWP.{plant_id}.{r['vesselId']}",
            pd.Timestamp(r["date"] + "-01"),
            float(r["hours"]),
            "h",
            json.dumps({k: r.get(k) for k in META}, ensure_ascii=False),
        )
        for r in rows
        if r.get("vesselId") and r.get("hours") is not None
    ]
    return pd.DataFrame(out, columns=COLUMNS)


class GFWPresence(Source):
    name = "gfw_presence"
    env_key = "GFW_API_TOKEN"

    def _report(self, feature: dict, start: dt.date, end: dt.date) -> list[dict]:
        params = {
            "spatial-resolution": "HIGH",
            "format": "JSON",
            "group-by": "VESSEL_ID",
            "temporal-resolution": "MONTHLY",
            "datasets[0]": DATASET,
            "date-range": f"{start.isoformat()},{(end + dt.timedelta(days=1)).isoformat()}",
            "spatial-aggregation": "true",
        }
        body = {"geojson": {"type": "FeatureCollection", "features": [feature]}}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for wait in (*RETRY_WAIT_S, None):
            r = self.client.post(URL, params=params, json=body, headers=headers, timeout=240)
            if r.status_code in (429, 524) and wait is not None:
                time.sleep(wait)
                continue
            r.raise_for_status()
            self.keep_raw(r.content, "json")
            return next(iter(r.json()["entries"][0].values()))
        raise AssertionError("unreachable")

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames = [
            to_canonical(self._report(f, start, end), f["properties"]["plant_id"]) for f in zones()
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
