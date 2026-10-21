"""Landsat 8 and 9 surface temperature around each ammonia plant, one row set per scene.

Google Earth Engine, collections ``LANDSAT/LC08/C02/T1_L2`` and ``LANDSAT/LC09/C02/T1_L2``
(Collection 2, Level 2), thermal band ``ST_B10``. Plant locations are the GEM coordinates in
``config/plants.csv``. Needs ``GEE_PROJECT`` and a one-time ``earthengine authenticate``.

For each scene I store the mean surface temperature of clear pixels in a 300 m disc around
the plant point and in a ring from 1 to 2 km, and the clear share of each area. A pixel is
clear if the QA_PIXEL bits for dilated cloud, cirrus, cloud and cloud shadow are all zero.
The archive applies no threshold: the core-minus-ring statistic and the 70 % clear rule are
applied later, in the open.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from nsqueeze.io.sources.base import Source
from nsqueeze.io.sources.sentinel2 import COLUMNS, plants_with_coordinates

COLLECTIONS = ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2")
SCALE_M = 30  # ST_B10 is delivered on a 30 m grid
ST_MULT, ST_ADD = 0.00341802, 149.0  # Collection 2 Level 2 scale factors to kelvin
CLOUD_BITS = 0b11110  # QA_PIXEL bits 1-4: dilated cloud, cirrus, cloud, cloud shadow
CORE_M, RING_IN_M, RING_OUT_M = 300, 1000, 2000
STATS = {
    "st_core_mean": "K",
    "st_ring_mean": "K",
    "core_clear_frac": "share",
    "ring_clear_frac": "share",
}


def to_canonical(features: list[dict], plant_id: str) -> pd.DataFrame:
    """Earth Engine feature properties (one feature per scene) → canonical rows."""
    rows = []
    for f in features:
        p = f["properties"]
        t = pd.to_datetime(p["time"], unit="ms")
        meta = json.dumps({"scene_id": p.get("scene_id")})
        for stat, unit in STATS.items():
            if p.get(stat) is not None:
                rows.append((f"LS.{plant_id}.{stat}", t, float(p[stat]), unit, meta))
    return pd.DataFrame(rows, columns=COLUMNS)


def _scene_stats(ee, core, ring):
    def stats(img):
        clear = img.select("QA_PIXEL").bitwiseAnd(CLOUD_BITS).eq(0).rename("clear")
        st = img.select("ST_B10").multiply(ST_MULT).add(ST_ADD).updateMask(clear).rename("st")
        mean = ee.Reducer.mean()
        return ee.Feature(
            None,
            {
                "time": img.get("system:time_start"),
                "scene_id": img.get("LANDSAT_PRODUCT_ID"),
                "st_core_mean": st.reduceRegion(mean, core, SCALE_M).get("st"),
                "st_ring_mean": st.reduceRegion(mean, ring, SCALE_M).get("st"),
                "core_clear_frac": clear.reduceRegion(mean, core, SCALE_M).get("clear"),
                "ring_clear_frac": clear.reduceRegion(mean, ring, SCALE_M).get("clear"),
            },
        )

    return stats


class LandsatThermal(Source):
    name = "landsat_st"
    env_key = "GEE_PROJECT"

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        import ee  # optional dependency (geo extra)

        ee.Initialize(project=self.api_key)
        frames = []
        for r in plants_with_coordinates().itertuples():
            pt = ee.Geometry.Point(float(r.lon), float(r.lat))
            core = pt.buffer(CORE_M)
            ring = pt.buffer(RING_OUT_M).difference(pt.buffer(RING_IN_M))
            col = (
                ee.ImageCollection(COLLECTIONS[0])
                .merge(ee.ImageCollection(COLLECTIONS[1]))
                .filterBounds(pt.buffer(RING_OUT_M))
                .filterDate(start.isoformat(), (end + dt.timedelta(days=1)).isoformat())
            )
            fc = ee.FeatureCollection(col.map(_scene_stats(ee, core, ring))).getInfo()
            frames.append(to_canonical(fc["features"], r.plant_id))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
