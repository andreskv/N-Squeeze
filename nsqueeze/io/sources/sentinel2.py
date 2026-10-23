"""Sentinel-2 SWIR statistics in a box around each ammonia plant, one row set per scene.

Google Earth Engine, collection ``COPERNICUS/S2_HARMONIZED`` (Level-1C, top-of-atmosphere
reflectance; hot pixels are not distorted by an atmospheric correction). Plant locations are
the GEM coordinates in ``config/plants.csv``; plants without coordinates are skipped. Needs
``GEE_PROJECT`` and a one-time ``earthengine authenticate`` on the machine that runs it.

I store raw statistics only. The archive applies no hot-spot threshold, so detection rules
are chosen later and in the open. Steam plumes can look like clouds, so the clear
fraction (Cloud Score+ ``cs_cdf`` at or above ``plant_activity.s2_clear_cs_cdf``) is a separate
statistic and masks nothing.

NHI_SWIR = (B12 - B11) / (B12 + B11), a normalized hot-spot index on the two SWIR bands.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from nsqueeze.config import config_dir
from nsqueeze.io.sources.base import Source
from nsqueeze.params import ParameterSet

COLLECTION = "COPERNICUS/S2_HARMONIZED"
CLOUD_SCORE = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
SCALE_M = 20  # native resolution of B11 and B12
DN_PER_REFLECTANCE = 1e4  # the collection stores TOA reflectance scaled by 10000
STATS = {  # statistic -> unit
    "b11_max": "reflectance",
    "b12_max": "reflectance",
    "b12_p99": "reflectance",
    "nhi_swir_max": "index",
    "nhi_swir_pos_px": "pixels",
    "clear_frac": "share",
    "n_px": "pixels",
}
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]


def plants_with_coordinates() -> pd.DataFrame:
    p = pd.read_csv(config_dir() / "plants.csv")
    return p.dropna(subset=["lat", "lon"])


def to_canonical(features: list[dict], plant_id: str) -> pd.DataFrame:
    """Earth Engine feature properties (one feature per scene) → canonical rows."""
    rows = []
    for f in features:
        p = f["properties"]
        t = pd.to_datetime(p["time"], unit="ms")
        meta = json.dumps({"product_id": p.get("product_id")})
        for stat, unit in STATS.items():
            if p.get(stat) is not None:
                rows.append((f"S2.{plant_id}.{stat}", t, float(p[stat]), unit, meta))
    return pd.DataFrame(rows, columns=COLUMNS)


def _scene_stats(ee, box, clear_cs_cdf: float):
    def stats(img):
        b11 = img.select("B11").divide(DN_PER_REFLECTANCE).rename("b11")
        b12 = img.select("B12").divide(DN_PER_REFLECTANCE).rename("b12")
        nhi = b12.subtract(b11).divide(b12.add(b11)).rename("nhi_swir")
        mx = ee.Image.cat([b11, b12, nhi]).reduceRegion(ee.Reducer.max(), box, SCALE_M)
        p99 = b12.reduceRegion(ee.Reducer.percentile([99]), box, SCALE_M)
        pos = nhi.gt(0).reduceRegion(ee.Reducer.sum(), box, SCALE_M)
        clear = img.select("cs_cdf").gte(clear_cs_cdf).reduceRegion(ee.Reducer.mean(), box, SCALE_M)
        n = b12.reduceRegion(ee.Reducer.count(), box, SCALE_M)
        return ee.Feature(
            None,
            {
                "time": img.get("system:time_start"),
                "product_id": img.get("PRODUCT_ID"),
                "b11_max": mx.get("b11"),
                "b12_max": mx.get("b12"),
                "b12_p99": p99.get("b12"),
                "nhi_swir_max": mx.get("nhi_swir"),
                "nhi_swir_pos_px": pos.get("nhi_swir"),
                "clear_frac": clear.get("cs_cdf"),
                "n_px": n.get("b12"),
            },
        )

    return stats


class Sentinel2SWIR(Source):
    name = "s2_swir"
    env_key = "GEE_PROJECT"
    verified = True  # prototype run on 2026-10-16 (Sluiskil, Aug 2022)

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        import ee  # optional dependency (geo extra)

        ee.Initialize(project=self.api_key)
        P = ParameterSet.load()
        radius = float(P.value("plant_activity.s2_box_radius_m"))
        clear = float(P.value("plant_activity.s2_clear_cs_cdf"))
        cs = ee.ImageCollection(CLOUD_SCORE)
        frames = []
        for r in plants_with_coordinates().itertuples():
            box = ee.Geometry.Point(float(r.lon), float(r.lat)).buffer(radius)
            col = (
                ee.ImageCollection(COLLECTION)
                .filterBounds(box)
                .filterDate(start.isoformat(), (end + dt.timedelta(days=1)).isoformat())
                .linkCollection(cs, ["cs_cdf"])
            )
            fc = ee.FeatureCollection(col.map(_scene_stats(ee, box, clear))).getInfo()
            frames.append(to_canonical(fc["features"], r.plant_id))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


PLUME_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
PLUME_CORE_M, PLUME_RING_IN_M, PLUME_RING_OUT_M = 500, 1500, 3000
PLUME_STATS = {
    "core_cloud_frac": "share",
    "ring_cloud_frac": "share",
    "core_px": "pixels",
    "ring_px": "pixels",
}


def plume_to_canonical(features: list[dict], plant_id: str) -> pd.DataFrame:
    """Scene features → canonical rows ``S2P.<plant>.<stat>``."""
    rows = []
    for f in features:
        p = f["properties"]
        t = pd.to_datetime(p["time"], unit="ms")
        meta = json.dumps({"product_id": p.get("product_id")})
        for stat, unit in PLUME_STATS.items():
            if p.get(stat) is not None:
                rows.append((f"S2P.{plant_id}.{stat}", t, float(p[stat]), unit, meta))
    return pd.DataFrame(rows, columns=COLUMNS)


class Sentinel2Plume(Source):
    """Share of pixels in scene classes 8 and 9 (cloud, medium and high probability) in a
    500 m disc around the plant and in a ring from 1.5 to 3 km, per Sentinel-2 Level-2A scene.

    A condensation plume over a running plant in cold weather is classed as cloud. The cold-day
    filter and the core-minus-ring statistic are applied later, in the open.
    """

    name = "s2_plume"
    env_key = "GEE_PROJECT"

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        import ee  # optional dependency (geo extra)

        ee.Initialize(project=self.api_key)
        frames = []
        for r in plants_with_coordinates().itertuples():
            pt = ee.Geometry.Point(float(r.lon), float(r.lat))
            core = pt.buffer(PLUME_CORE_M)
            ring = pt.buffer(PLUME_RING_OUT_M).difference(pt.buffer(PLUME_RING_IN_M))

            def stats(img, core=core, ring=ring):
                scl = img.select("SCL")
                cloud = scl.eq(8).Or(scl.eq(9)).rename("c")
                mean, count = ee.Reducer.mean(), ee.Reducer.count()
                return ee.Feature(
                    None,
                    {
                        "time": img.get("system:time_start"),
                        "product_id": img.get("PRODUCT_ID"),
                        "core_cloud_frac": cloud.reduceRegion(mean, core, SCALE_M).get("c"),
                        "ring_cloud_frac": cloud.reduceRegion(mean, ring, SCALE_M).get("c"),
                        "core_px": cloud.reduceRegion(count, core, SCALE_M).get("c"),
                        "ring_px": cloud.reduceRegion(count, ring, SCALE_M).get("c"),
                    },
                )

            col = (
                ee.ImageCollection(PLUME_COLLECTION)
                .filterBounds(pt)
                .filterDate(start.isoformat(), (end + dt.timedelta(days=1)).isoformat())
            )
            fc = ee.FeatureCollection(col.map(stats)).getInfo()
            frames.append(plume_to_canonical(fc["features"], r.plant_id))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
