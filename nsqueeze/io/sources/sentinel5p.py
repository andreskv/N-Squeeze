"""Sentinel-5P TROPOMI tropospheric NO2 around each ammonia plant, one row set per ISO week.

Google Earth Engine, collection ``COPERNICUS/S5P/OFFL/L3_NO2`` (one image per orbit). For each
week I take the per-pixel mean of the week's images, then its mean over a disc around the plant
point and over a ring around it (radii in ``plant_activity.no2_*``). A running ammonia plant burns
gas in its reformer, and the nitric acid units next to it stop when the ammonia stops: both emit
NOx. The ring gives the regional background on the same days.

I store the two means and the mean number of valid observations over each area. The ratio and
the anomaly are computed later, in the open.
"""

from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from nsqueeze.io.sources.base import Source
from nsqueeze.io.sources.sentinel2 import COLUMNS, plants_with_coordinates
from nsqueeze.params import ParameterSet

COLLECTION = "COPERNICUS/S5P/OFFL/L3_NO2"
BAND = "tropospheric_NO2_column_number_density"
SCALE_M = 1113.2  # grid of the Level-3 collection
PARALLEL_WEEKS = 6  # Earth Engine requests in flight at once


def week_to_canonical(features: list[dict], monday: pd.Timestamp) -> pd.DataFrame:
    """reduceRegions features (one per plant and area) → rows ``S5P.<plant>.<area>``."""
    rows = [
        (
            f"S5P.{p['plant_id']}.{p['area']}",
            monday,
            float(p["no2"]),
            "mol/m2",
            json.dumps({"n_obs": p.get("n_obs")}),
        )
        for p in (f["properties"] for f in features)
        if p.get("no2") is not None
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


class Sentinel5PNO2(Source):
    name = "s5p_no2"
    env_key = "GEE_PROJECT"

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        import ee  # optional dependency (geo extra)

        ee.Initialize(project=self.api_key)
        P = ParameterSet.load()
        disc = float(P.value("plant_activity.no2_disc_radius_m"))
        ring_in, ring_out = (float(x) for x in P.value("plant_activity.no2_ring_m"))
        areas = []
        for r in plants_with_coordinates().itertuples():
            pt = ee.Geometry.Point(float(r.lon), float(r.lat))
            ring = pt.buffer(ring_out).difference(pt.buffer(ring_in))
            areas += [
                ee.Feature(pt.buffer(disc), {"plant_id": r.plant_id, "area": "disc"}),
                ee.Feature(ring, {"plant_id": r.plant_id, "area": "ring"}),
            ]
        fc = ee.FeatureCollection(areas)
        empty = ee.Image.constant([0, 0]).rename(["no2", "n_obs"]).updateMask(0)

        def week(monday: pd.Timestamp) -> pd.DataFrame:
            col = (
                ee.ImageCollection(COLLECTION)
                .select(BAND)
                .filterDate(
                    monday.date().isoformat(), (monday + pd.Timedelta(days=7)).date().isoformat()
                )
            )
            img = col.mean().rename("no2").addBands(col.count().rename("n_obs"))
            img = ee.Image(ee.Algorithms.If(col.size().gt(0), img, empty))  # a week with no orbit
            out = img.reduceRegions(fc, ee.Reducer.mean(), SCALE_M).getInfo()
            return week_to_canonical(out["features"], monday)

        mondays = pd.date_range(start, end, freq="W-MON")
        with ThreadPoolExecutor(PARALLEL_WEEKS) as ex:
            frames = list(ex.map(week, mondays))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
