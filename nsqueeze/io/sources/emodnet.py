"""Vessel density at the berth of each coastal ammonia plant, from the EMODnet Human Activities
vessel density maps.

Cogea compiles monthly vessel density for EU waters from AIS and publishes it as GeoTIFF maps,
one map per month and ship type, for 2017-2024. Density is in hours per square kilometre per
month on a 1 km grid. Ship types follow the AIS type codes, so 10 is tankers, and one map covers
all ship types. The maps are free under CC BY 4.0 (data originator Cogea Srl) and are served by
the EMODnet Human Activities OGC endpoint.

I read one cell value per query point with WMS GetFeatureInfo, in EPSG:4326. The WMS is the only
route that takes lon/lat boxes without a projection library: the WCS subsets come in EPSG:3857
and the ZIP downloads hold whole years. The query points are the centres of a fixed 0.01 degree
lattice, about 0.7 x 1.1 km at the plant latitudes and close to the published 1 km cell.

One row per berth zone, ship type and month. ``value`` is the mean density over the lattice cells
whose centre lies inside the berth zone polygon. ``meta`` has the number of cells used. Cells with
no data are dropped, and a month without any data keeps a missing value, which stays apart from a
true zero.
"""

from __future__ import annotations

import datetime as dt
import json
import time

import httpx
import numpy as np
import pandas as pd

from nsqueeze.config import config_dir
from nsqueeze.io.sources.base import Source

URL = "https://ows.emodnet-humanactivities.eu/wms"
SHIP_TYPES = {"tanker": "vesseldensity_10", "all": "vesseldensity_all"}
UNIT = "h/km2/month"
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]
GRID_DEG = 0.01  # query lattice step in degrees
HALF_DEG = GRID_DEG / 2


def zones() -> list[dict]:
    return json.loads((config_dir() / "berth_zones.geojson").read_text())["features"]


class EmodnetVesselDensity(Source):
    name = "emodnet_density"

    def _cell_value(self, layer: str, lon: float, lat: float, month: pd.Timestamp) -> float | None:
        """One GetFeatureInfo at a lattice cell centre -> the map value, or None where no data."""
        params = {
            "service": "WMS",
            "request": "GetFeatureInfo",
            "version": "1.3.0",
            "layers": layer,
            "query_layers": layer,
            "styles": "",
            "crs": "EPSG:4326",
            "bbox": (
                f"{lat - HALF_DEG:.6f},{lon - HALF_DEG:.6f},"
                f"{lat + HALF_DEG:.6f},{lon + HALF_DEG:.6f}"
            ),
            "width": "11",
            "height": "11",
            "i": "5",
            "j": "5",
            "info_format": "application/json",
            "time": f"{month:%Y-%m-%d}T00:00:00.000Z",
        }
        # A back-fill sends thousands of small requests, and the server sometimes answers one with
        # an error status or a body that is not JSON. Retry that one request, not the whole chunk.
        for wait in (10, 60, 300, None):
            try:
                r = self.client.get(URL, params=params)
                r.raise_for_status()
                return parse_value(r.text)
            except (httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError):
                if wait is None:
                    raise
                time.sleep(wait)
        raise AssertionError("unreachable")

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        # Rows are monthly: a range that starts mid-month would split a month into two vintages.
        months = pd.date_range(start.replace(day=1), end, freq="MS")
        rows = []
        for feature in zones():
            plant_id = feature["properties"]["plant_id"]
            ring = np.asarray(feature["geometry"]["coordinates"][0], dtype=float)
            points = cell_centres(ring)
            points = points[point_in_polygon(points, ring)]
            for ship, layer in SHIP_TYPES.items():
                for month in months:
                    found = [self._cell_value(layer, lon, lat, month) for lon, lat in points]
                    used = [v for v in found if v is not None]
                    rows.append(
                        (
                            f"EMODNET.{plant_id}.{ship}",
                            month,
                            float(np.mean(used)) if used else float("nan"),
                            UNIT,
                            json.dumps({"cells": len(used)}),
                        )
                    )
        return pd.DataFrame(rows, columns=COLUMNS)


def cell_centres(ring: np.ndarray, step: float = GRID_DEG) -> np.ndarray:
    """Centres (lon, lat) of the fixed lattice cells that cover the ring's bounding box."""
    lon = np.arange(np.floor(ring[:, 0].min() / step), np.ceil(ring[:, 0].max() / step))
    lat = np.arange(np.floor(ring[:, 1].min() / step), np.ceil(ring[:, 1].max() / step))
    grid_lon, grid_lat = np.meshgrid(lon * step + step / 2, lat * step + step / 2)
    return np.column_stack([grid_lon.ravel(), grid_lat.ravel()])


def point_in_polygon(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """True where each (lon, lat) point lies inside the ring (ray casting along longitude)."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x1, y1 = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for i in range(len(ring)):
        spans = (y1[i] > y) != (y2[i] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = (x2[i] - x1[i]) * (y - y1[i]) / (y2[i] - y1[i]) + x1[i]
        inside ^= spans & (x < cross)
    return inside


def parse_value(body: str) -> float | None:
    """One GetFeatureInfo body -> the cell value, or None where there is no data.

    The maps mark missing cells with -9999 or with the lowest float32, and density is never
    negative, so any negative value is no data.
    """
    features = json.loads(body).get("features", [])
    value = features[0]["properties"].get("GRAY_INDEX") if features else None
    return None if value is None or value < 0 else float(value)
