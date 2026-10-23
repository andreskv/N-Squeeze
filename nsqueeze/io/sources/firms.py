"""NASA FIRMS VIIRS 375 m active-fire detections near each ammonia plant (S-NPP only).

Area API, one request per 5 days over a box around the plant panel. The standard product
(``VIIRS_SNPP_SP``) is used up to its last date, the near-real-time one (``VIIRS_SNPP_NRT``)
after it. One satellite only, so the detection rate does not change when NOAA-20 and NOAA-21
start. Needs ``FIRMS_MAP_KEY`` (free).

One row per plant and overpass (the store keeps one value per series and time): value = the
distance in km of the nearest detection within 5 km of the plant point. ``meta`` has the day/night
flag, the number of detections within 5 km and within 1 km, and the largest FRP. The 1 km night
count is applied later, in the open.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import time

import httpx
import numpy as np
import pandas as pd

from nsqueeze.io.sources.base import Source
from nsqueeze.io.sources.sentinel2 import COLUMNS, plants_with_coordinates

API = "https://firms.modaps.eosdis.nasa.gov/api"
BOX = "-8,36,29,61"  # west, south, east, north: all plants in config/plants.csv
KEEP_KM = 5.0
EARTH_KM = 6371.0


def near_plants(det: pd.DataFrame, plants: pd.DataFrame, source: str) -> pd.DataFrame:
    """Detections within KEEP_KM of a plant → one row per plant and overpass, ``FIRMS.<plant>``."""
    if det.empty:
        return pd.DataFrame(columns=COLUMNS)
    la0, lo0 = (
        np.radians(plants.lat.to_numpy())[:, None],
        np.radians(plants.lon.to_numpy())[:, None],
    )
    la, lo = (
        np.radians(det.latitude.to_numpy())[None, :],
        np.radians(det.longitude.to_numpy())[None, :],
    )
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    km = 2 * EARTH_KM * np.arcsin(np.sqrt(a))
    i, j = np.nonzero(km <= KEEP_KM)
    if len(i) == 0:
        return pd.DataFrame(columns=COLUMNS)
    d = det.iloc[j].reset_index(drop=True)
    hits = pd.DataFrame(
        {
            "series_id": "FIRMS." + plants.plant_id.iloc[i].reset_index(drop=True),
            "observed_at": pd.to_datetime(
                d.acq_date + " " + d.acq_time.astype(int).map("{:04d}".format),
                format="%Y-%m-%d %H%M",
            ),
            "km": km[i, j],
            "daynight": d.daynight,
            "frp": d.frp,
        }
    )
    rows = []
    for (sid, t), g in hits.groupby(["series_id", "observed_at"]):
        meta = {
            "daynight": g.daynight.iloc[0],
            "n_5km": len(g),
            "n_1km": int((g.km <= 1.0).sum()),
            "max_frp": float(g.frp.max()),
            "source": source,
        }
        rows.append((sid, t, float(g.km.min()), "km", json.dumps(meta)))
    return pd.DataFrame(rows, columns=COLUMNS)


class FirmsVIIRS(Source):
    name = "firms_viirs"
    env_key = "FIRMS_MAP_KEY"

    def _sp_end(self) -> dt.date:
        r = self.client.get(f"{API}/data_availability/csv/{self.api_key}/VIIRS_SNPP_SP")
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text)).max_date.map(pd.Timestamp).max().date()

    def _get(self, url: str):
        """One request with its own retries: an area request can take minutes when FIRMS is busy,
        and a chunk-level retry would refetch the whole year."""
        for wait in (30, 120, 300, None):
            try:
                r = self.client.get(url, timeout=300)
                r.raise_for_status()
                return r
            except (httpx.TimeoutException, httpx.HTTPStatusError):
                if wait is None:
                    raise
                time.sleep(wait)
        raise AssertionError("unreachable")

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        plants = plants_with_coordinates()
        sp_end = self._sp_end()
        frames, a = [], start
        while a <= end:
            source = "VIIRS_SNPP_SP" if a <= sp_end else "VIIRS_SNPP_NRT"
            days = min(5, (end - a).days + 1, (sp_end - a).days + 1 if a <= sp_end else 5)
            r = self._get(f"{API}/area/csv/{self.api_key}/{source}/{BOX}/{days}/{a}")
            if not r.text.startswith("latitude"):
                raise ValueError(f"FIRMS: unexpected reply for {source} {a}: {r.text[:120]}")
            self.keep_raw(r.content, "csv")
            frames.append(near_plants(pd.read_csv(io.StringIO(r.text)), plants, source))
            a += dt.timedelta(days=days)
            time.sleep(0.5)  # well under the 5,000 transactions per 10 minutes
        return pd.concat(frames, ignore_index=True)
