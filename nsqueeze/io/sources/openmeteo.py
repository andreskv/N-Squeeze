"""Daily 2 m temperature at one point per EU-27 country, Open-Meteo historical API (ERA5).

Key-free and checked live on 2026-10-12. Points are in ``config/weather_points.csv``
(capitals, grade C as a proxy for each country's population-weighted temperature).
The proxy is tested, not assumed: ``gas_balance.hdd_check`` compares the HDD built
from these points with Eurostat's monthly HDD (``eurostat_hdd``).

Tmax and Tmin are kept because the Eurostat definition uses
Tm = (Tmax + Tmin) / 2, not the 24-hour mean. ERA5 reaches back to 1940, which is what
the fan chart's weather-year resampling needs.
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from nsqueeze.config import config_dir
from nsqueeze.io.sources.base import Source

URL = "https://archive-api.open-meteo.com/v1/archive"
# Tmax and Tmin only: the Eurostat Tm needs nothing else, and each extra variable adds
# to the request weight that Open-Meteo's fair-use limits count.
VARIABLES = ("temperature_2m_max", "temperature_2m_min")
CHUNK_YEARS = 5  # long back-fills are split and stored chunk by chunk
BACKOFF_SECONDS = (65, 300, 900, 1800, 3600)  # 429 = minute, hour or day quota reached


def weather_points() -> pd.DataFrame:
    return pd.read_csv(config_dir() / "weather_points.csv")


def to_canonical(payload: list[dict] | dict, geos: list[str]) -> pd.DataFrame:
    payload = payload if isinstance(payload, list) else [payload]
    frames = []
    for geo, loc in zip(geos, payload, strict=True):
        d = loc["daily"]
        for var in VARIABLES:
            frames.append(
                pd.DataFrame(
                    {
                        "series_id": f"OM.{geo}.{var}",
                        "observed_at": pd.to_datetime(d["time"]),
                        "value": d[var],
                        "unit": "degC",
                    }
                )
            )
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["value"])


class OpenMeteoWeather(Source):
    name = "weather"
    verified = True

    def _get(self, params: dict):
        for wait in (*BACKOFF_SECONDS, None):
            r = self.client.get(URL, params=params)
            if r.status_code != 429 or wait is None:
                r.raise_for_status()
                return r
            time.sleep(wait)
        raise AssertionError("unreachable")

    def chunks(self, start: dt.date, end: dt.date):
        pts = weather_points()
        a = start
        while a <= end:
            b = min(end, dt.date(a.year + CHUNK_YEARS, 1, 1) - dt.timedelta(days=1))
            r = self._get(
                {
                    "latitude": ",".join(map(str, pts.lat)),
                    "longitude": ",".join(map(str, pts.lon)),
                    "start_date": a.isoformat(),
                    "end_date": b.isoformat(),
                    "daily": ",".join(VARIABLES),
                    "timezone": "UTC",
                }
            )
            self.keep_raw(r.content, "json")
            yield to_canonical(r.json(), list(pts.geo))
            a = b + dt.timedelta(days=1)

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        return pd.concat(self.chunks(start, end), ignore_index=True)

    def archive(self, store, start: dt.date, end: dt.date) -> int:
        """Write each chunk as its own vintage so a quota stop loses nothing already fetched."""
        n = 0
        for frame in self.chunks(start, end):
            store.write(self.name, frame)
            n += len(frame)
        return n
