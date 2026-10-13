"""ECB euro foreign-exchange reference rate, USD per EUR (key-free SDMX API)."""

from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from nsqueeze.io.sources.base import Source

URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"


class ECBFx(Source):
    name = "ecb_fx"
    verified = True  # endpoint exercised and parsed on 2026-10-12

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        r = self.client.get(
            URL,
            params={
                "startPeriod": start.isoformat(),
                "endPeriod": end.isoformat(),
                "format": "csvdata",
            },
        )
        r.raise_for_status()
        self.keep_raw(r.content, "csv")
        raw = pd.read_csv(io.BytesIO(r.content))
        return pd.DataFrame(
            {
                "series_id": "EXR.D.USD.EUR.SP00.A",
                "observed_at": pd.to_datetime(raw["TIME_PERIOD"]),
                "value": raw["OBS_VALUE"],
                "unit": "USD/EUR",
            }
        )
