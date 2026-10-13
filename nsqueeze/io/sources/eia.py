"""EIA API v2 — Henry Hub natural gas spot price, daily ($/MMBtu). Needs EIA_API_KEY."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from nsqueeze.io.sources.base import Source

URL = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"


class HenryHub(Source):
    name = "henry_hub"
    env_key = "EIA_API_KEY"
    verified = True  # fetched and parsed with a key on 2026-10-13

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        r = self.client.get(
            URL,
            params={
                "api_key": self.api_key,
                "frequency": "daily",
                "data[0]": "value",
                "facets[series][]": "RNGWHHD",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "length": 5000,
            },
        )
        r.raise_for_status()
        rows = r.json()["response"]["data"]
        if not rows:
            return pd.DataFrame(columns=["series_id", "observed_at", "value", "unit"])
        df = pd.DataFrame(rows)
        return pd.DataFrame(
            {
                "series_id": "EIA.RNGWHHD",
                "observed_at": pd.to_datetime(df["period"]),
                "value": pd.to_numeric(df["value"], errors="coerce"),
                "unit": "USD/MMBtu",
            }
        )
