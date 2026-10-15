"""Candidate daily TTF: Yahoo Finance ``TTF=F`` front-month close (EUR/MWh), key-free.

Grade C until it passes the monthly reconciliation test against the World Bank Pink
Sheet (validation.reconciliation_tolerance). History on Yahoo starts on 2017-10-23.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from nsqueeze.io.sources.base import Source

URL = "https://query1.finance.yahoo.com/v8/finance/chart/TTF=F"
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]


def chart_to_canonical(body: dict) -> pd.DataFrame:
    """Yahoo chart JSON → one row per trading day, dated in the exchange time zone."""
    res = body["chart"]["result"][0]
    meta = res["meta"]
    ts = res.get("timestamp") or []
    close = res["indicators"]["quote"][0]["close"]
    day = pd.to_datetime(pd.Series(ts, dtype="int64") + meta["gmtoffset"], unit="s").dt.normalize()
    df = pd.DataFrame({"observed_at": day, "value": pd.Series(close, dtype="float64")}).dropna()
    df = df.drop_duplicates("observed_at", keep="last")
    return pd.DataFrame(
        {
            "series_id": "YF.TTF=F",
            "observed_at": df.observed_at,
            "value": df.value,
            "unit": f"{meta['currency']}/MWh",
            "meta": json.dumps({"exchange": meta.get("exchangeName")}),
        },
        columns=COLUMNS,
    )


class YahooTTF(Source):
    name = "ttf_daily_candidate"
    verified = True  # fetched and parsed on 2026-10-15

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        t0 = int(dt.datetime.combine(start, dt.time(), dt.UTC).timestamp())
        t1 = int(dt.datetime.combine(end + dt.timedelta(days=1), dt.time(), dt.UTC).timestamp())
        r = self.client.get(URL, params={"period1": t0, "period2": t1, "interval": "1d"})
        r.raise_for_status()
        self.keep_raw(r.content, "json")
        return chart_to_canonical(r.json())
