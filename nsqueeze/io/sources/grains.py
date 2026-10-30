"""CBOT corn, soybean and wheat futures on Yahoo Finance, daily close (key-free).

Two kinds of series. The continuous front month (``ZC=F``, ``ZS=F``, ``ZW=F``) has history
from 2000. Each individual contract of the study months (corn July and December, soybeans
July and November, wheat July and December) is archived while it trades. Yahoo removes a
contract when it expires, so the daily archive is the only record of those closes. Expired
contracts come from another source later.

One row per symbol and trading day: value = the close as published, in US cents per bushel
(``USX/bu``, the currency that Yahoo reports in ``meta``). Grade C.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time

import pandas as pd

from nsqueeze.io.sources.base import Source
from nsqueeze.io.sources.yahoo import CHART, COLUMNS

# Study delivery months per root. Month codes: N = July, X = November, Z = December.
MONTHS = {"ZC": ((7, "N"), (12, "Z")), "ZS": ((7, "N"), (11, "X")), "ZW": ((7, "N"), (12, "Z"))}
CONTINUOUS = ("ZC=F", "ZS=F", "ZW=F")


def contract_symbols(today: dt.date) -> list[str]:
    """Yahoo symbols of the study contracts whose delivery month is not before the month of
    ``today``, at most three years ahead. ``ZCZ27.CBT`` = corn, December 2027, CBOT."""
    last = (today.year + 3, today.month)
    return [
        f"{root}{letter}{y % 100:02d}.CBT"
        for root, months in MONTHS.items()
        for y in range(today.year, today.year + 4)
        for m, letter in months
        if (today.year, today.month) <= (y, m) <= last
    ]


def chart_close_to_canonical(body: dict, symbol: str) -> pd.DataFrame:
    """Yahoo chart JSON → one row per trading day, dated in the exchange time zone."""
    res = (body["chart"].get("result") or [None])[0]
    if res is None:  # Yahoo answers with a null result when the symbol has no data
        return pd.DataFrame(columns=COLUMNS)
    meta = res["meta"]
    ts = res.get("timestamp") or []
    close = res["indicators"]["quote"][0]["close"]
    day = pd.to_datetime(pd.Series(ts, dtype="int64") + meta["gmtoffset"], unit="s").dt.normalize()
    df = pd.DataFrame({"observed_at": day, "value": pd.Series(close, dtype="float64")}).dropna()
    df = df.drop_duplicates("observed_at", keep="last")
    return pd.DataFrame(
        {
            "series_id": f"YF.{symbol}.close",
            "observed_at": df.observed_at,
            "value": df.value,
            "unit": f"{meta['currency']}/bu",
            "meta": json.dumps({"exchange": meta.get("exchangeName")}),
        },
        columns=COLUMNS,
    )


class GrainFutures(Source):
    name = "grain_futures"
    verified = True  # symbols and parsing checked live on 2026-10-27

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        t0 = int(dt.datetime.combine(start, dt.time(), dt.UTC).timestamp())
        t1 = int(dt.datetime.combine(end + dt.timedelta(days=1), dt.time(), dt.UTC).timestamp())
        # The archiver runs with end = today, so the symbol list covers the live contracts.
        symbols = contract_symbols(end) + list(CONTINUOUS)
        frames, missing = [], []
        for symbol in symbols:
            r = self.client.get(
                CHART + symbol, params={"period1": t0, "period2": t1, "interval": "1d"}
            )
            if r.status_code == 404:  # Yahoo removes a contract when it expires
                frame = pd.DataFrame(columns=COLUMNS)
            else:
                r.raise_for_status()
                self.keep_raw(r.content, "json")
                frame = chart_close_to_canonical(r.json(), symbol)
            if frame.empty:
                missing.append(symbol)
            else:
                frames.append(frame)
            time.sleep(0.5)  # one request every 0.5 s stays far below any published limit
        print(
            f"grain_futures: {len(missing)} of {len(symbols)} symbols gave no rows"
            f"{': ' + ', '.join(missing) if missing else ''}",
            file=sys.stderr,
        )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
