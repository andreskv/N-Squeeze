"""ENTSO-E Transparency Platform: actual generation from fossil gas, per EU-27 country.

The gas balance estimates gas burn for power (G_t) from ENTSO-E generation by fuel. The fetcher
keeps only fossil gas (psrType B04) at native resolution (15, 30 or 60 min) in MW.
Daily sums and the efficiency conversion are model decisions, not archive decisions.

Query from the ENTSO-E Postman collection (16.1.B&C, read on 2026-10-13):
``documentType=A75``, ``processType=A16``, ``in_Domain=<EIC>``, ``periodStart`` and
``periodEnd`` as yyyyMMddHHmm UTC, at most one year per request. Areas are the country
EIC codes in ``config/entsoe_areas.csv``. A time series with ``outBiddingZone_Domain``
is consumption, not generation, so the parser skips it.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import xml.etree.ElementTree as ET

import pandas as pd

from nsqueeze.config import config_dir
from nsqueeze.io.sources.base import Source

URL = "https://web-api.tp.entsoe.eu/api"
GAS = "B04"  # psrType: Fossil Gas
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(el: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in el.iter() if _local(c.tag) == name), None)


def to_canonical(xml: bytes, geo: str) -> pd.DataFrame:
    """One row per interval start (UTC, naive). Empty frame for an acknowledgement (no data).

    With curveType A03 the API leaves out a position when its value equals the one
    before it, so a missing position repeats the last value until the period ends.
    """
    root = ET.fromstring(xml)
    if "Acknowledgement" in _local(root.tag):
        reason = _find(root, "text")
        print(f"entsoe {geo}: {reason.text if reason is not None else 'no data'}", file=sys.stderr)
        return pd.DataFrame(columns=COLUMNS)
    out = []
    for ts in (e for e in root if _local(e.tag) == "TimeSeries"):
        if _find(ts, "inBiddingZone_Domain.mRID") is None:
            continue
        psr = _find(ts, "psrType")
        if psr is None or psr.text != GAS:
            continue
        unit = _find(ts, "quantity_Measure_Unit.name").text
        for period in (e for e in ts if _local(e.tag) == "Period"):
            start = pd.Timestamp(_find(period, "start").text).tz_convert(None)
            end = pd.Timestamp(_find(period, "end").text).tz_convert(None)
            res = _find(period, "resolution").text
            step = pd.Timedelta(res)
            points = {
                int(_find(p, "position").text): float(_find(p, "quantity").text)
                for p in period
                if _local(p.tag) == "Point"
            }
            meta = json.dumps({"resolution": res})
            last = None
            for i in range(1, int((end - start) / step) + 1):
                last = points.get(i, last)
                if last is None:
                    continue
                out.append(
                    {
                        "series_id": f"ENTSOE.{geo}.{GAS}",
                        "observed_at": start + (i - 1) * step,
                        "value": last,
                        "unit": "MW" if unit == "MAW" else unit,
                        "meta": meta,
                    }
                )
    return pd.DataFrame(out, columns=COLUMNS)


def areas() -> pd.DataFrame:
    return pd.read_csv(config_dir() / "entsoe_areas.csv")


class ENTSOEGasGeneration(Source):
    name = "entsoe"
    env_key = "ENTSOE_TOKEN"
    verified = True  # live call parsed on 2026-10-13: 26 of 27 countries (MT: no data)

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames = []
        for geo, eic in areas()[["geo", "eic"]].itertuples(index=False):
            d = start
            while d <= end:  # one year per request at most
                stop = min(end, (pd.Timestamp(d) + pd.DateOffset(years=1, days=-1)).date())
                r = self.client.get(
                    URL,
                    params={
                        "securityToken": self.api_key,
                        "documentType": "A75",
                        "processType": "A16",
                        "in_Domain": eic,
                        "psrType": GAS,
                        "periodStart": f"{d:%Y%m%d}0000",
                        "periodEnd": f"{stop + dt.timedelta(days=1):%Y%m%d}0000",
                    },
                )
                # A query with no data comes back as an acknowledgement document,
                # sometimes with HTTP 400. Parse it rather than fail the whole run.
                if r.status_code != 400 or b"Acknowledgement" not in r.content:
                    r.raise_for_status()
                self.keep_raw(r.content, "xml")
                frames.append(to_canonical(r.content, geo))
                d = stop + dt.timedelta(days=1)
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
