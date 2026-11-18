"""Stock-exchange filings on Oslo Børs (Newsweb), one row per filing (no key).

Listed producers must publish inside information at once (Market Abuse Regulation), and they
publish press releases on the same channel. Yara's curtailments of 2021-2022, for example, are
filings with a time stamp to the minute. The public Newsweb API lists the filings of an issuer
by date range and returns each filing's text.

value = 1 per filing. ``meta`` holds the title, the category and the link. For a filing whose
title names production (curtailment, ammonia, shutdown, restart and similar terms), ``meta``
also holds the body text. Every filing is opened, so that ``meta`` lists its attachments, and
each report PDF (quarterly, annual or integrated report) goes to the raw archive: the reports
hold plant decisions and production volumes. Release titles vary too much to select them by
title. The texts are kept for analysis. A published data set gives only dates, titles and links.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time

import pandas as pd

from nsqueeze.io.sources.base import Source

API = "https://api3.oslo.oslobors.no/v1/newsreader"
ISSUERS = ("YAR",)  # Yara International. Other Oslo issuers make no ammonia in the EU
PRODUCTION = re.compile(
    r"curtail|ammonia|production|produksjon|shut ?down|restart|\bclos(e|es|ing|ure)\b"
    r"|\bplants?\b|gas price|outage",
    re.I,
)
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]


REPORT_FILE = re.compile(r"report", re.I)  # the quarterly or annual report, not the slides


def to_canonical(
    messages: list[dict], bodies: dict[int, str], attachments: dict[int, list] | None = None
) -> pd.DataFrame:
    attachments = attachments or {}
    rows = []
    for m in messages:
        mid = int(m["messageId"])
        meta = {
            "title": m["title"],
            "category": (m.get("category") or [{}])[0].get("category_en"),
            "issuer": m.get("issuerSign"),
            "url": f"https://newsweb.oslobors.no/message/{mid}",
        }
        if mid in bodies:
            meta["body"] = bodies[mid]
        if attachments.get(mid):
            meta["attachments"] = attachments[mid]
        rows.append(
            {
                "series_id": f"NEWSWEB.{m.get('issuerSign')}.{mid}",
                "observed_at": pd.Timestamp(m["publishedTime"]).tz_convert("UTC").tz_localize(None),
                "value": 1.0,
                "unit": "filing",
                "meta": json.dumps(meta, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def year_windows(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """One window per calendar year, so that no list call grows large."""
    out, a = [], start
    while a <= end:
        b = min(end, dt.date(a.year, 12, 31))
        out.append((a, b))
        a = b + dt.timedelta(days=1)
    return out


class NewswebFilings(Source):
    name = "newsweb_filings"
    verified = True  # list and message endpoints checked live on 2026-11-18 (Yara, 2021-2022)

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        messages, bodies, files = [], {}, {}
        for issuer in ISSUERS:
            for a, b in year_windows(start, end):
                r = self.client.get(
                    f"{API}/list",
                    params={"issuer": issuer, "fromDate": a.isoformat(), "toDate": b.isoformat()},
                )
                r.raise_for_status()
                self.keep_raw(r.content, "json")
                messages += r.json()["data"]["messages"]
                time.sleep(0.5)
        for m in messages:
            mid = int(m["messageId"])
            r = self.client.get(f"{API}/message", params={"messageId": mid})
            r.raise_for_status()
            self.keep_raw(r.content, "json")
            msg = r.json()["data"]["message"]
            if PRODUCTION.search(m["title"]):
                bodies[mid] = msg.get("body", "")
            files[mid] = [{"id": a["id"], "name": a["name"]} for a in msg.get("attachments") or []]
            time.sleep(0.5)
            for a in files[mid]:
                if REPORT_FILE.search(a["name"]) and a["name"].lower().endswith(".pdf"):
                    f = self.client.get(
                        f"{API}/attachment", params={"messageId": mid, "attachmentId": a["id"]}
                    )
                    f.raise_for_status()
                    self.keep_raw(f.content, "pdf")
                    time.sleep(0.5)
        return to_canonical(messages, bodies, files)
