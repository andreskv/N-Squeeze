"""Press releases of listed producers from their own web sites, one row per release (no key).

Companies that are not listed in Oslo publish plant decisions (curtailments, closures,
restarts) as press releases on their own sites. Two sites have a machine-readable archive:

- OCI: the WordPress API of oci-global.com lists every post with its time and text.
- BASF: the site map of basf.com lists every news release by month, and the content API
  of the site returns the title and the release time of each one.

observed_at is the publication day in the company's local time, at 00:00. Many older OCI
posts and all BASF pages give the day with no time of day, and a midnight converted to UTC
falls on the day before. ``meta`` keeps the time stamp as the site gives it.

value = 1 per release. ``meta`` holds the title and the link. For a release whose title
names production (see ``newsweb.PRODUCTION``), ``meta`` also holds the text (OCI) or the raw
page goes to the archive (BASF). The texts are kept for analysis. A published data set
gives only dates, titles and links.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import time

import pandas as pd

from nsqueeze.io.sources.base import Source
from nsqueeze.io.sources.newsweb import PRODUCTION

COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]
OCI_API = "https://oci-global.com/wp-json/wp/v2/posts"
BASF_SITEMAP = "https://www.basf.com/dam/sitemaps/basf/www/global/en.xml"
BASF_API = "https://www.basf.com/api/content/delivery/content/basf/www/"
BASF_PATH = re.compile(r"global/en/media/news-releases/(\d{4})/(\d{2})/p-\d{2}-[0-9a-z-]+")


def _row(company: str, rid: str, when: str, meta: dict) -> dict:
    meta = {**meta, "published": when}
    return {
        "series_id": f"NEWS.{company}.{rid}",
        "observed_at": pd.Timestamp(when[:10]),
        "value": 1.0,
        "unit": "release",
        "meta": json.dumps(meta, ensure_ascii=False),
    }


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def oci_canonical(posts: list[dict]) -> pd.DataFrame:
    rows = []
    for p in posts:
        title = strip_html(p["title"]["rendered"])
        meta = {"title": title, "url": p["link"]}
        if PRODUCTION.search(title):
            meta["body"] = strip_html(p["content"]["rendered"])
        # "date" is the local time. "date_gmt" was reset when the site moved (Nov 2022).
        rows.append(_row("OCI", str(p["id"]), p["date"], meta))
    return pd.DataFrame(rows, columns=COLUMNS)


def basf_paths(sitemap: str, start: dt.date, end: dt.date) -> list[str]:
    """Release paths whose month (in the path) overlaps the window, without repeats."""
    out = []
    for m in BASF_PATH.finditer(sitemap):
        month = dt.date(int(m[1]), int(m[2]), 1)
        if dt.date(start.year, start.month, 1) <= month <= end and m[0] not in out:
            out.append(m[0])
    return out


class OCINews(Source):
    name = "oci_news"
    verified = True  # WordPress API checked live on 2026-11-18: 422 posts since 2013

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        posts, page = [], 1
        while True:
            r = self.client.get(
                OCI_API,
                params={
                    "after": f"{start.isoformat()}T00:00:00",
                    "before": f"{(end + dt.timedelta(days=1)).isoformat()}T00:00:00",
                    "per_page": 100,
                    "page": page,
                    "orderby": "date",
                    "order": "asc",
                    "_fields": "id,date,link,title,content",
                },
            )
            r.raise_for_status()
            self.keep_raw(r.content, "json")
            posts += r.json()
            if page >= int(r.headers.get("X-WP-TotalPages", 1)):
                break
            page += 1
            time.sleep(0.5)
        return oci_canonical(posts)


class BASFNews(Source):
    name = "basf_news"
    verified = True  # site map and content API checked live on 2026-11-18

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        r = self.client.get(BASF_SITEMAP)
        r.raise_for_status()
        rows = []
        for path in basf_paths(r.text, start, end):
            p = self.client.get(BASF_API + path)
            time.sleep(0.5)
            if p.status_code == 404:  # listed in the site map but removed
                continue
            p.raise_for_status()
            page = p.json()
            when = page["@effectiveDate"]  # local day at 00:00, with its offset
            if not start.isoformat() <= when[:10] <= end.isoformat():
                continue
            title = page["@title"]
            if PRODUCTION.search(title):
                self.keep_raw(p.content, "json")
            meta = {"title": title, "url": page["@canonicalUrl"]}
            rows.append(_row("BASF", page["@name"], when, meta))
        return pd.DataFrame(rows, columns=COLUMNS)
