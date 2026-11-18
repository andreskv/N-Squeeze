"""Company press releases (OCI, BASF): local publication day, bodies for production titles only.

No network. The posts and pages are invented.
"""

import datetime as dt
import json

import httpx

from nsqueeze.io.sources import company_news
from nsqueeze.io.sources.company_news import BASFNews, basf_paths, oci_canonical

POSTS = [
    {
        "id": 5,
        "date": "2022-08-02T00:00:00",  # migrated post: local day only
        "link": "https://oci-global.com/news-stories/press-releases/a/",
        "title": {"rendered": "OCI curtails ammonia production at Geleen"},
        "content": {"rendered": "<p>Due to high gas prices &amp; more.</p>"},
    },
    {
        "id": 6,
        "date": "2023-05-11T07:00:00",
        "link": "https://oci-global.com/news-stories/press-releases/b/",
        "title": {"rendered": "OCI appoints a new CFO"},
        "content": {"rendered": "<p>text</p>"},
    },
]


def test_oci_rows_use_the_local_day_and_keep_bodies_for_production_titles():
    df = oci_canonical(POSTS)
    assert df.series_id.tolist() == ["NEWS.OCI.5", "NEWS.OCI.6"]
    assert df.observed_at.tolist() == [dt.datetime(2022, 8, 2), dt.datetime(2023, 5, 11)]
    m0, m1 = (json.loads(m) for m in df.meta)
    assert m0["body"] == "Due to high gas prices & more."
    assert "body" not in m1 and m1["published"] == "2023-05-11T07:00:00"


def test_basf_paths_keep_the_window_months_once():
    base = "https://www.basf.com/global/en/media/news-releases/"
    sitemap = " ".join(
        base + p
        for p in ("2022/08/p-22-300", "2022/09/p-22-313", "2022/09/p-22-313", "2022/10/p-22-400")
    )
    got = basf_paths(sitemap, dt.date(2022, 9, 15), dt.date(2022, 9, 30))
    assert got == ["global/en/media/news-releases/2022/09/p-22-313"]


def test_basf_fetch_skips_removed_pages_and_days_outside_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("NSQ_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(company_news.time, "sleep", lambda s: None)
    base = "global/en/media/news-releases/2022/09/"
    pages = {
        "p-22-1": ("2022-09-05T00:00:00.000+02:00", "BASF cuts ammonia production"),
        "p-22-2": ("2022-09-30T00:00:00.000+02:00", "BASF opens an office"),
        "p-22-3": ("2022-09-01T00:00:00.000+02:00", "Outside the window"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith(".xml"):
            names = [*pages, "p-22-9", "p-22-4-BASF-YPC"]
            return httpx.Response(200, text=" ".join(base + n for n in names))
        name = path.rsplit("/", 1)[-1]
        if name == "p-22-4-BASF-YPC":
            return httpx.Response(200, json={})  # an empty node
        if name not in pages:
            return httpx.Response(404)
        when, title = pages[name]
        page = {"@name": name, "@title": title, "@effectiveDate": when, "@canonicalUrl": path}
        return httpx.Response(200, json=page)

    src = BASFNews(client=httpx.Client(transport=httpx.MockTransport(handler)))
    df = src.fetch(dt.date(2022, 9, 2), dt.date(2022, 9, 30))
    assert df.series_id.tolist() == ["NEWS.BASF.p-22-1", "NEWS.BASF.p-22-2"]
    assert df.observed_at.tolist() == [dt.datetime(2022, 9, 5), dt.datetime(2022, 9, 30)]
    assert len(list((tmp_path / "raw" / "basf_news").rglob("*.json.gz"))) == 1
