"""Newsweb filings: one row per filing, bodies only for production titles, year windows.

No network. The messages are invented.
"""

import datetime as dt
import json

import httpx

from nsqueeze.io.sources import newsweb
from nsqueeze.io.sources.newsweb import NewswebFilings, to_canonical, year_windows

MSGS = [
    {
        "messageId": 1,
        "title": "Yara curtails ammonia production",
        "publishedTime": "2022-08-25T06:00:00.000Z",
        "issuerSign": "YAR",
        "category": [{"category_en": "NON-REGULATORY PRESS RELEASES"}],
    },
    {
        "messageId": 2,
        "title": "Flagging i Yara International ASA",
        "publishedTime": "2022-09-12T13:34:51.579Z",
        "issuerSign": "YAR",
        "category": [{"category_en": "MAJOR SHAREHOLDINGS NOTIFICATION"}],
    },
]


def test_canonical_rows_and_body_only_where_given():
    df = to_canonical(MSGS, {1: "Oslo, 25 August 2022: ..."})
    assert df.series_id.tolist() == ["NEWSWEB.YAR.1", "NEWSWEB.YAR.2"]
    assert df.observed_at[0] == dt.datetime(2022, 8, 25, 6, 0)
    m0, m1 = (json.loads(m) for m in df.meta)
    assert m0["body"].startswith("Oslo") and "body" not in m1
    assert m0["url"].endswith("/message/1")


def test_year_windows_cover_the_range_once():
    w = year_windows(dt.date(2021, 6, 1), dt.date(2023, 2, 1))
    assert w == [
        (dt.date(2021, 6, 1), dt.date(2021, 12, 31)),
        (dt.date(2022, 1, 1), dt.date(2022, 12, 31)),
        (dt.date(2023, 1, 1), dt.date(2023, 2, 1)),
    ]


def test_fetch_gets_bodies_for_production_titles_only(tmp_path, monkeypatch):
    monkeypatch.setenv("NSQ_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(newsweb.time, "sleep", lambda s: None)
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/list"):
            return httpx.Response(200, json={"data": {"messages": MSGS}})
        asked.append(int(request.url.params["messageId"]))
        return httpx.Response(200, json={"data": {"message": {"body": "text"}}})

    src = NewswebFilings(client=httpx.Client(transport=httpx.MockTransport(handler)))
    df = src.fetch(dt.date(2022, 1, 1), dt.date(2022, 12, 31))
    assert asked == [1]
    assert len(df) == 2


def test_results_release_keeps_the_report_pdf_not_the_slides(tmp_path, monkeypatch):
    monkeypatch.setenv("NSQ_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(newsweb.time, "sleep", lambda s: None)
    msg = {**MSGS[0], "messageId": 7, "title": "Yara reports improved earnings"}
    pulled = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/list"):
            return httpx.Response(200, json={"data": {"messages": [msg]}})
        if path.endswith("/attachment"):
            pulled.append(int(request.url.params["attachmentId"]))
            return httpx.Response(200, content=b"%PDF-1.7 test")
        files = [
            {"id": 1, "name": "Yara 3Q Report.pdf"},
            {"id": 2, "name": "Yara 3Q Presentation.pdf"},
        ]
        return httpx.Response(200, json={"data": {"message": {"body": "b", "attachments": files}}})

    src = NewswebFilings(client=httpx.Client(transport=httpx.MockTransport(handler)))
    df = src.fetch(dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    assert pulled == [1]
    assert [a["name"] for a in json.loads(df.meta[0])["attachments"]] == [
        "Yara 3Q Report.pdf",
        "Yara 3Q Presentation.pdf",
    ]
    assert list((tmp_path / "raw" / "newsweb_filings").rglob("*.pdf.gz"))
