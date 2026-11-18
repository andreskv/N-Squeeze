"""Comext by reporting country: series IDs, flows, the Greece code and the month cut.

No network. The payload is a small JSON-stat body with invented values.
"""

import datetime as dt
import json

import httpx

from nsqueeze.io.sources.eurostat import EU27_GEO, ComextReporter, reporter_canonical


def body(reporter: str) -> dict:
    dims = {
        "freq": ["M"],
        "reporter": [reporter],
        "partner": ["DZ", "EXT_EU27_2020"],
        "product": ["2814"],
        "flow": ["1", "2"],
        "indicators": ["QUANTITY_IN_100KG"],
        "time": ["2025-11", "2025-12"],
    }
    ids = list(dims)
    sizes = [len(v) for v in dims.values()]
    n = 1
    for s in sizes:
        n *= s
    return {
        "id": ids,
        "size": sizes,
        "dimension": {
            d: {"category": {"index": {c: i for i, c in enumerate(v)}}} for d, v in dims.items()
        },
        "value": {str(i): float(i) for i in range(n)},
        "updated": "2026-11-15",
    }


def test_series_ids_name_reporter_flow_product_partner_indicator():
    df = reporter_canonical(body("NL"), dt.date(2025, 11, 30))
    assert set(df.series_id) == {
        f"COMEXT.NL.{f}.2814.{p}.QUANTITY_IN_100KG"
        for f in ("IMP", "EXP")
        for p in ("DZ", "EXT_EU27_2020")
    }
    assert df.observed_at.max() == dt.datetime(2025, 11, 1)  # December is after the end date
    assert df.unit.eq("100kg").all() and not df.duplicated(["series_id", "observed_at"]).any()
    assert json.loads(df.meta.iloc[0])["updated"] == "2026-11-15"


def test_fetch_asks_each_reporter_and_writes_greece_as_gr(tmp_path, monkeypatch):
    monkeypatch.setenv("NSQ_DATA_DIR", str(tmp_path))
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        rep = request.url.params["reporter"]
        asked.append(rep)
        return httpx.Response(200, json=body(rep))

    src = ComextReporter(client=httpx.Client(transport=httpx.MockTransport(handler)))
    df = src.fetch(dt.date(2025, 11, 1), dt.date(2025, 12, 31))
    assert len(asked) == len(EU27_GEO) and "GR" in asked and "EL" not in asked
    assert df.series_id.str.startswith("COMEXT.GR.").any()
