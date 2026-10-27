"""Eurostat PRODCOM fetcher: the parser and the fetch against a real response sample.

No network here. The sample in tests/fixtures/prodcom_sample.json is a DS-059359
response for two products, three countries and two years.
"""

import datetime as dt
import json
from pathlib import Path

from nsqueeze.io.sources.prodcom import EurostatProdcom, to_canonical

SAMPLE = Path(__file__).parent / "fixtures" / "prodcom_sample.json"


def _sample() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_prodcom_rows_from_the_sample_response():
    df = to_canonical(_sample())
    assert len(df) == 7  # the confidential cells of BE and DE 2023 anhydrous are absent
    row = df[df.series_id == "ESTAT.PRODCOM.20151075.APRODQNT.PL"].set_index("observed_at")
    assert row.value.loc["2023-01-01"] == 1480576000
    assert row.value.loc["2024-01-01"] == 1729526000
    de = df[df.series_id == "ESTAT.PRODCOM.20151077.APRODQNT.DE"].set_index("observed_at")
    assert de.value.loc["2023-01-01"] == 108411000 and de.unit.iloc[0] == "KG_N"
    assert (df.unit == "KG_N").all() and (df.observed_at.dt.day == 1).all()
    meta = json.loads(df.meta.iloc[0])
    assert meta == {"flag": None, "updated": "2026-10-20T11:00:00+0200"}


def test_prodcom_confidential_cells_stay_out():
    body = _sample()
    body["value"]["0"] = 123456789  # BE anhydrous 2023, whose flag says :C
    df = to_canonical(body)
    assert len(df) == 7
    assert not df.series_id.str.contains(".BE.").any()


def test_prodcom_flag_of_the_value_reaches_meta():
    body = {
        "updated": "2026-10-20T11:00:00+0200",
        "id": ["reporter", "product", "indicators", "time"],
        "size": [2, 1, 3, 1],
        "dimension": {
            "reporter": {"category": {"index": {"BE": 0, "DE": 1}}},
            "product": {"category": {"index": {"20151075": 0}}},
            "indicators": {"category": {"index": {"APRODQNT": 0, "QNTUNIT": 1, "APQNTFLAG": 2}}},
            "time": {"category": {"index": {"2023": 0}}},
        },
        # cells of one year: quantity, unit and flag per country
        "value": {"0": 100, "1": "KG_N", "2": ":C", "3": 200, "4": "KG_N", "5": ":E"},
    }
    df = to_canonical(body)
    assert len(df) == 1  # BE stays out although its quantity cell is present
    assert df.series_id.iloc[0] == "ESTAT.PRODCOM.20151075.APRODQNT.DE"
    assert json.loads(df.meta.iloc[0])["flag"] == ":E"


def test_prodcom_empty_response_gives_no_rows():
    body = _sample()
    body["value"] = {}
    df = to_canonical(body)
    assert df.empty
    assert list(df.columns) == ["series_id", "observed_at", "value", "unit", "meta"]


def test_prodcom_fetch_reads_both_lists_and_stops_at_end():
    import httpx

    seen = []

    def handler(req):
        seen.append(str(req.url))
        body = (
            _sample()
            if "DS-059359" in str(req.url)
            else {
                "updated": "2026-10-20T11:00:00+0200",
                "id": ["reporter", "product", "indicators", "time"],
                "size": [1, 1, 3, 1],
                "dimension": {
                    "reporter": {"category": {"index": {"DE": 0}}},
                    "product": {"category": {"index": {"20151075": 0}}},
                    "indicators": {
                        "category": {"index": {"APRODQNT": 0, "QNTUNIT": 1, "APQNTFLAG": 2}}
                    },
                    "time": {"category": {"index": {"2025": 0}}},
                },
                "value": {"0": 1675443000, "1": "KG_N"},
            }
        )
        return httpx.Response(200, json=body)

    src = EurostatProdcom(client=httpx.Client(transport=httpx.MockTransport(handler)))
    src.keep_raw = lambda *a: None
    df = src.fetch(dt.date(2023, 1, 1), dt.date(2024, 12, 31))
    assert len(seen) == 2 and "DS-059359" in seen[0] and "DS-059368" in seen[1]
    assert "sinceTimePeriod=2023" in seen[0] and "EU27_2020" in seen[0]
    assert len(df) == 7 and set(df.observed_at.dt.year) == {2023, 2024}
