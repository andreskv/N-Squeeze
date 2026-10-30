"""Grain futures fetcher: contract symbols, the parser and the fetch against a real sample.

No network here. tests/fixtures/grains_chart_sample.json is a Yahoo chart response for
ZCZ27.CBT, cut to its first three trading days.
"""

import datetime as dt
import json
from pathlib import Path

import httpx

from nsqueeze.io.sources.grains import GrainFutures, chart_close_to_canonical, contract_symbols

SAMPLE = Path(__file__).parent / "fixtures" / "grains_chart_sample.json"


def test_contract_symbols_in_december_and_after_the_year_change():
    # December keeps its own December contract. The January list drops it and the
    # soybean November contract of the old year.
    assert contract_symbols(dt.date(2026, 12, 15)) == [
        "ZCZ26.CBT", "ZCN27.CBT", "ZCZ27.CBT", "ZCN28.CBT", "ZCZ28.CBT", "ZCN29.CBT", "ZCZ29.CBT",
        "ZSN27.CBT", "ZSX27.CBT", "ZSN28.CBT", "ZSX28.CBT", "ZSN29.CBT", "ZSX29.CBT",
        "ZWZ26.CBT", "ZWN27.CBT", "ZWZ27.CBT", "ZWN28.CBT", "ZWZ28.CBT", "ZWN29.CBT", "ZWZ29.CBT",
    ]  # fmt: skip
    assert contract_symbols(dt.date(2027, 1, 5)) == [
        "ZCN27.CBT", "ZCZ27.CBT", "ZCN28.CBT", "ZCZ28.CBT", "ZCN29.CBT", "ZCZ29.CBT",
        "ZSN27.CBT", "ZSX27.CBT", "ZSN28.CBT", "ZSX28.CBT", "ZSN29.CBT", "ZSX29.CBT",
        "ZWN27.CBT", "ZWZ27.CBT", "ZWN28.CBT", "ZWZ28.CBT", "ZWN29.CBT", "ZWZ29.CBT",
    ]  # fmt: skip


def test_grain_rows_from_the_sample_response():
    body = json.loads(SAMPLE.read_text(encoding="utf-8"))
    df = chart_close_to_canonical(body, "ZCZ27.CBT")
    assert len(df) == 3 and df.observed_at.is_unique  # one row per trading day
    assert df.observed_at.dt.strftime("%Y-%m-%d").tolist() == [
        "2026-10-15",
        "2026-10-16",
        "2026-10-19",
    ]
    assert df.value.tolist() == [509.25, 506.75, 504.25]
    assert df.series_id.unique().tolist() == ["YF.ZCZ27.CBT.close"]
    assert df.unit.unique().tolist() == ["USX/bu"]
    assert json.loads(df.meta.iloc[0]) == {"exchange": "CBT"}


def test_grain_null_result_gives_no_rows():
    # the body of the 404 that Yahoo returns for a removed contract
    body = {
        "chart": {
            "result": None,
            "error": {"code": "Not Found", "description": "No data found, symbol may be delisted"},
        }
    }
    df = chart_close_to_canonical(body, "ZCZ24.CBT")
    assert df.empty
    assert list(df.columns) == ["series_id", "observed_at", "value", "unit", "meta"]


def test_grain_fetch_continues_after_a_404(monkeypatch, capsys):
    monkeypatch.setattr("nsqueeze.io.sources.grains.time.sleep", lambda s: None)
    sample = SAMPLE.read_text(encoding="utf-8")
    seen = []

    def handler(req):
        symbol = req.url.path.rsplit("/", 1)[-1]
        seen.append(symbol)
        if symbol == "ZCN28.CBT":
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, text=sample)

    src = GrainFutures(client=httpx.Client(transport=httpx.MockTransport(handler)))
    src.keep_raw = lambda *a: None
    end = dt.date(2026, 12, 15)
    df = src.fetch(end - dt.timedelta(days=10), end)

    assert seen == [*contract_symbols(end), "ZC=F", "ZS=F", "ZW=F"]
    expected = {f"YF.{s}.close" for s in seen if s != "ZCN28.CBT"}
    assert set(df.series_id) == expected and len(df) == 3 * len(expected)
    assert "grain_futures: 1 of 23 symbols gave no rows: ZCN28.CBT" in capsys.readouterr().err
