"""GUS monthly production: edition links, the two Table 2 layouts, latest edition wins.

No network. The workbooks are built here with invented values.
"""

import datetime as dt
import io
import json

import httpx
import pandas as pd

from nsqueeze.io.sources import gus
from nsqueeze.io.sources.gus import GUSProduction, editions, table2


def book(rows: list[list]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        pd.DataFrame([["Spis treści"]]).to_excel(w, sheet_name="Spis", header=False, index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="Tablica 2", header=False, index=False)
    return buf.getvalue()


def new_layout(prev: int, cur: int, scale: float) -> bytes:
    """2020s: year in column 2, months as 1, "02", 3.0 ... in every second column."""
    head = ["WYROBY", "J.m.", "Rok", "01-12", None]
    for m in range(1, 13):
        head += [m if m % 2 else f"{m:02d}", None]
    name = ["Amoniak bezwodny w przeliczeniu na 100%", "tys. t", prev, 12 * scale, None]
    nxt = [None, None, cur, 12 * scale, None]
    for _m in range(12):
        name += [scale, None]
        nxt += [scale + 1, None]
    return book([["TABL. 2. PRODUKCJA"], head, name, nxt, ["Kwas azotowy", "tys. t", prev, 1]])


def old_layout() -> bytes:
    """2016-2019: year in column 3, months as Roman numerals."""
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
    return book(
        [
            ["TABL. 2 PRODUKCJA"],
            [None, None, None, None, "I-XII", *roman],
            ["Amoniak bezwodny", None, None, 2015, 2400, *[200] * 12],
            [None, None, None, 2016, 2412, *[201] * 12],
        ]
    )


def test_both_layouts_give_24_months():
    new = table2(new_layout(2021, 2022, 10.0), dt.date(2022, 12, 1))
    old = table2(old_layout(), dt.date(2016, 12, 1))
    assert len(new) == len(old) == 24
    assert new.observed_at.min() == pd.Timestamp(2021, 1, 1)
    assert new.value.tolist() == [10.0] * 12 + [11.0] * 12
    assert old.value.tolist() == [200.0] * 12 + [201.0] * 12
    assert json.loads(old.meta[0]) == {"edition": "2016-12"}


def test_edition_links():
    html = (
        '<a href="/obszary-tematyczne/x/produkcja-wazniejszych-wyrobow-przemyslowych-'
        'we-wrzesniu-2016-roku,2,54.html">'
        '<a href="/obszary-tematyczne/x/produkcja-wazniejszych-wyrobow-przemyslowych-'
        'w-grudniu-2025-r-,2,165.html#archive">'
    )
    assert sorted(editions(html)) == [dt.date(2016, 9, 1), dt.date(2025, 12, 1)]


def test_fetch_keeps_the_latest_edition_per_month(tmp_path, monkeypatch):
    monkeypatch.setenv("NSQ_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(gus.time, "sleep", lambda s: None)
    page = (
        "/obszary-tematyczne/x/produkcja-wazniejszych-wyrobow-przemyslowych-w-{}-2022-r-,2,{}.html"
    )
    links = f'<a href="{page.format("grudniu", 2)}"> <a href="{page.format("listopadzie", 1)}">'
    books = {"/1.xlsx": new_layout(2021, 2022, 5.0), "/2.xlsx": new_layout(2021, 2022, 7.0)}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in books:
            return httpx.Response(200, content=books[path])
        if path.endswith(",1.html"):
            return httpx.Response(200, text='<a href="/1.xlsx">' + links)
        if path.endswith(",2.html"):
            return httpx.Response(200, text='<a href="/2.xlsx">' + links)
        return httpx.Response(200, text=links)  # the index and the archive of the latest edition

    src = GUSProduction(client=httpx.Client(transport=httpx.MockTransport(handler)))
    df = src.fetch(dt.date(2022, 11, 1), dt.date(2022, 12, 31))
    assert len(df) == 24
    assert set(df.value) == {7.0, 8.0}  # the December edition revised every month
    assert {json.loads(m)["edition"] for m in df.meta} == {"2022-12"}
