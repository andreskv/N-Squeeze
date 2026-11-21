"""Enagás industrial gas: sector codes, sheet rows, which workbook each window reads.

No network. The values are invented.
"""

import datetime as dt
import io

import httpx
import pandas as pd

from nsqueeze.io.sources.enagas import EnagasIndustrial, daily_rows, sector_code


def book(year: int) -> bytes:
    rows = [
        [None, None, None],
        [year, "TOTAL AÑO", 999.0],
        ["(GWh/día)", "Sector actividad", None],
        [pd.Timestamp(year, 1, 1), "QUÍMICA/FARMACEÚTICA", 80.5],
        [pd.Timestamp(year, 1, 1), "REFINO", 30.0],
        [pd.Timestamp(year, 1, 2), "QUÍMICA/FARMACEÚTICA", 79.0],
    ]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        pd.DataFrame(rows).to_excel(w, sheet_name=f"{year}_diario", header=False, index=False)
    return buf.getvalue()


def test_sector_code_strips_accents_and_signs():
    assert sector_code("QUÍMICA/FARMACEÚTICA") == "QUIMICA_FARMACEUTICA"
    assert sector_code("RESTO INDUSTRIA") == "RESTO_INDUSTRIA"


def test_daily_rows_skip_headers_and_the_year_total():
    df = daily_rows(book(2021))
    assert len(df) == 3
    assert df.series_id.tolist()[:2] == ["ENAGAS.IGIG.QUIMICA_FARMACEUTICA", "ENAGAS.IGIG.REFINO"]
    assert df.observed_at[2] == pd.Timestamp(2021, 1, 2) and df.value[2] == 79.0


def test_fetch_reads_the_workbook_of_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("NSQ_DATA_DIR", str(tmp_path))
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        asked.append(name)
        return httpx.Response(200, content=book(2026 if name == "DatosIGIG.xlsx" else 2021))

    src = EnagasIndustrial(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert src.fetch(dt.date(2023, 1, 1), dt.date(2024, 12, 31)).empty and asked == []
    old = src.fetch(dt.date(2021, 1, 1), dt.date(2021, 1, 1))
    assert len(old) == 2 and len(asked) == 1
    new = src.fetch(dt.date(2026, 1, 2), dt.date(2026, 1, 31))
    assert asked[-1] == "DatosIGIG.xlsx" and len(new) == 1
