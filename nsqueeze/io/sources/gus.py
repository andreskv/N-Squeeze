"""Monthly production of ammonia in Poland, from Statistics Poland (GUS), no key.

GUS publishes "Produkcja ważniejszych wyrobów przemysłowych" each month as a workbook.
Table 2 gives the production of each listed product in each month of the current and the
previous year, in thousand tonnes. Anhydrous ammonia (100 % NH3) is one of the products.
Poland's ammonia comes from five plants, four of them Grupa Azoty, so the national month is
close to a plant-group truth.

Each edition is a vintage: a later edition can revise a month. fetch() reads the editions
whose reference month is in the window and keeps, for each month, the value of the latest
edition read. ``meta`` names that edition. The edition list comes from the archive links on
the GUS industry pages.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import time

import pandas as pd

from nsqueeze.io.sources.base import Source

BASE = "https://stat.gov.pl"
INDEX = BASE + "/obszary-tematyczne/przemysl-budownictwo-srodki-trwale/przemysl/"
MONTHS = {
    "styczniu": 1, "lutym": 2, "marcu": 3, "kwietniu": 4, "maju": 5, "czerwcu": 6,
    "lipcu": 7, "sierpniu": 8, "wrzesniu": 9, "pazdzierniku": 10, "listopadzie": 11,
    "grudniu": 12,
}  # fmt: skip
EDITION = re.compile(
    r"/obszary-tematyczne/[^\"#]*produkcja-wazniejszych-wyrobow-przemyslowych-we?-"
    r"(" + "|".join(MONTHS) + r")-(20\d\d)-r[^\"#]*\.html"
)
PRODUCTS = {"Amoniak bezwodny": "GUS.PROD.NH3_ANHYDROUS"}
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]


def editions(html: str) -> dict[dt.date, str]:
    """Edition reference month -> page path."""
    return {dt.date(int(m[2]), MONTHS[m[1]], 1): m[0] for m in EDITION.finditer(html)}


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def month_columns(sheet: pd.DataFrame) -> dict[int, int]:
    """Column -> month, from the first header row that names all 12 months.

    Editions write the months as Roman numerals (2016-2019) or numbers (1, "02", 3.0).
    """
    for i in range(min(12, len(sheet))):
        cols = {}
        for c, v in sheet.iloc[i].items():
            s = str(v).strip().removesuffix(".0")
            if s in ROMAN:
                cols[c] = ROMAN.index(s) + 1
            elif s.isdigit() and 1 <= int(s) <= 12:
                cols[c] = int(s)
        if sorted(cols.values()) == list(range(1, 13)):
            return cols
    return {}


def year_of(row: pd.Series) -> int | None:
    for v in row.iloc[1:5]:
        y = pd.to_numeric(v, errors="coerce")
        if pd.notna(y) and 2000 <= y <= 2040 and y == int(y):
            return int(y)
    return None


def table2(book: bytes, edition: dt.date) -> pd.DataFrame:
    """Monthly rows of the PRODUCTS from Table 2 (months by year) of one edition workbook.

    The layout changes between editions, so the parser looks in every sheet for a header row
    with the 12 months and for the product rows below it. A product has one row per year.
    """
    rows = []
    for sheet in pd.read_excel(io.BytesIO(book), sheet_name=None, header=None).values():
        months = month_columns(sheet)
        if not months:
            continue
        for i in range(len(sheet)):
            name = str(sheet.iloc[i, 0])
            sid = next((v for k, v in PRODUCTS.items() if name.startswith(k)), None)
            if sid is not None:
                rows += _product_rows(sheet, i, sid, months, edition)
    return pd.DataFrame(rows, columns=COLUMNS)


def _product_rows(sheet, i, sid, months, edition) -> list[dict]:
    rows = []
    for j in range(i, min(i + 4, len(sheet))):  # the year rows follow the name row
        if j > i and str(sheet.iloc[j, 0]) != "nan" and not str(sheet.iloc[j, 0]).startswith("w "):
            break  # the next product
        year = year_of(sheet.iloc[j])
        if year is None:
            continue
        for c, m in months.items():
            v = pd.to_numeric(sheet.iloc[j, c], errors="coerce")
            if pd.notna(v):
                rows.append(
                    {
                        "series_id": sid,
                        "observed_at": pd.Timestamp(year, m, 1),
                        "value": float(v),
                        "unit": "kt",
                        "meta": json.dumps({"edition": edition.strftime("%Y-%m")}),
                    }
                )
    return rows


class GUSProduction(Source):
    name = "gus_production"
    verified = True  # edition pages and workbooks checked live on 2026-11-18 (2016-2025)

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        r = self.client.get(INDEX)
        r.raise_for_status()
        latest = EDITION.search(r.text)
        r = self.client.get(BASE + latest[0])  # its archive lists every edition
        r.raise_for_status()
        frames = []
        first = dt.date(start.year, start.month, 1)
        for month, path in sorted(editions(r.text).items()):
            if not first <= month <= end:
                continue
            page = self.client.get(BASE + path)
            page.raise_for_status()
            link = re.search(r'href="([^"]+\.xlsx?)"', page.text)
            if link is None:
                continue
            book = self.client.get(BASE + link[1] if link[1].startswith("/") else link[1])
            book.raise_for_status()
            self.keep_raw(book.content, "xlsx")
            frames.append(table2(book.content, month))
            time.sleep(0.5)
        if not frames:
            return pd.DataFrame(columns=COLUMNS)
        out = pd.concat(frames)  # editions in date order, so the last one per month is latest
        return out.drop_duplicates(["series_id", "observed_at"], keep="last").reset_index(drop=True)
