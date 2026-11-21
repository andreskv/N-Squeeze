"""Daily gas use of large Spanish industrial consumers by sector, from Enagás, no key.

Enagás publishes an index of large industrial gas consumers (IGIG) with daily use in GWh/d
for each sector of activity: chemical and pharmaceutical (Fertiberia's plants are in it),
refining, metals, paper, food and others. Two workbooks hold the daily rows:

- a file with one sheet per year for 2019-2021 ("demanda por sectores y provincia diario");
- DatosIGIG.xlsx, the current year, which Enagás replaces each year.

For 2022-2025 Enagás publishes only monthly PDF reports, so these years have no daily rows
here. Archiving DatosIGIG from now on keeps each new year. series_id = ENAGAS.IGIG.<sector>.
"""

from __future__ import annotations

import datetime as dt
import io
import re
import unicodedata

import pandas as pd

from nsqueeze.io.sources.base import Source

BASE = (
    "https://www.enagas.es/content/dam/enagas/en/files/gestion-tecnica-del-sistema/"
    "energy-data/publicaciones/demanda-de-gas/informe-demanda-industrial/"
)
FILE_2019_2021 = BASE + "demanda%20por%20sectores%20y%20provincia%20diario_envio.xlsx"
FILE_CURRENT = BASE + "DatosIGIG.xlsx"
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]


def sector_code(name: str) -> str:
    """'QUÍMICA/FARMACEÚTICA' -> 'QUIMICA_FARMACEUTICA'."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]+", "_", s.upper()).strip("_")


def daily_rows(book: bytes) -> pd.DataFrame:
    """Rows of every sheet laid out as date | sector | GWh/d (headers and totals skipped)."""
    frames = []
    for sheet in pd.read_excel(io.BytesIO(book), sheet_name=None, header=None).values():
        if sheet.shape[1] < 3:
            continue
        day = pd.to_datetime(sheet[0], errors="coerce")
        value = pd.to_numeric(sheet[2], errors="coerce")
        ok = day.notna() & (day.dt.year > 2000) & value.notna() & sheet[1].notna()
        frames.append(
            pd.DataFrame(
                {
                    "series_id": "ENAGAS.IGIG." + sheet.loc[ok, 1].astype(str).map(sector_code),
                    "observed_at": day[ok].dt.normalize(),
                    "value": value[ok].astype(float),
                    "unit": "GWh/d",
                    "meta": "{}",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


class EnagasIndustrial(Source):
    name = "enagas_industrial"
    verified = True  # both workbooks checked live on 2026-11-19

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        urls = []
        if start <= dt.date(2021, 12, 31) and end >= dt.date(2019, 1, 1):
            urls.append(FILE_2019_2021)
        if end >= dt.date(2026, 1, 1):  # DatosIGIG holds the current year only
            urls.append(FILE_CURRENT)
        frames = []
        for url in urls:
            r = self.client.get(url)
            r.raise_for_status()
            self.keep_raw(r.content, "xlsx")
            frames.append(daily_rows(r.content))
        if not frames:
            return pd.DataFrame(columns=COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        day = out.observed_at.dt.date
        return out[(day >= start) & (day <= end)].reset_index(drop=True)
