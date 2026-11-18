"""Daily German gas consumption by customer type, from Trading Hub Europe (THE), no key.

THE publishes, for each gas day, the allocated consumption of its market area by type of
exit point: standard load profiles (SLP, synthetic and analytic) and metered exits (RLM,
with and without a daily band), each for H-gas and L-gas. Large industrial users, ammonia
plants among them, are metered (RLM), so the RLM series is the daily industrial demand of
Germany. It is an aggregate, not one plant.

THE started on 1 October 2021. Before that, Germany had two market areas. THE keeps their
files: NetConnect Germany (NCG, from April 2011, kWh) and GASPOOL (GPL, from October 2016,
MWh). This source reads the API for days from October 2021 and the two archive files for the
days before. series_id = <area>.<type>, for example THE.RLMmT_H. Units stay as published.
``meta`` holds the allocation status (preliminary or final) where the source gives one.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

import pandas as pd

from nsqueeze.io.sources.base import Source

API = "https://datenservice-api.tradinghub.eu/api/evoq/GetAggregierteVerbrauchsdatenTabelle"
NCG_ZIP = "https://www.tradinghub.eu/Portals/0/Aggregierte%20Verbrauchsdaten%20NCG_31012022.zip"
GPL_ZIP = "https://www.tradinghub.eu/Portals/0/verbrauchsdaten-gesamt%5B1%5D.zip"
THE_START = dt.date(2021, 10, 1)
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]
API_FIELDS = {
    "slPsyn_H_Gas": "SLPsyn_H", "slPana_H_Gas": "SLPana_H",
    "slPsyn_L_Gas": "SLPsyn_L", "slPana_L_Gas": "SLPana_L",
    "rlMmT_H_Gas": "RLMmT_H", "rlMmT_L_Gas": "RLMmT_L",
    "rlMoT_H_Gas": "RLMoT_H", "rlMoT_L_Gas": "RLMoT_L",
}  # fmt: skip


def api_canonical(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        day = pd.Timestamp(r["gastag"][:10])
        meta = json.dumps({"status": r.get("statusEN"), "type": r.get("gasXType")})
        for field, name in API_FIELDS.items():
            if r.get(field) is not None:
                rows.append([f"THE.{name}", day, float(r[field]), "kWh", meta])
    return pd.DataFrame(rows, columns=COLUMNS)


def ncg_canonical(text: str) -> pd.DataFrame:
    """NCG file: one row per day, a value column and a unit column for each type."""
    df = pd.read_csv(io.StringIO(text), sep=";")
    rows = []
    for r in df.itertuples(index=False):
        r = r._asdict()
        day = pd.to_datetime(r["DayOfUse"], format="%d.%m.%Y")
        meta = json.dumps({"status": r.get("Status")})
        for col in df.columns:
            if col.startswith(("HGas", "LGas")):  # the unit columns start with "Unit"
                gas, kind = col[0], col[4:]  # "HGasRLMmT" -> "H", "RLMmT"
                rows.append([f"NCG.{kind}_{gas}", day, float(r[col]), r[f"Unit{col}"], meta])
    return pd.DataFrame(rows, columns=COLUMNS)


def gpl_canonical(text: str) -> pd.DataFrame:
    """GASPOOL file: one row per day, columns such as "RLMmT_H [MWh]"."""
    df = pd.read_csv(io.StringIO(text), sep=";")
    long = df.melt(id_vars="Datum", var_name="col", value_name="value")
    name = long.col.str.extract(r"^(\w+) \[(\w+)\]$")
    return pd.DataFrame(
        {
            "series_id": "GPL." + name[0],
            "observed_at": pd.to_datetime(long.Datum, format="%d.%m.%Y"),
            "value": pd.to_numeric(long.value).astype(float),
            "unit": name[1],
            "meta": json.dumps({}),
        }
    )[COLUMNS]


class THEConsumption(Source):
    name = "the_consumption"
    verified = True  # API and both archive files checked live on 2026-11-19

    def _zip_text(self, url: str, encoding: str) -> str:
        r = self.client.get(url)
        r.raise_for_status()
        self.keep_raw(r.content, "zip")
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            return z.read(z.namelist()[0]).decode(encoding)

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames = []
        if start < THE_START:
            frames.append(ncg_canonical(self._zip_text(NCG_ZIP, "latin-1")))
            frames.append(gpl_canonical(self._zip_text(GPL_ZIP, "latin-1")))
        if end >= THE_START:
            a = max(start, THE_START)
            r = self.client.get(
                API,
                params={
                    "DatumStart": a.strftime("%m-%d-%Y"),
                    "DatumEnde": end.strftime("%m-%d-%Y"),
                    "GasXType_Id": "all",
                },
            )
            r.raise_for_status()
            self.keep_raw(r.content, "json")
            frames.append(api_canonical(r.json()))
        out = pd.concat(frames, ignore_index=True)
        day = out.observed_at.dt.date
        return out[(day >= start) & (day <= end)].reset_index(drop=True)
