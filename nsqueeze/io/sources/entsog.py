"""ENTSOG Transparency Platform: daily physical flows at every reported point (key-free).

The gas balance needs Norwegian entry points (N_t), LNG entry points (L_t) and other import
pipelines (P_t). The fetcher keeps *all* points and their metadata rather than a
hand-picked list: point classification is a model decision (``gas_balance``), and a
PIT archive should not bake one in. About 1,000 rows per gas day.

Endpoint, fields and units checked live on 2026-10-12: ``/api/v1/operationaldatas``,
``indicator=Physical Flow``, ``periodType=day``, values in kWh/d. ``flowStatus``
(Provisional / Confirmed) is kept so revisions show up as new vintages.

The API serves only about the last five years ("archived on a rolling basis"); older
days are in the TP Archives download section. The PIT store now holds
2021-01-01 onward, so days rolling out of the API are kept here.
"""

from __future__ import annotations

import datetime as dt
import json
import sys

import pandas as pd

from nsqueeze.io.sources.base import Source

URL = "https://transparency.entsog.eu/api/v1/operationaldatas"
META = ("pointLabel", "pointType", "operatorLabel", "flowStatus", "lastUpdateDateTime")
FLOW_STATUS_RANK = {"Confirmed": 0, "Provisional": 1}


def to_canonical(rows: list[dict]) -> pd.DataFrame:
    """One row per (operator, point, direction, gas day). Gas day = local date of periodFrom.

    Some points report a day twice, as "Confirmed" and as "Provisional". I keep the confirmed
    value, then the provisional one, then a row with no status."""
    rows = sorted(rows, key=lambda r: FLOW_STATUS_RANK.get(r.get("flowStatus"), 2))
    out = [
        {
            "series_id": f"ENTSOG.{r['operatorKey']}.{r['pointKey']}.{r['directionKey']}",
            "observed_at": r["periodFrom"][:10],
            "value": r["value"],
            "unit": r["unit"],
            "meta": json.dumps({k: r.get(k) for k in META}, ensure_ascii=False),
        }
        for r in rows
        if r.get("value") is not None and r.get("pointKey")
    ]
    df = pd.DataFrame(out, columns=["series_id", "observed_at", "value", "unit", "meta"])
    return df.drop_duplicates(["series_id", "observed_at"]).reset_index(drop=True)


class ENTSOGFlows(Source):
    name = "entsog"
    verified = True  # endpoint exercised and parsed on 2026-10-12

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames, d = [], start
        while d <= end:  # one month per call keeps responses < ~30k rows
            stop = min(end, (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date())
            r = self.client.get(
                URL,
                params={
                    "from": d.isoformat(),
                    "to": stop.isoformat(),
                    "indicator": "Physical Flow",
                    "periodType": "day",
                    "timezone": "CET",
                    "limit": -1,
                },
            )
            if r.status_code == 404:
                # 404 means either "No result found" or, for windows older than about five
                # years, "archived on a rolling basis" (TP Archives only). Say which.
                msg = r.json().get("message", "") if r.content else ""
                if "archived" in msg:
                    print(f"entsog: {d}..{stop} not served by the API: {msg}", file=sys.stderr)
                d = stop + dt.timedelta(days=1)
                continue
            r.raise_for_status()
            self.keep_raw(r.content, "json")
            frames.append(to_canonical(r.json().get("operationaldatas", [])))
            d = stop + dt.timedelta(days=1)
        return pd.concat(frames, ignore_index=True) if frames else to_canonical([])


def parse_archive_value(s: str) -> float | None:
    """Value as written in the TP Archives yearly files.

    2016-2019 files write most values with the decimal point removed and a dot every
    three digits ("93.698.000.000.000"); values kept intact always have six decimals
    ("0.000000"). So a value with more than one dot is its digits / 10^6.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    if s.count(".") > 1:
        neg = s.startswith("-")
        return (-1 if neg else 1) * int(s.lstrip("-").replace(".", "")) / 1e6
    return float(s)


def api_keymap(payloads) -> dict[tuple, tuple[str, str]]:
    """Archive point -> (operatorKey, pointKey), built from API payloads.

    Two kinds of key: ``("eic", operator EIC, point EIC, direction)`` and, for archive
    rows with no point EIC ("?"), ``("label", operator EIC, point label, direction)``.
    """
    out = {}
    for p in payloads:
        for r in p.get("operationaldatas", []):
            if not (r.get("pointKey") and r.get("tsoEicCode")):
                continue
            keys = (r["operatorKey"], r["pointKey"])
            if r.get("tsoItemIdentifier"):
                out[("eic", r["tsoEicCode"], r["tsoItemIdentifier"], r["directionKey"])] = keys
            if r.get("pointLabel"):
                label = r["pointLabel"].strip().lower()
                out[("label", r["tsoEicCode"], label, r["directionKey"])] = keys
    return out


def _archive_id(keymap: dict, op: str, point_eic: str, label: str, direction: str) -> str:
    if point_eic.startswith("21"):
        if m := keymap.get(("eic", op, point_eic, direction)):
            return f"ENTSOG.{m[0]}.{m[1]}.{direction}"
        return f"ENTSOG.EIC.{op}.{point_eic}.{direction}"
    if m := keymap.get(("label", op, label.strip().lower(), direction)):
        return f"ENTSOG.{m[0]}.{m[1]}.{direction}"
    return f"ENTSOG.NAME.{op}.{label.strip()}.{direction}"


def archive_to_canonical(path, keymap: dict) -> pd.DataFrame:
    """A TP Archives ``AllTSOs_CalendarYear<YYYY>_PhysicalFlow.csv`` in canonical form.

    Series IDs match the API fetcher where the point maps to an API key (by EIC, else
    by label). Other points get ``ENTSOG.EIC.…`` or ``ENTSOG.NAME.…`` IDs. The gas day
    is the date of "Period From": the files use UTC, operator-local time or an explicit
    offset by year, and the gas day starts at 04:00-07:00 in all of them.

    The files also hold hourly and cumulative intraday rows; only 23-25 h periods are
    kept. Where one point has two daily rows with different values for the same gas
    day, the file does not say which is right, so the day is dropped and counted.
    """
    try:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", index_col=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="cp1252", index_col=False)
    df.columns = [c.strip(",").lower() for c in df.columns]
    df["value"] = df["value"].map(parse_archive_value)
    hours = (
        pd.to_datetime(df["period to"].str[:19]) - pd.to_datetime(df["period from"].str[:19])
    ).dt.total_seconds() / 3600
    df = df[df["value"].notna() & hours.between(23, 25)]
    df = df.fillna({"point eic": "?", "point name": ""})
    out = pd.DataFrame(
        {
            "series_id": [
                _archive_id(keymap, o, p, n, d)
                for o, p, n, d in df[
                    ["operator eic", "point eic", "point name", "direction"]
                ].itertuples(index=False)
            ],
            "observed_at": df["period from"].str[:10].to_numpy(),
            "value": df["value"].to_numpy(),
            "unit": df["unit"].to_numpy(),
            "meta": [
                json.dumps({"pointLabel": n, "operatorLabel": o, "flowStatus": "TP Archives"})
                for n, o in zip(df["point name"], df["operator name"], strict=True)
            ],
        }
    ).drop_duplicates(["series_id", "observed_at", "value"])
    conflict = out.duplicated(["series_id", "observed_at"], keep=False)
    if conflict.any():
        n = out[conflict][["series_id", "observed_at"]].drop_duplicates().shape[0]
        print(f"entsog archive {path}: dropped {n} conflicting point-days", file=sys.stderr)
    return out[~conflict].reset_index(drop=True)
