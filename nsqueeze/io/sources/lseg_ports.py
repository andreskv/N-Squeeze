"""Port calls from a licensed LSEG Workspace export, one row per call (no network).

I export Port Search workbooks by hand in a licensed session. The raw values stay in
``data/``: the licence permits analysis, not redistribution. Port departures and arrivals
report draft in metres. Berth stays report their duration in hours.

The parser removes live-vessel fields and every event in 2026 or later. It also removes
next-location details with a 2026 arrival and previous-location details that postdate the
current arrival. This keeps the open files inside the 2019-2025 observation window.
At touching port polygons of one site, a previous departure can postdate arrival, so I blank
those previous-location fields.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nsqueeze.config import data_dir
from nsqueeze.io.pit import PITStore
from nsqueeze.io.sources.base import Source
from nsqueeze.io.sources.lseg import RETRIEVED_AT, SESSION

CUTOFF = pd.Timestamp("2026-01-01")
MASKED = ("Last Seen", "Destination (current)", "ETA (GMT)")
NEXT_ARRIVAL = "Next Location Arrival Date"
PREVIOUS_DEPARTURE = "Previous Location Departure Date (GMT)"
BASE_META = (
    "Vessel",
    "Type",
    "Sub-type",
    "Sub-sub-type",
    "IMO",
    "DWT",
    "Max Draft (m)",
    "Clean / Dirty (tankers only)",
)
NEXT_META = (
    "Next Country/Region",
    "Next Location",
    "Next Location Draft (m) (Entering port)",
    "Next Location Draft (m) (Exiting port)",
    "Berth Type.1",
)
PREVIOUS_META = (
    "Previous Country/Region",
    "Previous Location",
    PREVIOUS_DEPARTURE,
    "Previous Location Draft (m) (Entering Location)",
    "Previous Location Draft (m) (Exiting Location)",
    "Berth Type.1",
)
META_FIELDS = {
    "dep": (
        *BASE_META,
        "Country/Region",
        "Polygon Type",
        "Polygon Name",
        "Berth Type",
        *NEXT_META,
        "Cubic Capacity (m3)",
    ),
    "arr": (
        *BASE_META,
        "Country/Region",
        "Port",
        "Berth Type",
        *PREVIOUS_META,
        "Cubic Capacity (m3)",
    ),
    "stay": (*BASE_META, "Departure Date (GMT)", "Length (m)", "Beam (m)"),
}


def read_port(path: Path) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Read one workbook, mask it, and return its canonical rows and row counts."""
    raw = pd.read_excel(path, header=None, dtype=object)
    header_at = raw.index[raw.iloc[:, 0].eq("Vessel")].tolist()
    if len(header_at) != 1:
        raise ValueError(f"lseg_portcalls: {path.name} must have one Vessel header")
    header_at = header_at[0]
    names: list[str] = []
    seen: dict[str, int] = {}
    for value in raw.iloc[header_at]:
        name = str(value)
        names.append(name if name not in seen else f"{name}.{seen[name]}")
        seen[name] = seen.get(name, 0) + 1
    data = raw.iloc[header_at + 1 :].copy()
    data.columns = names
    data = data.drop(columns=[c for c in MASKED if c in data])
    data = data.dropna(how="all")
    # Footer lines describe the export. Only rows with an IMO number are port calls.
    data = data.loc[pd.to_numeric(data["IMO"], errors="coerce").notna()].copy()

    kind = "dep" if "Polygon Name" in data else "arr" if "Port" in data else "stay"
    event_column = "Departure Date (GMT)" if kind == "dep" else "Arrival Date (GMT) (Entering port)"
    required = {event_column, "IMO", *META_FIELDS[kind]}
    if missing := sorted(required - set(data.columns)):
        raise ValueError(f"lseg_portcalls: {path.name} lacks columns {missing}")

    rows_read = len(data)
    next_mask = pd.Series(False, index=data.index)
    previous_mask = pd.Series(False, index=data.index)
    if kind == "dep":
        next_mask = _dates(data[NEXT_ARRIVAL]).ge(CUTOFF)
        data.loc[next_mask, list(NEXT_META)] = pd.NA
        data = data.drop(columns=NEXT_ARRIVAL)
    elif kind == "arr":
        previous_dates = _dates(data[PREVIOUS_DEPARTURE])
        event_dates = _dates(data[event_column])
        previous_mask = previous_dates.ge(CUTOFF) | previous_dates.gt(event_dates)
        data.loc[previous_mask, list(PREVIOUS_META)] = pd.NA

    event_dates = _dates(data[event_column])
    if event_dates.isna().any():
        raise ValueError(f"lseg_portcalls: {path.name} has a missing or invalid event time")
    departures = _dates(data["Departure Date (GMT)"]) if kind == "stay" else None
    drop_2026 = event_dates.ge(CUTOFF)
    if departures is not None:
        drop_2026 |= departures.ge(CUTOFF)
    no_place = pd.Series(False, index=data.index)
    if kind in {"dep", "arr"}:
        place_column = "Polygon Name" if kind == "dep" else "Port"
        no_place = data[place_column].astype(str).str.strip().eq("-")
    keep = ~drop_2026 & ~no_place

    meta = data.loc[keep, list(META_FIELDS[kind])].copy()
    renames = {column: _snake(column) for column in meta}
    place_column = "Polygon Name" if kind == "dep" else "Port" if kind == "arr" else None
    if place_column:
        renames[place_column] = "place"
    if "Berth Type.1" in renames:
        renames["Berth Type.1"] = "next_berth_type" if kind == "dep" else "previous_berth_type"
    meta = meta.rename(columns=renames)
    if kind == "stay":
        meta["place"] = _filter_place(raw, header_at)
        meta["departure_date_gmt"] = departures.loc[keep]
        duration = (departures - event_dates).dt.total_seconds() / 3600
        values = duration.loc[keep]
        unit = "h"
    else:
        if kind == "arr" and PREVIOUS_DEPARTURE in data:
            previous_dates = _dates(data[PREVIOUS_DEPARTURE])
            meta["previous_location_departure_date_gmt"] = previous_dates.loc[keep]
        draft_column = "Draft (m)" if kind == "dep" else "Draft (m) (Entering port)"
        values = pd.to_numeric(data.loc[keep, draft_column], errors="coerce")
        unit = "m"
    meta["file"] = path.stem

    imos = pd.to_numeric(data.loc[keep, "IMO"], errors="raise").astype("int64").astype(str)
    places = meta["place"].map(str).map(_snake)
    if places.eq("").any():
        raise ValueError(f"lseg_portcalls: {path.name} has an empty place slug")
    frame = pd.DataFrame(
        {
            "series_id": "IMO" + imos + "." + kind + "." + places,
            "observed_at": event_dates.loc[keep],
            "value": values.astype("float64"),
            "unit": unit,
            "meta": [_json(record) for record in meta.to_dict("records")],
        }
    ).reset_index(drop=True)
    counts: dict[str, int | str] = {
        "file": path.stem,
        "kind": kind,
        "rows_read": rows_read,
        "rows_after_masking": len(frame),
        "dropped_2026": int(drop_2026.sum()),
        "dropped_no_place": int(no_place.sum()),
        "next_blanked": int(next_mask.sum()),
        "previous_blanked": int(previous_mask.sum()),
    }
    return frame, counts


def read_session(folder: Path) -> pd.DataFrame:
    """Read files in filename order and keep the first row of each duplicate call."""
    paths = sorted(folder.glob("*.xlsx"))
    if not paths:
        raise ValueError(f"lseg_portcalls: no workbooks in {folder}")
    frames = [read_port(path)[0] for path in paths]
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["series_id", "observed_at"], keep="first")
        .reset_index(drop=True)
    )


class LSEGPortCalls(Source):
    name = "lseg_portcalls"
    verified = True  # files and licence terms checked by the author on 2026-11-16

    def __init__(self, folder: Path | None = None) -> None:
        super().__init__()
        self.folder = folder or data_dir() / "inbox" / "lseg" / SESSION / "ports"

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        return read_session(self.folder)

    def archive(self, store: PITStore, start: dt.date, end: dt.date) -> int:
        # The vintage is the session, not the day of the ingest.
        frame = self.fetch(start, end)
        store.write(self.name, frame, retrieved_at=RETRIEVED_AT)
        return len(frame)


def _dates(values: pd.Series) -> pd.Series:
    """The workbooks write dates as day/month/year text or Excel datetimes, all GMT."""
    return pd.to_datetime(values, format="mixed", dayfirst=True, errors="coerce")


def _filter_place(raw: pd.DataFrame, header_at: int) -> str:
    for value in raw.iloc[:header_at, 0].dropna():
        match = re.search(r"Port, Anchorage, Berth or Location:\s*\[([^]]+)\]", str(value))
        if match:
            return match.group(1).strip()
    raise ValueError("lseg_portcalls: berth file has no place in its filter line")


def _snake(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _json(record: dict) -> str:
    fields = {
        key: value
        for key, value in record.items()
        if value is not None
        and (not isinstance(value, str) or value.strip())
        and not (not isinstance(value, str) and pd.isna(value))
    }
    return json.dumps(fields, ensure_ascii=False, default=_json_default)


def _json_default(value):
    if isinstance(value, pd.Timestamp | dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")
