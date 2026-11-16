"""Ship registry data from an LSEG Workspace export, one row per IMO number (no network).

LSEG Workspace has no public API, so I export the data by hand in a licensed session and
put the files in ``data/inbox/lseg/<session date>/``. The raw values stay in ``data/``:
the licence allows analysis, not redistribution.

Two files of the 16 Nov 2026 session cover the same 2,846 IMO numbers.
``ship_details.csv`` (in the last CodeBook zip) gives the vessel type, the size class, the
deadweight, the cargo volume, the flag and the dimensions. ``ship_details2.csv`` gives the
build date, the status, the owner, the manager and the hull.

value = the cargo volume in m³ (blank for 19 ships), the quantity a tank-fill check
needs. Every other field goes in ``meta``. The attributes are as LSEG shows them on the
session date, so a ship that changed type since an earlier year keeps today's type.
``ex_name`` helps to find such ships.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from nsqueeze.config import data_dir
from nsqueeze.io.pit import PITStore
from nsqueeze.io.sources.base import Source

SESSION = "2026-11-16"
# End of the export: ship_details2.csv was saved at 15:14 CET.
RETRIEVED_AT = dt.datetime(2026, 11, 16, 14, 14, tzinfo=dt.UTC)
DETAILS_ZIP = "codebook/06_lseg_export_final_gas_freight_iv_equity.zip"
DETAILS2 = "codebook/ship_details2.csv"

FIELDS = {  # LSEG column → meta key
    "Name": "name",
    "Instrument": "ric",
    "Sub-Type": "sub_type",
    "Sub-Sub-Type": "size_class",
    "Dead weight tonnage": "dwt_t",
    "Flag": "flag",
    "Overall Length": "length_m",
    "Beam": "beam_m",
    "Ex Name": "ex_name",
    "Status": "status",
    "Built": "built",
    "Registered Owner": "owner",
    "Manager": "manager",
    "Hull Type": "hull",
    "Coated": "coated",
    "Class": "class",
    "ICE Class": "ice_class",
    "Built Demolition": "demolished",
}


def to_canonical(details: pd.DataFrame, details2: pd.DataFrame, session: str) -> pd.DataFrame:
    """Join the two ship files on IMO. One canonical row per ship."""
    for label, df in (("ship_details", details), ("ship_details2", details2)):
        if df["IMO"].duplicated().any():
            raise ValueError(f"lseg_vessels: {label} has a repeated IMO number")
    extra = details2.drop(columns=["Instrument", "Name"])
    ships = details.merge(extra, on="IMO", how="outer", validate="one_to_one", indicator=True)
    if (ships["_merge"] != "both").any():
        n = int((ships["_merge"] != "both").sum())
        raise ValueError(f"lseg_vessels: {n} IMO numbers are in one ship file only")
    ships = ships.sort_values("IMO")
    meta = ships[list(FIELDS)].rename(columns=FIELDS)
    if "built" in meta:
        meta["built"] = meta["built"].str[:10]
    return pd.DataFrame(
        {
            "series_id": "IMO" + ships["IMO"].astype(str),
            "observed_at": pd.Timestamp(session),
            "value": ships["Cubic Capacity"].astype("float64"),
            "unit": "m3",
            "meta": [
                json.dumps({k: v for k, v in rec.items() if pd.notna(v)}, ensure_ascii=False)
                for rec in meta.to_dict("records")
            ],
        }
    ).reset_index(drop=True)


def read_session(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(folder / DETAILS_ZIP) as z:
        details = pd.read_csv(io.BytesIO(z.read("ship_details.csv")))
    return details, pd.read_csv(folder / DETAILS2)


class LSEGVessels(Source):
    name = "lseg_vessels"
    verified = True  # files and licence terms checked by the author on 2026-11-16

    def __init__(self, folder: Path | None = None) -> None:
        super().__init__()
        self.folder = folder or data_dir() / "inbox" / "lseg" / SESSION

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        return to_canonical(*read_session(self.folder), SESSION)

    def archive(self, store: PITStore, start: dt.date, end: dt.date) -> int:
        # The vintage is the session, not the day of the ingest.
        frame = self.fetch(start, end)
        store.write(self.name, frame, retrieved_at=RETRIEVED_AT)
        return len(frame)
