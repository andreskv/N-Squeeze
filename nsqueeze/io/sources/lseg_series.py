"""Daily time series from licensed LSEG Workspace files, without network access.

I read each file name as a RIC and melt its selected LSEG fields into canonical rows. Raw
values stay in ``data/``: the licence permits analysis, not redistribution. The parser
first drops every observation after 31 December 2025: I decide later how to use the 2026
rows. The files give no unit, so ``ROOTS`` gives one per instrument root, with the check
behind it. A unit that I could not check is "unverified".
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from nsqueeze.config import data_dir
from nsqueeze.io.pit import PITStore
from nsqueeze.io.sources.base import Source
from nsqueeze.io.sources.lseg import RETRIEVED_AT, SESSION

CUTOFF = pd.Timestamp("2026-01-01")
ZIP_NAME = "06_lseg_export_final_gas_freight_iv_equity.zip"
FIELDS = (
    "SETTLE",
    "TRDPRC_1",
    "OFF_CLOSE",
    "FRGT_RATE1",
    "MID_PRICE",
    "FAIR_VALUE",
    "ACVOL_UNS",
    "OPINT_1",
)
PRICE_FIELDS = frozenset(FIELDS[:6])
EXCLUDED_ZIP_FILES = {"x_EMAc1_last20days.csv", "x_EMAc2_last20days.csv"}
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]

# The unit of each root, and how I checked it. "Archive" means that the series matches a
# series of another source in the point-in-time store, day by day, at a ratio near 1.
# "TTF" means a ratio near 1 to the LSEG TTF front month, which matches the archive.
# CME fertilizer units: CBOT rulebook chapters 42 and 44, the 2017 listing notice SER-7922 and
# the CBOT fertilizer product sheet (https://www.cmegroup.com/trading/agricultural/files/
# cbot-fertilizer-product.pdf). SGX ammonia: Argus report of the SGX launch, Oct 2023
# (https://www.argusmedia.com/en/news/2494842-singapore-exchange-starts-ammonia-derivative-contracts).
ROOTS: dict[str, tuple[str, str]] = {
    # Archive: equal to the Yahoo continuous contract (US cents per bushel).
    "C": ("CBOT corn", "USD/bu cents"),
    "S": ("CBOT soybeans", "USD/bu cents"),
    "W": ("CBOT wheat", "USD/bu cents"),
    # CME documents: 100 metric tons per contract.
    "UFE": ("CBOT urea FOB Egypt", "USD/t"),
    "UME": ("CBOT urea FOB Middle East", "USD/t"),
    "UFB": ("CBOT urea CFR Brazil", "USD/t"),
    # CME product sheet: 100 short tons per contract.
    "UFV": ("CBOT urea FOB US Gulf", "USD/short ton"),
    "NOU": ("CBOT UAN FOB NOLA", "USD/short ton"),
    "DNQ": ("CBOT DAP FOB NOLA", "USD/short ton"),
    # Argus report of the SGX launch: lots of 100 t, priced per tonne.
    "SAMMF": ("SGX Argus ammonia FOB Middle East", "USD/t"),
    # Archive: ratio 1.004 to the EEX EUA auction price.
    "CFI2": ("ICE EUA futures", "EUR/t CO2"),
    # Archive: equal to the Yahoo TTF front month.
    "TFMBM": ("ICE TTF monthly gas", "EUR/MWh"),
    # TTF (the other gas roots below).
    "TFMBY": ("ICE TTF yearly gas", "EUR/MWh"),
    "TRNLTTFFVM": ("TTF fair-value monthly gas", "EUR/MWh"),
    "TRNLTTFFVY": ("TTF fair-value yearly gas", "EUR/MWh"),
    "TTFDA": ("TTF day-ahead gas", "EUR/MWh"),
    "PNX-TTFEOD-DA": ("TTF day-ahead, end of day", "EUR/MWh"),
    "PNX-TTFEGSI-DA": ("TTF day-ahead, EGSI", "EUR/MWh"),
    "TRDENCGY": ("THE yearly gas", "EUR/MWh"),
    "PNCGM": ("THE month gas", "EUR/MWh"),
    "PNCGQ": ("THE quarter gas", "EUR/MWh"),
    "PNCGS": ("THE season gas", "EUR/MWh"),
    "PNCGY": ("THE year gas", "EUR/MWh"),
    "NGMM": ("THE monthly gas (ICE)", "EUR/MWh"),
    "ICPPEGNQ": ("PEG quarterly gas", "EUR/MWh"),
    "ICPPEGNY": ("PEG yearly gas", "EUR/MWh"),
    "PPEGM": ("PEG month gas", "EUR/MWh"),
    "PPEGQ": ("PEG quarter gas", "EUR/MWh"),
    "PPEGS": ("PEG season gas", "EUR/MWh"),
    "PPEGY": ("PEG year gas", "EUR/MWh"),
    "TRITPSVM": ("PSV monthly gas", "EUR/MWh"),
    "TRITPSVQ": ("PSV quarterly gas", "EUR/MWh"),
    "PSVM": ("PSV month gas", "EUR/MWh"),
    "PSVQ": ("PSV quarter gas", "EUR/MWh"),
    "PSVS": ("PSV season gas", "EUR/MWh"),
    "PSVY": ("PSV year gas", "EUR/MWh"),
    "OMIPFGESM": ("PVB monthly gas", "EUR/MWh"),
    "OMIPFGESQ": ("PVB quarterly gas", "EUR/MWh"),
    "OMIPFGESY": ("PVB yearly gas", "EUR/MWh"),
    "VTPM": ("CEGH VTP month gas", "EUR/MWh"),
    "VTPQ": ("CEGH VTP quarter gas", "EUR/MWh"),
    "VTPS": ("CEGH VTP season gas", "EUR/MWh"),
    "VTPY": ("CEGH VTP year gas", "EUR/MWh"),
    # TTF: ratio 2.53, which fits pence per therm (1 therm = 0.0293 MWh).
    "TRGBNBPM": ("NBP monthly gas", "GBp/therm"),
    # Spreads between two prices: the unit of a difference is not stated, so not checked.
    "TRNCGTTFFVD": ("THE-TTF day-ahead spread", "unverified"),
    "TRPSVTTFFVD": ("PSV-TTF day-ahead spread", "unverified"),
    "TRNBPTTFQ": ("NBP-TTF quarterly spread", "unverified"),
    "TRNBPTTFS": ("NBP-TTF seasonal spread", "unverified"),
    "TRNBPGSPY": ("NBP-GASPOOL yearly spread", "unverified"),
    "TRTFUTTFFVD": ("TTF-US day-ahead spread", "unverified"),
    "TRGBCSBM": ("GB monthly clean spark spread", "unverified"),
    "TRGBCSBQ": ("GB quarterly clean spark spread", "unverified"),
    # Not checked: no other source in the archive.
    "BL2": ("Euronext milling wheat", "unverified"),
    "COM": ("Euronext rapeseed", "unverified"),
    ".BLPG": ("Baltic LPG route BLPG1 (AG East)", "unverified"),
    ".BLPG2": ("Baltic LPG route BLPG2", "unverified"),
    ".BLPG3": ("Baltic LPG route BLPG3", "unverified"),
    ".BALPG": ("Baltic LPG index", "unverified"),
    "GFIBALBLPGY": ("BLPG1 yearly FFA (GFI)", "unverified"),
    ".BOPI": ("Baltic LPG operating cost index", "unverified"),
    "GBSLPGFHCUSGJPN": ("LPG freight Houston-Chiba (Gibson)", "unverified"),
    "GBSLPGFRTCMEGJPN": ("LPG freight Ras Tanura-Chiba (Gibson)", "unverified"),
    "GBSLPGFHFUSGNLD": ("LPG freight Houston-Flushing (Gibson)", "unverified"),
    "GBSFTCVLGC1Y": ("VLGC one-year time charter (Gibson)", "unverified"),
    "GBSLPGFCHNPHL": ("LPG freight South China-Philippines (Gibson)", "unverified"),
    # Unadjusted close in the listing currency. Archive: matches the Yahoo dividend-adjusted
    # close at a ratio that is near 1 in 2025 and grows back in time (YAR.OL, CF.N).
    "YAR.OL": ("Yara International", "NOK"),
    "CF.N": ("CF Industries", "USD"),
    # Listing currency of the exchange (Euronext Amsterdam, Toronto, NYSE, Xetra). Not checked.
    "OCI.AS": ("OCI", "EUR"),
    "NTR.TO": ("Nutrien", "CAD"),
    "MOS.N": ("Mosaic", "USD"),
    "SDFGn.DE": ("K+S", "EUR"),
    "BASFn.DE": ("BASF", "EUR"),
    # Implied volatility surface: SETTLE holds the volatility in percent (median about 23 for
    # corn and 17 for soybeans). The other fields hold other things (TRDPRC_1 the underlying
    # price, ACVOL_UNS the expiry date as a number), so only SETTLE is kept (FIELDS_BY_ROOT).
    "CATMIVM": ("CBOT corn ATM implied volatility", "%"),
    "SATMIVM": ("CBOT soybean ATM implied volatility", "%"),
}
FIELDS_BY_ROOT = {"CATMIVM": ("SETTLE",), "SATMIVM": ("SETTLE",)}
ROOT_ORDER = sorted(ROOTS, key=len, reverse=True)


def filename_to_ric(name: str) -> str:
    """Strip the export prefix and CSV suffix, then restore an expired-contract caret."""
    stem = re.sub(r"^(?:extra|x)_", "", Path(name).name)
    if not stem.lower().endswith(".csv"):
        raise ValueError(f"lseg_series: {name} is not a CSV file")
    return re.sub(r"_(\d)$", r"^\1", stem[:-4])


def root_of(ric: str) -> str:
    """Return the ROOTS key before continuation, delivery month, year and expiry suffix."""
    for root in ROOT_ORDER:
        if ric == root or re.fullmatch(
            rf"{re.escape(root)}(?:c\d+|[FGHJKMNQUVXZ]\d{{0,2}}(?:\^\d+)?)", ric
        ):
            return root
    raise ValueError(f"lseg_series: no ROOTS entry covers {ric}")


def to_canonical(raw: pd.DataFrame, ric: str, filename: str) -> pd.DataFrame:
    """Melt selected fields after removing all observations in 2026 or later."""
    dates = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise")
    keep = dates.lt(CUTOFF)
    dates = dates.loc[keep].reset_index(drop=True)
    data = raw.loc[keep].reset_index(drop=True)
    root = root_of(ric)
    short_name, price_unit = ROOTS[root]
    frames = []
    for field in FIELDS_BY_ROOT.get(root, FIELDS):
        if field not in data:
            continue
        values = pd.to_numeric(data[field], errors="coerce")
        present = values.notna()
        meta = {"file": filename}
        if field in PRICE_FIELDS:
            meta["name"] = short_name
        frames.append(
            pd.DataFrame(
                {
                    "series_id": f"{ric}.{field}",
                    "observed_at": dates.loc[present],
                    "value": values.loc[present].astype("float64"),
                    "unit": "contracts" if field not in PRICE_FIELDS else price_unit,
                    "meta": json.dumps(meta, ensure_ascii=False),
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    frame.attrs["dropped_2026"] = int((~keep).sum())
    return frame


def read_session(folder: Path) -> pd.DataFrame:
    """Read the selected ZIP members and loose CSVs and reject duplicate RICs."""
    loose = folder / "lseg_export_folder"
    paths = sorted(
        path
        for path in loose.glob("*.csv")
        if path.name not in {"grain_report.csv", "search_results.csv"}
    )
    with zipfile.ZipFile(folder / ZIP_NAME) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if re.match(r"^(?:extra|x)_.+\.csv$", Path(name).name)
            and Path(name).name not in EXCLUDED_ZIP_FILES
        )
        records = [
            (filename_to_ric(Path(name).name), Path(name).name, name, True) for name in members
        ] + [(filename_to_ric(path.name), path.name, path, False) for path in paths]
        by_ric: dict[str, list[str]] = {}
        for ric, filename, _, _ in records:
            by_ric.setdefault(ric, []).append(filename)
        repeated = {ric: names for ric, names in by_ric.items() if len(names) > 1}
        if repeated:
            first_ric, names = next(iter(repeated.items()))
            raise ValueError(f"lseg_series: {first_ric} comes from duplicate files {names}")
        frames = []
        for ric, filename, location, in_zip in records:
            content = archive.read(location) if in_zip else location.read_bytes()
            raw = pd.read_csv(io.BytesIO(content))
            if "Date" not in raw:
                raise ValueError(f"lseg_series: {filename} has no Date column")
            frames.append(to_canonical(raw, ric, filename))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    frame.attrs.update(
        files_read=[record[1] for record in records],
        rics=[record[0] for record in records],
        dropped_2026=sum(part.attrs["dropped_2026"] for part in frames),
    )
    return frame


class LSEGSeries(Source):
    name = "lseg_series"
    verified = True  # files and licence terms checked by the author on 2026-11-16

    def __init__(self, folder: Path | None = None) -> None:
        super().__init__()
        self.folder = folder or data_dir() / "inbox" / "lseg" / SESSION / "codebook"

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        return read_session(self.folder)

    def archive(self, store: PITStore, start: dt.date, end: dt.date) -> int:
        # The vintage is the session, not the day of the ingest.
        frame = self.fetch(start, end)
        store.write(self.name, frame, retrieved_at=RETRIEVED_AT)
        return len(frame)
