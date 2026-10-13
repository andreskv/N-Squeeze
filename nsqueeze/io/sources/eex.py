"""EEX primary-market auction results for EU allowances (EUA), 2012 → today.

Auction clearing prices are the building block of the regulatory CBAM price (the
quarterly, from 2027 weekly, average of auction prices), so a monthly
volume-weighted mean of them is the methodologically consistent carbon price here.

History comes as one zip of yearly workbooks (legacy .xls before 2020, .xlsx after);
the current year as a single .xlsx. Header rows move between years, so parsing
locates the row whose second cell is ``Date``. Only successful auctions of general
allowances are kept (aviation allowances, EUAA, are excluded).
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import pandas as pd

from nsqueeze.io.sources.base import Source

HISTORY_URL = (
    "https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/"
    "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
    "emission-spot-primary-market-auction-report-2012-2025-data.zip"
)
CURRENT_URL = (
    "https://public.eex-group.com/eex/eua-auction-report/"
    "emission-spot-primary-market-auction-report-{year}-data.xlsx"
)
# Header text varies by year ("€/tCO2" vs "EUR/tCO2", "Auction Volume" with or
# without units), so columns are matched by prefix.
PRICE_PREFIX, VOLUME_PREFIX = "Auction Price", "Auction Volume"


def parse_report(content: bytes, *, legacy: bool = False) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(content), header=None, engine="xlrd" if legacy else None)
    hdr = next(i for i in range(15) if "Date" in [str(v).strip() for v in raw.iloc[i].tolist()])
    df = raw.iloc[hdr + 1 :].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr]]
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()]
    if "Status" in df:
        df = df[df["Status"].astype(str).str.lower().str.startswith("success")]
    price_col = next(c for c in df.columns if c.startswith(PRICE_PREFIX))
    volume_col = next(c for c in df.columns if c.startswith(VOLUME_PREFIX))
    contract = df.get("Contract", pd.Series("", index=df.index)).astype(str)
    name = df.get("Auction Name", pd.Series("", index=df.index)).astype(str)
    aviation = contract.str.contains(
        "AA", regex=False
    )  # EAA2/EAA3 = aviation allowances | name.str.contains("aviation", case=False)
    df = df[~aviation]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"]).dt.normalize(),
            "price": pd.to_numeric(df[price_col], errors="coerce"),
            "volume": pd.to_numeric(df[volume_col], errors="coerce"),
            "auction": name.loc[df.index].to_numpy(),
        }
    ).dropna(subset=["price", "volume"])


def to_canonical(auctions: pd.DataFrame) -> pd.DataFrame:
    # Several auctions (EU, DE, PL) can clear on one day: aggregate to one row per day.
    day = (
        auctions.groupby("date")
        .apply(
            lambda g: pd.Series(
                {"price": (g.price * g.volume).sum() / g.volume.sum(), "volume": g.volume.sum()}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    price = pd.DataFrame(
        {
            "series_id": "EEX.EUA.auction_price",
            "observed_at": day.date,
            "value": day.price,
            "unit": "EUR/tCO2",
        }
    )
    vol = pd.DataFrame(
        {
            "series_id": "EEX.EUA.auction_volume",
            "observed_at": day.date,
            "value": day.volume,
            "unit": "tCO2",
        }
    )
    return pd.concat([price, vol], ignore_index=True)


class EUAAuctions(Source):
    name = "eua_auctions"
    verified = True  # downloaded and parsed 2026-10-12

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        parts = []
        if start.year <= 2025:
            r = self.client.get(HISTORY_URL)
            r.raise_for_status()
            self.keep_raw(r.content, "zip")
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for info in z.infolist():
                    if info.filename.endswith((".xls", ".xlsx")):
                        parts.append(
                            parse_report(z.read(info), legacy=info.filename.endswith(".xls"))
                        )
        for year in range(max(start.year, 2026), end.year + 1):
            r = self.client.get(CURRENT_URL.format(year=year))
            if r.status_code == 404:
                continue
            r.raise_for_status()
            self.keep_raw(r.content, "xlsx")
            parts.append(parse_report(r.content))
        auctions = pd.concat(parts, ignore_index=True).drop_duplicates()
        m = (auctions.date >= pd.Timestamp(start)) & (auctions.date <= pd.Timestamp(end))
        return to_canonical(auctions[m])
