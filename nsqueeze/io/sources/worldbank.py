"""World Bank Commodity Price Data ("Pink Sheet"), monthly, nominal US dollars.

Official monthly anchors: Henry Hub, European gas, Japanese LNG, urea, DAP.

Series definitions change over time and the workbook's own footnotes say so; the
breaks that matter here are recorded as ``BREAKS`` and carried into ``meta``:

* Natural gas, Europe: Netherlands TTF only **from April 2015**; before that an
  import-border-price blend. Analyses therefore start in 2015-04.
* Urea: f.o.b. **Black Sea until February 2022, Middle East from March 2022**.
  A change of origin mid-crisis, which import parity must model explicitly.

The download URL embeds a yearly document id, so it lives in ``config/sources.yaml``
(``pink_sheet_monthly.url``). A stale mirror at worldbank.org still serves a 2016
file; ``fetch`` rejects any workbook whose "Updated on" stamp is older than a year.
"""

from __future__ import annotations

import datetime as dt
import io
import json

import pandas as pd
import yaml

from nsqueeze.config import config_dir
from nsqueeze.io.sources.base import Source

SERIES = {  # column header in the workbook → (series_id, unit)
    "Natural gas, US": ("WB.NGAS_US", "USD/MMBtu"),
    "Natural gas, Europe": ("WB.NGAS_EUR", "USD/MMBtu"),
    "Liquefied natural gas, Japan": ("WB.LNG_JP", "USD/MMBtu"),
    "Urea": ("WB.UREA", "USD/t"),
    "DAP": ("WB.DAP", "USD/t"),
}
BREAKS = {
    "WB.NGAS_EUR": [("2015-04-01", "TTF from April 2015; earlier import-border blend")],
    "WB.UREA": [("2022-03-01", "f.o.b. Middle East from March 2022; earlier f.o.b. Black Sea")],
}


def _url() -> str:
    reg = yaml.safe_load((config_dir() / "sources.yaml").read_text(encoding="utf-8"))
    return reg["pink_sheet_monthly"]["url"]


def parse_workbook(content: bytes) -> tuple[pd.DataFrame, dt.date]:
    """Canonical long frame plus the workbook's own "Updated on" date."""
    raw = pd.read_excel(io.BytesIO(content), sheet_name="Monthly Prices", header=None)
    stamp = next(
        (str(v) for v in raw.iloc[:10, 0] if isinstance(v, str) and v.startswith("Updated on")),
        None,
    )
    if stamp is None:
        raise ValueError("Pink Sheet: no 'Updated on' stamp found")
    updated = dt.datetime.strptime(stamp.replace("Updated on", "").strip(), "%B %d, %Y").date()

    hdr_row = next(i for i in range(15) if str(raw.iloc[i, 1]).startswith("Crude oil"))
    headers = [str(h).strip() if pd.notna(h) else "" for h in raw.iloc[hdr_row]]
    units = raw.iloc[hdr_row + 1]
    body = raw.iloc[hdr_row + 2 :]
    months = body.iloc[:, 0].astype(str)
    keep = months.str.fullmatch(r"\d{4}M\d{2}")
    body, months = body[keep], months[keep]
    observed = pd.to_datetime(months.str.replace("M", "-") + "-01")

    frames = []
    for name, (sid, unit) in SERIES.items():
        col = next((j for j, h in enumerate(headers) if h == name), None)
        if col is None:
            raise ValueError(f"Pink Sheet: column {name!r} not found")
        if unit.replace("USD", "$").replace("/t", "/mt").lower() not in str(units[col]).lower():
            raise ValueError(f"Pink Sheet: unit for {name!r} changed to {units[col]!r}")
        vals = pd.to_numeric(body.iloc[:, col], errors="coerce")
        frames.append(
            pd.DataFrame(
                {
                    "series_id": sid,
                    "observed_at": observed.to_numpy(),
                    "value": vals.to_numpy(),
                    "unit": unit,
                    "meta": json.dumps({"updated": updated.isoformat(), "breaks": BREAKS.get(sid)}),
                }
            ).dropna(subset=["value"])
        )
    return pd.concat(frames, ignore_index=True), updated


class PinkSheetMonthly(Source):
    name = "pink_sheet_monthly"
    verified = True  # downloaded and parsed 2026-10-12 (workbook updated 2026-10-02)

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        r = self.client.get(_url())
        r.raise_for_status()
        self.keep_raw(r.content, "xlsx")
        frame, updated = parse_workbook(r.content)
        if (dt.date.today() - updated).days > 365:
            raise ValueError(f"Pink Sheet workbook is stale (updated {updated}); fix the URL")
        m = (frame.observed_at >= pd.Timestamp(start)) & (frame.observed_at <= pd.Timestamp(end))
        return frame[m].reset_index(drop=True)
