"""EU ETS verified emissions and free allocation per stationary installation, 2008 on (no key).

The Commission publishes one compliance file a year on its Union Registry page,
``verified_emissions_<year>_en.xlsx``. The latest file holds every year since 2008, one
row per account. Checked on 2026-10-14: the 2025 file (extracted 2026-04-01) has 18,993
accounts, of which 13,490 are stationary installations. Main activity code 41 is
ammonia production, 38 nitric acid; integrated chemical sites often report under
another code, so the archive keeps every installation and plant matching happens later.

"Excluded" (out of scope that year) and blank cells are not emissions and are dropped.
Values are tonnes CO2-equivalent. Free allocation (``ALLOCATION_<year>``, in allowances) is kept as
its own series: since 2021 it is adjusted when an installation's activity level changes, so it
also carries information on output.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re

import numpy as np
import pandas as pd

from nsqueeze.io.sources.base import Source

BASE = "https://climate.ec.europa.eu"
PAGE = BASE + "/eu-action/carbon-markets/eu-emissions-trading-system-eu-ets/union-registry_en"
LINK = re.compile(r'href="(/document/download/[^"]+filename=verified_emissions_(\d{4})_en\.xlsx)"')
COLUMNS = ["series_id", "observed_at", "value", "unit", "meta"]


def latest_link(html: str) -> tuple[str, int]:
    """The newest ``verified_emissions_<year>_en.xlsx`` link on the registry page."""
    links = [(m.group(1), int(m.group(2))) for m in LINK.finditer(html)]
    if not links:
        raise ValueError("no verified_emissions_<year>_en.xlsx link on the Union Registry page")
    return max(links, key=lambda x: x[1])


def to_canonical(xlsx: bytes) -> pd.DataFrame:
    """Stationary installations × years → canonical rows: verified emissions ``ETS.<registry>.<id>``
    and free allocation ``ETSALLOC.<registry>.<id>``."""
    readme = pd.read_excel(io.BytesIO(xlsx), "Read Me", header=None, nrows=1)
    extracted = str(pd.Timestamp(readme.iloc[0, 1]).date())
    d = pd.read_excel(io.BytesIO(xlsx), "data", header=2)
    d = d[d.ACCOUNT_TYPE == "OPERATOR HOLDING ACCOUNT"]
    years = [c for c in d.columns if re.fullmatch(r"(VERIFIED_EMISSIONS|ALLOCATION)_\d{4}", str(c))]
    long = d.melt(
        id_vars=["REGISTRY_CODE", "INSTALLATION_IDENTIFIER", "INSTALLATION_NAME",
                 "MAIN_ACTIVITY_TYPE_CODE", "PERMIT_IDENTIFIER"],
        value_vars=years, var_name="col", value_name="value",
    )  # fmt: skip
    long["value"] = pd.to_numeric(long.value, errors="coerce")  # "Excluded" -> NaN
    long = long.dropna(subset=["value"])
    meta = {
        (r, i): json.dumps(
            {"name": n, "activity": int(a), "permit": p, "extracted": extracted},
            ensure_ascii=False,
        )
        for r, i, n, a, p in d[
            ["REGISTRY_CODE", "INSTALLATION_IDENTIFIER", "INSTALLATION_NAME",
             "MAIN_ACTIVITY_TYPE_CODE", "PERMIT_IDENTIFIER"]
        ].itertuples(index=False)
    }  # fmt: skip
    keys = list(zip(long.REGISTRY_CODE, long.INSTALLATION_IDENTIFIER, strict=True))
    return pd.DataFrame(
        {
            "series_id": [
                f"{'ETSALLOC' if c.startswith('ALLOC') else 'ETS'}.{r}.{i}"
                for (r, i), c in zip(keys, long.col, strict=True)
            ],
            "observed_at": pd.to_datetime(long.col.str[-4:] + "-01-01").to_numpy(),
            "value": long.value.to_numpy(),
            "unit": np.where(long.col.str.startswith("ALLOC"), "EUA", "tCO2e"),
            "meta": [meta[k] for k in keys],
        },
        columns=COLUMNS,
    )


class ETSVerifiedEmissions(Source):
    name = "ets_registry"
    verified = True  # page, link and file layout checked on 2026-10-14

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        page = self.client.get(PAGE)
        page.raise_for_status()
        href, _ = latest_link(page.text)
        r = self.client.get(BASE + href)
        r.raise_for_status()
        self.keep_raw(r.content, "xlsx")
        df = to_canonical(r.content)
        year = df.observed_at.dt.year
        return df[(year >= start.year) & (year <= end.year)].reset_index(drop=True)
