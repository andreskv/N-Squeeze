"""GIE AGSI+ (storage) and ALSI+ (LNG terminals). Needs AGSI_API_KEY (free registration).

AGSI+ data are revised after publication; only a PIT archive keeps the vintage that
was public on a given morning, which is what a nowcast must be scored against.
"""

from __future__ import annotations

import datetime as dt
import json
import time

import pandas as pd

from nsqueeze.io.sources.base import Source

AGSI_URL = "https://agsi.gie.eu/api"
ALSI_URL = "https://alsi.gie.eu/api"

# Fields kept per gas day. Units as published by GIE (TWh for energy, % for fill).
AGSI_FIELDS = {
    "gasInStorage": "TWh",
    "workingGasVolume": "TWh",
    "injection": "GWh/d",
    "withdrawal": "GWh/d",
    "full": "%",
    "trend": "pp",
}
EU_COUNTRIES = (
    "AT",
    "BE",
    "BG",
    "HR",
    "CZ",
    "DK",
    "FR",
    "DE",
    "HU",
    "IE",
    "IT",
    "LV",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "ES",
    "SE",
)
EU27 = tuple(
    [
        "AT",
        "BE",
        "BG",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "ES",
        "FI",
        "FR",
        "GR",
        "HR",
        "HU",
        "IE",
        "IT",
        "LT",
        "LU",
        "LV",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SE",
        "SI",
        "SK",
    ]
)


# ALSI+ fields, from the GIE API manual v013 (March 2025), §2.4. Since January 2024
# ``inventory`` and ``dtmi`` are objects {lng: 10³ m³ LNG, gwh: GWh}; flattened here.
ALSI_FIELDS = {
    "inventory.lng": "1e3 m3 LNG",
    "inventory.gwh": "GWh",
    "sendOut": "GWh/d",
    "dtmi.lng": "1e3 m3 LNG",
    "dtmi.gwh": "GWh",
    "dtrs": "GWh/d",
}
# GIE rate limit (manual §7.4): > 60 calls/min → 60 s lock-out, repeated abuse → ban.
MIN_SECONDS_BETWEEN_CALLS = 1.1


def _field(row: dict, path: str):
    v = row
    for k in path.split("."):
        v = v.get(k) if isinstance(v, dict) else None
    return None if v in (None, "", "-") else v


def rows_to_canonical(rows: list[dict], prefix: str, scope: str, fields: dict) -> list[dict]:
    """Flatten GIE facility-report rows into canonical PIT rows."""
    out = []
    for row in rows:
        meta = json.dumps({"status": row.get("status"), "updatedAt": row.get("updatedAt")})
        for path, unit in fields.items():
            v = _field(row, path)
            if v is None:
                continue
            out.append(
                {
                    "series_id": f"{prefix}.{scope.upper()}.{path}",
                    "observed_at": row["gasDayStart"],
                    "value": v,
                    "unit": unit,
                    "meta": meta,
                }
            )
    return out


class _GIE(Source):
    url: str
    fields: dict
    prefix: str
    env_key = "AGSI_API_KEY"  # one GIE key serves AGSI+ and ALSI+
    verified = True  # AGSI+ and ALSI+ fetched and parsed with a key on 2026-10-13
    default_countries: tuple[str, ...] = ("eu", *EU_COUNTRIES)

    def __init__(self, *args, countries: tuple[str, ...] | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.countries = countries or self.default_countries
        self._last_call = 0.0

    def _get(self, params: dict):
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        r = self.client.get(self.url, params=params, headers={"x-key": self.api_key})
        self._last_call = time.monotonic()
        r.raise_for_status()
        return r

    def _pages(self, params: dict):
        page = 1
        while True:
            r = self._get({**params, "page": page, "size": 300})
            body = r.json()
            self.keep_raw(r.content, "json")
            yield from body.get("data", [])
            if page >= int(body.get("last_page", 1) or 1):
                break
            page += 1

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        out = []
        for c in self.countries:
            scope = {"type": "eu"} if c == "eu" else {"country": c}
            rows = list(self._pages({**scope, "from": start.isoformat(), "to": end.isoformat()}))
            out.extend(rows_to_canonical(rows, self.prefix, c, self.fields))
        return pd.DataFrame(out, columns=["series_id", "observed_at", "value", "unit", "meta"])


class AGSI(_GIE):
    name = "agsi"
    url = AGSI_URL
    fields = AGSI_FIELDS
    prefix = "AGSI"


class ALSI(_GIE):
    """LNG terminal inventory and send-out. Countries without a terminal return no rows."""

    name = "alsi"
    url = ALSI_URL
    fields = ALSI_FIELDS
    prefix = "ALSI"
    # Terminal countries change (new FSRUs), so ask for all EU-27; empty answers are cheap.
    default_countries = ("eu", *EU27)
