"""VIIRS Nightfire (VNF) night-time detections near each ammonia plant, S-NPP only.

The Earth Observation Group at the Colorado School of Mines publishes one global CSV per
night and satellite. I download the S-NPP file of each night and keep the detections within
5 km of a plant. One satellite only, so the detection rate does not change when NOAA-20 and
NOAA-21 start. Access needs an account under the VIIRS Nightfire Data Use License: set
``EOG_USER`` and ``EOG_PASSWORD`` (the site is a Keycloak login, not an API key).

Product versions: VNF v2.1 nights until 2017-11, VNF v3.0 from 2017-12 (the near-real-time
pass for the newest month, the reprocessed file once it appears). The version goes into
``meta`` so a reader can split the two eras.

One row per plant and overpass (the store keeps one value per series and time): value = the
distance in km of the nearest detection within 5 km of the plant point. ``meta`` has the
numbers of detections within 5 km and within 1 km and the largest fitted hot-source
temperature of the overpass. The raw archive keeps these plant rows only; the global nightly
file is re-downloadable, so it is not stored.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import re
import ssl
import time
from html import unescape

import httpx
import numpy as np
import pandas as pd
import truststore

from nsqueeze.io.sources.base import USER_AGENT, MissingCredential, Source
from nsqueeze.io.sources.sentinel2 import COLUMNS, plants_with_coordinates

BASE = "https://eogdata.mines.edu/wwwdata/viirs_products/vnf"
AUTH_URL = (
    "https://eogauth.mines.edu/realms/eog/protocol/openid-connect/auth"
    "?response_type=code&scope=openid%20email&client_id=eogdata-new-apache"
    "&redirect_uri=https%3A%2F%2Feogdata.mines.edu%2Foauth2callback&state=nq&nonce=nq"
)
V21_END = dt.date(2017, 11, 30)  # VNF v3.0 starts in December 2017
KEEP_KM = 5.0
EARTH_KM = 6371.0
NO_FIT = 999999  # the CSV marks a missing temperature fit with this value
USECOLS = ["Lat_GMTCO", "Lon_GMTCO", "Date_Mscan", "Temp_BB"]


def parse_night(content: bytes) -> pd.DataFrame:
    """One gzipped nightly CSV → detections with UTC time and fitted temperature in kelvin."""
    raw = gzip.decompress(content) if content[:2] == b"\x1f\x8b" else content
    det = pd.read_csv(io.BytesIO(raw), usecols=USECOLS)
    det = det.rename(
        columns={
            "Lat_GMTCO": "latitude",
            "Lon_GMTCO": "longitude",
            "Date_Mscan": "time",
            "Temp_BB": "temp_k",
        }
    )
    det["time"] = pd.to_datetime(det["time"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    det["temp_k"] = pd.to_numeric(det["temp_k"], errors="coerce").where(lambda s: s < NO_FIT)
    return det.dropna(subset=["latitude", "longitude", "time"])


def near_plants(det: pd.DataFrame, plants: pd.DataFrame, product: str) -> pd.DataFrame:
    """Detections within KEEP_KM of a plant → one row per plant and overpass, ``VNF.<plant>``."""
    if det.empty:
        return pd.DataFrame(columns=COLUMNS)
    la0, lo0 = (
        np.radians(plants.lat.to_numpy())[:, None],
        np.radians(plants.lon.to_numpy())[:, None],
    )
    la, lo = (
        np.radians(det.latitude.to_numpy())[None, :],
        np.radians(det.longitude.to_numpy())[None, :],
    )
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    km = 2 * EARTH_KM * np.arcsin(np.sqrt(a))
    i, j = np.nonzero(km <= KEEP_KM)
    if len(i) == 0:
        return pd.DataFrame(columns=COLUMNS)
    d = det.iloc[j].reset_index(drop=True)
    hits = pd.DataFrame(
        {
            "series_id": "VNF." + plants.plant_id.iloc[i].reset_index(drop=True),
            # one overpass crosses a plant within one minute; the store keeps one value per time
            "observed_at": d.time.dt.floor("min"),
            "km": km[i, j],
            "temp_k": d.temp_k,
        }
    )
    rows = []
    for (sid, t), g in hits.groupby(["series_id", "observed_at"]):
        meta = {
            "n_5km": len(g),
            "n_1km": int((g.km <= 1.0).sum()),
            "max_temp_k": None if g.temp_k.isna().all() else float(g.temp_k.max()),
            "product": product,
        }
        rows.append((sid, t, float(g.km.min()), "km", json.dumps(meta)))
    return pd.DataFrame(rows, columns=COLUMNS)


class NightfireVIIRS(Source):
    name = "viirs_nightfire"
    env_key = "EOG_USER"  # EOG_PASSWORD is also required

    def __init__(self, client: httpx.Client | None = None) -> None:
        # The EOG login host chains to a root that the certifi bundle dropped in 2026-10 and
        # the operating system still trusts, so this client verifies against the system store.
        super().__init__(
            client
            or httpx.Client(
                timeout=60,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            )
        )

    def _login(self) -> None:
        pw = os.environ.get("EOG_PASSWORD")
        if not pw:
            raise MissingCredential(
                f"{self.name}: set EOG_USER and EOG_PASSWORD (see .env.example)"
            )
        r = self.client.get(AUTH_URL)
        m = re.search(r'<form id="kc-form-login"[^>]*action="([^"]+)"', r.text)
        if not m and "KEYCLOAK_IDENTITY" in self.client.cookies:
            return  # the Keycloak session is still valid: the redirect renewed the site session
        if not m:
            raise ValueError("VNF login: no Keycloak login form; the auth endpoint moved")
        r = self.client.post(
            unescape(m.group(1)),
            data={"username": self.api_key, "password": pw, "credentialId": ""},
            follow_redirects=True,
        )
        if "KEYCLOAK_IDENTITY" not in self.client.cookies:
            raise MissingCredential(f"{self.name}: EOG rejected the credentials of {self.env_key}")

    def _get(self, url: str) -> bytes | None:
        """One nightly file, or None when it does not exist. One re-login: a chunk can outlive
        the login session, and an expired session answers with the login page, not with 401."""
        for attempt in (0, 1):
            r = self.client.get(url, timeout=300)
            if r.status_code == 404:
                return None
            if r.content[:2] == b"\x1f\x8b":
                return r.content
            if attempt == 0:
                self._login()
                continue
            raise ValueError(f"VNF: no CSV at {url} (status {r.status_code})")
        raise AssertionError("unreachable")

    def _night_urls(self, day: dt.date) -> list[tuple[str, str]]:
        """Candidate (url, product) for one night, best first: reprocessed, then near-real-time."""
        ymd = day.strftime("%Y%m%d")
        if day <= V21_END:
            return [(f"{BASE}/v21/VNF_npp_d{ymd}_noaa_v21.csv.gz", "v21")]
        return [
            (f"{BASE}/v30/rearrange/{day:%Y}/{day:%m}/npp/VNF_npp_d{ymd}_noaa_v30.csv.gz", "v30"),
            # The monthly folders miss some months (10 June to 31 December 2024 on 2026-10-27).
            # The flat folder keeps the same reprocessed files.
            (f"{BASE}/v30/VNF_npp_d{ymd}_noaa_v30.csv.gz", "v30"),
            (f"{BASE}/v30_nrp/VNF_npp_d{ymd}_eog_nrp_v30.csv.gz", "v30-nrp"),
        ]

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        plants = plants_with_coordinates()
        self._login()
        rows, extract = [], []
        for ts in pd.date_range(start, end):
            day = ts.date()
            content, product, fname = None, None, None
            for url, p in self._night_urls(day):
                if (content := self._get(url)) is not None:
                    product, fname = p, url.rsplit("/", 1)[-1]
                    break
            if content is None:
                continue  # no file that night
            det = parse_night(content)
            frame = near_plants(det, plants, product)
            if not frame.empty:
                rows.append(frame)
                extract.append(frame.assign(source_file=fname))
            time.sleep(0.5)  # a nightly file every 0.5 s is far below any published limit
        if extract:
            self.keep_raw(pd.concat(extract, ignore_index=True).to_csv(index=False).encode(), "csv")
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLUMNS)
