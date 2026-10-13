"""Common fetcher contract.

A :class:`Source` turns one API call into a canonical frame (see ``nsqueeze.io.pit``)
and optionally keeps the raw payload for audit. Fetchers never transform values
beyond parsing: unit conversions happen in the model layer.
"""

from __future__ import annotations

import datetime as dt
import gzip
import os
import uuid
from abc import ABC, abstractmethod
from typing import ClassVar

import httpx
import pandas as pd

from nsqueeze.config import data_dir
from nsqueeze.io.pit import PITStore

USER_AGENT = "n-squeeze-archiver/0.1 (research; contact via repository)"


class MissingCredential(RuntimeError):
    pass


class Source(ABC):
    name: ClassVar[str]
    env_key: ClassVar[str | None] = None
    verified: ClassVar[bool] = False  # flip once a human has checked endpoint + terms

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(
            timeout=60, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )

    @property
    def api_key(self) -> str:
        if not self.env_key:
            return ""
        key = os.environ.get(self.env_key)
        if not key:
            raise MissingCredential(f"{self.name}: set {self.env_key} (see .env.example)")
        return key

    @abstractmethod
    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Return canonical columns: series_id, observed_at, value, unit[, meta]."""

    def keep_raw(self, payload: bytes, ext: str) -> None:
        now = dt.datetime.now(dt.UTC)
        d = data_dir() / "raw" / self.name / now.date().isoformat()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{now:%H%M%S}-{uuid.uuid4().hex[:8]}.{ext}.gz").write_bytes(gzip.compress(payload))

    def archive(self, store: PITStore, start: dt.date, end: dt.date) -> int:
        frame = self.fetch(start, end)
        if frame.empty:
            return 0
        store.write(self.name, frame)
        return len(frame)


class NotYetImplemented(Source):
    """Placeholder for a registered source whose fetcher is scheduled for a later phase."""

    phase: ClassVar[int] = 0

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError(
            f"{self.name}: fetcher not written yet (phase {self.phase}). See config/sources.yaml."
        )
