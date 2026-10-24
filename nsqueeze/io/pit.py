"""Point-in-time (PIT) store: append-only Parquet under DuckDB.

Every value is stored with the moment it was retrieved and is never overwritten.
``read(as_of=t)`` reconstructs exactly what was knowable at ``t`` — the latest
vintage of each (series, observation date) retrieved on or before ``t``. Some
histories (AGSI+ vintages, live AIS) cannot be rebuilt later, which is why the
archivers start on day one.

Layout::

    data/pit/source=<name>/retrieved_date=YYYY-MM-DD/<HHMMSS>-<uuid8>.parquet

Canonical columns (one row per observation)::

    series_id    str        e.g. "EXR.D.USD.EUR"           (stable, source-specific)
    observed_at  timestamp  the time the value refers to   (gas day, fixing date…)
    value        float64
    unit         str        e.g. "USD/EUR", "TWh", "EUR/MWh"
    meta         str|null   JSON with source-specific extras
    retrieved_at timestamp  UTC, added by the store
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from nsqueeze.config import data_dir

REQUIRED = ("series_id", "observed_at", "value", "unit")


class PITStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else data_dir() / "pit"
        self.root.mkdir(parents=True, exist_ok=True)

    # ── write ────────────────────────────────────────────────────────────────
    def write(
        self, source: str, frame: pd.DataFrame, *, retrieved_at: dt.datetime | None = None
    ) -> Path:
        missing = [c for c in REQUIRED if c not in frame.columns]
        if missing:
            raise ValueError(f"{source}: frame lacks canonical columns {missing}")
        if not source.replace("_", "").isalnum():
            raise ValueError(f"source name must be [A-Za-z0-9_]: {source!r}")

        ts = (retrieved_at or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        df = frame.loc[:, [*REQUIRED, *(["meta"] if "meta" in frame else [])]].copy()
        if "meta" not in df:
            df["meta"] = None
        df["observed_at"] = pd.to_datetime(df["observed_at"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
        df["retrieved_at"] = pd.Timestamp(ts).tz_localize(None)  # stored as naive UTC
        dup = df.duplicated(["series_id", "observed_at"], keep=False)
        if dup.any():  # a read keeps one row per key, so the others would be lost
            first = df[dup].iloc[0]
            raise ValueError(
                f"{source}: {int(dup.sum())} rows share a (series_id, observed_at) key, "
                f"for example {first.series_id} at {first.observed_at}. Write one row per key"
            )

        part = self.root / f"source={source}" / f"retrieved_date={ts.date().isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        path = part / f"{ts:%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"
        with path.open("xb") as fh:  # "x": fail rather than ever overwrite a vintage
            df.to_parquet(fh, index=False)
        return path

    # ── read ─────────────────────────────────────────────────────────────────
    def _glob(self, source: str) -> str:
        return (self.root / f"source={source}" / "**" / "*.parquet").as_posix()

    def read(
        self,
        source: str,
        *,
        as_of: dt.datetime | str | None = None,
        series: list[str] | None = None,
    ) -> pd.DataFrame:
        """Latest vintage per (series_id, observed_at) known at ``as_of`` (UTC; default: now)."""
        cutoff = pd.Timestamp(as_of or dt.datetime.now(dt.UTC))
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        where, args = ["retrieved_at <= ?"], [cutoff.to_pydatetime()]
        if series:
            where.append(f"series_id IN ({', '.join('?' * len(series))})")
            args.extend(series)
        sql = f"""
            SELECT series_id, observed_at, value, unit, meta, retrieved_at
            FROM read_parquet('{self._glob(source)}', hive_partitioning = true,
                              union_by_name = true)
            WHERE {" AND ".join(where)}
            QUALIFY row_number() OVER (
                PARTITION BY series_id, observed_at ORDER BY retrieved_at DESC) = 1
            ORDER BY series_id, observed_at
        """
        with duckdb.connect() as con:
            return con.execute(sql, args).df()

    def vintages(self, source: str) -> pd.DataFrame:
        """One row per retrieval: when, how many rows, observation span covered."""
        sql = f"""
            SELECT retrieved_at, count(*) AS n_rows,
                   min(observed_at) AS first_obs, max(observed_at) AS last_obs
            FROM read_parquet('{self._glob(source)}', hive_partitioning = true)
            GROUP BY retrieved_at ORDER BY retrieved_at
        """
        with duckdb.connect() as con:
            return con.execute(sql).df()

    def sources(self) -> list[str]:
        return sorted(p.name.split("=", 1)[1] for p in self.root.glob("source=*") if p.is_dir())
