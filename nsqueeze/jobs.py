"""Long archive jobs (back-fills) that pause, resume and show progress.

A job splits a date range into chunks and archives one chunk at a time.
After each chunk it saves its state to ``data/logs/jobs/<name>.json``, so a stop loses at most the
chunk in progress. ``resume`` starts again at the first chunk that is not done.

    nsq jobs new weather-2000 --source weather --since 2000-01-01
    nsq jobs run weather-2000        # runs until done, paused or failed
    nsq jobs pause weather-2000      # from another terminal: stops after the current chunk
    nsq jobs resume weather-2000
    nsq jobs status                  # progress of every job
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from nsqueeze.config import data_dir

# Waits (seconds) before retrying a chunk after a transient network error. An
# unattended back-fill should not stop for one slow or dropped request.
RETRY_WAITS = (60, 300, 900)


def _transient(e: Exception) -> bool:
    if isinstance(e, httpx.TimeoutException | httpx.TransportError):
        return True
    return isinstance(e, httpx.HTTPStatusError) and (
        e.response.status_code == 429 or e.response.status_code >= 500
    )


def _archive_with_retry(src, store, a: dt.date, b: dt.date, log) -> int:
    for wait in (*RETRY_WAITS, None):
        try:
            return src.archive(store, a, b)
        except Exception as e:
            if wait is None or not _transient(e):
                raise
            log(f"chunk {a}..{b}: {type(e).__name__}, retry in {wait} s")
            time.sleep(wait)
    raise AssertionError("unreachable")


# Chunk length per source, in days. One chunk = one API window = one PIT vintage.
CHUNK_DAYS = {
    "weather": 1826,
    "entsog": 31,
    "agsi": 92,
    "alsi": 92,
    "portwatch": 366,
    "entsoe": 366,
    "s2_swir": 92,  # one quarter: about 20 scenes per plant, one Earth Engine call each
    "firms_viirs": 366,  # one year: 74 requests of 5 days
    "s2_plume": 366,  # one year per chunk: about 70 Level-2A scenes per plant
    "weather_plants": 1826,
    "landsat_st": 366,  # one year: about 45 scenes per plant (two satellites, 8-day revisit)
}


def jobs_dir() -> Path:
    d = data_dir() / "logs" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def plan_chunks(start: dt.date, end: dt.date, days: int | None) -> list[list[str]]:
    """Consecutive [start, end] windows that cover the range exactly once."""
    if not days:
        return [[start.isoformat(), end.isoformat()]]
    out, a = [], start
    while a <= end:
        b = min(end, a + dt.timedelta(days=days - 1))
        out.append([a.isoformat(), b.isoformat()])
        a = b + dt.timedelta(days=1)
    return out


@dataclass
class Job:
    name: str
    source: str
    chunks: list[list[str]]
    done: list[bool] = field(default_factory=list)
    rows: int = 0
    status: str = "ready"  # ready | running | paused | done | failed
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    last_error: str | None = None

    @property
    def path(self) -> Path:
        return jobs_dir() / f"{self.name}.json"

    @property
    def pause_flag(self) -> Path:
        return jobs_dir() / f"{self.name}.pause"

    @classmethod
    def new(cls, name: str, source: str, start: dt.date, end: dt.date) -> Job:
        job = cls(name, source, plan_chunks(start, end, CHUNK_DAYS.get(source)))
        if job.path.exists():
            raise FileExistsError(f"job {name!r} exists; resume it or pick another name")
        job.done = [False] * len(job.chunks)
        job.save()
        return job

    @classmethod
    def load(cls, name: str) -> Job:
        path = jobs_dir() / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"no job {name!r} in {jobs_dir()}")
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self) -> None:
        self.updated_at = _now()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        tmp.replace(self.path)  # atomic: a crash never leaves a half-written state file

    def log(self, msg: str) -> None:
        with (jobs_dir() / f"{self.name}.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {msg}\n")
        print(f"[{self.name}] {msg}", flush=True)

    @property
    def progress(self) -> str:
        n = sum(self.done)
        return f"{n}/{len(self.chunks)} chunks ({100 * n / len(self.chunks):.0f}%)"

    def run(self, store=None, source=None) -> str:
        """Archive the pending chunks in order. Returns the final status."""
        from nsqueeze.io import PITStore
        from nsqueeze.io.sources import get

        if self.status == "done":
            return self.status
        store, src = store or PITStore(), source or get(self.source)
        self.status, self.last_error = "running", None
        self.save()
        try:
            for i, (a, b) in enumerate(self.chunks):
                if self.done[i]:
                    continue
                if self.pause_flag.exists():
                    self.status = "paused"
                    self.log(f"paused before chunk {a}..{b}; {self.progress}")
                    return self._finish()
                n = _archive_with_retry(
                    src, store, dt.date.fromisoformat(a), dt.date.fromisoformat(b), self.log
                )
                self.done[i], self.rows = True, self.rows + n
                self.save()
                self.log(f"chunk {a}..{b}: {n} rows; {self.progress}")
            self.status = "done"
            self.log(f"done; {self.rows} rows in total")
        except KeyboardInterrupt:
            self.status = "paused"
            self.log(f"interrupted; {self.progress}. Resume with `nsq jobs resume {self.name}`")
        except Exception as e:  # quota, network, parse: keep the state, report, allow resume
            self.status, self.last_error = "failed", f"{type(e).__name__}: {e}"[:500]
            self.log(f"failed: {self.last_error}. Resume with `nsq jobs resume {self.name}`")
        return self._finish()

    def _finish(self) -> str:
        self.save()
        return self.status


def all_jobs() -> list[Job]:
    return [Job.load(p.stem) for p in sorted(jobs_dir().glob("*.json"))]
