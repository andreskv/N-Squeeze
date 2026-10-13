"""Live AIS archiver (aisstream.io).
Vessels en route, with declared destinations and ETAs, are the only real source of multi-day LNG lead time.
Messages are written verbatim as gzipped JSON lines, one file per UTC hour:
    data/raw/aisstream/YYYY-MM-DD/HH.jsonl.gz
Parsing into the PIT store happens later. Endpoint and subscription format follow aisstream.io's public documentation.
Run:  pip install -e ".[ais]"   then   python archivers/ais_stream.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import gzip
import json
import os
import signal

import websockets

from nsqueeze.config import data_dir
from nsqueeze.maritime.ais import BOXES

URL = "wss://stream.aisstream.io/v0/stream"
MESSAGE_TYPES = ["PositionReport", "ShipStaticData"]  # static data carries destination + ETA


class HourlyWriter:
    def __init__(self) -> None:
        self.hour: str | None = None
        self.fh = None

    def write(self, line: str) -> None:
        now = dt.datetime.now(dt.UTC)
        key = now.strftime("%Y-%m-%d/%H")
        if key != self.hour:
            self.close()
            path = data_dir() / "raw" / "aisstream" / f"{key}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = gzip.open(path, "at", encoding="utf-8")
            self.hour = key
        self.fh.write(line + "\n")

    def close(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None


async def run() -> None:
    key = os.environ.get("AISSTREAM_API_KEY")
    if not key:
        raise SystemExit("Set AISSTREAM_API_KEY (see .env.example)")
    sub = {
        "APIKey": key,
        "BoundingBoxes": list(BOXES.values()),
        "FilterMessageTypes": MESSAGE_TYPES,
    }
    writer, backoff = HourlyWriter(), 1
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    while not stop.is_set():
        try:
            async with websockets.connect(URL, ping_interval=30) as ws:
                await ws.send(json.dumps(sub))
                backoff = 1
                async for msg in ws:
                    writer.write(msg if isinstance(msg, str) else msg.decode())
                    if stop.is_set():
                        break
        except (OSError, websockets.ConnectionClosed) as exc:
            print(f"{dt.datetime.now(dt.UTC):%H:%M:%S} reconnect in {backoff}s: {exc}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)
    writer.close()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
