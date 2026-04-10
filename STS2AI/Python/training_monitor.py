"""Stream training metrics from metrics.jsonl to WebSocket overlay.

Tails a metrics.jsonl file via polling and broadcasts new entries
to connected overlay clients. Can run standalone or integrated
with demo_play.py.

Usage (standalone):
    python training_monitor.py --metrics-file path/to/metrics.jsonl --port 8765

Usage (integrated with demo_play.py):
    monitor = TrainingMetricsMonitor("path/to/metrics.jsonl", broadcaster)
    monitor.start()  # runs in daemon thread
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TrainingMetricsMonitor:
    """Polls metrics.jsonl and calls broadcast_fn with new entries."""

    def __init__(
        self,
        metrics_file: str | Path,
        broadcast_fn: Callable[[dict], None],
        poll_interval: float = 5.0,
        buffer_size: int = 500,
    ):
        self.metrics_file = Path(metrics_file)
        self.broadcast_fn = broadcast_fn
        self.poll_interval = poll_interval
        self.buffer: list[dict] = []
        self.buffer_size = buffer_size
        self._offset = 0
        self._stop = threading.Event()

    def start(self):
        """Start monitoring in a daemon thread."""
        t = threading.Thread(target=self._poll_loop, daemon=True, name="MetricsMonitor")
        t.start()
        logger.info("Training metrics monitor started: %s", self.metrics_file)
        return t

    def stop(self):
        self._stop.set()

    def get_history(self) -> list[dict]:
        """Return buffered history for initial client sync."""
        return list(self.buffer)

    def _poll_loop(self):
        """Poll file for new lines."""
        # Initial read of existing content
        self._read_new_lines(initial=True)

        while not self._stop.is_set():
            self._stop.wait(self.poll_interval)
            if self._stop.is_set():
                break
            self._read_new_lines(initial=False)

    def _read_new_lines(self, initial: bool = False):
        if not self.metrics_file.exists():
            return

        try:
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                f.seek(self._offset)
                new_lines = f.readlines()
                self._offset = f.tell()
        except (OSError, IOError) as e:
            logger.warning("Failed to read metrics file: %s", e)
            return

        new_entries = []
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                new_entries.append(entry)
            except json.JSONDecodeError:
                continue

        if not new_entries:
            return

        # Update buffer
        self.buffer.extend(new_entries)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

        # Broadcast
        if initial:
            # Send full history on startup
            self.broadcast_fn({
                "msg_type": "training_history",
                "data": self.buffer,
            })
            logger.info("Sent %d historical metrics entries", len(self.buffer))
        else:
            # Send new entries incrementally
            for entry in new_entries:
                self.broadcast_fn({
                    "msg_type": "training",
                    "data": entry,
                })
            logger.info("Streamed %d new metrics entries", len(new_entries))


if __name__ == "__main__":
    import argparse
    import asyncio
    import websockets.server

    parser = argparse.ArgumentParser(description="Training Metrics WebSocket Streamer")
    parser.add_argument("--metrics-file", type=str, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    state = {"clients": set(), "loop": None}

    def broadcast(msg: dict):
        if not state["clients"] or state["loop"] is None:
            return
        text = json.dumps(msg, ensure_ascii=False, default=str)
        asyncio.run_coroutine_threadsafe(_send_all(text), state["loop"])

    async def _send_all(text: str):
        dead = set()
        for ws in state["clients"]:
            try:
                await ws.send(text)
            except Exception:
                dead.add(ws)
        state["clients"].difference_update(dead)

    monitor = TrainingMetricsMonitor(args.metrics_file, broadcast, args.poll_interval)

    async def handler(ws):
        state["clients"].add(ws)
        logger.info("Client connected (%d total)", len(state["clients"]))
        history = monitor.get_history()
        if history:
            await ws.send(json.dumps({
                "msg_type": "training_history",
                "data": history,
            }, ensure_ascii=False, default=str))
        try:
            async for _ in ws:
                pass
        finally:
            state["clients"].discard(ws)

    async def main_async():
        state["loop"] = asyncio.get_event_loop()
        monitor.start()
        async with websockets.server.serve(handler, "0.0.0.0", args.port):
            logger.info("WebSocket server on port %d", args.port)
            await asyncio.Future()

    asyncio.run(main_async())
