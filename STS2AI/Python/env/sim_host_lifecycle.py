"""Lifecycle manager for launching and tearing down headless sim host processes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from env.headless_sim_runner import DEFAULT_DLL_PATH, DEFAULT_REPO_ROOT, start_headless_sim, stop_process


def transport_launch_protocol(transport: str) -> str | None:
    normalized = str(transport or "").strip().lower()
    if normalized == "pipe-binary":
        return "bin"
    if normalized == "pipe":
        return "json"
    return None


@dataclass
class SimHostLifecycleManager:
    ports: list[int]
    transport: str
    auto_launch: bool = False
    repo_root: str | Path = DEFAULT_REPO_ROOT
    dll_path: str | Path = DEFAULT_DLL_PATH
    connect_timeout_s: float = 15.0
    _procs: list[Any] = field(default_factory=list, init=False, repr=False)

    def start(self) -> list[Any]:
        if not self.auto_launch:
            return []
        protocol = transport_launch_protocol(self.transport)
        if protocol is None:
            return []
        repo_root = Path(self.repo_root)
        dll_path = Path(self.dll_path)
        for port in self.ports:
            self._procs.append(
                start_headless_sim(
                    port=int(port),
                    repo_root=repo_root,
                    dll_path=dll_path,
                    connect_timeout_s=float(self.connect_timeout_s),
                    protocol=protocol,
                )
            )
        return list(self._procs)

    def stop(self) -> None:
        while self._procs:
            stop_process(self._procs.pop())

    def __enter__(self) -> "SimHostLifecycleManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
