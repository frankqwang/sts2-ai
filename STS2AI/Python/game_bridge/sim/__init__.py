"""sim 启动与生命周期管理。"""

from __future__ import annotations

from pathlib import Path

from game_bridge.sim.launcher import (
    DEFAULT_DLL_PATH,
    DEFAULT_HOST_PATH,
    DEFAULT_REPO_ROOT,
    ensure_host_binary_is_fresh,
    start_headless_sim,
    stop_process,
)
from game_bridge.sim.consistency import build_consistency_report, inspect_state_consistency, static_consistency_report
from game_bridge.sim.process import SimProcessHandle


def launch_headless_sim(
    *,
    port: int,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
    host_path: str | Path = DEFAULT_HOST_PATH,
    connect_timeout_s: float = 15.0,
    protocol: str = "proto",
    dll_path: str | Path | None = None,
) -> SimProcessHandle:
    proc = start_headless_sim(
        port=port,
        repo_root=repo_root,
        host_path=dll_path or host_path,
        connect_timeout_s=connect_timeout_s,
        protocol=protocol,
    )
    return SimProcessHandle(
        process=proc,
        port=int(port),
        protocol=str(protocol),
        log_dir=Path(repo_root) / "Artifacts" / "sim_logs",
    )


__all__ = [
    "DEFAULT_DLL_PATH",
    "DEFAULT_HOST_PATH",
    "DEFAULT_REPO_ROOT",
    "SimProcessHandle",
    "build_consistency_report",
    "inspect_state_consistency",
    "static_consistency_report",
    "ensure_host_binary_is_fresh",
    "launch_headless_sim",
    "start_headless_sim",
    "stop_process",
]
