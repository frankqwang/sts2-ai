from __future__ import annotations

"""Replay 训练任务级共享 sim 进程。

为什么要单独放这一层：
- `GameBridgeCombatRuntime(auto_launch=True)` 会让每个 runtime 自己拉起并在 close 时回收 sim
- skada replay / ordered-run 训练会频繁创建短生命周期 runtime
- 如果不把 sim 进程提升到“任务级共享”，评估时就会退化成“每打一把战斗就重启一次 sim”

这里集中做一件事：
- 整个训练任务先拉起一个 proto sim
- 后续所有 runtime 都用 `auto_launch=False` 连接同一个 port
- 任务结束后统一回收
"""

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator


def _ensure_python_bridge_path() -> None:
    sts2ai_root = Path(__file__).resolve().parents[2]
    for root in (sts2ai_root / "Python", sts2ai_root / "bridge"):
        path = str(root)
        if path not in sys.path:
            sys.path.insert(0, path)


@contextmanager
def launch_shared_proto_sim(
    *,
    port: int,
    connect_timeout_s: float = 45.0,
    host_path: str | Path | None = None,
) -> Iterator[dict[str, object]]:
    _ensure_python_bridge_path()
    from game_bridge.sim.launcher import start_headless_sim, stop_process

    start_kwargs = {
        "port": port,
        "connect_timeout_s": connect_timeout_s,
        "protocol": "proto",
    }
    if host_path is not None:
        start_kwargs["host_path"] = str(host_path)
    proc = start_headless_sim(**start_kwargs)
    try:
        yield {
            "pid": int(proc.pid),
            "log_path": str(getattr(proc, "_sim_log_path", "")),
            "port": int(port),
            "host_path": str(host_path) if host_path is not None else "",
        }
    finally:
        stop_process(proc)
