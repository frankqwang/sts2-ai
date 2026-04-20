"""sim 进程句柄与清理工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SimProcessHandle:
    """运行中的 sim 进程句柄。"""

    process: Any
    port: int
    protocol: str
    log_dir: Path | None = None

    @property
    def pid(self) -> int | None:
        return getattr(self.process, "pid", None)
