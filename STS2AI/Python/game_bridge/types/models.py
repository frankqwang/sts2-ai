"""运行时公共 dataclass。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SessionConfig:
    port: int = 15527
    auto_launch: bool = False
    connect_timeout_s: float = 15.0
    repo_root: str | Path | None = None
    host_path: str | Path | None = None
    base_url: str = "http://127.0.0.1:15526"
    use_pipe: bool = True
    transport: str = "proto"


@dataclass(slots=True)
class StateView:
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def state_type(self) -> str:
        return str(self.raw.get("state_type") or "")

    @property
    def legal_actions(self) -> list[dict[str, Any]]:
        legal = self.raw.get("legal_actions")
        return list(legal) if isinstance(legal, list) else []

    @property
    def terminal(self) -> bool:
        return bool(self.raw.get("terminal", False))

    @property
    def run_outcome(self) -> str:
        return str(self.raw.get("run_outcome") or "")


@dataclass(slots=True)
class PolicyContext:
    step_index: int
    character_id: str
    seed: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
