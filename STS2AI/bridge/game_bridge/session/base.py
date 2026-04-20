"""session 抽象与公共工厂类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from game_bridge.types import SessionConfig


class BaseSession(Protocol):
    def close(self) -> None: ...
    def get_state(self) -> dict[str, Any]: ...


SessionKind = Literal["combat", "full_run"]


@dataclass(slots=True)
class SessionFactory:
    kind: SessionKind
    config: SessionConfig

    def create(self):
        from game_bridge.session import create_combat_session, create_full_run_session

        if self.kind == "combat":
            return create_combat_session(
                port=self.config.port,
                auto_launch=self.config.auto_launch,
                connect_timeout_s=self.config.connect_timeout_s,
                repo_root=self.config.repo_root,
                host_path=self.config.host_path,
            )
        return create_full_run_session(
            port=self.config.port,
            base_url=self.config.base_url,
            use_pipe=self.config.use_pipe,
            transport=self.config.transport,
            auto_launch=self.config.auto_launch,
            ready_timeout_s=self.config.connect_timeout_s,
            repo_root=self.config.repo_root,
            host_path=self.config.host_path,
        )
