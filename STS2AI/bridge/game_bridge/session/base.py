"""session 抽象与公共工厂类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from game_bridge.types import SessionConfig
from game_bridge.session.singleplayer_api import SingleplayerApiError


PIPE_SNAPSHOT_RPC_METHODS = frozenset(
    {
        "save_state",
        "export_state",
        "import_state",
        "load_state",
        "delete_state",
    }
)


class BaseSession(Protocol):
    def close(self) -> None: ...
    def get_state(self) -> dict[str, Any]: ...


class SnapshotCapableSession(BaseSession, Protocol):
    def save_state(self) -> str: ...
    def export_state(self, path: str, *, state_id: str | None = None) -> str: ...
    def import_state(self, path: str) -> dict[str, Any]: ...
    def load_state(self, state_id: str) -> dict[str, Any]: ...
    def delete_state(self, state_id: str) -> bool: ...


class PipeSnapshotMixin:
    """Shared save/load helpers for pipe-backed sessions.

    The underlying proto/json pipe codec already supports snapshot opcodes.
    Session wrappers should reuse this mixin instead of hand-copying
    `save_state/load_state/export/import/delete` one by one.
    """

    _pipe_snapshot_scope_name = "Pipe session"

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any: ...

    def save_state(self) -> str:
        result = self._call("save_state")
        state_id = result.get("state_id") if isinstance(result, dict) else None
        if isinstance(state_id, str) and state_id:
            return state_id
        raise SingleplayerApiError(f"{self._pipe_snapshot_scope_name} save_state response did not include a state_id.")

    def export_state(self, path: str, *, state_id: str | None = None) -> str:
        params: dict[str, Any] = {"path": str(path)}
        if state_id:
            params["state_id"] = str(state_id)
        result = self._call("export_state", params)
        written_path = result.get("path") if isinstance(result, dict) else None
        if isinstance(written_path, str) and written_path:
            return written_path
        raise SingleplayerApiError(f"{self._pipe_snapshot_scope_name} export_state response did not include a path.")

    def import_state(self, path: str) -> dict[str, Any]:
        result = self._call("import_state", {"path": str(path)})
        if isinstance(result, dict):
            return result
        raise SingleplayerApiError(f"{self._pipe_snapshot_scope_name} import_state response did not include a state payload.")

    def load_state(self, state_id: str) -> dict[str, Any]:
        result = self._call("load_state", {"state_id": str(state_id)})
        if isinstance(result, dict):
            return result
        raise SingleplayerApiError(f"{self._pipe_snapshot_scope_name} load_state response did not include a state payload.")

    def delete_state(self, state_id: str) -> bool:
        result = self._call("delete_state", {"state_id": str(state_id)})
        return bool(result.get("deleted", False)) if isinstance(result, dict) else False


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
