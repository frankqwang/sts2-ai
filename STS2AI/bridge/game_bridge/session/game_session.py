"""统一 GameSession：业务 API 与传输协议解耦。"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from google.protobuf import json_format

from game_bridge.generated import game_state_pb2 as pb
from game_bridge.session.base import PipeSnapshotMixin
from game_bridge.session.build_spec import BuildSpecPy, normalize_build_spec
from game_bridge.session.singleplayer_api import SingleplayerApiError
from game_bridge.session.state_semantics import is_failure_outcome, is_victory_outcome, normalize_run_outcome
from game_bridge.sim.launcher import DEFAULT_HOST_PATH, DEFAULT_REPO_ROOT, start_headless_sim, stop_process
from game_bridge.transport import (
    PipeConnection,
    PipeConnectionConfig,
    ProtoCodec,
    SimulatorApiError as TransportSimulatorApiError,
    TransportTimeoutError,
)


@dataclass(slots=True)
class SettlePolicy:
    """动作 settle 策略。

    sim 后端通常同步 settle；spectator 后端的 UI settle 等待在 C# runtime 内完成。
    这个对象保留在 Python API 中，避免调用方再把 settle 语义和 transport 混在一起。
    """

    wait_for_ui: bool = False
    timeout_s: float = 20.0
    poll_interval_s: float = 0.05


@runtime_checkable
class RpcTransport(Protocol):
    @property
    def transport_name(self) -> str: ...

    @property
    def last_call_metrics(self) -> dict[str, Any]: ...

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def call(self, method: str, params: dict[str, Any] | None = None, *, timeout_s: float | None = None) -> dict[str, Any]: ...


class PipeProtoTransport:
    """Named Pipe + protobuf binary transport."""

    transport_name = "pipe_proto"

    def __init__(
        self,
        *,
        port: int,
        auto_launch: bool = False,
        connect_timeout_s: float = 15.0,
        repo_root: str | Path | None = None,
        host_path: str | Path | None = None,
        dll_path: str | Path | None = None,
    ) -> None:
        self.port = int(port)
        self.repo_root = Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT
        self.host_path = Path(host_path or dll_path) if (host_path or dll_path) else DEFAULT_HOST_PATH
        self.connect_timeout_s = float(connect_timeout_s)
        self.auto_launch = bool(auto_launch)
        self._owned_proc: Any | None = None

        def _launcher(launch_port: int):
            proc = start_headless_sim(
                port=launch_port,
                repo_root=self.repo_root,
                host_path=self.host_path,
                connect_timeout_s=max(20.0, self.connect_timeout_s),
                protocol="proto",
            )
            self._owned_proc = proc
            return proc

        self._conn = PipeConnection(
            PipeConnectionConfig(
                port=self.port,
                protocol="proto",
                connect_timeout_s=self.connect_timeout_s,
                auto_launch=self.auto_launch,
                sim_launcher=_launcher if self.auto_launch else None,
                sim_stopper=stop_process if self.auto_launch else None,
                codec=ProtoCodec(),
            )
        )

    def connect(self) -> None:
        if not self._conn.is_connected():
            self._conn.connect()

    def close(self) -> None:
        self._conn.close()
        if self._owned_proc is not None:
            try:
                stop_process(self._owned_proc)
            finally:
                self._owned_proc = None

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.connect()
        return self._conn.safe_call(method, params, timeout_s=timeout_s)

    @property
    def last_call_metrics(self) -> dict[str, Any]:
        return dict(self._conn.last_call_metrics)


class HttpProtoJsonTransport:
    """HTTP + protobuf JSON mapping transport."""

    transport_name = "http_json"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:15526",
        request_timeout_s: float = 10.0,
    ) -> None:
        normalized = str(base_url).strip().rstrip("/")
        if normalized.endswith("/api/game_bridge/rpc"):
            self.rpc_url = normalized
        else:
            self.rpc_url = f"{normalized}/api/game_bridge/rpc"
        self.request_timeout_s = float(request_timeout_s)
        self._codec = ProtoCodec()
        self._last_call_metrics: dict[str, Any] = {}

    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        request_msg = self._codec.build_request_message(method, params or {})
        body = json_format.MessageToJson(request_msg).encode("utf-8")
        timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
        started = time.perf_counter()
        req = urllib.request.Request(
            self.rpc_url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp is not None else str(exc)
            raise SingleplayerApiError(f"HTTP protobuf JSON RPC failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise SingleplayerApiError(f"HTTP protobuf JSON RPC connection failed: {exc}") from exc

        response_msg = pb.BridgeResponseEnvelope()
        try:
            json_format.Parse(response_bytes.decode("utf-8"), response_msg)
        except Exception as exc:
            raise SingleplayerApiError(f"HTTP protobuf JSON RPC returned invalid protobuf JSON: {exc}") from exc
        result = self._codec.decode_response_message(response_msg)
        self._last_call_metrics = {
            "method": str(method),
            "request_bytes": len(body),
            "response_bytes": len(response_bytes),
            "total_duration_s": time.perf_counter() - started,
            "transport": self.transport_name,
        }
        if isinstance(result, dict) and result.get("error"):
            raise SingleplayerApiError(str(result["error"]))
        return result

    @property
    def last_call_metrics(self) -> dict[str, Any]:
        return dict(self._last_call_metrics)


def _unwrap_state(result: Any) -> dict[str, Any]:
    if isinstance(result, dict) and isinstance(result.get("state"), dict):
        return dict(result["state"])
    return dict(result) if isinstance(result, dict) else {}


def _terminal_reward(state: dict[str, Any]) -> float:
    if not bool(state.get("terminal")):
        return 0.0
    outcome = normalize_run_outcome(state.get("run_outcome"))
    if is_victory_outcome(outcome):
        return 1.0
    if is_failure_outcome(outcome):
        return -1.0
    return 0.0


class GameSession(PipeSnapshotMixin):
    """统一业务 session。

    transport 只负责传输，backend 只负责运行和 settle，Python 调用方只用
    reset/get_state/act/batch_act。
    """

    _pipe_snapshot_scope_name = "GameSession"

    def __init__(
        self,
        *,
        mode: str = "full_run",
        transport: str | RpcTransport = "pipe_proto",
        backend: str = "sim",
        port: int = 15527,
        base_url: str = "http://127.0.0.1:15526",
        auto_launch: bool = False,
        connect_timeout_s: float = 15.0,
        request_timeout_s: float = 10.0,
        repo_root: str | Path | None = None,
        host_path: str | Path | None = None,
        dll_path: str | Path | None = None,
        settle_policy: SettlePolicy | None = None,
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"full_run", "combat"}:
            raise ValueError(f"Unsupported GameSession mode: {mode!r}")
        self.mode = normalized_mode
        self.backend = str(backend).strip().lower()
        self.settle_policy = settle_policy or SettlePolicy(wait_for_ui=self.backend == "spectator")
        self._transport = (
            transport
            if isinstance(transport, RpcTransport)
            else _build_transport(
                transport=transport,
                backend=self.backend,
                port=port,
                base_url=base_url,
                auto_launch=auto_launch,
                connect_timeout_s=connect_timeout_s,
                request_timeout_s=request_timeout_s,
                repo_root=repo_root,
                host_path=host_path,
                dll_path=dll_path,
            )
        )
        self._current_state: dict[str, Any] = {}
        self._last_step_info: dict[str, Any] | None = None
        self._last_settlement_events: list[dict[str, Any]] = []

    def close(self) -> None:
        self._transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def reset(
        self,
        *,
        character_id: str | None = None,
        character: str | None = None,
        encounter_id: str | None = None,
        encounter: str | None = None,
        ascension_level: int | None = None,
        ascension: int | None = None,
        seed: str | None = None,
        build: BuildSpecPy | dict[str, Any] | None = None,
        floor: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        normalized_build = normalize_build_spec(build)
        params: dict[str, Any] = {
            "character_id": str(character_id or character or "IRONCLAD"),
            "ascension_level": int(ascension_level if ascension_level is not None else (ascension or 0)),
        }
        if seed:
            params["seed"] = str(seed)
        if normalized_build is not None:
            params["build"] = normalized_build
        if floor is not None:
            params["floor"] = int(floor)
        if encounter_id or encounter:
            params["encounter_id"] = str(encounter_id or encounter)
        method = "combat_reset" if self.mode == "combat" else "reset"
        result = self._call(method, params, timeout_s=timeout_s)
        self._current_state = _unwrap_state(result)
        return self.current_state

    def get_state(self) -> dict[str, Any]:
        method = "combat_state" if self.mode == "combat" else "state"
        self._current_state = _unwrap_state(self._call(method))
        return self.current_state

    def act(self, action: dict[str, Any] | pb.LegalAction, *, timeout_s: float | None = None) -> dict[str, Any]:
        self._last_step_info = None
        self._last_settlement_events = []
        params = _legal_action_to_dict(action)
        method = "combat_act" if self.mode == "combat" else "act"
        result = self._call(method, params, timeout_s=timeout_s)
        events = result.get("settlement_events") if isinstance(result, dict) else None
        if isinstance(events, list):
            self._last_settlement_events = [dict(item) for item in events if isinstance(item, dict)]
        if not bool(result.get("accepted", True)):
            self._raise_rejected_action(result)
        state = result.get("state") if isinstance(result, dict) else None
        if not isinstance(state, dict):
            raise SingleplayerApiError("Bridge act response did not include a state payload.")
        info = result.get("info")
        if isinstance(info, dict):
            self._last_step_info = dict(info)
            self._last_step_info["settlement_events"] = list(self._last_settlement_events)
        self._current_state = dict(state)
        return self.current_state

    def batch_act(self, actions: list[dict[str, Any]], *, timeout_s: float | None = None) -> dict[str, Any]:
        result = self._call("batch_act", {"actions": actions}, timeout_s=timeout_s)
        events = result.get("settlement_events") if isinstance(result, dict) else None
        self._last_settlement_events = [dict(item) for item in events if isinstance(item, dict)] if isinstance(events, list) else []
        if not bool(result.get("accepted", True)):
            self._raise_rejected_action(result)
        state = result.get("state")
        if isinstance(state, dict):
            self._current_state = dict(state)
        return result

    def act_gym(self, action: dict[str, Any] | pb.LegalAction) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Zero/RL runtime 需要 gym-style tuple 时使用；内部仍走 act。"""
        state = self.act(action)
        done = bool(state.get("terminal"))
        info = {
            "accepted": True,
            "state_type": state.get("state_type"),
            "run_outcome": normalize_run_outcome(state.get("run_outcome")) or None,
            "settlement_events": self.last_settlement_events,
        }
        return state, _terminal_reward(state), done, info

    def clear_state_cache(self) -> bool:
        result = self._call("clear_state_cache")
        return bool(result.get("deleted", False))

    def save_state(self) -> str:
        method = "save_search_state" if self.mode == "combat" else "save_state"
        result = self._call(method)
        state_id = result.get("state_id") if isinstance(result, dict) else None
        if isinstance(state_id, str) and state_id:
            return state_id
        raise SingleplayerApiError("Bridge save_state response did not include a state_id.")

    def load_state(self, state_id: str) -> dict[str, Any]:
        result = self._call("load_state", {"state_id": str(state_id)})
        self._current_state = _unwrap_state(result)
        return self.current_state

    def export_state(self, path: str, *, state_id: str | None = None) -> str:
        params: dict[str, Any] = {"path": str(path)}
        if state_id:
            params["state_id"] = str(state_id)
        result = self._call("export_state", params)
        written_path = result.get("path") if isinstance(result, dict) else None
        if isinstance(written_path, str) and written_path:
            return written_path
        raise SingleplayerApiError("Bridge export_state response did not include a path.")

    def import_state(self, path: str) -> dict[str, Any]:
        result = self._call("import_state", {"path": str(path)})
        self._current_state = _unwrap_state(result)
        return self.current_state

    def delete_state(self, state_id: str) -> bool:
        result = self._call("delete_state", {"state_id": str(state_id)})
        return bool(result.get("deleted", False)) if isinstance(result, dict) else False

    def perf_stats(self) -> dict[str, Any]:
        result = self._call("perf_stats")
        return result if isinstance(result, dict) else {}

    def reset_perf_stats(self) -> bool:
        result = self._call("reset_perf_stats")
        return bool(result.get("reset", False))

    @property
    def supports_local_ort(self) -> bool:
        return self.transport_name == "pipe_proto" and self.backend == "sim"

    def load_ort_model(self, path: str) -> bool:
        result = self._call("load_ort_model", {"path": str(path)})
        return bool(result.get("loaded", False))

    def run_combat_local(self, *, max_steps: int = 600) -> dict[str, Any]:
        result = self._call("run_combat_local", {"max_steps": int(max_steps)})
        return result if isinstance(result, dict) else {}

    def search_combat_mcts(self, **kwargs: Any) -> dict[str, Any]:
        result = self._call("search_combat_mcts", kwargs)
        return result if isinstance(result, dict) else {}

    @property
    def current_state(self) -> dict[str, Any]:
        return dict(self._current_state)

    @property
    def terminal(self) -> bool:
        return bool(self._current_state.get("terminal"))

    @property
    def run_outcome(self) -> str:
        return str(normalize_run_outcome(self._current_state.get("run_outcome")) or "")

    @property
    def legal_actions(self) -> list[dict[str, Any]]:
        legal = self._current_state.get("legal_actions")
        return list(legal) if isinstance(legal, list) else []

    @property
    def transport_name(self) -> str:
        return self._transport.transport_name

    @property
    def last_step_info(self) -> dict[str, Any] | None:
        return dict(self._last_step_info) if isinstance(self._last_step_info, dict) else None

    @property
    def last_settlement_events(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._last_settlement_events]

    def get_last_transport_metrics(self) -> dict[str, Any]:
        return dict(self._transport.last_call_metrics)

    def _call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        try:
            return self._transport.call(method, params, timeout_s=timeout_s)
        except TransportSimulatorApiError as exc:
            raise SingleplayerApiError(str(exc)) from exc
        except TransportTimeoutError as exc:
            raise TimeoutError(str(exc)) from exc

    def _raise_rejected_action(self, result: dict[str, Any]) -> None:
        error = SingleplayerApiError(str(result.get("error") or "Bridge act rejected"))
        state = result.get("state")
        if isinstance(state, dict):
            self._current_state = dict(state)
            setattr(error, "latest_state", self.current_state)
        info = result.get("info")
        if isinstance(info, dict):
            self._last_step_info = dict(info)
            setattr(error, "step_info", dict(info))
        events = result.get("settlement_events")
        if isinstance(events, list):
            self._last_settlement_events = [dict(item) for item in events if isinstance(item, dict)]
            setattr(error, "settlement_events", self.last_settlement_events)
        raise error


def _legal_action_to_dict(action: dict[str, Any] | pb.LegalAction) -> dict[str, Any]:
    if isinstance(action, pb.LegalAction):
        result: dict[str, Any] = {
            "action": action.action,
            "index": action.index,
            "card_index": action.card_index,
            "target_id": action.target_id,
            "col": action.col,
            "row": action.row,
            "slot": action.slot,
            "label": action.label,
            "card_id": action.card_id,
        }
        return {key: value for key, value in result.items() if value not in ("", None)}
    return dict(action)


def _build_transport(
    *,
    transport: str,
    backend: str,
    port: int,
    base_url: str,
    auto_launch: bool,
    connect_timeout_s: float,
    request_timeout_s: float,
    repo_root: str | Path | None,
    host_path: str | Path | None,
    dll_path: str | Path | None,
) -> RpcTransport:
    t = str(transport or "").strip().lower()
    if not t:
        t = "http_json" if backend == "spectator" else "pipe_proto"
    if t in {"pipe_proto", "pipe-proto", "proto", "protobuf"}:
        return PipeProtoTransport(
            port=port,
            auto_launch=auto_launch,
            connect_timeout_s=connect_timeout_s,
            repo_root=repo_root,
            host_path=host_path,
            dll_path=dll_path,
        )
    if t in {"http_json", "http-json", "pb_json", "pb-json", "protobuf_json", "protobuf-json"}:
        return HttpProtoJsonTransport(base_url=base_url, request_timeout_s=request_timeout_s)
    raise ValueError(f"Unsupported GameSession transport: {transport!r}")


def create_game_session(
    *,
    mode: str = "full_run",
    transport: str = "pipe_proto",
    backend: str = "sim",
    **kwargs: Any,
) -> GameSession:
    return GameSession(mode=mode, transport=transport, backend=backend, **kwargs)


__all__ = [
    "GameSession",
    "HttpProtoJsonTransport",
    "PipeProtoTransport",
    "RpcTransport",
    "SettlePolicy",
    "create_game_session",
]
