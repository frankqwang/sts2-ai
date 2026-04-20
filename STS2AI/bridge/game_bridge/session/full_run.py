"""全局运行环境客户端。

- `ApiBackedFullRunClient`: HTTP 观战 mod。
- `PipeBackedFullRunClient` / `BinaryBackedFullRunClient`: 训练主通道（Named Pipe, proto）。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from game_bridge.session.singleplayer_api import (
    SingleplayerApiError,
    SingleplayerClient,
    SingleplayerTimeoutError,
)
from game_bridge.sim.launcher import DEFAULT_HOST_PATH, DEFAULT_REPO_ROOT, start_headless_sim, stop_process
# 2026-04-18: game_bridge 统一走 PipeConnection + proto。
# - proto 协议 (ProtoCodec) : 训练主通道,schema 稳定
# - json  协议 (JsonCodec)  : 观战 / 诊断路径 (Godot mod 仍在用)
# - bin   协议              : **废弃** — 手写二进制解码器已停,禁止再加新功能
from game_bridge.transport import (
    PipeConnection,
    PipeConnectionConfig,
    JsonCodec,
    ProtoCodec,
    SimulatorApiError as TransportSimulatorApiError,
    TransportTimeoutError,
)
from game_bridge.sim.simulator_api_error import SimulatorApiError

logger = logging.getLogger(__name__)


def _unwrap_envelope(result: Any) -> Any:
    """PipeConnection 的 binary codec 可能返回 {status,opcode,payload} envelope;
    提取内部 payload dict。JSON codec 返回纯 dict 时直接透传。"""
    if (isinstance(result, dict)
            and "payload" in result
            and isinstance(result.get("payload"), dict)
            and "status" in result
            and "opcode" in result):
        return result["payload"]
    return result


def _state_type(state: dict[str, Any] | None) -> str:
    return str((state or {}).get("state_type") or "").lower()


def _extract_run_outcome(state: dict[str, Any]) -> str | None:
    for key in ("run_outcome", "outcome"):
        value = state.get(key)
        if value is not None:
            text = str(value).strip().lower()
            if text:
                return text

    game_over = state.get("game_over")
    if isinstance(game_over, dict):
        for key in ("run_outcome", "outcome", "result"):
            value = game_over.get(key)
            if value is None:
                continue
            text = str(value).strip().lower()
            if text:
                return text
    return None


def _is_menu_ready_for_v2_reset(state: dict[str, Any]) -> bool:
    if _state_type(state) != "menu":
        return True
    menu = state.get("menu")
    if not isinstance(menu, dict):
        return True
    return bool(menu.get("is_main_menu_visible"))


def _looks_like_missing_endpoint(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    return any(
        marker in text
        for marker in (
            "http 404",
            "not found",
            "unsupported full run env",
            "unknown api",
        )
    )

def _normalize_build_spec(build: dict[str, Any] | None) -> dict[str, Any] | None:
    if build is None:
        return None
    if not isinstance(build, dict):
        raise TypeError("build must be a dict when provided")

    def _normalize_card_entry(entry: Any) -> dict[str, Any]:
        if isinstance(entry, str):
            card_id = entry.strip()
            if not card_id:
                raise ValueError("build.deck contains an empty card id")
            return {"id": card_id}
        if not isinstance(entry, dict):
            raise TypeError("build.deck entries must be strings or dicts")
        card_id = str(entry.get("id") or entry.get("card_id") or entry.get("name") or "").strip()
        if not card_id:
            raise ValueError("build.deck entry is missing id")
        normalized: dict[str, Any] = {"id": card_id}
        upgrade_level = entry.get("upgrade_level", entry.get("upgrades", entry.get("current_upgrade_level")))
        if upgrade_level is None and bool(entry.get("is_upgraded")):
            upgrade_level = 1
        if upgrade_level is not None:
            normalized["upgrade_level"] = max(0, int(upgrade_level))
        floor_added = entry.get("floor_added_to_deck")
        if floor_added is not None:
            normalized["floor_added_to_deck"] = int(floor_added)
        props = entry.get("props")
        if props is not None:
            if not isinstance(props, dict):
                raise TypeError("build.deck entry props must be a dict")
            normalized["props"] = props
        return normalized

    def _normalize_relic_entry(entry: Any) -> dict[str, Any]:
        if isinstance(entry, str):
            relic_id = entry.strip()
            if not relic_id:
                raise ValueError("build.relics contains an empty relic id")
            return {"id": relic_id}
        if not isinstance(entry, dict):
            raise TypeError("build.relics entries must be strings or dicts")
        relic_id = str(entry.get("id") or entry.get("relic_id") or entry.get("name") or "").strip()
        if not relic_id:
            raise ValueError("build.relics entry is missing id")
        normalized: dict[str, Any] = {"id": relic_id}
        floor_added = entry.get("floor_added_to_deck")
        if floor_added is not None:
            normalized["floor_added_to_deck"] = int(floor_added)
        return normalized

    normalized_build: dict[str, Any] = {}
    deck_entries = build.get("deck", build.get("cards"))
    if deck_entries is not None:
        if not isinstance(deck_entries, list):
            raise TypeError("build.deck must be a list")
        normalized_build["deck"] = [_normalize_card_entry(entry) for entry in deck_entries]

    relic_entries = build.get("relics", build.get("relic_ids"))
    if relic_entries is not None:
        if not isinstance(relic_entries, list):
            raise TypeError("build.relics must be a list")
        normalized_build["relics"] = [_normalize_relic_entry(entry) for entry in relic_entries]

    scalar_aliases = {
        "current_hp": ("current_hp", "hp"),
        "max_hp": ("max_hp",),
        "max_energy": ("max_energy", "energy"),
        "gold": ("gold",),
    }
    for target_key, aliases in scalar_aliases.items():
        for alias in aliases:
            if alias in build and build[alias] is not None:
                normalized_build[target_key] = int(build[alias])
                break

    return normalized_build or None


@dataclass(slots=True)
class ApiBackedFullRunClient:
    base_url: str = "http://127.0.0.1:15526"
    poll_interval_s: float = 0.05
    request_timeout_s: float = 10.0
    ready_timeout_s: float = 20.0
    prefer_v2: bool = True
    _singleplayer: SingleplayerClient = field(init=False, repr=False)
    _use_v2: bool | None = field(default=None, init=False, repr=False)
    _last_step_info: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._singleplayer = SingleplayerClient(
            base_url=self.base_url,
            poll_interval_s=self.poll_interval_s,
            request_timeout_s=self.request_timeout_s,
            ready_timeout_s=self.ready_timeout_s,
        )

    def get_state(self) -> dict[str, Any]:
        if self._should_use_v2():
            return self._request_v2_state()
        return self._singleplayer.get_state()

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._last_step_info = None
        if self._should_use_v2():
            result = self._request_v2("POST", "/api/v2/full_run_env/step", payload)
            info = result.get("info")
            if isinstance(info, dict):
                self._last_step_info = dict(info)
            if not bool(result.get("accepted", False)):
                error = SingleplayerApiError(str(result.get("error") or "Unknown full-run env step error"))
                state = result.get("state")
                if isinstance(state, dict):
                    setattr(error, "latest_state", state)
                if isinstance(info, dict):
                    setattr(error, "step_info", info)
                raise error
            state = result.get("state")
            if isinstance(state, dict):
                return state
            raise SingleplayerApiError("Full-run env step response did not include a state payload.")
        return self._singleplayer.act(payload)

    def batch_act(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Execute multiple actions in a single HTTP request.

        All actions are executed sequentially on the game side within one
        main-thread call, with no per-action HTTP or frame overhead.

        Returns the state after all actions (or after the first rejection/terminal).
        Raises SingleplayerApiError if the batch was rejected.
        """
        result = self._request_v2("POST", "/api/v2/full_run_env/batch_step", {"actions": actions})
        if not bool(result.get("accepted", False)):
            error = SingleplayerApiError(str(result.get("error") or "Batch step error"))
            setattr(error, "latest_state", result)
            setattr(error, "steps_executed", result.get("steps_executed", 0))
            raise error
        return result

    def reset(
        self,
        *,
        character_id: str = "IRONCLAD",
        ascension_level: int = 0,
        seed: str | None = None,
        build: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        normalized_build = _normalize_build_spec(build)
        if self._should_use_v2():
            payload: dict[str, Any] = {
                "character_id": str(character_id),
                "ascension": int(ascension_level),
            }
            if seed:
                payload["seed"] = str(seed)
            if normalized_build is not None:
                payload["build"] = normalized_build
            wait_timeout = self.ready_timeout_s if timeout_s is None else float(timeout_s)
            initial_state = self._request_v2_state()
            if not _is_menu_ready_for_v2_reset(initial_state):
                initial_state = self.wait_until(
                    _is_menu_ready_for_v2_reset,
                    timeout_s=wait_timeout,
                    initial_state=initial_state,
                )
            payload["timeout_ms"] = max(100, int(wait_timeout * 1000))
            state = self._request_v2("POST", "/api/v2/full_run_env/reset", payload)
            if isinstance(state, dict):
                return state
            raise SingleplayerApiError("Full-run env reset response did not include a state payload.")

        payload = {
            "action": "start_run",
            "character_id": str(character_id),
            "ascension": int(ascension_level),
        }
        if seed:
            payload["seed"] = str(seed)
        if normalized_build is not None:
            payload["build"] = normalized_build
        try:
            state = self._singleplayer.act(payload)
        except SingleplayerApiError:
            state = self._singleplayer.act({"action": "start_run"})
        wait_timeout = self.ready_timeout_s if timeout_s is None else float(timeout_s)
        return self.wait_until(
            lambda current: _state_type(current) != "menu",
            timeout_s=wait_timeout,
            initial_state=state,
        )

    def wait_until(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout_s: float | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timeout_s = self.ready_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        state = initial_state if initial_state is not None else self.get_state()
        while time.monotonic() < deadline:
            if predicate(state):
                return state
            time.sleep(self.poll_interval_s)
            state = self.get_state()
        raise SingleplayerTimeoutError(
            "Full-run env did not reach the requested state before timeout. "
            f"Last state: {json.dumps(state, ensure_ascii=True)}"
        )

    def wait_for_state_change(
        self,
        previous_state: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        previous_signature = json.dumps(previous_state, ensure_ascii=True, sort_keys=True)
        return self.wait_until(
            lambda current: json.dumps(current, ensure_ascii=True, sort_keys=True) != previous_signature,
            timeout_s=timeout_s,
            initial_state=previous_state,
        )

    def save_state(self) -> str:
        result = self._request_v2("POST", "/api/v2/full_run_env/save_state", {})
        state_id = result.get("state_id")
        if isinstance(state_id, str) and state_id:
            return state_id
        raise SingleplayerApiError("Full-run env save_state response did not include a state_id.")

    def export_state(self, path: str, *, state_id: str | None = None) -> str:
        payload: dict[str, Any] = {"path": str(path)}
        if state_id:
            payload["state_id"] = str(state_id)
        result = self._request_v2("POST", "/api/v2/full_run_env/export_state", payload)
        written_path = result.get("path")
        if isinstance(written_path, str) and written_path:
            return written_path
        raise SingleplayerApiError("Full-run env export_state response did not include a path.")

    def import_state(self, path: str) -> dict[str, Any]:
        result = self._request_v2("POST", "/api/v2/full_run_env/import_state", {"path": str(path)})
        if isinstance(result, dict) and isinstance(result.get("state"), dict):
            return result["state"]
        if isinstance(result, dict):
            return result
        raise SingleplayerApiError("Full-run env import_state response did not include a state payload.")

    def load_state(self, state_id: str) -> dict[str, Any]:
        result = self._request_v2("POST", "/api/v2/full_run_env/load_state", {"state_id": str(state_id)})
        if isinstance(result, dict) and isinstance(result.get("state"), dict):
            return result["state"]
        if isinstance(result, dict):
            return result
        raise SingleplayerApiError("Full-run env load_state response did not include a state payload.")

    def delete_state(self, state_id: str) -> bool:
        result = self._request_v2("POST", "/api/v2/full_run_env/delete_state", {"state_id": str(state_id)})
        return bool(result.get("deleted", False))

    def clear_state_cache(self) -> bool:
        result = self._request_v2("POST", "/api/v2/full_run_env/delete_state", {"clear_all": True})
        return bool(result.get("deleted", False))

    def legal_actions(self) -> list[dict[str, Any]]:
        state = self.get_state()
        legal = state.get("legal_actions")
        return legal if isinstance(legal, list) else []

    def perf_stats(self) -> dict[str, Any]:
        return {}

    def reset_perf_stats(self) -> bool:
        return False

    @property
    def supports_local_ort(self) -> bool:
        return False

    def load_ort_model(self, path: str) -> bool:
        raise SingleplayerApiError("Local ORT rollout requires proto pipe transport.")

    def run_combat_local(self, *, max_steps: int = 600) -> dict[str, Any]:
        raise SingleplayerApiError("Local ORT rollout requires proto pipe transport.")

    def search_combat_mcts(self, **kwargs: Any) -> dict[str, Any]:
        raise SingleplayerApiError("C# combat MCTS requires proto pipe transport.")

    def close(self) -> None:
        self._singleplayer.close()

    @property
    def transport_name(self) -> str:
        if self._use_v2 is None:
            return "http"
        return "http-v2-full-run-env" if self._use_v2 else "http-v1-singleplayer"

    @property
    def last_step_info(self) -> dict[str, Any] | None:
        if not isinstance(self._last_step_info, dict):
            return None
        return dict(self._last_step_info)

    def _should_use_v2(self) -> bool:
        if not self.prefer_v2:
            self._use_v2 = False
            return False
        if self._use_v2 is not None:
            return self._use_v2

        try:
            self._request_v2_state()
            self._use_v2 = True
        except SingleplayerApiError as exc:
            if _looks_like_missing_endpoint(exc):
                self._use_v2 = False
            else:
                raise
        return bool(self._use_v2)

    def _request_v2_state(self) -> dict[str, Any]:
        state = self._request_v2("GET", "/api/v2/full_run_env/state")
        if not isinstance(state, dict):
            raise SingleplayerApiError("Full-run env state response was not a JSON object.")
        return state

    def _request_v2(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._singleplayer._request_json(method, path, payload)


class FullRunClientLike(Protocol):
    poll_interval_s: float

    def reset(
        self,
        *,
        character_id: str = "IRONCLAD",
        ascension_level: int = 0,
        seed: str | None = None,
        build: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    def get_state(self) -> dict[str, Any]: ...

    def act(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def batch_act(self, actions: list[dict[str, Any]]) -> dict[str, Any]: ...

    def save_state(self) -> str: ...

    def export_state(self, path: str, *, state_id: str | None = None) -> str: ...

    def import_state(self, path: str) -> dict[str, Any]: ...

    def load_state(self, state_id: str) -> dict[str, Any]: ...

    def delete_state(self, state_id: str) -> bool: ...

    def clear_state_cache(self) -> bool: ...

    def legal_actions(self) -> list[dict[str, Any]]: ...

    def perf_stats(self) -> dict[str, Any]: ...

    def reset_perf_stats(self) -> bool: ...

    @property
    def supports_local_ort(self) -> bool: ...

    def load_ort_model(self, path: str) -> bool: ...

    def run_combat_local(self, *, max_steps: int = 600) -> dict[str, Any]: ...

    def search_combat_mcts(self, **kwargs: Any) -> dict[str, Any]: ...

    def wait_for_state_change(
        self,
        previous_state: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...

    @property
    def transport_name(self) -> str: ...

    @property
    def last_step_info(self) -> dict[str, Any] | None: ...


@dataclass(slots=True)
class PipeBackedFullRunClient:
    """Full-run client using named pipe IPC (~0.5ms/call vs ~24ms HTTP).

    Requires the Godot simulator to be running with pipe server enabled.
    In pure-sim mode all game logic is synchronous, with no polling needed.

    **2026-04-18 重构**:协议统一走
    `game_bridge.transport.PipeConnection` + codec (proto / json)。
    连接/重连/锁/heartbeat 全下沉,本类只管业务调用语义。

    **bin 协议废弃**:老 `BinaryPipeClient` 手写二进制 wire 已停止维护。
    诊断需求走 json,训练主路径走 proto。不再接受 `protocol="bin"` 构造。

    **不再有 HTTP fallback**:之前的 `_http_fallback` 字段从未被实例化过,是死代码
    2026-04-18 清理。HTTP 路径只在观战 mod (ApiBackedFullRunClient 独立使用)
    场景保留,sim 训练主路径纯 pipe。
    """
    port: int = 15527
    protocol: str = "proto"
    poll_interval_s: float = 0.0  # not used, kept for FullRunClientLike compat
    connect_timeout_s: float = 10.0
    auto_launch: bool = False
    repo_root: str | None = None
    host_path: str | None = None
    dll_path: str | None = None
    _conn: PipeConnection = field(init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)
    _last_step_info: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _owned_host_proc: Any | None = field(default=None, init=False, repr=False)

    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _dead: bool = field(default=False, init=False, repr=False)
    _DEAD_THRESHOLD: int = 3  # mark dead after N consecutive total failures

    def __post_init__(self) -> None:
        if self.repo_root is None:
            self.repo_root = str(DEFAULT_REPO_ROOT)
        if self.host_path is None:
            self.host_path = str(self.dll_path or DEFAULT_HOST_PATH)
        self._conn = self._build_client()

    def _normalized_protocol(self) -> str:
        """归一化 protocol 字段。只接受 proto / json;bin 明确拒绝。"""
        p = str(self.protocol).strip().lower()
        if p in {"proto", "protobuf", "pipe-proto"}:
            return "proto"
        if p in {"json", "pipe", "pipe-json"}:
            return "json"
        if p in {"bin", "binary", "pipe-binary"}:
            raise ValueError(
                "PipeBackedFullRunClient 不再支持 'bin' 协议(手写二进制 wire 已废弃)。"
                "训练请用 'proto',诊断用 'json'。"
            )
        raise ValueError(f"Unknown pipe protocol: {p!r}")

    def _build_client(self) -> PipeConnection:
        proto = self._normalized_protocol()

        def _launcher(port: int):
            proc = start_headless_sim(
                port=port,
                repo_root=self.repo_root,
                host_path=self.host_path,
                connect_timeout_s=max(15.0, float(self.connect_timeout_s)),
                protocol=proto,
            )
            self._owned_host_proc = proc
            return proc

        codec = ProtoCodec() if proto == "proto" else JsonCodec()
        cfg = PipeConnectionConfig(
            port=self.port,
            protocol=proto,
            connect_timeout_s=float(self.connect_timeout_s),
            auto_launch=bool(self.auto_launch and self.repo_root and self.host_path),
            sim_launcher=_launcher if (self.auto_launch and self.repo_root) else None,
            sim_stopper=stop_process if self.auto_launch else None,
            codec=codec,
        )
        return PipeConnection(cfg)

    @property
    def is_dead(self) -> bool:
        """True if this env has failed too many times and should be skipped."""
        return self._dead

    def _ensure_connected(self) -> None:
        if self._connected and self._conn_is_live():
            return
        try:
            self._connect_fresh(timeout_s=self.connect_timeout_s)
            self._consecutive_failures = 0
            self._dead = False
        except Exception:
            self._reconnect()

    def _conn_is_live(self) -> bool:
        try:
            return bool(self._conn.is_connected())
        except Exception:
            return False

    def _close_pipe_quietly(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        self._connected = False

    def _connect_fresh(self, *, timeout_s: float) -> None:
        new_client = self._build_client()
        try:
            new_client.cfg.connect_timeout_s = float(timeout_s)
            new_client.connect()
        except Exception:
            try:
                new_client.close()
            except Exception:
                pass
            raise
        self._conn = new_client
        self._connected = True

    def _restart_host_process(self) -> None:
        if not self.auto_launch or not self.repo_root or not self.host_path:
            raise RuntimeError("auto-launch host recovery is not configured")
        stop_process(self._owned_host_proc)
        self._owned_host_proc = start_headless_sim(
            port=self.port,
            repo_root=self.repo_root,
            host_path=self.host_path,
            connect_timeout_s=max(15.0, float(self.connect_timeout_s)),
            protocol=self._normalized_protocol(),
        )

    def _try_pipe_reconnect_cycle(self, *, timeouts: list[float]) -> None:
        last_error: Exception | None = None
        for attempt, timeout_s in enumerate(timeouts, start=1):
            try:
                self._connect_fresh(timeout_s=timeout_s)
                return
            except Exception as exc:
                last_error = exc
                if attempt < len(timeouts):
                    time.sleep(min(0.25 * attempt, 1.0))
        if last_error is None:
            raise ConnectionError(f"Pipe reconnect cycle failed on port {self.port}")
        raise last_error

    def _reconnect(self) -> None:
        """Force reconnect after pipe error (timeout, broken pipe, etc)."""
        log = logging.getLogger(__name__)
        self._close_pipe_quietly()
        last_error: Exception | None = None
        quick_timeouts = [1.0, 2.0, min(max(float(self.connect_timeout_s), 3.0), 5.0)]
        try:
            self._try_pipe_reconnect_cycle(timeouts=quick_timeouts)
            self._consecutive_failures = 0
            self._dead = False
            return
        except Exception as exc:
            last_error = exc
            log.warning("Pipe reconnect attempt failed on port %d: %s", self.port, exc)

        if self.auto_launch and self.repo_root and self.host_path:
            try:
                log.warning("Restarting HeadlessSim host on port %d after reconnect failures", self.port)
                self._restart_host_process()
                self._try_pipe_reconnect_cycle(timeouts=[2.0, 3.0, 5.0])
                self._consecutive_failures = 0
                self._dead = False
                log.info("Pipe recovered on port %d after host restart", self.port)
                return
            except Exception as exc:
                last_error = exc
                log.error("Host restart recovery failed on port %d: %s", self.port, exc)

        self._consecutive_failures += 1
        self._dead = self._consecutive_failures >= self._DEAD_THRESHOLD
        if self._dead:
            log.error(
                "Port %d marked DEAD after %d failed recovery cycles: %s",
                self.port,
                self._consecutive_failures,
                last_error,
            )
            raise ConnectionError(f"Port {self.port} is dead")
        log.warning(
            "Port %d recovery cycle failed (%d/%d); env remains retryable: %s",
            self.port,
            self._consecutive_failures,
            self._DEAD_THRESHOLD,
            last_error,
        )
        raise ConnectionError(f"Port {self.port} reconnect failed")

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """业务 RPC 入口,带自动重连。统一走 PipeConnection.safe_call。

        返回:业务 payload dict (和旧 API 一致)。
        """
        self._ensure_connected()
        try:
            try:
                result = self._conn.safe_call(method, params)
            except TransportSimulatorApiError as exc:
                raise SimulatorApiError(str(exc), error_code=exc.error_code) from exc
            except TransportTimeoutError as exc:
                raise TimeoutError(str(exc)) from exc
            return _unwrap_envelope(result)
        except (TimeoutError, ConnectionError, BrokenPipeError, ValueError, json.JSONDecodeError):
            self._reconnect()
            try:
                result = self._conn.safe_call(method, params)
            except TransportSimulatorApiError as exc:
                raise SimulatorApiError(str(exc), error_code=exc.error_code) from exc
            except TransportTimeoutError as exc:
                raise TimeoutError(str(exc)) from exc
            return _unwrap_envelope(result)

    def get_state(self) -> dict[str, Any]:
        return self._call("state")

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._last_step_info = None
        result = self._call("step", payload)
        info = result.get("info")
        if isinstance(info, dict):
            self._last_step_info = dict(info)
        if not bool(result.get("accepted", False)):
            state = result.get("state")
            failure_code = (
                result.get("failure_code")
                or result.get("failureCode")
                or result.get("FailureCode")
            )
            error_text = str(result.get("error") or "Unknown full-run env step error")
            if failure_code:
                error_text = f"{error_text} [{failure_code}]"
            error = SingleplayerApiError(
                error_text
            )
            if isinstance(state, dict):
                setattr(error, "latest_state", state)
            if isinstance(info, dict):
                setattr(error, "step_info", info)
            if isinstance(failure_code, str) and failure_code:
                setattr(error, "failure_code", failure_code)
            raise error
        state = result.get("state")
        if isinstance(state, dict):
            return state
        raise SingleplayerApiError("Pipe step response did not include a state payload.")

    def batch_act(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        result = self._call("batch_step", {"actions": actions})
        if not bool(result.get("accepted", False)):
            error = SingleplayerApiError(str(result.get("error") or "Batch step error"))
            setattr(error, "latest_state", result)
            setattr(error, "steps_executed", result.get("steps_executed", 0))
            raise error
        return result

    def reset(
        self,
        *,
        character_id: str = "IRONCLAD",
        ascension_level: int = 0,
        seed: str | None = None,
        build: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        normalized_build = _normalize_build_spec(build)
        params: dict[str, Any] = {
            "character_id": str(character_id),
            "ascension_level": int(ascension_level),
        }
        if seed:
            params["seed"] = str(seed)
        if normalized_build is not None:
            params["build"] = normalized_build
        state = self._call("reset", params)
        if isinstance(state, dict):
            return state
        raise SingleplayerApiError("Pipe reset response did not include a state payload.")

    def wait_for_state_change(
        self,
        previous_state: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        # In pure-sim mode, state is already settled after step/reset.
        # Just return current state.
        return self.get_state()

    def save_state(self) -> str:
        result = self._call("save_state")
        state_id = result.get("state_id")
        if isinstance(state_id, str) and state_id:
            return state_id
        raise SingleplayerApiError("Pipe save_state response did not include a state_id.")

    def export_state(self, path: str, *, state_id: str | None = None) -> str:
        params: dict[str, Any] = {"path": str(path)}
        if state_id:
            params["state_id"] = str(state_id)
        result = self._call("export_state", params)
        written_path = result.get("path")
        if isinstance(written_path, str) and written_path:
            return written_path
        raise SingleplayerApiError("Pipe export_state response did not include a path.")

    def import_state(self, path: str) -> dict[str, Any]:
        result = self._call("import_state", {"path": str(path)})
        if isinstance(result, dict):
            return result
        raise SingleplayerApiError("Pipe import_state response did not include a state payload.")

    def load_state(self, state_id: str) -> dict[str, Any]:
        result = self._call("load_state", {"state_id": str(state_id)})
        if isinstance(result, dict):
            return result
        raise SingleplayerApiError("Pipe load_state response did not include a state payload.")

    def delete_state(self, state_id: str) -> bool:
        result = self._call("delete_state", {"state_id": str(state_id)})
        return bool(result.get("deleted", False))

    def clear_state_cache(self) -> bool:
        result = self._call("delete_state", {"clear_all": True})
        return bool(result.get("deleted", False))

    def legal_actions(self) -> list[dict[str, Any]]:
        result = self._call("legal_actions")
        legal = result.get("legal_actions")
        return legal if isinstance(legal, list) else []

    def perf_stats(self) -> dict[str, Any]:
        result = self._call("perf_stats")
        return result if isinstance(result, dict) else {}

    def reset_perf_stats(self) -> bool:
        result = self._call("reset_perf_stats")
        return bool(result.get("reset", False))

    @property
    def supports_local_ort(self) -> bool:
        # proto 走 opcode-based request 编码,可驱动 sim 的本地 ORT 接口
        return self._normalized_protocol() == "proto"

    def load_ort_model(self, path: str) -> bool:
        if not self.supports_local_ort:
            raise SingleplayerApiError("Local ORT rollout requires proto pipe transport.")
        result = self._call("load_ort_model", {"path": str(path)})
        return bool(result.get("loaded", False))

    def run_combat_local(self, *, max_steps: int = 600) -> dict[str, Any]:
        if not self.supports_local_ort:
            raise SingleplayerApiError("Local ORT rollout requires proto pipe transport.")
        result = self._call("run_combat_local", {"max_steps": int(max_steps)})
        return result if isinstance(result, dict) else {}

    def search_combat_mcts(self, **kwargs: Any) -> dict[str, Any]:
        if not self.supports_local_ort:
            raise SingleplayerApiError("C# combat MCTS requires proto pipe transport.")
        result = self._call("search_combat_mcts", kwargs)
        return result if isinstance(result, dict) else {}

    def close(self) -> None:
        self._close_pipe_quietly()
        stop_process(self._owned_host_proc)
        self._owned_host_proc = None
        # reset dead state so client 重新构造 / resume 训练时可以再次尝试连接
        self._consecutive_failures = 0
        self._dead = False

    @property
    def transport_name(self) -> str:
        proto = self._normalized_protocol()
        return "pipe-proto" if proto == "proto" else "pipe"

    @property
    def last_step_info(self) -> dict[str, Any] | None:
        if not isinstance(self._last_step_info, dict):
            return None
        return dict(self._last_step_info)


@dataclass(slots=True)
class BinaryBackedFullRunClient(PipeBackedFullRunClient):
    """训练主客户端,协议固定为 proto。

    class 名保留 "Binary" 前缀仅向后兼容旧 import;底层已是 proto pipe,
    **不会** 走废弃的手写二进制 wire。
    """
    protocol: str = "proto"


def _resolve_pipe_protocol(transport: str | None) -> str:
    """归一化 transport → pipe protocol。只支持 proto / json。

    2026-04-18: bin 协议(手写二进制 wire)正式废弃;传 "bin" 会 raise。
    """
    t = str(transport or "").strip().lower()
    if t in {"pipe-bin", "pipe-binary", "bin", "binary"}:
        raise ValueError(
            "pipe-binary transport is deprecated. Use 'proto' (training) or 'json' (diagnostics)."
        )
    if t in {"pipe-proto", "pipe-protobuf", "proto", "protobuf"}:
        return "proto"
    if t in {"pipe", "pipe-json", "json"}:
        return "json"
    # 默认(空或含糊值)→ proto
    return "proto"


def create_full_run_client(
    *,
    base_url: str = "http://127.0.0.1:15526",
    port: int | None = None,
    use_pipe: bool = False,
    transport: str | None = None,
    poll_interval_s: float = 0.05,
    request_timeout_s: float = 10.0,
    ready_timeout_s: float = 20.0,
    prefer_v2: bool = True,
    auto_launch: bool = False,
    repo_root: str | None = None,
    host_path: str | None = None,
    dll_path: str | None = None,
) -> FullRunClientLike:
    if auto_launch and not use_pipe:
        raise ValueError("auto_launch requires use_pipe=True for game_bridge full-run sessions.")
    if use_pipe:
        pipe_port = port if port is not None else int(base_url.rsplit(":", 1)[-1].split("/")[0])
        return PipeBackedFullRunClient(
            port=pipe_port,
            connect_timeout_s=ready_timeout_s,
            protocol=_resolve_pipe_protocol(transport),
            auto_launch=auto_launch,
            repo_root=repo_root,
            host_path=host_path,
            dll_path=dll_path,
        )
    return ApiBackedFullRunClient(
        base_url=base_url,
        poll_interval_s=poll_interval_s,
        request_timeout_s=request_timeout_s,
        ready_timeout_s=ready_timeout_s,
        prefer_v2=prefer_v2,
    )
