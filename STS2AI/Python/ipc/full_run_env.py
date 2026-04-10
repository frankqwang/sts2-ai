from __future__ import annotations

import argparse
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from sts2_singleplayer_env import (
    SingleplayerApiError,
    SingleplayerClient,
    SingleplayerConnectionError,
    SingleplayerTimeoutError,
)
from binary_pipe_client import BinaryPipeClient
from pipe_client import PipeClient
from simulator_api_error import SimulatorApiError

logger = logging.getLogger(__name__)


class FullRunEnvError(RuntimeError):
    pass


class FullRunEnv(ABC):
    @abstractmethod
    def reset(
        self,
        *,
        character_id: str = "IRONCLAD",
        ascension_level: int = 0,
        seed: str | None = None,
        auto_start_from_menu: bool | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Reset env for a new full-run episode and return initial state."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Fetch the latest raw environment state snapshot."""

    @abstractmethod
    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Apply one action and return (state, reward, done, info)."""

    @abstractmethod
    def close(self) -> None:
        """Release client resources."""


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
        main-thread call — no per-action HTTP or frame overhead.

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
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        if self._should_use_v2():
            payload: dict[str, Any] = {
                "character_id": str(character_id),
                "ascension": int(ascension_level),
            }
            if seed:
                payload["seed"] = str(seed)
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
        raise SingleplayerApiError("Local ORT rollout is only supported on pipe-binary clients.")

    def run_combat_local(self, *, max_steps: int = 600) -> dict[str, Any]:
        raise SingleplayerApiError("Local ORT rollout is only supported on pipe-binary clients.")

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
    In pure-sim mode all game logic is synchronous — no polling needed.

    If pipe connection fails, temporarily falls back to HTTP and periodically
    retries pipe reconnection (every ``_PIPE_RETRY_INTERVAL`` calls).
    """
    port: int = 15527
    protocol: str = "json"
    poll_interval_s: float = 0.0  # not used, kept for FullRunClientLike compat
    connect_timeout_s: float = 10.0
    _pipe: PipeClient | BinaryPipeClient = field(init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)
    _last_step_info: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _http_fallback: ApiBackedFullRunClient | None = field(default=None, init=False, repr=False)
    _call_count_since_fallback: int = field(default=0, init=False, repr=False)

    _PIPE_RETRY_INTERVAL: int = 50  # retry pipe every N calls while on HTTP
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _dead: bool = field(default=False, init=False, repr=False)
    _DEAD_THRESHOLD: int = 3  # mark dead after N consecutive total failures

    def __post_init__(self) -> None:
        self._pipe = self._new_pipe_client()

    def _normalized_protocol(self) -> str:
        return "bin" if str(self.protocol).strip().lower() in {"bin", "binary", "pipe-binary"} else "json"

    def _new_pipe_client(self) -> PipeClient | BinaryPipeClient:
        return BinaryPipeClient(port=self.port) if self._normalized_protocol() == "bin" else PipeClient(port=self.port)

    @property
    def is_dead(self) -> bool:
        """True if this env has failed too many times and should be skipped."""
        return self._dead

    def _ensure_connected(self) -> None:
        if not self._connected:
            self._pipe.connect(timeout_s=self.connect_timeout_s)
            self._connected = True

    def _maybe_retry_pipe(self) -> None:
        """Periodically try to recover pipe connection while on HTTP fallback."""
        if self._http_fallback is None:
            return
        self._call_count_since_fallback += 1
        if self._call_count_since_fallback < self._PIPE_RETRY_INTERVAL:
            return
        self._call_count_since_fallback = 0
        try:
            new_pipe = self._new_pipe_client()
            new_pipe.connect(timeout_s=3.0)
            # Success — switch back to pipe
            self._pipe = new_pipe
            self._http_fallback = None
            self._connected = True
            import logging
            logging.getLogger(__name__).info(
                "Pipe recovered on port %d, switching back from HTTP", self.port)
        except Exception:
            pass  # stay on HTTP, will retry later

    def _reconnect(self) -> None:
        """Force reconnect after pipe error (timeout, broken pipe, etc).

        Tries pipe first (fast, 2 attempts); if that fails, tries HTTP.
        If both fail repeatedly, marks this env as dead to avoid blocking.
        """
        try:
            self._pipe.close()
        except Exception:
            pass
        import time, logging
        log = logging.getLogger(__name__)

        # Try pipe reconnect — one shot, 1s timeout (normal connect <50ms)
        try:
            time.sleep(0.3)
            self._pipe = self._new_pipe_client()
            self._pipe.connect(timeout_s=1.0)
            self._connected = True
            self._consecutive_failures = 0
            return
        except Exception:
            try:
                self._pipe.close()
            except Exception:
                pass

        # Pipe failed — mark dead immediately, no HTTP fallback
        self._consecutive_failures += 1
        self._dead = True
        log.error("Port %d marked DEAD (pipe reconnect failed, attempt %d)",
                  self.port, self._consecutive_failures)
        raise ConnectionError(f"Port {self.port} is dead")

    def get_state(self) -> dict[str, Any]:
        self._ensure_connected()
        self._maybe_retry_pipe()
        if self._http_fallback is not None:
            return self._http_fallback.get_state()
        try:
            return self._pipe.call("state")
        except (TimeoutError, ConnectionError, BrokenPipeError):
            self._reconnect()
            if self._http_fallback is not None:
                return self._http_fallback.get_state()
            return self._pipe.call("state")

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_connected()
        self._last_step_info = None
        self._maybe_retry_pipe()
        if self._http_fallback is not None:
            return self._http_fallback.act(payload)
        try:
            result = self._pipe.call("step", payload)
        except (TimeoutError, ConnectionError, BrokenPipeError):
            self._reconnect()
            if self._http_fallback is not None:
                return self._http_fallback.act(payload)
            result = self._pipe.call("step", payload)
        info = result.get("info")
        if isinstance(info, dict):
            self._last_step_info = dict(info)
        if not bool(result.get("accepted", False)):
            # C# sometimes returns accepted=False even when the action executed
            # successfully (e.g., choose_map_node transitions to combat but reports
            # not accepted). If we got a valid state back, use it anyway.
            state = result.get("state")
            if isinstance(state, dict) and state.get("state_type"):
                logger.debug("Pipe step not accepted but got valid state: %s",
                             state.get("state_type"))
                return state
            error = SingleplayerApiError(
                str(result.get("error") or "Unknown full-run env step error")
            )
            if isinstance(state, dict):
                setattr(error, "latest_state", state)
            if isinstance(info, dict):
                setattr(error, "step_info", info)
            raise error
        state = result.get("state")
        if isinstance(state, dict):
            return state
        raise SingleplayerApiError("Pipe step response did not include a state payload.")

    def batch_act(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_connected()
        result = self._pipe.call("batch_step", {"actions": actions})
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
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self._ensure_connected()
        params: dict[str, Any] = {
            "character_id": str(character_id),
            "ascension_level": int(ascension_level),
        }
        if seed:
            params["seed"] = str(seed)
        self._maybe_retry_pipe()
        if self._http_fallback is not None:
            return self._http_fallback.reset(
                character_id=character_id, ascension_level=ascension_level,
                seed=seed, timeout_s=timeout_s)
        try:
            state = self._pipe.call("reset", params)
        except (TimeoutError, ConnectionError, BrokenPipeError):
            self._reconnect()
            if self._http_fallback is not None:
                return self._http_fallback.reset(
                    character_id=character_id, ascension_level=ascension_level,
                    seed=seed, timeout_s=timeout_s)
            state = self._pipe.call("reset", params)
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
        self._ensure_connected()
        result = self._pipe.call("save_state")
        state_id = result.get("state_id")
        if isinstance(state_id, str) and state_id:
            return state_id
        raise SingleplayerApiError("Pipe save_state response did not include a state_id.")

    def export_state(self, path: str, *, state_id: str | None = None) -> str:
        self._ensure_connected()
        params: dict[str, Any] = {"path": str(path)}
        if state_id:
            params["state_id"] = str(state_id)
        result = self._pipe.call("export_state", params)
        written_path = result.get("path")
        if isinstance(written_path, str) and written_path:
            return written_path
        raise SingleplayerApiError("Pipe export_state response did not include a path.")

    def import_state(self, path: str) -> dict[str, Any]:
        self._ensure_connected()
        result = self._pipe.call("import_state", {"path": str(path)})
        if isinstance(result, dict):
            return result
        raise SingleplayerApiError("Pipe import_state response did not include a state payload.")

    def load_state(self, state_id: str) -> dict[str, Any]:
        self._ensure_connected()
        result = self._pipe.call("load_state", {"state_id": str(state_id)})
        if isinstance(result, dict):
            return result
        raise SingleplayerApiError("Pipe load_state response did not include a state payload.")

    def delete_state(self, state_id: str) -> bool:
        self._ensure_connected()
        result = self._pipe.call("delete_state", {"state_id": str(state_id)})
        return bool(result.get("deleted", False))

    def clear_state_cache(self) -> bool:
        self._ensure_connected()
        result = self._pipe.call("delete_state", {"clear_all": True})
        return bool(result.get("deleted", False))

    def legal_actions(self) -> list[dict[str, Any]]:
        self._ensure_connected()
        result = self._pipe.call("legal_actions")
        legal = result.get("legal_actions")
        return legal if isinstance(legal, list) else []

    def perf_stats(self) -> dict[str, Any]:
        self._ensure_connected()
        result = self._pipe.call("perf_stats")
        return result if isinstance(result, dict) else {}

    def reset_perf_stats(self) -> bool:
        self._ensure_connected()
        result = self._pipe.call("reset_perf_stats")
        return bool(result.get("reset", False))

    @property
    def supports_local_ort(self) -> bool:
        return self._normalized_protocol() == "bin"

    def load_ort_model(self, path: str) -> bool:
        if not self.supports_local_ort:
            raise SingleplayerApiError("Local ORT rollout requires pipe-binary transport.")
        self._ensure_connected()
        result = self._pipe.call("load_ort_model", {"path": str(path)})
        return bool(result.get("loaded", False))

    def run_combat_local(self, *, max_steps: int = 600) -> dict[str, Any]:
        if not self.supports_local_ort:
            raise SingleplayerApiError("Local ORT rollout requires pipe-binary transport.")
        self._ensure_connected()
        result = self._pipe.call("run_combat_local", {"max_steps": int(max_steps)})
        return result if isinstance(result, dict) else {}

    def close(self) -> None:
        if self._connected:
            self._pipe.close()
            self._http_fallback = None
            self._connected = False

    @property
    def transport_name(self) -> str:
        return "pipe-binary" if self._normalized_protocol() == "bin" else "pipe"

    @property
    def last_step_info(self) -> dict[str, Any] | None:
        if not isinstance(self._last_step_info, dict):
            return None
        return dict(self._last_step_info)


@dataclass(slots=True)
class BinaryBackedFullRunClient(PipeBackedFullRunClient):
    protocol: str = "bin"


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
) -> FullRunClientLike:
    if use_pipe:
        pipe_port = port if port is not None else int(base_url.rsplit(":", 1)[-1].split("/")[0])
        return PipeBackedFullRunClient(
            port=pipe_port,
            connect_timeout_s=ready_timeout_s,
            protocol="bin" if str(transport or "").strip().lower() == "pipe-binary" else "json",
        )
    return ApiBackedFullRunClient(
        base_url=base_url,
        poll_interval_s=poll_interval_s,
        request_timeout_s=request_timeout_s,
        ready_timeout_s=ready_timeout_s,
        prefer_v2=prefer_v2,
    )


@dataclass(slots=True)
class SingleplayerFullRunEnv(FullRunEnv):
    base_url: str = "http://127.0.0.1:15526"
    poll_interval_s: float = 0.05
    request_timeout_s: float = 10.0
    ready_timeout_s: float = 20.0
    auto_start_from_menu: bool = True
    prefer_v2: bool = True
    _client: ApiBackedFullRunClient = field(init=False, repr=False)
    _has_entered_run: bool = field(default=False, init=False, repr=False)
    _last_state: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = ApiBackedFullRunClient(
            base_url=self.base_url,
            poll_interval_s=self.poll_interval_s,
            request_timeout_s=self.request_timeout_s,
            ready_timeout_s=self.ready_timeout_s,
            prefer_v2=self.prefer_v2,
        )

    def reset(
        self,
        *,
        character_id: str = "IRONCLAD",
        ascension_level: int = 0,
        seed: str | None = None,
        auto_start_from_menu: bool | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        state = self.get_state()
        if _state_type(state) != "menu":
            self._has_entered_run = True
            return state

        should_auto_start = self.auto_start_from_menu if auto_start_from_menu is None else bool(auto_start_from_menu)
        if not should_auto_start:
            return state
        state = self._client.reset(
            character_id=character_id,
            ascension_level=int(ascension_level),
            seed=seed,
            timeout_s=timeout_s,
        )
        self._has_entered_run = True
        self._last_state = state
        return state

    def get_state(self) -> dict[str, Any]:
        state = self._client.get_state()
        if _state_type(state) != "menu":
            self._has_entered_run = True
        self._last_state = state
        return state

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        next_state = self._client.act(action)
        state_type = _state_type(next_state)
        if state_type != "menu":
            self._has_entered_run = True

        run_outcome = _extract_run_outcome(next_state)
        done = bool(state_type == "game_over" or (state_type == "menu" and self._has_entered_run))
        reward = 0.0
        if done:
            if run_outcome and ("victory" in run_outcome or run_outcome == "win"):
                reward = 1.0
            elif run_outcome:
                reward = -1.0

        info: dict[str, Any] = {
            "accepted": True,
            "state_type": state_type,
            "run_outcome": run_outcome,
            "transport_name": self._client.transport_name,
        }
        step_info = self._client.last_step_info
        if isinstance(step_info, dict):
            info["step_info"] = step_info
        self._last_state = next_state
        return next_state, reward, done, info

    def close(self) -> None:
        self._client.close()


def connect(base_url: str = "http://127.0.0.1:15526") -> SingleplayerFullRunEnv:
    env = SingleplayerFullRunEnv(base_url=base_url)
    env.get_state()
    return env


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-run env adapter that prefers /api/v2/full_run_env and falls back to /api/v1/singleplayer.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:15526", help="STS2MCP HTTP base URL.")
    parser.add_argument("--character-id", default="IRONCLAD", help="Character for reset(start_run).")
    parser.add_argument("--ascension-level", type=int, default=0, help="Ascension for reset(start_run).")
    parser.add_argument("--seed", default=None, help="Optional seed for reset(start_run).")
    parser.add_argument("--no-auto-start", action="store_true", help="Do not start a run from menu on reset().")
    parser.add_argument("--force-v1", action="store_true", help="Disable /api/v2/full_run_env and always use /api/v1/singleplayer.")
    parser.add_argument("--step-json", default=None, help="Optional JSON action to send through step().")
    parser.add_argument("--print-state-json", action="store_true", help="Print full state JSON payload(s).")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    env = SingleplayerFullRunEnv(
        base_url=args.base_url,
        auto_start_from_menu=not args.no_auto_start,
        prefer_v2=not args.force_v1,
    )
    try:
        state = env.reset(
            character_id=args.character_id,
            ascension_level=int(args.ascension_level),
            seed=args.seed,
        )
        summary = {
            "event": "reset",
            "state_type": _state_type(state),
            "run": state.get("run"),
        }
        print(json.dumps(summary, ensure_ascii=True))
        if args.print_state_json:
            print(json.dumps(state, ensure_ascii=True))

        if args.step_json:
            action = json.loads(args.step_json)
            if not isinstance(action, dict):
                raise FullRunEnvError("--step-json must decode to a JSON object.")
            next_state, reward, done, info = env.step(action)
            step_summary = {
                "event": "step",
                "reward": reward,
                "done": done,
                "info": info,
                "state_type": _state_type(next_state),
                "run": next_state.get("run"),
            }
            print(json.dumps(step_summary, ensure_ascii=True))
            if args.print_state_json:
                print(json.dumps(next_state, ensure_ascii=True))
        return 0
    except (SingleplayerConnectionError, SingleplayerApiError, SingleplayerTimeoutError, FullRunEnvError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
