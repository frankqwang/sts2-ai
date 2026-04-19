"""STS2 MCP 单人自动化 API 的 HTTP 客户端（state/act/wait）。"""
from __future__ import annotations

import json
import time
import http.client
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class SingleplayerAutomationError(RuntimeError):
    pass


class SingleplayerConnectionError(SingleplayerAutomationError):
    pass


class SingleplayerApiError(SingleplayerAutomationError):
    pass


class SingleplayerTimeoutError(SingleplayerAutomationError):
    pass


COMBAT_STATE_TYPES = {"monster", "elite", "boss", "hand_select"}


def _lower(value: Any) -> str:
    return str(value or "").lower()


@dataclass(slots=True)
class SingleplayerClient:
    base_url: str = "http://127.0.0.1:15526"
    poll_interval_s: float = 0.15
    request_timeout_s: float = 10.0
    ready_timeout_s: float = 20.0
    request_retry_count: int = 2
    request_retry_delay_s: float = 0.2

    def get_state(self) -> dict[str, Any]:
        return self._request_json("GET", "/api/v1/singleplayer")

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        # The v1 API returns an action status object on POST, not the updated game state.
        # Follow the action with a fresh GET so callers always receive a state snapshot.
        self._request_json("POST", "/api/v1/singleplayer", payload)
        state = self.get_state()
        if self._should_wait_for_post_action_combat_settle(state):
            state = self.wait_until(
                lambda current: self._is_post_action_combat_settled(current),
                timeout_s=self.ready_timeout_s,
                initial_state=state,
            )
        return state

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
            "Singleplayer API did not reach the requested state before timeout. "
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

    def wait_until_actionable_combat(
        self,
        *,
        timeout_s: float | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.wait_until(
            lambda current: self._is_actionable_combat_state(current),
            timeout_s=timeout_s,
            initial_state=initial_state,
        )

    def close(self) -> None:
        return None

    def _should_wait_for_post_action_combat_settle(self, state: dict[str, Any]) -> bool:
        return self._is_combat_state(state) and not self._is_actionable_combat_state(state)

    def _is_post_action_combat_settled(self, state: dict[str, Any]) -> bool:
        return not self._is_combat_state(state) or self._is_actionable_combat_state(state)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        retry_budget = max(0, int(self.request_retry_count)) if method.upper() == "GET" else 0
        body = ""
        for attempt in range(retry_budget + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                    body = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                # Retry transient server errors (500) on GET requests
                if exc.code >= 500 and attempt < retry_budget:
                    time.sleep(max(0.0, float(self.request_retry_delay_s)))
                    continue
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("error"):
                    raise SingleplayerApiError(parsed["error"]) from exc
                raise SingleplayerApiError(f"HTTP {exc.code}: {body}") from exc
            except urllib.error.URLError as exc:
                if attempt < retry_budget:
                    time.sleep(max(0.0, float(self.request_retry_delay_s)))
                    continue
                raise SingleplayerConnectionError(
                    f"Could not connect to singleplayer API at {url}. Make sure STS2MCP is loaded."
                ) from exc
            except (ConnectionResetError, ConnectionAbortedError, http.client.RemoteDisconnected, TimeoutError, OSError) as exc:
                if attempt < retry_budget:
                    time.sleep(max(0.0, float(self.request_retry_delay_s)))
                    continue
                raise SingleplayerConnectionError(
                    f"Singleplayer API request to {url} was interrupted."
                ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SingleplayerApiError(f"Invalid JSON response from {url}: {body}") from exc

        if isinstance(parsed, dict) and parsed.get("status") == "error":
            raise SingleplayerApiError(parsed.get("error", "Unknown singleplayer API error"))
        if not isinstance(parsed, dict):
            raise SingleplayerApiError(f"Expected JSON object from {url}, got {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _is_actionable_combat_state(state: dict[str, Any]) -> bool:
        state_type = _lower(state.get("state_type"))
        if state_type not in COMBAT_STATE_TYPES:
            return False
        if state_type == "hand_select":
            return True
        if isinstance(state.get("card_selection"), dict):
            return True
        battle = state.get("battle") or {}
        if not (bool(battle.get("is_play_phase")) and _lower(battle.get("turn")) == "player"):
            return False

        player = battle.get("player") or state.get("player") or {}
        hand = list(player.get("hand") or [])
        if not hand:
            return True

        if any(bool(card.get("can_play")) for card in hand):
            return True

        reasons = {_lower(card.get("unplayable_reason")) for card in hand if card.get("unplayable_reason") is not None}
        if reasons and reasons.issubset({"playeractionsdisabled", "disabled", "none"}):
            return False

        return True

    @staticmethod
    def _is_combat_state(state: dict[str, Any]) -> bool:
        return _lower(state.get("state_type")) in COMBAT_STATE_TYPES
