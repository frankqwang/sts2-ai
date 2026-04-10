from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from full_run_env import ApiBackedFullRunClient, FullRunClientLike
from sts2_singleplayer_env import SingleplayerApiError


def is_wait_action(action: dict[str, Any] | None) -> bool:
    return str((action or {}).get("action") or "").lower() == "wait"


@dataclass(slots=True)
class FullRunBackendAdapter:
    """Backend-agnostic state advancement helper.

    Upper layers should depend on this adapter instead of hard-coding visible
    UI quirks or pure-sim assumptions. The goal is:
    - same branching / replay / rollout logic across backends
    - backend-specific settle / stale-state recovery only here
    """

    client: FullRunClientLike
    wait_timeout_s: float = 1.0

    @property
    def is_http_backend(self) -> bool:
        return isinstance(self.client, ApiBackedFullRunClient)

    def apply_action(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
    ) -> dict[str, Any]:
        if is_wait_action(action):
            return self._apply_wait(state)

        try:
            return self.client.act(action)
        except SingleplayerApiError as exc:
            if self._should_retry_after_stale(exc):
                return self._recover_after_stale(state)
            raise

    def _apply_wait(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.is_http_backend:
            # Preferred path once the HTTP server supports explicit wait.
            try:
                return self.client.act({"action": "wait"})
            except Exception:
                pass
            try:
                return self.client.wait_for_state_change(state, timeout_s=self.wait_timeout_s)
            except Exception:
                return self.client.get_state()
        return self.client.act({"action": "wait"})

    def _should_retry_after_stale(self, exc: SingleplayerApiError) -> bool:
        text = str(exc).strip().lower()
        return any(
            marker in text
            for marker in (
                "screen is not open",
                "state change timeout",
                "not in combat",
                "player actions are currently disabled",
                "turn may already be ending",
            )
        )

    def _recover_after_stale(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.client.wait_for_state_change(state, timeout_s=self.wait_timeout_s)
        except Exception:
            return self.client.get_state()


def apply_backend_action(
    client: FullRunClientLike,
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    wait_timeout_s: float = 1.0,
) -> dict[str, Any]:
    return FullRunBackendAdapter(client=client, wait_timeout_s=wait_timeout_s).apply_action(state, action)
