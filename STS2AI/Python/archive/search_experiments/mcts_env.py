"""MCTS environment wrapper for STS2 simulator.

Provides save/load state for tree search, wrapping the HTTP API.

Usage:
    env = MctsEnv(port=15527)
    env.reset(seed="TEST")
    sid = env.save()          # snapshot
    env.step(action)          # take action
    env.load(sid)             # restore to snapshot
    actions = env.legal_actions()  # get legal moves
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

try:
    from .simulator_api_error import SimulatorApiError
except ImportError:
    from simulator_api_error import SimulatorApiError


class MctsEnv:
    """MCTS-compatible environment backed by STS2 simulator HTTP API."""

    def __init__(self, base_url: str = "http://127.0.0.1:15527",
                 timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._state: dict[str, Any] | None = None

    def _request(self, method: str, path: str,
                 payload: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read()
            message = exc.reason or f"HTTP {exc.code}"
            error_code = None
            try:
                payload = json.loads(body.decode("utf-8"))
                if isinstance(payload, dict):
                    message = payload.get("error") or message
                    error_code = payload.get("error_code")
            except Exception:
                pass
            raise SimulatorApiError(message, error_code=error_code, status_code=exc.code) from exc

    def reset(self, character_id: str = "IRONCLAD",
              ascension_level: int = 0,
              seed: str | None = None) -> dict:
        """Start a new run. Returns initial state."""
        payload: dict[str, Any] = {
            "character_id": character_id,
            "ascension_level": ascension_level,
        }
        if seed:
            payload["seed"] = seed
        self._state = self._request("POST", "/api/v2/full_run_env/reset", payload)
        return self._state

    def step(self, action: dict) -> dict:
        """Execute an action. Returns next state."""
        # Strip non-action fields (label, note, is_enabled, etc.) to avoid C# parsing issues
        clean = {k: v for k, v in action.items()
                 if k in ("action", "index", "card_index", "hand_index",
                          "slot", "target_id", "target", "col", "row", "value")}
        result = self._request("POST", "/api/v2/full_run_env/step", clean)
        # The step endpoint wraps state in a result envelope
        if "state" in result and isinstance(result["state"], dict):
            self._state = result["state"]
        else:
            self._state = result
        return self._state

    def get_state(self) -> dict:
        """Get current state without acting."""
        self._state = self._request("GET", "/api/v2/full_run_env/state")
        return self._state

    def save(self) -> str:
        """Save current game state. Returns state_id for later restore."""
        result = self._request("POST", "/api/v2/full_run_env/save_state")
        return result["state_id"]

    def load(self, state_id: str) -> dict:
        """Restore game to a previously saved state."""
        self._state = self._request(
            "POST", "/api/v2/full_run_env/load_state",
            {"state_id": state_id},
        )
        return self._state

    def delete(self, state_id: str) -> bool:
        """Delete a saved state snapshot."""
        result = self._request(
            "POST", "/api/v2/full_run_env/delete_state",
            {"state_id": state_id},
        )
        return result.get("deleted", False)

    def clear_cache(self) -> None:
        """Clear all saved state snapshots."""
        self._request(
            "POST", "/api/v2/full_run_env/delete_state",
            {"clear_all": True},
        )

    def legal_actions(self) -> list[dict]:
        """Get legal actions from cached state."""
        if self._state is None:
            self.get_state()
        actions = self._state.get("legal_actions") or []
        # Filter out unsupported actions
        return [a for a in actions
                if isinstance(a, dict) and a.get("is_enabled") is not False]

    @property
    def state_type(self) -> str:
        if self._state is None:
            return ""
        return (self._state.get("state_type") or "").lower()

    @property
    def is_terminal(self) -> bool:
        if self._state is None:
            return False
        return bool(self._state.get("terminal"))

    @property
    def run_outcome(self) -> str | None:
        if self._state is None:
            return None
        return self._state.get("run_outcome")


def test_save_load(env: MctsEnv, seed: str = "MCTS_TEST") -> bool:
    """Test save/load consistency."""
    print("Testing save/load consistency...")
    env.reset(seed=seed)

    state_before = env.get_state()
    if not _supports_exact_snapshot(state_before.get("state_type")):
        raise RuntimeError(f"Expected non-combat state after reset, got {state_before.get('state_type')}")

    sid = env.save()
    sig_before = build_state_signature(state_before)

    print(f"  Saved at state_type={env.state_type}, legal={len(env.legal_actions())}")

    actions = env.legal_actions()
    if actions and not env.is_terminal:
        env.step(actions[0])

    print(f"  After one step: state_type={env.state_type}")

    # Restore
    state_after = env.load(sid)
    sig_after = build_state_signature(state_after)

    print(f"  After load: state_type={env.state_type}, legal={len(env.legal_actions())}")

    matches = sig_before == sig_after
    print(f"  Signature match: {matches}")

    # Clean up
    env.delete(sid)
    print(f"  Deleted snapshot {sid}")

    return matches


def _supports_exact_snapshot(state_type: str | None) -> bool:
    return (state_type or "").lower() not in {"monster", "elite", "boss", "hand_select"}


def build_state_signature(state: dict) -> str:
    """Stable signature for non-combat save/load exactness checks."""
    run = state.get("run") or {}
    player = _extract_player(state)
    legal_actions = state.get("legal_actions") or []

    payload: dict[str, Any] = {
        "state_type": state.get("state_type"),
        "act": run.get("act"),
        "floor": run.get("floor"),
        "seed": run.get("seed"),
        "player": {
            "hp": player.get("hp", player.get("current_hp")),
            "max_hp": player.get("max_hp"),
            "gold": player.get("gold"),
        },
        "legal_actions": [
            {
                "action": action.get("action"),
                "index": action.get("index"),
                "card_index": action.get("card_index"),
                "col": action.get("col"),
                "row": action.get("row"),
                "slot": action.get("slot"),
                "target_id": action.get("target_id"),
                "target": action.get("target"),
                "label": action.get("label"),
            }
            for action in legal_actions
            if isinstance(action, dict)
        ],
        "screen": _extract_screen_payload(state),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _extract_player(state: dict) -> dict:
    player = state.get("player")
    if isinstance(player, dict) and player:
        return player

    for key in ("map", "event", "rest_site", "shop", "treasure",
                "rewards", "card_reward", "card_select", "relic_select",
                "battle"):
        nested = state.get(key) or {}
        player = nested.get("player")
        if isinstance(player, dict) and player:
            return player

    return {}


def _extract_screen_payload(state: dict) -> dict:
    state_type = (state.get("state_type") or "").lower()
    if state_type == "map":
        map_state = state.get("map") or {}
        return {
            "next_options": [
                {
                    "index": option.get("index"),
                    "col": option.get("col"),
                    "row": option.get("row"),
                    "point_type": option.get("point_type"),
                }
                for option in map_state.get("next_options") or []
                if isinstance(option, dict)
            ]
        }
    if state_type == "event":
        event_state = state.get("event") or {}
        return {
            "in_dialogue": event_state.get("in_dialogue"),
            "is_finished": event_state.get("is_finished"),
            "options": [
                {
                    "index": option.get("index"),
                    "text": option.get("text"),
                    "is_locked": option.get("is_locked"),
                    "is_chosen": option.get("is_chosen"),
                    "is_proceed": option.get("is_proceed"),
                }
                for option in event_state.get("options") or []
                if isinstance(option, dict)
            ],
        }
    if state_type == "rest_site":
        rest_state = state.get("rest_site") or {}
        return {
            "can_proceed": rest_state.get("can_proceed"),
            "options": [
                {
                    "index": option.get("index"),
                    "id": option.get("id"),
                    "is_enabled": option.get("is_enabled"),
                }
                for option in rest_state.get("options") or []
                if isinstance(option, dict)
            ],
        }
    if state_type == "shop":
        shop_state = state.get("shop") or {}
        return {
            "can_proceed": shop_state.get("can_proceed"),
            "items": [
                {
                    "index": item.get("index"),
                    "category": item.get("category"),
                    "cost": item.get("cost"),
                    "can_afford": item.get("can_afford"),
                    "is_stocked": item.get("is_stocked"),
                    "on_sale": item.get("on_sale"),
                    "item_id": item.get("card_id") or item.get("relic_id")
                    or item.get("potion_id") or item.get("name"),
                }
                for item in shop_state.get("items") or []
                if isinstance(item, dict)
            ],
        }
    if state_type == "combat_rewards":
        rewards_state = state.get("rewards") or {}
        return {
            "can_proceed": rewards_state.get("can_proceed"),
            "items": [
                {
                    "index": item.get("index"),
                    "type": item.get("type"),
                    "label": item.get("label"),
                }
                for item in rewards_state.get("items") or []
                if isinstance(item, dict)
            ],
        }
    if state_type == "card_reward":
        reward_state = state.get("card_reward") or {}
        return {
            "can_skip": reward_state.get("can_skip"),
            "cards": [
                {
                    "index": card.get("index"),
                    "id": card.get("id"),
                    "name": card.get("name"),
                    "type": card.get("type"),
                    "rarity": card.get("rarity"),
                    "cost": card.get("cost"),
                    "is_upgraded": card.get("is_upgraded"),
                }
                for card in reward_state.get("cards") or []
                if isinstance(card, dict)
            ],
        }
    return {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=15527)
    args = parser.parse_args()

    env = MctsEnv(base_url=f"http://127.0.0.1:{args.port}")
    ok = test_save_load(env)
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
