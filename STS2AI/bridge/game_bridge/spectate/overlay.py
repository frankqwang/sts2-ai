"""观战 overlay 输出。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _action_type(action: dict[str, Any]) -> str:
    return str(action.get("action") or action.get("action_type") or action.get("type") or "").lower()


def _action_label(action: dict[str, Any]) -> str:
    atype = _action_type(action)
    if atype == "play_card":
        card = str(action.get("card_id") or "CARD")
        card_index = action.get("card_index", action.get("hand_index"))
        prefix = f"play hand[{card_index}] {card}" if card_index is not None else f"play {card}"
        target_id = action.get("target_id", action.get("target"))
        return f"{prefix} -> enemy{target_id}" if target_id is not None else prefix
    if atype == "end_turn":
        return "end_turn"
    label = str(action.get("label") or "").strip()
    card_id = str(action.get("card_id") or "").strip()
    if atype in {"select_card", "select_card_reward"} and card_id:
        return f"{atype} {card_id}"
    if atype == "choose_map_node":
        col = action.get("col")
        row = action.get("row")
        location = f" col={col} row={row}" if col is not None and row is not None else ""
        return f"{atype}{location}" + (f" {label}" if label else "")
    if atype and label:
        return f"{atype} {label}"
    if atype:
        return atype
    return str(action.get("id") or action.get("label") or "action")


def _policy_reason(action: dict[str, Any]) -> str:
    return str(
        action.get("_policy_reason")
        or action.get("_teacher_reason")
        or action.get("reason")
        or ""
    )


def _policy_source(action: dict[str, Any]) -> str:
    return str(
        action.get("_policy_route")
        or ("heuristic" if "_teacher_reason" in action else "policy")
    )


def _format_for_decision_overlay(data: dict[str, Any]) -> dict[str, Any]:
    state = _as_dict(data.get("state"))
    legal = [dict(a) for a in _as_list(data.get("legal_actions")) if isinstance(a, dict)]
    chosen = _as_dict(data.get("chosen_action"))
    chosen_idx = chosen.get("_policy_action_index")
    if chosen_idx is None:
        for idx, action in enumerate(legal):
            if action == chosen:
                chosen_idx = idx
                break

    return {
        "title": "LLM Decision",
        "state_type": data.get("state_type") or state.get("state_type"),
        "step": data.get("step_index"),
        "step_index": data.get("step_index"),
        "action_source": _policy_source(chosen),
        "chosen_action": chosen,
        "chosen_action_text": (
            f"[{chosen_idx}] {_action_label(chosen)}"
            if chosen and chosen_idx is not None
            else _action_label(chosen) if chosen else "-"
        ),
        "reason": _policy_reason(chosen),
    }


@dataclass
class OverlayWriter:
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, data: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(_format_for_decision_overlay(data), handle, ensure_ascii=False, indent=2)
