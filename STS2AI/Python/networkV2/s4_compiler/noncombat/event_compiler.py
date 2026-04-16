"""Event Compiler: 事件选项。

游戏数据格式 (event state):
  {
    "event_id": str,
    "options": [
      {"index": int, "text": str, "label": str, "is_locked": bool, ...},
    ]
  }
"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.actions import ActionCandidate


class EventCompiler:
    """编译 event 选项为 ActionCandidate 列表。"""

    def compile(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        event = obs.get("event") or {}
        event_id = str(event.get("event_id", "") or "").lower()

        for i, action in enumerate(legal_actions):
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action", "") or "").lower()
            label = str(action.get("label", "") or "")

            if action_type == "choose_event_option":
                option_idx = int(action.get("index", 0) or 0)

                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label,
                    family="event_option",
                    source_card_id=f"{event_id}_opt{option_idx}",
                    target_scope="event",
                    roles=["resource"],  # 事件选项的角色后续可细化
                ))

            elif action_type in ("proceed", "advance_dialogue"):
                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label or "Continue",
                    family="event_option",
                    roles=["terminal"],
                    ends_turn=True,
                ))

        return candidates
