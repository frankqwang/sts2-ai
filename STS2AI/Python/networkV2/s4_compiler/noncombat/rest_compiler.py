"""Rest Compiler: 休息站选项 (heal/smith/recall/dig/lift/toke)。

游戏数据格式 (rest_site state):
  {
    "options": [
      {"id": "rest"/"smith"/"recall"/..., "name": str, "is_enabled": bool},
    ]
  }
"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.actions import ActionCandidate


class RestCompiler:
    """编译 rest_site 选项为 ActionCandidate 列表。"""

    def compile(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []

        for i, action in enumerate(legal_actions):
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action", "") or "").lower()
            label = str(action.get("label", "") or "")

            if action_type == "choose_rest_option":
                rest_id = str(action.get("id", action.get("rest_id", "")) or "").lower()

                roles = []
                if rest_id == "rest":
                    roles.append("heal")
                elif rest_id == "smith":
                    roles.append("setup")
                elif rest_id in ("recall", "dig", "lift", "toke"):
                    roles.append("resource")

                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label or rest_id,
                    family="rest",
                    source_card_id=rest_id,
                    roles=roles,
                    target_scope="none",
                ))

            elif action_type == "proceed":
                candidates.append(ActionCandidate(
                    action_type="proceed",
                    action_index=i,
                    label=label or "Proceed",
                    family="rest",
                    roles=["terminal"],
                    ends_turn=True,
                ))

        return candidates
