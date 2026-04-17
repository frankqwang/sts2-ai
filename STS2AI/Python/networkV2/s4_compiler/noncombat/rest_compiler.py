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


# U1 修复：原先 rest option 的 token numeric 几乎空（只靠 family="rest" + 3-4 维
# role one-hot），recall/dig/lift/toke 都是 "resource" role，在 token 层完全无差别。
# 复用 ActionCandidate 的 rarity_weight 字段给每种 rest 选项一个"相对价值" soft 权重。
# 不改字段数，不占新通道。
#
# 价值排序来自经验：
# - smith: 永久升级一张卡，整局收益最稳 → 1.0
# - rest:  HP 恢复 30%，context-dependent（低 HP 时价值极高，通过 objective_bank 的
#          hp_ratio / survival_priority 给决策层额外条件化；这里只给 baseline）
# - recall: 取出 key relic（STS2 里某些 relic 的特殊交互），中偏高
# - dig / lift / toke: 依赖 relic 效果，平均价值中等
_REST_OPTION_WEIGHT = {
    "rest":   0.5,
    "smith":  1.0,
    "recall": 0.7,
    "dig":    0.5,
    "lift":   0.4,
    "toke":   0.4,
}


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
                    # U1: 给每种 rest 选项一个相对价值权重（编进 action token 的 rarity_weight 通道）
                    rarity_weight=_REST_OPTION_WEIGHT.get(rest_id, 0.3),
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
