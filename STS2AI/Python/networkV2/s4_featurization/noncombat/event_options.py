"""Event Option Builder: 事件选项。

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


# R2.3: 从 option label / effect_text 推断 event kind（token_bank_builder 做 one-hot）
# 原先所有 event option 的 roles 都硬编码 ["resource"]，网络完全无法区分
# "掉血换 relic" vs "给金币" vs "+max_hp"。用关键词匹配做 best-effort 分类。
#
# 每个 pattern 是一个 token 列表，全部命中（in substring）才算匹配。
# 这样 "Lose 10 HP" 会被 ["lose", "hp"] 捕获（单词 "lose hp" 连续 substring 匹配不到）。
# 匹配顺序先于 gain_hp / gain_curse 等，避免 "Lose HP to gain relic" 被误判为 gain_relic。
_EVENT_KIND_PATTERNS: list[tuple[str, list[list[str]]]] = [
    ("gain_gold",     [["gold"], ["coin"], ["金币"]]),
    ("gain_relic",    [["relic"], ["遗物"]]),
    ("gain_potion",   [["potion"], ["药水"], ["elixir"]]),
    ("remove_card",   [["remove", "card"], ["transform"], ["移除"], ["转化"], ["净化"], ["purge"]]),
    ("upgrade_card",  [["upgrade"], ["升级"]]),
    ("gain_curse",    [["curse"], ["诅咒"]]),
    ("lose_hp",       [["lose", "hp"], ["lose", "health"], ["take", "damage"],
                       ["掉血"], ["受伤"], ["自伤"]]),
    ("gain_hp",       [["heal"], ["restore", "hp"], ["max", "hp"], ["max", "health"],
                       ["恢复"], ["治疗"], ["生命上限"]]),
]


def _infer_event_kind(label: str, text: str = "") -> str:
    """label + effect 文本关键词匹配 → EVENT_KINDS 中的一个。查不到 → 'unknown'。"""
    blob = f"{label} {text}".lower()
    if not blob.strip():
        return "unknown"
    for kind, patterns in _EVENT_KIND_PATTERNS:
        for pattern in patterns:
            if all(tok in blob for tok in pattern):
                return kind
    return "unknown"


class EventOptionBuilder:
    """构建 event 选项为 ActionCandidate 列表。"""

    def build(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        event = obs.get("event") or {}
        event_id = str(event.get("event_id", "") or "").lower()
        # 预取 options 的 effect text（如果 obs 提供）
        options_by_idx = {
            int(opt.get("index", i) or 0): opt
            for i, opt in enumerate(event.get("options", []) or [])
            if isinstance(opt, dict)
        }

        for i, action in enumerate(legal_actions):
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action", "") or "").lower()
            label = str(action.get("label", "") or "")

            if action_type == "choose_event_option":
                option_idx = int(action.get("index", 0) or 0)
                # 从 obs.event.options 拿 effect text（如果有）
                opt = options_by_idx.get(option_idx, {})
                effect_text = str(
                    opt.get("text") or opt.get("effect_text") or opt.get("description") or ""
                )
                kind = _infer_event_kind(label, effect_text)

                candidates.append(ActionCandidate(
                    action_type=action_type,
                    action_index=i,
                    label=label,
                    family="event_option",
                    source_card_id=f"{event_id}_opt{option_idx}",
                    target_scope="event",
                    roles=["resource"],
                    # R2.3: 用 event_kind 补齐 token 信号（9 维 one-hot）
                    event_kind=kind,
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
