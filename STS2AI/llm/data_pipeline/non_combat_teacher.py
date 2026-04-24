"""非战斗状态的启发式老师。

覆盖 game_bridge full_run session 里常见的 state_type：
- event             事件选项
- card_select       选牌（棋盘抽卡 / 剧情选牌）
- card_reward       战斗后选奖励卡
- map               地图节点选路
- rest/campfire     休息点（回血 vs 升级）
- shop              商店

设计原则：
- 返回 action_index + reason 字符串，和战斗版一致
- 规则宁简勿繁，先给 LLM 一个**有信号**的老师，不追求最优
- 所有决策都能用 `legal_actions[*]` 里的 label / col / row / card_id 等字段判断，
  不需要访问 state 外层字段（state 外层字段做 bonus 判据时才用）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _pick(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


@dataclass(slots=True, frozen=True)
class NonCombatDecision:
    action_index: int
    reason: str


# --- Ironclad 攻击 / 防御 / 能量类偏好（和 card_effects.py 对齐）---
_IRONCLAD_PREFERRED_ATTACK = {
    "BLUDGEON", "UPPERCUT", "HEAVY_BLADE", "CLOTHESLINE", "IRON_WAVE",
    "TWIN_STRIKE", "POMMEL_STRIKE", "PERFECTED_STRIKE", "CINDER", "HEAVY_FORGE",
}
_IRONCLAD_PREFERRED_SKILL = {
    "SHRUG_IT_OFF", "TRUE_GRIT", "FLEX", "BATTLE_TRANCE", "SEEING_RED",
}
_IRONCLAD_PREFERRED_POWER = {
    "INFLAME", "METALLICIZE", "DEMON_FORM", "FEEL_NO_PAIN", "DARK_EMBRACE",
    "JUGGERNAUT", "FIRE_BREATHING", "COMBUST", "RUPTURE",
}
# 尽量避免的诅咒/status
_CURSES = {
    "DAZED", "SLIMED", "BURN", "WOUND", "REGRET", "SHAME", "DOUBT",
    "INJURY", "WRITHE", "NORMALITY", "PAIN", "PARASITE", "CLUMSY", "CURSE_OF_THE_BELL",
}


def _pick_map_node(state: dict[str, Any], legal: list[dict[str, Any]]) -> NonCombatDecision:
    player = _as_dict(state.get("player"))
    hp = float(_pick(player, "hp", "current_hp", default=0) or 0)
    max_hp = float(_pick(player, "max_hp", default=1) or 1)
    hp_ratio = hp / max(1.0, max_hp)
    gold = float(_pick(player, "gold", default=0) or 0)

    # 规则（按优先级）：
    # 1) HP < 40% → rest 最优
    # 2) HP > 75% 且没打过精英 → elite（换好物）
    # 3) 有 treasure 总选
    # 4) shop 如果 gold >= 150
    # 5) event（风险可控的免费效果）
    # 6) 默认 monster
    want_rest = hp_ratio < 0.40
    want_elite = hp_ratio > 0.75

    def _label(a: dict[str, Any]) -> str:
        return str(_pick(a, "label", default="") or "").lower()

    priorities = []  # (priority, index, reason)
    for idx, a in enumerate(legal):
        lab = _label(a)
        if "rest" in lab or "campfire" in lab:
            p = 0 if want_rest else 5
            priorities.append((p, idx, f"选 rest（HP={hp:.0f}/{max_hp:.0f}）"))
        elif "treasure" in lab or "chest" in lab:
            priorities.append((1, idx, "选宝箱"))
        elif "elite" in lab:
            priorities.append((2 if want_elite else 7, idx, f"选精英（HP 足={hp_ratio:.0%}）"))
        elif "shop" in lab or "merchant" in lab:
            priorities.append((3 if gold >= 150 else 6, idx, f"选商店（金币 {gold:.0f}）"))
        elif "event" in lab or "unknown" in lab or "question" in lab:
            priorities.append((4, idx, "选事件（低风险）"))
        elif "monster" in lab or "enemy" in lab:
            priorities.append((5, idx, "选普通怪"))
        elif "boss" in lab:
            priorities.append((9, idx, "只能推进到 boss"))
        else:
            priorities.append((8, idx, f"未识别节点 {lab or '?'}"))
    priorities.sort()
    best_p, best_idx, best_reason = priorities[0]
    return NonCombatDecision(action_index=best_idx, reason=best_reason)


def _pick_card_reward(state: dict[str, Any], legal: list[dict[str, Any]]) -> NonCombatDecision:
    """战斗后 3 选 1 + skip。Ironclad 偏好：大攻击 > power > skill > 过弱的小攻击"""
    def score(a: dict[str, Any]) -> tuple[int, int, str]:
        atype = str(_pick(a, "action", "type", default="")).lower()
        cid = str(_pick(a, "card_id", "label", default="")).upper()
        if atype.startswith("skip"):
            return (99, 0, "跳过奖励（只在没好牌时）")
        base_reason = f"拿 {cid}"
        if cid in _IRONCLAD_PREFERRED_POWER:
            return (0, -100, f"拿 power 卡 {cid}")
        if cid in _IRONCLAD_PREFERRED_ATTACK:
            return (1, -50, f"拿强攻击 {cid}")
        if cid in _IRONCLAD_PREFERRED_SKILL:
            return (2, -20, f"拿实用 skill {cid}")
        if cid in _CURSES:
            return (100, 0, f"避开诅咒 {cid}")
        if "STRIKE" in cid or "DEFEND" in cid:
            return (10, 0, f"基础 {cid}（补位）")
        return (5, 0, base_reason)

    scored = [(score(a), idx) for idx, a in enumerate(legal)]
    scored.sort()
    (_, _, reason), best_idx = scored[0]
    return NonCombatDecision(action_index=best_idx, reason=reason)


def _pick_event(state: dict[str, Any], legal: list[dict[str, Any]]) -> NonCombatDecision:
    """事件选项：label 里含 "GAIN"/"ADD"/"HEAL" 更安全；含 "LOSE"/"TAKE_DAMAGE"/"CURSE" 避免。"""
    def score(a: dict[str, Any]) -> tuple[int, str]:
        lab = str(_pick(a, "label", default="") or "").lower()
        if any(bad in lab for bad in ("curse", "lose_hp", "take_damage", "lose_max", "add_curse")):
            return (10, f"避开 {lab}")
        if any(good in lab for good in ("heal", "gain_gold", "add_relic", "free", "upgrade")):
            return (0, f"选好效果 {lab}")
        if "leave" in lab or "continue" in lab or "proceed" in lab:
            return (5, f"安全离开 {lab}")
        return (3, f"选事件 {lab or '选项'}")

    scored = [(score(a), idx) for idx, a in enumerate(legal)]
    scored.sort()
    (_, reason), best_idx = scored[0]
    return NonCombatDecision(action_index=best_idx, reason=reason)


def _pick_campfire(state: dict[str, Any], legal: list[dict[str, Any]]) -> NonCombatDecision:
    player = _as_dict(state.get("player"))
    hp = float(_pick(player, "hp", "current_hp", default=0) or 0)
    max_hp = float(_pick(player, "max_hp", default=1) or 1)
    hp_ratio = hp / max(1.0, max_hp)
    want_rest = hp_ratio < 0.50

    for idx, a in enumerate(legal):
        lab = str(_pick(a, "label", "action", default="")).lower()
        if want_rest and ("rest" in lab or "sleep" in lab):
            return NonCombatDecision(action_index=idx, reason=f"休息回血（HP={hp_ratio:.0%}）")
        if not want_rest and ("upgrade" in lab or "smith" in lab):
            return NonCombatDecision(action_index=idx, reason="升级一张牌")
    return NonCombatDecision(action_index=0, reason="默认选第一个休息选项")


def _pick_shop(state: dict[str, Any], legal: list[dict[str, Any]]) -> NonCombatDecision:
    """商店启发式：如果有 leave 就 leave（避免乱买），后续精细化再加。"""
    for idx, a in enumerate(legal):
        lab = str(_pick(a, "label", "action", default="")).lower()
        if "leave" in lab or "exit" in lab:
            return NonCombatDecision(action_index=idx, reason="离开商店（还没启发式买卡）")
    return NonCombatDecision(action_index=0, reason="默认第一个商店动作")


def pick_non_combat(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> NonCombatDecision | None:
    """根据 state_type 路由到相应启发式。不是非战斗状态返回 None。"""
    st = str(state.get("state_type") or "").lower()
    enabled = [a for a in legal_actions if isinstance(a, dict) and a.get("is_enabled") is not False]
    if not enabled:
        return NonCombatDecision(action_index=0, reason="no_legal_actions")

    # 只基于第一个动作的 action 字段来判断（不看 state_type 更稳）
    atype = str(_pick(enabled[0], "action", "type", default="")).lower()

    if atype in ("choose_map_node",):
        dec = _pick_map_node(state, enabled)
    elif atype in ("choose_event_option",):
        dec = _pick_event(state, enabled)
    elif atype in ("select_card_reward", "skip_card_reward"):
        dec = _pick_card_reward(state, enabled)
    elif st in ("campfire", "rest"):
        dec = _pick_campfire(state, enabled)
    elif st == "shop":
        dec = _pick_shop(state, enabled)
    elif atype in ("proceed", "claim_reward", "confirm_selection", "select_card"):
        # 单选 / 过场 —— 默认 idx=0
        reason = {
            "proceed": "推进游戏",
            "claim_reward": "领取奖励",
            "confirm_selection": "确认选择",
            "select_card": "选择默认卡",
        }.get(atype, "过场")
        dec = NonCombatDecision(action_index=0, reason=reason)
    else:
        # 不认识的非战斗动作
        return None

    # 保险：越界 fallback
    idx = max(0, min(dec.action_index, len(enabled) - 1))
    return NonCombatDecision(action_index=idx, reason=dec.reason)


__all__ = ["NonCombatDecision", "pick_non_combat"]
