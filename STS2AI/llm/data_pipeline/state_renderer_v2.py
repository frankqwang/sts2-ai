"""优化版状态渲染器（v2）。

三大改进：
1. 历史上下文：注入最近 1-2 回合的关键动作与状态变化摘要。
2. 紧凑牌库：默认只给 pile 计数 + 关键牌（升级牌、稀有牌），不再全部枚举。
3. JSON 模式：可选的结构化表示，用于对比实验（token 效率 vs 可理解性）。

接口与 v1（state_renderer.py）兼容，可直接替换 `render_state_text` 调用。
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from llm.data_pipeline.state_renderer import (
    _as_dict,
    _as_list,
    _clean_runtime_text,
    _fmt_powers,
    _is_localization_key,
    _item_description,
    _iter_hand_cards,
    _iter_powers,
    _iter_relics,
    _pick,
    _resolve_card_desc,
    render_enemies,
    render_hand,
    render_legal_actions,
    render_player,
    render_potions,
    render_relics,
    render_run_meta,
)


# ---------------------------------------------------------------------------
# 1. 历史上下文（History Buffer）
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    action_type: str
    card_id: str = ""
    target_id: str = ""
    outcome: str = ""  # 如 "enemy0 took 8 dmg, died"
    player_hp_delta: float = 0.0
    player_block: float = 0.0
    enemy_intent_change: str = ""


class HistoryBuffer:
    """维护最近 N 步的简史，供 renderer 注入 prompt。"""

    def __init__(self, max_len: int = 4) -> None:
        self._entries: list[HistoryEntry] = []
        self._max_len = max_len

    def push(self, entry: HistoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_len:
            self._entries.pop(0)

    def clear(self) -> None:
        self._entries.clear()

    def to_text(self) -> list[str]:
        if not self._entries:
            return []
        lines: list[str] = []
        for i, e in enumerate(self._entries, 1):
            parts: list[str] = [e.action_type]
            if e.card_id:
                parts.append(e.card_id)
            if e.target_id:
                parts.append(f"->{e.target_id}")
            if e.outcome:
                parts.append(f"({e.outcome})")
            lines.append(f"  t-{len(self._entries)-i}: {' '.join(parts)}")
        return lines


# ---------------------------------------------------------------------------
# 2. 紧凑牌库表示
# ---------------------------------------------------------------------------

def _summarize_cards_compact(card_list: list[Any], max_items: int = 5) -> str:
    """只显示最关键的几张牌，其余用 '...N more' 省略。"""
    if not card_list:
        return "-"
    counter: Counter = Counter()
    upgraded: set[str] = set()
    for c in card_list:
        if isinstance(c, dict):
            cid = str(_pick(c, "id", "card_id", default=""))
            upg = bool(_pick(c, "is_upgraded", default=False))
        elif isinstance(c, str):
            cid = c
            upg = False
        else:
            continue
        if not cid:
            continue
        key = f"{cid}+" if upg else cid
        counter[key] += 1
        if upg:
            upgraded.add(key)

    parts = []
    for cid, n in counter.most_common():
        label = f"{cid}x{n}" if n > 1 else cid
        parts.append(label)
        if len(parts) >= max_items:
            remaining = len(counter) - max_items
            if remaining > 0:
                parts.append(f"...{remaining}more")
            break
    return ", ".join(parts)


def render_piles_compact(state: dict[str, Any]) -> list[str]:
    battle = _as_dict(state.get("battle"))
    draw = _as_list(battle.get("draw_pile_cards"))
    discard = _as_list(battle.get("discard_pile_cards"))
    exhaust = _as_list(battle.get("exhaust_pile_cards"))

    lines = [f"  pile_stats: draw={len(draw)} discard={len(discard)} exhaust={len(exhaust)}"]

    # 只展示关键牌：升级牌、稀有牌（如果信息可得）
    notable_draw = _summarize_cards_compact(draw, max_items=4)
    notable_exhaust = _summarize_cards_compact(exhaust, max_items=3)
    if notable_draw != "-":
        lines.append(f"  notable_draws: {notable_draw}")
    if notable_exhaust != "-":
        lines.append(f"  exhausted: {notable_exhaust}")
    return lines


# ---------------------------------------------------------------------------
# 3. 主渲染接口（兼容 v1 + 新增历史 + 紧凑模式）
# ---------------------------------------------------------------------------

def render_state_text_v2(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    encounter_id: str = "",
    history: HistoryBuffer | None = None,
    compact: bool = True,
) -> str:
    """优化版状态文本渲染。

    Args:
        state: game_bridge state dict
        legal_actions: 当前合法动作列表
        encounter_id: 覆盖用的 encounter id
        history: 历史上下文 buffer（None 则不注入）
        compact: True 使用紧凑牌库表示；False 回退到 v1 的枚举风格
    """
    parts: list[str] = []

    parts.append(render_run_meta(state, encounter_id=encounter_id))
    parts.append(render_player(state))

    if history is not None:
        hist_lines = history.to_text()
        if hist_lines:
            parts.append("recent_history:")
            parts.extend(hist_lines)

    relic_lines = render_relics(state)
    if relic_lines:
        parts.append("relics:")
        parts.extend(relic_lines)

    potion_lines = render_potions(state)
    if potion_lines:
        parts.append("potions:")
        parts.extend(potion_lines)

    # deck 摘要（v1 已有 render_deck，直接复用逻辑）
    from llm.data_pipeline.state_renderer import render_deck
    parts.append(f"deck: {render_deck(state)}")

    # 牌库：紧凑 or 完整
    if compact:
        parts.append("piles:")
        parts.extend(render_piles_compact(state))
    else:
        from llm.data_pipeline.state_renderer import render_piles
        pile_lines = render_piles(state)
        parts.append("piles:")
        parts.extend(pile_lines)

    parts.append("enemies:")
    parts.extend(render_enemies(state))

    parts.append("hand:")
    parts.extend(render_hand(state))

    parts.append("legal_actions:")
    parts.extend(render_legal_actions(legal_actions, state))

    # glossary 只在需要时显示（v1 逻辑）
    from llm.data_pipeline.state_renderer import render_glossary
    glossary_lines = render_glossary(state)
    if glossary_lines:
        parts.append("glossary:")
        parts.extend(glossary_lines)

    parts.append(
        'Return one JSON line: {"action_index":N,"confidence":0.0,'
        '"action_scores":[{"action_index":N,"score":0.0}],"reason":"..."} '
        "using listed action_index values."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 4. JSON 模式（结构化表示，用于对比实验）
# ---------------------------------------------------------------------------

def _card_to_json(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _pick(card, "id", "card_id", default="?"),
        "cost": _pick(card, "cost", "cost_now", default=0),
        "type": str(_pick(card, "type", default="")).lower(),
        "upg": bool(_pick(card, "is_upgraded", default=False)),
        "dmg_preview": card.get("preview_damage_per_target") or {},
        "blk_preview": card.get("preview_block") or 0,
        "playable": bool(_pick(card, "can_play", default=True)),
    }


def _enemy_to_json(enemy: dict[str, Any], index: int) -> dict[str, Any]:
    target_id = _pick(enemy, "target_id", "combat_id", default=index)
    intent = str(_pick(enemy, "intent_type", "next_move_id", default="?"))
    dmg = _pick(enemy, "intent_damage", "move_base_damage", default=None)
    hits = _pick(enemy, "intent_hits", "move_hits", default=None)
    intent_str = intent
    if dmg:
        intent_str += f"({dmg}" + (f"x{hits}" if hits and int(hits) > 1 else "") + ")"
    return {
        "idx": int(target_id),
        "id": _pick(enemy, "monster_id", "entity_id", "id", default=f"enemy_{index}"),
        "hp": f"{_pick(enemy, 'hp', 'current_hp', default=0)}/{_pick(enemy, 'max_hp', default=0)}",
        "block": _pick(enemy, "block", default=0),
        "intent": intent_str,
        "powers": _fmt_powers(_as_list(enemy.get("powers")) or _as_list(enemy.get("buffs"))),
        "alive": bool(_pick(enemy, "is_alive", "alive", default=True)),
    }


def render_state_json(state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> str:
    """把状态渲染成紧凑 JSON 字符串。

    注意：JSON 的 token 效率不一定比精心设计的文本高（引号、括号占 token），
    但这提供了一种可程序解析的替代格式，适合测试哪种表示对模型更友好。
    """
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))

    payload = {
        "run": {
            "char": _pick(top_player, "character", default="IRONCLAD"),
            "floor": _pick(_as_dict(state.get("run")), "floor_reached", "floor", default="?"),
            "act": _pick(_as_dict(state.get("run")), "act", default="?"),
            "gold": _pick(top_player, "gold", default=0),
        },
        "player": {
            "hp": f"{_pick(top_player, 'hp', 'current_hp', default=0)}/{_pick(top_player, 'max_hp', default=0)}",
            "block": _pick(battle_player, "block", default=_pick(top_player, "block", default=0)),
            "energy": f"{_pick(battle, 'energy', default=_pick(top_player, 'energy', default=0))}/{_pick(battle, 'max_energy', default=_pick(top_player, 'max_energy', default=0))}",
            "powers": _fmt_powers(_as_list(battle_player.get("powers")) or _as_list(top_player.get("powers"))),
        },
        "piles": {
            "draw": len(_as_list(battle.get("draw_pile_cards"))),
            "discard": len(_as_list(battle.get("discard_pile_cards"))),
            "exhaust": len(_as_list(battle.get("exhaust_pile_cards"))),
        },
        "enemies": [_enemy_to_json(e, i) for i, e in enumerate(_as_list(state.get("enemies")) or _as_list(battle.get("enemies"))) if isinstance(e, dict)],
        "hand": [_card_to_json(c) for c in _iter_hand_cards(state)],
        "legal_actions": [
            {
                "idx": i,
                "type": str(_pick(a, "action", "action_type", "type", default="?")).lower(),
                "card": _pick(a, "card_id", default=None),
                "hand_idx": _pick(a, "card_index", "hand_index", default=None),
                "target": _pick(a, "target_id", "target", default=None),
            }
            for i, a in enumerate(legal_actions)
            if isinstance(a, dict)
        ],
    }
    # 追加一条指令
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + '\nChoose exactly one legal_actions[idx] and reply with JSON: {"action_index":idx,"confidence":0.0,'
        '"action_scores":[{"action_index":idx,"score":0.0}],"reason":"..."}'
    )


__all__ = [
    "HistoryBuffer",
    "HistoryEntry",
    "render_piles_compact",
    "render_state_json",
    "render_state_text_v2",
]
