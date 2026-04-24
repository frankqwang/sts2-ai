"""把 game_bridge 原始 state JSON + legal_actions 压成 LLM 能读的紧凑文本。

2026-04-24 大改：从静态 card_effects 表转向**sim 真实字段 + catalog 静态补充**。
- 卡牌描述：优先 sim 的 description（动态），localization key 时 fallback 到 catalog_loader
- preview_damage / preview_block：sim 直出（暂未全部填充，见 ProtoStateBuilder）
- 遗物 / 药水 / 牌堆内容 / 全牌组：从 state 直读渲染
- 玩家/敌人 powers：id + amount + 人类可读描述
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Iterable

from llm.data_pipeline.catalog_loader import (
    lookup_card,
    power_short,
    render_card_description,
    relic_short,
)

_LOCALIZATION_KEY_RE = re.compile(
    r"^[A-Za-z0-9_]+\.(?:description|desc)(?:$|\s|\[)"
)


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _pick(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _is_localization_key(text: str) -> bool:
    """sim 有时返回 'CARD_ID.description' 而不是解析后的文本。判断用。"""
    if not text:
        return True
    stripped = text.strip()
    return bool(_LOCALIZATION_KEY_RE.match(stripped))


def _fmt_powers(powers: Iterable[Any]) -> str:
    parts: list[str] = []
    for power in powers or []:
        if not isinstance(power, dict):
            continue
        pid = str(power.get("id") or power.get("power_id") or power.get("name") or "").upper()
        amount = power.get("amount")
        if not pid:
            continue
        desc = power_short(pid, amount=amount)
        parts.append(desc)
    return "; ".join(parts) if parts else "-"


def _resolve_card_desc(card: dict[str, Any]) -> str:
    """优先用 sim 动态 description；是 localization key 时退回 catalog 静态描述。"""
    sim_desc = str(card.get("description") or "")
    cid = str(card.get("id") or card.get("card_id") or "")
    is_upgraded = bool(_pick(card, "is_upgraded", default=False))
    if sim_desc and not _is_localization_key(sim_desc):
        return render_card_description(
            cid,
            sim_desc,
            is_upgraded=is_upgraded,
            runtime_values=card,
        )
    info = lookup_card(cid)
    if info and info.get("description"):
        return render_card_description(
            cid,
            str(info["description"]),
            is_upgraded=is_upgraded,
            runtime_values=card,
        )
    return ""


def render_player(state: dict[str, Any]) -> str:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    hp = _pick(top_player, "hp", "current_hp", default=0)
    max_hp = _pick(top_player, "max_hp", default=0)
    block = _pick(battle_player, "block", default=_pick(top_player, "block", default=0))
    energy = _pick(battle, "energy", default=_pick(top_player, "energy", default=0))
    max_energy = _pick(battle, "max_energy", default=_pick(top_player, "max_energy", default=0))
    buffs = _fmt_powers(_as_list(battle_player.get("powers")) or _as_list(top_player.get("powers")))
    return f"player: hp={hp}/{max_hp} block={block} energy={energy}/{max_energy} buffs={buffs}"


def render_enemies(state: dict[str, Any]) -> list[str]:
    battle = _as_dict(state.get("battle"))
    enemies_raw = _as_list(state.get("enemies")) or _as_list(battle.get("enemies"))
    lines: list[str] = []
    for index, enemy in enumerate(enemies_raw):
        if not isinstance(enemy, dict):
            continue
        eid = _pick(enemy, "monster_id", "entity_id", "id", default=f"enemy_{index}")
        target_id = _pick(enemy, "target_id", "combat_id", default=index)
        hp = _pick(enemy, "hp", "current_hp", default=0)
        max_hp = _pick(enemy, "max_hp", default=0)
        block = _pick(enemy, "block", default=0)
        intent = _pick(enemy, "intent_type", "next_move_id", default="?")
        dmg = _pick(enemy, "intent_damage", "move_base_damage", default=None)
        hits = _pick(enemy, "intent_hits", "move_hits", default=None)
        intent_str = str(intent)
        if dmg:
            intent_str += f"({dmg}"
            if hits and int(hits) > 1:
                intent_str += f"x{hits}"
            intent_str += ")"
        buffs = _fmt_powers(_as_list(enemy.get("powers")) or _as_list(enemy.get("buffs")))
        alive = bool(_pick(enemy, "is_alive", "alive", default=True))
        tag = "" if alive else " [dead]"
        lines.append(
            f"  id={target_id} {eid} hp={hp}/{max_hp} block={block} intent={intent_str} buffs={buffs}{tag}"
        )
    return lines


def render_hand(state: dict[str, Any]) -> list[str]:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    hand_raw = (
        _as_list(battle.get("hand"))
        or _as_list(battle_player.get("hand"))
        or _as_list(state.get("hand"))
        or _as_list(top_player.get("hand"))
    )
    lines: list[str] = []
    for index, card in enumerate(hand_raw):
        if not isinstance(card, dict):
            continue
        cid = _pick(card, "id", "card_id", default="?")
        cost = _pick(card, "cost", "cost_now", default=0)
        card_type = str(_pick(card, "type", default="")).lower()
        is_upg = bool(_pick(card, "is_upgraded", default=False))
        can_play = bool(_pick(card, "can_play", default=True))

        # 优先显示 sim 实时 preview
        preview_dmg = card.get("preview_damage_per_target") or {}
        preview_block = card.get("preview_block") or 0
        desc = _resolve_card_desc(card)

        # 预算数值
        dmg_str = ""
        if preview_dmg:
            vals = [f"→{tid}:{d}" for tid, d in preview_dmg.items() if d]
            if vals:
                dmg_str = " dmg(actual)=" + ",".join(vals)
        blk_str = f" block(actual)={preview_block}" if preview_block else ""

        tags = [card_type] if card_type else []
        if is_upg:
            tags.append("upg")
        tag_str = ",".join(tags) if tags else "-"
        flag = "" if can_play else " [unplayable]"

        desc_snippet = (desc[:60] + "…") if desc and len(desc) > 60 else desc
        desc_part = f" | {desc_snippet}" if desc_snippet else ""
        lines.append(
            f"  [{index}] {cid} cost={cost}{dmg_str}{blk_str} tags={tag_str}{flag}{desc_part}"
        )
    return lines


def render_relics(state: dict[str, Any]) -> list[str]:
    """渲染玩家遗物 + 简要效果。"""
    player_top = _as_dict(state.get("player"))
    battle = _as_dict(state.get("battle"))
    battle_player = _as_dict(battle.get("player"))
    relics_raw = _as_list(player_top.get("relics")) or _as_list(battle_player.get("relics"))
    lines: list[str] = []
    for index, relic in enumerate(relics_raw):
        if not isinstance(relic, dict):
            continue
        rid = str(_pick(relic, "id", "relic_id", default="?"))
        desc = relic_short(rid)
        lines.append(f"  [{index}] {rid} — {desc}")
    return lines


def render_potions(state: dict[str, Any]) -> list[str]:
    player_top = _as_dict(state.get("player"))
    potions = _as_list(player_top.get("potions"))
    if not potions:
        return []
    out = []
    for idx, p in enumerate(potions):
        if not isinstance(p, dict):
            continue
        pid = _pick(p, "id", "potion_id", default="?")
        out.append(f"  [{idx}] {pid}")
    return out


def _summarize_cards(card_list: list[Any]) -> str:
    """把卡牌列表压缩成 'STRIKE_IRONCLAD×4, DEFEND_IRONCLAD×3, BASH, BLUDGEON+' 形式。"""
    if not card_list:
        return "-"
    counter: Counter = Counter()
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
    parts = []
    for cid, n in counter.most_common():
        parts.append(f"{cid}×{n}" if n > 1 else cid)
    return ", ".join(parts)


def render_piles(state: dict[str, Any]) -> list[str]:
    battle = _as_dict(state.get("battle"))
    draw = _as_list(battle.get("draw_pile_cards"))
    discard = _as_list(battle.get("discard_pile_cards"))
    exhaust = _as_list(battle.get("exhaust_pile_cards"))
    lines = [
        f"  draw[{len(draw)}]: {_summarize_cards(draw)}",
        f"  discard[{len(discard)}]: {_summarize_cards(discard)}",
        f"  exhaust[{len(exhaust)}]: {_summarize_cards(exhaust)}",
    ]
    return lines


def render_deck(state: dict[str, Any]) -> str:
    player_top = _as_dict(state.get("player"))
    deck = _as_list(player_top.get("deck"))
    if not deck:
        return "-"
    return _summarize_cards(deck)


def render_legal_actions(legal_actions: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, action in enumerate(legal_actions):
        if not isinstance(action, dict):
            lines.append(f"  [{index}] <non-dict>")
            continue
        atype = str(_pick(action, "action", "action_type", "type", default="?")).lower()
        card_id = _pick(action, "card_id", default=None)
        card_index = _pick(action, "card_index", "hand_index", default=None)
        target = _pick(action, "target_id", "target", default=None)
        label_text = _pick(action, "label", default=None)
        extras: list[str] = []
        if card_id:
            extras.append(f"card={card_id}")
        if card_index is not None:
            extras.append(f"hand_idx={card_index}")
        if target not in (None, 0, ""):
            extras.append(f"target={target}")
        if atype not in ("play_card", "end_turn"):
            for key in ("node_type", "room_type", "col", "row", "reward_type", "option_text"):
                v = action.get(key)
                if v not in (None, "", 0):
                    extras.append(f"{key}={v}")
            if label_text and str(label_text).strip().lower() != atype:
                extras.append(f'label="{label_text}"')
        out = atype
        if extras:
            out += " " + " ".join(extras)
        lines.append(f"  [{index}] {out}")
    return lines


def render_run_meta(state: dict[str, Any], *, encounter_id: str = "") -> str:
    run = _as_dict(state.get("run"))
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    character = _pick(top_player, "character", default="IRONCLAD")
    floor = _pick(run, "floor_reached", "floor", default="?")
    act = _pick(run, "act", default="?")
    turn_round = _pick(battle, "round_number_raw", "round_number", default="?")
    encounter = encounter_id or _pick(battle, "encounter_id", "encounter", default="?") or "?"
    gold = _pick(top_player, "gold", default=0)
    return f"run: char={character} act={act} floor={floor} encounter={encounter} round={turn_round} gold={gold}"


@dataclass(slots=True, frozen=True)
class RenderedState:
    run_line: str
    player_line: str
    pile_lines: list[str]
    deck_line: str
    relic_lines: list[str]
    potion_lines: list[str]
    enemy_lines: list[str]
    hand_lines: list[str]
    action_lines: list[str]

    def to_user_message(self) -> str:
        parts = [
            self.run_line,
            self.player_line,
        ]
        if self.relic_lines:
            parts.append("relics:")
            parts.extend(self.relic_lines)
        if self.potion_lines:
            parts.append("potions:")
            parts.extend(self.potion_lines)
        parts.append(f"deck: {self.deck_line}")
        parts.append("piles:")
        parts.extend(self.pile_lines)
        parts.append("enemies:")
        parts.extend(self.enemy_lines)
        parts.append("hand:")
        parts.extend(self.hand_lines)
        parts.append("legal_actions:")
        parts.extend(self.action_lines)
        parts.append("请输出一行 JSON，只能从 legal_actions 中选一个 action_index。")
        return "\n".join(parts)


def render_state(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    encounter_id: str = "",
) -> RenderedState:
    return RenderedState(
        run_line=render_run_meta(state, encounter_id=encounter_id),
        player_line=render_player(state),
        pile_lines=render_piles(state),
        deck_line=render_deck(state),
        relic_lines=render_relics(state),
        potion_lines=render_potions(state),
        enemy_lines=render_enemies(state),
        hand_lines=render_hand(state),
        action_lines=render_legal_actions(legal_actions),
    )


def render_state_text(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    encounter_id: str = "",
) -> str:
    return render_state(state, legal_actions, encounter_id=encounter_id).to_user_message()


__all__ = [
    "RenderedState",
    "render_deck",
    "render_enemies",
    "render_hand",
    "render_legal_actions",
    "render_piles",
    "render_player",
    "render_potions",
    "render_relics",
    "render_run_meta",
    "render_state",
    "render_state_text",
]
