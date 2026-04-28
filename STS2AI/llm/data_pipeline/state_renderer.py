"""Render game_bridge state into compact English LLM text.

The renderer intentionally keeps stable game IDs (card/relic/power IDs) and
uses runtime card descriptions from the simulator. It does not translate card
text in Python; HeadlessSim should run with the desired game locale.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Iterable

from llm.data_pipeline.experience_library import (
    DEFAULT_EXPERIENCE_PATH,
    ExperienceEntry,
    load_experience,
    render_experience_block,
    retrieve_experience,
)
from llm.data_pipeline.strategy_context import inject_strategy_context

_LOCALIZATION_KEY_RE = re.compile(
    r"^[A-Za-z0-9_]+\.(?:description|desc)(?:$|\s|\[)"
)
_MARKUP_TAG_RE = re.compile(r"\[/?(?:gold|yellow|red|blue|green|cyan|magenta|white|orange|purple)\]")
_IMG_TAG_RE = re.compile(r"\[img\].*?\[/img\]")
_REPEATED_ENERGY_RE = re.compile(r"(?:\benergy\b\s*){2,}", re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_DAMAGE_DESC_RE = re.compile(
    r"^\s*Deal\s+\d+(?:\s*x\s*\d+)?\s+damage\.\s*",
    re.IGNORECASE,
)
_LEADING_BLOCK_DESC_RE = re.compile(
    r"^\s*Gain\s+\d+\s+Block\.\s*",
    re.IGNORECASE,
)
_SELF_HP_LOSS_RE = re.compile(r"\bLose\s+(\d+)\s+HP\b", re.IGNORECASE)

_RELIC_GLOSSARY = {
    "BURNING_BLOOD": "heal 6 HP after combat.",
    "HAND_DRILL": "when you break an enemy's Block, apply 2 Vulnerable.",
    "MINIATURE_CANNON": "at combat start, deal 7 damage to all enemies.",
    "SILVER_CRUCIBLE": "after combat, 50% chance to heal 4 HP.",
}

_POWER_GLOSSARY = {
    "ARTIFACT_POWER": "negates the next debuff, then loses 1 stack.",
    "VULNERABLE_POWER": "takes 50% more attack damage; usually decreases by 1 each turn.",
    "WEAK_POWER": "deals 25% less attack damage; usually decreases by 1 each turn.",
    "FRAIL_POWER": "gains 25% less Block; usually decreases by 1 each turn.",
    "STRENGTH_POWER": "modifies attack damage by its stack amount.",
    "DEXTERITY_POWER": "modifies Block gained by its stack amount.",
    "CONSTRICT_POWER": "lose HP at end of turn equal to its stack amount; Block does not stop it.",
}

_POTION_GLOSSARY = {
    "FORTIFIER": "gain Block equal to twice your current Block.",
}

_KEYWORD_GLOSSARY = {
    "Artifact": "negates the next debuff, then loses 1 stack.",
    "Vulnerable": "target takes 50% more attack damage.",
    "Weak": "target deals 25% less attack damage.",
    "Frail": "target gains 25% less Block.",
    "Strength": "changes attack damage by its amount.",
    "Dexterity": "changes Block gained by its amount.",
    "Block": "damage shield that is usually lost at end of turn.",
    "Exhaust": "remove the card from combat.",
    "Ethereal": "if still in hand at end of turn, exhaust it.",
    "Retain": "the card stays in hand at end of turn.",
    "Status": "a non-standard deck card, usually harmful or unplayable.",
    "Dazed": "an unplayable Ethereal Status card.",
    "Wound": "an unplayable Status card.",
    "Burn": "a Status card that punishes you if it stays in hand.",
}

_KEYWORD_PATTERNS = {
    key: re.compile(rf"\b{re.escape(key)}(?:ed)?\b", re.IGNORECASE)
    for key in _KEYWORD_GLOSSARY
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _experience_path() -> Path:
    raw = os.environ.get("STS2_LLM_EXPERIENCE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_EXPERIENCE_PATH


_GENERIC_EXPERIENCE_TAGS = {
    "attack",
    "block",
    "combat",
    "defense",
    "draw",
    "end_turn",
    "energy",
    "enemy_intent",
    "incoming_damage",
    "lethal",
    "overkill",
    "reason",
    "target_priority",
    "targeting",
    "tempo",
}

_NON_COMBAT_EXPERIENCE_TAGS = {
    "card_reward",
    "event",
    "map",
    "relic_reward",
    "rest",
    "shop",
}


def _experience_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text or "")}


def _specific_experience_tags(entry: ExperienceEntry) -> list[str]:
    tags: list[str] = []
    for tag in entry.tags:
        clean = str(tag or "").strip().lower()
        if not clean or clean in _GENERIC_EXPERIENCE_TAGS:
            continue
        tags.append(clean)
    return tags


def _filter_relevant_experience(user_message: str, entries: list[ExperienceEntry], *, limit: int) -> list[ExperienceEntry]:
    if not entries or limit <= 0:
        return []
    visible = _experience_tokens(user_message)
    looks_combat = bool(
        re.search(r"^\s+enemy\d+:", user_message, flags=re.MULTILINE)
        or re.search(r"^\s+\[\d+\]\s+[A-Z0-9_+]+\s+cost=", user_message, flags=re.MULTILINE)
        or "play_card" in user_message
    )
    state_type = "combat" if looks_combat else "non_combat"
    out: list[ExperienceEntry] = []
    for entry in entries:
        tags = {str(tag or "").strip().lower() for tag in entry.tags}
        specific_tags = _specific_experience_tags(entry)
        if specific_tags and not any(tag in visible for tag in specific_tags):
            continue
        if state_type != "combat" and not (tags & _NON_COMBAT_EXPERIENCE_TAGS):
            continue
        out.append(entry)
        if len(out) >= limit:
            break
    return out


def inject_experience_context(user_message: str, *, limit: int | None = None) -> str:
    max_entries = _env_int("STS2_LLM_EXPERIENCE_LIMIT", 1) if limit is None else int(limit)
    if max_entries <= 0:
        return user_message
    try:
        candidates = retrieve_experience(
            user_message,
            load_experience(_experience_path()),
            limit=max_entries * 4,
        )
        entries = _filter_relevant_experience(user_message, candidates, limit=max_entries)
    except Exception:
        return user_message
    block = render_experience_block(entries)
    if not block:
        return user_message
    lines = user_message.splitlines()
    if not lines:
        return block
    insert_at = 1
    for index, line in enumerate(lines):
        if line.startswith("player:"):
            insert_at = index
            break
    return "\n".join([*lines[:insert_at], block, *lines[insert_at:]])


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _pick(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _indexed_item(items: list[Any], raw_index: Any) -> dict[str, Any] | None:
    index = _to_int(raw_index)
    if index is None:
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        item_index = _to_int(_pick(item, "index", default=None))
        if item_index == index:
            return item
    if 0 <= index < len(items) and isinstance(items[index], dict):
        return items[index]
    return None


def _enemy_target_id(enemy: dict[str, Any], fallback: int) -> Any:
    target_id = _pick(enemy, "target_id", "combat_id", default=None)
    if target_id not in (None, ""):
        return target_id
    raw_id = _pick(enemy, "id", default=None)
    if isinstance(raw_id, int) or (isinstance(raw_id, str) and raw_id.isdigit()):
        return raw_id
    return fallback


def _is_localization_key(text: str) -> bool:
    if not text:
        return True
    stripped = text.strip()
    return bool(_LOCALIZATION_KEY_RE.match(stripped))


def _clean_runtime_text(text: str) -> str:
    if not text:
        return ""

    def _replace_img(match: re.Match[str]) -> str:
        tag = match.group(0).lower()
        if "star_icon" in tag:
            return " star "
        if "energy_icon" in tag:
            return " energy "
        return ""

    text = _IMG_TAG_RE.sub(_replace_img, text)
    text = _MARKUP_TAG_RE.sub("", text)
    text = text.replace("\\n", " ").replace("\n", " ")
    text = text.replace("；", ";")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _REPEATED_ENERGY_RE.sub(
        lambda match: f"{len(re.findall(r'energy', match.group(0), re.IGNORECASE))} Energy ",
        text,
    )
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text


def _fmt_powers(powers: Iterable[Any]) -> str:
    parts: list[str] = []
    for power in powers or []:
        if not isinstance(power, dict):
            continue
        pid = str(power.get("id") or power.get("power_id") or power.get("name") or "").upper()
        amount = power.get("amount")
        if not pid:
            continue
        if amount in (None, "", 0):
            parts.append(pid)
        else:
            parts.append(f"{pid}={amount}")
    return "; ".join(parts) if parts else "-"


def _resolve_card_desc(card: dict[str, Any]) -> str:
    sim_desc = str(card.get("description") or "")
    if sim_desc and not _is_localization_key(sim_desc):
        return _clean_runtime_text(sim_desc)
    return ""


def _strip_preview_repeated_card_text(card: dict[str, Any], desc: str) -> str:
    if not desc:
        return ""
    out = desc
    if card.get("preview_damage_per_target"):
        out = _LEADING_DAMAGE_DESC_RE.sub("", out)
    if card.get("preview_block"):
        out = _LEADING_BLOCK_DESC_RE.sub("", out)
    return out.strip()


def _item_description(item: dict[str, Any], fallback_by_id: dict[str, str], item_id: str) -> str:
    for key in ("description", "desc", "tooltip", "text"):
        desc = str(item.get(key) or "")
        if desc and not _is_localization_key(desc):
            return _clean_runtime_text(desc)
    return fallback_by_id.get(item_id.upper(), "")


def _iter_relics(state: dict[str, Any]) -> list[dict[str, Any]]:
    player_top = _as_dict(state.get("player"))
    battle = _as_dict(state.get("battle"))
    battle_player = _as_dict(battle.get("player"))
    return [r for r in (_as_list(player_top.get("relics")) or _as_list(battle_player.get("relics"))) if isinstance(r, dict)]


def _iter_hand_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    hand_raw = (
        _as_list(battle.get("hand"))
        or _as_list(battle_player.get("hand"))
        or _as_list(state.get("hand"))
        or _as_list(top_player.get("hand"))
    )
    return [card for card in hand_raw if isinstance(card, dict)]


def _iter_powers(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    powers = list(_as_list(battle_player.get("powers")) or _as_list(top_player.get("powers")))
    for enemy in _as_list(state.get("enemies")) or _as_list(battle.get("enemies")):
        if isinstance(enemy, dict):
            powers.extend(_as_list(enemy.get("powers")) or _as_list(enemy.get("buffs")))
    return [power for power in powers if isinstance(power, dict)]


def render_player(state: dict[str, Any]) -> str:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    energy_state = _as_dict(state.get("energy"))
    hp = _pick(top_player, "hp", "current_hp", default=0)
    max_hp = _pick(top_player, "max_hp", default=0)
    block = _pick(battle_player, "block", default=_pick(top_player, "block", default=0))
    energy = _pick(battle, "energy", default=_pick(top_player, "energy", default=_pick(energy_state, "current", default=0)))
    max_energy = _pick(battle, "max_energy", default=_pick(top_player, "max_energy", default=_pick(energy_state, "max", default=0)))
    powers = _fmt_powers(_as_list(battle_player.get("powers")) or _as_list(top_player.get("powers")))
    return f"player: hp={hp}/{max_hp} block={block} energy={energy}/{max_energy} powers={powers}"


def render_enemies(state: dict[str, Any]) -> list[str]:
    battle = _as_dict(state.get("battle"))
    enemies_raw = _as_list(state.get("enemies")) or _as_list(battle.get("enemies"))
    lines: list[str] = []
    for index, enemy in enumerate(enemies_raw):
        if not isinstance(enemy, dict):
            continue
        eid = _pick(enemy, "monster_id", "entity_id", "name", "id", default=f"enemy_{index}")
        target_id = _enemy_target_id(enemy, index + 1)
        hp = _pick(enemy, "hp", "current_hp", default=0)
        max_hp = _pick(enemy, "max_hp", default=0)
        block = _pick(enemy, "block", default=0)
        intent = _pick(enemy, "intent_type", "next_move_id", "intent", default="?")
        dmg = _pick(enemy, "intent_damage", "move_base_damage", default=None)
        hits = _pick(enemy, "intent_hits", "move_hits", default=None)
        intent_str = str(intent)
        if dmg:
            intent_str += f"({dmg}"
            if hits and int(hits) > 1:
                intent_str += f"x{hits}"
            intent_str += ")"
        powers = _fmt_powers(_as_list(enemy.get("powers")) or _as_list(enemy.get("buffs")))
        alive = bool(_pick(enemy, "is_alive", "alive", default=True))
        tag = "" if alive else " [dead]"
        lines.append(
            f"  enemy{target_id}: {eid} hp={hp}/{max_hp} block={block} intent={intent_str} powers={powers}{tag}"
        )
    return lines


def _structured_play_hints(legal_actions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    hints: dict[int, dict[str, Any]] = {}
    for action in legal_actions:
        if not isinstance(action, dict) or action.get("is_enabled") is False:
            continue
        if _action_type(action) != "play_card":
            continue
        ci = _action_card_index(action)
        if ci is None or ci < 0:
            continue
        hint = hints.setdefault(ci, {"self": False, "targets": set()})
        target_id = _action_target_id(action)
        if target_id is None:
            hint["self"] = True
        else:
            hint["targets"].add(target_id)
    return hints


def render_hand(
    state: dict[str, Any],
    *,
    play_hints: dict[int, dict[str, Any]] | None = None,
    include_previews: bool = False,
    strip_preview_text: bool = False,
) -> list[str]:
    lines: list[str] = []
    for index, card in enumerate(_iter_hand_cards(state)):
        cid = _pick(card, "id", "card_id", default="?")
        cost = _pick(card, "cost", "cost_now", default=0)
        card_type = str(_pick(card, "type", default="")).strip()
        is_upg = bool(_pick(card, "is_upgraded", default=False))
        can_play = bool(_pick(card, "can_play", default=True))

        # Prefer real-time simulator previews.
        preview_dmg = card.get("preview_damage_per_target") or {}
        preview_block = card.get("preview_block") or 0
        desc = _resolve_card_desc(card)
        if strip_preview_text and not include_previews:
            desc = _strip_preview_repeated_card_text(card, desc)

        damage_preview_str = ""
        if preview_dmg:
            vals = [f"enemy{tid}={d}" for tid, d in preview_dmg.items() if d]
            if vals:
                damage_preview_str = " damage_preview=" + ",".join(vals)
        block_preview_str = f" block_preview={preview_block}" if preview_block else ""

        attrs = [f"cost={cost}"]
        if card_type:
            attrs.append(f"type={card_type}")
        if is_upg:
            attrs.append("upgraded=true")
        if include_previews and damage_preview_str:
            attrs.append(damage_preview_str.strip())
        if include_previews and block_preview_str:
            attrs.append(block_preview_str.strip())
        if include_previews and not can_play:
            attrs.append("playable=false")
        if play_hints is not None:
            hint = play_hints.get(index)
            if hint is not None:
                targets = sorted(hint.get("targets") or [])
                if targets:
                    attrs.append("legal_targets=" + ",".join(f"enemy{tid}" for tid in targets))
                elif hint.get("self"):
                    attrs.append("legal_target=self")

        desc_snippet = (desc[:80] + "...") if desc and len(desc) > 80 else desc
        desc_part = f" | {desc_snippet}" if desc_snippet else ""
        lines.append(
            f"  [{index}] {cid} {' '.join(attrs)}{desc_part}"
        )
    return lines


def render_relics(state: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for index, relic in enumerate(_iter_relics(state)):
        rid = str(_pick(relic, "id", "relic_id", default="?"))
        counter = _pick(relic, "counter", "amount", default=None)
        suffix = f":{counter}" if counter not in (None, "", 0) else ""
        desc = _item_description(relic, _RELIC_GLOSSARY, rid)
        desc_part = f" | {desc}" if desc else ""
        lines.append(f"  [{index}] {rid}{suffix}{desc_part}")
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
        desc = _item_description(p, _POTION_GLOSSARY, str(pid))
        desc_part = f" | {desc}" if desc else ""
        out.append(f"  [{idx}] {pid}{desc_part}")
    return out


def _visible_text_fragments(state: dict[str, Any]) -> list[str]:
    battle = _as_dict(state.get("battle"))
    fragments: list[str] = []

    for relic in _iter_relics(state):
        rid = str(_pick(relic, "id", "relic_id", default=""))
        fragments.append(rid)
        fragments.append(_item_description(relic, _RELIC_GLOSSARY, rid))

    for power in _iter_powers(state):
        pid = str(power.get("id") or power.get("power_id") or power.get("name") or "")
        fragments.append(pid)
        fragments.append(_item_description(power, _POWER_GLOSSARY, pid))

    def add_card(card: Any) -> None:
        if isinstance(card, dict):
            fragments.append(str(_pick(card, "id", "card_id", default="")))
            fragments.append(_resolve_card_desc(card))
            for keyword in _as_list(card.get("keywords")):
                fragments.append(str(keyword))
        elif isinstance(card, str):
            fragments.append(card)

    for card in _iter_hand_cards(state):
        add_card(card)
    for card in _as_list(_as_dict(state.get("card_reward")).get("cards")):
        add_card(card)
    for card in _as_list(_as_dict(state.get("card_select")).get("cards")):
        add_card(card)
    for key in ("draw_pile_cards", "discard_pile_cards", "exhaust_pile_cards"):
        for card in _as_list(battle.get(key)):
            add_card(card)

    return [fragment for fragment in fragments if fragment]


def render_glossary(state: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()

    for power in _iter_powers(state):
        pid = str(power.get("id") or power.get("power_id") or power.get("name") or "").upper()
        if not pid or pid in seen:
            continue
        desc = _item_description(power, _POWER_GLOSSARY, pid)
        if desc:
            lines.append(f"  {pid}: {desc}")
            seen.add(pid)

    visible_text = "\n".join(_visible_text_fragments(state))
    for keyword, pattern in _KEYWORD_PATTERNS.items():
        if keyword in seen:
            continue
        if pattern.search(visible_text):
            lines.append(f"  {keyword}: {_KEYWORD_GLOSSARY[keyword]}")
            seen.add(keyword)

    return lines


def _summarize_cards(card_list: list[Any]) -> str:
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
        parts.append(f"{cid}x{n}" if n > 1 else cid)
    return ", ".join(parts)


def render_piles(state: dict[str, Any]) -> list[str]:
    battle = _as_dict(state.get("battle"))
    draw = _as_list(battle.get("draw_pile_cards"))
    discard = _as_list(battle.get("discard_pile_cards"))
    exhaust = _as_list(battle.get("exhaust_pile_cards"))
    lines = [f"  draw={len(draw)} discard={len(discard)} exhaust={len(exhaust)}"]
    if 0 < len(draw) <= 8:
        lines.append(f"  draw_cards: {_summarize_cards(draw)}")
    if len(exhaust) > 0:
        lines.append(f"  exhaust_cards: {_summarize_cards(exhaust)}")
    return lines


def render_deck(state: dict[str, Any]) -> str:
    player_top = _as_dict(state.get("player"))
    deck = _as_list(player_top.get("deck"))
    if not deck:
        return "-"
    return _summarize_cards(deck)


def _compact_inline_text(value: Any, *, max_chars: int = 180) -> str:
    text = _WHITESPACE_RE.sub(" ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def render_event(state: dict[str, Any]) -> list[str]:
    event = _as_dict(state.get("event"))
    if not event:
        return []
    lines: list[str] = []
    event_id = _compact_inline_text(_pick(event, "event_id", "id", default=""))
    event_name = _compact_inline_text(_pick(event, "event_name", "name", "title", default=""))
    header = event_name or event_id
    if header:
        suffix = f" id={event_id}" if event_id and event_id != header else ""
        lines.append(f"  name={header}{suffix}")
    body = _compact_inline_text(_pick(event, "body", "description", "text", default=""), max_chars=260)
    if body:
        lines.append(f"  body={body}")
    options = [option for option in _as_list(event.get("options")) if isinstance(option, dict)]
    if options:
        lines.append("  options:")
        for option in options[:8]:
            index = _pick(option, "index", default="?")
            label = _compact_inline_text(_pick(option, "label", "title", "text", default=""))
            text = _compact_inline_text(_pick(option, "text", "title", default=""))
            attrs = [f"[{index}]"]
            if label:
                attrs.append(f'label="{label}"')
            if text and text != label:
                attrs.append(f'text="{text}"')
            if bool(_pick(option, "is_locked", default=False)):
                attrs.append("locked=true")
            if bool(_pick(option, "is_proceed", default=False)):
                attrs.append("proceed=true")
            lines.append("    " + " ".join(attrs))
    return lines


def render_legal_actions(legal_actions: list[dict[str, Any]], state: dict[str, Any] | None = None) -> list[str]:
    hand_by_index: dict[int, str] = {}
    hand_cards_by_index: dict[int, dict[str, Any]] = {}
    enemies_by_target_id: dict[int, dict[str, Any]] = {}
    card_reward_cards: list[Any] = []
    card_select: dict[str, Any] = {}
    event_options: list[Any] = []
    player_hp = 0.0
    if state is not None:
        battle = _as_dict(state.get("battle"))
        top_player = _as_dict(state.get("player"))
        battle_player = _as_dict(battle.get("player"))
        card_select = _as_dict(state.get("card_select"))
        event_options = _as_list(_as_dict(state.get("event")).get("options"))
        try:
            player_hp = float(_pick(top_player, "hp", "current_hp", default=_pick(battle_player, "hp", default=0)) or 0)
        except (TypeError, ValueError):
            player_hp = 0.0
        card_reward_cards = _as_list(_as_dict(state.get("card_reward")).get("cards"))
        hand = (
            _as_list(battle.get("hand"))
            or _as_list(battle_player.get("hand"))
            or _as_list(state.get("hand"))
            or _as_list(top_player.get("hand"))
        )
        for idx, card in enumerate(hand):
            if isinstance(card, dict):
                hand_by_index[idx] = str(_pick(card, "id", "card_id", default=f"card{idx}"))
                hand_cards_by_index[idx] = card
        enemies = _as_list(state.get("enemies")) or _as_list(battle.get("enemies"))
        for fallback_idx, enemy in enumerate(enemies, start=1):
            if not isinstance(enemy, dict):
                continue
            raw_target = _pick(enemy, "target_id", "combat_id", "id", default=fallback_idx)
            try:
                target_id = int(raw_target)
            except (TypeError, ValueError):
                target_id = fallback_idx
            enemies_by_target_id[target_id] = enemy

    def fmt_target(value: Any) -> str:
        if value in (None, 0, "", -1, "0", "-1"):
            return "self"
        return f"enemy{value}"

    def fmt_num(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:.1f}".rstrip("0").rstrip(".")

    def enemy_for_target(value: Any) -> dict[str, Any]:
        try:
            return enemies_by_target_id.get(int(value), {})
        except (TypeError, ValueError):
            return {}

    def selection_purpose(action: dict[str, Any] | None = None) -> str:
        texts: list[str] = []
        if action:
            texts.extend(str(_pick(action, key, default="")) for key in ("label", "purpose", "operation"))
        texts.extend(str(_pick(card_select, key, default="")) for key in ("purpose", "operation", "prompt", "screen_type"))
        merged = " ".join(texts).lower()
        if any(token in merged for token in ("remove", "purge", "delete", "移除", "删除")):
            return "remove_card"
        if any(token in merged for token in ("transform", "转化", "变化", "变换")):
            return "transform_card"
        if any(token in merged for token in ("upgrade", "升级", "强化", "smith")):
            return "upgrade_card"
        return ""

    def select_card_detail(index: int, action: dict[str, Any]) -> str:
        atype = str(_pick(action, "action", "action_type", "type", default="?")).lower()
        card_id = str(_pick(action, "card_id", default="") or "").strip()
        choice_index = _pick(action, "index", "card_index", "hand_index", default=None)
        source_card_index = _pick(action, "card_index", "hand_index", default=None)
        label = str(_pick(action, "label", default="") or "").strip()
        if not card_id:
            card = _indexed_item(_as_list(card_select.get("cards")), choice_index)
            card_id = str(_pick(card or {}, "id", "card_id", default="") or "").strip()
        if not card_id:
            card = _indexed_item(_as_list(card_select.get("cards")), source_card_index)
            card_id = str(_pick(card or {}, "id", "card_id", default="") or "").strip()
        if not card_id and label:
            card_id = label.split()[-1]
        purpose = selection_purpose(action)
        extras: list[str] = []
        if purpose:
            extras.append(f"purpose={purpose}")
        if card_id:
            extras.append(f"card={card_id}")
        if choice_index is not None:
            extras.append(f"choice_idx={choice_index}")
        if source_card_index not in (None, choice_index):
            extras.append(f"source_hand_idx={source_card_index}")
        if purpose == "remove_card":
            upper = card_id.upper()
            if upper.startswith("STRIKE_") or upper.startswith("DEFEND_"):
                extras.append("priority=starter_basic")
            elif upper in {"BASH", "NEUTRALIZE", "SURVIVOR", "ZAP", "DUALCAST"}:
                extras.append("priority=protected_starter_key")
        return f"  [{index}] {atype}" + ((" " + " ".join(extras)) if extras else "")

    def action_preview(ci: int, target: Any) -> str:
        card = hand_cards_by_index.get(ci)
        if not card:
            return ""
        parts: list[str] = []
        if target in (None, 0, "", -1, "0", "-1"):
            block = card.get("preview_block") or 0
            if block:
                parts.append(f"block={block}")
            match = _SELF_HP_LOSS_RE.search(_resolve_card_desc(card))
            if match:
                hp_loss = int(match.group(1))
                parts.append(f"self_hp_loss={hp_loss}")
                if player_hp > 0:
                    parts.append(f"self_hp_after={fmt_num(max(0.0, player_hp - hp_loss))}")
        else:
            preview_dmg = card.get("preview_damage_per_target") or {}
            damage = None
            for key in (target, str(target)):
                if key in preview_dmg:
                    damage = preview_dmg[key]
                    break
            if damage not in (None, "", 0):
                parts.append(f"damage={damage}")
                enemy = enemy_for_target(target)
                if enemy:
                    try:
                        damage_value = float(damage)
                        hp = float(_pick(enemy, "hp", "current_hp", default=0) or 0)
                        block = float(_pick(enemy, "block", default=0) or 0)
                    except (TypeError, ValueError):
                        hp = 0.0
                        block = 0.0
                        damage_value = 0.0
                    hp_damage = min(hp, max(0.0, damage_value - block))
                    lethal = hp > 0 and hp_damage >= hp
                    parts.append(f"hp={fmt_num(hp)}")
                    if block > 0:
                        parts.append(f"block={fmt_num(block)}")
                        parts.append(f"hp_damage={fmt_num(hp_damage)}")
                    parts.append(f"lethal={str(lethal).lower()}")
        return (" " + " ".join(parts)) if parts else ""

    def card_reward_detail(action: dict[str, Any]) -> tuple[list[str], str]:
        option_index = _pick(action, "index", "card_index", "hand_index", default=None)
        card = _indexed_item(card_reward_cards, option_index)
        if not card:
            return [], ""
        card_id = str(_pick(card, "id", "card_id", default="")).strip()
        attrs: list[str] = []
        cost = _pick(card, "cost", "cost_now", default=None)
        if cost not in (None, ""):
            attrs.append(f"cost={cost}")
        card_type = str(_pick(card, "type", "card_type", default="")).strip()
        if card_type:
            attrs.append(f"type={card_type}")
        rarity = str(_pick(card, "rarity", default="")).strip()
        if rarity:
            attrs.append(f"rarity={rarity}")
        if bool(_pick(card, "is_upgraded", default=False)):
            attrs.append("upgraded=true")
        keywords = [str(k) for k in _as_list(card.get("keywords")) if str(k).strip()]
        if keywords:
            attrs.append("keywords=" + ",".join(keywords[:6]))
        desc = _resolve_card_desc(card)
        desc_part = f" | {desc}" if desc else ""
        if card_id and not action.get("card_id"):
            attrs.insert(0, f"card={card_id}")
        return attrs, desc_part

    def fmt_other(index: int, action: dict[str, Any]) -> str:
        atype = str(_pick(action, "action", "action_type", "type", default="?")).lower()
        if atype in {"select_card", "select_card_option", "combat_select_card"}:
            return select_card_detail(index, action)
        card_id = _pick(action, "card_id", default=None)
        card_index = _pick(action, "card_index", "hand_index", default=None)
        target = _pick(action, "target_id", "target", default=None)
        label_text = _pick(action, "label", default=None)
        extras: list[str] = []
        option_index = _pick(action, "index", default=None)
        event_option = (
            _indexed_item(event_options, option_index)
            if atype == "choose_event_option"
            else None
        )
        if atype in {"confirm_selection", "cancel_selection"}:
            purpose = selection_purpose(action)
            selected = [
                str(_pick(card, "id", "card_id", default="")).strip()
                for card in _as_list(card_select.get("selected_cards"))
                if isinstance(card, dict) and str(_pick(card, "id", "card_id", default="")).strip()
            ]
            if purpose:
                extras.append(f"purpose={purpose}")
            if selected:
                extras.append("selected=" + ",".join(selected[:5]))
        if card_id:
            extras.append(f"card={card_id}")
        if atype not in ("end_turn", "select_card_reward") and card_index is not None:
            extras.append(f"hand_idx={card_index}")
        if target not in (None, 0, ""):
            extras.append(f"target={target}")
        desc_part = ""
        if atype == "select_card_reward":
            reward_attrs, desc_part = card_reward_detail(action)
            existing = {part.split("=", 1)[0] for part in extras if "=" in part}
            for attr in reward_attrs:
                key = attr.split("=", 1)[0]
                if key not in existing:
                    extras.append(attr)
                    existing.add(key)
        if atype not in ("play_card", "end_turn"):
            for key in ("node_type", "room_type", "col", "row", "reward_type", "option_text"):
                v = action.get(key)
                if v not in (None, "", 0):
                    extras.append(f"{key}={v}")
            if event_option:
                option_text = _compact_inline_text(_pick(event_option, "text", "title", default=""))
                option_label = _compact_inline_text(_pick(event_option, "label", default=""))
                if option_text and option_text != str(label_text or "").strip():
                    extras.append(f'option_text="{option_text}"')
                if option_label and not label_text:
                    label_text = option_label
            if label_text and str(label_text).strip().lower() != atype:
                extras.append(f'label="{label_text}"')
        out = atype
        if extras:
            out += " " + " ".join(extras)
        return f"  [{index}] {out}{desc_part}"

    entries: list[tuple[str, int | str]] = []
    groups: dict[int, dict[str, Any]] = {}
    for index, action in enumerate(legal_actions):
        if not isinstance(action, dict):
            entries.append(("other", f"  [{index}] <non-dict>"))
            continue
        atype = str(_pick(action, "action", "action_type", "type", default="?")).lower()
        card_id = _pick(action, "card_id", default=None)
        card_index = _pick(action, "card_index", "hand_index", default=None)
        target = _pick(action, "target_id", "target", default=None)
        if atype == "play_card" and card_index is not None:
            try:
                ci = int(card_index)
            except (TypeError, ValueError):
                ci = -1
            if ci >= 0:
                if ci not in groups:
                    entries.append(("card", ci))
                    groups[ci] = {
                        "card_id": str(card_id or hand_by_index.get(ci, f"card{ci}")),
                        "actions": [],
                    }
                groups[ci]["actions"].append((index, target))
                continue
        entries.append(("other", fmt_other(index, action)))

    lines: list[str] = []
    for kind, value in entries:
        if kind == "other":
            lines.append(str(value))
            continue
        ci = int(value)
        group = groups[ci]
        cid = str(group["card_id"])
        actions = list(group["actions"])
        if len(actions) == 1:
            action_index, target = actions[0]
            preview = action_preview(ci, target)
            lines.append(f"  [{action_index}] {cid} hand[{ci}] target={fmt_target(target)}{preview}")
            continue
        lines.append(f"  {cid} hand[{ci}]:")
        for action_index, target in actions:
            preview = action_preview(ci, target)
            lines.append(f"    [{action_index}] target={fmt_target(target)}{preview}")
    return lines


def _action_type(action: dict[str, Any]) -> str:
    raw = str(_pick(action, "action", "action_type", "type", default="?")).strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"play", "playcard", "play_card", "card"}:
        return "play_card"
    if raw in {"end", "endturn", "end_turn", "pass", "done"}:
        return "end_turn"
    return raw


def _action_card_index(action: dict[str, Any]) -> int | None:
    raw = _pick(action, "card_index", "hand_index", default=None)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _action_target_id(action: dict[str, Any]) -> int | None:
    raw = _pick(action, "target_id", "target", default=None)
    if raw in (None, "", 0, -1, "0", "-1"):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def can_render_structured_actions(legal_actions: list[dict[str, Any]]) -> bool:
    """Whether the compressed structured-action prompt can cover this action set."""
    enabled = [
        action for action in (legal_actions or [])
        if isinstance(action, dict) and action.get("is_enabled") is not False
    ]
    if not enabled:
        return False
    for action in enabled:
        atype = _action_type(action)
        if atype == "end_turn":
            continue
        if atype == "play_card" and _action_card_index(action) is not None:
            continue
        return False
    return True


def render_structured_legal_actions(
    legal_actions: list[dict[str, Any]],
    state: dict[str, Any] | None = None,
) -> list[str]:
    """Render only command schemas; per-card legality lives in `hand:` lines."""
    has_play_card = False
    has_end_turn = False
    for action in legal_actions:
        if not isinstance(action, dict) or action.get("is_enabled") is False:
            continue
        atype = _action_type(action)
        if atype == "end_turn":
            has_end_turn = True
            continue
        if atype == "play_card" and _action_card_index(action) is not None:
            has_play_card = True

    lines: list[str] = []
    if has_play_card:
        lines.append("  play_card: use a hand index with legal_target or legal_targets")
    if has_end_turn:
        lines.append("  end_turn")
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
    gold = _pick(top_player, "gold", default=_pick(run, "gold", default=0))
    return f"run: char={character} act={act} floor={floor} encounter={encounter} round={turn_round} gold={gold}"


@dataclass(slots=True, frozen=True)
class RenderedState:
    run_line: str
    player_line: str
    event_lines: list[str]
    pile_lines: list[str]
    deck_line: str
    relic_lines: list[str]
    potion_lines: list[str]
    enemy_lines: list[str]
    hand_lines: list[str]
    action_lines: list[str]
    glossary_lines: list[str]

    def to_user_message(self) -> str:
        return self._to_user_message(
              action_heading="legal_actions:",
              return_line=(
                'Return strict JSON only: {"action_index":N,"confidence":0.0,"reason":"..."} '
                  "using one listed action_index. Do not output multiple objects or candidates."
              ),
        )

    def to_structured_action_user_message(self) -> str:
        return self._to_user_message(
            action_heading="commands:",
            return_line=(
                "Return one JSON line using only commands and hand legality. "
                'Targeted card schema: {"action":"play_card","hand_index":HAND,"target_id":ENEMY,"reason":"..."}. '
                'Self/no-target schema: {"action":"play_card","hand_index":HAND,"reason":"..."}. '
                'End turn schema: {"action":"end_turn","reason":"..."}.'
            ),
        )

    def _to_user_message(self, *, action_heading: str, return_line: str) -> str:
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
        if self.event_lines:
            parts.append("event:")
            parts.extend(self.event_lines)
        parts.append(f"deck: {self.deck_line}")
        parts.append("piles:")
        parts.extend(self.pile_lines)
        parts.append("enemies:")
        parts.extend(self.enemy_lines)
        parts.append("hand:")
        parts.extend(self.hand_lines)
        parts.append(action_heading)
        parts.extend(self.action_lines)
        if self.glossary_lines:
            parts.append("glossary:")
            parts.extend(self.glossary_lines)
        parts.append(return_line)
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
        event_lines=render_event(state),
        pile_lines=render_piles(state),
        deck_line=render_deck(state),
        relic_lines=render_relics(state),
        potion_lines=render_potions(state),
        enemy_lines=render_enemies(state),
        hand_lines=render_hand(state),
        action_lines=render_legal_actions(legal_actions, state),
        glossary_lines=render_glossary(state),
    )


def render_state_text(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    encounter_id: str = "",
    strategy_context: str = "",
) -> str:
    return inject_experience_context(inject_strategy_context(
        render_state(state, legal_actions, encounter_id=encounter_id).to_user_message(),
        strategy_context,
    ))


def render_structured_action_state(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    encounter_id: str = "",
) -> RenderedState:
    return RenderedState(
        run_line=render_run_meta(state, encounter_id=encounter_id),
        player_line=render_player(state),
        event_lines=render_event(state),
        pile_lines=render_piles(state),
        deck_line=render_deck(state),
        relic_lines=render_relics(state),
        potion_lines=render_potions(state),
        enemy_lines=render_enemies(state),
        hand_lines=render_hand(state, play_hints=_structured_play_hints(legal_actions)),
        action_lines=render_structured_legal_actions(legal_actions, state),
        glossary_lines=render_glossary(state),
    )


def render_structured_action_state_text(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    encounter_id: str = "",
    strategy_context: str = "",
) -> str:
    return inject_experience_context(inject_strategy_context(
        render_structured_action_state(
            state,
            legal_actions,
            encounter_id=encounter_id,
        ).to_structured_action_user_message(),
        strategy_context,
    ))


__all__ = [
    "RenderedState",
    "can_render_structured_actions",
    "render_deck",
    "render_enemies",
    "render_hand",
    "inject_experience_context",
    "render_legal_actions",
    "render_glossary",
    "render_piles",
    "render_player",
    "render_potions",
    "render_relics",
    "render_run_meta",
    "render_state",
    "render_state_text",
    "render_structured_action_state",
    "render_structured_action_state_text",
    "render_structured_legal_actions",
]
