"""Planner-hint helpers.

Planner-hint is a combat-level strategy note for the combat policy prompt. It
is not an executor: it must not contain action_index or an action sequence.
The combat LoRA still selects exactly one current legal action.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from llm.data_pipeline.guide_knowledge import render_retrieved_knowledge_for_state
from llm.data_pipeline.state_renderer import (
    render_deck,
    render_enemies,
    render_glossary,
    render_hand,
    render_piles,
    render_player,
    render_potions,
    render_relics,
    render_run_meta,
)


# ---------------------------------------------------------------------------
# Defaults shared by all CLI entry points (grpo_rollout / policy_eval /
# fullrun_eval / self_iterate / self_train_loop) and runtime LLM policy.
# Keep this as the single source of truth so we don't drift defaults across
# argparse declarations.
# ---------------------------------------------------------------------------
PLANNER_HINT_REFRESH_CHOICES: tuple[str, ...] = ("combat", "turn")
DEFAULT_PLANNER_HINT_REFRESH: str = "turn"


TEXT_KEYS = (
    "battle_objective",
    "enemy_focus",
    "deck_usage",
    "risk_tradeoff",
    "resource_timing",
    "potion_stance",
)
LIST_KEYS = ("kill_order", "danger_notes")
ALLOWED_KEYS = set(TEXT_KEYS) | set(LIST_KEYS)
LEGACY_KEYS = {
    "combat_plan",
    "encounter_guide",
    "defense_policy",
    "resource_policy",
    "potion_policy",
}
FORBIDDEN_KEYS = {
    "action",
    "action_index",
    "actions",
    "best_action_index",
    "legal_action_index",
    "sequence",
    "turn_plan",
}

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_FINAL_JSON_RE = re.compile(r"<FINAL_JSON>\s*(.*?)\s*</FINAL_JSON>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _pick(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _id(value: Any) -> str:
    if isinstance(value, dict):
        return str(_pick(value, "id", "card_id", "relic_id", "monster_id", "entity_id", default="")).upper()
    return str(value or "").upper()


def _clean_text(value: Any, *, max_chars: int = 220) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text[:max_chars].rstrip()


def _clean_list(value: Any, *, max_items: int = 6, max_chars: int = 120) -> list[str]:
    raw_items = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in raw_items:
        text = _clean_text(item, max_chars=max_chars)
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in FORBIDDEN_KEYS:
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _contains_cjk_text(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_cjk_text(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_cjk_text(item) for item in value)
    return bool(_CJK_RE.search(str(value or "")))


def _strip_model_wrappers(raw_text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", raw_text or "").strip()
    final = _FINAL_JSON_RE.search(text)
    if final:
        return final.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _json_object_from_text(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    text = _strip_model_wrappers(raw_text)
    try:
        payload = json.loads(text)
        return (payload, "ok") if isinstance(payload, dict) else (None, "json_not_object")
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, "ok"
    return None, "json_parse_failed"


def normalize_planner_hint(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only the approved battle-level hint fields."""
    hint: dict[str, Any] = {}
    for key in TEXT_KEYS:
        value = _clean_text(payload.get(key), max_chars=240)
        if value:
            hint[key] = value
    for key in LIST_KEYS:
        values = _clean_list(payload.get(key), max_items=6, max_chars=120)
        if values:
            hint[key] = values
    return hint


def parse_planner_hint_json(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    payload, status = _json_object_from_text(raw_text)
    if payload is None:
        return None, status
    if _contains_forbidden_key(payload):
        return None, "forbidden_action_fields"
    keys = {str(key).strip() for key in payload.keys()}
    if keys & LEGACY_KEYS:
        return None, "legacy_planner_hint_fields"
    if keys - ALLOWED_KEYS:
        return None, "unknown_planner_hint_fields"
    hint = normalize_planner_hint(payload)
    if not hint:
        return None, "empty_hint"
    if _contains_cjk_text(hint):
        return None, "non_english_text"
    return hint, "ok"


def format_planner_hint(hint: dict[str, Any] | str) -> str:
    """Render only the inner planner_hint lines, without the heading."""
    if isinstance(hint, str):
        text = hint.strip()
        if not text:
            return ""
        if text.startswith("planner_hint:"):
            lines = text.splitlines()[1:]
            return "\n".join(line[2:] if line.startswith("  ") else line for line in lines).strip()
        return text

    lines: list[str] = []
    for key in TEXT_KEYS:
        value = _clean_text(hint.get(key), max_chars=260)
        if value:
            lines.append(f"{key}: {value}")
    kill_order = _clean_list(hint.get("kill_order"), max_items=6)
    if kill_order:
        lines.append("kill_order: " + " -> ".join(kill_order))
    danger_notes = _clean_list(hint.get("danger_notes"), max_items=6)
    if danger_notes:
        lines.append("danger_notes: " + "; ".join(danger_notes))
    return "\n".join(lines)


def _state_fingerprint(state: dict[str, Any]) -> str:
    player = _as_dict(state.get("player"))
    battle = _as_dict(state.get("battle"))
    run = _as_dict(state.get("run"))
    deck = _as_list(player.get("deck"))
    deck_ids = [_id(card) for card in deck[:80]]
    payload = {
        "act": _pick(run, "act", default="?"),
        "floor": _pick(run, "floor_reached", "floor", default="?"),
        "seed": _pick(run, "seed", "run_seed", default=""),
        "combat": _pick(
            battle,
            "combat_key",
            "encounter_key",
            "combat_id",
            default=_pick(state, "combat_key", "encounter_key", "combat_id", default=""),
        ),
        "encounter": _pick(battle, "encounter_id", "encounter", default=""),
        "deck": deck_ids,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def planner_hint_cache_key(state: dict[str, Any], *, refresh: str = DEFAULT_PLANNER_HINT_REFRESH) -> str:
    battle = _as_dict(state.get("battle"))
    base = _state_fingerprint(state)
    if refresh == "turn":
        turn = _pick(battle, "round_number_raw", "round_number", default="?")
        return f"{base}:turn={turn}"
    return base


def _append_section(lines: list[str], title: str, section_lines: list[str]) -> None:
    if not section_lines:
        return
    lines.append(f"{title}:")
    lines.extend(section_lines)


def _memory_block(memory: str) -> list[str]:
    clean = (memory or "").strip()
    if not clean:
        return []
    out = ["memory_context:"]
    out.extend(f"  {line}" if line else "" for line in clean.splitlines())
    return out


def _knowledge_block(knowledge: str) -> list[str]:
    clean = (knowledge or "").strip()
    return clean.splitlines() if clean else []


def render_planner_hint_user_message(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]] | None = None,
    *,
    memory: str = "",
    previous_hint: str = "",
    knowledge: str = "",
    include_knowledge: bool = True,
    require_knowledge: bool = False,
) -> str:
    """Render combat state for planner-hint generation.

    `legal_actions` is intentionally unused in the output. The planner should
    produce a combat-level hint, not pick an action from current legal indices.
    """
    _ = legal_actions
    knowledge_text = knowledge
    if include_knowledge and not knowledge_text:
        knowledge_text, _entries = render_retrieved_knowledge_for_state(state)
    if require_knowledge and not (knowledge_text or "").strip():
        raise ValueError("retrieved_knowledge_required")
    lines = [
        render_run_meta(state),
        *_memory_block(memory),
        *_knowledge_block(knowledge_text),
        render_player(state),
    ]
    _append_section(lines, "relics", render_relics(state))
    _append_section(lines, "potions", render_potions(state))
    lines.append(f"deck: {render_deck(state)}")
    _append_section(lines, "piles", render_piles(state))
    _append_section(lines, "enemies", render_enemies(state))
    _append_section(lines, "hand", render_hand(state, include_previews=True))
    glossary = render_glossary(state)
    if glossary:
        _append_section(lines, "glossary", glossary)
    prior = format_planner_hint(previous_hint)
    if prior:
        lines.append("previous_planner_hint:")
        lines.extend(f"  {line}" for line in prior.splitlines())
    lines.append(
        "Task: write a short battle-level planner_hint for the combat policy. "
        "Use English values and original game IDs. Use retrieved_knowledge as evidence, not as hard rules. "
        "Do not choose a legal action, "
        "do not output action_index, and do not output an action sequence."
    )
    return "\n".join(line for line in lines if line is not None)


__all__ = [
    "ALLOWED_KEYS",
    "FORBIDDEN_KEYS",
    "LEGACY_KEYS",
    "LIST_KEYS",
    "TEXT_KEYS",
    "format_planner_hint",
    "normalize_planner_hint",
    "parse_planner_hint_json",
    "planner_hint_cache_key",
    "render_planner_hint_user_message",
]
