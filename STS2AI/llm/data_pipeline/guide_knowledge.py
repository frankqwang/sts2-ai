"""Local guide knowledge retrieval for planner/non-combat prompts.

This is intentionally small and deterministic. It is closer to a local RAG
evidence layer than a rule engine: entries are retrieved by entities/tags from
the visible state, rendered with source metadata, and left for the planner LoRA
to interpret.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from llm.paths import LLM_ROOT


DEFAULT_GUIDE_PATH = LLM_ROOT / "knowledge" / "guide_corpus.jsonl"

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "off", "no"}


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text or "") if len(match.group(0)) >= 2}


def _clean_text(value: Any, *, max_chars: int = 420) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text[:max_chars].rstrip()


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
        return str(_pick(value, "id", "card_id", "relic_id", "monster_id", "entity_id", "potion_id", default="")).upper()
    return str(value or "").upper()


def _normal_ids(values: list[Any]) -> set[str]:
    return {identifier for identifier in (_id(value) for value in values) if identifier}


@dataclass(frozen=True)
class GuideEntry:
    entry_id: str
    scope: str
    entity_ids: list[str]
    tags: list[str]
    text: str
    source: str = ""
    confidence: float = 0.6

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def guide_path(path: Path | str | None = None) -> Path:
    if path:
        return Path(path)
    raw = os.environ.get("STS2_LLM_GUIDE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_GUIDE_PATH


def guide_rag_enabled() -> bool:
    return _env_bool("STS2_LLM_GUIDE_RAG", True)


def load_guide_entries(path: Path | str | None = None) -> list[GuideEntry]:
    resolved = guide_path(path)
    if not resolved.exists():
        return []
    entries: list[GuideEntry] = []
    for line in resolved.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        text = _clean_text(raw.get("text"))
        if not text:
            continue
        entity_ids = [
            str(item).strip().upper()
            for item in (raw.get("entity_ids") or raw.get("entities") or [])
            if str(item).strip()
        ]
        tags = [str(item).strip().lower() for item in (raw.get("tags") or []) if str(item).strip()]
        try:
            confidence = float(raw.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        entry_id = str(raw.get("id") or raw.get("entry_id") or f"guide_{len(entries) + 1}").strip()
        entries.append(GuideEntry(
            entry_id=entry_id,
            scope=str(raw.get("scope") or "general").strip().lower(),
            entity_ids=entity_ids,
            tags=tags,
            text=text,
            source=str(raw.get("source") or "").strip(),
            confidence=max(0.0, min(1.0, confidence)),
        ))
    return entries


def state_query_terms(state: dict[str, Any]) -> dict[str, set[str]]:
    run = _as_dict(state.get("run"))
    battle = _as_dict(state.get("battle"))
    player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    enemies = [dict(item) for item in (_as_list(state.get("enemies")) or _as_list(battle.get("enemies"))) if isinstance(item, dict)]
    hand = (
        _as_list(battle.get("hand"))
        or _as_list(battle_player.get("hand"))
        or _as_list(state.get("hand"))
        or _as_list(player.get("hand"))
    )
    deck = _as_list(player.get("deck"))
    relics = _as_list(player.get("relics")) or _as_list(battle_player.get("relics"))
    potions = _as_list(player.get("potions")) or _as_list(battle_player.get("potions"))
    powers = _as_list(player.get("powers")) or _as_list(battle_player.get("powers"))

    entities: set[str] = set()
    encounter = str(_pick(battle, "encounter_id", "encounter", default="")).upper()
    if encounter:
        entities.add(encounter)
    boss = str(_pick(run, "boss", "act_boss", "current_boss", default="")).upper()
    if boss:
        entities.add(boss)
    entities.update(_normal_ids(enemies))
    entities.update(_normal_ids(hand))
    entities.update(_normal_ids(deck))
    entities.update(_normal_ids(relics))
    entities.update(_normal_ids(potions))
    entities.update(_normal_ids(powers))

    tags: set[str] = set()
    act = _pick(run, "act", default=None)
    floor = _pick(run, "floor", "floor_reached", default=None)
    if act not in (None, ""):
        tags.add(f"act{act}")
    try:
        floor_i = int(floor)
    except (TypeError, ValueError):
        floor_i = 0
    if 1 <= floor_i <= 6:
        tags.add("act1_early")
    elif 7 <= floor_i <= 12:
        tags.add("act1_midrun")
    elif floor_i > 12:
        tags.add("act1_late")
    if any("BASH" == item or item.endswith("_BASH") for item in entities):
        tags.add("vulnerable")
    if any("DEFEND" in item for item in entities):
        tags.add("block")
    if any("STRIKE" in item for item in entities):
        tags.add("attack")
    if potions:
        tags.add("potion")
    if boss:
        tags.add("boss")

    text_parts = [*entities, *tags]
    for enemy in enemies:
        text_parts.extend(str(_pick(enemy, key, default="")) for key in ("intent_type", "next_move_id", "intent"))
    for card in [item for item in hand if isinstance(item, dict)]:
        text_parts.extend(str(_pick(card, key, default="")) for key in ("description", "desc", "type"))

    return {
        "entities": {item for item in entities if item},
        "tags": {item for item in tags if item},
        "tokens": _tokens(" ".join(text_parts)),
    }


def text_query_terms(text: str) -> dict[str, set[str]]:
    tokens = _tokens(text)
    entities = {token.upper() for token in tokens if "_" in token or token.isupper()}
    tags = {token.lower() for token in tokens}
    return {"entities": entities, "tags": tags, "tokens": tokens}


def retrieve_guides(
    query: dict[str, set[str]],
    entries: list[GuideEntry],
    *,
    limit: int = 4,
) -> list[GuideEntry]:
    if limit <= 0 or not entries:
        return []
    query_entities = {item.upper() for item in query.get("entities", set())}
    query_tags = {item.lower() for item in query.get("tags", set())}
    query_tokens = {item.lower() for item in query.get("tokens", set())}

    scored: list[tuple[float, GuideEntry]] = []
    for entry in entries:
        entry_entities = {item.upper() for item in entry.entity_ids}
        entry_tags = {item.lower() for item in entry.tags}
        entry_tokens = _tokens(" ".join([entry.scope, *entry.entity_ids, *entry.tags, entry.text]))
        entity_hits = query_entities & entry_entities
        tag_hits = query_tags & entry_tags
        token_hits = query_tokens & entry_tokens
        if not entity_hits and not tag_hits and not token_hits:
            continue
        score = (
            5.0 * len(entity_hits)
            + 1.5 * len(tag_hits)
            + 0.35 * len(token_hits)
            + 0.5 * entry.confidence
        )
        if score <= 0:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda item: (item[0], item[1].confidence, item[1].entry_id), reverse=True)
    return [entry for _score, entry in scored[:limit]]


def render_guide_block(entries: list[GuideEntry]) -> str:
    if not entries:
        return ""
    lines = ["retrieved_knowledge:"]
    for index, entry in enumerate(entries, start=1):
        entities = ",".join(entry.entity_ids[:5]) if entry.entity_ids else "-"
        source = entry.source or entry.entry_id
        lines.append(
            f"  [{index}] scope={entry.scope} entities={entities} source={source} confidence={entry.confidence:.2f}"
        )
        lines.append(f"      {entry.text}")
    return "\n".join(lines)


def retrieve_guides_for_state(
    state: dict[str, Any],
    *,
    path: Path | str | None = None,
    limit: int | None = None,
) -> list[GuideEntry]:
    if not guide_rag_enabled():
        return []
    resolved_limit = _env_int("STS2_LLM_GUIDE_LIMIT", 4) if limit is None else int(limit)
    return retrieve_guides(state_query_terms(state), load_guide_entries(path), limit=max(0, resolved_limit))


def retrieve_guides_for_text(
    text: str,
    *,
    path: Path | str | None = None,
    limit: int | None = None,
) -> list[GuideEntry]:
    if not guide_rag_enabled():
        return []
    resolved_limit = _env_int("STS2_LLM_GUIDE_LIMIT", 4) if limit is None else int(limit)
    return retrieve_guides(text_query_terms(text), load_guide_entries(path), limit=max(0, resolved_limit))


def render_retrieved_knowledge_for_state(state: dict[str, Any], *, limit: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    entries = retrieve_guides_for_state(state, limit=limit)
    return render_guide_block(entries), [entry.to_json() for entry in entries]


def render_retrieved_knowledge_for_text(text: str, *, limit: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    entries = retrieve_guides_for_text(text, limit=limit)
    return render_guide_block(entries), [entry.to_json() for entry in entries]


__all__ = [
    "DEFAULT_GUIDE_PATH",
    "GuideEntry",
    "guide_rag_enabled",
    "guide_path",
    "load_guide_entries",
    "render_guide_block",
    "render_retrieved_knowledge_for_state",
    "render_retrieved_knowledge_for_text",
    "retrieve_guides",
    "retrieve_guides_for_state",
    "retrieve_guides_for_text",
    "state_query_terms",
    "text_query_terms",
]
