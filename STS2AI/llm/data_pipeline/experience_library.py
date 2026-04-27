"""Small local experience library for post-game lessons and guide snippets.

Entries are intentionally compact. They can come from manual notes, local
review, self-rerank outputs, or later API teacher reviews. Retrieval is simple
lexical matching for now so the rest of the pipeline does not depend on an
embedding service.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path
from typing import Any

from llm.paths import ARTIFACTS_ROOT


DEFAULT_EXPERIENCE_PATH = ARTIFACTS_ROOT / "experience" / "lessons.jsonl"

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text or "") if len(match.group(0)) >= 2}


def _message_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content") or "") for message in messages if isinstance(message, dict))


@dataclass(frozen=True)
class ExperienceEntry:
    tags: list[str]
    applies_when: str
    advice: str
    avoid: str = ""
    source: str = ""
    confidence: float = 0.5

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_experience(path: Path = DEFAULT_EXPERIENCE_PATH) -> list[ExperienceEntry]:
    if not path.exists():
        return []
    entries: list[ExperienceEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        advice = str(raw.get("advice") or "").strip()
        applies_when = str(raw.get("applies_when") or "").strip()
        if not advice or not applies_when:
            continue
        tags = [str(tag) for tag in (raw.get("tags") or []) if str(tag).strip()]
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        entries.append(ExperienceEntry(
            tags=tags,
            applies_when=applies_when,
            advice=advice,
            avoid=str(raw.get("avoid") or ""),
            source=str(raw.get("source") or ""),
            confidence=max(0.0, min(1.0, confidence)),
        ))
    return entries


def append_experience(entries: list[ExperienceEntry], path: Path = DEFAULT_EXPERIENCE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")


def retrieve_experience(
    text: str,
    entries: list[ExperienceEntry],
    *,
    limit: int = 4,
) -> list[ExperienceEntry]:
    query = _tokens(text)
    if not query:
        return []

    scored: list[tuple[float, ExperienceEntry]] = []
    for entry in entries:
        entry_text = " ".join([*entry.tags, entry.applies_when, entry.advice, entry.avoid])
        entry_tokens = _tokens(entry_text)
        if not entry_tokens:
            continue
        overlap = len(query & entry_tokens)
        if overlap <= 0:
            continue
        score = overlap + 0.5 * float(entry.confidence)
        scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _score, entry in scored[:limit]]


def render_experience_block(entries: list[ExperienceEntry], *, compact: bool = True) -> str:
    if not entries:
        return ""
    lines = ["experience:"]
    for index, entry in enumerate(entries, start=1):
        avoid = f"; avoid={entry.avoid}" if entry.avoid else ""
        if compact:
            lines.append(f"  [{index}] {entry.advice}{avoid}")
        else:
            tags = ",".join(entry.tags[:4]) if entry.tags else "-"
            lines.append(
                f"  [{index}] tags={tags}; when={entry.applies_when}; advice={entry.advice}{avoid}"
            )
    return "\n".join(lines)


def retrieve_experience_for_messages(
    messages: list[dict[str, Any]],
    *,
    path: Path = DEFAULT_EXPERIENCE_PATH,
    limit: int = 4,
) -> list[ExperienceEntry]:
    return retrieve_experience(_message_text(messages), load_experience(path), limit=limit)


__all__ = [
    "DEFAULT_EXPERIENCE_PATH",
    "ExperienceEntry",
    "append_experience",
    "load_experience",
    "render_experience_block",
    "retrieve_experience",
    "retrieve_experience_for_messages",
]
