"""Validate the local planner guide corpus.

The corpus is part of the prompt contract. Invalid rows should be fixed or
removed instead of being silently skipped by retrieval.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.guide_knowledge import (  # noqa: E402
    DEFAULT_GUIDE_PATH,
    GuideEntry,
    retrieve_guides,
    text_query_terms,
)
from llm.paths import EVALS_ROOT, ensure_dirs  # noqa: E402


DEFAULT_REQUIRED_ENTITIES = (
    "BASH",
    "CULTISTS_NORMAL",
    "BURNING_BLOOD",
    "HAND_DRILL",
    "SOUL_FYSH",
    "THE_INSATIABLE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(DEFAULT_GUIDE_PATH))
    parser.add_argument("--out-dir", default="", help="default: Artifacts/llm/evals/guide_corpus_check_<timestamp>")
    parser.add_argument("--require-entity", action="append", default=list(DEFAULT_REQUIRED_ENTITIES))
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_ok(source: str) -> bool:
    if source in {"seed_guide", "teacher_review", "manual"}:
        return True
    return source.startswith("https://")


def _validate_row(payload: Any, *, seen_ids: set[str]) -> tuple[GuideEntry | None, list[str]]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return None, ["json_not_object"]
    entry_id = str(payload.get("id") or payload.get("entry_id") or "").strip()
    if not entry_id:
        errors.append("missing_id")
    elif entry_id in seen_ids:
        errors.append("duplicate_id")
    else:
        seen_ids.add(entry_id)

    scope = str(payload.get("scope") or "").strip().lower()
    if not scope:
        errors.append("missing_scope")

    raw_entities = payload.get("entity_ids")
    if not isinstance(raw_entities, list) or not raw_entities or not all(str(item).strip() for item in raw_entities):
        errors.append("bad_entity_ids")
        entity_ids: list[str] = []
    else:
        entity_ids = [str(item).strip().upper() for item in raw_entities]

    raw_tags = payload.get("tags")
    if not isinstance(raw_tags, list) or not all(str(item).strip() for item in raw_tags):
        errors.append("bad_tags")
        tags: list[str] = []
    else:
        tags = [str(item).strip().lower() for item in raw_tags]

    text = str(payload.get("text") or "").strip()
    if not text:
        errors.append("missing_text")
    elif len(text) > 520:
        errors.append("text_too_long")

    source = str(payload.get("source") or "").strip()
    if not source:
        errors.append("missing_source")
    elif not _source_ok(source):
        errors.append("bad_source")

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = -1.0
    if not (0.0 <= confidence <= 1.0):
        errors.append("bad_confidence")

    if errors:
        return None, errors
    return GuideEntry(
        entry_id=entry_id,
        scope=scope,
        entity_ids=entity_ids,
        tags=tags,
        text=text,
        source=source,
        confidence=confidence,
    ), []


def _read_entries(path: Path, *, examples: int) -> tuple[list[GuideEntry], Counter[str], list[dict[str, Any]], str]:
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    rows = raw_bytes.decode("utf-8-sig").splitlines()
    entries: list[GuideEntry] = []
    seen_ids: set[str] = set()
    counters: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for line_no, line in enumerate(rows, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            counters["json_parse_failed"] += 1
            if len(failures) < examples:
                failures.append({"line_no": line_no, "status": "json_parse_failed", "error": str(exc)})
            continue
        entry, errors = _validate_row(payload, seen_ids=seen_ids)
        if errors:
            for error in errors:
                counters[error] += 1
            if len(failures) < examples:
                failures.append({
                    "line_no": line_no,
                    "status": ",".join(errors),
                    "id": payload.get("id") if isinstance(payload, dict) else None,
                })
            continue
        assert entry is not None
        entries.append(entry)
        counters["ok"] += 1
    return entries, counters, failures, digest


def _retrieval_checks(entries: list[GuideEntry], required_entities: list[str]) -> tuple[Counter[str], list[dict[str, Any]]]:
    counters: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for entity in required_entities:
        hits = retrieve_guides(text_query_terms(entity), entries, limit=3)
        if hits:
            counters["retrieval_ok"] += 1
            continue
        counters["retrieval_missing"] += 1
        failures.append({"entity": entity, "status": "retrieval_missing"})
    return counters, failures


def main() -> int:
    args = parse_args()
    ensure_dirs()
    corpus = Path(args.corpus).resolve()
    if not corpus.exists():
        raise SystemExit(f"missing corpus: {corpus}")

    entries, row_counts, row_failures, digest = _read_entries(corpus, examples=max(0, args.examples))
    retrieval_counts, retrieval_failures = _retrieval_checks(entries, list(dict.fromkeys(args.require_entity)))
    invalid = sum(count for key, count in row_counts.items() if key != "ok") + retrieval_counts.get("retrieval_missing", 0)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else EVALS_ROOT / f"guide_corpus_check_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    summary = {
        "kind": "guide_corpus_check",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": str(corpus),
        "sha256": digest,
        "entries": len(entries),
        "row_status_counts": dict(row_counts),
        "retrieval_status_counts": dict(retrieval_counts),
        "invalid": invalid,
        "failures": row_failures[: args.examples] + retrieval_failures[: args.examples],
    }
    _write_json(out_dir / "summary.json", summary)
    print(out_dir)
    if invalid:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
