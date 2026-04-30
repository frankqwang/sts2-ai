"""Build planner-hint SFT data from combat review JSON files.

Input reviews are produced by `kimi_review_turn_order.py` or
`run_kimi_combat_review_batch.py`. The review must contain a top-level
`planner_hint` object. The output is a normal chat SFT dataset for a planner
adapter that returns battle-level JSON hints, not actions.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.guide_knowledge import render_retrieved_knowledge_for_text  # noqa: E402
from llm.data_pipeline.planner_hint import parse_planner_hint_json  # noqa: E402
from llm.paths import DATASETS_ROOT, ensure_dirs  # noqa: E402
from llm.prompts import load_system_prompt  # noqa: E402


_RETURN_RE = re.compile(r"^Return (?:one JSON line|strict JSON only): .*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="append", default=[], help="turn_order_review.json")
    parser.add_argument("--episode-input", action="append", default=[], help="matching episode_input.json")
    parser.add_argument(
        "--review-root",
        action="append",
        default=[],
        help="Directory containing turn_order_review.json and sibling episode_input.json files.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DATASETS_ROOT / f"planner_hint_sft_{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
    )
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-overall-score",
        type=float,
        default=0.0,
        help=(
            "Skip reviews whose top-level ``overall_score`` is below this. "
            "Use this to keep planner SFT focused on turns the teacher "
            "considered competently judged. Default 0 = no filtering."
        ),
    )
    parser.add_argument(
        "--include-phase-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the review carries a top-level ``phase_plan_zh`` (turn-by-turn "
            "tactical breakdown), splice it into the planner_hint assistant "
            "label as ``phase_plan``. Lets the planner LoRA learn temporal "
            "phase reasoning, not just episode-level hints."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _review_pairs_from_roots(roots: list[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if not root.exists():
            continue
        for review_path in sorted(root.rglob("turn_order_review.json")):
            episode_path = review_path.with_name("episode_input.json")
            if not episode_path.exists():
                continue
            key = (review_path.resolve(), episode_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def _strip_block(lines: list[str], heading: str) -> list[str]:
    out: list[str] = []
    skip = False
    prefix = f"{heading}:"
    for line in lines:
        if line.startswith(prefix):
            skip = True
            continue
        if skip and line and not line.startswith((" ", "\t")):
            skip = False
        if not skip:
            out.append(line)
    return out


def _planner_user_from_state_text(user_message: str) -> str:
    lines = str(user_message or "").splitlines()
    lines = _strip_block(lines, "strategy_context")
    lines = _strip_block(lines, "legal_actions")
    text = "\n".join(lines)
    text = _RETURN_RE.sub("", text).strip()
    knowledge_block, _entries = render_retrieved_knowledge_for_text(text)
    if knowledge_block:
        split_lines = text.splitlines()
        if split_lines:
            text = "\n".join([split_lines[0], knowledge_block, *split_lines[1:]])
        else:
            text = knowledge_block
    return (
        f"{text}\n"
        "Task: write a short battle-level planner_hint for the combat policy. "
        "Use retrieved_knowledge as evidence, not as hard rules. "
        "Do not choose a legal action, do not output action_index, and do not output an action sequence."
    ).strip()


def _first_decision_state(episode: dict[str, Any]) -> str:
    for turn in episode.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        for decision in turn.get("decisions") or []:
            if not isinstance(decision, dict):
                continue
            state = str(decision.get("pre_decision_state") or decision.get("state") or decision.get("compact_state") or "")
            if state:
                return state
    return ""


def _meta_str(value: Any) -> str:
    """规范化 dataset meta 字段为 str；None / list / dict / bool / int / float 一律 cast。

    pyarrow.from_pydict 按列推断 schema：同列不同 row 类型不一致会报
    'Could not convert ... with type str: tried to convert to int64'。
    将 LLM/teacher 派生字段统一 str 化是最稳的兜底。
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)
    return str(value)


def _coerce_overall_score(value: Any) -> float | None:
    """Review ``overall_score`` is sometimes int, sometimes str ("8" / "8/10").
    Return a float in [0, 10] or None when un-parseable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    # "8/10" → "8"
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def _row_from_pair(
    review_path: Path,
    episode_path: Path,
    *,
    system_prompt: str,
    min_overall_score: float = 0.0,
    include_phase_plan: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    review = _read_json(review_path)
    episode = _read_json(episode_path)
    raw_hint = review.get("planner_hint")
    if not isinstance(raw_hint, dict):
        return None, "missing_planner_hint"

    # Optional quality gate: skip reviews where the teacher's own
    # confidence in the episode read is below threshold. Keeps the planner
    # SFT pool focused on samples the teacher could competently distill.
    overall_score = _coerce_overall_score(review.get("overall_score"))
    if min_overall_score > 0 and overall_score is not None and overall_score < min_overall_score:
        return None, "below_min_overall_score"

    hint, status = parse_planner_hint_json(json.dumps(raw_hint, ensure_ascii=False))
    if status != "ok" or hint is None:
        return None, status
    source_state = _first_decision_state(episode)
    if not source_state:
        return None, "missing_source_state"

    # Splice teacher's per-turn ``phase_plan_zh`` into the planner_hint
    # label so the LoRA learns turn-by-turn pacing in addition to the
    # battle-level objective. Schema-wise we just add a sibling
    # ``phase_plan`` field; planner_hint downstream parser ignores
    # unknown keys (parse_planner_hint_json is permissive).
    if include_phase_plan:
        phase_plan = review.get("phase_plan_zh") or review.get("phase_plan")
        if isinstance(phase_plan, str) and phase_plan.strip():
            hint = {**hint, "phase_plan": phase_plan.strip()}

    assistant = json.dumps(hint, ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _planner_user_from_state_text(source_state)},
            {"role": "assistant", "content": assistant},
        ],
        # NOTE: meta 字段会进 jsonl，pyarrow 加载 dataset 时按列推断 schema；
        # Kimi/Claude 偶尔把同一字段返回 int/str/float/list 多种类型，会让 schema 推断失败
        # （iter02 因 overall_score int vs str 撞过一次）。统一强制 cast 成 str，
        # meta 不参与训练，类型不重要，只要列内一致就够。
        "meta": {
            "source": "turn_order_review_planner_hint",
            "source_review": str(review_path),
            "source_episode_input": str(episode_path),
            "episode_id": _meta_str(episode.get("episode_id") or review.get("episode_id")),
            "encounter_id": _meta_str(episode.get("encounter_id") or review.get("encounter_id")),
            "outcome": _meta_str(episode.get("outcome") or review.get("outcome")),
            "overall_score": _meta_str(review.get("overall_score")),
        },
    }, "ok"


def main() -> int:
    args = parse_args()
    ensure_dirs()
    if len(args.review) != len(args.episode_input):
        raise SystemExit("--review and --episode-input counts must match")
    pairs = [(Path(review).resolve(), Path(episode).resolve()) for review, episode in zip(args.review, args.episode_input)]
    pairs.extend(_review_pairs_from_roots(args.review_root))
    if not pairs:
        raise SystemExit("no review/episode-input pairs found")

    system_prompt = load_system_prompt("planner_hint")
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    for review_path, episode_path in pairs:
        row, status = _row_from_pair(
            review_path, episode_path,
            system_prompt=system_prompt,
            min_overall_score=float(args.min_overall_score),
            include_phase_plan=bool(args.include_phase_plan),
        )
        counters[status] += 1
        if row is None:
            invalid.append({
                "review": str(review_path),
                "episode_input": str(episode_path),
                "status": status,
            })
            continue
        rows.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_n = int(round(len(rows) * max(0.0, min(0.9, args.eval_ratio))))
    eval_rows = rows[:eval_n]
    train_rows = rows[eval_n:]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    _write_jsonl(out_dir / "invalid.jsonl", invalid)
    _write_json(out_dir / "summary.json", {
        "kind": "planner_hint_sft_dataset",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "pairs": len(pairs),
        "rows": len(rows),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "invalid": len(invalid),
        "status_counts": dict(counters),
        "reviews": [str(path) for path, _episode in pairs],
        "episode_inputs": [str(path) for _review, path in pairs],
    })
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
