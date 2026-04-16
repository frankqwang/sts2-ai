"""Merge multiple combat-teacher JSONL datasets into one, dedup by sample_id.

Keeps the first occurrence (earlier `--source` arguments win on duplicates).
Re-assigns train/holdout split using `stable_split(sample_id)`.
Writes a manifest + summary beside the output JSONL.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PYTHON_ROOT = _THIS_FILE.parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))


import argparse
import json
from collections import Counter
from dataclasses import replace

from search.combat_teacher_dataset import (
    CombatTeacherSample,
    dedupe_samples_by_id,
    load_combat_teacher_samples,
    stable_split,
    write_combat_teacher_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge combat-teacher datasets with dedup.")
    parser.add_argument("--source", action="append", required=True, help="Input JSONL files (repeatable).")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--summary", default="", help="Optional override summary JSON path.")
    args = parser.parse_args()

    all_samples: list[CombatTeacherSample] = []
    sources = [str(Path(src)) for src in args.source]
    for src in sources:
        loaded = load_combat_teacher_samples(src)
        print(f"loaded {len(loaded):4d} samples from {src}")
        all_samples.extend(loaded)

    before = len(all_samples)
    deduped = dedupe_samples_by_id(all_samples)
    print(f"deduped {before} -> {len(deduped)}")

    # Re-assign stable split
    refreshed: list[CombatTeacherSample] = []
    for sample in deduped:
        refreshed.append(replace(sample, split=stable_split(sample.sample_id or "")))

    bucket_counts: Counter[str] = Counter()
    floor_counts: Counter[int] = Counter()
    state_type_counts: Counter[str] = Counter()
    motif_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for sample in refreshed:
        bucket_counts[str(sample.source_bucket or "")] += 1
        state = sample.state if isinstance(sample.state, dict) else {}
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        floor_value = run.get("floor")
        if floor_value is None:
            floor_value = state.get("floor")
        try:
            floor_int = int(floor_value)
        except (TypeError, ValueError):
            floor_int = 0
        floor_counts[floor_int] += 1
        state_type = str(state.get("state_type") or run.get("room_type") or "").lower()
        state_type_counts[state_type] += 1
        for motif in sample.motif_labels or []:
            motif_counts[str(motif)] += 1
        split_counts[str(sample.split or "")] += 1

    summary = {
        "output": str(Path(args.output)),
        "sources": sources,
        "sample_count": len(refreshed),
        "split": dict(sorted(split_counts.items())),
        "bucket": dict(sorted(bucket_counts.items())),
        "floor": {str(floor): count for floor, count in sorted(floor_counts.items())},
        "state_type": dict(sorted(state_type_counts.items())),
        "top_motifs": motif_counts.most_common(30),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "merged_combat_teacher_dataset.v1",
        "sources": sources,
        "sample_count": len(refreshed),
        "bucket": dict(bucket_counts),
        "floor_counts": {str(k): v for k, v in floor_counts.items()},
        "motif_counts": dict(motif_counts),
        "split_counts": dict(split_counts),
    }
    write_combat_teacher_samples(str(output_path), refreshed, metadata=metadata)

    summary_path = Path(args.summary) if args.summary else output_path.with_name("mixed_dataset_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# Merged Combat Teacher Dataset",
        "",
        f"- samples: {summary['sample_count']}",
        f"- train: {split_counts.get('train', 0)}",
        f"- holdout: {split_counts.get('holdout', 0)}",
        "",
        "## Floors",
    ]
    for floor, count in sorted(floor_counts.items()):
        md_lines.append(f"- floor {floor}: {count}")
    md_lines.extend(["", "## Motifs"])
    for motif, count in motif_counts.most_common(30):
        md_lines.append(f"- {motif}: {count}")
    summary_md_path = summary_path.with_suffix(".md")
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
