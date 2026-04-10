from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    tools_python_dir = Path(__file__).resolve().parents[2]
    if str(tools_python_dir) not in sys.path:
        sys.path.insert(0, str(tools_python_dir))

import _path_init  # noqa: F401

from data.derived.build_llm_views import (
    build_preference_pair_from_ranking_records,
    build_sft_dialogue_from_ranking_records,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive LLM SFT/preference views from ranking/card_ranking JSONL")
    parser.add_argument("--ranking-jsonl", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--source-schema-version", type=str, default="card_ranking.v1")
    args = parser.parse_args()

    ranking_path = Path(args.ranking_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(ranking_path)

    sft_path, sft_manifest = build_sft_dialogue_from_ranking_records(
        ranking_records=records,
        output_dir=out_dir / "sft",
        source_schema_version=str(args.source_schema_version),
        source_path=ranking_path,
    )
    pref_path, pref_manifest = build_preference_pair_from_ranking_records(
        ranking_records=records,
        output_dir=out_dir / "preference",
        source_schema_version=str(args.source_schema_version),
        source_path=ranking_path,
    )
    summary = {
        "ranking_jsonl": str(ranking_path),
        "num_roots": len(records),
        "sft_path": str(sft_path),
        "sft_manifest": str(sft_manifest),
        "preference_path": str(pref_path),
        "preference_manifest": str(pref_manifest),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
