from __future__ import annotations

import sys
import argparse
import json
import random
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    tools_python_dir = Path(__file__).resolve().parents[2]
    if str(tools_python_dir) not in sys.path:
        sys.path.insert(0, str(tools_python_dir))

import _path_init  # noqa: F401

from data.manifest import build_manifest, write_manifest


LLM_BUNDLE_SCHEMA_VERSION = "llm_bundle.v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return path


def _split_records(
    records: list[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if val_fraction > 0 else 0
    val_count = min(val_count, max(0, len(shuffled) - 1))
    if val_count <= 0:
        return shuffled, []
    return shuffled[val_count:], shuffled[:val_count]


def build_llm_bundle(
    *,
    output_dir: str | Path,
    sft_sources: list[str | Path],
    preference_sources: list[str | Path],
    val_fraction: float = 0.05,
    seed: int = 7,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_records: list[dict[str, Any]] = []
    sft_source_paths: list[str] = []
    for source in sft_sources:
        source_path = Path(source)
        records = _read_jsonl(source_path)
        if records:
            sft_records.extend(records)
            sft_source_paths.append(str(source_path))

    pref_records: list[dict[str, Any]] = []
    pref_source_paths: list[str] = []
    for source in preference_sources:
        source_path = Path(source)
        records = _read_jsonl(source_path)
        if records:
            pref_records.extend(records)
            pref_source_paths.append(str(source_path))

    sft_train, sft_val = _split_records(sft_records, val_fraction=val_fraction, seed=seed)
    pref_train, pref_val = _split_records(pref_records, val_fraction=val_fraction, seed=seed)

    sft_train_path = _write_jsonl(out_dir / "sft" / "train.sft_dialogue.jsonl", sft_train)
    sft_val_path = _write_jsonl(out_dir / "sft" / "val.sft_dialogue.jsonl", sft_val)
    pref_train_path = _write_jsonl(out_dir / "preference" / "train.preference_pair.jsonl", pref_train)
    pref_val_path = _write_jsonl(out_dir / "preference" / "val.preference_pair.jsonl", pref_val)

    summary = {
        "sft_total": len(sft_records),
        "sft_train": len(sft_train),
        "sft_val": len(sft_val),
        "preference_total": len(pref_records),
        "preference_train": len(pref_train),
        "preference_val": len(pref_val),
    }
    manifest = build_manifest(
        dataset_kind="llm_bundle",
        schema_version=LLM_BUNDLE_SCHEMA_VERSION,
        output_dir=str(out_dir),
        status="complete",
        source_raw_paths=[],
        derived_from=["sft_dialogue.v1", "preference_pair.v1"],
        generator_config={
            "val_fraction": float(val_fraction),
            "seed": int(seed),
            "sft_sources": sft_source_paths,
            "preference_sources": pref_source_paths,
        },
        summary=summary,
        extra={
            "sft_train_path": str(sft_train_path),
            "sft_val_path": str(sft_val_path),
            "preference_train_path": str(pref_train_path),
            "preference_val_path": str(pref_val_path),
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable LLM bundle from derived offline datasets")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--sft-source", action="append", default=[], help="Path to sft_dialogue.jsonl (repeatable)")
    parser.add_argument("--preference-source", action="append", default=[], help="Path to preference_pair.jsonl (repeatable)")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    manifest = build_llm_bundle(
        output_dir=args.output_dir,
        sft_sources=args.sft_source,
        preference_sources=args.preference_source,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
