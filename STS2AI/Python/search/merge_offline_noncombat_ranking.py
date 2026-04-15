#!/usr/bin/env python3
"""Merge two offline_noncombat_ranking data dirs into a single dataset.

The output layout mirrors what a single generator run produces
(`card_ranking.jsonl` at top level + `tensors/*.npz`), so
`MatchupRankingDataset` loads it with no special handling. To avoid
`sample_NNNNN.npz` filename collisions between the two source dirs, the
second dir's tensors are placed in a side-car subdirectory (default
`tensors_rankb2/`) and the corresponding samples' `state_tensors_path`
fields are rewritten on the fly.

Typical use:

    python STS2AI/Python/search/merge_offline_noncombat_ranking.py \\
      --dir-a STS2AI/Artifacts/offline_noncombat_ranking/from_hybrid_02293_20260415-031531 \\
      --dir-b STS2AI/Artifacts/offline_noncombat_ranking/from_hybrid_02293_rankb2_20260415-110759 \\
      --output STS2AI/Artifacts/offline_noncombat_ranking/from_hybrid_02293_merged_20260415 \\
      --tag-b rankb2

Output `manifest.json` records the merge provenance (source dirs + per-source
sample counts) so the dataset is traceable back to its generators.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def merge(dir_a: Path, dir_b: Path, out: Path, tag_b: str) -> dict[str, Any]:
    """Merge dir_a and dir_b into out. Returns the written manifest dict."""
    out.mkdir(parents=True, exist_ok=True)

    # --- Copy tensors, placing dir_b's in a side-car subdir to avoid collision
    tensors_a_dst = out / "tensors"
    tensors_b_dst = out / f"tensors_{tag_b}"
    a_count = 0
    b_count = 0
    if (dir_a / "tensors").exists():
        shutil.copytree(dir_a / "tensors", tensors_a_dst, dirs_exist_ok=True)
        a_count = sum(1 for _ in tensors_a_dst.glob("*.npz"))
    if (dir_b / "tensors").exists():
        shutil.copytree(dir_b / "tensors", tensors_b_dst, dirs_exist_ok=True)
        b_count = sum(1 for _ in tensors_b_dst.glob("*.npz"))

    # --- Merge card_ranking.jsonl. dir_b samples need tensor_path rewritten.
    lines_a = _load_jsonl(dir_a / "card_ranking.jsonl")
    raw_lines_b = _load_jsonl(dir_b / "card_ranking.jsonl")

    rewritten_b: list[str] = []
    for raw in raw_lines_b:
        sample = json.loads(raw)
        tp = sample.get("state_tensors_path", "")
        if isinstance(tp, str) and tp.startswith("tensors/"):
            sample["state_tensors_path"] = f"tensors_{tag_b}/" + tp[len("tensors/") :]
        rewritten_b.append(json.dumps(sample, ensure_ascii=False) + "\n")

    with (out / "card_ranking.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(lines_a)
        f.writelines(rewritten_b)

    # --- Compose a traceable merge manifest
    mf_a = _load_manifest(dir_a / "manifest.json")
    mf_b = _load_manifest(dir_b / "manifest.json")
    sum_a = int((mf_a.get("summary") or {}).get("total_samples", len(lines_a)) or 0)
    sum_b = int((mf_b.get("summary") or {}).get("total_samples", len(rewritten_b)) or 0)
    merged_manifest: dict[str, Any] = {
        "dataset_kind": "card_ranking",
        "dataset_schema_version": mf_a.get("dataset_schema_version")
        or mf_b.get("dataset_schema_version"),
        "status": "complete",
        "merged": True,
        "source_dirs": [str(dir_a), str(dir_b)],
        "tag_b": tag_b,
        "summary": {
            "total_samples": sum_a + sum_b,
            "source_a_samples": sum_a,
            "source_b_samples": sum_b,
            "source_a_manifest_summary": mf_a.get("summary"),
            "source_b_manifest_summary": mf_b.get("summary"),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(merged_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"merged {len(lines_a)} + {len(rewritten_b)} = "
        f"{len(lines_a) + len(rewritten_b)} samples into {out}"
    )
    print(f"  tensors_a: {tensors_a_dst} ({a_count} npz)")
    print(f"  tensors_b: {tensors_b_dst} ({b_count} npz)")
    return merged_manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Merge two offline_noncombat_ranking dirs.")
    p.add_argument(
        "--dir-a", required=True,
        help="First teacher dir. Its tensors stay under tensors/."
    )
    p.add_argument(
        "--dir-b", required=True,
        help="Second teacher dir. Its tensors go into tensors_<tag>/ to avoid collision."
    )
    p.add_argument("--output", required=True, help="Output merged dir.")
    p.add_argument("--tag-b", default="rankb2", help="Subdir suffix for dir-b tensors. Default: rankb2.")
    args = p.parse_args()
    merge(Path(args.dir_a), Path(args.dir_b), Path(args.output), tag_b=args.tag_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
