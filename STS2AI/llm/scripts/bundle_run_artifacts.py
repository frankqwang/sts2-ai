"""Collect one LLM experiment into a single human-readable run directory.

Example:

    python STS2AI/llm/scripts/bundle_run_artifacts.py \
        --run-name act1_explicit_legal_winnable_v2 \
        --dataset-name act1_explicit_legal_winnable_v2 \
        --sft-name act1_explicit_legal_winnable_v2_sft \
        --pattern "*winnable_v2*" \
        --overwrite

Output:
    STS2AI/Artifacts/llm/runs/<run-name>/
      dataset/
      sft/
      runtime/
      logs/
      README.md
      manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.paths import ARTIFACTS_ROOT, DATASETS_ROOT, LOGS_ROOT, RUNTIME_CWD, RUNS_ROOT, SFT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="Centralized bundle name under Artifacts/llm/runs.")
    parser.add_argument("--dataset-name", default="", help="Dataset subdir under Artifacts/llm/datasets.")
    parser.add_argument("--sft-name", default="", help="SFT subdir under Artifacts/llm/sft.")
    parser.add_argument("--pattern", action="append", default=[], help="Runtime/log filename glob. Can repeat.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing bundle directory.")
    return parser.parse_args()


def _safe_bundle_root(run_name: str) -> Path:
    root = (RUNS_ROOT / run_name).resolve()
    runs_root = RUNS_ROOT.resolve()
    if root == runs_root or runs_root not in root.parents:
        raise ValueError(f"unsafe run bundle path: {root}")
    return root


def _copy_tree(src: Path, dst: Path, manifest: dict[str, object]) -> None:
    if not src.exists():
        manifest.setdefault("missing", []).append(str(src))
        return
    shutil.copytree(src, dst)
    manifest.setdefault("copied_dirs", []).append({"from": str(src), "to": str(dst)})


def _copy_matching(src_dir: Path, dst_dir: Path, patterns: list[str], manifest: dict[str, object]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for src in sorted(src_dir.glob(pattern)):
            if not src.is_file() or src in seen:
                continue
            seen.add(src)
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied.append({"from": str(src), "to": str(dst)})
    if copied:
        manifest.setdefault("copied_files", []).extend(copied)


def _write_readme(root: Path, args: argparse.Namespace, manifest: dict[str, object]) -> None:
    lines = [
        f"# LLM Run: {args.run_name}",
        "",
        "这个目录是一次实验的人类可读入口。旧的 `datasets/`、`sft/`、`runtime/`、`logs/` 目录仍保留给脚本兼容；这里集中复制一份，方便检查。",
        "",
        "## Contents",
        "",
        "- `dataset/`: training/eval JSONL, meta, expanded prompt samples if present.",
        "- `sft/`: LoRA adapter, trainer checkpoints, run metadata.",
        "- `runtime/`: episode eval summaries, traces, ad-hoc eval runners.",
        "- `logs/`: rollout/training/eval stdout and stderr.",
        "- `manifest.json`: copy source map and missing paths.",
        "",
        "## Source",
        "",
        f"- artifacts root: `{ARTIFACTS_ROOT}`",
        f"- dataset name: `{args.dataset_name or '-'}`",
        f"- sft name: `{args.sft_name or '-'}`",
        f"- runtime/log patterns: `{', '.join(args.pattern) if args.pattern else '-'}`",
        "",
        "## Copied Summary",
        "",
        f"- copied dirs: `{len(manifest.get('copied_dirs', []))}`",
        f"- copied files: `{len(manifest.get('copied_files', []))}`",
        f"- missing sources: `{len(manifest.get('missing', []))}`",
        "",
    ]
    root.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = _safe_bundle_root(args.run_name)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"bundle already exists: {root} (use --overwrite)")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    patterns = args.pattern or [f"*{args.run_name}*"]
    manifest: dict[str, object] = {
        "run_name": args.run_name,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bundle_root": str(root),
        "patterns": patterns,
        "copied_dirs": [],
        "copied_files": [],
        "missing": [],
    }

    if args.dataset_name:
        _copy_tree(DATASETS_ROOT / args.dataset_name, root / "dataset", manifest)
    if args.sft_name:
        _copy_tree(SFT_ROOT / args.sft_name, root / "sft", manifest)
    _copy_matching(RUNTIME_CWD, root / "runtime", patterns, manifest)
    _copy_matching(LOGS_ROOT, root / "logs", patterns, manifest)

    root.joinpath("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(root, args, manifest)

    print(f"BUNDLE={root}")
    print(f"README={root / 'README.md'}")
    print(f"MANIFEST={root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
