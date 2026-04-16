"""窗口挖掘运行器：批量运行训练窗口分析。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_SEQUENCE: list[tuple[str, str, bool]] = [
    ("training_window", "analyze_training_window.py", True),
    ("episode_classification", "classify_episode_failures.py", True),
    ("route_shop_patterns", "mine_route_and_shop_patterns.py", True),
    ("boss_cases", "mine_boss_cases.py", True),
    ("build_outcomes", "analyze_build_outcomes.py", True),
    ("training_vs_skada", "compare_training_vs_skada_builds.py", True),
    ("case_patterns", "mine_case_patterns.py", True),
    ("training_dashboard", "render_training_dashboard.py", False),
    ("training_trends", "render_training_trends.py", True),
]


def _run_script(
    script_path: Path,
    *,
    training_dir: Path,
    iter_start: int | None,
    iter_end: int | None,
    supports_window: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(script_path), str(training_dir)]
    if supports_window and iter_start is not None:
        cmd.extend(["--iter-start", str(iter_start)])
    if supports_window and iter_end is not None:
        cmd.extend(["--iter-end", str(iter_end)])
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )


def _collect_artifacts(training_dir: Path, iter_start: int | None, iter_end: int | None) -> list[str]:
    analysis_dir = training_dir / "analysis"
    if not analysis_dir.exists():
        return []
    suffix = ""
    if iter_start is not None and iter_end is not None:
        suffix = f"_{iter_start:05d}_to_{iter_end:05d}"
    elif iter_start is not None:
        suffix = f"_from_{iter_start:05d}"
    elif iter_end is not None:
        suffix = f"_to_{iter_end:05d}"
    files: list[str] = []
    for path in sorted(analysis_dir.iterdir()):
        if suffix and suffix in path.stem:
            files.append(str(path))
    for name in ("training_dashboard.png", "training_trends.png"):
        path = analysis_dir / name
        if path.exists() and str(path) not in files:
            files.append(str(path))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full training-window mining pipeline.")
    parser.add_argument("training_dir", help="Training output directory that contains metrics.jsonl and replays/")
    parser.add_argument("--iter-start", type=int, default=None)
    parser.add_argument("--iter-end", type=int, default=None)
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    script_dir = Path(__file__).resolve().parent
    analysis_dir = training_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for label, script_name, supports_window in SCRIPT_SEQUENCE:
        script_path = script_dir / script_name
        completed = _run_script(
            script_path,
            training_dir=training_dir,
            iter_start=args.iter_start,
            iter_end=args.iter_end,
            supports_window=supports_window,
        )
        results.append(
            {
                "label": label,
                "script": script_name,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )

    summary = {
        "training_dir": str(training_dir),
        "iter_start": args.iter_start,
        "iter_end": args.iter_end,
        "scripts": results,
        "artifacts": _collect_artifacts(training_dir, args.iter_start, args.iter_end),
    }
    suffix = "all"
    if args.iter_start is not None or args.iter_end is not None:
        start = "start" if args.iter_start is None else f"{args.iter_start:05d}"
        end = "end" if args.iter_end is None else f"{args.iter_end:05d}"
        suffix = f"{start}_to_{end}"
    output_path = analysis_dir / f"window_mining_{suffix}.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON: {output_path}")


if __name__ == "__main__":
    main()
