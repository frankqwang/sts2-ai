from __future__ import annotations
# --- wizardly cleanup 2026-04-08: tools/python subdir sys.path bootstrap ---
# Moved out of tools/python/ root; bootstrap below re-adds the parent dir so
# flat `from combat_nn import X` style imports still resolve.
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # noqa: E402
# --- end bootstrap ---

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from sts2ai_paths import ARTIFACTS_ROOT, REPO_ROOT
from sim_semantic_audit_common import (
    DEFAULT_GODOT_EXE,
    DEFAULT_HEADLESS_DLL,
    DEFAULT_REPO_ROOT,
    build_seed_list,
)
from training_semantic_audit import (
    DEFAULT_COMBAT_CHECKPOINT,
    DEFAULT_HYBRID_CHECKPOINT,
    _launch_backend_fleet,
    _resolve_checkpoint_path,
    _stop_backend_fleet,
    _transport_for_backend,
)


DEFAULT_BASELINE_PORT = 15980
DEFAULT_CANDIDATE_PORT = 15990
DEFAULT_REPO_ROOT = REPO_ROOT
DEFAULT_OUTPUT_ROOT = ARTIFACTS_ROOT / "verification"


def _run_evaluate(
    *,
    python_exe: Path,
    repo_root: Path,
    output_dir: Path,
    backend: str,
    port: int,
    checkpoint: Path,
    combat_checkpoint: Path | None,
    seed_file: Path,
    seed_suite: str,
    num_games: int,
    max_steps: int,
    trace_seeds: list[str],
) -> tuple[int, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "evaluate.stdout.log"
    stderr_path = log_dir / "evaluate.stderr.log"
    result_path = output_dir / "eval_result.json"
    command = [
        str(python_exe),
        "STS2AI/Python/evaluate_ai.py",
        "--checkpoint",
        str(checkpoint),
        "--transport",
        _transport_for_backend(backend),
        "--port",
        str(port),
        "--num-games",
        str(num_games),
        "--seed-suite",
        seed_suite,
        "--seeds-file",
        str(seed_file),
        "--max-steps",
        str(max_steps),
        "--output",
        str(result_path),
    ]
    if combat_checkpoint is not None:
        command.extend(["--combat-checkpoint", str(combat_checkpoint)])
    if trace_seeds:
        trace_dir = output_dir / "traces"
        command.extend(
            [
                "--save-trace-dir",
                str(trace_dir),
                "--trace-seeds",
                ",".join(trace_seeds),
            ]
        )
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
    return proc.returncode, result_path, stdout_path, stderr_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_values(baseline: float, candidate: float) -> dict[str, float]:
    abs_diff = float(candidate - baseline)
    rel_diff = 0.0
    if abs(baseline) > 1e-9:
        rel_diff = abs_diff / baseline
    return {
        "baseline": round(float(baseline), 6),
        "candidate": round(float(candidate), 6),
        "abs_diff": round(abs_diff, 6),
        "rel_diff": round(rel_diff, 6),
    }


def _index_results(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = payload.get("results") or {}
    nn_results = results.get("nn") or []
    indexed: dict[str, dict[str, Any]] = {}
    for item in nn_results:
        seed = str(item.get("seed") or "").strip()
        if seed:
            indexed[seed] = item
    return indexed


def _compare_seed_records(
    baseline_results: dict[str, dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
    seeds: list[str],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for seed in seeds:
        baseline = baseline_results.get(seed)
        candidate = candidate_results.get(seed)
        if baseline is None or candidate is None:
            mismatches.append(
                {
                    "seed": seed,
                    "reason": "missing_result",
                    "baseline_present": baseline is not None,
                    "candidate_present": candidate is not None,
                }
            )
            continue
        diffs: dict[str, Any] = {}
        for key in ("outcome", "max_floor", "boss_reached", "act1_cleared", "total_steps", "timeout_state_type"):
            if baseline.get(key) != candidate.get(key):
                diffs[key] = {
                    "baseline": baseline.get(key),
                    "candidate": candidate.get(key),
                }
        if diffs:
            mismatches.append({"seed": seed, "reason": "seed_result_diff", "diffs": diffs})
    return mismatches


def _build_comparison(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    seeds: list[str],
) -> dict[str, Any]:
    baseline_summary = (baseline_payload.get("summaries") or {}).get("nn") or {}
    candidate_summary = (candidate_payload.get("summaries") or {}).get("nn") or {}
    baseline_results = _index_results(baseline_payload)
    candidate_results = _index_results(candidate_payload)
    per_seed_mismatches = _compare_seed_records(baseline_results, candidate_results, seeds)
    metrics = {
        "avg_floor": _compare_values(baseline_summary.get("avg_floor", 0.0), candidate_summary.get("avg_floor", 0.0)),
        "boss_reach_rate": _compare_values(
            baseline_summary.get("boss_reach_rate", 0.0), candidate_summary.get("boss_reach_rate", 0.0)
        ),
        "act1_clear_rate": _compare_values(
            baseline_summary.get("act1_clear_rate", 0.0), candidate_summary.get("act1_clear_rate", 0.0)
        ),
        "avg_steps": _compare_values(baseline_summary.get("avg_steps", 0.0), candidate_summary.get("avg_steps", 0.0)),
        "timeout_count": _compare_values(
            baseline_summary.get("timeout_count", 0.0), candidate_summary.get("timeout_count", 0.0)
        ),
    }
    thresholds = {
        "avg_floor_abs": 1.0,
        "boss_reach_abs": 0.10,
        "act1_clear_abs": 0.10,
        "avg_steps_rel": 0.20,
        "timeout_count_abs": 2.0,
        "seed_mismatch_count": 0,
    }
    failures: list[str] = []
    if abs(metrics["avg_floor"]["abs_diff"]) > thresholds["avg_floor_abs"]:
        failures.append("avg_floor")
    if abs(metrics["boss_reach_rate"]["abs_diff"]) > thresholds["boss_reach_abs"]:
        failures.append("boss_reach_rate")
    if abs(metrics["act1_clear_rate"]["abs_diff"]) > thresholds["act1_clear_abs"]:
        failures.append("act1_clear_rate")
    if abs(metrics["avg_steps"]["rel_diff"]) > thresholds["avg_steps_rel"]:
        failures.append("avg_steps")
    if abs(metrics["timeout_count"]["abs_diff"]) > thresholds["timeout_count_abs"]:
        failures.append("timeout_count")
    if len(per_seed_mismatches) > thresholds["seed_mismatch_count"]:
        failures.append("per_seed_mismatch")
    return {
        "metrics": metrics,
        "thresholds": thresholds,
        "failed_metrics": failures,
        "per_seed_mismatch_count": len(per_seed_mismatches),
        "per_seed_mismatch_examples": per_seed_mismatches[:20],
    }


def run_policy_rollout_audit(
    *,
    repo_root: Path,
    python_exe: Path,
    godot_exe: Path,
    headless_dll: Path,
    checkpoint: Path,
    combat_checkpoint: Path | None,
    baseline_backend: str,
    candidate_backend: str,
    baseline_port: int,
    candidate_port: int,
    seeds: list[str],
    max_steps: int,
    output_root: Path,
    trace_count: int,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = output_root / f"policy_rollout_audit_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    seed_file = run_root / "seed_suite.json"
    seed_payload = {"benchmark": [{"seed": seed} for seed in seeds]}
    seed_file.write_text(json.dumps(seed_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    trace_seeds = seeds[: max(0, trace_count)]

    def run_one(label: str, backend: str, port: int) -> dict[str, Any]:
        label_root = run_root / label
        procs = _launch_backend_fleet(
            backend=backend,
            start_port=port,
            num_envs=1,
            repo_root=repo_root,
            godot_exe=godot_exe,
            headless_dll=headless_dll,
        )
        try:
            returncode, result_path, stdout_path, stderr_path = _run_evaluate(
                python_exe=python_exe,
                repo_root=repo_root,
                output_dir=label_root,
                backend=backend,
                port=port,
                checkpoint=checkpoint,
                combat_checkpoint=combat_checkpoint,
                seed_file=seed_file,
                seed_suite="benchmark",
                num_games=len(seeds),
                max_steps=max_steps,
                trace_seeds=trace_seeds,
            )
        finally:
            _stop_backend_fleet(procs)
        payload = _load_json(result_path)
        return {
            "backend": backend,
            "returncode": returncode,
            "output_root": str(label_root),
            "result_json": str(result_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "payload": payload,
        }

    report = {
        "run_root": str(run_root),
        "checkpoint": str(checkpoint),
        "combat_checkpoint": str(combat_checkpoint) if combat_checkpoint is not None else None,
        "baseline_backend": baseline_backend,
        "candidate_backend": candidate_backend,
        "seed_count": len(seeds),
        "max_steps": max_steps,
        "trace_seeds": trace_seeds,
        "seeds": seeds,
    }
    report["baseline"] = run_one("baseline", baseline_backend, baseline_port)
    report["candidate"] = run_one("candidate", candidate_backend, candidate_port)
    comparison = _build_comparison(
        report["baseline"]["payload"],
        report["candidate"]["payload"],
        seeds=seeds,
    )
    report["comparison"] = comparison
    report["passed"] = (
        report["baseline"]["returncode"] == 0
        and report["candidate"]["returncode"] == 0
        and not comparison["failed_metrics"]
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare greedy checkpoint rollouts between Godot and headless backends.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--godot-exe", type=Path, default=DEFAULT_GODOT_EXE)
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_HEADLESS_DLL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_HYBRID_CHECKPOINT)
    parser.add_argument("--combat-checkpoint", type=Path, default=DEFAULT_COMBAT_CHECKPOINT)
    parser.add_argument("--baseline-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="godot-http")
    parser.add_argument("--candidate-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="headless-pipe")
    parser.add_argument("--baseline-port", type=int, default=DEFAULT_BASELINE_PORT)
    parser.add_argument("--candidate-port", type=int, default=DEFAULT_CANDIDATE_PORT)
    parser.add_argument("--seed-prefix", type=str, default="ROLLOUT_AUDIT")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--include-default-seeds", action="store_true", default=False)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--trace-count", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.checkpoint = _resolve_checkpoint_path(
        args.checkpoint,
        repo_root=args.repo_root,
        glob_patterns=[
            "STS2AI/Artifacts/**/hybrid_final.pt",
            "STS2AI/Artifacts/**/hybrid_*.pt",
            "artifacts/**/hybrid_final.pt",
            "artifacts/**/hybrid_*.pt",
        ],
    )
    if args.combat_checkpoint is not None:
        args.combat_checkpoint = _resolve_checkpoint_path(
            args.combat_checkpoint,
            repo_root=args.repo_root,
            glob_patterns=[
                "STS2AI/Artifacts/**/combat_*.pt",
                "artifacts/**/combat_*.pt",
                ".claude/worktrees/**/combat_*.pt",
            ],
        )
    seeds = build_seed_list(
        explicit_seeds=args.seeds,
        seed_prefix=args.seed_prefix,
        start_index=args.seed_start,
        count=args.seed_count,
        include_default=args.include_default_seeds,
    )
    report = run_policy_rollout_audit(
        repo_root=args.repo_root,
        python_exe=args.python_exe,
        godot_exe=args.godot_exe,
        headless_dll=args.headless_dll,
        checkpoint=args.checkpoint,
        combat_checkpoint=args.combat_checkpoint,
        baseline_backend=args.baseline_backend,
        candidate_backend=args.candidate_backend,
        baseline_port=args.baseline_port,
        candidate_port=args.candidate_port,
        seeds=seeds,
        max_steps=args.max_steps,
        output_root=args.output_root,
        trace_count=args.trace_count,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
