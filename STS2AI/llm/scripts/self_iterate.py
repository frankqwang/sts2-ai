"""One-command self-iteration loop for the LLM policy.

Pipeline:
  current adapter -> policy rollout -> GRPO-lite candidate -> fixed-seed eval
  current/candidate -> promotion gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.paths import ARTIFACTS_ROOT, DATASETS_ROOT, GRPO_ROOT, RUNS_ROOT, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-adapter", type=str, required=True)
    parser.add_argument("--python-exe", type=str, default="", help="运行 rollout/train/eval 的 Python；默认使用 unsloth studio venv。")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--rollout-generations", type=int, default=2)
    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--rollout-max-steps", type=int, default=120)
    parser.add_argument("--rollout-port-base", type=int, default=16040)
    parser.add_argument("--encounter-filter", type=str, default="", help="只迭代 encounter_id 包含此字符串的战斗；空则跑 winnable pool。")
    parser.add_argument("--case-index", type=str, default="", help="Skada combat cases.jsonl；设置后 rollout 使用真实 Skada combat reset case。")
    parser.add_argument("--case-character", type=str, default="IRONCLAD")
    parser.add_argument("--case-floor-min", type=int, default=1)
    parser.add_argument("--case-floor-max", type=int, default=17)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-sample-seed", type=int, default=0)
    parser.add_argument("--case-sample-mode", choices=["file", "random", "stratified"], default="stratified")
    parser.add_argument("--include-lost-cases", action="store_true")
    parser.add_argument("--eval-episodes-per-encounter", type=int, default=2)
    parser.add_argument("--eval-max-steps", type=int, default=120)
    parser.add_argument("--eval-port-base", type=int, default=16140)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--allow-json-like-rollout", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument(
        "--grpo-loss-scope",
        choices=["full_text", "assistant"],
        default="full_text",
        help="GRPO-lite 训练 loss 范围。full_text 先保证自迭代闭环可运行；assistant 用于 response-only mask 稳定后的训练。",
    )
    parser.add_argument("--min-win-rate-delta", type=float, default=0.0)
    parser.add_argument("--max-reward-regression", type=float, default=0.05)
    parser.add_argument("--max-per-encounter-reward-regression", type=float, default=0.15)
    parser.add_argument("--max-per-encounter-win-rate-regression", type=float, default=0.001)
    parser.add_argument("--max-invalid-output-rate", type=float, default=0.02)
    parser.add_argument("--max-mechanism-score-regression", type=float, default=0.03)
    parser.add_argument("--max-missed-visible-lethal-increase", type=int, default=0)
    parser.add_argument("--max-reason-math-contradiction-increase", type=int, default=0)
    parser.add_argument("--max-reason-lethal-claim-error-increase", type=int, default=0)
    parser.add_argument("--max-action-score-lethal-math-contradiction-increase", type=int, default=0)
    parser.add_argument("--max-strict-json-failure-rate", type=float, default=0.05)
    parser.add_argument("--allow-missing-eval-keys", action="store_true")
    parser.add_argument("--promote", action="store_true", help="达标时写 current_adapter.json 指针。")
    parser.add_argument("--dry-run", action="store_true", help="只写计划，不执行命令。")
    return parser.parse_args()


def _default_python_exe() -> str:
    env_value = os.environ.get("STS2_LLM_PYTHON_EXE", "").strip()
    if env_value:
        return env_value
    unsloth_python = Path.home() / ".unsloth" / "studio" / "unsloth_studio" / "Scripts" / "python.exe"
    if unsloth_python.exists():
        return str(unsloth_python)
    return sys.executable


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(
    command: list[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    dry_run: bool,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[self-iterate] cmd: {' '.join(command)}")
    print(f"[self-iterate] stdout -> {stdout_log}")
    print(f"[self-iterate] stderr -> {stderr_log}")
    if dry_run:
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        return 0
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["UNSLOTH_RETURN_LOGITS"] = "1"
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=str(cwd), stdout=out, stderr=err, env=env)
    if proc.returncode != 0:
        print(f"[self-iterate] failed code={proc.returncode}. See {stderr_log}", file=sys.stderr)
    return int(proc.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metrics(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _avg_reward(metrics: dict[str, Any]) -> float:
    value = ((metrics.get("reward") or {}).get("avg"))
    return float(value or 0.0)


def _win_rate(metrics: dict[str, Any]) -> float:
    value = metrics.get("win_rate")
    return float(value or 0.0)


def _invalid_rate(metrics: dict[str, Any]) -> float:
    value = metrics.get("invalid_output_episode_rate")
    return float(value or 0.0)


def _metric_avg(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    if isinstance(value, dict):
        raw = value.get("avg")
        return float(raw if raw is not None else default)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _quality_count(metrics: dict[str, Any], flag: str) -> int:
    value = metrics.get("action_quality")
    if isinstance(value, dict):
        try:
            return int(value.get(flag) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _strict_json_failure_rate(metrics: dict[str, Any]) -> float:
    stats = metrics.get("policy_stats")
    if not isinstance(stats, dict):
        return 0.0
    try:
        failures = int(stats.get("strict_json_failures") or 0)
        ok = int(stats.get("strict_json_ok") or 0)
    except (TypeError, ValueError):
        return 0.0
    total = failures + ok
    return failures / total if total else 0.0


def _by_encounter(metrics: dict[str, Any]) -> dict[str, Any]:
    value = metrics.get("by_encounter")
    return value if isinstance(value, dict) else {}


def _payload_reward(payload: dict[str, Any]) -> float:
    return float(((payload.get("reward") or {}).get("avg")) or 0.0)


def _payload_win(payload: dict[str, Any]) -> float:
    return float(payload.get("win_rate") or 0.0)


def _candidate_passes(
    *,
    current: dict[str, Any],
    candidate: dict[str, Any],
    min_win_rate_delta: float,
    max_reward_regression: float,
    max_per_encounter_reward_regression: float,
    max_per_encounter_win_rate_regression: float,
    max_invalid_output_rate: float,
    max_mechanism_score_regression: float,
    max_missed_visible_lethal_increase: int,
    max_reason_math_contradiction_increase: int,
    max_reason_lethal_claim_error_increase: int,
    max_action_score_lethal_math_contradiction_increase: int,
    max_strict_json_failure_rate: float,
    allow_missing_eval_keys: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    current_win = _win_rate(current)
    candidate_win = _win_rate(candidate)
    current_reward = _avg_reward(current)
    candidate_reward = _avg_reward(candidate)
    candidate_invalid = _invalid_rate(candidate)
    current_mechanism = _metric_avg(current, "mechanism_score", default=1.0)
    candidate_mechanism = _metric_avg(candidate, "mechanism_score", default=1.0)
    current_missed_lethal = _quality_count(current, "missed_visible_lethal")
    candidate_missed_lethal = _quality_count(candidate, "missed_visible_lethal")
    current_reason_math = _quality_count(current, "reason_math_contradiction")
    candidate_reason_math = _quality_count(candidate, "reason_math_contradiction")
    current_lethal_claim_error = _quality_count(current, "reason_claims_lethal_but_action_not_lethal")
    candidate_lethal_claim_error = _quality_count(candidate, "reason_claims_lethal_but_action_not_lethal")
    current_score_lethal_math = _quality_count(current, "action_score_lethal_math_contradiction")
    candidate_score_lethal_math = _quality_count(candidate, "action_score_lethal_math_contradiction")
    candidate_strict_json_failure_rate = _strict_json_failure_rate(candidate)

    if candidate_invalid > max_invalid_output_rate:
        reasons.append(f"candidate invalid rate {candidate_invalid:.4f} > {max_invalid_output_rate:.4f}")
    if candidate_strict_json_failure_rate > max_strict_json_failure_rate:
        reasons.append(
            f"candidate strict JSON failure rate {candidate_strict_json_failure_rate:.4f} "
            f"> {max_strict_json_failure_rate:.4f}"
        )
    if candidate_win + 1e-9 < current_win + min_win_rate_delta:
        reasons.append(
            f"candidate win_rate {candidate_win:.4f} < current {current_win:.4f} + delta {min_win_rate_delta:.4f}"
        )
    if candidate_reward + max_reward_regression + 1e-9 < current_reward:
        reasons.append(
            f"candidate reward {candidate_reward:.4f} regressed below current {current_reward:.4f} "
            f"by more than {max_reward_regression:.4f}"
        )
    if candidate_mechanism + max_mechanism_score_regression + 1e-9 < current_mechanism:
        reasons.append(
            f"candidate mechanism_score {candidate_mechanism:.4f} regressed below current "
            f"{current_mechanism:.4f} by more than {max_mechanism_score_regression:.4f}"
        )
    if candidate_missed_lethal > current_missed_lethal + max_missed_visible_lethal_increase:
        reasons.append(
            f"candidate missed_visible_lethal increased {current_missed_lethal} -> {candidate_missed_lethal}"
        )
    if candidate_reason_math > current_reason_math + max_reason_math_contradiction_increase:
        reasons.append(
            "candidate reason_math_contradiction increased "
            f"{current_reason_math} -> {candidate_reason_math}"
        )
    if candidate_lethal_claim_error > current_lethal_claim_error + max_reason_lethal_claim_error_increase:
        reasons.append(
            "candidate reason_claims_lethal_but_action_not_lethal increased "
            f"{current_lethal_claim_error} -> {candidate_lethal_claim_error}"
        )
    if candidate_score_lethal_math > current_score_lethal_math + max_action_score_lethal_math_contradiction_increase:
        reasons.append(
            "candidate action_score_lethal_math_contradiction increased "
            f"{current_score_lethal_math} -> {candidate_score_lethal_math}"
        )

    current_by = _by_encounter(current)
    candidate_by = _by_encounter(candidate)
    current_keys = set(current_by.keys())
    candidate_keys = set(candidate_by.keys())
    if not allow_missing_eval_keys and current_keys != candidate_keys:
        missing = sorted(current_keys - candidate_keys)
        extra = sorted(candidate_keys - current_keys)
        if missing:
            reasons.append(f"candidate eval missing encounter keys: {missing[:5]}")
        if extra:
            reasons.append(f"candidate eval has unexpected encounter keys: {extra[:5]}")

    for key in sorted(current_keys & candidate_keys):
        cur_payload = current_by[key] if isinstance(current_by.get(key), dict) else {}
        cand_payload = candidate_by[key] if isinstance(candidate_by.get(key), dict) else {}
        cur_win = _payload_win(cur_payload)
        cand_win = _payload_win(cand_payload)
        cur_reward = _payload_reward(cur_payload)
        cand_reward = _payload_reward(cand_payload)
        label = str(cur_payload.get("encounter_label") or key)
        if cand_win + max_per_encounter_win_rate_regression + 1e-9 < cur_win:
            reasons.append(
                f"{label}: win_rate regressed {cur_win:.4f} -> {cand_win:.4f}"
            )
        if cand_reward + max_per_encounter_reward_regression + 1e-9 < cur_reward:
            reasons.append(
                f"{label}: reward regressed {cur_reward:.4f} -> {cand_reward:.4f}"
            )
    return (len(reasons) == 0), reasons


def main() -> None:
    args = parse_args()
    ensure_dirs()
    current_adapter = Path(args.current_adapter).resolve()
    if not current_adapter.exists():
        raise FileNotFoundError(f"current adapter not found: {current_adapter}")

    run_name = args.run_name or f"self_iterate_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root = RUNS_ROOT / run_name
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = f"{run_name}_rollout"
    candidate_name = f"{run_name}_candidate"
    current_eval_name = f"{run_name}_current_eval"
    candidate_eval_name = f"{run_name}_candidate_eval"
    dataset_dir = DATASETS_ROOT / dataset_name
    candidate_run_dir = GRPO_ROOT / candidate_name
    candidate_adapter = candidate_run_dir / "adapter"
    audit_dir = ARTIFACTS_ROOT / "reviews" / f"{run_name}_rollout_audit"

    py = str(Path(args.python_exe or _default_python_exe()).resolve())
    if not Path(py).exists():
        raise FileNotFoundError(f"python exe not found: {py}")
    common_model_flags = ["--parse-retries", str(args.parse_retries)]
    if args.load_in_4bit:
        common_model_flags.append("--load-in-4bit")

    rollout_cmd = [
        py, "-m", "llm.training.grpo_rollout",
        "--adapter-dir", str(current_adapter),
        "--out-subdir", dataset_name,
        "--num-generations", str(args.rollout_generations),
        "--max-steps", str(args.rollout_max_steps),
        "--port-base", str(args.rollout_port_base),
        "--temperature", str(args.rollout_temperature),
        "--seed", str(args.seed),
        *common_model_flags,
    ]
    if args.encounter_filter:
        rollout_cmd += ["--encounter-filter", args.encounter_filter]
    if args.case_index:
        rollout_cmd += [
            "--case-index", args.case_index,
            "--case-character", args.case_character,
            "--case-floor-min", str(args.case_floor_min),
            "--case-floor-max", str(args.case_floor_max),
            "--case-limit", str(args.case_limit),
            "--case-sample-seed", str(args.case_sample_seed or args.seed),
            "--case-sample-mode", args.case_sample_mode,
        ]
        if args.include_lost_cases:
            rollout_cmd.append("--include-lost-cases")
    if args.no_thinking:
        rollout_cmd.append("--no-thinking")
    if args.allow_json_like_rollout:
        rollout_cmd.append("--allow-json-like-rollout")

    audit_cmd = [
        py, "-m", "llm.scripts.audit_rollout_failures",
        "--dataset-dir", str(dataset_dir),
        "--out-dir", str(audit_dir),
        "--log", str(logs_dir / "rollout.stderr.log"),
        "--log", str(logs_dir / "rollout.stdout.log"),
    ]
    pool_ingest_dataset_cmd = [
        py, "-m", "llm.scripts.manage_dataset_pool",
        "ingest-dataset",
        "--dataset-dir", str(dataset_dir),
        "--source-name", run_name,
    ]
    pool_ingest_audit_cmd = [
        py, "-m", "llm.scripts.manage_dataset_pool",
        "ingest-audit",
        "--audit-dir", str(audit_dir),
        "--dataset-dir", str(dataset_dir),
        "--source-name", run_name,
    ]

    train_cmd = [
        py, "-m", "llm.training.grpo_lite",
        "--adapter-dir", str(current_adapter),
        "--dataset-dir", str(dataset_dir),
        "--run-name", candidate_name,
        "--num-epochs", str(args.num_epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--max-seq-length", str(args.max_seq_length),
        "--loss-scope", args.grpo_loss_scope,
    ]
    if args.load_in_4bit:
        train_cmd.append("--load-in-4bit")

    current_eval_cmd = [
        py, "-m", "llm.eval.policy_eval",
        "--adapter-dir", str(current_adapter),
        "--run-name", current_eval_name,
        "--episodes-per-encounter", str(args.eval_episodes_per_encounter),
        "--max-steps", str(args.eval_max_steps),
        "--port-base", str(args.eval_port_base),
        "--seed", str(args.seed),
        *common_model_flags,
    ]
    if args.encounter_filter:
        current_eval_cmd += ["--encounter-filter", args.encounter_filter]
    candidate_eval_cmd = [
        py, "-m", "llm.eval.policy_eval",
        "--adapter-dir", str(candidate_adapter),
        "--run-name", candidate_eval_name,
        "--episodes-per-encounter", str(args.eval_episodes_per_encounter),
        "--max-steps", str(args.eval_max_steps),
        "--port-base", str(args.eval_port_base + 100),
        "--seed", str(args.seed),
        *common_model_flags,
    ]
    if args.encounter_filter:
        candidate_eval_cmd += ["--encounter-filter", args.encounter_filter]

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_root": str(run_root),
        "current_adapter": str(current_adapter),
        "dataset_dir": str(dataset_dir),
        "rollout_audit_summary": str(audit_dir / "summary.json"),
        "candidate_adapter": str(candidate_adapter),
        "current_eval_metrics": str(ARTIFACTS_ROOT / "evals" / current_eval_name / "metrics.json"),
        "candidate_eval_metrics": str(ARTIFACTS_ROOT / "evals" / candidate_eval_name / "metrics.json"),
        "args": vars(args),
        "python_exe": py,
        "commands": {
            "rollout": rollout_cmd,
            "rollout_audit": audit_cmd,
            "pool_ingest_dataset": pool_ingest_dataset_cmd,
            "pool_ingest_audit": pool_ingest_audit_cmd,
            "train": train_cmd,
            "current_eval": current_eval_cmd,
            "candidate_eval": candidate_eval_cmd,
        },
        "status": "planned" if args.dry_run else "running",
    }
    _write_json(run_root / "manifest.json", manifest)

    steps = [
        ("rollout", rollout_cmd),
        ("rollout_audit", audit_cmd),
        ("pool_ingest_dataset", pool_ingest_dataset_cmd),
        ("pool_ingest_audit", pool_ingest_audit_cmd),
        ("train", train_cmd),
        ("current_eval", current_eval_cmd),
        ("candidate_eval", candidate_eval_cmd),
    ]
    for label, cmd in steps:
        code = _run(
            cmd,
            cwd=Path(__file__).resolve().parents[2],
            stdout_log=logs_dir / f"{label}.stdout.log",
            stderr_log=logs_dir / f"{label}.stderr.log",
            dry_run=args.dry_run,
        )
        manifest.setdefault("step_results", {})[label] = {
            "returncode": code,
            "stdout": str(logs_dir / f"{label}.stdout.log"),
            "stderr": str(logs_dir / f"{label}.stderr.log"),
        }
        _write_json(run_root / "manifest.json", manifest)
        if code != 0:
            manifest["status"] = "failed"
            manifest["failed_step"] = label
            _write_json(run_root / "manifest.json", manifest)
            raise SystemExit(code)

    if args.dry_run:
        manifest["status"] = "dry_run"
        _write_json(run_root / "manifest.json", manifest)
        print(f"[self-iterate] dry-run manifest -> {run_root / 'manifest.json'}")
        return

    current_metrics_path = Path(manifest["current_eval_metrics"])
    candidate_metrics_path = Path(manifest["candidate_eval_metrics"])
    current_metrics = _metrics(current_metrics_path)
    candidate_metrics = _metrics(candidate_metrics_path)
    passed, reasons = _candidate_passes(
        current=current_metrics,
        candidate=candidate_metrics,
        min_win_rate_delta=args.min_win_rate_delta,
        max_reward_regression=args.max_reward_regression,
        max_per_encounter_reward_regression=args.max_per_encounter_reward_regression,
        max_per_encounter_win_rate_regression=args.max_per_encounter_win_rate_regression,
        max_invalid_output_rate=args.max_invalid_output_rate,
        max_mechanism_score_regression=args.max_mechanism_score_regression,
        max_missed_visible_lethal_increase=args.max_missed_visible_lethal_increase,
        max_reason_math_contradiction_increase=args.max_reason_math_contradiction_increase,
        max_reason_lethal_claim_error_increase=args.max_reason_lethal_claim_error_increase,
        max_action_score_lethal_math_contradiction_increase=args.max_action_score_lethal_math_contradiction_increase,
        max_strict_json_failure_rate=args.max_strict_json_failure_rate,
        allow_missing_eval_keys=args.allow_missing_eval_keys,
    )

    promotion = {
        "passed": passed,
        "reasons": reasons,
        "current": {
            "adapter": str(current_adapter),
            "win_rate": _win_rate(current_metrics),
            "reward_avg": _avg_reward(current_metrics),
            "invalid_output_rate": _invalid_rate(current_metrics),
            "mechanism_score": _metric_avg(current_metrics, "mechanism_score", default=1.0),
            "missed_visible_lethal": _quality_count(current_metrics, "missed_visible_lethal"),
            "reason_math_contradiction": _quality_count(current_metrics, "reason_math_contradiction"),
            "reason_claims_lethal_but_action_not_lethal": _quality_count(
                current_metrics,
                "reason_claims_lethal_but_action_not_lethal",
            ),
            "action_score_lethal_math_contradiction": _quality_count(
                current_metrics,
                "action_score_lethal_math_contradiction",
            ),
            "strict_json_failure_rate": _strict_json_failure_rate(current_metrics),
        },
        "candidate": {
            "adapter": str(candidate_adapter),
            "win_rate": _win_rate(candidate_metrics),
            "reward_avg": _avg_reward(candidate_metrics),
            "invalid_output_rate": _invalid_rate(candidate_metrics),
            "mechanism_score": _metric_avg(candidate_metrics, "mechanism_score", default=1.0),
            "missed_visible_lethal": _quality_count(candidate_metrics, "missed_visible_lethal"),
            "reason_math_contradiction": _quality_count(candidate_metrics, "reason_math_contradiction"),
            "reason_claims_lethal_but_action_not_lethal": _quality_count(
                candidate_metrics,
                "reason_claims_lethal_but_action_not_lethal",
            ),
            "action_score_lethal_math_contradiction": _quality_count(
                candidate_metrics,
                "action_score_lethal_math_contradiction",
            ),
            "strict_json_failure_rate": _strict_json_failure_rate(candidate_metrics),
        },
        "by_encounter": {
            "current": _by_encounter(current_metrics),
            "candidate": _by_encounter(candidate_metrics),
        },
    }
    manifest["promotion"] = promotion
    manifest["status"] = "completed"

    if args.promote and passed:
        pointer = ARTIFACTS_ROOT / "current_adapter.json"
        _write_json(pointer, {
            "adapter_dir": str(candidate_adapter),
            "promoted_at": datetime.now().isoformat(timespec="seconds"),
            "source_run": str(run_root),
            "promotion": promotion,
        })
        manifest["promoted_pointer"] = str(pointer)
    elif args.promote and not passed:
        manifest["promoted_pointer"] = None

    _write_json(run_root / "manifest.json", manifest)
    _write_json(run_root / "promotion.json", promotion)
    print(f"[self-iterate] promotion passed={passed}")
    if reasons:
        for reason in reasons:
            print(f"[self-iterate] gate: {reason}")
    print(f"[self-iterate] manifest -> {run_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
