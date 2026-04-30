"""Multi-iteration curriculum loop for LLM self-training.

This is the higher-level loop above ``self_iterate.py``:

  current adapter
    -> rollout/train/eval candidate for one curriculum stage
    -> promote candidate only if gated eval passes
    -> repeat with the promoted adapter or focus hard cases after failure

It deliberately avoids save/load MCTS. The only improvement signal is model
rollout quality measured by fixed-seed policy eval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.planner_hint import DEFAULT_PLANNER_HINT_REFRESH, PLANNER_HINT_REFRESH_CHOICES
from llm.paths import ARTIFACTS_ROOT, RUNS_ROOT, ensure_dirs, resolve_default_python_exe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-adapter", type=str, required=True)
    parser.add_argument("--python-exe", type=str, default="", help="默认使用 STS2_LLM_PYTHON_EXE 或 STS2AI/llm/.venv311。")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--stages",
        type=str,
        default="CHOMPERS;SLIMES,CULTISTS;skada_floor_06,skada_floor_07,skada_floor_08;",
        help="分号分隔的 Skada case curriculum filter；空 stage 表示全部已加载 Skada cases。",
    )
    parser.add_argument("--focus-hard-cases", action="store_true", help="晋级失败后下一轮优先跑 hard_cases。")
    parser.add_argument("--hard-case-limit", type=int, default=4)
    parser.add_argument("--rollout-generations", type=int, default=4)
    parser.add_argument("--rollout-temperature", type=float, default=0.75)
    parser.add_argument("--rollout-max-steps", type=int, default=120)
    parser.add_argument("--eval-episodes-per-encounter", type=int, default=2)
    parser.add_argument("--eval-max-steps", type=int, default=120)
    parser.add_argument("--port-base", type=int, default=16440)
    parser.add_argument("--case-index", type=str, required=True, help="Skada combat cases.jsonl。")
    parser.add_argument("--case-character", type=str, default="IRONCLAD")
    parser.add_argument("--case-floor-min", type=int, default=1)
    parser.add_argument("--case-floor-max", type=int, default=17)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-sample-seed", type=int, default=0)
    parser.add_argument("--case-sample-mode", choices=["file", "random", "stratified"], default="stratified")
    parser.add_argument("--include-lost-cases", action="store_true")
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--allow-json-like-rollout", action="store_true")
    parser.add_argument("--planner-hint-adapter-dir", type=str, default="", help="可选 planner-hint LoRA adapter；每轮 self_iterate 透传给 rollout/eval。")
    parser.add_argument(
        "--planner-hint-refresh",
        choices=list(PLANNER_HINT_REFRESH_CHOICES),
        default=DEFAULT_PLANNER_HINT_REFRESH,
    )
    parser.add_argument("--planner-hint-max-new-tokens", type=int, default=240)
    parser.add_argument("--co-train-planner", action="store_true", help="每轮同步训练 planner candidate，并用 joint eval 决定是否一起晋级。")
    parser.add_argument("--planner-train-dataset-dir", type=str, default="", help="可选 planner-hint SFT 数据集；为空时使用本轮 teacher 产出的 planner_hint 数据。")
    parser.add_argument("--planner-min-train-rows", type=int, default=1)
    parser.add_argument("--planner-num-epochs", type=int, default=1)
    parser.add_argument("--planner-batch-size", type=int, default=1)
    parser.add_argument("--planner-grad-accum", type=int, default=4)
    parser.add_argument("--planner-lr", type=float, default=1e-4)
    parser.add_argument("--planner-max-seq-length", type=int, default=2048)
    parser.add_argument("--planner-load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--kimi-teacher", action="store_true", help="每轮 rollout 后调用 teacher 复盘，并构建 combat/planner 两套 teacher 数据。")
    parser.add_argument(
        "--teacher-provider",
        choices=["deepseek", "kimi", "kimi_code", "claude_cli"],
        default=os.environ.get("TEACHER_PROVIDER", "deepseek"),
        help="teacher 底层 provider，默认 deepseek-v4-pro；也支持 kimi / claude_cli。",
    )
    parser.add_argument("--teacher-model", type=str, default=os.environ.get("TEACHER_MODEL", ""))
    parser.add_argument("--teacher-max-workers", type=int, default=1)
    parser.add_argument("--teacher-skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-claude-command", type=str, default=os.environ.get("CLAUDE_CLI_COMMAND", "claude"))
    parser.add_argument("--teacher-claude-proxy", type=str, default=os.environ.get("CLAUDE_PROXY", "http://127.0.0.1:7897"))
    parser.add_argument("--kimi-limit-episodes", type=int, default=20)
    parser.add_argument("--kimi-max-api-calls", type=int, default=20)
    parser.add_argument("--kimi-model", type=str, default=os.environ.get("KIMI_MODEL", "kimi-k2.6"))
    parser.add_argument("--kimi-base-url", type=str, default=os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"))
    parser.add_argument("--kimi-api-key-env", type=str, default="MOONSHOT_API_KEY")
    parser.add_argument("--kimi-max-tokens", type=int, default=4096)
    parser.add_argument("--kimi-thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--kimi-timeout-s", type=float, default=180.0)
    parser.add_argument("--kimi-sleep-s", type=float, default=0.2)
    parser.add_argument("--kimi-max-decision-state-chars", type=int, default=7000)
    parser.add_argument("--kimi-damage-turns", type=int, default=2)
    parser.add_argument("--kimi-min-confidence", type=float, default=0.75)
    parser.add_argument("--kimi-min-review-ok-rate", type=float, default=0.5)
    parser.add_argument("--kimi-min-teacher-rows", type=int, default=0)
    parser.add_argument("--kimi-fail-on-quality-gate", action="store_true")
    parser.add_argument("--kimi-dry-run", action="store_true")
    parser.add_argument("--kimi-append-experience", action="store_true")
    parser.add_argument("--train-from-pool-after-teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pool-train-target-size", type=int, default=5000)
    parser.add_argument("--pool-gold-min-ratio", type=float, default=0.15)
    parser.add_argument("--grpo-loss-scope", choices=["full_text", "assistant"], default="full_text")
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
    parser.add_argument("--write-current-pointer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args()


def _default_python_exe() -> str:
    return str(resolve_default_python_exe())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def _run(
    command: list[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    dry_run: bool,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[self-train-loop] cmd: {' '.join(command)}")
    print(f"[self-train-loop] stdout -> {stdout_log}")
    print(f"[self-train-loop] stderr -> {stderr_log}")
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
    return int(proc.returncode)


def _slug(value: str) -> str:
    text = value.strip() or "all"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")[:48] or "stage"


def _parse_stages(raw: str) -> list[str]:
    stages = [part.strip() for part in raw.split(";")]
    if not stages:
        return [""]
    return stages


def _hard_case_filter(metrics_path: Path, limit: int) -> str:
    metrics = _read_json(metrics_path)
    cases = metrics.get("hard_cases")
    if not isinstance(cases, list):
        return ""
    keys: list[str] = []
    for item in cases:
        if isinstance(item, dict) and item.get("encounter_key"):
            keys.append(str(item["encounter_key"]))
        if len(keys) >= limit:
            break
    return ",".join(keys)


def _promotion_passed(run_name: str) -> tuple[bool, dict[str, Any]]:
    promotion_path = RUNS_ROOT / run_name / "promotion.json"
    promotion = _read_json(promotion_path)
    return bool(promotion.get("passed")), promotion


def _candidate_adapter(run_name: str) -> Path:
    manifest = _read_json(RUNS_ROOT / run_name / "manifest.json")
    return Path(str(manifest.get("candidate_adapter") or "")).resolve()


def _planner_candidate_adapter(run_name: str) -> Path | None:
    manifest = _read_json(RUNS_ROOT / run_name / "manifest.json")
    raw = manifest.get("planner_candidate_adapter")
    return Path(str(raw)).resolve() if raw else None


def _eval_metrics_path(run_name: str, which: str) -> Path:
    manifest = _read_json(RUNS_ROOT / run_name / "manifest.json")
    key = "candidate_eval_metrics" if which == "candidate" else "current_eval_metrics"
    return Path(str(manifest.get(key) or "")).resolve()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    current_adapter = Path(args.current_adapter).resolve()
    if not current_adapter.exists():
        raise FileNotFoundError(f"current adapter not found: {current_adapter}")
    current_planner_adapter = Path(args.planner_hint_adapter_dir).resolve() if args.planner_hint_adapter_dir else None
    if current_planner_adapter is not None and not current_planner_adapter.exists():
        raise FileNotFoundError(f"planner-hint adapter not found: {current_planner_adapter}")

    py = str(Path(args.python_exe or _default_python_exe()).resolve())
    if not Path(py).exists():
        raise FileNotFoundError(f"python exe not found: {py}")

    run_name = args.run_name or f"self_train_loop_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root = RUNS_ROOT / run_name
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    stages = _parse_stages(args.stages)
    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_root": str(run_root),
        "initial_adapter": str(current_adapter),
        "current_adapter": str(current_adapter),
        "initial_planner_adapter": str(current_planner_adapter) if current_planner_adapter is not None else None,
        "current_planner_adapter": str(current_planner_adapter) if current_planner_adapter is not None else None,
        "python_exe": py,
        "args": vars(args),
        "stages": stages,
        "iterations": [],
        "status": "planned" if args.dry_run else "running",
    }
    _write_json(run_root / "manifest.json", manifest)

    hard_filter = ""
    cwd = Path(__file__).resolve().parents[3]
    for i in range(max(0, args.iterations)):
        base_filter = stages[i % len(stages)]
        stage_filter = hard_filter if hard_filter else base_filter
        iter_name = f"{run_name}_iter{i:02d}_{_slug(stage_filter)}"
        rollout_port_base = args.port_base + i * 400
        eval_port_base = rollout_port_base + 200

        cmd = [
            py, "-m", "llm.scripts.automation.self_iterate",
            "--current-adapter", str(current_adapter),
            "--python-exe", py,
            "--run-name", iter_name,
            "--rollout-generations", str(args.rollout_generations),
            "--rollout-temperature", str(args.rollout_temperature),
            "--rollout-max-steps", str(args.rollout_max_steps),
            "--rollout-port-base", str(rollout_port_base),
            "--eval-episodes-per-encounter", str(args.eval_episodes_per_encounter),
            "--eval-max-steps", str(args.eval_max_steps),
            "--eval-port-base", str(eval_port_base),
            "--seed", str(args.seed + i),
            "--parse-retries", str(args.parse_retries),
            "--num-epochs", str(args.num_epochs),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--lr", str(args.lr),
            "--max-seq-length", str(args.max_seq_length),
            "--grpo-loss-scope", args.grpo_loss_scope,
            "--min-win-rate-delta", str(args.min_win_rate_delta),
            "--max-reward-regression", str(args.max_reward_regression),
            "--max-per-encounter-reward-regression", str(args.max_per_encounter_reward_regression),
            "--max-per-encounter-win-rate-regression", str(args.max_per_encounter_win_rate_regression),
            "--max-invalid-output-rate", str(args.max_invalid_output_rate),
            "--max-mechanism-score-regression", str(args.max_mechanism_score_regression),
            "--max-missed-visible-lethal-increase", str(args.max_missed_visible_lethal_increase),
            "--max-reason-math-contradiction-increase", str(args.max_reason_math_contradiction_increase),
            "--max-reason-lethal-claim-error-increase", str(args.max_reason_lethal_claim_error_increase),
            "--max-action-score-lethal-math-contradiction-increase",
            str(args.max_action_score_lethal_math_contradiction_increase),
            "--max-strict-json-failure-rate", str(args.max_strict_json_failure_rate),
        ]
        if stage_filter:
            cmd += ["--encounter-filter", stage_filter]
        cmd += [
            "--case-index", args.case_index,
            "--case-character", args.case_character,
            "--case-floor-min", str(args.case_floor_min),
            "--case-floor-max", str(args.case_floor_max),
            "--case-limit", str(args.case_limit),
            "--case-sample-seed", str(args.case_sample_seed or (args.seed + i)),
            "--case-sample-mode", args.case_sample_mode,
        ]
        if args.include_lost_cases:
            cmd.append("--include-lost-cases")
        if args.load_in_4bit:
            cmd.append("--load-in-4bit")
        if args.no_thinking:
            cmd.append("--no-thinking")
        if args.allow_json_like_rollout:
            cmd.append("--allow-json-like-rollout")
        if current_planner_adapter is not None:
            cmd += [
                "--planner-hint-adapter-dir", str(current_planner_adapter),
                "--planner-hint-refresh", args.planner_hint_refresh,
                "--planner-hint-max-new-tokens", str(args.planner_hint_max_new_tokens),
            ]
        if args.co_train_planner:
            cmd += [
                "--co-train-planner",
                "--planner-min-train-rows", str(args.planner_min_train_rows),
                "--planner-num-epochs", str(args.planner_num_epochs),
                "--planner-batch-size", str(args.planner_batch_size),
                "--planner-grad-accum", str(args.planner_grad_accum),
                "--planner-lr", str(args.planner_lr),
                "--planner-max-seq-length", str(args.planner_max_seq_length),
            ]
            if args.planner_train_dataset_dir:
                cmd += ["--planner-train-dataset-dir", args.planner_train_dataset_dir]
            cmd.append("--planner-load-in-4bit" if args.planner_load_in_4bit else "--no-planner-load-in-4bit")
        if args.kimi_teacher:
            cmd += [
                "--kimi-teacher",
                "--teacher-provider", args.teacher_provider,
                "--teacher-model", args.teacher_model,
                "--teacher-max-workers", str(args.teacher_max_workers),
                "--teacher-claude-command", args.teacher_claude_command,
                "--teacher-claude-proxy", args.teacher_claude_proxy,
                "--kimi-limit-episodes", str(args.kimi_limit_episodes),
                "--kimi-max-api-calls", str(args.kimi_max_api_calls),
                "--kimi-model", args.kimi_model,
                "--kimi-base-url", args.kimi_base_url,
                "--kimi-api-key-env", args.kimi_api_key_env,
                "--kimi-max-tokens", str(args.kimi_max_tokens),
                "--kimi-thinking", args.kimi_thinking,
                "--kimi-timeout-s", str(args.kimi_timeout_s),
                "--kimi-sleep-s", str(args.kimi_sleep_s),
                "--kimi-max-decision-state-chars", str(args.kimi_max_decision_state_chars),
                "--kimi-damage-turns", str(args.kimi_damage_turns),
                "--kimi-min-confidence", str(args.kimi_min_confidence),
                "--kimi-min-review-ok-rate", str(args.kimi_min_review_ok_rate),
                "--kimi-min-teacher-rows", str(args.kimi_min_teacher_rows),
                "--pool-train-target-size", str(args.pool_train_target_size),
                "--pool-gold-min-ratio", str(args.pool_gold_min_ratio),
            ]
            cmd.append("--teacher-skip-existing" if args.teacher_skip_existing else "--no-teacher-skip-existing")
            cmd.append(
                "--train-from-pool-after-teacher"
                if args.train_from_pool_after_teacher
                else "--no-train-from-pool-after-teacher"
            )
            if args.kimi_fail_on_quality_gate:
                cmd.append("--kimi-fail-on-quality-gate")
            if args.kimi_dry_run:
                cmd.append("--kimi-dry-run")
            if args.kimi_append_experience:
                cmd.append("--kimi-append-experience")
        if args.allow_missing_eval_keys:
            cmd.append("--allow-missing-eval-keys")
        if args.dry_run:
            cmd.append("--dry-run")

        code = _run(
            cmd,
            cwd=cwd,
            stdout_log=logs_dir / f"iter{i:02d}.stdout.log",
            stderr_log=logs_dir / f"iter{i:02d}.stderr.log",
            dry_run=False,
        )
        iter_record: dict[str, Any] = {
            "index": i,
            "run_name": iter_name,
            "stage_filter": stage_filter,
            "used_hard_filter": bool(hard_filter),
            "returncode": code,
            "stdout": str(logs_dir / f"iter{i:02d}.stdout.log"),
            "stderr": str(logs_dir / f"iter{i:02d}.stderr.log"),
            "adapter_before": str(current_adapter),
            "planner_adapter_before": str(current_planner_adapter) if current_planner_adapter is not None else None,
        }

        if code != 0:
            iter_record["status"] = "failed"
            manifest["iterations"].append(iter_record)
            manifest["status"] = "failed"
            manifest["failed_iteration"] = i
            _write_json(run_root / "manifest.json", manifest)
            if args.continue_on_failure:
                hard_filter = ""
                continue
            raise SystemExit(code)

        if args.dry_run:
            iter_record["status"] = "dry_run"
            manifest["iterations"].append(iter_record)
            _write_json(run_root / "manifest.json", manifest)
            continue

        passed, promotion = _promotion_passed(iter_name)
        candidate_adapter = _candidate_adapter(iter_name)
        planner_candidate_adapter = _planner_candidate_adapter(iter_name)
        iter_record["promotion"] = promotion
        iter_record["candidate_adapter"] = str(candidate_adapter)
        iter_record["planner_candidate_adapter"] = (
            str(planner_candidate_adapter) if planner_candidate_adapter is not None else None
        )
        if passed and candidate_adapter.exists():
            current_adapter = candidate_adapter
            if args.co_train_planner and planner_candidate_adapter is not None and planner_candidate_adapter.exists():
                current_planner_adapter = planner_candidate_adapter
            hard_filter = ""
            iter_record["status"] = "promoted"
            iter_record["adapter_after"] = str(current_adapter)
            iter_record["planner_adapter_after"] = (
                str(current_planner_adapter) if current_planner_adapter is not None else None
            )
        else:
            iter_record["status"] = "rejected"
            iter_record["adapter_after"] = str(current_adapter)
            iter_record["planner_adapter_after"] = (
                str(current_planner_adapter) if current_planner_adapter is not None else None
            )
            if args.focus_hard_cases:
                current_metrics = _eval_metrics_path(iter_name, "current")
                hard_filter = _hard_case_filter(current_metrics, args.hard_case_limit)
                iter_record["next_hard_filter"] = hard_filter
            else:
                hard_filter = ""

        manifest["iterations"].append(iter_record)
        manifest["current_adapter"] = str(current_adapter)
        manifest["current_planner_adapter"] = str(current_planner_adapter) if current_planner_adapter is not None else None
        _write_json(run_root / "manifest.json", manifest)

    manifest["status"] = "completed" if manifest.get("status") != "failed" else "failed"
    manifest["final_adapter"] = str(current_adapter)
    manifest["final_planner_adapter"] = str(current_planner_adapter) if current_planner_adapter is not None else None

    if args.write_current_pointer and not args.dry_run:
        pointer = ARTIFACTS_ROOT / "current_adapter.json"
        _write_json(pointer, {
            "adapter_dir": str(current_adapter),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source_run": str(run_root),
        })
        manifest["current_pointer"] = str(pointer)
        if current_planner_adapter is not None:
            planner_pointer = ARTIFACTS_ROOT / "current_planner_hint_adapter.json"
            _write_json(planner_pointer, {
                "adapter_dir": str(current_planner_adapter),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "source_run": str(run_root),
            })
            manifest["current_planner_pointer"] = str(planner_pointer)

    _write_json(run_root / "manifest.json", manifest)
    print(f"[self-train-loop] final adapter: {current_adapter}")
    print(f"[self-train-loop] manifest -> {run_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
