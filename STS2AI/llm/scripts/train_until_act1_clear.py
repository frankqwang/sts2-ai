"""Run full-run validation and self-training until Act 1 is cleared.

This script is intentionally an orchestrator only. It does not add a new
training algorithm. Each cycle:

1. Run a visible spectator full-run with the current combat/non-combat LoRA.
2. Stop when the trace reaches act >= 2 or a victory outcome.
3. If not clear, run one self_train_loop block and update the combat adapter
   from that loop's final_adapter.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.paths import LLM_ROOT, REPO_ROOT, RUNS_ROOT, STS2AI_ROOT, ensure_dirs, resolve_default_python_exe


RUN_LINE_RE = re.compile(r"\brun:\s+.*?\bact=(\d+)\s+floor=(\d+|\?)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-adapter", required=True)
    parser.add_argument("--non-combat-adapter", required=True)
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026042610)
    parser.add_argument("--spectate-max-steps", type=int, default=500)
    parser.add_argument("--spectate-step-delay", type=float, default=0.05)
    parser.add_argument("--spectate-max-new-tokens", type=int, default=180)
    parser.add_argument("--mcp-port-base", type=int, default=15526)
    parser.add_argument("--train-iterations", type=int, default=1)
    parser.add_argument(
        "--train-stages",
        default="CHOMPERS;SLIMES,CULTISTS;BOWLBUGS,GREMLIN;act1_midrun;",
    )
    parser.add_argument("--rollout-generations", type=int, default=3)
    parser.add_argument("--rollout-temperature", type=float, default=0.75)
    parser.add_argument("--rollout-max-steps", type=int, default=120)
    parser.add_argument("--eval-episodes-per-encounter", type=int, default=2)
    parser.add_argument("--eval-max-steps", type=int, default=120)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--allow-json-like-rollout", action="store_true")
    parser.add_argument("--case-index", type=str, default="", help="Skada combat cases.jsonl；设置后训练 rollout 使用真实 Skada reset case。")
    parser.add_argument("--case-character", type=str, default="IRONCLAD")
    parser.add_argument("--case-floor-min", type=int, default=1)
    parser.add_argument("--case-floor-max", type=int, default=17)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-sample-seed", type=int, default=0)
    parser.add_argument("--case-sample-mode", choices=["file", "random", "stratified"], default="stratified")
    parser.add_argument("--include-lost-cases", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _default_python_exe() -> str:
    return str(resolve_default_python_exe())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run(
    command: list[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    dry_run: bool,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        stdout_log.write_text("\n".join(command) + "\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        return 0
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=str(cwd), stdout=out, stderr=err, env=env)
    return int(proc.returncode)


def _last_jsonl_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if not last:
        return {}
    return json.loads(last)


def _last_trace_position(trace_path: Path) -> dict[str, Any]:
    row = _last_jsonl_row(trace_path)
    user_message = str(row.get("user_message") or "")
    match = RUN_LINE_RE.search(user_message)
    if not match:
        return {"act": None, "floor": None, "step": row.get("step")}
    floor_raw = match.group(2)
    return {
        "act": int(match.group(1)),
        "floor": None if floor_raw == "?" else int(floor_raw),
        "step": row.get("step"),
    }


def _act1_status(metrics_path: Path, trace_path: Path) -> dict[str, Any]:
    metrics = _read_json(metrics_path)
    result = metrics.get("episode_result") if isinstance(metrics.get("episode_result"), dict) else {}
    trace = metrics.get("policy_trace") if isinstance(metrics.get("policy_trace"), dict) else {}
    position = _last_trace_position(trace_path)
    outcome = str(result.get("run_outcome") or "").strip().lower()
    act = position.get("act")
    cleared = outcome == "victory" or (isinstance(act, int) and act >= 2)
    return {
        "cleared": cleared,
        "outcome": outcome or None,
        "terminal": result.get("terminal"),
        "stopped": result.get("stopped"),
        "state_type": result.get("state_type"),
        "act": act,
        "floor": position.get("floor"),
        "last_step": position.get("step"),
        "steps": trace.get("steps"),
        "invalid_outputs": trace.get("invalid_outputs"),
        "invalid_output_rate": trace.get("invalid_output_rate"),
        "first_attempt_invalid": trace.get("first_attempt_invalid"),
        "action_index_out_of_range_attempts": trace.get("action_index_out_of_range_attempts"),
        "routes": trace.get("routes"),
        "quality_flags": trace.get("quality_flags"),
    }


def _post_eval_artifacts(
    *,
    python_exe: Path,
    trace_path: Path,
    eval_dir: Path,
    logs_dir: Path,
    cycle: int,
    dry_run: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    if not dry_run and not trace_path.exists():
        record["skipped"] = "missing_trace"
        return record

    review_dir = eval_dir / "review"
    review_stdout = logs_dir / f"cycle{cycle:02d}.review.stdout.log"
    review_stderr = logs_dir / f"cycle{cycle:02d}.review.stderr.log"
    review_cmd = [
        str(python_exe),
        "-m",
        "llm.scripts.review_step_trace",
        "--trace",
        str(trace_path),
        "--out-dir",
        str(review_dir),
        "--append-experience",
    ]
    review_code = _run(
        review_cmd,
        cwd=STS2AI_ROOT,
        stdout_log=review_stdout,
        stderr_log=review_stderr,
        dry_run=dry_run,
    )
    record.update({
        "review_returncode": review_code,
        "review_dir": str(review_dir),
        "review_summary": str(review_dir / "summary.json"),
        "review_stdout": str(review_stdout),
        "review_stderr": str(review_stderr),
    })

    viewer_path = eval_dir / "step_trace_viewer.html"
    viewer_stdout = logs_dir / f"cycle{cycle:02d}.viewer.stdout.log"
    viewer_stderr = logs_dir / f"cycle{cycle:02d}.viewer.stderr.log"
    viewer_cmd = [
        str(python_exe),
        "-m",
        "llm.scripts.trace_viewer_html",
        "--trace",
        str(trace_path),
        "--out",
        str(viewer_path),
        "--title",
        f"Act1 fullrun cycle {cycle:02d}",
    ]
    viewer_code = _run(
        viewer_cmd,
        cwd=STS2AI_ROOT,
        stdout_log=viewer_stdout,
        stderr_log=viewer_stderr,
        dry_run=dry_run,
    )
    record.update({
        "viewer_returncode": viewer_code,
        "viewer_html": str(viewer_path),
        "viewer_stdout": str(viewer_stdout),
        "viewer_stderr": str(viewer_stderr),
    })
    return record


def _spectate_command(
    *,
    combat_adapter: Path,
    non_combat_adapter: Path,
    output_dir: Path,
    seed: int,
    mcp_port: int,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(LLM_ROOT / "scripts" / "spectate_llm.ps1"),
        "-StopExistingGodot",
        "-AdapterDir",
        str(combat_adapter),
        "-CombatAdapterDir",
        str(combat_adapter),
        "-NonCombatAdapterDir",
        str(non_combat_adapter),
        "-MaxNewTokens",
        str(args.spectate_max_new_tokens),
        "-Temperature",
        "0",
        "-ActionMode",
        "index",
        "-Seed",
        str(seed),
        "-MaxSteps",
        str(args.spectate_max_steps),
        "-StepDelay",
        f"{args.spectate_step_delay:.3f}",
        "-McpPort",
        str(mcp_port),
        "-OutputDir",
        str(output_dir),
        "-MuteAudio",
    ]
    return command


def _train_command(
    *,
    python_exe: Path,
    current_adapter: Path,
    run_name: str,
    seed: int,
    port_base: int,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(python_exe),
        "-m",
        "llm.scripts.self_train_loop",
        "--current-adapter",
        str(current_adapter),
        "--python-exe",
        str(python_exe),
        "--run-name",
        run_name,
        "--iterations",
        str(args.train_iterations),
        "--stages",
        args.train_stages,
        "--focus-hard-cases",
        "--rollout-generations",
        str(args.rollout_generations),
        "--rollout-temperature",
        str(args.rollout_temperature),
        "--rollout-max-steps",
        str(args.rollout_max_steps),
        "--eval-episodes-per-encounter",
        str(args.eval_episodes_per_encounter),
        "--eval-max-steps",
        str(args.eval_max_steps),
        "--port-base",
        str(port_base),
        "--seed",
        str(seed),
        "--parse-retries",
        str(args.parse_retries),
        "--num-epochs",
        str(args.num_epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--max-seq-length",
        str(args.max_seq_length),
        "--write-current-pointer",
    ]
    if args.case_index:
        command += [
            "--case-index", args.case_index,
            "--case-character", args.case_character,
            "--case-floor-min", str(args.case_floor_min),
            "--case-floor-max", str(args.case_floor_max),
            "--case-limit", str(args.case_limit),
            "--case-sample-seed", str(args.case_sample_seed or seed),
            "--case-sample-mode", args.case_sample_mode,
        ]
        if args.include_lost_cases:
            command.append("--include-lost-cases")
    if args.no_thinking:
        command.append("--no-thinking")
    if args.allow_json_like_rollout:
        command.append("--allow-json-like-rollout")
    if args.load_in_4bit:
        command.append("--load-in-4bit")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    args = parse_args()
    ensure_dirs()

    python_exe = Path(args.python_exe or _default_python_exe()).resolve()
    if not python_exe.exists():
        raise FileNotFoundError(f"python exe not found: {python_exe}")

    current_adapter = Path(args.current_adapter).resolve()
    non_combat_adapter = Path(args.non_combat_adapter).resolve()
    if not current_adapter.exists():
        raise FileNotFoundError(f"current adapter not found: {current_adapter}")
    if not non_combat_adapter.exists():
        raise FileNotFoundError(f"non-combat adapter not found: {non_combat_adapter}")

    run_name = args.run_name or f"act1_until_clear_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root = RUNS_ROOT / run_name
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_root": str(run_root),
        "initial_combat_adapter": str(current_adapter),
        "current_combat_adapter": str(current_adapter),
        "non_combat_adapter": str(non_combat_adapter),
        "python_exe": str(python_exe),
        "args": vars(args),
        "cycles": [],
        "status": "planned" if args.dry_run else "running",
    }
    _write_json(run_root / "manifest.json", manifest)

    for cycle in range(max(0, args.max_cycles)):
        cycle_name = f"{run_name}_cycle{cycle:02d}"
        eval_dir = STS2AI_ROOT / "Artifacts" / "llm" / "spectate_llm" / f"{cycle_name}_fullrun"
        eval_stdout = logs_dir / f"cycle{cycle:02d}.spectate.stdout.log"
        eval_stderr = logs_dir / f"cycle{cycle:02d}.spectate.stderr.log"
        eval_seed = args.seed + cycle
        mcp_port = args.mcp_port_base + cycle
        eval_cmd = _spectate_command(
            combat_adapter=current_adapter,
            non_combat_adapter=non_combat_adapter,
            output_dir=eval_dir,
            seed=eval_seed,
            mcp_port=mcp_port,
            args=args,
        )
        print(f"[act1-loop] cycle={cycle} eval -> {eval_dir}")
        eval_code = _run(
            eval_cmd,
            cwd=REPO_ROOT,
            stdout_log=eval_stdout,
            stderr_log=eval_stderr,
            dry_run=args.dry_run,
        )
        metrics_path = eval_dir / "metrics.json"
        trace_path = eval_dir / "step_trace.jsonl"
        status = _act1_status(metrics_path, trace_path) if not args.dry_run else {"cleared": False}
        post_eval = _post_eval_artifacts(
            python_exe=python_exe,
            trace_path=trace_path,
            eval_dir=eval_dir,
            logs_dir=logs_dir,
            cycle=cycle,
            dry_run=args.dry_run,
        )
        cycle_record: dict[str, Any] = {
            "cycle": cycle,
            "adapter_before": str(current_adapter),
            "eval_returncode": eval_code,
            "eval_run_root": str(eval_dir),
            "eval_stdout": str(eval_stdout),
            "eval_stderr": str(eval_stderr),
            "eval_metrics": str(metrics_path),
            "act1_status": status,
            "post_eval_artifacts": post_eval,
        }
        manifest["cycles"].append(cycle_record)
        manifest["current_combat_adapter"] = str(current_adapter)
        _write_json(run_root / "manifest.json", manifest)

        if eval_code == 0 and status.get("cleared"):
            manifest["status"] = "act1_cleared"
            manifest["final_adapter"] = str(current_adapter)
            manifest["cleared_cycle"] = cycle
            _write_json(run_root / "manifest.json", manifest)
            print(f"[act1-loop] Act 1 cleared at cycle {cycle}")
            return

        if eval_code != 0:
            cycle_record["eval_failed"] = True
            manifest["status"] = "failed"
            manifest["failed_cycle"] = cycle
            manifest["failed_step"] = "spectate_eval"
            _write_json(run_root / "manifest.json", manifest)
            raise SystemExit(eval_code)

        if cycle >= args.max_cycles - 1:
            break

        train_name = f"{cycle_name}_train"
        train_stdout = logs_dir / f"cycle{cycle:02d}.train.stdout.log"
        train_stderr = logs_dir / f"cycle{cycle:02d}.train.stderr.log"
        train_cmd = _train_command(
            python_exe=python_exe,
            current_adapter=current_adapter,
            run_name=train_name,
            seed=args.seed + 1000 + cycle,
            port_base=16440 + cycle * 1000,
            args=args,
        )
        print(f"[act1-loop] cycle={cycle} train -> {train_name}")
        train_code = _run(
            train_cmd,
            cwd=STS2AI_ROOT,
            stdout_log=train_stdout,
            stderr_log=train_stderr,
            dry_run=args.dry_run,
        )
        train_manifest_path = RUNS_ROOT / train_name / "manifest.json"
        train_manifest = _read_json(train_manifest_path)
        final_adapter = train_manifest.get("final_adapter")
        cycle_record.update(
            {
                "train_returncode": train_code,
                "train_run_name": train_name,
                "train_manifest": str(train_manifest_path),
                "train_stdout": str(train_stdout),
                "train_stderr": str(train_stderr),
                "train_final_adapter": final_adapter,
            }
        )
        if train_code != 0:
            manifest["status"] = "failed"
            manifest["failed_cycle"] = cycle
            manifest["failed_step"] = "self_train_loop"
            _write_json(run_root / "manifest.json", manifest)
            raise SystemExit(train_code)
        if final_adapter:
            current_adapter = Path(str(final_adapter)).resolve()
        cycle_record["adapter_after"] = str(current_adapter)
        manifest["current_combat_adapter"] = str(current_adapter)
        _write_json(run_root / "manifest.json", manifest)

    manifest["status"] = "max_cycles_reached"
    manifest["final_adapter"] = str(current_adapter)
    _write_json(run_root / "manifest.json", manifest)
    print(f"[act1-loop] max cycles reached. final adapter: {current_adapter}")


if __name__ == "__main__":
    main()
