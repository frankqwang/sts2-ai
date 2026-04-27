"""Run a conservative Kimi-canonical candidate train/eval/gate experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.paths import ARTIFACTS_ROOT, DATASETS_ROOT, EVALS_ROOT, GRPO_ROOT, RUNS_ROOT, ensure_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--dataset-dir",
        default=str(DATASETS_ROOT / "kimi_teacher_100_filtered_canonical_20260425"),
    )
    parser.add_argument("--current-adapter", default="")
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--candidate-run-name", default="")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--eval-episodes-per-encounter", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=120)
    parser.add_argument("--eval-port-base", type=int, default=17400)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--skip-train", action="store_true", help="Reuse an existing candidate adapter and run eval/gate only.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _default_python_exe() -> str:
    env_value = os.environ.get("STS2_LLM_PYTHON_EXE", "").strip()
    if env_value:
        return env_value
    unsloth_python = Path.home() / ".unsloth" / "studio" / "unsloth_studio" / "Scripts" / "python.exe"
    if unsloth_python.exists():
        return str(unsloth_python)
    return sys.executable


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _current_adapter_from_pointer() -> Path:
    pointer = ARTIFACTS_ROOT / "current_adapter.json"
    payload = _read_json(pointer)
    adapter = Path(str(payload.get("adapter_dir") or "")).resolve()
    if not adapter.exists():
        raise FileNotFoundError(f"current adapter not found from pointer: {adapter}")
    return adapter


def _run_step(
    *,
    label: str,
    command: list[str],
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    env: dict[str, str],
    dry_run: bool,
    allow_returncodes: set[int] | None = None,
) -> int:
    allow_returncodes = allow_returncodes or {0}
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[kimi-gate] {label}: {' '.join(command)}")
    print(f"[kimi-gate] stdout -> {stdout_log}")
    print(f"[kimi-gate] stderr -> {stderr_log}")
    if dry_run:
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        return 0
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=str(cwd), stdout=out, stderr=err, env=env)
    if proc.returncode not in allow_returncodes:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    ensure_dirs()
    run_name = args.run_name or f"kimi_canonical_gate_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root = RUNS_ROOT / run_name
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    py = Path(args.python_exe or _default_python_exe()).resolve()
    if not py.exists():
        raise FileNotFoundError(f"python exe not found: {py}")
    current_adapter = Path(args.current_adapter).resolve() if args.current_adapter else _current_adapter_from_pointer()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not (dataset_dir / "train.jsonl").exists():
        raise FileNotFoundError(f"dataset train.jsonl not found: {dataset_dir}")

    candidate_run = args.candidate_run_name or f"{run_name}_candidate"
    candidate_adapter = GRPO_ROOT / candidate_run / "adapter"
    if args.skip_train and not candidate_adapter.exists():
        raise FileNotFoundError(f"--skip-train candidate adapter not found: {candidate_adapter}")
    current_eval = f"{run_name}_current_eval"
    candidate_eval = f"{run_name}_candidate_eval"
    promotion_path = run_root / "promotion.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["UNSLOTH_RETURN_LOGITS"] = "1"

    train_cmd = [
        str(py), "-m", "llm.training.grpo_lite",
        "--adapter-dir", str(current_adapter),
        "--dataset-dir", str(dataset_dir),
        "--run-name", candidate_run,
        "--num-epochs", str(args.num_epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--max-seq-length", str(args.max_seq_length),
        "--loss-scope", "assistant",
    ]
    current_eval_cmd = [
        str(py), "-m", "llm.eval.policy_eval",
        "--adapter-dir", str(current_adapter),
        "--run-name", current_eval,
        "--episodes-per-encounter", str(args.eval_episodes_per_encounter),
        "--max-steps", str(args.eval_max_steps),
        "--port-base", str(args.eval_port_base),
        "--seed", str(args.seed),
        "--parse-retries", str(args.parse_retries),
    ]
    candidate_eval_cmd = [
        str(py), "-m", "llm.eval.policy_eval",
        "--adapter-dir", str(candidate_adapter),
        "--run-name", candidate_eval,
        "--episodes-per-encounter", str(args.eval_episodes_per_encounter),
        "--max-steps", str(args.eval_max_steps),
        "--port-base", str(args.eval_port_base + 100),
        "--seed", str(args.seed),
        "--parse-retries", str(args.parse_retries),
    ]
    compare_cmd = [
        str(py), "-m", "llm.scripts.compare_policy_eval",
        "--current-metrics", str(EVALS_ROOT / current_eval / "metrics.json"),
        "--candidate-metrics", str(EVALS_ROOT / candidate_eval / "metrics.json"),
        "--candidate-adapter", str(candidate_adapter),
        "--out", str(promotion_path),
    ]

    manifest: dict[str, Any] = {
        "run_name": run_name,
        "run_root": str(run_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python_exe": str(py),
        "current_adapter": str(current_adapter),
        "dataset_dir": str(dataset_dir),
        "candidate_run": candidate_run,
        "candidate_adapter": str(candidate_adapter),
        "current_eval_metrics": str(EVALS_ROOT / current_eval / "metrics.json"),
        "candidate_eval_metrics": str(EVALS_ROOT / candidate_eval / "metrics.json"),
        "promotion": str(promotion_path),
        "args": vars(args),
        "commands": {
        "train": train_cmd,
            "current_eval": current_eval_cmd,
            "candidate_eval": candidate_eval_cmd,
            "compare": compare_cmd,
        },
        "status": "planned" if args.dry_run else "running",
        "steps": {},
    }
    _write_json(run_root / "manifest.json", manifest)

    cwd = Path(__file__).resolve().parents[2]
    steps = [
        *([] if args.skip_train else [("train", train_cmd, {0})]),
        ("current_eval", current_eval_cmd, {0}),
        ("candidate_eval", candidate_eval_cmd, {0}),
        ("compare", compare_cmd, {0, 2}),
    ]
    try:
        for label, command, allowed in steps:
            code = _run_step(
                label=label,
                command=command,
                cwd=cwd,
                stdout_log=logs_dir / f"{label}.stdout.log",
                stderr_log=logs_dir / f"{label}.stderr.log",
                env=env,
                dry_run=args.dry_run,
                allow_returncodes=allowed,
            )
            manifest["steps"][label] = {
                "returncode": code,
                "stdout": str(logs_dir / f"{label}.stdout.log"),
                "stderr": str(logs_dir / f"{label}.stderr.log"),
            }
            _write_json(run_root / "manifest.json", manifest)
    except subprocess.CalledProcessError as exc:
        manifest["status"] = "failed"
        manifest["failed_step"] = next((label for label, command, _allowed in steps if command == exc.cmd), "")
        manifest["failed_returncode"] = int(exc.returncode)
        _write_json(run_root / "manifest.json", manifest)
        raise

    manifest["status"] = "dry_run" if args.dry_run else "completed"
    if promotion_path.exists():
        manifest["promotion_result"] = _read_json(promotion_path)
    _write_json(run_root / "manifest.json", manifest)
    print(f"[kimi-gate] manifest -> {run_root / 'manifest.json'}")
    print(f"[kimi-gate] promotion -> {promotion_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
