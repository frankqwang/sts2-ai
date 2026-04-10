from __future__ import annotations
# --- wizardly cleanup 2026-04-08: tools/python subdir sys.path bootstrap ---
# Moved out of tools/python/ root; bootstrap below re-adds the parent dir so
# flat `from combat_nn import X` style imports still resolve.
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # noqa: E402
# --- end bootstrap ---
import _path_init  # noqa: F401

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from headless_sim_runner import start_headless_sim, stop_process
from sim_semantic_audit_common import DEFAULT_GODOT_EXE, DEFAULT_HEADLESS_DLL, DEFAULT_REPO_ROOT
from sts2ai_paths import ARTIFACTS_ROOT, MAINLINE_CHECKPOINT, REPO_ROOT


DEFAULT_BASELINE_START_PORT = 15640
DEFAULT_CANDIDATE_START_PORT = 15740
DEFAULT_REPO_ROOT = REPO_ROOT
DEFAULT_HYBRID_CHECKPOINT = MAINLINE_CHECKPOINT
DEFAULT_COMBAT_CHECKPOINT = MAINLINE_CHECKPOINT
DEFAULT_OUTPUT_ROOT = ARTIFACTS_ROOT / "verification"
DEFAULT_TRAINING_AUDIT_SEED = 20260403
TRACE_STEP_RE = re.compile(r"^\[(\d+)\]")
TRACE_STATE_RE = re.compile(r"^\[(\d+)\]\s+([A-Za-z_]+)(?:\s|\[|:)")
TRACE_END_STEP_RE = re.compile(r"steps=(\d+)")
HEADER_RE = re.compile(
	 r"# outcome=(?P<outcome>\S+) floor=(?P<floor>-?\d+) combats=(?P<combats>-?\d+) time=(?P<time>[0-9.]+)s(?: end_reason=(?P<end_reason>\S+))? error=(?P<error>.*)"
)
CHECKPOINT_ITER_RE = re.compile(r"_(\d{4,6})\.pt$")
NOISY_CHECKPOINT_DIR_MARKERS = (
	"sts2ai/artifacts/verification/",
	"sts2ai/artifacts/recording/",
	"artifacts/verification/",
	"artifacts/recording/",
)


def _resolve_checkpoint_path(preferred: Path, *, repo_root: Path, glob_patterns: list[str]) -> Path:
	if preferred.exists():
		return preferred
	for pattern in glob_patterns:
		candidates = sorted(
			(
				path
				for path in repo_root.glob(pattern)
				if not any(marker in path.as_posix().lower() for marker in NOISY_CHECKPOINT_DIR_MARKERS)
			),
			key=lambda path: path.stat().st_mtime,
			reverse=True,
		)
		if candidates:
			return candidates[0]
	raise FileNotFoundError(f"Checkpoint not found. Preferred={preferred}; patterns={glob_patterns}")


def _infer_resume_start_iteration(checkpoint_path: Path) -> int:
	try:
		import torch  # type: ignore

		payload = torch.load(checkpoint_path, map_location="cpu")
		iteration = payload.get("iteration") if isinstance(payload, dict) else None
		if isinstance(iteration, int) and iteration >= 0:
			return iteration + 1
	except Exception:
		pass

	match = CHECKPOINT_ITER_RE.search(checkpoint_path.name)
	if match:
		return int(match.group(1)) + 1
	return 0


def _transport_for_backend(backend: str) -> str:
	return {
		"godot-http": "http",
		"headless-pipe": "pipe",
		"headless-binary": "pipe-binary",
	}[backend]


def _godot_full_run_server_args(*, repo_root: Path, port: int) -> list[str]:
	return [
		"--headless",
		"--fixed-fps",
		"1000",
		"--path",
		str(repo_root),
		"--",
		"--mcp-port",
		str(port),
		"--full-run-sim-server",
	]


def _launch_backend_fleet(
	*,
	backend: str,
	start_port: int,
	num_envs: int,
	repo_root: Path,
	godot_exe: Path,
	headless_dll: Path,
) -> list[subprocess.Popen]:
	procs: list[subprocess.Popen] = []
	create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
	for offset in range(num_envs):
		port = start_port + offset
		if backend == "godot-http":
			proc = subprocess.Popen(
				[str(godot_exe), *_godot_full_run_server_args(repo_root=repo_root, port=port)],
				stdout=subprocess.DEVNULL,
				stderr=subprocess.DEVNULL,
				creationflags=create_no_window,
			)
		elif backend == "headless-pipe":
			proc = start_headless_sim(port=port, repo_root=repo_root, dll_path=headless_dll)
		elif backend == "headless-binary":
			proc = start_headless_sim(port=port, repo_root=repo_root, dll_path=headless_dll, protocol="bin")
		else:
			raise ValueError(f"Unsupported backend: {backend}")
		procs.append(proc)
	if backend == "godot-http":
		time.sleep(12)
	else:
		time.sleep(2)
	return procs


def _stop_backend_fleet(procs: list[subprocess.Popen]) -> None:
	for proc in procs:
		stop_process(proc)


def _find_training_output_dir(root: Path) -> Path:
	candidates = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("hybrid_")]
	if not candidates:
		raise FileNotFoundError(f"No hybrid_* output dir found under {root}")
	return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_training(
	*,
	python_exe: Path,
	repo_root: Path,
	output_root: Path,
	backend: str,
	start_port: int,
	num_envs: int,
	iterations: int,
	episodes_per_iter: int,
	hybrid_checkpoint: Path,
	combat_checkpoint: Path | None,
	max_episode_steps: int,
	no_mcts: bool,
	ppo_minibatch: int,
	seed: int | None,
	deterministic_policy: bool,
) -> tuple[int, Path, Path, Path]:
	log_dir = output_root / "logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	stdout_path = log_dir / "train.stdout.log"
	stderr_path = log_dir / "train.stderr.log"
	resume_start_iter = _infer_resume_start_iteration(hybrid_checkpoint)
	target_max_iterations = max(iterations, resume_start_iter + iterations)
	command = [
		str(python_exe),
		"STS2AI/Python/train_hybrid.py",
		"--transport",
		_transport_for_backend(backend),
		"--num-envs",
		str(num_envs),
		"--start-port",
		str(start_port),
		"--max-iterations",
		str(target_max_iterations),
		"--episodes-per-iter",
		str(episodes_per_iter),
		"--max-episode-steps",
		str(max_episode_steps),
		"--ppo-minibatch",
		str(ppo_minibatch),
		"--resume",
		str(hybrid_checkpoint),
		"--output-dir",
		str(output_root),
		"--no-save-offline-data",
		"--save-replay-traces",
		"--save-metrics-log",
	]
	if not deterministic_policy:
		command.append("--multi-process")
	else:
		command.append("--deterministic-policy")
	if seed is not None:
		command.extend(["--seed", str(seed)])
	if no_mcts:
		command.append("--no-mcts")
	if combat_checkpoint is not None:
		command.extend(["--resume-mcts", str(combat_checkpoint)])
	with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
		proc = subprocess.run(
			command,
			cwd=repo_root,
			stdout=stdout_file,
			stderr=stderr_file,
			text=True,
			env={**os.environ, "PYTHONUTF8": "1"},
		)
	return proc.returncode, _find_training_output_dir(output_root), stdout_path, stderr_path


def _load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
	if not metrics_path.exists():
		return []
	entries: list[dict[str, Any]] = []
	for line in metrics_path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			entries.append(json.loads(line))
		except json.JSONDecodeError:
			continue
	return entries


def _mean(values: list[float]) -> float:
	return float(sum(values) / len(values)) if values else 0.0


def _tail(values: list[float], last_n: int) -> float:
	return _mean(values[-last_n:]) if values else 0.0


def _summarize_metrics(entries: list[dict[str, Any]], *, tail_n: int) -> dict[str, Any]:
	avg_floors = [float(entry.get("avg_floor", 0.0)) for entry in entries]
	boss_rates = [float(entry.get("boss_reach_rate", 0.0)) for entry in entries]
	act1_rates = [float(entry.get("act1_clear_rate", 0.0)) for entry in entries]
	iter_times = [float(entry.get("iter_time_s", 0.0)) for entry in entries]
	card_skip_rates = [float(entry.get("card_reward_skip_rate", 0.0)) for entry in entries]
	return {
		"iterations": len(entries),
		"avg_floor_mean": round(_mean(avg_floors), 4),
		"avg_floor_tail_mean": round(_tail(avg_floors, tail_n), 4),
		"boss_reach_mean": round(_mean(boss_rates), 4),
		"boss_reach_tail_mean": round(_tail(boss_rates, tail_n), 4),
		"act1_clear_mean": round(_mean(act1_rates), 4),
		"act1_clear_tail_mean": round(_tail(act1_rates, tail_n), 4),
		"iter_time_mean_s": round(_mean(iter_times), 4),
		"iter_time_tail_mean_s": round(_tail(iter_times, tail_n), 4),
		"card_reward_skip_rate_mean": round(_mean(card_skip_rates), 4),
		"card_reward_skip_rate_tail_mean": round(_tail(card_skip_rates, tail_n), 4),
	}


def _parse_replay_file(path: Path) -> dict[str, Any]:
	lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
	header = lines[1].strip() if len(lines) > 1 else ""
	match = HEADER_RE.match(header)
	outcome = None
	floor = None
	combats = None
	duration_s = None
	end_reason = None
	error = None
	if match:
		outcome = match.group("outcome")
		floor = int(match.group("floor"))
		combats = int(match.group("combats"))
		duration_s = float(match.group("time"))
		end_reason = match.group("end_reason")
		error_text = str(match.group("error") or "").strip()
		error = None if error_text in {"", "None", "null"} else error_text
	tag_match = re.search(r"_([A-Z]{3})_f", path.name)
	tag = tag_match.group(1) if tag_match else "UNK"
	max_step = 0
	state_sequence: list[tuple[int, str]] = []
	for line in lines:
		stripped = line.strip()
		state_match = TRACE_STATE_RE.match(stripped)
		if state_match:
			state_sequence.append((int(state_match.group(1)), state_match.group(2).lower()))
		step_match = TRACE_STEP_RE.match(stripped)
		if step_match:
			max_step = max(max_step, int(step_match.group(1)))
		elif "[END]" in line:
			end_match = TRACE_END_STEP_RE.search(line)
			if end_match:
				max_step = max(max_step, int(end_match.group(1)))
	reward_hits = sum(1 for _, state_type in state_sequence if state_type == "combat_rewards")
	card_reward_hits = sum(1 for _, state_type in state_sequence if state_type == "card_reward")
	auto_claim_hits = sum(1 for line in lines if "combat_rewards: auto-claim" in line)
	boss_reached = any("map: boss" in line.lower() or " st=boss" in line.lower() for line in lines)
	return {
		"path": str(path),
		"tag": tag,
		"outcome": outcome,
		"end_reason": end_reason,
		"error": error,
		"floor": floor,
		"combats": combats,
		"duration_s": duration_s,
		"steps": max_step,
		"reward_hits": reward_hits,
		"card_reward_hits": card_reward_hits,
		"auto_claim_hits": auto_claim_hits,
		"boss_reached": boss_reached,
	}


def _summarize_replays(replay_dir: Path) -> dict[str, Any]:
	files = sorted(replay_dir.glob("*.txt")) if replay_dir.exists() else []
	records = [_parse_replay_file(path) for path in files]
	tag_counts = Counter(record["tag"] for record in records)
	outcome_counts = Counter(str(record["outcome"] or "None") for record in records)
	end_reason_counts = Counter(str(record["end_reason"] or "None") for record in records)
	error_count = sum(1 for record in records if record["error"])
	boss_count = sum(1 for record in records if record["boss_reached"])
	reward_hits = [record["reward_hits"] for record in records]
	card_reward_hits = [record["card_reward_hits"] for record in records]
	auto_claim_hits = [record["auto_claim_hits"] for record in records]
	steps = [record["steps"] for record in records]
	floors = [record["floor"] or 0 for record in records]
	return {
		"episode_count": len(records),
		"tag_counts": dict(sorted(tag_counts.items())),
		"outcome_counts": dict(sorted(outcome_counts.items())),
		"end_reason_counts": dict(sorted(end_reason_counts.items())),
		"error_count": error_count,
		"boss_reach_ratio": round(boss_count / max(1, len(records)), 4),
		"avg_episode_steps": round(_mean([float(value) for value in steps]), 4),
		"avg_floor": round(_mean([float(value) for value in floors]), 4),
		"avg_reward_hits": round(_mean([float(value) for value in reward_hits]), 4),
		"avg_card_reward_hits": round(_mean([float(value) for value in card_reward_hits]), 4),
		"avg_auto_claim_hits": round(_mean([float(value) for value in auto_claim_hits]), 4),
		"max_ratio": round(tag_counts.get("MAX", 0) / max(1, len(records)), 4),
		"unk_ratio": round(tag_counts.get("UNK", 0) / max(1, len(records)), 4),
		"err_ratio": round(tag_counts.get("ERR", 0) / max(1, len(records)), 4),
		"sample_records": records[:10],
	}


def _compare_values(left: float, right: float) -> dict[str, Any]:
	abs_diff = right - left
	rel_diff = 0.0 if abs(left) < 1e-9 else abs_diff / abs(left)
	return {
		"baseline": round(left, 4),
		"candidate": round(right, 4),
		"abs_diff": round(abs_diff, 4),
		"rel_diff": round(rel_diff, 4),
	}


def _evaluate_drift(baseline: dict[str, Any], candidate: dict[str, Any], *, thresholds: dict[str, float]) -> tuple[bool, dict[str, Any]]:
	metrics = {
		"avg_floor_tail_mean": _compare_values(baseline["metrics"]["avg_floor_tail_mean"], candidate["metrics"]["avg_floor_tail_mean"]),
		"boss_reach_tail_mean": _compare_values(baseline["metrics"]["boss_reach_tail_mean"], candidate["metrics"]["boss_reach_tail_mean"]),
		"act1_clear_tail_mean": _compare_values(baseline["metrics"]["act1_clear_tail_mean"], candidate["metrics"]["act1_clear_tail_mean"]),
		"avg_episode_steps": _compare_values(baseline["replays"]["avg_episode_steps"], candidate["replays"]["avg_episode_steps"]),
		"avg_reward_hits": _compare_values(baseline["replays"]["avg_reward_hits"], candidate["replays"]["avg_reward_hits"]),
		"avg_card_reward_hits": _compare_values(baseline["replays"]["avg_card_reward_hits"], candidate["replays"]["avg_card_reward_hits"]),
		"max_ratio": _compare_values(baseline["replays"]["max_ratio"], candidate["replays"]["max_ratio"]),
		"unk_ratio": _compare_values(baseline["replays"]["unk_ratio"], candidate["replays"]["unk_ratio"]),
		"err_ratio": _compare_values(baseline["replays"]["err_ratio"], candidate["replays"]["err_ratio"]),
	}
	failures: list[str] = []
	if abs(metrics["avg_floor_tail_mean"]["abs_diff"]) > thresholds["avg_floor_abs"]:
		failures.append("avg_floor_tail_mean")
	if abs(metrics["boss_reach_tail_mean"]["abs_diff"]) > thresholds["boss_reach_abs"]:
		failures.append("boss_reach_tail_mean")
	if abs(metrics["act1_clear_tail_mean"]["abs_diff"]) > thresholds["act1_clear_abs"]:
		failures.append("act1_clear_tail_mean")
	if abs(metrics["avg_episode_steps"]["rel_diff"]) > thresholds["episode_steps_rel"]:
		failures.append("avg_episode_steps")
	if abs(metrics["avg_reward_hits"]["rel_diff"]) > thresholds["reward_hits_rel"]:
		failures.append("avg_reward_hits")
	if abs(metrics["avg_card_reward_hits"]["rel_diff"]) > thresholds["card_reward_hits_rel"]:
		failures.append("avg_card_reward_hits")
	if abs(metrics["max_ratio"]["abs_diff"]) > thresholds["max_ratio_abs"]:
		failures.append("max_ratio")
	if abs(metrics["unk_ratio"]["abs_diff"]) > thresholds["unk_ratio_abs"]:
		failures.append("unk_ratio")
	if abs(metrics["err_ratio"]["abs_diff"]) > thresholds["err_ratio_abs"]:
		failures.append("err_ratio")
	return len(failures) == 0, {"metrics": metrics, "failed_metrics": failures, "thresholds": thresholds}


def _backend_report(
	*,
	name: str,
	backend: str,
	output_root: Path,
	returncode: int,
	train_output_dir: Path,
	stdout_path: Path,
	stderr_path: Path,
) -> dict[str, Any]:
	metrics_entries = _load_metrics(train_output_dir / "metrics.jsonl")
	replays = _summarize_replays(train_output_dir / "replays")
	return {
		"name": name,
		"backend": backend,
		"transport": _transport_for_backend(backend),
		"returncode": returncode,
		"output_root": str(output_root),
		"train_output_dir": str(train_output_dir),
		"stdout_log": str(stdout_path),
		"stderr_log": str(stderr_path),
		"metrics": _summarize_metrics(metrics_entries, tail_n=min(10, max(1, len(metrics_entries)))),
		"replays": replays,
	}


def run_training_semantic_audit(
	*,
	repo_root: Path,
	python_exe: Path,
	godot_exe: Path,
	headless_dll: Path,
	hybrid_checkpoint: Path,
	combat_checkpoint: Path | None,
	baseline_backend: str,
	candidate_backend: str,
	baseline_start_port: int,
	candidate_start_port: int,
	num_envs: int,
	iterations: int,
	episodes_per_iter: int,
	max_episode_steps: int,
	output_root: Path,
	no_mcts: bool,
	ppo_minibatch: int,
	seed: int | None,
	deterministic_policy: bool,
) -> dict[str, Any]:
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	run_root = output_root / f"training_semantic_audit_{timestamp}"
	run_root.mkdir(parents=True, exist_ok=True)

	thresholds = {
		"avg_floor_abs": 1.0,
		"boss_reach_abs": 0.15,
		"act1_clear_abs": 0.10,
		"episode_steps_rel": 0.25,
		"reward_hits_rel": 0.25,
		"card_reward_hits_rel": 0.25,
		"max_ratio_abs": 0.10,
		"unk_ratio_abs": 0.02,
		"err_ratio_abs": 0.02,
	}

	report: dict[str, Any] = {
		"run_root": str(run_root),
		"baseline_backend": baseline_backend,
		"candidate_backend": candidate_backend,
		"num_envs": num_envs,
		"iterations": iterations,
		"episodes_per_iter": episodes_per_iter,
		"max_episode_steps": max_episode_steps,
		"checkpoint": str(hybrid_checkpoint),
		"combat_checkpoint": str(combat_checkpoint) if combat_checkpoint is not None else None,
		"seed": seed,
		"deterministic_policy": deterministic_policy,
	}

	def run_one(label: str, backend: str, start_port: int) -> dict[str, Any]:
		label_root = run_root / label
		label_root.mkdir(parents=True, exist_ok=True)
		procs = _launch_backend_fleet(
			backend=backend,
			start_port=start_port,
			num_envs=num_envs,
			repo_root=repo_root,
			godot_exe=godot_exe,
			headless_dll=headless_dll,
		)
		try:
			returncode, train_output_dir, stdout_path, stderr_path = _run_training(
				python_exe=python_exe,
				repo_root=repo_root,
				output_root=label_root,
				backend=backend,
				start_port=start_port,
				num_envs=num_envs,
				iterations=iterations,
				episodes_per_iter=episodes_per_iter,
				hybrid_checkpoint=hybrid_checkpoint,
				combat_checkpoint=combat_checkpoint,
				max_episode_steps=max_episode_steps,
				no_mcts=no_mcts,
				ppo_minibatch=ppo_minibatch,
				seed=seed,
				deterministic_policy=deterministic_policy,
			)
		finally:
			_stop_backend_fleet(procs)
		return _backend_report(
			name=label,
			backend=backend,
			output_root=label_root,
			returncode=returncode,
			train_output_dir=train_output_dir,
			stdout_path=stdout_path,
			stderr_path=stderr_path,
		)

	report["baseline"] = run_one("baseline", baseline_backend, baseline_start_port)
	report["candidate"] = run_one("candidate", candidate_backend, candidate_start_port)
	pass_drift, drift_report = _evaluate_drift(report["baseline"], report["candidate"], thresholds=thresholds)
	report["comparison"] = drift_report
	data_quality_failures: list[str] = []
	if int(report["baseline"]["metrics"].get("iterations", 0)) <= 0:
		data_quality_failures.append("baseline_zero_iterations")
	if int(report["candidate"]["metrics"].get("iterations", 0)) <= 0:
		data_quality_failures.append("candidate_zero_iterations")
	if int(report["baseline"]["replays"].get("episode_count", 0)) <= 0:
		data_quality_failures.append("baseline_zero_episodes")
	if int(report["candidate"]["replays"].get("episode_count", 0)) <= 0:
		data_quality_failures.append("candidate_zero_episodes")
	report["comparison"]["data_quality_failures"] = data_quality_failures
	report["passed"] = (
		report["baseline"]["returncode"] == 0
		and report["candidate"]["returncode"] == 0
		and pass_drift
		and not data_quality_failures
	)
	return report


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Short-train semantic parity audit between Godot and headless full-run backends.")
	parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
	parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
	parser.add_argument("--godot-exe", type=Path, default=DEFAULT_GODOT_EXE)
	parser.add_argument("--headless-dll", type=Path, default=DEFAULT_HEADLESS_DLL)
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_HYBRID_CHECKPOINT)
	parser.add_argument("--combat-checkpoint", type=Path, default=DEFAULT_COMBAT_CHECKPOINT)
	parser.add_argument("--baseline-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="godot-http")
	parser.add_argument("--candidate-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="headless-pipe")
	parser.add_argument("--baseline-start-port", type=int, default=DEFAULT_BASELINE_START_PORT)
	parser.add_argument("--candidate-start-port", type=int, default=DEFAULT_CANDIDATE_START_PORT)
	parser.add_argument("--num-envs", type=int, default=4)
	parser.add_argument("--iterations", type=int, default=20)
	parser.add_argument("--episodes-per-iter", type=int, default=4)
	parser.add_argument("--max-episode-steps", type=int, default=600)
	parser.add_argument("--ppo-minibatch", type=int, default=32)
	parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_AUDIT_SEED)
	parser.add_argument("--deterministic-policy", action="store_true", default=True)
	parser.add_argument("--stochastic-policy", dest="deterministic_policy", action="store_false")
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
	parser.add_argument("--report-json", type=Path, default=None)
	parser.add_argument("--no-mcts", action="store_true", default=True)
	parser.add_argument("--with-mcts", dest="no_mcts", action="store_false")
	parser.set_defaults(no_mcts=True)
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
	report = run_training_semantic_audit(
		repo_root=args.repo_root,
		python_exe=args.python_exe,
		godot_exe=args.godot_exe,
		headless_dll=args.headless_dll,
		hybrid_checkpoint=args.checkpoint,
		combat_checkpoint=args.combat_checkpoint,
		baseline_backend=args.baseline_backend,
		candidate_backend=args.candidate_backend,
		baseline_start_port=args.baseline_start_port,
		candidate_start_port=args.candidate_start_port,
		num_envs=args.num_envs,
		iterations=args.iterations,
		episodes_per_iter=args.episodes_per_iter,
		max_episode_steps=args.max_episode_steps,
		output_root=args.output_root,
		no_mcts=args.no_mcts,
		ppo_minibatch=args.ppo_minibatch,
		seed=args.seed,
		deterministic_policy=args.deterministic_policy,
	)
	payload = json.dumps(report, indent=2, ensure_ascii=False)
	if args.report_json is not None:
		args.report_json.parent.mkdir(parents=True, exist_ok=True)
		args.report_json.write_text(payload, encoding="utf-8")
	print(payload)
	return 0 if report.get("passed") else 1


if __name__ == "__main__":
	raise SystemExit(main())
