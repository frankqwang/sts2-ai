from __future__ import annotations
# --- wizardly cleanup 2026-04-08: tools/python subdir sys.path bootstrap ---
# Moved out of tools/python/ root; bootstrap below re-adds the parent dir so
# flat `from combat_nn import X` style imports still resolve.
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # noqa: E402
# --- end bootstrap ---

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from sim_semantic_audit_common import (
	COMBAT_STATE_TYPES,
	DEFAULT_GODOT_EXE,
	DEFAULT_HEADLESS_DLL,
	DEFAULT_PORT,
	DEFAULT_REPO_ROOT,
	backend_client,
	build_seed_list,
	choose_policy_action,
	classify_capped_reason,
	compact_state_record,
	print_json_report,
)


def run_cap_episode(
	client: Any,
	*,
	seed: str,
	policy: str,
	max_steps: int,
	rng_seed: int = 42,
) -> dict[str, Any]:
	rng = random.Random(rng_seed)
	state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
	steps = 0
	outcome = "max_steps"
	error: str | None = None
	boss_reached = False
	first_boss_step: int | None = None
	last_state = compact_state_record(state, step=0)

	for step in range(max_steps):
		steps = step + 1
		last_state = compact_state_record(state, step=step)
		state_type = str(last_state["state_type"] or "").lower()
		if state_type == "boss":
			boss_reached = True
			first_boss_step = first_boss_step if first_boss_step is not None else step
		if bool(state.get("terminal", False)) or state_type == "game_over":
			outcome = str(state.get("run_outcome") or "game_over")
			break
		action = choose_policy_action(policy, state, rng)
		try:
			state = client.act(action)
		except Exception as exc:
			error = str(exc)
			outcome = "error"
			break
	else:
		outcome = "max_steps"

	end_reason = "terminal" if outcome not in {"max_steps", "error"} else outcome
	capped_reason = classify_capped_reason(last_state, boss_reached=boss_reached) if outcome == "max_steps" else None
	return {
		"seed": seed,
		"policy": policy,
		"max_steps": max_steps,
		"steps": steps,
		"outcome": outcome,
		"end_reason": end_reason,
		"error": error,
		"boss_reached": boss_reached,
		"first_boss_step": first_boss_step,
		"last_state": last_state,
		"capped_reason": capped_reason,
	}


def compare_cap_result(
	baseline_cap: dict[str, Any],
	candidate_cap: dict[str, Any],
	baseline_long: dict[str, Any] | None,
	candidate_long: dict[str, Any] | None,
) -> list[str]:
	mismatches: list[str] = []
	for key in ("outcome", "boss_reached", "capped_reason"):
		if baseline_cap.get(key) != candidate_cap.get(key):
			mismatches.append(f"{key}: {baseline_cap.get(key)!r} != {candidate_cap.get(key)!r}")
	for key in ("state_type", "floor", "act"):
		left = (baseline_cap.get("last_state") or {}).get(key)
		right = (candidate_cap.get("last_state") or {}).get(key)
		if left != right:
			mismatches.append(f"last_state.{key}: {left!r} != {right!r}")
	if (baseline_long is None) != (candidate_long is None):
		mismatches.append("long_cap_followup presence differs")
	elif baseline_long is not None and candidate_long is not None:
		for key in ("outcome", "boss_reached", "capped_reason"):
			if baseline_long.get(key) != candidate_long.get(key):
				mismatches.append(f"long_cap.{key}: {baseline_long.get(key)!r} != {candidate_long.get(key)!r}")
	return mismatches


def run_audit(
	*,
	baseline_backend: str,
	candidate_backend: str,
	baseline_port: int,
	candidate_port: int,
	auto_launch: bool,
	repo_root: Path,
	godot_exe: Path,
	headless_dll: Path,
	seeds: list[str],
	policies: list[str],
	step_cap: int,
	long_cap: int,
) -> dict[str, Any]:
	report: dict[str, Any] = {
		"baseline_backend": baseline_backend,
		"candidate_backend": candidate_backend,
		"step_cap": step_cap,
		"long_cap": long_cap,
		"seed_count": len(seeds),
		"policies": policies,
		"results": [],
	}

	with backend_client(
		backend=baseline_backend,
		port=baseline_port,
		auto_launch=auto_launch,
		repo_root=repo_root,
		godot_exe=godot_exe,
		headless_dll=headless_dll,
	) as baseline, backend_client(
		backend=candidate_backend,
		port=candidate_port,
		auto_launch=auto_launch,
		repo_root=repo_root,
		godot_exe=godot_exe,
		headless_dll=headless_dll,
	) as candidate:
		for policy in policies:
			for seed in seeds:
				baseline_cap = run_cap_episode(baseline, seed=seed, policy=policy, max_steps=step_cap)
				candidate_cap = run_cap_episode(candidate, seed=seed, policy=policy, max_steps=step_cap)
				baseline_long = None
				candidate_long = None
				if baseline_cap["outcome"] == "max_steps":
					baseline_long = run_cap_episode(baseline, seed=seed, policy=policy, max_steps=long_cap)
				if candidate_cap["outcome"] == "max_steps":
					candidate_long = run_cap_episode(candidate, seed=seed, policy=policy, max_steps=long_cap)
				report["results"].append(
					{
						"seed": seed,
						"policy": policy,
						"baseline": {"cap": baseline_cap, "long_cap": baseline_long},
						"candidate": {"cap": candidate_cap, "long_cap": candidate_long},
						"mismatches": compare_cap_result(baseline_cap, candidate_cap, baseline_long, candidate_long),
					}
				)

	report["summary"] = summarize_report(report["results"])
	report["pass"] = report["summary"]["mismatch_count"] == 0
	return report


def summarize_report(results: list[dict[str, Any]]) -> dict[str, Any]:
	def summarize_backend(entries: list[dict[str, Any]], backend_key: str) -> dict[str, Any]:
		outcomes = Counter()
		capped_reasons = Counter()
		boss_unk = 0
		resolved_after_long_cap = 0
		for entry in entries:
			cap = entry[backend_key]["cap"]
			long_cap = entry[backend_key]["long_cap"]
			outcomes[str(cap["outcome"])] += 1
			if cap["outcome"] == "max_steps":
				reason = str(cap["capped_reason"] or "unknown")
				capped_reasons[reason] += 1
				if reason == "boss":
					boss_unk += 1
				if long_cap is not None and long_cap["outcome"] != "max_steps":
					resolved_after_long_cap += 1
		return {
			"outcomes_at_cap": dict(sorted(outcomes.items())),
			"capped_reasons_at_cap": dict(sorted(capped_reasons.items())),
			"boss_unk_at_cap": boss_unk,
			"resolved_after_long_cap": resolved_after_long_cap,
		}

	mismatch_examples: list[dict[str, Any]] = []
	for entry in results:
		if entry["mismatches"] and len(mismatch_examples) < 25:
			mismatch_examples.append(
				{
					"seed": entry["seed"],
					"policy": entry["policy"],
					"mismatches": entry["mismatches"],
					"baseline_cap": entry["baseline"]["cap"],
					"candidate_cap": entry["candidate"]["cap"],
				}
			)

	return {
		"baseline": summarize_backend(results, "baseline"),
		"candidate": summarize_backend(results, "candidate"),
		"mismatch_count": sum(1 for entry in results if entry["mismatches"]),
		"mismatch_examples": mismatch_examples,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Audit boss/UNK outcome parity across full-run backends.")
	parser.add_argument("--baseline-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="godot-http")
	parser.add_argument("--candidate-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="headless-pipe")
	parser.add_argument("--baseline-port", type=int, default=DEFAULT_PORT + 130)
	parser.add_argument("--candidate-port", type=int, default=DEFAULT_PORT + 131)
	parser.add_argument("--auto-launch", action="store_true")
	parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
	parser.add_argument("--godot-exe", type=Path, default=DEFAULT_GODOT_EXE)
	parser.add_argument("--headless-dll", type=Path, default=DEFAULT_HEADLESS_DLL)
	parser.add_argument("--seed-prefix", default="COVERAGE_SCAN")
	parser.add_argument("--start-index", type=int, default=0)
	parser.add_argument("--count", type=int, default=200)
	parser.add_argument("--seed", dest="explicit_seeds", action="append", default=None)
	parser.add_argument("--include-default-seeds", action="store_true")
	parser.add_argument("--policy", choices=["coverage", "exit", "training", "deterministic"], action="append", default=None)
	parser.add_argument("--step-cap", type=int, default=600)
	parser.add_argument("--long-cap", type=int, default=1200)
	parser.add_argument("--report-json", type=Path, default=None)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	policies = args.policy or ["exit", "coverage"]
	seeds = build_seed_list(
		explicit_seeds=args.explicit_seeds,
		seed_prefix=args.seed_prefix,
		start_index=args.start_index,
		count=args.count,
		include_default=args.include_default_seeds,
	)
	report = run_audit(
		baseline_backend=args.baseline_backend,
		candidate_backend=args.candidate_backend,
		baseline_port=args.baseline_port,
		candidate_port=args.candidate_port,
		auto_launch=args.auto_launch,
		repo_root=args.repo_root,
		godot_exe=args.godot_exe,
		headless_dll=args.headless_dll,
		seeds=seeds,
		policies=policies,
		step_cap=args.step_cap,
		long_cap=args.long_cap,
	)
	print_json_report(report, args.report_json)
	return 0 if report["pass"] else 1


if __name__ == "__main__":
	raise SystemExit(main())
