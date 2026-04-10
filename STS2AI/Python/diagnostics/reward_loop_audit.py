from __future__ import annotations
# --- wizardly cleanup 2026-04-08: tools/python subdir sys.path bootstrap ---
# Moved out of tools/python/ root; bootstrap below re-adds the parent dir so
# flat `from combat_nn import X` style imports still resolve.
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # noqa: E402
# --- end bootstrap ---

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from sim_semantic_audit_common import (
	DEFAULT_GODOT_EXE,
	DEFAULT_HEADLESS_DLL,
	DEFAULT_PORT,
	DEFAULT_REPO_ROOT,
	REWARD_STATE_TYPES,
	backend_client,
	build_seed_list,
	choose_policy_action,
	compact_state_record,
	print_json_report,
)


def choose_reward_audit_action(
	policy: str,
	state: dict[str, Any],
	rng: random.Random,
	*,
	active_chain: dict[str, Any] | None = None,
) -> dict[str, Any]:
	state_type = str(state.get("state_type") or "").lower()
	legal = [action for action in state.get("legal_actions") or [] if bool(action.get("is_enabled", True))]
	if not legal:
		return {"action": "wait"}

	if state_type == "combat_rewards":
		claims = [action for action in legal if (action.get("action") or "").lower() == "claim_reward"]
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if claims:
			combat_reward_events = [
				event for event in (active_chain or {}).get("events") or []
				if str(event.get("state_type") or "").lower() == "combat_rewards"
			]
			card_reward_seen = any(
				str(event.get("state_type") or "").lower() == "card_reward"
				for event in (active_chain or {}).get("events") or []
			)
			if card_reward_seen:
				if len(combat_reward_events) >= 2:
					last_count = int(combat_reward_events[-1].get("claim_reward_count") or 0)
					prev_count = int(combat_reward_events[-2].get("claim_reward_count") or 0)
					if proceed is not None and last_count == prev_count:
						# We already came back from card_reward and a reward claim did not
						# shrink the remaining reward set. Treat the rest as skippable to
						# avoid manufacturing an endless post-reward audit loop.
						return proceed
				return min(claims, key=lambda action: int(action.get("index") or 999))
			return max(claims, key=lambda action: int(action.get("index") or -1))
		if proceed is not None:
			return proceed

	if state_type == "card_reward":
		selects = [action for action in legal if (action.get("action") or "").lower() == "select_card_reward"]
		if selects:
			return min(selects, key=lambda action: int(action.get("index") or 999))
		skip = next((action for action in legal if (action.get("action") or "").lower() == "skip_card_reward"), None)
		if skip is not None:
			return skip

	return choose_policy_action(policy, state, rng)


def run_reward_loop_episode(
	client: Any,
	*,
	seed: str,
	policy: str,
	max_steps: int,
	rng_seed: int = 42,
) -> dict[str, Any]:
	rng = random.Random(rng_seed)
	state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
	chains: list[dict[str, Any]] = []
	active_chain: dict[str, Any] | None = None
	steps = 0
	outcome = "max_steps"
	error: str | None = None
	last_state = compact_state_record(state, step=0)

	for step in range(max_steps):
		steps = step + 1
		record = compact_state_record(state, step=step)
		last_state = record
		state_type = str(record["state_type"] or "").lower()
		if state_type in REWARD_STATE_TYPES:
			if active_chain is None:
				active_chain = {
					"seed": seed,
					"policy": policy,
					"start_step": step,
					"start_floor": record["floor"],
					"events": [],
					"select_counts": [],
					"claim_counts": [],
				}
			active_chain["events"].append(record)
			if state_type == "card_reward":
				active_chain["select_counts"].append(int(record["select_card_reward_count"] or 0))
			if state_type == "combat_rewards":
				active_chain["claim_counts"].append(int(record["claim_reward_count"] or 0))
		elif active_chain is not None:
			active_chain["end_step"] = step
			active_chain["exit_state_type"] = state_type
			chains.append(classify_reward_chain(active_chain))
			active_chain = None

		if bool(state.get("terminal", False)) or state_type == "game_over":
			outcome = str(state.get("run_outcome") or "game_over")
			break

		action = choose_reward_audit_action(policy, state, rng, active_chain=active_chain)
		try:
			state = client.act(action)
		except Exception as exc:
			error = str(exc)
			outcome = "error"
			break
	else:
		outcome = "max_steps"

	if active_chain is not None:
		active_chain["end_step"] = steps
		active_chain["exit_state_type"] = None
		active_chain["capped_in_reward_state"] = True
		chains.append(classify_reward_chain(active_chain))

	suspicious = [chain for chain in chains if chain["classification"] != "healthy"]
	return {
		"seed": seed,
		"policy": policy,
		"steps": steps,
		"outcome": outcome,
		"error": error,
		"last_state": last_state,
		"reward_chain_count": len(chains),
		"suspicious_chain_count": len(suspicious),
		"chains": chains,
	}


def classify_reward_chain(chain: dict[str, Any]) -> dict[str, Any]:
	events = chain["events"]
	select_counts = list(chain["select_counts"])
	claim_counts = list(chain["claim_counts"])
	issues: list[str] = []

	if any(count < 0 for count in select_counts + claim_counts):
		issues.append("negative_action_count")
	if any(select_counts[index] > select_counts[index - 1] for index in range(1, len(select_counts))):
		issues.append("card_reward_options_increased")
	if any(select_counts[index] == select_counts[index - 1] and select_counts[index] > 0 for index in range(1, len(select_counts))):
		issues.append("card_reward_options_stalled")
	if any(count <= 0 for count in claim_counts):
		issues.append("combat_rewards_without_claim")
	if bool(chain.get("capped_in_reward_state")):
		issues.append("capped_in_reward_state")
	if len(events) > 16:
		issues.append("reward_chain_too_long")
	if not events:
		issues.append("empty_reward_chain")

	classification = "healthy" if not issues else "suspicious"
	return {
		"seed": chain["seed"],
		"policy": chain["policy"],
		"start_step": chain["start_step"],
		"end_step": chain.get("end_step"),
		"start_floor": chain["start_floor"],
		"exit_state_type": chain.get("exit_state_type"),
		"event_count": len(events),
		"select_counts": select_counts,
		"claim_counts": claim_counts,
		"classification": classification,
		"issues": issues,
		"events": events,
	}


def compare_episode_pair(baseline_episode: dict[str, Any], candidate_episode: dict[str, Any]) -> list[str]:
	mismatches: list[str] = []
	for key in ("outcome", "reward_chain_count", "suspicious_chain_count"):
		if baseline_episode.get(key) != candidate_episode.get(key):
			mismatches.append(f"{key}: {baseline_episode.get(key)!r} != {candidate_episode.get(key)!r}")

	baseline_chains = [
		(chain["select_counts"], chain["claim_counts"], chain["exit_state_type"], chain["classification"])
		for chain in baseline_episode["chains"]
	]
	candidate_chains = [
		(chain["select_counts"], chain["claim_counts"], chain["exit_state_type"], chain["classification"])
		for chain in candidate_episode["chains"]
	]
	if baseline_chains != candidate_chains:
		mismatches.append("reward_chains differ")
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
	max_steps: int,
) -> dict[str, Any]:
	report: dict[str, Any] = {
		"baseline_backend": baseline_backend,
		"candidate_backend": candidate_backend,
		"max_steps": max_steps,
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
				baseline_episode = run_reward_loop_episode(baseline, seed=seed, policy=policy, max_steps=max_steps)
				candidate_episode = run_reward_loop_episode(candidate, seed=seed, policy=policy, max_steps=max_steps)
				report["results"].append(
					{
						"seed": seed,
						"policy": policy,
						"baseline": baseline_episode,
						"candidate": candidate_episode,
						"mismatches": compare_episode_pair(baseline_episode, candidate_episode),
					}
				)

	summary = summarize_report(report["results"])
	report["summary"] = summary
	report["pass"] = summary["mismatch_count"] == 0 and summary["candidate_suspicious_chain_count"] == 0
	return report


def summarize_report(results: list[dict[str, Any]]) -> dict[str, Any]:
	mismatch_examples: list[dict[str, Any]] = []
	baseline_outcomes: Counter[str] = Counter()
	candidate_outcomes: Counter[str] = Counter()
	policy_counts: dict[str, Counter[str]] = defaultdict(Counter)
	candidate_suspicious: list[dict[str, Any]] = []
	baseline_suspicious: list[dict[str, Any]] = []
	baseline_suspicious_count = 0
	candidate_suspicious_count = 0

	for entry in results:
		seed = entry["seed"]
		policy = entry["policy"]
		baseline = entry["baseline"]
		candidate = entry["candidate"]
		mismatches = entry["mismatches"]

		baseline_outcomes[str(baseline["outcome"])] += 1
		candidate_outcomes[str(candidate["outcome"])] += 1
		if mismatches:
			policy_counts[policy]["mismatches"] += 1
			if len(mismatch_examples) < 20:
				mismatch_examples.append(
					{
						"seed": seed,
						"policy": policy,
						"mismatches": mismatches,
						"baseline_last_state": baseline["last_state"],
						"candidate_last_state": candidate["last_state"],
					}
				)
		for chain in baseline["chains"]:
			if chain["classification"] != "healthy":
				baseline_suspicious_count += 1
				if len(baseline_suspicious) < 20:
					baseline_suspicious.append({"seed": seed, "policy": policy, "chain": chain})
		for chain in candidate["chains"]:
			if chain["classification"] != "healthy":
				candidate_suspicious_count += 1
				if len(candidate_suspicious) < 20:
					candidate_suspicious.append({"seed": seed, "policy": policy, "chain": chain})

	return {
		"baseline_outcomes": dict(sorted(baseline_outcomes.items())),
		"candidate_outcomes": dict(sorted(candidate_outcomes.items())),
		"mismatch_count": sum(policy_counter["mismatches"] for policy_counter in policy_counts.values()),
		"policy_counts": {policy: dict(counter) for policy, counter in sorted(policy_counts.items())},
		"baseline_suspicious_chain_count": baseline_suspicious_count,
		"candidate_suspicious_chain_count": candidate_suspicious_count,
		"mismatch_examples": mismatch_examples,
		"baseline_suspicious_examples": baseline_suspicious,
		"candidate_suspicious_examples": candidate_suspicious,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Audit reward-loop semantics across full-run backends.")
	parser.add_argument("--baseline-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="godot-http")
	parser.add_argument("--candidate-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="headless-pipe")
	parser.add_argument("--baseline-port", type=int, default=DEFAULT_PORT + 120)
	parser.add_argument("--candidate-port", type=int, default=DEFAULT_PORT + 121)
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
	parser.add_argument("--max-steps", type=int, default=800)
	parser.add_argument("--report-json", type=Path, default=None)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	policies = args.policy or ["coverage", "exit", "training"]
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
		max_steps=args.max_steps,
	)
	print_json_report(report, args.report_json)
	return 0 if report["pass"] else 1


if __name__ == "__main__":
	raise SystemExit(main())
