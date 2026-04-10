from __future__ import annotations

import _path_init  # noqa: F401

import json
import random
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).parent))

from sts2ai_paths import REPO_ROOT
from headless_sim_runner import stop_process
from test_simulator_consistency import (
	BOSS_PARITY_SEEDS,
	COVERAGE_SEEDS,
	DEFAULT_GODOT_EXE,
	DEFAULT_HEADLESS_DLL,
	DEFAULT_PARITY_SEEDS,
	DEFAULT_PORT,
	STRESS_SEEDS,
	create_client,
	launch_backend,
	pick_coverage_action,
	pick_deterministic_action,
	pick_exit_hunting_action,
)


REWARD_STATE_TYPES = {"combat_rewards", "card_reward"}
COMBAT_STATE_TYPES = {"monster", "elite", "boss", "hand_select", "combat_pending"}
DEFAULT_REPO_ROOT = REPO_ROOT


def build_seed_list(
	*,
	explicit_seeds: list[str] | None,
	seed_prefix: str,
	start_index: int,
	count: int,
	include_default: bool,
) -> list[str]:
	seeds: list[str] = []
	normalized_prefix = seed_prefix.rstrip("_")
	if include_default:
		seeds.extend(DEFAULT_PARITY_SEEDS)
		seeds.extend(BOSS_PARITY_SEEDS)
		seeds.extend(COVERAGE_SEEDS)
		seeds.extend(STRESS_SEEDS)
	if explicit_seeds:
		seeds.extend(explicit_seeds)
	seeds.extend(f"{normalized_prefix}_{index:04d}" for index in range(start_index, start_index + count))
	ordered: list[str] = []
	seen: set[str] = set()
	for seed in seeds:
		if seed and seed not in seen:
			seen.add(seed)
			ordered.append(seed)
	return ordered


def count_legal_actions(state: dict[str, Any], action_name: str) -> int:
	return sum(
		1
		for action in (state.get("legal_actions") or [])
		if (action.get("action") or "").lower() == action_name and bool(action.get("is_enabled", True))
	)


def choose_policy_action(policy: str, state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
	state_type = str(state.get("state_type") or "").lower()
	legal = [action for action in state.get("legal_actions") or [] if bool(action.get("is_enabled", True))]
	if not legal:
		return {"action": "wait"}

	if policy == "coverage":
		return pick_coverage_action(state, rng)
	if policy == "exit":
		return pick_exit_hunting_action(state, rng)
	if policy == "deterministic":
		return pick_deterministic_action(state_type, legal, rng)
	if policy != "training":
		raise ValueError(f"Unsupported policy '{policy}'.")

	selection_actions = {
		"select_card",
		"combat_select_card",
		"confirm_selection",
		"combat_confirm_selection",
		"cancel_selection",
		"skip_relic_selection",
	}
	legal_action_names = {(action.get("action") or "").lower() for action in legal}
	if state_type in {"card_select", "hand_select", "relic_select"} or legal_action_names & selection_actions:
		confirm = next(
			(
				action
				for action in legal
				if (action.get("action") or "").lower()
				in {"confirm_selection", "combat_confirm_selection", "skip_relic_selection", "cancel_selection"}
			),
			None,
		)
		if confirm is not None:
			return confirm
		select = next(
			(
				action
				for action in legal
				if "select" in (action.get("action") or "").lower()
			),
			None,
		)
		if select is not None:
			return select

	if state_type == "combat_rewards":
		claim = next((action for action in legal if (action.get("action") or "").lower() == "claim_reward"), None)
		if claim is not None:
			return claim
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if proceed is not None:
			return proceed

	if state_type == "card_reward":
		select = next((action for action in legal if (action.get("action") or "").lower() == "select_card_reward"), None)
		if select is not None:
			return select
		skip = next((action for action in legal if (action.get("action") or "").lower() == "skip_card_reward"), None)
		if skip is not None:
			return skip

	if state_type == "treasure":
		claim = next((action for action in legal if (action.get("action") or "").lower() == "claim_treasure_relic"), None)
		if claim is not None:
			return claim
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if proceed is not None:
			return proceed

	if state_type == "shop":
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if proceed is not None:
			return proceed

	return pick_deterministic_action(state_type, legal, rng)


def compact_state_record(state: dict[str, Any], *, step: int) -> dict[str, Any]:
	run = state.get("run") or {}
	return {
		"step": step,
		"state_type": state.get("state_type"),
		"terminal": bool(state.get("terminal", False)),
		"run_outcome": state.get("run_outcome"),
		"floor": run.get("floor"),
		"act": run.get("act"),
		"claim_reward_count": count_legal_actions(state, "claim_reward"),
		"select_card_reward_count": count_legal_actions(state, "select_card_reward"),
		"proceed_count": count_legal_actions(state, "proceed"),
		"skip_card_reward_count": count_legal_actions(state, "skip_card_reward"),
	}


def classify_capped_reason(record: dict[str, Any], *, boss_reached: bool) -> str:
	state_type = str(record.get("state_type") or "").lower()
	if boss_reached or state_type == "boss":
		return "boss"
	if state_type in REWARD_STATE_TYPES:
		return "reward"
	if state_type in COMBAT_STATE_TYPES:
		return "combat"
	if not state_type:
		return "unknown"
	return state_type


def _kill_stale_headless_processes(port: int) -> None:
	if sys.platform != "win32":
		return
	command = (
		"$procs = Get-CimInstance Win32_Process | "
		f"Where-Object {{ $_.Name -eq 'dotnet.exe' -and $_.CommandLine -like '*HeadlessSim.dll*' -and $_.CommandLine -like '*--port {port}*' }}; "
		"foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force }"
	)
	try:
		subprocess.run(
			["powershell", "-NoProfile", "-Command", command],
			check=False,
			capture_output=True,
			text=True,
		)
	except Exception:
		pass


@contextmanager
def backend_client(
	*,
	backend: str,
	port: int,
	auto_launch: bool,
	repo_root: Path = DEFAULT_REPO_ROOT,
	godot_exe: Path = DEFAULT_GODOT_EXE,
	headless_dll: Path = DEFAULT_HEADLESS_DLL,
) -> Iterator[Any]:
	proc = None
	client = None
	try:
		if auto_launch:
			if backend.startswith("headless-"):
				_kill_stale_headless_processes(port)
			proc = launch_backend(
				backend=backend,
				port=port,
				repo_root=repo_root,
				godot_exe=godot_exe,
				headless_dll=headless_dll,
			)
		client = create_client(backend, port)
		yield client
	finally:
		if client is not None:
			try:
				client.close()
			except Exception:
				pass
		stop_process(proc)


def print_json_report(report: dict[str, Any], report_json: Path | None) -> None:
	if report_json is not None:
		report_json = report_json.resolve()
		report_json.parent.mkdir(parents=True, exist_ok=True)
		report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print(json.dumps(report, indent=2))
