from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/{core,ipc,search} to sys.path)

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from full_run_env import ApiBackedFullRunClient, BinaryBackedFullRunClient, FullRunClientLike, PipeBackedFullRunClient
from headless_sim_runner import DEFAULT_DLL_PATH, start_headless_sim, stop_process


DEFAULT_PORT = 15527
DEFAULT_REPO_ROOT = Path(r"D:/dev/ai-slay-sts2/sts2")
DEFAULT_GODOT_EXE = Path(r"D:/dev/Godot_v4.5.1-stable_mono_win64/Godot_v4.5.1-stable_mono_win64_console.exe")
DEFAULT_HEADLESS_DLL = DEFAULT_DLL_PATH
TEST_SEEDS = ["CONSIST_A", "CONSIST_B", "CONSIST_C", "CONSIST_D", "CONSIST_E"]
REWARD_PARITY_SEEDS = ["CARD_SCAN_24", "CARD_SCAN_169"]
BOSS_PARITY_SEEDS = ["COVERAGE_SCAN_0210"]
NONCOMBAT_PARITY_SEEDS = ["SHOP_EXIT_000", "SHOP_EXIT_001", "SHOP_EXIT_002", "SHOP_EXIT_005", "SHOP_EXIT_009", "SHOP_EXIT_010", "SHOP_EXIT_011", "SHOP_EXIT_014"]
DEFAULT_PARITY_SEEDS = TEST_SEEDS + REWARD_PARITY_SEEDS + BOSS_PARITY_SEEDS + NONCOMBAT_PARITY_SEEDS
DISCOVERED_COVERAGE_SEEDS = ["COVERAGE_SCAN_0040", "COVERAGE_SCAN_0210"]
COVERAGE_SEEDS = DEFAULT_PARITY_SEEDS + DISCOVERED_COVERAGE_SEEDS
STRESS_SEEDS = ["BENCH_A", "BENCH_B", "BENCH_C"]
EXACT_SAVELOAD_STATES = {"monster", "elite", "boss", "combat_rewards", "card_reward", "map", "event", "rest_site", "shop", "treasure", "game_over"}
AUDIT_STATES = ["map", "event", "rest_site", "shop", "treasure", "monster", "elite", "boss", "combat_pending", "combat_rewards", "card_reward", "card_select", "relic_select", "game_over"]
TRACKED_TRANSITIONS = [("event", "card_select"), ("event", "relic_select"), ("event", "monster"), ("event", "elite"), ("event", "boss"), ("combat_pending", "combat_rewards"), ("combat_pending", "map"), ("combat_pending", "game_over"), ("combat_rewards", "card_reward"), ("combat_rewards", "map"), ("combat_rewards", "event"), ("combat_rewards", "game_over"), ("treasure", "relic_select")]
REQUIRED_AUDIT_TRANSITIONS = [("event", "card_select"), ("event", "monster"), ("combat_pending", "map"), ("combat_rewards", "card_reward")]
STATIC_AUDIT_NOTES = {
	"relic_select": "No non-simulation call sites for RelicSelectCmd.FromChooseARelicScreen/GetSelectedRelicAsync found in src/Core.",
	("event", "relic_select"): "Current event paths use direct relic obtain/reward flows; they do not route through RelicSelectCmd or NChooseARelicSelection.",
	("treasure", "relic_select"): "TreasureRoom uses TreasureRoomRelicSynchronizer voting/current relic state instead of RelicSelectCmd or relic_select screen state.",
	("event", "elite"): "All current *EventEncounter classes in src/Core/Models/Encounters declare RoomType.Monster.",
	("event", "boss"): "All current *EventEncounter classes in src/Core/Models/Encounters declare RoomType.Monster.",
}
UNPRODUCED_AUDIT_STATES = {"relic_select"}
DISCOVERY_REQUIRED_STATE_TARGETS = ["boss"]
DISCOVERY_OPTIONAL_STATE_TARGETS = ["relic_select"]
DISCOVERY_REQUIRED_TRANSITION_TARGETS: list[tuple[str, str]] = []
DISCOVERY_OPTIONAL_TRANSITION_TARGETS = [("event", "relic_select"), ("treasure", "relic_select"), ("event", "elite"), ("event", "boss")]
DISCOVERY_SEED_PREFIX = "COVERAGE_SCAN"
MAX_STEPS_PER_RUN = 800
COMBAT_TYPES = {"monster", "elite", "boss"}
COMBAT_DETAIL_TYPES = COMBAT_TYPES | {"hand_select"}
VALID_STATE_TYPES = {"menu", "map", "monster", "elite", "boss", "event", "rest_site", "shop", "treasure", "combat_rewards", "card_reward", "card_select", "relic_select", "hand_select", "combat_pending", "game_over", "run_bootstrap"}
VALID_TRANSITIONS = {
	"menu": {"map", "run_bootstrap"},
	"run_bootstrap": {"map"},
	"map": COMBAT_TYPES | {"event", "rest_site", "shop", "treasure"},
	"monster": {"monster", "combat_pending", "combat_rewards", "game_over", "hand_select", "card_select"},
	"elite": {"elite", "combat_pending", "combat_rewards", "game_over", "hand_select", "card_select"},
	"boss": {"boss", "combat_pending", "combat_rewards", "game_over", "hand_select", "card_select"},
	"hand_select": COMBAT_TYPES | {"combat_pending", "combat_rewards", "game_over"},
	"card_select": COMBAT_TYPES | {"combat_pending", "combat_rewards", "game_over", "card_select", "map"},
	"combat_pending": {"combat_rewards", "map", "game_over"} | COMBAT_TYPES,
	"combat_rewards": {"combat_rewards", "card_reward", "card_select", "relic_select", "map", "game_over", "event"},
	"card_reward": {"combat_rewards", "map", "card_select", "combat_pending"},
	"relic_select": {"combat_rewards", "map", "relic_select"},
	"event": {"event", "map", "combat_rewards", "card_reward", "card_select", "relic_select", "game_over", "rest_site", "shop", "treasure"} | COMBAT_TYPES,
	"rest_site": {"map", "card_select", "rest_site"},
	"shop": {"shop", "map", "card_select"},
	"treasure": {"treasure", "map", "relic_select"},
	"game_over": set(),
}


def _deep_copy_jsonable(value: Any) -> Any:
	return json.loads(json.dumps(value, ensure_ascii=True, default=str))


def _safe_perf_stats(client: FullRunClientLike) -> dict[str, Any]:
	try:
		stats = client.perf_stats()
		return stats if isinstance(stats, dict) else {}
	except Exception as exc:
		return {"error": str(exc)}


def _report_transition_key(transition: tuple[str, str]) -> str:
	return f"{transition[0]}->{transition[1]}"


class Violation:
	def __init__(self, step: int, category: str, message: str, state_type: str = "?"):
		self.step = step
		self.category = category
		self.message = message
		self.state_type = state_type

	def __str__(self) -> str:
		return f"  [Step {self.step:4d}] [{self.category}] ({self.state_type}) {self.message}"


def create_client(backend: str, port: int) -> FullRunClientLike:
	if backend == "godot-http":
		return ApiBackedFullRunClient(base_url=f"http://127.0.0.1:{port}", request_timeout_s=30.0, ready_timeout_s=30.0)
	if backend == "headless-binary":
		return BinaryBackedFullRunClient(port=port, connect_timeout_s=15.0)
	return PipeBackedFullRunClient(port=port, connect_timeout_s=15.0)


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


def launch_backend(*, backend: str, port: int, repo_root: Path, godot_exe: Path, headless_dll: Path) -> subprocess.Popen | None:
	if backend == "headless-pipe":
		return start_headless_sim(port=port, repo_root=repo_root, dll_path=headless_dll)
	if backend == "headless-binary":
		return start_headless_sim(port=port, repo_root=repo_root, dll_path=headless_dll, protocol="bin")
	if backend in {"godot-http", "godot-pipe"}:
		proc = subprocess.Popen(
			[str(godot_exe), *_godot_full_run_server_args(repo_root=repo_root, port=port)],
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
			creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
		)
		time.sleep(12)
		return proc
	return None


def _kill_stale_headless_processes(port: int) -> None:
	try:
		import psutil  # type: ignore
	except Exception:
		return
	pipe_name = f"sts2_mcts_pipe_{port}"
	binary_pipe_name = f"sts2_mcts_bin_{port}"
	for proc in psutil.process_iter(["pid", "name", "cmdline"]):
		try:
			cmdline = " ".join(proc.info.get("cmdline") or [])
		except Exception:
			continue
		if "HeadlessSim.dll" not in cmdline:
			continue
		if pipe_name not in cmdline and binary_pipe_name not in cmdline and f"--port {port}" not in cmdline:
			continue
		try:
			proc.terminate()
			proc.wait(timeout=5)
		except Exception:
			try:
				proc.kill()
			except Exception:
				pass


def normalize_legal_action(action: dict[str, Any]) -> tuple[Any, ...]:
	return (action.get("action"), action.get("index"), action.get("card_index"), action.get("target_id"), action.get("col"), action.get("row"), action.get("slot"), bool(action.get("is_enabled", True)))


def extract_player(state: dict[str, Any]) -> dict[str, Any]:
	battle = state.get("battle") or {}
	player = battle.get("player") or {}
	if player:
		return player
	for key in ("map", "event", "rest_site", "shop", "treasure", "rewards", "card_reward", "card_select", "relic_select"):
		player = ((state.get(key) or {}).get("player") or {})
		if player:
			return player
	return state.get("player") or {}


def _normalize_cost_value(value: Any) -> Any:
	if value is None:
		return None
	text = str(value).strip()
	if text == "":
		return None
	if text.upper() == "X":
		return "X"
	try:
		num = int(text)
		return None if num < 0 else num
	except ValueError:
		return text


def _normalize_bool_flag(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if value is None:
		return False
	text = str(value).strip().lower()
	if text in {"true", "1", "yes"}:
		return True
	if text in {"false", "0", "no", ""}:
		return False
	return bool(value)


def _normalize_card_record(card: dict[str, Any], *, include_index: bool) -> tuple[Any, ...]:
	record = []
	if include_index:
		record.append(card.get("index"))
	record.extend(
		[
			card.get("id"),
			str(card.get("type") or "").lower() or None,
			_normalize_cost_value(card.get("cost")),
			_normalize_bool_flag(card.get("is_upgraded")),
			card.get("target_type"),
			_normalize_bool_flag(card.get("can_play", True)),
		]
	)
	return tuple(record)


def _normalize_deck_card(card: dict[str, Any]) -> tuple[Any, ...]:
	return (
		card.get("id"),
		str(card.get("type") or "").lower() or None,
		str(card.get("rarity") or "").lower() or None,
		_normalize_bool_flag(card.get("is_upgraded")),
	)


def _normalize_relic_record(relic: dict[str, Any]) -> tuple[Any, ...]:
	counter = relic.get("counter")
	if counter in (0, "0", "", None):
		counter = None
	return (
		relic.get("id"),
		counter,
	)


def _normalize_potion_record(potion: dict[str, Any]) -> tuple[Any, ...]:
	if not isinstance(potion, dict):
		return (None,)
	return (
		potion.get("id"),
	)


def _normalize_enemy_record(enemy: dict[str, Any]) -> tuple[Any, ...]:
	intents = tuple(str((intent or {}).get("type") or "").lower() for intent in (enemy.get("intents") or []))
	status_names = tuple(sorted(str((status or {}).get("id") or (status or {}).get("name") or "") for status in (enemy.get("status") or [])))
	raw_is_alive = enemy.get("is_alive")
	if raw_is_alive is None:
		hp_value = enemy.get("hp")
		if hp_value is None:
			hp_value = enemy.get("current_hp")
		raw_is_alive = (hp_value or 0) > 0
	return (
		enemy.get("combat_id"),
		enemy.get("hp"),
		enemy.get("max_hp"),
		enemy.get("block"),
		_normalize_bool_flag(raw_is_alive),
		intents,
		status_names,
	)


def _normalize_status_record(status: dict[str, Any]) -> tuple[Any, ...]:
	if not isinstance(status, dict):
		return (None, None, None)
	amount = status.get("amount")
	if amount is None:
		amount = status.get("count")
	if amount is None:
		amount = status.get("value")
	return (
		status.get("id") or status.get("name"),
		amount,
		_normalize_bool_flag(status.get("is_negative")),
	)


def _player_scalar_summary(player: dict[str, Any], *, state_type: str) -> dict[str, Any]:
	scalars: dict[str, Any] = {
		"hp": player.get("hp") if player.get("hp") is not None else player.get("current_hp"),
		"max_hp": player.get("max_hp"),
		"gold": player.get("gold"),
		"open_potion_slots": player.get("open_potion_slots"),
	}
	if state_type in COMBAT_DETAIL_TYPES:
		scalars.update(
			{
				"block": player.get("block"),
				"energy": player.get("energy"),
				"max_energy": player.get("max_energy"),
				"draw_pile_count": player.get("draw_pile_count"),
				"discard_pile_count": player.get("discard_pile_count"),
				"exhaust_pile_count": player.get("exhaust_pile_count"),
			}
		)
	return scalars


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
	run = state.get("run") or {}
	battle = state.get("battle") or {}
	state_type = str(state.get("state_type") or "")
	player = extract_player(state)
	enemies = [_normalize_enemy_record(enemy) for enemy in battle.get("enemies") or []]
	hand_cards = [_normalize_card_record(card, include_index=True) for card in player.get("hand") or []]
	deck_cards = [_normalize_deck_card(card) for card in player.get("deck") or []]
	relics = [_normalize_relic_record(relic) for relic in player.get("relics") or []]
	potions = [_normalize_potion_record(potion) for potion in player.get("potions") or []]
	player_statuses = [_normalize_status_record(status) for status in player.get("status") or []]
	rewards = state.get("rewards") or {}
	card_reward = state.get("card_reward") or {}
	event = state.get("event") or {}
	shop = state.get("shop") or {}
	treasure = state.get("treasure") or {}
	card_select = state.get("card_select") or {}
	relic_select = state.get("relic_select") or {}
	return {
		"state_type": state_type,
		"terminal": state.get("terminal"),
		"run_outcome": state.get("run_outcome"),
		"floor": run.get("floor"),
		"act": run.get("act"),
		"player": _player_scalar_summary(player, state_type=state_type),
		"deck_cards": deck_cards,
		"relics": relics,
		"potions": potions,
		"player_statuses": player_statuses,
		"hand_cards": hand_cards,
		"enemy_count": len(enemies),
		"enemies": enemies,
		"legal_actions": [normalize_legal_action(action) for action in state.get("legal_actions") or []],
		"event_payload": {"present": bool(event), "event_id": event.get("event_id"), "in_dialogue": event.get("in_dialogue"), "is_finished": event.get("is_finished"), "options": [(item.get("index"), item.get("is_locked"), item.get("is_chosen"), item.get("is_proceed"), item.get("text") or item.get("label")) for item in event.get("options") or []]},
		"shop_payload": {"present": bool(shop), "is_open": shop.get("is_open"), "can_proceed": shop.get("can_proceed"), "items": [(item.get("index"), item.get("category"), item.get("cost"), item.get("can_afford"), item.get("is_stocked"), item.get("on_sale"), item.get("card_id"), item.get("relic_id"), item.get("potion_id"), item.get("name")) for item in shop.get("items") or []]},
		"treasure_payload": {"present": bool(treasure), "can_proceed": treasure.get("can_proceed"), "relics": [(item.get("index"), item.get("id"), item.get("name"), item.get("rarity")) for item in treasure.get("relics") or []]},
		"reward_payload": {
			"present": bool(rewards),
			"can_proceed": rewards.get("can_proceed"),
			"items": [
				(
					item.get("index"),
					item.get("type"),
					item.get("label"),
					item.get("reward_key"),
					item.get("reward_source"),
					item.get("claimable"),
					item.get("claim_block_reason"),
				)
				for item in rewards.get("items") or []
			],
		},
		"card_reward_payload": {"present": bool(card_reward), "can_skip": card_reward.get("can_skip"), "cards": [(item.get("index"), item.get("id"), item.get("name"), item.get("rarity"), item.get("type"), item.get("cost"), item.get("is_upgraded")) for item in card_reward.get("cards") or []]},
		"card_select_payload": {"present": bool(card_select), "screen_type": card_select.get("screen_type"), "selected_count": card_select.get("selected_count"), "can_confirm": card_select.get("can_confirm"), "can_cancel": card_select.get("can_cancel"), "cards": [(item.get("index"), item.get("id"), item.get("name"), item.get("type"), item.get("rarity"), item.get("cost"), item.get("is_upgraded"), item.get("is_selected")) for item in card_select.get("cards") or []], "selected_cards": [(item.get("index"), item.get("id"), item.get("name"), item.get("type"), item.get("rarity"), item.get("cost"), item.get("is_upgraded"), item.get("is_selected")) for item in card_select.get("selected_cards") or []]},
		"relic_select_payload": {"present": bool(relic_select), "can_skip": relic_select.get("can_skip"), "relics": [(item.get("index"), item.get("id"), item.get("name"), item.get("rarity")) for item in relic_select.get("relics") or []]},
	}


def state_signature(state: dict[str, Any]) -> str:
	return hashlib.md5(json.dumps(state_summary(state), sort_keys=True, default=str).encode()).hexdigest()


def _count_legal_actions(summary: dict[str, Any], action_name: str) -> int:
	return sum(1 for action in (summary.get("legal_actions") or []) if action and action[0] == action_name)


def _normalize_event_payload(payload: dict[str, Any], summary: dict[str, Any] | None = None, *, full: bool = False) -> dict[str, Any]:
	event_id = str(payload.get("event_id") or "")
	if "." in event_id:
		event_id = event_id.split(".")[-1]
	options = payload.get("options") or []
	enabled_option_count = sum(1 for item in options if not bool(item[1]))
	if summary is not None:
		enabled_option_count = max(enabled_option_count, _count_legal_actions(summary, "choose_event_option"))
	result = {
		"present": bool(payload.get("present")),
		"event_id": event_id,
		"in_dialogue": bool(payload.get("in_dialogue", False)),
		"option_count": enabled_option_count,
	}
	if full:
		# Public trace parity should focus on actionable event semantics, not
		# locked placeholder rows whose UI realization can differ between
		# Godot-backed screens and headless shims.
		result["options"] = [
			(
				item[0],
				bool(item[3]),
			)
			for item in options
			if not bool(item[1])
		]
	return result


def _normalize_shop_payload(payload: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
	def normalize_category(value: Any) -> Any:
		text = str(value or "").strip().lower()
		if text in {"card_removal", "remove_card"}:
			return "remove_card"
		return text

	def normalize_symbol(value: Any) -> Any:
		text = str(value or "").strip()
		return text or None

	result = {
		"present": bool(payload.get("present")),
		"items": [
			(
				item[0],
				normalize_category(item[1]),
				item[2],
				item[3],
				item[4],
				bool(item[5]),
				normalize_symbol(item[6]),
				normalize_symbol(item[7]),
				normalize_symbol(item[8]),
			)
			for item in (payload.get("items") or [])
		],
	}
	if full:
		result["is_open"] = bool(payload.get("is_open", False))
		result["can_proceed"] = bool(payload.get("can_proceed", False))
	return result


def _normalize_treasure_payload(payload: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
	relics = []
	for item in (payload.get("relics") or []):
		record = [item[0], item[1]]
		if full:
			record.append(str(item[3] or "").lower() or None)
		relics.append(tuple(record))
	return {
		"present": bool(payload.get("present")),
		"relics": relics,
	}


def _normalize_card_reward_payload(payload: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
	return {
		"present": bool(payload.get("present")),
		"can_skip": bool(payload.get("can_skip", False)),
		"cards": [
			(
				item[0],
				item[1],
				str(item[3] or "").lower(),
				str(item[4] or "").lower(),
				item[5],
				bool(item[6]),
			)
			for item in (payload.get("cards") or [])
		],
	}


def _normalize_reward_payload(payload: dict[str, Any], summary: dict[str, Any] | None = None, *, full: bool = False) -> dict[str, Any]:
	items = []
	for item in (payload.get("items") or []):
		reward_type = item[1]
		label = item[2]
		reward_key = item[3] if len(item) > 3 else None
		reward_source = item[4] if len(item) > 4 else None
		claimable = _normalize_bool_flag(item[5]) if len(item) > 5 else True
		claim_block_reason = item[6] if len(item) > 6 else None
		if reward_type in {"card", "special_card", "remove_card"}:
			label = None
		elif label is not None:
			label = str(label).strip()
			if label.endswith(".title"):
				label = label[:-6]
			label = label or None
		if reward_key:
			reward_key = str(reward_key).strip().lower()
			parts = reward_key.split("|", 2)
			reward_key = "|".join(parts[:2]) if parts else None
		record: list[Any] = [item[0], reward_type]
		if reward_key:
			record.append(reward_key)
		else:
			record.append(None)
		record.extend([reward_source, claimable, claim_block_reason if claim_block_reason else None])
		if full:
			record.append(label)
		items.append(tuple(record))
	claim_count = len(items)
	can_proceed = bool(payload.get("can_proceed", False))
	if summary is not None:
		claim_count = max(claim_count, _count_legal_actions(summary, "claim_reward"))
		can_proceed = can_proceed or _count_legal_actions(summary, "proceed") > 0
	return {
		"present": bool(payload.get("present")) or claim_count > 0 or can_proceed,
		"can_proceed": can_proceed,
		"claim_count": claim_count,
		"items": items,
	}


def _normalize_card_select_payload(payload: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
	def normalize_card_tuple(item: Any) -> tuple[Any, ...]:
		record = [item[0], item[1]]
		if full:
			record.extend(
				[
					str(item[3] or "").lower() or None,
					str(item[4] or "").lower() or None,
					_normalize_cost_value(item[5]),
					_normalize_bool_flag(item[6]),
					_normalize_bool_flag(item[7]),
				]
			)
		return tuple(record)

	return {
		"present": bool(payload.get("present")),
		"screen_type": payload.get("screen_type"),
		"selected_count": payload.get("selected_count"),
		"can_confirm": bool(payload.get("can_confirm", False)),
		"can_cancel": bool(payload.get("can_cancel", False)),
		"cards": [normalize_card_tuple(item) for item in (payload.get("cards") or [])],
		"selected_cards": [normalize_card_tuple(item) for item in (payload.get("selected_cards") or [])],
	}


def _normalize_relic_select_payload(payload: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
	relics = []
	for item in (payload.get("relics") or []):
		record = [item[0], item[1]]
		if full:
			record.append(str(item[3] or "").lower() or None)
		relics.append(tuple(record))
	return {
		"present": bool(payload.get("present")),
		"can_skip": bool(payload.get("can_skip", False)),
		"relics": relics,
	}


def _normalize_event_display_payload(payload: dict[str, Any]) -> dict[str, Any]:
	options = []
	for item in (payload.get("options") or []):
		text = str(item[4] if len(item) > 4 else "" or "").strip() or None
		options.append((item[0], text))
	return {
		"present": bool(payload.get("present")),
		"event_id": str(payload.get("event_id") or "").split(".")[-1] or None,
		"options": options,
	}


def _normalize_shop_display_payload(payload: dict[str, Any]) -> dict[str, Any]:
	return {
		"present": bool(payload.get("present")),
		"items": [
			(
				item[0],
				str(item[1] or "").lower() or None,
				str(item[9] or "").strip() or None,
			)
			for item in (payload.get("items") or [])
		],
	}


def _normalize_treasure_display_payload(payload: dict[str, Any]) -> dict[str, Any]:
	return {
		"present": bool(payload.get("present")),
		"relics": [
			(
				item[0],
				item[1],
				str(item[2] or "").strip() or None,
			)
			for item in (payload.get("relics") or [])
		],
	}


def _normalize_reward_display_payload(payload: dict[str, Any]) -> dict[str, Any]:
	items = []
	for item in (payload.get("items") or []):
		label = str(item[2] or "").strip() or None
		items.append((item[0], str(item[1] or "").lower() or None, label))
	return {
		"present": bool(payload.get("present")),
		"items": items,
	}


def _normalize_card_reward_display_payload(payload: dict[str, Any]) -> dict[str, Any]:
	return {
		"present": bool(payload.get("present")),
		"cards": [
			(
				item[0],
				item[1],
				str(item[2] or "").strip() or None,
			)
			for item in (payload.get("cards") or [])
		],
	}


def _normalize_card_select_display_payload(payload: dict[str, Any]) -> dict[str, Any]:
	def normalize_cards(items: list[Any]) -> list[tuple[Any, ...]]:
		return [
			(
				item[0],
				item[1],
				str(item[2] or "").strip() or None,
				bool(item[7]) if len(item) > 7 else False,
			)
			for item in items
		]

	return {
		"present": bool(payload.get("present")),
		"screen_type": payload.get("screen_type"),
		"cards": normalize_cards(payload.get("cards") or []),
		"selected_cards": normalize_cards(payload.get("selected_cards") or []),
	}


def _normalize_relic_select_display_payload(payload: dict[str, Any]) -> dict[str, Any]:
	return {
		"present": bool(payload.get("present")),
		"relics": [
			(
				item[0],
				item[1],
				str(item[2] or "").strip() or None,
			)
			for item in (payload.get("relics") or [])
		],
	}


def compare_display_payloads(baseline_summary: dict[str, Any], candidate_summary: dict[str, Any]) -> list[str]:
	diffs: list[str] = []
	state_type = str(baseline_summary.get("state_type") or "")
	for key in ("state_type", "floor", "act"):
		if baseline_summary.get(key) != candidate_summary.get(key):
			diffs.append(f"{key}: {baseline_summary.get(key)!r} != {candidate_summary.get(key)!r}")
	for key in _active_payload_keys(state_type):
		left = baseline_summary.get(key) or {}
		right = candidate_summary.get(key) or {}
		if key == "event_payload":
			left = _normalize_event_display_payload(left)
			right = _normalize_event_display_payload(right)
		elif key == "shop_payload":
			left = _normalize_shop_display_payload(left)
			right = _normalize_shop_display_payload(right)
		elif key == "treasure_payload":
			left = _normalize_treasure_display_payload(left)
			right = _normalize_treasure_display_payload(right)
		elif key == "reward_payload":
			left = _normalize_reward_display_payload(left)
			right = _normalize_reward_display_payload(right)
		elif key == "card_reward_payload":
			left = _normalize_card_reward_display_payload(left)
			right = _normalize_card_reward_display_payload(right)
		elif key == "card_select_payload":
			left = _normalize_card_select_display_payload(left)
			right = _normalize_card_select_display_payload(right)
		elif key == "relic_select_payload":
			left = _normalize_relic_select_display_payload(left)
			right = _normalize_relic_select_display_payload(right)
		if left.get("present") and right.get("present") and left != right:
			diffs.append(f"display.{key}: {left!r} != {right!r}")
	return diffs


def _active_payload_keys(state_type: str) -> list[str]:
	return {
		"event": ["event_payload"],
		"shop": ["shop_payload"],
		"treasure": ["treasure_payload"],
		"combat_rewards": ["reward_payload"],
		"card_reward": ["card_reward_payload"],
		"card_select": ["card_select_payload"],
		"relic_select": ["relic_select_payload"],
	}.get(state_type, [])


def compare_state_summaries(baseline_summary: dict[str, Any], candidate_summary: dict[str, Any], *, detail_level: str = "summary") -> list[str]:
	diffs: list[str] = []
	if detail_level == "display":
		return compare_display_payloads(baseline_summary, candidate_summary)
	state_type = str(baseline_summary.get("state_type") or "")
	for key in ("state_type", "terminal", "run_outcome", "floor", "act", "enemy_count", "legal_actions"):
		if baseline_summary.get(key) != candidate_summary.get(key):
			diffs.append(f"{key}: {baseline_summary.get(key)!r} != {candidate_summary.get(key)!r}")
	if state_type in COMBAT_DETAIL_TYPES and baseline_summary.get("enemies") != candidate_summary.get("enemies"):
		diffs.append(f"enemies: {baseline_summary.get('enemies')!r} != {candidate_summary.get('enemies')!r}")
	for key in _active_payload_keys(state_type):
		if key == "event_payload":
			left = _normalize_event_payload(baseline_summary.get(key) or {}, baseline_summary)
			right = _normalize_event_payload(candidate_summary.get(key) or {}, candidate_summary)
			if left != right:
				diffs.append(f"{key}: {left!r} != {right!r}")
			continue
		left = baseline_summary.get(key) or {}
		right = candidate_summary.get(key) or {}
		if key == "reward_payload":
			left = _normalize_reward_payload(left, baseline_summary)
			right = _normalize_reward_payload(right, candidate_summary)
			left_core = {name: left.get(name) for name in ("present", "can_proceed", "claim_count")}
			right_core = {name: right.get(name) for name in ("present", "can_proceed", "claim_count")}
			if left_core != right_core:
				diffs.append(f"{key}: {left_core!r} != {right_core!r}")
			elif left.get("items") and right.get("items") and left.get("items") != right.get("items"):
				diffs.append(f"{key}: {left!r} != {right!r}")
			continue
		if key == "shop_payload":
			left = _normalize_shop_payload(left)
			right = _normalize_shop_payload(right)
		elif key == "treasure_payload":
			left = _normalize_treasure_payload(left)
			right = _normalize_treasure_payload(right)
			left_relics = left.get("relics") or []
			right_relics = right.get("relics") or []
			if left_relics and right_relics and left_relics != right_relics:
				diffs.append(f"{key}: {left!r} != {right!r}")
			continue
		elif key == "card_reward_payload":
			left = _normalize_card_reward_payload(left)
			right = _normalize_card_reward_payload(right)
		elif key == "card_select_payload":
			left = _normalize_card_select_payload(left)
			right = _normalize_card_select_payload(right)
		elif key == "relic_select_payload":
			left = _normalize_relic_select_payload(left)
			right = _normalize_relic_select_payload(right)
			left_relics = left.get("relics") or []
			right_relics = right.get("relics") or []
			if left_relics and right_relics and left_relics != right_relics:
				diffs.append(f"{key}: {left!r} != {right!r}")
			continue
		if "present" in left or "present" in right:
			if not (left.get("present") and right.get("present")):
				continue
		if left != right:
			diffs.append(f"{key}: {left!r} != {right!r}")
	if state_type in COMBAT_DETAIL_TYPES and str(candidate_summary.get("state_type") or "") in COMBAT_DETAIL_TYPES:
		for key in ("hp", "max_hp", "block", "energy"):
			left = (baseline_summary.get("player") or {}).get(key)
			right = (candidate_summary.get("player") or {}).get(key)
			if left is None or right is None:
				continue
			if left != right:
				diffs.append(f"player.{key}: {left!r} != {right!r}")
	else:
		for key in ("hp", "max_hp", "gold", "open_potion_slots"):
			left = (baseline_summary.get("player") or {}).get(key)
			right = (candidate_summary.get("player") or {}).get(key)
			if left is None or right is None:
				continue
			if left != right:
				diffs.append(f"player.{key}: {left!r} != {right!r}")
	if detail_level in {"strict", "full"}:
		for key in ("deck_cards", "relics", "potions"):
			left = baseline_summary.get(key) or []
			right = candidate_summary.get(key) or []
			if left and right and left != right:
				diffs.append(f"{key}: {left!r} != {right!r}")
		if state_type in COMBAT_DETAIL_TYPES and baseline_summary.get("hand_cards") != candidate_summary.get("hand_cards"):
			diffs.append(f"hand_cards: {baseline_summary.get('hand_cards')!r} != {candidate_summary.get('hand_cards')!r}")
		if state_type in COMBAT_DETAIL_TYPES and str(candidate_summary.get("state_type") or "") in COMBAT_DETAIL_TYPES:
			for key in ("gold", "open_potion_slots", "max_energy", "draw_pile_count", "discard_pile_count", "exhaust_pile_count"):
				left = (baseline_summary.get("player") or {}).get(key)
				right = (candidate_summary.get("player") or {}).get(key)
				if left is None or right is None:
					continue
				if left != right:
					diffs.append(f"player.{key}: {left!r} != {right!r}")
	if detail_level == "full":
		for key in ("deck_cards", "relics", "potions", "player_statuses"):
			left = baseline_summary.get(key) or []
			right = candidate_summary.get(key) or []
			if left and right and left != right:
				diffs.append(f"{key}: {left!r} != {right!r}")
		if state_type in COMBAT_DETAIL_TYPES and baseline_summary.get("hand_cards") != candidate_summary.get("hand_cards"):
			diffs.append(f"hand_cards: {baseline_summary.get('hand_cards')!r} != {candidate_summary.get('hand_cards')!r}")
		for key in _active_payload_keys(state_type):
			left = baseline_summary.get(key) or {}
			right = candidate_summary.get(key) or {}
			if key == "event_payload":
				left = _normalize_event_payload(left, baseline_summary, full=True)
				right = _normalize_event_payload(right, candidate_summary, full=True)
			elif key == "reward_payload":
				left = _normalize_reward_payload(left, baseline_summary, full=True)
				right = _normalize_reward_payload(right, candidate_summary, full=True)
			elif key == "shop_payload":
				left = _normalize_shop_payload(left, full=True)
				right = _normalize_shop_payload(right, full=True)
			elif key == "treasure_payload":
				left = _normalize_treasure_payload(left, full=True)
				right = _normalize_treasure_payload(right, full=True)
			elif key == "card_reward_payload":
				left = _normalize_card_reward_payload(left, full=True)
				right = _normalize_card_reward_payload(right, full=True)
			elif key == "card_select_payload":
				left = _normalize_card_select_payload(left, full=True)
				right = _normalize_card_select_payload(right, full=True)
			elif key == "relic_select_payload":
				left = _normalize_relic_select_payload(left, full=True)
				right = _normalize_relic_select_payload(right, full=True)
			if left.get("present") and right.get("present") and left != right:
				diffs.append(f"{key}.full: {left!r} != {right!r}")
		for key in ("gold", "open_potion_slots", "max_energy", "draw_pile_count", "discard_pile_count", "exhaust_pile_count"):
			left = (baseline_summary.get("player") or {}).get(key)
			right = (candidate_summary.get("player") or {}).get(key)
			if left is None or right is None:
				continue
			if left != right:
				diffs.append(f"player.{key}: {left!r} != {right!r}")
	return diffs


def classify_summary_diffs(diffs: list[str]) -> str:
	if any(entry.startswith("display.") for entry in diffs):
		return "display_divergence"
	if any(entry.startswith(("state_type:", "floor:", "act:", "terminal:", "run_outcome:")) for entry in diffs):
		return "state_transition"
	if any(entry.startswith(("legal_actions:", "player.", "enemies:", "hand_cards:", "deck_cards:", "relics:", "potions:")) or "_payload:" in entry for entry in diffs):
		return "payload_divergence"
	return "representation_only"


def action_key(action: dict[str, Any]) -> tuple[Any, ...]:
	return (action.get("action") or action.get("type") or "", action.get("index", action.get("card_index", action.get("col", 0))) or 0, action.get("target_id", 0) or 0)


def pick_deterministic_action(state_type: str, legal: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
	if not legal:
		return {"action": "wait"}
	sorted_legal = sorted(legal, key=action_key)
	if state_type in COMBAT_TYPES:
		play_cards = [action for action in sorted_legal if (action.get("action") or "").lower() == "play_card"]
		end_turns = [action for action in sorted_legal if (action.get("action") or "").lower() == "end_turn"]
		if play_cards:
			return rng.choice(play_cards)
		if end_turns:
			return end_turns[0]
	if state_type == "card_select":
		for action in sorted_legal:
			if (action.get("action") or "").lower() in {"confirm_selection", "cancel_selection"}:
				return action
	return rng.choice(sorted_legal)


def pick_coverage_action(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
	state_type = str(state.get("state_type") or "")
	legal = [action for action in state.get("legal_actions") or [] if bool(action.get("is_enabled", True))]
	if not legal:
		return {"action": "wait"}
	if state_type == "map":
		options = sorted(
			(action for action in legal if (action.get("action") or "").lower() == "choose_map_node"),
			key=lambda action: (
				0 if "treasure" in str(action.get("label") or "").lower() else
				1 if "shop" in str(action.get("label") or "").lower() else
				2 if "rest" in str(action.get("label") or "").lower() else
				3 if "unknown" in str(action.get("label") or "").lower() or "event" in str(action.get("label") or "").lower() else
				4 if "monster" in str(action.get("label") or "").lower() else
				5 if "elite" in str(action.get("label") or "").lower() else
				6 if "boss" in str(action.get("label") or "").lower() else
				7,
				action.get("row") or 999,
				action.get("col") or 999,
			),
		)
		if options:
			return options[0]
	if state_type == "event":
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if proceed is not None:
			return proceed
		options = [action for action in legal if (action.get("action") or "").lower() == "choose_event_option"]
		if options:
			return sorted(options, key=lambda action: (-(action.get("index") or 0), action.get("label") or ""))[0]
		advance = next((action for action in legal if (action.get("action") or "").lower() == "advance_dialogue"), None)
		if advance is not None:
			return advance
	if state_type == "combat_rewards":
		reward_by_index = {item.get("index"): item for item in (state.get("rewards") or {}).get("items") or []}
		claims = [action for action in legal if (action.get("action") or "").lower() == "claim_reward"]
		if claims:
			def reward_priority(action: dict[str, Any]) -> tuple[int, int]:
				reward = reward_by_index.get(action.get("index")) or {}
				return (
					{"relic": 0, "card": 1, "gold": 2, "linked": 3, "potion": 4}.get(str(reward.get("type") or "").lower(), 5),
					action.get("index") or 999,
				)
			return sorted(claims, key=reward_priority)[0]
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if proceed is not None:
			return proceed
	if state_type == "card_reward":
		selects = [action for action in legal if (action.get("action") or "").lower() == "select_card_reward"]
		if selects:
			return sorted(selects, key=lambda action: action.get("index") or 999)[0]
		skip = next((action for action in legal if (action.get("action") or "").lower() == "skip_card_reward"), None)
		if skip is not None:
			return skip
	if state_type == "relic_select":
		selects = [action for action in legal if (action.get("action") or "").lower() == "select_relic"]
		if selects:
			return sorted(selects, key=lambda action: action.get("index") or 999)[0]
		skip = next((action for action in legal if (action.get("action") or "").lower() == "skip_relic_selection"), None)
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
	if state_type == "rest_site":
		rest = next((action for action in legal if (action.get("action") or "").lower() == "choose_rest_option"), None)
		if rest is not None:
			return rest
	if state_type == "card_select":
		confirm = next((action for action in legal if (action.get("action") or "").lower() == "confirm_selection"), None)
		if confirm is not None:
			return confirm
	return pick_deterministic_action(state_type, legal, rng)


def pick_exit_hunting_action(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
	state_type = str(state.get("state_type") or "")
	legal = [action for action in state.get("legal_actions") or [] if bool(action.get("is_enabled", True))]
	if not legal:
		return {"action": "wait"}
	if state_type == "map":
		options = sorted(
			(action for action in legal if (action.get("action") or "").lower() == "choose_map_node"),
			key=lambda action: (
				0 if "boss" in str(action.get("label") or "").lower() else
				1 if "elite" in str(action.get("label") or "").lower() else
				2 if "monster" in str(action.get("label") or "").lower() else
				3 if "unknown" in str(action.get("label") or "").lower() or "event" in str(action.get("label") or "").lower() else
				4 if "shop" in str(action.get("label") or "").lower() else
				5 if "treasure" in str(action.get("label") or "").lower() else
				6 if "rest" in str(action.get("label") or "").lower() else
				7,
				action.get("row") or 999,
				action.get("col") or 999,
			),
		)
		if options:
			return options[0]
	if state_type == "event":
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if proceed is not None:
			return proceed
		advance = next((action for action in legal if (action.get("action") or "").lower() == "advance_dialogue"), None)
		if advance is not None:
			return advance
		options = [action for action in legal if (action.get("action") or "").lower() == "choose_event_option"]
		if options:
			return sorted(options, key=lambda action: (action.get("index") or 999, action.get("label") or ""))[0]
	if state_type == "combat_rewards":
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if proceed is not None:
			return proceed
		claims = [action for action in legal if (action.get("action") or "").lower() == "claim_reward"]
		if claims:
			return sorted(claims, key=lambda action: action.get("index") or 999)[0]
	if state_type == "card_reward":
		skip = next((action for action in legal if (action.get("action") or "").lower() == "skip_card_reward"), None)
		if skip is not None:
			return skip
		selects = [action for action in legal if (action.get("action") or "").lower() == "select_card_reward"]
		if selects:
			return sorted(selects, key=lambda action: action.get("index") or 999)[0]
	if state_type == "shop":
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if proceed is not None:
			return proceed
	if state_type == "rest_site":
		options = [action for action in legal if (action.get("action") or "").lower() == "choose_rest_option"]
		if options:
			return sorted(options, key=lambda action: action.get("index") or 999)[0]
	if state_type == "card_select":
		selects = [action for action in legal if (action.get("action") or "").lower() == "select_card"]
		if selects:
			return sorted(selects, key=lambda action: action.get("index") or 999)[0]
		confirm = next((action for action in legal if (action.get("action") or "").lower() == "confirm_selection"), None)
		if confirm is not None:
			return confirm
		cancel = next((action for action in legal if (action.get("action") or "").lower() == "cancel_selection"), None)
		if cancel is not None:
			return cancel
	if state_type == "treasure":
		claim = next((action for action in legal if (action.get("action") or "").lower() == "claim_treasure_relic"), None)
		if claim is not None:
			return claim
		proceed = next((action for action in legal if (action.get("action") or "").lower() == "proceed"), None)
		if proceed is not None:
			return proceed
	return pick_deterministic_action(state_type, legal, rng)


def pick_training_like_action(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
	state_type = str(state.get("state_type") or "")
	legal = [action for action in state.get("legal_actions") or [] if bool(action.get("is_enabled", True))]
	if not legal:
		return {"action": "wait"}

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
		select = next((action for action in legal if "select" in (action.get("action") or "").lower()), None)
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


def check_state_invariants(step: int, state: dict[str, Any], prev_state: dict[str, Any] | None) -> list[Violation]:
	violations: list[Violation] = []
	state_type = str(state.get("state_type") or "unknown")
	terminal = bool(state.get("terminal", False))
	legal = state.get("legal_actions") or []
	run = state.get("run") or {}
	floor_val = run.get("floor", 0)
	if state_type not in VALID_STATE_TYPES and state_type != "unknown":
		violations.append(Violation(step, "STATE_TYPE", f"Unknown state_type: {state_type}", state_type))
	if terminal and state.get("run_outcome") not in {"victory", "defeat"}:
		violations.append(Violation(step, "OUTCOME", f"Terminal but run_outcome={state.get('run_outcome')!r}", state_type))
	if not terminal and state_type not in {"game_over", "combat_pending"} and len(legal) == 0 and state_type not in COMBAT_TYPES | {"hand_select"}:
		violations.append(Violation(step, "NO_ACTIONS", f"Non-terminal {state_type} with 0 legal actions", state_type))
	player = extract_player(state)
	hp = player.get("current_hp", player.get("hp"))
	max_hp = player.get("max_hp")
	if isinstance(hp, (int, float)) and isinstance(max_hp, (int, float)):
		if hp < 0:
			violations.append(Violation(step, "HP_NEGATIVE", f"HP is negative: {hp}", state_type))
		if hp > max_hp:
			violations.append(Violation(step, "HP_OVERFLOW", f"HP ({hp}) > max_hp ({max_hp})", state_type))
		if max_hp <= 0:
			violations.append(Violation(step, "MAX_HP", f"max_hp is non-positive: {max_hp}", state_type))
	gold = player.get("gold", run.get("gold"))
	if isinstance(gold, (int, float)) and gold < 0:
		violations.append(Violation(step, "GOLD_NEGATIVE", f"Gold is negative: {gold}", state_type))
	if prev_state is not None:
		prev_floor = (prev_state.get("run") or {}).get("floor", 0)
		if floor_val < prev_floor and not terminal:
			violations.append(Violation(step, "FLOOR_DECREASE", f"Floor decreased: {prev_floor} -> {floor_val}", state_type))
		prev_state_type = prev_state.get("state_type", "unknown")
		allowed = VALID_TRANSITIONS.get(prev_state_type)
		if allowed is not None and state_type not in allowed and state_type != prev_state_type:
			violations.append(Violation(step, "TRANSITION", f"Invalid transition: {prev_state_type} -> {state_type}", state_type))
	return violations


def resolve_seed_set(name: str) -> list[str]:
	return {"default": TEST_SEEDS, "parity": DEFAULT_PARITY_SEEDS, "coverage": COVERAGE_SEEDS, "stress": STRESS_SEEDS}[name]


def run_episode(client: FullRunClientLike, seed: str, *, rng_seed: int = 42, max_steps: int = MAX_STEPS_PER_RUN, check_invariants_enabled: bool = True) -> tuple[list[str], list[Violation], dict[str, Any]]:
	rng = random.Random(rng_seed)
	signatures: list[str] = []
	violations: list[Violation] = []
	summary = {"seed": seed, "max_floor": 0, "steps": 0, "outcome": None, "state_counts": Counter(), "wait_count": 0, "error_count": 0}
	state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
	prev_state = None
	for step in range(max_steps):
		signatures.append(state_signature(state))
		state_type = state.get("state_type", "?")
		terminal = bool(state.get("terminal", False))
		legal = state.get("legal_actions") or []
		summary["state_counts"][state_type] += 1
		summary["steps"] = step + 1
		summary["max_floor"] = max(summary["max_floor"], (state.get("run") or {}).get("floor", 0))
		if check_invariants_enabled:
			violations.extend(check_state_invariants(step, state, prev_state))
		if terminal or state_type == "game_over":
			summary["outcome"] = state.get("run_outcome") or "game_over"
			break
		prev_state = state
		try:
			if not legal:
				summary["wait_count"] += 1
				state = client.act({"action": "wait"})
			else:
				state = client.act(pick_deterministic_action(str(state_type), legal, rng))
		except Exception:
			summary["error_count"] += 1
			try:
				state = client.get_state()
			except Exception:
				break
	else:
		summary["outcome"] = "max_steps"
	return signatures, violations, summary


def run_coverage_episode(client: FullRunClientLike, seed: str, *, rng_seed: int = 42, max_steps: int = MAX_STEPS_PER_RUN, chooser: Any | None = None) -> dict[str, Any]:
	rng = random.Random(rng_seed)
	state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
	first_seen: dict[str, int] = {}
	transitions: Counter[tuple[str, str]] = Counter()
	prev_state_type: str | None = None
	for step in range(max_steps):
		state_type = str(state.get("state_type") or "")
		first_seen.setdefault(state_type, step)
		if prev_state_type is not None and prev_state_type != state_type:
			transitions[(prev_state_type, state_type)] += 1
		if state.get("terminal") or state_type == "game_over":
			break
		if chooser is None:
			legal = state.get("legal_actions") or []
			action = {"action": "wait"} if not legal else pick_deterministic_action(state_type, legal, rng)
		else:
			action = chooser(state, rng)
		prev_state_type = state_type
		state = client.act(action)
	return {"seed": seed, "first_seen": first_seen, "transitions": transitions}


def test_determinism(client: FullRunClientLike, seeds: list[str], max_steps: int, *, label: str | None = None) -> bool:
	print("=" * 60)
	print("TEST: DETERMINISM" if label is None else f"TEST: DETERMINISM ({label})")
	print("=" * 60)
	passed = True
	for seed in seeds[:2]:
		print(f"\n  Seed: {seed}")
		sigs1, _, summary1 = run_episode(client, seed, rng_seed=42, max_steps=max_steps, check_invariants_enabled=False)
		sigs2, _, summary2 = run_episode(client, seed, rng_seed=42, max_steps=max_steps, check_invariants_enabled=False)
		print(f"    run1: steps={len(sigs1)} floor={summary1['max_floor']} outcome={summary1['outcome']}")
		print(f"    run2: steps={len(sigs2)} floor={summary2['max_floor']} outcome={summary2['outcome']}")
		if sigs1 != sigs2:
			print("    FAIL: trajectories diverged")
			passed = False
		else:
			print("    PASS: trajectories match")
	return passed


def test_invariants(client: FullRunClientLike, seeds: list[str], max_steps: int) -> bool:
	print("=" * 60)
	print("TEST: STATE INVARIANTS")
	print("=" * 60)
	passed = True
	for seed in seeds:
		print(f"\n  Seed: {seed}")
		_, violations, summary = run_episode(client, seed, max_steps=max_steps, check_invariants_enabled=True)
		print(f"    steps={summary['steps']} floor={summary['max_floor']} outcome={summary['outcome']} waits={summary['wait_count']} errors={summary['error_count']}")
		if violations:
			print(f"    FAIL: {len(violations)} violations")
			for violation in violations[:5]:
				print(str(violation))
			passed = False
		else:
			print("    PASS: no invariant violations")
	return passed


def test_outcome(client: FullRunClientLike, seeds: list[str], max_steps: int) -> bool:
	print("=" * 60)
	print("TEST: RUN OUTCOME REPORTING")
	print("=" * 60)
	passed = True
	for seed in seeds:
		print(f"\n  Seed: {seed}")
		_, _, summary = run_episode(client, seed, max_steps=max_steps, check_invariants_enabled=False)
		print(f"    steps={summary['steps']} floor={summary['max_floor']} outcome={summary['outcome']}")
		if summary["outcome"] not in {"victory", "defeat", "max_steps"}:
			print("    FAIL: invalid outcome")
			passed = False
		else:
			print("    PASS")
	return passed


def run_parity_episode(
	baseline: FullRunClientLike,
	candidate: FullRunClientLike,
	*,
	seed: str,
	max_steps: int,
	action_driver: str = "baseline",
	parity_detail: str = "summary",
) -> tuple[bool, list[str], dict[str, Any]]:
	logs: list[str] = []
	baseline_state = baseline.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
	candidate_state = candidate.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
	rng = random.Random(42)
	previous_action: dict[str, Any] | None = None
	report: dict[str, Any] = {
		"seed": seed,
		"action_driver": action_driver,
		"parity_detail": parity_detail,
		"passed": False,
		"steps_executed": 0,
		"terminal": False,
		"outcome": None,
		"first_divergence": None,
	}
	for step in range(max_steps):
		baseline_summary = state_summary(baseline_state)
		candidate_summary = state_summary(candidate_state)
		summary_diffs = compare_state_summaries(baseline_summary, candidate_summary, detail_level=parity_detail)
		if summary_diffs:
			diff_kind = classify_summary_diffs(summary_diffs)
			if diff_kind == "representation_only":
				logs.append(f"step {step}: representation_only (ignored)")
			else:
				logs.append(f"step {step}: {diff_kind}")
				if previous_action is not None:
					logs.append(f"  previous_action={json.dumps(previous_action, ensure_ascii=True, sort_keys=True)}")
				logs.extend(f"  diff={entry}" for entry in summary_diffs[:8])
				logs.append(f"  baseline={json.dumps(baseline_summary, ensure_ascii=True, sort_keys=True)}")
				logs.append(f"  candidate={json.dumps(candidate_summary, ensure_ascii=True, sort_keys=True)}")
				report["first_divergence"] = {
					"step": step,
					"category": diff_kind,
					"previous_action": _deep_copy_jsonable(previous_action),
					"diffs": summary_diffs[:32],
					"baseline_summary": _deep_copy_jsonable(baseline_summary),
					"candidate_summary": _deep_copy_jsonable(candidate_summary),
				}
				return False, logs, report
		if baseline_state.get("terminal") or baseline_state.get("state_type") == "game_over":
			logs.append(f"terminal at step {step} outcome={baseline_state.get('run_outcome')}")
			report["passed"] = True
			report["steps_executed"] = step
			report["terminal"] = True
			report["outcome"] = baseline_state.get("run_outcome")
			return True, logs, report
		driver_state = baseline_state if action_driver == "baseline" else candidate_state
		legal = driver_state.get("legal_actions") or []
		action = {"action": "wait"} if not legal else pick_deterministic_action(str(driver_state.get("state_type") or ""), legal, rng)
		previous_action = dict(action)
		logs.append(f"step {step}: action={json.dumps(action, ensure_ascii=True, sort_keys=True)}")
		try:
			baseline_state = baseline.act(action)
		except Exception as exc:
			logs.append(f"step {step}: baseline protocol/runtime failure: {exc}")
			report["first_divergence"] = {
				"step": step,
				"category": "protocol/runtime",
				"driver_state_type": str(driver_state.get("state_type") or ""),
				"driver_backend": action_driver,
				"action": _deep_copy_jsonable(action),
				"failure_backend": "baseline",
				"error": str(exc),
			}
			return False, logs, report
		try:
			candidate_state = candidate.act(action)
		except Exception as exc:
			logs.append(f"step {step}: candidate protocol/runtime failure: {exc}")
			report["first_divergence"] = {
				"step": step,
				"category": "protocol/runtime",
				"driver_state_type": str(driver_state.get("state_type") or ""),
				"driver_backend": action_driver,
				"action": _deep_copy_jsonable(action),
				"failure_backend": "candidate",
				"error": str(exc),
			}
			return False, logs, report
		report["steps_executed"] = step + 1
	report["first_divergence"] = {
		"step": max_steps,
		"category": "max_steps",
		"driver_backend": action_driver,
	}
	return False, [*logs, f"max_steps reached ({max_steps})"], report


def _resolve_parity_directions(mode: str) -> list[tuple[str, str]]:
	return {
		"forward": [("baseline", "baseline->candidate")],
		"reverse": [("candidate", "candidate->baseline")],
		"bidirectional": [("baseline", "baseline->candidate"), ("candidate", "candidate->baseline")],
	}[mode]


def run_parity_suite(
	baseline: FullRunClientLike,
	candidate: FullRunClientLike,
	*,
	seeds: list[str],
	max_steps: int,
	parity_mode: str,
	parity_detail: str,
) -> tuple[bool, dict[str, Any]]:
	all_pass = True
	report: dict[str, Any] = {
		"parity_mode": parity_mode,
		"parity_detail": parity_detail,
		"directions": [],
	}
	for action_driver, label in _resolve_parity_directions(parity_mode):
		print(f"\n  Direction: {label}")
		direction_report: dict[str, Any] = {
			"label": label,
			"action_driver": action_driver,
			"passed": True,
			"seed_results": [],
		}
		for seed in seeds:
			print(f"    Seed: {seed}")
			ok, logs, episode_report = run_parity_episode(
				baseline,
				candidate,
				seed=seed,
				max_steps=max_steps,
				action_driver=action_driver,
				parity_detail=parity_detail,
			)
			print(f"      {'PASS' if ok else 'FAIL'}")
			if ok:
				if logs:
					print(f"      {logs[-1]}")
			else:
				for line in logs[:8]:
					print(f"      {line}")
				if len(logs) > 8:
					print(f"      ... {len(logs) - 8} more log lines")
			direction_report["seed_results"].append(episode_report)
			direction_report["passed"] = direction_report["passed"] and ok
			all_pass = all_pass and ok
		report["directions"].append(direction_report)
	report["passed"] = all_pass
	return all_pass, report


def test_parity(
	baseline_backend: str,
	candidate_backend: str,
	*,
	baseline_port: int,
	candidate_port: int,
	seeds: list[str],
	max_steps: int,
	auto_launch: bool,
	repo_root: Path,
	godot_exe: Path,
	headless_dll: Path,
	parity_mode: str = "bidirectional",
	parity_detail: str = "summary",
) -> tuple[bool, dict[str, Any]]:
	baseline_proc = None
	candidate_proc = None
	try:
		if auto_launch:
			if baseline_backend.startswith("headless-"):
				_kill_stale_headless_processes(baseline_port)
			if candidate_backend.startswith("headless-"):
				_kill_stale_headless_processes(candidate_port)
			baseline_proc = launch_backend(backend=baseline_backend, port=baseline_port, repo_root=repo_root, godot_exe=godot_exe, headless_dll=headless_dll)
			candidate_proc = launch_backend(backend=candidate_backend, port=candidate_port, repo_root=repo_root, godot_exe=godot_exe, headless_dll=headless_dll)
		baseline = create_client(baseline_backend, baseline_port)
		candidate = create_client(candidate_backend, candidate_port)
		try:
			print("=" * 60)
			print("TEST: PARITY")
			print("=" * 60)
			ok, report = run_parity_suite(
				baseline,
				candidate,
				seeds=seeds,
				max_steps=max_steps,
				parity_mode=parity_mode,
				parity_detail=parity_detail,
			)
			report.update({
				"baseline_backend": baseline_backend,
				"candidate_backend": candidate_backend,
			})
			return ok, report
		finally:
			baseline.close()
			candidate.close()
	finally:
		stop_process(baseline_proc)
		stop_process(candidate_proc)


def test_coverage_audit(
	client: FullRunClientLike,
	*,
	parity_seeds: list[str],
	coverage_seeds: list[str],
	stress_seeds: list[str],
	max_steps: int,
	enforce_required_coverage: bool = True,
) -> tuple[bool, dict[str, Any]]:
	print("=" * 60)
	print("TEST: COVERAGE AUDIT")
	print("=" * 60)
	parity_presence: dict[str, set[str]] = defaultdict(set)
	coverage_presence: dict[str, set[str]] = defaultdict(set)
	stress_presence: dict[str, set[str]] = defaultdict(set)
	first_seen_by_state: dict[str, list[tuple[str, int]]] = defaultdict(list)
	entry_sources: dict[str, set[str]] = defaultdict(set)
	tracked_transition_counts: Counter[tuple[str, str]] = Counter()
	transition_hits_by_group: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
	first_hit_by_transition: dict[tuple[str, str], tuple[str, int, str]] = {}
	runtime_failures: list[tuple[str, str]] = []

	coverage_policies = [("branch", pick_coverage_action), ("exit", pick_exit_hunting_action), ("training", pick_training_like_action)]

	def collect(seed_group: str, seeds: list[str], sink: dict[str, set[str]] | None) -> None:
		policies = coverage_policies if seed_group in {"coverage", "stress"} else [("default", None)]
		for policy_name, chooser in policies:
			for seed in seeds:
				seed_key = seed if policy_name == "default" else f"{seed}@{policy_name}"
				try:
					episode = run_coverage_episode(client, seed, max_steps=max_steps, chooser=chooser)
				except Exception as exc:
					runtime_failures.append((f"{seed_group}:{seed_key}", str(exc)))
					continue
				for state_name, first_step in episode["first_seen"].items():
					if sink is not None:
						sink[state_name].add(seed_key)
					if seed_group == "coverage":
						first_seen_by_state[state_name].append((seed_key, first_step))
				for transition, count in episode["transitions"].items():
					if seed_group == "coverage":
						entry_sources[transition[1]].add(transition[0])
						existing = first_hit_by_transition.get(transition)
						candidate = (seed_key, count, policy_name)
						if existing is None:
							first_hit_by_transition[transition] = candidate
					if transition in TRACKED_TRANSITIONS:
						tracked_transition_counts[transition] += count
						transition_hits_by_group[seed_group][transition] += count

	collect("parity", parity_seeds, parity_presence)
	collect("coverage", coverage_seeds, coverage_presence)
	collect("stress", stress_seeds, stress_presence)
	print("\n  State Coverage Matrix")
	for state_name in AUDIT_STATES:
		first_hits = sorted(first_seen_by_state.get(state_name, []), key=lambda item: (item[1], item[0]))
		first_hit = first_hits[0] if first_hits else None
		first_hit_text = f"{first_hit[0]}@{first_hit[1]}" if first_hit is not None else "-"
		sources = sorted(entry_sources.get(state_name, set()))
		print(f"    {state_name:16s} coverage={'Y' if state_name in coverage_presence else 'N'} parity={'Y' if state_name in parity_presence else 'N'} exact={'Y' if state_name in EXACT_SAVELOAD_STATES else 'N'} stress={'Y' if state_name in stress_presence else 'N'} first_hit={first_hit_text} sources={','.join(sources) if sources else '-'}")
	print("\n  High-Risk Transition Coverage")
	for transition in TRACKED_TRANSITIONS:
		first_hit = first_hit_by_transition.get(transition)
		first_hit_text = "-" if first_hit is None else f"{first_hit[0]} ({first_hit[2]})"
		required = "Y" if transition in REQUIRED_AUDIT_TRANSITIONS else "N"
		print(
			f"    {transition[0]} -> {transition[1]}: total={tracked_transition_counts.get(transition, 0)} "
			f"coverage={transition_hits_by_group['coverage'].get(transition, 0)} "
			f"stress={transition_hits_by_group['stress'].get(transition, 0)} "
			f"required={required} first_hit={first_hit_text}"
		)
	print(f"\n  runtime_failures: {len(runtime_failures)}")
	report: dict[str, Any] = {
		"states": {},
		"transitions": {},
		"runtime_failures": [{"seed": seed, "error": error} for seed, error in runtime_failures],
		"required_coverage_enforced": enforce_required_coverage,
	}
	for state_name in AUDIT_STATES:
		first_hits = sorted(first_seen_by_state.get(state_name, []), key=lambda item: (item[1], item[0]))
		first_hit = first_hits[0] if first_hits else None
		report["states"][state_name] = {
			"coverage": state_name in coverage_presence,
			"parity": state_name in parity_presence,
			"exact_save_load": state_name in EXACT_SAVELOAD_STATES,
			"stress": state_name in stress_presence,
			"first_hit_seed": first_hit[0] if first_hit is not None else None,
			"first_hit_step": first_hit[1] if first_hit is not None else None,
			"entry_sources": sorted(entry_sources.get(state_name, set())),
			"produced_by_full_run": state_name not in UNPRODUCED_AUDIT_STATES,
			"note": STATIC_AUDIT_NOTES.get(state_name),
		}
	for transition in TRACKED_TRANSITIONS:
		report["transitions"][_report_transition_key(transition)] = {
			"count": tracked_transition_counts.get(transition, 0),
			"coverage_count": transition_hits_by_group["coverage"].get(transition, 0),
			"stress_count": transition_hits_by_group["stress"].get(transition, 0),
			"first_hit_seed": (first_hit_by_transition.get(transition) or (None, None, None))[0],
			"first_hit_policy": (first_hit_by_transition.get(transition) or (None, None, None))[2],
			"note": STATIC_AUDIT_NOTES.get(transition),
			"required": transition in REQUIRED_AUDIT_TRANSITIONS,
		}
	if runtime_failures:
		print(f"  first_failure: {runtime_failures[0][0]} :: {runtime_failures[0][1]}")
		report["passed"] = False
		return False, report
	missing_states = [state_name for state_name in AUDIT_STATES if state_name not in coverage_presence]
	missing_required_states = [state_name for state_name in missing_states if state_name not in UNPRODUCED_AUDIT_STATES]
	missing_unproduced_states = [state_name for state_name in missing_states if state_name in UNPRODUCED_AUDIT_STATES]
	report["missing_required_states"] = missing_required_states
	report["missing_unproduced_states"] = missing_unproduced_states
	if missing_unproduced_states:
		print("\n  Not Currently Produced By Full-Run")
		for state_name in missing_unproduced_states:
			print(f"    {state_name}")
			note = STATIC_AUDIT_NOTES.get(state_name)
			if note:
				print(f"      note: {note}")
	if missing_required_states and enforce_required_coverage:
		print("\n  FAIL: missing coverage for states:")
		for state_name in missing_required_states:
			print(f"    {state_name}")
			note = STATIC_AUDIT_NOTES.get(state_name)
			if note:
				print(f"      note: {note}")
		report["passed"] = False
		return False, report
	missing_required_transitions = [transition for transition in REQUIRED_AUDIT_TRANSITIONS if tracked_transition_counts.get(transition, 0) <= 0]
	report["missing_required_transitions"] = [_report_transition_key(transition) for transition in missing_required_transitions]
	if missing_required_states and not enforce_required_coverage:
		print("\n  Advisory: missing coverage for states:")
		for state_name in missing_required_states:
			print(f"    {state_name}")
			note = STATIC_AUDIT_NOTES.get(state_name)
			if note:
				print(f"      note: {note}")
	if missing_required_transitions and enforce_required_coverage:
		print("\n  FAIL: missing coverage for required transitions:")
		for transition in missing_required_transitions:
			label = f"{transition[0]} -> {transition[1]}"
			print(f"    {label}")
		report["passed"] = False
		return False, report
	if missing_required_transitions and not enforce_required_coverage:
		print("\n  Advisory: missing coverage for required transitions:")
		for transition in missing_required_transitions:
			label = f"{transition[0]} -> {transition[1]}"
			print(f"    {label}")
	if missing_required_states or missing_required_transitions:
		print("\n  PASS: coverage gaps recorded as advisory-only")
	else:
		print("\n  PASS: coverage matrix populated for all audited states")
	report["passed"] = True
	return True, report


def test_discover(client: FullRunClientLike, *, start_index: int, count: int, max_steps: int, seed_prefix: str) -> tuple[bool, dict[str, Any]]:
	print("=" * 60)
	print("TEST: DISCOVER COVERAGE TARGETS")
	print("=" * 60)
	found_states: dict[str, tuple[str, int]] = {}
	found_transitions: dict[tuple[str, str], tuple[str, int]] = {}
	failures: list[tuple[str, str]] = []
	all_state_targets = DISCOVERY_REQUIRED_STATE_TARGETS + DISCOVERY_OPTIONAL_STATE_TARGETS
	all_transition_targets = DISCOVERY_REQUIRED_TRANSITION_TARGETS + DISCOVERY_OPTIONAL_TRANSITION_TARGETS
	for index in range(start_index, start_index + count):
		seed = f"{seed_prefix}_{index:04d}"
		try:
			episode = run_coverage_episode(client, seed, max_steps=max_steps, chooser=pick_coverage_action)
		except Exception as exc:
			failures.append((seed, str(exc)))
			continue
		first_seen = episode["first_seen"]
		transitions = episode["transitions"]
		for state_name in all_state_targets:
			if state_name in first_seen and state_name not in found_states:
				found_states[state_name] = (seed, first_seen[state_name])
		for transition in all_transition_targets:
			if transition in transitions and transition not in found_transitions:
				found_transitions[transition] = (seed, transitions[transition])
		if len(found_states) == len(all_state_targets) and len(found_transitions) == len(all_transition_targets):
			break
	print("\n  Required States")
	for state_name in DISCOVERY_REQUIRED_STATE_TARGETS:
		entry = found_states.get(state_name)
		print(f"    {state_name:20s} {entry[0]}@{entry[1]}" if entry else f"    {state_name:20s} -")
	if DISCOVERY_OPTIONAL_STATE_TARGETS:
		print("\n  Not Currently Produced States")
		for state_name in DISCOVERY_OPTIONAL_STATE_TARGETS:
			entry = found_states.get(state_name)
			print(f"    {state_name:20s} {entry[0]}@{entry[1]}" if entry else f"    {state_name:20s} -")
			note = STATIC_AUDIT_NOTES.get(state_name)
			if note:
				print(f"      note: {note}")
	print("\n  Required Transitions")
	for transition in DISCOVERY_REQUIRED_TRANSITION_TARGETS:
		entry = found_transitions.get(transition)
		label = f"{transition[0]} -> {transition[1]}"
		print(f"    {label:20s} {entry[0]} x{entry[1]}" if entry else f"    {label:20s} -")
	if DISCOVERY_OPTIONAL_TRANSITION_TARGETS:
		print("\n  Not Currently Produced Transitions")
		for transition in DISCOVERY_OPTIONAL_TRANSITION_TARGETS:
			entry = found_transitions.get(transition)
			label = f"{transition[0]} -> {transition[1]}"
			print(f"    {label:20s} {entry[0]} x{entry[1]}" if entry else f"    {label:20s} -")
			note = STATIC_AUDIT_NOTES.get(transition)
			if note:
				print(f"      note: {note}")
	print(f"\n  runtime_failures: {len(failures)}")
	if failures:
		print(f"  first_failure: {failures[0][0]} :: {failures[0][1]}")
	all_required_states_found = all(state_name in found_states for state_name in DISCOVERY_REQUIRED_STATE_TARGETS)
	all_required_transitions_found = all(transition in found_transitions for transition in DISCOVERY_REQUIRED_TRANSITION_TARGETS)
	report = {
		"passed": all_required_states_found and all_required_transitions_found and not failures,
		"required_state_targets": DISCOVERY_REQUIRED_STATE_TARGETS,
		"optional_state_targets": DISCOVERY_OPTIONAL_STATE_TARGETS,
		"required_transition_targets": [_report_transition_key(transition) for transition in DISCOVERY_REQUIRED_TRANSITION_TARGETS],
		"optional_transition_targets": [_report_transition_key(transition) for transition in DISCOVERY_OPTIONAL_TRANSITION_TARGETS],
		"found_states": {state_name: {"seed": found_states[state_name][0], "step": found_states[state_name][1]} for state_name in found_states},
		"found_transitions": {_report_transition_key(transition): {"seed": found_transitions[transition][0], "count": found_transitions[transition][1]} for transition in found_transitions},
		"static_notes": {
			_report_transition_key(key) if isinstance(key, tuple) else key: value
			for key, value in STATIC_AUDIT_NOTES.items()
		},
		"runtime_failures": [{"seed": seed, "error": error} for seed, error in failures],
	}
	return report["passed"], report


def test_audit(
	*,
	baseline_backend: str,
	candidate_backend: str,
	baseline_port: int,
	candidate_port: int,
	parity_seeds: list[str],
	coverage_seeds: list[str],
	stress_seeds: list[str],
	max_steps: int,
	auto_launch: bool,
	repo_root: Path,
	godot_exe: Path,
	headless_dll: Path,
	parity_mode: str = "bidirectional",
	parity_detail: str = "summary",
	coverage_enforcement: str = "required",
) -> tuple[bool, dict[str, Any]]:
	baseline_proc = None
	candidate_proc = None
	try:
		if auto_launch:
			if baseline_backend.startswith("headless-"):
				_kill_stale_headless_processes(baseline_port)
			if candidate_backend.startswith("headless-"):
				_kill_stale_headless_processes(candidate_port)
			baseline_proc = launch_backend(backend=baseline_backend, port=baseline_port, repo_root=repo_root, godot_exe=godot_exe, headless_dll=headless_dll)
			candidate_proc = launch_backend(backend=candidate_backend, port=candidate_port, repo_root=repo_root, godot_exe=godot_exe, headless_dll=headless_dll)
		baseline = create_client(baseline_backend, baseline_port)
		candidate = create_client(candidate_backend, candidate_port)
		try:
			baseline.reset_perf_stats()
			candidate.reset_perf_stats()
			ok_baseline_det = test_determinism(baseline, TEST_SEEDS, max_steps, label=baseline_backend)
			ok_candidate_det = test_determinism(candidate, TEST_SEEDS, max_steps, label=candidate_backend)
			ok_parity, parity_report = run_parity_suite(
				baseline,
				candidate,
				seeds=parity_seeds,
				max_steps=max_steps,
				parity_mode=parity_mode,
				parity_detail=parity_detail,
			)
			parity_report.update({
				"baseline_backend": baseline_backend,
				"candidate_backend": candidate_backend,
			})
			ok_coverage, coverage_report = test_coverage_audit(
				candidate,
				parity_seeds=parity_seeds,
				coverage_seeds=coverage_seeds,
				stress_seeds=stress_seeds,
				max_steps=max_steps,
				enforce_required_coverage=coverage_enforcement == "required",
			)
			report = {
				"baseline_backend": baseline_backend,
				"candidate_backend": candidate_backend,
				"baseline_determinism": ok_baseline_det,
				"candidate_determinism": ok_candidate_det,
				"parity": parity_report,
				"coverage": coverage_report,
				"baseline_perf_stats": _safe_perf_stats(baseline),
				"candidate_perf_stats": _safe_perf_stats(candidate),
			}
			print("\n" + "=" * 60)
			print("AUDIT SUMMARY")
			print("=" * 60)
			print(f"  baseline_determinism: {'PASS' if ok_baseline_det else 'FAIL'}")
			print(f"  candidate_determinism: {'PASS' if ok_candidate_det else 'FAIL'}")
			print(f"  parity: {'PASS' if ok_parity else 'FAIL'}")
			print(f"  coverage: {'PASS' if ok_coverage else 'FAIL'}")
			report["passed"] = ok_baseline_det and ok_candidate_det and ok_parity and ok_coverage
			return report["passed"], report
		finally:
			baseline.close()
			candidate.close()
	finally:
		stop_process(baseline_proc)
		stop_process(candidate_proc)


def main() -> int:
	parser = argparse.ArgumentParser(description="Audit parity, determinism, invariants, and coverage for full-run simulator backends.")
	parser.add_argument("--test", choices=["determinism", "invariants", "outcome", "parity", "coverage", "discover", "audit", "all"], default="all")
	parser.add_argument("--backend", choices=["godot-http", "godot-pipe", "headless-pipe", "headless-binary"], default="headless-pipe")
	parser.add_argument("--baseline-backend", choices=["godot-http", "godot-pipe", "headless-pipe", "headless-binary"], default="godot-http")
	parser.add_argument("--candidate-backend", choices=["godot-http", "godot-pipe", "headless-pipe", "headless-binary"], default="headless-pipe")
	parser.add_argument("--port", type=int, default=DEFAULT_PORT)
	parser.add_argument("--baseline-port", type=int, default=15526)
	parser.add_argument("--candidate-port", type=int, default=15527)
	parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
	parser.add_argument("--godot-exe", type=Path, default=DEFAULT_GODOT_EXE)
	parser.add_argument("--headless-dll", type=Path, default=DEFAULT_HEADLESS_DLL)
	parser.add_argument("--auto-launch", action="store_true")
	parser.add_argument("--seed-set", choices=["default", "parity", "coverage", "stress"], default="default")
	parser.add_argument("--seeds", nargs="+")
	parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_RUN)
	parser.add_argument("--discover-start", type=int, default=0)
	parser.add_argument("--discover-count", type=int, default=200)
	parser.add_argument("--discover-prefix", default=DISCOVERY_SEED_PREFIX)
	parser.add_argument("--parity-mode", choices=["forward", "reverse", "bidirectional"], default="bidirectional")
	parser.add_argument("--parity-detail", choices=["summary", "strict", "full", "display"], default="summary")
	parser.add_argument("--coverage-enforcement", choices=["required", "advisory"], default="required")
	parser.add_argument("--report-json", type=Path, default=None)
	args = parser.parse_args()

	def selected_seeds(default_group: str | None = None) -> list[str]:
		if args.seeds:
			return list(args.seeds)
		return resolve_seed_set(default_group or args.seed_set)

	if args.test in {"parity", "audit"}:
		parity_seeds = selected_seeds("parity")
		if args.test == "parity":
			ok, report = test_parity(
				args.baseline_backend,
				args.candidate_backend,
				baseline_port=args.baseline_port,
				candidate_port=args.candidate_port,
				seeds=parity_seeds,
				max_steps=args.max_steps,
				auto_launch=args.auto_launch,
				repo_root=args.repo_root,
				godot_exe=args.godot_exe,
				headless_dll=args.headless_dll,
				parity_mode=args.parity_mode,
				parity_detail=args.parity_detail,
			)
			if args.report_json is not None:
				args.report_json.parent.mkdir(parents=True, exist_ok=True)
				args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
			return 0 if ok else 1
		ok, report = test_audit(
			baseline_backend=args.baseline_backend,
			candidate_backend=args.candidate_backend,
			baseline_port=args.baseline_port,
			candidate_port=args.candidate_port,
			parity_seeds=parity_seeds,
			coverage_seeds=selected_seeds("coverage"),
			stress_seeds=selected_seeds("stress"),
			max_steps=args.max_steps,
			auto_launch=args.auto_launch,
			repo_root=args.repo_root,
			godot_exe=args.godot_exe,
			headless_dll=args.headless_dll,
			parity_mode=args.parity_mode,
			parity_detail=args.parity_detail,
			coverage_enforcement=args.coverage_enforcement,
		)
		if args.report_json is not None:
			args.report_json.parent.mkdir(parents=True, exist_ok=True)
			args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
		return 0 if ok else 1

	proc = None
	client: FullRunClientLike | None = None
	try:
		if args.auto_launch:
			proc = launch_backend(
				backend=args.backend,
				port=args.port,
				repo_root=args.repo_root,
				godot_exe=args.godot_exe,
				headless_dll=args.headless_dll,
			)
		client = create_client(args.backend, args.port)
		tests = [args.test] if args.test != "all" else ["determinism", "invariants", "outcome", "coverage"]
		results: list[tuple[str, bool]] = []
		for test_name in tests:
			if test_name == "determinism":
				results.append((test_name, test_determinism(client, selected_seeds("default"), args.max_steps)))
			elif test_name == "invariants":
				results.append((test_name, test_invariants(client, selected_seeds("default"), args.max_steps)))
			elif test_name == "outcome":
				results.append((test_name, test_outcome(client, selected_seeds("default"), args.max_steps)))
			elif test_name == "coverage":
				ok, report = test_coverage_audit(
					client,
					parity_seeds=selected_seeds("parity"),
					coverage_seeds=selected_seeds("coverage"),
					stress_seeds=selected_seeds("stress"),
					max_steps=args.max_steps,
					enforce_required_coverage=args.coverage_enforcement == "required",
				)
				results.append((test_name, ok))
				if args.report_json is not None:
					args.report_json.parent.mkdir(parents=True, exist_ok=True)
					args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
			elif test_name == "discover":
				ok, report = test_discover(client, start_index=args.discover_start, count=args.discover_count, max_steps=args.max_steps, seed_prefix=args.discover_prefix)
				results.append((test_name, ok))
				if args.report_json is not None:
					args.report_json.parent.mkdir(parents=True, exist_ok=True)
					args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
		print("\n" + "=" * 60)
		print("SUMMARY")
		print("=" * 60)
		for test_name, ok in results:
			print(f"  {test_name}: {'PASS' if ok else 'FAIL'}")
		return 0 if all(ok for _, ok in results) else 1
	finally:
		if client is not None:
			client.close()
		stop_process(proc)


if __name__ == "__main__":
	raise SystemExit(main())
