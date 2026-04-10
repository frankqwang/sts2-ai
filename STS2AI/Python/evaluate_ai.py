#!/usr/bin/env python3
"""Evaluate STS2 AI agent on fixed-seed games for reproducible benchmarking.

Loads a hybrid checkpoint (PPO + Combat NN) and runs N games sequentially
against a single Godot instance, collecting per-game statistics.  Optionally
compares against random and/or heuristic baselines.

Usage:
    # Evaluate latest checkpoint
    python evaluate_ai.py --checkpoint path/to/hybrid_XXXXX.pt --port 15527 --num-games 50

    # Compare with baselines
    python evaluate_ai.py --checkpoint path.pt --port 15527 --num-games 50 \
        --baseline random --baseline heuristic

    # Custom timeout and character
    python evaluate_ai.py --checkpoint path.pt --port 15527 --num-games 20 \
        --game-timeout 180 --character SILENT
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds STS2AI/Python library dirs to sys.path)

import argparse
import atexit
import json
import logging
import os
import random
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vocab import load_vocab, Vocab
from full_run_env import PipeBackedFullRunClient, ApiBackedFullRunClient, create_full_run_client
from combat_nn import (
    CombatPolicyValueNetwork,
    build_combat_features,
    build_combat_action_features,
    MAX_ACTIONS,
)
from rl_policy_v2 import FullRunPolicyNetworkV2
from rl_encoder_v2 import build_structured_state, build_structured_actions
from rl_policy_v2 import (
    _structured_state_to_numpy_dict,
    _structured_actions_to_numpy_dict,
)
from rl_reward_shaping import boss_readiness_score, extract_next_boss_token
from heuristic_combat import heuristic_combat_action
from combat_mcts_agent import CombatMCTSAgent, PipeCombatForwardModel
from headless_sim_runner import DEFAULT_DLL_PATH, start_headless_sim, stop_process
from mcts_core import MCTSConfig
from data.raw.raw_dataset_writer import write_raw_full_run_exports
from data.derived.build_rl_views import build_transition_view
from data.derived.build_llm_views import build_sft_dialogue_view
from sts2_singleplayer_env import (
    adapt_v1_state_for_combat_policy,
)
from archive.combat_actions import normalize_action
from archive.combat_bc import BehaviorCloningLinearPolicy
from combat_teacher_common import BODY_SLAM_TOKENS, _card_for_action, _card_slug, detect_motif_labels
from card_tags import load_card_tags
from sts2ai_paths import ARTIFACTS_ROOT, MAINLINE_CHECKPOINT, REPO_ROOT, SEEDS_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
DEFAULT_REPO_ROOT = REPO_ROOT
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "eval"

# State types considered combat
COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}
SELECTION_SCREENS = {"card_select", "hand_select", "relic_select"}
SELECTION_ACTION_NAMES = {
    "select_card",
    "combat_select_card",
    "combat_confirm_selection",
    "confirm_selection",
    "cancel_selection",
    "skip_relic_selection",
}
ESCAPE_ACTION_NAMES = ("proceed", "skip", "cancel_selection")

# Floor threshold for "act 1 boss beaten"
WIN_FLOOR = 17
DEFAULT_SEED_FILE = SEEDS_ROOT / "full_run_benchmark_seeds_act1_exit20.json"
TRAJECTORY_SCHEMA_VERSION = "full_run_trajectory.v1"
DEFAULT_COMBAT_BC_GATE_CONSTRAINTS = {
    "min_floor": 14,
    "room_types": ["elite", "boss"],
}
DEFAULT_COMBAT_BC_PATCH_CONFIG = {
    "mode": "rerank",
    "alpha": 0.5,
    "require_margin": False,
    "min_margin_zscore": 0.0,
    "max_base_top_prob": 1.0,
}
DEFAULT_COMBAT_MCTS_TACTICAL_BLEND_WEIGHT = 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _estimate_boss_hp_fraction(state: dict[str, Any]) -> float:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or []
    total_hp = 0
    total_max_hp = 0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        total_hp += _safe_int(enemy.get("hp", enemy.get("current_hp", 0)))
        total_max_hp += max(1, _safe_int(enemy.get("max_hp", 1)))
    if total_max_hp <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - total_hp / total_max_hp))


def _estimate_incoming_damage(state: dict[str, Any]) -> float:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or []
    total = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        if hp <= 0:
            continue
        intents = enemy.get("intents") if isinstance(enemy.get("intents"), list) else []
        if intents and isinstance(intents[0], dict):
            intent = intents[0]
            damage = float(intent.get("damage") or 0.0)
            hits = float(intent.get("hits") or intent.get("multiplier") or 1.0)
            total += damage * max(1.0, hits)
        else:
            total += float(enemy.get("intent_damage") or 0.0)
    return total


def _combat_tactical_leaf_value(state: dict[str, Any]) -> float:
    """Cheap local tactical value used only for eval-time MCTS experiments."""
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []

    player_hp = float(player.get("hp", player.get("current_hp", 0)) or 0.0)
    player_max_hp = max(1.0, float(player.get("max_hp", 1) or 1.0))
    player_block = float(player.get("block", 0) or 0.0)
    player_hp_frac = max(0.0, min(1.0, player_hp / player_max_hp))

    total_enemy_hp = 0.0
    total_enemy_max_hp = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        hp = float(enemy.get("hp", enemy.get("current_hp", 0)) or 0.0)
        max_hp = max(1.0, float(enemy.get("max_hp", 1) or 1.0))
        if hp <= 0:
            continue
        total_enemy_hp += hp
        total_enemy_max_hp += max_hp

    enemy_hp_frac = 1.0 if total_enemy_max_hp <= 0 else max(0.0, min(1.0, total_enemy_hp / total_enemy_max_hp))
    incoming_damage = _estimate_incoming_damage(state)
    pressure = max(0.0, incoming_damage - player_block)
    pressure_frac = max(0.0, min(1.0, pressure / player_max_hp))
    block_credit = max(0.0, min(1.0, min(player_block, incoming_damage) / player_max_hp))

    tactical = (
        (player_hp_frac - enemy_hp_frac)
        + 0.35 * block_credit
        - 0.45 * pressure_frac
    )
    return float(np.clip(tactical, -1.0, 1.0))


def _alive_enemy_names_from_live_state(state: dict[str, Any]) -> list[str]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
    names: list[str] = []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        if hp <= 0:
            continue
        name = _lower(enemy.get("name") or enemy.get("id"))
        if name:
            names.append(name)
    return names


def _load_seed_list(
    seeds_file: str | Path | None,
    seed_suite: str,
    num_games: int,
) -> list[str]:
    if seeds_file:
        path = Path(seeds_file)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                suite_entries = payload.get(seed_suite) or []
                seeds: list[str] = []
                for item in suite_entries:
                    if isinstance(item, dict):
                        seed = str(item.get("seed") or "").strip()
                        if seed:
                            seeds.append(seed)
                    elif item:
                        seeds.append(str(item))
                if seeds:
                    return seeds[:num_games]
            elif isinstance(payload, list):
                return [str(item) for item in payload[:num_games]]
    return [f"EVAL_{i + 1:03d}" for i in range(num_games)]


def _legal_action_name_set(legal: list[dict[str, Any]]) -> set[str]:
    return {
        str(action.get("action") or "")
        for action in legal
        if isinstance(action, dict)
    }


def _is_selection_screen(state_type: str, legal: list[dict[str, Any]]) -> bool:
    st = (state_type or "").strip().lower()
    return st in SELECTION_SCREENS or bool(
        _legal_action_name_set(legal) & SELECTION_ACTION_NAMES
    )


def _choose_auto_progress_action(
    state: dict[str, Any],
    state_type: str,
    legal: list[dict[str, Any]],
    last_reward_claim_sig: str | None = None,
) -> dict[str, Any] | None:
    st = (state_type or "").strip().lower()
    last_reward_claim_sig = str(last_reward_claim_sig or "").strip().lower()

    if _is_selection_screen(st, legal):
        for action in legal:
            action_name = str(action.get("action") or "")
            if "confirm" in action_name or "skip" in action_name:
                return action
        for action in legal:
            if "select" in str(action.get("action") or ""):
                return action

    if st == "combat_rewards":
        claim_action = _choose_claimable_reward_action(state, legal)
        repeated_claim = False
        if claim_action is not None:
            claim_sig = _reward_claim_signature(state, claim_action)
            repeated_claim = bool(claim_sig and claim_sig == last_reward_claim_sig)
        if repeated_claim:
            for action in legal:
                if action.get("action") in ("proceed", "skip"):
                    return action
        if claim_action is not None:
            return claim_action
        for action in legal:
            if action.get("action") in ("proceed", "skip"):
                return action

    return None


def _combat_rewards_state(state: dict[str, Any]) -> dict[str, Any]:
    rewards_state = state.get("combat_rewards")
    if isinstance(rewards_state, dict):
        return rewards_state
    rewards_state = state.get("rewards")
    if isinstance(rewards_state, dict):
        return rewards_state
    return {}


def _reward_item_claimable(state: dict[str, Any], reward_item: dict[str, Any] | None) -> bool:
    if not isinstance(reward_item, dict):
        return True
    explicit = reward_item.get("claimable")
    if explicit is not None:
        return bool(explicit)
    reward_type = str(reward_item.get("type") or "").strip().lower()
    if reward_type != "potion":
        return True
    rewards_state = _combat_rewards_state(state)
    player = rewards_state.get("player") or state.get("player") or {}
    try:
        open_slots = int(player.get("open_potion_slots", 0) or 0)
    except Exception:
        open_slots = 0
    return open_slots > 0


def _choose_claimable_reward_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> dict[str, Any] | None:
    rewards_state = _combat_rewards_state(state)
    items = rewards_state.get("items")
    indexed_items: dict[int, dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except Exception:
                continue
            indexed_items[idx] = item
    fallback: dict[str, Any] | None = None
    for action in legal:
        if action.get("action") != "claim_reward":
            continue
        reward_item = indexed_items.get(int(action.get("index", -1)))
        enriched_action = dict(action)
        for src_key, dst_key in (
            ("reward_type", "reward_type"),
            ("reward_id", "reward_id"),
            ("reward_key", "reward_key"),
            ("reward_source", "reward_source"),
            ("claimable", "claimable"),
            ("claim_block_reason", "claim_block_reason"),
        ):
            if action.get(src_key) is not None and enriched_action.get(dst_key) is None:
                enriched_action[dst_key] = action.get(src_key)
        if isinstance(reward_item, dict):
            for src_key, dst_key in (
                ("type", "reward_type"),
                ("id", "reward_id"),
                ("reward_key", "reward_key"),
                ("reward_source", "reward_source"),
                ("claimable", "claimable"),
                ("claim_block_reason", "claim_block_reason"),
            ):
                if reward_item.get(src_key) is not None and not enriched_action.get(dst_key):
                    enriched_action[dst_key] = reward_item.get(src_key)
        if fallback is None:
            fallback = enriched_action
        explicit_claimable = enriched_action.get("claimable")
        if explicit_claimable is not None:
            if bool(explicit_claimable):
                return enriched_action
            continue
        if _reward_item_claimable(state, reward_item):
            return enriched_action
    return fallback


def _reward_claim_signature(state: dict[str, Any], action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    if str(action.get("action") or "").strip().lower() != "claim_reward":
        return ""
    remaining_claim_actions = 0
    try:
        remaining_claim_actions = sum(
            1
            for legal_action in (state.get("legal_actions") or [])
            if isinstance(legal_action, dict)
            and str(legal_action.get("action") or "").strip().lower() == "claim_reward"
            and legal_action.get("is_enabled") is not False
        )
    except Exception:
        remaining_claim_actions = 0
    parts = [
        str(action.get("action") or "").strip().lower(),
        str(action.get("label") or "").strip().lower(),
        str(action.get("reward_type") or "").strip().lower(),
        str(action.get("reward_id") or action.get("id") or "").strip().lower(),
        str(action.get("reward_key") or "").strip().lower(),
        str(remaining_claim_actions),
    ]
    return "|".join(parts)


def _next_reward_claim_signature(
    state_type: str,
    state: dict[str, Any],
    action: dict[str, Any] | None,
) -> str:
    if (state_type or "").strip().lower() != "combat_rewards":
        return ""
    return _reward_claim_signature(state, action)


def _choose_repeat_escape_action(
    legal: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not legal:
        return None
    for action_name in ESCAPE_ACTION_NAMES:
        for action in legal:
            if action.get("action") == action_name:
                return action
    if len(legal) > 1:
        return legal[1]
    return legal[0]


@dataclass
class RepeatLoopTracker:
    trigger_count: int = 3
    max_repeats: int = 20
    last_state_key: str = ""
    repeat_count: int = 0

    def observe(self, state_type: str, legal: list[dict[str, Any]]) -> int:
        current_key = f"{(state_type or '').strip().lower()}:{len(legal)}"
        if current_key == self.last_state_key:
            self.repeat_count += 1
        else:
            self.last_state_key = current_key
            self.repeat_count = 0
        return self.repeat_count

    def choose_escape_action(
        self,
        legal: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if self.repeat_count < self.trigger_count:
            return None
        return _choose_repeat_escape_action(legal)

    def should_abort(self) -> bool:
        return self.repeat_count >= self.max_repeats


def _parse_trace_seed_arg(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    seeds = {
        item.strip()
        for item in str(raw).split(",")
        if item and item.strip()
    }
    return seeds or None


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _json_safe_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe_tree(inner)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [_json_safe_tree(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_tree(item) for item in value]
    return _json_safe_value(value)


def _sanitize_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    sanitized: dict[str, Any] = {}
    for key in (
        "action",
        "label",
        "node_type",
        "index",
        "target_id",
        "target",
        "card_index",
        "card_id",
        "slot",
        "screen_type",
    ):
        if key in action:
            sanitized[key] = _json_safe_value(action.get(key))
    return sanitized


def _make_trace_step(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    chosen_action: dict[str, Any] | None,
    action_source: str,
    step_index: int,
    combat_mcts_trace: CombatMctsTrace | None = None,
) -> dict[str, Any]:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    step = {
        "step": int(step_index),
        "state_type": str(state.get("state_type") or "").strip().lower(),
        "floor": _safe_int(run.get("floor", 0), 0),
        "act": _safe_int(run.get("act", 0), 0),
        "hp": _safe_int(player.get("hp", player.get("current_hp", 0)), 0),
        "max_hp": _safe_int(player.get("max_hp", 0), 0),
        "gold": _safe_int(player.get("gold", 0), 0),
        "legal_action_count": len(legal),
        "chosen_action": _sanitize_action(chosen_action),
        "action_source": action_source,
        "next_boss_token": extract_next_boss_token(state),
        "boss_readiness": round(boss_readiness_score(state), 6),
        "boss_hp_fraction_seen": round(_estimate_boss_hp_fraction(state), 6),
    }
    if combat_mcts_trace is not None:
        step["combat_mcts"] = {
            "chosen_action": _sanitize_action(combat_mcts_trace.chosen_action),
            "top_actions": [
                {
                    **{k: _json_safe_value(v) for k, v in item.items() if k in {"prior", "visits", "visit_frac", "q"}},
                    "action": _sanitize_action(item.get("action")),
                }
                for item in combat_mcts_trace.top_actions
            ],
            "sims": int(combat_mcts_trace.sims),
            "root_value": float(combat_mcts_trace.root_value),
        }
    return step


def _extract_player_snapshot(state: dict[str, Any]) -> dict[str, Any] | None:
    battle = state.get("battle")
    if isinstance(battle, dict) and isinstance(battle.get("player"), dict):
        return battle["player"]

    for key in (
        "map",
        "shop",
        "rest_site",
        "event",
        "rewards",
        "card_reward",
        "card_select",
        "relic_select",
        "treasure",
        "menu",
    ):
        container = state.get(key)
        if isinstance(container, dict) and isinstance(container.get("player"), dict):
            return container["player"]

    if isinstance(state.get("player"), dict):
        return state["player"]

    return None


def _extract_progress(state: dict[str, Any]) -> dict[str, Any]:
    run_state = state.get("run") if isinstance(state.get("run"), dict) else {}
    player = _extract_player_snapshot(state) or {}
    deck = player.get("deck") if isinstance(player.get("deck"), list) else []
    relics = player.get("relics") if isinstance(player.get("relics"), list) else []
    potions = player.get("potions") if isinstance(player.get("potions"), list) else []
    return {
        "state_type": _lower(state.get("state_type")),
        "act": _safe_int(run_state.get("act"), 0),
        "floor": _safe_int(run_state.get("floor"), 0),
        "hp": _safe_int(player.get("hp", player.get("current_hp")), 0),
        "max_hp": max(1, _safe_int(player.get("max_hp"), 1)),
        "gold": _safe_int(player.get("gold"), 0),
        "deck_count": len(deck),
        "relic_count": len(relics),
        "potion_count": len(potions),
    }


def _compute_delta(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    before_progress: dict[str, Any],
    after_progress: dict[str, Any],
) -> dict[str, Any]:
    before_safe = _json_safe_tree(before_state)
    after_safe = _json_safe_tree(after_state)
    changed_keys: list[str] = []
    before_keys = set(before_safe.keys()) if isinstance(before_safe, dict) else set()
    after_keys = set(after_safe.keys()) if isinstance(after_safe, dict) else set()
    for key in sorted(before_keys | after_keys):
        before_value = before_safe.get(key) if isinstance(before_safe, dict) else None
        after_value = after_safe.get(key) if isinstance(after_safe, dict) else None
        if before_value != after_value:
            changed_keys.append(key)

    return {
        "state_changed": before_safe != after_safe,
        "state_type_changed": before_progress["state_type"] != after_progress["state_type"],
        "changed_top_level_keys": changed_keys,
        "act_delta": after_progress["act"] - before_progress["act"],
        "floor_delta": after_progress["floor"] - before_progress["floor"],
        "hp_delta": after_progress["hp"] - before_progress["hp"],
        "max_hp_delta": after_progress["max_hp"] - before_progress["max_hp"],
        "gold_delta": after_progress["gold"] - before_progress["gold"],
        "deck_count_delta": after_progress["deck_count"] - before_progress["deck_count"],
        "relic_count_delta": after_progress["relic_count"] - before_progress["relic_count"],
        "potion_count_delta": after_progress["potion_count"] - before_progress["potion_count"],
    }


def _run_outcome_from_state(state: dict[str, Any]) -> str:
    game_over = state.get("game_over") if isinstance(state.get("game_over"), dict) else {}
    return str(game_over.get("run_outcome") or game_over.get("outcome") or "").strip().lower()


def _make_trajectory_record(
    *,
    run_id: str,
    seed: str,
    step_index: int,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    chosen_action: dict[str, Any],
    action_source: str,
    next_state: dict[str, Any],
) -> dict[str, Any]:
    before_progress = _extract_progress(state)
    after_progress = _extract_progress(next_state)
    terminal = bool(next_state.get("terminal")) or _lower(next_state.get("state_type")) == "game_over"
    record = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "run_id": run_id,
        "step_index": int(step_index),
        "timestamp_utc": _utc_now(),
        "seed": seed,
        "act": before_progress["act"],
        "floor": before_progress["floor"],
        "state_type": before_progress["state_type"],
        "env_api_mode": "v1_singleplayer",
        "raw_state": _json_safe_tree(state),
        "candidate_actions": _json_safe_tree(legal),
        "chosen_action": _json_safe_tree(chosen_action),
        "action_source": action_source,
        "next_state": _json_safe_tree(next_state),
        "terminal": terminal,
        "delta": _compute_delta(state, next_state, before_progress, after_progress),
    }
    if terminal:
        run_outcome = _run_outcome_from_state(next_state)
        if run_outcome:
            record["run_outcome"] = run_outcome
    return record


def _write_trace_outputs(
    output_dir: Path,
    strategy: str,
    results: list[GameResult],
    trace_payloads: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    index_payload = {
        **metadata,
        "strategy": strategy,
        "captured_trace_count": len(trace_payloads),
        "captured_seeds": sorted(trace_payloads.keys()),
        "results": [asdict(result) for result in results],
    }
    (output_dir / "index.json").write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for result in results:
        steps = trace_payloads.get(result.seed)
        if steps is None:
            continue
        payload = {
            "summary": asdict(result),
            "trace": steps,
        }
        (output_dir / f"{result.seed}_trace.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _write_trajectory_outputs(
    output_dir: Path,
    strategy: str,
    results: list["GameResult"],
    trajectory_payloads: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        **metadata,
        "strategy": strategy,
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "captured_trajectory_count": len(trajectory_payloads),
        "captured_seeds": sorted(trajectory_payloads.keys()),
        "results": [asdict(result) for result in results],
    }
    (output_dir / "trajectory_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for result in results:
        records = trajectory_payloads.get(result.seed)
        if records is None:
            continue
        trajectory_path = output_dir / f"{result.seed}_trajectory.jsonl"
        with trajectory_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")

    raw_path, _ = write_raw_full_run_exports(
        output_dir=output_dir,
        trajectory_payloads=trajectory_payloads,
        metadata={
            **metadata,
            "strategy": strategy,
            "backend_kind": metadata.get("backend_kind") or metadata.get("env_api_mode") or "evaluate_ai",
            "transport": metadata.get("transport"),
            "port": metadata.get("port"),
        },
        checkpoint_path=metadata.get("checkpoint"),
        checkpoint_sha256=metadata.get("checkpoint_sha256"),
    )
    raw_records: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    build_transition_view(
        raw_full_run_records=raw_records,
        output_dir=output_dir / "derived" / "rl",
    )
    build_sft_dialogue_view(
        raw_full_run_records=raw_records,
        output_dir=output_dir / "derived" / "llm",
    )


# ---------------------------------------------------------------------------
# Per-game result
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    game_id: int = 0
    seed: str = ""
    strategy: str = ""  # "nn", "random", "heuristic"
    max_floor: int = 0
    final_hp: int = 0
    final_max_hp: int = 0
    num_combats_won: int = 0
    total_steps: int = 0
    time_taken_s: float = 0.0
    outcome: str = ""  # "victory", "death", "timeout", "error"
    error_msg: str = ""
    boss_reached: bool = False
    act1_cleared: bool = False
    boss_hp_fraction_dealt: float = 0.0
    next_boss_token: str = "unknown"
    boss_readiness_at_floor_8: float | None = None
    boss_readiness_at_floor_12: float | None = None
    boss_readiness_at_floor_16: float | None = None
    card_reward_screens: int = 0
    card_reward_skips: int = 0
    floor_at_death: int = 0
    last_state_type: str = ""
    last_state_floor: int = 0
    last_state_legal_action_count: int = 0
    timeout_state_type: str = ""
    timeout_floor: int = 0
    timeout_legal_action_count: int = 0
    action_source_counts: dict[str, int] = field(default_factory=dict)
    combat_teacher_override_counts: dict[str, int] = field(default_factory=dict)
    # Boss-entry cohort capture (first time `state_type == "boss"` is observed)
    boss_entry_floor: int = 0
    boss_entry_hp: int = 0
    boss_entry_max_hp: int = 0
    boss_entry_deck_count: int = 0
    boss_entry_relic_count: int = 0
    boss_entry_potion_count: int = 0
    boss_entry_gold: int = 0
    boss_entry_token: str = "unknown"


@dataclass(slots=True)
class CombatBcPatchConfig:
    mode: str = "rerank"
    alpha: float = 0.5
    require_margin: bool = False
    min_margin_zscore: float = 0.0
    max_base_top_prob: float = 1.0


@dataclass(slots=True)
class CombatBcOverride:
    policy: BehaviorCloningLinearPolicy
    model_path: str
    gate_constraints: dict[str, Any] = field(default_factory=dict)
    patch_config: CombatBcPatchConfig = field(default_factory=CombatBcPatchConfig)


@dataclass(slots=True)
class CombatMctsTrace:
    chosen_action: dict[str, Any]
    top_actions: list[dict[str, Any]]
    sims: int
    root_value: float


@dataclass(slots=True)
class CombatTeacherOverride:
    network: CombatPolicyValueNetwork
    vocab: Vocab
    device: torch.device
    mode: str = "replace"
    lethal_logit_blend_alpha: float = 0.0
    direct_lethal_probe_top_k: int = 4
    direct_lethal_step_budget: int = 24


@dataclass(slots=True)
class CombatMctsTacticalBlendConfig:
    weight: float = 0.0


class CombatMctsTacticalBlendEvaluator:
    """Eval-time wrapper that nudges MCTS leaf values toward local tactical progress."""

    def __init__(self, base_evaluator: Any, config: CombatMctsTacticalBlendConfig):
        self.base_evaluator = base_evaluator
        self.config = config

    def _blend_value(self, state: dict[str, Any], nn_value: float) -> float:
        weight = max(0.0, min(1.0, float(self.config.weight)))
        if weight <= 0.0:
            return float(nn_value)
        tactical = _combat_tactical_leaf_value(state)
        return float(np.clip((1.0 - weight) * float(nn_value) + weight * tactical, -1.0, 1.0))

    def evaluate(self, state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
        policy, value = self.base_evaluator.evaluate(state, legal_actions)
        return policy, self._blend_value(state, value)

    def evaluate_batch(
        self,
        states: list[dict[str, Any]],
        legal_actions_list: list[list[dict[str, Any]]],
    ) -> list[tuple[np.ndarray, float]]:
        base_results = self.base_evaluator.evaluate_batch(states, legal_actions_list)
        return [
            (policy, self._blend_value(state, value))
            for (policy, value), state in zip(base_results, states)
        ]


# ---------------------------------------------------------------------------
# Tensor helpers (mirror train_hybrid.py patterns)
# ---------------------------------------------------------------------------

def _build_ppo_tensors(
    state: dict, legal: list[dict], vocab: Vocab, device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build PPO state/action tensors from raw game state."""
    ss = build_structured_state(state, vocab)
    sa = build_structured_actions(state, legal, vocab)

    state_t: dict[str, torch.Tensor] = {}
    for k, v in _structured_state_to_numpy_dict(ss).items():
        t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
        if "ids" in k or "idx" in k or "types" in k or "count" in k:
            t = t.long()
        elif "mask" in k:
            t = t.bool()
        else:
            t = t.float()
        state_t[k] = t.to(device)

    action_t: dict[str, torch.Tensor] = {}
    for k, v in _structured_actions_to_numpy_dict(sa).items():
        t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
        if "ids" in k or "types" in k or "indices" in k:
            t = t.long()
        elif "mask" in k:
            t = t.bool()
        else:
            t = t.float()
        action_t[k] = t.to(device)

    return state_t, action_t


def _build_combat_tensors(
    state: dict, legal: list[dict], vocab: Vocab, device: torch.device,
    ppo_net: FullRunPolicyNetworkV2 | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build combat NN state/action tensors from raw game state."""
    sf = build_combat_features(state, vocab)
    af = build_combat_action_features(state, legal, vocab)

    # Inject deck_repr from PPO brain if available (build_plan_z bridge)
    if ppo_net is not None and hasattr(ppo_net, "compute_deck_repr"):
        try:
            from rl_encoder_v2 import build_structured_state as _bss
            ss = _bss(state, vocab)
            deck_t = {
                "deck_ids": torch.tensor(ss.deck_ids).unsqueeze(0).to(device),
                "deck_aux": torch.tensor(ss.deck_aux).unsqueeze(0).float().to(device),
                "deck_mask": torch.tensor(ss.deck_mask).unsqueeze(0).bool().to(device),
            }
            with torch.no_grad():
                sf["deck_repr"] = ppo_net.compute_deck_repr(deck_t).squeeze(0).cpu().numpy()
        except Exception:
            pass

    sf_t: dict[str, torch.Tensor] = {}
    for k, v in sf.items():
        t = torch.tensor(v).unsqueeze(0)
        if v.dtype in (np.int64, np.int32):
            t = t.long()
        elif v.dtype == bool:
            t = t.bool()
        else:
            t = t.float()
        sf_t[k] = t.to(device)

    af_t: dict[str, torch.Tensor] = {}
    for k, v in af.items():
        t = torch.tensor(v).unsqueeze(0)
        if v.dtype in (np.int64, np.int32):
            t = t.long()
        elif v.dtype == bool:
            t = t.bool()
        else:
            t = t.float()
        af_t[k] = t.to(device)

    return sf_t, af_t


def _combat_probe_is_victory(state: dict[str, Any]) -> bool:
    st = _lower(state.get("state_type"))
    if st == "game_over" or state.get("terminal"):
        outcome = _lower((state.get("game_over") or {}).get("run_outcome") or (state.get("game_over") or {}).get("outcome"))
        return "victory" in outcome or outcome == "win"
    if st not in COMBAT_SCREENS and st not in SELECTION_SCREENS:
        return True
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
    living = [
        enemy for enemy in enemies
        if isinstance(enemy, dict) and _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0) > 0
    ]
    return bool(enemies) and not living


def _combat_teacher_probe_candidate_indices(
    legal: list[dict[str, Any]],
    masked_scores: np.ndarray,
    masked_logits: np.ndarray,
    *,
    top_k: int,
) -> list[int]:
    supported = [
        idx for idx, action in enumerate(legal)
        if _lower(action.get("action")) in {"play_card", "use_potion"}
    ]
    if not supported:
        return []
    top_k = max(1, int(top_k))
    ordered_scores = sorted(supported, key=lambda idx: float(masked_scores[idx]), reverse=True)
    ordered_logits = sorted(supported, key=lambda idx: float(masked_logits[idx]), reverse=True)
    candidate_indices: list[int] = []
    for idx in ordered_scores[:top_k] + ordered_logits[:top_k]:
        if idx not in candidate_indices:
            candidate_indices.append(idx)
    return candidate_indices


def _probe_direct_lethal_indices(
    *,
    legal: list[dict[str, Any]],
    pipe_getter: Any | None,
    candidate_indices: list[int],
    step_budget: int,
) -> set[int]:
    if pipe_getter is None or not candidate_indices:
        return set()
    fm = None
    lethal_indices: set[int] = set()
    try:
        fm = PipeCombatForwardModel.from_current_state(
            pipe_getter,
            max_step_budget=max(4, int(step_budget)),
        )
        for idx in candidate_indices:
            if idx < 0 or idx >= len(legal):
                continue
            child = fm.clone()
            try:
                child.step(legal[idx])
                if _combat_probe_is_victory(child.get_state_dict()):
                    lethal_indices.add(idx)
            except Exception:
                continue
        return lethal_indices
    except Exception:
        return set()
    finally:
        if fm is not None:
            try:
                restored = fm.cleanup_and_restore()
                if restored is None:
                    fm.cleanup()
            except Exception:
                try:
                    fm.cleanup()
                except Exception:
                    pass


def _select_combat_teacher_index(
    *,
    legal: list[dict[str, Any]],
    masked_scores: np.ndarray,
    masked_logits: np.ndarray,
    lethal_logit_blend_alpha: float,
    direct_lethal_probe_top_k: int,
    direct_lethal_probe: Any | None = None,
) -> tuple[int, str]:
    if len(legal) == 0:
        return 0, "combat_teacher_scores"
    score_idx = int(np.argmax(masked_scores))
    alpha = float(max(0.0, lethal_logit_blend_alpha))
    if alpha <= 0.0 or direct_lethal_probe is None:
        return score_idx, "combat_teacher_scores"
    logit_idx = int(np.argmax(masked_logits))
    if score_idx == logit_idx:
        return score_idx, "combat_teacher_scores"
    candidate_indices = _combat_teacher_probe_candidate_indices(
        legal,
        masked_scores,
        masked_logits,
        top_k=direct_lethal_probe_top_k,
    )
    if not candidate_indices:
        return score_idx, "combat_teacher_scores"
    lethal_indices = {
        idx for idx in (direct_lethal_probe(candidate_indices) or set())
        if 0 <= int(idx) < len(legal)
    }
    if not lethal_indices:
        return score_idx, "combat_teacher_scores"
    blended = masked_scores + alpha * masked_logits
    best_idx = max(lethal_indices, key=lambda idx: float(blended[idx]))
    return int(best_idx), "combat_teacher_direct_lethal_blend"


def _combat_teacher_forward_arrays(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    combat_teacher_override: CombatTeacherOverride,
) -> tuple[np.ndarray, np.ndarray]:
    sf_t, af_t = _build_combat_tensors(
        state,
        legal,
        combat_teacher_override.vocab,
        combat_teacher_override.device,
    )
    with torch.no_grad():
        logits, _value, action_scores, _continuation = combat_teacher_override.network.forward_teacher(sf_t, af_t)
    action_count = len(legal)
    mask = af_t["action_mask"].squeeze(0).detach().cpu().numpy().astype(bool)
    masked_scores = action_scores.squeeze(0).detach().cpu().float().numpy()[:action_count]
    masked_logits = logits.squeeze(0).detach().cpu().float().numpy()[:action_count]
    if action_count > 0:
        masked_scores = np.where(mask[:action_count], masked_scores, -1e9)
        masked_logits = np.where(mask[:action_count], masked_logits, -1e9)
    return masked_scores, masked_logits


def _combat_teacher_runtime_override_source(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    baseline_idx: int,
    teacher_idx: int,
    runtime_labels: set[str],
) -> str | None:
    if not (0 <= baseline_idx < len(legal) and 0 <= teacher_idx < len(legal)):
        return None
    if baseline_idx == teacher_idx:
        return None

    baseline_action = legal[baseline_idx]
    teacher_action = legal[teacher_idx]
    baseline_name = _lower(baseline_action.get("action"))
    teacher_name = _lower(teacher_action.get("action"))

    if "bad_end_turn" in runtime_labels and baseline_name == "end_turn" and teacher_name != "end_turn":
        return "combat_teacher_rerank_bad_end_turn"

    if "bash_before_strike" in runtime_labels:
        baseline_card = _card_for_action(state, baseline_action)
        teacher_card = _card_for_action(state, teacher_action)
        baseline_slug = _card_slug(baseline_card)
        teacher_slug = _card_slug(teacher_card)
        card_tags = load_card_tags()
        baseline_tags = set(card_tags.get(baseline_slug, []))
        teacher_tags = set(card_tags.get(teacher_slug, []))
        if teacher_name == "play_card" and "vulnerable" in teacher_tags and "vulnerable" not in baseline_tags:
            return "combat_teacher_rerank_bash_setup"

    if "bodyslam_before_block" in runtime_labels:
        baseline_slug = _card_slug(_card_for_action(state, baseline_action))
        teacher_slug = _card_slug(_card_for_action(state, teacher_action))
        card_tags = load_card_tags()
        teacher_tags = set(card_tags.get(teacher_slug, []))
        if (
            baseline_name == "play_card"
            and baseline_slug in BODY_SLAM_TOKENS
            and teacher_name == "play_card"
            and "block" in teacher_tags
            and teacher_slug not in BODY_SLAM_TOKENS
        ):
            return "combat_teacher_rerank_body_slam"

    return None


def _select_action_combat_teacher_rerank(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    baseline_logits: np.ndarray,
    combat_teacher_override: CombatTeacherOverride,
    pipe_getter: Any | None,
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    if not legal:
        return None, None, None
    masked_scores, teacher_logits = _combat_teacher_forward_arrays(
        state=state,
        legal=legal,
        combat_teacher_override=combat_teacher_override,
    )
    score_idx = int(np.argmax(masked_scores))
    baseline_idx = int(np.argmax(baseline_logits[:len(legal)])) if len(legal) > 0 else 0

    if pipe_getter is not None and combat_teacher_override.lethal_logit_blend_alpha > 0.0:
        candidate_indices = _combat_teacher_probe_candidate_indices(
            legal,
            masked_scores,
            baseline_logits[:len(legal)],
            top_k=combat_teacher_override.direct_lethal_probe_top_k,
        )
        lethal_indices = _probe_direct_lethal_indices(
            legal=legal,
            pipe_getter=pipe_getter,
            candidate_indices=candidate_indices,
            step_budget=combat_teacher_override.direct_lethal_step_budget,
        )
        if lethal_indices:
            best_idx = max(lethal_indices, key=lambda idx: float(baseline_logits[idx]))
            return int(best_idx), legal[int(best_idx)], "combat_teacher_rerank_direct_lethal"

    runtime_labels = set(detect_motif_labels(state, legal))
    override_source = _combat_teacher_runtime_override_source(
        state=state,
        legal=legal,
        baseline_idx=baseline_idx,
        teacher_idx=score_idx,
        runtime_labels=runtime_labels,
    )
    if override_source is not None:
        return score_idx, legal[score_idx], override_source
    return None, None, None


def _select_action_combat_teacher(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    combat_teacher_override: CombatTeacherOverride,
    pipe_getter: Any | None,
) -> tuple[int, dict[str, Any], str]:
    masked_scores, masked_logits = _combat_teacher_forward_arrays(
        state=state,
        legal=legal,
        combat_teacher_override=combat_teacher_override,
    )
    direct_lethal_probe = None
    if pipe_getter is not None and combat_teacher_override.lethal_logit_blend_alpha > 0.0:
        direct_lethal_probe = lambda candidate_indices: _probe_direct_lethal_indices(
            legal=legal,
            pipe_getter=pipe_getter,
            candidate_indices=candidate_indices,
            step_budget=combat_teacher_override.direct_lethal_step_budget,
        )
    action_idx, source = _select_combat_teacher_index(
        legal=legal,
        masked_scores=masked_scores,
        masked_logits=masked_logits,
        lethal_logit_blend_alpha=combat_teacher_override.lethal_logit_blend_alpha,
        direct_lethal_probe_top_k=combat_teacher_override.direct_lethal_probe_top_k,
        direct_lethal_probe=direct_lethal_probe,
    )
    if action_idx < len(legal):
        return action_idx, legal[action_idx], source
    return 0, legal[0], source


# ---------------------------------------------------------------------------
# Action selection strategies
# ---------------------------------------------------------------------------

def _select_action_nn(
    state: dict,
    legal: list[dict],
    ppo_net: FullRunPolicyNetworkV2,
    combat_net: CombatPolicyValueNetwork,
    combat_teacher_override: CombatTeacherOverride | None,
    combat_bc_override: CombatBcOverride | None,
    combat_mcts_agent: CombatMCTSAgent | None,
    combat_pipe_getter: Any | None,
    vocab: Vocab,
    device: torch.device,
    *,
    lethal_probe: bool = False,
    beam_search: int = 0,
    turn_planner: Any | None = None,
) -> tuple[int, dict, str, CombatMctsTrace | None]:
    """Select action using NN (argmax / deterministic).

    Non-combat: PPO network argmax.
    Combat: Combat NN argmax.
    Falls back to heuristic on any error.
    """
    st = (state.get("state_type") or "").lower()

    try:
        if st in COMBAT_SCREENS:
            if combat_mcts_agent is not None and combat_pipe_getter is not None:
                fm = None
                try:
                    fm = PipeCombatForwardModel.from_current_state(
                        combat_pipe_getter,
                        max_step_budget=getattr(combat_mcts_agent, "_max_step_budget", 200),
                    )
                    action, root = combat_mcts_agent.choose_action(fm)
                    action_idx = _match_legal_action_index(legal, action)
                    top_actions, root_value = _summarize_mcts_root(root)
                    trace_meta = CombatMctsTrace(
                        chosen_action=dict(action) if isinstance(action, dict) else {},
                        top_actions=top_actions,
                        sims=int(getattr(combat_mcts_agent.config, "num_simulations", 0)),
                        root_value=root_value,
                    )
                    if action_idx < len(legal):
                        return action_idx, legal[action_idx], "combat_mcts", trace_meta
                    return 0, legal[0], "combat_mcts", trace_meta
                finally:
                    if fm is not None:
                        try:
                            restored = fm.cleanup_and_restore()
                            if restored is None:
                                fm.cleanup()
                        except Exception:
                            try:
                                fm.cleanup()
                            except Exception:
                                pass
            if combat_teacher_override is not None and combat_teacher_override.mode == "replace":
                action_idx, action, action_source = _select_action_combat_teacher(
                    state=state,
                    legal=legal,
                    combat_teacher_override=combat_teacher_override,
                    pipe_getter=combat_pipe_getter,
                )
                return action_idx, action, action_source, None
            sf_t, af_t = _build_combat_tensors(state, legal, vocab, device, ppo_net=ppo_net)
            with torch.no_grad():
                logits, _value = combat_net(sf_t, af_t)
            # Mask invalid actions and argmax
            mask = af_t["action_mask"].float()
            logits = logits.squeeze(0)  # (MAX_ACTIONS,)
            logits = logits + (1.0 - mask.squeeze(0)) * (-1e9)
            base_logits = logits[:len(legal)].detach().cpu().numpy()
            if combat_bc_override is not None:
                bc_choice = _select_action_combat_bc(
                    state=state,
                    legal=legal,
                    combat_bc_override=combat_bc_override,
                    base_logits=base_logits,
                )
                if bc_choice is not None:
                    action_idx, action, action_source = bc_choice
                    return action_idx, action, action_source, None
            if combat_teacher_override is not None and combat_teacher_override.mode == "rerank":
                teacher_choice = _select_action_combat_teacher_rerank(
                    state=state,
                    legal=legal,
                    baseline_logits=base_logits,
                    combat_teacher_override=combat_teacher_override,
                    pipe_getter=combat_pipe_getter,
                )
                if teacher_choice[0] is not None and teacher_choice[1] is not None and teacher_choice[2] is not None:
                    action_idx, action, action_source = teacher_choice
                    return action_idx, action, action_source, None
            # Standalone lethal probe — only fires when NN chose end_turn
            # but play_card options exist (minimal probe to catch missed lethals)
            if lethal_probe and combat_pipe_getter is not None:
                nn_choice = int(np.argmax(base_logits)) if len(base_logits) > 0 else 0
                chosen_action_type = (legal[nn_choice].get("action") or "").lower() if nn_choice < len(legal) else ""
                if chosen_action_type == "end_turn":
                    play_indices = [
                        i for i, a in enumerate(legal)
                        if (a.get("action") or "").lower() == "play_card"
                    ]
                    if play_indices:
                        raw_pipe = combat_pipe_getter()
                        best_lethal_idx: int | None = None
                        sid = None
                        try:
                            sid = raw_pipe.call("save_state").get("state_id", "")
                            for idx in play_indices[:4]:
                                raw_pipe.call("load_state", {"state_id": sid})
                                probe_result = raw_pipe.call("step", legal[idx])
                                probe_state = probe_result.get("state", probe_result)
                                if _combat_probe_is_victory(probe_state):
                                    if best_lethal_idx is None or base_logits[idx] > base_logits[best_lethal_idx]:
                                        best_lethal_idx = idx
                            raw_pipe.call("load_state", {"state_id": sid})
                        except Exception:
                            best_lethal_idx = None
                        finally:
                            if sid:
                                try:
                                    raw_pipe.call("delete_state", {"state_id": sid})
                                except Exception:
                                    pass
                        if best_lethal_idx is not None:
                            return best_lethal_idx, legal[best_lethal_idx], "nn_lethal", None
            # Turn-level planner (policy-guided beam search over complete turns,
            # OR DFS turn solver, OR multi-turn lookahead — see turn_solver_planner.py
            # and combat_turn_planner.py for the available implementations).
            if turn_planner is not None and combat_pipe_getter is not None:
                planner_mode = getattr(turn_planner, "_mode", "boss_elite")
                should_plan = (
                    planner_mode == "always"
                    or (planner_mode == "boss" and st == "boss")
                    or (planner_mode == "elite" and st == "elite")
                    or (planner_mode == "boss_elite" and st in ("boss", "elite"))
                )
                if should_plan:
                    try:
                        result = turn_planner.select_action(
                            combat_pipe_getter, state, legal,
                        )
                        if result is not None:
                            pidx, psource = result
                            if pidx < len(legal):
                                return pidx, legal[pidx], psource, None
                    except Exception as e:
                        logger.debug("Turn planner failed: %s", e)

            # NOTE 2026-04-08 (wizardly cleanup): legacy --beam-search branch
            # removed; beam_search_combat.py archived. See parser comment.
            action_idx = int(np.argmax(base_logits)) if len(base_logits) > 0 else 0
        else:
            state_t, action_t = _build_ppo_tensors(state, legal, vocab, device)
            with torch.no_grad():
                logits, _values, _dq, _boss_ready, _action_adv = ppo_net(state_t, action_t)
            # Argmax (deterministic)
            logits = logits.squeeze(0)  # (MAX_ACTIONS,)
            action_idx = int(logits.argmax().item())

        if action_idx < len(legal):
            return action_idx, legal[action_idx], "nn", None
        return 0, legal[0], "nn", None

    except Exception as e:
        logger.debug("NN inference error, falling back to heuristic: %s", e)
        if st in COMBAT_SCREENS:
            action_idx, action = heuristic_combat_action(legal, state)
            return action_idx, action, "heuristic_fallback", None
        return 0, legal[0], "heuristic_fallback", None


def _select_action_random(
    state: dict, legal: list[dict],
) -> tuple[int, dict]:
    """Select a random legal action."""
    idx = random.randrange(len(legal))
    return idx, legal[idx]


def _select_action_heuristic(
    state: dict, legal: list[dict],
) -> tuple[int, dict]:
    """Select action using heuristic strategy.

    Combat: use the rule-based heuristic from train_hybrid.
    Non-combat: pick first legal action (usually a reasonable default).
    """
    st = (state.get("state_type") or "").lower()
    if st in COMBAT_SCREENS:
        return heuristic_combat_action(legal, state)
    # Non-combat: first action is usually "proceed" or a reasonable choice
    return 0, legal[0]


def _raw_action_key(action: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if not isinstance(action, dict):
        return None
    return (
        _lower(action.get("action")),
        action.get("card_index"),
        action.get("slot"),
        action.get("target"),
        action.get("index"),
        action.get("screen_type"),
    )


def _match_legal_action_index(legal: list[dict[str, Any]], action: dict[str, Any] | None) -> int:
    if not legal or not isinstance(action, dict):
        return 0
    raw_key = _raw_action_key(action)
    if raw_key is not None:
        for idx, legal_action in enumerate(legal):
            if _raw_action_key(legal_action) == raw_key:
                return idx
    normalized = normalize_action(action)
    normalized_key = _raw_action_key(normalized)
    if normalized_key is not None:
        for idx, legal_action in enumerate(legal):
            if _raw_action_key(normalize_action(legal_action)) == normalized_key:
                return idx
    action_name = _lower(action.get("action"))
    label = str(action.get("label") or "")
    for idx, legal_action in enumerate(legal):
        if _lower(legal_action.get("action")) == action_name and str(legal_action.get("label") or "") == label:
            return idx
    for idx, legal_action in enumerate(legal):
        if _lower(legal_action.get("action")) == action_name:
            return idx
    return 0


def _summarize_mcts_root(root: Any, k: int = 3) -> tuple[list[dict[str, Any]], float]:
    if root is None or not getattr(root, "children", None):
        return [], 0.0
    total_visits = max(1, int(sum(child.visit_count for child in root.children.values())))
    ranked = sorted(
        root.children.values(),
        key=lambda child: child.visit_count,
        reverse=True,
    )[: max(1, int(k))]
    summary: list[dict[str, Any]] = []
    for child in ranked:
        action = child.action if isinstance(child.action, dict) else {}
        summary.append(
            {
                "action": dict(action),
                "action_name": str(action.get("action") or ""),
                "label": str(action.get("label") or action.get("action") or ""),
                "target": action.get("target") or action.get("target_id"),
                "visits": int(child.visit_count),
                "visit_frac": round(float(child.visit_count) / total_visits, 4),
                "q": round(float(child.q_value), 4),
                "prior": round(float(child.prior), 4),
            }
        )
    return summary, round(float(root.q_value), 4)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_combat_bc_patch_config(payload: dict[str, Any]) -> CombatBcPatchConfig:
    raw_patch = payload.get("patch_config")
    if not isinstance(raw_patch, dict):
        raw_patch = DEFAULT_COMBAT_BC_PATCH_CONFIG
    return CombatBcPatchConfig(
        mode=str(raw_patch.get("mode") or DEFAULT_COMBAT_BC_PATCH_CONFIG["mode"]).strip().lower(),
        alpha=_safe_float(raw_patch.get("alpha"), DEFAULT_COMBAT_BC_PATCH_CONFIG["alpha"]),
        require_margin=bool(raw_patch.get("require_margin", DEFAULT_COMBAT_BC_PATCH_CONFIG["require_margin"])),
        min_margin_zscore=_safe_float(
            raw_patch.get("min_margin_zscore"),
            DEFAULT_COMBAT_BC_PATCH_CONFIG["min_margin_zscore"],
        ),
        max_base_top_prob=_safe_float(
            raw_patch.get("max_base_top_prob"),
            DEFAULT_COMBAT_BC_PATCH_CONFIG["max_base_top_prob"],
        ),
    )


def _load_combat_bc_override(path: str | Path) -> CombatBcOverride:
    model_path = Path(path)
    payload = json.loads(model_path.read_text(encoding="utf-8-sig"))
    gate_constraints = payload.get("gate_constraints")
    if not isinstance(gate_constraints, dict):
        gate_constraints = dict(DEFAULT_COMBAT_BC_GATE_CONSTRAINTS)
    return CombatBcOverride(
        policy=BehaviorCloningLinearPolicy.load(model_path),
        model_path=str(model_path),
        gate_constraints=gate_constraints,
        patch_config=_parse_combat_bc_patch_config(payload),
    )


def _normalize_live_combat_action(
    raw_action: dict[str, Any],
    raw_state: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(raw_action, dict):
        return None

    action_name = _lower(raw_action.get("action"))
    if not action_name:
        normalized = normalize_action(raw_action)
        return normalized if normalized.get("type") else None

    if action_name == "play_card":
        normalized: dict[str, Any] = {
            "type": "play_card",
            "hand_index": _safe_int(raw_action.get("card_index"), -1),
        }
        target = raw_action.get("target")
        if target is not None:
            for enemy in (raw_state.get("battle") or {}).get("enemies", []):
                if str(enemy.get("entity_id")) == str(target):
                    normalized["target_id"] = _safe_int(enemy.get("combat_id"), -1)
                    break
        return normalize_action(normalized)

    if action_name == "use_potion":
        normalized = {
            "type": "use_potion",
            "slot": _safe_int(raw_action.get("slot"), -1),
        }
        target = raw_action.get("target")
        if target is not None:
            for enemy in (raw_state.get("battle") or {}).get("enemies", []):
                if str(enemy.get("entity_id")) == str(target):
                    normalized["target_id"] = _safe_int(enemy.get("combat_id"), -1)
                    break
        return normalize_action(normalized)

    if action_name == "end_turn":
        return {"type": "end_turn"}

    if action_name in {"combat_select_card", "select_card"}:
        return normalize_action(
            {
                "type": "select_card_option",
                "option_index": _safe_int(raw_action.get("card_index"), -1),
            }
        )

    if action_name in {"combat_confirm_selection", "confirm_selection"}:
        return {"type": "confirm_selection"}

    if action_name == "cancel_selection":
        return {"type": "cancel_selection"}

    normalized = normalize_action(raw_action)
    return normalized if normalized.get("type") else None


def _combat_bc_gate_allows_state(
    state: dict[str, Any],
    gate_constraints: dict[str, Any] | None,
) -> bool:
    if not isinstance(gate_constraints, dict) or not gate_constraints:
        return True

    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    floor = _safe_int(run.get("floor"), 0)
    state_type = _lower(state.get("state_type"))

    min_floor = gate_constraints.get("min_floor")
    if min_floor is not None and floor < _safe_int(min_floor, 0):
        return False

    max_floor = gate_constraints.get("max_floor")
    if max_floor is not None and floor > _safe_int(max_floor, floor):
        return False

    room_types = gate_constraints.get("room_types")
    if isinstance(room_types, list) and room_types:
        allowed = {_lower(item) for item in room_types if str(item).strip()}
        if allowed and state_type not in allowed:
            return False

    alive_enemy_names = _alive_enemy_names_from_live_state(state)
    alive_enemy_count = len(alive_enemy_names)

    min_alive_enemies = gate_constraints.get("min_alive_enemies")
    if min_alive_enemies is not None and alive_enemy_count < _safe_int(min_alive_enemies, 0):
        return False

    max_alive_enemies = gate_constraints.get("max_alive_enemies")
    if max_alive_enemies is not None and alive_enemy_count > _safe_int(max_alive_enemies, alive_enemy_count):
        return False

    enemy_name_tokens_any = gate_constraints.get("enemy_name_tokens_any")
    if isinstance(enemy_name_tokens_any, list) and enemy_name_tokens_any:
        wanted_tokens = [_lower(item) for item in enemy_name_tokens_any if str(item).strip()]
        if wanted_tokens and not any(
            token in enemy_name
            for token in wanted_tokens
            for enemy_name in alive_enemy_names
        ):
            return False

    return True


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64, copy=True)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def _safe_softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64, copy=True)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if total <= 0.0:
        return np.full_like(values, fill_value=1.0 / max(1, values.size), dtype=np.float64)
    return exp_values / total


def _build_combat_bc_bonus(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    combat_bc_override: CombatBcOverride,
) -> np.ndarray | None:
    adapted_state = adapt_v1_state_for_combat_policy(state)
    supported_indices: list[int] = []
    normalized_actions: list[dict[str, Any]] = []

    for idx, candidate in enumerate(legal):
        normalized = _normalize_live_combat_action(candidate, state)
        if not isinstance(normalized, dict) or not normalized.get("type"):
            continue
        supported_indices.append(idx)
        normalized_actions.append(normalized)

    if not normalized_actions:
        return None

    bc_scores = combat_bc_override.policy.score_actions(adapted_state, normalized_actions)
    bc_bonus = np.zeros((len(legal),), dtype=np.float64)
    supported_bonus = _safe_zscore(np.asarray(bc_scores, dtype=np.float64))
    for idx, bonus in zip(supported_indices, supported_bonus, strict=False):
        bc_bonus[idx] = float(bonus)

    return bc_bonus


def _select_action_combat_bc(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    combat_bc_override: CombatBcOverride,
    base_logits: np.ndarray,
) -> tuple[int, dict[str, Any], str] | None:
    if not _combat_bc_gate_allows_state(state, combat_bc_override.gate_constraints):
        return None

    patch_cfg = combat_bc_override.patch_config
    if patch_cfg.mode != "rerank" or patch_cfg.alpha <= 0.0:
        return None

    bonus_payload = _build_combat_bc_bonus(
        state=state,
        legal=legal,
        combat_bc_override=combat_bc_override,
    )
    if bonus_payload is None:
        return None

    bc_bonus = bonus_payload
    if bc_bonus.size != len(legal):
        return None

    top_bc_idx = int(np.argmax(bc_bonus)) if bc_bonus.size else 0
    if patch_cfg.require_margin:
        if bc_bonus.size < 2:
            return None
        ordered = np.sort(bc_bonus)
        margin = float(ordered[-1] - ordered[-2])
        if margin < patch_cfg.min_margin_zscore:
            return None

    if patch_cfg.max_base_top_prob < 0.999999:
        base_probs = _safe_softmax(np.asarray(base_logits, dtype=np.float64))
        if float(np.max(base_probs)) > patch_cfg.max_base_top_prob:
            return None

    combined_logits = np.asarray(base_logits, dtype=np.float64) + patch_cfg.alpha * bc_bonus
    action_idx = int(np.argmax(combined_logits))
    action_source = "combat_bc_rerank_gated" if patch_cfg.require_margin else "combat_bc_rerank"
    if top_bc_idx == action_idx:
        return action_idx, legal[action_idx], action_source

    return action_idx, legal[action_idx], action_source


def _record_last_state_snapshot(
    result: GameResult,
    state_type: str,
    floor: int,
    legal_count: int,
) -> None:
    result.last_state_type = (state_type or "").strip().lower()
    result.last_state_floor = int(floor)
    result.last_state_legal_action_count = int(legal_count)


def _record_timeout_snapshot(result: GameResult) -> None:
    result.timeout_state_type = result.last_state_type
    result.timeout_floor = int(result.last_state_floor)
    result.timeout_legal_action_count = int(result.last_state_legal_action_count)


def _increment_named_count(bucket: dict[str, int], name: str | None) -> None:
    key = str(name or "").strip()
    if not key:
        return
    bucket[key] = int(bucket.get(key, 0)) + 1


def _infer_ppo_embed_dim(
    state_dict: dict[str, Any] | None,
    fallback: int,
) -> int:
    if isinstance(state_dict, dict):
        weight = state_dict.get("entity_emb.card_embed.weight")
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[1])
    return int(fallback)


def _infer_combat_dims(
    state_dict: dict[str, Any] | None,
    fallback_embed_dim: int,
    fallback_hidden_dim: int,
) -> tuple[int, int]:
    embed_dim = int(fallback_embed_dim)
    hidden_dim = int(fallback_hidden_dim)
    if isinstance(state_dict, dict):
        card_weight = state_dict.get("entity_emb.card_embed.weight")
        action_proj = state_dict.get("action_proj.weight")
        if isinstance(card_weight, torch.Tensor) and card_weight.ndim == 2:
            embed_dim = int(card_weight.shape[1])
        if isinstance(action_proj, torch.Tensor) and action_proj.ndim == 2:
            hidden_dim = int(action_proj.shape[0])
    return embed_dim, hidden_dim


def _safe_load_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    label: str,
) -> None:
    """Strict=False load that filters out shape-mismatched keys.

    Note 2026-04-08 PM (wizardly merge): the previous `_grow_linear_input_dim`
    backward-compat shim (zero-pad trailing input dims to handle the
    `ENEMY_AUX_DIM 16 → 32` and `COMBAT_EXTRA_SCALAR_DIM=14` end-grows) was
    removed along with all PPO900-era checkpoints. Per the no-compat memory,
    this loader now only handles "shape matches exactly", "key doesn't exist
    on this side, skip it" — and (2026-04-09) a narrow Linear INPUT-dim
    partial-copy path for the retrieval-head encoder expansion, matching the
    logic in train_hybrid.py::_safe_load_state_dict.
    """
    current = model.state_dict()
    filtered: dict[str, Any] = {}
    skipped: list[str] = []
    partial: list[str] = []
    for key, value in state_dict.items():
        if key not in current:
            continue
        if current[key].shape == value.shape:
            filtered[key] = value
            continue
        cur = current[key]
        # Linear weight: (out, in_new) vs (out, in_old) — partial copy old cols,
        # zero-init new cols. This is what makes retrieval-on checkpoints load
        # into a retrieval-off model (though we should normally auto-detect
        # and build retrieval-on, see _infer_retrieval_proj_dim below).
        if value.ndim == 2 and cur.ndim == 2 and cur.shape[0] == value.shape[0]:
            if cur.shape[1] > value.shape[1]:
                new_w = torch.zeros_like(cur)
                new_w[:, :value.shape[1]] = value
                filtered[key] = new_w
                partial.append(f"{key}: {list(value.shape)}->{list(cur.shape)}")
                continue
            if cur.shape[1] < value.shape[1]:
                # Narrow truncation: model is smaller than ckpt. Take the
                # first model.in_features columns — only safe if downstream
                # behavior expects a narrower input. For retrieval eval this
                # happens when an old non-retrieval script loads a retrieval
                # checkpoint; we LOG it but still copy.
                new_w = value[:, :cur.shape[1]].contiguous().clone()
                filtered[key] = new_w
                partial.append(f"{key}: {list(value.shape)}->truncated to {list(cur.shape)}")
                continue
        skipped.append(
            f"{key}: ckpt={list(value.shape)} vs model={list(current[key].shape)}"
        )
    if partial:
        logger.info(
            "Partial-loaded %d expanded params in %s: %s",
            len(partial), label, "; ".join(partial[:5]),
        )
    if skipped:
        logger.warning(
            "Skipped %d mismatched keys in %s: %s",
            len(skipped),
            label,
            "; ".join(skipped[:5]),
        )
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        logger.info("New params in %s (random init): %d keys", label, len(missing))
    if unexpected:
        logger.warning("Unexpected params ignored in %s: %d keys", label, len(unexpected))


def _infer_retrieval_proj_dim(state_dict: dict[str, Any]) -> int:
    """Detect whether a checkpoint was trained with --retrieval-head.

    Returns the symbolic proj_dim if the head is present, else 0.
    Looks for `symbolic_head.out_proj.weight` which has shape (proj_dim, embed_dim).
    """
    if not isinstance(state_dict, dict):
        return 0
    out_proj = state_dict.get("symbolic_head.out_proj.weight")
    if isinstance(out_proj, torch.Tensor) and out_proj.ndim == 2:
        return int(out_proj.shape[0])
    return 0


def _dispatch_action_with_fallbacks(
    client: PipeBackedFullRunClient | ApiBackedFullRunClient,
    action: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    attempts: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append_attempt(candidate: dict[str, Any] | None) -> None:
        if not isinstance(candidate, dict):
            return
        key = json.dumps(_json_safe_tree(candidate), sort_keys=True, ensure_ascii=True)
        if key in seen:
            return
        seen.add(key)
        attempts.append(candidate)

    _append_attempt(action)
    stripped = {k: v for k, v in action.items() if k not in ("target_id", "slot", "target")}
    _append_attempt(stripped)
    _append_attempt({"action": "end_turn"})

    errors: list[str] = []
    for candidate in attempts:
        try:
            return client.act(candidate), candidate, ""
        except Exception as exc:
            errors.append(f"{candidate.get('action', '?')}:{exc}")

    try:
        state = client.get_state()
        return state, None, "; ".join(errors)
    except Exception as exc:
        errors.append(f"get_state:{exc}")
        return None, None, "; ".join(errors)


# ---------------------------------------------------------------------------
# Single game runner
# ---------------------------------------------------------------------------

def run_single_game(
    client: PipeBackedFullRunClient,
    strategy: str,
    game_id: int,
    seed: str,
    character_id: str,
    ascension_level: int,
    game_timeout: float,
    max_steps: int,
    # NN-specific (only used when strategy == "nn")
    ppo_net: FullRunPolicyNetworkV2 | None = None,
    combat_net: CombatPolicyValueNetwork | None = None,
    combat_teacher_override: CombatTeacherOverride | None = None,
    combat_bc_override: CombatBcOverride | None = None,
    combat_mcts_agent: CombatMCTSAgent | None = None,
    vocab: Vocab | None = None,
    device: torch.device | None = None,
    capture_trace: bool = False,
    capture_trajectory: bool = False,
    lethal_probe: bool = False,
    beam_search: int = 0,
    turn_planner: Any | None = None,
) -> tuple[GameResult, list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    """Run one game and return the result."""
    result = GameResult(
        game_id=game_id, seed=seed, strategy=strategy,
    )
    start_time = time.monotonic()
    run_id = f"eval-{strategy}-{seed}-{uuid.uuid4().hex[:12]}"
    trace_steps: list[dict[str, Any]] | None = [] if capture_trace else None
    trajectory_records: list[dict[str, Any]] | None = [] if capture_trajectory else None

    try:
        state = client.reset(
            character_id=character_id,
            ascension_level=ascension_level,
            seed=seed,
            timeout_s=30.0,
        )
    except Exception as e:
        # Retry once after reconnect
        try:
            if hasattr(client, "_reconnect"):
                client._reconnect()
            state = client.reset(
                character_id=character_id,
                ascension_level=ascension_level,
                seed=seed,
                timeout_s=30.0,
            )
        except Exception as e2:
            result.outcome = "error"
            result.error_msg = f"reset failed: {e2}"
            result.time_taken_s = time.monotonic() - start_time
            return result, trace_steps, trajectory_records

    in_combat = False
    combats_entered = 0
    loop_tracker = RepeatLoopTracker()
    last_reward_claim_sig = ""
    st = (state.get("state_type") or "").lower()
    floor = _safe_int((state.get("run") or {}).get("floor", 0), 0)
    legal: list[dict[str, Any]] = []
    result.next_boss_token = extract_next_boss_token(state)

    def _record_build_diagnostics(current_state: dict[str, Any]) -> None:
        token = extract_next_boss_token(current_state)
        if token != "unknown":
            result.next_boss_token = token
        readiness = boss_readiness_score(current_state)
        run_state = current_state.get("run") if isinstance(current_state.get("run"), dict) else {}
        current_floor = _safe_int(run_state.get("floor", 0), 0)
        if current_floor >= 8 and result.boss_readiness_at_floor_8 is None:
            result.boss_readiness_at_floor_8 = readiness
        if current_floor >= 12 and result.boss_readiness_at_floor_12 is None:
            result.boss_readiness_at_floor_12 = readiness
        if current_floor >= 16 and result.boss_readiness_at_floor_16 is None:
            result.boss_readiness_at_floor_16 = readiness

    for step_i in range(max_steps):
        elapsed = time.monotonic() - start_time
        if elapsed > game_timeout:
            _record_last_state_snapshot(
                result,
                state.get("state_type") or "",
                _safe_int((state.get("run") or {}).get("floor", result.max_floor), result.max_floor),
                len(state.get("legal_actions") or []),
            )
            result.outcome = "timeout"
            _record_timeout_snapshot(result)
            break

        st = (state.get("state_type") or "").lower()

        # Terminal check
        if st == "game_over" or state.get("terminal"):
            if trace_steps is not None:
                trace_steps.append(
                    _make_trace_step(
                        state,
                        state.get("legal_actions") or [],
                        None,
                        "terminal",
                        result.total_steps,
                    )
                )
            go = state.get("game_over") or {}
            outcome_str = (go.get("run_outcome") or go.get("outcome") or "").lower()
            if "victory" in outcome_str or outcome_str == "win":
                result.outcome = "victory"
            else:
                result.outcome = "death"
                result.floor_at_death = result.max_floor
            # Extract final HP
            player = go.get("player") or state.get("player") or {}
            result.final_hp = int(player.get("hp", player.get("current_hp", 0)))
            result.final_max_hp = int(player.get("max_hp", 0))
            if result.boss_reached:
                result.boss_hp_fraction_dealt = max(
                    result.boss_hp_fraction_dealt, _estimate_boss_hp_fraction(state),
                )
            if result.outcome == "victory":
                result.act1_cleared = True
                if result.boss_reached:
                    result.boss_hp_fraction_dealt = 1.0
            break

        # Track floor
        run = state.get("run") or {}
        floor = int(run.get("floor", 0))
        result.max_floor = max(result.max_floor, floor)
        _record_build_diagnostics(state)
        if _safe_int(run.get("act", 1), 1) > 1 or result.max_floor >= WIN_FLOOR:
            result.act1_cleared = True

        if st == "boss":
            if not result.boss_reached:
                # Capture boss-entry cohort snapshot once
                _be_progress = _extract_progress(state)
                result.boss_entry_floor = _safe_int(_be_progress.get("floor"), 0)
                result.boss_entry_hp = _safe_int(_be_progress.get("hp"), 0)
                result.boss_entry_max_hp = _safe_int(_be_progress.get("max_hp"), 0)
                result.boss_entry_deck_count = _safe_int(_be_progress.get("deck_count"), 0)
                result.boss_entry_relic_count = _safe_int(_be_progress.get("relic_count"), 0)
                result.boss_entry_potion_count = _safe_int(_be_progress.get("potion_count"), 0)
                result.boss_entry_gold = _safe_int(_be_progress.get("gold"), 0)
                result.boss_entry_token = extract_next_boss_token(state)
            result.boss_reached = True
            result.boss_hp_fraction_dealt = max(
                result.boss_hp_fraction_dealt, _estimate_boss_hp_fraction(state),
            )

        # Track combat transitions
        if st in COMBAT_SCREENS:
            if not in_combat:
                in_combat = True
                combats_entered += 1
        else:
            if in_combat:
                in_combat = False
                result.num_combats_won += 1  # survived combat

        # Get legal actions
        legal = state.get("legal_actions", [])
        legal = [
            a for a in legal
            if isinstance(a, dict) and a.get("is_enabled") is not False
        ]
        _record_last_state_snapshot(result, st, floor, len(legal))
        if not legal:
            dispatched_action = {"action": "wait"}
            next_state, executed_action, dispatch_error = _dispatch_action_with_fallbacks(
                client,
                dispatched_action,
            )
            if next_state is None:
                result.outcome = "error"
                result.error_msg = f"step {step_i}: {dispatch_error}"
                break
            action_for_logging = executed_action or dispatched_action
            action_source = "auto_wait"
            if executed_action is not None and executed_action != dispatched_action:
                action_source = "auto_wait:fallback"
            elif executed_action is None and dispatch_error:
                action_source = "auto_wait:state_refresh"
            if trace_steps is not None:
                trace_steps.append(
                    _make_trace_step(
                        state,
                        legal,
                        action_for_logging,
                        action_source,
                        result.total_steps,
                    )
                )
            if trajectory_records is not None and executed_action is not None:
                trajectory_records.append(
                    _make_trajectory_record(
                        run_id=run_id,
                        seed=seed,
                        step_index=result.total_steps,
                        state=state,
                        legal=legal,
                        chosen_action=executed_action,
                        action_source=action_source,
                        next_state=next_state,
                    )
                )
            state = next_state
            last_reward_claim_sig = ""
            result.total_steps += 1
            continue

        auto_action = _choose_auto_progress_action(state, st, legal, last_reward_claim_sig)
        abort_after_action = False
        action_source = strategy
        combat_mcts_trace = None
        if auto_action is not None:
            action_idx, action = 0, auto_action
            action_source = "auto_progress"
        else:
            loop_tracker.observe(st, legal)
            escape_action = loop_tracker.choose_escape_action(legal)
            if escape_action is not None:
                action_idx, action = 0, escape_action
                abort_after_action = loop_tracker.should_abort()
                action_source = "repeat_escape"
            elif strategy == "nn":
                action_idx, action, action_source, combat_mcts_trace = _select_action_nn(
                    state,
                    legal,
                    ppo_net,
                    combat_net,
                    combat_teacher_override,
                    combat_bc_override,
                    combat_mcts_agent,
                    (lambda c=client: c._pipe) if hasattr(client, "_pipe") else None,
                    vocab,
                    device,
                    lethal_probe=lethal_probe,
                    beam_search=beam_search,
                    turn_planner=turn_planner,
                )
            elif strategy == "random":
                action_idx, action = _select_action_random(state, legal)
                combat_mcts_trace = None
            elif strategy == "heuristic":
                action_idx, action = _select_action_heuristic(state, legal)
                combat_mcts_trace = None
            else:
                action_idx, action = 0, legal[0]
                combat_mcts_trace = None

        next_state, executed_action, dispatch_error = _dispatch_action_with_fallbacks(
            client,
            action,
        )
        if next_state is None:
            result.outcome = "error"
            result.error_msg = f"step {step_i}: {dispatch_error}"
            break

        action_for_logging = executed_action or action
        effective_action_source = action_source
        if executed_action is not None and executed_action != action:
            effective_action_source = f"{action_source}:fallback"
        elif executed_action is None and dispatch_error:
            effective_action_source = f"{action_source}:state_refresh"
        _increment_named_count(result.action_source_counts, effective_action_source)
        if effective_action_source.startswith("combat_teacher_rerank"):
            _increment_named_count(
                result.combat_teacher_override_counts,
                effective_action_source,
            )

        if trace_steps is not None:
            trace_steps.append(
                _make_trace_step(
                    state,
                    legal,
                    action_for_logging,
                    effective_action_source,
                    result.total_steps,
                    combat_mcts_trace=combat_mcts_trace,
                )
            )
        if st == "card_reward":
            result.card_reward_screens += 1
            executed_name = _lower((action_for_logging or {}).get("action"))
            if executed_name in {"skip", "skip_card_reward"}:
                result.card_reward_skips += 1
        if trajectory_records is not None and executed_action is not None:
            trajectory_records.append(
                _make_trajectory_record(
                    run_id=run_id,
                    seed=seed,
                    step_index=result.total_steps,
                    state=state,
                    legal=legal,
                    chosen_action=executed_action,
                    action_source=effective_action_source,
                    next_state=next_state,
                )
            )
        last_reward_claim_sig = _next_reward_claim_signature(st, state, action_for_logging)
        state = next_state

        result.total_steps += 1
        if abort_after_action:
            result.outcome = "timeout"
            result.error_msg = f"repeat_loop:{st}"
            _record_timeout_snapshot(result)
            break

    else:
        # Exhausted max_steps without terminal
        if not result.outcome:
            result.outcome = "timeout"
            _record_timeout_snapshot(result)

    # Finalize timing
    result.time_taken_s = time.monotonic() - start_time

    # If we never set outcome (loop broke without terminal)
    if not result.outcome:
        _record_last_state_snapshot(
            result,
            state.get("state_type") or "",
            _safe_int((state.get("run") or {}).get("floor", result.max_floor), result.max_floor),
            len(state.get("legal_actions") or []),
        )
        result.outcome = "timeout"
        _record_timeout_snapshot(result)

    # Extract final HP from last state if not yet set
    if result.final_hp == 0 and result.outcome != "error":
        try:
            player = state.get("player") or {}
            result.final_hp = int(player.get("hp", player.get("current_hp", 0)))
            result.final_max_hp = int(player.get("max_hp", 0))
        except Exception:
            pass

    if result.act1_cleared and result.boss_hp_fraction_dealt <= 0:
        result.boss_hp_fraction_dealt = 1.0

    return result, trace_steps, trajectory_records


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    client: PipeBackedFullRunClient,
    strategy: str,
    seeds: list[str],
    character_id: str,
    ascension_level: int,
    game_timeout: float,
    max_steps: int,
    ppo_net: FullRunPolicyNetworkV2 | None = None,
    combat_net: CombatPolicyValueNetwork | None = None,
    combat_teacher_override: CombatTeacherOverride | None = None,
    combat_bc_override: CombatBcOverride | None = None,
    combat_mcts_agent: CombatMCTSAgent | None = None,
    vocab: Vocab | None = None,
    device: torch.device | None = None,
    trace_seeds: set[str] | None = None,
    trajectory_seeds: set[str] | None = None,
    lethal_probe: bool = False,
    beam_search: int = 0,
    turn_planner: Any | None = None,
) -> tuple[
    list[GameResult],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    """Run N games and return results.  Prints progress every 10 games."""
    results: list[GameResult] = []
    trace_payloads: dict[str, list[dict[str, Any]]] = {}
    trajectory_payloads: dict[str, list[dict[str, Any]]] = {}
    num_games = len(seeds)

    logger.info(
        "Starting evaluation: strategy=%s, num_games=%d, character=%s, asc=%d",
        strategy, num_games, character_id, ascension_level,
    )

    for i, seed in enumerate(seeds):
        try:
            result, trace_steps, trajectory_records = run_single_game(
                client=client,
                strategy=strategy,
                game_id=i + 1,
                seed=seed,
                character_id=character_id,
                ascension_level=ascension_level,
                game_timeout=game_timeout,
                max_steps=max_steps,
                ppo_net=ppo_net,
                combat_net=combat_net,
                combat_teacher_override=combat_teacher_override,
                combat_bc_override=combat_bc_override,
                combat_mcts_agent=combat_mcts_agent,
                vocab=vocab,
                device=device,
                capture_trace=bool(trace_seeds and seed in trace_seeds),
                capture_trajectory=bool(trajectory_seeds and seed in trajectory_seeds),
                lethal_probe=lethal_probe,
                beam_search=beam_search,
                turn_planner=turn_planner,
            )
        except Exception as e:
            logger.warning("Game %d/%d crashed: %s", i + 1, num_games, e)
            result = GameResult(
                game_id=i + 1, seed=seed, strategy=strategy,
                outcome="error", error_msg=str(e),
            )
            trace_steps = None
            trajectory_records = None
        results.append(result)
        if trace_steps is not None:
            trace_payloads[seed] = trace_steps
        if trajectory_records is not None:
            trajectory_payloads[seed] = trajectory_records

        # Progress log every 10 games
        if (i + 1) % 10 == 0 or (i + 1) == num_games:
            completed = [r for r in results if r.outcome not in ("error",)]
            avg_floor = (
                sum(r.max_floor for r in completed) / len(completed)
                if completed else 0
            )
            errors = sum(1 for r in results if r.outcome == "error")
            logger.info(
                "[%s] %d/%d games done | avg_floor=%.1f | errors=%d",
                strategy, i + 1, num_games, avg_floor, errors,
            )

    return results, trace_payloads, trajectory_payloads


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(results: list[GameResult]) -> dict[str, Any]:
    """Compute summary statistics from a list of GameResults."""
    valid = [r for r in results if r.outcome != "error"]
    if not valid:
        return {
            "strategy": results[0].strategy if results else "?",
            "total_games": len(results),
            "valid_games": 0,
            "error_count": len(results),
            "death_count": 0,
            "timeout_count": 0,
            "boss_reach_count": 0,
            "act1_clear_count": 0,
            "avg_floor": 0.0,
            "median_floor": 0,
            "max_floor": 0,
            "min_floor": 0,
            "std_floor": 0.0,
            "win_rate": 0.0,
            "wins": 0,
            "boss_reach_rate": 0.0,
            "act1_clear_rate": 0.0,
            "avg_boss_hp_fraction_dealt": 0.0,
            "avg_boss_readiness_at_floor_8": 0.0,
            "avg_boss_readiness_at_floor_12": 0.0,
            "avg_boss_readiness_at_floor_16": 0.0,
            "avg_floor_at_death": 0.0,
            "avg_floor_at_timeout": 0.0,
            "avg_hp_at_death": 0.0,
            "avg_combats_won": 0.0,
            "avg_steps": 0.0,
            "avg_time_s": 0.0,
            "total_time_s": 0.0,
            "card_reward_skip_rate_by_boss": {},
            "action_source_counts": {},
            "combat_teacher_override_counts": {},
            "avg_combat_teacher_overrides_per_game": 0.0,
            "games_with_combat_teacher_override": 0,
        }

    floors = [r.max_floor for r in valid]
    hps = [r.final_hp for r in valid]
    times = [r.time_taken_s for r in valid]
    steps = [r.total_steps for r in valid]
    act1_clears = [r for r in valid if r.act1_cleared]
    boss_games = [r for r in valid if r.boss_reached]
    deaths = [r for r in valid if r.outcome == "death"]
    timeouts = [r for r in valid if r.outcome == "timeout"]
    readiness_8 = [r.boss_readiness_at_floor_8 for r in valid if r.boss_readiness_at_floor_8 is not None]
    readiness_12 = [r.boss_readiness_at_floor_12 for r in valid if r.boss_readiness_at_floor_12 is not None]
    readiness_16 = [r.boss_readiness_at_floor_16 for r in valid if r.boss_readiness_at_floor_16 is not None]
    skip_by_boss: dict[str, dict[str, int]] = {}
    action_source_counts: dict[str, int] = {}
    combat_teacher_override_counts: dict[str, int] = {}
    games_with_teacher_override = 0
    for result in valid:
        token = result.next_boss_token or "unknown"
        bucket = skip_by_boss.setdefault(token, {"screens": 0, "skips": 0})
        bucket["screens"] += int(result.card_reward_screens)
        bucket["skips"] += int(result.card_reward_skips)
        if result.combat_teacher_override_counts:
            games_with_teacher_override += 1
        for source, count in (result.action_source_counts or {}).items():
            action_source_counts[str(source)] = action_source_counts.get(str(source), 0) + int(count)
        for source, count in (result.combat_teacher_override_counts or {}).items():
            combat_teacher_override_counts[str(source)] = (
                combat_teacher_override_counts.get(str(source), 0) + int(count)
            )

    return {
        "strategy": valid[0].strategy if valid else "?",
        "total_games": len(results),
        "valid_games": len(valid),
        "error_count": len(results) - len(valid),
        "death_count": len(deaths),
        "timeout_count": len(timeouts),
        "boss_reach_count": len(boss_games),
        "act1_clear_count": len(act1_clears),
        "avg_floor": round(float(np.mean(floors)), 2),
        "median_floor": int(np.median(floors)),
        "max_floor": max(floors),
        "min_floor": min(floors),
        "std_floor": round(float(np.std(floors)), 2),
        "win_rate": round(len(act1_clears) / len(valid), 4),
        "wins": len(act1_clears),
        "boss_reach_rate": round(len(boss_games) / len(valid), 4),
        "act1_clear_rate": round(len(act1_clears) / len(valid), 4),
        "avg_boss_hp_fraction_dealt": round(
            float(np.mean([r.boss_hp_fraction_dealt for r in boss_games])), 4,
        ) if boss_games else 0.0,
        "avg_boss_readiness_at_floor_8": round(float(np.mean(readiness_8)), 4) if readiness_8 else 0.0,
        "avg_boss_readiness_at_floor_12": round(float(np.mean(readiness_12)), 4) if readiness_12 else 0.0,
        "avg_boss_readiness_at_floor_16": round(float(np.mean(readiness_16)), 4) if readiness_16 else 0.0,
        "avg_floor_at_death": round(
            float(np.mean([r.floor_at_death for r in deaths])), 2,
        ) if deaths else 0.0,
        "avg_floor_at_timeout": round(
            float(np.mean([r.timeout_floor for r in timeouts])), 2,
        ) if timeouts else 0.0,
        "avg_hp_at_death": round(float(np.mean(hps)), 1),
        "avg_combats_won": round(
            float(np.mean([r.num_combats_won for r in valid])), 1
        ),
        "avg_steps": round(float(np.mean(steps)), 1),
        "avg_time_s": round(float(np.mean(times)), 2),
        "total_time_s": round(sum(times), 1),
        "card_reward_skip_rate_by_boss": {
            token: round(bucket["skips"] / max(1, bucket["screens"]), 4)
            for token, bucket in sorted(skip_by_boss.items())
        },
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "combat_teacher_override_counts": dict(sorted(combat_teacher_override_counts.items())),
        "avg_combat_teacher_overrides_per_game": round(
            float(sum(combat_teacher_override_counts.values())) / max(1, len(valid)),
            2,
        ),
        "games_with_combat_teacher_override": games_with_teacher_override,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a formatted summary block."""
    s = summary
    strategy = s.get("strategy", "?")
    print(f"\n{'=' * 60}")
    print(f"  Strategy: {strategy.upper()}")
    print(f"{'=' * 60}")
    print(f"  Games:       {s['valid_games']} valid / {s['total_games']} total "
          f"({s['error_count']} errors)")
    print(f"  Outcomes:    {s['death_count']} death / {s['timeout_count']} timeout")
    print(f"  Win rate:    {s['win_rate'] * 100:.1f}%  "
          f"({s['wins']} wins, floor >= {WIN_FLOOR})")
    print(f"  Boss reach:  {s['boss_reach_rate'] * 100:.1f}%")
    print(f"  Act1 clear:  {s['act1_clear_rate'] * 100:.1f}%")
    print(f"  Boss HP:     {s['avg_boss_hp_fraction_dealt']:.2f} avg dealt")
    print(
        f"  Readiness:   f8={s.get('avg_boss_readiness_at_floor_8', 0.0):.2f} "
        f"f12={s.get('avg_boss_readiness_at_floor_12', 0.0):.2f} "
        f"f16={s.get('avg_boss_readiness_at_floor_16', 0.0):.2f}"
    )
    print(f"  Death floor: {s['avg_floor_at_death']:.1f} avg")
    print(f"  Timeout floor:{s['avg_floor_at_timeout']:.1f} avg")
    print(f"  Avg floor:   {s['avg_floor']:.1f}  "
          f"(median={s['median_floor']}, max={s['max_floor']}, "
          f"min={s['min_floor']}, std={s['std_floor']:.1f})")
    print(f"  Avg HP:      {s['avg_hp_at_death']:.0f}")
    print(f"  Combats won: {s['avg_combats_won']:.1f} avg")
    print(f"  Steps:       {s['avg_steps']:.0f} avg")
    print(f"  Time:        {s['avg_time_s']:.1f}s avg, "
          f"{s['total_time_s']:.0f}s total")
    print(f"{'=' * 60}")


def print_floor_histogram(results: list[GameResult], strategy: str) -> None:
    """Print a text-based floor distribution histogram."""
    valid = [r for r in results if r.outcome != "error"]
    if not valid:
        print(f"  [{strategy}] No valid results for histogram.")
        return

    floors = [r.max_floor for r in valid]
    max_f = max(floors) if floors else 0

    # Bucket floors
    buckets: dict[int, int] = {}
    for f in floors:
        buckets[f] = buckets.get(f, 0) + 1

    max_count = max(buckets.values()) if buckets else 1
    bar_width = 40

    print(f"\n  Floor distribution [{strategy.upper()}]:")
    print(f"  {'Floor':>5}  {'Count':>5}  Bar")
    print(f"  {'-' * 5}  {'-' * 5}  {'-' * bar_width}")

    for floor_num in range(0, max_f + 1):
        count = buckets.get(floor_num, 0)
        if count == 0:
            continue
        bar_len = int(count / max_count * bar_width)
        bar = "#" * max(bar_len, 1)
        print(f"  {floor_num:>5}  {count:>5}  {bar}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate STS2 AI agent on fixed-seed games",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(MAINLINE_CHECKPOINT),
        help="Path to hybrid checkpoint (hybrid_XXXXX.pt). "
             "Required unless only running baselines.",
    )
    parser.add_argument(
        "--combat-checkpoint", type=str, default=None,
        help="Optional standalone combat checkpoint to override the combat brain",
    )
    parser.add_argument(
        "--combat-bc-model", type=str, default=None,
        help="Optional combat behavior-clone JSON to rerank combat actions on top of the active combat network",
    )
    parser.add_argument(
        "--combat-teacher-checkpoint", type=str, default=None,
        help="Optional offline combat-teacher checkpoint to use as the combat policy during evaluation",
    )
    parser.add_argument(
        "--combat-teacher-mode", choices=["replace", "rerank"], default="replace",
        help="How to apply the combat teacher during evaluation: full replace or baseline-first selective rerank",
    )
    parser.add_argument(
        "--combat-teacher-lethal-logit-blend-alpha", type=float, default=0.0,
        help="Selective direct-lethal blend weight for combat-teacher eval (0 disables)",
    )
    parser.add_argument(
        "--combat-teacher-direct-probe-top-k", type=int, default=4,
        help="How many top score/logit play actions to probe for direct lethal when using combat teacher eval",
    )
    parser.add_argument(
        "--combat-teacher-direct-probe-step-budget", type=int, default=24,
        help="Max simulated combat steps per direct-lethal probe branch",
    )
    parser.add_argument(
        "--lethal-probe", action="store_true", default=False,
        help="Enable standalone lethal probe for combat: check if any play_card "
             "directly kills all enemies and force that action (no teacher required)",
    )
    # NOTE 2026-04-08 (wizardly cleanup): --beam-search and its lazy import of
    # `beam_search_combat.py` were removed. It was a legacy research planner
    # superseded by --combat-turn-planner / --combat-turn-solver /
    # --combat-boss-expert. No tests or scripts invoked it; the file was
    # archived to tools/python/archive/beam_search_combat.py.
    parser.add_argument(
        "--combat-turn-planner", action="store_true", default=False,
        help="Enable policy-guided turn-level planner for combat decisions.",
    )
    parser.add_argument(
        "--planner-mode", choices=["boss", "elite", "boss_elite", "always"],
        default="boss_elite",
        help="When to enable planner (default: boss_elite).",
    )
    parser.add_argument(
        "--combat-turn-solver", action="store_true", default=False,
        help="Enable CombatTurnSolver (DFS within a turn) at boss/elite. "
             "Mutually exclusive with --combat-turn-planner.",
    )
    parser.add_argument(
        "--turn-solver-mode", choices=["boss", "elite", "boss_elite", "always"],
        default="boss",
        help="When to enable turn solver (default: boss only).",
    )
    parser.add_argument(
        "--turn-solver-max-actions", type=int, default=12,
        help="Max player actions per turn for the DFS solver (default: 12).",
    )
    parser.add_argument(
        "--turn-solver-hp-loss-weight", type=float, default=0.05,
        help="Weight applied to expected HP loss in the leaf eval. "
             "Default 0.05 (vs solver hardcoded 0.15) — lower values let the NN baseline value drive damage decisions.",
    )
    parser.add_argument(
        "--turn-solver-heuristic-blend", type=float, default=0.0,
        help="Blend alpha for heuristic state value at the leaf. "
             "0.0 = NN value only (default), 1.0 = heuristic only. "
             "Use this to bypass the flat NN value head signal at the boss.",
    )
    parser.add_argument(
        "--turn-solver-mc-head", type=str, default=None,
        help="Optional path to a trained MC value head (combat_mc_value_head.pt) "
             "to use as part of the leaf evaluator.",
    )
    parser.add_argument(
        "--turn-solver-mc-blend", type=float, default=0.0,
        help="Blend alpha for MC value head at the leaf (0..1). "
             "Combined with heuristic blend.",
    )
    parser.add_argument(
        "--turn-solver-heuristic-enemy-hp-weight", type=float, default=1.0,
        help="P4/P5 fix (2026-04-07): scale enemy HP weight in the heuristic state value. "
             "Default 1.0 = original behavior. Try 2.0 or 3.0 to compensate the structural "
             "defensive bias caused by player_max_hp << boss_max_hp.",
    )
    parser.add_argument(
        "--turn-solver-heuristic-use-absolute-hp", action="store_true", default=False,
        help="P4/P5 fix: use absolute HP units instead of fractions in the heuristic state value. "
             "Removes the asymmetric scaling bias entirely.",
    )
    parser.add_argument(
        "--turn-solver-boss-token-whitelist", type=str, default="",
        help="P1-2 (2026-04-07): comma-separated list of boss tokens (e.g. "
             "'CEREMONIAL_BEAST'). When set, the turn solver only fires on boss "
             "states whose enemy.id matches one of these tokens — other bosses "
             "and all non-boss states fall through to NN argmax. Use this to "
             "disable search on bosses where it regresses (the_kin, vantom).",
    )
    parser.add_argument(
        "--combat-boss-expert", type=str, default="",
        help="Phase 4 Stage 4 (2026-04-07): path to a trained boss expert checkpoint "
             "(`train_boss_expert.py` output). When set, the expert REPLACES the turn "
             "solver for the trained boss_token (e.g. CEREMONIAL_BEAST), giving zero-cost "
             "inference. For non-trained boss states the expert is silent and the planner "
             "falls through to NN argmax. Mutually exclusive with --combat-turn-solver "
             "(don't pass both).",
    )
    parser.add_argument(
        "--combat-boss-experts", type=str, default="",
        help="Phase 4 Stage 9 (2026-04-08): comma-separated list of boss expert checkpoints "
             "to chain via fallback. Use this when you have multiple per-boss experts "
             "(e.g. v4_ceremonial.pt,v4_vantom.pt,v4_kin.pt). The first expert that matches "
             "the current boss_token is used; otherwise falls through to the next. "
             "Mutually exclusive with --combat-boss-expert (use one or the other).",
    )
    parser.add_argument(
        "--combat-boss-expert-rerank-topk", type=int, default=0,
        help="Phase 4 Stage 5 (2026-04-07): when > 0 AND --combat-boss-expert is set, wraps "
             "the expert in a rerank planner. Expert proposes top-K candidates, each is "
             "verified via 1-step lookahead + abs HP heuristic leaf eval, best is picked. "
             "Recommended: topk=3 or 4. topk=0 disables rerank (pure expert).",
    )
    parser.add_argument(
        "--combat-boss-expert-fallback-search", type=str, default="",
        help="Phase 4 Stage 8 (2026-04-08): when set AND expert+rerank is active, runs this "
             "search config as a fallback when the expert doesn't fire (wrong boss_token). "
             "Value is a turn-solver config name: 'frac_leaf' (fraction leaf, no abs HP, no whitelist) "
             "is the main supported option. Routes: expert+rerank → ceremonial, "
             "fallback_search → everything else, NN argmax → all fail.",
    )
    parser.add_argument(
        "--multi-turn-solver", action="store_true", default=False,
        help="Enable 2-turn lookahead solver (P1b). Mutually exclusive with --combat-turn-solver and --combat-turn-planner.",
    )
    # NOTE 2026-04-08 (wizardly merge): --boss-turn-search and its companion
    # CLI flags were removed together with `boss_turn_search_planner.py`. The
    # Phase 3 boss-only turn-level search experiment was rejected (5.5% same
    # as Stage 5 expert+rerank, but worse wallclock; see
    # `docs/diagnostics/phase3_boss_turn_search_20260407.md`). Stage 5 expert
    # rerank (`--combat-boss-expert ... --combat-boss-expert-rerank-topk 3`)
    # is the canonical replacement. The QG leaf evaluator runtime is now
    # plugged into `--combat-turn-solver` directly via the `boss_leaf_evaluator`
    # hook in `turn_solver_planner.build_turn_solver_planner` (Step 4 merge).
    parser.add_argument(
        "--multi-turn-topk", type=int, default=3,
        help="Top-K candidate first actions to lookahead from (default: 3).",
    )
    parser.add_argument(
        "--multi-turn-gamma", type=float, default=0.9,
        help="Discount factor for next-turn lookahead value (default: 0.9).",
    )
    parser.add_argument("--planner-beam-width", type=int, default=8)
    parser.add_argument("--planner-expand-topk", type=int, default=5)
    parser.add_argument("--planner-max-turn-steps", type=int, default=20)
    parser.add_argument("--planner-max-final-seqs", type=int, default=8)
    parser.add_argument("--planner-prior-bonus", type=float, default=0.05)
    parser.add_argument("--planner-hp-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--combat-mcts-sims", type=int, default=0,
        help="Optional number of combat MCTS simulations to run per combat decision during evaluation",
    )
    parser.add_argument(
        "--combat-mcts-c-puct", type=float, default=1.5,
        help="PUCT exploration constant for combat MCTS evaluation",
    )
    parser.add_argument(
        "--combat-mcts-step-budget", type=int, default=200,
        help="Max simulated combat steps per MCTS branch before forcing terminal",
    )
    parser.add_argument(
        "--combat-mcts-tactical-weight", type=float, default=DEFAULT_COMBAT_MCTS_TACTICAL_BLEND_WEIGHT,
        help="Optional eval-time blend weight for local tactical leaf value (0 disables, recommended 0.1-0.3)",
    )
    parser.add_argument(
        "--combat-mcts-continuation-value", action="store_true", default=False,
        help="Use teacher-trained continuation_value_head (win_prob) instead of PPO value_head for MCTS leaf evaluation",
    )
    parser.add_argument("--port", type=int, default=15527,
                        help="Godot simulator pipe port (default: 15527)")
    parser.add_argument(
        "--transport",
        choices=["http", "pipe", "pipe-binary"],
        default="pipe",
        help="Simulator transport to use for evaluation (default: pipe)",
    )
    parser.add_argument("--auto-launch", action="store_true",
                        help="Auto-launch a fresh Sim host for pipe transports.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                        help="Repo root used when auto-launching Sim hosts.")
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_DLL_PATH,
                        help="Path to headless_sim_host_0991.exe/.dll (or legacy HeadlessSim.dll) for auto-launch.")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to evaluate (default: 100)")
    parser.add_argument(
        "--seeds-file", type=str, default=str(DEFAULT_SEED_FILE),
        help="Seed suite JSON file (default: Act1 benchmark seeds)",
    )
    parser.add_argument(
        "--seed-suite", type=str, default="benchmark",
        choices=["smoke", "regression", "benchmark"],
        help="Seed suite to load from --seeds-file (default: benchmark)",
    )
    parser.add_argument("--game-timeout", type=float, default=120.0,
                        help="Per-game timeout in seconds (default: 120)")
    parser.add_argument("--max-steps", type=int, default=800,
                        help="Max steps per game (default: 800)")
    parser.add_argument("--character", type=str, default="IRONCLAD",
                        help="Character ID (default: IRONCLAD)")
    parser.add_argument("--ascension", type=int, default=0,
                        help="Ascension level (default: 0)")
    parser.add_argument("--baseline", type=str, action="append", default=[],
                        choices=["random", "heuristic"],
                        help="Also run baseline strategy (can specify multiple)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: auto-generated)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: auto)")
    parser.add_argument(
        "--save-trace-dir",
        type=str,
        default=None,
        help="Optional directory to save per-seed trace exports and index.json",
    )
    parser.add_argument(
        "--trace-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to export full step traces for",
    )
    parser.add_argument(
        "--save-trajectory-dir",
        type=str,
        default=None,
        help="Optional directory to save full_run_trajectory.v1 JSONL exports",
    )
    parser.add_argument(
        "--trajectory-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to export trajectory JSONL for",
    )

    args = parser.parse_args()

    # Determine strategies to run
    strategies: list[str] = []
    if args.checkpoint:
        strategies.append("nn")
    for b in args.baseline:
        if b not in strategies:
            strategies.append(b)

    if not strategies:
        logger.error("No strategies to evaluate. Provide --checkpoint and/or --baseline.")
        return 1

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    vocab = load_vocab()

    # Load NN checkpoint if needed
    ppo_net: FullRunPolicyNetworkV2 | None = None
    combat_net: CombatPolicyValueNetwork | None = None
    combat_teacher_override: CombatTeacherOverride | None = None
    combat_bc_override: CombatBcOverride | None = None
    combat_mcts_agent: CombatMCTSAgent | None = None

    if "nn" in strategies:
        if not args.checkpoint:
            logger.error("--checkpoint is required for NN evaluation")
            return 1
        if not Path(args.checkpoint).exists():
            logger.error("Checkpoint not found: %s", args.checkpoint)
            return 1

        logger.info("Loading checkpoint: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        # PPO network
        ppo_state = ckpt.get("ppo_model") or ckpt.get("model_state_dict")
        ppo_config = ckpt.get("ppo_config", {})
        ppo_embed_dim = _infer_ppo_embed_dim(
            ppo_state, ppo_config.get("embed_dim", 32),
        )
        if ppo_config.get("embed_dim") not in (None, ppo_embed_dim):
            logger.warning(
                "Checkpoint PPO config embed_dim=%s disagrees with weights; using inferred embed_dim=%d",
                ppo_config.get("embed_dim"), ppo_embed_dim,
            )
        # Auto-detect retrieval head from checkpoint (2026-04-09)
        _retrieval_proj_dim = _infer_retrieval_proj_dim(ppo_state or {})
        _use_retrieval = _retrieval_proj_dim > 0
        if _use_retrieval:
            logger.info(
                "Auto-detected SymbolicFeaturesHead in checkpoint (proj_dim=%d); "
                "building retrieval-enabled network.", _retrieval_proj_dim,
            )
        ppo_net = FullRunPolicyNetworkV2(
            vocab=vocab, embed_dim=ppo_embed_dim,
            use_symbolic_features=_use_retrieval,
            symbolic_proj_dim=_retrieval_proj_dim if _use_retrieval else 16,
        )
        if "ppo_model" in ckpt:
            _safe_load_state_dict(ppo_net, ckpt["ppo_model"], "PPO")
            logger.info("Loaded PPO model (embed_dim=%d)", ppo_embed_dim)
        elif "model_state_dict" in ckpt:
            _safe_load_state_dict(ppo_net, ckpt["model_state_dict"], "PPO")
            logger.info("Loaded PPO model from model_state_dict")
        else:
            logger.warning("No PPO model found in checkpoint!")

        # Combat NN
        combat_state = ckpt.get("mcts_model")
        mcts_config = ckpt.get("mcts_config", {})
        combat_embed_dim, combat_hidden_dim = _infer_combat_dims(
            combat_state,
            mcts_config.get("embed_dim", ppo_embed_dim),
            mcts_config.get("hidden_dim", 128),
        )
        if mcts_config.get("embed_dim") not in (None, combat_embed_dim) or (
            mcts_config.get("hidden_dim") not in (None, combat_hidden_dim)
        ):
            logger.warning(
                "Checkpoint combat config embed=%s hidden=%s disagrees with weights; using inferred embed=%d hidden=%d",
                mcts_config.get("embed_dim"),
                mcts_config.get("hidden_dim"),
                combat_embed_dim,
                combat_hidden_dim,
            )
        # Auto-detect deck_repr_dim from checkpoint keys
        mcts_sd = ckpt.get("mcts_model", {})
        _deck_repr_dim = 0
        if any("deck_encoder" in k for k in mcts_sd):
            # Infer from deck_encoder output dim
            for k, v in mcts_sd.items():
                if k == "deck_encoder.norm.weight":
                    _deck_repr_dim = v.shape[0]
                    break
        combat_net = CombatPolicyValueNetwork(
            vocab=vocab,
            embed_dim=combat_embed_dim,
            hidden_dim=combat_hidden_dim,
            entity_embeddings=ppo_net.entity_emb,  # shared embeddings
            deck_repr_dim=_deck_repr_dim,
            symbolic_head=ppo_net.symbolic_head,  # shared retrieval head (may be None)
        )
        if "mcts_model" in ckpt:
            _safe_load_state_dict(combat_net, ckpt["mcts_model"], "combat")
            logger.info("Loaded combat NN (embed=%d, hidden=%d)",
                        combat_embed_dim, combat_hidden_dim)
        else:
            logger.warning("No combat NN model found in checkpoint!")

        if args.combat_checkpoint:
            combat_path = Path(args.combat_checkpoint)
            if not combat_path.exists():
                logger.error("Combat checkpoint not found: %s", args.combat_checkpoint)
                return 1
            logger.info("Loading combat override checkpoint: %s", combat_path)
            combat_ckpt = torch.load(combat_path, map_location="cpu", weights_only=False)
            combat_state_override = (
                combat_ckpt.get("model_state_dict")
                or combat_ckpt.get("mcts_model")
            )
            if not isinstance(combat_state_override, dict):
                logger.error(
                    "Combat checkpoint has no model_state_dict or mcts_model: %s",
                    combat_path,
                )
                return 1
            override_cfg = combat_ckpt.get("config", {})
            override_embed_dim, override_hidden_dim = _infer_combat_dims(
                combat_state_override,
                override_cfg.get("embed_dim", combat_embed_dim),
                override_cfg.get("hidden_dim", combat_hidden_dim),
            )
            combat_net = CombatPolicyValueNetwork(
                vocab=vocab,
                embed_dim=override_embed_dim,
                hidden_dim=override_hidden_dim,
            )
            _safe_load_state_dict(combat_net, combat_state_override, "combat_override")
            logger.info(
                "Loaded combat override (embed=%d, hidden=%d)",
                override_embed_dim,
                override_hidden_dim,
            )

        if args.combat_teacher_checkpoint:
            teacher_path = Path(args.combat_teacher_checkpoint)
            if not teacher_path.exists():
                logger.error("Combat teacher checkpoint not found: %s", teacher_path)
                return 1
            logger.info("Loading combat teacher override checkpoint: %s", teacher_path)
            teacher_ckpt = torch.load(teacher_path, map_location="cpu", weights_only=False)
            teacher_state = teacher_ckpt.get("model_state_dict") or teacher_ckpt.get("mcts_model")
            if not isinstance(teacher_state, dict):
                logger.error(
                    "Combat teacher checkpoint has no model_state_dict or mcts_model: %s",
                    teacher_path,
                )
                return 1
            teacher_cfg = teacher_ckpt.get("config", {})
            teacher_embed_dim, teacher_hidden_dim = _infer_combat_dims(
                teacher_state,
                teacher_cfg.get("embed_dim", combat_embed_dim),
                teacher_cfg.get("hidden_dim", combat_hidden_dim),
            )
            teacher_net = CombatPolicyValueNetwork(
                vocab=vocab,
                embed_dim=teacher_embed_dim,
                hidden_dim=teacher_hidden_dim,
            )
            _safe_load_state_dict(teacher_net, teacher_state, "combat_teacher_override")
            teacher_net.to(device).eval()
            combat_teacher_override = CombatTeacherOverride(
                network=teacher_net,
                vocab=vocab,
                device=device,
                mode=str(args.combat_teacher_mode),
                lethal_logit_blend_alpha=float(max(0.0, args.combat_teacher_lethal_logit_blend_alpha)),
                direct_lethal_probe_top_k=max(1, int(args.combat_teacher_direct_probe_top_k)),
                direct_lethal_step_budget=max(4, int(args.combat_teacher_direct_probe_step_budget)),
            )
            logger.info(
                "Loaded combat teacher override (mode=%s, embed=%d, hidden=%d, lethal_blend=%.2f, probe_top_k=%d, probe_budget=%d)",
                combat_teacher_override.mode,
                teacher_embed_dim,
                teacher_hidden_dim,
                combat_teacher_override.lethal_logit_blend_alpha,
                combat_teacher_override.direct_lethal_probe_top_k,
                combat_teacher_override.direct_lethal_step_budget,
            )
            if args.combat_bc_model and combat_teacher_override.mode == "replace":
                logger.warning("combat-teacher override is active in replace mode; combat BC rerank will be ignored for combat decisions")
            if args.combat_mcts_sims > 0:
                logger.warning("combat-teacher override is active, but combat MCTS still takes priority when enabled")

        if args.combat_bc_model:
            combat_bc_path = Path(args.combat_bc_model)
            if not combat_bc_path.exists():
                logger.error("Combat BC model not found: %s", args.combat_bc_model)
                return 1
            combat_bc_override = _load_combat_bc_override(combat_bc_path)
            logger.info(
                "Loaded combat BC override: %s (gate=%s, patch=%s)",
                combat_bc_path,
                json.dumps(combat_bc_override.gate_constraints, ensure_ascii=True, sort_keys=True),
                json.dumps(asdict(combat_bc_override.patch_config), ensure_ascii=True, sort_keys=True),
            )

        if args.combat_mcts_sims > 0:
            combat_mcts_agent = CombatMCTSAgent(
                network=combat_net,
                vocab=vocab,
                config=MCTSConfig(
                    num_simulations=max(1, int(args.combat_mcts_sims)),
                    c_puct=float(args.combat_mcts_c_puct),
                    temperature=0.0,
                    dirichlet_alpha=0.0,
                    dirichlet_fraction=0.0,
                    num_determinizations=1,
                ),
                training=False,
                device=device,
            )
            combat_mcts_agent._max_step_budget = int(args.combat_mcts_step_budget)
            if getattr(args, "combat_mcts_continuation_value", False):
                from combat_nn import CombatNNEvaluator as _CombatNNEvaluator
                combat_mcts_agent.evaluator = _CombatNNEvaluator(
                    combat_net, vocab, device=device, use_continuation_value=True,
                )
                logger.info("MCTS evaluator: using continuation_value_head (win_prob)")
            tactical_weight = max(0.0, min(1.0, float(args.combat_mcts_tactical_weight)))
            if tactical_weight > 0.0:
                combat_mcts_agent.evaluator = CombatMctsTacticalBlendEvaluator(
                    combat_mcts_agent.evaluator,
                    CombatMctsTacticalBlendConfig(weight=tactical_weight),
                )
            logger.info(
                "Loaded combat MCTS eval override: sims=%d c_puct=%.2f step_budget=%d tactical_weight=%.2f",
                args.combat_mcts_sims,
                args.combat_mcts_c_puct,
                args.combat_mcts_step_budget,
                tactical_weight,
            )

        ppo_net.to(device).eval()
        combat_net.to(device).eval()

        logger.info("Models loaded on %s", device)

    # Connect to Godot
    spawned_sim_proc = None
    if args.auto_launch:
        if args.transport == "http":
            logger.warning("--auto-launch is only supported for Sim pipe transports; ignoring for transport=http")
        else:
            launch_protocol = "bin" if args.transport == "pipe-binary" else "json"
            logger.info(
                "Auto-launching fresh Sim host from %s on port %d (%s)",
                Path(args.headless_dll).resolve(),
                args.port,
                launch_protocol,
            )
            spawned_sim_proc = start_headless_sim(
                port=args.port,
                repo_root=args.repo_root,
                dll_path=args.headless_dll,
                protocol=launch_protocol,
            )
            atexit.register(lambda: stop_process(spawned_sim_proc))

    logger.info("Connecting to simulator on port %d via %s...", args.port, args.transport)
    client: PipeBackedFullRunClient | ApiBackedFullRunClient
    if args.transport == "http":
        client = ApiBackedFullRunClient(base_url=f"http://127.0.0.1:{args.port}")
        try:
            client.get_state()
        except Exception as http_exc:
            logger.error("Failed to connect to HTTP simulator: %s", http_exc)
            return 1
    elif args.transport == "pipe":
        try:
            client = create_full_run_client(
                port=args.port,
                use_pipe=True,
                transport=args.transport,
                ready_timeout_s=15.0,
            )
            client._ensure_connected()
        except Exception as e:
            logger.warning("Pipe unavailable on port %d (%s). Falling back to HTTP.", args.port, e)
            client = ApiBackedFullRunClient(base_url=f"http://127.0.0.1:{args.port}")
            try:
                client.get_state()
            except Exception as http_exc:
                logger.error("Failed to connect to Godot: %s", http_exc)
                return 1
    else:
        client = create_full_run_client(
            port=args.port,
            use_pipe=True,
            transport=args.transport,
            ready_timeout_s=15.0,
        )
        try:
            client._ensure_connected()
        except Exception as e:
            logger.error("Failed to connect to simulator: %s", e)
            return 1
    logger.info("Connected via %s", getattr(client, "transport_name", args.transport))

    seeds = _load_seed_list(args.seeds_file, args.seed_suite, args.num_games)
    trace_seeds = _parse_trace_seed_arg(args.trace_seeds)
    trajectory_seeds = _parse_trace_seed_arg(args.trajectory_seeds)
    logger.info(
        "Using %d evaluation seeds from %s [%s]",
        len(seeds), args.seeds_file, args.seed_suite,
    )

    # Create turn planner if requested
    _turn_planner = None
    if getattr(args, "combat_turn_planner", False) and combat_net is not None:
        from combat_turn_planner import (
            TurnPlanner, PolicyBeamCandidateGenerator, RolloutCombatEvaluator, BeamConfig,
        )
        beam_cfg = BeamConfig(
            beam_width=getattr(args, "planner_beam_width", 8),
            expand_topk=getattr(args, "planner_expand_topk", 5),
            max_turn_steps=getattr(args, "planner_max_turn_steps", 20),
            max_final_sequences=getattr(args, "planner_max_final_seqs", 8),
            prior_bonus=getattr(args, "planner_prior_bonus", 0.05),
        )
        generator = PolicyBeamCandidateGenerator(
            combat_net=combat_net, vocab=vocab, device=device, config=beam_cfg,
        )
        evaluator = RolloutCombatEvaluator(
            hp_loss_weight=getattr(args, "planner_hp_loss_weight", 0.2),
        )
        _turn_planner = TurnPlanner(generator, evaluator, beam_cfg)
        _turn_planner._mode = getattr(args, "planner_mode", "boss_elite")
        logger.info("Turn planner enabled: mode=%s beam=%d topk=%d",
                     _turn_planner._mode, beam_cfg.beam_width, beam_cfg.expand_topk)
    elif (getattr(args, "combat_boss_expert", "") or getattr(args, "combat_boss_experts", "")) and combat_net is not None:
        rerank_topk = int(getattr(args, "combat_boss_expert_rerank_topk", 0) or 0)
        if rerank_topk > 0:
            from boss_expert_rerank_planner import build_boss_expert_rerank_planner
            # Phase 4 Stage 8: optional fallback search for non-expert bosses
            _fallback_planner = None
            fb_name = getattr(args, "combat_boss_expert_fallback_search", "") or ""
            if fb_name:
                from turn_solver_planner import build_turn_solver_planner
                if fb_name == "frac_leaf":
                    # Fraction leaf, no abs HP, no whitelist — runs on all bosses
                    _fallback_planner = build_turn_solver_planner(
                        combat_net=combat_net,
                        vocab=vocab,
                        device=device,
                        mode="boss",
                        hp_loss_weight=0.0,
                        heuristic_blend_alpha=1.0,
                        heuristic_use_absolute_hp=False,
                        boss_token_whitelist=None,
                    )
                    logger.info("Stage 8 fallback planner built: frac_leaf (no abs HP, no whitelist)")
                else:
                    logger.warning("Unknown fallback search config: %s (only 'frac_leaf' supported)", fb_name)

            # Phase 4 Stage 9 (2026-04-08): multi-expert chain via fallback.
            # If --combat-boss-experts (plural) is provided, build a chain of
            # per-boss rerank planners, each falling back to the next.
            multi_paths = getattr(args, "combat_boss_experts", "") or ""
            if multi_paths:
                expert_paths = [p.strip() for p in multi_paths.split(",") if p.strip()]
                logger.info("Building multi-expert chain: %d experts", len(expert_paths))
                # Build innermost first (chained fallback)
                chained: Any = _fallback_planner
                for expert_path in reversed(expert_paths):
                    next_planner = build_boss_expert_rerank_planner(
                        expert_path,
                        device=device,
                        mode="boss",
                        combat_net=combat_net,
                        ppo_net=ppo_net,
                        sts_vocab=vocab,
                        topk=rerank_topk,
                        use_absolute_hp=True,
                        fallback_planner=chained,
                    )
                    if next_planner is None:
                        logger.error("Failed to load boss expert %s", expert_path)
                        return 1
                    logger.info("  → loaded %s (target=%s)", expert_path, next_planner.target_boss_token)
                    chained = next_planner
                _turn_planner = chained
            else:
                _turn_planner = build_boss_expert_rerank_planner(
                    args.combat_boss_expert,
                    device=device,
                    mode="boss",
                    combat_net=combat_net,
                    ppo_net=ppo_net,
                    sts_vocab=vocab,
                    topk=rerank_topk,
                    use_absolute_hp=True,
                    fallback_planner=_fallback_planner,
                )
            if _turn_planner is None:
                logger.error("Failed to load boss expert rerank planner from %s", args.combat_boss_expert)
                return 1
            logger.info(
                "Boss expert + rerank planner enabled: %s (target=%s, v2=%s, topk=%d, fallback=%s)",
                args.combat_boss_expert,
                _turn_planner.target_boss_token,
                _turn_planner.is_v2,
                rerank_topk,
                fb_name or "(none)",
            )
        else:
            from boss_expert_planner import load_boss_expert_planner
            _turn_planner = load_boss_expert_planner(
                args.combat_boss_expert,
                device=device,
                mode="boss",
                combat_net=combat_net,
                ppo_net=ppo_net,
                sts_vocab=vocab,
            )
            if _turn_planner is None:
                logger.error("Failed to load boss expert from %s", args.combat_boss_expert)
                return 1
            logger.info(
                "Boss expert planner enabled: %s (target=%s, v2=%s)",
                args.combat_boss_expert,
                _turn_planner.target_boss_token,
                _turn_planner.is_v2,
            )
    elif getattr(args, "combat_turn_solver", False) and combat_net is not None:
        from turn_solver_planner import build_turn_solver_planner
        _whitelist_raw = getattr(args, "turn_solver_boss_token_whitelist", "") or ""
        _whitelist = [t.strip() for t in _whitelist_raw.split(",") if t.strip()] if _whitelist_raw else None
        _turn_planner = build_turn_solver_planner(
            combat_net=combat_net,
            vocab=vocab,
            device=device,
            mode=getattr(args, "turn_solver_mode", "boss"),
            max_player_actions=getattr(args, "turn_solver_max_actions", 12),
            hp_loss_weight=getattr(args, "turn_solver_hp_loss_weight", 0.05),
            heuristic_blend_alpha=getattr(args, "turn_solver_heuristic_blend", 0.0),
            mc_value_head_path=getattr(args, "turn_solver_mc_head", None),
            mc_value_blend=getattr(args, "turn_solver_mc_blend", 0.0),
            heuristic_enemy_hp_weight=getattr(args, "turn_solver_heuristic_enemy_hp_weight", 1.0),
            heuristic_use_absolute_hp=getattr(args, "turn_solver_heuristic_use_absolute_hp", False),
            boss_token_whitelist=_whitelist,
        )
        logger.info(
            "Combat turn solver enabled: mode=%s max_actions=%d hp_loss_weight=%.3f heur_blend=%.2f mc_blend=%.2f e_w=%.2f abs_hp=%s whitelist=%s",
            _turn_planner._mode,
            getattr(args, "turn_solver_max_actions", 12),
            getattr(args, "turn_solver_hp_loss_weight", 0.05),
            getattr(args, "turn_solver_heuristic_blend", 0.0),
            getattr(args, "turn_solver_mc_blend", 0.0),
            getattr(args, "turn_solver_heuristic_enemy_hp_weight", 1.0),
            getattr(args, "turn_solver_heuristic_use_absolute_hp", False),
            _whitelist or "(none)",
        )
    elif getattr(args, "multi_turn_solver", False) and combat_net is not None:
        from multi_turn_solver_planner import build_multi_turn_solver_planner
        _turn_planner = build_multi_turn_solver_planner(
            combat_net=combat_net,
            vocab=vocab,
            device=device,
            mode=getattr(args, "turn_solver_mode", "boss"),
            max_player_actions=getattr(args, "turn_solver_max_actions", 12),
            hp_loss_weight=getattr(args, "turn_solver_hp_loss_weight", 0.05),
            heuristic_blend_alpha=getattr(args, "turn_solver_heuristic_blend", 0.5),
            topk_candidates=getattr(args, "multi_turn_topk", 3),
            gamma=getattr(args, "multi_turn_gamma", 0.9),
            heuristic_enemy_hp_weight=getattr(args, "turn_solver_heuristic_enemy_hp_weight", 1.0),
            heuristic_use_absolute_hp=getattr(args, "turn_solver_heuristic_use_absolute_hp", False),
        )
        logger.info(
            "Multi-turn solver enabled: mode=%s topk=%d gamma=%.2f hp_loss=%.3f heur_blend=%.2f",
            _turn_planner._mode,
            getattr(args, "multi_turn_topk", 3),
            getattr(args, "multi_turn_gamma", 0.9),
            getattr(args, "turn_solver_hp_loss_weight", 0.05),
            getattr(args, "turn_solver_heuristic_blend", 0.5),
        )
    # NOTE 2026-04-08 (wizardly merge): the `--boss-turn-search` branch was
    # removed (see comment near the parser argument block above).

    # Run evaluations
    all_results: dict[str, list[GameResult]] = {}
    all_summaries: dict[str, dict[str, Any]] = {}

    for strategy in strategies:
        results, trace_payloads, trajectory_payloads = evaluate_batch(
            client=client,
            strategy=strategy,
            seeds=seeds,
            character_id=args.character,
            ascension_level=args.ascension,
            game_timeout=args.game_timeout,
            max_steps=args.max_steps,
            ppo_net=ppo_net,
            combat_net=combat_net,
            combat_teacher_override=combat_teacher_override,
            combat_bc_override=combat_bc_override,
            combat_mcts_agent=combat_mcts_agent,
            vocab=vocab,
            device=device,
            trace_seeds=trace_seeds,
            trajectory_seeds=trajectory_seeds,
            lethal_probe=getattr(args, "lethal_probe", False),
            beam_search=getattr(args, "beam_search", 0),
            turn_planner=_turn_planner,
        )
        all_results[strategy] = results
        summary = compute_summary(results)
        all_summaries[strategy] = summary

        if args.save_trace_dir:
            base_trace_dir = Path(args.save_trace_dir)
            trace_dir = (
                base_trace_dir / strategy
                if len(strategies) > 1
                else base_trace_dir
            )
            _write_trace_outputs(
                trace_dir,
                strategy,
                results,
                trace_payloads,
                {
                    "timestamp": datetime.now().isoformat(),
                    "checkpoint": args.checkpoint,
                    "combat_checkpoint": args.combat_checkpoint,
                    "combat_bc_model": args.combat_bc_model,
                    "combat_bc_patch_config": (
                        asdict(combat_bc_override.patch_config)
                        if combat_bc_override is not None
                        else None
                    ),
                    "character": args.character,
                    "ascension": args.ascension,
                    "seeds_file": args.seeds_file,
                    "seed_suite": args.seed_suite,
                    "num_games": len(seeds),
                    "trace_seeds": sorted(trace_seeds) if trace_seeds else [],
                },
            )

        if args.save_trajectory_dir and strategy == "nn":
            base_trajectory_dir = Path(args.save_trajectory_dir)
            trajectory_dir = (
                base_trajectory_dir / strategy
                if len(strategies) > 1
                else base_trajectory_dir
            )
            _write_trajectory_outputs(
                trajectory_dir,
                strategy,
                results,
                trajectory_payloads,
                {
                    "timestamp": datetime.now().isoformat(),
                    "checkpoint": args.checkpoint,
                    "combat_checkpoint": args.combat_checkpoint,
                    "combat_bc_model": args.combat_bc_model,
                    "combat_bc_patch_config": (
                        asdict(combat_bc_override.patch_config)
                        if combat_bc_override is not None
                        else None
                    ),
                    "character": args.character,
                    "ascension": args.ascension,
                    "seeds_file": args.seeds_file,
                    "seed_suite": args.seed_suite,
                    "num_games": len(seeds),
                    "trajectory_seeds": sorted(trajectory_seeds) if trajectory_seeds else [],
                },
            )

    # Print summaries
    for strategy in strategies:
        print_summary(all_summaries[strategy])
        print_floor_histogram(all_results[strategy], strategy)

    # Comparison table (if multiple strategies)
    if len(strategies) > 1:
        print(f"\n{'=' * 60}")
        print("  COMPARISON")
        print(f"{'=' * 60}")
        header = f"  {'Metric':<20}"
        for s in strategies:
            header += f"  {s.upper():>12}"
        print(header)
        print(f"  {'-' * 20}" + f"  {'-' * 12}" * len(strategies))

        metrics = [
            ("Win rate", lambda sm: f"{sm['win_rate'] * 100:.1f}%"),
            ("Boss reach", lambda sm: f"{sm['boss_reach_rate'] * 100:.1f}%"),
            ("Act1 clear", lambda sm: f"{sm['act1_clear_rate'] * 100:.1f}%"),
            ("Boss HP", lambda sm: f"{sm['avg_boss_hp_fraction_dealt']:.2f}"),
            ("Death floor", lambda sm: f"{sm['avg_floor_at_death']:.1f}"),
            ("Timeout floor", lambda sm: f"{sm['avg_floor_at_timeout']:.1f}"),
            ("Timeout count", lambda sm: f"{sm['timeout_count']}"),
            ("Avg floor", lambda sm: f"{sm['avg_floor']:.1f}"),
            ("Max floor", lambda sm: f"{sm['max_floor']}"),
            ("Avg HP", lambda sm: f"{sm['avg_hp_at_death']:.0f}"),
            ("Avg combats won", lambda sm: f"{sm['avg_combats_won']:.1f}"),
            ("Avg steps", lambda sm: f"{sm['avg_steps']:.0f}"),
            ("Avg time (s)", lambda sm: f"{sm['avg_time_s']:.1f}"),
        ]
        for label, fmt_fn in metrics:
            row = f"  {label:<20}"
            for s in strategies:
                row += f"  {fmt_fn(all_summaries[s]):>12}"
            print(row)
        print(f"{'=' * 60}")

    # Save to JSON
    output_path = args.output
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"eval_{ts}.json")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "combat_checkpoint": args.combat_checkpoint,
        "combat_bc_model": args.combat_bc_model,
        "combat_bc_patch_config": (
            asdict(combat_bc_override.patch_config)
            if combat_bc_override is not None
            else None
        ),
        "num_games": len(seeds),
        "seeds_file": args.seeds_file,
        "seed_suite": args.seed_suite,
        "seeds": seeds,
        "character": args.character,
        "ascension": args.ascension,
        "game_timeout": args.game_timeout,
        "max_steps": args.max_steps,
        "port": args.port,
        "strategies": strategies,
        "trace_seeds": sorted(trace_seeds) if trace_seeds else [],
        "trajectory_seeds": sorted(trajectory_seeds) if trajectory_seeds else [],
        "summaries": all_summaries,
        "results": {
            strategy: [asdict(r) for r in results]
            for strategy, results in all_results.items()
        },
        # Task 3 forensic: dump turn planner telemetry if exposed
        "turn_planner_forensic": (
            getattr(_turn_planner, "forensic", None)
            if _turn_planner is not None
            else None
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)

    if _turn_planner is not None:
        _planner_calls = getattr(_turn_planner, "calls", None)
        _planner_solver = getattr(_turn_planner, "solver_calls", None)
        _planner_cache = getattr(_turn_planner, "cache_hits", None)
        _planner_unsupported = getattr(_turn_planner, "unsupported", None)
        _planner_fallbacks = getattr(_turn_planner, "fallbacks", None)
        _planner_errors = getattr(_turn_planner, "errors", None)
        if _planner_calls is not None:
            logger.info(
                "Turn planner telemetry: calls=%s solver=%s cache=%s unsupported=%s fallbacks=%s errors=%s",
                _planner_calls, _planner_solver, _planner_cache,
                _planner_unsupported, _planner_fallbacks, _planner_errors,
            )
        _la_attempts = getattr(_turn_planner, "lookahead_attempts", None)
        _la_successes = getattr(_turn_planner, "lookahead_successes", None)
        if _la_attempts is not None:
            logger.info(
                "Multi-turn lookahead telemetry: attempts=%s successes=%s",
                _la_attempts, _la_successes,
            )

    # Cleanup
    try:
        client.close()
    except Exception:
        pass
    stop_process(spawned_sim_proc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
