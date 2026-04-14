#!/usr/bin/env python3
"""Offline dataset generator for non-combat ranking via branch rollout.

At each card_reward screen during an episode, saves state, evaluates each
option (take card A/B/C or skip) by playing forward through the next combat
with deterministic policy, then records combat outcomes as ranking targets.

Uses common random numbers: all options play from the same save point with
the same built-in RNG state, so variance comes only from the card difference.

Examples:
    # Generate dataset from 100 episodes
    python generate_offline_noncombat_ranking_data.py \\
        --pipe --port 15527 --episodes 100 \\
        --output STS2AI/Artifacts/offline_noncombat_ranking_v1/

    # With multiple parallel envs
    python generate_offline_noncombat_ranking_data.py \\
        --pipe --start-port 15527 --num-envs 4 --episodes 200 \\
        --output STS2AI/Artifacts/offline_noncombat_ranking_v1/
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    # Allow direct execution from the repo root with STS2AI/Python as the root.
    python_root = Path(__file__).resolve().parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

import _path_init  # noqa: F401  (adds STS2AI/Python library dirs to sys.path)

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import random
import threading
import time
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from full_run_env import FullRunClientLike, create_full_run_client
from test_simulator_consistency import COMBAT_TYPES
from verify_save_load import choose_default_action
from backends.full_run_backend import apply_backend_action
from ipc.headless_sim_runner import DEFAULT_DLL_PATH
from ipc.sim_host_lifecycle import SimHostLifecycleManager
from runtime.full_run_action_semantics import (
    RolloutDecision,
    choose_auto_progress_action,
    choose_rollout_decision,
    claim_reward_action_count,
    next_reward_claim_signature,
)
from runtime.run_outcome_vocab import RUN_OUTCOME_DEATH, RUN_OUTCOME_TIMEOUT, RUN_OUTCOME_VICTORY, is_failure_outcome, normalize_run_outcome
from card_reward_tree import RewardTreeConfig, evaluate_card_reward_tree
from map_route_tree import MapRouteConfig, evaluate_map_route_tree
from noncombat_deterministic import (
    choose_deterministic_card_select_action,
    choose_deterministic_rest_action,
    choose_deterministic_shop_action,
)
from data.raw.branch_schema import make_raw_branch_rollout_record
from data.raw.raw_dataset_writer import write_raw_branch_exports
from data.derived.build_rl_views import build_ranking_view
from data.derived.build_llm_views import build_preference_pair_view
from sts2ai_paths import ARTIFACTS_ROOT


def _peek_config_path(argv: list[str]) -> str | None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    known, _unknown = pre_parser.parse_known_args(argv)
    return known.config


def _flatten_config_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict):
                _visit(value)
            else:
                flat[str(key).replace("-", "_")] = value

    _visit(payload)
    return flat


def _load_generator_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a TOML table: {config_path}")
    return _flatten_config_mapping(payload)


def _load_seed_list(path_like: str | Path, *, limit: int = 0) -> list[str]:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    seeds: list[str] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("seeds", "items", "entries", "queue"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                seed = str(item.get("seed") or "").strip()
                if seed:
                    seeds.append(seed)
            else:
                seed = str(item).strip()
                if seed:
                    seeds.append(seed)
    else:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                seeds.append(stripped)
    ordered: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        if seed and seed not in seen:
            seen.add(seed)
            ordered.append(seed)
    if limit > 0:
        ordered = ordered[:limit]
    return ordered

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CombatOutcome:
    won: bool
    hp_after: int
    hp_lost: int
    turns: int
    terminal_state_type: str
    end_floor: int = 0
    boss_reached: bool = False
    run_outcome: str | None = None


@dataclass
class CardRankingSample:
    deck_ids: list[str]
    relic_ids: list[str]
    floor: int
    act: int
    options: list[dict[str, Any]]
    scores: list[float]
    best_idx: int
    combat_outcomes: dict[str, dict[str, Any]]
    sample_type: str = "card_reward"  # "card_reward" or "remove_card"
    label_source: str = "single_step"
    option_tree_values: list[dict[str, Any]] | None = None
    tree_summary: dict[str, Any] | None = None
    state_tensors_path: str | None = None
    _encoded_tensors: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        return d


@dataclass
class EpisodeGenerationSummary:
    seed: str
    status: str
    sample_count: int
    end_floor: int
    boss_reached: bool
    act1_cleared: bool
    outcome: str
    map_decisions_seen: int = 0
    map_samples_recorded: int = 0
    card_reward_decisions_seen: int = 0
    card_reward_samples_recorded: int = 0
    sampling_skip_reasons: dict[str, int] = field(default_factory=dict)


def _sha256_file(path_like: str | Path | None) -> str | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def _utc_now() -> str:
    return datetime.now(UTC).replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _build_stats(
    all_samples: list["CardRankingSample"],
    episode_logs: list[dict[str, Any]],
    *,
    episodes: int,
    num_envs: int,
    transport: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    stats = {
        "total_samples": len(all_samples),
        "tensor_samples": int(sum(1 for sample in all_samples if sample._encoded_tensors)),
        "episodes": int(episodes),
        "avg_samples_per_episode": len(all_samples) / max(int(episodes), 1),
        "elapsed_seconds": round(float(elapsed_seconds), 1),
        "num_envs": int(num_envs),
        "transport": str(transport),
    }
    if all_samples:
        all_scores = [s.scores for s in all_samples]
        score_spreads = [max(s) - min(s) for s in all_scores]
        sample_type_counts = Counter(str(s.sample_type or "unknown") for s in all_samples)
        floor_counts = Counter(int(s.floor or 0) for s in all_samples)
        act_counts = Counter(int(s.act or 0) for s in all_samples)
        best_idx_counts = Counter(int(s.best_idx) for s in all_samples)
        usable_floor_counts = Counter(
            int(sample.floor or 0)
            for sample, spread in zip(all_samples, score_spreads)
            if spread >= 0.01
        )
        usable_sample_count = int(sum(usable_floor_counts.values()))
        usable_ge5_sample_count = int(
            sum(count for floor, count in usable_floor_counts.items() if int(floor) >= 5)
        )
        zero_spread = sum(1 for spread in score_spreads if abs(spread) < 1e-8)
        stats["avg_score_spread"] = round(sum(score_spreads) / len(score_spreads), 4)
        stats["skip_best_rate"] = round(
            sum(1 for s in all_samples if s.best_idx == len(s.scores) - 1) / len(all_samples),
            4,
        )
        stats["zero_spread_samples"] = zero_spread
        stats["nonzero_spread_samples"] = len(all_samples) - zero_spread
        stats["nonzero_spread_rate"] = round(
            (len(all_samples) - zero_spread) / len(all_samples),
            4,
        )
        stats["sample_type_counts"] = {
            str(k): int(v) for k, v in sorted(sample_type_counts.items(), key=lambda kv: kv[0])
        }
        stats["act_counts"] = {
            str(k): int(v) for k, v in sorted(act_counts.items(), key=lambda kv: kv[0])
        }
        stats["best_idx_counts"] = {
            str(k): int(v) for k, v in sorted(best_idx_counts.items(), key=lambda kv: kv[0])
        }
        stats["floor_counts"] = {
            str(k): int(v)
            for k, v in sorted(floor_counts.items(), key=lambda kv: int(kv[0]))
        }
        stats["sample_floor_counts"] = dict(stats["floor_counts"])
        stats["usable_samples"] = usable_sample_count
        stats["usable_ge5_samples"] = usable_ge5_sample_count
        stats["usable_floor_counts"] = {
            str(k): int(v)
            for k, v in sorted(usable_floor_counts.items(), key=lambda kv: int(kv[0]))
        }

    if episode_logs:
        end_floor_counts = Counter(int(log.get("end_floor") or 0) for log in episode_logs)
        outcome_counts = Counter(str(log.get("outcome") or "unknown") for log in episode_logs)
        map_decisions_seen = sum(int(log.get("map_decisions_seen") or 0) for log in episode_logs)
        map_samples_recorded = sum(int(log.get("map_samples_recorded") or 0) for log in episode_logs)
        card_reward_decisions_seen = sum(int(log.get("card_reward_decisions_seen") or 0) for log in episode_logs)
        card_reward_samples_recorded = sum(int(log.get("card_reward_samples_recorded") or 0) for log in episode_logs)
        sampling_skip_reasons: Counter[str] = Counter()
        for log in episode_logs:
            for key, value in (log.get("sampling_skip_reasons") or {}).items():
                try:
                    sampling_skip_reasons[str(key)] += int(value or 0)
                except Exception:
                    continue
        stats["episode_end_floor_counts"] = {
            str(k): int(v)
            for k, v in sorted(end_floor_counts.items(), key=lambda kv: int(kv[0]))
        }
        stats["episode_outcome_counts"] = {
            str(k): int(v)
            for k, v in sorted(outcome_counts.items(), key=lambda kv: kv[0])
        }
        stats["episodes_boss_reached"] = int(sum(1 for log in episode_logs if log.get("boss_reached")))
        stats["episodes_act1_cleared"] = int(sum(1 for log in episode_logs if log.get("act1_cleared")))
        stats["map_decisions_seen"] = int(map_decisions_seen)
        stats["map_samples_recorded"] = int(map_samples_recorded)
        stats["card_reward_decisions_seen"] = int(card_reward_decisions_seen)
        stats["card_reward_samples_recorded"] = int(card_reward_samples_recorded)
        stats["sampling_skip_reasons"] = {
            str(k): int(v) for k, v in sorted(sampling_skip_reasons.items(), key=lambda kv: kv[0])
        }
    return stats


def _write_progress_snapshot(
    *,
    out_dir: Path,
    total_episodes: int,
    completed_episodes: int,
    all_samples: list["CardRankingSample"],
    episode_logs: list[dict[str, Any]],
    num_envs: int,
    transport: str,
    t0: float,
) -> Path:
    progress_path = out_dir / "progress.json"
    stats = _build_stats(
        all_samples,
        episode_logs,
        episodes=completed_episodes,
        num_envs=num_envs,
        transport=transport,
        elapsed_seconds=time.time() - t0,
    )
    payload = {
        "status": "recording",
        "updated_at_utc": _utc_now(),
        "completed_episodes": int(completed_episodes),
        "total_episodes": int(total_episodes),
        "progress_fraction": round(completed_episodes / max(total_episodes, 1), 4),
        "summary": stats,
        "last_episode": episode_logs[-1] if episode_logs else None,
    }
    progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return progress_path


def _materialize_dataset_snapshot(
    *,
    out_dir: Path,
    all_samples: list["CardRankingSample"],
    raw_branch_records: list[dict[str, Any]],
    episode_logs: list[dict[str, Any]],
    episodes: int,
    total_episodes: int,
    num_envs: int,
    transport: str,
    t0: float,
    args: argparse.Namespace,
    partial: bool,
) -> tuple[Path, Path, Path, Path]:
    jsonl_path = out_dir / "card_ranking.jsonl"
    tensors_dir = out_dir / "tensors"
    stats = _build_stats(
        all_samples,
        episode_logs,
        episodes=episodes,
        num_envs=num_envs,
        transport=transport,
        elapsed_seconds=time.time() - t0,
    )
    raw_path, raw_manifest_path = write_raw_branch_exports(
        output_dir=out_dir,
        branch_records=raw_branch_records,
        metadata={
            "episodes": int(total_episodes),
            "completed_episodes": int(episodes),
            "num_envs": int(num_envs),
            "transport": str(transport),
            "port": int(args.port),
            "start_port": None if args.start_port is None else int(args.start_port),
            "seed_prefix": str(args.seed_prefix),
            "common_random_numbers": True,
            "debug_rollout_trace_dir": str(args.debug_rollout_trace_dir) if args.debug_rollout_trace_dir else None,
            "rollout_max_combats": int(args.rollout_max_combats),
            "rollout_max_steps": int(args.rollout_max_steps),
            "rerun_low_spread_threshold": float(args.rerun_low_spread_threshold),
            "rerun_max_combats": int(args.rerun_max_combats),
            "rerun_max_steps": int(args.rerun_max_steps),
            "label_mode": str(args.label_mode),
            "map_max_depth": int(args.map_max_depth),
            "map_beam_width": int(args.map_beam_width),
            "map_advance_max_steps": int(args.map_advance_max_steps),
            "map_max_option_seconds": None if float(args.map_max_option_seconds) <= 0 else float(args.map_max_option_seconds),
            "episode_stop_floor": None if int(args.episode_stop_floor) <= 0 else int(args.episode_stop_floor),
            "tree_max_reward_depth": int(args.tree_max_reward_depth),
            "tree_beam_width": int(args.tree_beam_width),
            "tree_advance_max_steps": int(args.tree_advance_max_steps),
            "tree_local_weight": float(args.tree_local_weight),
            "tree_max_option_seconds": None if float(args.tree_max_option_seconds) <= 0 else float(args.tree_max_option_seconds),
            "tree_recurse_only_when_spread_below": None if float(args.tree_recurse_only_when_spread_below) <= 0 else float(args.tree_recurse_only_when_spread_below),
            "local_ort_rollout": bool(args.local_ort_rollout),
            "local_ort_max_combat_steps": int(args.local_ort_max_combat_steps),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "combat_checkpoint": str(args.combat_checkpoint) if args.combat_checkpoint else None,
            "combat_checkpoint_sha256": _sha256_file(args.combat_checkpoint),
            "sample_type_counts": stats.get("sample_type_counts", {}),
        },
        partial=partial,
    )
    _, ranking_summary = build_ranking_view(
        raw_branch_records=raw_branch_records,
        output_dir=out_dir / "derived" / "rl",
        compatibility_root=out_dir,
        partial=partial,
    )
    preference_path, preference_manifest_path = build_preference_pair_view(
        raw_branch_records=raw_branch_records,
        output_dir=out_dir / "derived" / "llm",
    )
    stats["tensor_samples"] = int(ranking_summary.get("tensor_samples", stats.get("tensor_samples", 0)))
    stats["derived_ranking_samples"] = int(ranking_summary.get("num_samples", 0))
    stats["derived_preference_pairs"] = int(
        json.loads(preference_manifest_path.read_text(encoding="utf-8")).get("summary", {}).get("num_pairs", 0)
    )

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    episode_logs_path = out_dir / "episode_logs.json"
    episode_logs_path.write_text(json.dumps(episode_logs, indent=2, ensure_ascii=False), encoding="utf-8")

    progress_path = out_dir / "progress.json"
    manifest = {
        "dataset_kind": "card_ranking",
        "dataset_schema_version": "card_ranking.v1",
        "dataset_kind_raw": "raw_branch_rollout",
        "dataset_kind_derived_rl": "ranking_sample",
        "dataset_kind_derived_llm": "preference_pair",
        "generated_at_utc": _utc_now(),
        "output_dir": str(out_dir),
        "jsonl_path": str(jsonl_path),
        "tensors_dir": str(tensors_dir),
        "stats_path": str(stats_path),
        "episode_logs_path": str(episode_logs_path),
        "progress_path": str(progress_path),
        "raw_path": str(raw_path),
        "raw_manifest_path": str(raw_manifest_path),
        "source_raw_paths": [str(raw_path)],
        "derived_from": ["raw_branch_rollout.v1"],
        "derived_rl_manifest_path": str(out_dir / "derived" / "rl" / "manifest.json"),
        "derived_llm_manifest_path": str(preference_manifest_path),
        "derived_llm_preference_path": str(preference_path),
        "status": "partial" if partial else "complete",
        "generation_config": {
            "episodes": int(total_episodes),
            "completed_episodes": int(episodes),
            "num_envs": int(num_envs),
            "transport": str(transport),
            "port": int(args.port),
            "start_port": None if args.start_port is None else int(args.start_port),
            "seed_prefix": str(args.seed_prefix),
            "common_random_numbers": True,
            "debug_rollout_trace_dir": str(args.debug_rollout_trace_dir) if args.debug_rollout_trace_dir else None,
            "rollout_max_combats": int(args.rollout_max_combats),
            "rollout_max_steps": int(args.rollout_max_steps),
            "rerun_low_spread_threshold": float(args.rerun_low_spread_threshold),
            "rerun_max_combats": int(args.rerun_max_combats),
            "rerun_max_steps": int(args.rerun_max_steps),
            "label_mode": str(args.label_mode),
            "map_max_depth": int(args.map_max_depth),
            "map_beam_width": int(args.map_beam_width),
            "map_advance_max_steps": int(args.map_advance_max_steps),
            "map_max_option_seconds": None if float(args.map_max_option_seconds) <= 0 else float(args.map_max_option_seconds),
            "episode_stop_floor": None if int(args.episode_stop_floor) <= 0 else int(args.episode_stop_floor),
            "tree_max_reward_depth": int(args.tree_max_reward_depth),
            "tree_beam_width": int(args.tree_beam_width),
            "tree_advance_max_steps": int(args.tree_advance_max_steps),
            "tree_local_weight": float(args.tree_local_weight),
            "tree_max_option_seconds": None if float(args.tree_max_option_seconds) <= 0 else float(args.tree_max_option_seconds),
            "tree_recurse_only_when_spread_below": None if float(args.tree_recurse_only_when_spread_below) <= 0 else float(args.tree_recurse_only_when_spread_below),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "combat_checkpoint": str(args.combat_checkpoint) if args.combat_checkpoint else None,
            "combat_checkpoint_sha256": _sha256_file(args.combat_checkpoint),
        },
        "summary": stats,
    }
    manifest_path = Path(args.manifest_out) if (args.manifest_out and not partial) else (out_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonl_path, stats_path, episode_logs_path, manifest_path


# ---------------------------------------------------------------------------
# Combat simulation helpers
# ---------------------------------------------------------------------------

def _extract_deck_ids(state: dict[str, Any]) -> list[str]:
    player = state.get("player") or {}
    deck = player.get("deck") or []
    return [c.get("id") or c.get("label") or "?" for c in deck]


def _extract_relic_ids(state: dict[str, Any]) -> list[str]:
    player = state.get("player") or {}
    relics = player.get("relics") or []
    return [r.get("id") or r.get("name") or "?" for r in relics]


def _extract_player_hp(state: dict[str, Any]) -> int:
    player = state.get("player") or {}
    return player.get("hp") or player.get("current_hp") or 0


def _extract_floor(state: dict[str, Any]) -> int:
    return int((state.get("run") or {}).get("floor") or 0)


def _reached_stop_floor(state: dict[str, Any], stop_floor: int | None) -> bool:
    if stop_floor is None:
        return False
    return _extract_floor(state) >= int(stop_floor)


def _compact_hand_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    player = state.get("player") or {}
    battle = state.get("battle") or {}
    hand = player.get("hand") or battle.get("hand") or []
    compact: list[dict[str, Any]] = []
    for idx, card in enumerate(hand):
        compact.append(
            {
                "index": int(idx),
                "id": card.get("id") or card.get("name") or "?",
                "name": card.get("name") or card.get("id") or "?",
                "cost": card.get("cost"),
                "upgraded": card.get("upgraded"),
            }
        )
    return compact


def _compact_enemy_summary(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or []
    compact: list[dict[str, Any]] = []
    for enemy in enemies:
        compact.append(
            {
                "id": enemy.get("id") or enemy.get("name") or "?",
                "name": enemy.get("name") or enemy.get("id") or "?",
                "hp": enemy.get("hp"),
                "max_hp": enemy.get("max_hp"),
                "block": enemy.get("block"),
                "intent": enemy.get("intent"),
            }
        )
    return compact


def _compact_debug_state(state: dict[str, Any]) -> dict[str, Any]:
    run = state.get("run") or {}
    player = state.get("player") or {}
    battle = state.get("battle") or {}
    legal = state.get("legal_actions") or []
    return {
        "state_type": str(state.get("state_type") or ""),
        "act": int(run.get("act") or 0),
        "floor": int(run.get("floor") or 0),
        "terminal": bool(state.get("terminal", False)),
        "run_outcome": state.get("run_outcome"),
        "hp": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "energy": player.get("energy"),
        "block": player.get("block"),
        "gold": player.get("gold"),
        "round_number": battle.get("round_number") or battle.get("round"),
        "hand": _compact_hand_cards(state),
        "draw_pile": len(player.get("draw_pile") or []),
        "discard_pile": len(player.get("discard_pile") or []),
        "exhaust_pile": len(player.get("exhaust_pile") or []),
        "enemies": _compact_enemy_summary(state),
        "legal_count": len(legal),
    }


def _compact_debug_action(
    action: dict[str, Any] | None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not action:
        return None
    keep = (
        "action",
        "label",
        "index",
        "card_index",
        "target_id",
        "target",
        "slot",
        "col",
        "row",
        "reward_type",
        "reward_key",
        "card_id",
        "card_name",
    )
    compact = {key: action.get(key) for key in keep if key in action}
    if state is not None and str(action.get("action") or "").lower() == "play_card":
        card_index = action.get("card_index")
        hand = ((state.get("player") or {}).get("hand") or (state.get("battle") or {}).get("hand") or [])
        if isinstance(card_index, int) and 0 <= card_index < len(hand):
            card = hand[card_index] or {}
            compact["played_card_id"] = card.get("id") or card.get("name") or "?"
            compact["played_card_name"] = card.get("name") or card.get("id") or "?"
            compact["played_card_cost"] = card.get("cost")
    return compact or {"action": str(action.get("action") or "?")}


def _sanitize_trace_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))
    return safe[:48] or "option"


def _write_rollout_trace(
    trace_dir: str | Path | None,
    *,
    seed: str,
    sample_index: int,
    sample_type: str,
    floor: int,
    option_index: int,
    option_label: str,
    trace_entries: list[dict[str, Any]],
) -> str | None:
    if not trace_dir or not trace_entries:
        return None
    base = Path(trace_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / (
        f"sample_{sample_index:05d}_f{int(floor):02d}_{sample_type}"
        f"_opt{int(option_index):02d}_{_sanitize_trace_label(option_label)}.jsonl"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "kind": "meta",
            "seed": seed,
            "sample_index": int(sample_index),
            "sample_type": sample_type,
            "floor": int(floor),
            "option_index": int(option_index),
            "option_label": option_label,
        }, ensure_ascii=False) + "\n")
        for entry in trace_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return str(path)


def _did_reach_boss(state: dict[str, Any]) -> bool:
    st = str(state.get("state_type") or "").lower()
    return st == "boss" or _extract_floor(state) >= 16


def _did_clear_act1(state: dict[str, Any]) -> bool:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    act = _safe_int(run.get("act", 0), 0)
    floor = _extract_floor(state)
    return act > 1 or floor >= 18


def _build_raw_branch_record(
    *,
    seed: str,
    sample_index: int,
    sample_type: str,
    label_source: str,
    root_state: dict[str, Any],
    options: list[dict[str, Any]],
    scores: list[float],
    best_idx: int,
    combat_outcomes: dict[int, CombatOutcome] | dict[str, Any],
    option_traces: dict[int, list[dict[str, Any]]] | dict[str, list[dict[str, Any]]],
    tree_summary: dict[str, Any] | None,
    option_tree_values: list[dict[str, Any]] | None,
    port: int | None,
    transport: str,
    checkpoint_path: str | None,
    checkpoint_sha256: str | None,
    combat_checkpoint_path: str | None,
    combat_checkpoint_sha256: str | None,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    normalized_outcomes = {
        str(k): asdict(v) if isinstance(v, CombatOutcome) else v
        for k, v in combat_outcomes.items()
    }
    normalized_traces = {str(k): v for k, v in option_traces.items()}
    return make_raw_branch_rollout_record(
        episode_id=seed,
        seed=seed,
        sample_index=sample_index,
        sample_type=sample_type,
        label_source=label_source,
        root_state=root_state,
        options=options,
        scores=scores,
        best_idx=best_idx,
        combat_outcomes=normalized_outcomes,
        option_traces=normalized_traces,
        tree_summary=tree_summary,
        option_tree_values=option_tree_values,
        port=port,
        transport=transport,
        backend_kind=f"full_run_{transport}",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        combat_checkpoint_path=combat_checkpoint_path,
        combat_checkpoint_sha256=combat_checkpoint_sha256,
        generator_config=generator_config,
    )


def _extract_card_reward_options(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract card reward options from state."""
    legal = state.get("legal_actions") or []
    options = []
    for action in legal:
        act_type = action.get("action") or ""
        if act_type == "select_card_reward":
            card_reward = state.get("card_reward") or {}
            cards = card_reward.get("cards") or []
            idx = action.get("index")
            card_info = cards[idx] if idx is not None and idx < len(cards) else {}
            options.append({
                "type": "select",
                "card_id": (
                    card_info.get("id")
                    or card_info.get("name")
                    or action.get("card_id")
                    or action.get("card_name")
                    or action.get("label")
                    or "?"
                ),
                "card_name": (
                    card_info.get("name")
                    or action.get("card_name")
                    or action.get("label")
                    or ""
                ),
                "index": idx,
                "action": action,
            })
        elif act_type == "skip_card_reward":
            options.append({
                "type": "skip",
                "card_id": "SKIP",
                "index": action.get("index"),
                "action": action,
            })
    return options


def _extract_remove_options(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract remove card options from a card_select screen."""
    legal = state.get("legal_actions") or []
    card_select = state.get("card_select") or {}
    cards = card_select.get("cards") or []
    options = []
    for action in legal:
        act_type = action.get("action") or ""
        if act_type in ("select_card", "combat_select_card"):
            idx = action.get("index")
            card_info = cards[idx] if idx is not None and idx < len(cards) else {}
            options.append({
                "type": "remove",
                "card_id": card_info.get("id") or card_info.get("name") or "?",
                "card_name": card_info.get("name") or "",
                "index": idx,
                "action": action,
            })
        elif act_type in ("cancel", "skip", "confirm_selection"):
            # Cancel/skip = don't remove anything (keep all cards)
            options.append({
                "type": "skip_remove",
                "card_id": "NO_REMOVE",
                "index": action.get("index"),
                "action": action,
            })
    return options


def _extract_map_options(state: dict[str, Any]) -> list[dict[str, Any]]:
    legal = state.get("legal_actions") or []
    map_state = state.get("map") or {}
    next_options = map_state.get("next_options") or []
    by_index: dict[int, dict[str, Any]] = {}
    for option in next_options:
        if not isinstance(option, dict):
            continue
        try:
            by_index[int(option.get("index", -1))] = option
        except Exception:
            continue
    options: list[dict[str, Any]] = []
    for action in legal:
        if str(action.get("action") or "").strip().lower() != "choose_map_node":
            continue
        try:
            idx = int(action.get("index", -1))
        except Exception:
            continue
        option = by_index.get(idx, {})
        node_type = option.get("type") or option.get("point_type") or option.get("label") or "unknown"
        options.append(
            {
                "type": str(node_type),
                "label": f"{node_type}@{option.get('col', '?')},{option.get('row', '?')}",
                "index": idx,
                "col": option.get("col"),
                "row": option.get("row"),
                "action": action,
            }
        )
    return options


def _choose_deterministic_screen_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> dict[str, Any] | None:
    st = str(state.get("state_type") or "").strip().lower()
    if st == "rest_site":
        return choose_deterministic_rest_action(state, legal, hp_rest_threshold=0.5)
    if st == "shop":
        return choose_deterministic_shop_action(state, legal)
    if st == "card_select":
        return choose_deterministic_card_select_action(state, legal)
    return None


def _resolve_best_card_reward_choice(
    *,
    client: FullRunClientLike,
    seed: str,
    state: dict[str, Any],
    sample_index: int,
    combat_evaluator: Any | None,
    ppo_policy: Any | None,
    rollout_max_combats: int,
    rollout_max_steps: int,
    rerun_low_spread_threshold: float,
    rerun_max_combats: int,
    rerun_max_steps: int,
    rollout_goal: str,
    label_mode: str,
    reward_tree_config: RewardTreeConfig | None,
    use_local_ort_rollout: bool,
    local_ort_max_combat_steps: int,
) -> tuple[dict[str, Any], float]:
    options = _extract_card_reward_options(state)
    if not options:
        legal = state.get("legal_actions") or []
        fallback = legal[0] if legal else {"action": "wait"}
        return fallback, 0.0

    floor = _extract_floor(state)
    hp_before = _extract_player_hp(state)
    with tempfile.TemporaryDirectory(prefix=f"sts2_card_reward_pick_f{int(floor):02d}_") as tmpdir:
        snapshot_path = str(Path(tmpdir) / "pick_snapshot.json")
        client.export_state(snapshot_path)
        if label_mode == "reward_tree" and reward_tree_config is not None:
            tree_result = evaluate_card_reward_tree(
                client=client,
                seed=seed,
                state=client.get_state(),
                root_options=options,
                sample_index=sample_index,
                debug_rollout_trace_dir=None,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                config=reward_tree_config,
                apply_action=_apply_action,
                settle_after_choice=_settle_after_choice,
                extract_floor=_extract_floor,
                extract_player_hp=_extract_player_hp,
                extract_card_reward_options=_extract_card_reward_options,
                did_reach_boss=_did_reach_boss,
                choose_rollout_decision=_choose_rollout_decision,
                evaluate_branch_outcomes=_evaluate_branch_outcomes,
                compute_option_scores=compute_option_scores,
            )
            scores = list(tree_result.scores)
        else:
            def _restore(option_index: int) -> None:
                base_state = client.import_state(snapshot_path)
                branch_state = _apply_action(client, base_state, options[option_index]["action"])
                _settle_after_choice(
                    client,
                    branch_state,
                    previous_state_type="card_reward",
                    previous_floor=int(floor),
                )

            outcomes = _evaluate_branch_outcomes(
                client=client,
                seed=seed,
                floor=int(floor),
                hp_before=hp_before,
                sample_index=sample_index,
                sample_type="card_reward",
                options=options,
                restore_fn=_restore,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                debug_rollout_trace_dir=None,
                rollout_goal=rollout_goal,
                max_combats=rollout_max_combats,
                max_steps=rollout_max_steps,
                use_local_ort_rollout=use_local_ort_rollout,
                local_ort_max_combat_steps=local_ort_max_combat_steps,
                trace_store=None,
            )
            scores = compute_option_scores(outcomes, max_hp=max(hp_before, 1))
            if (
                rerun_low_spread_threshold > 0
                and _score_spread(scores) < rerun_low_spread_threshold
                and rerun_max_combats > rollout_max_combats
            ):
                outcomes = _evaluate_branch_outcomes(
                    client=client,
                    seed=seed,
                    floor=int(floor),
                    hp_before=hp_before,
                    sample_index=sample_index,
                    sample_type="card_reward",
                    options=options,
                    restore_fn=_restore,
                    combat_evaluator=combat_evaluator,
                    ppo_policy=ppo_policy,
                    debug_rollout_trace_dir=None,
                    rollout_goal=rollout_goal,
                    max_combats=rerun_max_combats,
                    max_steps=rerun_max_steps,
                    use_local_ort_rollout=use_local_ort_rollout,
                    local_ort_max_combat_steps=local_ort_max_combat_steps,
                    trace_store=None,
                )
                scores = compute_option_scores(outcomes, max_hp=max(hp_before, 1))
        restored_state = client.import_state(snapshot_path)
        restored_state_type = str(restored_state.get("state_type") or "").strip().lower()
        if restored_state_type != "card_reward":
            raise RuntimeError(
                f"_resolve_best_card_reward_choice expected restore to card_reward, got {restored_state_type or 'unknown'}"
            )
    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    return options[best_idx]["action"], float(scores[best_idx])


def _choose_rollout_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    rng: random.Random,
    *,
    combat_evaluator: Any | None = None,
    ppo_policy: Any | None = None,
) -> dict[str, Any]:
    return _choose_rollout_decision(
        state,
        legal,
        rng,
        combat_evaluator=combat_evaluator,
        ppo_policy=ppo_policy,
    ).action


def _choose_rollout_decision(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    rng: random.Random,
    *,
    combat_evaluator: Any | None = None,
    ppo_policy: Any | None = None,
) -> RolloutDecision:
    return choose_rollout_decision(
        state,
        legal,
        rng,
        combat_evaluator=combat_evaluator,
        ppo_policy=ppo_policy,
    )


def _apply_action(
    client: FullRunClientLike,
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    wait_timeout_s: float = 1.0,
) -> dict[str, Any]:
    return apply_backend_action(client, state, action, wait_timeout_s=wait_timeout_s)


def _encode_option_tensors(
    state: dict[str, Any],
    option_actions: list[dict[str, Any]],
) -> dict[str, np.ndarray] | None:
    """Pre-encode state + candidate option tensors for offline ranking."""
    if not option_actions:
        return None
    try:
        from rl_encoder_v2 import build_structured_actions, build_structured_state
        from vocab import load_vocab as _load_vocab

        vocab = _load_vocab()
        ss = build_structured_state(state, vocab)
        sa = build_structured_actions(state, option_actions, vocab)
        encoded: dict[str, np.ndarray] = {}
        for fname in [
            "scalars", "deck_ids", "deck_aux", "deck_mask",
            "relic_ids", "relic_aux", "relic_mask",
            "potion_ids", "potion_mask",
            "hand_ids", "hand_aux", "hand_mask",
            "enemy_ids", "enemy_aux", "enemy_mask",
            "map_node_types", "map_node_mask",
            "reward_card_ids", "reward_card_aux", "reward_card_mask",
            "shop_card_ids", "shop_relic_ids", "shop_potion_ids",
            "shop_prices", "shop_mask",
            "rest_option_ids", "rest_option_mask",
        ]:
            val = getattr(ss, fname, None)
            if val is not None:
                encoded[f"state_{fname}"] = np.asarray(val)
        encoded["state_screen_type_idx"] = np.array(ss.screen_type_idx, dtype=np.int64)
        encoded["state_next_boss_idx"] = np.array(ss.next_boss_idx, dtype=np.int64)
        encoded["state_event_option_count"] = np.array(ss.event_option_count, dtype=np.int64)

        for fname in [
            "action_type_ids", "target_card_ids", "target_enemy_ids",
            "target_node_types", "target_indices", "action_mask",
        ]:
            val = getattr(sa, fname, None)
            if val is not None:
                encoded[f"action_{fname}"] = np.asarray(val)
        return encoded
    except Exception:
        return None


def _settle_after_choice(
    client: FullRunClientLike,
    state: dict[str, Any],
    *,
    previous_state_type: str,
    previous_floor: int,
    max_followups: int = 4,
) -> dict[str, Any]:
    """Advance through same-screen confirmation states after committing a choice."""
    current = state
    for _ in range(max_followups):
        if str(current.get("state_type") or "") != previous_state_type:
            break
        if _extract_floor(current) != previous_floor:
            break
        legal = current.get("legal_actions") or []
        if not legal:
            current = _apply_action(client, current, {"action": "wait"})
            continue
        current = _apply_action(client, current, choose_default_action(current))
    return current


def _advance_to_sampling_point(
    client: FullRunClientLike,
    state: dict[str, Any],
    rng: random.Random,
    *,
    combat_evaluator: Any | None = None,
    ppo_policy: Any | None = None,
    use_local_ort_rollout: bool = False,
    local_ort_max_combat_steps: int = 600,
    max_steps: int = 512,
    stop_floor: int | None = None,
) -> dict[str, Any]:
    """Advance through non-sampled screens until the next sampleable decision point.

    Stop when we reach:
      - terminal
      - map
      - card_reward
      - deterministic non-combat screens (rest/shop/card_select)
    """
    current = state
    last_action_name: str | None = None
    last_reward_claim_sig: str | None = None
    last_reward_claim_count: int | None = None
    reward_chain_card_reward_seen = False

    for _ in range(max_steps):
        if current.get("terminal"):
            return current

        st = str(current.get("state_type") or "").strip().lower()
        if _reached_stop_floor(current, stop_floor) and st not in {"map", "card_reward", "rest_site", "shop", "card_select"}:
            return current
        if st in {"map", "card_reward", "rest_site", "shop", "card_select"}:
            return current

        legal = current.get("legal_actions") or []

        if st in COMBAT_TYPES:
            if use_local_ort_rollout and getattr(client, "supports_local_ort", False):
                result = client.run_combat_local(max_steps=local_ort_max_combat_steps)
                post_state = result.get("state") if isinstance(result, dict) else None
                current = post_state if isinstance(post_state, dict) else client.get_state()
                last_action_name = "run_combat_local"
                last_reward_claim_sig = None
                last_reward_claim_count = None
                reward_chain_card_reward_seen = False
                continue

            if not legal:
                current = _apply_action(client, current, {"action": "wait"})
                last_action_name = "wait"
                continue

            decision = _choose_rollout_decision(
                current,
                legal,
                rng,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
            )
            current = _apply_action(client, current, decision.action)
            last_action_name = str(decision.action.get("action") or "")
            continue

        if not legal:
            current = _apply_action(client, current, {"action": "wait"})
            last_action_name = "wait"
            continue

        auto_action = choose_auto_progress_action(
            current,
            legal,
            last_action_name=last_action_name,
            last_reward_claim_sig=last_reward_claim_sig,
            last_reward_claim_count=last_reward_claim_count,
            reward_chain_card_reward_seen=reward_chain_card_reward_seen,
        )
        if auto_action is not None:
            previous_floor = int(_extract_floor(current))
            current_claim_sig = next_reward_claim_signature(st, current, auto_action)
            current_claim_count = (
                claim_reward_action_count(legal)
                if st == "combat_rewards" else None
            )
            current = _apply_action(client, current, auto_action)
            current = _settle_after_choice(
                client,
                current,
                previous_state_type=st,
                previous_floor=previous_floor,
            )
            last_action_name = str(auto_action.get("action") or "")
            last_reward_claim_sig = current_claim_sig or None
            last_reward_claim_count = current_claim_count
            reward_chain_card_reward_seen = reward_chain_card_reward_seen or st == "card_reward"
            continue

        deterministic_action = _choose_deterministic_screen_action(current, legal)
        if deterministic_action is not None:
            return current

        decision = _choose_rollout_decision(
            current,
            legal,
            rng,
            combat_evaluator=combat_evaluator,
            ppo_policy=ppo_policy,
        )
        previous_floor = int(_extract_floor(current))
        current = _apply_action(client, current, decision.action)
        current = _settle_after_choice(
            client,
            current,
            previous_state_type=st,
            previous_floor=previous_floor,
        )
        last_action_name = str(decision.action.get("action") or "")

    return current


def _apply_action_and_advance(
    client: FullRunClientLike,
    state: dict[str, Any],
    action: dict[str, Any],
    rng: random.Random,
    *,
    combat_evaluator: Any | None = None,
    ppo_policy: Any | None = None,
    use_local_ort_rollout: bool = False,
    local_ort_max_combat_steps: int = 600,
    stop_floor: int | None = None,
) -> dict[str, Any]:
    previous_state_type = str(state.get("state_type") or "")
    previous_floor = _extract_floor(state)
    current = _apply_action(client, state, action)
    current = _settle_after_choice(
        client,
        current,
        previous_state_type=previous_state_type,
        previous_floor=previous_floor,
    )
    return _advance_to_sampling_point(
        client,
        current,
        rng,
        combat_evaluator=combat_evaluator,
        ppo_policy=ppo_policy,
        use_local_ort_rollout=use_local_ort_rollout,
        local_ort_max_combat_steps=local_ort_max_combat_steps,
        stop_floor=stop_floor,
    )


def play_forward_multi_combat(
    client: FullRunClientLike,
    hp_at_start: int,
    rng: random.Random | None = None,
    rollout_goal: str = "terminal",
    max_combats: int = 3,
    max_steps: int = 500,
    combat_evaluator: Any | None = None,
    ppo_policy: Any | None = None,
    trace_sink: list[dict[str, Any]] | None = None,
    use_local_ort_rollout: bool = False,
    local_ort_max_combat_steps: int = 600,
) -> CombatOutcome:
    """Play forward until the configured rollout goal is reached."""
    if rng is None:
        rng = random.Random(42)
    normalized_goal = str(rollout_goal or "combat_horizon").strip().lower()
    if normalized_goal not in {"combat_horizon", "boss_reached_or_terminal", "terminal"}:
        raise ValueError(f"Unsupported rollout_goal: {rollout_goal}")

    state = client.get_state()
    in_combat = False
    combats_completed = 0
    total_turns = 0
    total_hp_lost = 0
    combat_entry_hp = hp_at_start
    current_turns = 0

    for step_idx in range(max_steps):
        st = str(state.get("state_type") or "")
        terminal = state.get("terminal", False)
        current_floor = _extract_floor(state)
        current_run_outcome = state.get("run_outcome")
        current_boss_reached = _did_reach_boss(state)

        if trace_sink is not None:
            trace_sink.append(
                {
                    "kind": "step",
                    "step_index": int(step_idx),
                    "combats_completed": int(combats_completed),
                    "in_combat": bool(in_combat),
                    "combat_entry_hp": int(combat_entry_hp),
                    "current_turns": int(current_turns),
                    "state": _compact_debug_state(state),
                }
            )

        if terminal:
            hp_now = _extract_player_hp(state)
            if in_combat:
                total_hp_lost += max(0, combat_entry_hp - hp_now)
                total_turns += current_turns
            outcome = CombatOutcome(
                won=(state.get("run_outcome") == "victory"),
                hp_after=hp_now,
                hp_lost=total_hp_lost,
                turns=total_turns,
                terminal_state_type=st,
                end_floor=current_floor,
                boss_reached=current_boss_reached,
                run_outcome=(str(current_run_outcome) if current_run_outcome is not None else None),
            )
            if trace_sink is not None:
                trace_sink.append(
                    {
                        "kind": "terminal",
                        "step_index": int(step_idx),
                        "outcome": asdict(outcome),
                    }
                )
            return outcome

        if normalized_goal == "boss_reached_or_terminal" and current_boss_reached:
            hp_now = _extract_player_hp(state)
            if in_combat:
                total_hp_lost += max(0, combat_entry_hp - hp_now)
                total_turns += current_turns
            outcome = CombatOutcome(
                won=False,
                hp_after=hp_now,
                hp_lost=total_hp_lost,
                turns=total_turns,
                terminal_state_type="boss_reached",
                end_floor=current_floor,
                boss_reached=True,
                run_outcome=(str(current_run_outcome) if current_run_outcome is not None else None),
            )
            if trace_sink is not None:
                trace_sink.append(
                    {
                        "kind": "goal_reached",
                        "step_index": int(step_idx),
                        "goal": normalized_goal,
                        "outcome": asdict(outcome),
                    }
                )
            return outcome

        if st in COMBAT_TYPES:
            if use_local_ort_rollout and client.supports_local_ort:
                if not in_combat:
                    in_combat = True
                    combat_entry_hp = _extract_player_hp(state)
                    current_turns = 0
                result = client.run_combat_local(max_steps=local_ort_max_combat_steps)
                post_state = result.get("state") if isinstance(result, dict) else None
                if not isinstance(post_state, dict):
                    post_state = client.get_state()
                hp_now = _extract_player_hp(post_state)
                combat_steps = int((result or {}).get("combat_steps", 0) or 0)
                total_hp_lost += max(0, combat_entry_hp - hp_now)
                total_turns += max(1, combat_steps)
                combats_completed += 1
                in_combat = False
                state = post_state
                if trace_sink is not None:
                    trace_sink.append(
                        {
                            "kind": "local_ort_combat",
                            "step_index": int(step_idx),
                            "combat_steps": int(combat_steps),
                            "hp_after": int(hp_now),
                            "timing": dict((result or {}).get("timing") or {}),
                            "state": _compact_debug_state(post_state),
                        }
                    )
                if state.get("terminal"):
                    outcome = CombatOutcome(
                        won=(state.get("run_outcome") == "victory"),
                        hp_after=hp_now,
                        hp_lost=total_hp_lost,
                        turns=total_turns,
                        terminal_state_type=str(state.get("state_type") or "game_over"),
                        end_floor=_extract_floor(state),
                        boss_reached=_did_reach_boss(state),
                        run_outcome=(str(state.get("run_outcome")) if state.get("run_outcome") is not None else None),
                    )
                    if trace_sink is not None:
                        trace_sink.append(
                            {
                                "kind": "terminal",
                                "step_index": int(step_idx),
                                "outcome": asdict(outcome),
                            }
                        )
                    return outcome
                if normalized_goal == "combat_horizon" and combats_completed >= max_combats:
                    outcome = CombatOutcome(
                        won=True,
                        hp_after=hp_now,
                        hp_lost=total_hp_lost,
                        turns=total_turns,
                        terminal_state_type=f"completed_{combats_completed}_combats",
                        end_floor=_extract_floor(state),
                        boss_reached=_did_reach_boss(state),
                        run_outcome=(str(state.get("run_outcome")) if state.get("run_outcome") is not None else None),
                    )
                    if trace_sink is not None:
                        trace_sink.append(
                            {
                                "kind": "completed",
                                "step_index": int(step_idx),
                                "outcome": asdict(outcome),
                            }
                        )
                    return outcome
                continue
            if not in_combat:
                in_combat = True
                combat_entry_hp = _extract_player_hp(state)
                current_turns = 0
            battle = state.get("battle") or {}
            current_turns = battle.get("round_number") or battle.get("round") or current_turns
        elif in_combat:
            # Just exited a combat
            hp_now = _extract_player_hp(state)
            total_hp_lost += max(0, combat_entry_hp - hp_now)
            total_turns += current_turns
            combats_completed += 1
            in_combat = False

            if normalized_goal == "combat_horizon" and combats_completed >= max_combats:
                outcome = CombatOutcome(
                    won=True,
                    hp_after=hp_now,
                    hp_lost=total_hp_lost,
                    turns=total_turns,
                    terminal_state_type=f"completed_{combats_completed}_combats",
                    end_floor=_extract_floor(state),
                    boss_reached=_did_reach_boss(state),
                    run_outcome=(str(state.get("run_outcome")) if state.get("run_outcome") is not None else None),
                )
                if trace_sink is not None:
                    trace_sink.append(
                        {
                            "kind": "completed",
                            "step_index": int(step_idx),
                            "outcome": asdict(outcome),
                        }
                    )
                return outcome

        legal = state.get("legal_actions") or []
        if not legal:
            if trace_sink is not None:
                trace_sink.append(
                    {
                        "kind": "action",
                        "step_index": int(step_idx),
                        "action": {"action": "wait"},
                        "reason": "no_legal_actions",
                    }
                )
            state = _apply_action(client, state, {"action": "wait"})
            continue

        decision = _choose_rollout_decision(
            state,
            legal,
            rng,
            combat_evaluator=combat_evaluator,
            ppo_policy=ppo_policy,
        )
        if trace_sink is not None:
            trace_sink.append(
                {
                    "kind": "action",
                    "step_index": int(step_idx),
                    "source": decision.source,
                    "action": _compact_debug_action(decision.action, state),
                }
            )
        state = _apply_action(client, state, decision.action)

    # Timeout
    hp_now = _extract_player_hp(state)
    if in_combat:
        total_hp_lost += max(0, combat_entry_hp - hp_now)
        total_turns += current_turns
    outcome = CombatOutcome(
        won=(str(state.get("run_outcome") or "").lower() == "victory"),
        hp_after=hp_now,
        hp_lost=total_hp_lost,
        turns=total_turns,
        terminal_state_type=f"timeout_after_{combats_completed}_combats",
        end_floor=_extract_floor(state),
        boss_reached=_did_reach_boss(state),
        run_outcome=(str(state.get("run_outcome")) if state.get("run_outcome") is not None else None),
    )
    if trace_sink is not None:
        trace_sink.append(
            {
                "kind": "timeout",
                "step_index": int(max_steps),
                "outcome": asdict(outcome),
                "final_state": _compact_debug_state(state),
            }
        )
    return outcome


def compute_option_scores(
    outcomes: dict[int, CombatOutcome],
    max_hp: int = 80,
) -> list[float]:
    """Compute boss-oriented ranking scores from branch outcomes.

    Goal order:
    1. Win the act / run.
    2. If not winning, prefer branches that at least reach the boss.
    3. Among boss-loss branches, prefer deeper progress with less cumulative HP bleed.
    4. Only use speed as a weak tie-breaker.
    """
    scores = []
    max_idx = max(outcomes.keys()) + 1

    for i in range(max_idx):
        outcome = outcomes.get(i)
        if outcome is None:
            scores.append(0.0)
            continue

        run_outcome = normalize_run_outcome(outcome.run_outcome, default="")
        victory = 1.0 if run_outcome == RUN_OUTCOME_VICTORY else 0.0
        boss_reached = 1.0 if outcome.boss_reached else 0.0
        floor_progress = min(max(int(outcome.end_floor), 0), 18) / 18.0
        hp_ratio = max(0, outcome.hp_after) / max(1, max_hp)
        hp_loss_efficiency = max(0.0, 1.0 - (max(0, outcome.hp_lost) / max(1.0, float(max_hp) * 3.0)))
        max_expected_turns = 30.0
        speed = max(0.0, 1.0 - outcome.turns / max_expected_turns)
        boss_loss_bonus = boss_reached * (1.0 - victory)
        preboss_defeat_penalty = 1.0 if is_failure_outcome(run_outcome) and boss_reached <= 0.0 else 0.0

        score = (
            1.6 * victory
            + 0.7 * boss_reached
            + 0.45 * floor_progress
            + 0.15 * hp_ratio
            + 0.30 * hp_loss_efficiency
            + 0.10 * boss_loss_bonus * hp_loss_efficiency
            + 0.03 * speed
            - 0.15 * preboss_defeat_penalty
        )
        scores.append(round(score, 4))

    return scores


def _score_spread(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return float(max(scores) - min(scores))


def _evaluate_branch_outcomes(
    *,
    client: FullRunClientLike,
    seed: str,
    floor: int,
    hp_before: int,
    sample_index: int,
    sample_type: str,
    options: list[dict[str, Any]],
    restore_fn: Any,
    combat_evaluator: Any | None,
    ppo_policy: Any | None,
    debug_rollout_trace_dir: str | None,
    rollout_goal: str,
    max_combats: int,
    max_steps: int,
    use_local_ort_rollout: bool,
    local_ort_max_combat_steps: int,
    trace_store: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[int, CombatOutcome]:
    combat_outcomes: dict[int, CombatOutcome] = {}
    for i, option in enumerate(options):
        try:
            restore_fn(i)
            combat_rng = random.Random(f"{seed}_f{floor}_shared")
            trace_entries: list[dict[str, Any]] | None = [] if debug_rollout_trace_dir else None
            outcome = play_forward_multi_combat(
                client,
                hp_before,
                rng=combat_rng,
                rollout_goal=rollout_goal,
                max_combats=max_combats,
                max_steps=max_steps,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                trace_sink=trace_entries,
                use_local_ort_rollout=use_local_ort_rollout,
                local_ort_max_combat_steps=local_ort_max_combat_steps,
            )
            _write_rollout_trace(
                debug_rollout_trace_dir,
                seed=seed,
                sample_index=sample_index,
                sample_type=sample_type,
                floor=int(floor),
                option_index=i,
                option_label=str(option.get("card_name") or option.get("card_id") or option.get("label") or option.get("type") or i),
                trace_entries=trace_entries or [],
            )
            if trace_store is not None:
                trace_store[i] = list(trace_entries or [])
            combat_outcomes[i] = outcome
        except Exception as exc:
            if trace_store is not None:
                trace_store[i] = []
            combat_outcomes[i] = CombatOutcome(
                won=False,
                hp_after=0,
                hp_lost=hp_before,
                turns=0,
                terminal_state_type=f"error:{exc}",
                end_floor=int(floor),
                boss_reached=False,
                run_outcome="error",
            )
    return combat_outcomes


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_from_episode(
    client: FullRunClientLike,
    seed: str,
    *,
    port: int | None = None,
    transport: str = "pipe-binary",
    checkpoint_path: str | None = None,
    checkpoint_sha256: str | None = None,
    combat_checkpoint_path: str | None = None,
    combat_checkpoint_sha256: str | None = None,
    generator_config: dict[str, Any] | None = None,
    max_steps: int = 600,
    combat_evaluator: Any | None = None,
    ppo_policy: Any | None = None,
    debug_rollout_trace_dir: str | None = None,
    rollout_max_combats: int = 3,
    rollout_max_steps: int = 500,
    rerun_low_spread_threshold: float = 0.0,
    rerun_max_combats: int = 5,
    rerun_max_steps: int = 900,
    rollout_goal: str = "terminal",
    label_mode: str = "single_step",
    reward_tree_config: RewardTreeConfig | None = None,
    map_route_config: MapRouteConfig | None = None,
    use_local_ort_rollout: bool = False,
    local_ort_max_combat_steps: int = 600,
    stop_floor: int | None = None,
    sample_types: set[str] | None = None,
) -> tuple[list[CardRankingSample], EpisodeGenerationSummary, list[dict[str, Any]]]:
    """Run one episode, intercept card_reward screens, evaluate each option."""
    samples: list[CardRankingSample] = []
    raw_branch_records: list[dict[str, Any]] = []
    rng = random.Random(seed)

    state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
    boss_reached = _did_reach_boss(state)
    final_status = "ok"
    map_decisions_seen = 0
    map_samples_recorded = 0
    card_reward_decisions_seen = 0
    card_reward_samples_recorded = 0
    sampling_skip_reasons: Counter[str] = Counter()

    enabled_sample_types = {str(sample_type).strip().lower() for sample_type in (sample_types or {"map", "card_reward"})}

    for step_i in range(max_steps):
        st = str(state.get("state_type") or "")
        if state.get("terminal"):
            break
        if _reached_stop_floor(state, stop_floor) and st not in {"map", "card_reward"}:
            final_status = "floor_cap"
            break
        boss_reached = boss_reached or _did_reach_boss(state)

        if st == "map" and "map" in enabled_sample_types and map_route_config is not None:
            map_options = _extract_map_options(state)
            if len(map_options) >= 2:
                map_decisions_seen += 1
                try:
                    deck_ids = _extract_deck_ids(state)
                    relic_ids = _extract_relic_ids(state)
                    floor = (state.get("run") or {}).get("floor") or 0
                    act = (state.get("run") or {}).get("act") or 1
                    encoded_tensors = _encode_option_tensors(
                        state,
                        [opt["action"] for opt in map_options],
                    )
                    map_result = evaluate_map_route_tree(
                        client=client,
                        seed=seed,
                        state=state,
                        root_options=map_options,
                        config=map_route_config,
                        apply_action=_apply_action,
                        settle_after_choice=_settle_after_choice,
                        extract_floor=_extract_floor,
                        extract_player_hp=_extract_player_hp,
                        did_reach_boss=_did_reach_boss,
                        choose_rollout_decision=_choose_rollout_decision,
                        choose_deterministic_screen_action=_choose_deterministic_screen_action,
                        resolve_card_reward_choice=lambda _client, _branch_seed, _branch_state: _resolve_best_card_reward_choice(
                            client=_client,
                            seed=_branch_seed,
                            state=_branch_state,
                            sample_index=len(samples),
                            combat_evaluator=combat_evaluator,
                            ppo_policy=ppo_policy,
                            rollout_max_combats=rollout_max_combats,
                            rollout_max_steps=rollout_max_steps,
                            rerun_low_spread_threshold=rerun_low_spread_threshold,
                            rerun_max_combats=rerun_max_combats,
                            rerun_max_steps=rerun_max_steps,
                            rollout_goal=rollout_goal,
                            label_mode=label_mode,
                            reward_tree_config=reward_tree_config,
                            use_local_ort_rollout=use_local_ort_rollout,
                            local_ort_max_combat_steps=local_ort_max_combat_steps,
                        ),
                        extract_map_options=_extract_map_options,
                        use_local_ort_rollout=use_local_ort_rollout,
                        local_ort_max_combat_steps=local_ort_max_combat_steps,
                    )
                    best_idx = int(max(range(len(map_result.scores)), key=lambda i: map_result.scores[i]))
                    tensor_rel_path = f"tensors/sample_{len(samples):05d}.npz" if encoded_tensors else None
                    sample = CardRankingSample(
                        deck_ids=deck_ids,
                        relic_ids=relic_ids,
                        floor=floor,
                        act=act,
                        options=[{k: v for k, v in opt.items() if k != "action"} for opt in map_options],
                        scores=list(map_result.scores),
                        best_idx=best_idx,
                        combat_outcomes={str(k): v for k, v in map_result.route_outcomes.items()},
                        sample_type="map",
                        label_source="map_route_tree",
                        option_tree_values=[asdict(v) for v in map_result.option_values],
                        tree_summary=map_result.summary,
                        state_tensors_path=tensor_rel_path,
                        _encoded_tensors=encoded_tensors,
                    )
                    samples.append(sample)
                    map_samples_recorded += 1
                    raw_branch_records.append(
                        _build_raw_branch_record(
                            seed=seed,
                            sample_index=len(samples) - 1,
                            sample_type="map",
                            label_source="map_route_tree",
                            root_state=state,
                            options=map_options,
                            scores=list(map_result.scores),
                            best_idx=best_idx,
                            combat_outcomes=map_result.route_outcomes,
                            option_traces={},
                            tree_summary=map_result.summary,
                            option_tree_values=[asdict(v) for v in map_result.option_values],
                            port=port,
                            transport=transport,
                            checkpoint_path=checkpoint_path,
                            checkpoint_sha256=checkpoint_sha256,
                            combat_checkpoint_path=combat_checkpoint_path,
                            combat_checkpoint_sha256=combat_checkpoint_sha256,
                            generator_config=generator_config or {},
                        )
                    )
                    state = _apply_action_and_advance(
                        client,
                        state,
                        map_options[best_idx]["action"],
                        rng,
                        combat_evaluator=combat_evaluator,
                        ppo_policy=ppo_policy,
                        use_local_ort_rollout=use_local_ort_rollout,
                        local_ort_max_combat_steps=local_ort_max_combat_steps,
                        stop_floor=stop_floor,
                    )
                except Exception:
                    sampling_skip_reasons["search_error"] += 1
                    state = _apply_action_and_advance(
                        client,
                        state,
                        map_options[0]["action"],
                        rng,
                        combat_evaluator=combat_evaluator,
                        ppo_policy=ppo_policy,
                        use_local_ort_rollout=use_local_ort_rollout,
                        local_ort_max_combat_steps=local_ort_max_combat_steps,
                        stop_floor=stop_floor,
                    )
                continue
            if len(map_options) == 1:
                sampling_skip_reasons["single_path_map"] += 1
                state = _apply_action_and_advance(
                    client,
                    state,
                    map_options[0]["action"],
                    rng,
                    combat_evaluator=combat_evaluator,
                    ppo_policy=ppo_policy,
                    use_local_ort_rollout=use_local_ort_rollout,
                    local_ort_max_combat_steps=local_ort_max_combat_steps,
                    stop_floor=stop_floor,
                )
                continue

        if st == "card_reward" and "card_reward" in enabled_sample_types:
            options = _extract_card_reward_options(state)
            if len(options) >= 2:
                card_reward_decisions_seen += 1
                deck_ids = _extract_deck_ids(state)
                relic_ids = _extract_relic_ids(state)
                floor = (state.get("run") or {}).get("floor") or 0
                act = (state.get("run") or {}).get("act") or 1
                hp_before = _extract_player_hp(state)

                encoded_tensors = _encode_option_tensors(
                    state,
                    [opt["action"] for opt in options],
                )

                combat_outcomes: dict[int, CombatOutcome] = {}
                card_reward_handled = False
                tree_result = None
                option_traces: dict[int, list[dict[str, Any]]] = {}
                continuation_state_id: str | None = None
                with tempfile.TemporaryDirectory(prefix=f"sts2_card_reward_f{int(floor):02d}_") as tmpdir:
                    snapshot_path = str(Path(tmpdir) / "branch_snapshot.json")
                    try:
                        continuation_state_id = client.save_state()
                        client.export_state(snapshot_path)
                        if label_mode == "reward_tree" and reward_tree_config is not None:
                            tree_result = evaluate_card_reward_tree(
                                client=client,
                                seed=seed,
                                state=client.get_state(),
                                root_options=options,
                                sample_index=len(samples),
                                debug_rollout_trace_dir=debug_rollout_trace_dir,
                                combat_evaluator=combat_evaluator,
                                ppo_policy=ppo_policy,
                                config=reward_tree_config,
                                apply_action=_apply_action,
                                settle_after_choice=_settle_after_choice,
                                extract_floor=_extract_floor,
                                extract_player_hp=_extract_player_hp,
                                extract_card_reward_options=_extract_card_reward_options,
                                did_reach_boss=_did_reach_boss,
                                choose_rollout_decision=_choose_rollout_decision,
                                evaluate_branch_outcomes=_evaluate_branch_outcomes,
                                compute_option_scores=compute_option_scores,
                            )
                            combat_outcomes = tree_result.root_outcomes
                        else:
                            def _restore_reward_export(option_index: int) -> None:
                                branch_base = client.import_state(snapshot_path)
                                branch_state = _apply_action(client, branch_base, options[option_index]["action"])
                                _settle_after_choice(
                                    client,
                                    branch_state,
                                    previous_state_type=st,
                                    previous_floor=int(floor),
                                )

                            combat_outcomes = _evaluate_branch_outcomes(
                                client=client,
                                seed=seed,
                                floor=int(floor),
                                hp_before=hp_before,
                                sample_index=len(samples),
                                sample_type="card_reward",
                                options=options,
                                restore_fn=_restore_reward_export,
                                combat_evaluator=combat_evaluator,
                                ppo_policy=ppo_policy,
                                debug_rollout_trace_dir=debug_rollout_trace_dir,
                                rollout_goal=rollout_goal,
                                max_combats=rollout_max_combats,
                                max_steps=rollout_max_steps,
                                use_local_ort_rollout=use_local_ort_rollout,
                                local_ort_max_combat_steps=local_ort_max_combat_steps,
                                trace_store=option_traces,
                            )
                            scores = compute_option_scores(combat_outcomes, max_hp=max(hp_before, 1))
                            if (
                                rerun_low_spread_threshold > 0
                                and _score_spread(scores) < rerun_low_spread_threshold
                                and rerun_max_combats > rollout_max_combats
                            ):
                                combat_outcomes = _evaluate_branch_outcomes(
                                    client=client,
                                    seed=seed,
                                    floor=int(floor),
                                    hp_before=hp_before,
                                    sample_index=len(samples),
                                    sample_type="card_reward",
                                    options=options,
                                    restore_fn=_restore_reward_export,
                                    combat_evaluator=combat_evaluator,
                                    ppo_policy=ppo_policy,
                                    debug_rollout_trace_dir=debug_rollout_trace_dir,
                                    rollout_goal=rollout_goal,
                                    max_combats=rerun_max_combats,
                                    max_steps=rerun_max_steps,
                                    use_local_ort_rollout=use_local_ort_rollout,
                                    local_ort_max_combat_steps=local_ort_max_combat_steps,
                                    trace_store=option_traces,
                                )

                        if continuation_state_id:
                            restored_state = client.load_state(continuation_state_id)
                            if str(restored_state.get("state_type") or "").strip().lower() == "card_reward":
                                state = restored_state
                                card_reward_handled = True
                    except Exception:
                        card_reward_handled = False
                    finally:
                        pass

                if card_reward_handled:
                    scores = tree_result.scores if tree_result is not None else compute_option_scores(combat_outcomes, max_hp=max(hp_before, 1))
                    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
                    tensor_rel_path = f"tensors/sample_{len(samples):05d}.npz" if encoded_tensors else None
                    sample = CardRankingSample(
                        deck_ids=deck_ids,
                        relic_ids=relic_ids,
                        floor=floor,
                        act=act,
                        options=[{k: v for k, v in opt.items() if k != "action"} for opt in options],
                        scores=scores,
                        best_idx=best_idx,
                        combat_outcomes={str(k): asdict(v) for k, v in combat_outcomes.items()},
                        label_source="reward_tree" if tree_result is not None else "single_step",
                        option_tree_values=(
                            [asdict(v) for v in tree_result.option_values]
                            if tree_result is not None else None
                        ),
                        tree_summary=tree_result.summary if tree_result is not None else None,
                        state_tensors_path=tensor_rel_path,
                        _encoded_tensors=encoded_tensors,
                    )
                    samples.append(sample)
                    raw_branch_records.append(
                        _build_raw_branch_record(
                            seed=seed,
                            sample_index=len(samples) - 1,
                            sample_type="card_reward",
                            label_source=sample.label_source,
                            root_state=state,
                            options=options,
                            scores=scores,
                            best_idx=best_idx,
                            combat_outcomes=combat_outcomes,
                            option_traces=option_traces,
                            tree_summary=tree_result.summary if tree_result is not None else None,
                            option_tree_values=(
                                [asdict(v) for v in tree_result.option_values]
                                if tree_result is not None else None
                            ),
                            port=port,
                            transport=transport,
                            checkpoint_path=checkpoint_path,
                            checkpoint_sha256=checkpoint_sha256,
                            combat_checkpoint_path=combat_checkpoint_path,
                            combat_checkpoint_sha256=combat_checkpoint_sha256,
                            generator_config=generator_config or {},
                        )
                    )

                    best_action = options[best_idx]["action"]
                    live_state = client.get_state()
                    if str(live_state.get("state_type") or "").strip().lower() != "card_reward":
                        final_status = "post_card_reward_restore_failed"
                        state = live_state
                        break
                    card_reward_samples_recorded += 1
                    state = _apply_action_and_advance(
                        client,
                        state,
                        best_action,
                        rng,
                        combat_evaluator=combat_evaluator,
                        ppo_policy=ppo_policy,
                        use_local_ort_rollout=use_local_ort_rollout,
                        local_ort_max_combat_steps=local_ort_max_combat_steps,
                        stop_floor=stop_floor,
                    )
                    if continuation_state_id:
                        try:
                            client.delete_state(continuation_state_id)
                        except Exception:
                            pass
                    continue

                # Save one snapshot PER option so each state_id is loaded
                # exactly once. C# enforces strict signature verification for
                # card_reward states — reusing the same state_id after the game
                # state has drifted triggers a signature mismatch error.
                num_saves = len(options) + 1  # +1 for the restore-to-continue save
                state_ids: list[str] = []
                if continuation_state_id:
                    try:
                        restored_state = client.load_state(continuation_state_id)
                        if isinstance(restored_state, dict):
                            state = restored_state
                    except Exception:
                        pass
                try:
                    for _ in range(num_saves):
                        state_ids.append(client.save_state())
                except Exception:
                    # Save not supported — skip this sample
                    for sid in state_ids:
                        try:
                            client.delete_state(sid)
                        except Exception:
                            pass
                    action = _choose_rollout_action(
                        state,
                        state.get("legal_actions") or [],
                        rng,
                        combat_evaluator=combat_evaluator,
                        ppo_policy=ppo_policy,
                    )
                    state = _apply_action(client, state, action)
                    continue

                deck_ids = _extract_deck_ids(state)
                relic_ids = _extract_relic_ids(state)
                floor = (state.get("run") or {}).get("floor") or 0
                act = (state.get("run") or {}).get("act") or 1
                hp_before = _extract_player_hp(state)

                encoded_tensors = _encode_option_tensors(
                    state,
                    [opt["action"] for opt in options],
                )

                def _restore_reward_save(option_index: int) -> None:
                    client.load_state(state_ids[option_index])
                    branch_state = _apply_action(client, client.get_state(), options[option_index]["action"])
                    _settle_after_choice(
                        client,
                        branch_state,
                        previous_state_type=st,
                        previous_floor=int(floor),
                    )

                combat_outcomes = _evaluate_branch_outcomes(
                    client=client,
                    seed=seed,
                    floor=int(floor),
                    hp_before=hp_before,
                    sample_index=len(samples),
                    sample_type="card_reward",
                    options=options,
                    restore_fn=_restore_reward_save,
                    combat_evaluator=combat_evaluator,
                    ppo_policy=ppo_policy,
                    debug_rollout_trace_dir=debug_rollout_trace_dir,
                    rollout_goal=rollout_goal,
                    max_combats=rollout_max_combats,
                    max_steps=rollout_max_steps,
                    use_local_ort_rollout=use_local_ort_rollout,
                    local_ort_max_combat_steps=local_ort_max_combat_steps,
                    trace_store=option_traces,
                )
                scores = compute_option_scores(combat_outcomes, max_hp=max(hp_before, 1))
                if (
                    rerun_low_spread_threshold > 0
                    and _score_spread(scores) < rerun_low_spread_threshold
                    and rerun_max_combats > rollout_max_combats
                ):
                    combat_outcomes = _evaluate_branch_outcomes(
                        client=client,
                        seed=seed,
                        floor=int(floor),
                        hp_before=hp_before,
                        sample_index=len(samples),
                        sample_type="card_reward",
                        options=options,
                        restore_fn=_restore_reward_save,
                        combat_evaluator=combat_evaluator,
                        ppo_policy=ppo_policy,
                        debug_rollout_trace_dir=debug_rollout_trace_dir,
                        rollout_goal=rollout_goal,
                        max_combats=rerun_max_combats,
                        max_steps=rerun_max_steps,
                        use_local_ort_rollout=use_local_ort_rollout,
                        local_ort_max_combat_steps=local_ort_max_combat_steps,
                        trace_store=option_traces,
                    )

                # Restore to original state using the last (unused) save.
                # Keep that continuation snapshot alive until after we apply the
                # chosen card; deleting the active loaded state before stepping
                # leaves the backend out of sync with the local `state` dict.
                continuation_sid = state_ids[-1] if state_ids else None
                try:
                    if continuation_sid:
                        state = client.load_state(continuation_sid)
                except Exception:
                    pass
                # Clean up branch-only snapshots, but keep the continuation one
                # until the selected reward action has been applied.
                for sid in state_ids[:-1]:
                    try:
                        client.delete_state(sid)
                    except Exception:
                        pass

                scores = compute_option_scores(combat_outcomes, max_hp=max(hp_before, 1))
                best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))

                # Determine tensor path (will be saved later by caller)
                tensor_rel_path = f"tensors/sample_{len(samples):05d}.npz" if encoded_tensors else None

                sample = CardRankingSample(
                    deck_ids=deck_ids,
                    relic_ids=relic_ids,
                    floor=floor,
                    act=act,
                    options=[{k: v for k, v in opt.items() if k != "action"} for opt in options],
                    scores=scores,
                    best_idx=best_idx,
                    combat_outcomes={
                        str(k): asdict(v) for k, v in combat_outcomes.items()
                    },
                    state_tensors_path=tensor_rel_path,
                    _encoded_tensors=encoded_tensors,
                )
                samples.append(sample)
                raw_branch_records.append(
                    _build_raw_branch_record(
                        seed=seed,
                        sample_index=len(samples) - 1,
                        sample_type="card_reward",
                        label_source="single_step",
                        root_state=state,
                        options=options,
                        scores=scores,
                        best_idx=best_idx,
                        combat_outcomes=combat_outcomes,
                        option_traces=option_traces,
                        tree_summary=None,
                        option_tree_values=None,
                        port=port,
                        transport=transport,
                        checkpoint_path=checkpoint_path,
                        checkpoint_sha256=checkpoint_sha256,
                        combat_checkpoint_path=combat_checkpoint_path,
                        combat_checkpoint_sha256=combat_checkpoint_sha256,
                        generator_config=generator_config or {},
                    )
                )

                # Continue episode: select best option
                best_action = options[best_idx]["action"]
                live_state = client.get_state()
                if str(live_state.get("state_type") or "").strip().lower() != "card_reward":
                    final_status = "post_card_reward_restore_failed"
                    state = live_state
                    break
                card_reward_samples_recorded += 1
                state = _apply_action_and_advance(
                    client,
                    state,
                    best_action,
                    rng,
                    combat_evaluator=combat_evaluator,
                    ppo_policy=ppo_policy,
                    use_local_ort_rollout=use_local_ort_rollout,
                    local_ort_max_combat_steps=local_ort_max_combat_steps,
                    stop_floor=stop_floor,
                )
                if continuation_sid:
                    try:
                        client.delete_state(continuation_sid)
                    except Exception:
                        pass
                if continuation_state_id:
                    try:
                        client.delete_state(continuation_state_id)
                    except Exception:
                        pass
                continue
            sampling_skip_reasons["invalid_card_reward_options"] += 1
            legal = state.get("legal_actions") or []
            fallback_action = options[0]["action"] if len(options) == 1 else (legal[0] if legal else {"action": "wait"})
            state = _apply_action_and_advance(
                client,
                state,
                fallback_action,
                rng,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                use_local_ort_rollout=use_local_ort_rollout,
                local_ort_max_combat_steps=local_ort_max_combat_steps,
                stop_floor=stop_floor,
            )
            continue

        deterministic_action = _choose_deterministic_screen_action(state, state.get("legal_actions") or [])
        if deterministic_action is not None:
            state = _apply_action_and_advance(
                client,
                state,
                deterministic_action,
                rng,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                use_local_ort_rollout=use_local_ort_rollout,
                local_ort_max_combat_steps=local_ort_max_combat_steps,
                stop_floor=stop_floor,
            )
            continue

        # Default: take an action and continue
        legal = state.get("legal_actions") or []
        if not legal:
            state = _advance_to_sampling_point(
                client,
                _apply_action(client, state, {"action": "wait"}),
                rng,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                use_local_ort_rollout=use_local_ort_rollout,
                local_ort_max_combat_steps=local_ort_max_combat_steps,
                stop_floor=stop_floor,
            )
            continue

        if st in COMBAT_TYPES and use_local_ort_rollout and client.supports_local_ort:
            result = client.run_combat_local(max_steps=local_ort_max_combat_steps)
            post_state = result.get("state") if isinstance(result, dict) else None
            state = _advance_to_sampling_point(
                client,
                post_state if isinstance(post_state, dict) else client.get_state(),
                rng,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                use_local_ort_rollout=use_local_ort_rollout,
                local_ort_max_combat_steps=local_ort_max_combat_steps,
                stop_floor=stop_floor,
            )
            continue

        action = _choose_rollout_action(
            state,
            legal,
            rng,
            combat_evaluator=combat_evaluator,
            ppo_policy=ppo_policy,
        )
        state = _apply_action_and_advance(
            client,
            state,
            action,
            rng,
            combat_evaluator=combat_evaluator,
            ppo_policy=ppo_policy,
            use_local_ort_rollout=use_local_ort_rollout,
            local_ort_max_combat_steps=local_ort_max_combat_steps,
            stop_floor=stop_floor,
        )

    if not state.get("terminal") and final_status == "ok":
        final_status = "timeout"
    if final_status == "floor_cap":
        final_outcome = "floor_cap"
    else:
        final_outcome = normalize_run_outcome(
            state.get("run_outcome") or (RUN_OUTCOME_TIMEOUT if final_status == "timeout" else RUN_OUTCOME_DEATH)
        )
    summary = EpisodeGenerationSummary(
        seed=seed,
        status=final_status,
        sample_count=len(samples),
        end_floor=_extract_floor(state),
        boss_reached=boss_reached or _did_reach_boss(state),
        act1_cleared=_did_clear_act1(state),
        outcome=final_outcome,
        map_decisions_seen=int(map_decisions_seen),
        map_samples_recorded=int(map_samples_recorded),
        card_reward_decisions_seen=int(card_reward_decisions_seen),
        card_reward_samples_recorded=int(card_reward_samples_recorded),
        sampling_skip_reasons={
            key: int(sampling_skip_reasons.get(key, 0))
            for key in ("single_path_map", "invalid_card_reward_options", "search_error", "post_terminal")
        },
    )
    return samples, summary, raw_branch_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _safe_load_state_dict(model: Any, state_dict: dict[str, Any] | None) -> None:
    if not state_dict:
        return
    current = model.state_dict()
    filtered: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None):
            filtered[key] = value
    model.load_state_dict(filtered, strict=False)


def _infer_ppo_embed_dim(
    state_dict: dict[str, Any] | None,
    fallback: int = 32,
) -> int:
    if isinstance(state_dict, dict):
        weight = state_dict.get("entity_emb.card_embed.weight")
        if getattr(weight, "ndim", None) == 2:
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
        if getattr(card_weight, "ndim", None) == 2:
            embed_dim = int(card_weight.shape[1])
        if getattr(action_proj, "ndim", None) == 2:
            hidden_dim = int(action_proj.shape[0])
    return embed_dim, hidden_dim


def _infer_deck_repr_dim(state_dict: dict[str, Any] | None) -> int:
    if not isinstance(state_dict, dict):
        return 0
    norm_weight = state_dict.get("deck_encoder.norm.weight")
    if getattr(norm_weight, "ndim", None) == 1:
        return int(norm_weight.shape[0])
    return 0


def _infer_retrieval_proj_dim(state_dict: dict[str, Any] | None) -> int:
    if not isinstance(state_dict, dict):
        return 0
    out_proj = state_dict.get("symbolic_head.out_proj.weight")
    if getattr(out_proj, "ndim", None) == 2:
        return int(out_proj.shape[0])
    return 0


def _load_combat_evaluator(
    checkpoint_path: str | None,
    combat_checkpoint_path: str | None = None,
) -> Any | None:
    """Load CombatNNEvaluator from a hybrid and/or standalone combat checkpoint."""
    if not checkpoint_path and not combat_checkpoint_path:
        return None
    try:
        import torch
        from vocab import load_vocab
        from combat_nn import CombatPolicyValueNetwork, CombatNNEvaluator
        from rl_policy_v2 import FullRunPolicyNetworkV2

        vocab = load_vocab()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt: dict[str, Any] = {}
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ppo_state = ckpt.get("ppo_model") or ckpt.get("ppo_state_dict") or ckpt.get("model_state_dict")
        combat_state = ckpt.get("mcts_model") or ckpt.get("mcts_state_dict")
        ppo_embed_dim = _infer_ppo_embed_dim(ppo_state, 32)
        retrieval_proj_dim = _infer_retrieval_proj_dim(ppo_state)
        use_retrieval = retrieval_proj_dim > 0

        ppo_net = FullRunPolicyNetworkV2(
            vocab=vocab,
            embed_dim=ppo_embed_dim,
            use_symbolic_features=use_retrieval,
            symbolic_proj_dim=retrieval_proj_dim if use_retrieval else 16,
        )
        if ppo_state:
            _safe_load_state_dict(ppo_net, ppo_state)

        combat_embed_dim, combat_hidden_dim = _infer_combat_dims(
            combat_state, ppo_embed_dim, 128,
        )
        deck_repr_dim = _infer_deck_repr_dim(combat_state)
        combat_net = CombatPolicyValueNetwork(
            vocab=vocab,
            embed_dim=combat_embed_dim,
            hidden_dim=combat_hidden_dim,
            entity_embeddings=ppo_net.entity_emb,
            deck_repr_dim=deck_repr_dim,
            symbolic_head=ppo_net.symbolic_head,
        )
        if combat_state:
            _safe_load_state_dict(combat_net, combat_state)
        if combat_checkpoint_path:
            combat_ckpt = torch.load(combat_checkpoint_path, map_location="cpu", weights_only=False)
            _safe_load_state_dict(
                combat_net,
                combat_ckpt.get("mcts_model") or combat_ckpt.get("model_state_dict"),
            )

        evaluator = CombatNNEvaluator(combat_net, vocab, device=device)
        print(
            "Combat NN loaded "
            f"(hybrid={checkpoint_path or '-'} combat={combat_checkpoint_path or '-'}) "
            f"(device={device})"
        )
        return evaluator
    except Exception as exc:
        print(f"WARNING: Failed to load combat NN: {exc}")
        return None


def _load_ppo_rollout_policy(checkpoint_path: str | None) -> Any | None:
    """Load the non-combat champion policy for dataset rollouts."""
    if not checkpoint_path:
        return None
    try:
        import torch
        from vocab import load_vocab
        from rl_policy_v2 import FullRunPolicyNetworkV2, RLFullRunPolicyV2

        vocab = load_vocab()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ppo_state = ckpt.get("ppo_model") or ckpt.get("model_state_dict")
        if not isinstance(ppo_state, dict):
            raise ValueError("checkpoint has no ppo_model/model_state_dict")

        embed_dim = _infer_ppo_embed_dim(ppo_state, 32)
        retrieval_proj_dim = _infer_retrieval_proj_dim(ppo_state)
        use_retrieval = retrieval_proj_dim > 0
        network = FullRunPolicyNetworkV2(
            vocab=vocab,
            embed_dim=embed_dim,
            use_symbolic_features=use_retrieval,
            symbolic_proj_dim=retrieval_proj_dim if use_retrieval else 16,
        )
        _safe_load_state_dict(network, ppo_state)
        network.eval()
        policy = RLFullRunPolicyV2(network=network, vocab=vocab, deterministic=True)
        print(f"PPO rollout policy loaded (checkpoint={checkpoint_path})")
        return policy
    except Exception as exc:
        print(f"WARNING: Failed to load PPO rollout policy: {exc}")
        return None


def _prepare_local_ort_rollout_model(
    *,
    checkpoint_path: str | None,
    combat_checkpoint_path: str | None,
    output_dir: Path,
) -> str | None:
    if not checkpoint_path and not combat_checkpoint_path:
        return None
    try:
        import torch
        from export_actor_onnx import export_from_training_snapshot
        from vocab import load_vocab

        hybrid_ckpt = {}
        if checkpoint_path:
            hybrid_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ppo_state = hybrid_ckpt.get("ppo_model") or hybrid_ckpt.get("ppo_state_dict") or hybrid_ckpt.get("model_state_dict") or {}
        combat_state = hybrid_ckpt.get("mcts_model") or hybrid_ckpt.get("mcts_state_dict") or {}
        if combat_checkpoint_path:
            combat_ckpt = torch.load(combat_checkpoint_path, map_location="cpu", weights_only=False)
            combat_state = combat_ckpt.get("mcts_model") or combat_ckpt.get("model_state_dict") or combat_state
        if not isinstance(combat_state, dict) or not combat_state:
            return None

        ort_dir = output_dir / "ort_rollout"
        ort_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = ort_dir / "actor_combat.onnx"
        vocab_path = ort_dir / "vocab_mapping.json"
        vocab = load_vocab()
        export_from_training_snapshot(ppo_state if isinstance(ppo_state, dict) else {}, combat_state, vocab, str(onnx_path), policy_version=0)
        vocab_path.write_text(
            json.dumps(
                {
                    "card_to_idx": vocab.card_to_idx,
                    "monster_to_idx": vocab.monster_to_idx,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(onnx_path.resolve())
    except Exception as exc:
        print(f"WARNING: Failed to prepare local ORT rollout model: {exc}")
        return None


def _build_client(port: int, transport: str) -> FullRunClientLike:
    use_pipe = str(transport).strip().lower() in {"pipe", "pipe-binary"}
    base_url = f"http://127.0.0.1:{int(port)}"
    client = create_full_run_client(
        base_url=base_url,
        port=port,
        use_pipe=use_pipe,
        transport=transport,
        ready_timeout_s=15.0,
    )
    if hasattr(client, "_ensure_connected"):
        client._ensure_connected()
    return client


def _generate_episode_batch(
    *,
    port: int,
    transport: str,
    seeds: list[str],
    checkpoint_path: str | None,
    combat_checkpoint_path: str | None,
    debug_rollout_trace_dir: str | None = None,
    progress_callback: Any | None = None,
    rollout_max_combats: int = 3,
    rollout_max_steps: int = 500,
    rerun_low_spread_threshold: float = 0.0,
    rerun_max_combats: int = 5,
    rerun_max_steps: int = 900,
    rollout_goal: str = "terminal",
    label_mode: str = "single_step",
    reward_tree_config: RewardTreeConfig | None = None,
    map_route_config: MapRouteConfig | None = None,
    use_local_ort_rollout: bool = False,
    local_ort_model_path: str | None = None,
    local_ort_max_combat_steps: int = 600,
    sample_types: set[str] | None = None,
) -> tuple[list[CardRankingSample], list[dict[str, Any]], list[dict[str, Any]]]:
    generator_config = {
        "transport": str(transport),
        "port": int(port),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "combat_checkpoint": combat_checkpoint_path,
        "combat_checkpoint_sha256": _sha256_file(combat_checkpoint_path),
        "rollout_max_combats": int(rollout_max_combats),
        "rollout_max_steps": int(rollout_max_steps),
        "rerun_low_spread_threshold": float(rerun_low_spread_threshold),
        "rerun_max_combats": int(rerun_max_combats),
        "rerun_max_steps": int(rerun_max_steps),
        "rollout_goal": str(rollout_goal),
        "label_mode": str(label_mode),
        "map_max_depth": int(map_route_config.max_map_depth) if map_route_config is not None else 0,
        "map_beam_width": int(map_route_config.beam_width) if map_route_config is not None else 0,
        "map_advance_max_steps": int(map_route_config.advance_max_steps) if map_route_config is not None else 0,
        "map_max_option_seconds": (
            float(map_route_config.max_option_seconds)
            if map_route_config is not None and map_route_config.max_option_seconds is not None else None
        ),
        "episode_stop_floor": int(map_route_config.stop_floor) if map_route_config is not None and map_route_config.stop_floor is not None else None,
        "tree_max_reward_depth": int(reward_tree_config.max_reward_depth) if reward_tree_config is not None else 0,
        "tree_beam_width": int(reward_tree_config.beam_width) if reward_tree_config is not None else 0,
        "tree_route_search": bool(reward_tree_config.enable_route_search) if reward_tree_config is not None else False,
        "tree_advance_max_steps": int(reward_tree_config.advance_max_steps) if reward_tree_config is not None else 0,
        "tree_local_weight": float(reward_tree_config.blend_local_weight) if reward_tree_config is not None else None,
        "tree_max_option_seconds": (
            float(reward_tree_config.max_option_seconds)
            if reward_tree_config is not None and reward_tree_config.max_option_seconds is not None else None
        ),
        "tree_recurse_only_when_spread_below": (
            float(reward_tree_config.recurse_only_when_spread_below)
            if reward_tree_config is not None and reward_tree_config.recurse_only_when_spread_below is not None else None
        ),
        "local_ort_rollout": bool(use_local_ort_rollout),
        "local_ort_model_path": local_ort_model_path,
        "local_ort_max_combat_steps": int(local_ort_max_combat_steps),
        "sample_types": sorted(str(sample_type).strip().lower() for sample_type in (sample_types or {"map", "card_reward"})),
    }
    client = _build_client(port, transport)
    combat_evaluator = _load_combat_evaluator(checkpoint_path, combat_checkpoint_path)
    ppo_policy = _load_ppo_rollout_policy(checkpoint_path)
    if use_local_ort_rollout and local_ort_model_path and client.supports_local_ort:
        loaded = client.load_ort_model(local_ort_model_path)
        if loaded:
            print(f"Local ORT rollout loaded on port {port}: {local_ort_model_path}")
    samples: list[CardRankingSample] = []
    raw_branch_records: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, seed in enumerate(seeds, start=1):
        try:
            episode_samples, summary, episode_raw_branch_records = generate_from_episode(
                client,
                seed,
                port=port,
                transport=transport,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=generator_config["checkpoint_sha256"],
                combat_checkpoint_path=combat_checkpoint_path,
                combat_checkpoint_sha256=generator_config["combat_checkpoint_sha256"],
                generator_config=generator_config,
                combat_evaluator=combat_evaluator,
                ppo_policy=ppo_policy,
                debug_rollout_trace_dir=debug_rollout_trace_dir,
                rollout_max_combats=rollout_max_combats,
                rollout_max_steps=rollout_max_steps,
                rerun_low_spread_threshold=rerun_low_spread_threshold,
                rerun_max_combats=rerun_max_combats,
                rerun_max_steps=rerun_max_steps,
                rollout_goal=rollout_goal,
                label_mode=label_mode,
                reward_tree_config=reward_tree_config,
                map_route_config=map_route_config,
                use_local_ort_rollout=use_local_ort_rollout,
                local_ort_max_combat_steps=local_ort_max_combat_steps,
                stop_floor=(map_route_config.stop_floor if map_route_config is not None else None),
                sample_types=sample_types,
            )
            log_entry = {
                "episode_index": idx,
                "seed": seed,
                "port": port,
                "status": summary.status,
                "samples": len(episode_samples),
                "sample_count": summary.sample_count,
                "end_floor": summary.end_floor,
                "boss_reached": summary.boss_reached,
                "outcome": summary.outcome,
                "map_decisions_seen": summary.map_decisions_seen,
                "map_samples_recorded": summary.map_samples_recorded,
                "card_reward_decisions_seen": summary.card_reward_decisions_seen,
                "card_reward_samples_recorded": summary.card_reward_samples_recorded,
                "sampling_skip_reasons": dict(summary.sampling_skip_reasons),
                "elapsed_s": round(time.time() - t0, 2),
            }
            samples.extend(episode_samples)
            raw_branch_records.extend(episode_raw_branch_records)
            logs.append(log_entry)
            if progress_callback is not None:
                progress_callback(episode_samples, log_entry, episode_raw_branch_records)
        except Exception as exc:
            log_entry = {
                "episode_index": idx,
                "seed": seed,
                "port": port,
                "status": f"error:{exc}",
                "samples": 0,
                "sample_count": 0,
                "end_floor": 0,
                "boss_reached": False,
                "outcome": "error",
                "elapsed_s": round(time.time() - t0, 2),
            }
            logs.append(log_entry)
            if progress_callback is not None:
                progress_callback([], log_entry, [])
    return samples, logs, raw_branch_records


def main() -> None:
    config_path = _peek_config_path(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Generate offline non-combat ranking dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Optional TOML config file. Leaf keys should match argparse dest names.",
    )
    parser.add_argument("--pipe", action="store_true")
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--start-port", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--auto-launch",
        action="store_true",
        default=False,
        help="Auto-launch local headless simulator hosts for pipe/pipe-binary transports.",
    )
    parser.add_argument(
        "--headless-dll",
        type=str,
        default=str(DEFAULT_DLL_PATH),
        help="HeadlessSim host binary path used when --auto-launch is enabled.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default=str(ARTIFACTS_ROOT / "offline_noncombat_ranking_v1"))
    parser.add_argument("--seed-prefix", type=str, default="RANK")
    parser.add_argument("--seed-file", type=str, default="",
                        help="Optional JSON/TXT seed list. If set, explicit seeds replace generated seed_prefix episodes.")
    parser.add_argument("--seed", action="append", default=[],
                        help="Explicit seed(s). Can be repeated. Combined with --seed-file and deduplicated.")
    parser.add_argument("--num-seeds", type=int, default=0,
                        help="Optional cap when using --seed/--seed-file. <=0 keeps all explicit seeds.")
    parser.add_argument(
        "--sample-types",
        type=str,
        default="map,card_reward",
        help="Comma-separated sampled decision types. Supported: map,card_reward (default: map,card_reward)",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Hybrid checkpoint for NN combat policy (improves data quality)")
    parser.add_argument("--combat-checkpoint", type=str, default=None,
                        help="Optional standalone combat checkpoint to override combat weights")
    parser.add_argument(
        "--transport",
        choices=["http", "pipe", "pipe-binary"],
        default="pipe-binary",
        help="Simulator transport for dataset generation (default: pipe-binary)",
    )
    parser.add_argument(
        "--manifest-out",
        type=str,
        default="",
        help="Optional manifest JSON path. Defaults to <output>/manifest.json",
    )
    parser.add_argument(
        "--debug-rollout-trace-dir",
        type=str,
        default="",
        help="Optional directory for per-option rollout step traces (JSONL).",
    )
    parser.add_argument("--rollout-max-combats", type=int, default=3,
                        help="Primary branch rollout horizon in combats (default: 3)")
    parser.add_argument("--rollout-max-steps", type=int, default=500,
                        help="Primary branch rollout step cap (default: 500)")
    parser.add_argument(
        "--rollout-goal",
        choices=["combat_horizon", "boss_reached_or_terminal", "terminal"],
        default="terminal",
        help=(
            "Card reward branch rollout stopping rule. "
            "'terminal' runs to game_over/victory, "
            "'boss_reached_or_terminal' stops once boss is reached, "
            "'combat_horizon' preserves the old fixed-combat horizon behavior."
        ),
    )
    parser.add_argument("--rerun-low-spread-threshold", type=float, default=0.0,
                        help="If primary rollout spread is below this threshold, rerun card_reward branches with a longer horizon (default: off)")
    parser.add_argument("--rerun-max-combats", type=int, default=5,
                        help="Long-horizon rerun combats for low-spread card_reward samples (default: 5)")
    parser.add_argument("--rerun-max-steps", type=int, default=900,
                        help="Long-horizon rerun step cap for low-spread card_reward samples (default: 900)")
    parser.add_argument("--label-mode", choices=["single_step", "reward_tree"], default="single_step",
                        help="Card reward label mode: local single-step rollout or recursive reward-tree continuation")
    parser.add_argument("--tree-max-reward-depth", type=int, default=3,
                        help="Max recursive card_reward depth for reward_tree mode (default: 3)")
    parser.add_argument("--tree-beam-width", type=int, default=2,
                        help="Beam width for reward_tree mode (default: 2)")
    parser.add_argument(
        "--tree-route-search",
        action="store_true",
        default=False,
        help=(
            "Enable short-horizon card_reward route search. When combined with "
            "--label-mode reward_tree and terminal rollout, recursively plans "
            "across future card_reward screens instead of greedily scoring only the current one."
        ),
    )
    parser.add_argument("--tree-advance-max-steps", type=int, default=900,
                        help="Step cap while advancing to the next card_reward in reward_tree mode (default: 900)")
    parser.add_argument("--tree-local-weight", type=float, default=0.6,
                        help="Blend weight for local rollout score vs downstream reward-tree continuation (default: 0.6)")
    parser.add_argument("--tree-max-option-seconds", type=float, default=4.0,
                        help="Per-root-option wall-clock budget for reward_tree recursion; <=0 disables time budgeting (default: 4.0)")
    parser.add_argument("--tree-recurse-only-when-spread-below", type=float, default=0.25,
                        help="Skip deeper reward_tree recursion when current-layer local score spread is already this large or larger; <=0 disables the gate (default: 0.25)")
    parser.add_argument("--map-max-depth", type=int, default=5,
                        help="Max recursive map depth for map_route_tree labels (default: 5)")
    parser.add_argument("--map-beam-width", type=int, default=2,
                        help="Beam width for internal map route search (default: 2)")
    parser.add_argument("--map-advance-max-steps", type=int, default=1600,
                        help="Step cap while advancing map route continuations (default: 1600)")
    parser.add_argument("--map-max-option-seconds", type=float, default=3.0,
                        help="Per-root-option wall-clock budget for map_route_tree recursion; <=0 disables time budgeting (default: 3.0)")
    parser.add_argument("--episode-stop-floor", type=int, default=0,
                        help="If > 0, stop the episode and internal search continuations once this floor is reached (default: off)")
    parser.add_argument("--flush-every-episodes", type=int, default=0,
                        help="If > 0, periodically materialize a trainable partial dataset every N completed episodes")
    parser.add_argument("--local-ort-rollout", action="store_true", default=False,
                        help="Run combat rollout continuation in C# via local ORT actor when using pipe-binary transport")
    parser.add_argument("--local-ort-max-combat-steps", type=int, default=600,
                        help="Step cap passed to C# run_combat_local during local ORT rollout (default: 600)")
    loaded_config = _load_generator_config(config_path)
    if loaded_config:
        parser.set_defaults(**loaded_config)
    args = parser.parse_args()

    if args.local_ort_rollout and str(args.transport).strip().lower() != "pipe-binary":
        print("WARNING: --local-ort-rollout requires pipe-binary; disabling local ORT rollout")
        args.local_ort_rollout = False

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"

    explicit_seeds = [str(seed).strip() for seed in (args.seed or []) if str(seed).strip()]
    if args.seed_file:
        explicit_seeds.extend(_load_seed_list(args.seed_file, limit=max(0, int(args.num_seeds))))
    seeds: list[str]
    if explicit_seeds:
        ordered: list[str] = []
        seen: set[str] = set()
        for seed in explicit_seeds:
            if seed and seed not in seen:
                seen.add(seed)
                ordered.append(seed)
        if int(args.num_seeds) > 0:
            ordered = ordered[: int(args.num_seeds)]
        seeds = ordered
    else:
        seeds = [f"{args.seed_prefix}_{ep:05d}" for ep in range(args.episodes)]
    if not seeds:
        raise ValueError("No seeds resolved. Provide --episodes > 0 or pass --seed/--seed-file.")
    args.episodes = len(seeds)

    num_envs = max(1, int(args.num_envs))
    base_port = args.start_port or args.port
    progress_every_episodes = 16
    flush_every_episodes = max(0, int(args.flush_every_episodes))
    sample_types = {
        token.strip().lower()
        for token in str(args.sample_types or "").split(",")
        if token.strip()
    } or {"map", "card_reward"}
    unsupported_sample_types = sorted(sample_types - {"map", "card_reward"})
    if unsupported_sample_types:
        raise ValueError(
            f"Unsupported --sample-types values: {', '.join(unsupported_sample_types)}. "
            "Supported values are: map, card_reward."
        )
    reward_tree_config = RewardTreeConfig(
        max_reward_depth=int(args.tree_max_reward_depth),
        beam_width=int(args.tree_beam_width),
        rollout_goal=str(args.rollout_goal),
        enable_route_search=bool(args.tree_route_search),
        rollout_max_combats=int(args.rollout_max_combats),
        rollout_max_steps=int(args.rollout_max_steps),
        rerun_low_spread_threshold=float(args.rerun_low_spread_threshold),
        rerun_max_combats=int(args.rerun_max_combats),
        rerun_max_steps=int(args.rerun_max_steps),
        advance_max_steps=int(args.tree_advance_max_steps),
        blend_local_weight=float(args.tree_local_weight),
        use_local_ort_rollout=bool(args.local_ort_rollout),
        local_ort_max_combat_steps=int(args.local_ort_max_combat_steps),
        stop_floor=(int(args.episode_stop_floor) if int(args.episode_stop_floor) > 0 else None),
        max_option_seconds=(float(args.tree_max_option_seconds) if float(args.tree_max_option_seconds) > 0 else None),
        recurse_only_when_spread_below=(
            float(args.tree_recurse_only_when_spread_below)
            if float(args.tree_recurse_only_when_spread_below) > 0 else None
        ),
    )
    map_route_config = MapRouteConfig(
        max_map_depth=int(args.map_max_depth),
        beam_width=int(args.map_beam_width),
        advance_max_steps=int(args.map_advance_max_steps),
        stop_floor=(int(args.episode_stop_floor) if int(args.episode_stop_floor) > 0 else None),
        max_option_seconds=(float(args.map_max_option_seconds) if float(args.map_max_option_seconds) > 0 else None),
    )
    local_ort_model_path = None
    if args.local_ort_rollout:
        local_ort_model_path = _prepare_local_ort_rollout_model(
            checkpoint_path=args.checkpoint,
            combat_checkpoint_path=args.combat_checkpoint,
            output_dir=out_dir,
        )
        if not local_ort_model_path:
            print("WARNING: Local ORT rollout requested but ONNX export failed; falling back to Python combat inference")

    if not args.checkpoint and not args.combat_checkpoint:
        print("WARNING: No combat checkpoint provided; using heuristic combat policy (lower data quality)")
        print("WARNING: No checkpoint — using heuristic combat policy (lower data quality)")

    all_samples: list[CardRankingSample] = []
    all_raw_branch_records: list[dict[str, Any]] = []
    episode_logs: list[dict[str, Any]] = []
    t0 = time.time()
    progress_lock = threading.Lock()
    seed_batches: list[list[str]] = [[] for _ in range(num_envs)]
    for idx, seed in enumerate(seeds):
        seed_batches[idx % num_envs].append(seed)

    def _on_episode_complete(
        episode_samples: list[CardRankingSample],
        log_entry: dict[str, Any],
        episode_raw_branch_records: list[dict[str, Any]],
    ) -> None:
        with progress_lock:
            all_samples.extend(episode_samples)
            all_raw_branch_records.extend(episode_raw_branch_records)
            episode_logs.append(log_entry)
            completed_episodes = len(episode_logs)
            if (
                completed_episodes == 1
                or completed_episodes % progress_every_episodes == 0
                or completed_episodes >= args.episodes
            ):
                _write_progress_snapshot(
                    out_dir=out_dir,
                    total_episodes=args.episodes,
                    completed_episodes=completed_episodes,
                    all_samples=all_samples,
                    episode_logs=episode_logs,
                    num_envs=num_envs,
                    transport=args.transport,
                    t0=t0,
                )
            if (
                flush_every_episodes > 0
                and (
                    completed_episodes % flush_every_episodes == 0
                    or completed_episodes >= args.episodes
                )
            ):
                _materialize_dataset_snapshot(
                    out_dir=out_dir,
                    all_samples=all_samples,
                    raw_branch_records=all_raw_branch_records,
                    episode_logs=episode_logs,
                    episodes=completed_episodes,
                    total_episodes=args.episodes,
                    num_envs=num_envs,
                    transport=args.transport,
                    t0=t0,
                    args=args,
                    partial=(completed_episodes < args.episodes),
                )

    host_ports = [base_port + env_idx for env_idx, batch in enumerate(seed_batches) if batch]
    with SimHostLifecycleManager(
        ports=host_ports,
        transport=args.transport,
        auto_launch=bool(args.auto_launch),
        repo_root=Path.cwd(),
        dll_path=args.headless_dll,
    ):
        with ThreadPoolExecutor(max_workers=num_envs) as pool:
            futures = [
                pool.submit(
                    _generate_episode_batch,
                    port=base_port + env_idx,
                    transport=args.transport,
                    seeds=batch,
                    checkpoint_path=args.checkpoint,
                    combat_checkpoint_path=args.combat_checkpoint,
                    debug_rollout_trace_dir=(args.debug_rollout_trace_dir or None),
                    progress_callback=_on_episode_complete,
                    rollout_max_combats=int(args.rollout_max_combats),
                    rollout_max_steps=int(args.rollout_max_steps),
                    rerun_low_spread_threshold=float(args.rerun_low_spread_threshold),
                    rerun_max_combats=int(args.rerun_max_combats),
                    rerun_max_steps=int(args.rerun_max_steps),
                    rollout_goal=str(args.rollout_goal),
                    label_mode=str(args.label_mode),
                    reward_tree_config=reward_tree_config,
                    map_route_config=map_route_config,
                    use_local_ort_rollout=bool(local_ort_model_path),
                    local_ort_model_path=local_ort_model_path,
                    local_ort_max_combat_steps=int(args.local_ort_max_combat_steps),
                    sample_types=sample_types,
                )
                for env_idx, batch in enumerate(seed_batches)
                if batch
            ]
            for future in as_completed(futures):
                _, batch_logs, _ = future.result()
                completed = len(episode_logs)
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                last = batch_logs[-1] if batch_logs else {}
                print(
                    f"[{completed}/{args.episodes}] port={last.get('port')} "
                    f"seed={last.get('seed')} status={last.get('status')} "
                    f"samples={last.get('samples', 0)} total={len(all_samples)} "
                    f"({rate:.2f} ep/s)"
                )

    jsonl_path, stats_path, episode_logs_path, manifest_path = _materialize_dataset_snapshot(
        out_dir=out_dir,
        all_samples=all_samples,
        raw_branch_records=all_raw_branch_records,
        episode_logs=episode_logs,
        episodes=args.episodes,
        total_episodes=args.episodes,
        num_envs=num_envs,
        transport=args.transport,
        t0=t0,
        args=args,
        partial=False,
    )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    progress_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "updated_at_utc": _utc_now(),
                "completed_episodes": int(args.episodes),
                "total_episodes": int(args.episodes),
                "progress_fraction": 1.0,
                "summary": stats,
                "last_episode": episode_logs[-1] if episode_logs else None,
                "manifest_path": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nDone. {len(all_samples)} samples -> {jsonl_path}")
    print(f"Stats: {json.dumps(stats, indent=2)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
