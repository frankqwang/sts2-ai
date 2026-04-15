#!/usr/bin/env python3
"""Unified PPO + MCTS hybrid training loop.

Trains both brains simultaneously in shared episodes:
- Non-combat screens → PPO inference + data collection
- Combat screens → MCTS search + data collection

Usage:
    # Single env (testing):
    python train_hybrid.py --pipe --num-envs 1 --start-port 15527 --max-iterations 5

    # 8 parallel envs (production):
    python train_hybrid.py --pipe --num-envs 8 --start-port 15527 --max-iterations 500

    # Resume from checkpoints:
    python train_hybrid.py --pipe --num-envs 8 --resume-ppo ppo_best.pt --resume-mcts mcts_best.pt

    # Or use the launch script:
    .\\scripts\\start-mcts-training.ps1  (update to call train_hybrid.py)
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds STS2AI/Python library dirs to sys.path)

import argparse
import atexit
import json
import logging
import gc
import random
import re
import signal
import sqlite3
import sys
import threading
import time
import tomllib
import traceback
from collections import deque
from contextlib import nullcontext
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.vocab import load_vocab, Vocab
from core.rl_encoder_v2 import build_structured_state, build_structured_actions
from core.rl_policy_v2 import (
    FullRunPolicyNetworkV2,
    PPOTrainerV2,
    StructuredRolloutBuffer,
    _structured_state_to_numpy_dict,
    _structured_actions_to_numpy_dict,
)
from core.rl_reward_shaping import (
    extract_next_boss_token,
    boss_readiness_score,
    shaped_reward,
    combat_step_reward,
    combat_local_tactical_reward,
    compute_combat_feedback,
    screen_local_delta_reward,
    _extract_player,
    _safe_int,
)
from combat_safety import rerank_combat_logits_with_safety
from rl_segment_buffer import SegmentRolloutBuffer, Segment
from segment_collector import NonCombatSegmentCollector
from search.counterfactual_scoring import compute_counterfactual_reward
from core.combat_nn import (
    CombatPolicyValueNetwork,
    build_combat_features,
    build_combat_action_features,
    MAX_ACTIONS,
)
from search.mcts_core import MCTSConfig, action_key, mcts_search
from search.combat_mcts_agent import CombatMCTSAgent, PipeCombatForwardModel
from full_run_env import ApiBackedFullRunClient, PipeBackedFullRunClient, create_full_run_client
from headless_sim_runner import DEFAULT_DLL_PATH, stop_process
from ipc.sim_host_lifecycle import SimHostLifecycleManager, transport_launch_protocol
from sts2ai_paths import ARTIFACTS_ROOT, DATASETS_ROOT, MAINLINE_CHECKPOINT, REPO_ROOT
from training_health import TrainingHealthMonitor
from episode_data_saver import EpisodeDataSaver
from runtime.run_outcome_vocab import (
    RUN_OUTCOME_DEATH,
    RUN_OUTCOME_VICTORY,
    normalize_run_outcome,
)
from runtime.full_run_action_semantics import (
    legal_action_name_set as _shared_legal_action_name_set,
    is_selection_screen as _shared_is_selection_screen,
    choose_auto_progress_action as _shared_choose_auto_progress_action,
    combat_rewards_state as _shared_combat_rewards_state,
    reward_item_claimable as _shared_reward_item_claimable,
    choose_claimable_reward_action as _shared_choose_claimable_reward_action,
    claim_reward_action_count as _shared_claim_reward_action_count,
    reward_claim_signature as _shared_reward_claim_signature,
    next_reward_claim_signature as _shared_next_reward_claim_signature,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

COMBAT_PENDING_STATE_TYPES = {
    "combat_pending",
    "combat_start_pending",
    "combat_post_end_pending",
}

_shutdown_requested = False
_TRACE_NAME_MAPS: dict[str, dict[str, str]] | None = None
_RUNTIME_SKADA_PRIORS: Any | bool | None = None
DEFAULT_REPO_ROOT = REPO_ROOT
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "hybrid_training"
DEFAULT_OFFLINE_NONCOMBAT_RANKING_DATA_DIR = DATASETS_ROOT / "offline_noncombat_ranking_post_wizardly"
DEFAULT_MATCHUP_DATA_DIR = DEFAULT_OFFLINE_NONCOMBAT_RANKING_DATA_DIR
DEFAULT_COMBAT_TEACHER_DATA = DATASETS_ROOT / "combat_teacher_post_wizardly" / "teacher.jsonl"

# Backward-compatible config aliases for older opaque names. The right-hand side
# is the argparse dest / canonical internal name actually consumed by training.
_CONFIG_ALIASES = {
    "offline_noncombat_ranking_data_dir": "matchup_data_dir",
    "offline_noncombat_ranking_batch_size": "matchup_batch_size",
    "offline_noncombat_ranking_loss_weight": "matchup_loss_weight",
    "offline_noncombat_ranking_updates_per_iter": "matchup_updates_per_iter",
    "offline_noncombat_ranking_warmup_iters": "matchup_warmup_iters",
    "offline_noncombat_ranking_loss_decay_tau": "matchup_loss_decay_tau",
    "offline_noncombat_ranking_blend_beta": "matchup_blend_beta",
    "offline_noncombat_ranking_min_spread": "matchup_min_spread",
    "saved_offline_episodes_enabled": "save_offline_data",
    "saved_offline_episodes_min_floor": "offline_min_floor",
    "saved_offline_replay_traces_enabled": "save_replay_traces",
    "saved_offline_metrics_log_enabled": "save_metrics_log",
    "offline_combat_teacher_data_dir": "combat_teacher_data_dir",
    "offline_combat_teacher_loss_weight": "combat_teacher_loss_weight",
    "offline_combat_teacher_batch_size": "combat_teacher_batch_size",
    "offline_combat_teacher_updates_per_iter": "combat_teacher_updates_per_iter",
    "offline_combat_teacher_warmup_iters": "combat_teacher_warmup_iters",
}


def _load_trace_name_maps() -> dict[str, dict[str, str]]:
    global _TRACE_NAME_MAPS
    if _TRACE_NAME_MAPS is not None:
        return _TRACE_NAME_MAPS
    maps: dict[str, dict[str, str]] = {
        "cards": {},
        "encounters": {},
        "relics": {},
        "campfire": {},
    }
    db_path = Path(__file__).resolve().parents[1] / "Assets" / "datasets" / "skada" / "skada_analytics.sqlite"
    if not db_path.exists():
        _TRACE_NAME_MAPS = maps
        return maps
    try:
        conn = sqlite3.connect(db_path)
        try:
            for card_id, name_zh in conn.execute("SELECT card_id, name_zh FROM cards"):
                key = str(card_id or "").strip().upper()
                val = str(name_zh or "").strip()
                if key and val:
                    maps["cards"][key] = val
            for encounter, name_zh in conn.execute("SELECT encounter, name_zh FROM encounters"):
                key = str(encounter or "").strip().upper()
                val = str(name_zh or "").strip()
                if key and val:
                    maps["encounters"][key] = val
            for relic_id, name_zh in conn.execute("SELECT relic_id, name_zh FROM relics"):
                key = str(relic_id or "").strip().upper()
                val = str(name_zh or "").strip()
                if key and val:
                    maps["relics"][key] = val
            for action, name_zh in conn.execute("SELECT action, name_zh FROM campfire_decisions"):
                key = str(action or "").strip().upper()
                val = str(name_zh or "").strip()
                if key and val:
                    maps["campfire"][key] = val
        finally:
            conn.close()
    except Exception:
        pass
    _TRACE_NAME_MAPS = maps
    return maps


def _trace_pretty_token(token: Any) -> str:
    text = str(token or "").strip()
    if not text:
        return "未知"
    if all(ch.isupper() or ch.isdigit() or ch == "_" for ch in text):
        return " ".join(part.capitalize() for part in text.split("_") if part)
    return text


def _trace_resolve_name(token: Any, *, category: str = "generic") -> str:
    text = str(token or "").strip()
    if not text:
        return "未知"
    key = text.upper()
    maps = _load_trace_name_maps()
    if category == "card":
        return maps["cards"].get(key) or _trace_pretty_token(text)
    if category == "encounter":
        return maps["encounters"].get(key) or _trace_pretty_token(text)
    if category == "relic":
        return maps["relics"].get(key) or _trace_pretty_token(text)
    if category == "campfire":
        return maps["campfire"].get(key) or _trace_pretty_token(text)
    generic_known = {
        "monster": "普通战斗",
        "unknown": "事件",
        "rest_site": "火堆",
        "treasure": "宝箱",
        "shop": "商店",
        "elite": "精英",
        "boss": "Boss",
        "skip_card_reward": "跳过卡奖",
        "rest": "休息",
        "smith": "锻造",
        "proceed": "离开/继续",
        "remove_card": "删牌",
        "play_card": "出牌",
        "use_potion": "使用药水",
        "end_turn": "结束回合",
    }
    return (
        maps["cards"].get(key)
        or maps["encounters"].get(key)
        or maps["relics"].get(key)
        or generic_known.get(text.lower())
        or _trace_pretty_token(text)
    )


def _lower_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_combat_pending_state(state_type: Any) -> bool:
    return _lower_text(state_type) in COMBAT_PENDING_STATE_TYPES


def _optional_lock(lock: threading.Lock | None):
    return lock if lock is not None else nullcontext()


def _sanitize_run_tag(tag: Any) -> str:
    text = str(tag or "").strip().lower()
    if not text:
        return ""
    text = text.replace("\\", "_").replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._-")


def _build_run_output_dir_name(*, num_envs: int, timestamp: str, run_tag: str | None) -> str:
    parts = [timestamp, f"{num_envs}env"]
    sanitized_tag = _sanitize_run_tag(run_tag)
    if sanitized_tag:
        parts.append(sanitized_tag)
    return "_".join(parts)


def _advance_combat_pending_transition(
    client: Any,
    current_streak: int,
    *,
    poll_sleep_s: float = 0.02,
    wait_every: int = 5,
) -> tuple[dict[str, Any], str]:
    """Advance async combat transitions without spamming step(wait).

    Most combat_pending frames are passive state transitions where a state poll is
    enough to observe the next actionable screen. For pipe backends we still send
    an occasional wait as a kick in case the host expects a step to flush the
    transition.
    """

    streak = max(1, int(current_streak))
    should_wait = wait_every > 0 and streak % wait_every == 0
    if not should_wait:
        getter = getattr(client, "get_state", None)
        if callable(getter):
            if poll_sleep_s > 0:
                time.sleep(poll_sleep_s)
            return getter(), "refresh"
    return client.act({"action": "wait"}), "wait"


def _snapshot_state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _mp_worker_create_ort_session(
    model_path: str | None,
    *,
    worker_id: int,
    label: str,
    log: Any,
):
    model_path = str(model_path or "").strip()
    if not model_path:
        return None
    try:
        import onnxruntime as ort

        ort_opts = ort.SessionOptions()
        ort_opts.intra_op_num_threads = 1
        ort_opts.inter_op_num_threads = 1
        ort_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(
            model_path,
            ort_opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as ort_err:
        log.warning(
            "Worker %d %s ORT session bootstrap failed, falling back to PyTorch CPU: %s",
            worker_id,
            label,
            ort_err,
        )
        return None


def _apply_mp_worker_refresh_message(
    *,
    worker_id: int,
    log: Any,
    worker_config: dict[str, Any],
    ppo_net: FullRunPolicyNetworkV2,
    mcts_net: CombatPolicyValueNetwork,
    refresh_task: dict[str, Any],
    ppo_ort_session: Any,
    combat_ort_session: Any,
) -> tuple[Any, Any]:
    worker_config_updates = refresh_task.get("worker_config_updates")
    if isinstance(worker_config_updates, dict):
        worker_config.update(worker_config_updates)

    ppo_state_dict = refresh_task.get("ppo_state_dict")
    if isinstance(ppo_state_dict, dict):
        ppo_net.load_state_dict(ppo_state_dict, strict=False)

    mcts_state_dict = refresh_task.get("mcts_state_dict")
    if isinstance(mcts_state_dict, dict):
        mcts_net.load_state_dict(mcts_state_dict, strict=False)

    ppo_net.eval()
    mcts_net.eval()

    ppo_onnx_path = str(
        refresh_task.get("ppo_onnx_path")
        or worker_config.get("ppo_onnx_path", "")
        or ""
    ).strip()
    combat_onnx_path = str(
        refresh_task.get("combat_onnx_path")
        or worker_config.get("combat_onnx_path", "")
        or ""
    ).strip()
    worker_config["ppo_onnx_path"] = ppo_onnx_path
    worker_config["combat_onnx_path"] = combat_onnx_path

    if bool(refresh_task.get("disable_ort", False)):
        return None, None

    if bool(refresh_task.get("reload_ort", False)):
        ppo_ort_session = _mp_worker_create_ort_session(
            ppo_onnx_path,
            worker_id=worker_id,
            label="PPO",
            log=log,
        )
        combat_ort_session = _mp_worker_create_ort_session(
            combat_onnx_path,
            worker_id=worker_id,
            label="combat",
            log=log,
        )

    return ppo_ort_session, combat_ort_session


def _extract_terminal_outcome(state: dict[str, Any]) -> str:
    go = state.get("game_over") if isinstance(state.get("game_over"), dict) else {}
    for value in (
        go.get("run_outcome"),
        go.get("outcome"),
        state.get("run_outcome"),
        state.get("outcome"),
    ):
        text = normalize_run_outcome(value, default="")
        if text:
            return text
    return ""


def _terminal_value_from_outcome(state: dict[str, Any], *, default_on_unknown: float = -1.0) -> float:
    outcome_text = _extract_terminal_outcome(state)
    if outcome_text == RUN_OUTCOME_VICTORY:
        return 1.0
    if outcome_text:
        return -1.0
    return float(default_on_unknown)


def _is_episode_terminal_state(
    state: dict[str, Any],
    *,
    has_entered_run: bool,
) -> bool:
    st = _lower_text(state.get("state_type"))
    if st == "game_over" or bool(state.get("terminal")):
        return True
    return st == "menu" and has_entered_run


def _is_act1_map_state(state: dict[str, Any]) -> bool:
    if _lower_text(state.get("state_type")) != "map":
        return False
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    return _safe_int(run.get("act", 0), 0) == 1


def _build_map_node_lookup(map_state: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for node in map_state.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        try:
            lookup[(int(node.get("col", 0)), int(node.get("row", 0)))] = node
        except Exception:
            continue
    return lookup


def _resolve_map_option_coord(
    action: dict[str, Any],
    next_options: list[dict[str, Any]],
) -> tuple[int, int] | None:
    try:
        if action.get("col") is not None and action.get("row") is not None:
            return int(action.get("col", 0)), int(action.get("row", 0))
    except Exception:
        pass

    action_idx = _safe_int(action.get("index", -1), -1)
    if 0 <= action_idx < len(next_options):
        option = next_options[action_idx]
        try:
            return int(option.get("col", 0)), int(option.get("row", 0))
        except Exception:
            return None
    return None


def _map_option_has_elite_free_path(
    map_state: dict[str, Any],
    action: dict[str, Any],
) -> bool:
    next_options = map_state.get("next_options") or []
    node_lookup = _build_map_node_lookup(map_state)
    start_coord = _resolve_map_option_coord(action, next_options)
    if start_coord is None:
        return False

    seen: set[tuple[int, int]] = set()

    def _dfs(coord: tuple[int, int]) -> bool:
        if coord in seen:
            return False
        seen.add(coord)
        node = node_lookup.get(coord)
        if node is None:
            return True
        if "elite" in _lower_text(node.get("type")):
            return False
        children = node.get("children") or []
        valid_children: list[tuple[int, int]] = []
        for child in children:
            if not isinstance(child, (list, tuple)) or len(child) != 2:
                continue
            try:
                valid_children.append((int(child[0]), int(child[1])))
            except Exception:
                continue
        if not valid_children:
            return True
        return any(_dfs(child_coord) for child_coord in valid_children)

    return _dfs(start_coord)


def _map_option_route_stats(
    state: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any] | None:
    map_state = state.get("map") if isinstance(state.get("map"), dict) else {}
    next_options = map_state.get("next_options") or []
    node_lookup = _build_map_node_lookup(map_state)
    start_coord = _resolve_map_option_coord(action, next_options)
    if start_coord is None:
        return None

    boss = map_state.get("boss") if isinstance(map_state.get("boss"), dict) else {}
    boss_row = int(boss.get("row", 0) or 0)
    start_node = node_lookup.get(start_coord, {})
    option_type = _lower_text(start_node.get("type") or action.get("label") or action.get("action"))

    child_coords: list[tuple[int, int]] = []
    for child in start_node.get("children") or []:
        if not isinstance(child, (list, tuple)) or len(child) != 2:
            continue
        try:
            child_coords.append((int(child[0]), int(child[1])))
        except Exception:
            continue
    child_types = [_lower_text((node_lookup.get(coord) or {}).get("type")) for coord in child_coords]
    non_empty_child_types = [child_type for child_type in child_types if child_type]

    stack: list[tuple[tuple[int, int], dict[str, int]]] = [
        (start_coord, {"elite": 0, "shop": 0, "restsite": 0, "monster": 0})
    ]
    path_stats: list[dict[str, int]] = []
    max_paths = 64
    while stack and len(path_stats) < max_paths:
        coord, counts = stack.pop()
        node = node_lookup.get(coord)
        if node is None:
            path_stats.append(dict(counts))
            continue
        ntype = _lower_text(node.get("type"))
        new_counts = dict(counts)
        if ntype in new_counts:
            new_counts[ntype] += 1
        children = node.get("children") or []
        valid_children: list[tuple[int, int]] = []
        for child in children:
            if not isinstance(child, (list, tuple)) or len(child) != 2:
                continue
            try:
                valid_children.append((int(child[0]), int(child[1])))
            except Exception:
                continue
        if not valid_children or coord[1] >= boss_row:
            path_stats.append(new_counts)
            continue
        for child_coord in valid_children:
            stack.append((child_coord, new_counts))

    if not path_stats:
        path_stats = [{"elite": 0, "shop": 0, "restsite": 0, "monster": 0}]

    elites = [float(p.get("elite", 0)) for p in path_stats]
    shops = [float(p.get("shop", 0)) for p in path_stats]
    rests = [float(p.get("restsite", 0)) for p in path_stats]
    monsters = [float(p.get("monster", 0)) for p in path_stats]
    rows_to_boss = max(0.0, float(boss_row - start_coord[1]))
    return {
        "min_elite": min(elites) if elites else 0.0,
        "max_shop": max(shops) if shops else 0.0,
        "max_rest": max(rests) if rests else 0.0,
        "avg_monster": (sum(monsters) / max(1, len(monsters))) if monsters else 0.0,
        "rows_to_boss": rows_to_boss,
        "option_type": option_type,
        "forced_next_elite": bool(non_empty_child_types) and all(t == "elite" for t in non_empty_child_types),
        "has_rest_child": any(t == "restsite" for t in non_empty_child_types),
        "has_shop_child": any(t == "shop" for t in non_empty_child_types),
        "elite_free_path": _map_option_has_elite_free_path(map_state, action),
    }


def _score_act1_route_plan(
    state: dict[str, Any],
    route_stats: dict[str, Any],
) -> float:
    player = _extract_player(state)
    hp = float(player.get("hp", player.get("current_hp", 0)) or 0.0)
    max_hp = max(1.0, float(player.get("max_hp", 1) or 1.0))
    hp_ratio = max(0.0, min(1.0, hp / max_hp))
    gold = int(player.get("gold", 0) or 0)
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    floor = int(run.get("floor", 0) or 0)

    min_elite = float(route_stats.get("min_elite", 0.0) or 0.0)
    max_shop = float(route_stats.get("max_shop", 0.0) or 0.0)
    max_rest = float(route_stats.get("max_rest", 0.0) or 0.0)
    avg_monster = float(route_stats.get("avg_monster", 0.0) or 0.0)
    rows_to_boss = float(route_stats.get("rows_to_boss", 0.0) or 0.0)

    low_hp = max(0.0, (0.72 - hp_ratio) / 0.42)
    rich_enough = 1.0 if gold >= 120 else 0.65 if gold >= 75 else 0.25 if gold >= 50 else 0.0
    early_act = 1.0 if floor <= 8 else 0.55 if floor <= 11 else 0.2
    boss_pressure = max(0.0, (6.0 - rows_to_boss) / 6.0)

    elite_penalty = min_elite * (3.0 - 1.2 * hp_ratio + 0.6 * early_act + 0.5 * low_hp)
    rest_bonus = max_rest * (2.2 * low_hp + 0.45 * boss_pressure)
    shop_bonus = max_shop * (1.15 * rich_enough + 0.30 * boss_pressure)
    monster_penalty = avg_monster * (0.16 + 0.65 * low_hp)

    option_type = _lower_text(route_stats.get("option_type"))
    immediate_type_bonus = 0.0
    if option_type == "restsite":
        immediate_type_bonus += 0.35 + 0.30 * low_hp
    elif option_type == "shop":
        immediate_type_bonus += 0.25 + 0.35 * rich_enough
    elif option_type == "unknown":
        immediate_type_bonus += 0.05
    elif option_type == "elite":
        immediate_type_bonus -= 1.4 + 0.8 * low_hp

    if bool(route_stats.get("forced_next_elite")):
        elite_penalty += 3.5 + 1.5 * low_hp
    if bool(route_stats.get("has_rest_child")):
        immediate_type_bonus += 0.10 + 0.10 * low_hp
    if bool(route_stats.get("has_shop_child")):
        immediate_type_bonus += 0.08 + 0.10 * rich_enough
    if bool(route_stats.get("elite_free_path")):
        immediate_type_bonus += 0.30

    return float(-elite_penalty + rest_bonus + shop_bonus - monster_penalty + immediate_type_bonus)


def _choose_act1_no_elite_map_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    *,
    action_logits: np.ndarray | None = None,
    fallback_idx: int | None = None,
) -> tuple[int, dict[str, Any], str] | None:
    if not _is_act1_map_state(state) or not legal:
        return None
    map_state = state.get("map") if isinstance(state.get("map"), dict) else {}
    if not map_state:
        return None

    route_candidates: list[tuple[int, dict[str, Any], float]] = []
    safe_candidates: list[tuple[int, dict[str, Any], float]] = []
    for idx, action in enumerate(legal):
        if _lower_text(action.get("action")) != "choose_map_node":
            continue
        route_stats = _map_option_route_stats(state, action)
        if route_stats is None:
            continue
        score = _score_act1_route_plan(state, route_stats)
        route_candidates.append((idx, action, score))
        if bool(route_stats.get("elite_free_path")):
            safe_candidates.append((idx, action, score))

    if not route_candidates:
        return None

    candidates = safe_candidates or route_candidates
    if action_logits is not None and len(action_logits) >= len(legal):
        best_idx, best_action, _best_score = max(
            candidates,
            key=lambda item: float(action_logits[item[0]]) + item[2],
        )
        return int(best_idx), best_action, "act1_route_plan"
    if fallback_idx is not None:
        for idx, action, _score in candidates:
            if idx == fallback_idx:
                return int(idx), action, "act1_route_plan_keep"
    best_idx, best_action, _best_score = max(candidates, key=lambda item: item[2])
    return int(best_idx), best_action, "act1_route_plan_fallback"


_SHOP_REMOVE_PRIORITY = (
    "STRIKE_IRONCLAD",
    "DEFEND_IRONCLAD",
    "BASH",
)


def _choose_shop_remove_purchase_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> tuple[int, dict[str, Any], str] | None:
    """If at a shop screen with an affordable, in-stock remove_card service,
    return a forced shop_purchase action targeting it.

    Historical bug (pre-fix): legal actions from binary protocol always have
    action="shop_purchase" with no Label field — the original check
    `action.get("action") == "remove_card"` never matched, so this hard rule
    silently failed (only ~14% of shops actually used remove vs the intended
    100% when a remove offer was available and affordable). The correct path
    is to look up state.shop.items[index].category == "remove_card" and match
    the legal action by index.
    """
    if _lower_text(state.get("state_type")) != "shop":
        return None
    shop_items = (state.get("shop") or {}).get("items") or []
    remove_indices: set[int] = set()
    for item in shop_items:
        if _lower_text(item.get("category")) != "remove_card":
            continue
        if not bool(item.get("can_afford")):
            continue
        if not bool(item.get("is_stocked")):
            continue
        try:
            remove_indices.add(int(item.get("index", -1)))
        except (TypeError, ValueError):
            continue
    if not remove_indices:
        return None
    for idx, action in enumerate(legal):
        if _lower_text(action.get("action")) != "shop_purchase":
            continue
        try:
            action_index = int(action.get("index", -1))
        except (TypeError, ValueError):
            continue
        if action_index in remove_indices:
            return int(idx), action, "shop_force_remove"
    return None


def _choose_shop_remove_target_action(
    legal: list[dict[str, Any]],
) -> tuple[int, dict[str, Any], str] | None:
    if not legal:
        return None

    def _action_card_key(action: dict[str, Any]) -> str:
        for key in ("card_id", "label", "name", "note"):
            text = str(action.get(key) or "").strip()
            if text:
                return text.upper()
        return ""

    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for idx, action in enumerate(legal):
        action_name = _lower_text(action.get("action"))
        if "select" not in action_name:
            continue
        card_key = _action_card_key(action)
        priority = len(_SHOP_REMOVE_PRIORITY)
        for order, prefix in enumerate(_SHOP_REMOVE_PRIORITY):
            if card_key.startswith(prefix):
                priority = order
                break
        ranked.append((priority, idx, action))

    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    priority, idx, action = ranked[0]
    source = "shop_remove_basic_fallback" if priority >= len(_SHOP_REMOVE_PRIORITY) else "shop_remove_basic"
    return int(idx), action, source


def _build_shop_session_snapshot(state: dict[str, Any], *, step_i: int, floor: int) -> dict[str, Any]:
    shop_state = (state.get("shop") or {}) if isinstance(state, dict) else {}
    player = (shop_state.get("player") or state.get("player") or {}) if isinstance(state, dict) else {}
    offers: list[dict[str, Any]] = []
    for item in shop_state.get("items") or []:
        if not isinstance(item, dict):
            continue
        offers.append(
            {
                "index": int(item.get("index", -1) or -1),
                "category": str(item.get("category") or "unknown"),
                "id": str(item.get("id") or ""),
                "name": str(item.get("name") or item.get("id") or item.get("category") or "unknown"),
                "cost": int(item.get("cost", item.get("price", 0)) or 0),
                "can_afford": bool(item.get("can_afford")),
                "is_stocked": bool(item.get("is_stocked", True)),
                "on_sale": bool(item.get("on_sale")),
            }
        )
    return {
        "enter_step": step_i,
        "enter_floor": floor,
        "enter_gold": int(player.get("gold", 0) or 0),
        "offers": offers,
        "actions": [],
    }


def _normalize_card_slug(value: Any) -> str:
    text = _lower_text(value).replace(".title", "")
    for old, new in ((" ", "_"), ("-", "_"), ("/", "_")):
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _action_card_slug(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    for key in ("card_id", "id", "label", "name", "note"):
        slug = _normalize_card_slug(action.get(key))
        if slug:
            return slug
    return ""


def _runtime_skada_priors():
    global _RUNTIME_SKADA_PRIORS
    if _RUNTIME_SKADA_PRIORS is False:
        return None
    if _RUNTIME_SKADA_PRIORS is None:
        try:
            from skada.skada_priors import SkadaPriors
            priors = SkadaPriors()
            _RUNTIME_SKADA_PRIORS = priors if priors.loaded else False
        except Exception:
            _RUNTIME_SKADA_PRIORS = False
    return _RUNTIME_SKADA_PRIORS if _RUNTIME_SKADA_PRIORS is not False else None


def _lookup_boss_best_cards(boss_token: str) -> set[str]:
    priors = _runtime_skada_priors()
    if priors is None or not boss_token or boss_token == "unknown":
        return set()
    encounter_keys = (
        boss_token,
        boss_token.upper(),
        f"{boss_token}_boss",
        f"{boss_token.upper()}_BOSS",
    )
    for encounter in encounter_keys:
        boss = priors.boss(encounter)
        if boss is not None:
            return {str(card).strip().lower() for card in boss.best_cards if str(card).strip()}
    return set()


_BOSS_CARD_PREFS: dict[str, dict[str, set[str]]] = {
    "waterfall_giant": {
        "prefer": {
            "armaments", "barricade", "battle_trance", "body_slam", "disarm",
            "entrench", "flame_barrier", "ghostly_armor", "impervious",
            "inflame", "power_through", "rage", "second_wind", "shockwave",
            "shrug_it_off", "true_grit",
        },
        "avoid": {
            "setup_strike", "thunderclap", "twin_strike", "wild_strike",
        },
    },
    "soul_fysh": {
        "prefer": {
            "anger", "armaments", "battle_trance", "carnage", "dropkick",
            "headbutt", "hemokinesis", "inflame", "offering", "pommel_strike",
            "shrug_it_off", "thunderclap", "twin_strike", "uppercut",
        },
        "avoid": {
            "barricade", "entrench", "power_through",
        },
    },
}


def _boss_conditioned_card_bonus(
    state: dict[str, Any],
    action: dict[str, Any],
) -> float:
    action_name = _lower_text(action.get("action"))
    if action_name not in {"select_card_reward", "shop_purchase"}:
        return 0.0

    boss_token = extract_next_boss_token(state)
    if boss_token == "unknown":
        return 0.0

    slug = _action_card_slug(action)
    if not slug:
        return 0.0

    score = 0.0
    if slug in _lookup_boss_best_cards(boss_token):
        score += 2.0

    prefs = _BOSS_CARD_PREFS.get(boss_token) or {}
    if slug in prefs.get("prefer", set()):
        score += 1.0
    if slug in prefs.get("avoid", set()):
        score -= 0.75

    if boss_token == "waterfall_giant":
        if any(token in slug for token in ("barrier", "armor", "block", "shrug", "power_through", "grit")):
            score += 0.35
        if any(token in slug for token in ("inflame", "armaments", "barricade", "entrench")):
            score += 0.30
    elif boss_token == "soul_fysh":
        if any(token in slug for token in ("anger", "pommel", "headbutt", "uppercut", "strike", "slash")):
            score += 0.30
        if any(token in slug for token in ("inflame", "battle_trance", "offering")):
            score += 0.25
    return float(score)


def _choose_boss_conditioned_card_reward_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    *,
    action_logits: np.ndarray | None = None,
    fallback_idx: int | None = None,
    guidance_weight: float = 0.0,
) -> tuple[int, dict[str, Any], str] | None:
    if guidance_weight <= 0.0 or _lower_text(state.get("state_type")) != "card_reward" or not legal:
        return None

    pick_bonuses: dict[int, float] = {}
    best_pick_bonus = 0.0
    for idx, action in enumerate(legal):
        if _lower_text(action.get("action")) != "select_card_reward":
            continue
        bonus = _boss_conditioned_card_bonus(state, action) * guidance_weight
        pick_bonuses[idx] = bonus
        best_pick_bonus = max(best_pick_bonus, bonus)

    scored: list[tuple[float, int, dict[str, Any], float]] = []
    for idx, action in enumerate(legal):
        action_name = _lower_text(action.get("action"))
        base = float(action_logits[idx]) if action_logits is not None and idx < len(action_logits) else 0.0
        bonus = 0.0
        if action_name == "select_card_reward":
            bonus = pick_bonuses.get(idx, 0.0)
        elif action_name in {"skip", "skip_card_reward"}:
            bonus = -0.65 if best_pick_bonus >= 0.8 else -0.20 if best_pick_bonus >= 0.35 else 0.0
        scored.append((base + bonus, idx, action, bonus))

    if not scored:
        return None
    best_score, best_idx, best_action, best_bonus = max(scored, key=lambda item: item[0])
    if fallback_idx is not None:
        for score, idx, action, _bonus in scored:
            if idx == fallback_idx and idx == best_idx:
                return int(idx), action, "boss_card_guidance_keep"
            if idx == fallback_idx and best_score <= score + 0.05:
                return int(idx), action, "boss_card_guidance_keep"
    if abs(best_bonus) < 1e-5 and fallback_idx is not None and 0 <= fallback_idx < len(legal):
        return int(fallback_idx), legal[fallback_idx], "boss_card_guidance_keep"
    boss_token = extract_next_boss_token(state) or "unknown"
    return int(best_idx), best_action, f"boss_card_guidance_{boss_token}"


def _build_card_reward_decision_details(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    *,
    action_logits: np.ndarray | None,
    guidance_weight: float,
    selected_idx: int | None = None,
    source: str = "",
) -> dict[str, Any]:
    boss_token = extract_next_boss_token(state) or "unknown"
    probs: np.ndarray | None = None
    if action_logits is not None and len(action_logits) >= len(legal) and len(legal) > 0:
        logits = np.asarray(action_logits[:len(legal)], dtype=np.float32)
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        denom = float(exp.sum())
        if denom > 0.0:
            probs = exp / denom

    choices: list[dict[str, Any]] = []
    for idx, action in enumerate(legal):
        action_name = _lower_text(action.get("action"))
        boss_bonus = 0.0
        if action_name in {"select_card_reward", "skip", "skip_card_reward"}:
            if action_name == "select_card_reward":
                boss_bonus = _boss_conditioned_card_bonus(state, action) * max(0.0, guidance_weight)
            elif guidance_weight > 0.0:
                best_pick_bonus = 0.0
                for candidate in legal:
                    if _lower_text(candidate.get("action")) == "select_card_reward":
                        best_pick_bonus = max(
                            best_pick_bonus,
                            _boss_conditioned_card_bonus(state, candidate) * max(0.0, guidance_weight),
                        )
                boss_bonus = -0.65 if best_pick_bonus >= 0.8 else -0.20 if best_pick_bonus >= 0.35 else 0.0
        raw_logit = float(action_logits[idx]) if action_logits is not None and idx < len(action_logits) else 0.0
        final_score = raw_logit + boss_bonus
        choices.append(
            {
                "idx": int(idx),
                "action": str(action.get("action") or ""),
                "label": str(action.get("label") or action.get("card_id") or action.get("name") or action.get("action") or ""),
                "boss_bonus": round(float(boss_bonus), 4),
                "raw_logit": round(float(raw_logit), 4),
                "final_score": round(float(final_score), 4),
                "prob": round(float(probs[idx]), 4) if probs is not None and idx < len(probs) else None,
                "selected": bool(selected_idx == idx),
            }
        )
    choices.sort(key=lambda item: item["final_score"], reverse=True)
    return {
        "boss_token": boss_token,
        "guidance_weight": round(float(guidance_weight), 4),
        "source": source,
        "selected_idx": int(selected_idx) if selected_idx is not None else None,
        "choices": choices,
    }


def _build_shop_decision_details(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    *,
    action_logits: np.ndarray | None,
    selected_idx: int | None = None,
    source: str = "",
) -> dict[str, Any]:
    shop_snapshot = _build_shop_session_snapshot(
        state,
        step_i=0,
        floor=_safe_int(((state.get("run") or {}) if isinstance(state, dict) else {}).get("floor", 0), 0),
    )
    offers = shop_snapshot.get("offers") or []
    offer_by_slug: dict[str, dict[str, Any]] = {}
    for offer in offers:
        slug = _normalize_card_slug(offer.get("id") or offer.get("name") or offer.get("category"))
        if slug and slug not in offer_by_slug:
            offer_by_slug[slug] = offer

    probs: np.ndarray | None = None
    if action_logits is not None and len(action_logits) >= len(legal) and len(legal) > 0:
        logits = np.asarray(action_logits[:len(legal)], dtype=np.float32)
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        denom = float(exp.sum())
        if denom > 0.0:
            probs = exp / denom

    choices: list[dict[str, Any]] = []
    for idx, action in enumerate(legal):
        action_name = _lower_text(action.get("action"))
        label = str(action.get("label") or action.get("name") or action.get("action") or "")
        slug = _normalize_card_slug(action.get("card_id") or action.get("id") or label)
        offer = offer_by_slug.get(slug)
        if offer is None and action_name == "remove_card":
            offer = next((item for item in offers if _lower_text(item.get("category")) == "remove_card"), None)
        raw_logit = float(action_logits[idx]) if action_logits is not None and idx < len(action_logits) else 0.0
        entry = {
            "idx": int(idx),
            "action": str(action.get("action") or ""),
            "label": label,
            "raw_logit": round(float(raw_logit), 4),
            "prob": round(float(probs[idx]), 4) if probs is not None and idx < len(probs) else None,
            "selected": bool(selected_idx == idx),
        }
        if offer is not None:
            entry.update(
                {
                    "offer_category": str(offer.get("category") or ""),
                    "offer_name": str(offer.get("name") or offer.get("id") or ""),
                    "offer_cost": _safe_int(offer.get("cost", 0), 0),
                    "can_afford": bool(offer.get("can_afford")),
                    "is_stocked": bool(offer.get("is_stocked", True)),
                }
            )
        choices.append(entry)
    choices.sort(key=lambda item: item["raw_logit"], reverse=True)
    return {
        "enter_gold": int(shop_snapshot.get("enter_gold", 0) or 0),
        "source": source,
        "selected_idx": int(selected_idx) if selected_idx is not None else None,
        "choices": choices,
    }


def _combat_round_number(state: dict[str, Any]) -> int:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    return _safe_int(state.get("round_number") or battle.get("round_number"), 0)


def _combat_should_use_mcts(
    *,
    use_mcts_combat: bool,
    mcts_warmup_active: bool,
    combat_room_type: str,
    turn_action_index: int,
    mcts_first_n_actions_per_turn: int,
    mcts_full_search_on_elite_boss: bool,
) -> bool:
    if not use_mcts_combat or mcts_warmup_active:
        return False
    if mcts_full_search_on_elite_boss and combat_room_type in {"elite", "boss"}:
        return True
    if mcts_first_n_actions_per_turn <= 0:
        return True
    return turn_action_index < mcts_first_n_actions_per_turn


def _peek_config_path(argv: list[str]) -> str | None:
    """Read --config early so file values can become parser defaults."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    known, _unknown = pre_parser.parse_known_args(argv)
    return known.config


def _flatten_config_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested TOML tables into parser dest -> value pairs.

    Config files keep related settings grouped in sections, but leaf keys are
    still expected to match argparse dest names such as `ppo_lr` or
    `combat_teacher_loss_weight`.
    """
    flat: dict[str, Any] = {}

    def _visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict):
                _visit(value)
            else:
                normalized_key = str(key).replace("-", "_")
                flat[_CONFIG_ALIASES.get(normalized_key, normalized_key)] = value

    _visit(payload)
    return flat


def _load_train_hybrid_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a TOML table: {config_path}")
    return _flatten_config_mapping(payload)


def _configure_main_combat_path_mode(
    network: CombatPolicyValueNetwork,
    mode: str,
) -> str:
    """Choose whether the main combat rollout path trains residual attention.

    `mlp` preserves the legacy main path by freezing the residual attention
    parameters and forcing both gates to 0.
    `light_attention` leaves the residual branch trainable.
    """
    normalized = str(mode or "mlp").strip().lower()
    if normalized not in {"mlp", "light_attention"}:
        raise ValueError(f"Unsupported combat_main_path_mode: {mode}")

    attention_names = {
        "main_action_context_gate",
        "main_state_context_gate",
    }
    attention_prefixes = (
        "main_action_context_attn.",
        "main_action_context_norm.",
        "main_action_context_ffn.",
        "main_action_context_ffn_norm.",
    )
    use_attention = normalized == "light_attention"
    with torch.no_grad():
        if hasattr(network, "main_action_context_gate"):
            network.main_action_context_gate.fill_(0.0)
        if hasattr(network, "main_state_context_gate"):
            network.main_state_context_gate.fill_(0.0)
    for name, param in network.named_parameters():
        if name in attention_names or any(name.startswith(prefix) for prefix in attention_prefixes):
            param.requires_grad = use_attention
    return normalized


def _configure_offline_noncombat_ranking_head_mode(
    network: FullRunPolicyNetworkV2,
    mode: str,
) -> str:
    """Choose whether offline non-combat ranking uses residual attention."""
    return network.configure_offline_noncombat_ranking_head_mode(mode)


def _training_data_source_summary(
    *,
    args,
    effective_counterfactual_scoring: bool,
    effective_counterfactual_weight: float,
    matchup_dataset_size: int,
    combat_teacher_dataset_size: int,
) -> dict[str, Any]:
    """Describe which data sources currently influence training and how much."""
    return {
        "summary": {
            "character_id": str(args.character_id),
            "transport": str(args.transport),
            "num_envs": int(args.num_envs),
            "max_iterations": int(args.max_iterations),
            "combat_main_path_mode": str(getattr(args, "combat_main_path_mode", "mlp")),
            "offline_noncombat_ranking_head_mode": str(
                getattr(args, "offline_noncombat_ranking_head_mode", "mlp")
            ),
        },
        "online_data": {
            "noncombat_ppo_rollouts": {
                "enabled": not bool(getattr(args, "freeze_ppo", False)),
                "role": "Primary non-combat learning signal from live self-play episodes.",
                "notes": [
                    "Collected into ppo_buffer from live simulator episodes.",
                    "Merged with a floor-depth weight, so deeper runs count more.",
                ],
            },
            "combat_ppo_rollouts": {
                "enabled": not bool(getattr(args, "freeze_combat", False)),
                "role": "Online combat policy/value updates from NN-selected combat actions.",
                "notes": [
                    "Only NN-selected combat steps are stored; MCTS-chosen steps are excluded.",
                    f"Monster hallway combat reward merge weight: {float(args.combat_monster_reward_weight):.3f}.",
                ],
            },
            "combat_search_examples": {
                "enabled": bool(getattr(args, "mcts", False)),
                "role": "Higher-quality combat supervision generated by MCTS search.",
                "notes": [
                    f"MCTS sims per search: {int(args.mcts_sims)}.",
                    "Used to train the combat search/value head when MCTS is enabled.",
                ],
            },
        },
        "offline_or_auxiliary_data": {
            "offline_noncombat_ranking": {
                "enabled": matchup_dataset_size > 0,
                "canonical_arg": "matchup_data_dir",
                "recommended_alias": "offline_noncombat_ranking_data_dir",
                "samples": int(matchup_dataset_size),
                "weight": float(args.matchup_loss_weight),
                "warmup_iters": int(args.matchup_warmup_iters),
                "updates_per_iter": int(args.matchup_updates_per_iter),
                "notes": [
                    "Supervises non-combat card preference / ranking score head.",
                    "Legacy internal name is 'matchup'; the clearer config alias is 'offline_noncombat_ranking_*'.",
                    f"Ranking head mode: {getattr(args, 'offline_noncombat_ranking_head_mode', 'mlp')}.",
                    "Loss weight is fixed unless matchup_loss_decay_tau is enabled.",
                ],
            },
            "offline_combat_teacher": {
                "enabled": combat_teacher_dataset_size > 0,
                "canonical_arg": "combat_teacher_data_dir",
                "recommended_alias": "offline_combat_teacher_data_dir",
                "samples": int(combat_teacher_dataset_size),
                "weight": float(args.combat_teacher_loss_weight),
                "warmup_iters": int(args.combat_teacher_warmup_iters),
                "updates_per_iter": int(args.combat_teacher_updates_per_iter),
                "notes": [
                    "Offline turn-solver teacher data for combat reranking.",
                    "The clearer config alias is 'offline_combat_teacher_*'.",
                    "As the dataset grows, fixed per-iter updates reduce per-sample revisit frequency.",
                ],
            },
            "counterfactual_reward": {
                "enabled": bool(effective_counterfactual_scoring),
                "weight": float(effective_counterfactual_weight),
                "notes": [
                    "Reward shaping term blended into non-combat PPO rewards.",
                    f"Skada prior blend gamma: {float(args.skada_prior_weight):.3f}.",
                ],
            },
            "saved_offline_episodes": {
                "enabled": bool(args.save_offline_data),
                "min_floor": int(args.offline_min_floor),
                "canonical_arg": "save_offline_data",
                "recommended_alias": "saved_offline_episodes_enabled",
                "notes": [
                    "Saves high-quality episodes to disk for future offline training.",
                    "This saver does not automatically feed back into the current run.",
                ],
            },
        },
    }


def _write_training_flow_snapshot(
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    (output_dir / "training_sources.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_lines = [
        "# Hybrid Training Data Flow",
        "",
        "This file is generated at launch time so each training run records",
        "which data sources are active and how strongly they influence updates.",
        "",
        "## Online Data",
    ]
    for name, payload in summary.get("online_data", {}).items():
        md_lines.append(f"- `{name}`: enabled={payload.get('enabled')}")
        md_lines.append(f"  role: {payload.get('role')}")
        for note in payload.get("notes", []):
            md_lines.append(f"  note: {note}")
    md_lines.extend(["", "## Offline / Auxiliary Data"])
    for name, payload in summary.get("offline_or_auxiliary_data", {}).items():
        detail_bits = []
        if "samples" in payload:
            detail_bits.append(f"samples={payload['samples']}")
        if "weight" in payload:
            detail_bits.append(f"weight={payload['weight']}")
        if "updates_per_iter" in payload:
            detail_bits.append(f"updates_per_iter={payload['updates_per_iter']}")
        suffix = f" ({', '.join(detail_bits)})" if detail_bits else ""
        md_lines.append(f"- `{name}`: enabled={payload.get('enabled')}{suffix}")
        for note in payload.get("notes", []):
            md_lines.append(f"  note: {note}")
    (output_dir / "training_flow.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested (signal %d)...", signum)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}
ACT1_CLEAR_FLOOR = 17
SELECTION_ACTION_NAMES = {
    "select_card",
    "combat_select_card",
    "combat_confirm_selection",
    "confirm_selection",
    "cancel_selection",
    "skip_relic_selection",
}
SELECTION_SCREENS = {"card_select", "hand_select", "relic_select"}
POST_CARD_REWARD_ACTIONS = {"select_card_reward", "skip_card_reward"}
REWARD_FLOW_SCREENS = {"combat_rewards", "card_reward"}


def _legal_action_name_set(legal: list[dict[str, Any]]) -> set[str]:
    return _shared_legal_action_name_set(legal)


def _is_selection_screen(state_type: str, legal: list[dict[str, Any]]) -> bool:
    return _shared_is_selection_screen(state_type, legal)


def _choose_auto_progress_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    last_action_name: str | None = None,
    last_reward_claim_sig: str | None = None,
    last_reward_claim_count: int | None = None,
    reward_chain_card_reward_seen: bool = False,
) -> dict[str, Any] | None:
    return _shared_choose_auto_progress_action(
        state,
        legal,
        last_action_name=last_action_name,
        last_reward_claim_sig=last_reward_claim_sig,
        last_reward_claim_count=last_reward_claim_count,
        reward_chain_card_reward_seen=reward_chain_card_reward_seen,
    )


def _choose_empty_legal_recovery_action(
    state: dict[str, Any],
    last_action_name: str | None = None,
) -> dict[str, Any] | None:
    st = (state.get("state_type") or "").strip().lower()
    last_action_name = str(last_action_name or "").strip().lower()

    if st == "event":
        event_state = state.get("event") or {}
        if event_state.get("in_dialogue"):
            return {"action": "advance_dialogue"}
        if event_state.get("can_proceed") or event_state.get("is_finished"):
            return {"action": "proceed"}
        if last_action_name in POST_CARD_REWARD_ACTIONS and (
            event_state.get("can_proceed") or event_state.get("is_finished")
        ):
            return {"action": "proceed"}

    if st == "combat_rewards":
        rewards_state = state.get("combat_rewards") or state.get("rewards") or {}
        if rewards_state.get("can_proceed"):
            return {"action": "proceed"}

    if st in {"treasure", "rest", "rest_site", "shop"}:
        return {"action": "proceed"}

    return None


def _choose_rest_site_repeat_escape_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> dict[str, Any] | None:
    st = (state.get("state_type") or "").strip().lower()
    if st != "rest_site":
        return None

    rest_state = state.get("rest_site")
    if not isinstance(rest_state, dict):
        rest_state = {}

    option_by_index: dict[int, dict[str, Any]] = {}
    for option in rest_state.get("options") or []:
        if not isinstance(option, dict):
            continue
        try:
            option_index = int(option.get("index", -1))
        except Exception:
            continue
        option_by_index[option_index] = option

    def _action_name(action: dict[str, Any]) -> str:
        return str(action.get("action") or "").strip().lower()

    def _normalized_action_text(action: dict[str, Any]) -> str:
        parts: list[str] = []
        parts.append(str(action.get("label") or ""))
        try:
            option_index = int(action.get("index", -1))
        except Exception:
            option_index = -1
        option = option_by_index.get(option_index) or {}
        for key in ("id", "name", "description"):
            parts.append(str(option.get(key) or ""))
        return " ".join(part.strip().lower() for part in parts if str(part).strip())

    proceed = next((action for action in legal if _action_name(action) == "proceed"), None)
    rest_actions = [action for action in legal if _action_name(action) == "choose_rest_option"]
    if not rest_actions:
        return proceed

    for action in rest_actions:
        normalized = _normalized_action_text(action)
        if any(token in normalized for token in ("rest", "heal", "sleep", "recover")):
            return action

    if proceed is not None:
        return proceed

    return min(rest_actions, key=lambda action: int(action.get("index", 0)))


def _combat_rewards_state(state: dict[str, Any]) -> dict[str, Any]:
    return _shared_combat_rewards_state(state)


def _reward_item_claimable(state: dict[str, Any], reward_item: dict[str, Any] | None) -> bool:
    return _shared_reward_item_claimable(state, reward_item)


def _choose_claimable_reward_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    *,
    prefer_highest_index: bool = False,
) -> dict[str, Any] | None:
    return _shared_choose_claimable_reward_action(
        state,
        legal,
        prefer_highest_index=prefer_highest_index,
    )


def _claim_reward_action_count(legal: list[dict[str, Any]]) -> int:
    return _shared_claim_reward_action_count(legal)


def _reward_claim_signature(state: dict[str, Any], action: dict[str, Any] | None) -> str:
    return _shared_reward_claim_signature(state, action)


def _next_reward_claim_signature(
    state_type: str,
    state: dict[str, Any],
    action: dict[str, Any] | None,
) -> str:
    return _shared_next_reward_claim_signature(state_type, state, action)


def _combat_player_view(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle")
    if isinstance(battle, dict):
        player = battle.get("player")
        if isinstance(player, dict):
            return player
    player = state.get("player")
    return player if isinstance(player, dict) else {}


def _combat_hand_summary(state: dict[str, Any], max_cards: int = 6) -> str:
    battle = state.get("battle") or {}
    hand = battle.get("hand") or _combat_player_view(state).get("hand") or []
    if not isinstance(hand, list) or not hand:
        return "-"
    parts: list[str] = []
    for card in hand[:max_cards]:
        if not isinstance(card, dict):
            continue
        name = str(card.get("name") or card.get("label") or card.get("id") or "?")
        cost = card.get("cost")
        parts.append(f"{name}({cost})")
    more = "" if len(hand) <= max_cards else f"+{len(hand) - max_cards}"
    return ",".join(parts) + more


def _combat_enemy_intent_summary(state: dict[str, Any], max_enemies: int = 3) -> str:
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    if not isinstance(enemies, list) or not enemies:
        return "-"
    parts: list[str] = []
    for enemy in enemies[:max_enemies]:
        if not isinstance(enemy, dict):
            continue
        name = str(enemy.get("name") or enemy.get("id") or "?")
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        block = _safe_int(enemy.get("block", 0), 0)
        intents = enemy.get("intents") or []
        if isinstance(intents, list) and intents:
            intent0 = intents[0] if isinstance(intents[0], dict) else {}
        else:
            intent0 = {}
        intent_type = str(intent0.get("type") or intent0.get("label") or "?")
        dmg = _safe_int(intent0.get("damage", 0), 0)
        hits = max(1, _safe_int(intent0.get("hits", 1), 1))
        dmg_str = f"{dmg}x{hits}" if dmg > 0 else intent_type
        parts.append(f"{name}[{hp}/{block}:{dmg_str}]")
    more = "" if len(enemies) <= max_enemies else f"+{len(enemies) - max_enemies}"
    return ",".join(parts) + more


def _combat_enemy_group_key(state: dict[str, Any]) -> str:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    names: list[str] = []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        name = str(enemy.get("id") or enemy.get("name") or "").strip().upper()
        if name:
            names.append(name)
    return "+".join(names) if names else "UNKNOWN"


def _combat_hand_summary_zh(state: dict[str, Any], max_cards: int = 6) -> str:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = _combat_player_view(state)
    hand = battle.get("hand") or player.get("hand") or []
    parts: list[str] = []
    for card in hand[:max_cards]:
        if not isinstance(card, dict):
            continue
        name = _trace_resolve_name(card.get("name") or card.get("id"), category="card")
        cost = card.get("cost_for_turn", card.get("cost", "?"))
        parts.append(f"{name}({cost})")
    if not parts:
        return "-"
    more = "" if len(hand) <= max_cards else f" +{len(hand) - max_cards}"
    return "，".join(parts) + more


def _combat_enemy_intent_summary_zh(state: dict[str, Any], max_enemies: int = 3) -> str:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    parts: list[str] = []
    for enemy in enemies[:max_enemies]:
        if not isinstance(enemy, dict):
            continue
        name = _trace_resolve_name(enemy.get("name") or enemy.get("id"), category="encounter")
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        block = _safe_int(enemy.get("block", 0), 0)
        intents = enemy.get("intents") or []
        if isinstance(intents, list) and intents:
            intent0 = intents[0] if isinstance(intents[0], dict) else {}
        else:
            intent0 = {}
        intent_type = str(intent0.get("type") or intent0.get("label") or "?")
        dmg = _safe_int(intent0.get("damage", 0), 0)
        hits = max(1, _safe_int(intent0.get("hits", 1), 1))
        if dmg > 0:
            intent_desc = f"攻击 {dmg}x{hits}"
        else:
            intent_desc = _trace_pretty_token(intent_type)
        parts.append(f"{name} {hp}/{block}，意图 {intent_desc}")
    if not parts:
        return "-"
    more = "" if len(enemies) <= max_enemies else f" +{len(enemies) - max_enemies}"
    return "；".join(parts) + more


def _combat_enemy_map(state: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    out: dict[Any, dict[str, Any]] = {}
    for idx, enemy in enumerate(enemies):
        if not isinstance(enemy, dict):
            continue
        key = enemy.get("combat_id", enemy.get("target_id", enemy.get("entity_id", idx)))
        out[key] = enemy
    return out


def _combat_enemy_change_items(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    action: dict[str, Any] | None,
    *,
    max_items: int = 6,
) -> list[dict[str, Any]]:
    pre_map = _combat_enemy_map(pre_state)
    post_map = _combat_enemy_map(post_state)
    target_key = None if not isinstance(action, dict) else action.get("target_id", action.get("enemy_id", action.get("target")))
    items: list[dict[str, Any]] = []
    for key, pre_enemy in pre_map.items():
        post_enemy = post_map.get(key)
        pre_hp = _safe_int(pre_enemy.get("hp", pre_enemy.get("current_hp", 0)), 0)
        pre_blk = _safe_int(pre_enemy.get("block", 0), 0)
        name = _trace_resolve_name((post_enemy or pre_enemy).get("name") or (post_enemy or pre_enemy).get("id"), category="encounter")
        if post_enemy is None:
            items.append({
                "key": key,
                "name": name,
                "pre_hp": pre_hp,
                "post_hp": 0,
                "pre_block": pre_blk,
                "post_block": 0,
                "defeated": True,
                "targeted": key == target_key,
            })
            continue
        post_hp = _safe_int(post_enemy.get("hp", post_enemy.get("current_hp", 0)), 0)
        post_blk = _safe_int(post_enemy.get("block", 0), 0)
        if pre_hp != post_hp or pre_blk != post_blk or key == target_key:
            items.append({
                "key": key,
                "name": name,
                "pre_hp": pre_hp,
                "post_hp": post_hp,
                "pre_block": pre_blk,
                "post_block": post_blk,
                "defeated": False,
                "targeted": key == target_key,
            })
    return items[:max_items]


def _trace_enemy_change_summary(pre_state: dict[str, Any], post_state: dict[str, Any], action: dict[str, Any] | None) -> str:
    items = _combat_enemy_change_items(pre_state, post_state, action, max_items=3)
    parts: list[str] = []
    for item in items:
        if item.get("defeated"):
            parts.append(f"{item['name']} 被击败")
        else:
            parts.append(
                f"{item['name']} 血量 {item['pre_hp']}->{item['post_hp']}，"
                f"格挡 {item['pre_block']}->{item['post_block']}"
            )
    return "；".join(parts) if parts else "敌方数值无明显变化"


def _combat_step_structured_summary(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    action: dict[str, Any] | None,
) -> dict[str, Any]:
    pre_player = _combat_player_view(pre_state)
    post_player = _combat_player_view(post_state)
    target_key = None if not isinstance(action, dict) else action.get("target_id", action.get("enemy_id", action.get("target")))
    target_enemy = _combat_enemy_map(pre_state).get(target_key)
    return {
        "pre_hp": _safe_int(pre_player.get("hp", pre_player.get("current_hp", 0)), 0),
        "post_hp": _safe_int(post_player.get("hp", post_player.get("current_hp", 0)), 0),
        "pre_block": _safe_int(pre_player.get("block", 0), 0),
        "post_block": _safe_int(post_player.get("block", 0), 0),
        "pre_energy": _safe_int(pre_player.get("energy", 0), 0),
        "post_energy": _safe_int(post_player.get("energy", 0), 0),
        "target_key": target_key,
        "target_name": (
            _trace_resolve_name(target_enemy.get("name") or target_enemy.get("id"), category="encounter")
            if isinstance(target_enemy, dict)
            else ""
        ),
        "enemy_changes": _combat_enemy_change_items(pre_state, post_state, action),
        "pre_intent": _combat_enemy_intent_summary(pre_state),
        "pre_intent_zh": _combat_enemy_intent_summary_zh(pre_state),
        "next_intent": _combat_enemy_intent_summary(post_state),
        "next_intent_zh": _combat_enemy_intent_summary_zh(post_state),
        "post_state_type": _lower_text(post_state.get("state_type")),
    }


def _combat_action_label_zh(action: dict[str, Any] | None, state: dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return "未知动作"
    action_name = _combat_action_name(action)
    label = _combat_action_label(action)
    if action_name == "play_card":
        label = _trace_resolve_name(label, category="card")
        return f"打出「{label}」"
    if action_name == "use_potion":
        player = _combat_player_view(state)
        potions = player.get("potions") or []
        slot = action.get("slot")
        potion_name = None
        for potion in potions:
            if isinstance(potion, dict) and potion.get("slot") == slot:
                potion_name = potion.get("name") or potion.get("id")
                break
        potion_label = _trace_resolve_name(potion_name or label)
        return f"使用药水「{potion_label}」"
    if action_name == "end_turn":
        return "结束回合"
    return _trace_resolve_name(label)


def _combat_target_label_zh(action: dict[str, Any] | None, state: dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return ""
    target_key = action.get("target_id", action.get("enemy_id", action.get("target")))
    if target_key in (None, ""):
        return ""
    enemy = _combat_enemy_map(state).get(target_key)
    if isinstance(enemy, dict):
        name = _trace_resolve_name(enemy.get("name") or enemy.get("id"), category="encounter")
        return f" -> 目标「{name}」"
    return f" -> 目标 `{target_key}`"


def _topk_action_summary_zh(
    legal: list[dict[str, Any]],
    logits_or_probs: np.ndarray | list[float] | torch.Tensor,
    k: int = 3,
    already_probs: bool = False,
) -> str:
    raw = _topk_action_summary(legal, logits_or_probs, k=k, already_probs=already_probs)
    if raw == "-" or not raw:
        return raw
    parts: list[str] = []
    for chunk in raw.split(" | "):
        if ":" not in chunk:
            parts.append(_trace_resolve_name(chunk))
            continue
        label, prob = chunk.rsplit(":", 1)
        parts.append(f"{_trace_resolve_name(label, category='card')}:{prob}")
    return " | ".join(parts)


def _combat_result_summary_zh(pre_state: dict[str, Any], post_state: dict[str, Any], action: dict[str, Any] | None) -> str:
    pre_player = _combat_player_view(pre_state)
    post_player = _combat_player_view(post_state)
    pre_hp = _safe_int(pre_player.get("hp", pre_player.get("current_hp", 0)), 0)
    post_hp = _safe_int(post_player.get("hp", post_player.get("current_hp", 0)), 0)
    pre_blk = _safe_int(pre_player.get("block", 0), 0)
    post_blk = _safe_int(post_player.get("block", 0), 0)
    pre_energy = _safe_int(pre_player.get("energy", 0), 0)
    post_energy = _safe_int(post_player.get("energy", 0), 0)
    enemy_delta = _trace_enemy_change_summary(pre_state, post_state, action)
    next_intent = _combat_enemy_intent_summary_zh(post_state)
    if _lower_text(post_state.get("state_type")) not in COMBAT_SCREENS:
        next_intent = "敌人全部击败，战斗结束"
    return (
        f"结果：我方生命 {pre_hp}->{post_hp}，格挡 {pre_blk}->{post_blk}，能量 {pre_energy}->{post_energy}；"
        f"{enemy_delta}；下拍：{next_intent}"
    )


def _action_target_summary(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    for key in ("target", "target_id", "enemy_id", "slot"):
        value = action.get(key)
        if value not in (None, ""):
            return f" target={value}"
    return ""


def _topk_action_summary(
    legal: list[dict[str, Any]],
    logits_or_probs: np.ndarray | list[float] | torch.Tensor,
    k: int = 3,
    already_probs: bool = False,
) -> str:
    if not legal:
        return "-"
    if isinstance(logits_or_probs, torch.Tensor):
        arr = logits_or_probs.detach().float().cpu().numpy()
    else:
        arr = np.asarray(logits_or_probs, dtype=np.float64)
    arr = np.ravel(arr)
    if arr.size == 0:
        return "-"
    arr = arr[:len(legal)]
    if arr.size == 0:
        return "-"
    if already_probs:
        probs = arr
    else:
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        denom = np.sum(exp)
        probs = exp / denom if denom > 0 else np.zeros_like(arr)
    order = np.argsort(-probs)[: min(k, len(legal))]
    parts: list[str] = []
    for idx in order:
        action = legal[int(idx)]
        label = str(action.get("label") or action.get("action") or idx)
        parts.append(f"{label}:{float(probs[int(idx)]):.2f}")
    return " | ".join(parts)


def _combat_action_label(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return "?"
    return str(action.get("label") or action.get("action") or "?")


def _combat_action_name(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("action") or action.get("type") or "").strip().lower()


def _combat_is_end_turn_action(action: dict[str, Any] | None) -> bool:
    return _combat_action_name(action) == "end_turn"


def _combat_is_use_potion_action(action: dict[str, Any] | None) -> bool:
    return _combat_action_name(action) == "use_potion"


def _combat_action_looks_defensive(action: dict[str, Any] | None) -> bool:
    label = _combat_action_label(action).strip().lower()
    if not label or _combat_is_end_turn_action(action) or _combat_is_use_potion_action(action):
        return False
    defense_tokens = (
        "defend",
        "block",
        "barrier",
        "armor",
        "shield",
        "shrug",
        "panic",
        "ghostly",
        "power through",
        "flame barrier",
        "iron wave",
        "entrench",
    )
    return any(token in label for token in defense_tokens)


def _combat_action_looks_attack(action: dict[str, Any] | None) -> bool:
    if _combat_action_name(action) != "play_card":
        return False
    if _combat_action_looks_defensive(action):
        return False
    if any(action.get(key) not in (None, "") for key in ("target", "target_id", "enemy_id")):
        return True
    label = _combat_action_label(action).strip().lower()
    attack_tokens = (
        "strike",
        "bash",
        "anger",
        "boomerang",
        "slash",
        "whirlwind",
        "pummel",
        "headbutt",
        "uppercut",
        "hemokinesis",
        "carnage",
        "bludgeon",
        "perfected",
        "clothesline",
        "dropkick",
        "pommel",
        "twin",
        "sword",
        "sever",
        "thunderclap",
        "wild strike",
        "body slam",
    )
    return any(token in label for token in attack_tokens)


def _combat_played_card_from_action(
    state: dict[str, Any],
    action: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if _combat_action_name(action) != "play_card" or not isinstance(action, dict):
        return None
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = _extract_player(state)
    hand = battle.get("hand") or player.get("hand") or []
    card_idx = _safe_int(
        action.get("card_index", action.get("hand_index", action.get("index", -1))),
        -1,
    )
    if 0 <= card_idx < len(hand) and isinstance(hand[card_idx], dict):
        return dict(hand[card_idx])
    action_label = str(action.get("label") or "").strip().lower()
    if action_label:
        for card in hand:
            if not isinstance(card, dict):
                continue
            card_name = str(card.get("name") or card.get("id") or "").strip().lower()
            if card_name and card_name in action_label:
                return dict(card)
    return None


def _combat_card_type(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get("type") or card.get("card_type") or card.get("cardType") or "").strip().lower()


def _combat_card_effect_summary(card: dict[str, Any] | None) -> tuple[float, float, float, float, float, float, float]:
    if not isinstance(card, dict):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    cost = _to_float(card.get("cost_for_turn", card.get("cost", card.get("energy_cost", 0))))
    damage = _to_float(card.get("damage", card.get("base_damage", card.get("attack_damage", 0))))
    block = _to_float(card.get("block", card.get("base_block", 0)))
    draw = _to_float(card.get("draw", card.get("cards_to_draw", card.get("draw_amount", 0))))
    magic = _to_float(card.get("magic_number", card.get("magic", 0)))
    text = _lower_text(card.get("description") or card.get("raw_description") or card.get("text") or "")
    if draw <= 0 and "draw" in text:
        draw = max(draw, magic)
    discard = 1.0 if any(tok in text for tok in ("discard", "put a card from your hand")) else 0.0
    exhaust = 1.0 if card.get("exhaust") or "exhaust" in text else 0.0
    create = 1.0 if any(tok in text for tok in ("add ", "create ", "shuffle")) else 0.0
    return cost, damage, block, draw, discard, exhaust, create


def _new_combat_turn_prefix() -> dict[str, Any]:
    return {
        "action_count": 0,
        "cards_played": 0,
        "attack_count": 0,
        "skill_count": 0,
        "power_count": 0,
        "targeted_count": 0,
        "non_card_count": 0,
        "energy_spent": 0.0,
        "potion_count": 0,
        "selection_count": 0,
        "damage_est": 0.0,
        "block_est": 0.0,
        "draw_est": 0.0,
        "discard_count": 0.0,
        "exhaust_count": 0.0,
        "create_count": 0.0,
        "last_action_attack": 0.0,
        "last_action_skill": 0.0,
        "last_action_power": 0.0,
        "last_action_non_card": 0.0,
        "recent_cards": [],
    }


def _attach_combat_turn_prefix(state: dict[str, Any], prefix: dict[str, Any]) -> None:
    if isinstance(state, dict):
        state["_combat_turn_prefix"] = {
            "action_count": int(prefix.get("action_count", 0)),
            "cards_played": int(prefix.get("cards_played", 0)),
            "attack_count": int(prefix.get("attack_count", 0)),
            "skill_count": int(prefix.get("skill_count", 0)),
            "power_count": int(prefix.get("power_count", 0)),
            "targeted_count": int(prefix.get("targeted_count", 0)),
            "non_card_count": int(prefix.get("non_card_count", 0)),
            "energy_spent": float(prefix.get("energy_spent", 0.0)),
            "potion_count": int(prefix.get("potion_count", 0)),
            "selection_count": int(prefix.get("selection_count", 0)),
            "damage_est": float(prefix.get("damage_est", 0.0)),
            "block_est": float(prefix.get("block_est", 0.0)),
            "draw_est": float(prefix.get("draw_est", 0.0)),
            "discard_count": float(prefix.get("discard_count", 0.0)),
            "exhaust_count": float(prefix.get("exhaust_count", 0.0)),
            "create_count": float(prefix.get("create_count", 0.0)),
            "last_action_attack": float(prefix.get("last_action_attack", 0.0)),
            "last_action_skill": float(prefix.get("last_action_skill", 0.0)),
            "last_action_power": float(prefix.get("last_action_power", 0.0)),
            "last_action_non_card": float(prefix.get("last_action_non_card", 0.0)),
            "recent_cards": [dict(card) for card in prefix.get("recent_cards", []) if isinstance(card, dict)][-4:],
        }


def _update_combat_turn_prefix(
    prefix: dict[str, Any],
    *,
    state: dict[str, Any],
    action: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = {
        "action_count": int(prefix.get("action_count", 0)),
        "cards_played": int(prefix.get("cards_played", 0)),
        "attack_count": int(prefix.get("attack_count", 0)),
        "skill_count": int(prefix.get("skill_count", 0)),
        "power_count": int(prefix.get("power_count", 0)),
        "targeted_count": int(prefix.get("targeted_count", 0)),
        "non_card_count": int(prefix.get("non_card_count", 0)),
        "energy_spent": float(prefix.get("energy_spent", 0.0)),
        "potion_count": int(prefix.get("potion_count", 0)),
        "selection_count": int(prefix.get("selection_count", 0)),
        "damage_est": float(prefix.get("damage_est", 0.0)),
        "block_est": float(prefix.get("block_est", 0.0)),
        "draw_est": float(prefix.get("draw_est", 0.0)),
        "discard_count": float(prefix.get("discard_count", 0.0)),
        "exhaust_count": float(prefix.get("exhaust_count", 0.0)),
        "create_count": float(prefix.get("create_count", 0.0)),
        "last_action_attack": 0.0,
        "last_action_skill": 0.0,
        "last_action_power": 0.0,
        "last_action_non_card": 0.0,
        "recent_cards": [dict(card) for card in prefix.get("recent_cards", []) if isinstance(card, dict)][-4:],
    }
    if not isinstance(action, dict):
        return updated
    updated["action_count"] += 1
    if any(action.get(key) not in (None, "") for key in ("target", "target_id", "enemy_id")):
        updated["targeted_count"] += 1
    played_card = _combat_played_card_from_action(state, action)
    if played_card is None:
        updated["non_card_count"] += 1
        updated["last_action_non_card"] = 1.0
        action_name = _combat_action_name(action)
        if action_name == "use_potion":
            updated["potion_count"] += 1
        elif action_name in {"select_hand_card", "select_card_option", "confirm_selection", "cancel_selection"}:
            updated["selection_count"] += 1
        return updated
    updated["cards_played"] += 1
    updated["recent_cards"].append(played_card)
    updated["recent_cards"] = updated["recent_cards"][-4:]
    card_type = _combat_card_type(played_card)
    if "attack" in card_type:
        updated["attack_count"] += 1
        updated["last_action_attack"] = 1.0
    elif "skill" in card_type:
        updated["skill_count"] += 1
        updated["last_action_skill"] = 1.0
    elif "power" in card_type:
        updated["power_count"] += 1
        updated["last_action_power"] = 1.0
    cost, damage, block, draw, discard, exhaust, create = _combat_card_effect_summary(played_card)
    updated["energy_spent"] += max(0.0, cost)
    updated["damage_est"] += max(0.0, damage)
    updated["block_est"] += max(0.0, block)
    updated["draw_est"] += max(0.0, draw)
    updated["discard_count"] += max(0.0, discard)
    updated["exhaust_count"] += max(0.0, exhaust)
    updated["create_count"] += max(0.0, create)
    return updated


def _combat_root_topk_summary(root: Any, k: int = 5) -> str:
    children_map = getattr(root, "children", None)
    if not isinstance(children_map, dict) or not children_map:
        return "-"
    children = list(children_map.values())
    total_visits = max(1, sum(max(0, int(getattr(child, "visit_count", 0))) for child in children))
    ranked = sorted(
        children,
        key=lambda child: (
            int(getattr(child, "visit_count", 0)),
            float(getattr(child, "prior", 0.0)),
            float(getattr(child, "q_value", 0.0)),
        ),
        reverse=True,
    )[: min(k, len(children))]
    parts: list[str] = []
    for child in ranked:
        label = _combat_action_label(getattr(child, "action", None))
        visits = max(0, int(getattr(child, "visit_count", 0)))
        visit_frac = visits / total_visits
        q_value = float(getattr(child, "q_value", 0.0))
        prior = float(getattr(child, "prior", 0.0))
        parts.append(f"{label}:n={visits}/{visit_frac:.2f},q={q_value:.2f},p={prior:.2f}")
    return " | ".join(parts)


def _combat_root_action_summary(root: Any, action: dict[str, Any] | None) -> str:
    children_map = getattr(root, "children", None)
    if not isinstance(children_map, dict) or not children_map or not isinstance(action, dict):
        return "chosen[missing]"
    child = children_map.get(action_key(action))
    if child is None:
        for candidate in children_map.values():
            cand_action = getattr(candidate, "action", None)
            if (
                isinstance(cand_action, dict)
                and cand_action.get("action") == action.get("action")
                and cand_action.get("label") == action.get("label")
                and cand_action.get("target") == action.get("target")
            ):
                child = candidate
                break
    if child is None:
        return "chosen[missing]"
    total_visits = max(1, sum(max(0, int(getattr(node, "visit_count", 0))) for node in children_map.values()))
    visits = max(0, int(getattr(child, "visit_count", 0)))
    visit_frac = visits / total_visits
    q_value = float(getattr(child, "q_value", 0.0))
    prior = float(getattr(child, "prior", 0.0))
    return f"chosen[n={visits}/{visit_frac:.2f},q={q_value:.2f},p={prior:.2f}]"


def _combat_mcts_suspect_reasons(
    *,
    action: dict[str, Any] | None,
    legal: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(action, dict):
        return reasons
    player = _combat_player_view(state)
    energy = _safe_int(player.get("energy", 0), 0)
    has_attack_option = any(_combat_action_looks_attack(candidate) for candidate in legal)
    has_defense_option = any(_combat_action_looks_defensive(candidate) for candidate in legal)
    if _combat_is_end_turn_action(action) and energy > 0:
        remaining_plays = [
            candidate for candidate in legal
            if not _combat_is_end_turn_action(candidate)
            and _combat_action_name(candidate) not in {"confirm_selection", "cancel_selection"}
        ]
        if remaining_plays:
            reasons.append("end_turn_with_energy")
        if has_attack_option:
            reasons.append("end_turn_skips_attack")
        if has_defense_option:
            reasons.append("end_turn_skips_block")
    if _combat_is_use_potion_action(action):
        reasons.append("use_potion")
    if _combat_action_looks_defensive(action) and energy > 0 and has_attack_option:
        reasons.append("defense_bias")
    return reasons


_COMBAT_HARD_STATE_WEIGHTS = {
    "potion_decision": 1.5,
    "premature_end_turn": 3.0,
    "repeat_loop_entry": 2.5,
    "order_sensitive_play": 1.75,
}


def _combat_hard_state_tags(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    action: dict[str, Any] | None,
    repeat_count: int,
    turn_prefix: dict[str, Any] | None,
) -> list[str]:
    tags: list[str] = []
    if any(_combat_is_use_potion_action(candidate) for candidate in legal):
        tags.append("potion_decision")
    suspect_reasons = _combat_mcts_suspect_reasons(action=action, legal=legal, state=state)
    if any(reason.startswith("end_turn_") for reason in suspect_reasons):
        tags.append("premature_end_turn")
    if repeat_count >= 2:
        tags.append("repeat_loop_entry")
    play_card_options = sum(1 for candidate in legal if _combat_action_name(candidate) == "play_card")
    prefix_actions = _safe_int((turn_prefix or {}).get("action_count", 0), 0)
    if (
        _combat_action_name(action) == "play_card"
        and play_card_options >= 2
        and (prefix_actions > 0 or play_card_options >= 3)
    ):
        tags.append("order_sensitive_play")
    return tags


def _combat_hard_state_weight(tags: list[str]) -> float:
    weight = 1.0
    for tag in tags:
        weight = max(weight, float(_COMBAT_HARD_STATE_WEIGHTS.get(tag, 1.0)))
    return weight


def _combat_room_conditioned_continuation_loss(
    continuation_pred: torch.Tensor,
    continuation_target: torch.Tensor,
    room_type_onehot: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train continuation targets with explicit survival/cost semantics.

    Column layout:
    - [:, 0] win_prob
    - [:, 1] expected_hp_loss
    - [:, 2] expected_potion_cost
    """
    if room_type_onehot is None:
        room_type_onehot = torch.zeros(
            continuation_pred.shape[0], 3,
            dtype=continuation_pred.dtype, device=continuation_pred.device,
        )
        room_type_onehot[:, 0] = 1.0
    room_type_onehot = room_type_onehot.to(dtype=continuation_pred.dtype, device=continuation_pred.device)
    hallway = room_type_onehot[:, 0]
    elite = room_type_onehot[:, 1]
    boss = room_type_onehot[:, 2]

    survival_loss = F.binary_cross_entropy(
        continuation_pred[:, 0].clamp(1e-5, 1.0 - 1e-5),
        continuation_target[:, 0].clamp(0.0, 1.0),
        reduction="none",
    )
    hp_loss = F.smooth_l1_loss(
        continuation_pred[:, 1],
        continuation_target[:, 1],
        reduction="none",
    )
    potion_loss = F.smooth_l1_loss(
        continuation_pred[:, 2],
        continuation_target[:, 2],
        reduction="none",
    )

    survival_weight = hallway * 0.9 + elite * 1.0 + boss * 1.25
    hp_weight = hallway * 1.0 + elite * 0.75 + boss * 0.10
    potion_weight = hallway * 1.0 + elite * 0.80 + boss * 0.15

    total = (
        survival_weight * survival_loss
        + hp_weight * hp_loss
        + potion_weight * potion_loss
    ).mean()
    return total, survival_loss.mean(), hp_loss.mean(), potion_loss.mean()


def _resolve_counterfactual_runtime(
    use_segment_collector: bool,
    counterfactual_scoring: bool,
    counterfactual_weight: float,
) -> tuple[bool, float, list[str]]:
    """Resolve the actually effective counterfactual settings."""
    warnings: list[str] = []
    effective_scoring = bool(counterfactual_scoring and use_segment_collector)
    effective_weight = float(counterfactual_weight) if effective_scoring else 0.0

    if counterfactual_weight > 0 and not counterfactual_scoring:
        warnings.append(
            "counterfactual_weight > 0 but counterfactual_scoring is disabled; "
            "effective counterfactual weight is 0.0."
        )
    if counterfactual_scoring and not use_segment_collector:
        warnings.append(
            "counterfactual_scoring requires --use-segment-collector; legacy step-by-step PPO "
            "does not consume counterfactual reward, so effective counterfactual scoring is disabled."
        )

    return effective_scoring, effective_weight, warnings


def _configure_boss_aware_warmup(model: FullRunPolicyNetworkV2) -> tuple[int, int]:
    """Freeze PPO backbone and train only newly added boss-aware modules."""
    trainable_prefixes = (
        "entity_emb.text_token_embed",
        "boss_screen_adapter",
        "boss_readiness_head",
    )
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        allow = any(name.startswith(prefix) for prefix in trainable_prefixes)
        param.requires_grad = allow
        if allow:
            trainable_params += param.numel()
    return trainable_params, total_params


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


def _checkpoint_retrieval_proj_dim(ckpt: dict[str, Any]) -> int:
    """Infer retrieval proj dim from any supported checkpoint payload layout."""
    if not isinstance(ckpt, dict):
        return 0
    for key in ("ppo_model", "model_state_dict", "mcts_model"):
        proj_dim = _infer_retrieval_proj_dim(ckpt.get(key, {}))
        if proj_dim > 0:
            return proj_dim
    return 0


def _extract_deck_size(state: dict[str, Any]) -> int:
    player = _extract_player(state)
    deck = player.get("deck") if isinstance(player.get("deck"), list) else []
    return len(deck)


def _detect_combat_room_type(state_type: str, state: dict[str, Any]) -> str:
    st = str(state_type or "").strip().lower()
    if st in ("boss", "elite", "monster"):
        return st
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or []
    for enemy in enemies:
        etype = str((enemy or {}).get("type", "")).strip().lower()
        if etype.startswith("boss"):
            return "boss"
        if etype.startswith("elite"):
            return "elite"
    return "monster"


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


# ---------------------------------------------------------------------------
# MCTS replay buffer (same as train_combat_mcts.py)
# ---------------------------------------------------------------------------

@dataclass
class MCTSTrainingExample:
    state_features: dict[str, np.ndarray]
    action_features: dict[str, np.ndarray]
    mcts_policy: np.ndarray
    outcome: float


class MCTSReplayBuffer:
    def __init__(self, max_size: int = 50000):
        self.buffer: deque[MCTSTrainingExample] = deque(maxlen=max_size)

    def add(self, ex: MCTSTrainingExample):
        self.buffer.append(ex)

    def sample(self, n: int) -> list[MCTSTrainingExample]:
        idx = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Combat PPO rollout buffer (per-step data for PPO training of combat NN)
# ---------------------------------------------------------------------------

@dataclass
class CombatRolloutBuffer:
    """Lightweight buffer for combat PPO steps.

    Stores per-step combat data: state/action features, chosen action index,
    log_prob from sampling, per-step shaped reward, value estimate, done flag.
    GAE is computed before training.
    """

    state_features: list[dict[str, np.ndarray]] = field(default_factory=list)
    action_features: list[dict[str, np.ndarray]] = field(default_factory=list)
    action_indices: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    screen_types: list[str] = field(default_factory=list)  # encounter type per step
    sample_weights: list[float] = field(default_factory=list)
    hard_state_tags: list[list[str]] = field(default_factory=list)

    # Computed after collection
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def add(
        self,
        sf: dict[str, np.ndarray],
        af: dict[str, np.ndarray],
        action_idx: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        screen_type: str = "",
        sample_weight: float = 1.0,
        hard_state_tags: list[str] | None = None,
    ) -> None:
        self.state_features.append(sf)
        self.action_features.append(af)
        self.action_indices.append(action_idx)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.screen_types.append(screen_type)
        self.sample_weights.append(float(sample_weight))
        self.hard_state_tags.append(list(hard_state_tags or []))

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        """Compute GAE advantages and returns.

        Note: combat NN value head uses Tanh (output in [-1, 1]).
        GAE computation is standard — the bounded output just means
        value targets (returns) will naturally stay in a reasonable range.
        """
        n = len(self.rewards)
        self.advantages = [0.0] * n
        self.returns = [0.0] * n
        last_gae = 0.0

        for t in reversed(range(n)):
            if self.dones[t]:
                next_value = 0.0
                last_gae = 0.0
            elif t + 1 < n:
                next_value = self.values[t + 1]
            else:
                next_value = 0.0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * lam * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def to_tensors(self, device: torch.device | None = None) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Convert buffer to tensors for training."""
        n = len(self.rewards)

        # Stack state tensors
        state_tensors: dict[str, torch.Tensor] = {}
        if n > 0:
            for key in self.state_features[0]:
                arrays = [s[key] for s in self.state_features]
                arr = np.stack(arrays)
                if arr.dtype in (np.int64, np.int32):
                    state_tensors[key] = torch.tensor(arr, dtype=torch.long)
                elif arr.dtype == bool:
                    state_tensors[key] = torch.tensor(arr, dtype=torch.bool)
                else:
                    state_tensors[key] = torch.tensor(arr, dtype=torch.float32)

        # Stack action tensors
        action_tensors: dict[str, torch.Tensor] = {}
        if n > 0:
            for key in self.action_features[0]:
                arrays = [a[key] for a in self.action_features]
                arr = np.stack(arrays)
                if arr.dtype in (np.int64, np.int32):
                    action_tensors[key] = torch.tensor(arr, dtype=torch.long)
                elif arr.dtype == bool:
                    action_tensors[key] = torch.tensor(arr, dtype=torch.bool)
                else:
                    action_tensors[key] = torch.tensor(arr, dtype=torch.float32)

        result = {
            "state_tensors": state_tensors,
            "action_tensors": action_tensors,
            "actions": torch.tensor(self.action_indices, dtype=torch.long),
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "advantages": torch.tensor(self.advantages, dtype=torch.float32),
            "returns": torch.tensor(self.returns, dtype=torch.float32),
            "sample_weights": torch.tensor(self.sample_weights, dtype=torch.float32),
        }
        if device is not None:
            for k, v in result.items():
                if isinstance(v, dict):
                    result[k] = {kk: vv.to(device) for kk, vv in v.items()}
                else:
                    result[k] = v.to(device)
        return result

    def clear(self) -> None:
        for attr in ("state_features", "action_features", "action_indices",
                      "log_probs", "rewards", "values", "dones",
                      "advantages", "returns", "screen_types",
                      "sample_weights", "hard_state_tags"):
            getattr(self, attr).clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Combat PPO Trainer
# ---------------------------------------------------------------------------

class CombatPPOTrainer:
    """PPO update for the combat neural network.

    Uses the same clipped surrogate + GAE approach as PPOTrainerV2,
    adapted for the combat NN's input format (combat features, not structured state).
    """

    def __init__(
        self,
        network: CombatPolicyValueNetwork,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.05,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        target_kl: float = 0.0,
    ):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl

    def update(self, buffer: CombatRolloutBuffer) -> dict[str, float]:
        """Run PPO update on the combat buffer. Returns loss metrics."""
        buffer.compute_gae()
        device = next(self.network.parameters()).device
        data = buffer.to_tensors(device)

        state_tensors = data["state_tensors"]
        action_tensors = data["action_tensors"]
        old_actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]
        sample_weights = data["sample_weights"]

        # Normalize advantages
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        n = len(old_actions)
        total_ploss = 0.0
        total_vloss = 0.0
        total_entropy = 0.0
        total_ratio_mean = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        num_updates = 0
        early_stop = False

        for _epoch in range(self.ppo_epochs):
            indices = torch.randperm(n, device=device)
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                mb_idx = indices[start:end]

                # Slice minibatch
                mb_state = {k: v[mb_idx] for k, v in state_tensors.items()}
                mb_action = {k: v[mb_idx] for k, v in action_tensors.items()}
                mb_old_actions = old_actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_sample_weights = sample_weights[mb_idx]
                mb_sample_weights = mb_sample_weights / mb_sample_weights.mean().clamp_min(1e-6)

                # Forward
                logits, values = self.network(mb_state, mb_action)

                # Compute new log_probs from Categorical
                mask = mb_action["action_mask"].float()
                logits_masked = logits + (1.0 - mask) * (-1e9)
                dist = torch.distributions.Categorical(logits=logits_masked)
                new_log_probs = dist.log_prob(mb_old_actions)
                entropy = dist.entropy().mean()

                # PPO clipped ratio
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -(torch.min(surr1, surr2) * mb_sample_weights).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                # Value loss (clamp returns to [-1, 1] to match Tanh output)
                mb_returns_clamped = mb_returns.clamp(-1.0, 1.0)
                value_loss = F.mse_loss(values, mb_returns_clamped, reduction="none")
                value_loss = (value_loss * mb_sample_weights).mean()

                # Combined loss
                entropy = (dist.entropy() * mb_sample_weights).mean()
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                total_entropy += entropy.item()
                total_ratio_mean += ratio.mean().item()
                total_clip_fraction += clip_fraction.item()
                approx_kl = (mb_old_log_probs - new_log_probs).mean().abs()
                total_approx_kl += approx_kl.item()
                num_updates += 1

                if self.target_kl > 0 and approx_kl.item() > self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        num_updates = max(num_updates, 1)
        return {
            "combat_ppo_ploss": total_ploss / num_updates,
            "combat_ppo_vloss": total_vloss / num_updates,
            "combat_entropy": total_entropy / num_updates,
            "combat_ppo_ratio_mean": total_ratio_mean / num_updates,
            "combat_ppo_clip_fraction": total_clip_fraction / num_updates,
            "combat_ppo_approx_kl": total_approx_kl / num_updates,
            "combat_ppo_early_stop": float(early_stop),
        }


# ---------------------------------------------------------------------------
# Multi-process episode worker
# ---------------------------------------------------------------------------

def _mp_episode_worker(
    worker_id: int,
    port: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    worker_config: dict[str, Any],
    ppo_state_dict: dict[str, Any],
    mcts_state_dict: dict[str, Any],
):
    """Worker process: owns one env + one CPU model snapshot."""
    import logging
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger(f"worker-{worker_id}")
    from full_run_env import create_full_run_client
    ppo_ort_session = None
    combat_ort_session = None

    try:
        vocab = load_vocab()
        use_symbolic_features = bool(worker_config.get("use_symbolic_features", False))
        symbolic_proj_dim = int(worker_config.get("symbolic_proj_dim", 16))
        offline_mode = str(worker_config.get("offline_noncombat_ranking_head_mode", "mlp"))
        combat_main_path_mode = str(worker_config.get("combat_main_path_mode", "mlp"))
        deck_repr_dim = int(worker_config.get("deck_repr_dim", 0))
        residual_adapter = bool(worker_config.get("residual_adapter", False))

        ppo_net = FullRunPolicyNetworkV2(
            vocab=vocab,
            embed_dim=int(worker_config.get("embed_dim", 32)),
            use_symbolic_features=use_symbolic_features,
            symbolic_proj_dim=symbolic_proj_dim,
            offline_noncombat_ranking_head_mode=offline_mode,
        ).cpu().eval()
        mcts_net = CombatPolicyValueNetwork(
            vocab=vocab,
            embed_dim=int(worker_config.get("embed_dim", 32)),
            hidden_dim=int(worker_config.get("combat_hidden_dim", 128)),
            entity_embeddings=ppo_net.entity_emb,
            deck_repr_dim=deck_repr_dim,
            residual_adapter=residual_adapter,
            symbolic_head=ppo_net.symbolic_head,
        ).cpu().eval()
        _configure_offline_noncombat_ranking_head_mode(ppo_net, offline_mode)
        _configure_main_combat_path_mode(mcts_net, combat_main_path_mode)
        ppo_net.load_state_dict(ppo_state_dict, strict=False)
        mcts_net.load_state_dict(mcts_state_dict, strict=False)
        for param in ppo_net.parameters():
            param.requires_grad_(False)
        for param in mcts_net.parameters():
            param.requires_grad_(False)
        ppo_onnx_path = str(worker_config.get("ppo_onnx_path", "") or "").strip()
        if ppo_onnx_path:
            ppo_ort_session = _mp_worker_create_ort_session(
                ppo_onnx_path,
                worker_id=worker_id,
                label="PPO",
                log=log,
            )
        combat_onnx_path = str(worker_config.get("combat_onnx_path", "") or "").strip()
        if combat_onnx_path:
            combat_ort_session = _mp_worker_create_ort_session(
                combat_onnx_path,
                worker_id=worker_id,
                label="combat",
                log=log,
            )
        mcts_agent = CombatMCTSAgent(
            network=mcts_net,
            vocab=vocab,
            config=MCTSConfig(num_simulations=int(worker_config.get("mcts_sims", 50))),
            training=False,
            device=torch.device("cpu"),
            ppo_net=ppo_net,
            backend=str(worker_config.get("combat_mcts_backend", "python")),
            use_continuation_value=bool(worker_config.get("combat_mcts_continuation_value", False)),
        )
        client = create_full_run_client(
            port=port,
            use_pipe=True,
            transport=str(worker_config.get("transport", "pipe")),
            ready_timeout_s=15.0,
            auto_launch=False,
        )
        client._ensure_connected()
    except Exception as e:
        log.error("Worker %d bootstrap failed: %s", worker_id, e)
        result_queue.put((worker_id, None, None, {"error": f"connect: {e}"}))
        return

    while True:
        try:
            task = task_queue.get(timeout=5.0)
        except Exception:
            continue

        if task is None:  # shutdown sentinel
            break

        try:
            task_cmd = str(task.get("cmd", "collect_episode")) if isinstance(task, dict) else "collect_episode"
            if task_cmd == "refresh_weights":
                ppo_ort_session, combat_ort_session = _apply_mp_worker_refresh_message(
                    worker_id=worker_id,
                    log=log,
                    worker_config=worker_config,
                    ppo_net=ppo_net,
                    mcts_net=mcts_net,
                    refresh_task=task if isinstance(task, dict) else {},
                    ppo_ort_session=ppo_ort_session,
                    combat_ort_session=combat_ort_session,
                )
                result_queue.put({
                    "type": "refresh_ack",
                    "worker_id": worker_id,
                    "iteration": task.get("iteration") if isinstance(task, dict) else None,
                    "ok": True,
                })
                continue
            if task_cmd != "collect_episode":
                result_queue.put({
                    "type": "refresh_ack",
                    "worker_id": worker_id,
                    "iteration": task.get("iteration") if isinstance(task, dict) else None,
                    "ok": False,
                    "error": f"unknown_cmd:{task_cmd}",
                })
                continue

            episode_seed = task.get("seed") if isinstance(task, dict) else None
            ep_ppo, ep_mcts, ep_stats = collect_unified_episode(
                ppo_network=ppo_net,
                mcts_agent=mcts_agent,
                vocab=vocab,
                pipe=(lambda c=client: getattr(c, "_pipe", None)),
                client=client,
                character_id=str(worker_config.get("character_id", "IRONCLAD")),
                seed=episode_seed,
                episode_timeout=float(worker_config.get("episode_timeout", 90.0)),
                max_steps=int(worker_config.get("max_episode_steps", 600)),
                use_mcts_combat=bool(worker_config.get("mcts", False)),
                mcts_warmup_active=False,
                mcts_first_n_actions_per_turn=int(worker_config.get("mcts_first_n_actions_per_turn", 0)),
                mcts_full_search_on_elite_boss=bool(worker_config.get("mcts_full_search_on_elite_boss", True)),
                act1_no_elite_routes=bool(worker_config.get("act1_no_elite_routes", False)),
                combat_pending_stall_threshold=int(worker_config.get("combat_pending_stall_threshold", 100)),
                combat_buffer=CombatRolloutBuffer(),
                inference_client=None,
                use_segment_collector=bool(worker_config.get("use_segment_collector", False)),
                counterfactual_scoring=bool(worker_config.get("counterfactual_scoring", False)),
                counterfactual_weight=float(worker_config.get("counterfactual_weight", 0.0)),
                screen_local_delta=bool(worker_config.get("screen_local_delta", True)),
                deterministic_policy=bool(worker_config.get("deterministic_policy", False)),
                episode_saver=None,
                use_local_ort=False,
                ppo_ort_session=ppo_ort_session,
                combat_ort_session=combat_ort_session,
                boss_entry_quality_weight=float(worker_config.get("boss_entry_quality_weight", 0.0)),
                early_damage_potion_penalty_weight=float(worker_config.get("early_damage_potion_penalty_weight", 0.0)),
                boss_conditioned_card_guidance_weight=float(worker_config.get("boss_conditioned_card_guidance_weight", 0.0)),
                combat_safety_rerank_weight=float(worker_config.get("combat_safety_rerank_weight", 0.0)),
                build_mode=bool(worker_config.get("build_mode", False)),
                model_forward_lock=None,
            )
            result_queue.put((worker_id, ep_ppo, ep_mcts, ep_stats))
        except BaseException as e:
            if isinstance(task, dict) and str(task.get("cmd", "")) == "refresh_weights":
                result_queue.put({
                    "type": "refresh_ack",
                    "worker_id": worker_id,
                    "iteration": task.get("iteration"),
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
            log.warning("Worker %d episode failed: %s", worker_id, e)
            result_queue.put((worker_id, None, None, {"error": f"{type(e).__name__}: {e}"}))

    try:
        client.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Unified episode collection
# ---------------------------------------------------------------------------

def _export_ppo_actor_onnx(net, output_path: str, vocab) -> None:
    """Export PPO actor-only ONNX for ORT CPU inference."""
    import torch.nn as nn
    from core.rl_encoder_v2 import (SCALAR_DIM, MAX_DECK_SIZE, MAX_RELICS, MAX_POTIONS,
        MAX_HAND_SIZE, MAX_ENEMIES, MAX_MAP_NODES, MAX_CARD_REWARDS, MAX_SHOP_ITEMS,
        MAX_REST_OPTIONS, MAX_ACTIONS, CARD_AUX_DIM, ENEMY_AUX_DIM, NUM_RELIC_TAGS, MAP_ROUTE_DIM)

    class Wrapper(nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network
        def forward(self, scalars, deck_ids, deck_aux, deck_mask, relic_ids, relic_aux, relic_mask,
                    potion_ids, potion_mask, hand_ids, hand_aux, hand_mask,
                    enemy_ids, enemy_aux, enemy_mask,
                    screen_type_idx, next_boss_idx,
                    map_node_types, map_node_mask, map_route_features,
                    reward_card_ids, reward_card_aux, reward_card_mask,
                    shop_card_ids, shop_relic_ids, shop_potion_ids, shop_prices, shop_mask,
                    event_option_count, rest_option_ids, rest_option_mask,
                    action_type_ids, target_card_ids, target_enemy_ids,
                    target_node_types, target_indices, action_mask):
            ss = {'scalars': scalars, 'deck_ids': deck_ids, 'deck_aux': deck_aux,
                  'deck_mask': deck_mask.bool(), 'relic_ids': relic_ids, 'relic_aux': relic_aux,
                  'relic_mask': relic_mask.bool(), 'potion_ids': potion_ids,
                  'potion_mask': potion_mask.bool(), 'hand_ids': hand_ids, 'hand_aux': hand_aux,
                  'hand_mask': hand_mask.bool(), 'enemy_ids': enemy_ids, 'enemy_aux': enemy_aux,
                  'enemy_mask': enemy_mask.bool(), 'screen_type_idx': screen_type_idx.long(),
                  'next_boss_idx': next_boss_idx.long(),
                  'map_node_types': map_node_types, 'map_node_mask': map_node_mask.bool(),
                  'map_route_features': map_route_features,
                  'reward_card_ids': reward_card_ids, 'reward_card_aux': reward_card_aux,
                  'reward_card_mask': reward_card_mask.bool(),
                  'shop_card_ids': shop_card_ids, 'shop_relic_ids': shop_relic_ids,
                  'shop_potion_ids': shop_potion_ids, 'shop_prices': shop_prices,
                  'shop_mask': shop_mask.bool(),
                  'event_option_count': event_option_count.long(),
                  'rest_option_ids': rest_option_ids, 'rest_option_mask': rest_option_mask.bool()}
            sa = {'action_type_ids': action_type_ids, 'target_card_ids': target_card_ids,
                  'target_enemy_ids': target_enemy_ids, 'target_node_types': target_node_types,
                  'target_indices': target_indices, 'action_mask': action_mask.bool()}
            logits = self.network(ss, sa)[0]
            return logits

    wrapper = Wrapper(net).cpu().eval()
    B = 1
    inputs = [
        torch.randn(B, SCALAR_DIM), torch.zeros(B, MAX_DECK_SIZE, dtype=torch.int64),
        torch.zeros(B, MAX_DECK_SIZE, CARD_AUX_DIM), torch.ones(B, MAX_DECK_SIZE),
        torch.zeros(B, MAX_RELICS, dtype=torch.int64), torch.zeros(B, MAX_RELICS, NUM_RELIC_TAGS),
        torch.zeros(B, MAX_RELICS), torch.zeros(B, MAX_POTIONS, dtype=torch.int64),
        torch.zeros(B, MAX_POTIONS), torch.zeros(B, MAX_HAND_SIZE, dtype=torch.int64),
        torch.zeros(B, MAX_HAND_SIZE, CARD_AUX_DIM), torch.zeros(B, MAX_HAND_SIZE),
        torch.zeros(B, MAX_ENEMIES, dtype=torch.int64), torch.zeros(B, MAX_ENEMIES, ENEMY_AUX_DIM),
        torch.zeros(B, MAX_ENEMIES), torch.tensor([4], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64), torch.zeros(B, MAX_MAP_NODES, dtype=torch.int64),
        torch.ones(B, MAX_MAP_NODES), torch.zeros(B, MAX_MAP_NODES, MAP_ROUTE_DIM),
        torch.zeros(B, MAX_CARD_REWARDS, dtype=torch.int64), torch.zeros(B, MAX_CARD_REWARDS, CARD_AUX_DIM),
        torch.zeros(B, MAX_CARD_REWARDS), torch.zeros(B, MAX_SHOP_ITEMS, dtype=torch.int64),
        torch.zeros(B, MAX_SHOP_ITEMS, dtype=torch.int64), torch.zeros(B, MAX_SHOP_ITEMS, dtype=torch.int64),
        torch.zeros(B, MAX_SHOP_ITEMS), torch.zeros(B, MAX_SHOP_ITEMS),
        torch.tensor([0], dtype=torch.int64), torch.zeros(B, MAX_REST_OPTIONS, dtype=torch.int64),
        torch.zeros(B, MAX_REST_OPTIONS), torch.zeros(B, MAX_ACTIONS, dtype=torch.int64),
        torch.zeros(B, MAX_ACTIONS, dtype=torch.int64), torch.zeros(B, MAX_ACTIONS, dtype=torch.int64),
        torch.zeros(B, MAX_ACTIONS, dtype=torch.int64), torch.zeros(B, MAX_ACTIONS, dtype=torch.int64),
        torch.ones(B, MAX_ACTIONS),
    ]
    names = ['scalars', 'deck_ids', 'deck_aux', 'deck_mask', 'relic_ids', 'relic_aux', 'relic_mask',
             'potion_ids', 'potion_mask', 'hand_ids', 'hand_aux', 'hand_mask',
             'enemy_ids', 'enemy_aux', 'enemy_mask', 'screen_type_idx', 'next_boss_idx',
             'map_node_types', 'map_node_mask', 'map_route_features',
             'reward_card_ids', 'reward_card_aux', 'reward_card_mask',
             'shop_card_ids', 'shop_relic_ids', 'shop_potion_ids', 'shop_prices', 'shop_mask',
             'event_option_count', 'rest_option_ids', 'rest_option_mask',
             'action_type_ids', 'target_card_ids', 'target_enemy_ids',
             'target_node_types', 'target_indices', 'action_mask']
    torch.onnx.export(wrapper, tuple(inputs), output_path, input_names=names,
                      output_names=['policy_logits'], opset_version=17, do_constant_folding=True)


def collect_unified_episode(
    ppo_network: FullRunPolicyNetworkV2,
    mcts_agent: CombatMCTSAgent,
    vocab: Vocab,
    pipe,
    client,
    character_id: str = "IRONCLAD",
    ascension_level: int = 0,
    seed: str | None = None,
    episode_timeout: float = 90.0,
    max_steps: int = 600,
    use_mcts_combat: bool = False,
    mcts_warmup_active: bool = False,
    mcts_first_n_actions_per_turn: int = 0,
    mcts_full_search_on_elite_boss: bool = True,
    act1_no_elite_routes: bool = False,
    combat_buffer: CombatRolloutBuffer | None = None,
    inference_client=None,
    # Phase 2-4 options
    use_segment_collector: bool = False,
    counterfactual_scoring: bool = False,
    counterfactual_weight: float = 0.3,
    screen_local_delta: bool = True,
    deterministic_policy: bool = False,
    episode_saver: EpisodeDataSaver | None = None,
    use_local_ort: bool = False,
    ppo_ort_session=None,  # ORT CPU session for non-combat actor (Branch C)
    combat_ort_session=None,  # ORT CPU session for combat actor/value
    # Step 2 / Phase 5 options
    boss_entry_quality_weight: float = 0.0,
    early_damage_potion_penalty_weight: float = 0.0,
    boss_conditioned_card_guidance_weight: float = 0.0,
    combat_safety_rerank_weight: float = 0.0,
    # Build mode
    build_mode: bool = False,
    combat_pending_stall_threshold: int = 100,
    model_forward_lock: threading.Lock | None = None,
) -> tuple[StructuredRolloutBuffer, list[MCTSTrainingExample], dict]:
    """Collect one episode with PPO for non-combat and MCTS/PPO for combat.

    Args:
        combat_buffer: If provided, NN combat steps are collected for PPO training.
            Only NN episodes (use_random_this_step=False) contribute PPO data.
            If None, a local buffer is created (returned via stats["_combat_buffer"]).

    Returns:
        ppo_buffer: non-combat steps for PPO training
        mcts_examples: combat decisions for behavior cloning training (outcomes backfilled)
        stats: episode statistics (includes "_combat_buffer" key for merging)
    """

    ppo_buffer = StructuredRolloutBuffer()
    segment_buffer = SegmentRolloutBuffer() if use_segment_collector else None
    segment_collector = NonCombatSegmentCollector() if use_segment_collector else None
    mcts_pending: list[dict] = []  # outcome=0 until episode end
    if combat_buffer is None:
        combat_buffer = CombatRolloutBuffer()
    stats: dict[str, Any] = {
        "floors": 0, "combats": 0, "combats_won": 0, "ppo_steps": 0,
        "mcts_decisions": 0, "mcts_searches": 0, "combat_ppo_steps": 0,
        "combat_random_warmup_steps": 0,
        "combat_mcts_turn_limited_steps": 0,
        "wait_steps": 0,
        "combat_pending_steps": 0,
        "combat_pending_wait_steps": 0,
        "combat_pending_refresh_steps": 0,
        "combat_pending_stall": False,
        "repeat_max": 0,
        "hard_state_potion_decision_steps": 0,
        "hard_state_premature_end_turn_steps": 0,
        "hard_state_repeat_loop_steps": 0,
        "hard_state_order_sensitive_steps": 0,
        "combat_hard_state_weight_sum": 0.0,
        "outcome": None, "error": None, "end_reason": None,
        "cards_taken": [], "cards_skipped": 0,
        "card_reward_screens": 0, "card_reward_skips": 0,
        "hp_timeline": [],  # HP after each combat
        "death_enemy": None,
        "boss_reached": False,
        "act1_cleared": False,
        "boss_hp_fraction_dealt": [],
        "deck_size_at_boss": [],
    }
    def _mark_boss_reached() -> None:
        stats["boss_reached"] = True

    def _mark_act1_cleared() -> None:
        # Act1 clear is a strict superset of boss reach.
        stats["act1_cleared"] = True
        _mark_boss_reached()
    # Per-step timing diagnostics
    _slow_step_threshold = 1.0  # seconds
    _slow_steps = 0
    _max_step_ms = 0.0
    _timeout_count = 0
    _pipe_time = 0.0
    _pipe_calls = 0
    _feature_time = 0.0
    _feature_calls = 0
    _inference_time = 0.0
    _inference_calls = 0
    _buffer_time = 0.0
    _buffer_calls = 0
    # Episode trace for debugging (kept lightweight — only key events)
    _episode_trace: list[str] = []
    _episode_trace_zh: list[str] = []
    _episode_summary: dict[str, Any] = {
        "seed": seed,
        "character_id": character_id,
        "max_steps": max_steps,
        "route_choices": [],
        "event_choices": [],
        "card_rewards": [],
        "shop_sessions": [],
        "rest_sessions": [],
        "combats": [],
        "auto_actions": [],
        "combat_pending_spans": [],
        "combat_pending_debug": [],
        "counters": {
            "wait_steps": 0,
            "combat_pending_steps": 0,
            "combat_pending_wait_steps": 0,
            "combat_pending_refresh_steps": 0,
            "empty_legal_recovery_steps": 0,
            "auto_progress_steps": 0,
            "event_refresh_after_empty_legal_steps": 0,
            "map_override_steps": 0,
            "repeat_escape_steps": 0,
        },
    }
    _current_shop_session: dict[str, Any] | None = None
    _current_rest_session: dict[str, Any] | None = None
    _current_combat_summary: dict[str, Any] | None = None
    _combat_pending_streak = 0
    _combat_pending_start_step: int | None = None

    def _flush_shop_session() -> None:
        nonlocal _current_shop_session
        if _current_shop_session is not None:
            _episode_summary["shop_sessions"].append(_current_shop_session)
        _current_shop_session = None

    def _flush_rest_session() -> None:
        nonlocal _current_rest_session
        if _current_rest_session is not None:
            _episode_summary["rest_sessions"].append(_current_rest_session)
        _current_rest_session = None

    def _flush_combat_pending_span(end_step: int, end_state_type: str) -> None:
        nonlocal _combat_pending_streak, _combat_pending_start_step
        if _combat_pending_streak <= 0:
            return
        _episode_summary["combat_pending_spans"].append(
            {
                "start_step": _combat_pending_start_step,
                "end_step": end_step,
                "count": _combat_pending_streak,
                "end_state_type": end_state_type,
            }
        )
        _combat_pending_streak = 0
        _combat_pending_start_step = None

    def _finalize_current_combat(
        post_state: dict[str, Any],
        *,
        won: bool,
        end_reason: str,
        end_step: int,
    ) -> None:
        nonlocal _current_combat_summary
        if _current_combat_summary is None:
            return
        player = _extract_player(post_state)
        _current_combat_summary["won"] = bool(won)
        _current_combat_summary["end_reason"] = end_reason
        _current_combat_summary["end_step"] = end_step
        _current_combat_summary["end_state_type"] = _lower_text(post_state.get("state_type"))
        _current_combat_summary["end_hp"] = _safe_int(player.get("hp", player.get("current_hp", 0)), 0)
        _current_combat_summary["end_max_hp"] = _safe_int(player.get("max_hp", 0), 0)
        _current_combat_summary["end_floor"] = _safe_int((post_state.get("run") or {}).get("floor", 0), 0)
        _current_combat_summary["action_count"] = len(_current_combat_summary.get("actions") or [])
        _current_combat_summary["last_intent"] = _combat_enemy_intent_summary(post_state)
        _current_combat_summary["last_intent_zh"] = _combat_enemy_intent_summary_zh(post_state)
        _current_combat_summary = None

    _last_action_key = ""
    _repeat_count = 0
    _MAX_REPEATS = 20  # bail out if same action repeated this many times
    _last_action_name = ""
    _last_reward_claim_sig = ""
    _last_reward_claim_count: int | None = None
    _reward_chain_card_reward_seen = False
    _combat_round_no: int | None = None
    _combat_turn_action_index = 0

    episode_start = time.monotonic()

    try:
        _t0 = time.monotonic()
        state = client.reset(character_id=character_id, ascension_level=ascension_level, seed=seed)
        _dt = time.monotonic() - _t0
        _pipe_time += _dt
        _pipe_calls += 1
        _max_step_ms = max(_max_step_ms, _dt * 1000)
        if _dt >= _slow_step_threshold:
            _slow_steps += 1
            logger.debug("Slow reset: %.0fms", _dt * 1000)
    except Exception as e:
        # Try reconnect + retry once
        try:
            if hasattr(client, "_reconnect"):
                client._reconnect()
            state = client.reset(character_id=character_id, ascension_level=ascension_level, seed=seed)
        except Exception as e2:
            stats["error"] = f"reset: {e2}"
            stats["end_reason"] = "error"
            return ppo_buffer, [], stats

    prev_state = state
    _resolved_seed = str(seed or ((state.get("run") or {}).get("seed") or "")).strip()
    if _resolved_seed:
        _episode_summary["seed"] = _resolved_seed
    in_combat = False
    _prev_combat_state = state  # init; updated when entering/during combat
    _hp_at_combat_start = 80  # init; updated when entering combat
    _combat_room_type = "monster"
    _boss_hp_frac_peak = 0.0
    _combat_turn_prefix = _new_combat_turn_prefix()
    _pending_boss_deck_size: int | None = None
    _combat_ppo_pending = None  # pending combat PPO data awaiting next_state
    _has_entered_run = _lower_text(state.get("state_type")) != "menu"

    for step_i in range(max_steps):
        if time.monotonic() - episode_start > episode_timeout:
            stats["error"] = "timeout"
            stats["end_reason"] = "timeout"
            break
        if _shutdown_requested:
            break

        st = (state.get("state_type") or "").lower()
        run = state.get("run") or {}
        current_floor = _safe_int(run.get("floor", 0), 0)
        if st != "menu":
            _has_entered_run = True
        if _is_combat_pending_state(st):
            stats["combat_pending_steps"] = _safe_int(stats.get("combat_pending_steps", 0), 0) + 1
            _episode_summary["counters"]["combat_pending_steps"] += 1
            if _combat_pending_streak == 0:
                _combat_pending_start_step = step_i
            _combat_pending_streak += 1
            if _combat_pending_streak in (5, 20, 100):
                _episode_trace.append(
                    f"[{step_i}] combat_pending: stall x{_combat_pending_streak} floor={current_floor}"
                )
            if combat_pending_stall_threshold > 0 and _combat_pending_streak >= combat_pending_stall_threshold:
                stats["combat_pending_stall"] = True
                stats["end_reason"] = "combat_pending_stall"
                _episode_trace.append(
                    f"[END] combat_pending_stall reached (count={_combat_pending_streak}, floor={current_floor}, st={st})"
                )
                break
        else:
            _flush_combat_pending_span(step_i - 1, st)
        if st == "card_reward":
            _reward_chain_card_reward_seen = True
        elif st not in REWARD_FLOW_SCREENS:
            _reward_chain_card_reward_seen = False
            _last_reward_claim_count = None

        # Terminal
        if _is_episode_terminal_state(state, has_entered_run=_has_entered_run):
            terminal_value = _terminal_value_from_outcome(state, default_on_unknown=-1.0)
            stats["outcome"] = RUN_OUTCOME_VICTORY if terminal_value > 0 else RUN_OUTCOME_DEATH
            stats["end_reason"] = "terminal"
            _p = (state.get("player") or {})
            # Track death info
            if terminal_value < 0:
                _battle = state.get("battle") or {}
                _enemies = _battle.get("enemies") or []
                if _enemies:
                    stats["death_enemy"] = "+".join(
                        [e.get("id", "?")[:12] for e in _enemies[:4]])
            _episode_trace.append(
                f"[{step_i}] TERMINAL: {stats['outcome']} floor={stats['floors']} "
                f"hp={_p.get('hp',0)}/{_p.get('max_hp',0)} death_by={stats.get('death_enemy','N/A')}")

            # Final PPO step (done=True)
            reward = shaped_reward(
                prev_state, state, terminal_value, done=True,
                boss_entry_quality_weight=boss_entry_quality_weight,
                early_damage_potion_penalty_weight=early_damage_potion_penalty_weight,
            )

            if segment_collector is not None and segment_collector.is_open and segment_buffer is not None:
                # Phase 2: accumulate terminal reward and close segment
                segment_collector.add_reward(reward, tag="terminal", steps=1)
                seg = segment_collector.close_segment(done=True)
                if seg is not None:
                    segment_buffer.add(seg)
            elif len(ppo_buffer) > 0:
                # Legacy: mark last step done
                ppo_buffer.dones[-1] = True
                ppo_buffer.rewards[-1] = reward

            # Backfill MCTS outcomes
            for ex_data in mcts_pending:
                ex_data["outcome"] = terminal_value

            # Combat death feedback → non-combat decisions
            if in_combat and terminal_value < 0:
                # Determine room type and boss damage for shaped feedback
                _death_room_type = _combat_room_type or "monster"
                _boss_hp_frac_dealt = 0.0
                if _death_room_type == "boss":
                    _boss_hp_frac_dealt = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))
                    _mark_boss_reached()
                    stats["boss_hp_fraction_dealt"].append(_boss_hp_frac_dealt)
                _feedback = compute_combat_feedback(
                    _hp_at_combat_start, 0, _safe_int(_extract_player(state).get("max_hp", 80)),
                    won=False, room_type=_death_room_type,
                    boss_hp_fraction_dealt=_boss_hp_frac_dealt)
                if segment_collector is not None and segment_collector.is_open and segment_buffer is not None:
                    segment_collector.add_reward(_feedback, tag="fight_death", steps=1)
                    seg = segment_collector.close_segment(done=True)
                    if seg is not None:
                        segment_buffer.add(seg)
                elif len(ppo_buffer) > 0:
                    _fb_steps = min(10, len(ppo_buffer))
                    for _fi in range(len(ppo_buffer) - _fb_steps, len(ppo_buffer)):
                        ppo_buffer.rewards[_fi] += _feedback / max(1, _fb_steps)

            # Flush any pending combat PPO step before termination
            if in_combat and _combat_ppo_pending is not None and combat_buffer is not None:
                combat_won = terminal_value > 0
                # Partial credit for boss damage dealt on terminal loss (see rl_reward_shaping.combat_terminal_reward)
                _br_flush = 0.0
                if (not combat_won) and _combat_room_type == "boss":
                    _br_flush = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))
                step_reward = combat_step_reward(
                    _prev_combat_state, state, combat_won=combat_won,
                    hp_at_combat_start=_hp_at_combat_start,
                    boss_damage_ratio=_br_flush)
                step_reward += combat_local_tactical_reward(
                    _prev_combat_state,
                    action,
                    legal,
                )
                combat_buffer.add(
                    sf=_combat_ppo_pending["sf"],
                    af=_combat_ppo_pending["af"],
                    action_idx=_combat_ppo_pending["action_idx"],
                    log_prob=_combat_ppo_pending["log_prob"],
                    reward=step_reward,
                    value=_combat_ppo_pending["value"],
                    done=True,
                    screen_type=_combat_ppo_pending.get("screen_type", ""),
                    sample_weight=_combat_ppo_pending.get("sample_weight", 1.0),
                    hard_state_tags=_combat_ppo_pending.get("hard_state_tags"),
                )
                stats["combat_ppo_steps"] += 1
                _combat_ppo_pending = None
            # Mark last combat PPO step as done (death during combat)
            elif combat_buffer is not None and len(combat_buffer) > 0 and in_combat:
                combat_buffer.dones[-1] = True
                combat_won = terminal_value > 0
                _br_last = 0.0
                if (not combat_won) and _combat_room_type == "boss":
                    _br_last = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))
                combat_buffer.rewards[-1] = combat_step_reward(
                    prev_state, state, combat_won=combat_won,
                    hp_at_combat_start=_hp_at_combat_start,
                    boss_damage_ratio=_br_last)
            if in_combat:
                _finalize_current_combat(
                    state,
                    won=terminal_value > 0,
                    end_reason="terminal",
                    end_step=step_i,
                )
            _flush_shop_session()
            _flush_rest_session()
            _flush_combat_pending_span(step_i, st)
            break

        run = state.get("run") or {}
        stats["floors"] = max(stats["floors"], int(run.get("floor", 0)))
        if _safe_int(run.get("act", 1), 1) > 1:
            _mark_act1_cleared()

        legal = state.get("legal_actions", [])
        legal = [a for a in legal if isinstance(a, dict) and a.get("is_enabled") is not False]
        _current_reward_claim_count = _claim_reward_action_count(legal) if st == "combat_rewards" else None

        if not legal and st == "event":
            try:
                refreshed_state = client.get_state()
            except Exception:
                refreshed_state = None
            if isinstance(refreshed_state, dict):
                refreshed_legal = refreshed_state.get("legal_actions", [])
                refreshed_legal = [
                    a for a in refreshed_legal
                    if isinstance(a, dict) and a.get("is_enabled") is not False
                ]
                refreshed_event = refreshed_state.get("event") or {}
                original_event = state.get("event") or {}
                if (
                    refreshed_state.get("state_type") != st
                    or refreshed_legal
                    or refreshed_event != original_event
                ):
                    _episode_trace.append(f"[{step_i}] event: refresh-after-empty-legal")
                    _episode_summary["counters"]["event_refresh_after_empty_legal_steps"] += 1
                    _episode_summary["auto_actions"].append(
                        {
                            "step": step_i,
                            "state_type": st,
                            "kind": "event_refresh_after_empty_legal",
                            "floor": current_floor,
                        }
                    )
                    state = refreshed_state
                    continue

        if not legal:
            if _is_combat_pending_state(st):
                try:
                    if _current_reward_claim_count is not None:
                        _last_reward_claim_count = _current_reward_claim_count
                    state, pending_method = _advance_combat_pending_transition(
                        client,
                        _combat_pending_streak,
                    )
                    if pending_method == "refresh":
                        stats["combat_pending_refresh_steps"] = _safe_int(stats.get("combat_pending_refresh_steps", 0), 0) + 1
                        _episode_summary["counters"]["combat_pending_refresh_steps"] += 1
                    else:
                        stats["wait_steps"] = _safe_int(stats.get("wait_steps", 0), 0) + 1
                        _episode_summary["counters"]["wait_steps"] += 1
                        stats["combat_pending_wait_steps"] = _safe_int(stats.get("combat_pending_wait_steps", 0), 0) + 1
                        _episode_summary["counters"]["combat_pending_wait_steps"] += 1

                    next_st = str((state.get("state_type") or "")).lower()
                    next_legal = state.get("legal_actions", [])
                    next_legal = [a for a in next_legal if isinstance(a, dict) and a.get("is_enabled") is not False]
                    method_counter = (
                        _episode_summary["counters"]["combat_pending_refresh_steps"]
                        if pending_method == "refresh"
                        else _episode_summary["counters"]["combat_pending_wait_steps"]
                    )
                    debug_entry = {
                        "step": step_i,
                        "floor": current_floor,
                        "streak": _combat_pending_streak,
                        "method": pending_method,
                        "next_state_type": next_st,
                        "next_legal_count": len(next_legal),
                        "terminal": bool(_is_episode_terminal_state(state, has_entered_run=_has_entered_run)),
                    }
                    if len(_episode_summary["combat_pending_debug"]) < 24:
                        _episode_summary["combat_pending_debug"].append(debug_entry)
                    if method_counter in (1, 5, 20, 100) or next_st != st or next_legal:
                        _episode_trace.append(
                            f"[{step_i}] combat_pending: {pending_method} x{method_counter} "
                            f"floor={current_floor} -> st={next_st or '?'} legal={len(next_legal)} "
                            f"terminal={int(debug_entry['terminal'])}"
                        )
                        _episode_trace_zh.append(
                            f"[第{step_i}步][过渡] combat_pending: {pending_method} x{method_counter} "
                            f"楼层={current_floor} -> 状态={next_st or '?'} 可行动作={len(next_legal)} "
                            f"终局={int(debug_entry['terminal'])}"
                        )
                    _last_reward_claim_sig = ""
                except Exception:
                    break
                continue
            recovery_action = _choose_empty_legal_recovery_action(state, _last_action_name)
            if recovery_action is not None:
                _episode_trace.append(
                    f"[{step_i}] {st}: empty-legal recovery via {recovery_action.get('action','?')}"
                )
                _episode_summary["counters"]["empty_legal_recovery_steps"] += 1
                _episode_summary["auto_actions"].append(
                    {
                        "step": step_i,
                        "state_type": st,
                        "kind": "empty_legal_recovery",
                        "action": str(recovery_action.get("action") or ""),
                        "label": str(recovery_action.get("label") or recovery_action.get("action") or ""),
                        "floor": current_floor,
                    }
                )
                try:
                    prev_state = state
                    state = client.act(recovery_action)
                    _last_action_name = str(recovery_action.get("action") or "").strip().lower()
                    _last_reward_claim_sig = _next_reward_claim_signature(st, prev_state, recovery_action)
                except Exception:
                    break
                continue

        auto_action = _choose_auto_progress_action(
            state,
            legal,
            _last_action_name,
            _last_reward_claim_sig,
            _last_reward_claim_count,
            _reward_chain_card_reward_seen,
        )
        if auto_action is not None:
            auto_name = str(auto_action.get("action") or "?")
            auto_label = str(auto_action.get("label") or auto_name)
            _episode_trace.append(f"[{step_i}] {st}: auto-progress {auto_name} ({auto_label})")
            _episode_summary["counters"]["auto_progress_steps"] += 1
            _episode_summary["auto_actions"].append(
                {
                    "step": step_i,
                    "state_type": st,
                    "kind": "auto_progress",
                    "action": auto_name,
                    "label": auto_label,
                    "floor": current_floor,
                }
            )
            try:
                if _current_reward_claim_count is not None:
                    _last_reward_claim_count = _current_reward_claim_count
                prev_state = state
                state = client.act(auto_action)
                _last_action_name = auto_name.strip().lower()
                _last_reward_claim_sig = _next_reward_claim_signature(st, prev_state, auto_action)
            except Exception:
                break
            continue

        legal = state.get("legal_actions", [])
        legal = [a for a in legal if isinstance(a, dict) and a.get("is_enabled") is not False]

        if not legal:
            # No actions available — wait
            try:
                if _current_reward_claim_count is not None:
                    _last_reward_claim_count = _current_reward_claim_count
                stats["wait_steps"] = _safe_int(stats.get("wait_steps", 0), 0) + 1
                _episode_summary["counters"]["wait_steps"] += 1
                if _is_combat_pending_state(st):
                    stats["combat_pending_wait_steps"] = _safe_int(stats.get("combat_pending_wait_steps", 0), 0) + 1
                    _episode_summary["counters"]["combat_pending_wait_steps"] += 1
                    if _episode_summary["counters"]["combat_pending_wait_steps"] in (1, 5, 20, 100):
                        _episode_trace.append(
                            f"[{step_i}] combat_pending: wait x{_episode_summary['counters']['combat_pending_wait_steps']} floor={current_floor}"
                        )
                state = client.act({"action": "wait"})
                _last_reward_claim_sig = ""
            except Exception:
                break
            continue

        # --- Auto-handle selection flows (select → confirm) ---
        # Detect by BOTH state_type AND legal_action content, because C# can
        # return select/confirm actions even under "elite"/"monster" state_type.
        _legal_action_names = {a.get("action", "") for a in legal}
        _is_selection = (
            st in ("card_select", "hand_select", "relic_select")
            or _legal_action_names & {"select_card", "combat_select_card",
                                      "combat_confirm_selection", "confirm_selection",
                                      "cancel_selection", "skip_relic_selection"}
        )
        if _is_selection:
            confirm = [a for a in legal if "confirm" in a.get("action", "") or "skip" in a.get("action", "")]
            if confirm:
                _episode_trace.append(f"[{step_i}] {st}: auto-confirm ({confirm[0].get('action')})")
                _episode_summary["auto_actions"].append(
                    {
                        "step": step_i,
                        "state_type": st,
                        "kind": "auto_confirm",
                        "action": str(confirm[0].get("action") or ""),
                        "label": str(confirm[0].get("label") or confirm[0].get("action") or ""),
                        "floor": current_floor,
                    }
                )
                try:
                    state = client.act(confirm[0])
                    _last_reward_claim_sig = ""
                except Exception:
                    break
                continue
            _shop_remove_target_override = None
            if (
                _current_shop_session is not None
                and (_current_shop_session.get("actions") or [])
                and str((_current_shop_session.get("actions") or [])[-1].get("action") or "").strip().lower() == "remove_card"
            ):
                _shop_remove_target_override = _choose_shop_remove_target_action(legal)
            select = [a for a in legal if "select" in a.get("action", "")]
            if select:
                chosen_select = select[0]
                _auto_select_kind = "auto_select"
                if _shop_remove_target_override is not None:
                    _, chosen_select, _auto_select_kind = _shop_remove_target_override
                _episode_trace.append(f"[{step_i}] {st}: auto-select {chosen_select.get('label','?')}")
                _auto_select_entry = {
                    "step": step_i,
                    "state_type": st,
                    "kind": _auto_select_kind,
                    "action": str(chosen_select.get("action") or ""),
                    "label": str(chosen_select.get("label") or chosen_select.get("action") or ""),
                    "floor": current_floor,
                }
                _episode_summary["auto_actions"].append(
                    _auto_select_entry
                )
                if (
                    _current_shop_session is not None
                    and (_current_shop_session.get("actions") or [])
                    and str((_current_shop_session.get("actions") or [])[-1].get("action") or "").strip().lower() == "remove_card"
                ):
                    _shop_remove_entry = dict(_auto_select_entry)
                    _shop_remove_entry["action"] = "remove_target"
                    _current_shop_session["actions"].append(_shop_remove_entry)
                try:
                    state = client.act(chosen_select)
                    _last_reward_claim_sig = ""
                except Exception:
                    break
                continue

        # --- Auto-claim combat rewards (gold/potions always worth taking) ---
        # PPO decides card selection in card_reward screen. Here we just auto-claim
        # non-card rewards and auto-proceed to reach the card_reward screen.
        if st == "combat_rewards":
            claim = [a for a in legal if a.get("action") == "claim_reward"]
            if claim:
                _episode_trace.append(f"[{step_i}] combat_rewards: auto-claim {claim[0].get('label','?')}")
                _episode_summary["auto_actions"].append(
                    {
                        "step": step_i,
                        "state_type": st,
                        "kind": "combat_reward_claim",
                        "action": "claim_reward",
                        "label": str(claim[0].get("label") or "claim_reward"),
                        "floor": current_floor,
                    }
                )
                try:
                    _t0 = time.monotonic()
                    state = client.act(claim[0])
                    _pipe_time += time.monotonic() - _t0
                    _pipe_calls += 1
                except Exception:
                    break
                continue
            # All items claimed — auto-proceed to card_reward screen.
            proceed = [a for a in legal if a.get("action") in ("proceed", "skip")]
            if proceed:
                _episode_trace.append(f"[{step_i}] combat_rewards: auto-proceed to card_reward")
                _episode_summary["auto_actions"].append(
                    {
                        "step": step_i,
                        "state_type": st,
                        "kind": "combat_reward_proceed",
                        "action": str(proceed[0].get("action") or ""),
                        "label": str(proceed[0].get("label") or proceed[0].get("action") or ""),
                        "floor": current_floor,
                    }
                )
                try:
                    _t0 = time.monotonic()
                    state = client.act(proceed[0])
                    _pipe_time += time.monotonic() - _t0
                    _pipe_calls += 1
                except Exception:
                    break
                continue

        # --- Repeat-action detection: try different action before abort ---
        _cur_action_key = f"{st}:{len(legal)}"
        if _cur_action_key == _last_action_key:
            _repeat_count += 1
            if _repeat_count >= 3:
                rest_escape = _choose_rest_site_repeat_escape_action(state, legal)
                if rest_escape is not None:
                    chosen = rest_escape
                else:
                # Try 1: proceed/skip
                    escape = [a for a in legal if a.get("action") in ("proceed", "skip", "cancel_selection")]
                    if not escape:
                        # Try 2: pick a different action than what PPO keeps choosing
                        import random as _rng
                        escape = [a for a in legal if a != legal[0]]  # anything different
                        if not escape:
                            escape = legal  # only one option, stuck
                    chosen = _rng.choice(escape) if len(escape) > 1 else escape[0]
                _episode_trace.append(f"[{step_i}] REPEAT x{_repeat_count}: escape via {chosen.get('action','?')}: {chosen.get('label','?')}")
                stats["repeat_max"] = max(_safe_int(stats.get("repeat_max", 0), 0), _repeat_count)
                _episode_summary["counters"]["repeat_escape_steps"] += 1
                if _current_combat_summary is not None:
                    _current_combat_summary["repeat_hits"] = max(
                        _safe_int(_current_combat_summary.get("repeat_hits", 0), 0),
                        _repeat_count,
                    )
                _episode_summary["auto_actions"].append(
                    {
                        "step": step_i,
                        "state_type": st,
                        "kind": "repeat_escape",
                        "action": str(chosen.get("action") or ""),
                        "label": str(chosen.get("label") or chosen.get("action") or ""),
                        "repeat_count": _repeat_count,
                        "floor": current_floor,
                    }
                )
                try:
                    if _current_reward_claim_count is not None:
                        _last_reward_claim_count = _current_reward_claim_count
                    _t0 = time.monotonic()
                    state = client.act(chosen)
                    _pipe_time += time.monotonic() - _t0
                    _pipe_calls += 1
                    _last_reward_claim_sig = _next_reward_claim_signature(st, state, chosen)
                except Exception:
                    break
                if _repeat_count >= _MAX_REPEATS:
                    _episode_trace.append(f"[{step_i}] ABORT: still stuck after {_repeat_count} repeats")
                    stats["error"] = f"repeat_loop:{st}"
                    stats["end_reason"] = "repeat_loop"
                    break
                continue
        else:
            _repeat_count = 0
            _last_action_key = _cur_action_key

        # ----- BUILD MODE: skip non-boss combat instantly -----
        if build_mode and st in COMBAT_SCREENS:
            _combat_room_type = _detect_combat_room_type(st, state)
            if _pending_boss_deck_size is not None:
                _combat_room_type = "boss"

            if _combat_room_type != "boss":
                # Non-boss combat: skip instantly via SkipCombat opcode
                if not in_combat:
                    in_combat = True
                    stats["combats"] += 1

                raw_pipe = pipe() if callable(pipe) else pipe
                try:
                    _t0 = time.monotonic()
                    result = raw_pipe.call("skip_combat")
                    _pipe_time += time.monotonic() - _t0
                    _pipe_calls += 1
                    post_state = result.get("state", result)
                    post_st = (post_state.get("state_type") or "").lower()
                    # Give small positive reward for surviving
                    if len(ppo_buffer) > 0:
                        ppo_buffer.rewards[-1] += 0.05
                    prev_state = state
                    state = post_state
                    in_combat = False
                    stats.setdefault("build_mode_skips", 0)
                    stats["build_mode_skips"] += 1
                    _episode_trace.append(
                        f"[{step_i}] BUILD_MODE_SKIP #{stats['combats']} {_combat_room_type}")
                    continue
                except Exception as skip_err:
                    _episode_trace.append(f"[{step_i}] BUILD_MODE_SKIP_ERROR: {skip_err}")
                    # Fall through to normal combat path

        # ----- COMBAT: LOCAL ORT (skip per-step Python inference) -----
        if st in COMBAT_SCREENS and use_local_ort and pipe is not None:
            if not in_combat:
                in_combat = True
                stats["combats"] += 1
                _combat_room_type = _detect_combat_room_type(st, state)
                if _pending_boss_deck_size is not None:
                    _combat_room_type = "boss"
                player = state.get("player") or {}
                _hp_at_combat_start = int(player.get("hp", player.get("current_hp", 0)) or 0)
                if _combat_room_type == "boss":
                    _mark_boss_reached()
                    stats["deck_size_at_boss"].append(
                        _pending_boss_deck_size if _pending_boss_deck_size is not None
                        else _extract_deck_size(state))
                    _pending_boss_deck_size = None

            # Run entire combat in C# with local ORT actor
            raw_pipe = pipe() if callable(pipe) else pipe
            try:
                _ort_t0 = time.monotonic()
                result = raw_pipe.call("run_combat_local", {"max_steps": 600})
                combat_steps = result.get("combat_steps", 0)
                post_state = result.get("state", result)
                _ort_elapsed = time.monotonic() - _ort_t0
                _pipe_time += _ort_elapsed
                _pipe_calls += 1
                stats.setdefault("_ort_combat_time", 0.0)
                stats["_ort_combat_time"] += _ort_elapsed
                stats.setdefault("_ort_combat_calls", 0)
                stats["_ort_combat_calls"] += 1
                mcts_decisions += combat_steps
                combat_ppo_steps += combat_steps

                # Compute fight summary reward for non-combat PPO
                post_player = post_state.get("player") or {}
                _hp_now = int(post_player.get("hp", post_player.get("current_hp", 0)) or 0)
                _max_hp = int(post_player.get("max_hp", 80) or 80)
                post_st = (post_state.get("state_type") or "").lower()
                won = post_st not in COMBAT_SCREENS and post_st != "game_over"
                _boss_hp_frac = 0.0
                if _combat_room_type == "boss":
                    _boss_hp_frac = _estimate_boss_hp_fraction(post_state if not won else state)
                    stats["boss_hp_fracs"].append(max(_boss_hp_frac, _boss_hp_frac_peak if '_boss_hp_frac_peak' in dir() else 0))
                _feedback = compute_combat_feedback(
                    _hp_at_combat_start, _hp_now if won else 0, _max_hp,
                    won=won, room_type=_combat_room_type,
                    boss_hp_fraction_dealt=_boss_hp_frac)
                if len(ppo_buffer) > 0:
                    ppo_buffer.rewards[-1] += _feedback

                prev_state = state
                state = post_state
                in_combat = False
                _episode_trace.append(
                    f"[{step_i}] ORT_COMBAT #{stats['combats']} {_combat_room_type} "
                    f"steps={combat_steps} won={won} hp={_hp_at_combat_start}->{_hp_now}")
                continue
            except Exception as ort_err:
                _episode_trace.append(f"[{step_i}] ORT_COMBAT_ERROR: {ort_err}")
                # Fall through to normal combat path

        # ----- COMBAT: MCTS/PPO -----
        if st in COMBAT_SCREENS:
            if not in_combat:
                in_combat = True
                stats["combats"] += 1
                _flush_shop_session()
                _flush_rest_session()
                _combat_room_type = _detect_combat_room_type(st, state)
                if _pending_boss_deck_size is not None:
                    _combat_room_type = "boss"
                _boss_hp_frac_peak = 0.0
                _prev_combat_state = state  # track for combat step reward
                _combat_round_no = None
                _combat_turn_action_index = 0
                _combat_turn_prefix = _new_combat_turn_prefix()
                player = state.get("player") or {}
                battle = state.get("battle") or {}
                enemies = battle.get("enemies") or []
                _hp_at_combat_start = int(player.get("hp", player.get("current_hp", 0)) or 0)
                _current_combat_summary = {
                    "combat_index": stats["combats"],
                    "room_type": _combat_room_type,
                    "start_step": step_i,
                    "floor": _safe_int(run.get("floor", 0), 0),
                    "start_hp": _hp_at_combat_start,
                    "start_max_hp": _safe_int(player.get("max_hp", 0), 0),
                    "start_block": _safe_int(player.get("block", 0), 0),
                    "start_energy": _safe_int(player.get("energy", 0), 0),
                    "start_deck_size": _extract_deck_size(state),
                    "start_potion_count": len(player.get("potions") if isinstance(player.get("potions"), list) else []),
                    "start_relic_count": len(player.get("relics") if isinstance(player.get("relics"), list) else []),
                    "start_enemy_count": len(enemies),
                    "start_enemy_hp_total": sum(_safe_int(e.get("hp", 0), 0) for e in enemies if isinstance(e, dict)),
                    "enemy_group": _combat_enemy_group_key(state),
                    "start_intent": _combat_enemy_intent_summary(state),
                    "start_intent_zh": _combat_enemy_intent_summary_zh(state),
                    "potion_uses": 0,
                    "repeat_hits": 0,
                    "actions": [],
                }
                _episode_summary["combats"].append(_current_combat_summary)
                if _combat_room_type == "boss":
                    _mark_boss_reached()
                    stats["deck_size_at_boss"].append(
                        _pending_boss_deck_size
                        if _pending_boss_deck_size is not None
                        else _extract_deck_size(state)
                    )
                    _pending_boss_deck_size = None
                _episode_trace.append(
                    f"[{step_i}] COMBAT #{stats['combats']} floor={run.get('floor',0)} "
                    f"hp={player.get('hp',0)} blk={player.get('block',0)} e={player.get('energy',0)} "
                    f"hand=[{_combat_hand_summary(state)}] "
                    f"intent=[{_combat_enemy_intent_summary(state)}]"
                )
                _episode_trace_zh.extend([
                    f"[第{step_i}步][战斗开始] 第{stats['combats']}场，层数 {run.get('floor', 0)}，房间 {_trace_resolve_name(_combat_room_type)}",
                    f"我方：{player.get('hp', 0)}/{player.get('max_hp', 0) or 80} 血，{player.get('block', 0)} 格挡，{player.get('energy', 0)} 能量",
                    f"手牌：{_combat_hand_summary_zh(state)}",
                    f"敌方：{_combat_enemy_intent_summary_zh(state)}",
                ])

                # Flush PPO's last non-combat step as "entering combat"
                if len(ppo_buffer) > 0:
                    reward = shaped_reward(
                        prev_state, state, 0.0, done=False,
                        boss_entry_quality_weight=boss_entry_quality_weight,
                        early_damage_potion_penalty_weight=early_damage_potion_penalty_weight,
                    )
                    ppo_buffer.rewards[-1] = reward

            _round_no = _combat_round_number(state)
            if _combat_round_no != _round_no:
                _combat_round_no = _round_no
                _combat_turn_action_index = 0
                _combat_turn_prefix = _new_combat_turn_prefix()

            # Combat decision: MCTS search (if pipe available) + Combat PPO data
            import random as _random
            _combat_trace_payload: dict[str, Any] | None = None
            try:
                _feat_t0 = time.monotonic()
                _attach_combat_turn_prefix(state, _combat_turn_prefix)
                sf = build_combat_features(state, vocab)
                af = build_combat_action_features(state, legal, vocab)
                _feature_time += time.monotonic() - _feat_t0
                _feature_calls += 1

                # Note: deck_ids/aux/mask are now included in build_combat_features()
                # automatically. CombatPolicyValueNetwork with deck_repr_dim > 0
                # encodes them internally via its own deck_encoder.

                # Try MCTS search first (high-quality decision + behavior cloning target)
                mcts_used = False
                _mcts_nn_policy = None
                _mcts_nn_value = None
                _mcts_root_value = None
                _mcts_root_top = "-"
                _mcts_root_chosen = "chosen[missing]"
                _mcts_suspect_reasons: list[str] = []
                _mcts_allowed_this_step = _combat_should_use_mcts(
                    use_mcts_combat=use_mcts_combat,
                    mcts_warmup_active=mcts_warmup_active,
                    combat_room_type=_combat_room_type,
                    turn_action_index=_combat_turn_action_index,
                    mcts_first_n_actions_per_turn=mcts_first_n_actions_per_turn,
                    mcts_full_search_on_elite_boss=mcts_full_search_on_elite_boss,
                )
                if use_mcts_combat and not mcts_warmup_active and not _mcts_allowed_this_step:
                    stats["combat_mcts_turn_limited_steps"] += 1

                if pipe is not None and _mcts_allowed_this_step:
                    try:
                        with _optional_lock(model_forward_lock):
                            try:
                                _mcts_nn_policy, _mcts_nn_value = mcts_agent.evaluator.evaluate(state, legal)
                            except Exception as eval_error:
                                logger.debug("MCTS root NN eval failed at step %d: %s", step_i, eval_error)
                                _mcts_nn_policy, _mcts_nn_value = None, None
                            fm = PipeCombatForwardModel.from_current_state(
                                pipe() if callable(pipe) else pipe)
                            action, root = mcts_agent.choose_action(fm)
                        _mcts_root_value = float(getattr(root, "q_value", 0.0))
                        _mcts_root_top = _combat_root_topk_summary(root, k=5)
                        from search.combat_mcts_agent import _reconcile_action
                        action = _reconcile_action(action, legal)
                        _mcts_root_chosen = _combat_root_action_summary(root, action)
                        # Find action_idx in legal list
                        action_idx = 0
                        action_label = action.get("label", "")
                        for ai, la in enumerate(legal):
                            if la.get("label") == action_label and la.get("action") == action.get("action"):
                                action_idx = ai
                                break

                        # Collect MCTS behavior cloning target (visit distribution)
                        if af["action_mask"].any():
                            _, mcts_policy = root.visit_distribution()
                            padded = np.zeros(MAX_ACTIONS, dtype=np.float32)
                            padded[:len(mcts_policy)] = mcts_policy
                            mcts_pending.append({
                                "state_features": sf,
                                "action_features": af,
                                "mcts_policy": padded,
                                "outcome": 0.0,
                            })
                        mcts_used = True
                        stats["mcts_searches"] += 1
                        _mcts_suspect_reasons = _combat_mcts_suspect_reasons(
                            action=action,
                            legal=legal,
                            state=state,
                        )
                    except Exception as e:
                        logger.debug("MCTS search failed at step %d: %s", step_i, e)
                    finally:
                        try:
                            if 'fm' in dir() and fm is not None:
                                fm.cleanup_and_restore()
                        except Exception:
                            try:
                                if 'fm' in dir() and fm is not None:
                                    fm.cleanup()
                            except Exception:
                                pass

                if use_mcts_combat and mcts_warmup_active:
                    action_idx = _random.randrange(len(legal))
                    action = legal[action_idx]
                    mcts_used = False
                    stats["combat_random_warmup_steps"] += 1

                # Fallback: NN PPO sampling (also used for Combat PPO data)
                if not mcts_used and not (use_mcts_combat and mcts_warmup_active):
                    _infer_t0 = time.monotonic()
                    if inference_client is not None:
                        logits_np, value = inference_client.combat_inference(sf, af)
                        mask = af["action_mask"].astype(np.float32)
                        logits_masked = logits_np + (1.0 - mask) * (-1e9)
                        logits_t = torch.tensor(logits_masked)
                        dist = torch.distributions.Categorical(logits=logits_t)
                        if deterministic_policy:
                            action_idx_t = logits_t.argmax(dim=-1)
                        else:
                            action_idx_t = dist.sample()
                        log_prob = dist.log_prob(action_idx_t).item()
                        action_idx = action_idx_t.item()
                    elif combat_ort_session is not None:
                        ort_inputs = {}
                        for inp in combat_ort_session.get_inputs():
                            name = inp.name
                            arr = sf.get(name, af.get(name))
                            if arr is None:
                                continue
                            arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
                            if arr.dtype == bool:
                                arr = arr.astype(np.float32)
                            elif arr.dtype in (np.int64, np.int32):
                                arr = arr.astype(np.int64)
                            else:
                                arr = arr.astype(np.float32)
                            ort_inputs[name] = arr.reshape(1, *arr.shape) if arr.ndim > 0 else arr.reshape(1)
                        ort_outputs = combat_ort_session.run(None, ort_inputs)
                        logits_np = np.asarray(ort_outputs[0][0], dtype=np.float32)
                        value_arr = np.asarray(ort_outputs[1], dtype=np.float32).reshape(-1)
                        value = float(value_arr[0]) if value_arr.size > 0 else 0.0
                        mask = af["action_mask"].astype(np.float32)
                        logits_masked = logits_np + (1.0 - mask) * (-1e9)
                        if combat_safety_rerank_weight > 0.0:
                            logits_masked, _safety_adjustments = rerank_combat_logits_with_safety(
                                state,
                                legal,
                                logits_masked,
                                weight=float(combat_safety_rerank_weight),
                            )
                        logits_t = torch.tensor(logits_masked)
                        dist = torch.distributions.Categorical(logits=logits_t)
                        if deterministic_policy:
                            action_idx_t = logits_t.argmax(dim=-1)
                        else:
                            action_idx_t = dist.sample()
                        log_prob = dist.log_prob(action_idx_t).item()
                        action_idx = action_idx_t.item()
                    else:
                        combat_device = next(mcts_agent.network.parameters()).device
                        sf_t = {}
                        for k, v in sf.items():
                            t = torch.tensor(v).unsqueeze(0)
                            if v.dtype in (np.int64, np.int32): t = t.long()
                            elif v.dtype == bool: t = t.bool()
                            else: t = t.float()
                            sf_t[k] = t.to(combat_device)
                        af_t = {}
                        for k, v in af.items():
                            t = torch.tensor(v).unsqueeze(0)
                            if v.dtype in (np.int64, np.int32): t = t.long()
                            elif v.dtype == bool: t = t.bool()
                            else: t = t.float()
                            af_t[k] = t.to(combat_device)
                        with _optional_lock(model_forward_lock):
                            with torch.no_grad():
                                logits, value_t = mcts_agent.network(sf_t, af_t)
                        mask = af_t["action_mask"].float()
                        logits_masked = logits + (1.0 - mask) * (-1e9)
                        logits_masked_np = logits_masked.squeeze(0).detach().cpu().numpy()
                        if combat_safety_rerank_weight > 0.0:
                            logits_masked_np, _safety_adjustments = rerank_combat_logits_with_safety(
                                state,
                                legal,
                                logits_masked_np,
                                weight=float(combat_safety_rerank_weight),
                            )
                        logits_masked_t = torch.tensor(logits_masked_np, dtype=torch.float32, device=combat_device)
                        dist = torch.distributions.Categorical(logits=logits_masked_t)
                        if deterministic_policy:
                            action_idx_t = logits_masked_t.argmax(dim=-1)
                        else:
                            action_idx_t = dist.sample()
                        log_prob = dist.log_prob(action_idx_t).cpu().item()
                        action_idx = action_idx_t.cpu().item()
                        value = value_t.squeeze(0).cpu().item()
                        logits_masked = logits_masked_np
                    _inference_time += time.monotonic() - _infer_t0
                    _inference_calls += 1

                    if action_idx < len(legal):
                        action = legal[action_idx]
                    else:
                        action = legal[0]; action_idx = 0

                # Combat PPO data: only from NN episodes (not MCTS/heuristic).
                # MCTS actions have extremely low NN log_prob → ratio explosion in PPO.
                stats["mcts_decisions"] += 1
                _act_label = action.get("label", action.get("action", "?"))
                _src = "mcts" if mcts_used else "nn"
                _target_suffix = _action_target_summary(action)
                _player_view = _combat_player_view(state)
                _combat_ctx = (
                    f"hp={_player_view.get('hp',0)} blk={_player_view.get('block',0)} "
                    f"e={_player_view.get('energy',0)} "
                    f"hand=[{_combat_hand_summary(state)}] "
                    f"intent=[{_combat_enemy_intent_summary(state)}]"
                )
                _combat_ctx_zh = (
                    f"我方：{_player_view.get('hp', 0)}/{_player_view.get('max_hp', 0) or 80} 血，"
                    f"{_player_view.get('block', 0)} 格挡，{_player_view.get('energy', 0)} 能量；"
                    f"手牌：{_combat_hand_summary_zh(state)}；"
                    f"敌方：{_combat_enemy_intent_summary_zh(state)}"
                )

                if not mcts_used and not (use_mcts_combat and mcts_warmup_active):
                    _top_actions = _topk_action_summary(legal, logits_masked, k=3)
                    _top_actions_zh = _topk_action_summary_zh(legal, logits_masked, k=3)
                    _hard_tags = _combat_hard_state_tags(
                        state=state,
                        legal=legal,
                        action=action,
                        repeat_count=_repeat_count,
                        turn_prefix=_combat_turn_prefix,
                    )
                    _hard_weight = _combat_hard_state_weight(_hard_tags)
                    for _tag in _hard_tags:
                        if _tag == "potion_decision":
                            stats["hard_state_potion_decision_steps"] += 1
                        elif _tag == "premature_end_turn":
                            stats["hard_state_premature_end_turn_steps"] += 1
                        elif _tag == "repeat_loop_entry":
                            stats["hard_state_repeat_loop_steps"] += 1
                        elif _tag == "order_sensitive_play":
                            stats["hard_state_order_sensitive_steps"] += 1
                    stats["combat_hard_state_weight_sum"] += _hard_weight
                    _episode_trace.append(
                        f"[{step_i}] COMBAT {_src}: {_act_label}{_target_suffix} "
                        f"(idx={action_idx} v={value:.2f} lp={log_prob:.2f} "
                        f"w={_hard_weight:.2f}{' tags=' + ','.join(_hard_tags) if _hard_tags else ''}) "
                        f"{_combat_ctx} top=[{_top_actions}]"
                    )
                    # Room-type weighting for combat PPO gradient balance
                    # (2026-04-15). A typical Act 1 run has ~85% monster combat
                    # transitions, ~5% boss transitions. Without weighting, the
                    # PPO gradient is dominated by "fast hallway attack" plays
                    # and the policy never properly learns "stack block for
                    # boss" regimes. Multiply per-room weight into the existing
                    # _hard_weight (which carries combat_safety_rerank etc.).
                    # PPOTrainer normalizes sample_weights per minibatch
                    # (mb_sample_weights /= mb_sample_weights.mean()) so the
                    # absolute scale doesn't break update dynamics — relative
                    # scale of boss vs monster is what matters.
                    _room_type_weight = {
                        "monster": 1.0,
                        "elite": 2.5,
                        "boss": 5.0,
                    }.get(str(_combat_room_type or "monster").lower(), 1.0)
                    _combat_ppo_pending = {
                        "sf": sf, "af": af,
                        "action_idx": action_idx,
                        "log_prob": log_prob,
                        "value": value,
                        "screen_type": st,
                        "sample_weight": _hard_weight * _room_type_weight,
                        "hard_state_tags": _hard_tags,
                    }
                    _combat_trace_payload = {
                        "source": _src,
                        "context_zh": _combat_ctx_zh,
                        "diag_zh": (
                            f"idx={action_idx}，V={value:.2f}，logp={log_prob:.2f}，权重={_hard_weight:.2f}"
                            + (f"，标签={','.join(_hard_tags)}" if _hard_tags else "")
                        ),
                        "top_actions_zh": _top_actions_zh,
                    }
                elif use_mcts_combat and mcts_warmup_active:
                    _episode_trace.append(
                        f"[{step_i}] COMBAT warmup-random: {_act_label}{_target_suffix} "
                        f"(idx={action_idx}) {_combat_ctx}"
                    )
                    _combat_trace_payload = {
                        "source": "warmup-random",
                        "context_zh": _combat_ctx_zh,
                        "diag_zh": f"idx={action_idx}，来源=warmup-random",
                        "top_actions_zh": "",
                    }
                    _combat_ppo_pending = None
                else:
                    _diag_bits: list[str] = [f"idx={action_idx}"]
                    if _mcts_nn_value is not None:
                        _diag_bits.append(f"nn_v={float(_mcts_nn_value):.2f}")
                    if _mcts_root_value is not None:
                        _diag_bits.append(f"root_v={float(_mcts_root_value):.2f}")
                    if _mcts_suspect_reasons:
                        _diag_bits.append(f"suspect={','.join(_mcts_suspect_reasons)}")
                    _episode_trace.append(
                        f"[{step_i}] COMBAT {_src}: {_act_label}{_target_suffix} "
                        f"({' '.join(_diag_bits)}) {_combat_ctx}"
                    )
                    if _mcts_nn_policy is not None or _mcts_root_value is not None:
                        _nn_top = (
                            _topk_action_summary(legal, _mcts_nn_policy, k=5, already_probs=True)
                            if _mcts_nn_policy is not None
                            else "-"
                        )
                        _episode_trace.append(
                            f"[{step_i}] COMBAT mcts-dbg: "
                            f"nn_top=[{_nn_top}] root_top=[{_mcts_root_top}] {_mcts_root_chosen}"
                        )
                    _mcts_diag_zh = "，".join(_diag_bits)
                    _combat_trace_payload = {
                        "source": _src,
                        "context_zh": _combat_ctx_zh,
                        "diag_zh": _mcts_diag_zh,
                        "top_actions_zh": (
                            _topk_action_summary_zh(legal, _mcts_nn_policy, k=5, already_probs=True)
                            if _mcts_nn_policy is not None
                            else ""
                        ),
                        "root_top": _mcts_root_top,
                        "root_choice": _mcts_root_chosen,
                    }
                    _combat_ppo_pending = None  # MCTS data goes to mcts_pending only

                _combat_turn_action_index += 1

            except Exception as e:
                logger.warning("Combat eval failed at step %d: %s", step_i, e)
                action = _random.choice(legal)
                _combat_trace_payload = {
                    "source": "fallback-random",
                    "context_zh": f"我方：{_combat_hand_summary_zh(state)}；敌方：{_combat_enemy_intent_summary_zh(state)}",
                    "diag_zh": f"combat eval failed，随机回退：{e}",
                    "top_actions_zh": "",
                }
                _combat_ppo_pending = None

            # Execute action via client (pipe or HTTP).
            _t0 = time.monotonic()
            _act_desc = action.get("action", "?") if isinstance(action, dict) else "?"
            executed_action = action if isinstance(action, dict) else None
            pre_action_state = state
            try:
                state = client.act(action)
            except Exception:
                try:
                    fallback = {k: v for k, v in action.items()
                                if k not in ("target_id", "slot", "target")}
                    state = client.act(fallback)
                    executed_action = fallback
                except Exception:
                    try:
                        executed_action = {"action": "end_turn"}
                        state = client.act(executed_action)
                    except Exception:
                        try:
                            state = client.get_state()
                            executed_action = None
                        except Exception as e2:
                            stats["error"] = f"combat step: {e2}"
                            _timeout_count += 1
                            break
            _combat_turn_prefix = _update_combat_turn_prefix(
                _combat_turn_prefix,
                state=pre_action_state,
                action=executed_action,
            )
            _attach_combat_turn_prefix(state, _combat_turn_prefix)
            _dt = time.monotonic() - _t0
            _pipe_time += _dt
            _pipe_calls += 1
            _max_step_ms = max(_max_step_ms, _dt * 1000)
            if _dt >= _slow_step_threshold:
                _slow_steps += 1
                logger.debug("Slow combat step %d (%s): %.0fms", step_i, _act_desc, _dt * 1000)
            if _combat_room_type == "boss":
                _boss_hp_frac_peak = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))

            if _combat_trace_payload is not None:
                _zh_action = _combat_action_label_zh(executed_action or action, pre_action_state)
                _zh_target = _combat_target_label_zh(executed_action or action, pre_action_state)
                _source = str(_combat_trace_payload.get("source") or "nn")
                _episode_trace_zh.append(f"[第{step_i}步][战斗][{_source}] {_zh_action}{_zh_target}")
                _context_zh = str(_combat_trace_payload.get("context_zh") or "").strip()
                if _context_zh:
                    _episode_trace_zh.append(_context_zh)
                _diag_zh = str(_combat_trace_payload.get("diag_zh") or "").strip()
                if _diag_zh:
                    _episode_trace_zh.append(f"模型：{_diag_zh}")
                _top_actions_zh = str(_combat_trace_payload.get("top_actions_zh") or "").strip()
                if _top_actions_zh:
                    _episode_trace_zh.append(f"候选：{_top_actions_zh}")
                _root_top = str(_combat_trace_payload.get("root_top") or "").strip()
                if _root_top:
                    _episode_trace_zh.append(f"MCTS root_top：{_root_top}")
                _root_choice = str(_combat_trace_payload.get("root_choice") or "").strip()
                if _root_choice:
                    _episode_trace_zh.append(f"MCTS chosen：{_root_choice}")
                _episode_trace_zh.append(_combat_result_summary_zh(pre_action_state, state, executed_action or action))
                _episode_trace_zh.append("")
                if _current_combat_summary is not None:
                    _step_summary = _combat_step_structured_summary(pre_action_state, state, executed_action or action)
                    _current_combat_summary["actions"].append(
                        {
                            "step": step_i,
                            "source": _source,
                            "action_name": _combat_action_name(executed_action or action),
                            "action_label": _combat_action_label(executed_action or action),
                            "action_label_zh": _zh_action,
                            "target_label_zh": _zh_target,
                            "hard_tags": list((_combat_ppo_pending or {}).get("hard_state_tags") or []),
                            "top_actions_zh": str(_combat_trace_payload.get("top_actions_zh") or ""),
                            "diag_zh": str(_combat_trace_payload.get("diag_zh") or ""),
                            "result": _step_summary,
                        }
                    )
                    _current_combat_summary["last_intent"] = _step_summary.get("next_intent", "")
                    _current_combat_summary["last_intent_zh"] = _step_summary.get("next_intent_zh", "")
                    if _combat_action_name(executed_action or action) == "use_potion":
                        _current_combat_summary["potion_uses"] = _safe_int(_current_combat_summary.get("potion_uses", 0), 0) + 1

            # Add combat PPO step now that we have next_state
            if (combat_buffer is not None
                    and _combat_ppo_pending is not None):
                next_st = (state.get("state_type") or "").lower()
                # Check if combat just ended
                combat_just_ended = next_st not in COMBAT_SCREENS
                combat_won = None
                if combat_just_ended:
                    # If next state is game_over with death, combat was lost
                    if next_st == "game_over" or state.get("terminal"):
                        go = state.get("game_over") or {}
                        outcome_str = (go.get("run_outcome") or go.get("outcome") or "").lower()
                        combat_won = "victory" in outcome_str or outcome_str == "win"
                        if not combat_won:
                            combat_won = False
                    else:
                        combat_won = True  # survived combat → reward screen or map
                # Partial credit for boss damage dealt on terminal loss (see rl_reward_shaping.combat_terminal_reward)
                _br_main = 0.0
                if combat_just_ended and combat_won is False and _combat_room_type == "boss":
                    _br_main = max(_boss_hp_frac_peak, _estimate_boss_hp_fraction(state))
                step_reward = combat_step_reward(
                    _prev_combat_state, state, combat_won=combat_won,
                    hp_at_combat_start=_hp_at_combat_start,
                    boss_damage_ratio=_br_main)
                _buf_t0 = time.monotonic()
                combat_buffer.add(
                    sf=_combat_ppo_pending["sf"],
                    af=_combat_ppo_pending["af"],
                    action_idx=_combat_ppo_pending["action_idx"],
                    log_prob=_combat_ppo_pending["log_prob"],
                    reward=step_reward,
                    value=_combat_ppo_pending["value"],
                    done=bool(combat_just_ended),
                    screen_type=_combat_ppo_pending.get("screen_type", ""),
                    sample_weight=_combat_ppo_pending.get("sample_weight", 1.0),
                    hard_state_tags=_combat_ppo_pending.get("hard_state_tags"),
                )
                _buffer_time += time.monotonic() - _buf_t0
                _buffer_calls += 1
                stats["combat_ppo_steps"] += 1
                _combat_ppo_pending = None

            _prev_combat_state = state  # update for next step's reward

        # ----- NON-COMBAT: PPO -----
        else:
            if in_combat:
                in_combat = False
                stats["combats_won"] += 1
                player = state.get("player") or {}
                _hp_now = int(player.get("hp", player.get("current_hp", 0)))
                _max_hp = int(player.get("max_hp", 80))
                stats["hp_timeline"].append(_hp_now)
                _finalize_current_combat(state, won=True, end_reason="won", end_step=step_i)
                _episode_trace.append(f"[{step_i}] COMBAT WON -> {st} hp={_hp_now}/{_max_hp}")
                _boss_hp_frac_dealt = 1.0 if _combat_room_type == "boss" else 0.0
                if _combat_room_type == "boss":
                    _mark_act1_cleared()
                    stats["boss_hp_fraction_dealt"].append(max(_boss_hp_frac_peak, _boss_hp_frac_dealt))

                # --- Combat feedback → non-combat decisions ---
                _feedback = compute_combat_feedback(
                    _hp_at_combat_start, _hp_now, _max_hp, won=True,
                    room_type=_combat_room_type,
                    boss_hp_fraction_dealt=_boss_hp_frac_dealt,
                )
                if segment_collector is not None and segment_collector.is_open:
                    # Phase 2: accumulate into current segment (direct attribution)
                    segment_collector.add_reward(_feedback, tag="fight_summary", steps=5)
                    _episode_trace.append(
                        f"[{step_i}] COMBAT FEEDBACK: {_feedback:+.3f} → segment")
                else:
                    # Legacy: spread to last N non-combat PPO steps
                    _fb_steps = min(10, len(ppo_buffer))
                    for _fi in range(len(ppo_buffer) - _fb_steps, len(ppo_buffer)):
                        ppo_buffer.rewards[_fi] += _feedback / max(1, _fb_steps)
                    if _fb_steps > 0:
                        _episode_trace.append(
                            f"[{step_i}] COMBAT FEEDBACK: {_feedback:+.3f} spread to {_fb_steps} PPO steps")
                # Combat ended → backfill pending MCTS examples with HP-based value
                if mcts_pending:
                    player = state.get("player") or {}
                    hp = float(player.get("hp", player.get("current_hp", 0)))
                    max_hp = max(1.0, float(player.get("max_hp", 1)))
                    hp_frac = hp / max_hp
                    combat_value = 1.0 if _combat_room_type == "boss" else (0.8 + 0.2 * hp_frac)
                    for ex_data in mcts_pending:
                        if ex_data["outcome"] == 0.0:
                            ex_data["outcome"] = combat_value

            # PPO inference — pending-step pattern:
            # 1. Observe state, sample action
            # 2. Execute action → get next_state
            # 3. Compute reward = shaped(state, next_state)
            # 4. Write (state, action, reward) to buffer
            _ppo_pending = None
            action = None
            try:
                _shop_remove_purchase_override = _choose_shop_remove_purchase_action(state, legal)
                if _shop_remove_purchase_override is not None:
                    action_idx, action, map_override_source = _shop_remove_purchase_override
                    log_prob = 0.0
                    value = 0.0
                    _card_reward_debug = None
                else:
                    _feat_t0 = time.monotonic()
                    ss = build_structured_state(state, vocab)
                    sa = build_structured_actions(state, legal, vocab)

                    sf_np = _structured_state_to_numpy_dict(ss)
                    af_np = _structured_actions_to_numpy_dict(sa)
                    _feature_time += time.monotonic() - _feat_t0
                    _feature_calls += 1
                    policy_logits_np: np.ndarray | None = None
                    map_override_source = ""

                    _nc_fwd_t0 = time.monotonic()
                    if inference_client is not None:
                        action_idx, log_prob, _, value = inference_client.ppo_inference(sf_np, af_np)
                    elif ppo_ort_session is not None:
                        # Branch C: ORT CPU actor-only (0.5ms vs PyTorch 6.5ms)
                        ort_inputs = {}
                        for inp in ppo_ort_session.get_inputs():
                            name = inp.name
                            arr = sf_np.get(name, af_np.get(name))
                            if arr is None:
                                continue
                            arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
                            if arr.dtype == bool:
                                arr = arr.astype(np.float32)
                            elif arr.dtype in (np.int64, np.int32):
                                arr = arr.astype(np.int64)
                            else:
                                arr = arr.astype(np.float32)
                            ort_inputs[name] = arr.reshape(1, *arr.shape) if arr.ndim > 0 else arr.reshape(1)
                        ort_logits = ppo_ort_session.run(None, ort_inputs)[0][0]  # (MAX_ACTIONS,)
                        mask = af_np["action_mask"].astype(np.float32)
                        masked = ort_logits + (1.0 - mask) * (-1e9)
                        if deterministic_policy:
                            action_idx = int(np.argmax(masked))
                        else:
                            shifted = masked - masked.max()
                            probs = np.exp(shifted)
                            probs = probs / probs.sum()
                            action_idx = int(np.random.choice(len(probs), p=probs))
                        log_prob = float(np.log(max(probs[action_idx], 1e-8))) if not deterministic_policy else 0.0
                        value = 0.0
                        policy_logits_np = masked.astype(np.float32, copy=False)
                    else:
                        ppo_device = next(ppo_network.parameters()).device
                        state_t = {}
                        for k, v in sf_np.items():
                            t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
                            if "ids" in k or "idx" in k or "types" in k or "count" in k:
                                t = t.long()
                            elif "mask" in k:
                                t = t.bool()
                            else:
                                t = t.float()
                            state_t[k] = t.to(ppo_device)
                        action_t = {}
                        for k, v in af_np.items():
                            t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
                            if "ids" in k or "types" in k or "indices" in k:
                                t = t.long()
                            elif "mask" in k:
                                t = t.bool()
                            else:
                                t = t.float()
                            action_t[k] = t.to(ppo_device)
                        with _optional_lock(model_forward_lock):
                            with torch.no_grad():
                                action_idx_t, log_prob_t, _, value_t = ppo_network.get_action_and_value(
                                    state_t, action_t, deterministic=deterministic_policy)
                                logits_t, _values_t, _deck_quality_t, _boss_readiness_t, _action_adv_t = ppo_network.forward(
                                    state_t, action_t
                                )
                                mask_t = action_t["action_mask"].float()
                                logits_masked_t = logits_t + (1.0 - mask_t) * (-1e9)
                        action_idx = action_idx_t.cpu().item()
                        log_prob = log_prob_t.cpu().item()
                        value = value_t.cpu().item()
                        policy_logits_np = logits_masked_t.squeeze(0).detach().cpu().numpy()

                    if st == "card_reward":
                        override = _choose_boss_conditioned_card_reward_action(
                            state,
                            legal,
                            action_logits=policy_logits_np,
                            fallback_idx=action_idx,
                            guidance_weight=boss_conditioned_card_guidance_weight,
                        )
                        if override is not None:
                            override_idx, override_action, map_override_source = override
                            if override_idx != action_idx:
                                action_idx = override_idx
                                if policy_logits_np is not None:
                                    _dist = torch.distributions.Categorical(
                                        logits=torch.tensor(policy_logits_np[:len(legal)], dtype=torch.float32)
                                    )
                                    log_prob = _dist.log_prob(torch.tensor(action_idx)).item()
                                action = override_action
                        _card_reward_debug = _build_card_reward_decision_details(
                            state,
                            legal,
                            action_logits=policy_logits_np,
                            guidance_weight=boss_conditioned_card_guidance_weight,
                            selected_idx=action_idx,
                            source=map_override_source,
                        )
                    else:
                        _card_reward_debug = None

                    if act1_no_elite_routes and _is_act1_map_state(state):
                        override = _choose_act1_no_elite_map_action(
                            state,
                            legal,
                            action_logits=policy_logits_np,
                            fallback_idx=action_idx,
                        )
                        if override is not None:
                            override_idx, override_action, map_override_source = override
                            if override_idx != action_idx:
                                action_idx = override_idx
                                if policy_logits_np is not None:
                                    _dist = torch.distributions.Categorical(
                                        logits=torch.tensor(policy_logits_np[:len(legal)], dtype=torch.float32)
                                    )
                                    log_prob = _dist.log_prob(torch.tensor(action_idx)).item()
                                action = override_action

                    _nc_fwd_elapsed = time.monotonic() - _nc_fwd_t0
                    _inference_time += _nc_fwd_elapsed
                    _inference_calls += 1
                    stats.setdefault("_nc_forward_time", 0.0)
                    stats["_nc_forward_time"] += _nc_fwd_elapsed
                    stats.setdefault("_nc_forward_calls", 0)
                    stats["_nc_forward_calls"] += 1

                    if action is None and action_idx < len(legal):
                        action = legal[action_idx]
                    elif action is None:
                        action = legal[0]
                        action_idx = 0

                # Track card selections
                _act_name = action.get("action", "")
                _act_label = action.get("label", _act_name)
                if st == "map" and _act_name == "choose_map_node":
                    _node_hint = str(
                        action.get("node_type")
                        or action.get("note")
                        or action.get("label")
                        or ""
                    ).strip().lower()
                    if "boss" in _node_hint:
                        _pending_boss_deck_size = _extract_deck_size(state)
                if st == "card_reward":
                    stats["card_reward_screens"] += 1
                if _act_name == "select_card_reward":
                    stats["cards_taken"].append(_act_label)
                elif _act_name in ("skip", "skip_card_reward"):
                    stats["cards_skipped"] += 1
                    if st == "card_reward":
                        stats["card_reward_skips"] += 1

                # Extract player HP from nested state (player may be under event/map/shop etc.)
                _p = state.get("player") or {}
                if not _p.get("hp"):
                    for _container_key in ("event", "map", "shop", "rest_site", "rewards",
                                           "card_reward", "treasure", "combat_rewards"):
                        _container = state.get(_container_key)
                        if isinstance(_container, dict) and isinstance(_container.get("player"), dict):
                            _p = _container["player"]
                            break
                _evt_id = ""
                if st == "event":
                    _evt = state.get("event") or {}
                    _evt_id = f" [{_evt.get('event_id', '?')}]"
                _episode_trace.append(
                    f"[{step_i}] {st}{_evt_id}: {_act_label} "
                    f"(idx={action_idx} v={value:.2f}"
                    f"{' src=' + map_override_source if map_override_source else ''}) "
                    f"floor={run.get('floor',0)} hp={_p.get('hp', _p.get('current_hp', '?'))}")

                # Save pending — reward computed AFTER act()
                _summary_entry = {
                    "step": step_i,
                    "floor": _safe_int(run.get("floor", 0), 0),
                    "state_type": st,
                    "action": _act_name,
                    "label": _act_label,
                    "value": float(value),
                }
                if map_override_source:
                    _summary_entry["source"] = map_override_source
                    _episode_summary["counters"]["decision_override_steps"] = (
                        _safe_int(_episode_summary["counters"].get("decision_override_steps", 0), 0) + 1
                    )
                    if st == "map":
                        _episode_summary["counters"]["map_override_steps"] += 1
                if st == "map" and _act_name == "choose_map_node":
                    _summary_entry["node_type"] = str(
                        action.get("node_type")
                        or action.get("note")
                        or action.get("label")
                        or ""
                    )
                    _episode_summary["route_choices"].append(_summary_entry)
                    _flush_shop_session()
                    _flush_rest_session()
                    _choice_label = _lower_text(_summary_entry.get("node_type") or _act_label)
                    if "shop" in _choice_label:
                        _current_shop_session = _build_shop_session_snapshot(
                            state,
                            step_i=step_i,
                            floor=_summary_entry["floor"],
                        )
                    elif "rest" in _choice_label or "camp" in _choice_label:
                        _current_rest_session = {
                            "enter_step": step_i,
                            "enter_floor": _summary_entry["floor"],
                            "actions": [],
                        }
                elif st == "event":
                    _summary_entry["event_id"] = str((state.get("event") or {}).get("event_id") or "")
                    _episode_summary["event_choices"].append(_summary_entry)
                    _flush_shop_session()
                    _flush_rest_session()
                elif st == "card_reward":
                    _summary_entry["skipped"] = _act_name in ("skip", "skip_card_reward")
                    if _card_reward_debug is not None:
                        _summary_entry["decision_debug"] = _card_reward_debug
                    _episode_summary["card_rewards"].append(_summary_entry)
                    _flush_shop_session()
                    _flush_rest_session()
                elif st == "shop":
                    if _current_shop_session is None:
                        _current_shop_session = _build_shop_session_snapshot(
                            state,
                            step_i=step_i,
                            floor=_summary_entry["floor"],
                        )
                    elif not (_current_shop_session.get("offers") or []):
                        _shop_snapshot = _build_shop_session_snapshot(
                            state,
                            step_i=int(_current_shop_session.get("enter_step", step_i) or step_i),
                            floor=int(_current_shop_session.get("enter_floor", _summary_entry["floor"]) or _summary_entry["floor"]),
                        )
                        _current_shop_session["enter_gold"] = _shop_snapshot.get("enter_gold", 0)
                        _current_shop_session["offers"] = _shop_snapshot.get("offers", [])
                    _current_shop_session["actions"].append(_summary_entry)
                elif st == "rest_site":
                    if _current_rest_session is None:
                        _current_rest_session = {
                            "enter_step": step_i,
                            "enter_floor": _summary_entry["floor"],
                            "actions": [],
                        }
                    _current_rest_session["actions"].append(_summary_entry)
                else:
                    _flush_shop_session()
                    _flush_rest_session()

                if _shop_remove_purchase_override is not None:
                    # FIX: force_remove samples must NOT enter PPO buffer.
                    # They carry synthetic log_prob=0.0 while the actor's real
                    # log P_new(remove_card) is very small, so ratio collapses
                    # toward 0 and approx_kl = |0 - new_log_prob| explodes
                    # (observed KL 0.3-0.87 in validation runs even at epoch=1).
                    # Skipping buffer here fully decouples the deterministic
                    # hard rule from PPO learning while keeping the action
                    # being actually executed in the environment.
                    _ppo_pending = None
                else:
                    _ppo_pending = {
                        "ss": ss, "sa": sa,
                        "action_idx": action_idx,
                        "log_prob": log_prob,
                        "value": value,
                        "pre_state": state,  # state BEFORE action
                        "screen_type": st,
                    }

            except Exception as e:
                logger.warning("PPO inference failed at step %d: %s", step_i, e)
                action = legal[0]

            prev_state = state

            # Execute action via client (pipe or HTTP)
            _t0 = time.monotonic()
            _act_desc = action.get("action", "?") if isinstance(action, dict) else "?"
            try:
                if _current_reward_claim_count is not None:
                    _last_reward_claim_count = _current_reward_claim_count
                state = client.act(action)
                _last_action_name = str(_act_desc or "").strip().lower()
                _last_reward_claim_sig = _next_reward_claim_signature(st, prev_state, action)
            except Exception as e1:
                err_str = str(e1)
                _episode_trace.append(f"[{step_i}] ACT FAILED ({_act_desc}): {err_str[:80]}")
                # Retry once for transient errors (file rename race condition)
                if "rename" in err_str.lower() or "file" in err_str.lower():
                    import time as _t
                    _t.sleep(0.1)
                    try:
                        if _current_reward_claim_count is not None:
                            _last_reward_claim_count = _current_reward_claim_count
                        state = client.act(action)
                        _last_action_name = str(_act_desc or "").strip().lower()
                        _last_reward_claim_sig = _next_reward_claim_signature(st, prev_state, action)
                        _episode_trace.append(f"[{step_i}] RETRY OK")
                    except Exception:
                        stats["error"] = f"rename_retry_failed: {err_str[:50]}"
                        stats["end_reason"] = "step_error"
                        break
                else:
                    pass
                try:
                    state = client.get_state()
                    fresh_legal = [a for a in state.get("legal_actions", [])
                                   if isinstance(a, dict) and a.get("is_enabled") is not False]
                    if fresh_legal:
                        fallback_action = (
                            _choose_auto_progress_action(
                                state,
                                fresh_legal,
                                _last_action_name,
                                _last_reward_claim_sig,
                                _last_reward_claim_count,
                                _reward_chain_card_reward_seen,
                            )
                            or fresh_legal[0]
                        )
                        fallback_claim_count = _claim_reward_action_count(fresh_legal) if st == "combat_rewards" else None
                        if fallback_claim_count is not None:
                            _last_reward_claim_count = fallback_claim_count
                        prev_state = state
                        state = client.act(fallback_action)
                        _last_action_name = str(fallback_action.get("action") or "").strip().lower()
                        _last_reward_claim_sig = _next_reward_claim_signature(
                            st,
                            prev_state,
                            fallback_action,
                        )
                    else:
                        try:
                            recovery_action = _choose_empty_legal_recovery_action(state, _last_action_name)
                            if recovery_action is not None:
                                prev_state = state
                                state = client.act(recovery_action)
                                _last_action_name = str(recovery_action.get("action") or "").strip().lower()
                                _last_reward_claim_sig = _next_reward_claim_signature(
                                    st,
                                    prev_state,
                                    recovery_action,
                                )
                            else:
                                state = client.act({"action": "wait"})
                                _last_action_name = "wait"
                                _last_reward_claim_sig = ""
                        except Exception:
                            state = client.get_state()
                except Exception as e2:
                    _episode_trace.append(f"[{step_i}] FALLBACK FAILED: {str(e2)[:80]}")
                    stats["error"] = f"non-combat step: {e2}"
                    stats["end_reason"] = "step_error"
                    _timeout_count += 1
                    break
            _dt = time.monotonic() - _t0
            _pipe_time += _dt
            _pipe_calls += 1
            _max_step_ms = max(_max_step_ms, _dt * 1000)
            if _dt >= _slow_step_threshold:
                _slow_steps += 1
                logger.debug("Slow non-combat step %d (%s): %.0fms", step_i, _act_desc, _dt * 1000)

            # Now compute reward with CORRECT (pre_state → post_state) and add to buffer
            if _ppo_pending is not None:
                reward = shaped_reward(
                    _ppo_pending["pre_state"], state, 0.0, done=False,
                    boss_entry_quality_weight=boss_entry_quality_weight,
                    action=action,
                    early_damage_potion_penalty_weight=early_damage_potion_penalty_weight,
                )
                local_delta = 0.0
                if screen_local_delta and segment_collector is None:
                    local_delta = screen_local_delta_reward(
                        _ppo_pending["pre_state"], state, _ppo_pending.get("screen_type", ""),
                    )
                    reward += local_delta

                if segment_collector is not None and segment_buffer is not None:
                    _buf_t0 = time.monotonic()
                    # Phase 2: Close previous segment (if any) and open new one
                    if segment_collector.is_open:
                        seg = segment_collector.close_segment(done=False)
                        if seg is not None:
                            segment_buffer.add(seg)
                    # Compute counterfactual reward (Phase 3)
                    _cf_teacher = None
                    if counterfactual_scoring:
                        try:
                            _cf_reward, _cf_teacher = compute_counterfactual_reward(
                                st, _ppo_pending["pre_state"], legal,
                                _ppo_pending["action_idx"],
                            )
                        except Exception:
                            _cf_reward = 0.0
                    else:
                        _cf_reward = 0.0
                    # Open new segment
                    segment_collector.open_segment(
                        state=_structured_state_to_numpy_dict(_ppo_pending["ss"]),
                        actions=_structured_actions_to_numpy_dict(_ppo_pending["sa"]),
                        action_idx=_ppo_pending["action_idx"],
                        log_prob=_ppo_pending["log_prob"],
                        value=_ppo_pending["value"],
                        screen_type_idx=_ppo_pending["ss"].screen_type_idx,
                        teacher_logits=_cf_teacher,
                    )
                    # Add PBRS reward + counterfactual
                    segment_collector.add_reward(reward, tag="pbrs", steps=1)
                    if _cf_reward != 0.0:
                        segment_collector.add_reward(
                            _cf_reward * counterfactual_weight, tag="counterfactual", steps=0)
                    _buffer_time += time.monotonic() - _buf_t0
                    _buffer_calls += 1
                    stats["ppo_steps"] += 1
                else:
                    # Legacy step-by-step path
                    _buf_t0 = time.monotonic()
                    ppo_buffer.add(
                        _ppo_pending["ss"], _ppo_pending["sa"],
                        _ppo_pending["action_idx"], _ppo_pending["log_prob"],
                        reward, _ppo_pending["value"], done=False,
                        boss_readiness_target=boss_readiness_score(_ppo_pending["pre_state"]))
                    stats["ppo_steps"] += 1
                    # Save for offline data
                    if episode_saver is not None:
                        episode_saver.add_step(
                            _structured_state_to_numpy_dict(_ppo_pending["ss"]),
                            _structured_actions_to_numpy_dict(_ppo_pending["sa"]),
                            _ppo_pending["action_idx"], reward,
                            _ppo_pending.get("screen_type", "unknown"),
                            _ppo_pending["log_prob"], _ppo_pending["value"],
                        )
                    _buffer_time += time.monotonic() - _buf_t0
                    _buffer_calls += 1

    # Close any remaining open segment
    if segment_collector is not None and segment_collector.is_open and segment_buffer is not None:
        seg = segment_collector.close_segment(done=True)
        if seg is not None:
            segment_buffer.add(seg)

    # Set floor targets for PPO deck quality head
    # Richer target: progress + win bonus + combat efficiency
    final_floor = stats["floors"]
    total_floor = float(final_floor)
    won = stats.get("outcome") == "victory"
    combats_won = stats.get("combats_won", 0)
    combats_total = max(1, stats.get("combats", 1))
    dq_target = (
        (total_floor / 51.0) * 0.5                      # progress [0, 0.5]
        + (1.0 if won else 0.0) * 0.3                   # win bonus [0, 0.3]
        + (combats_won / combats_total) * 0.2            # combat efficiency [0, 0.2]
    )
    dq_target = min(dq_target, 1.0)
    if len(ppo_buffer) > 0:
        ppo_buffer.set_floor_targets(dq_target)
    if segment_buffer is not None and len(segment_buffer) > 0:
        segment_buffer.set_floor_targets(dq_target)

    # Build MCTS examples
    mcts_examples = [MCTSTrainingExample(**d) for d in mcts_pending]

    # End-of-episode reason
    if not stats.get("error") and not stats.get("outcome"):
        if stats.get("end_reason") == "combat_pending_stall":
            if in_combat:
                _finalize_current_combat(state, won=False, end_reason="combat_pending_stall", end_step=step_i)
        else:
            stats["end_reason"] = "max_steps"
            _episode_trace.append(f"[END] max_steps reached (steps={step_i}, st={st})")
            if in_combat:
                _finalize_current_combat(state, won=False, end_reason="max_steps", end_step=step_i)
    _flush_shop_session()
    _flush_rest_session()
    _flush_combat_pending_span(step_i, st)
    if stats.get("act1_cleared"):
        _mark_boss_reached()
    stats["episode_time_s"] = time.monotonic() - episode_start
    stats["slow_steps"] = _slow_steps
    stats["max_step_ms"] = _max_step_ms
    stats["timeout_count"] = _timeout_count
    stats["pipe_time_s"] = _pipe_time
    stats["pipe_calls"] = _pipe_calls
    stats["feature_time_s"] = _feature_time
    stats["feature_calls"] = _feature_calls
    stats["inference_time_s"] = _inference_time
    stats["inference_calls"] = _inference_calls
    stats["buffer_time_s"] = _buffer_time
    stats["buffer_calls"] = _buffer_calls
    # Save high-quality episodes for offline RL
    if episode_saver is not None:
        episode_saver.finish_episode(
            floor=final_floor,
            outcome=stats.get("outcome"),
            combats_won=combats_won,
            extra_stats=stats,
        )

    _episode_summary["outcome"] = stats.get("outcome")
    _episode_summary["error"] = stats.get("error")
    _episode_summary["end_reason"] = stats.get("end_reason")
    _episode_summary["final_floor"] = stats.get("floors", 0)
    _episode_summary["combat_details"] = list(_episode_summary.get("combats") or [])
    _episode_summary["combats"] = stats.get("combats", 0)
    _episode_summary["combats_won"] = stats.get("combats_won", 0)
    _episode_summary["boss_reached"] = bool(stats.get("boss_reached"))
    _episode_summary["act1_cleared"] = bool(stats.get("act1_cleared"))
    _episode_summary["cards_taken"] = list(stats.get("cards_taken") or [])
    _episode_summary["cards_skipped"] = _safe_int(stats.get("cards_skipped", 0), 0)
    _episode_summary["death_enemy"] = stats.get("death_enemy")
    _episode_summary["repeat_max"] = _safe_int(stats.get("repeat_max", 0), 0)

    stats["_combat_buffer"] = combat_buffer  # for merging in main loop
    stats["_episode_trace"] = _episode_trace  # for replay dump
    stats["_episode_trace_zh"] = _episode_trace_zh  # for replay dump
    stats["_episode_summary"] = _episode_summary  # for replay dump
    stats["_segment_buffer"] = segment_buffer  # for Phase 2 merging
    if segment_buffer is not None and len(segment_buffer) > 0:
        stats["segment_stats"] = segment_buffer.get_segment_stats()
    return ppo_buffer, mcts_examples, stats


# ---------------------------------------------------------------------------
# MCTS train step (from train_combat_mcts.py)
# ---------------------------------------------------------------------------

def mcts_train_step(
    network: CombatPolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[MCTSTrainingExample],
    device: torch.device | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    if device is None:
        device = next(network.parameters()).device

    state_tensors = {}
    action_tensors = {}
    for k in batch[0].state_features:
        arr = np.stack([ex.state_features[k] for ex in batch])
        if arr.dtype in (np.int64, np.int32):
            state_tensors[k] = torch.tensor(arr, dtype=torch.long, device=device)
        elif arr.dtype == bool:
            state_tensors[k] = torch.tensor(arr, dtype=torch.bool, device=device)
        else:
            state_tensors[k] = torch.tensor(arr, dtype=torch.float32, device=device)
    for k in batch[0].action_features:
        arr = np.stack([ex.action_features[k] for ex in batch])
        if arr.dtype in (np.int64, np.int32):
            action_tensors[k] = torch.tensor(arr, dtype=torch.long, device=device)
        elif arr.dtype == bool:
            action_tensors[k] = torch.tensor(arr, dtype=torch.bool, device=device)
        else:
            action_tensors[k] = torch.tensor(arr, dtype=torch.float32, device=device)

    target_policy = torch.tensor(np.stack([ex.mcts_policy for ex in batch]),
                                  dtype=torch.float32, device=device)
    target_value = torch.tensor([ex.outcome for ex in batch],
                                 dtype=torch.float32, device=device)

    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, value = network.forward(state_tensors, action_tensors)
            logits_safe = logits.float().clamp(min=-30.0)
            log_probs = F.log_softmax(logits_safe, dim=-1)
            mask = action_tensors["action_mask"].float()
            policy_loss = -(target_policy * (log_probs * mask)).sum(dim=-1).mean()
            value_loss = F.mse_loss(value.float(), target_value)
            loss = policy_loss + value_loss
    else:
        logits, value = network.forward(state_tensors, action_tensors)
        logits_safe = logits.clamp(min=-30.0)
        log_probs = F.log_softmax(logits_safe, dim=-1)
        mask = action_tensors["action_mask"].float()
        policy_loss = -(target_policy * (log_probs * mask)).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, target_value)
        loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()

    return {"mcts_ploss": policy_loss.item(), "mcts_vloss": value_loss.item()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    config_path = _peek_config_path(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Unified PPO + MCTS hybrid training")
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Optional TOML config file. Leaf keys should match argparse dest names.",
    )
    # Environment
    parser.add_argument("--pipe", action="store_true", help="Use pipe for MCTS (recommended)")
    parser.add_argument(
        "--transport",
        choices=["http", "pipe", "pipe-binary"],
        default=None,
        help="Simulator transport override. Defaults to 'pipe' when --pipe is set, otherwise 'http'.",
    )
    parser.add_argument("--auto-launch", action="store_true",
                        help="Auto-launch one fresh Sim host per env port for pipe transports.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                        help="Repo root used when auto-launching Sim hosts.")
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_DLL_PATH,
                        help="Path to headless_sim_host_0991.exe/.dll (or legacy HeadlessSim.dll) for auto-launch.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--start-port", type=int, default=15527)
    parser.add_argument("--character-id", default="IRONCLAD")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional global seed for Python/NumPy/Torch and per-episode env reset seeds")

    # Training
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--episodes-per-iter", type=int, default=0,
                        help="0 = auto (same as num-envs)")
    parser.add_argument("--episode-timeout", type=float, default=90.0)
    parser.add_argument("--max-episode-steps", type=int, default=600)

    # PPO hyperparams
    parser.add_argument("--ppo-lr", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch", type=int, default=32)
    parser.add_argument("--target-kl", type=float, default=0.0,
                        help="Target KL for non-combat PPO early stop (0=disabled, 0.01 recommended).")
    parser.add_argument("--ppo-entropy-coeff", type=float, default=0.1)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--boss-readiness-coeff", type=float, default=0.05,
                        help="Auxiliary loss weight for boss-aware build readiness head")
    parser.add_argument("--boss-aware-warmup-only", action="store_true", default=False,
                        help="Freeze old PPO backbone and train only new boss-aware adapter/head params")

    # MCTS hyperparams
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--mcts-lr", type=float, default=1e-3)
    parser.add_argument("--mcts-batch-size", type=int, default=128)
    parser.add_argument("--mcts-train-steps", type=int, default=10)
    parser.add_argument("--mcts-warmup-iters", type=int, default=50,
                        help="Use random combat actions for first N iters (untrained NN is worse than random)")
    parser.add_argument("--mcts-replay-size", type=int, default=50000)
    parser.add_argument("--mcts-first-n-actions-per-turn", type=int, default=0,
                        help="If >0, hallway combat only uses MCTS for the first N player actions each turn (0=all actions).")
    parser.add_argument("--mcts-full-search-on-elite-boss", action="store_true", default=True,
                        help="Always keep full-turn MCTS search on elite/boss combats even when hallway turns are capped.")
    parser.add_argument("--no-mcts-full-search-on-elite-boss", dest="mcts_full_search_on_elite_boss", action="store_false")
    parser.add_argument("--act1-no-elite-routes", action="store_true", default=False,
                        help="On Act 1 map screens, override PPO to only choose branches whose future path can avoid elites.")

    # Combat PPO hyperparams
    parser.add_argument("--combat-ppo-lr", type=float, default=3e-4,
                        help="Combat PPO learning rate (default: 3e-4)")
    parser.add_argument("--combat-ppo-epochs", type=int, default=4,
                        help="Combat PPO epochs per update (default: 4)")
    parser.add_argument("--combat-ppo-minibatch", type=int, default=64,
                        help="Combat PPO minibatch size (default: 64)")
    parser.add_argument("--combat-target-kl", type=float, default=0.0,
                        help="Target KL for combat PPO early stop (0=disabled, 0.01 recommended).")
    parser.add_argument("--combat-ppo-entropy-coeff", type=float, default=0.05,
                        help="Combat PPO entropy coefficient (default: 0.05)")
    parser.add_argument("--combat-ppo-clip", type=float, default=0.2,
                        help="Combat PPO clip epsilon (default: 0.2)")

    # Network architecture
    parser.add_argument("--embed-dim", type=int, default=48,
                        help="Entity embedding dimension (default: 48)")
    parser.add_argument("--combat-hidden-dim", type=int, default=192,
                        help="Combat NN hidden dimension (default: 192)")
    parser.add_argument("--combat-main-path-mode", choices=["mlp", "light_attention"], default="mlp",
                        help="Main combat rollout path structure. 'mlp' keeps the legacy policy/value path; "
                             "'light_attention' enables a residual state/action attention branch in the "
                             "main combat policy/value path. Recommended to test only in hybrid/PPO training.")
    parser.add_argument("--deck-repr-dim", type=int, default=0,
                        help="Deck embedding dimension for build_plan_z bridge (0=disabled, 64=recommended)")
    parser.add_argument("--vectorized", action="store_true", default=False,
                        help="Use vectorized episode collection (parallel pipe I/O + batch NN inference)")
    parser.add_argument("--local-ort", action="store_true", default=False,
                        help="Use C# local ORT CPU actor for combat (3x+ throughput, requires ONNX model loaded in sim)")
    parser.add_argument("--ort-model-path", type=str, default=None,
                        help="Path to ONNX actor model (auto-loads into each HeadlessSim on startup)")
    parser.add_argument(
        "--combat-mcts-backend",
        choices=["python", "csharp"],
        default="python",
        help="Combat MCTS backend. 'csharp' requires pipe-binary transport and an ORT model loaded in the simulator.",
    )
    parser.add_argument(
        "--combat-mcts-continuation-value",
        action="store_true",
        default=False,
        help="Use continuation_value_head for combat MCTS leaf evaluation. When backend=csharp this is forwarded to the simulator search request.",
    )
    parser.add_argument("--iter-time-budget", type=float, default=0,
                        help="Max seconds per iter for episode collection (0=no limit, 4.0=recommended with --local-ort)")
    parser.add_argument("--zero-cuda-collector", action="store_true", default=False,
                        help="Use CPU policy snapshots for rollout workers (zero CUDA in collector threads)")
    parser.add_argument("--residual-adapter", action="store_true", default=False,
                        help="Use residual adapter mode: freeze backbone, train only deck adapter heads")
    parser.add_argument("--freeze-embeddings", action="store_true", default=False,
                        help="Freeze shared entity embeddings to prevent combat/ranking gradient conflict")
    parser.add_argument("--retrieval-head", action="store_true", default=False,
                        help="Enable SymbolicFeaturesHead cross-attention over "
                             "source_knowledge.sqlite. Adds ~18K trainable params "
                             "and a zero-shot symbolic prior for rare entities. "
                             "Default off so the champion is untouched. See "
                             "docs/HANDOFF_2026-04-09.md §7.2.D for design.")
    parser.add_argument("--retrieval-proj-dim", type=int, default=16,
                        help="Output projection dim of SymbolicFeaturesHead "
                             "(default: 16). Only used when --retrieval-head is set.")
    parser.add_argument("--freeze-combat", action="store_true", default=False,
                        help="Freeze entire combat brain (mcts_net). Only train PPO/non-combat side.")
    parser.add_argument("--freeze-ppo", action="store_true", default=False,
                        help="Freeze entire PPO brain (ppo_net). Only train combat side.")
    parser.add_argument("--combat-boss-only", action="store_true", default=False,
                        help="Only train combat on boss/elite encounters (skip hallway). "
                             "Requires --freeze-ppo.")
    parser.add_argument("--combat-monster-reward-weight", type=float, default=1.0,
                        help="Reward weight for monster combat data during merge (default 1.0). "
                             "Setting <1.0 keeps monster data in buffer but dampens its loss contribution. "
                             "Ignored when --combat-boss-only is set. Recommended: 0.1 for focus on boss/elite.")
    parser.add_argument("--matchup-loss-decay-tau", "--offline-noncombat-ranking-loss-decay-tau", type=float, default=0.0,
                        help="Exponential decay tau for offline non-combat ranking loss weight (0=no decay, 300=recommended)")

    # Checkpoints
    parser.add_argument("--resume", type=str, default=str(MAINLINE_CHECKPOINT),
                        help="Resume both networks from a hybrid checkpoint (hybrid_XXXXX.pt)")
    parser.add_argument("--resume-ppo", type=str, default=None,
                        help="Resume PPO only (standalone PPO checkpoint)")
    parser.add_argument("--resume-mcts", type=str, default=None,
                        help="Resume MCTS only (standalone MCTS checkpoint)")
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Optional run label appended after the timestamp in the per-run output folder name. "
            "Keep --output-dir as a stable experiment root; put short experiment notes such as "
            "'acttransitionfix_resume2275' in --run-tag."
        ),
    )
    parser.add_argument("--multi-process", action="store_true",
                        help="Use multi-process workers + batch inference (bypasses GIL)")
    parser.add_argument("--deterministic-policy", action="store_true", default=False,
                        help="Use argmax action selection for PPO/combat policy (audit/demo only)")
    parser.add_argument("--batch-timeout-ms", type=float, default=5.0,
                        help="Batch inference server timeout in ms")
    parser.add_argument("--mcts", action="store_true",
                        help="Enable MCTS combat search (slower but higher quality decisions)")
    parser.add_argument("--no-mcts", action="store_true",
                        help="(deprecated, MCTS off by default)")

    # --- GPT-design optimizations (Phase 1-4) ---
    parser.add_argument("--screen-value-heads", action="store_true", default=True,
                        help="Use screen-specific value heads (Phase 1A, default: True)")
    parser.add_argument("--no-screen-value-heads", dest="screen_value_heads", action="store_false")
    parser.add_argument("--per-screen-adv-norm", action="store_true", default=True,
                        help="Normalize advantages per screen type (Phase 1B)")
    parser.add_argument("--no-per-screen-adv-norm", dest="per_screen_adv_norm", action="store_false")
    parser.add_argument("--weighted-screen-sampling", action="store_true", default=True,
                        help="Weight minibatch sampling by screen frequency (Phase 1C)")
    parser.add_argument("--no-weighted-screen-sampling", dest="weighted_screen_sampling", action="store_false")
    parser.add_argument("--use-segment-collector", action="store_true", default=False,
                        help="Use semi-MDP segment collector for non-combat (Phase 2)")
    parser.add_argument("--counterfactual-scoring", action="store_true", default=False,
                        help="Enable screen-local counterfactual scoring (Phase 3)")
    parser.add_argument("--counterfactual-weight", type=float, default=0.0,
                        help="Blend weight for counterfactual reward (Phase 3)")
    parser.add_argument("--kl-warmstart", action="store_true", default=False,
                        help="Enable KL warm-start from heuristic teacher (Phase 4)")
    parser.add_argument("--kl-beta-start", type=float, default=0.5,
                        help="Initial KL loss coefficient (Phase 4)")
    parser.add_argument("--kl-beta-decay", type=int, default=2000,
                        help="Number of iterations for KL beta decay (Phase 4)")

    # --- Offline data saving ---
    parser.add_argument("--save-offline-data", "--saved-offline-episodes", dest="save_offline_data", action="store_true", default=True,
                        help="Save high-quality episodes for future offline use (default: True)")
    parser.add_argument("--no-save-offline-data", "--no-saved-offline-episodes", dest="save_offline_data", action="store_false")
    parser.add_argument("--offline-min-floor", "--saved-offline-episodes-min-floor", type=int, default=14,
                        help="Min floor required before saved offline episodes are written (default: 14)")
    parser.add_argument("--save-replay-traces", action="store_true", default=True,
                        help="Write per-episode replay trace files (default: True)")
    parser.add_argument("--no-save-replay-traces", dest="save_replay_traces", action="store_false")
    parser.add_argument("--save-replay-human", action="store_true", default=True,
                        help="Write human-readable replay text files (*.txt/*.zh.txt) (default: True)")
    parser.add_argument("--no-save-replay-human", dest="save_replay_human", action="store_false")
    parser.add_argument("--save-replay-structured", action="store_true", default=True,
                        help="Write machine-readable replay summary files (*.summary.json) (default: True)")
    parser.add_argument("--no-save-replay-structured", dest="save_replay_structured", action="store_false")
    parser.add_argument("--combat-pending-stall-threshold", type=int, default=100,
                        help="Abort episode early when combat_pending repeats this many steps (0=disabled).")
    parser.add_argument("--save-metrics-log", action="store_true", default=True,
                        help="Append per-iteration metrics to metrics.jsonl (default: True)")
    parser.add_argument("--no-save-metrics-log", dest="save_metrics_log", action="store_false")
    parser.add_argument("--screen-local-delta", action="store_true", default=True,
                        help="Add small immediate screen-local delta reward on legacy PPO path (default: True)")
    parser.add_argument("--no-screen-local-delta", dest="screen_local_delta", action="store_false")

    # --- Offline non-combat ranking data (legacy internal name: matchup) ---
    parser.add_argument("--matchup-data-dir", "--offline-noncombat-ranking-data-dir",
                        type=str, default=str(DEFAULT_OFFLINE_NONCOMBAT_RANKING_DATA_DIR),
                        help="Offline non-combat ranking dataset path (legacy name: matchup-data-dir)")
    parser.add_argument("--matchup-batch-size", "--offline-noncombat-ranking-batch-size",
                        type=int, default=32,
                        help="Batch size for offline non-combat ranking loss (default: 32)")
    parser.add_argument("--matchup-loss-weight", "--offline-noncombat-ranking-loss-weight",
                        type=float, default=0.1,
                        help="Weight for offline non-combat ranking loss (default: 0.1)")
    parser.add_argument("--matchup-updates-per-iter", "--offline-noncombat-ranking-updates-per-iter",
                        type=int, default=1,
                        help="How many offline non-combat ranking updates to run per iteration (default: 1)")
    parser.add_argument("--matchup-warmup-iters", "--offline-noncombat-ranking-warmup-iters",
                        type=int, default=100,
                        help="Skip offline non-combat ranking loss for first N iterations (default: 100)")
    parser.add_argument("--matchup-blend-beta", "--offline-noncombat-ranking-blend-beta", type=float, default=0.0,
                        help="Blend the non-combat ranking score head into teacher signal (0=off, 0.3=recommended)")
    parser.add_argument("--matchup-min-spread", "--offline-noncombat-ranking-min-spread", type=float, default=0.001,
                        help="Filter out offline non-combat ranking samples with spread below this (default: 0.001)")
    parser.add_argument(
        "--offline-noncombat-ranking-head-mode",
        "--matchup-head-mode",
        type=str,
        choices=["mlp", "light_attention", "transformer"],
        default="mlp",
        help=(
            "Structure used by the offline non-combat ranking scorer. "
            "'mlp' keeps the legacy option scorer; 'light_attention' adds a "
            "residual attention block over screen context plus candidate options; "
            "'transformer' uses a deeper transformer-style residual scorer."
        ),
    )

    # --- Skada community priors ---
    parser.add_argument("--skada-prior-weight", type=float, default=0.15,
                        help="Blend weight for Skada community priors in counterfactual scoring (0=off, 0.15=recommended)")
    parser.add_argument("--skada-boss-weights", action="store_true", default=False,
                        help="Use Skada boss wipe rates to scale boss-entry quality bonus")
    parser.add_argument("--skada-db", type=str, default=None,
                        help="Path to Skada analytics SQLite DB (default: auto-detect)")

    # --- Combat teacher data (offline turn-solver teacher) ---
    parser.add_argument("--combat-teacher-data-dir", "--offline-combat-teacher-data-dir",
                        type=str, default=str(DEFAULT_COMBAT_TEACHER_DATA),
                        help="Offline combat teacher dataset path (JSONL or directory)")
    parser.add_argument("--combat-teacher-loss-weight", "--offline-combat-teacher-loss-weight",
                        type=float, default=0.1,
                        help="Weight for offline combat teacher loss (default: 0.1)")
    parser.add_argument("--combat-teacher-batch-size", "--offline-combat-teacher-batch-size",
                        type=int, default=32,
                        help="Batch size for offline combat teacher loss (default: 32)")
    parser.add_argument("--combat-teacher-updates-per-iter", "--offline-combat-teacher-updates-per-iter",
                        type=int, default=1,
                        help="How many offline combat teacher updates to run per iteration (default: 1)")
    parser.add_argument("--combat-teacher-warmup-iters", "--offline-combat-teacher-warmup-iters",
                        type=int, default=0,
                        help="Skip offline combat teacher loss for first N iterations (default: 0)")

    # --- Build Mode (non-boss combat auto-win) ---
    parser.add_argument("--build-mode", action="store_true", default=False,
                        help="Build Mode: auto-win non-boss combat (monster/elite) via save/load "
                             "state. Only boss fights are played normally. Use this to isolate "
                             "non-combat brain training and test deck-building quality.")
    parser.add_argument("--build-mode-hp-restore", type=float, default=1.0,
                        help="HP fraction to restore after auto-win combat in build mode "
                             "(1.0 = full HP, 0.8 = 80%% of max HP). Default: 1.0")

    # --- Step 2 / Phase 5: Macro Milestone PPO (boss-entry build quality) ---
    parser.add_argument("--boss-entry-quality-weight", type=float, default=0.0,
                        help="Step 2 / Phase 5: scale boss-entry quality milestone bonus "
                             "(potions/HP/relics, fires once when crossing floor 14->15+). "
                             "0.0 = disabled (default), 1.0 = full weight (~+0.95 max).")
    parser.add_argument("--early-damage-potion-penalty-weight", type=float, default=0.0,
                        help="Step 2 / Phase 5: scale per-use penalty for using a damage potion "
                             "before reaching the boss zone. 0.0 = disabled (default), 1.0 = -0.05/use.")
    parser.add_argument("--boss-conditioned-card-guidance-weight", type=float, default=0.0,
                        help="Apply a lightweight boss-conditioned rerank on card rewards using next boss token "
                             "(0=off, ~0.6-1.0 recommended for short tests).")
    parser.add_argument("--combat-safety-rerank-weight", type=float, default=0.0,
                        help="Apply low-HP / high-incoming combat safety rerank on NN logits "
                             "(0=off, 1.0 recommended).")

    if config_path:
        config_overrides = _load_train_hybrid_config(config_path)
        known_dests = {action.dest for action in parser._actions}
        unknown_keys = sorted(key for key in config_overrides if key not in known_dests)
        if unknown_keys:
            logger.warning("Ignoring unknown config keys from %s: %s", config_path, ", ".join(unknown_keys))
        parser.set_defaults(**{key: value for key, value in config_overrides.items() if key in known_dests})

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if args.deterministic_policy and args.multi_process:
        logger.warning("Config: --deterministic-policy forces single-process inference for audit stability.")
        args.multi_process = False
    if args.episodes_per_iter == 0:
        args.episodes_per_iter = max(args.num_envs, 2)

    effective_counterfactual_scoring, effective_counterfactual_weight, cf_warnings = (
        _resolve_counterfactual_runtime(
            use_segment_collector=args.use_segment_collector,
            counterfactual_scoring=args.counterfactual_scoring,
            counterfactual_weight=args.counterfactual_weight,
        )
    )
    for warning_msg in cf_warnings:
        logger.warning("Config: %s", warning_msg)
    if args.ppo_minibatch != 32:
        logger.warning(
            "Config: ppo_minibatch=%d is outside the current ACT1 first-win profile (recommended: 32).",
            args.ppo_minibatch,
        )
    if args.use_segment_collector or args.counterfactual_scoring or args.kl_warmstart:
        logger.warning(
            "Config: running outside the current ACT1 first-win profile "
            "(segment=%s counterfactual=%s kl_warmstart=%s).",
            args.use_segment_collector,
            args.counterfactual_scoring,
            args.kl_warmstart,
        )
    if args.boss_entry_quality_weight != 0.0:
        logger.warning(
            "Config: Step 2 / Phase 5 boss-entry quality milestone enabled "
            "(weight=%.2f, max bonus ~%.2f at floor crossing 14->15+, "
            "early-dmg-pot penalty weight=%.2f)",
            args.boss_entry_quality_weight,
            0.95 * args.boss_entry_quality_weight,
            args.early_damage_potion_penalty_weight,
        )

    vocab = load_vocab()
    env_ports = [args.start_port + i for i in range(args.num_envs)]
    env_urls = [f"http://127.0.0.1:{p}" for p in env_ports]

    # Output
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_dir)
    run_dir_name = _build_run_output_dir_name(
        num_envs=args.num_envs,
        timestamp=timestamp,
        run_tag=args.run_tag,
    )
    output_dir = output_root / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = vars(args).copy()
    for _key, _value in list(config_payload.items()):
        if isinstance(_value, Path):
            config_payload[_key] = str(_value)
    config_payload["output_root"] = str(output_root)
    config_payload["run_dir_name"] = run_dir_name
    config_payload["run_tag_sanitized"] = _sanitize_run_tag(args.run_tag)
    config_payload["effective_counterfactual_scoring"] = effective_counterfactual_scoring
    config_payload["effective_counterfactual_weight"] = effective_counterfactual_weight
    if args.config:
        config_payload["config"] = str(Path(args.config).resolve())
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    metrics_log = output_dir / "metrics.jsonl" if args.save_metrics_log else None
    metrics_history: list[dict[str, Any]] = []
    health_monitor = TrainingHealthMonitor()
    health_check_interval = 25

    # Offline non-combat ranking data mainly shapes card/shop/campfire preference.
    # It complements live PPO data instead of replacing it.
    matchup_dataset = None
    if args.matchup_data_dir:
        from search.matchup_dataset import MatchupRankingDataset
        matchup_dataset = MatchupRankingDataset(
            args.matchup_data_dir,
            min_spread=args.matchup_min_spread,
        )
        logger.info("Offline non-combat ranking dataset: %d samples from %s (filtered %d with spread < %.4f)",
                     len(matchup_dataset), args.matchup_data_dir,
                     matchup_dataset._filtered_count, args.matchup_min_spread)
        if len(matchup_dataset) > 0:
            stats = matchup_dataset.get_stats()
            logger.info("  avg_options=%.1f score_spread=%.4f skip_best=%.1f%%",
                         stats.get("avg_options", 0),
                         stats.get("score_std", 0),
                         stats.get("skip_best_rate", 0) * 100)

    # Offline combat teacher data is separate from live combat PPO/MCTS data.
    # Keep its batch/update schedule explicit so dataset growth does not
    # silently reduce its influence on the combat network.
    combat_teacher_dataset = None
    if args.combat_teacher_data_dir:
        from search.combat_teacher_dataset import load_combat_teacher_samples
        from search.train_combat_teacher import CombatTeacherTorchDataset
        ct_path = Path(args.combat_teacher_data_dir)
        ct_samples: list = []
        if ct_path.is_file():
            ct_samples = load_combat_teacher_samples(ct_path)
        elif ct_path.is_dir():
            for jsonl_file in sorted(ct_path.glob("*.jsonl")):
                ct_samples.extend(load_combat_teacher_samples(jsonl_file))
        if ct_samples:
            # Only use train-split samples
            ct_samples = [s for s in ct_samples if str(s.split or "train") == "train"]
            combat_teacher_dataset = CombatTeacherTorchDataset(ct_samples, vocab=vocab)
            logger.info("Offline combat teacher dataset: %d samples from %s",
                         len(combat_teacher_dataset), args.combat_teacher_data_dir)
        else:
            logger.warning("Offline combat teacher dataset: 0 samples found in %s", args.combat_teacher_data_dir)

    # Offline data saver
    episode_saver = None
    if args.save_offline_data:
        offline_dir = output_dir / "offline_data"
        episode_saver = EpisodeDataSaver(
            output_dir=offline_dir,
            min_floor=args.offline_min_floor,
        )
        logger.info("Saved offline episodes: floor >= %d → %s", args.offline_min_floor, offline_dir)

    training_source_summary = _training_data_source_summary(
        args=args,
        effective_counterfactual_scoring=effective_counterfactual_scoring,
        effective_counterfactual_weight=effective_counterfactual_weight,
        matchup_dataset_size=len(matchup_dataset) if matchup_dataset is not None else 0,
        combat_teacher_dataset_size=len(combat_teacher_dataset) if combat_teacher_dataset is not None else 0,
    )
    _write_training_flow_snapshot(output_dir, training_source_summary)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info("Device: %s (AMP: %s)", device, use_amp)

    # For the STS2AI mainline, champion checkpoints are authoritative. If a
    # resume checkpoint carries retrieval-head metadata, automatically adopt the
    # matching retrieval architecture instead of requiring the caller to repeat
    # those flags on every invocation.
    use_symbolic_features = getattr(args, "retrieval_head", False)
    symbolic_proj_dim = getattr(args, "retrieval_proj_dim", 16)
    _resume_sources = [
        ("--resume", args.resume),
        ("--resume-ppo", args.resume_ppo),
        ("--resume-mcts", args.resume_mcts),
    ]
    for _arg_name, _ckpt_path in _resume_sources:
        if not _ckpt_path:
            continue
        try:
            _peek_ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
        except Exception as _e:
            logger.warning("Could not inspect %s for retrieval-head metadata: %s", _ckpt_path, _e)
            continue
        _ckpt_retrieval_proj_dim = _checkpoint_retrieval_proj_dim(_peek_ckpt)
        if _ckpt_retrieval_proj_dim > 0 and not use_symbolic_features:
            use_symbolic_features = True
            symbolic_proj_dim = _ckpt_retrieval_proj_dim
            logger.info(
                "Auto-enabled retrieval-head from %s checkpoint metadata (proj_dim=%d).",
                _arg_name,
                _ckpt_retrieval_proj_dim,
            )
        if _ckpt_retrieval_proj_dim > 0 and symbolic_proj_dim != _ckpt_retrieval_proj_dim:
            raise SystemExit(
                f"{_arg_name} checkpoint '{_ckpt_path}' expects retrieval proj_dim="
                f"{_ckpt_retrieval_proj_dim}, but this run requested "
                f"--retrieval-proj-dim {symbolic_proj_dim}. Re-run with "
                f"--retrieval-proj-dim {_ckpt_retrieval_proj_dim}."
            )

    # Load networks with shared embeddings
    ppo_net = FullRunPolicyNetworkV2(
        vocab=vocab,
        embed_dim=args.embed_dim,
        use_symbolic_features=use_symbolic_features,
        symbolic_proj_dim=symbolic_proj_dim,
        offline_noncombat_ranking_head_mode=getattr(
            args,
            "offline_noncombat_ranking_head_mode",
            "mlp",
        ),
    )
    deck_repr_dim = getattr(args, "deck_repr_dim", 0)

    # Auto-detect deck_repr_dim from checkpoint if --resume is provided so we
    # build a network architecture that matches the checkpoint exactly. Without
    # this, --resume of a deck-aware checkpoint silently drops the deck/pile
    # encoders and the state_encoder mismatches.
    if args.resume and deck_repr_dim == 0:
        try:
            _peek_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            _peek_state = _peek_ckpt.get("mcts_model", {})
            if isinstance(_peek_state, dict):
                for _k, _v in _peek_state.items():
                    if "deck_encoder" in _k and "attn.in_proj_weight" in _k:
                        deck_repr_dim = int(_v.shape[0]) // 3
                        logger.info("Auto-detected deck_repr_dim=%d from checkpoint %s",
                                    deck_repr_dim, args.resume)
                        break
            del _peek_ckpt, _peek_state
        except Exception as _e:
            logger.warning("Could not peek checkpoint for deck_repr_dim auto-detect: %s", _e)

    mcts_net = CombatPolicyValueNetwork(
        vocab=vocab, embed_dim=args.embed_dim,
        hidden_dim=args.combat_hidden_dim,
        entity_embeddings=ppo_net.entity_emb,  # shared embeddings
        deck_repr_dim=deck_repr_dim,
        residual_adapter=getattr(args, "residual_adapter", False),
        symbolic_head=ppo_net.symbolic_head,  # shared (may be None)
    )
    if use_symbolic_features:
        sym_param_count = sum(p.numel() for p in ppo_net.symbolic_head.parameters())
        logger.info(
            "SymbolicFeaturesHead enabled: %d trainable params (owned by PPO optimizer, "
            "combat optimizer will exclude symbolic_head.* to avoid double-step)",
            sym_param_count,
        )
        if getattr(args, "freeze_embeddings", False):
            logger.info(
                "SymbolicFeaturesHead: query side (entity_emb) is frozen via "
                "--freeze-embeddings; symbol side (symbol_embed + cross_attn) "
                "remains trainable."
            )
    start_iter = 0

    def _safe_load_state_dict(model, state_dict, label="model"):
        """Load state dict, handling shape mismatches with partial copy.

        For Linear weights where model dim > checkpoint dim (e.g. deck_repr_dim
        expansion), copies checkpoint columns into the first N columns of the
        model weight and zero-inits the rest.
        """
        current = model.state_dict()
        filtered = {}
        skipped = []
        partial = []
        for k, v in state_dict.items():
            if k in current and current[k].shape == v.shape:
                filtered[k] = v
            elif k in current and v.dim() == 2 and current[k].dim() == 2:
                # Linear weight: (out, in_new) vs (out, in_old)
                if current[k].shape[0] == v.shape[0] and current[k].shape[1] > v.shape[1]:
                    # Partial copy: old columns + zero-init new columns
                    new_w = torch.zeros_like(current[k])
                    new_w[:, :v.shape[1]] = v
                    filtered[k] = new_w
                    partial.append(f"{k}: {list(v.shape)}->{list(current[k].shape)}")
                else:
                    skipped.append(f"{k}: ckpt={list(v.shape)} vs model={list(current[k].shape)}")
            elif k in current:
                skipped.append(f"{k}: ckpt={list(v.shape)} vs model={list(current[k].shape)}")
        if partial:
            logger.info("Partial-loaded %d expanded params in %s: %s", len(partial), label, "; ".join(partial[:5]))
        if skipped:
            logger.warning("Skipped %d mismatched keys in %s: %s", len(skipped), label, "; ".join(skipped[:5]))
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            logger.info("New params in %s (randomly init): %d keys", label, len(missing))

    # --resume: load hybrid checkpoint (both networks)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "ppo_model" in ckpt:
            _safe_load_state_dict(ppo_net, ckpt["ppo_model"], "PPO")
            logger.info("Loaded PPO from hybrid checkpoint")
        if "mcts_model" in ckpt:
            _safe_load_state_dict(mcts_net, ckpt["mcts_model"], "combat")
            logger.info("Loaded combat policy from hybrid checkpoint")
        start_iter = ckpt.get("iteration", 0) + 1

    # --resume-ppo / --resume-mcts: load standalone checkpoints (override)
    if args.resume_ppo:
        ckpt = torch.load(args.resume_ppo, map_location="cpu", weights_only=False)
        if "ppo_model" in ckpt:
            _safe_load_state_dict(ppo_net, ckpt["ppo_model"], "PPO")
        elif "model_state_dict" in ckpt:
            _safe_load_state_dict(ppo_net, ckpt["model_state_dict"], "PPO")
        logger.info("Loaded PPO from %s", args.resume_ppo)

    offline_noncombat_ranking_head_mode = _configure_offline_noncombat_ranking_head_mode(
        ppo_net,
        getattr(args, "offline_noncombat_ranking_head_mode", "mlp"),
    )
    logger.info(
        "Offline non-combat ranking head mode: %s",
        offline_noncombat_ranking_head_mode,
    )

    if args.resume_mcts:
        ckpt = torch.load(args.resume_mcts, map_location="cpu", weights_only=False)
        if "mcts_model" in ckpt:
            _safe_load_state_dict(mcts_net, ckpt["mcts_model"], "combat")
        elif "model_state_dict" in ckpt:
            _safe_load_state_dict(mcts_net, ckpt["model_state_dict"], "combat")
        logger.info("Loaded combat policy from %s", args.resume_mcts)

    combat_main_path_mode = _configure_main_combat_path_mode(
        mcts_net,
        getattr(args, "combat_main_path_mode", "mlp"),
    )
    logger.info("Combat main rollout path mode: %s", combat_main_path_mode)

    # Initialize combat deck_encoder from PPO deck_encoder (transfer learned representation)
    if deck_repr_dim > 0 and hasattr(mcts_net, 'deck_encoder') and hasattr(ppo_net, 'deck_encoder'):
        try:
            ppo_deck_sd = {k: v for k, v in ppo_net.deck_encoder.state_dict().items()}
            combat_deck_sd = mcts_net.deck_encoder.state_dict()
            matched = 0
            for k in combat_deck_sd:
                if k in ppo_deck_sd and combat_deck_sd[k].shape == ppo_deck_sd[k].shape:
                    combat_deck_sd[k] = ppo_deck_sd[k]
                    matched += 1
            if matched > 0:
                mcts_net.deck_encoder.load_state_dict(combat_deck_sd)
                logger.info("Initialized combat deck_encoder from PPO deck_encoder (%d/%d params copied)",
                            matched, len(combat_deck_sd))
        except Exception as e:
            logger.debug("Could not copy deck_encoder weights: %s", e)

    if args.boss_aware_warmup_only:
        trainable_params, total_params = _configure_boss_aware_warmup(ppo_net)
        logger.info(
            "Boss-aware warmup: training only new PPO boss-aware params (%d / %d trainable)",
            trainable_params,
            total_params,
        )

    ppo_net.to(device)
    mcts_net.to(device)

    ppo_trainer = PPOTrainerV2(
        network=ppo_net, lr=args.ppo_lr, ppo_epochs=args.ppo_epochs,
        minibatch_size=args.ppo_minibatch, entropy_coeff=args.ppo_entropy_coeff,
        clip_epsilon=args.ppo_clip, max_grad_norm=1.0,
        boss_readiness_coeff=args.boss_readiness_coeff,
        target_kl=args.target_kl,
    )

    requested_combat_mcts_backend = str(getattr(args, "combat_mcts_backend", "python") or "python").strip().lower()
    initial_transport = args.transport or ("pipe-binary" if args.pipe else "http")
    combat_mcts_backend = requested_combat_mcts_backend
    if requested_combat_mcts_backend == "csharp" and initial_transport != "pipe-binary":
        logger.warning(
            "combat_mcts_backend=csharp requires pipe-binary transport; falling back to python backend for transport=%s",
            initial_transport,
        )
        combat_mcts_backend = "python"

    mcts_config = MCTSConfig(num_simulations=args.mcts_sims, c_puct=1.5,
                              temperature=1.0, dirichlet_alpha=0.3, dirichlet_fraction=0.25)
    mcts_agent = CombatMCTSAgent(network=mcts_net, vocab=vocab, config=mcts_config,
                                  training=True, device=device, ppo_net=ppo_net,
                                  backend=combat_mcts_backend,
                                  use_continuation_value=bool(args.combat_mcts_continuation_value))
    # Exclude shared symbolic_head params from the combat optimizer — the PPO
    # optimizer owns them. Combat's backward still accumulates gradients on
    # those params via autograd; they are consumed at PPO step time. This
    # avoids two Adam states updating the same parameter with independent
    # moving averages, which would cause thrashing.
    def _combat_trainable_params():
        return [
            p for n, p in mcts_net.named_parameters()
            if not n.startswith("symbolic_head.")
        ]
    if use_symbolic_features:
        excluded_combat = sum(
            p.numel() for n, p in mcts_net.named_parameters()
            if n.startswith("symbolic_head.")
        )
        logger.info(
            "SymbolicFeaturesHead: excluded %d params from combat optimizer "
            "(owned by PPO optimizer instead)", excluded_combat,
        )
    mcts_optimizer = torch.optim.Adam(
        _combat_trainable_params() if use_symbolic_features else mcts_net.parameters(),
        lr=args.mcts_lr,
        weight_decay=1e-4,
    )
    mcts_replay = MCTSReplayBuffer(max_size=args.mcts_replay_size)

    # Combat PPO trainer (shares mcts_net, uses its own optimizer with separate lr)
    # NOTE: CombatPPOTrainer builds its own optimizer from network.parameters()
    # in its __init__. When symbolic features are enabled we need to rebuild
    # that optimizer with the filtered param list. Done immediately below.
    combat_ppo_trainer = CombatPPOTrainer(
        network=mcts_net,
        lr=args.combat_ppo_lr,
        clip_epsilon=args.combat_ppo_clip,
        entropy_coeff=args.combat_ppo_entropy_coeff,
        ppo_epochs=args.combat_ppo_epochs,
        minibatch_size=args.combat_ppo_minibatch,
        target_kl=args.combat_target_kl,
    )
    if use_symbolic_features:
        # Swap CombatPPOTrainer's internal optimizer to exclude symbolic_head.*
        # See _combat_trainable_params() above for the rationale. Match the
        # Adam defaults from CombatPPOTrainer.__init__ (line ~727).
        combat_ppo_trainer.optimizer = torch.optim.Adam(
            _combat_trainable_params(),
            lr=args.combat_ppo_lr,
        )

    # Residual adapter: freeze backbone, only train adapter heads
    if getattr(args, "residual_adapter", False) and deck_repr_dim > 0:
        adapter_prefixes = ("deck_encoder.", "delta_logits_head.", "delta_value_head.",
                            "adapter_alpha", "adapter_beta")
        frozen_count = 0
        trainable_count = 0
        for name, param in mcts_net.named_parameters():
            if any(name.startswith(p) for p in adapter_prefixes):
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        logger.info("Residual adapter: frozen %d params, trainable %d params (adapter only)",
                    frozen_count, trainable_count)

    # Freeze shared entity embeddings (GPT Pro: prevent combat/ranking gradient conflict)
    if getattr(args, "freeze_embeddings", False):
        frozen_emb = 0
        for name, param in ppo_net.named_parameters():
            if name.startswith("entity_emb."):
                param.requires_grad = False
                frozen_emb += param.numel()
        # Combat net shares entity_emb, so freezing ppo_net's also freezes combat's
        logger.info("Frozen shared entity embeddings: %d params", frozen_emb)

    # Freeze entire combat brain (splice diagnostic finding: combat475 > combat600)
    if getattr(args, "freeze_combat", False):
        frozen_combat = 0
        for name, param in mcts_net.named_parameters():
            param.requires_grad = False
            frozen_combat += param.numel()
        logger.info("Frozen entire combat brain: %d params (PPO-only training)", frozen_combat)

    # Freeze entire PPO brain (combat-only training to improve boss conversion)
    if getattr(args, "freeze_ppo", False):
        frozen_ppo = 0
        for name, param in ppo_net.named_parameters():
            if not name.startswith("entity_emb."):  # embeddings already handled above
                param.requires_grad = False
                frozen_ppo += param.numel()
        logger.info("Frozen entire PPO brain: %d params (combat-only training)", frozen_ppo)

    logger.info("PPO params: %d | MCTS params: %d | Envs: %d",
                ppo_net.param_count(), mcts_net.param_count(), len(env_ports))

    # Pipe clients — reuse from PipeBackedFullRunClient (single session per port)
    transport = initial_transport
    use_pipe_transport = transport in {"pipe", "pipe-binary"}
    pipe_clients: dict[int, Any] = {}
    spawned_env_procs: list[Any] = []

    logger.info("Starting hybrid training from iter %d (output: %s)", start_iter, output_dir)

    def _cleanup_spawned_envs() -> None:
        while spawned_env_procs:
            stop_process(spawned_env_procs.pop())

    if args.auto_launch:
        launch_protocol = transport_launch_protocol(transport)
        if launch_protocol is None:
            logger.warning("--auto-launch is only supported for pipe transports; ignoring for transport=%s", transport)
        else:
            atexit.register(_cleanup_spawned_envs)
            logger.info(
                "Auto-launching %d fresh Sim hosts from %s on ports %s (%s)",
                len(env_ports),
                Path(args.headless_dll).resolve(),
                ",".join(str(port) for port in env_ports),
                launch_protocol,
            )
            host_manager = SimHostLifecycleManager(
                ports=list(env_ports),
                transport=transport,
                auto_launch=True,
                repo_root=args.repo_root,
                dll_path=args.headless_dll,
            )
            spawned_env_procs.extend(host_manager.start())

    # Create persistent clients (one per env). PipeBackedFullRunClient
    # owns the pipe connection; MCTS reuses it via client._pipe.
    env_clients: dict[int, Any] = {}
    if not args.multi_process:
        for port in env_ports:
            if use_pipe_transport:
                client = create_full_run_client(
                    port=port,
                    use_pipe=True,
                    transport=transport,
                    ready_timeout_s=15.0,
                    auto_launch=bool(args.auto_launch),
                    repo_root=str(args.repo_root),
                    dll_path=str(args.headless_dll),
                )
                client._ensure_connected()
                env_clients[port] = client
                pipe_clients[port] = client._pipe  # share pipe for MCTS
                logger.info("Pipe client ready: port %d transport=%s", port, transport)
            else:
                url = f"http://127.0.0.1:{port}"
                env_clients[port] = ApiBackedFullRunClient(
                    base_url=url, poll_interval_s=0.005, request_timeout_s=60.0)

    if use_pipe_transport and not args.multi_process and not pipe_clients:
        logger.error("No pipe connections!")
        _cleanup_spawned_envs()
        return 1

    # --- Load ORT model into each HeadlessSim (if --local-ort) ---
    need_sim_ort = bool(args.local_ort or combat_mcts_backend == "csharp")
    if need_sim_ort and args.ort_model_path and use_pipe_transport:
        import os
        ort_abs = os.path.abspath(args.ort_model_path)
        for port, raw_pipe in pipe_clients.items():
            try:
                result = raw_pipe.call("load_ort_model", {"path": ort_abs})
                loaded = result.get("loaded", False)
                logger.info("ORT model loaded on port %d: %s", port, "OK" if loaded else "FAILED")
            except Exception as e:
                logger.warning("ORT model load failed on port %d: %s", port, e)
    elif combat_mcts_backend == "csharp" and use_pipe_transport:
        logger.warning(
            "combat_mcts_backend=csharp is enabled but --ort-model-path was not provided; simulator search will only work if the host already has an ORT model loaded."
        )

    # --- Multi-process batch inference setup ---
    inf_server = None
    inf_clients: dict[int, Any] = {}
    mp_ctx = None
    mp_task_queues: list[Any] = []
    mp_result_queue = None
    mp_workers: list[Any] = []
    mp_ppo_onnx_path: str | None = None
    mp_combat_onnx_path: str | None = None
    if args.multi_process and not args.local_ort:
        mp_ctx = mp.get_context("spawn")
        mp_result_queue = mp_ctx.Queue()
        try:
            import copy
            _onnx_export_net = copy.deepcopy(ppo_net).cpu().eval()
            mp_ppo_onnx_path = str(output_dir / "ppo_actor_worker.onnx")
            _export_ppo_actor_onnx(_onnx_export_net, mp_ppo_onnx_path, vocab)
            logger.info("Multi-process collector: exported PPO actor ONNX to %s", mp_ppo_onnx_path)
        except Exception as mp_ort_err:
            logger.warning("Multi-process collector ORT export failed, workers will use PyTorch CPU: %s", mp_ort_err)
            mp_ppo_onnx_path = None
        try:
            from export_actor_onnx import export_from_training_snapshot
            mp_combat_onnx_path = str(output_dir / "combat_actor_worker.onnx")
            export_from_training_snapshot(
                ppo_net.state_dict(),
                mcts_net.state_dict(),
                vocab,
                mp_combat_onnx_path,
                export_mode="legacy",
                include_continuation=False,
            )
            logger.info("Multi-process collector: exported combat actor ONNX to %s", mp_combat_onnx_path)
        except Exception as mp_combat_ort_err:
            logger.warning("Multi-process combat ORT export failed, workers will use PyTorch CPU: %s", mp_combat_ort_err)
            mp_combat_onnx_path = None
        worker_config = {
            "transport": transport,
            "character_id": args.character_id,
            "episode_timeout": args.episode_timeout,
            "max_episode_steps": args.max_episode_steps,
            "boss_entry_quality_weight": args.boss_entry_quality_weight,
            "early_damage_potion_penalty_weight": args.early_damage_potion_penalty_weight,
            "boss_conditioned_card_guidance_weight": args.boss_conditioned_card_guidance_weight,
            "combat_safety_rerank_weight": args.combat_safety_rerank_weight,
            "embed_dim": args.embed_dim,
            "combat_hidden_dim": args.combat_hidden_dim,
            "use_symbolic_features": use_symbolic_features,
            "symbolic_proj_dim": symbolic_proj_dim,
            "offline_noncombat_ranking_head_mode": offline_noncombat_ranking_head_mode,
            "combat_main_path_mode": combat_main_path_mode,
            "deck_repr_dim": deck_repr_dim,
            "residual_adapter": getattr(args, "residual_adapter", False),
            "combat_mcts_backend": combat_mcts_backend,
            "combat_mcts_continuation_value": bool(args.combat_mcts_continuation_value),
            "mcts_sims": args.mcts_sims,
            "mcts": bool(args.mcts),
            "mcts_first_n_actions_per_turn": args.mcts_first_n_actions_per_turn,
            "mcts_full_search_on_elite_boss": args.mcts_full_search_on_elite_boss,
            "act1_no_elite_routes": args.act1_no_elite_routes,
            "combat_pending_stall_threshold": args.combat_pending_stall_threshold,
            "use_segment_collector": args.use_segment_collector,
            "counterfactual_scoring": effective_counterfactual_scoring,
            "counterfactual_weight": effective_counterfactual_weight,
            "screen_local_delta": args.screen_local_delta,
            "deterministic_policy": args.deterministic_policy,
            "build_mode": getattr(args, "build_mode", False),
            "ppo_onnx_path": mp_ppo_onnx_path,
            "combat_onnx_path": mp_combat_onnx_path,
        }
        ppo_worker_state = {k: v.detach().cpu() for k, v in ppo_net.state_dict().items()}
        mcts_worker_state = {k: v.detach().cpu() for k, v in mcts_net.state_dict().items()}
        for i, port in enumerate(env_ports):
            task_q = mp_ctx.Queue()
            worker = mp_ctx.Process(
                target=_mp_episode_worker,
                args=(i, port, task_q, mp_result_queue, worker_config, ppo_worker_state, mcts_worker_state),
                daemon=True,
            )
            worker.start()
            mp_task_queues.append(task_q)
            mp_workers.append(worker)
        logger.info("Multi-process collector enabled (%d worker processes)", len(env_ports))
    elif args.local_ort and args.multi_process:
        # ORT mode: combat in C#, non-combat uses CPU policy snapshot.
        # Zero CUDA in worker threads — eliminates multi-thread CUDA contention.
        logger.info("Local ORT mode: combat in C#, non-combat CPU snapshot (%d workers)", len(env_ports))

    def _collect(env_idx: int,
                 combat_buf: CombatRolloutBuffer | None = None):
        port = env_ports[env_idx % len(env_ports)]
        episode_seed = None if args.seed is None else f"audit-{args.seed}-iter{iteration:05d}-ep{env_idx:03d}"
        client = env_clients.get(port)
        # Skip dead envs immediately
        if client is not None and hasattr(client, 'is_dead') and client.is_dead:
            return StructuredRolloutBuffer(), [], {"error": f"port {port} is dead", "floors": 0}
        pipe_getter = (lambda c=client: getattr(c, "_pipe", None)) if hasattr(client, "_pipe") else None
        pipe = pipe_getter if pipe_getter is not None else pipe_clients.get(port)
        # Use inference client for multi-process mode
        inf_client = inf_clients.get(env_idx % len(env_ports)) if args.multi_process else None
        # Use CPU snapshots if zero-CUDA mode (no CUDA in worker threads)
        if _use_zero_cuda:
            _ppo = _cpu_ppo_net
            _mcts = _cpu_mcts_agent
            _model_forward_lock = _collector_forward_lock
        elif _collector_ppo_nets is not None and _collector_mcts_agents is not None:
            _ppo = _collector_ppo_nets[env_idx % len(_collector_ppo_nets)]
            _mcts = _collector_mcts_agents[env_idx % len(_collector_mcts_agents)]
            _model_forward_lock = _collector_forward_lock
        else:
            _ppo = ppo_net
            _mcts = mcts_agent
            _model_forward_lock = _collector_forward_lock
        return collect_unified_episode(
            _ppo, _mcts, vocab, pipe, client,
            character_id=args.character_id,
            seed=episode_seed,
            episode_timeout=args.episode_timeout,
            max_steps=args.max_episode_steps,
            use_mcts_combat=args.mcts,
            mcts_warmup_active=bool(args.mcts and iteration < args.mcts_warmup_iters),
            mcts_first_n_actions_per_turn=args.mcts_first_n_actions_per_turn,
            mcts_full_search_on_elite_boss=args.mcts_full_search_on_elite_boss,
            act1_no_elite_routes=args.act1_no_elite_routes,
            combat_pending_stall_threshold=args.combat_pending_stall_threshold,
            combat_buffer=combat_buf,
            inference_client=inf_client,
            use_segment_collector=args.use_segment_collector,
            counterfactual_scoring=effective_counterfactual_scoring,
            counterfactual_weight=effective_counterfactual_weight,
            screen_local_delta=args.screen_local_delta,
            deterministic_policy=args.deterministic_policy,
            episode_saver=episode_saver,
            use_local_ort=args.local_ort,
            ppo_ort_session=_ppo_ort_session,
            boss_entry_quality_weight=args.boss_entry_quality_weight,
            early_damage_potion_penalty_weight=args.early_damage_potion_penalty_weight,
            boss_conditioned_card_guidance_weight=args.boss_conditioned_card_guidance_weight,
            combat_safety_rerank_weight=args.combat_safety_rerank_weight,
            build_mode=getattr(args, "build_mode", False),
            model_forward_lock=_model_forward_lock,
        )

    # Load Skada community priors (card quality, synergies, boss difficulty)
    _skada_priors_obj = None
    if args.skada_prior_weight > 0 or args.skada_boss_weights:
        try:
            from skada.skada_priors import SkadaPriors
            _skada_priors_obj = SkadaPriors(args.skada_db)
            if _skada_priors_obj.loaded:
                logger.info("Skada priors loaded: %d cards, %d relics, %d synergies, %d bosses",
                            _skada_priors_obj.num_cards, _skada_priors_obj.num_relics,
                            _skada_priors_obj.num_synergies, _skada_priors_obj.num_bosses)
                if args.skada_boss_weights:
                    from core.rl_reward_shaping import load_skada_boss_difficulty
                    load_skada_boss_difficulty(_skada_priors_obj)
            else:
                logger.warning("Skada DB not found — skada priors disabled")
                _skada_priors_obj = None
        except Exception as e:
            logger.warning("Failed to load Skada priors: %s", e)
            _skada_priors_obj = None

    # Register learned card evaluator for counterfactual scoring
    if effective_counterfactual_scoring or args.matchup_blend_beta > 0 or args.skada_prior_weight > 0:
        from search.counterfactual_scoring import set_learned_evaluator
        # Blend alpha ramps up over training: start with heuristic, gradually trust learned
        _initial_alpha = 0.3 if start_iter > 200 else 0.0
        _matchup_beta = args.matchup_blend_beta
        _skada_gamma = args.skada_prior_weight if _skada_priors_obj is not None else 0.0
        set_learned_evaluator(
            ppo_net, vocab,
            alpha=_initial_alpha,
            matchup_beta=_matchup_beta,
            skada_priors=_skada_priors_obj,
            skada_gamma=_skada_gamma,
        )
        logger.info("Learned card evaluator registered (alpha=%.2f, matchup_beta=%.2f, skada_gamma=%.2f)",
                     _initial_alpha, _matchup_beta, _skada_gamma)

    # --- Zero-CUDA collector: CPU policy snapshot for worker threads ---
    _cpu_ppo_net = None
    _cpu_mcts_net = None
    _cpu_mcts_agent = None
    _collector_ppo_nets = None
    _collector_mcts_nets = None
    _collector_mcts_agents = None
    _use_zero_cuda = args.zero_cuda_collector and args.multi_process
    _ppo_ort_session = None  # ORT CPU session for non-combat (Branch C)
    _collector_forward_lock: threading.Lock | None = None
    if len(env_ports) > 1 and not args.multi_process:
        _collector_forward_lock = threading.Lock()
    if _use_zero_cuda:
        import copy
        _cpu_ppo_net = copy.deepcopy(ppo_net).cpu().eval()
        _cpu_mcts_net = copy.deepcopy(mcts_net).cpu().eval()
        _cpu_mcts_agent = CombatMCTSAgent(
            network=_cpu_mcts_net, vocab=vocab, config=mcts_config,
            training=False, device=torch.device("cpu"), ppo_net=_cpu_ppo_net,
            backend=combat_mcts_backend,
            use_continuation_value=bool(args.combat_mcts_continuation_value))

        # Export PPO actor to ONNX for ORT CPU inference (Branch C)
        try:
            import onnxruntime as ort
            _ppo_onnx_path = str(output_dir / "ppo_actor.onnx")
            _export_ppo_actor_onnx(_cpu_ppo_net, _ppo_onnx_path, vocab)
            _ppo_ort_opts = ort.SessionOptions()
            _ppo_ort_opts.intra_op_num_threads = 1
            _ppo_ort_opts.inter_op_num_threads = 1
            _ppo_ort_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            _ppo_ort_session = ort.InferenceSession(_ppo_onnx_path, _ppo_ort_opts, providers=["CPUExecutionProvider"])
            logger.info("Zero-CUDA collector: PPO ORT CPU session created (0.5ms vs PyTorch 6.5ms)")
        except Exception as _ort_err:
            logger.warning("PPO ORT export failed, falling back to CPU PyTorch: %s", _ort_err)
    elif len(env_ports) > 1 and not args.multi_process:
        import copy
        _collector_ppo_nets = []
        _collector_mcts_nets = []
        _collector_mcts_agents = []
        for _ in env_ports:
            _ppo_copy = copy.deepcopy(ppo_net).cpu().eval()
            _mcts_copy = copy.deepcopy(mcts_net).cpu().eval()
            _collector_ppo_nets.append(_ppo_copy)
            _collector_mcts_nets.append(_mcts_copy)
            _collector_mcts_agents.append(
                CombatMCTSAgent(
                    network=_mcts_copy,
                    vocab=vocab,
                    config=mcts_config,
                    training=False,
                    device=torch.device("cpu"),
                    ppo_net=_ppo_copy,
                    backend=combat_mcts_backend,
                    use_continuation_value=bool(args.combat_mcts_continuation_value),
                )
            )
        logger.info(
            "Per-env CPU collector snapshots created for %d envs (avoid shared-model threaded inference)",
            len(env_ports),
        )
        logger.info(
            "Collector model forward lock enabled for %d envs (serialize PyTorch attention across worker threads)",
            len(env_ports),
        )

    _mp_worker_ort_refresh_interval = 25
    _mp_worker_active_ppo_onnx_path = mp_ppo_onnx_path
    _mp_worker_active_combat_onnx_path = mp_combat_onnx_path

    def _remove_onnx_artifact_set(onnx_path: str | None) -> None:
        if not onnx_path:
            return
        for _suffix in ("", ".data"):
            try:
                Path(f"{onnx_path}{_suffix}").unlink(missing_ok=True)
            except Exception:
                pass

    def _broadcast_mp_worker_refresh(iteration: int) -> None:
        nonlocal _mp_worker_active_ppo_onnx_path, _mp_worker_active_combat_onnx_path
        if not (args.multi_process and not args.local_ort):
            return
        if not mp_task_queues or mp_result_queue is None:
            return

        refresh_task: dict[str, Any] = {
            "cmd": "refresh_weights",
            "iteration": int(iteration),
            "ppo_state_dict": _snapshot_state_dict_cpu(ppo_net),
            "mcts_state_dict": _snapshot_state_dict_cpu(mcts_net),
        }

        has_worker_ort = bool(mp_ppo_onnx_path or mp_combat_onnx_path)
        should_reload_worker_ort = has_worker_ort and (
            iteration > start_iter and iteration % _mp_worker_ort_refresh_interval == 0
        )

        if has_worker_ort:
            if should_reload_worker_ort:
                try:
                    import copy
                    from export_actor_onnx import export_from_training_snapshot

                    if mp_ppo_onnx_path:
                        _next_ppo_onnx_path = str(output_dir / f"ppo_actor_worker_iter{iteration:05d}.onnx")
                        _ppo_export_net = copy.deepcopy(ppo_net).cpu().eval()
                        _export_ppo_actor_onnx(_ppo_export_net, _next_ppo_onnx_path, vocab)
                        refresh_task["ppo_onnx_path"] = _next_ppo_onnx_path
                    if mp_combat_onnx_path:
                        _next_combat_onnx_path = str(output_dir / f"combat_actor_worker_iter{iteration:05d}.onnx")
                        export_from_training_snapshot(
                            ppo_net.state_dict(),
                            mcts_net.state_dict(),
                            vocab,
                            _next_combat_onnx_path,
                            export_mode="legacy",
                            include_continuation=False,
                        )
                        refresh_task["combat_onnx_path"] = _next_combat_onnx_path
                    refresh_task["reload_ort"] = True
                except Exception as mp_refresh_ort_err:
                    logger.warning(
                        "Multi-process worker ORT refresh failed iter %d; workers will fall back to PyTorch CPU: %s",
                        iteration,
                        mp_refresh_ort_err,
                    )
                    refresh_task["disable_ort"] = True
                else:
                    _old_ppo_onnx_path = _mp_worker_active_ppo_onnx_path
                    _old_combat_onnx_path = _mp_worker_active_combat_onnx_path
                    _mp_worker_active_ppo_onnx_path = str(refresh_task.get("ppo_onnx_path") or "") or None
                    _mp_worker_active_combat_onnx_path = str(refresh_task.get("combat_onnx_path") or "") or None
                    if _old_ppo_onnx_path and _old_ppo_onnx_path != _mp_worker_active_ppo_onnx_path:
                        _remove_onnx_artifact_set(_old_ppo_onnx_path)
                    if _old_combat_onnx_path and _old_combat_onnx_path != _mp_worker_active_combat_onnx_path:
                        _remove_onnx_artifact_set(_old_combat_onnx_path)
            elif iteration > start_iter:
                # Avoid silently using stale ONNX snapshots between refreshes.
                refresh_task["disable_ort"] = True

        for task_q in mp_task_queues:
            task_q.put(refresh_task)

        pending_workers = set(range(len(mp_task_queues)))
        refresh_failures: dict[int, str] = {}
        deadline = time.monotonic() + max(30.0, float(args.episode_timeout) + 15.0)
        while pending_workers and time.monotonic() < deadline:
            timeout_s = max(1.0, min(10.0, deadline - time.monotonic()))
            try:
                msg = mp_result_queue.get(timeout=timeout_s)
            except Exception:
                break

            if isinstance(msg, dict) and msg.get("type") == "refresh_ack":
                worker_id = _safe_int(msg.get("worker_id"), -1)
                if worker_id >= 0:
                    pending_workers.discard(worker_id)
                    if not bool(msg.get("ok", False)):
                        refresh_failures[worker_id] = str(msg.get("error") or "refresh_failed")
                continue

            if isinstance(msg, tuple) and len(msg) == 4:
                worker_id = _safe_int(msg[0], -1)
                ep_stats = msg[3] if isinstance(msg[3], dict) else {}
                err_text = str(ep_stats.get("error") or "unexpected_worker_result")
                if worker_id >= 0:
                    pending_workers.discard(worker_id)
                    refresh_failures[worker_id] = err_text
                continue

        if pending_workers:
            missing = ",".join(str(wid) for wid in sorted(pending_workers))
            raise RuntimeError(f"multi-process worker refresh ack timeout: workers={missing}")
        if refresh_failures:
            details = ", ".join(f"{wid}:{err}" for wid, err in sorted(refresh_failures.items()))
            raise RuntimeError(f"multi-process worker refresh failed: {details}")

    try:
        end_iter = start_iter + args.max_iterations
        for iteration in range(start_iter, end_iter):
            if _shutdown_requested:
                break
            iter_start = time.monotonic()

            # Refresh CPU snapshots + ORT sessions every iteration (zero-CUDA mode)
            if _use_zero_cuda and _cpu_ppo_net is not None:
                _cpu_ppo_net.load_state_dict({k: v.cpu() for k, v in ppo_net.state_dict().items()})
                _cpu_mcts_net.load_state_dict({k: v.cpu() for k, v in mcts_net.state_dict().items()})
                # Re-export PPO ORT every 25 iter (amortize ~800ms export cost)
                if _ppo_ort_session is not None and iteration % 25 == 0 and iteration > 0:
                    try:
                        _export_ppo_actor_onnx(_cpu_ppo_net, _ppo_onnx_path, vocab)
                        _ppo_ort_session = ort.InferenceSession(_ppo_onnx_path, _ppo_ort_opts, providers=["CPUExecutionProvider"])
                    except Exception:
                        pass
            elif _collector_ppo_nets is not None and _collector_mcts_nets is not None:
                _ppo_sd = {k: v.cpu() for k, v in ppo_net.state_dict().items()}
                _mcts_sd = {k: v.cpu() for k, v in mcts_net.state_dict().items()}
                for _ppo_copy in _collector_ppo_nets:
                    _ppo_copy.load_state_dict(_ppo_sd)
                for _mcts_copy in _collector_mcts_nets:
                    _mcts_copy.load_state_dict(_mcts_sd)

                # Export fresh combat ONNX and hot-reload into C# sims
                # Only export every 25 iterations to amortize overhead (~1-2s per export)
                _ort_refresh_interval = 25
                if args.local_ort and use_pipe_transport and iteration % _ort_refresh_interval == 0:
                    import os
                    from export_actor_onnx import export_from_training_snapshot
                    _onnx_path = str(output_dir / f"actor_v{iteration:05d}.onnx")
                    try:
                        _export_ms = export_from_training_snapshot(
                            ppo_net.state_dict(), mcts_net.state_dict(), vocab, _onnx_path,
                            policy_version=iteration)
                        for _port, _pipe in pipe_clients.items():
                            _pipe.call("load_ort_model", {"path": os.path.abspath(_onnx_path)})
                        if iteration == 0 or iteration % 25 == 0:
                            logger.info("ORT v%d exported (%.0fms) + loaded into %d sims",
                                        iteration, _export_ms, len(pipe_clients))
                        # Clean up old ONNX files (keep only current)
                        if iteration > 0:
                            _old = str(output_dir / f"actor_v{iteration-1:05d}.onnx")
                            try:
                                os.remove(_old)
                            except OSError:
                                pass
                    except Exception as _e:
                        logger.warning("ORT export/reload failed iter %d: %s", iteration, _e)

            if args.multi_process and not args.local_ort:
                _broadcast_mp_worker_refresh(iteration)

            # --- Collect episodes ---

            # Combat: MCTS search (high-quality actions) + Combat PPO (per-step reward learning)
            if iteration == 0:
                if args.mcts:
                    logger.info("Combat search mode: MCTS selects actions, PPO learns value/aux targets")
                else:
                    logger.info("Pure PPO mode: combat NN selects actions directly (no search)")
            ppo_net.eval()
            mcts_net.eval()
            ppo_buffer = StructuredRolloutBuffer()
            combat_buffer = CombatRolloutBuffer()
            _ep_counter = 0
            new_mcts = 0
            total_floors = 0
            total_combats = 0
            victories = 0
            ppo_steps = 0
            mcts_decisions = 0
            combat_ppo_steps = 0
            boss_reached_eps = 0
            act1_cleared_eps = 0
            boss_hp_fracs: list[float] = []
            _iter_ort_combat_time = 0.0
            _iter_ort_combat_calls = 0
            _iter_nc_forward_time = 0.0
            _iter_nc_forward_calls = 0
            _iter_pipe_time = 0.0
            _iter_pipe_calls = 0
            _iter_feature_time = 0.0
            _iter_feature_calls = 0
            _iter_inference_time = 0.0
            _iter_inference_calls = 0
            _iter_buffer_time = 0.0
            _iter_buffer_calls = 0
            deck_sizes_at_boss: list[int] = []
            card_reward_screens = 0
            card_reward_skips = 0
            ep_times = []
            iter_slow_steps = 0
            iter_max_step_ms = 0.0
            iter_timeout_count = 0
            combat_random_warmup_steps = 0
            combat_mcts_turn_limited_steps = 0
            wait_steps_total = 0
            combat_pending_steps_total = 0
            combat_pending_wait_steps_total = 0
            combat_pending_refresh_steps_total = 0
            combat_pending_stall_count = 0
            hard_state_potion_steps = 0
            hard_state_end_turn_steps = 0
            hard_state_repeat_steps = 0
            hard_state_order_steps = 0
            combat_hard_state_weight_sum = 0.0

            def _merge_episode(ep_ppo, ep_mcts, ep_stats):
                """Merge single episode results into iteration-level accumulators."""
                nonlocal new_mcts, total_floors, total_combats, victories
                nonlocal ppo_steps, mcts_decisions, combat_ppo_steps
                nonlocal boss_reached_eps, act1_cleared_eps
                nonlocal boss_hp_fracs, deck_sizes_at_boss, card_reward_screens, card_reward_skips
                nonlocal iter_slow_steps, iter_max_step_ms, iter_timeout_count
                nonlocal combat_random_warmup_steps, combat_mcts_turn_limited_steps
                nonlocal wait_steps_total, combat_pending_steps_total
                nonlocal combat_pending_wait_steps_total, combat_pending_refresh_steps_total
                nonlocal combat_pending_stall_count
                nonlocal hard_state_potion_steps, hard_state_end_turn_steps
                nonlocal hard_state_repeat_steps, hard_state_order_steps
                nonlocal combat_hard_state_weight_sum
                nonlocal _ep_counter
                nonlocal _iter_ort_combat_time, _iter_ort_combat_calls
                nonlocal _iter_nc_forward_time, _iter_nc_forward_calls
                nonlocal _iter_pipe_time, _iter_pipe_calls
                nonlocal _iter_feature_time, _iter_feature_calls
                nonlocal _iter_inference_time, _iter_inference_calls
                nonlocal _iter_buffer_time, _iter_buffer_calls

                # Merge PPO buffer — weight rewards by floor reached
                # Higher floor episodes get amplified rewards (learn from better runs)
                _ep_floor = ep_stats.get("floors", 0)
                _floor_weight = 1.0 + max(0, _ep_floor - 3) * 0.15
                # floor 3: 1.0x, floor 5: 1.3x, floor 8: 1.75x, floor 10: 2.05x
                for i in range(len(ep_ppo)):
                    ppo_buffer.states.append(ep_ppo.states[i])
                    ppo_buffer.actions_data.append(ep_ppo.actions_data[i])
                    ppo_buffer.action_indices.append(ep_ppo.action_indices[i])
                    ppo_buffer.log_probs.append(ep_ppo.log_probs[i])
                    ppo_buffer.rewards.append(ep_ppo.rewards[i] * _floor_weight)
                    ppo_buffer.values.append(ep_ppo.values[i])
                    ppo_buffer.dones.append(ep_ppo.dones[i])
                    ppo_buffer.floor_targets.append(ep_ppo.floor_targets[i])
                    ppo_buffer.boss_readiness_targets.append(ep_ppo.boss_readiness_targets[i])

                # Merge combat PPO buffer
                ep_combat = ep_stats.get("_combat_buffer")
                boss_only = getattr(args, "combat_boss_only", False)
                monster_reward_weight = float(getattr(args, "combat_monster_reward_weight", 1.0))
                boss_screens = {"boss", "elite"}
                if ep_combat is not None:
                    for i in range(len(ep_combat)):
                        # Encounter-gated: skip hallway if --combat-boss-only
                        step_screen = ep_combat.screen_types[i] if ep_combat.screen_types and i < len(ep_combat.screen_types) else ""
                        is_boss_or_elite = step_screen in boss_screens
                        if boss_only and ep_combat.screen_types and not is_boss_or_elite:
                            continue
                        combat_buffer.state_features.append(ep_combat.state_features[i])
                        combat_buffer.action_features.append(ep_combat.action_features[i])
                        combat_buffer.action_indices.append(ep_combat.action_indices[i])
                        combat_buffer.log_probs.append(ep_combat.log_probs[i])
                        # Apply reward weight: monster fights get down-weighted if < 1.0
                        step_reward = ep_combat.rewards[i]
                        if (not boss_only) and (monster_reward_weight != 1.0) and (not is_boss_or_elite) and step_screen:
                            step_reward = step_reward * monster_reward_weight
                        combat_buffer.rewards.append(step_reward)
                        combat_buffer.values.append(ep_combat.values[i])
                        combat_buffer.dones.append(ep_combat.dones[i])
                        if ep_combat.screen_types:
                            combat_buffer.screen_types.append(step_screen)
                        if ep_combat.sample_weights:
                            combat_buffer.sample_weights.append(ep_combat.sample_weights[i])
                        if ep_combat.hard_state_tags:
                            combat_buffer.hard_state_tags.append(ep_combat.hard_state_tags[i])

                # Merge segment buffer (Phase 2) — segment data goes into ppo_buffer
                # by converting segments to step-equivalent entries
                ep_seg = ep_stats.get("_segment_buffer")
                if ep_seg is not None and len(ep_seg) > 0:
                    for seg in ep_seg.segments:
                        ppo_buffer.states.append(seg.state)
                        ppo_buffer.actions_data.append(seg.actions)
                        ppo_buffer.action_indices.append(seg.action_idx)
                        ppo_buffer.log_probs.append(seg.log_prob)
                        ppo_buffer.rewards.append(seg.reward_sum * _floor_weight)
                        ppo_buffer.values.append(seg.value)
                        ppo_buffer.dones.append(seg.done)
                        ppo_buffer.floor_targets.append(seg.floor_target)
                        ppo_buffer.boss_readiness_targets.append(0.0)

                for ex in ep_mcts:
                    mcts_replay.add(ex)
                    new_mcts += 1

                total_floors += ep_stats.get("floors", 0)
                total_combats += ep_stats.get("combats", 0)
                ppo_steps += ep_stats.get("ppo_steps", 0)
                mcts_decisions += ep_stats.get("mcts_decisions", 0)
                combat_ppo_steps += ep_stats.get("combat_ppo_steps", 0)
                combat_random_warmup_steps += ep_stats.get("combat_random_warmup_steps", 0)
                combat_mcts_turn_limited_steps += ep_stats.get("combat_mcts_turn_limited_steps", 0)
                wait_steps_total += ep_stats.get("wait_steps", 0)
                combat_pending_steps_total += ep_stats.get("combat_pending_steps", 0)
                combat_pending_wait_steps_total += ep_stats.get("combat_pending_wait_steps", 0)
                combat_pending_refresh_steps_total += ep_stats.get("combat_pending_refresh_steps", 0)
                combat_pending_stall_count += int(bool(ep_stats.get("combat_pending_stall")))
                hard_state_potion_steps += ep_stats.get("hard_state_potion_decision_steps", 0)
                hard_state_end_turn_steps += ep_stats.get("hard_state_premature_end_turn_steps", 0)
                hard_state_repeat_steps += ep_stats.get("hard_state_repeat_loop_steps", 0)
                hard_state_order_steps += ep_stats.get("hard_state_order_sensitive_steps", 0)
                combat_hard_state_weight_sum += float(ep_stats.get("combat_hard_state_weight_sum", 0.0))
                boss_reached_eps += int(bool(ep_stats.get("boss_reached")))
                act1_cleared_eps += int(bool(ep_stats.get("act1_cleared")))
                boss_hp_fracs.extend(ep_stats.get("boss_hp_fraction_dealt", []))
                deck_sizes_at_boss.extend(ep_stats.get("deck_size_at_boss", []))
                card_reward_screens += ep_stats.get("card_reward_screens", 0)
                card_reward_skips += ep_stats.get("card_reward_skips", 0)

                # Per-episode summary (sample log every 50 iter)
                if iteration % 50 == 0:
                    _fl = ep_stats.get("floors", 0)
                    _cw = ep_stats.get("combats_won", 0)
                    _ct = ep_stats.get("combats", 0)
                    _cards = ep_stats.get("cards_taken", [])
                    _skip = ep_stats.get("cards_skipped", 0)
                    _de = ep_stats.get("death_enemy", "N/A")
                    _hp = ep_stats.get("hp_timeline", [])
                    _out = ep_stats.get("outcome", "?")
                    logger.info("  EP: %s f%d | %dW/%d combats | cards: %s skip:%d | hp:%s | died_to:%s",
                                _out, _fl, _cw, _ct,
                                ",".join(_cards[:5]) if _cards else "none", _skip,
                                "->".join(str(h) for h in _hp[-5:]) if _hp else "?",
                                _de)
                ep_times.append(ep_stats.get("episode_time_s", 0))
                iter_slow_steps += ep_stats.get("slow_steps", 0)
                iter_max_step_ms = max(iter_max_step_ms, ep_stats.get("max_step_ms", 0))
                iter_timeout_count += ep_stats.get("timeout_count", 0)
                if ep_stats.get("outcome") == "victory":
                    victories += 1

                # Accumulate timing stats from zero-CUDA paths
                _iter_ort_combat_time += ep_stats.get("_ort_combat_time", 0.0)
                _iter_ort_combat_calls += ep_stats.get("_ort_combat_calls", 0)
                _iter_nc_forward_time += ep_stats.get("_nc_forward_time", 0.0)
                _iter_nc_forward_calls += ep_stats.get("_nc_forward_calls", 0)
                _iter_pipe_time += ep_stats.get("pipe_time_s", 0.0)
                _iter_pipe_calls += ep_stats.get("pipe_calls", 0)
                _iter_feature_time += ep_stats.get("feature_time_s", 0.0)
                _iter_feature_calls += ep_stats.get("feature_calls", 0)
                _iter_inference_time += ep_stats.get("inference_time_s", 0.0)
                _iter_inference_calls += ep_stats.get("inference_calls", 0)
                _iter_buffer_time += ep_stats.get("buffer_time_s", 0.0)
                _iter_buffer_calls += ep_stats.get("buffer_calls", 0)

                # --- Episode trace: dump EVERY episode to file ---
                trace = ep_stats.get("_episode_trace", [])
                trace_zh = ep_stats.get("_episode_trace_zh", [])
                summary = ep_stats.get("_episode_summary")
                ep_error = ep_stats.get("error")
                ep_outcome = ep_stats.get("outcome", "unknown")
                ep_end_reason = ep_stats.get("end_reason")
                ep_floor = ep_stats.get("floors", 0)
                ep_combats = ep_stats.get("combats", 0)
                ep_time = ep_stats.get("episode_time_s", 0)

                if ep_error and trace:
                    logger.warning("Episode ERROR (%s) trace:\n  %s",
                                   ep_error, "\n  ".join(trace[-30:]))

                # Write every episode trace to replays/ subdirectory
                if args.save_replay_traces and (trace or trace_zh or summary):
                    replay_dir = output_dir / "replays"
                    replay_dir.mkdir(exist_ok=True)
                    if ep_error:
                        tag = "ERR"
                    elif ep_outcome:
                        tag = str(ep_outcome).upper()[:3]
                    elif ep_end_reason == "combat_pending_stall":
                        tag = "STL"
                    elif ep_end_reason == "max_steps":
                        tag = "MAX"
                    else:
                        tag = "UNK"
                    trace_path = replay_dir / f"i{iteration:05d}_e{_ep_counter:03d}_{tag}_f{ep_floor}.txt"
                    trace_zh_path = replay_dir / f"i{iteration:05d}_e{_ep_counter:03d}_{tag}_f{ep_floor}.zh.txt"
                    trace_summary_path = replay_dir / f"i{iteration:05d}_e{_ep_counter:03d}_{tag}_f{ep_floor}.summary.json"
                    if args.save_replay_human and trace:
                        try:
                            with open(trace_path, "w", encoding="utf-8") as f:
                                f.write(f"# Iter {iteration} episode {_ep_counter}\n")
                                f.write(f"# outcome={ep_outcome} floor={ep_floor} "
                                        f"combats={ep_combats} time={ep_time:.1f}s "
                                        f"end_reason={ep_end_reason} error={ep_error}\n\n")
                                f.write("\n".join(trace))
                        except Exception:
                            pass
                    if args.save_replay_human and trace_zh:
                        try:
                            with open(trace_zh_path, "w", encoding="utf-8") as f:
                                f.write(f"# 第 {iteration} 轮，第 {_ep_counter} 局\n")
                                f.write(
                                    f"# outcome={ep_outcome} floor={ep_floor} "
                                    f"combats={ep_combats} time={ep_time:.1f}s "
                                    f"end_reason={ep_end_reason} error={ep_error}\n\n"
                                )
                                f.write("\n".join(trace_zh))
                        except Exception:
                            pass
                    if args.save_replay_structured and isinstance(summary, dict):
                        try:
                            summary_payload = dict(summary)
                            summary_payload.update(
                                {
                                    "iteration": iteration,
                                    "episode": _ep_counter,
                                    "tag": tag,
                                    "trace_path": str(trace_path.resolve()) if trace else "",
                                    "trace_zh_path": str(trace_zh_path.resolve()) if trace_zh else "",
                                }
                            )
                            trace_summary_path.write_text(
                                json.dumps(summary_payload, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                _ep_counter += 1

            num_workers = min(len(env_ports), args.episodes_per_iter)
            if args.multi_process and not args.local_ort:
                assert mp_result_queue is not None
                for ep in range(args.episodes_per_iter):
                    episode_seed = None if args.seed is None else f"audit-{args.seed}-iter{iteration:05d}-ep{ep:03d}"
                    mp_task_queues[ep % len(mp_task_queues)].put({
                        "cmd": "collect_episode",
                        "seed": episode_seed,
                    })
                for _ in range(args.episodes_per_iter):
                    try:
                        _worker_id, ep_ppo, ep_mcts, ep_stats = mp_result_queue.get(
                            timeout=max(10.0, float(args.episode_timeout) + 10.0)
                        )
                    except Exception as e:
                        logger.error("Multi-process collector timeout/failure: %s", e)
                        ep_ppo, ep_mcts, ep_stats = StructuredRolloutBuffer(), [], {"error": f"mp_collect: {e}"}
                    if ep_ppo is None or ep_mcts is None:
                        ep_ppo, ep_mcts = StructuredRolloutBuffer(), []
                    _merge_episode(ep_ppo, ep_mcts, ep_stats)
            elif args.vectorized:
                # Vectorized: all envs step in lockstep (parallel pipe I/O + batch NN)
                from vectorized_collector import collect_vectorized_episodes
                vec_clients = [env_clients[p] for p in env_ports]
                vec_ppo, vec_combat, vec_stats = collect_vectorized_episodes(
                    ppo_net=ppo_net,
                    combat_net=mcts_net,
                    vocab=vocab,
                    clients=vec_clients,
                    character_id=args.character_id,
                    max_steps=args.max_episode_steps,
                    episode_timeout=args.episode_timeout,
                    screen_local_delta=args.screen_local_delta,
                    device=device,
                )
                # Merge vectorized results into iteration buffers
                env_floor_map = {s["env_id"]: s["floors"] / 16.0 for s in vec_stats}
                for t in vec_ppo:
                    if t.get("type") == "nc_step" and "state_features" in t:
                        floor_target = env_floor_map.get(t.get("env_id", -1), 0.0)
                        ppo_buffer.states.append(t["state_features"])
                        ppo_buffer.actions_data.append(t["action_features"])
                        ppo_buffer.action_indices.append(t["action_idx"])
                        ppo_buffer.log_probs.append(t["log_prob"])
                        ppo_buffer.rewards.append(t["reward"])
                        ppo_buffer.values.append(t["value"])
                        ppo_buffer.dones.append(False)
                        ppo_buffer.floor_targets.append(floor_target)
                        ppo_buffer.boss_readiness_targets.append(0.0)
                        ppo_steps += 1
                for t in vec_combat:
                    if "state_features" in t:
                        combat_buffer.add(
                            sf=t["state_features"],
                            af=t["action_features"],
                            action_idx=t["action_idx"],
                            log_prob=t["log_prob"],
                            reward=0.0,
                            value=t["value"],
                            done=False,
                        )
                        combat_ppo_steps += 1
                for s in vec_stats:
                    total_floors += s["floors"]
                    total_combats += s["combats"]
                    if s["outcome"] == "victory":
                        victories += 1
                        act1_cleared_eps += 1
                    if s["boss_reached"]:
                        boss_reached_eps += 1
                        if s["boss_hp_peak"] > 0:
                            boss_hp_fracs.append(s["boss_hp_peak"])
                        for ds in s["deck_size_at_boss"]:
                            deck_sizes_at_boss.append(ds)
                    card_reward_skips += s["cards_skipped"]
                    card_reward_screens += s["cards_skipped"] + len(s["cards_taken"])
                    ep_times.append(0.0)
            elif num_workers > 1:
                # Time-budget collection: submit all episodes, harvest completed
                # ones within a time limit. Prevents straggler episodes from
                # blocking the entire iteration (critical for --local-ort mode
                # where combat is coarse-grained).
                iter_budget_s = args.iter_time_budget if hasattr(args, 'iter_time_budget') and args.iter_time_budget > 0 else 0
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    env_idx = 0
                    futures = {}
                    for _ in range(min(num_workers, args.episodes_per_iter)):
                        futures[executor.submit(_collect, env_idx)] = env_idx
                        env_idx += 1
                    remaining = args.episodes_per_iter - len(futures)
                    harvest_deadline = time.monotonic() + iter_budget_s if iter_budget_s > 0 else float("inf")
                    harvested = 0
                    skipped = 0

                    while futures:
                        if _shutdown_requested:
                            break
                        # Check time budget
                        time_left = harvest_deadline - time.monotonic()
                        if iter_budget_s > 0 and time_left <= 0 and harvested >= 4:
                            # Time's up and we have enough episodes — skip remaining
                            skipped = len(futures)
                            for f in futures:
                                f.cancel()
                            break
                        budget_timeout = max(0.1, min(time_left, 5.0)) if iter_budget_s > 0 else None
                        try:
                            for f in as_completed(futures, timeout=budget_timeout):
                                fidx = futures.pop(f)
                                try:
                                    ep_ppo, ep_mcts, ep_stats = f.result()
                                except BaseException as e:
                                    logger.error(
                                        "Episode %d failed with %s\n%s",
                                        fidx,
                                        repr(e),
                                        traceback.format_exc(),
                                    )
                                    ep_ppo, ep_mcts, ep_stats = StructuredRolloutBuffer(), [], {
                                        "error": f"{type(e).__name__}: {e}"
                                    }

                                _merge_episode(ep_ppo, ep_mcts, ep_stats)
                                harvested += 1

                                if remaining > 0 and not _shutdown_requested and time.monotonic() < harvest_deadline:
                                    futures[executor.submit(_collect, env_idx)] = env_idx
                                    env_idx += 1
                                    remaining -= 1
                                break
                        except TimeoutError:
                            # Time budget exceeded — skip remaining futures
                            if harvested >= 4:
                                skipped = len(futures)
                                for f in futures:
                                    f.cancel()
                                futures.clear()
                    if skipped > 0:
                        iter_timeout_count += skipped
            else:
                for ep in range(args.episodes_per_iter):
                    if _shutdown_requested:
                        break
                    try:
                        ep_ppo, ep_mcts, ep_stats = _collect(ep)
                    except BaseException as e:
                        logger.error(
                            "Episode %d failed with %s\n%s",
                            ep,
                            repr(e),
                            traceback.format_exc(),
                        )
                        ep_ppo, ep_mcts, ep_stats = StructuredRolloutBuffer(), [], {
                            "error": f"{type(e).__name__}: {e}"
                        }
                    _merge_episode(ep_ppo, ep_mcts, ep_stats)

            _collect_end = time.monotonic()

            # --- Train PPO ---
            ppo_metrics = {"ppo_ploss": 0, "ppo_vloss": 0, "ppo_entropy": 0, "boss_readiness_loss": 0}
            if len(ppo_buffer) >= 4 and not getattr(args, "freeze_ppo", False):
                ppo_net.train()
                ppo_buffer.compute_gae()
                # Phase 4: KL warm-start beta schedule
                _kl_beta = 0.0
                if args.kl_warmstart:
                    _progress = min(iteration / max(1, args.kl_beta_decay), 1.0)
                    _kl_beta = args.kl_beta_start * max(0, 1.0 - _progress) + 0.05 * _progress
                ppo_metrics = ppo_trainer.update(
                    ppo_buffer,
                    per_screen_adv_norm=args.per_screen_adv_norm,
                    weighted_screen_sampling=args.weighted_screen_sampling,
                    kl_beta=_kl_beta,
                )
                ppo_buffer.clear()

            # --- Train MCTS (behavior cloning from MCTS visit distributions) ---
            mcts_metrics = {"mcts_ploss": 0, "mcts_vloss": 0}
            if (
                iteration >= args.mcts_warmup_iters
                and len(mcts_replay) >= args.mcts_batch_size
                and new_mcts > 0
            ):
                mcts_net.train()
                for _ in range(args.mcts_train_steps):
                    batch = mcts_replay.sample(args.mcts_batch_size)
                    mcts_metrics = mcts_train_step(mcts_net, mcts_optimizer, batch,
                                                       device=device, use_amp=use_amp)

            # --- Train Combat PPO ---
            combat_ppo_metrics = {"combat_ppo_ploss": 0, "combat_ppo_vloss": 0, "combat_entropy": 0}
            if len(combat_buffer) >= 32 and not getattr(args, "freeze_combat", False):
                mcts_net.train()
                combat_ppo_metrics = combat_ppo_trainer.update(combat_buffer)
                combat_buffer.clear()
            elif len(combat_buffer) >= 32:
                combat_buffer.clear()  # discard combat data when frozen

            # --- Train matchup ranking (offline card ranking data) ---
            offline_noncombat_ranking_loss_val = 0.0
            if (matchup_dataset is not None
                    and iteration >= args.matchup_warmup_iters
                    and len(matchup_dataset) > 0
                    and not getattr(args, "freeze_ppo", False)):
                ppo_net.train()
                matchup_updates = max(1, int(getattr(args, "matchup_updates_per_iter", 1)))
                offline_noncombat_ranking_losses: list[float] = []
                for _ in range(matchup_updates):
                    mb = matchup_dataset.sample_batch(args.matchup_batch_size, device=device)
                    if mb is None or "state_tensors" not in mb or "action_tensors" not in mb:
                        continue
                    pred_scores_full = ppo_net.compute_matchup_scores(
                        mb["state_tensors"], mb["action_tensors"])
                    # pred_scores is (B, MAX_ACTIONS=30), target is (B, MAX_OPTIONS=4)
                    # Slice to match option count
                    n_opts = mb["target_scores"].shape[1]
                    pred_scores = pred_scores_full[:, :n_opts]
                    from search.ranking_loss import listwise_ranking_loss
                    rank_loss = listwise_ranking_loss(
                        pred_scores, mb["target_scores"], mb["option_mask"])
                    decay_tau = getattr(args, "matchup_loss_decay_tau", 0.0)
                    if decay_tau > 0:
                        import math
                        matchup_w = args.matchup_loss_weight * math.exp(-iteration / decay_tau)
                    else:
                        matchup_w = args.matchup_loss_weight
                    total_rloss = matchup_w * rank_loss
                    ppo_trainer.optimizer.zero_grad()
                    total_rloss.backward()
                    ppo_trainer.optimizer.step()
                    offline_noncombat_ranking_losses.append(rank_loss.item())
                if offline_noncombat_ranking_losses:
                    offline_noncombat_ranking_loss_val = float(
                        sum(offline_noncombat_ranking_losses) / len(offline_noncombat_ranking_losses)
                    )

            # --- Train combat teacher (offline turn-solver teacher data) ---
            ct_loss_val = 0.0
            ct_ce_val = 0.0
            ct_rank_val = 0.0
            ct_cont_val = 0.0
            ct_survival_val = 0.0
            ct_hp_cost_val = 0.0
            ct_potion_cost_val = 0.0
            if (combat_teacher_dataset is not None
                    and iteration >= args.combat_teacher_warmup_iters
                    and len(combat_teacher_dataset) > 0
                    and not getattr(args, "freeze_combat", False)):
                from search.train_combat_teacher import (
                    _regret_weighted_pairwise_ranking,
                    _stack_batch as _ct_stack_batch,
                )
                mcts_net.train()
                ct_updates = max(1, int(getattr(args, "combat_teacher_updates_per_iter", 1)))
                ct_loss_history: list[float] = []
                ct_ce_history: list[float] = []
                ct_rank_history: list[float] = []
                ct_cont_history: list[float] = []
                ct_survival_history: list[float] = []
                ct_hp_cost_history: list[float] = []
                ct_potion_cost_history: list[float] = []
                for _ in range(ct_updates):
                    # Sample a random batch each update so large teacher datasets
                    # can keep meaningful influence on combat training.
                    ct_bs = min(args.combat_teacher_batch_size, len(combat_teacher_dataset))
                    ct_indices = random.sample(range(len(combat_teacher_dataset)), ct_bs)
                    ct_raw_batch = [combat_teacher_dataset[i] for i in ct_indices]
                    ct_batch = _ct_stack_batch(ct_raw_batch, device)

                    ct_logits, _ct_value, ct_action_scores, ct_continuation = mcts_net.forward_teacher(
                        ct_batch["state_features"], ct_batch["action_features"])
                    ct_action_mask = ct_batch["action_features"]["action_mask"]
                    ct_masked_scores = ct_action_scores.masked_fill(~ct_action_mask, -1e9)

                    # Teacher best-action CE (on action_score head)
                    ct_ce = F.cross_entropy(ct_masked_scores, ct_batch["teacher_best_action_index"])

                    # Regret-weighted pairwise ranking (clamp regrets to avoid numerical explosion)
                    ct_regrets_clamped = ct_batch["regrets"].clamp(max=10.0)
                    ct_rank = _regret_weighted_pairwise_ranking(
                        ct_masked_scores, ct_regrets_clamped,
                        ct_batch["teacher_best_action_index"], ct_action_mask,
                        ct_batch["sample_weight"])

                    # Continuation value regression (explicit survival + cost)
                    ct_cont, ct_survival, ct_hp_cost, ct_potion_cost = _combat_room_conditioned_continuation_loss(
                        ct_continuation,
                        ct_batch["continuation_targets"],
                        ct_batch["state_features"].get("room_type_onehot"),
                    )

                    ct_total = args.combat_teacher_loss_weight * (ct_ce + ct_rank + ct_cont)
                    combat_ppo_trainer.optimizer.zero_grad()
                    ct_total.backward()
                    torch.nn.utils.clip_grad_norm_(mcts_net.parameters(), 1.0)
                    combat_ppo_trainer.optimizer.step()

                    ct_loss_history.append(ct_total.item())
                    ct_ce_history.append(ct_ce.item())
                    ct_rank_history.append(ct_rank.item())
                    ct_cont_history.append(ct_cont.item())
                    ct_survival_history.append(ct_survival.item())
                    ct_hp_cost_history.append(ct_hp_cost.item())
                    ct_potion_cost_history.append(ct_potion_cost.item())

                if ct_loss_history:
                    ct_loss_val = float(sum(ct_loss_history) / len(ct_loss_history))
                    ct_ce_val = float(sum(ct_ce_history) / len(ct_ce_history))
                    ct_rank_val = float(sum(ct_rank_history) / len(ct_rank_history))
                    ct_cont_val = float(sum(ct_cont_history) / len(ct_cont_history))
                    ct_survival_val = float(sum(ct_survival_history) / len(ct_survival_history))
                    ct_hp_cost_val = float(sum(ct_hp_cost_history) / len(ct_hp_cost_history))
                    ct_potion_cost_val = float(sum(ct_potion_cost_history) / len(ct_potion_cost_history))

            _update_end = time.monotonic()
            iter_time = _update_end - iter_start
            _collect_time = _collect_end - iter_start
            _update_time = _update_end - _collect_end
            avg_floor = total_floors / max(1, args.episodes_per_iter)
            avg_ep = sum(ep_times) / max(1, len(ep_times))
            boss_reach_rate = boss_reached_eps / max(1, args.episodes_per_iter)
            act1_clear_rate = act1_cleared_eps / max(1, args.episodes_per_iter)
            boss_hp_fraction_mean = (
                float(np.mean(boss_hp_fracs)) if boss_hp_fracs else 0.0
            )
            deck_size_at_boss_mean = (
                float(np.mean(deck_sizes_at_boss)) if deck_sizes_at_boss else 0.0
            )
            card_reward_skip_rate = (
                card_reward_skips / max(1, card_reward_screens)
            )

            entry = {
                "iteration": iteration,
                "avg_floor": avg_floor,
                "victories": victories,
                "episodes": args.episodes_per_iter,
                "combats": total_combats,
                "ppo_steps": ppo_steps,
                "mcts_decisions": mcts_decisions,
                "combat_decisions": mcts_decisions,
                "mcts_replay": len(mcts_replay),
                "combat_search_replay": len(mcts_replay),
                "new_mcts": new_mcts,
                "new_combat_search_samples": new_mcts,
                "ppo_ploss": ppo_metrics.get("policy_loss", ppo_metrics.get("ppo_ploss", 0)),
                "ppo_vloss": ppo_metrics.get("value_loss", ppo_metrics.get("ppo_vloss", 0)),
                "ppo_entropy": ppo_metrics.get("entropy", ppo_metrics.get("ppo_entropy", 0)),
                "boss_readiness_loss": ppo_metrics.get("boss_readiness_loss", 0),
                "boss_readiness_weighted": ppo_metrics.get("boss_readiness_loss", 0) * args.boss_readiness_coeff,
                "ppo_ratio_mean": ppo_metrics.get("ratio_mean", 0),
                "ppo_clip_fraction": ppo_metrics.get("clip_fraction", 0),
                "ppo_approx_kl": ppo_metrics.get("approx_kl", 0),
                "ppo_early_stop": ppo_metrics.get("early_stop", 0),
                "mcts_ploss": mcts_metrics.get("mcts_ploss", 0),
                "mcts_vloss": mcts_metrics.get("mcts_vloss", 0),
                "combat_search_ploss": mcts_metrics.get("mcts_ploss", 0),
                "combat_search_vloss": mcts_metrics.get("mcts_vloss", 0),
                "combat_ppo_steps": combat_ppo_steps,
                "combat_random_warmup_steps": combat_random_warmup_steps,
                "combat_mcts_turn_limited_steps": combat_mcts_turn_limited_steps,
                "combat_ppo_ploss": combat_ppo_metrics.get("combat_ppo_ploss", 0),
                "combat_ppo_vloss": combat_ppo_metrics.get("combat_ppo_vloss", 0),
                "combat_entropy": combat_ppo_metrics.get("combat_entropy", 0),
                "combat_ppo_ratio_mean": combat_ppo_metrics.get("combat_ppo_ratio_mean", 0),
                "combat_ppo_clip_fraction": combat_ppo_metrics.get("combat_ppo_clip_fraction", 0),
                "combat_ppo_approx_kl": combat_ppo_metrics.get("combat_ppo_approx_kl", 0),
                "combat_ppo_early_stop": combat_ppo_metrics.get("combat_ppo_early_stop", 0),
                "offline_noncombat_ranking_loss": round(offline_noncombat_ranking_loss_val, 6),
                "combat_teacher_loss": round(ct_loss_val, 6),
                "combat_teacher_ce": round(ct_ce_val, 6),
                "combat_teacher_rank": round(ct_rank_val, 6),
                "combat_teacher_cont": round(ct_cont_val, 6),
                "combat_teacher_survival": round(ct_survival_val, 6),
                "combat_teacher_hp_cost": round(ct_hp_cost_val, 6),
                "combat_teacher_potion_cost": round(ct_potion_cost_val, 6),
                "offline_noncombat_ranking_head_mode": offline_noncombat_ranking_head_mode,
                "combat_main_path_mode": combat_main_path_mode,
                "avg_ep_time": avg_ep,
                "iter_time_s": iter_time,
                "collect_time_s": round(_collect_time, 3),
                "update_time_s": round(_update_time, 3),
                "ort_combat_time_s": round(_iter_ort_combat_time, 3),
                "ort_combat_calls": _iter_ort_combat_calls,
                "nc_forward_time_s": round(_iter_nc_forward_time, 3),
                "nc_forward_calls": _iter_nc_forward_calls,
                "pipe_time_s": round(_iter_pipe_time, 3),
                "pipe_calls": _iter_pipe_calls,
                "feature_time_s": round(_iter_feature_time, 3),
                "feature_calls": _iter_feature_calls,
                "inference_time_s": round(_iter_inference_time, 3),
                "inference_calls": _iter_inference_calls,
                "buffer_time_s": round(_iter_buffer_time, 3),
                "buffer_calls": _iter_buffer_calls,
                "slow_steps": iter_slow_steps,
                "max_step_ms": round(iter_max_step_ms, 1),
                "timeout_count": iter_timeout_count,
                "wait_steps": wait_steps_total,
                "combat_pending_steps": combat_pending_steps_total,
                "combat_pending_wait_steps": combat_pending_wait_steps_total,
                "combat_pending_refresh_steps": combat_pending_refresh_steps_total,
                "combat_pending_stall_count": combat_pending_stall_count,
                "hard_state_potion_steps": hard_state_potion_steps,
                "hard_state_premature_end_turn_steps": hard_state_end_turn_steps,
                "hard_state_repeat_loop_steps": hard_state_repeat_steps,
                "hard_state_order_sensitive_steps": hard_state_order_steps,
                "combat_hard_state_weight_mean": round(
                    combat_hard_state_weight_sum / max(1, combat_ppo_steps), 4
                ),
                "boss_reach_rate": round(boss_reach_rate, 4),
                "act1_clear_rate": round(act1_clear_rate, 4),
                "boss_hp_fraction_dealt_mean": round(boss_hp_fraction_mean, 4),
                "deck_size_at_boss_mean": round(deck_size_at_boss_mean, 2),
                "card_reward_skip_rate": round(card_reward_skip_rate, 4),
            }

            if hasattr(ppo_net, "offline_ranking_action_context_gate"):
                entry["offline_ranking_action_context_gate"] = round(
                    float(ppo_net.offline_ranking_action_context_gate.detach().item()), 6
                )
            if hasattr(ppo_net, "offline_ranking_state_context_gate"):
                entry["offline_ranking_state_context_gate"] = round(
                    float(ppo_net.offline_ranking_state_context_gate.detach().item()), 6
                )
            if hasattr(mcts_net, "main_action_context_gate"):
                entry["combat_main_action_context_gate"] = round(
                    float(mcts_net.main_action_context_gate.detach().item()), 6
                )
            if hasattr(mcts_net, "action_aux_gate"):
                entry["combat_action_aux_gate"] = round(
                    float(mcts_net.action_aux_gate.detach().item()), 6
                )
            if hasattr(mcts_net, "action_family_gate"):
                entry["combat_action_family_gate"] = round(
                    float(mcts_net.action_family_gate.detach().item()), 6
                )
            if hasattr(mcts_net, "stop_continue_gate"):
                entry["combat_stop_continue_gate"] = round(
                    float(mcts_net.stop_continue_gate.detach().item()), 6
                )
            if hasattr(mcts_net, "resource_gate"):
                entry["combat_resource_gate"] = round(
                    float(mcts_net.resource_gate.detach().item()), 6
                )
            if hasattr(mcts_net, "main_state_context_gate"):
                entry["combat_main_state_context_gate"] = round(
                    float(mcts_net.main_state_context_gate.detach().item()), 6
                )
            if hasattr(mcts_net, "teacher_action_context_gate"):
                entry["combat_teacher_action_context_gate"] = round(
                    float(mcts_net.teacher_action_context_gate.detach().item()), 6
                )
            metrics_history.append(entry)
            if metrics_log is not None:
                with open(metrics_log, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            logger.info(
                "Iter %3d | floor %.1f | vic %d/%d | boss %.0f%% act1 %.0f%% boss_hp %.2f deck@boss %.1f skip %.0f%% | ppo %d combat %d cppo %d | "
                "pend r/w/stl %d/%d/%d | "
                "ppo_pl %.4f ppo_vl %.4f ppo_ent %.3f ppo_clip %.2f ratio %.3f boss_r %.4f (w %.5f) | "
                "search_pl %.4f search_vl %.4f offr %.4f | cppo_pl %.4f cppo_vl %.4f cbt_ent %.3f cppo_clip %.2f | "
                "rank[%s] a_gate %.4f s_gate %.4f | combat[%s] a_gate %.4f s_gate %.4f t_gate %.4f | %.0fs",
                iteration, avg_floor, victories, args.episodes_per_iter,
                boss_reach_rate * 100.0, act1_clear_rate * 100.0,
                boss_hp_fraction_mean, deck_size_at_boss_mean, card_reward_skip_rate * 100.0,
                ppo_steps, mcts_decisions, combat_ppo_steps,
                entry.get("combat_pending_refresh_steps", 0),
                entry.get("combat_pending_wait_steps", 0),
                entry.get("combat_pending_stall_count", 0),
                entry["ppo_ploss"], entry["ppo_vloss"], entry.get("ppo_entropy", 0),
                entry.get("ppo_clip_fraction", 0), entry.get("ppo_ratio_mean", 0),
                entry["boss_readiness_loss"], entry.get("boss_readiness_weighted", 0),
                entry["mcts_ploss"], entry["mcts_vloss"],
                entry.get("offline_noncombat_ranking_loss", 0.0),
                entry["combat_ppo_ploss"], entry["combat_ppo_vloss"],
                entry["combat_entropy"], entry.get("combat_ppo_clip_fraction", 0),
                entry.get("offline_noncombat_ranking_head_mode", "mlp"),
                entry.get("offline_ranking_action_context_gate", 0.0),
                entry.get("offline_ranking_state_context_gate", 0.0),
                entry.get("combat_main_path_mode", "mlp"),
                entry.get("combat_main_action_context_gate", 0.0),
                entry.get("combat_main_state_context_gate", 0.0),
                entry.get("combat_teacher_action_context_gate", 0.0),
                iter_time,
            )

            # Free per-iter refcycle garbage and return torch CUDA caching allocator
            # pool to the driver. Without this, bigbatch configs (e.g. 2000 ep/iter)
            # let the allocator pool grow to ~15GB and main RSS to ~20GB within 1-2
            # iterations, causing throughput to fall 3x and risking CUDA OOM.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Health check
            # Ramp up learned card evaluator blend (every 100 iter)
            if effective_counterfactual_scoring and iteration > 0 and iteration % 100 == 0:
                from search.counterfactual_scoring import set_learned_evaluator
                _alpha = min(0.7, 0.1 + iteration * 0.3 / 1000)  # 0→0.7 over 2000 iter
                set_learned_evaluator(
                    ppo_net, vocab, alpha=_alpha,
                    skada_priors=_skada_priors_obj,
                    skada_gamma=args.skada_prior_weight if _skada_priors_obj is not None else 0.0,
                )

            if iteration > 0 and iteration % health_check_interval == 0:
                try:
                    alerts = health_monitor.check_all(metrics_history)
                    for alert in alerts:
                        logger.warning("HEALTH: %s", alert)
                except Exception:
                    pass

            # Periodic diagnostic dump
            if iteration > 0 and iteration % 50 == 0:
                try:
                    recent = metrics_history[-50:] if len(metrics_history) >= 50 else metrics_history
                    floors = [e.get("avg_floor", 0) for e in recent]
                    search_plosses = [
                        e.get("combat_search_ploss", e.get("mcts_ploss", 0))
                        for e in recent
                        if e.get("combat_search_ploss", e.get("mcts_ploss", 0)) > 0
                    ]
                    total_slow = sum(e.get("slow_steps", 0) for e in recent)
                    total_to = sum(e.get("timeout_count", 0) for e in recent)
                    ppo_buf_sizes = [e.get("ppo_steps", 0) for e in recent]
                    logger.info("=== DIAGNOSTIC iter %d ===", iteration)
                    logger.info("  Floor trend (last %d): avg=%.1f min=%.0f max=%.0f",
                                len(recent), np.mean(floors) if floors else 0,
                                min(floors) if floors else 0, max(floors) if floors else 0)
                    if search_plosses and len(search_plosses) >= 2:
                        logger.info("  Combat search ploss trend: %.3f -> %.3f",
                                    search_plosses[0], search_plosses[-1])
                    cppo_plosses = [e.get("combat_ppo_ploss", 0) for e in recent if e.get("combat_ppo_ploss", 0) > 0]
                    cppo_entropies = [e.get("combat_entropy", 0) for e in recent if e.get("combat_entropy", 0) > 0]
                    if cppo_plosses and len(cppo_plosses) >= 2:
                        logger.info("  Combat PPO ploss trend: %.3f -> %.3f",
                                    cppo_plosses[0], cppo_plosses[-1])
                    if cppo_entropies and len(cppo_entropies) >= 2:
                        logger.info("  Combat entropy trend: %.3f -> %.3f",
                                    cppo_entropies[0], cppo_entropies[-1])
                    logger.info("  Slow steps total: %d, timeouts: %d", total_slow, total_to)
                    logger.info("  PPO buffer avg size: %d",
                                int(np.mean(ppo_buf_sizes)) if ppo_buf_sizes else 0)
                    cppo_buf_sizes = [e.get("combat_ppo_steps", 0) for e in recent]
                    logger.info("  Combat PPO buffer avg size: %d",
                                int(np.mean(cppo_buf_sizes)) if cppo_buf_sizes else 0)

                    # --- Feature activity monitor ---
                    # Check if any feature groups are all-zero (dead features = schema bug)
                    if len(ppo_buffer) > 0:
                        dead_features = []
                        sample_ss = ppo_buffer.states[0]
                        for key in ["deck_mask", "relic_mask", "map_node_mask",
                                     "hand_mask", "enemy_mask"]:
                            vals = [getattr(s, key, None) for s in ppo_buffer.states[-20:]]
                            vals = [v for v in vals if v is not None]
                            if vals and not any(v.any() for v in vals):
                                dead_features.append(key)
                        if dead_features:
                            logger.warning("  DEAD FEATURES (all-zero): %s — schema bug?",
                                           ", ".join(dead_features))
                        else:
                            logger.info("  Feature activity: all feature groups active")

                    # --- Reward distribution monitor ---
                    if len(ppo_buffer) > 4:
                        rewards = np.array(ppo_buffer.rewards[-50:])
                        r_mean, r_std = rewards.mean(), rewards.std()
                        logger.info("  PPO reward: mean=%.4f std=%.4f min=%.4f max=%.4f",
                                    r_mean, r_std, rewards.min(), rewards.max())
                        if r_std < 1e-6:
                            logger.warning("  PPO REWARD FLAT (std=0) — reward shaping broken?")

                    if combat_buffer is not None and len(combat_buffer) > 4:
                        c_rewards = np.array(combat_buffer.rewards[-50:])
                        logger.info("  Combat reward: mean=%.4f std=%.4f min=%.4f max=%.4f",
                                    c_rewards.mean(), c_rewards.std(),
                                    c_rewards.min(), c_rewards.max())

                    # --- Advantage monitor ---
                    if len(ppo_buffer) > 4 and ppo_buffer.advantages:
                        advs = np.array(ppo_buffer.advantages[-50:])
                        logger.info("  PPO advantages: mean=%.4f std=%.4f",
                                    advs.mean(), advs.std())
                        if advs.std() < 1e-6:
                            logger.warning("  PPO ADVANTAGES FLAT — value function not learning?")

                except Exception:
                    pass

            # Save
            if iteration % args.save_interval == 0:
                torch.save({
                    "ppo_model": ppo_net.state_dict(),
                    "mcts_model": mcts_net.state_dict(),
                    "iteration": iteration,
                    "ppo_config": {
                        "embed_dim": args.embed_dim,
                        "offline_noncombat_ranking_head_mode": offline_noncombat_ranking_head_mode,
                    },
                    "mcts_config": {
                        "embed_dim": args.embed_dim,
                        "hidden_dim": args.combat_hidden_dim,
                        "combat_main_path_mode": combat_main_path_mode,
                    },
                }, output_dir / f"hybrid_{iteration:05d}.pt")

    except BaseException as e:
        logger.error("Crash: %s\n%s", e, traceback.format_exc())
        torch.save({
            "ppo_model": ppo_net.state_dict(),
            "mcts_model": mcts_net.state_dict(),
            "crash": str(e),
            "ppo_config": {
                "embed_dim": args.embed_dim,
                "offline_noncombat_ranking_head_mode": offline_noncombat_ranking_head_mode,
            },
            "mcts_config": {
                "embed_dim": args.embed_dim,
                "hidden_dim": args.combat_hidden_dim,
                "combat_main_path_mode": combat_main_path_mode,
            },
        }, output_dir / "hybrid_crash.pt")
        raise
    finally:
        if inf_server is not None:
            inf_server.stop()
        for _task_q in mp_task_queues:
            try:
                _task_q.put(None)
            except Exception:
                pass
        for _worker in mp_workers:
            try:
                _worker.join(timeout=2.0)
            except Exception:
                pass
            try:
                if _worker.is_alive():
                    _worker.terminate()
            except Exception:
                pass
        for client in env_clients.values():
            try:
                if hasattr(client, '_pipe') and client._pipe is not None:
                    client._pipe.call("delete_state", {"clear_all": True})
                elif hasattr(client, 'act'):
                    pass  # HTTP client — no cleanup needed
            except Exception:
                pass
            try:
                if hasattr(client, 'close'):
                    client.close()
            except Exception:
                pass
        _cleanup_spawned_envs()

    # Final save — use the SAME config keys as the periodic save above so the
    # downstream loader doesn't trip the "config disagrees with weights"
    # warning. The previous version hardcoded embed_dim=32 / hidden_dim=128
    # (legacy defaults from a much earlier training era) which is now wrong
    # because the current defaults are embed_dim=48 / hidden_dim=192.
    torch.save({
        "ppo_model": ppo_net.state_dict(),
        "mcts_model": mcts_net.state_dict(),
        "iteration": end_iter - 1,
        "ppo_config": {
            "embed_dim": args.embed_dim,
            "offline_noncombat_ranking_head_mode": offline_noncombat_ranking_head_mode,
        },
        "mcts_config": {
            "embed_dim": args.embed_dim,
            "hidden_dim": args.combat_hidden_dim,
            "combat_main_path_mode": combat_main_path_mode,
        },
    }, output_dir / "hybrid_final.pt")

    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
