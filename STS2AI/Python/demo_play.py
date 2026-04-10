"""AI demo player with real-time decision visualization overlay.

Supports two runtime modes:
1. Front-UI mode over STS2MCP HTTP singleplayer/full-run endpoints.
2. Legacy simulator mode over named pipe/full-run-sim-server.

Recommended for recording:
    Godot.exe --path . -- --mcp-port 15600 --mcp-decision-overlay-file STS2AI/Artifacts/demo_overlay/live_overlay.json

    python STS2AI/Python/demo_play.py --checkpoint STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt \\
                                      --transport http --port 15600 \\
                                      --decision-overlay-file STS2AI/Artifacts/demo_overlay/live_overlay.json
"""
from __future__ import annotations

import _path_init  # noqa: F401

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from sts2ai_paths import ARTIFACTS_ROOT, MAINLINE_CHECKPOINT

DEFAULT_HYBRID_CHECKPOINT = MAINLINE_CHECKPOINT
DEFAULT_COMBAT_CHECKPOINT = MAINLINE_CHECKPOINT.with_name("_no_default_combat_override.pt")
DEFAULT_OUTPUT_ROOT = ARTIFACTS_ROOT / "demo_overlay"

import numpy as np
import torch
import websockets
import websockets.server

from vocab import load_vocab
from combat_nn import (
    CombatPolicyValueNetwork, build_combat_features, build_combat_action_features,
)
from rl_policy_v2 import (
    FullRunPolicyNetworkV2, _structured_state_to_numpy_dict, _structured_actions_to_numpy_dict,
)
from rl_encoder_v2 import build_structured_state, build_structured_actions
try:
    from rl_reward_shaping import deck_score
except ImportError:
    from rl_reward_shaping import problem_score as deck_score
from rl_reward_shaping import (
    compute_problem_vector, problem_score, survival_margin,
    economy_score, potential, boss_readiness_score, extract_next_boss_token,
)
from nn_hooks import NNInternalsCollector, format_internals_for_broadcast
from training_monitor import TrainingMetricsMonitor
from full_run_env import create_full_run_client
from sts2_singleplayer_env import adapt_v1_state_for_combat_policy
from demo_action_candidates import _combat_candidate_actions, _non_combat_candidate_actions

import io

# Force UTF-8 output on Windows to avoid Chinese garbling
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", encoding="utf-8")
logger = logging.getLogger(__name__)

COMBAT_SCREENS = {"combat", "monster", "elite", "boss", "hand_select", "card_select"}
SELECTION_SCREENS = {"hand_select", "card_select"}
ESCAPE_ACTION_NAMES = (
    "confirm_selection",
    "combat_confirm_selection",
    "claim_reward",
    "proceed",
    "skip",
    "end_turn",
    "wait",
)

_shutdown = False
def _handle_signal(sig, frame):
    global _shutdown
    _shutdown = True
signal.signal(signal.SIGINT, _handle_signal)

PROBLEM_VECTOR_LABELS = [
    "frontload", "aoe", "block", "draw", "energy",
    "scaling", "consistency", "elite_ready", "boss_answer",
]


def _safe_load_state_dict(model: torch.nn.Module, state_dict: dict[str, Any], label: str) -> None:
    """Load only compatible checkpoint keys so older models keep working."""
    model_state = model.state_dict()
    filtered: dict[str, Any] = {}
    skipped_shape: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped_shape:
        logger.warning("%s shape mismatches skipped: %d keys", label, len(skipped_shape))
    if missing:
        logger.info("New params in %s (randomly init): %d keys", label, len(missing))
    if unexpected:
        logger.warning("%s unexpected keys ignored: %d", label, len(unexpected))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_required_checkpoint(
    explicit_path: str | None,
    *,
    env_var: str,
    default_path: Path,
    label: str,
) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(default_path)

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.exists():
            return resolved

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        f"{label} checkpoint not found.\n"
        f"Checked:\n{checked_text}\n"
        f"Pass the explicit CLI flag or set {env_var}."
    )


def _resolve_optional_checkpoint(
    explicit_path: str | None,
    *,
    env_var: str,
    default_path: Path,
) -> Path | None:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(default_path)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def _state_signature(state: dict[str, Any], legal: list[dict[str, Any]]) -> str:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
    enemy_sig = [
        (
            str(enemy.get("name") or enemy.get("id") or "?"),
            int(enemy.get("hp", enemy.get("current_hp", 0)) or 0),
            int(enemy.get("block", 0) or 0),
        )
        for enemy in enemies
        if isinstance(enemy, dict)
    ]
    legal_sig = [
        (
            str(action.get("action") or ""),
            str(action.get("label") or ""),
            int(action.get("index", -1) or -1),
        )
        for action in legal
        if isinstance(action, dict)
    ]
    signature = {
        "state_type": str(state.get("state_type") or "").lower(),
        "floor": int(run.get("floor", 0) or 0),
        "act": int(run.get("act", 0) or 0),
        "round": int(battle.get("round", 0) or 0),
        "hp": int((player.get("hp", player.get("current_hp", 0)) or 0)),
        "energy": int((player.get("energy", 0) or 0)),
        "enemy_sig": enemy_sig,
        "legal_sig": legal_sig,
    }
    return json.dumps(signature, ensure_ascii=False, sort_keys=True)


def _get_card_select_state(state: dict[str, Any]) -> dict[str, Any]:
    card_select = state.get("card_select")
    return card_select if isinstance(card_select, dict) else {}


def _indexed_action_sort_key(action: dict[str, Any]) -> tuple[int, int]:
    raw_index = action.get("index")
    try:
        index = int(raw_index)
    except Exception:
        index = 1_000_000
    return (index, len(str(action.get("label") or "")))


def _card_select_selected_indexes(state: dict[str, Any]) -> set[int]:
    card_select = _get_card_select_state(state)
    selected_indexes: set[int] = set()

    selected_cards = card_select.get("selected_cards")
    if isinstance(selected_cards, list):
        for item in selected_cards:
            if not isinstance(item, dict):
                continue
            try:
                selected_indexes.add(int(item.get("index")))
            except Exception:
                continue

    cards = card_select.get("cards")
    if isinstance(cards, list):
        for item in cards:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("is_selected")):
                continue
            try:
                selected_indexes.add(int(item.get("index")))
            except Exception:
                continue

    return selected_indexes


def _normalize_card_select_legal_actions(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    card_select = _get_card_select_state(state)
    if not card_select or not legal:
        return legal

    def _action_name(action: dict[str, Any]) -> str:
        return str(action.get("action") or "").strip().lower()

    try:
        remaining_picks = int(card_select.get("remaining_picks"))
    except Exception:
        remaining_picks = -1

    try:
        selected_count = int(card_select.get("selected_count"))
    except Exception:
        selected_count = len(_card_select_selected_indexes(state))

    try:
        max_select = int(card_select.get("max_select"))
    except Exception:
        max_select = -1

    can_confirm = bool(card_select.get("can_confirm"))
    can_cancel = bool(card_select.get("can_cancel"))

    if max_select > 0 and selected_count > max_select and can_cancel:
        cancel_only = [action for action in legal if _action_name(action) == "cancel_selection"]
        if cancel_only:
            return cancel_only

    if remaining_picks == 0:
        confirm_cancel_only = [
            action for action in legal
            if _action_name(action) in {"confirm_selection", "cancel_selection"}
        ]
        if can_confirm and confirm_cancel_only:
            return confirm_cancel_only
        if confirm_cancel_only:
            return confirm_cancel_only

    return legal


def _choose_card_select_escape_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
) -> dict[str, Any] | None:
    card_select = _get_card_select_state(state)
    if not card_select:
        return None

    legal = _normalize_card_select_legal_actions(state, legal)
    selected_indexes = _card_select_selected_indexes(state)
    remaining_picks_raw = card_select.get("remaining_picks")
    try:
        remaining_picks = int(remaining_picks_raw)
    except Exception:
        remaining_picks = -1
    try:
        selected_count = int(card_select.get("selected_count"))
    except Exception:
        selected_count = len(selected_indexes)
    try:
        max_select = int(card_select.get("max_select"))
    except Exception:
        max_select = -1

    select_actions = [
        action
        for action in legal
        if str(action.get("action") or "").strip().lower() == "select_card"
    ]
    confirm = next(
        (
            action
            for action in legal
            if str(action.get("action") or "").strip().lower() == "confirm_selection"
        ),
        None,
    )
    cancel = next(
        (
            action
            for action in legal
            if str(action.get("action") or "").strip().lower() == "cancel_selection"
        ),
        None,
    )

    if max_select > 0 and selected_count > max_select and cancel is not None:
        return cancel

    if remaining_picks > 0:
        unselected = [
            action for action in select_actions
            if action.get("index") not in selected_indexes
        ]
        if unselected:
            return min(unselected, key=_indexed_action_sort_key)

    if confirm is not None and (selected_indexes or remaining_picks == 0):
        return confirm

    if select_actions:
        return min(select_actions, key=_indexed_action_sort_key)

    return confirm or cancel


def _choose_auto_progress_action(state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
    st = str(state.get("state_type") or "").strip().lower()
    if st == "card_select":
        card_select_action = _choose_card_select_escape_action(state, legal)
        if card_select_action is not None:
            return card_select_action

    if st in SELECTION_SCREENS:
        for preferred in ("confirm_selection", "combat_confirm_selection", "skip", "cancel_selection"):
            for action in legal:
                if action.get("action") == preferred:
                    return action
        for action in legal:
            if "select" in str(action.get("action") or ""):
                return action

    if st == "combat_rewards":
        for preferred in ("claim_reward", "proceed", "skip"):
            for action in legal:
                if action.get("action") == preferred:
                    return action
    return None


def _choose_combat_selection_progress_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    *,
    repeat_count: int,
) -> dict[str, Any] | None:
    card_selection = state.get("card_selection") if isinstance(state.get("card_selection"), dict) else {}
    if not card_selection:
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        card_selection = battle.get("card_selection") if isinstance(battle.get("card_selection"), dict) else {}
    if not card_selection:
        return None

    selected = card_selection.get("selected_options")
    has_selected = isinstance(selected, list) and len(selected) > 0
    can_confirm = bool(card_selection.get("can_confirm"))
    confirm = next(
        (
            action
            for action in legal
            if str(action.get("action") or "").strip().lower() in {"combat_confirm_selection", "confirm_selection"}
        ),
        None,
    )
    if confirm is None:
        return None

    if has_selected and can_confirm:
        return confirm

    if repeat_count >= 2 and can_confirm:
        return confirm

    return None


def _choose_rest_site_stall_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    last_action_name: str,
    repeat_count: int,
) -> dict[str, Any] | None:
    st = str(state.get("state_type") or "").strip().lower()
    if st != "rest_site" or repeat_count < 1:
        return None

    proceed = next(
        (action for action in legal if str(action.get("action") or "").strip().lower() == "proceed"),
        None,
    )
    rest_options = [
        action
        for action in legal
        if str(action.get("action") or "").strip().lower() == "choose_rest_option"
    ]

    if last_action_name == "choose_rest_option":
        if len(rest_options) > 1:
            return max(rest_options, key=_indexed_action_sort_key)
        if proceed is not None:
            return proceed

    return None


def _should_wait_for_visible_transition(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    last_action_name: str,
    repeat_count: int,
    *,
    use_pipe: bool,
) -> bool:
    if use_pipe or repeat_count < 1:
        return False

    st = str(state.get("state_type") or "").strip().lower()
    if st != "rest_site":
        return False

    if last_action_name != "proceed":
        return False

    non_wait_actions = [
        action for action in legal
        if str(action.get("action") or "").strip().lower() != "wait"
    ]
    return (
        len(non_wait_actions) == 1
        and str(non_wait_actions[0].get("action") or "").strip().lower() == "proceed"
    )


def _is_benign_visible_dispatch_failure(
    state_type: str,
    action_name: str,
    exc: Exception,
    *,
    use_pipe: bool,
) -> bool:
    if use_pipe:
        return False
    text = str(exc).strip().lower()
    return (
        state_type == "event"
        and action_name == "advance_dialogue"
        and (
            "dialogue hitbox not available" in text
            or "no ancient dialogue active" in text
        )
    )


def _choose_repeat_escape_action(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    last_action_name: str = "",
) -> dict[str, Any] | None:
    rest_site_stall = _choose_rest_site_stall_action(state, legal, last_action_name, repeat_count=3)
    if rest_site_stall is not None:
        return rest_site_stall
    auto = _choose_auto_progress_action(state, legal)
    if auto is not None:
        return auto
    for preferred in ESCAPE_ACTION_NAMES:
        for action in legal:
            if action.get("action") == preferred:
                return action
    if len(legal) > 1:
        return legal[1]
    return legal[0] if legal else None


def _is_combat_context(state: dict[str, Any]) -> bool:
    st = str(state.get("state_type") or "").strip().lower()
    if st in {"combat", "monster", "elite", "boss", "hand_select"}:
        return True
    if st != "card_select":
        return False
    if isinstance(state.get("battle"), dict):
        return True
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    room_type = str(run.get("room_type") or "").strip().lower()
    if room_type in {"monster", "elite", "boss"}:
        return True
    card_select = state.get("card_select") if isinstance(state.get("card_select"), dict) else {}
    screen_type = str(card_select.get("screen_type") or "").strip().lower()
    return screen_type in {"combat", "combat_reward", "combat_select"}


def _is_actionable_visible_combat_state(state: dict[str, Any]) -> bool:
    st = str(state.get("state_type") or "").strip().lower()
    if st not in {"combat", "monster", "elite", "boss", "hand_select", "card_select"}:
        return False
    if st == "hand_select":
        return True
    if isinstance(state.get("card_selection"), dict):
        return True

    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    if not (bool(battle.get("is_play_phase")) and str(battle.get("turn") or "").strip().lower() == "player"):
        return False

    player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
    if not player:
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
    hand = list(player.get("hand") or [])
    if not hand:
        return True
    if any(bool(card.get("can_play")) for card in hand if isinstance(card, dict)):
        return True

    reasons = {
        str(card.get("unplayable_reason") or "").strip().lower()
        for card in hand
        if isinstance(card, dict) and card.get("unplayable_reason") is not None
    }
    if reasons and reasons.issubset({"playeractionsdisabled", "disabled", "none"}):
        return False

    return True


def _visible_combat_needs_wait(state: dict[str, Any], *, use_pipe: bool) -> bool:
    return not use_pipe and _is_combat_context(state) and not _is_actionable_visible_combat_state(state)


SUSPICIOUS_TEXT_MARKERS = (
    "LocString",
    "鈥",
    "鍐",
    "璁",
    "妫",
    "鐥",
    "鎵",
    "琛",
    "闃",
    "鍙",
    "鍧",
    "鐔",
    "閫",
    "涓",
    "浜",
    "鍟",
    "浣",
    "浼",
    "閿",
    "鑰",
    "鎰",
    "鏀",
)


DISPLAY_SUSPICIOUS_TEXT_MARKERS = (
    "LocString",
    "鈥",
    "鍐",
    "璁",
    "妫",
    "鐥",
    "鎵",
    "琛",
    "闃",
    "鍙",
    "鍧",
    "鐔",
    "閫",
    "涓",
    "浜",
    "鍟",
    "浣",
    "浼",
    "閿",
    "鑰",
    "鎰",
    "鏀",
)


def _looks_garbled(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    if not text.strip():
        return False
    if any(marker in text for marker in DISPLAY_SUSPICIOUS_TEXT_MARKERS):
        return True
    if "?" in text and len(text) >= 3:
        return True
    return False


def _lookup_indexed_name(items: Any, index: Any) -> str | None:
    if not isinstance(items, list):
        return None
    try:
        idx = int(index)
    except Exception:
        return None
    if idx < 0 or idx >= len(items):
        return None
    item = items[idx]
    if not isinstance(item, dict):
        return None
    for key in ("name", "label", "option_text", "card_name", "id", "card_id", "title", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip() and not _looks_garbled(value):
            return value
    for key in ("id", "card_id", "name", "label", "option_text", "card_name", "title", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _pretty_action_name(action_name: str) -> str:
    text = str(action_name or "").strip().replace("_", " ")
    if not text:
        return "Action"
    return " ".join(part.capitalize() for part in text.split())


def _lookup_map_option_label(state: dict[str, Any], index: Any) -> str | None:
    next_options = ((state.get("map") or {}).get("next_options")) if isinstance(state.get("map"), dict) else None
    if not isinstance(next_options, list):
        return None
    try:
        idx = int(index)
    except Exception:
        return None
    option = next((item for item in next_options if isinstance(item, dict) and int(item.get("index", -1)) == idx), None)
    if not isinstance(option, dict):
        return None
    for key in ("label", "name", "title", "node_type", "room_type", "type"):
        value = option.get(key)
        if isinstance(value, str) and value.strip():
            raw = value.strip()
            return raw if any(ch > "\x7f" for ch in raw) else _pretty_action_name(raw)
    return None


def _clean_action_label(action: dict[str, Any], state: dict[str, Any]) -> str:
    act_name = str(action.get("action") or "")
    label = action.get("label")
    note = action.get("note")

    if isinstance(label, str) and label.strip() and not _looks_garbled(label):
        return label

    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    room_type = str(run.get("room_type") or "").lower()
    hand = None
    if isinstance(battle.get("player"), dict):
        hand = battle["player"].get("hand")
    if hand is None:
        hand = player.get("hand")

    if act_name == "play_card":
        card_name = _lookup_indexed_name(hand, action.get("card_index"))
        if not card_name:
            card_name = _lookup_indexed_name(hand, action.get("index"))
        if card_name:
            return card_name
        card_idx = action.get("card_index", action.get("index"))
        if card_idx is not None:
            return f"Play Card #{card_idx}"
        return "Play Card"

    if act_name == "use_potion":
        slot = action.get("slot")
        if slot is not None:
            return f"Use Potion #{slot}"
        return "Use Potion"

    if act_name in ("confirm_selection", "combat_confirm_selection"):
        return "Confirm"
    if act_name == "claim_reward":
        rewards = state.get("rewards") if isinstance(state.get("rewards"), dict) else {}
        reward_name = _lookup_indexed_name(rewards.get("items"), action.get("index"))
        return reward_name or "Claim Reward"
    if act_name == "proceed":
        state_type = str(state.get("state_type") or "").lower()
        proceed_labels = {
            "combat_rewards": "Leave Rewards",
            "treasure": "Leave Treasure",
            "shop": "Leave Shop",
            "event": "Continue",
            "rest_site": "Leave Rest Site",
        }
        return proceed_labels.get(state_type, "Proceed")
    if act_name == "skip":
        return "Skip"
    if act_name == "cancel_selection":
        return "Cancel"
    if act_name == "end_turn":
        return "End Turn"
    if act_name == "wait":
        return "Wait"

    state_type = str(state.get("state_type") or "").lower()
    if state_type == "event":
        option_name = _lookup_indexed_name((state.get("event") or {}).get("options"), action.get("index"))
        if option_name:
            return option_name
        if act_name == "advance_dialogue":
            return "Continue Event"
    if state_type == "card_reward":
        card_name = _lookup_indexed_name(
            (state.get("card_reward") or {}).get("cards"),
            action.get("index", action.get("card_index")),
        )
        if card_name:
            return card_name
    if state_type in ("hand_select", "card_select"):
        cards = (state.get("hand_select") or {}).get("cards") or (state.get("card_select") or {}).get("cards")
        card_name = _lookup_indexed_name(cards, action.get("index", action.get("card_index")))
        if card_name:
            return card_name
    if state_type == "shop":
        item_name = _lookup_indexed_name((state.get("shop") or {}).get("items"), action.get("index"))
        if item_name:
            return item_name
    if state_type == "rest_site":
        opt_name = _lookup_indexed_name((state.get("rest_site") or {}).get("options"), action.get("index"))
        if opt_name:
            return opt_name
    if state_type == "combat_rewards":
        rewards = state.get("rewards") if isinstance(state.get("rewards"), dict) else {}
        reward_name = _lookup_indexed_name(rewards.get("items"), action.get("index"))
        if reward_name:
            return reward_name
    if state_type == "treasure":
        relic_name = _lookup_indexed_name((state.get("treasure") or {}).get("relics"), action.get("index"))
        if relic_name:
            return relic_name
    if state_type == "relic_select":
        relic_name = _lookup_indexed_name((state.get("relic_select") or {}).get("relics"), action.get("index"))
        if relic_name:
            return relic_name
        if act_name == "skip_relic_selection":
            return "Skip Relic"
    if state_type == "map":
        option_label = _lookup_map_option_label(state, action.get("index"))
        if option_label:
            return option_label
        if room_type:
            return _pretty_action_name(room_type)

    if isinstance(note, str) and note.strip() and not _looks_garbled(note):
        return note
    if isinstance(label, str) and label.strip():
        return label
    return _pretty_action_name(act_name)


def _combat_hand(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    if isinstance(battle.get("hand"), list):
        return [card for card in battle["hand"] if isinstance(card, dict)]
    if isinstance(player.get("hand"), list):
        return [card for card in player["hand"] if isinstance(card, dict)]
    return []


def _combat_enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else state.get("enemies")
    if not isinstance(enemies, list):
        return []
    return [enemy for enemy in enemies if isinstance(enemy, dict)]


def _card_for_action(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    if str(action.get("action") or "").lower() != "play_card":
        return None
    try:
        idx = int(action.get("card_index", action.get("index", -1)))
    except Exception:
        return None
    hand = _combat_hand(state)
    if 0 <= idx < len(hand):
        return hand[idx]
    return None


def _normalize_action_payload_for_transport(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    use_pipe: bool,
) -> dict[str, Any]:
    payload = dict(action)
    act_name = str(payload.get("action") or "").strip().lower()
    state_type = str(state.get("state_type") or "").strip().lower()

    # The visible front-end selection screens expect grid-style indices even when
    # upstream candidate actions still carry legacy card_index fields.
    if act_name == "select_card" and "index" not in payload and "card_index" in payload:
        payload["index"] = payload["card_index"]
        if not use_pipe:
            payload.pop("card_index", None)

    if act_name == "select_card_reward":
        if "card_index" not in payload and "index" in payload:
            payload["card_index"] = payload["index"]
        elif "index" not in payload and "card_index" in payload:
            payload["index"] = payload["card_index"]

    # Keep front-UI payloads minimal so unexpected legacy fields do not leak into
    # visible-mode dispatch.
    if not use_pipe and state_type == "card_select":
        allowed = {"action", "index", "value"}
        payload = {key: value for key, value in payload.items() if key in allowed}

    if not use_pipe and state_type == "card_reward":
        allowed = {"action", "card_index", "index", "value"}
        payload = {key: value for key, value in payload.items() if key in allowed}

    return payload


def _should_retry_visible_action(state_type: str, exc: Exception) -> bool:
    text = str(exc).strip().lower()
    if not text:
        return False
    if state_type == "rest_site" and "not open" in text:
        return True
    if state_type == "event" and (
        "dialogue hitbox not available" in text
        or "no ancient dialogue active" in text
    ):
        return True
    if state_type in {"shop", "treasure"} and "no proceed button available or enabled" in text:
        return True
    if state_type in {"card_select", "card_reward"} and "missing 'index'" in text:
        return True
    if state_type == "card_reward" and "missing 'card_index'" in text:
        return True
    return False


def _enemy_for_action(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    target = action.get("target") or action.get("target_id")
    if target in (None, ""):
        return None
    for enemy in _combat_enemies(state):
        if target in {
            enemy.get("entity_id"),
            enemy.get("id"),
            enemy.get("monster_id"),
            enemy.get("combat_id"),
        }:
            return enemy
    return None


def _entity_status_amount(entity: dict[str, Any] | None, keyword: str) -> int:
    if not isinstance(entity, dict):
        return 0
    for status_key in ("status", "powers", "buffs"):
        statuses = entity.get(status_key)
        if not isinstance(statuses, list):
            continue
        for status in statuses:
            if not isinstance(status, dict):
                continue
            sid = str(status.get("id") or status.get("name") or status.get("title") or "").lower()
            if keyword in sid:
                try:
                    return int(status.get("amount") or status.get("stacks") or status.get("counter") or 0)
                except Exception:
                    return 0
    return 0


def _card_text(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    parts = [
        str(card.get("id") or ""),
        str(card.get("name") or ""),
        str(card.get("title") or ""),
        str(card.get("description") or ""),
        str(card.get("type") or ""),
    ]
    return " ".join(parts).lower()


def _is_vulnerable_setup(card: dict[str, Any] | None) -> bool:
    text = _card_text(card)
    return any(token in text for token in ("bash", "uppercut", "tremble", "taunt", "痛击", "上勾拳", "战栗", "挑衅", "易伤", "vulnerable"))


def _is_attack_card(card: dict[str, Any] | None) -> bool:
    text = _card_text(card)
    return "attack" in text or "造成" in text or "伤害" in text


def _is_block_card(card: dict[str, Any] | None) -> bool:
    if not isinstance(card, dict):
        return False
    if bool(card.get("gains_block")):
        return True
    text = _card_text(card)
    return any(token in text for token in ("defend", "block", "格挡", "获得"))


def _is_body_slam(card: dict[str, Any] | None) -> bool:
    text = _card_text(card)
    return any(token in text for token in ("body_slam", "body slam", "全身撞击", "当前格挡值"))


# Clean overrides for demo/recording mode so the on-screen reasoning stays readable.
def _is_vulnerable_setup(card: dict[str, Any] | None) -> bool:
    text = _card_text(card)
    return any(token in text for token in ("bash", "uppercut", "tremble", "taunt", "vulnerable"))


def _is_attack_card(card: dict[str, Any] | None) -> bool:
    text = _card_text(card)
    return "attack" in text or "damage" in text


def _is_block_card(card: dict[str, Any] | None) -> bool:
    if not isinstance(card, dict):
        return False
    if bool(card.get("gains_block")):
        return True
    text = _card_text(card)
    return any(token in text for token in ("defend", "block", "gain block"))


def _is_body_slam(card: dict[str, Any] | None) -> bool:
    text = _card_text(card)
    return any(token in text for token in ("body_slam", "body slam"))


def _combat_tactical_rerank(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    probs: np.ndarray,
    default_idx: int,
) -> tuple[int, str | None]:
    if not legal:
        return default_idx, None

    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    try:
        player_block = int(player.get("block", 0) or 0)
    except Exception:
        player_block = 0
    adjusted = np.array(probs[:len(legal)], dtype=np.float32)
    reasons: list[str] = []

    play_cards = [(_card_for_action(state, action), action) for action in legal]
    has_block_play = any(_is_block_card(card) for card, action in play_cards if str(action.get("action") or "").lower() == "play_card")
    has_followup_attack = any(_is_attack_card(card) and not _is_vulnerable_setup(card) for card, action in play_cards if str(action.get("action") or "").lower() == "play_card")

    for i, action in enumerate(legal):
        act_name = str(action.get("action") or "").lower()
        if act_name != "play_card":
            continue
        card = _card_for_action(state, action)
        enemy = _enemy_for_action(state, action)

        if _is_body_slam(card) and player_block <= 0 and has_block_play:
            adjusted[i] -= 1.25
            reasons.append("delay_zero_block_body_slam")

        if _is_block_card(card):
            body_slam_exists = any(_is_body_slam(other_card) for other_card, other_action in play_cards if str(other_action.get("action") or "").lower() == "play_card")
            if player_block <= 0 and body_slam_exists:
                adjusted[i] += 0.18

        if _is_vulnerable_setup(card) and enemy is not None:
            if _entity_status_amount(enemy, "vulnerable") <= 0 and has_followup_attack:
                adjusted[i] += 0.55
                reasons.append("prefer_vulnerable_setup")

        if _is_attack_card(card) and not _is_vulnerable_setup(card) and enemy is not None:
            if _entity_status_amount(enemy, "vulnerable") <= 0 and any(
                _is_vulnerable_setup(other_card) and (other_action.get("target") or other_action.get("target_id")) in {action.get("target"), action.get("target_id")}
                for other_card, other_action in play_cards
                if str(other_action.get("action") or "").lower() == "play_card"
            ):
                adjusted[i] -= 0.22

    best_idx = int(np.argmax(adjusted[:len(legal)]))
    if best_idx != int(default_idx):
        return best_idx, ",".join(sorted(set(reasons))) or "tactical_rerank"
    return default_idx, None


class DecisionOverlayFileWriter:
    """Best-effort JSON writer for the in-game AI HUD."""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None

    def write(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        temp_path = self.path.with_name(f"{self.path.name}.tmp")
        for _ in range(5):
            try:
                temp_path.write_text(raw, encoding="utf-8")
                temp_path.replace(self.path)
                return
            except PermissionError:
                time.sleep(0.03)
        temp_path.write_text(raw, encoding="utf-8")
        temp_path.replace(self.path)

    def clear(self) -> None:
        if self.path is None:
            return
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass


class DemoArtifactWriter:
    """Write lightweight recording artifacts for demo runs."""

    def __init__(self, output_dir: str | None):
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        self.trace_path = self.output_dir / "decision_trace.jsonl" if self.output_dir else None
        self.summary_path = self.output_dir / "demo_summary.json" if self.output_dir else None
        self._episodes: list[dict[str, Any]] = []
        self._session_meta: dict[str, Any] = {
            "generated_at": _utc_now_iso(),
        }

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_decision(self, payload: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        record = {
            "timestamp_utc": _utc_now_iso(),
            **payload,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str))
            handle.write("\n")

    def record_episode(self, summary: dict[str, Any]) -> None:
        self._episodes.append(summary)
        self._flush_summary()

    def finalize(self, **meta: Any) -> None:
        self._session_meta.update(meta)
        self._flush_summary()

    def _flush_summary(self) -> None:
        if self.summary_path is None:
            return
        payload = {
            **self._session_meta,
            "episode_count": len(self._episodes),
            "episodes": self._episodes,
        }
        self.summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# WebSocket broadcaster with bidirectional control
# ---------------------------------------------------------------------------

class DecisionBroadcaster:
    """WebSocket server that broadcasts AI decisions and accepts control commands."""

    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._latest: str = "{}"

        # Playback controls
        self.paused = threading.Event()
        self.paused.set()  # not paused initially
        self.step_once = False
        self.speed_multiplier = 1.0
        self._cmd_file: Path | None = None

    def start(self):
        """Start WebSocket + HTTP server in daemon threads."""
        threading.Thread(target=self._run_ws, daemon=True, name="WS-Broadcaster").start()
        overlay_dir = Path(__file__).parent / "overlay"
        if overlay_dir.exists():
            threading.Thread(target=self._run_http, args=(overlay_dir,),
                             daemon=True, name="HTTP-Overlay").start()
        logger.info("Overlay: http://localhost:%d  |  WebSocket: ws://localhost:%d/ws",
                     self.port, self.port + 1)

    def broadcast(self, decision: dict):
        """Send decision to all connected WebSocket clients."""
        if "msg_type" not in decision:
            decision["msg_type"] = "decision"
        msg = json.dumps(decision, ensure_ascii=False, default=str)
        self._latest = msg
        if self._loop and self.clients:
            asyncio.run_coroutine_threadsafe(self._send_all(msg), self._loop)

    async def _send_all(self, msg: str):
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    def _run_ws(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        start_server = websockets.server.serve(
            self._handler, "0.0.0.0", self.port + 1)
        self._loop.run_until_complete(start_server)
        self._loop.run_forever()

    async def _handler(self, ws):
        self.clients.add(ws)
        logger.info("Overlay client connected (%d total)", len(self.clients))
        try:
            await ws.send(self._latest)
            await ws.send(json.dumps({
                "msg_type": "playback_state",
                "paused": not self.paused.is_set(),
                "speed": self.speed_multiplier,
            }))
            async for msg in ws:
                try:
                    cmd = json.loads(msg)
                    self._handle_command(cmd)
                except (json.JSONDecodeError, KeyError):
                    pass
        finally:
            self.clients.discard(ws)

    def _handle_command(self, cmd: dict):
        command = cmd.get("command", "")
        if command == "pause":
            self.paused.clear()
            logger.info("Playback paused")
            self._broadcast_playback_state()
        elif command == "resume":
            self.paused.set()
            logger.info("Playback resumed")
            self._broadcast_playback_state()
        elif command == "step":
            self.step_once = True
            self.paused.set()
            logger.info("Single step")
        elif command == "speed":
            self.speed_multiplier = max(0.1, min(10.0, float(cmd.get("value", 1.0))))
            logger.info("Speed: %.1fx", self.speed_multiplier)
            self._broadcast_playback_state()

    def _broadcast_playback_state(self):
        self.broadcast({
            "msg_type": "playback_state",
            "paused": not self.paused.is_set(),
            "speed": self.speed_multiplier,
        })

    def set_cmd_file(self, overlay_file: str | None):
        """Set the playback.cmd file path (next to overlay JSON)."""
        if overlay_file:
            self._cmd_file = Path(overlay_file).parent / "playback.cmd"

    def _poll_cmd_file(self):
        """Check for file-based control commands from the game overlay."""
        if self._cmd_file is None or not self._cmd_file.exists():
            return
        try:
            cmd = self._cmd_file.read_text().strip()
            self._cmd_file.unlink(missing_ok=True)
            if cmd:
                logger.info("File command: %s", cmd)
                self._handle_command({"command": cmd})
        except Exception:
            pass

    def wait_if_paused(self):
        """Block until unpaused. Returns True when ready to continue."""
        self._poll_cmd_file()
        while not self.paused.is_set() and not _shutdown:
            self._poll_cmd_file()
            self.paused.wait(timeout=0.1)
        if self.step_once:
            self.step_once = False
            self.paused.clear()
            return True
        return True

    def _run_http(self, overlay_dir: Path):
        broadcaster = self

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/api/pause":
                    broadcaster._handle_command({"command": "pause"})
                    self._json_ok({"status": "paused"})
                elif path == "/api/resume":
                    broadcaster._handle_command({"command": "resume"})
                    self._json_ok({"status": "resumed"})
                elif path == "/api/step":
                    broadcaster._handle_command({"command": "step"})
                    self._json_ok({"status": "stepped"})
                elif path == "/api/status":
                    self._json_ok({
                        "paused": not broadcaster.paused.is_set(),
                        "speed": broadcaster.speed_multiplier,
                    })
                else:
                    os.chdir(str(overlay_dir))
                    super().do_GET()

            def _json_ok(self, data):
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("0.0.0.0", self.port), Handler)
        server.serve_forever()


# ---------------------------------------------------------------------------
# Reasoning generator (rule-based, bilingual)
# ---------------------------------------------------------------------------

def generate_reasoning(
    decision_type: str, action: dict, state: dict,
    legal: list[dict], value: float, probs: list[float],
) -> tuple[str, str]:
    """Generate bilingual reasoning text for the AI's decision."""
    act_name = action.get("action", "")
    label = _clean_action_label(action, state)
    player = state.get("player") or state.get("battle", {}).get("player") or {}
    hp = player.get("hp", player.get("current_hp", 0))
    max_hp = player.get("max_hp", 80)
    hp_ratio = hp / max(1, max_hp)
    energy = player.get("energy", 0)

    if decision_type == "combat":
        battle = state.get("battle") or {}
        enemies = battle.get("enemies") or []
        total_enemy_hp = sum(e.get("hp", 0) for e in enemies)

        if act_name == "end_turn":
            if energy == 0:
                return ("能量用尽，结束回合", "No energy left, ending turn")
            return ("当前没有更好的出牌选择", "No better card to play")

        if act_name == "play_card":
            if total_enemy_hp <= 15:
                return (f"打出 {label}，接近斩杀", f"Playing {label}, close to lethal")

            enemy_attacking = any(
                "attack" in (i.get("type", "")).lower()
                for e in enemies for i in (e.get("intents") or [])
            )

            card_type_guess = "attack"
            lbl_lower = label.lower() if label else ""
            if any(k in lbl_lower for k in ("defend", "block")):
                card_type_guess = "block"

            if card_type_guess == "block" and enemy_attacking:
                return (
                    f"打出 {label} 提供格挡，对手准备进攻",
                    f"Playing {label} for block, enemy attacking",
                )
            if card_type_guess == "attack":
                return (f"打出 {label} 进行输出", f"Playing {label}, dealing damage")
            return (f"打出 {label}", f"Playing {label}")

        if act_name == "use_potion":
            return (f"使用药水：{label}", f"Using potion: {label}")

    elif decision_type == "map":
        st_next = _clean_action_label(action, state)
        if "rest" in st_next.lower():
            if hp_ratio < 0.5:
                return ("血量偏低，优先休息点", "Low HP, choosing rest site")
            return ("前往休息点", "Heading to rest site")
        if "elite" in st_next.lower():
            return ("选择精英路线", "Challenging elite enemy")
        if "shop" in st_next.lower():
            return ("前往商店", "Visiting the shop")
        if "monster" in st_next.lower():
            return ("前往普通战斗", "Heading to combat")
        return (f"选择路线：{st_next}", f"Choosing path: {st_next}")

    elif decision_type == "card_reward":
        if "skip" in act_name.lower():
            return ("跳过卡牌奖励，保持牌组精简", "Skipping card reward, deck lean enough")
        return (f"选择卡牌：{label}", f"Picking card: {label}")

    elif decision_type == "rest":
        if "rest" in act_name.lower() or "rest" in label.lower():
            return ("选择休息回复生命", "Resting to recover HP")
        if "smith" in act_name.lower() or "upgrade" in act_name.lower():
            return ("选择锻造升级卡牌", "Upgrading a card at the smithy")
        return (f"{label}", f"{label}")

    elif decision_type == "shop":
        return (f"商店选择：{label}", f"Shop: {label}")

    elif decision_type == "event":
        return (f"事件选择：{label}", f"Event choice: {label}")

    return (f"{label}", f"{label}")

# Clean override for on-screen bilingual reasoning so recording mode never shows mojibake.
def generate_reasoning(
    decision_type: str, action: dict, state: dict,
    legal: list[dict], value: float, probs: list[float],
) -> tuple[str, str]:
    act_name = str(action.get("action") or "")
    label = _clean_action_label(action, state)
    player = state.get("player") or state.get("battle", {}).get("player") or {}
    hp = float(player.get("hp", player.get("current_hp", 0)) or 0)
    max_hp = max(1.0, float(player.get("max_hp", 80) or 80))
    hp_ratio = hp / max_hp
    energy = int(player.get("energy", 0) or 0)

    if decision_type == "combat":
        battle = state.get("battle") or {}
        enemies = battle.get("enemies") or []
        total_enemy_hp = sum(int(enemy.get("hp", 0) or 0) for enemy in enemies if isinstance(enemy, dict))
        enemy_attacking = any(
            "attack" in str(intent.get("type", "")).lower()
            for enemy in enemies if isinstance(enemy, dict)
            for intent in (enemy.get("intents") or [])
            if isinstance(intent, dict)
        )

        if act_name == "end_turn":
            if energy == 0:
                return ("能量已经打空，先结束回合。", "No energy remains, so we end the turn.")
            return ("这一拍没有更高价值的操作了。", "There is no higher-value play on this turn.")

        if act_name == "play_card":
            if total_enemy_hp <= 15:
                return (f"打出 {label}，这一拍已经接近斩杀。", f"Playing {label}; this line is close to lethal.")
            card = _card_for_action(state, action)
            if _is_block_card(card) and enemy_attacking:
                return (
                    f"先用 {label} 补格挡，挡住对手这一轮进攻。",
                    f"Using {label} for block because the enemy is attacking.",
                )
            if _is_attack_card(card):
                return (f"打出 {label} 主动推进伤害。", f"Playing {label} to push damage.")
            return (f"打出 {label}，维持当前节奏。", f"Playing {label} to keep the current tempo.")

        if act_name == "use_potion":
            return (f"使用药水：{label}。", f"Using potion: {label}.")

        if act_name in {"confirm_selection", "combat_confirm_selection"}:
            return ("确认当前战斗选择。", "Confirming the current combat selection.")

    if decision_type == "map":
        next_node = _clean_action_label(action, state)
        next_lower = next_node.lower()
        if "rest" in next_lower:
            if hp_ratio < 0.5:
                return ("血量偏低，优先走休息点。", "HP is low, so we route toward a rest site.")
            return ("路线偏向休息点，先把状态稳住。", "Heading toward a rest site to stabilize.")
        if "elite" in next_lower:
            return ("选择精英路线，主动换更高回报。", "Taking the elite route for a higher upside.")
        if "shop" in next_lower:
            return ("这条路线去商店，把资源转成强度。", "Routing into the shop to convert resources into power.")
        if "monster" in next_lower:
            return ("优先走普通战斗，继续稳步推进。", "Taking a normal combat for steady progress.")
        return (f"选择路线：{next_node}。", f"Choosing path: {next_node}.")

    if decision_type == "card_reward":
        if "skip" in act_name.lower():
            return ("这次跳过卡牌奖励，保持牌组更紧凑。", "Skipping the card reward to keep the deck tight.")
        return (f"选择这张卡：{label}。", f"Taking this card: {label}.")

    if decision_type == "rest":
        lower = label.lower()
        if "rest" in act_name.lower() or "rest" in lower:
            return ("在营火休息，先把血量补回来。", "Resting at the fire to recover HP.")
        if "smith" in act_name.lower() or "upgrade" in lower:
            return ("在营火升级卡牌，换更强的后续回合。", "Upgrading a card at the fire for stronger future turns.")
        return (f"营火选择：{label}。", f"Campfire choice: {label}.")

    if decision_type == "shop":
        return (f"商店选择：{label}。", f"Shop choice: {label}.")

    if decision_type == "event":
        return (f"事件选择：{label}。", f"Event choice: {label}.")

    if decision_type == "treasure":
        return (f"宝箱界面操作：{label}。", f"Treasure action: {label}.")

    return (f"当前动作：{label}。", f"Current action: {label}.")


# Final clean override for visible demo mode. The last definition wins and keeps
# all viewer-facing strings readable even if older legacy blocks still exist.
def generate_reasoning(
    decision_type: str, action: dict, state: dict,
    legal: list[dict], value: float, probs: list[float],
) -> tuple[str, str]:
    act_name = str(action.get("action") or "")
    label = _clean_action_label(action, state)
    player = state.get("player") or state.get("battle", {}).get("player") or {}
    hp = float(player.get("hp", player.get("current_hp", 0)) or 0)
    max_hp = max(1.0, float(player.get("max_hp", 80) or 80))
    hp_ratio = hp / max_hp
    energy = int(player.get("energy", 0) or 0)

    if decision_type == "combat":
        battle = state.get("battle") or {}
        enemies = battle.get("enemies") or []
        total_enemy_hp = sum(int(enemy.get("hp", 0) or 0) for enemy in enemies if isinstance(enemy, dict))
        enemy_attacking = any(
            "attack" in str(intent.get("type", "")).lower()
            for enemy in enemies if isinstance(enemy, dict)
            for intent in (enemy.get("intents") or [])
            if isinstance(intent, dict)
        )

        if act_name == "end_turn":
            if energy == 0:
                return ("能量已经打空了，先结束这一回合。", "No energy remains, so we end the turn.")
            return ("这一拍没有更高价值的操作了，先把回合交出去。", "There is no higher-value play on this turn.")

        if act_name == "play_card":
            if total_enemy_hp <= 15:
                return (f"打出 {label}，这一拍已经接近斩杀。", f"Playing {label}; this line is close to lethal.")
            card = _card_for_action(state, action)
            if _is_block_card(card) and enemy_attacking:
                return (
                    f"先用 {label} 补格挡，把这轮伤害接住。",
                    f"Using {label} for block because the enemy is attacking.",
                )
            if _is_attack_card(card):
                return (f"打出 {label} 主动推进伤害。", f"Playing {label} to push damage.")
            return (f"打出 {label}，维持当前节奏。", f"Playing {label} to keep the current tempo.")

        if act_name == "use_potion":
            return (f"使用药水：{label}。", f"Using potion: {label}.")

        if act_name in {"confirm_selection", "combat_confirm_selection"}:
            return ("确认当前战斗选择。", "Confirming the current combat selection.")

    if decision_type == "map":
        next_node = _clean_action_label(action, state)
        next_lower = next_node.lower()
        if "rest" in next_lower:
            if hp_ratio < 0.5:
                return ("血量偏低，路线优先去休息点。", "HP is low, so we route toward a rest site.")
            return ("这条路线会先去休息点，把状态稳住。", "Heading toward a rest site to stabilize.")
        if "elite" in next_lower:
            return ("选择精英路线，主动换更高回报。", "Taking the elite route for a higher upside.")
        if "shop" in next_lower:
            return ("路线先去商店，把资源转成强度。", "Routing into the shop to convert resources into power.")
        if "monster" in next_lower:
            return ("优先走普通战斗，继续稳定推进。", "Taking a normal combat for steady progress.")
        return (f"选择路线：{next_node}。", f"Choosing path: {next_node}.")

    if decision_type == "card_reward":
        if "skip" in act_name.lower():
            return ("这次跳过卡牌奖励，保持牌组更紧凑。", "Skipping the card reward to keep the deck tight.")
        return (f"选择这张牌：{label}。", f"Taking this card: {label}.")

    if decision_type == "rest":
        lower = label.lower()
        if "rest" in act_name.lower() or "rest" in lower:
            return ("在营火休息，先把血量补回来。", "Resting at the fire to recover HP.")
        if "smith" in act_name.lower() or "upgrade" in lower:
            return ("在营火升级卡牌，换更强的后续回合。", "Upgrading a card at the fire for stronger future turns.")
        return (f"营火选择：{label}。", f"Campfire choice: {label}.")

    if decision_type == "shop":
        return (f"商店选择：{label}。", f"Shop choice: {label}.")

    if decision_type == "event":
        return (f"事件选择：{label}。", f"Event choice: {label}.")

    if decision_type == "treasure":
        return (f"宝箱界面操作：{label}。", f"Treasure action: {label}.")

    return (f"当前动作：{label}。", f"Current action: {label}.")


# ---------------------------------------------------------------------------
# Extract expanded combat state for panorama
# ---------------------------------------------------------------------------

def extract_combat_state(state: dict) -> dict | None:
    """Extract detailed combat state for battle panorama visualization."""
    battle = state.get("battle")
    if not battle:
        return None

    player = battle.get("player") or {}
    hand = battle.get("hand") or []
    enemies = battle.get("enemies") or []

    hand_cards = []
    for card in hand:
        if not isinstance(card, dict):
            continue
        hand_cards.append({
            "name": card.get("name", card.get("id", "?")),
            "cost": card.get("cost", 0),
            "type": card.get("type", "unknown"),
            "playable": card.get("is_playable", True),
            "upgraded": card.get("upgraded", False),
        })

    player_powers = []
    for p in (player.get("powers") or []):
        if isinstance(p, dict):
            player_powers.append({
                "id": p.get("id", p.get("name", "?")),
                "amount": p.get("amount", 0),
            })

    enemies_detail = []
    for e in enemies:
        if not isinstance(e, dict):
            continue
        e_powers = []
        for p in (e.get("powers") or []):
            if isinstance(p, dict):
                e_powers.append({
                    "id": p.get("id", p.get("name", "?")),
                    "amount": p.get("amount", 0),
                })
        intents = e.get("intents") or []
        intent_type = intents[0].get("type", "?") if intents else "?"
        intent_dmg = intents[0].get("damage", 0) if intents else 0
        intent_hits = intents[0].get("hits", 1) if intents else 1

        enemies_detail.append({
            "name": e.get("name", e.get("entity_id", "?")),
            "hp": e.get("hp", 0),
            "max_hp": e.get("max_hp", 1),
            "block": e.get("block", 0),
            "intent": intent_type.lower(),
            "intent_damage": intent_dmg,
            "intent_hits": intent_hits,
            "powers": e_powers,
        })

    return {
        "hand": hand_cards,
        "draw_pile_size": len(battle.get("draw_pile") or []),
        "discard_pile_size": len(battle.get("discard_pile") or []),
        "exhaust_pile_size": len(battle.get("exhaust_pile") or []),
        "player_powers": player_powers,
        "player_block": player.get("block", 0),
        "player_energy": player.get("energy", 0),
        "player_max_energy": player.get("max_energy", 3),
        "enemies": enemies_detail,
        "round": battle.get("round", 0),
    }


# ---------------------------------------------------------------------------
# Build decision JSON
# ---------------------------------------------------------------------------

def build_decision(
    decision_type: str, state: dict, legal: list[dict],
    action: dict, action_idx: int, probs: np.ndarray, value: float,
    step: int,
    action_source: str,
    nn_internals: dict | None = None,
    reward_shaping_data: dict | None = None,
    combat_state: dict | None = None,
) -> dict:
    """Build the decision JSON for WebSocket broadcast."""
    run = state.get("run") or {}
    player = state.get("player") or state.get("battle", {}).get("player") or {}
    battle = state.get("battle") or {}

    hp = player.get("hp", player.get("current_hp", 0))
    max_hp = player.get("max_hp", 80)
    deck = player.get("deck") if isinstance(player.get("deck"), list) else []
    potions = player.get("potions") if isinstance(player.get("potions"), list) else []
    potion_names = [p.get("name", p.get("id", "Empty")) for p in potions
                    if isinstance(p, dict)]

    options = []
    for i, la in enumerate(legal):
        prob = float(probs[i]) if i < len(probs) else 0.0
        opt = {
            "label": _clean_action_label(la, state),
            "prob": round(prob, 3),
            "type": la.get("action", "?"),
            "chosen": bool(i == action_idx),
        }
        if nn_internals and "action_advantages" in nn_internals:
            advs = nn_internals["action_advantages"]
            if i < len(advs):
                opt["advantage"] = advs[i]
        cost = la.get("cost")
        if cost is not None:
            try:
                opt["cost"] = int(cost)
            except Exception:
                pass
        target = (
            la.get("target_name")
            or la.get("target_label")
            or la.get("target")
            or la.get("target_id")
        )
        if target not in (None, ""):
            opt["target"] = str(target)
        options.append(opt)
    options.sort(key=lambda x: -x["prob"])

    enemies_data = []
    for e in (battle.get("enemies") or []):
        intents = e.get("intents") or []
        intent_type = intents[0].get("type", "?") if intents else "?"
        intent_dmg = intents[0].get("damage", 0) if intents else 0
        enemies_data.append({
            "name": e.get("name", e.get("entity_id", "?")),
            "hp": e.get("hp", 0),
            "max_hp": e.get("max_hp", 1),
            "intent": intent_type.lower(),
            "intent_damage": intent_dmg,
            "block": e.get("block", 0),
        })

    reasoning_zh, reasoning_en = generate_reasoning(
        decision_type, action, state, legal, value, probs.tolist())

    ds = deck_score(state) if deck else 0.0
    run_next_boss = (
        run.get("next_boss_name")
        or run.get("boss_name")
        or run.get("next_boss")
        or run.get("boss")
    )
    boss_token = extract_next_boss_token(state)
    readiness_score = (
        reward_shaping_data.get("boss_readiness_score")
        if isinstance(reward_shaping_data, dict) and reward_shaping_data.get("boss_readiness_score") is not None
        else boss_readiness_score(state)
    )
    chosen_label = _clean_action_label(action, state)
    chosen_prob = float(probs[action_idx]) if action_idx < len(probs) else 0.0
    details = [
        f"candidates={len(legal)}",
        f"chosen_prob={chosen_prob:.2f}",
        f"value={value:.2f}",
        f"deck_score={ds:.2f}",
    ]

    result = {
        "msg_type": "decision",
        "title": "AI 决策透视 / AI Decision",
        "type": decision_type,
        "state_type": state.get("state_type", decision_type),
        "step": step,
        "floor": run.get("floor", 0),
        "act": run.get("act", 1),
        "screen": state.get("state_type", "?"),
        "action_source": action_source,
        "player": {
            "hp": hp, "max_hp": max_hp,
            "energy": player.get("energy", 0),
            "block": player.get("block", 0),
            "gold": player.get("gold", 0),
            "deck_size": len(deck),
            "deck_score": round(ds, 2),
            "potions": potion_names,
        },
        "enemies": enemies_data,
        "options": options[:12],
        "chosen": {
            "label": chosen_label,
            "index": int(action_idx),
        },
        "chosen_action": chosen_label,
        "value": round(value, 3),
        "reasoning_zh": reasoning_zh,
        "reasoning_en": reasoning_en,
        "reason": reasoning_zh or reasoning_en,
        "next_boss": run_next_boss or boss_token,
        "next_boss_name": run_next_boss,
        "next_boss_archetype": boss_token,
        "boss_readiness": round(float(readiness_score), 3),
        "details": details,
    }

    if nn_internals:
        result["nn_internals"] = nn_internals
    if reward_shaping_data:
        result["reward_shaping"] = reward_shaping_data
    if combat_state:
        result["combat_state"] = combat_state

    return result


def compute_reward_shaping_data(state: dict) -> dict:
    """Compute reward shaping components for visualization."""
    try:
        pv = compute_problem_vector(state)
        return {
            "problem_vector": {
                label: round(float(pv[i]), 3)
                for i, label in enumerate(PROBLEM_VECTOR_LABELS)
            },
            "problem_score": round(problem_score(state), 3),
            "survival_margin": round(survival_margin(state), 3),
            "economy_score": round(economy_score(state), 3),
            "potential": round(potential(state), 3),
            "boss_readiness_score": round(boss_readiness_score(state), 3),
        }
    except Exception as e:
        logger.debug("Reward shaping computation failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Demo Player with Overlay")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Hybrid checkpoint path. Defaults to the current documented demo champion.")
    parser.add_argument("--combat-checkpoint", type=str, default=None,
                        help="Optional combat override checkpoint. Defaults to the current documented demo champion override when available.")
    parser.add_argument("--port", type=int, default=15527, help="Godot MCP/pipe port")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Optional MCP HTTP base URL. Defaults to http://127.0.0.1:<port>")
    parser.add_argument("--transport", type=str, default="http",
                        choices=("http", "pipe", "pipe-binary"),
                        help="Runtime transport. Use http for visible front-UI recording, pipe/pipe-binary for simulator.")
    parser.add_argument("--overlay-port", type=int, default=8765, help="Overlay HTTP port")
    parser.add_argument("--decision-overlay-file", type=str, default=None,
                        help="Optional JSON file for the in-game AI overlay HUD.")
    parser.add_argument("--character-id", type=str, default="IRONCLAD")
    parser.add_argument("--seed", type=str, default=None,
                        help="Optional fixed seed for reproducible visible demo runs.")
    parser.add_argument("--step-delay", type=float, default=1.0,
                        help="Seconds to pause between steps (for human viewing)")
    parser.add_argument("--combat-delay", type=float, default=0.45,
                        help="Seconds between combat steps")
    parser.add_argument("--greedy", action="store_true",
                        help="Use argmax instead of sampling")
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to play (default: 1 for recording-friendly demo mode)")
    parser.add_argument("--metrics-file", type=str, default=None,
                        help="Path to metrics.jsonl for training dashboard")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Optional output directory for demo summary and decision trace artifacts.")
    args = parser.parse_args()

    vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    checkpoint_path = _resolve_required_checkpoint(
        args.checkpoint,
        env_var="STS2_DEMO_HYBRID_CHECKPOINT",
        default_path=DEFAULT_HYBRID_CHECKPOINT,
        label="Hybrid demo",
    )
    combat_checkpoint_path = _resolve_optional_checkpoint(
        args.combat_checkpoint,
        env_var="STS2_DEMO_COMBAT_CHECKPOINT",
        default_path=DEFAULT_COMBAT_CHECKPOINT,
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    artifact_writer = DemoArtifactWriter(str(output_dir) if output_dir else None)
    logger.info("Hybrid checkpoint: %s", checkpoint_path)
    if combat_checkpoint_path is not None:
        logger.info("Combat override checkpoint: %s", combat_checkpoint_path)
    else:
        logger.info("Combat override checkpoint: not found, using combat weights embedded in the hybrid checkpoint.")
    if output_dir is not None:
        logger.info("Demo artifacts: %s", output_dir)
    logger.info("Demo seed: %s", args.seed or "<random>")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ppo_net = FullRunPolicyNetworkV2(vocab=vocab, embed_dim=args.embed_dim)
    combat_net = CombatPolicyValueNetwork(vocab=vocab, embed_dim=args.embed_dim,
                                           hidden_dim=args.hidden_dim)
    if "ppo_model" in ckpt:
        _safe_load_state_dict(ppo_net, ckpt["ppo_model"], "PPO")
        if "mcts_model" in ckpt:
            _safe_load_state_dict(combat_net, ckpt["mcts_model"], "combat")
        logger.info("Loaded hybrid checkpoint")
    else:
        raise ValueError("--checkpoint must be a hybrid checkpoint containing ppo_model")

    if combat_checkpoint_path:
        combat_ckpt = torch.load(combat_checkpoint_path, map_location="cpu", weights_only=False)
        if "mcts_model" in combat_ckpt:
            _safe_load_state_dict(combat_net, combat_ckpt["mcts_model"], "combat_override")
        elif "model_state_dict" in combat_ckpt:
            _safe_load_state_dict(combat_net, combat_ckpt["model_state_dict"], "combat_override")
        else:
            raise ValueError("--combat-checkpoint must contain mcts_model or model_state_dict")
        logger.info("Loaded combat override from %s", combat_checkpoint_path)
    ppo_net.to(device).eval()
    combat_net.to(device).eval()

    # Register NN hooks for internals extraction
    collector = NNInternalsCollector(combat_net, ppo_net)
    logger.info("NN internals hooks registered")

    # Start overlay
    broadcaster = DecisionBroadcaster(port=args.overlay_port)
    broadcaster.set_cmd_file(args.decision_overlay_file)
    broadcaster.start()
    overlay_writer = DecisionOverlayFileWriter(args.decision_overlay_file)
    overlay_writer.clear()
    logger.info("Recording overlay URL: http://localhost:%d/?mode=recording", args.overlay_port)
    logger.info("Dashboard overlay URL: http://localhost:%d/", args.overlay_port)
    if args.decision_overlay_file:
        logger.info("Godot HUD payload file: %s", Path(args.decision_overlay_file).expanduser().resolve())

    def publish(payload: dict[str, Any]) -> None:
        broadcaster.broadcast(payload)
        overlay_writer.write(payload)
        artifact_writer.write_decision(payload)

    # Start training metrics monitor if requested
    if args.metrics_file:
        metrics_monitor = TrainingMetricsMonitor(
            args.metrics_file, broadcaster.broadcast)
        metrics_monitor.start()

    # Connect to Godot
    base_url = args.base_url or f"http://127.0.0.1:{args.port}"
    use_pipe = args.transport in {"pipe", "pipe-binary"}
    client = create_full_run_client(
        base_url=base_url,
        port=args.port,
        use_pipe=use_pipe,
        transport=args.transport,
        prefer_v2=True,
    )
    logger.info(
        "Connected to Godot via %s (%s)",
        getattr(client, "transport_name", args.transport),
        base_url if not use_pipe else f"port {args.port}",
    )

    episode = 0
    while not _shutdown:
        if args.episodes > 0 and episode >= args.episodes:
            break
        episode += 1

        state = client.reset(character_id=args.character_id, seed=args.seed)
        logger.info("=== Episode %d started ===", episode)
        step = 0
        last_state_key = ""
        repeat_count = 0
        last_floor = -1
        same_floor_steps = 0
        step_i = 0
        idle_polls = 0
        last_action_name = ""

        while step_i < 600:
            if _shutdown:
                break

            # Playback control
            broadcaster.wait_if_paused()
            if _shutdown:
                break

            st = (state.get("state_type") or "").lower()
            is_combat_context = _is_combat_context(state)
            if st == "game_over" or state.get("terminal"):
                go = state.get("game_over") or {}
                outcome = go.get("run_outcome", go.get("outcome", "?"))
                floor = (state.get("run") or {}).get("floor", 0)
                logger.info("Episode %d: %s at floor %d", episode, outcome, floor)
                publish({
                    "msg_type": "decision",
                    "type": "game_over",
                    "step": step_i,
                    "floor": floor,
                    "outcome": outcome,
                    "reasoning_zh": f"游戏结束: {outcome}",
                    "reasoning_en": f"Game Over: {outcome}",
                })
                time.sleep(3)
                break

            legal = [a for a in state.get("legal_actions", [])
                     if isinstance(a, dict) and a.get("is_enabled") is not False]
            transport_name = str(getattr(client, "transport_name", "") or "").strip().lower()
            use_v1_compat = transport_name == "http-v1-singleplayer"
            if not legal and not use_pipe:
                try:
                    if is_combat_context:
                        legal = [
                            action for action in _combat_candidate_actions(
                                adapt_v1_state_for_combat_policy(state) if use_v1_compat else state
                            )
                            if isinstance(action, dict)
                        ]
                    else:
                        legal = [
                            action for action in _non_combat_candidate_actions(state)
                            if isinstance(action, dict)
                        ]
                except Exception as exc:
                    logger.debug("Failed to derive front-UI candidate actions: %s", exc)
            if st == "card_select":
                legal = _normalize_card_select_legal_actions(state, legal)
            floor = int(((state.get("run") or {}).get("floor", 0) or 0))
            if floor == last_floor:
                same_floor_steps += 1
            else:
                last_floor = floor
                same_floor_steps = 0

            if not legal:
                idle_polls += 1
                if idle_polls >= 400:
                    logger.warning("Demo aborting idle transition stall at floor %s state=%s", floor, st)
                    publish({
                        "msg_type": "decision",
                        "type": "game_over",
                        "step": step_i,
                        "floor": floor,
                        "outcome": "stuck_abort",
                        "reasoning_zh": "界面过场持续过久，演示提前结束",
                        "reasoning_en": "UI transition took too long, aborting demo early",
                    })
                    break
                time.sleep(0.10 if not use_pipe else 0.02)
                state = client.get_state()
                continue
            idle_polls = 0

            state_key = _state_signature(state, legal)
            if state_key == last_state_key:
                repeat_count += 1
            else:
                last_state_key = state_key
                repeat_count = 0

            if repeat_count >= 25:
                logger.warning("Demo aborting repeated state loop at step %d floor %s", step_i, floor)
                publish({
                    "msg_type": "decision",
                    "type": "game_over",
                    "step": step_i,
                    "floor": floor,
                    "outcome": "stuck_abort",
                    "reasoning_zh": "检测到重复状态循环，演示提前结束",
                    "reasoning_en": "Detected repeated state loop, aborting demo early",
                })
                break

            if is_combat_context and same_floor_steps >= 140:
                logger.warning("Demo aborting combat stall at step %d floor %s", step_i, floor)
                publish({
                    "msg_type": "decision",
                    "type": "game_over",
                    "step": step_i,
                    "floor": floor,
                    "outcome": "stuck_abort",
                    "reasoning_zh": "该层战斗持续过久，演示提前结束",
                    "reasoning_en": "Combat on this floor ran too long, aborting demo early",
                })
                break

            # Compute reward shaping data
            rs_data = compute_reward_shaping_data(state)
            forced_action: dict[str, Any] | None = None
            forced_source: str | None = None
            if st == "combat_rewards" and last_action_name in {"select_card_reward", "skip_card_reward"}:
                proceed_action = next(
                    (item for item in legal if str(item.get("action") or "").lower() == "proceed"),
                    None,
                )
                if proceed_action is not None:
                    forced_action = proceed_action
                    forced_source = "post_card_reward_proceed"

            # --- Inference ---
            if is_combat_context:
                decision_type = "combat"
                action_source = "combat_net"
                sf = build_combat_features(state, vocab)
                af = build_combat_action_features(state, legal, vocab)
                sf_t = {k: (torch.tensor(v).unsqueeze(0).long() if v.dtype in (np.int64, np.int32)
                            else (torch.tensor(v).unsqueeze(0).bool() if v.dtype == bool
                                  else torch.tensor(v).unsqueeze(0).float())).to(device)
                        for k, v in sf.items()}
                af_t = {k: (torch.tensor(v).unsqueeze(0).long() if v.dtype in (np.int64, np.int32)
                            else (torch.tensor(v).unsqueeze(0).bool() if v.dtype == bool
                                  else torch.tensor(v).unsqueeze(0).float())).to(device)
                        for k, v in af.items()}
                with torch.no_grad():
                    logits, value_t = combat_net(sf_t, af_t)
                mask = af_t["action_mask"].float()
                logits_m = logits + (1 - mask) * (-1e9)
                probs_t = torch.softmax(logits_m.squeeze(0), dim=-1)
                probs = probs_t.cpu().numpy()
                value = value_t.squeeze(0).cpu().item()

                if args.greedy:
                    action_idx = probs[:len(legal)].argmax()
                else:
                    dist = torch.distributions.Categorical(probs=probs_t)
                    action_idx = dist.sample().item()

                reranked_idx, rerank_reason = _combat_tactical_rerank(
                    state,
                    legal,
                    probs,
                    int(action_idx),
                )
                if reranked_idx != int(action_idx):
                    action_idx = reranked_idx
                    action_source = f"combat_net+tactical_guard:{rerank_reason or 'rerank'}"

                # Hand/enemy names for attention labels
                battle = state.get("battle") or {}
                hand = battle.get("hand") or []
                hand_names = [c.get("name", c.get("id", "?")) for c in hand if isinstance(c, dict)]
                enemy_names = [e.get("name", e.get("entity_id", "?"))
                               for e in (battle.get("enemies") or []) if isinstance(e, dict)]

                raw_internals = collector.get_and_clear()
                nn_internals = format_internals_for_broadcast(
                    raw_internals, hand_names=hand_names, enemy_names=enemy_names)
                combat_state_data = extract_combat_state(state)
                delay = args.combat_delay

            else:
                if "card_reward" in st:
                    decision_type = "card_reward"
                elif st == "map":
                    decision_type = "map"
                elif st == "rest_site":
                    decision_type = "rest"
                elif st == "shop":
                    decision_type = "shop"
                elif st == "event":
                    decision_type = "event"
                else:
                    decision_type = st
                action_source = "ppo_net"

                ss = build_structured_state(state, vocab)
                sa = build_structured_actions(state, legal, vocab)
                sf_np = _structured_state_to_numpy_dict(ss)
                af_np = _structured_actions_to_numpy_dict(sa)

                st_t = {k: (torch.tensor(v).unsqueeze(0).long()
                            if ("ids" in k or "idx" in k or "types" in k or "count" in k)
                            else (torch.tensor(v).unsqueeze(0).bool() if "mask" in k
                                  else (torch.tensor(v).unsqueeze(0).float()
                                        if isinstance(v, np.ndarray)
                                        else torch.tensor([v]).float()))).to(device)
                        for k, v in sf_np.items()}
                at_t = {k: (torch.tensor(v).unsqueeze(0).long()
                            if ("ids" in k or "types" in k or "indices" in k)
                            else (torch.tensor(v).unsqueeze(0).bool() if "mask" in k
                                  else (torch.tensor(v).unsqueeze(0).float()
                                        if isinstance(v, np.ndarray)
                                        else torch.tensor([v]).float()))).to(device)
                        for k, v in af_np.items()}

                with torch.no_grad():
                    act_idx_t, _, _, value_t = ppo_net.get_action_and_value(st_t, at_t)
                action_idx = act_idx_t.item()
                value = value_t.item()

                # Full forward for aux predictions
                with torch.no_grad():
                    logits_raw, _, deck_quality_t, boss_readiness_t, action_adv_t = ppo_net.forward(st_t, at_t)
                mask_np = af_np.get("action_mask", np.ones(len(legal), dtype=bool))
                mask_t = torch.tensor(mask_np).unsqueeze(0).float().to(device)
                logits_m = logits_raw + (1 - mask_t) * (-1e9)
                probs = torch.softmax(logits_m.squeeze(0), dim=-1).cpu().numpy()

                if args.greedy:
                    action_idx = probs[:len(legal)].argmax()

                raw_internals = collector.get_and_clear()
                nn_internals = format_internals_for_broadcast(
                    raw_internals,
                    deck_quality=deck_quality_t.squeeze(0).cpu().item(),
                    boss_readiness=boss_readiness_t.squeeze(0).cpu().item(),
                    action_advantages=action_adv_t.squeeze(0).cpu().numpy()[:len(legal)],
                )
                combat_state_data = None
                delay = args.step_delay

            # Select action
            if action_idx < len(legal):
                action = legal[action_idx]
            else:
                action = legal[0]
                action_idx = 0

            if forced_action is not None:
                action = forced_action
                action_source = forced_source or action_source
                try:
                    action_idx = legal.index(action)
                except ValueError:
                    action_idx = 0
                probs = np.zeros(max(len(legal), 1), dtype=np.float32)
                if 0 <= action_idx < len(probs):
                    probs[action_idx] = 1.0

            if repeat_count >= 3:
                escape_action = _choose_repeat_escape_action(state, legal)
                if escape_action is not None:
                    try:
                        action_idx = legal.index(escape_action)
                    except ValueError:
                        action_idx = 0
                    action = escape_action
                    action_source = "loop_guard"
                    logger.warning(
                        "Demo loop guard forced action at step %d repeat=%d: %s",
                        step_i,
                        repeat_count,
                        action.get("action", "?"),
                    )

            # Build and broadcast
            decision = build_decision(
                decision_type, state, legal, action, action_idx, probs, value, step_i,
                action_source,
                nn_internals=nn_internals,
                reward_shaping_data=rs_data,
                combat_state=combat_state_data,
            )
            publish(decision)

            # Delay: let the player read the decision overlay before executing
            delay = args.combat_delay if decision_type == "combat" else args.step_delay
            broadcaster.wait_if_paused()
            if delay > 0:
                time.sleep(delay / max(0.1, broadcaster.speed_multiplier))

            # Log / execute payload
            clean = {
                k: v for k, v in action.items()
                if k in ("action", "index", "card_index", "target", "target_id", "slot", "col", "row", "value")
            }
            clean = _normalize_action_payload_for_transport(state, clean, use_pipe=use_pipe)
            label = _clean_action_label(action, state)
            prob = float(probs[action_idx]) if action_idx < len(probs) else 0
            logger.info(
                "Step %3d | %s | %s [%s] (%.0f%%) v=%.2f payload=%s",
                step_i,
                decision_type,
                label,
                clean.get("action", "?") if isinstance(action, dict) else "?",
                prob * 100,
                value,
                json.dumps(clean, ensure_ascii=False, sort_keys=True),
            )

            # Execute
            try:
                state = client.act(clean)
                last_action_name = str(clean.get("action") or "")
            except Exception as exc:
                if not use_pipe and _should_retry_visible_action(st, exc):
                    logger.info(
                        "Visible action retry at step %d state=%s payload=%s error=%s",
                        step_i,
                        st,
                        json.dumps(clean, ensure_ascii=False, sort_keys=True),
                        exc,
                    )
                    time.sleep(0.18)
                    try:
                        state = client.act(clean)
                        last_action_name = str(clean.get("action") or "")
                    except Exception as retry_exc:
                        logger.warning(
                            "Action dispatch failed at step %d state=%s payload=%s error=%s",
                            step_i,
                            st,
                            json.dumps(clean, ensure_ascii=False, sort_keys=True),
                            retry_exc,
                        )
                        time.sleep(0.10)
                        state = client.get_state()
                else:
                    logger.warning(
                        "Action dispatch failed at step %d state=%s payload=%s error=%s",
                        step_i,
                        st,
                        json.dumps(clean, ensure_ascii=False, sort_keys=True),
                        exc,
                    )
                    if use_pipe:
                        try:
                            state = client.act({"action": "end_turn"})
                        except Exception:
                            try:
                                state = client.act({"action": "wait"})
                            except Exception:
                                state = client.get_state()
                    else:
                        # In front-UI demo mode, silently converting failed actions
                        # into end_turn makes the run look obviously wrong on screen.
                        # Refresh state instead so we can inspect the real failure.
                        time.sleep(0.10)
                        state = client.get_state()

            step_i += 1
            step = step_i

    collector.cleanup()
    client.close()
    logger.info("Demo finished. %d episodes played.", episode)

def main():
    parser = argparse.ArgumentParser(description="AI Demo Player with Overlay")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Hybrid checkpoint path. Defaults to the current documented demo champion.")
    parser.add_argument("--combat-checkpoint", type=str, default=None,
                        help="Optional combat override checkpoint. Defaults to the current documented demo champion override when available.")
    parser.add_argument("--port", type=int, default=15527, help="Godot MCP/pipe port")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Optional MCP HTTP base URL. Defaults to http://127.0.0.1:<port>")
    parser.add_argument("--transport", type=str, default="http",
                        choices=("http", "pipe", "pipe-binary"),
                        help="Runtime transport. Use http for visible front-UI recording, pipe/pipe-binary for simulator.")
    parser.add_argument("--overlay-port", type=int, default=8765, help="Overlay HTTP port")
    parser.add_argument("--decision-overlay-file", type=str, default=None,
                        help="Optional JSON file for the in-game AI overlay HUD.")
    parser.add_argument("--character-id", type=str, default="IRONCLAD")
    parser.add_argument("--seed", type=str, default=None,
                        help="Optional fixed seed for reproducible visible demo runs.")
    parser.add_argument("--step-delay", type=float, default=1.0,
                        help="Seconds to pause between visible steps.")
    parser.add_argument("--combat-delay", type=float, default=0.45,
                        help="Seconds between combat decisions.")
    parser.add_argument("--greedy", action="store_true", help="Use argmax instead of sampling.")
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to play (default: 1 for recording-friendly demo mode).")
    parser.add_argument("--metrics-file", type=str, default=None,
                        help="Path to metrics.jsonl for the optional training panel.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Optional directory for demo summary and decision trace artifacts.")
    args = parser.parse_args()

    vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    checkpoint_path = _resolve_required_checkpoint(
        args.checkpoint,
        env_var="STS2_DEMO_HYBRID_CHECKPOINT",
        default_path=DEFAULT_HYBRID_CHECKPOINT,
        label="Hybrid demo",
    )
    combat_checkpoint_path = _resolve_optional_checkpoint(
        args.combat_checkpoint,
        env_var="STS2_DEMO_COMBAT_CHECKPOINT",
        default_path=DEFAULT_COMBAT_CHECKPOINT,
    )

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    artifact_writer = DemoArtifactWriter(str(output_dir) if output_dir else None)
    logger.info("Hybrid checkpoint: %s", checkpoint_path)
    if combat_checkpoint_path is not None:
        logger.info("Combat override checkpoint: %s", combat_checkpoint_path)
    else:
        logger.info("Combat override checkpoint: not found, using combat weights embedded in the hybrid checkpoint.")
    if output_dir is not None:
        logger.info("Demo artifacts: %s", output_dir)
    logger.info("Demo seed: %s", args.seed or "<random>")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ppo_net = FullRunPolicyNetworkV2(vocab=vocab, embed_dim=args.embed_dim)
    combat_net = CombatPolicyValueNetwork(vocab=vocab, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    if "ppo_model" in ckpt:
        _safe_load_state_dict(ppo_net, ckpt["ppo_model"], "PPO")
        if "mcts_model" in ckpt:
            _safe_load_state_dict(combat_net, ckpt["mcts_model"], "combat")
        logger.info("Loaded hybrid checkpoint")
    else:
        raise ValueError("--checkpoint must be a hybrid checkpoint containing ppo_model")

    if combat_checkpoint_path:
        combat_ckpt = torch.load(combat_checkpoint_path, map_location="cpu", weights_only=False)
        if "mcts_model" in combat_ckpt:
            _safe_load_state_dict(combat_net, combat_ckpt["mcts_model"], "combat_override")
        elif "model_state_dict" in combat_ckpt:
            _safe_load_state_dict(combat_net, combat_ckpt["model_state_dict"], "combat_override")
        else:
            raise ValueError("--combat-checkpoint must contain mcts_model or model_state_dict")
        logger.info("Loaded combat override from %s", combat_checkpoint_path)

    ppo_net.to(device).eval()
    combat_net.to(device).eval()

    collector = NNInternalsCollector(combat_net, ppo_net)
    logger.info("NN internals hooks registered")

    broadcaster = DecisionBroadcaster(port=args.overlay_port)
    broadcaster.set_cmd_file(args.decision_overlay_file)
    broadcaster.start()
    overlay_writer = DecisionOverlayFileWriter(args.decision_overlay_file)
    overlay_writer.clear()
    logger.info("Recording overlay URL: http://localhost:%d/?mode=recording", args.overlay_port)
    logger.info("Dashboard overlay URL: http://localhost:%d/", args.overlay_port)
    if args.decision_overlay_file:
        logger.info("Godot HUD payload file: %s", Path(args.decision_overlay_file).expanduser().resolve())

    def publish(payload: dict[str, Any]) -> None:
        broadcaster.broadcast(payload)
        overlay_writer.write(payload)
        artifact_writer.write_decision(payload)

    if args.metrics_file:
        metrics_monitor = TrainingMetricsMonitor(args.metrics_file, broadcaster.broadcast)
        metrics_monitor.start()

    base_url = args.base_url or f"http://127.0.0.1:{args.port}"
    use_pipe = args.transport in {"pipe", "pipe-binary"}
    client = create_full_run_client(
        base_url=base_url,
        port=args.port,
        use_pipe=use_pipe,
        transport=args.transport,
        prefer_v2=True,
    )
    logger.info(
        "Connected to Godot via %s (%s)",
        getattr(client, "transport_name", args.transport),
        base_url if not use_pipe else f"port {args.port}",
    )

    artifact_writer.finalize(
        started_at=_utc_now_iso(),
        base_url=base_url,
        transport=getattr(client, "transport_name", args.transport),
        character_id=args.character_id,
        seed=args.seed,
        checkpoint=str(checkpoint_path),
        combat_checkpoint=str(combat_checkpoint_path) if combat_checkpoint_path else None,
        recording_overlay_url=f"http://localhost:{args.overlay_port}/?mode=recording",
        dashboard_overlay_url=f"http://localhost:{args.overlay_port}/",
        overlay_file=str(Path(args.decision_overlay_file).expanduser().resolve()) if args.decision_overlay_file else None,
    )

    def finalize_episode(summary: dict[str, Any], *, pause_seconds: float = 0.0) -> None:
        artifact_writer.record_episode(summary)
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    episode = 0
    completed_episodes = 0
    try:
        while not _shutdown:
            if args.episodes > 0 and episode >= args.episodes:
                break
            episode += 1

            state = client.reset(character_id=args.character_id, seed=args.seed)
            logger.info("=== Episode %d started ===", episode)
            last_state_key = ""
            repeat_count = 0
            last_floor = -1
            same_floor_steps = 0
            step_i = 0
            idle_polls = 0
            last_action_name = ""
            episode_finished = False
            episode_summary: dict[str, Any] = {
                "episode": episode,
                "started_at": _utc_now_iso(),
                "seed": args.seed,
                "steps": 0,
                "outcome": None,
                "end_reason": None,
                "last_state_type": None,
                "last_floor": 0,
                "boss_reached": False,
                "loop_guard_count": 0,
                "auto_progress_count": 0,
                "action_failures": 0,
            }

            while step_i < 600:
                if _shutdown:
                    episode_summary["end_reason"] = "interrupted"
                    episode_summary["outcome"] = "interrupted"
                    break

                broadcaster.wait_if_paused()
                if _shutdown:
                    episode_summary["end_reason"] = "interrupted"
                    episode_summary["outcome"] = "interrupted"
                    break

                st = str(state.get("state_type") or "").lower()
                run = state.get("run") if isinstance(state.get("run"), dict) else {}
                room_type = str(run.get("room_type") or "").lower()
                floor = int(run.get("floor", 0) or 0)
                episode_summary["last_state_type"] = st
                episode_summary["last_floor"] = floor
                if st == "boss" or room_type == "boss":
                    episode_summary["boss_reached"] = True

                is_combat_context = _is_combat_context(state)
                if st == "game_over" or state.get("terminal"):
                    go = state.get("game_over") or {}
                    outcome = go.get("run_outcome", go.get("outcome", "?"))
                    episode_summary["steps"] = step_i
                    episode_summary["outcome"] = outcome
                    episode_summary["end_reason"] = "game_over"
                    logger.info("Episode %d: %s at floor %d", episode, outcome, floor)
                    publish({
                        "msg_type": "decision",
                        "type": "game_over",
                        "state_type": "game_over",
                        "step": step_i,
                        "floor": floor,
                        "act": run.get("act", 1),
                        "outcome": outcome,
                        "end_reason": "game_over",
                        "boss_reached": episode_summary["boss_reached"],
                        "reasoning_zh": f"游戏结束：{outcome}",
                        "reasoning_en": f"Game over: {outcome}",
                    })
                    finalize_episode(episode_summary, pause_seconds=3.0)
                    completed_episodes += 1
                    episode_finished = True
                    break

                if _visible_combat_needs_wait(state, use_pipe=use_pipe):
                    idle_polls += 1
                    if idle_polls >= 400:
                        logger.warning("Demo aborting combat wait stall at floor %s state=%s", floor, st)
                        episode_summary["steps"] = step_i
                        episode_summary["outcome"] = "stuck_abort"
                        episode_summary["end_reason"] = "combat_wait_timeout"
                        finalize_episode(episode_summary)
                        completed_episodes += 1
                        episode_finished = True
                        break
                    time.sleep(0.10)
                    state = client.get_state()
                    continue

                legal = [
                    action for action in state.get("legal_actions", [])
                    if isinstance(action, dict) and action.get("is_enabled") is not False
                ]
                transport_name = str(getattr(client, "transport_name", "") or "").strip().lower()
                use_v1_compat = transport_name == "http-v1-singleplayer"
                if not legal and not use_pipe:
                    try:
                        if is_combat_context:
                            legal = [
                                action for action in _combat_candidate_actions(
                                    adapt_v1_state_for_combat_policy(state) if use_v1_compat else state
                                )
                                if isinstance(action, dict)
                            ]
                        else:
                            legal = [
                                action for action in _non_combat_candidate_actions(state)
                                if isinstance(action, dict)
                            ]
                    except Exception as exc:
                        logger.debug("Failed to derive front-UI candidate actions: %s", exc)
                if st == "card_select":
                    legal = _normalize_card_select_legal_actions(state, legal)

                if floor == last_floor:
                    same_floor_steps += 1
                else:
                    last_floor = floor
                    same_floor_steps = 0

                if not legal:
                    idle_polls += 1
                    if idle_polls >= 400:
                        logger.warning("Demo aborting idle transition stall at floor %s state=%s", floor, st)
                        episode_summary["steps"] = step_i
                        episode_summary["outcome"] = "stuck_abort"
                        episode_summary["end_reason"] = "ui_transition_timeout"
                        publish({
                            "msg_type": "decision",
                            "type": "game_over",
                            "state_type": st,
                            "step": step_i,
                            "floor": floor,
                            "act": run.get("act", 1),
                            "outcome": "stuck_abort",
                            "end_reason": "ui_transition_timeout",
                            "boss_reached": episode_summary["boss_reached"],
                            "reasoning_zh": "界面过场持续过久，演示提前结束。",
                            "reasoning_en": "A UI transition took too long, so the demo ended early.",
                        })
                        finalize_episode(episode_summary)
                        completed_episodes += 1
                        episode_finished = True
                        break
                    time.sleep(0.10 if not use_pipe else 0.02)
                    state = client.get_state()
                    continue
                idle_polls = 0

                state_key = _state_signature(state, legal)
                if state_key == last_state_key:
                    repeat_count += 1
                else:
                    last_state_key = state_key
                    repeat_count = 0

                if _should_wait_for_visible_transition(
                    state,
                    legal,
                    last_action_name,
                    repeat_count,
                    use_pipe=use_pipe,
                ):
                    time.sleep(0.18)
                    state = client.get_state()
                    continue

                if repeat_count >= 25:
                    logger.warning("Demo aborting repeated state loop at step %d floor %s", step_i, floor)
                    episode_summary["steps"] = step_i
                    episode_summary["outcome"] = "stuck_abort"
                    episode_summary["end_reason"] = "repeated_state_loop"
                    publish({
                        "msg_type": "decision",
                        "type": "game_over",
                        "state_type": st,
                        "step": step_i,
                        "floor": floor,
                        "act": run.get("act", 1),
                        "outcome": "stuck_abort",
                        "end_reason": "repeated_state_loop",
                        "boss_reached": episode_summary["boss_reached"],
                        "reasoning_zh": "检测到重复状态循环，演示提前结束。",
                        "reasoning_en": "A repeated state loop was detected, so the demo ended early.",
                    })
                    finalize_episode(episode_summary)
                    completed_episodes += 1
                    episode_finished = True
                    break

                if is_combat_context and same_floor_steps >= 140:
                    logger.warning("Demo aborting combat stall at step %d floor %s", step_i, floor)
                    episode_summary["steps"] = step_i
                    episode_summary["outcome"] = "stuck_abort"
                    episode_summary["end_reason"] = "combat_floor_timeout"
                    publish({
                        "msg_type": "decision",
                        "type": "game_over",
                        "state_type": st,
                        "step": step_i,
                        "floor": floor,
                        "act": run.get("act", 1),
                        "outcome": "stuck_abort",
                        "end_reason": "combat_floor_timeout",
                        "boss_reached": episode_summary["boss_reached"],
                        "reasoning_zh": "该层战斗持续过久，演示提前结束。",
                        "reasoning_en": "Combat on this floor ran too long, so the demo ended early.",
                    })
                    finalize_episode(episode_summary)
                    completed_episodes += 1
                    episode_finished = True
                    break

                rs_data = compute_reward_shaping_data(state)
                forced_action: dict[str, Any] | None = None
                forced_source: str | None = None
                if st == "combat_rewards" and last_action_name in {"select_card_reward", "skip_card_reward"}:
                    proceed_action = next(
                        (item for item in legal if str(item.get("action") or "").lower() == "proceed"),
                        None,
                    )
                    if proceed_action is not None:
                        forced_action = proceed_action
                        forced_source = "post_card_reward_proceed"
                        episode_summary["auto_progress_count"] = int(episode_summary["auto_progress_count"]) + 1
                if forced_action is None and is_combat_context:
                    selection_progress_action = _choose_combat_selection_progress_action(
                        state,
                        legal,
                        repeat_count=repeat_count,
                    )
                    if selection_progress_action is not None:
                        forced_action = selection_progress_action
                        forced_source = "combat_selection_progress"
                        episode_summary["auto_progress_count"] = int(episode_summary["auto_progress_count"]) + 1

                if is_combat_context:
                    decision_type = "combat"
                    action_source = "combat_net"
                    sf = build_combat_features(state, vocab)
                    af = build_combat_action_features(state, legal, vocab)
                    sf_t = {
                        key: (
                            torch.tensor(value).unsqueeze(0).long()
                            if value.dtype in (np.int64, np.int32)
                            else (
                                torch.tensor(value).unsqueeze(0).bool()
                                if value.dtype == bool
                                else torch.tensor(value).unsqueeze(0).float()
                            )
                        ).to(device)
                        for key, value in sf.items()
                    }
                    af_t = {
                        key: (
                            torch.tensor(value).unsqueeze(0).long()
                            if value.dtype in (np.int64, np.int32)
                            else (
                                torch.tensor(value).unsqueeze(0).bool()
                                if value.dtype == bool
                                else torch.tensor(value).unsqueeze(0).float()
                            )
                        ).to(device)
                        for key, value in af.items()
                    }
                    with torch.no_grad():
                        logits, value_t = combat_net(sf_t, af_t)
                    mask = af_t["action_mask"].float()
                    logits_m = logits + (1 - mask) * (-1e9)
                    probs_t = torch.softmax(logits_m.squeeze(0), dim=-1)
                    probs = probs_t.cpu().numpy()
                    value = value_t.squeeze(0).cpu().item()

                    if args.greedy:
                        action_idx = probs[:len(legal)].argmax()
                    else:
                        dist = torch.distributions.Categorical(probs=probs_t)
                        action_idx = dist.sample().item()

                    reranked_idx, rerank_reason = _combat_tactical_rerank(state, legal, probs, int(action_idx))
                    if reranked_idx != int(action_idx):
                        action_idx = reranked_idx
                        action_source = f"combat_net+tactical_guard:{rerank_reason or 'rerank'}"

                    battle = state.get("battle") or {}
                    hand = battle.get("hand") or []
                    hand_names = [card.get("name", card.get("id", "?")) for card in hand if isinstance(card, dict)]
                    enemy_names = [
                        enemy.get("name", enemy.get("entity_id", "?"))
                        for enemy in (battle.get("enemies") or [])
                        if isinstance(enemy, dict)
                    ]
                    raw_internals = collector.get_and_clear()
                    nn_internals = format_internals_for_broadcast(
                        raw_internals,
                        hand_names=hand_names,
                        enemy_names=enemy_names,
                    )
                    combat_state_data = extract_combat_state(state)
                    delay = args.combat_delay
                else:
                    if "card_reward" in st:
                        decision_type = "card_reward"
                    elif st == "map":
                        decision_type = "map"
                    elif st == "rest_site":
                        decision_type = "rest"
                    elif st == "shop":
                        decision_type = "shop"
                    elif st == "event":
                        decision_type = "event"
                    elif st == "treasure":
                        decision_type = "treasure"
                    else:
                        decision_type = st
                    action_source = "ppo_net"

                    ss = build_structured_state(state, vocab)
                    sa = build_structured_actions(state, legal, vocab)
                    sf_np = _structured_state_to_numpy_dict(ss)
                    af_np = _structured_actions_to_numpy_dict(sa)

                    st_t = {
                        key: (
                            torch.tensor(value).unsqueeze(0).long()
                            if ("ids" in key or "idx" in key or "types" in key or "count" in key)
                            else (
                                torch.tensor(value).unsqueeze(0).bool()
                                if "mask" in key
                                else (
                                    torch.tensor(value).unsqueeze(0).float()
                                    if isinstance(value, np.ndarray)
                                    else torch.tensor([value]).float()
                                )
                            )
                        ).to(device)
                        for key, value in sf_np.items()
                    }
                    at_t = {
                        key: (
                            torch.tensor(value).unsqueeze(0).long()
                            if ("ids" in key or "types" in key or "indices" in key)
                            else (
                                torch.tensor(value).unsqueeze(0).bool()
                                if "mask" in key
                                else (
                                    torch.tensor(value).unsqueeze(0).float()
                                    if isinstance(value, np.ndarray)
                                    else torch.tensor([value]).float()
                                )
                            )
                        ).to(device)
                        for key, value in af_np.items()
                    }

                    with torch.no_grad():
                        act_idx_t, _, _, value_t = ppo_net.get_action_and_value(st_t, at_t)
                    action_idx = act_idx_t.item()
                    value = value_t.item()

                    with torch.no_grad():
                        logits_raw, _, deck_quality_t, boss_readiness_t, action_adv_t = ppo_net.forward(st_t, at_t)
                    mask_np = af_np.get("action_mask", np.ones(len(legal), dtype=bool))
                    mask_t = torch.tensor(mask_np).unsqueeze(0).float().to(device)
                    logits_m = logits_raw + (1 - mask_t) * (-1e9)
                    probs = torch.softmax(logits_m.squeeze(0), dim=-1).cpu().numpy()
                    if args.greedy:
                        action_idx = probs[:len(legal)].argmax()

                    raw_internals = collector.get_and_clear()
                    nn_internals = format_internals_for_broadcast(
                        raw_internals,
                        deck_quality=deck_quality_t.squeeze(0).cpu().item(),
                        boss_readiness=boss_readiness_t.squeeze(0).cpu().item(),
                        action_advantages=action_adv_t.squeeze(0).cpu().numpy()[:len(legal)],
                    )
                    combat_state_data = None
                    delay = args.step_delay

                action = legal[action_idx] if action_idx < len(legal) else legal[0]
                if action_idx >= len(legal):
                    action_idx = 0

                if forced_action is not None:
                    action = forced_action
                    action_source = forced_source or action_source
                    try:
                        action_idx = legal.index(action)
                    except ValueError:
                        action_idx = 0
                    probs = np.zeros(max(len(legal), 1), dtype=np.float32)
                    if 0 <= action_idx < len(probs):
                        probs[action_idx] = 1.0

                if repeat_count >= 3:
                    escape_action = _choose_repeat_escape_action(state, legal, last_action_name)
                    if escape_action is not None:
                        try:
                            action_idx = legal.index(escape_action)
                        except ValueError:
                            action_idx = 0
                        action = escape_action
                        action_source = "loop_guard"
                        episode_summary["loop_guard_count"] = int(episode_summary["loop_guard_count"]) + 1
                        logger.warning(
                            "Demo loop guard forced action at step %d repeat=%d: %s",
                            step_i,
                            repeat_count,
                            action.get("action", "?"),
                        )

                decision = build_decision(
                    decision_type,
                    state,
                    legal,
                    action,
                    action_idx,
                    probs,
                    value,
                    step_i,
                    action_source,
                    nn_internals=nn_internals,
                    reward_shaping_data=rs_data,
                    combat_state=combat_state_data,
                )
                publish(decision)

                # Show decision on overlay, then wait before executing
                delay = args.combat_delay if decision_type == "combat" else args.step_delay
                broadcaster.wait_if_paused()
                if delay > 0:
                    time.sleep(delay / max(0.1, broadcaster.speed_multiplier))

                clean = {
                    key: value
                    for key, value in action.items()
                    if key in ("action", "index", "card_index", "target", "target_id", "slot", "col", "row", "value")
                }
                clean = _normalize_action_payload_for_transport(state, clean, use_pipe=use_pipe)
                label = _clean_action_label(action, state)
                prob = float(probs[action_idx]) if action_idx < len(probs) else 0
                logger.info(
                    "Step %3d | %s | %s [%s] (%.0f%%) v=%.2f payload=%s",
                    step_i,
                    decision_type,
                    label,
                    clean.get("action", "?") if isinstance(action, dict) else "?",
                    prob * 100,
                    value,
                    json.dumps(clean, ensure_ascii=False, sort_keys=True),
                )

                try:
                    state = client.act(clean)
                    last_action_name = str(clean.get("action") or "")
                except Exception as exc:
                    if not use_pipe and _should_retry_visible_action(st, exc):
                        logger.info(
                            "Visible action retry at step %d state=%s payload=%s error=%s",
                            step_i,
                            st,
                            json.dumps(clean, ensure_ascii=False, sort_keys=True),
                            exc,
                        )
                        time.sleep(0.18)
                        try:
                            state = client.act(clean)
                            last_action_name = str(clean.get("action") or "")
                        except Exception as retry_exc:
                            if _is_benign_visible_dispatch_failure(
                                st,
                                str(clean.get("action") or ""),
                                retry_exc,
                                use_pipe=use_pipe,
                            ):
                                logger.info(
                                    "Visible timing mismatch tolerated at step %d state=%s payload=%s error=%s",
                                    step_i,
                                    st,
                                    json.dumps(clean, ensure_ascii=False, sort_keys=True),
                                    retry_exc,
                                )
                                time.sleep(0.10)
                                state = client.get_state()
                            else:
                                episode_summary["action_failures"] = int(episode_summary["action_failures"]) + 1
                                logger.warning(
                                    "Action dispatch failed at step %d state=%s payload=%s error=%s",
                                    step_i,
                                    st,
                                    json.dumps(clean, ensure_ascii=False, sort_keys=True),
                                    retry_exc,
                                )
                                time.sleep(0.10)
                                state = client.get_state()
                    else:
                        if _is_benign_visible_dispatch_failure(
                            st,
                            str(clean.get("action") or ""),
                            exc,
                            use_pipe=use_pipe,
                        ):
                            logger.info(
                                "Visible timing mismatch tolerated at step %d state=%s payload=%s error=%s",
                                step_i,
                                st,
                                json.dumps(clean, ensure_ascii=False, sort_keys=True),
                                exc,
                            )
                            time.sleep(0.10)
                            state = client.get_state()
                        else:
                            episode_summary["action_failures"] = int(episode_summary["action_failures"]) + 1
                            logger.warning(
                                "Action dispatch failed at step %d state=%s payload=%s error=%s",
                                step_i,
                                st,
                                json.dumps(clean, ensure_ascii=False, sort_keys=True),
                                exc,
                            )
                            if use_pipe:
                                try:
                                    state = client.act({"action": "end_turn"})
                                except Exception:
                                    try:
                                        state = client.act({"action": "wait"})
                                    except Exception:
                                        state = client.get_state()
                            else:
                                time.sleep(0.10)
                                state = client.get_state()

                step_i += 1
                episode_summary["steps"] = step_i

            if not episode_finished:
                if step_i >= 600 and episode_summary.get("end_reason") is None:
                    run = state.get("run") if isinstance(state.get("run"), dict) else {}
                    episode_summary["outcome"] = "max_steps"
                    episode_summary["end_reason"] = "step_cap"
                    publish({
                        "msg_type": "decision",
                        "type": "game_over",
                        "state_type": episode_summary.get("last_state_type"),
                        "step": step_i,
                        "floor": episode_summary.get("last_floor", 0),
                        "act": run.get("act", 1),
                        "outcome": "max_steps",
                        "end_reason": "step_cap",
                        "boss_reached": episode_summary["boss_reached"],
                        "reasoning_zh": "本局到达步数上限，演示在当前状态收尾。",
                        "reasoning_en": "This run reached the step cap, so the demo ended on the current state.",
                    })
                if episode_summary.get("end_reason") is not None:
                    finalize_episode(episode_summary)
                    completed_episodes += 1
    finally:
        artifact_writer.finalize(
            finished_at=_utc_now_iso(),
            requested_episodes=args.episodes,
            completed_episodes=completed_episodes,
            interrupted=_shutdown,
        )
        collector.cleanup()
        client.close()
        logger.info("Demo finished. %d episodes played.", episode)


# Final visible-demo overrides. The last definition wins, so we keep the demo
# copy readable without having to untangle every historical duplicate above.
def generate_reasoning(
    decision_type: str, action: dict, state: dict,
    legal: list[dict], value: float, probs: list[float],
) -> tuple[str, str]:
    act_name = str(action.get("action") or "")
    label = _clean_action_label(action, state)
    player = state.get("player") or state.get("battle", {}).get("player") or {}
    hp = float(player.get("hp", player.get("current_hp", 0)) or 0)
    max_hp = max(1.0, float(player.get("max_hp", 80) or 80))
    hp_ratio = hp / max_hp
    energy = int(player.get("energy", 0) or 0)

    if decision_type == "combat":
        battle = state.get("battle") or {}
        enemies = battle.get("enemies") or []
        total_enemy_hp = sum(int(enemy.get("hp", 0) or 0) for enemy in enemies if isinstance(enemy, dict))
        enemy_attacking = any(
            "attack" in str(intent.get("type", "")).lower()
            for enemy in enemies if isinstance(enemy, dict)
            for intent in (enemy.get("intents") or [])
            if isinstance(intent, dict)
        )

        if act_name == "end_turn":
            if energy == 0:
                return ("能量已经打空了，先结束这一回合。", "No energy remains, so we end the turn.")
            return ("这一拍没有更高价值的动作了，先把回合交出去。", "There is no higher-value play on this turn.")

        if act_name == "play_card":
            if total_enemy_hp <= 15:
                return (f"打出 {label}，这一拍已经接近斩杀。", f"Playing {label}; this line is close to lethal.")
            card = _card_for_action(state, action)
            if _is_block_card(card) and enemy_attacking:
                return (
                    f"先用 {label} 补格挡，把这轮伤害接住。",
                    f"Using {label} for block because the enemy is attacking.",
                )
            if _is_attack_card(card):
                return (f"打出 {label}，主动推进伤害。", f"Playing {label} to push damage.")
            return (f"打出 {label}，维持当前节奏。", f"Playing {label} to keep the current tempo.")

        if act_name == "use_potion":
            return (f"使用药水：{label}。", f"Using potion: {label}.")

        if act_name in {"confirm_selection", "combat_confirm_selection"}:
            return ("确认当前战斗选择。", "Confirming the current combat selection.")

    if decision_type == "map":
        next_node = _clean_action_label(action, state)
        next_lower = next_node.lower()
        if "rest" in next_lower:
            if hp_ratio < 0.5:
                return ("血量偏低，路线优先去休息点。", "HP is low, so we route toward a rest site.")
            return ("这条路线会先去休息点，把状态稳住。", "Heading toward a rest site to stabilize.")
        if "elite" in next_lower:
            return ("选择精英路线，主动换更高回报。", "Taking the elite route for a higher upside.")
        if "shop" in next_lower:
            return ("路线先去商店，把资源转成强度。", "Routing into the shop to convert resources into power.")
        if "monster" in next_lower:
            return ("优先走普通战斗，继续稳定推进。", "Taking a normal combat for steady progress.")
        return (f"选择路线：{next_node}。", f"Choosing path: {next_node}.")

    if decision_type == "card_reward":
        if "skip" in act_name.lower():
            return ("这次跳过卡牌奖励，保持牌组更紧凑。", "Skipping the card reward to keep the deck tight.")
        return (f"选择这张牌：{label}。", f"Taking this card: {label}.")

    if decision_type == "rest":
        lower = label.lower()
        if "rest" in act_name.lower() or "rest" in lower:
            return ("在营火休息，先把血量补回来。", "Resting at the fire to recover HP.")
        if "smith" in act_name.lower() or "upgrade" in lower:
            return ("在营火升级卡牌，换更强的后续回合。", "Upgrading a card at the fire for stronger future turns.")
        return (f"营火选择：{label}。", f"Campfire choice: {label}.")

    if decision_type == "shop":
        return (f"商店选择：{label}。", f"Shop choice: {label}.")

    if decision_type == "event":
        return (f"事件选择：{label}。", f"Event choice: {label}.")

    if decision_type == "treasure":
        return (f"宝箱界面操作：{label}。", f"Treasure action: {label}.")

    return (f"当前动作：{label}。", f"Current action: {label}.")


def _append_screen_specific_details(state: dict, details: list[str]) -> None:
    state_type = str(state.get("state_type") or "").strip().lower()

    if state_type == "card_select":
        card_select = state.get("card_select") if isinstance(state.get("card_select"), dict) else {}
        cards = card_select.get("cards") if isinstance(card_select.get("cards"), list) else []
        selected_cards = card_select.get("selected_cards") if isinstance(card_select.get("selected_cards"), list) else []
        screen_type = str(card_select.get("screen_type") or "").strip().lower()
        if screen_type:
            details.append(f"screen={screen_type}")
        details.append(f"selected={len(selected_cards)}/{len(cards)}")
        remaining = card_select.get("remaining_picks")
        if remaining is not None:
            try:
                details.append(f"remaining={int(remaining)}")
            except Exception:
                pass
        min_select = card_select.get("min_select")
        max_select = card_select.get("max_select")
        if min_select is not None or max_select is not None:
            try:
                details.append(f"quota={int(min_select or 0)}-{int(max_select or len(cards))}")
            except Exception:
                pass
        if card_select.get("can_confirm") is not None:
            details.append(f"confirm={'yes' if bool(card_select.get('can_confirm')) else 'no'}")
        if card_select.get("requires_manual_confirmation") is not None:
            details.append(
                f"manual_confirm={'yes' if bool(card_select.get('requires_manual_confirmation')) else 'no'}"
            )
        return

    if state_type == "event":
        event = state.get("event") if isinstance(state.get("event"), dict) else {}
        options = event.get("options") if isinstance(event.get("options"), list) else []
        if options:
            details.append(f"event_options={len(options)}")
        if event.get("in_dialogue") is not None:
            details.append(f"dialogue={'yes' if bool(event.get('in_dialogue')) else 'no'}")
        return

    if state_type == "shop":
        shop = state.get("shop") if isinstance(state.get("shop"), dict) else {}
        items = shop.get("items") if isinstance(shop.get("items"), list) else []
        if items:
            stocked = sum(1 for item in items if isinstance(item, dict) and item.get("is_stocked", True))
            affordable = sum(1 for item in items if isinstance(item, dict) and item.get("can_afford"))
            details.append(f"shop_items={stocked}")
            details.append(f"affordable={affordable}")
        return

    if state_type == "treasure":
        treasure = state.get("treasure") if isinstance(state.get("treasure"), dict) else {}
        relics = treasure.get("relics") if isinstance(treasure.get("relics"), list) else []
        if relics:
            details.append(f"relic_choices={len(relics)}")
        if treasure.get("can_proceed") is not None:
            details.append(f"can_proceed={'yes' if bool(treasure.get('can_proceed')) else 'no'}")
        return

    if state_type == "combat_rewards":
        rewards = state.get("rewards") if isinstance(state.get("rewards"), dict) else {}
        reward_items = rewards.get("items") if isinstance(rewards.get("items"), list) else []
        if reward_items:
            details.append(f"rewards={len(reward_items)}")
        if rewards.get("can_proceed") is not None:
            details.append(f"can_proceed={'yes' if bool(rewards.get('can_proceed')) else 'no'}")
        return

    if state_type == "card_reward":
        card_reward = state.get("card_reward") if isinstance(state.get("card_reward"), dict) else {}
        cards = card_reward.get("cards") if isinstance(card_reward.get("cards"), list) else []
        if cards:
            details.append(f"reward_cards={len(cards)}")
        if card_reward.get("can_skip") is not None:
            details.append(f"can_skip={'yes' if bool(card_reward.get('can_skip')) else 'no'}")
        return

    if state_type == "map":
        map_state = state.get("map") if isinstance(state.get("map"), dict) else {}
        next_options = map_state.get("next_options") if isinstance(map_state.get("next_options"), list) else []
        if next_options:
            details.append(f"next_nodes={len(next_options)}")


def build_decision(
    decision_type: str, state: dict, legal: list[dict],
    action: dict, action_idx: int, probs: np.ndarray, value: float,
    step: int,
    action_source: str,
    nn_internals: dict | None = None,
    reward_shaping_data: dict | None = None,
    combat_state: dict | None = None,
) -> dict:
    """Build the decision JSON for visible demo overlays."""
    run = state.get("run") or {}
    player = state.get("player") or state.get("battle", {}).get("player") or {}
    battle = state.get("battle") or {}

    hp = player.get("hp", player.get("current_hp", 0))
    max_hp = player.get("max_hp", 80)
    deck = player.get("deck") if isinstance(player.get("deck"), list) else []
    potions = player.get("potions") if isinstance(player.get("potions"), list) else []
    potion_names = [p.get("name", p.get("id", "Empty")) for p in potions if isinstance(p, dict)]

    options = []
    for i, la in enumerate(legal):
        prob = float(probs[i]) if i < len(probs) else 0.0
        opt = {
            "label": _clean_action_label(la, state),
            "prob": round(prob, 3),
            "type": la.get("action", "?"),
            "chosen": bool(i == action_idx),
        }
        if nn_internals and "action_advantages" in nn_internals:
            advs = nn_internals["action_advantages"]
            if i < len(advs):
                opt["advantage"] = advs[i]
        cost = la.get("cost")
        if cost is not None:
            try:
                opt["cost"] = int(cost)
            except Exception:
                pass
        target = (
            la.get("target_name")
            or la.get("target_label")
            or la.get("target")
            or la.get("target_id")
        )
        if target not in (None, ""):
            opt["target"] = str(target)
        options.append(opt)
    options.sort(key=lambda x: -x["prob"])

    enemies_data = []
    for e in (battle.get("enemies") or []):
        intents = e.get("intents") or []
        intent_type = intents[0].get("type", "?") if intents else "?"
        intent_dmg = intents[0].get("damage", 0) if intents else 0
        enemies_data.append({
            "name": e.get("name", e.get("entity_id", "?")),
            "hp": e.get("hp", 0),
            "max_hp": e.get("max_hp", 1),
            "intent": intent_type.lower(),
            "intent_damage": intent_dmg,
            "block": e.get("block", 0),
        })

    reasoning_zh, reasoning_en = generate_reasoning(
        decision_type, action, state, legal, value, probs.tolist())

    ds = deck_score(state) if deck else 0.0
    run_next_boss = (
        run.get("next_boss_name")
        or run.get("boss_name")
        or run.get("next_boss")
        or run.get("boss")
    )
    boss_token = extract_next_boss_token(state)
    readiness_score = (
        reward_shaping_data.get("boss_readiness_score")
        if isinstance(reward_shaping_data, dict) and reward_shaping_data.get("boss_readiness_score") is not None
        else boss_readiness_score(state)
    )
    chosen_label = _clean_action_label(action, state)
    chosen_prob = float(probs[action_idx]) if action_idx < len(probs) else 0.0
    # Problem vector: 9-dim deck capability assessment
    pv = compute_problem_vector(state)
    pv_labels = ["frontload", "aoe", "block", "draw", "energy", "scaling", "consistency", "elite_ready", "boss_answer"]
    pv_dict = {label: round(float(pv[i]), 2) for i, label in enumerate(pv_labels)}

    details = [
        f"candidates={len(legal)}",
        f"chosen_prob={chosen_prob:.2f}",
        f"value={value:.2f}",
        f"deck_score={ds:.2f}",
    ]
    _append_screen_specific_details(state, details)

    result = {
        "msg_type": "decision",
        "title": "AI 决策透视 / AI Decision",
        "type": decision_type,
        "state_type": state.get("state_type", decision_type),
        "step": step,
        "floor": run.get("floor", 0),
        "act": run.get("act", 1),
        "screen": state.get("state_type", "?"),
        "action_source": action_source,
        "player": {
            "hp": hp,
            "max_hp": max_hp,
            "energy": player.get("energy", 0),
            "block": player.get("block", 0),
            "gold": player.get("gold", 0),
            "deck_size": len(deck),
            "deck_score": round(ds, 2),
            "potions": potion_names,
        },
        "enemies": enemies_data,
        "options": options[:12],
        "chosen": {
            "label": chosen_label,
            "index": int(action_idx),
        },
        "chosen_action": chosen_label,
        "value": round(value, 3),
        "reasoning_zh": reasoning_zh,
        "reasoning_en": reasoning_en,
        "reason": reasoning_zh or reasoning_en,
        "next_boss": run_next_boss or boss_token,
        "next_boss_name": run_next_boss,
        "next_boss_archetype": boss_token,
        "boss_readiness": round(float(readiness_score), 3),
        "deck_quality": round(ds, 3),
        "problem_vector": pv_dict,
        "details": details,
    }

    if nn_internals:
        result["nn_internals"] = nn_internals
    if reward_shaping_data:
        result["reward_shaping"] = reward_shaping_data
    if combat_state:
        result["combat_state"] = combat_state

    return result


if __name__ == "__main__":
    main()


