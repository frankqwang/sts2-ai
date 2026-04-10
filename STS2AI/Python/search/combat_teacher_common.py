from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from card_tags import load_card_tags
from combat_nn import (
    CombatPolicyValueNetwork,
    build_combat_action_features,
    build_combat_features,
)
from rl_encoder_v2 import _lower, _safe_float, _safe_int
from vocab import Vocab, _slugify, load_vocab

COMBAT_TEACHER_SCHEMA_VERSION = "combat_teacher_dataset.v1"
COMBAT_TURN_SOLUTION_SCHEMA_VERSION = "combat_turn_solution.v1"
COMBAT_MICROBENCH_SCHEMA_VERSION = "combat_microbench.v1"

COMBAT_STATE_TYPES = {"combat", "monster", "elite", "boss"}
UNSUPPORTED_STATE_TYPES = {"hand_select", "card_select", "relic_select"}
SUPPORTED_SOLVER_ACTIONS = {"play_card", "use_potion", "end_turn"}
UNSUPPORTED_CARD_TAGS = {
    "draw",
    "generate_card",
    "upgrade_card",
    "discard",
    "random_target",
}
UNSUPPORTED_KEYWORDS = {
    "discover",
}
BODY_SLAM_TOKENS = {
    "body_slam",
    "bodyslam",
}
MOTIF_NAMES = (
    "missed_lethal",
    "direct_lethal_first_action",
    "turn_lethal_no_end_turn",
    "bash_before_strike",
    "bodyslam_before_block",
    "bad_end_turn",
    "potion_misuse",
)


@dataclass(slots=True)
class BaselineCombatPolicy:
    network: CombatPolicyValueNetwork
    vocab: Vocab
    device: torch.device

    def score(self, state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> dict[str, Any]:
        sf = build_combat_features(state, self.vocab)
        af = build_combat_action_features(state, legal_actions, self.vocab)

        state_t: dict[str, torch.Tensor] = {}
        for key, value in sf.items():
            tensor = torch.tensor(value).unsqueeze(0)
            if value.dtype in (np.int64, np.int32):
                tensor = tensor.long()
            elif value.dtype == bool:
                tensor = tensor.bool()
            else:
                tensor = tensor.float()
            state_t[key] = tensor.to(self.device)

        action_t: dict[str, torch.Tensor] = {}
        for key, value in af.items():
            tensor = torch.tensor(value).unsqueeze(0)
            if value.dtype in (np.int64, np.int32):
                tensor = tensor.long()
            elif value.dtype == bool:
                tensor = tensor.bool()
            else:
                tensor = tensor.float()
            action_t[key] = tensor.to(self.device)

        with torch.no_grad():
            logits, value = self.network(state_t, action_t)

        count = min(len(legal_actions), logits.shape[1])
        raw_logits = logits[0, :count].detach().cpu().float().numpy()
        if count > 0:
            shifted = raw_logits - raw_logits.max()
            probs = np.exp(shifted)
            probs = probs / max(np.sum(probs), 1e-8)
            best_index = int(np.argmax(raw_logits))
        else:
            probs = np.zeros(0, dtype=np.float32)
            best_index = 0
        return {
            "logits": raw_logits.astype(np.float32),
            "probs": probs.astype(np.float32),
            "value": float(value[0].detach().cpu().item()),
            "best_index": best_index,
        }


def _safe_load_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
) -> None:
    current = model.state_dict()
    filtered: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None):
            filtered[key] = value
    model.load_state_dict(filtered, strict=False)


def load_baseline_combat_policy(
    checkpoint_path: str | Path,
    *,
    vocab: Vocab | None = None,
    device: torch.device | None = None,
) -> BaselineCombatPolicy:
    vocab = vocab or load_vocab()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    combat_state = checkpoint.get("mcts_model") or checkpoint.get("model_state_dict")
    if not isinstance(combat_state, dict):
        raise ValueError(f"Combat checkpoint has no mcts_model/model_state_dict: {checkpoint_path}")

    embed_weight = combat_state.get("entity_emb.card_embed.weight")
    action_proj = combat_state.get("action_proj.weight")
    embed_dim = int(embed_weight.shape[1]) if isinstance(embed_weight, torch.Tensor) and embed_weight.ndim == 2 else 32
    hidden_dim = int(action_proj.shape[0]) if isinstance(action_proj, torch.Tensor) and action_proj.ndim == 2 else 128

    network = CombatPolicyValueNetwork(vocab=vocab, embed_dim=embed_dim, hidden_dim=hidden_dim)
    _safe_load_state_dict(network, combat_state)
    network.to(device).eval()
    return BaselineCombatPolicy(network=network, vocab=vocab, device=device)


def sanitize_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    clean: dict[str, Any] = {}
    for key in (
        "action",
        "label",
        "index",
        "card_index",
        "target_id",
        "target",
        "slot",
        "card_id",
    ):
        if key in action:
            clean[key] = action.get(key)
    return clean


def _battle(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle")
    return battle if isinstance(battle, dict) else {}


def _player(state: dict[str, Any]) -> dict[str, Any]:
    battle = _battle(state)
    player = battle.get("player")
    if isinstance(player, dict):
        return player
    root_player = state.get("player")
    return root_player if isinstance(root_player, dict) else {}


def _enabled_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    legal = state.get("legal_actions")
    if not isinstance(legal, list):
        return []
    return [action for action in legal if isinstance(action, dict) and action.get("is_enabled") is not False]


def _card_slug(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    return _slugify(card.get("id") or card.get("card_id") or card.get("name") or "").lower()


def _card_for_action(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    if _lower(action.get("action")) != "play_card":
        return None
    battle = _battle(state)
    player = _player(state)
    hand = battle.get("hand")
    if not isinstance(hand, list):
        hand = player.get("hand")
    if not isinstance(hand, list):
        return None
    idx = action.get("card_index")
    try:
        card_index = int(idx)
    except Exception:
        card_index = -1
    if 0 <= card_index < len(hand) and isinstance(hand[card_index], dict):
        return hand[card_index]
    action_card_id = _lower(action.get("card_id"))
    action_label = _lower(action.get("label"))
    for card in hand:
        if not isinstance(card, dict):
            continue
        if action_card_id and _lower(card.get("id")) == action_card_id:
            return card
        if action_label and _lower(card.get("name") or card.get("label")) == action_label:
            return card
    return None


def _enemy_target_lookup(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    battle = _battle(state)
    enemies = battle.get("enemies")
    lookup: dict[str, dict[str, Any]] = {}
    if isinstance(enemies, list):
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            for key in (
                enemy.get("entity_id"),
                enemy.get("combat_id"),
                enemy.get("id"),
                enemy.get("name"),
            ):
                if key is not None:
                    lookup[str(key)] = enemy
    return lookup


def _target_enemy_for_action(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    lookup = _enemy_target_lookup(state)
    target = action.get("target_id")
    if target is None:
        target = action.get("target")
    if target is None:
        return None
    return lookup.get(str(target))


def _power_amount(powers: Any, token: str) -> float:
    if not isinstance(powers, list):
        return 0.0
    for power in powers:
        if not isinstance(power, dict):
            continue
        pid = _lower(power.get("id") or power.get("power_id") or power.get("name"))
        if token in pid:
            return _safe_float(power.get("amount") or power.get("stacks") or 0.0, 0.0)
    return 0.0


def _enemy_has_vulnerable(enemy: dict[str, Any] | None) -> bool:
    if not isinstance(enemy, dict):
        return False
    powers = enemy.get("status") or enemy.get("powers") or []
    return _power_amount(powers, "vulnerable") > 0


def canonical_public_state_payload(
    state: dict[str, Any],
    *,
    action_budget_used: int = 0,
) -> dict[str, Any]:
    battle = _battle(state)
    player = _player(state)
    hand = battle.get("hand")
    if not isinstance(hand, list):
        hand = player.get("hand")
    if not isinstance(hand, list):
        hand = []
    hand_counter: Counter[tuple[str, int, int]] = Counter()
    for card in hand:
        if not isinstance(card, dict):
            continue
        hand_counter[
            (
                _card_slug(card),
                1 if bool(card.get("is_upgraded") or _safe_int(card.get("upgrades"), 0) > 0) else 0,
                _safe_int(card.get("cost"), 0),
            )
        ] += 1

    enemies_payload: list[dict[str, Any]] = []
    enemies = battle.get("enemies")
    if isinstance(enemies, list):
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            intents_payload = []
            intents = enemy.get("intents")
            if isinstance(intents, list):
                for intent in intents:
                    if not isinstance(intent, dict):
                        continue
                    intents_payload.append(
                        {
                            "type": _lower(intent.get("type")),
                            "label": str(intent.get("label") or ""),
                            "total_damage": _safe_int(
                                intent.get("total_damage")
                                or intent.get("damage")
                                or intent.get("intent_damage"),
                                0,
                            ),
                        }
                    )
            powers_payload = []
            for power in enemy.get("status") or enemy.get("powers") or []:
                if not isinstance(power, dict):
                    continue
                powers_payload.append(
                    (
                        _lower(power.get("id") or power.get("power_id") or power.get("name")),
                        _safe_int(power.get("amount") or power.get("stacks"), 0),
                    )
                )
            enemies_payload.append(
                {
                    "entity_id": str(enemy.get("entity_id") or enemy.get("combat_id") or enemy.get("id") or enemy.get("name") or ""),
                    "name": _lower(enemy.get("name") or enemy.get("id")),
                    "hp": _safe_int(enemy.get("hp", enemy.get("current_hp")), 0),
                    "max_hp": _safe_int(enemy.get("max_hp"), 0),
                    "block": _safe_int(enemy.get("block"), 0),
                    "is_alive": bool(enemy.get("is_alive", True)),
                    "powers": sorted(powers_payload),
                    "intents": intents_payload,
                }
            )

    player_powers_payload = []
    for power in player.get("status") or player.get("powers") or []:
        if not isinstance(power, dict):
            continue
        player_powers_payload.append(
            (
                _lower(power.get("id") or power.get("power_id") or power.get("name")),
                _safe_int(power.get("amount") or power.get("stacks"), 0),
            )
        )

    potions_payload = []
    for potion in player.get("potions") or []:
        if not isinstance(potion, dict):
            continue
        potions_payload.append(
            (
                _safe_int(potion.get("slot"), -1),
                _lower(potion.get("id") or potion.get("name")),
                bool(potion.get("can_use_in_combat", True)),
            )
        )

    return {
        "state_type": _lower(state.get("state_type")),
        "floor": _safe_int((state.get("run") or {}).get("floor"), 0),
        "round": _safe_int(battle.get("round") or battle.get("round_number"), 0),
        "turn": _lower(battle.get("turn")),
        "is_play_phase": bool(battle.get("is_play_phase", False)),
        "action_budget_used": int(action_budget_used),
        "player": {
            "hp": _safe_int(player.get("hp", player.get("current_hp")), 0),
            "max_hp": _safe_int(player.get("max_hp"), 0),
            "block": _safe_int(player.get("block"), 0),
            "energy": _safe_int(battle.get("energy") or player.get("energy"), 0),
            "max_energy": _safe_int(battle.get("max_energy") or player.get("max_energy"), 0),
            "draw_pile_count": _safe_int(battle.get("draw_pile_count") or player.get("draw_pile_count"), 0),
            "discard_pile_count": _safe_int(battle.get("discard_pile_count") or player.get("discard_pile_count"), 0),
            "exhaust_pile_count": _safe_int(battle.get("exhaust_pile_count") or player.get("exhaust_pile_count"), 0),
            "powers": sorted(player_powers_payload),
        },
        "hand_multiset": [
            {
                "card_slug": slug,
                "is_upgraded": upgraded,
                "cost": cost,
                "count": count,
            }
            for (slug, upgraded, cost), count in sorted(hand_counter.items())
        ],
        "potions": sorted(potions_payload),
        "enemies": enemies_payload,
    }


def canonical_public_state_hash(
    state: dict[str, Any],
    *,
    action_budget_used: int = 0,
) -> str:
    payload = canonical_public_state_payload(state, action_budget_used=action_budget_used)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def immediate_expected_hp_loss_this_enemy_turn(state: dict[str, Any]) -> float:
    battle = _battle(state)
    player = _player(state)
    block = _safe_float(player.get("block"), 0.0)
    incoming = 0.0
    enemies = battle.get("enemies")
    if isinstance(enemies, list):
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            if _safe_int(enemy.get("hp", enemy.get("current_hp")), 0) <= 0:
                continue
            intents = enemy.get("intents")
            if isinstance(intents, list):
                for intent in intents:
                    if not isinstance(intent, dict):
                        continue
                    damage = _safe_float(
                        intent.get("total_damage")
                        or intent.get("damage")
                        or intent.get("intent_damage"),
                        0.0,
                    )
                    incoming += damage
            else:
                incoming += _safe_float(enemy.get("intent_damage"), 0.0)
    return max(0.0, incoming - block)


def _has_non_end_turn_play(state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> bool:
    for action in legal_actions:
        if _lower(action.get("action")) in {"play_card", "use_potion"}:
            if is_action_supported_for_turn_solver(state, action):
                return True
    return False


def is_action_supported_for_turn_solver(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    card_tags: dict[str, list[str]] | None = None,
) -> bool:
    return unsupported_action_reason_for_turn_solver(state, action, card_tags=card_tags) is None


def unsupported_action_reason_for_turn_solver(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    card_tags: dict[str, list[str]] | None = None,
) -> str | None:
    """Return None if the action is supported by the turn solver, otherwise a
    short reason string (e.g. "unsupported_card_tag:draw"). Used by the QG
    diagnostic instrumentation in `solver_support_diagnostics`.
    """
    action_name = _lower(action.get("action"))
    if action_name not in SUPPORTED_SOLVER_ACTIONS:
        return f"unsupported_action_type:{action_name or 'unknown'}"
    if action_name == "end_turn":
        return None
    if action_name == "use_potion":
        return None
    card_tags = card_tags or load_card_tags()
    card = _card_for_action(state, action)
    slug = _card_slug(card)
    tags = set(card_tags.get(slug, []))
    keywords = {_lower(keyword) for keyword in (card.get("keywords") or [])} if isinstance(card, dict) else set()
    blocked_tags = sorted(tags & UNSUPPORTED_CARD_TAGS)
    if blocked_tags:
        return f"unsupported_card_tag:{blocked_tags[0]}"
    blocked_keywords = sorted(keywords & UNSUPPORTED_KEYWORDS)
    if blocked_keywords:
        return f"unsupported_keyword:{blocked_keywords[0]}"
    return None


def is_supported_solver_state(
    state: dict[str, Any],
    *,
    card_tags: dict[str, list[str]] | None = None,
) -> bool:
    return solver_support_diagnostics(state, card_tags=card_tags).get("supported", False)


def solver_support_diagnostics(
    state: dict[str, Any],
    *,
    card_tags: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Return a dict describing whether the turn solver can run on this state.
    Always populated keys: `supported` (bool), `reason` (str), and per-action
    counts/details. Used by `multi_turn_solver_planner` and the trace tools to
    explain why the solver fell back.
    """
    card_tags = card_tags or load_card_tags()
    state_type = _lower(state.get("state_type"))
    battle = _battle(state)
    legal_actions = _enabled_legal_actions(state)
    diagnostics: dict[str, Any] = {
        "supported": False,
        "reason": "",
        "state_type": state_type,
        "battle_is_play_phase": bool(battle.get("is_play_phase", False)),
        "battle_turn": _lower(battle.get("turn")),
        "has_card_selection": isinstance(battle.get("card_selection"), dict),
        "enabled_legal_action_count": len(legal_actions),
        "supported_action_count": 0,
        "unsupported_action_count": 0,
        "unsupported_action_reasons": [],
    }
    if state_type not in COMBAT_STATE_TYPES:
        diagnostics["reason"] = f"unsupported_state_type:{state_type or 'unknown'}"
        return diagnostics
    if not bool(battle.get("is_play_phase", False)):
        diagnostics["reason"] = "not_play_phase"
        return diagnostics
    if _lower(battle.get("turn")) not in {"player", ""}:
        diagnostics["reason"] = f"not_player_turn:{_lower(battle.get('turn')) or 'unknown'}"
        return diagnostics
    if isinstance(battle.get("card_selection"), dict):
        diagnostics["reason"] = "card_selection_active"
        return diagnostics
    if _lower(state.get("state_type")) in UNSUPPORTED_STATE_TYPES:
        diagnostics["reason"] = f"unsupported_state_type:{state_type}"
        return diagnostics
    if not legal_actions:
        diagnostics["reason"] = "no_enabled_legal_actions"
        return diagnostics
    unsupported_actions: list[dict[str, Any]] = []
    for action in legal_actions:
        reason = unsupported_action_reason_for_turn_solver(state, action, card_tags=card_tags)
        if reason is not None:
            unsupported_actions.append(
                {
                    "reason": reason,
                    "action": sanitize_action(action) or {},
                }
            )
    diagnostics["unsupported_action_count"] = len(unsupported_actions)
    diagnostics["unsupported_action_reasons"] = unsupported_actions
    diagnostics["supported_action_count"] = max(0, len(legal_actions) - len(unsupported_actions))
    if diagnostics["supported_action_count"] <= 0:
        diagnostics["reason"] = "unsupported_legal_actions"
        return diagnostics
    diagnostics["supported"] = True
    diagnostics["reason"] = "supported"
    return diagnostics


def detect_motif_labels(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    card_tags: dict[str, list[str]] | None = None,
) -> list[str]:
    card_tags = card_tags or load_card_tags()
    labels: list[str] = []

    vulnerable_actions = []
    strike_like_actions = []
    body_slam_actions = []
    block_actions = []
    for action in legal_actions:
        if _lower(action.get("action")) != "play_card":
            continue
        card = _card_for_action(state, action)
        slug = _card_slug(card)
        tags = set(card_tags.get(slug, []))
        if "vulnerable" in tags:
            vulnerable_actions.append(action)
        if "damage" in tags and "vulnerable" not in tags:
            strike_like_actions.append(action)
        if slug in BODY_SLAM_TOKENS:
            body_slam_actions.append(action)
        if "block" in tags and slug not in BODY_SLAM_TOKENS:
            block_actions.append(action)

    if vulnerable_actions and strike_like_actions:
        for action in vulnerable_actions:
            enemy = _target_enemy_for_action(state, action)
            if not _enemy_has_vulnerable(enemy):
                labels.append("bash_before_strike")
                break

    player = _player(state)
    if body_slam_actions and _safe_int(player.get("block"), 0) <= 0 and block_actions:
        labels.append("bodyslam_before_block")

    if any(_lower(action.get("action")) == "end_turn" for action in legal_actions) and _has_non_end_turn_play(state, legal_actions):
        labels.append("bad_end_turn")

    if any(_lower(action.get("action")) == "use_potion" for action in legal_actions):
        labels.append("potion_misuse")

    return sorted(set(labels))


def estimate_line_continuation_targets(
    *,
    terminal_state: dict[str, Any],
    baseline_value: float,
    total_potions_used: int,
) -> dict[str, float]:
    battle = _battle(terminal_state)
    enemies = battle.get("enemies")
    if isinstance(enemies, list):
        living = sum(1 for enemy in enemies if isinstance(enemy, dict) and _safe_int(enemy.get("hp", enemy.get("current_hp")), 0) > 0)
    else:
        living = 0
    if _lower(terminal_state.get("state_type")) not in COMBAT_STATE_TYPES:
        win_prob = 1.0
    elif living <= 0:
        win_prob = 1.0
    else:
        win_prob = max(0.0, min(1.0, 0.5 * (float(baseline_value) + 1.0)))
    return {
        "win_prob": float(win_prob),
        "expected_hp_loss": float(immediate_expected_hp_loss_this_enemy_turn(terminal_state)),
        "expected_potion_cost": float(max(0, total_potions_used)),
    }


def compute_immediate_action_components(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    action: dict[str, Any],
    next_state: dict[str, Any] | None,
    *,
    card_tags: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    card_tags = card_tags or load_card_tags()
    components = {
        "lethal_bonus": 0.0,
        "vulnerable_setup_bonus": 0.0,
        "body_slam_after_block_bonus": 0.0,
        "bad_end_turn_penalty": 0.0,
        "potion_waste_penalty": 0.0,
        "potion_cost": 0.0,
        # Step 1 (near-win analysis 2026-04-07): hand-crafted rules to fix
        # patterns observed in benchmark-38/12/49/33/40/11 near-win losses.
        "power_card_early_bonus": 0.0,
        "damage_potion_clutch_bonus": 0.0,
        "x_cost_first_in_turn_bonus": 0.0,
        "early_game_defend_penalty": 0.0,
    }
    action_name = _lower(action.get("action"))

    # Pre-compute boss HP fraction (for damage_potion_clutch_bonus rule)
    battle = _battle(state)
    enemies = battle.get("enemies") or []
    e_hp_total = 0
    e_max_total = 1
    for e in enemies:
        if isinstance(e, dict):
            e_hp_total += _safe_int(e.get("hp", e.get("current_hp", 0)), 0)
            e_max_total += _safe_int(e.get("max_hp", 1), 0)
    boss_hp_frac = e_hp_total / max(1, e_max_total)
    round_number = _safe_int(battle.get("round_number_raw", battle.get("round_number", 0)), 0)
    is_boss = _lower(state.get("state_type")) == "boss"

    if action_name == "end_turn" and _has_non_end_turn_play(state, legal_actions):
        components["bad_end_turn_penalty"] = 0.35
        return components

    if action_name == "use_potion":
        components["potion_cost"] = 1.0
        if immediate_expected_hp_loss_this_enemy_turn(state) <= 0.0:
            components["potion_waste_penalty"] = 0.25
        # Step 1 rule 2: damage_potion_clutch_bonus
        # Damage potions used early (boss > 50% HP) waste their burst potential.
        # Damage potions used in clutch (boss < 30% HP) often close near-wins.
        # We approximate "damage potion" by checking the potion id/label/name
        # for damage-related keywords. Heal/utility potions don't get the bonus.
        if is_boss:
            potion_label = _lower(action.get("label", "") or action.get("potion_id", ""))
            damage_potion_keywords = (
                "fire", "explosive", "strength", "dexterity", "energy",
                "attack", "ancient", "blood", "essence_of_steel",
            )
            looks_like_damage_potion = any(kw in potion_label for kw in damage_potion_keywords)
            if looks_like_damage_potion or "potion" in potion_label:
                if boss_hp_frac > 0.5:
                    components["damage_potion_clutch_bonus"] = -0.5
                elif boss_hp_frac < 0.3:
                    components["damage_potion_clutch_bonus"] = 0.6
        return components

    if action_name != "play_card":
        return components

    card = _card_for_action(state, action)
    slug = _card_slug(card)
    tags = set(card_tags.get(slug, []))
    player = _player(state)
    block_before = _safe_int(player.get("block"), 0)

    if "vulnerable" in tags:
        enemy = _target_enemy_for_action(state, action)
        if not _enemy_has_vulnerable(enemy):
            if any(
                _lower(other.get("action")) == "play_card"
                and other is not action
                and "damage" in set(card_tags.get(_card_slug(_card_for_action(state, other)), []))
                for other in legal_actions
            ):
                components["vulnerable_setup_bonus"] = 0.20

    if slug in BODY_SLAM_TOKENS:
        if block_before <= 0:
            if any(
                _lower(other.get("action")) == "play_card"
                and "block" in set(card_tags.get(_card_slug(_card_for_action(state, other)), []))
                and _card_slug(_card_for_action(state, other)) not in BODY_SLAM_TOKENS
                for other in legal_actions
            ):
                components["body_slam_after_block_bonus"] = -0.35
    elif "block" in tags and block_before <= 0:
        if any(
            _lower(other.get("action")) == "play_card"
            and _card_slug(_card_for_action(state, other)) in BODY_SLAM_TOKENS
            for other in legal_actions
        ):
            components["body_slam_after_block_bonus"] = 0.10

    if next_state is not None and _lower(next_state.get("state_type")) not in COMBAT_STATE_TYPES:
        components["lethal_bonus"] = 0.75
    elif next_state is not None:
        battle_n = _battle(next_state)
        enemies_n = battle_n.get("enemies")
        if isinstance(enemies_n, list) and all(_safe_int(enemy.get("hp", enemy.get("current_hp")), 0) <= 0 for enemy in enemies_n if isinstance(enemy, dict)):
            components["lethal_bonus"] = 0.75

    # Step 1 rule 1: power_card_early_bonus
    # Power cards (Inflame, Inferno, Stampede, Demon Form, etc.) compound over
    # the rest of combat. Playing them on round 0-2 gives ~5-10 turns of value;
    # playing them on round 5+ gives almost nothing. Reward early plays.
    if is_boss and "power" in tags:
        if round_number <= 2:
            components["power_card_early_bonus"] = 0.4
        elif round_number <= 4:
            components["power_card_early_bonus"] = 0.15

    # Step 1 rule 3: x_cost_first_in_turn_bonus
    # X-cost cards (Whirlwind, Spite, etc.) deal damage proportional to energy
    # spent. They MUST be played first in the turn (with full energy) to be
    # worth their slot. We detect "first action of turn" by checking that no
    # cards have been played yet (block_before == 0 AND energy == max_energy).
    if is_boss and "x_cost" in tags:
        cur_energy = _safe_int(battle.get("energy"), 0)
        max_energy = _safe_int(battle.get("max_energy"), 3)
        if cur_energy >= max_energy:
            components["x_cost_first_in_turn_bonus"] = 0.3
        else:
            # Played late - lost burst value
            components["x_cost_first_in_turn_bonus"] = -0.2

    # Step 1 rule 4: early_game_defend_penalty
    # At high HP (>70%) with low incoming damage (<10), playing Defend wastes
    # an energy that could deal 5-15 boss damage. This is the #1 mistake in
    # benchmark-38: defending at 80/80 HP burned 2-3 critical turns.
    if is_boss and "block" in tags and slug not in BODY_SLAM_TOKENS:
        p_hp = _safe_int(player.get("hp", player.get("current_hp", 0)), 0)
        p_max = max(1, _safe_int(player.get("max_hp", 1), 0))
        hp_frac = p_hp / p_max
        if hp_frac > 0.7:
            incoming = immediate_expected_hp_loss_this_enemy_turn(state)
            if incoming < 10:
                components["early_game_defend_penalty"] = -0.3

    return components


def combine_leaf_breakdown(
    terminal_state: dict[str, Any],
    *,
    baseline_value: float,
    total_action_components: dict[str, float] | None = None,
) -> dict[str, float]:
    total_action_components = total_action_components or {}
    expected_hp_loss = immediate_expected_hp_loss_this_enemy_turn(terminal_state)
    breakdown = {
        "combat_net_value": float(baseline_value),
        "immediate_expected_hp_loss_this_enemy_turn": float(expected_hp_loss),
        "lethal_bonus": float(total_action_components.get("lethal_bonus", 0.0)),
        "vulnerable_setup_bonus": float(total_action_components.get("vulnerable_setup_bonus", 0.0)),
        "body_slam_after_block_bonus": float(total_action_components.get("body_slam_after_block_bonus", 0.0)),
        "bad_end_turn_penalty": float(total_action_components.get("bad_end_turn_penalty", 0.0)),
        "potion_waste_penalty": float(total_action_components.get("potion_waste_penalty", 0.0)),
        "potion_cost": float(total_action_components.get("potion_cost", 0.0)),
        # Step 1 (2026-04-07 near-win analysis fix)
        "power_card_early_bonus": float(total_action_components.get("power_card_early_bonus", 0.0)),
        "damage_potion_clutch_bonus": float(total_action_components.get("damage_potion_clutch_bonus", 0.0)),
        "x_cost_first_in_turn_bonus": float(total_action_components.get("x_cost_first_in_turn_bonus", 0.0)),
        "early_game_defend_penalty": float(total_action_components.get("early_game_defend_penalty", 0.0)),
    }
    breakdown["total"] = (
        breakdown["combat_net_value"]
        - 0.15 * breakdown["immediate_expected_hp_loss_this_enemy_turn"]
        + breakdown["lethal_bonus"]
        + breakdown["vulnerable_setup_bonus"]
        + breakdown["body_slam_after_block_bonus"]
        - breakdown["bad_end_turn_penalty"]
        - breakdown["potion_waste_penalty"]
        + breakdown["power_card_early_bonus"]
        + breakdown["damage_potion_clutch_bonus"]
        + breakdown["x_cost_first_in_turn_bonus"]
        + breakdown["early_game_defend_penalty"]
    )
    return breakdown


def aggregate_action_components(parts: list[dict[str, float]]) -> dict[str, float]:
    totals = {
        "lethal_bonus": 0.0,
        "vulnerable_setup_bonus": 0.0,
        "body_slam_after_block_bonus": 0.0,
        "bad_end_turn_penalty": 0.0,
        "potion_waste_penalty": 0.0,
        "potion_cost": 0.0,
        # Step 1 (2026-04-07 near-win analysis fix)
        "power_card_early_bonus": 0.0,
        "damage_potion_clutch_bonus": 0.0,
        "x_cost_first_in_turn_bonus": 0.0,
        "early_game_defend_penalty": 0.0,
    }
    for part in parts:
        for key in totals:
            totals[key] += float(part.get(key, 0.0))
    return totals


def stable_sample_id(seed: str, state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> str:
    payload = {
        "seed": seed,
        "state_hash": canonical_public_state_hash(state),
        "legal_actions": [sanitize_action(action) for action in legal_actions],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()
