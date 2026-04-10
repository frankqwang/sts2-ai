"""TurnSolverPlanner: adapter that drives `CombatTurnSolver` from
`evaluate_ai.py`'s `_select_action_nn` hook.

Usage in `evaluate_ai.py`:
    from turn_solver_planner import build_turn_solver_planner
    planner = build_turn_solver_planner(combat_net=combat_net,
                                        vocab=vocab, device=device,
                                        mode="boss",
                                        max_player_actions=12)
    # then pass `turn_planner=planner` to evaluate_batch / run_single_game.

The planner has a `select_action(pipe_getter, state, legal)` method matching
the existing `combat_turn_planner.TurnPlanner` interface, and a `_mode`
attribute (`always` / `boss` / `elite` / `boss_elite`).

Caching: once the solver finds a full turn line, the cached actions are
returned for subsequent decisions in the same turn (until the cache is
exhausted, the next state hash diverges, or end_turn is reached).
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from boss_leaf_evaluator import load_boss_leaf_evaluator_runtime
from combat_nn import CombatPolicyValueNetwork
from combat_teacher_common import (
    BaselineCombatPolicy,
    combine_leaf_breakdown,
    sanitize_action,
)
from combat_turn_solver import (
    CombatTurnSolver,
    CombatTurnSolution,
    SolverLineResult,
    _is_terminal_combat_leaf,
    _zero_components,
)
from rl_encoder_v2 import Vocab

logger = logging.getLogger("turn_solver_planner")

_BOSS_SETUP_HEURISTIC_FLOOR = 0.8


@dataclass
class _CachedTurnPlan:
    actions: list[dict[str, Any]] = field(default_factory=list)
    next_idx: int = 0
    state_hash: str = ""

    def is_exhausted(self) -> bool:
        return self.next_idx >= len(self.actions)


class _PipeEnvAdapter:
    """Adapt a raw pipe object into a CombatTurnBranchEnv-compatible wrapper.
    Mimics PipeBackedFullRunClient's exact response handling so the solver
    sees identical save/load/act semantics as the live client.
    """

    def __init__(self, pipe: Any):
        self._pipe = pipe

    def save_state(self) -> str:
        result = self._pipe.call("save_state")
        state_id = result.get("state_id") if isinstance(result, dict) else None
        return str(state_id) if state_id else ""

    def load_state(self, state_id: str) -> dict[str, Any]:
        result = self._pipe.call("load_state", {"state_id": str(state_id)})
        return result if isinstance(result, dict) else {}

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._pipe.call("step", payload)
        if not isinstance(result, dict):
            return {}
        state = result.get("state")
        # PipeBackedFullRunClient returns inner state even on accepted=False
        # if the inner state is a valid combat/transition dict.
        if isinstance(state, dict) and state.get("state_type"):
            return state
        # Some C# transitions return accepted=True with state nested elsewhere
        if isinstance(state, dict):
            return state
        return result

    def delete_state(self, state_id: str) -> bool:
        try:
            r = self._pipe.call("delete_state", {"state_id": str(state_id)})
            return bool(r.get("deleted", False)) if isinstance(r, dict) else False
        except Exception:
            return False

    def clear_state_cache(self) -> bool:
        try:
            r = self._pipe.call("delete_state", {"clear_all": True})
            return bool(r.get("deleted", False)) if isinstance(r, dict) else False
        except Exception:
            return False


def _state_signature(state: dict[str, Any]) -> str:
    """Cheap signature for cache validity check.

    The cache is valid as long as we are in the SAME turn (same round number)
    and the state_type is still combat. We deliberately do NOT include HP /
    hand / energy because those change as we play through the cached action
    line — that's the whole point of caching the line.
    """
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    rnd = int(battle.get("round_number", battle.get("round", state.get("round_number", state.get("round", 0)))) or 0)
    st = (state.get("state_type") or "").lower()
    return f"st{st}_r{rnd}"


def _action_matches_legal(action: dict[str, Any], legal_action: dict[str, Any]) -> bool:
    if not isinstance(action, dict) or not isinstance(legal_action, dict):
        return False
    if (action.get("action") or "") != (legal_action.get("action") or ""):
        return False
    for key in ("card_index", "index", "slot"):
        a_value = action.get(key)
        b_value = legal_action.get(key)
        if a_value is None or b_value is None:
            continue
        if a_value != b_value:
            return False
    a_label = action.get("label") or ""
    b_label = legal_action.get("label") or ""
    if a_label and b_label and a_label != b_label:
        return False
    a_card = action.get("card_id") or ""
    b_card = legal_action.get("card_id") or ""
    if a_card and b_card and a_card != b_card:
        return False
    a_target = action.get("target_id") or action.get("target") or ""
    b_target = legal_action.get("target_id") or legal_action.get("target") or ""
    if a_target and b_target and a_target != b_target:
        return False
    return True


def _action_matches_legal_semantic(action: dict[str, Any], legal_action: dict[str, Any]) -> bool:
    if not isinstance(action, dict) or not isinstance(legal_action, dict):
        return False
    if (action.get("action") or "") != (legal_action.get("action") or ""):
        return False
    for key in ("label", "card_id", "screen_type", "node_type"):
        a_value = action.get(key)
        b_value = legal_action.get(key)
        if a_value is None or b_value is None or a_value == "" or b_value == "":
            continue
        if a_value != b_value:
            return False
    a_target = action.get("target_id") or action.get("target") or ""
    b_target = legal_action.get("target_id") or legal_action.get("target") or ""
    if a_target and b_target and a_target != b_target:
        return False
    return True


def _find_cache_match(
    action: dict[str, Any],
    legal: list[dict[str, Any]],
) -> tuple[int | None, str | None]:
    for idx, legal_action in enumerate(legal):
        if _action_matches_legal(action, legal_action):
            return idx, "exact"
    semantic_matches = [
        idx
        for idx, legal_action in enumerate(legal)
        if _action_matches_legal_semantic(action, legal_action)
    ]
    if len(semantic_matches) == 1:
        return semantic_matches[0], "semantic"
    return None, None


def _same_first_action(
    left_line: list[dict[str, Any]] | None,
    right_line: list[dict[str, Any]] | None,
) -> bool:
    left = left_line[0] if left_line else None
    right = right_line[0] if right_line else None
    return _action_matches_legal(left or {}, right or {})


def _incoming_damage_from_enemy(enemy: dict[str, Any]) -> float:
    """Estimate next-turn damage incoming from this enemy.

    Prefers `total_damage` (already pre-multiplied by hit count in C#)
    when present. Otherwise falls back to per-hit `damage * repeats`.
    The 2026-04-08 API DTO change added explicit `repeats` and per-hit
    `damage` fields so multi-hit attacks like Vantom INKY_LANCE_MOVE
    are no longer collapsed to a single 1-hit estimate.
    """
    if not isinstance(enemy, dict):
        return 0.0
    intent = enemy.get("intent")
    if isinstance(intent, dict):
        # Prefer total_damage if it's a positive value (already multiplied).
        total_damage = float(intent.get("total_damage", 0) or 0.0)
        if total_damage > 0:
            return total_damage
        damage = float(intent.get("damage", 0) or 0.0)
        hits = max(1.0, float(intent.get("repeats", intent.get("hits", intent.get("multiplier", 1))) or 1.0))
        return damage * hits
    intents = enemy.get("intents") if isinstance(enemy.get("intents"), list) else []
    if intents and isinstance(intents[0], dict):
        total = 0.0
        for item in intents:
            if not isinstance(item, dict):
                continue
            # Prefer total_damage when present (post-2026-04-08 C# API).
            it_total = float(item.get("total_damage", 0) or 0.0)
            if it_total > 0:
                total += it_total
                continue
            damage = float(item.get("damage", 0) or 0.0)
            hits = max(1.0, float(item.get("repeats", item.get("hits", item.get("multiplier", 1))) or 1.0))
            total += damage * hits
        if total > 0.0:
            return total
    damage = float(enemy.get("intent_damage", 0) or 0.0)
    hits = max(1.0, float(enemy.get("intent_hits", 1) or 1.0))
    return damage * hits


class _MCValueHead(torch.nn.Module):
    """Tiny MLP that scores combat states from the offline MC dataset."""

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


_MC_FEATURE_KEYS = [
    "round_number",
    "player_hp_frac",
    "player_block_frac",
    "player_energy",
    "hand_size",
    "draw_size",
    "discard_size",
    "exhaust_size",
    "enemy_count",
    "enemy_hp_frac",
    "enemy_max_hp_total_log",
]


def _featurize_for_mc(state: dict[str, Any]) -> torch.Tensor:
    import math
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") or {}
    enemies = battle.get("enemies") or []
    p_hp = float(player.get("hp", player.get("current_hp", 0)) or 0)
    p_max = max(1.0, float(player.get("max_hp", 1) or 1))
    p_block = float(player.get("block", 0) or 0)
    p_energy = float(player.get("energy", 0) or 0)
    rnd = float(battle.get("round_number", battle.get("round", 0)) or 0)
    hand = battle.get("hand") or []
    draw = battle.get("draw_pile") or []
    disc = battle.get("discard_pile") or []
    exh = battle.get("exhaust_pile") or []
    enemy_count = sum(1 for e in enemies if isinstance(e, dict) and float(e.get("hp", 0) or 0) > 0)
    enemy_hp_total = sum(float(e.get("hp", 0) or 0) for e in enemies if isinstance(e, dict))
    enemy_max_total = max(1.0, sum(float(e.get("max_hp", 1) or 1) for e in enemies if isinstance(e, dict)))
    feats = [
        rnd / 10.0,
        p_hp / p_max,
        p_block / p_max,
        p_energy / 5.0,
        len(hand) / 10.0,
        len(draw) / 30.0,
        len(disc) / 30.0,
        len(exh) / 30.0,
        enemy_count / 5.0,
        enemy_hp_total / enemy_max_total,
        math.log1p(enemy_max_total) / 10.0,
    ]
    return torch.tensor(feats, dtype=torch.float32)


def load_mc_value_head(path: str | None) -> _MCValueHead | None:
    if not path:
        return None
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        return None
    payload = torch.load(p, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict")
    hidden_dim = int(payload.get("hidden_dim", 64))
    feature_keys = payload.get("feature_keys", _MC_FEATURE_KEYS)
    head = _MCValueHead(input_dim=len(feature_keys), hidden=hidden_dim)
    if state_dict is not None:
        head.load_state_dict(state_dict, strict=False)
    head.eval()
    return head


def _enemy_power_stack(enemy: dict[str, Any], power_id_fragment: str) -> float:
    """Return the total stack count of any power whose id contains
    `power_id_fragment` (case-insensitive). Returns 0.0 if none present.

    Mirrors rl_encoder_v2._get_enemy_power but duplicated here to keep
    turn_solver_planner free of encoder-side imports.
    """
    powers = enemy.get("status") or enemy.get("powers") or enemy.get("buffs") or []
    if not isinstance(powers, list):
        return 0.0
    frag = power_id_fragment.lower()
    total = 0.0
    for p in powers:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or p.get("power_id", "")).lower()
        if frag in pid:
            total += float(p.get("amount", p.get("stacks", p.get("count", 0))) or 0)
    return total


# Powers that cap incoming damage to a small value per hit. Reading these
# is critical because the AI's combat search otherwise treats a
# "Bash for 32" line as worth 32 damage when the actual damage that lands
# is 1 (or the cap value). Without this signal, search degenerates into
# big-single-hit lines that are silently nullified.
#
# Mapping: power_id_fragment -> (typical opening stack, narrative)
_DAMAGE_CAP_POWERS = {
    "slippery":     (9.0, "Vantom opener: 9-stack shield, decrements per hit, cap=1"),
    "intangible":   (3.0, "STS canonical: cap=1 (or 5 with TheBoot relic), ticks per turn"),
    "hardtokill":   (5.0, "cap=stack count, doesn't auto-decrement"),
}


def _enemy_damage_cap_load(enemy: dict[str, Any]) -> float:
    """Return a [0, 1]+ weight representing how shielded this enemy is.

    Each known damage-cap power contributes its remaining stack count
    normalized by the typical opening stack. The result is the sum of
    contributions, capped at 2.0 so multiple stacked shields cannot
    explode the leaf score.
    """
    load = 0.0
    for frag, (norm, _) in _DAMAGE_CAP_POWERS.items():
        stack = _enemy_power_stack(enemy, frag)
        if stack > 0:
            load += min(1.0, stack / max(1.0, norm))
    return min(2.0, load)


def _is_minion(enemy: dict[str, Any]) -> bool:
    """True if this enemy carries MinionPower (kin follower marker)."""
    return _enemy_power_stack(enemy, "minion") > 0


def _heuristic_state_value(
    state: dict[str, Any],
    *,
    enemy_hp_weight: float = 1.0,
    use_absolute_hp_scaling: bool = False,
) -> float:
    """Hand-crafted power-aware leaf evaluator (Stage 5 production heuristic).

    Merge note 2026-04-08 (wizardly): this is the P0 power-aware heuristic
    layered on top of QG's helper API. The P0 / QG split:

      - QG contributed: `_enemy_power_stack`, `_is_minion`,
        `_incoming_damage_from_enemy` helpers (used below) plus the
        `_DAMAGE_CAP_POWERS` table (kept as auxiliary documentation, not
        consumed here directly because P0 already itemizes its own per-power
        penalty terms).
      - P0 contributed: the per-power components (slippery progress bonus,
        primary/minion HP split, metallicize/barricade/intangible/ritual/
        angry penalties), the absolute-HP scaling switch, and the score
        formula. P0's Stage 5 production champion uses
        `use_absolute_hp_scaling=True`.

    New scoring components (active in both abs-HP and fraction modes):
      - `slippery_progress_bonus`: each 1-HP hit delivered to a Slippery
        enemy is *progress* (one layer consumed) and worth more than the
        raw 1 HP damage suggests. Modeled as `0.10 * layers_cleared`.
      - `minion_damage_discount`: damage dealt to minion enemies (those
        with MinionPower) is heavily discounted because killing the leader
        cascades to them for free.
      - `metallicize_penalty`: enemy gains block_per_turn * N, so attacks
        effectively do less damage than shown — raise e_hp threshold.
      - `barricade_penalty`: enemy keeps block across turns (block scaling).
      - `intangible_penalty`: all damage capped to 1 while active.
      - `ritual_growth_penalty`: strength growth each turn end.
      - `angry_growth_penalty`: strength growth when damaged.

    These are heuristic signals (not exact physics), tuned to prevent the
    expert+rerank from selecting strictly-wrong lines on ceremonial_beast,
    the_kin, vantom in Act 1.
    """
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") or state.get("player") or {}
    p_hp = float(player.get("hp", player.get("current_hp", 0)) or 0)
    p_max = max(1.0, float(player.get("max_hp", 1) or 1))
    p_block = float(player.get("block", 0) or 0)

    enemies = battle.get("enemies") or []
    if not isinstance(enemies, list):
        enemies = []

    # ============ Per-enemy accumulation with power-aware discounts ============
    # Split enemies into minion vs primary so damage dealt to minions doesn't
    # equal damage dealt to the leader in the leader-cascade case (the_kin).
    primary_e_hp = 0.0
    primary_e_max = 0.0
    minion_e_hp = 0.0
    minion_e_max = 0.0
    alive_count = 0
    incoming = 0.0

    # Per-enemy power accumulators for bonus/penalty terms
    max_slippery_any = 0.0
    any_slippery_present = False
    total_metallicize = 0.0
    any_barricade = False
    any_intangible = False
    total_ritual = 0.0
    total_angry = 0.0
    has_minions = False
    has_primary_alive = False

    for e in enemies:
        if not isinstance(e, dict):
            continue
        ehp = float(e.get("hp", 0) or 0)
        emax = float(e.get("max_hp", 1) or 1)
        is_min = _is_minion(e)
        if is_min:
            minion_e_hp += max(0.0, ehp)
            minion_e_max += max(1.0, emax)
            has_minions = True
        else:
            primary_e_hp += max(0.0, ehp)
            primary_e_max += max(1.0, emax)
            if ehp > 0:
                has_primary_alive = True

        if ehp > 0:
            alive_count += 1
            # QG-supplied helper: prefers `total_damage` from the new C# DTO
            # (post-2026-04-08 binary protocol) and falls back to per-hit
            # `damage * repeats` for legacy formats.
            incoming += _incoming_damage_from_enemy(e)

            # Read powers from enemy.status / powers / buffs (QG helper).
            slip = _enemy_power_stack(e, "slippery")
            if slip > 0:
                any_slippery_present = True
                max_slippery_any = max(max_slippery_any, slip)
            total_metallicize += _enemy_power_stack(e, "metallicize")
            if _enemy_power_stack(e, "barricade") > 0:
                any_barricade = True
            if _enemy_power_stack(e, "intangible") > 0:
                any_intangible = True
            total_ritual += _enemy_power_stack(e, "ritual")
            total_angry += _enemy_power_stack(e, "angry")

    # Aggregate totals
    e_hp_total = primary_e_hp + minion_e_hp
    e_max_total = primary_e_max + minion_e_max

    p_hp_frac = p_hp / p_max
    incoming_through_block = max(0.0, incoming - p_block)
    incoming_frac = incoming_through_block / p_max

    # ============ Damage-dealt accounting with minion discount ============
    e_max_dealt_raw = e_max_total - e_hp_total
    primary_dealt = primary_e_max - primary_e_hp
    minion_dealt = minion_e_max - minion_e_hp

    # When leader is alive and minions exist, damage to minions is worth
    # only ~25% of damage to primary (primary death cascades to minions).
    # Once primary is dead, minion damage fully counts (they're alive
    # without support and still dangerous).
    if has_minions and has_primary_alive:
        effective_dealt = primary_dealt + 0.25 * minion_dealt
    else:
        effective_dealt = e_max_dealt_raw

    # ============ Slippery progress bonus ============
    # Slippery caps damage to 1/hit AND each hit removes 1 layer. So during
    # slippery phase, the raw damage numbers severely underreport progress.
    # Reward = monotonic in layers cleared (Vantom opens with 9 layers).
    SLIPPERY_INIT = 9.0
    slippery_bonus = 0.0
    if any_slippery_present:
        layers_cleared = max(0.0, SLIPPERY_INIT - max_slippery_any)
        slippery_bonus = 0.10 * layers_cleared  # 0.0 → 0.9 as we clear 9 layers
    elif primary_e_hp < primary_e_max:
        # Slippery was fully cleared AND we've started dealing real damage.
        slippery_bonus = 0.10 * SLIPPERY_INIT  # 0.90 — strictly better than mid-clear

    # ============ Penalties for enemy-side defensive/growth powers ============
    metallicize_penalty = 0.3 * (total_metallicize / 10.0) if total_metallicize > 0 else 0.0
    barricade_penalty = 0.2 if any_barricade else 0.0
    intangible_penalty = 0.5 if any_intangible else 0.0  # strong: damage capped to 1
    ritual_penalty = 0.15 * (total_ritual / 5.0) if total_ritual > 0 else 0.0
    angry_penalty = 0.10 * (total_angry / 5.0) if total_angry > 0 else 0.0

    if use_absolute_hp_scaling:
        # Production path — abs HP scaling (Stage 5 default).
        score = (
            + 0.5 * (p_hp / 100.0)              # alive matters (lower weight)
            + 0.8 * (effective_dealt / 100.0)   # damage dealt (minion-discounted)
            + 0.10 * (p_block / 100.0)          # block is mild positive
            - 0.80 * (incoming_through_block / 100.0)
            + 0.20 * (1.0 - min(1.0, alive_count / 3.0))
            + slippery_bonus                    # slippery progress ★
            - metallicize_penalty
            - barricade_penalty
            - intangible_penalty
            - ritual_penalty
            - angry_penalty
        )
    else:
        # Composite score in roughly [-2, +2] (fraction-scaled)
        effective_dealt_frac = effective_dealt / max(1.0, e_max_total)
        score = (
            + 1.0 * p_hp_frac                                     # alive matters
            - enemy_hp_weight * (1.0 - effective_dealt_frac)      # damage matters (minion-discounted)
            + 0.10 * (p_block / p_max)                            # block is mild positive
            - 0.80 * incoming_frac                                # taking damage next turn is bad
            + 0.20 * (1.0 - min(1.0, alive_count / 3.0))
            + slippery_bonus
            - metallicize_penalty
            - barricade_penalty
            - intangible_penalty
            - ritual_penalty
            - angry_penalty
        )
    # Soft clamp to wider range (power bonus/penalty can push past ±1.5)
    return max(-2.5, min(2.5, score))


class _TunedCombatTurnSolver(CombatTurnSolver):
    """Subclass that exposes a configurable hp_loss_weight at the leaf AND
    optionally swaps in a heuristic state-value evaluator that replaces the
    PPO value head signal.

    The default `CombatTurnSolver` hardcodes `total = baseline - 0.15 *
    expected_hp_loss`, which over-weights survival AND inherits the flat
    PPO value head signal. This subclass lets callers:
      - tune hp_loss_weight
      - blend NN value with a heuristic state value (Tier 1 leaf eval)
        via `heuristic_blend_alpha` (0.0 = NN only, 1.0 = heuristic only)
    """

    def __init__(
        self, *args,
        hp_loss_weight: float = 0.05,
        heuristic_blend_alpha: float = 0.0,
        mc_value_head: _MCValueHead | None = None,
        mc_value_blend: float = 0.0,
        boss_leaf_evaluator: Any | None = None,
        # P0 (Stage 5) heuristic kwargs — control the depth/scaling of
        # `_heuristic_state_value` so the production heuristic blend can use
        # absolute-HP scaling and a tunable enemy-HP weight.
        heuristic_enemy_hp_weight: float = 1.0,
        heuristic_use_absolute_hp: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hp_loss_weight = float(hp_loss_weight)
        self.heuristic_blend_alpha = float(heuristic_blend_alpha)
        self.mc_value_head = mc_value_head
        self.mc_value_blend = float(mc_value_blend)
        self.boss_leaf_evaluator = boss_leaf_evaluator
        self.heuristic_enemy_hp_weight = float(heuristic_enemy_hp_weight)
        self.heuristic_use_absolute_hp = bool(heuristic_use_absolute_hp)

    def _evaluate_leaf(self, state: dict[str, Any]) -> SolverLineResult:
        if _is_terminal_combat_leaf(state):
            state_type = str(state.get("state_type") or "").strip().lower()
            if state_type == "game_over":
                baseline_value = -1.0
                expected_hp_loss = 999.0
            else:
                baseline_value = 1.0
                expected_hp_loss = 0.0
            total = baseline_value - self.hp_loss_weight * expected_hp_loss
            self.stats.evaluated_leaves += 1
            return SolverLineResult(
                total_value=total,
                action_line=[],
                component_totals=_zero_components(),
                leaf_state=state,
                leaf_baseline_value=baseline_value,
                leaf_expected_hp_loss=expected_hp_loss,
                leaf_predicted_hp_loss_ratio=0.0,
                leaf_predicted_total_hp_loss=0.0,
            )

        legal_actions = [a for a in state.get("legal_actions") or [] if isinstance(a, dict)]
        baseline = self.baseline_policy.score(state, legal_actions)
        nn_value = float(baseline["value"])
        state_type = str(state.get("state_type") or "").strip().lower()
        immediate_expected_hp_loss = float(
            combine_leaf_breakdown(
                state,
                baseline_value=nn_value,
                total_action_components=_zero_components(),
            )["immediate_expected_hp_loss_this_enemy_turn"]
        )

        # Compute mixed value: nn + heuristic + mc_value
        mixed_value = nn_value
        leaf_hp_loss_ratio = 0.0
        leaf_predicted_total_hp_loss = 0.0
        leaf_runtime_used = False
        if self.boss_leaf_evaluator is not None:
            try:
                leaf_outputs = self.boss_leaf_evaluator.predict_state(state)
                mixed_value = float(leaf_outputs.get("leaf_value", mixed_value))
                leaf_hp_loss_ratio = float(leaf_outputs.get("hp_loss_ratio", 0.0) or 0.0)
                battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
                player = battle.get("player") or state.get("player") or {}
                max_hp = max(1.0, float(player.get("max_hp", 1) or 1))
                leaf_predicted_total_hp_loss = leaf_hp_loss_ratio * max_hp
                leaf_runtime_used = True
            except Exception:
                leaf_outputs = None
        else:
            leaf_outputs = None

        effective_heuristic_blend = self.heuristic_blend_alpha
        # Boss setup turns often have zero immediate danger but large
        # long-horizon pessimism from the learned leaf head. In that case,
        # enforce a stronger heuristic floor so opener setup lines like BASH
        # are not buried by raw leaf calibration noise.
        if (
            leaf_runtime_used
            and state_type == "boss"
            and immediate_expected_hp_loss <= 1e-6
        ):
            effective_heuristic_blend = max(
                effective_heuristic_blend,
                _BOSS_SETUP_HEURISTIC_FLOOR,
            )
        if effective_heuristic_blend > 0.0:
            h_value = _heuristic_state_value(
                state,
                enemy_hp_weight=self.heuristic_enemy_hp_weight,
                use_absolute_hp_scaling=self.heuristic_use_absolute_hp,
            )
            mixed_value = (
                (1.0 - effective_heuristic_blend) * mixed_value
                + effective_heuristic_blend * h_value
            )
        if self.mc_value_blend > 0.0 and self.mc_value_head is not None:
            try:
                with torch.no_grad():
                    mc_input = _featurize_for_mc(state).unsqueeze(0)
                    mc_v = float(self.mc_value_head(mc_input).item())
                mixed_value = (
                    (1.0 - self.mc_value_blend) * mixed_value
                    + self.mc_value_blend * mc_v
                )
            except Exception:
                pass

        baseline_value = mixed_value
        expected_hp_loss = immediate_expected_hp_loss
        total = baseline_value - self.hp_loss_weight * expected_hp_loss
        self.stats.evaluated_leaves += 1
        return SolverLineResult(
            total_value=total,
            action_line=[],
            component_totals=_zero_components(),
            leaf_state=state,
            leaf_baseline_value=baseline_value,
            leaf_expected_hp_loss=expected_hp_loss,
            leaf_predicted_hp_loss_ratio=leaf_hp_loss_ratio,
            leaf_predicted_total_hp_loss=leaf_predicted_total_hp_loss,
        )


class TurnSolverPlanner:
    """Drives CombatTurnSolver from the `_select_action_nn` hook."""

    def __init__(
        self,
        baseline_policy: BaselineCombatPolicy,
        *,
        mode: str = "boss",
        max_player_actions: int = 12,
        hp_loss_weight: float = 0.05,
        heuristic_blend_alpha: float = 0.0,
        mc_value_head: _MCValueHead | None = None,
        mc_value_blend: float = 0.0,
        boss_leaf_evaluator: Any | None = None,
        # P0 (Stage 5) heuristic kwargs — control depth/scaling of the
        # power-aware heuristic.
        heuristic_enemy_hp_weight: float = 1.0,
        heuristic_use_absolute_hp: bool = False,
        # P0 (P1-2) boss-token whitelist — only fire on these boss tokens.
        # Empty/None means fire on every boss state.
        boss_token_whitelist: list[str] | None = None,
    ):
        self.baseline_policy = baseline_policy
        self._mode = mode
        self.max_player_actions = max_player_actions
        self.hp_loss_weight = hp_loss_weight
        self.heuristic_blend_alpha = heuristic_blend_alpha
        self.mc_value_head = mc_value_head
        self.mc_value_blend = mc_value_blend
        self.boss_leaf_evaluator = boss_leaf_evaluator
        self.heuristic_enemy_hp_weight = float(heuristic_enemy_hp_weight)
        self.heuristic_use_absolute_hp = bool(heuristic_use_absolute_hp)
        # P0 P1-2: optional whitelist of boss tokens. When set, the planner
        # only fires on boss states whose enemy.id matches one of these tokens.
        self._boss_token_whitelist: set[str] = (
            {str(t).upper() for t in boss_token_whitelist} if boss_token_whitelist else set()
        )
        self._cache: _CachedTurnPlan | None = None
        # Telemetry
        self.calls = 0
        self.solver_calls = 0
        self.cache_hits = 0
        self.unsupported = 0
        self.fallbacks = 0
        self.errors = 0
        self.skipped_by_token_filter = 0
        self.last_attempt: dict[str, Any] = {}
        self.last_decision: dict[str, Any] = {}
        # P0 Phase 4 stage 2 trace recording (default off, set by trace generator)
        self.record_trace: bool = False
        self.trace_records: list[dict[str, Any]] = []

    def _build_solver(self, env: _PipeEnvAdapter, *, use_leaf_evaluator: bool) -> _TunedCombatTurnSolver:
        return _TunedCombatTurnSolver(
            env=env,
            baseline_policy=self.baseline_policy,
            max_player_actions=self.max_player_actions,
            hp_loss_weight=self.hp_loss_weight,
            heuristic_blend_alpha=(self.heuristic_blend_alpha if use_leaf_evaluator else 1.0),
            mc_value_head=(self.mc_value_head if use_leaf_evaluator else None),
            mc_value_blend=(self.mc_value_blend if use_leaf_evaluator else 0.0),
            boss_leaf_evaluator=(self.boss_leaf_evaluator if use_leaf_evaluator else None),
            heuristic_enemy_hp_weight=self.heuristic_enemy_hp_weight,
            heuristic_use_absolute_hp=self.heuristic_use_absolute_hp,
        )

    def _state_boss_token(self, state: dict[str, Any]) -> str:
        """Extract boss token from current state, uppercase. Returns '' if not boss state.

        STS2 enemy ids in boss states are simple uppercase tokens like
        'CEREMONIAL_BEAST', 'VANTOM', 'KIN_FOLLOWER'. When the state_type is
        'boss', the first non-empty enemy id IS the boss token.
        """
        if (state.get("state_type") or "").lower() != "boss":
            return ""
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        enemies = battle.get("enemies") or []
        if not isinstance(enemies, list):
            return ""
        for e in enemies:
            if not isinstance(e, dict):
                continue
            tok = (e.get("id") or e.get("name") or "").strip().upper()
            if tok:
                return tok
        return ""

    def _passes_boss_token_filter(self, state: dict[str, Any]) -> bool:
        """Return True if this state should run the planner under the whitelist.

        - Empty whitelist → always True (no filter active)
        - Non-boss state → always True (filter only applies to boss states)
        - Boss state with token in whitelist → True
        - Boss state with token NOT in whitelist → False
        """
        if not self._boss_token_whitelist:
            return True
        st = (state.get("state_type") or "").lower()
        if st != "boss":
            return True
        tok = self._state_boss_token(state)
        return tok in self._boss_token_whitelist

    def invalidate_cache(self) -> None:
        self._cache = None

    def _try_cache(
        self,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
    ) -> tuple[int, str] | None:
        if self._cache is None:
            self.last_attempt.update({"cache_state": "none"})
            return None
        if self._cache.is_exhausted():
            self.last_attempt.update({"cache_state": "exhausted"})
            self._cache = None
            return None
        sig = _state_signature(state)
        if sig != self._cache.state_hash:
            self.last_attempt.update(
                {
                    "cache_state": "state_hash_mismatch",
                    "cache_expected_state_hash": self._cache.state_hash,
                    "cache_actual_state_hash": sig,
                }
            )
            self._cache = None
            return None
        next_action = self._cache.actions[self._cache.next_idx]
        match_idx, match_kind = _find_cache_match(next_action, legal)
        if match_idx is not None:
            self._cache.next_idx += 1
            self.cache_hits += 1
            matched_action = legal[match_idx] if match_idx < len(legal) else {}
            self.last_attempt = {
                "planner": "turn_solver",
                "selected_source": "turn_solver_cache",
                "cache_hit": True,
                "cache_match": match_kind,
                "next_action": sanitize_action(next_action) or {},
                "matched_action": sanitize_action(matched_action) or {},
                "remaining_actions": max(0, len(self._cache.actions) - self._cache.next_idx),
            }
            self.last_decision = dict(self.last_attempt)
            return match_idx, "turn_solver_cache"
        # Action no longer legal — invalidate
        self.last_attempt.update(
            {
                "cache_state": "next_action_not_legal",
                "cache_next_action": sanitize_action(next_action) or {},
            }
        )
        self._cache = None
        return None

    def select_action(
        self,
        pipe_getter: Any,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
    ) -> tuple[int, str] | None:
        self.calls += 1
        self.last_attempt = {
            "planner": "turn_solver",
            "selected_source": None,
            "cache_hit": False,
            "status": "started",
        }
        # P0 P1-2: boss token filter — skip planner if state's boss token isn't
        # in the whitelist. Returning None falls through to NN argmax.
        if not self._passes_boss_token_filter(state):
            self.skipped_by_token_filter += 1
            self.last_attempt.update({"status": "skipped_by_token_filter"})
            return None
        # P0 Phase 4 stage 2 trace hook: when `record_trace` is True, capture
        # the solution and state snapshot. The trace generator reads
        # `trace_records` after each game. While recording, skip cache so every
        # decision triggers a fresh solve.
        record_trace_now = bool(getattr(self, "record_trace", False))
        # 1) Try cached plan first (unless recording trace)
        if not record_trace_now:
            cached = self._try_cache(state, legal)
            if cached is not None:
                return cached

        # 2) Run a fresh solve
        # pipe_getter is a callable returning the raw pipe object (NOT a client)
        try:
            raw_pipe = pipe_getter()
        except Exception as e:
            logger.debug("pipe_getter failed: %s", e)
            self.errors += 1
            self.last_attempt.update({"status": "pipe_getter_failed", "error": str(e)})
            return None
        env = _PipeEnvAdapter(raw_pipe)
        try:
            root_state_id = env.save_state()
        except Exception as e:
            logger.debug("turn solver save_state failed: %s", e)
            self.errors += 1
            self.last_attempt.update({"status": "save_state_failed", "error": str(e)})
            return None
        if not root_state_id:
            self.errors += 1
            self.last_attempt.update({"status": "empty_root_state_id"})
            return None

        solver = self._build_solver(env, use_leaf_evaluator=True)
        heuristic_solver: _TunedCombatTurnSolver | None = None
        solution: CombatTurnSolution | None = None
        continuation_source = "candidate_leaf"
        heuristic_reference_action: dict[str, Any] | None = None
        try:
            solution = solver.solve(state, root_state_id=root_state_id)
            self.solver_calls += 1
            if (
                solution is not None
                and solution.supported
                and solution.best_full_turn_line
                and self.boss_leaf_evaluator is not None
                and str(state.get("state_type") or "").strip().lower() == "boss"
            ):
                heuristic_solver = self._build_solver(env, use_leaf_evaluator=False)
                heuristic_root_state_id = env.save_state()
                heuristic_solution = heuristic_solver.solve(state, root_state_id=heuristic_root_state_id)
                self.solver_calls += 1
                if (
                    heuristic_solution is not None
                    and heuristic_solution.supported
                    and heuristic_solution.best_full_turn_line
                ):
                    heuristic_reference_action = sanitize_action(heuristic_solution.best_full_turn_line[0]) or {}
                    if _same_first_action(solution.best_full_turn_line, heuristic_solution.best_full_turn_line):
                        solution.best_full_turn_line = list(heuristic_solution.best_full_turn_line)
                        continuation_source = "heuristic_same_first_action"
        except Exception as e:
            logger.debug("turn solver solve failed: %s", e)
            self.errors += 1
            self.last_attempt.update({"status": "solve_failed", "error": str(e)})
            return None
        finally:
            try:
                solver.cleanup()
            except Exception:
                pass
            if heuristic_solver is not None:
                try:
                    heuristic_solver.cleanup()
                except Exception:
                    pass
            try:
                env.delete_state(root_state_id)
            except Exception:
                pass

        if solution is None or not solution.supported:
            self.unsupported += 1
            self.last_attempt.update(
                {
                    "status": "unsupported",
                    "unsupported_reason": getattr(solution, "unsupported_reason", "") if solution is not None else "none",
                    "unsupported_details": getattr(solution, "unsupported_details", {}) if solution is not None else {},
                }
            )
            return None
        if not solution.best_full_turn_line:
            self.fallbacks += 1
            self.last_attempt.update({"status": "empty_best_line"})
            return None

        # P0 Phase 4 stage 2 trace hook: append (state_snapshot, solution)
        # before caching the line. The caller is responsible for serializing
        # the state immediately if needed.
        if record_trace_now:
            try:
                self.trace_records.append({
                    "state": state,
                    "solution": solution,
                    "boss_token": self._state_boss_token(state),
                })
            except Exception as e:
                logger.debug("trace record append failed: %s", e)

        # 3) Cache the line and return the first action
        self._cache = _CachedTurnPlan(
            actions=list(solution.best_full_turn_line),
            next_idx=0,
            state_hash=_state_signature(state),
        )
        ranked = sorted(
            [item for item in (solution.per_action_score or []) if item.get("supported", True)],
            key=lambda item: -float(item.get("total_score", item.get("score", float("-inf"))) or float("-inf")),
        )
        self.last_attempt = {
            "planner": "turn_solver",
            "selected_source": "turn_solver",
            "cache_hit": False,
            "status": "selected",
            "root_value": float(solution.root_value),
            "continuation_source": continuation_source,
            "heuristic_reference_action": heuristic_reference_action,
            "selected_action": sanitize_action(solution.best_full_turn_line[0]) if solution.best_full_turn_line else None,
            "top_candidates": [
                {
                    "action": sanitize_action(item.get("action") or {}),
                    "total_score": float(item.get("total_score", item.get("score", 0.0)) or 0.0),
                    "component_score": float(item.get("component_score", 0.0) or 0.0),
                    "leaf_score": float(item.get("leaf_score", 0.0) or 0.0),
                    "expected_hp_loss": float(item.get("expected_hp_loss", 0.0) or 0.0),
                    "predicted_hp_loss_ratio": float(item.get("predicted_hp_loss_ratio", 0.0) or 0.0),
                    "predicted_total_hp_loss": float(item.get("predicted_total_hp_loss", 0.0) or 0.0),
                    "unsupported_reason": str(item.get("unsupported_reason") or ""),
                }
                for item in ranked[: min(5, len(ranked))]
            ],
        }
        self.last_decision = dict(self.last_attempt)
        first = solution.best_full_turn_line[0]
        for i, la in enumerate(legal):
            if _action_matches_legal(first, la):
                self._cache.next_idx = 1
                return i, "turn_solver"

        # First action not in legal list — bail out
        self._cache = None
        self.fallbacks += 1
        self.last_attempt.update({"status": "selected_action_not_legal"})
        return None


def build_turn_solver_planner(
    combat_net: CombatPolicyValueNetwork,
    vocab: Vocab,
    device: torch.device,
    *,
    mode: str = "boss",
    max_player_actions: int = 12,
    hp_loss_weight: float = 0.05,
    heuristic_blend_alpha: float = 0.0,
    mc_value_head_path: str | None = None,
    mc_value_blend: float = 0.0,
    boss_leaf_evaluator_path: str | None = None,
    # P0 (Stage 5) heuristic kwargs
    heuristic_enemy_hp_weight: float = 1.0,
    heuristic_use_absolute_hp: bool = False,
    # P0 (P1-2) boss-token whitelist
    boss_token_whitelist: list[str] | None = None,
) -> TurnSolverPlanner:
    """Wrap an existing `combat_net` into a `BaselineCombatPolicy` and build
    the planner. Reuses the in-memory combat NN — no separate checkpoint load.
    Optionally loads a Monte Carlo value head for leaf evaluation, or a Boss
    Leaf Evaluator runtime (QG Phase 2 contract).

    `boss_token_whitelist` (P1-2): if provided, the planner only fires on boss
    states whose enemy.id matches one of these tokens (case-insensitive).
    Other boss states (and all non-boss states) fall through to NN argmax.

    `heuristic_use_absolute_hp` / `heuristic_enemy_hp_weight`: control the
    P0 power-aware heuristic (`_heuristic_state_value`). Stage 5 production
    uses `heuristic_use_absolute_hp=True`.
    """
    baseline = BaselineCombatPolicy(network=combat_net, vocab=vocab, device=device)
    mc_head = load_mc_value_head(mc_value_head_path) if mc_value_blend > 0 else None
    boss_leaf_evaluator = load_boss_leaf_evaluator_runtime(boss_leaf_evaluator_path, device=device)
    if mc_value_blend > 0 and mc_head is None:
        logger.warning(
            "MC value blend > 0 requested but value head failed to load from %s",
            mc_value_head_path,
        )
    if boss_leaf_evaluator_path and boss_leaf_evaluator is None:
        logger.warning(
            "Boss leaf evaluator failed to load from %s",
            boss_leaf_evaluator_path,
        )
    return TurnSolverPlanner(
        baseline_policy=baseline,
        mode=mode,
        max_player_actions=max_player_actions,
        hp_loss_weight=hp_loss_weight,
        heuristic_blend_alpha=heuristic_blend_alpha,
        mc_value_head=mc_head,
        mc_value_blend=mc_value_blend,
        boss_leaf_evaluator=boss_leaf_evaluator,
        heuristic_enemy_hp_weight=heuristic_enemy_hp_weight,
        heuristic_use_absolute_hp=heuristic_use_absolute_hp,
        boss_token_whitelist=boss_token_whitelist,
    )
