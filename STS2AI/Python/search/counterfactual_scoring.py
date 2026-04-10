"""Screen-local counterfactual scoring for non-combat decisions.

Fused from original implementation + codex/reward branch best practices:
  - Boss archetype-aware problem weights (from codex)
  - Full screen coverage: card_reward, shop, rest_site, relic, map, event (from codex)
  - state_utility with deck/gold override for hypothetical evaluation (from codex)
  - Dispersion guard + z-score normalization (from original)
  - Conservative clip=0.12 (from codex, per gpt-design guidance)
  - Modular file structure with CLI flag control (from original)
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import copy
import logging
from math import exp
from typing import Any

import numpy as np

from rl_reward_shaping import (
    _extract_player,
    _extract_progress,
    _lower,
    _safe_int,
    canonicalize_boss_token,
    compute_problem_vector,
    extract_next_boss_token,
    problem_score,
    survival_margin,
    economy_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (conservative clips per gpt-design)
# ---------------------------------------------------------------------------

COUNTERFACTUAL_CLIP = 0.25  # Was 0.12, too conservative — only top-1 got positive reward
SOFT_BASELINE_TEMP = 1.5    # Was 0.70, raised to lower baseline so more actions get positive reward
TEACHER_TEMP = 0.35         # Was 0.45, sharper teacher distribution

# Screens with local scoring support
LOCAL_SCORER_SCREENS = {"card_reward", "shop", "rest_site", "relic_select", "treasure", "map", "event"}

# Card keyword sets for scoring (from codex)
FRONTLOAD_KEYWORDS = {
    "pommel_strike", "twin_strike", "wild_strike", "perfected_strike",
    "carnage", "bludgeon", "hemokinesis", "predator", "glass_knife",
    "riddle_with_holes", "ball_lightning", "sunder", "wallop",
    "consecrate", "bowling_bash", "carve_reality", "uppercut", "bash",
}
BLOCK_KEYWORDS = {
    "defend", "shrug_it_off", "flame_barrier", "ghostly_armor", "impervious",
    "backflip", "blur", "leg_sweep", "dash", "glacier", "charge_battery",
    "stack", "third_eye", "protect", "deceive_reality", "safety", "halt",
    "power_through",
}
DRAW_KEYWORDS = {
    "pommel_strike", "shrug_it_off", "burning_pact", "battle_trance",
    "backflip", "acrobatics", "prepared", "skim", "coolheaded",
    "compile_driver", "sanctity", "empty_mind", "rushdown",
}
ENERGY_KEYWORDS = {
    "offering", "adrenaline", "tactician", "double_energy", "turbo",
    "aggregate", "fission", "deus_ex_machina", "collect", "seeing_red",
}
SCALING_KEYWORDS = {
    "inflame", "spot_weakness", "demon_form", "limit_break", "corruption",
    "feel_no_pain", "noxious_fumes", "footwork", "accuracy", "after_image",
    "wraith_form", "catalyst", "electrodynamics", "defragment",
    "bias_cognition", "echo_form", "rushdown", "mental_fortress",
    "talk_to_the_hand", "devotion", "deva_form", "lesson_learned",
}
GOOD_UPGRADE_KEYWORDS = {
    "bash", "pommel_strike", "shrug_it_off", "carnage", "armaments",
    "leg_sweep", "backflip", "footwork", "noxious_fumes", "defragment",
    "electrodynamics", "glacier", "talk_to_the_hand", "rushdown", "consecrate",
}
REMOVE_BAD_KEYWORDS = {"strike", "defend", "wound", "dazed", "slimed", "burn", "regret", "curse"}


# ---------------------------------------------------------------------------
# Boss Archetype Weights (from codex — key domain knowledge)
# ---------------------------------------------------------------------------

ACT_BOSS_WEIGHTS: dict[int, dict[str, dict[str, float]]] = {
    1: {
        "unknown": {
            "frontload": 0.22, "aoe": 0.08, "block_consistency": 0.14, "draw": 0.08,
            "energy": 0.05, "scaling": 0.06, "consistency": 0.10, "elite_readiness": 0.17, "boss_answer": 0.10,
        },
        "guardian": {
            "frontload": 0.14, "aoe": 0.05, "block_consistency": 0.22, "draw": 0.08,
            "energy": 0.05, "scaling": 0.06, "consistency": 0.12, "elite_readiness": 0.14, "boss_answer": 0.14,
        },
        "slime": {
            "frontload": 0.22, "aoe": 0.16, "block_consistency": 0.10, "draw": 0.06,
            "energy": 0.05, "scaling": 0.04, "consistency": 0.08, "elite_readiness": 0.16, "boss_answer": 0.13,
        },
        "hexa": {
            "frontload": 0.18, "aoe": 0.04, "block_consistency": 0.12, "draw": 0.12,
            "energy": 0.08, "scaling": 0.11, "consistency": 0.11, "elite_readiness": 0.12, "boss_answer": 0.12,
        },
    },
    2: {
        "unknown": {
            "frontload": 0.10, "aoe": 0.18, "block_consistency": 0.18, "draw": 0.12,
            "energy": 0.10, "scaling": 0.10, "consistency": 0.10, "elite_readiness": 0.07, "boss_answer": 0.05,
        },
        "champ": {
            "frontload": 0.08, "aoe": 0.12, "block_consistency": 0.20, "draw": 0.12,
            "energy": 0.10, "scaling": 0.14, "consistency": 0.11, "elite_readiness": 0.06, "boss_answer": 0.07,
        },
        "collector": {
            "frontload": 0.10, "aoe": 0.22, "block_consistency": 0.16, "draw": 0.11,
            "energy": 0.09, "scaling": 0.08, "consistency": 0.10, "elite_readiness": 0.08, "boss_answer": 0.06,
        },
        "automaton": {
            "frontload": 0.09, "aoe": 0.16, "block_consistency": 0.17, "draw": 0.13,
            "energy": 0.11, "scaling": 0.11, "consistency": 0.11, "elite_readiness": 0.06, "boss_answer": 0.06,
        },
    },
    3: {
        "unknown": {
            "frontload": 0.06, "aoe": 0.07, "block_consistency": 0.16, "draw": 0.15,
            "energy": 0.14, "scaling": 0.18, "consistency": 0.12, "elite_readiness": 0.04, "boss_answer": 0.08,
        },
        "time_eater": {
            "frontload": 0.05, "aoe": 0.05, "block_consistency": 0.18, "draw": 0.10,
            "energy": 0.12, "scaling": 0.20, "consistency": 0.16, "elite_readiness": 0.04, "boss_answer": 0.10,
        },
        "awakened": {
            "frontload": 0.06, "aoe": 0.08, "block_consistency": 0.15, "draw": 0.14,
            "energy": 0.14, "scaling": 0.17, "consistency": 0.13, "elite_readiness": 0.04, "boss_answer": 0.09,
        },
        "donu_deca": {
            "frontload": 0.06, "aoe": 0.08, "block_consistency": 0.15, "draw": 0.15,
            "energy": 0.14, "scaling": 0.18, "consistency": 0.12, "elite_readiness": 0.04, "boss_answer": 0.08,
        },
    },
}


def infer_boss_archetype(state: dict[str, Any]) -> str:
    """Infer boss archetype from run/battle state."""
    return extract_next_boss_token(state)


def _boss_from_name(name: str) -> str:
    return canonicalize_boss_token(name)


# ---------------------------------------------------------------------------
# State utility with deck/gold override (from codex)
# ---------------------------------------------------------------------------

# Screen-specific mix: (problem, survival, economy) — from codex, tuned
_SCREEN_MIX = {
    "map":             (0.68, 0.20, 0.12),
    "card_reward":     (0.72, 0.18, 0.10),
    "shop":            (0.54, 0.12, 0.34),
    "rest_site":       (0.50, 0.42, 0.08),
    "campfire":        (0.50, 0.42, 0.08),
    "event":           (0.54, 0.36, 0.10),
    "relic_select":    (0.60, 0.22, 0.18),
    "treasure":        (0.60, 0.22, 0.18),
    "combat_rewards":  (0.55, 0.25, 0.20),
    "default":         (0.58, 0.28, 0.14),
}


def state_utility(
    state: dict[str, Any],
    deck_override: list[dict] | None = None,
    gold_override: int | None = None,
    screen_override: str | None = None,
) -> float:
    """Compute state utility with optional hypothetical overrides.

    This enables counterfactual evaluation: "what would utility be if we added this card?"
    """
    # Build hypothetical state if overrides provided
    if deck_override is not None or gold_override is not None:
        state = _apply_overrides(state, deck_override, gold_override)

    screen_type = _lower(screen_override or state.get("state_type") or "default")
    p = _extract_progress(state)
    act = min(max(p["act"], 1), 3)
    boss = infer_boss_archetype(state)

    # Problem score with boss-aware weights
    pv = compute_problem_vector(state)
    act_weights = ACT_BOSS_WEIGHTS.get(act, ACT_BOSS_WEIGHTS[3])
    weights = act_weights.get(boss, act_weights["unknown"])
    weight_sum = max(1e-6, sum(weights.values()))
    ps = sum(float(pv[i]) * weights.get(k, 0.0) for i, k in enumerate([
        "frontload", "aoe", "block_consistency", "draw", "energy",
        "scaling", "consistency", "elite_readiness", "boss_answer",
    ])) / weight_sum

    sm = survival_margin(state)
    es = economy_score(state)

    w_p, w_s, w_e = _SCREEN_MIX.get(screen_type, _SCREEN_MIX["default"])
    return max(-1.0, min(1.5, w_p * ps + w_s * sm + w_e * es))


def _apply_overrides(state: dict, deck_override: list | None, gold_override: int | None) -> dict:
    """Create shallow copy of state with deck/gold overrides applied."""
    hypo = copy.copy(state)
    player = _extract_player(state)
    if not player:
        return hypo
    hypo_player = copy.copy(player)
    if deck_override is not None:
        hypo_player["deck"] = deck_override
    if gold_override is not None:
        hypo_player["gold"] = gold_override
    # Place updated player in correct location
    if isinstance(state.get("player"), dict):
        hypo["player"] = hypo_player
    else:
        for key in ("battle", "map", "shop", "rest_site", "event",
                    "rewards", "card_reward", "card_select", "relic_select", "treasure"):
            container = state.get(key)
            if isinstance(container, dict) and isinstance(container.get("player"), dict):
                hypo[key] = copy.copy(container)
                hypo[key]["player"] = hypo_player
                break
    return hypo


# ---------------------------------------------------------------------------
# Core: relative reward + teacher distribution
# ---------------------------------------------------------------------------

def soft_baseline(scores: list[float], temperature: float = SOFT_BASELINE_TEMP) -> float:
    """Softmax-weighted average baseline (less noisy than max)."""
    if not scores:
        return 0.0
    mx = max(scores)
    weights = [exp((s - mx) / max(1e-6, temperature)) for s in scores]
    total = sum(weights)
    if total <= 0:
        return sum(scores) / len(scores)
    return sum(s * w for s, w in zip(scores, weights)) / total


def normalize_scores(scores: list[float], std_floor: float = 0.05) -> tuple[list[float], float]:
    """Z-score normalize within screen. Returns (norm_scores, dispersion)."""
    if len(scores) <= 1:
        return [0.0] * len(scores), 0.0
    s = np.array(scores, dtype=np.float64)
    mean_v = s.mean()
    std_v = max(float(s.std()), std_floor)
    norm = ((s - mean_v) / std_v).tolist()
    return norm, float(s.std())


def counterfactual_reward(
    chosen_idx: int,
    scores: list[float],
    clip_range: float = COUNTERFACTUAL_CLIP,
    min_dispersion: float = 0.005,
) -> tuple[float, list[float] | None]:
    """Compute relative reward and teacher distribution.

    Uses raw scores directly (not z-score normalized) with soft baseline.
    This preserves absolute magnitude differences between good cards and skip.
    """
    if not scores or chosen_idx < 0 or chosen_idx >= len(scores):
        return 0.0, None

    # Dispersion guard on raw scores
    raw_range = max(scores) - min(scores)
    if raw_range < min_dispersion:
        return 0.0, None

    # Use raw scores with soft baseline (not z-score normalized)
    # This way a card with score 0.10 vs skip at -0.03 gives clear positive signal
    baseline = soft_baseline(scores)
    reward = max(-clip_range, min(clip_range, scores[chosen_idx] - baseline))

    # Teacher distribution (Phase 4)
    teacher = teacher_distribution_from_scores(scores)
    return reward, teacher


def teacher_distribution_from_scores(scores: list[float], temperature: float = TEACHER_TEMP) -> list[float]:
    """Convert action scores to teacher probability distribution."""
    if not scores:
        return []
    mx = max(scores)
    logits = [exp((s - mx) / max(1e-6, temperature)) for s in scores]
    total = sum(logits)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [v / total for v in logits]


# ---------------------------------------------------------------------------
# Learned card evaluation (uses deck_quality_head from NN)
# ---------------------------------------------------------------------------

# Global reference to the PPO network — set by training loop
_ppo_network = None
_ppo_vocab = None
_learned_blend_alpha = 0.0  # 0 = pure heuristic, 1 = pure learned
_matchup_blend_beta = 0.0  # 0 = no matchup head, >0 = blend matchup scores
_skada_priors = None        # SkadaPriors instance (community card/relic stats)
_skada_blend_gamma = 0.0    # 0 = no skada, >0 = blend skada community priors


def set_learned_evaluator(
    network, vocab,
    alpha: float = 0.5,
    matchup_beta: float = 0.0,
    skada_priors=None,
    skada_gamma: float = 0.0,
) -> None:
    """Register the PPO network for learned deck quality evaluation.

    Called by training loop after network is initialized.
    alpha: blend factor for deck_quality_head (0=heuristic only, 0.5=equal mix, 1=learned only)
    matchup_beta: blend factor for matchup_score_head (0=disabled, 0.3=30% matchup)
    skada_priors: SkadaPriors instance with community card/relic quality data
    skada_gamma: blend factor for Skada priors (0=disabled, 0.15=recommended)
    """
    global _ppo_network, _ppo_vocab, _learned_blend_alpha, _matchup_blend_beta
    global _skada_priors, _skada_blend_gamma
    _ppo_network = network
    _ppo_vocab = vocab
    _learned_blend_alpha = alpha
    _matchup_blend_beta = matchup_beta
    _skada_priors = skada_priors
    _skada_blend_gamma = skada_gamma


def _learned_card_scores(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> list[float] | None:
    """Score card_reward actions using the NN's deck_quality_head.

    Returns None if learned evaluator is not available.
    """
    if _ppo_network is None or _ppo_vocab is None:
        return None

    try:
        from rl_encoder_v2 import build_structured_state, MAX_DECK_SIZE, CARD_AUX_DIM

        # Build base state encoding
        ss = build_structured_state(state, _ppo_vocab)
        base_quality = _ppo_network.evaluate_deck_quality(
            ss.scalars, ss.deck_ids, ss.deck_aux, ss.deck_mask)

        scores = []
        for action in legal_actions:
            action_type = _lower(action.get("action", ""))
            if action_type in ("skip", "skip_card_reward"):
                scores.append(0.0)
                continue

            if action_type == "select_card_reward":
                card_info = _get_action_card_info(action, state)
                if card_info:
                    # Build hypothetical state with card added
                    hypo_state = _apply_overrides(state, _extract_deck(state) + [card_info], None)
                    hypo_ss = build_structured_state(hypo_state, _ppo_vocab)
                    hypo_quality = _ppo_network.evaluate_deck_quality(
                        hypo_ss.scalars, hypo_ss.deck_ids, hypo_ss.deck_aux, hypo_ss.deck_mask)
                    # Marginal value: how much does this card improve deck quality?
                    delta = hypo_quality - base_quality
                    scores.append(delta)
                else:
                    scores.append(0.0)
                continue

            scores.append(0.0)
        return scores
    except Exception as e:
        logger.debug("Learned card scoring failed: %s", e)
        return None


def _skada_card_scores(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> list[float] | None:
    """Score card_reward actions using Skada community priors.

    Uses pick rates, win rate deltas, and synergy data from the Skada
    analytics database. Floor-conditional and deck-synergy-aware.
    Returns None if Skada priors are not loaded.
    """
    if _skada_priors is None or not _skada_priors.loaded:
        return None

    player = _extract_player(state)
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    floor = _safe_int(run.get("floor"), 1)
    deck = _extract_deck(state)
    deck_slugs = [_lower(c.get("id", "")) for c in deck if isinstance(c, dict)]

    scores = []
    card_scores = []  # collect card scores for skip calibration

    for action in legal_actions:
        action_type = _lower(action.get("action", ""))
        if action_type == "select_card_reward":
            card_info = _get_action_card_info(action, state)
            if card_info:
                card_slug = _lower(card_info.get("id", ""))
                s = _skada_priors.card_score_for_context(card_slug, floor, deck_slugs)
                # Center around 0 and scale to similar range as heuristic scores
                centered = (s - 0.5) * 0.15  # maps [0,1] → [-0.075, 0.075]
                scores.append(centered)
                card_scores.append(centered)
            else:
                scores.append(0.0)
        elif action_type in ("skip", "skip_card_reward"):
            scores.append(None)  # placeholder, filled after card scores
        else:
            scores.append(0.0)

    # Fill in skip scores: skip is good when cards are below average
    if card_scores:
        median_card = sorted(card_scores)[len(card_scores) // 2]
        skip_score = -median_card * 0.5  # inverse of median card score
    else:
        skip_score = 0.0

    for i, s in enumerate(scores):
        if s is None:
            scores[i] = skip_score

    return scores


def _skada_shop_card_score(item: dict, state: dict) -> float:
    """Score a shop card using Skada priors."""
    if _skada_priors is None or not _skada_priors.loaded:
        return 0.0
    card_slug = _lower(item.get("card_id", item.get("id", "")))
    if not card_slug:
        return 0.0
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    floor = _safe_int(run.get("floor"), 1)
    deck = _extract_deck(state)
    deck_slugs = [_lower(c.get("id", "")) for c in deck if isinstance(c, dict)]
    s = _skada_priors.card_score_for_context(card_slug, floor, deck_slugs)
    return (s - 0.5) * 0.12


def _skada_relic_score(relic_id: str) -> float:
    """Score a relic using Skada priors."""
    if _skada_priors is None or not _skada_priors.loaded:
        return 0.0
    slug = _lower(relic_id)
    rp = _skada_priors.relic(slug)
    if rp is None:
        return 0.0
    # Use win_rate_delta as quality signal, scaled to similar range
    return max(-0.1, min(0.1, rp.win_rate_delta * 0.001))


def _matchup_card_scores(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> list[float] | None:
    """Score card_reward actions using the matchup_score_head.

    This head is trained on offline combat simulation data.
    Returns None if matchup scorer is not available.
    """
    if _ppo_network is None or _ppo_vocab is None:
        return None
    try:
        from rl_encoder_v2 import encode_structured_state, encode_structured_actions
        state_tensors = encode_structured_state(state, _ppo_vocab)
        action_tensors = encode_structured_actions(state, legal_actions, _ppo_vocab)
        import torch
        device = next(_ppo_network.parameters()).device
        st = {k: torch.tensor(v, device=device).unsqueeze(0) if not isinstance(v, torch.Tensor) else v.unsqueeze(0)
              for k, v in state_tensors.items()}
        at = {k: torch.tensor(v, device=device).unsqueeze(0) if not isinstance(v, torch.Tensor) else v.unsqueeze(0)
              for k, v in action_tensors.items()}
        with torch.no_grad():
            scores = _ppo_network.compute_matchup_scores(st, at)  # (1, A)
        return scores.squeeze(0).cpu().tolist()[:len(legal_actions)]
    except Exception as e:
        logger.debug("Matchup card scoring failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Screen Adapters
# ---------------------------------------------------------------------------

def _card_text(card: dict[str, Any]) -> str:
    parts = [card.get("id"), card.get("name"), card.get("type"),
             card.get("rarity"), card.get("description")]
    keywords = card.get("keywords")
    if isinstance(keywords, list):
        parts.extend(keywords)
    return " ".join(_lower(p) for p in parts if p)


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(kw in text for kw in keywords)


def _is_basic_or_bad(card: dict[str, Any]) -> bool:
    text = _card_text(card)
    rarity = _lower(card.get("rarity"))
    ctype = _lower(card.get("type"))
    return rarity == "basic" or ctype == "curse" or _contains_any(text, REMOVE_BAD_KEYWORDS)


def _extract_deck(state: dict[str, Any]) -> list[dict]:
    player = _extract_player(state)
    deck = player.get("deck")
    return [c for c in deck if isinstance(c, dict)] if isinstance(deck, list) else []


def _screen_state(state: dict[str, Any], key: str) -> dict[str, Any]:
    container = state.get(key)
    return container if isinstance(container, dict) else {}


# --- Card Reward ---

def score_card_reward(state: dict, legal_actions: list[dict]) -> list[float]:
    """Score card reward options using heuristic + learned blend.

    If NN learned evaluator is available, blends:
      score = (1-alpha) * heuristic + alpha * learned
    Otherwise falls back to pure heuristic.
    """
    # Heuristic scores
    deck = _extract_deck(state)
    before = state_utility(state, deck_override=deck)
    heuristic_scores = []
    best_gain = 0.0
    reward_cards = _get_reward_cards(state)
    for card in reward_cards:
        if isinstance(card, dict):
            gain = state_utility(state, deck_override=deck + [card]) - before
            best_gain = max(best_gain, gain)

    for action in legal_actions:
        action_type = _lower(action.get("action", ""))
        if action_type in ("skip", "skip_card_reward"):
            if best_gain <= 0.015:
                heuristic_scores.append(0.035)
            elif best_gain >= 0.070:
                heuristic_scores.append(-0.035)
            else:
                heuristic_scores.append(0.0)
        elif action_type == "select_card_reward":
            card_info = _get_action_card_info(action, state)
            if card_info:
                gain = state_utility(state, deck_override=deck + [card_info]) - before
                heuristic_scores.append(gain * 1.3)
            else:
                heuristic_scores.append(0.0)
        else:
            heuristic_scores.append(0.0)

    # Blend with learned scores if available
    current_scores = heuristic_scores
    alpha = _learned_blend_alpha
    if alpha > 0:
        learned_scores = _learned_card_scores(state, legal_actions)
        if learned_scores is not None and len(learned_scores) == len(heuristic_scores):
            # Scale learned scores to similar range as heuristic (learned is [0,1] delta)
            learned_scale = 1.5
            current_scores = [
                (1 - alpha) * h + alpha * l * learned_scale
                for h, l in zip(heuristic_scores, learned_scores)
            ]

    # Blend with matchup_score_head if available and trained
    beta = _matchup_blend_beta
    if beta > 0 and _ppo_network is not None and _ppo_vocab is not None:
        try:
            matchup_scores = _matchup_card_scores(state, legal_actions)
            if matchup_scores is not None and len(matchup_scores) == len(current_scores):
                current_scores = [
                    (1 - beta) * c + beta * m
                    for c, m in zip(current_scores, matchup_scores)
                ]
        except Exception:
            pass  # Graceful fallback to non-matchup scores

    # Blend with Skada community priors
    gamma = _skada_blend_gamma
    if gamma > 0 and _skada_priors is not None:
        skada_scores = _skada_card_scores(state, legal_actions)
        if skada_scores is not None and len(skada_scores) == len(current_scores):
            current_scores = [
                (1 - gamma) * c + gamma * s
                for c, s in zip(current_scores, skada_scores)
            ]

    return current_scores


# --- Shop (from codex) ---

def score_shop(state: dict, legal_actions: list[dict]) -> list[float]:
    """Score shop actions using state_utility with deck/gold overrides."""
    shop = _screen_state(state, "shop")
    items = shop.get("items")
    if not isinstance(items, list):
        items = []
    deck = _extract_deck(state)
    p = _extract_progress(state)
    gold = p["gold"]
    before = state_utility(state, deck_override=deck, gold_override=gold)
    scores = []

    for action in legal_actions:
        action_name = _lower(action.get("action", ""))
        if action_name in ("shop_exit", "proceed"):
            # Penalize leaving if there's a good remove available
            remove_gain = _best_remove_delta(state)
            affordable = any(
                isinstance(item, dict)
                and _safe_int(item.get("cost"), 10**9) <= gold
                for item in items
            )
            if affordable and remove_gain >= 0.03:
                scores.append(-0.03)
            else:
                scores.append(0.0)
            continue

        idx = _safe_int(action.get("index", -1), -1)
        if idx < 0 or idx >= len(items) or not isinstance(items[idx], dict):
            scores.append(0.0)
            continue

        item = items[idx]
        cost = _safe_int(item.get("cost"), 0)
        gold_after = max(0, gold - cost)
        category = _lower(item.get("category", ""))

        if category == "card":
            pseudo_card = {k: item.get(f"card_{k}", item.get(k))
                          for k in ("id", "name", "type", "rarity", "description", "keywords")}
            after = state_utility(state, deck_override=deck + [pseudo_card], gold_override=gold_after)
            base = (after - before) * 1.5 - min(cost / 180.0, 0.14)
            # Blend with Skada prior for shop cards
            if _skada_blend_gamma > 0 and _skada_priors is not None:
                skada_s = _skada_shop_card_score(item, state)
                base = (1 - _skada_blend_gamma) * base + _skada_blend_gamma * skada_s
            scores.append(base)
        elif category == "card_removal":
            gain = _best_remove_delta(state)
            basics = sum(1 for c in deck if _is_basic_or_bad(c))
            gain += 0.05 if basics >= 7 else (0.03 if basics >= 5 else 0.0)
            reserve_penalty = 0.02 if gold_after < 25 else 0.0
            scores.append(gain * 2.2 - reserve_penalty)
        elif category == "relic":
            base_relic = _score_relic_blob(item, state) - min(cost / 220.0, 0.12)
            # Blend with Skada relic prior
            if _skada_blend_gamma > 0 and _skada_priors is not None:
                relic_id = item.get("id", item.get("relic_id", ""))
                skada_r = _skada_relic_score(relic_id)
                base_relic = (1 - _skada_blend_gamma) * base_relic + _skada_blend_gamma * skada_r
            scores.append(base_relic)
        elif category == "potion":
            scores.append(0.02 if p["hp"] < p["max_hp"] * 0.6 else 0.0)
        else:
            scores.append(0.0)

    return scores


# --- Campfire / Rest Site (enhanced from codex) ---

def score_campfire(state: dict, legal_actions: list[dict]) -> list[float]:
    """Score campfire options with survival-aware logic."""
    rest = _screen_state(state, "rest_site")
    options = rest.get("options")
    deck = _extract_deck(state)
    util = state_utility(state, deck_override=deck)
    sm = survival_margin(state)
    p = _extract_progress(state)
    near_boss = p["floor"] >= 14 and p["act"] <= 1
    safe_threshold = 0.48 if near_boss else 0.38
    scores = []

    for action in legal_actions:
        action_type = _lower(action.get("action", ""))
        if action_type != "choose_rest_option":
            scores.append(0.0)
            continue

        idx = _safe_int(action.get("index", -1), -1)
        option = None
        if isinstance(options, list) and 0 <= idx < len(options):
            option = options[idx]
        option_id = _lower((option or {}).get("id", ""))
        option_name = _lower((option or {}).get("name", ""))
        rest_opt = _lower(action.get("rest_option", ""))

        is_smith = any(k in s for k in ("smith", "upgrade") for s in (option_id, option_name, rest_opt))
        is_rest = any(k in s for k in ("rest", "heal") for s in (option_id, option_name, rest_opt))

        if is_smith:
            upgrade_value = _best_upgrade_value(deck)
            if sm >= safe_threshold:
                scores.append(0.07 + upgrade_value * 1.2)
            else:
                scores.append(upgrade_value - 0.04)
        elif is_rest:
            if sm < (0.45 if near_boss else 0.32):
                scores.append(0.08)
            else:
                scores.append(-0.05)
        else:
            scores.append(0.01)

    return scores


# --- Relic Select (from codex) ---

def score_relic(state: dict, legal_actions: list[dict]) -> list[float]:
    screen_type = _lower(state.get("state_type", ""))
    if screen_type == "treasure":
        relics = _screen_state(state, "treasure").get("relics")
    else:
        relics = _screen_state(state, "relic_select").get("relics")
    if not isinstance(relics, list):
        relics = []
    scores = []
    for action in legal_actions:
        action_name = _lower(action.get("action", ""))
        if action_name in ("skip_relic_selection", "proceed", "skip"):
            scores.append(0.0)
            continue
        idx = _safe_int(action.get("index", -1), -1)
        if 0 <= idx < len(relics) and isinstance(relics[idx], dict):
            scores.append(_score_relic_blob(relics[idx], state))
        else:
            scores.append(0.0)
    return scores


# --- Map (enhanced from codex) ---

def score_map_choice(state: dict, legal_actions: list[dict]) -> list[float]:
    util_val = state_utility(state)
    pv = compute_problem_vector(state)
    elite_ready = float(pv[7])
    sm = survival_margin(state)
    p = _extract_progress(state)
    near_boss = p["floor"] >= 14 and p["act"] == 1
    scores = []

    # Count excess basic cards for shop scoring
    deck = _extract_deck(state)
    act = int(p.get("act") or 1)
    basic_allowance = {1: 8, 2: 5, 3: 3}.get(act, 5)
    basics = sum(1 for c in deck if _is_basic_or_bad(c))
    excess_basics = max(0, basics - basic_allowance)

    for action in legal_actions:
        node_type = _lower(action.get("node_type", action.get("type", "")))
        if "elite" in node_type:
            score = -0.05 + 0.42 * (elite_ready - 0.35) + 0.10 * (sm - 0.35)
            # HP-aware avoidance: strongly discourage elite when survival is low
            if sm < 0.2:
                score -= 0.15
            elif sm < 0.35:
                score -= 0.08
            if near_boss and elite_ready < 0.58:
                score -= 0.04
            scores.append(score)
        elif "boss" in node_type:
            scores.append(0.08)
        elif "rest" in node_type:
            scores.append(0.06 if sm < 0.36 or near_boss else -0.03)
        elif "shop" in node_type:
            gold = p["gold"]
            base = 0.07 if gold >= 110 else (0.03 if gold >= 75 else 0.0)
            # Deck quality bonus: prefer shop when deck is clogged with basics
            remove_bonus = min(0.10, excess_basics * 0.03)
            scores.append(base + remove_bonus)
        elif "treasure" in node_type:
            scores.append(0.05)
        elif "event" in node_type:
            scores.append(0.012 - 0.02 * max(0.0, elite_ready - 0.52))
        elif "monster" in node_type or node_type == "":
            scores.append(0.015 - 0.03 * max(0.0, elite_ready - 0.46))
        else:
            scores.append(0.0)
    return scores


# --- Event (from codex) ---

def score_event(state: dict, legal_actions: list[dict]) -> list[float]:
    event = _screen_state(state, "event")
    options = event.get("options")
    if not isinstance(options, list):
        options = []
    scores = []
    for action in legal_actions:
        action_name = _lower(action.get("action", ""))
        if action_name in ("advance_dialogue", "proceed"):
            scores.append(0.01)
            continue
        idx = _safe_int(action.get("index", -1), -1)
        if idx < 0 or idx >= len(options) or not isinstance(options[idx], dict):
            scores.append(0.0)
            continue
        option = options[idx]
        score = 0.0
        # Parse effect tags if available
        effect_tags = option.get("effect_tags")
        if isinstance(effect_tags, list):
            tags = {_lower(t) for t in effect_tags}
            if "gain_hp" in tags: score += 0.04
            if "lose_hp" in tags: score -= 0.03
            if "lose_gold" in tags: score -= 0.02
            if "gain_gold" in tags: score += 0.02
            if "card_remove" in tags or "card_upgrade" in tags: score += 0.06
            if "add_curse" in tags: score -= 0.05
        # Parse numeric deltas
        hp_delta = option.get("hp_delta")
        if hp_delta is not None:
            score += _safe_int(hp_delta) / 100.0
        gold_delta = option.get("gold_delta")
        if gold_delta is not None:
            score += _safe_int(gold_delta) / 250.0
        # Text heuristics
        text = " ".join(_lower(option.get(k, "")) for k in ("text", "title", "description") if option.get(k))
        if "remove" in text or "upgrade" in text: score += 0.04
        if "curse" in text: score -= 0.04
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

SCREEN_ADAPTERS: dict[str, Any] = {
    "card_reward": score_card_reward,
    "shop": score_shop,
    "rest_site": score_campfire,
    "campfire": score_campfire,
    "relic_select": score_relic,
    "treasure": score_relic,
    "map": score_map_choice,
    "event": score_event,
}


def get_counterfactual_scores(
    screen_type: str,
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> list[float] | None:
    adapter = SCREEN_ADAPTERS.get(screen_type)
    if adapter is None:
        return None
    try:
        return adapter(state, legal_actions)
    except Exception as e:
        logger.debug("Counterfactual scoring failed for %s: %s", screen_type, e)
        return None


def compute_counterfactual_reward(
    screen_type: str,
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    chosen_idx: int,
    clip_range: float = COUNTERFACTUAL_CLIP,
) -> tuple[float, np.ndarray | None]:
    """Full counterfactual reward computation.

    Returns: (reward, teacher_logits_padded_to_MAX_ACTIONS_or_None)
    """
    from rl_encoder_v2 import MAX_ACTIONS

    scores = get_counterfactual_scores(screen_type, state, legal_actions)
    if scores is None:
        return 0.0, None

    reward, teacher_probs = counterfactual_reward(chosen_idx, scores, clip_range)

    teacher_padded = None
    if teacher_probs is not None:
        teacher_padded = np.zeros(MAX_ACTIONS, dtype=np.float32)
        for i, p in enumerate(teacher_probs):
            if i < MAX_ACTIONS:
                teacher_padded[i] = p

    return reward, teacher_padded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_action_card_info(action: dict, state: dict) -> dict | None:
    card = action.get("card")
    if isinstance(card, dict):
        return card
    idx = action.get("index", action.get("card_index"))
    if idx is not None:
        rewards = _get_reward_cards(state)
        idx = int(idx) if idx is not None else -1
        if 0 <= idx < len(rewards):
            return rewards[idx]
    return None


def _get_reward_cards(state: dict) -> list[dict]:
    for key in ("card_reward", "rewards", "combat_rewards"):
        container = state.get(key)
        if isinstance(container, dict):
            cards = container.get("cards", container.get("card_rewards", []))
            if isinstance(cards, list):
                return cards
    return []


def _best_remove_delta(state: dict) -> float:
    """Best possible utility gain from removing one bad card."""
    deck = _extract_deck(state)
    if not deck:
        return 0.0
    before = state_utility(state, deck_override=deck)
    best_gain = 0.0
    for i, card in enumerate(deck):
        if not _is_basic_or_bad(card):
            continue
        after = state_utility(state, deck_override=deck[:i] + deck[i + 1:])
        best_gain = max(best_gain, after - before)
    return best_gain


def _best_upgrade_value(deck: list[dict]) -> float:
    """Estimate best upgrade value from deck."""
    best = 0.0
    for card in deck:
        if _safe_int(card.get("upgrades"), 0) > 0 or bool(card.get("is_upgraded")):
            continue
        text = _card_text(card)
        score = 0.03
        if _contains_any(text, GOOD_UPGRADE_KEYWORDS):
            score += 0.05
        if _contains_any(text, DRAW_KEYWORDS | ENERGY_KEYWORDS | SCALING_KEYWORDS):
            score += 0.02
        best = max(best, score)
    return best


def _score_relic_blob(relic: dict, state: dict) -> float:
    text = " ".join(_lower(relic.get(k, "")) for k in ("id", "name", "description") if relic.get(k))
    p = _extract_progress(state)
    score = 0.04
    if "boss" in text: score += 0.03
    if any(k in text for k in ("energy", "draw", "card", "upgrade", "shop", "elite")): score += 0.04
    if p["act"] == 1 and any(k in text for k in ("damage", "strength", "attack")): score += 0.03
    if any(k in text for k in ("curse", "lose hp", "lose max")): score -= 0.04
    return score
