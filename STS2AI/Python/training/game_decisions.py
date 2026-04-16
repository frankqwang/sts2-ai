"""Game decision heuristics — map routing, card reward selection, shop decisions.

Extracted from train_hybrid.py. These functions implement rule-based or
heuristic-guided action selection for non-combat game screens (map, card
rewards, shop). They are called by collect_unified_episode() when the
PPO policy defers to hand-crafted rules.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.rl_encoder_v2 import _lower, _safe_float, _safe_int, _extract_player
from core.rl_reward_shaping import extract_next_boss_token

logger = logging.getLogger(__name__)

_RUNTIME_SKADA_PRIORS: Any = None


def _lower_text(value) -> str:
    """Normalize to lower-cased string."""
    return str(value or "").strip().lower()


_SHOP_REMOVE_PRIORITY = (
    "STRIKE_IRONCLAD",
    "DEFEND_IRONCLAD",
    "BASH",
)

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


def _is_act1_map_state(state: dict) -> bool:
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
            from data.skada.skada_priors import SkadaPriors
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

