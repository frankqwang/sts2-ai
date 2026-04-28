"""Conservative deterministic gates before LLM policy calls.

The simple gate only handles forced decisions. The survival gate handles a
small set of low-HP combat situations where a visible block, weak, or lethal
action clearly reduces immediate HP loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import string
from typing import Any


@dataclass(slots=True, frozen=True)
class SimplePolicyDecision:
    action_index: int
    reason: str
    route: str = "heuristic_forced"


def _action_type(action: dict[str, Any]) -> str:
    return str(action.get("action") or action.get("action_type") or action.get("type") or "").lower()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _pick(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    drop = set(string.whitespace + string.punctuation)
    return "".join(ch for ch in text if ch not in drop)


def _player_hp_block(state: dict[str, Any]) -> tuple[float, float]:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    try:
        hp = float(_pick(top_player, "hp", "current_hp", default=_pick(battle_player, "hp", default=0)) or 0)
        block = float(_pick(battle_player, "block", default=_pick(top_player, "block", default=0)) or 0)
    except (TypeError, ValueError):
        return 0.0, 0.0
    return hp, block


def _incoming_damage(state: dict[str, Any]) -> float:
    battle = _as_dict(state.get("battle"))
    total = 0.0
    for enemy in _as_list(state.get("enemies")) or _as_list(battle.get("enemies")):
        if not isinstance(enemy, dict):
            continue
        intent = str(_pick(enemy, "intent_type", "next_move_id", "intent", default="")).upper()
        if "ATTACK" not in intent:
            continue
        try:
            damage = float(_pick(enemy, "intent_damage", "move_base_damage", default=0) or 0)
            hits = max(1, int(_pick(enemy, "intent_hits", "move_hits", default=1) or 1))
        except (TypeError, ValueError):
            continue
        total += damage * hits
    return total


def _end_turn_hp_loss(state: dict[str, Any]) -> float:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    powers = _as_list(battle_player.get("powers")) or _as_list(top_player.get("powers"))
    loss = 0.0
    for power in powers:
        if not isinstance(power, dict):
            continue
        power_id = str(_pick(power, "id", "power_id", "name", default="")).upper()
        if power_id != "CONSTRICT_POWER" and "CONSTRICT" not in power_id:
            continue
        try:
            loss += max(0.0, float(_pick(power, "amount", "stacks", "stack", default=0) or 0))
        except (TypeError, ValueError):
            continue
    return loss


def _current_turn_can_kill_player(state: dict[str, Any]) -> bool:
    hp, block = _player_hp_block(state)
    if hp <= 0:
        return False
    hp_loss = max(0.0, _incoming_damage(state) - block) + _end_turn_hp_loss(state)
    return hp_loss >= hp


_URGENT_POTION_IDS = {
    "FORTIFIER",
    "BLOCK_POTION",
    "HEALTH_POTION",
    "REGEN_POTION",
}
_STARTER_BASIC_PREFIXES = ("STRIKE_", "DEFEND_")
_PROTECTED_STARTER_KEYS = {"BASH", "NEUTRALIZE", "SURVIVOR", "ZAP", "DUALCAST"}
_UPGRADE_PRIORITY = {
    "BASH": 0,
    "PERFECTED_STRIKE": 5,
    "FLAME_BARRIER": 8,
    "COLOSSUS": 10,
    "BLOOD_WALL": 12,
    "RAMPAGE": 14,
    "IRON_WAVE": 16,
    "DISMANTLE": 18,
    "SETUP_STRIKE": 20,
    "ANGER": 22,
}
_WEAK_RE = re.compile(r"\bApply\s+(\d+)\s+Weak\b", re.IGNORECASE)


def _is_urgent_potion_action(action: dict[str, Any], state: dict[str, Any]) -> bool:
    raw = " ".join(str(action.get(key) or "") for key in ("label", "potion_id", "id", "name")).upper()
    slot = action.get("slot")
    potions = _as_list(_as_dict(state.get("player")).get("potions"))
    try:
        potion = potions[int(slot)] if slot is not None else None
    except (TypeError, ValueError, IndexError):
        potion = None
    if isinstance(potion, dict):
        raw += " " + " ".join(str(potion.get(key) or "") for key in ("id", "potion_id", "name")).upper()
    return any(pid in raw for pid in _URGENT_POTION_IDS)


def _player_max_hp(state: dict[str, Any]) -> float:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    try:
        return float(_pick(top_player, "max_hp", default=_pick(battle_player, "max_hp", default=0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _iter_hand_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    raw = (
        _as_list(battle.get("hand"))
        or _as_list(battle_player.get("hand"))
        or _as_list(state.get("hand"))
        or _as_list(top_player.get("hand"))
    )
    return [card for card in raw if isinstance(card, dict)]


def _iter_enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = _as_dict(state.get("battle"))
    return [enemy for enemy in (_as_list(state.get("enemies")) or _as_list(battle.get("enemies"))) if isinstance(enemy, dict)]


def _target_id(enemy: dict[str, Any], fallback: int) -> int:
    raw = _pick(enemy, "target_id", "combat_id", default=None)
    if raw in (None, ""):
        raw_id = _pick(enemy, "id", default=None)
        raw = raw_id if isinstance(raw_id, int) or (isinstance(raw_id, str) and raw_id.isdigit()) else fallback
    try:
        return int(raw or fallback)
    except (TypeError, ValueError):
        return fallback


def _action_card_index(action: dict[str, Any]) -> int | None:
    raw = _pick(action, "card_index", "hand_index", default=None)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _action_target_id(action: dict[str, Any]) -> int | None:
    raw = _pick(action, "target_id", "target", default=None)
    if raw in (None, "", 0, -1, "0", "-1"):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _card_damage(card: dict[str, Any], target_id: int) -> float:
    preview = card.get("preview_damage_per_target")
    if not isinstance(preview, dict):
        return 0.0
    for key in (target_id, str(target_id)):
        if key in preview:
            try:
                return float(preview[key] or 0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _card_block(card: dict[str, Any]) -> float:
    try:
        return max(0.0, float(card.get("preview_block") or 0))
    except (TypeError, ValueError):
        return 0.0


def _action_block(action: dict[str, Any]) -> float:
    for key in ("block", "preview_block", "gain_block"):
        if action.get(key) in (None, ""):
            continue
        try:
            return max(0.0, float(action.get(key)))
        except (TypeError, ValueError):
            continue
    return 0.0


def _action_damage(action: dict[str, Any]) -> float:
    for key in ("hp_damage", "damage", "preview_damage"):
        if action.get(key) in (None, ""):
            continue
        try:
            return max(0.0, float(action.get(key)))
        except (TypeError, ValueError):
            continue
    return 0.0


def _action_self_hp_loss(action: dict[str, Any], card: dict[str, Any]) -> float:
    for key in ("self_hp_loss", "hp_loss", "self_damage"):
        if action.get(key) in (None, ""):
            continue
        try:
            return max(0.0, float(action.get(key)))
        except (TypeError, ValueError):
            continue
    text = " ".join(
        str(_pick(card, key, default="") or "")
        for key in ("description", "desc", "text", "raw_description")
    )
    match = re.search(r"\bLose\s+(\d+)\s+HP\b", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def _card_applies_weak(card: dict[str, Any]) -> bool:
    text = " ".join(
        str(card.get(key) or "")
        for key in ("description", "desc", "text", "raw_description")
    )
    if _WEAK_RE.search(text):
        return True
    card_id = str(_pick(card, "id", "card_id", "name", default="")).upper()
    return card_id in {"UPPERCUT", "SHOCKWAVE", "CLOTHESLINE", "THUNDERCLAP", "INTIMIDATE"}


def _enemy_effective_hp(enemy: dict[str, Any]) -> float:
    try:
        hp = float(_pick(enemy, "hp", "current_hp", default=0) or 0)
        block = float(_pick(enemy, "block", default=0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return hp + block


def _enemy_incoming_damage(enemy: dict[str, Any]) -> float:
    intent = str(_pick(enemy, "intent_type", "next_move_id", "intent", default="")).upper()
    if "ATTACK" not in intent:
        return 0.0
    try:
        damage = float(_pick(enemy, "intent_damage", "move_base_damage", default=0) or 0)
        hits = max(1, int(_pick(enemy, "intent_hits", "move_hits", default=1) or 1))
    except (TypeError, ValueError):
        return 0.0
    return damage * hits


def _survival_action_score(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    incoming: float,
    block: float,
) -> tuple[float, float, str] | None:
    if _action_type(action) != "play_card":
        return None
    hand = _iter_hand_cards(state)
    card_index = _action_card_index(action)
    if card_index is None or card_index < 0 or card_index >= len(hand):
        return None
    card = hand[card_index]
    hp, _ = _player_hp_block(state)
    self_hp_loss = _action_self_hp_loss(action, card)
    if self_hp_loss > 0 and hp - self_hp_loss <= 0:
        return None
    added_block = max(_card_block(card), _action_block(action))
    mitigated_incoming = incoming
    target_id = _action_target_id(action)
    enemies = {_target_id(enemy, idx): enemy for idx, enemy in enumerate(_iter_enemies(state), start=1)}
    if target_id is not None:
        enemy = enemies.get(target_id)
        if enemy:
            target_incoming = _enemy_incoming_damage(enemy)
            visible_damage = max(_card_damage(card, target_id), _action_damage(action))
            if target_incoming > 0 and visible_damage >= _enemy_effective_hp(enemy) > 0:
                mitigated_incoming = max(0.0, mitigated_incoming - target_incoming)
            elif target_incoming > 0 and _card_applies_weak(card):
                mitigated_incoming = max(0.0, mitigated_incoming - target_incoming * 0.25)
    if added_block <= 0 and mitigated_incoming >= incoming:
        return None
    hp_loss_after = self_hp_loss + max(0.0, mitigated_incoming - (block + added_block)) + _end_turn_hp_loss(state)
    improvement = max(0.0, max(0.0, incoming - block) + _end_turn_hp_loss(state) - hp_loss_after)
    if improvement <= 0:
        return None
    card_id = str(_pick(card, "id", "card_id", "name", default="card"))
    reason = f"survival: reduce immediate hp loss by {improvement:g} with {card_id}"
    return hp_loss_after, -improvement, reason


def choose_survival_action(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> SimplePolicyDecision | None:
    """Choose a visible mitigation action in low-HP combat.

    This is intentionally narrow: it only fires when current incoming damage
    would leave the player at a dangerous HP threshold, and the chosen card
    demonstrably lowers immediate HP loss through block, weak, or lethal.
    """
    hp, block = _player_hp_block(state)
    max_hp = _player_max_hp(state)
    incoming = _incoming_damage(state)
    current_loss = max(0.0, incoming - block) + _end_turn_hp_loss(state)
    if hp <= 0 or current_loss <= 0:
        return None
    danger_threshold = max(18.0, max_hp * 0.35) if max_hp > 0 else 18.0
    significant_loss_threshold = max(12.0, max_hp * 0.15) if max_hp > 0 else 12.0
    if hp - current_loss > danger_threshold and current_loss < significant_loss_threshold:
        return None

    enabled = [
        (index, action) for index, action in enumerate(legal_actions or [])
        if isinstance(action, dict) and action.get("is_enabled") is not False
    ]
    candidates: list[tuple[float, float, int, str]] = []
    for raw_index, action in enabled:
        score = _survival_action_score(state, action, incoming=incoming, block=block)
        if score is None:
            continue
        hp_loss_after, neg_improvement, reason = score
        candidates.append((hp_loss_after, neg_improvement, raw_index, reason))
    if not candidates:
        return None
    _, _, raw_index, reason = min(candidates)
    return SimplePolicyDecision(action_index=raw_index, reason=reason, route="heuristic_survival")


def _reward_items(state: dict[str, Any]) -> list[dict[str, Any]]:
    rewards = _as_dict(state.get("rewards")) or _as_dict(state.get("combat_rewards"))
    return [item for item in _as_list(rewards.get("items")) if isinstance(item, dict)]


def _reward_item_for_action(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    items = _reward_items(state)
    if not items:
        return {}

    action_label = _norm_text(action.get("label") or action.get("value"))
    if action_label:
        for item in items:
            item_label = _norm_text(item.get("label") or item.get("description") or item.get("type"))
            if item_label and item_label == action_label:
                return item
        for item in items:
            item_label = _norm_text(item.get("label") or item.get("description") or item.get("type"))
            if item_label and (item_label in action_label or action_label in item_label):
                return item

    action_index = _safe_int(action.get("index"))
    if action_index is not None:
        for item in items:
            if _safe_int(item.get("index")) == action_index:
                return item
        if 0 <= action_index < len(items):
            return items[action_index]
    return {}


def _open_potion_slots(state: dict[str, Any]) -> int | None:
    player = _as_dict(state.get("player"))
    if "open_potion_slots" in player:
        return _safe_int(player.get("open_potion_slots"))
    rewards = _as_dict(state.get("rewards")) or _as_dict(state.get("combat_rewards"))
    reward_player = _as_dict(rewards.get("player"))
    if "open_potion_slots" in reward_player:
        return _safe_int(reward_player.get("open_potion_slots"))
    return None


def _is_claimable_reward_action(state: dict[str, Any], action: dict[str, Any]) -> bool:
    item = _reward_item_for_action(state, action)
    if item and item.get("claimable") is False:
        return False

    reward_type = str(
        _pick(item, "type", "reward_type", default=_pick(action, "reward_type", "type", default=""))
        or ""
    ).lower()
    if reward_type == "potion":
        open_slots = _open_potion_slots(state)
        if open_slots is not None and open_slots <= 0:
            return False
    return True


def _selection_purpose(action: dict[str, Any] | None, state: dict[str, Any]) -> str:
    card_select = _as_dict(state.get("card_select"))
    texts: list[str] = []
    if action is not None:
        texts.extend(str(action.get(key) or "") for key in ("label", "purpose", "operation"))
    texts.extend(str(card_select.get(key) or "") for key in ("purpose", "operation", "prompt", "screen_type"))
    merged = " ".join(texts).lower()
    if any(token in merged for token in ("remove", "purge", "delete", "移除", "删除")):
        return "remove_card"
    if any(token in merged for token in ("transform", "转化", "变化", "变换")):
        return "transform_card"
    if any(token in merged for token in ("upgrade", "升级", "强化", "smith")):
        return "upgrade_card"
    return ""


def _selection_card_id(action: dict[str, Any], state: dict[str, Any]) -> str:
    raw = str(action.get("card_id") or action.get("id") or "").strip()
    if raw:
        return raw.upper()
    card_select = _as_dict(state.get("card_select"))
    cards = _as_list(card_select.get("cards"))
    index = action.get("index")
    if index is None:
        index = action.get("card_index")
    try:
        card = cards[int(index)]
    except (TypeError, ValueError, IndexError):
        card = None
    if isinstance(card, dict):
        raw = str(card.get("id") or card.get("card_id") or "").strip()
        if raw:
            return raw.upper()
    label = str(action.get("label") or "").strip()
    return label.split()[-1].upper() if label else ""


def _remove_priority(card_id: str) -> int:
    if not card_id:
        return 50
    if any(token in card_id for token in ("CURSE", "INJURY", "SHAME", "DOUBT", "REGRET", "PAIN", "PARASITE", "DECAY", "NORMALITY")):
        return 0
    if card_id.startswith("STRIKE_"):
        return 10
    if card_id.startswith("DEFEND_"):
        return 20
    if card_id in _PROTECTED_STARTER_KEYS:
        return 90
    if card_id.startswith(_STARTER_BASIC_PREFIXES):
        return 30
    return 50


def _upgrade_priority(card_id: str) -> int:
    if not card_id:
        return 100
    base = card_id.removesuffix("+")
    if base in _UPGRADE_PRIORITY:
        return _UPGRADE_PRIORITY[base]
    if base.startswith("DEFEND_"):
        return 60
    if base.startswith("STRIKE_"):
        return 70
    return 40


def _choose_card_removal_action(
    state: dict[str, Any],
    enabled: list[tuple[int, dict[str, Any]]],
) -> SimplePolicyDecision | None:
    selectable = [
        (raw_index, action)
        for raw_index, action in enabled
        if _action_type(action) in {"select_card", "select_card_option"}
        and _selection_purpose(action, state) == "remove_card"
    ]
    if selectable:
        raw_index, action = min(
            selectable,
            key=lambda item: (_remove_priority(_selection_card_id(item[1], state)), item[0]),
        )
        card_id = _selection_card_id(action, state) or str(action.get("label") or "card")
        return SimplePolicyDecision(
            action_index=raw_index,
            reason=f"remove weak starter/basic card before key cards: {card_id}",
        )

    action_types = {_action_type(action) for _, action in enabled}
    if action_types and action_types <= {"confirm_selection", "cancel_selection"}:
        card_select = _as_dict(state.get("card_select"))
        selected_ids = [
            str(card.get("id") or card.get("card_id") or "").upper()
            for card in _as_list(card_select.get("selected_cards"))
            if isinstance(card, dict)
        ]
        is_remove = any(_selection_purpose(action, state) == "remove_card" for _, action in enabled)
        if is_remove and any(card_id in _PROTECTED_STARTER_KEYS for card_id in selected_ids):
            for raw_index, action in enabled:
                if _action_type(action) == "cancel_selection":
                    return SimplePolicyDecision(
                        action_index=raw_index,
                        reason=f"cancel protected key-card removal: {','.join(selected_ids)}",
                    )
        for raw_index, action in enabled:
            if _action_type(action) == "confirm_selection":
                return SimplePolicyDecision(
                    action_index=raw_index,
                    reason="confirm selected card removal" if is_remove else "confirm selected cards",
                )
    return None


def _choose_upgrade_action(
    state: dict[str, Any],
    enabled: list[tuple[int, dict[str, Any]]],
) -> SimplePolicyDecision | None:
    selectable = [
        (raw_index, action)
        for raw_index, action in enabled
        if _action_type(action) in {"select_card", "select_card_option"}
        and _selection_purpose(action, state) == "upgrade_card"
    ]
    if not selectable:
        return None
    raw_index, action = min(
        selectable,
        key=lambda item: (_upgrade_priority(_selection_card_id(item[1], state)), item[0]),
    )
    card_id = _selection_card_id(action, state) or str(action.get("label") or "card")
    return SimplePolicyDecision(
        action_index=raw_index,
        reason=f"upgrade highest-impact available card: {card_id}",
    )


def _choose_rest_option_action(
    state: dict[str, Any],
    enabled: list[tuple[int, dict[str, Any]]],
) -> SimplePolicyDecision | None:
    rest_options = [
        (raw_index, action)
        for raw_index, action in enabled
        if _action_type(action) == "choose_rest_option"
    ]
    if not rest_options:
        return None
    hp, _ = _player_hp_block(state)
    max_hp = _player_max_hp(state)
    hp_ratio = hp / max(1.0, max_hp) if max_hp > 0 else 1.0
    wants_heal = hp_ratio < 0.50
    preferred_tokens = ("heal", "rest", "sleep") if wants_heal else ("smith", "upgrade")
    for raw_index, action in rest_options:
        label = str(_pick(action, "label", "id", "name", default="")).lower()
        if any(token in label for token in preferred_tokens):
            return SimplePolicyDecision(
                action_index=raw_index,
                reason=(
                    f"rest to recover before boss/elite: hp={hp:g}/{max_hp:g}"
                    if wants_heal
                    else "smith while HP is stable"
                ),
            )
    return None


def _choose_event_safety_action(
    state: dict[str, Any],
    enabled: list[tuple[int, dict[str, Any]]],
) -> SimplePolicyDecision | None:
    event = _as_dict(state.get("event"))
    event_name = str(_pick(event, "event_id", "id", "event_name", "name", default="")).upper()
    if "TABLET_OF_TRUTH" not in event_name:
        return None
    options = [
        (raw_index, action)
        for raw_index, action in enabled
        if _action_type(action) == "choose_event_option"
    ]
    if len(options) < 2:
        return None

    def label(action: dict[str, Any]) -> str:
        return str(_pick(action, "label", "text", "id", default="")).strip().lower()

    risky_tokens = ("continue", "keep", "deciphering", "losing everything")
    if not any(any(token in label(action) for token in risky_tokens) for _, action in options):
        return None
    for raw_index, action in options:
        text = label(action)
        if any(token in text for token in ("give up", "smash", "leave")):
            hp, _ = _player_hp_block(state)
            max_hp = _player_max_hp(state)
            return SimplePolicyDecision(
                action_index=raw_index,
                reason=f"event safety: stop Tablet of Truth max-HP loss at hp={hp:g}/{max_hp:g}",
            )
    return None


def choose_simple_action(
    _state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> SimplePolicyDecision | None:
    """Return a deterministic decision only when there is no real choice.

    `action_index` is relative to the input `legal_actions` list. Multi-card,
    multi-target, and play-vs-end-turn decisions intentionally return None.
    """
    enabled = [
        (index, action) for index, action in enumerate(legal_actions or [])
        if isinstance(action, dict) and action.get("is_enabled") is not False
    ]
    claimable_rewards = [
        (index, action) for index, action in enabled
        if _action_type(action) == "claim_reward"
        and _is_claimable_reward_action(_state, action)
    ]
    if claimable_rewards:
        raw_index, action = claimable_rewards[0]
        label = str(action.get("label") or action.get("reward_type") or "reward")
        return SimplePolicyDecision(
            action_index=raw_index,
            reason=f"claim visible reward before proceed: {label}",
        )
    if any(_action_type(action) == "claim_reward" for _, action in enabled):
        for raw_index, action in enabled:
            if _action_type(action) == "proceed":
                return SimplePolicyDecision(
                    action_index=raw_index,
                    reason="skip unclaimable visible rewards and proceed",
                )

    removal_decision = _choose_card_removal_action(_state, enabled)
    if removal_decision is not None:
        return removal_decision

    action_types = [_action_type(action) for _, action in enabled]
    upgrade_decision = _choose_upgrade_action(_state, enabled)
    if upgrade_decision is not None:
        return upgrade_decision

    rest_decision = _choose_rest_option_action(_state, enabled)
    if rest_decision is not None:
        return rest_decision

    event_safety_decision = _choose_event_safety_action(_state, enabled)
    if event_safety_decision is not None:
        return event_safety_decision

    if (
        "end_turn" in action_types
        and "use_potion" in action_types
        and all(atype in {"end_turn", "use_potion"} for atype in action_types)
    ):
        if _current_turn_can_kill_player(_state):
            for raw_index, action in enabled:
                if _action_type(action) == "use_potion" and _is_urgent_potion_action(action, _state):
                    label = str(action.get("label") or action.get("potion_id") or "potion")
                    return SimplePolicyDecision(
                        action_index=raw_index,
                        reason=f"use defensive potion before urgent end_turn: {label}",
                    )
        for raw_index, action in enabled:
            if _action_type(action) == "end_turn":
                return SimplePolicyDecision(
                    action_index=raw_index,
                    reason="avoid optional potion use; end_turn is available",
                )

    if "combat_confirm_selection" in action_types and "combat_select_card" in action_types:
        for raw_index, action in enabled:
            if _action_type(action) == "combat_confirm_selection":
                return SimplePolicyDecision(
                    action_index=raw_index,
                    reason="confirm available combat card selection",
                )

    state_type = str(_state.get("state_type") or "").strip().lower().replace("-", "_")
    if state_type == "shop" and "proceed" in action_types:
        for raw_index, action in enabled:
            if _action_type(action) == "proceed":
                return SimplePolicyDecision(
                    action_index=raw_index,
                    reason="leave shop to avoid repeated low-confidence purchases",
                )

    if len(enabled) != 1:
        return None

    raw_index, action = enabled[0]
    atype = _action_type(action)
    if atype == "end_turn":
        reason = "only legal action is end_turn"
    elif atype == "play_card":
        card_id = str(action.get("card_id") or "card")
        reason = f"only legal action is play {card_id}"
    else:
        reason = f"only legal action is {atype or 'action'}"
    return SimplePolicyDecision(action_index=raw_index, reason=reason)


__all__ = ["SimplePolicyDecision", "choose_simple_action", "choose_survival_action"]
