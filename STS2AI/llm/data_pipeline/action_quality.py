"""Lightweight action quality flags for rollout/eval diagnostics.

These flags are conservative. They are metrics, not hard rules: the policy can
still choose any legal action, but obvious misses become visible in artifacts.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


_KILL_CLAIM_RE = re.compile(r"\b(kill|kills|lethal|finish|dead|eliminate|eliminates)\b", re.IGNORECASE)
_MATH_GE_RE = re.compile(r"\b(-?\d+(?:\.\d+)?)\s*>=\s*(-?\d+(?:\.\d+)?)\b")
_DAMAGE_TARGET_HP_RE = re.compile(
    r"\bdamage\s*=\s*(-?\d+(?:\.\d+)?).*?\btarget_hp\s*=\s*(-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_SELF_HP_LOSS_RE = re.compile(r"\bLose\s+(\d+)\s+HP\b", re.IGNORECASE)

# These flags are diagnostics during rollout, but they must not become positive
# training targets. Dataset builders use this shared blocklist for consistent
# quarantine/drop decisions.
TRAINING_BLOCKLIST_FLAGS = frozenset({
    "dangerous_end_turn",
    "dangerous_self_damage",
    "low_hp_self_damage",
    "missed_visible_lethal",
    "reason_math_contradiction",
    "reason_claims_lethal_but_action_not_lethal",
    "action_score_lethal_math_contradiction",
})


def _pick(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _action_type(action: dict[str, Any]) -> str:
    return str(_pick(action, "action", "action_type", "type", default="")).lower()


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


def _enemy_effective_hp(enemy: dict[str, Any]) -> float:
    try:
        hp = float(_pick(enemy, "hp", "current_hp", default=0) or 0)
        block = float(_pick(enemy, "block", default=0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return hp + block


def _incoming_damage(state: dict[str, Any]) -> float:
    total = 0.0
    skipped = 0
    for enemy in _iter_enemies(state):
        intent = str(_pick(enemy, "intent_type", "next_move_id", "intent", default="")).upper()
        if "ATTACK" not in intent:
            continue
        try:
            damage = float(_pick(enemy, "intent_damage", "move_base_damage", default=0) or 0)
            hits = max(1, int(_pick(enemy, "intent_hits", "move_hits", default=1) or 1))
        except (TypeError, ValueError):
            # sim schema 漂移时静默跳过会让 dangerous_end_turn 漏判；
            # 累计计数，调用侧后续可在 metrics 里记录。
            skipped += 1
            continue
        total += damage * hits
    if skipped:
        # 暴露给调用者审计；不破坏现有签名。
        try:
            state.setdefault("_incoming_damage_skipped", 0)
            state["_incoming_damage_skipped"] = int(state.get("_incoming_damage_skipped") or 0) + skipped
        except Exception:
            pass
    return total


def _player_powers(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    return [
        power for power in (_as_list(battle_player.get("powers")) or _as_list(top_player.get("powers")))
        if isinstance(power, dict)
    ]


def _power_amount(power: dict[str, Any]) -> float:
    try:
        return float(_pick(power, "amount", "stacks", "stack", default=0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _end_turn_hp_loss(state: dict[str, Any]) -> float:
    loss = 0.0
    for power in _player_powers(state):
        power_id = str(_pick(power, "id", "power_id", "name", default="")).upper()
        if power_id == "CONSTRICT_POWER" or "CONSTRICT" in power_id:
            loss += max(0.0, _power_amount(power))
    return loss


def _player_block(state: dict[str, Any]) -> float:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    try:
        return float(_pick(battle_player, "block", default=_pick(top_player, "block", default=0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _player_hp(state: dict[str, Any]) -> float:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    try:
        return float(_pick(top_player, "hp", "current_hp", default=_pick(battle_player, "hp", default=0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _player_max_hp(state: dict[str, Any]) -> float:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    try:
        return float(_pick(top_player, "max_hp", default=_pick(battle_player, "max_hp", default=0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _player_energy(state: dict[str, Any]) -> float:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    energy_state = _as_dict(state.get("energy"))
    try:
        return float(_pick(battle, "energy", default=_pick(top_player, "energy", default=_pick(energy_state, "current", default=0))) or 0)
    except (TypeError, ValueError):
        return 0.0


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


def _card_self_hp_loss(card: dict[str, Any]) -> float:
    text = " ".join(
        str(card.get(key) or "")
        for key in ("description", "desc", "text", "raw_description")
    )
    match = _SELF_HP_LOSS_RE.search(text)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _lethal_action_indices(state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> set[int]:
    hand = _iter_hand_cards(state)
    enemies = {
        _target_id(enemy, idx): enemy
        for idx, enemy in enumerate(_iter_enemies(state), start=1)
        if bool(_pick(enemy, "is_alive", "alive", default=True))
    }
    lethal: set[int] = set()
    for index, action in enumerate(legal_actions):
        if not isinstance(action, dict) or action.get("is_enabled") is False:
            continue
        if _action_type(action) != "play_card":
            continue
        card_index = _action_card_index(action)
        target_id = _action_target_id(action)
        if card_index is None or target_id is None:
            continue
        if card_index < 0 or card_index >= len(hand):
            continue
        enemy = enemies.get(target_id)
        if not enemy:
            continue
        if _card_damage(hand[card_index], target_id) >= _enemy_effective_hp(enemy) > 0:
            lethal.add(index)
    return lethal


def _enabled_play_actions(legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        action
        for action in legal_actions
        if isinstance(action, dict)
        and action.get("is_enabled") is not False
        and _action_type(action) == "play_card"
    ]


def _sum_visible_damage(state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> float:
    hand = _iter_hand_cards(state)
    total = 0.0
    for action in legal_actions:
        if not isinstance(action, dict) or _action_type(action) != "play_card":
            continue
        card_index = _action_card_index(action)
        target_id = _action_target_id(action)
        if card_index is None or target_id is None:
            continue
        if 0 <= card_index < len(hand):
            total += _card_damage(hand[card_index], target_id)
    return total


@dataclass
class QualityReport:
    flags: list[str] = field(default_factory=list)
    opportunities: Counter[str] = field(default_factory=Counter)
    misses: Counter[str] = field(default_factory=Counter)
    metrics: dict[str, float] = field(default_factory=dict)

    def score(self) -> float:
        weights = {
            "visible_lethal": 2.0,
            "dangerous_turn": 1.0,
            "playable_before_end_turn": 0.75,
            "energy_use": 0.5,
            "reason_consistency": 1.0,
            "score_consistency": 0.5,
        }
        penalty = 0.0
        possible = 0.0
        for key, count in self.opportunities.items():
            possible += weights.get(key, 1.0) * count
        for key, count in self.misses.items():
            penalty += weights.get(key, 1.0) * count
        if possible <= 0:
            return 1.0
        return max(0.0, round(1.0 - penalty / possible, 4))

    def as_dict(self) -> dict[str, Any]:
        return {
            "flags": list(self.flags),
            "opportunities": {key: int(value) for key, value in self.opportunities.most_common()},
            "misses": {key: int(value) for key, value in self.misses.most_common()},
            "metrics": dict(self.metrics),
            "mechanism_score": self.score(),
        }


def assess_action_quality(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    chosen_index: int,
    reason: str = "",
    action_scores: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return conservative quality flags for a chosen action."""
    return assess_action_quality_report(
        state,
        legal_actions,
        chosen_index,
        reason=reason,
        action_scores=action_scores,
    ).flags


def assess_action_quality_report(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    chosen_index: int,
    *,
    reason: str = "",
    action_scores: list[dict[str, Any]] | None = None,
) -> QualityReport:
    """Return mechanism-understanding diagnostics for a chosen action."""
    report = QualityReport()
    incoming = _incoming_damage(state)
    end_turn_hp_loss = _end_turn_hp_loss(state)
    block = _player_block(state)
    hp = _player_hp(state)
    max_hp = _player_max_hp(state)
    energy = _player_energy(state)
    visible_damage = _sum_visible_damage(state, legal_actions)
    current_hp_loss = max(0.0, incoming - block) + end_turn_hp_loss
    report.metrics.update({
        "incoming_damage": incoming,
        "end_turn_hp_loss": end_turn_hp_loss,
        "current_hp_loss": current_hp_loss,
        "hp_after_current_threat": hp - current_hp_loss,
        "block": block,
        "energy": energy,
        "visible_damage": visible_damage,
    })

    if chosen_index < 0 or chosen_index >= len(legal_actions):
        report.flags.append("invalid_chosen_index")
        report.misses["valid_action"] += 1
        return report
    chosen = legal_actions[chosen_index]
    chosen_type = _action_type(chosen)
    play_actions = _enabled_play_actions(legal_actions)
    end_turn_available = any(
        isinstance(action, dict)
        and action.get("is_enabled") is not False
        and _action_type(action) == "end_turn"
        for action in legal_actions
    )
    claim_reward_indices = {
        index
        for index, action in enumerate(legal_actions)
        if isinstance(action, dict)
        and action.get("is_enabled") is not False
        and _action_type(action) == "claim_reward"
    }
    lethal = _lethal_action_indices(state, legal_actions)
    if reason:
        for match in _MATH_GE_RE.finditer(reason):
            report.opportunities["reason_consistency"] += 1
            try:
                if float(match.group(1)) + 1e-9 < float(match.group(2)):
                    report.flags.append("reason_math_contradiction")
                    report.misses["reason_consistency"] += 1
                    break
            except ValueError:
                continue
        if _KILL_CLAIM_RE.search(reason):
            report.opportunities["reason_consistency"] += 1
            if chosen_index not in lethal:
                report.flags.append("reason_claims_lethal_but_action_not_lethal")
                report.misses["reason_consistency"] += 1
    for score in action_scores or []:
        if not isinstance(score, dict):
            continue
        note = str(score.get("note") or "")
        if not note or not _KILL_CLAIM_RE.search(note):
            continue
        report.opportunities["score_consistency"] += 1
        for match in _DAMAGE_TARGET_HP_RE.finditer(note):
            try:
                if float(match.group(1)) + 1e-9 < float(match.group(2)):
                    report.flags.append("action_score_lethal_math_contradiction")
                    report.misses["score_consistency"] += 1
                    break
            except ValueError:
                continue
    if lethal:
        report.opportunities["visible_lethal"] += 1
        if chosen_index not in lethal:
            report.flags.append("missed_visible_lethal")
            report.misses["visible_lethal"] += 1
    if chosen_type == "end_turn" and play_actions:
        report.opportunities["playable_before_end_turn"] += 1
        report.flags.append("end_turn_with_playable_cards")
        report.misses["playable_before_end_turn"] += 1
    if chosen_type == "end_turn" and energy > 0 and play_actions:
        report.opportunities["energy_use"] += 1
        report.flags.append("floating_energy_end_turn")
        report.misses["energy_use"] += 1
    if claim_reward_indices:
        report.opportunities["unclaimed_reward"] += 1
        if chosen_type == "proceed":
            report.flags.append("proceed_with_unclaimed_rewards")
            report.misses["unclaimed_reward"] += 1
    if chosen_type == "use_potion" and end_turn_available and incoming <= block:
        report.opportunities["potion_conservation"] += 1
        report.flags.append("unnecessary_potion_use")
        report.misses["potion_conservation"] += 1
    if current_hp_loss > 0 and play_actions:
        report.opportunities["dangerous_turn"] += 1
        if chosen_type == "end_turn":
            report.flags.append("dangerous_end_turn")
            report.misses["dangerous_turn"] += 1
    if chosen_type == "play_card":
        chosen_card_index = _action_card_index(chosen)
        hand = _iter_hand_cards(state)
        if chosen_card_index is not None and 0 <= chosen_card_index < len(hand):
            self_hp_loss = _card_self_hp_loss(hand[chosen_card_index])
            if self_hp_loss > 0:
                report.opportunities["self_damage_safety"] += 1
                hp_after_self = hp - self_hp_loss
                report.metrics["chosen_self_hp_loss"] = self_hp_loss
                report.metrics["hp_after_chosen_self_damage"] = hp_after_self
                if hp_after_self <= max(0.0, incoming - block) + end_turn_hp_loss:
                    report.flags.append("dangerous_self_damage")
                    report.misses["self_damage_safety"] += 1
                elif current_hp_loss > 0 and hp_after_self <= max(10.0, max_hp * 0.15):
                    report.flags.append("low_hp_self_damage")
                    report.misses["self_damage_safety"] += 1
    return report


def count_quality_flags(steps: list[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for step in steps:
        flags = getattr(step, "quality_flags", None)
        if flags is None and isinstance(step, dict):
            flags = step.get("quality_flags")
        for flag in flags or []:
            counter[str(flag)] += 1
    return {key: int(value) for key, value in counter.most_common()}


def summarize_quality_reports(steps: list[Any], *, final_state: dict[str, Any] | None = None) -> dict[str, Any]:
    flags: Counter[str] = Counter()
    opportunities: Counter[str] = Counter()
    misses: Counter[str] = Counter()
    scores: list[float] = []
    hp_start: float | None = None
    hp_end: float | None = None
    turn_count = 0
    damage_tempo: list[float] = []
    for step in steps:
        report = getattr(step, "quality_report", None)
        if report is None and isinstance(step, dict):
            report = step.get("quality_report")
        if not isinstance(report, dict):
            continue
        flags.update(report.get("flags") or [])
        opportunities.update(report.get("opportunities") or {})
        misses.update(report.get("misses") or {})
        score = report.get("mechanism_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        damage = metrics.get("visible_damage")
        if isinstance(damage, (int, float)):
            damage_tempo.append(float(damage))

        state = getattr(step, "state", None)
        if state is None and isinstance(step, dict):
            state = step.get("state")
        if isinstance(state, dict):
            hp = _player_hp(state)
            if hp_start is None:
                hp_start = hp
            hp_end = hp
        chosen_index = getattr(step, "chosen_index", None)
        legal_actions = getattr(step, "legal_actions", None)
        if isinstance(step, dict):
            chosen_index = step.get("chosen_index", chosen_index)
            legal_actions = step.get("legal_actions", legal_actions)
        try:
            # NOTE: chosen_index=-1 表示 invalid_output；Python 负索引会取末尾元素
            # 而不是 IndexError，所以必须显式判 >=0，否则会把 invalid 步误数为 end_turn。
            if (
                isinstance(legal_actions, list)
                and chosen_index is not None
                and int(chosen_index) >= 0
                and int(chosen_index) < len(legal_actions)
            ):
                action = legal_actions[int(chosen_index)]
                if isinstance(action, dict) and _action_type(action) == "end_turn":
                    turn_count += 1
        except (TypeError, ValueError):
            pass

    # NOTE: sim 可能在战斗结束（玩家死/max_steps）时把 player.hp 重置回初始值，
    # 这会导致 final_state 报的 hp 反而高于 steps 末尾的真实 hp，从而把 hp_lost 算成 0。
    # 因此 final_state 仅在不大于 steps 末尾累加 hp_end 时才覆盖；否则保留 step 累加值。
    if final_state and isinstance(final_state, dict):
        fs_hp = _player_hp(final_state)
        if isinstance(fs_hp, (int, float)):
            if hp_end is None or fs_hp <= float(hp_end):
                hp_end = fs_hp
    hp_lost = max(0.0, (hp_start or 0.0) - (hp_end or hp_start or 0.0))
    total_opportunities = sum(opportunities.values())
    total_misses = sum(misses.values())
    sequence_opportunities = (
        opportunities.get("visible_lethal", 0)
        + opportunities.get("playable_before_end_turn", 0)
        + opportunities.get("energy_use", 0)
    )
    sequence_misses = (
        misses.get("visible_lethal", 0)
        + misses.get("playable_before_end_turn", 0)
        + misses.get("energy_use", 0)
    )
    defense_opportunities = opportunities.get("dangerous_turn", 0)
    defense_misses = misses.get("dangerous_turn", 0)
    sequence_score = 1.0 - sequence_misses / sequence_opportunities if sequence_opportunities else 1.0
    defense_score = 1.0 - defense_misses / defense_opportunities if defense_opportunities else 1.0
    return {
        "flags": {key: int(value) for key, value in flags.most_common()},
        "opportunities": {key: int(value) for key, value in opportunities.most_common()},
        "misses": {key: int(value) for key, value in misses.most_common()},
        "mechanism_score": round(sum(scores) / len(scores), 4) if scores else 1.0,
        "mechanism_miss_rate": round(total_misses / total_opportunities, 4) if total_opportunities else 0.0,
        "sequence_score": round(max(0.0, sequence_score), 4),
        "defense_score": round(max(0.0, defense_score), 4),
        "hp_lost": hp_lost,
        "turns": turn_count,
        "steps": len(steps),
        "steps_per_turn": round(len(steps) / max(1, turn_count), 4) if steps else 0.0,
        "visible_damage_per_step": round(sum(damage_tempo) / len(damage_tempo), 4) if damage_tempo else 0.0,
    }


__all__ = [
    "QualityReport",
    "TRAINING_BLOCKLIST_FLAGS",
    "assess_action_quality",
    "assess_action_quality_report",
    "count_quality_flags",
    "summarize_quality_reports",
]
