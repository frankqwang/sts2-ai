"""基于规则的启发式老师（Ironclad 优先版）。

关键适配点（2026-04-24 实测）：
- legal_actions 里 `action`/`type` = "play_card"；`card_id` 就在 action 里，不必回查手牌
- `card_index` = 手牌 index；`target_id` 是 int
- enemy.`target_id` 是 int；`intent_type` 大写（Attack / Defend / Buff 等）
- enemy powers 用 `id` 不是 `power_id`
- 手牌 state 里没有 damage_now/block_now，从 `card_effects.lookup_effect` 查表
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.data_pipeline.card_effects import (
    effective_block,
    effective_damage,
    lookup_effect,
)


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _pick(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _powers_map(source: list[Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for power in source or []:
        if not isinstance(power, dict):
            continue
        pid = str(power.get("id") or power.get("power_id") or power.get("name") or "").upper()
        if not pid:
            continue
        amount = power.get("amount")
        try:
            out[pid] = float(amount) if amount is not None else 1.0
        except (TypeError, ValueError):
            out[pid] = 1.0
    return out


@dataclass(slots=True)
class _TargetInfo:
    target_id: int
    name: str
    hp: float
    max_hp: float
    block: float
    intent: str
    incoming_damage: float
    vulnerable: bool
    weak: bool
    alive: bool


def _enumerate_enemies(state: dict[str, Any]) -> list[_TargetInfo]:
    battle = _as_dict(state.get("battle"))
    enemies_raw = _as_list(state.get("enemies")) or _as_list(battle.get("enemies"))
    out: list[_TargetInfo] = []
    for index, enemy in enumerate(enemies_raw):
        if not isinstance(enemy, dict):
            continue
        powers = _powers_map(_as_list(enemy.get("powers")) or _as_list(enemy.get("buffs")))
        intent = str(_pick(enemy, "intent_type", "next_move_id", default="")).upper()
        dmg = float(_pick(enemy, "intent_damage", "move_base_damage", default=0) or 0.0)
        hits = max(1, int(_pick(enemy, "intent_hits", "move_hits", default=1) or 1))
        incoming = dmg * hits if "ATTACK" in intent else 0.0
        try:
            target_id = int(_pick(enemy, "target_id", "combat_id", default=index) or index)
        except (TypeError, ValueError):
            target_id = index
        out.append(
            _TargetInfo(
                target_id=target_id,
                name=str(_pick(enemy, "monster_id", "entity_id", "id", default=f"enemy_{index}")),
                hp=float(_pick(enemy, "hp", "current_hp", default=0) or 0.0),
                max_hp=float(_pick(enemy, "max_hp", default=0) or 0.0),
                block=float(_pick(enemy, "block", default=0) or 0.0),
                intent=intent,
                incoming_damage=incoming,
                vulnerable=bool(powers.get("VULNERABLE_POWER") or powers.get("VULNERABLE")),
                weak=bool(powers.get("WEAK_POWER") or powers.get("WEAK")),
                alive=bool(_pick(enemy, "is_alive", "alive", default=True)),
            )
        )
    return out


def _resolve_incoming_total(enemies: list[_TargetInfo], player_weak: bool = False) -> float:
    total = sum(e.incoming_damage for e in enemies if e.alive)
    if player_weak:
        total *= 0.75
    return total


def _find_target(enemies: list[_TargetInfo], target_id: int | str | None) -> _TargetInfo | None:
    if target_id is None:
        return None
    try:
        tid = int(target_id)
    except (TypeError, ValueError):
        return None
    for e in enemies:
        if e.target_id == tid:
            return e
    return None


def _hand_card_for_action(action: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    battle = _as_dict(state.get("battle"))
    hand = (
        _as_list(battle.get("hand"))
        or _as_list(_as_dict(battle.get("player")).get("hand"))
        or _as_list(state.get("hand"))
    )
    card_index = _pick(action, "card_index", "hand_index", "index", default=-1)
    try:
        ci = int(card_index)
    except (TypeError, ValueError):
        return {}
    if 0 <= ci < len(hand) and isinstance(hand[ci], dict):
        return dict(hand[ci])
    return {}


@dataclass(slots=True, frozen=True)
class TeacherDecision:
    action_index: int
    score: float
    reason: str


def _score_play_card(
    action: dict[str, Any],
    state: dict[str, Any],
    enemies: list[_TargetInfo],
    incoming: float,
    player_hp: float,
    player_max_hp: float,
    player_block: float,
) -> tuple[float, str]:
    card_id = str(_pick(action, "card_id", default="")).upper()
    card_from_hand = _hand_card_for_action(action, state)
    is_upgraded = bool(_pick(card_from_hand, "is_upgraded", default=False))
    target_id = _pick(action, "target_id", "target", default=None)
    target = _find_target(enemies, target_id)

    dmg = effective_damage(card_id, is_upgraded=is_upgraded)
    blk = effective_block(card_id, is_upgraded=is_upgraded)
    eff = lookup_effect(card_id, is_upgraded=is_upgraded)

    score = 0.0
    reasons: list[str] = []

    hp_ratio = player_hp / max(1.0, player_max_hp)
    need_block = incoming - player_block
    defensive_mode = hp_ratio < 0.35 or need_block > 0.5 * max(1.0, player_max_hp)

    # 攻击
    if dmg > 0:
        effective_target = target
        if effective_target is None and enemies:
            alive = [e for e in enemies if e.alive]
            if alive:
                effective_target = min(alive, key=lambda e: e.hp)
        if effective_target is not None:
            actual = dmg
            if effective_target.vulnerable:
                actual *= 1.5
            actual -= effective_target.block
            actual = max(0.0, actual)
            score += 1.0 + actual * 0.25
            if actual >= effective_target.hp:
                score += 6.0  # 点杀单位极高优先级
                reasons.append(f"点杀 {effective_target.name}({actual:.0f}≥{effective_target.hp:.0f})")
            else:
                reasons.append(f"打 {effective_target.name} {actual:.0f}")
            # 打攻击 intent 的怪，减压
            if "ATTACK" in effective_target.intent and effective_target.incoming_damage > 0:
                score += 0.6
        # AOE 额外加分
        if eff.is_aoe:
            alive_n = sum(1 for e in enemies if e.alive)
            score += 0.8 * alive_n
            reasons.append(f"AOE×{alive_n}")

    # 防御
    if blk > 0:
        if need_block > 0:
            score += 1.5 + min(need_block, blk) * 0.4
            reasons.append(f"补 {min(need_block, blk):.0f} 挡")
        else:
            score += 0.3  # 没压力时小加分
        if defensive_mode:
            score += 2.5

    # Debuff
    if eff.applies_vulnerable or eff.applies_weak:
        effective_target = target or max(
            (e for e in enemies if e.alive), key=lambda e: e.hp, default=None
        )
        if effective_target is not None and effective_target.alive:
            if eff.applies_vulnerable and not effective_target.vulnerable and effective_target.hp > 8:
                score += 2.5 + 0.05 * effective_target.hp
                reasons.append(f"脆弱→{effective_target.name}")
            if eff.applies_weak and not effective_target.weak and "ATTACK" in effective_target.intent:
                score += 1.5
                reasons.append(f"虚弱→{effective_target.name}")

    # Power
    if eff.is_power:
        score += 2.0
        reasons.append(f"上 {card_id}")

    if score == 0.0 and dmg == 0 and blk == 0:
        # 未知卡 fallback
        score = 0.5
        reasons.append(f"打未知卡 {card_id}")

    if not reasons:
        reasons.append(f"打 {card_id}")

    return score, "; ".join(reasons)


def _score_end_turn(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> tuple[float, str]:
    battle = _as_dict(state.get("battle"))
    energy = float(_pick(battle, "energy", default=_pick(_as_dict(state.get("player")), "energy", default=0)) or 0)
    has_playable = any(
        (str(a.get("action") or a.get("action_type") or a.get("type") or "").lower() == "play_card"
         and a.get("is_enabled") is not False)
        for a in legal_actions
    )
    if not has_playable:
        return 6.0, "没有可打的牌，结束回合"
    if energy <= 0:
        return 4.0, "能量耗尽，结束回合"
    return -0.5 - energy * 2.5, f"浪费 {energy:.0f} 点能量 end_turn"


def pick_action(state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> TeacherDecision:
    enabled = [
        (index, action)
        for index, action in enumerate(legal_actions)
        if isinstance(action, dict) and action.get("is_enabled") is not False
    ]
    if not enabled:
        return TeacherDecision(action_index=0, score=0.0, reason="no_enabled_actions")

    top_player = _as_dict(state.get("player"))
    battle = _as_dict(state.get("battle"))
    battle_player = _as_dict(battle.get("player"))
    player_powers = _powers_map(_as_list(battle_player.get("powers")) or _as_list(top_player.get("powers")))
    player_hp = float(_pick(top_player, "hp", "current_hp", default=0) or 0)
    player_max_hp = float(_pick(top_player, "max_hp", default=1) or 1)
    player_block = float(_pick(battle_player, "block", default=_pick(top_player, "block", default=0)) or 0)
    player_weak = bool(player_powers.get("WEAK_POWER") or player_powers.get("WEAK"))

    enemies = _enumerate_enemies(state)
    incoming = _resolve_incoming_total(enemies, player_weak=player_weak)

    best_index = enabled[0][0]
    best_score = float("-inf")
    best_reason = "default"

    for raw_index, action in enabled:
        atype = str(_pick(action, "action", "action_type", "type", default="")).lower()
        if atype == "play_card":
            score, reason = _score_play_card(
                action, state, enemies, incoming, player_hp, player_max_hp, player_block
            )
        elif atype == "end_turn":
            score, reason = _score_end_turn(state, legal_actions)
        else:
            score, reason = 0.3, f"兜底选 {atype or 'action'}"
        if score > best_score:
            best_score = score
            best_index = raw_index
            best_reason = reason

    return TeacherDecision(action_index=best_index, score=float(best_score), reason=best_reason)


__all__ = ["TeacherDecision", "pick_action"]
