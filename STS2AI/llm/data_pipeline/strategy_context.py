"""Structured strategy context for LLM decision prompts.

The context is intentionally concise and deterministic. It gives the policy
global/build, combat, turn, and turn-history cues without relying on a second
planner model. A later planner/RAG layer can replace these summaries while
keeping the same prompt shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _pick(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _id(value: Any) -> str:
    if isinstance(value, dict):
        return str(_pick(value, "id", "card_id", "relic_id", "monster_id", "entity_id", default="")).upper()
    return str(value or "").upper()


def _hand_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    raw = (
        _as_list(battle.get("hand"))
        or _as_list(battle_player.get("hand"))
        or _as_list(state.get("hand"))
        or _as_list(top_player.get("hand"))
    )
    return [dict(card) for card in raw if isinstance(card, dict)]


def _deck_cards(state: dict[str, Any]) -> list[Any]:
    return _as_list(_as_dict(state.get("player")).get("deck"))


def _relics(state: dict[str, Any]) -> list[Any]:
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(_as_dict(state.get("battle")).get("player"))
    return _as_list(top_player.get("relics")) or _as_list(battle_player.get("relics"))


def _enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = _as_dict(state.get("battle"))
    return [dict(e) for e in (_as_list(state.get("enemies")) or _as_list(battle.get("enemies"))) if isinstance(e, dict)]


def _round_number(state: dict[str, Any]) -> str:
    battle = _as_dict(state.get("battle"))
    combat = _as_dict(state.get("combat"))
    return str(_pick(battle, "round_number_raw", "round_number", default=_pick(combat, "round_number", "turn", default="?")))


def _combat_key(state: dict[str, Any]) -> str:
    battle = _as_dict(state.get("battle"))
    enemies = ",".join(_id(enemy) for enemy in _enemies(state))
    encounter = str(_pick(battle, "encounter_id", "encounter", default=""))
    return encounter or enemies or "combat"


def _target_id(enemy: dict[str, Any], fallback: int) -> int:
    raw = _pick(enemy, "target_id", "combat_id", default=None)
    if raw in (None, ""):
        raw_id = _pick(enemy, "id", default=None)
        if isinstance(raw_id, int) or (isinstance(raw_id, str) and raw_id.isdigit()):
            raw = raw_id
        else:
            raw = fallback
    try:
        return int(raw or fallback)
    except (TypeError, ValueError):
        return fallback


def _enemy_hp(enemy: dict[str, Any]) -> float:
    try:
        return float(_pick(enemy, "hp", "current_hp", default=0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _enemy_block(enemy: dict[str, Any]) -> float:
    try:
        return float(_pick(enemy, "block", default=0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _enemy_incoming(enemy: dict[str, Any]) -> float:
    intent = str(_pick(enemy, "intent_type", "next_move_id", "intent", default="")).upper()
    if "ATTACK" not in intent:
        return 0.0
    try:
        damage = float(_pick(enemy, "intent_damage", "move_base_damage", default=0) or 0)
        hits = max(1, int(_pick(enemy, "intent_hits", "move_hits", default=1) or 1))
    except (TypeError, ValueError):
        return 0.0
    return damage * hits


def _player_powers(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    return [
        dict(power)
        for power in (_as_list(battle_player.get("powers")) or _as_list(top_player.get("powers")))
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
        power_id = _id(power)
        if power_id == "CONSTRICT_POWER" or "CONSTRICT" in power_id:
            loss += max(0.0, _power_amount(power))
    return loss


def _player_line_values(state: dict[str, Any]) -> tuple[float, float, float, float]:
    battle = _as_dict(state.get("battle"))
    top_player = _as_dict(state.get("player"))
    battle_player = _as_dict(battle.get("player"))
    energy_state = _as_dict(state.get("energy"))
    try:
        hp = float(_pick(top_player, "hp", "current_hp", default=0) or 0)
        max_hp = float(_pick(top_player, "max_hp", default=1) or 1)
        block = float(_pick(battle_player, "block", default=_pick(top_player, "block", default=0)) or 0)
        energy = float(
            _pick(
                battle,
                "energy",
                default=_pick(top_player, "energy", default=_pick(energy_state, "current", default=0)),
            )
            or 0
        )
    except (TypeError, ValueError):
        return 0.0, 1.0, 0.0, 0.0
    return hp, max_hp, block, energy


def _player_hp(state: dict[str, Any]) -> float:
    hp, _max_hp, _block, _energy = _player_line_values(state)
    return hp


def _card_preview_damage(card: dict[str, Any]) -> dict[int, float]:
    preview = card.get("preview_damage_per_target")
    if not isinstance(preview, dict):
        return {}
    out: dict[int, float] = {}
    for raw_key, raw_value in preview.items():
        try:
            target_id = int(raw_key)
            damage = float(raw_value)
        except (TypeError, ValueError):
            continue
        out[target_id] = damage
    return out


def _action_label(action: dict[str, Any]) -> str:
    atype = str(_pick(action, "action", "action_type", "type", default="?")).lower()
    if atype == "play_card":
        card = str(_pick(action, "card_id", default="?"))
        hand_idx = _pick(action, "card_index", "hand_index", default="?")
        target = _pick(action, "target_id", "target", default=None)
        if target not in (None, "", 0, -1, "0", "-1"):
            return f"played {card} hand[{hand_idx}] -> enemy{target}"
        return f"played {card} hand[{hand_idx}]"
    if atype == "end_turn":
        return "ended turn"
    return atype or "action"


def _deck_summary(state: dict[str, Any]) -> str:
    ids = [_id(card) for card in _deck_cards(state)]
    if not ids:
        ids = [_id(card) for card in _hand_cards(state)]
    attacks = sum(1 for cid in ids if "STRIKE" in cid or cid in {"BASH", "POMMEL_STRIKE", "BLUDGEON", "CINDER"})
    blocks = sum(1 for cid in ids if "DEFEND" in cid)
    key_cards = [cid for cid in ids if cid in {"BASH", "POMMEL_STRIKE", "BLUDGEON", "CINDER", "FORGOTTEN_RITUAL"}]
    if key_cards:
        key_text = ",".join(dict.fromkeys(key_cards))
    else:
        key_text = "basic strikes/defends"
    if "BLUDGEON" in ids or "CINDER" in ids:
        style = "burst damage build"
    elif attacks >= blocks:
        style = "basic attack deck"
    else:
        style = "defensive starter deck"
    return f"{style}; attacks={attacks} blocks={blocks}; key_cards={key_text}"


def _relic_summary(state: dict[str, Any]) -> str:
    ids = [_id(relic) for relic in _relics(state)]
    if not ids:
        return "none"
    return ",".join(dict.fromkeys(ids))


def _legal_playable_count(legal_actions: list[dict[str, Any]]) -> int:
    return sum(
        1
        for action in legal_actions
        if isinstance(action, dict)
        and action.get("is_enabled") is not False
        and str(_pick(action, "action", "action_type", "type", default="")).lower() == "play_card"
    )


def _lethal_lines(state: dict[str, Any]) -> list[str]:
    hand = _hand_cards(state)
    enemies = {_target_id(enemy, idx): enemy for idx, enemy in enumerate(_enemies(state), start=1)}
    lines: list[str] = []
    for idx, card in enumerate(hand):
        cid = _id(card)
        for target_id, damage in _card_preview_damage(card).items():
            enemy = enemies.get(target_id)
            if not enemy:
                continue
            hp_after_block = _enemy_hp(enemy) + _enemy_block(enemy)
            if damage >= hp_after_block and hp_after_block > 0:
                lines.append(f"hand[{idx}] {cid} kills enemy{target_id} with damage={damage:g}")
    return lines[:3]


def _join_limited(values: list[str], *, max_items: int, fallback: str = "none") -> str:
    clean = [value for value in values if value]
    if not clean:
        return fallback
    return "; ".join(clean[-max_items:])


def _memory_lines(memory: str) -> list[str]:
    clean = (memory or "").strip()
    if not clean:
        return ["  memory:", "    run_prev_combats: none", "    combat: none", "    turn: none"]
    if "\n" in clean:
        return [f"  {line}" if idx == 0 else f"  {line}" for idx, line in enumerate(clean.splitlines())]
    return [f"  memory: {clean}"]


def build_strategy_context(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    history: list[str] | None = None,
    *,
    memory: str = "",
) -> str:
    hp, max_hp, block, energy = _player_line_values(state)
    enemies = _enemies(state)
    incoming = sum(_enemy_incoming(enemy) for enemy in enemies)
    end_turn_hp_loss = _end_turn_hp_loss(state)
    attack_hp_loss = max(0.0, incoming - block)
    block_deficit = attack_hp_loss
    total_hp_loss = attack_hp_loss + end_turn_hp_loss
    alive = [
        (idx, enemy)
        for idx, enemy in enumerate(enemies, start=1)
        if bool(_pick(enemy, "is_alive", "alive", default=True))
    ]
    low_hp_targets = sorted(
        (
            (_target_id(enemy, idx), _enemy_hp(enemy), _id(enemy))
            for idx, enemy in alive
        ),
        key=lambda item: item[1],
    )
    lethal = _lethal_lines(state)
    playable = _legal_playable_count(legal_actions)
    hp_ratio = hp / max(1.0, max_hp)
    is_combat = bool(enemies) or playable > 0 or bool(_hand_cards(state))

    threat = "none"
    if incoming > 0:
        threat = f"incoming_damage={incoming:g}; block_deficit={block_deficit:g}"
        if end_turn_hp_loss > 0:
            threat += f"; end_turn_hp_loss={end_turn_hp_loss:g}; total_hp_loss={total_hp_loss:g}"
    elif end_turn_hp_loss > 0:
        threat = f"end_turn_hp_loss={end_turn_hp_loss:g}; total_hp_loss={total_hp_loss:g}"
    elif any(str(_pick(enemy, "intent_type", "next_move_id", default="")).upper() for _, enemy in alive):
        threat = "no immediate attack; use energy for damage/setup"

    target_plan = "no enemy target"
    if lethal:
        target_plan = "take lethal if available: " + "; ".join(lethal)
    elif low_hp_targets:
        tid, hp_value, name = low_hp_targets[0]
        target_plan = f"focus lowest practical enemy: enemy{tid} ({name}) hp={hp_value:g}"

    turn_goal_parts: list[str] = []
    if lethal:
        turn_goal_parts.append("convert lethal")
    if total_hp_loss > 0 and hp_ratio < 0.5:
        turn_goal_parts.append("reduce incoming damage before greed")
    elif incoming > 0:
        turn_goal_parts.append("balance damage with enough block")
    if playable and energy > 0:
        turn_goal_parts.append("spend energy on high-impact legal cards")
    if not turn_goal_parts:
        turn_goal_parts.append("advance combat without wasting resources")

    avoid: list[str] = []
    if playable and energy > 0:
        avoid.append("ending turn while useful playable cards remain")
    if lethal:
        avoid.append("missing visible lethal")
    if total_hp_loss >= hp and hp > 0:
        avoid.append("dying to current attack/end-turn HP loss")
    elif total_hp_loss > 0:
        avoid.append("ignoring lethal incoming damage")
    if not avoid:
        avoid.append("low-value actions")

    hist = history or []
    history_text = _join_limited(hist, max_items=4, fallback="none")
    memory_text = memory.strip() or "\n".join([
        "memory:",
        "  run_prev_combats: none",
        "  combat: none",
        f"  turn: actions={history_text}",
    ])
    if not is_combat:
        return "\n".join([
            "strategy_context:",
            *_memory_lines(memory_text),
            f"  plan: deck={_deck_summary(state)}; relic_signals={_relic_summary(state)}",
            "  turn: goal=choose the best listed legal non-combat option",
            "  rule: legal_actions override context.",
        ])
    turn_parts = [f"round={_round_number(state)}", f"energy={energy:g}", f"goal={'; '.join(turn_goal_parts)}"]
    if avoid != ["low-value actions"]:
        turn_parts.append(f"avoid={'; '.join(avoid)}")
    lines = [
        "strategy_context:",
        *_memory_lines(memory_text),
        f"  plan: deck={_deck_summary(state)}; relic_signals={_relic_summary(state)}",
        f"  threat: {threat}",
        f"  target: {target_plan}",
        f"  turn: {'; '.join(turn_parts)}",
        "  rule: legal_actions override context.",
    ]
    return "\n".join(lines)


def inject_strategy_context(user_message: str, strategy_context: str) -> str:
    context = (strategy_context or "").strip()
    if not context:
        return user_message
    lines = user_message.splitlines()
    if not lines:
        return context
    return "\n".join([lines[0], context, *lines[1:]])


@dataclass
class StrategyMemory:
    combat_key: str = ""
    turn_key: str = ""
    turn_actions: list[str] = field(default_factory=list)
    combat_actions: list[str] = field(default_factory=list)
    run_notes: list[str] = field(default_factory=list)
    combat_start_hp: float | None = None
    last_hp: float | None = None

    def reset(self) -> None:
        self.combat_key = ""
        self.turn_key = ""
        self.turn_actions.clear()
        self.combat_actions.clear()
        self.run_notes.clear()
        self.combat_start_hp = None
        self.last_hp = None

    def _finish_combat_note(self) -> None:
        if not self.combat_key or self.combat_start_hp is None or self.last_hp is None:
            return
        lost_hp = max(0.0, self.combat_start_hp - self.last_hp)
        if lost_hp <= 0 and not self.combat_actions:
            return
        action_text = _join_limited(self.combat_actions, max_items=2, fallback="none")
        self.run_notes.append(f"prev_combat lost_hp={lost_hp:g} last_actions={action_text}")
        del self.run_notes[:-2]

    def _memory_text(self, state: dict[str, Any]) -> str:
        current_hp = _player_hp(state)
        combat_lost = 0.0
        if self.combat_start_hp is not None:
            combat_lost = max(0.0, self.combat_start_hp - current_hp)
        run_text = _join_limited(self.run_notes, max_items=2, fallback="none")
        combat_prior_actions = list(self.combat_actions)
        if self.turn_actions and combat_prior_actions[-len(self.turn_actions):] == self.turn_actions:
            combat_prior_actions = combat_prior_actions[:-len(self.turn_actions)]
        combat_text = f"lost_hp={combat_lost:g}; prior_actions={_join_limited(combat_prior_actions, max_items=4, fallback='none')}"
        turn_text = _join_limited(self.turn_actions, max_items=4, fallback="none")
        self.last_hp = current_hp
        return "\n".join([
            "memory:",
            f"  run_prev_combats: {run_text}",
            f"  combat: {combat_text}",
            f"  turn: actions={turn_text}",
        ])

    def context_text(self, state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> str:
        combat = _combat_key(state)
        turn = f"{combat}::round={_round_number(state)}"
        if combat != self.combat_key:
            self._finish_combat_note()
            self.combat_key = combat
            self.turn_key = turn
            self.turn_actions.clear()
            self.combat_actions.clear()
            self.combat_start_hp = _player_hp(state)
            self.last_hp = self.combat_start_hp
        elif turn != self.turn_key:
            self.turn_key = turn
            self.turn_actions.clear()
        return build_strategy_context(
            state,
            legal_actions,
            self.turn_actions,
            memory=self._memory_text(state),
        )

    def record_action(self, action: dict[str, Any] | None) -> None:
        if not isinstance(action, dict):
            return
        label = _action_label(action)
        self.turn_actions.append(label)
        self.combat_actions.append(label)
        del self.turn_actions[:-6]
        del self.combat_actions[:-8]


__all__ = [
    "StrategyMemory",
    "build_strategy_context",
    "inject_strategy_context",
]
