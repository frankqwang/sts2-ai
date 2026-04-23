from __future__ import annotations

"""Feature extraction from structured domain objects to fixed-width tensors."""

import hashlib
from dataclasses import dataclass
from typing import Iterable

from ..config import EncoderConfig
from ..domain import BattleState, HistoryStep, LegalAction, TrainingSample, TransitionDelta


PLAYER_SEMANTIC_DIM = 4
HAND_SEMANTIC_DIM = 8
# 13 semantic flags from _action_semantic_flags(...)
ACTION_SEMANTIC_DIM = 13
PILE_SEMANTIC_DIM = 4
# 18 semantic/history extras on top of the 17 scalar history fields
HISTORY_SEMANTIC_DIM = 18
HISTORY_TOKEN_DIM = 17 + HISTORY_SEMANTIC_DIM
STATIC_NUMERIC_DIM = 19

_ENGINE_POWER_IDS = {
    "BARRICADE",
    "CORRUPTION",
    "DARK_EMBRACE",
    "DEMON_FORM",
    "EVOLVE",
    "FEEL_NO_PAIN",
    "INFLAME",
    "METALLICIZE",
    "PYRE",
    "RUPTURE",
}
_EXHAUST_ENABLER_IDS = {
    "BURNING_PACT",
    "FIEND_FIRE",
    "PURITY",
    "SECOND_WIND",
    "SEVER_SOUL",
    "TRUE_GRIT",
}
_EXHAUST_PAYOFF_IDS = {
    "DARK_EMBRACE",
    "FEEL_NO_PAIN",
    "PACTS_END",
    "PYRE",
}
_RESOURCE_CARD_IDS = {
    "BLOODLETTING",
    "BURNING_PACT",
    "INFERNAL_BLADE",
    "OFFERING",
    "POMMEL_STRIKE",
    "SHRUG_IT_OFF",
}
@dataclass(slots=True)
class EncodedSample:
    player_numeric: list[float]
    static_numeric: list[float]
    static_ids: list[int]
    relic_ids: list[int]
    deck_card_ids: list[int]
    potion_ids: list[int]
    draw_pile_ids: list[int]
    discard_pile_ids: list[int]
    exhaust_pile_ids: list[int]
    player_buff_ids: list[int]
    player_buff_values: list[float]
    enemy_numeric: list[list[float]]
    enemy_ids: list[int]
    enemy_target_ids: list[int]
    enemy_intent_ids: list[int]
    enemy_buff_ids: list[list[int]]
    enemy_buff_values: list[list[float]]
    hand_numeric: list[list[float]]
    hand_card_ids: list[int]
    pile_numeric: list[float]
    history_numeric: list[list[float]]
    action_numeric: list[list[float]]
    action_type_ids: list[int]
    action_card_ids: list[int]
    action_card_indices: list[int]
    action_target_ids: list[int]
    behavior_action_index: int
    fight_targets: list[float]
    delta_targets: list[float]
    sample_weight: float
    fight_quality_score: float
    behavior_ce_scale: float
    old_logprob: float
    old_value: float
    old_intent_logprob: float
    old_intent_value: float
    ppo_return: float
    ppo_advantage: float
    turn_id: int
    turn_start_mask: float
    active_intent: int
    turn_return: float
    turn_advantage: float
    chosen_action_future_targets: list[float]
    submenu_confirm_target: float
    submenu_has_confirm: float


class FeatureExtractor:
    def __init__(self, config: EncoderConfig):
        self._config = config

    def encode_sample(self, sample: TrainingSample) -> EncodedSample:
        return EncodedSample(
            player_numeric=self._encode_player(sample.state),
            static_numeric=self._encode_static_numeric(sample.state),
            static_ids=self._encode_static_ids(sample.state),
            relic_ids=self._encode_id_list(sample.state.context.relics),
            deck_card_ids=self._encode_id_list(sample.state.context.deck_cards),
            potion_ids=self._encode_id_list(sample.state.player.potions),
            draw_pile_ids=self._encode_id_list(sample.state.piles.draw_cards),
            discard_pile_ids=self._encode_id_list(sample.state.piles.discard_cards),
            exhaust_pile_ids=self._encode_id_list(sample.state.piles.exhaust_cards),
            player_buff_ids=self._encode_mapping_ids(sample.state.player.buffs),
            player_buff_values=self._encode_mapping_values(sample.state.player.buffs),
            enemy_numeric=self._encode_enemies(sample.state),
            enemy_ids=[self._hash_id(enemy.enemy_id) for enemy in sample.state.enemies],
            enemy_target_ids=self._encode_enemy_target_ids(sample.state),
            enemy_intent_ids=[self._hash_id(enemy.intent_id) for enemy in sample.state.enemies],
            enemy_buff_ids=[self._encode_mapping_ids(enemy.buffs) for enemy in sample.state.enemies],
            enemy_buff_values=[self._encode_mapping_values(enemy.buffs) for enemy in sample.state.enemies],
            hand_numeric=self._encode_hand(sample.state),
            hand_card_ids=[self._hash_id(card.card_id) for card in sample.state.hand],
            pile_numeric=self._encode_piles(sample.state),
            history_numeric=self._encode_history(sample.history),
            action_numeric=self._encode_actions(sample.legal_actions),
            action_type_ids=[self._hash_id(action.action_type) for action in sample.legal_actions],
            action_card_ids=[self._hash_id(action.card_id or action.potion_id or action.special_id) for action in sample.legal_actions],
            action_card_indices=self._encode_action_card_indices(sample.legal_actions),
            action_target_ids=self._encode_action_target_ids(sample.legal_actions),
            behavior_action_index=sample.behavior_action_index,
            fight_targets=[
                sample.fight_label.fight_win,
                sample.fight_label.enemy_hp_fraction_dealt,
                sample.fight_label.self_hp_fraction_remaining,
            ],
            delta_targets=self._encode_delta(sample.delta),
            sample_weight=max(0.1, float(sample.sample_weight)),
            fight_quality_score=float(sample.fight_score),
            behavior_ce_scale=float(sample.metadata.get("behavior_ce_scale", 1.0) or 1.0),
            old_logprob=float(sample.old_logprob),
            old_value=float(sample.old_value),
            old_intent_logprob=float(sample.old_intent_logprob),
            old_intent_value=float(sample.old_intent_value),
            ppo_return=float(sample.ppo_return),
            ppo_advantage=float(sample.ppo_advantage),
            turn_id=int(sample.turn_id),
            turn_start_mask=float(sample.turn_start_mask),
            active_intent=int(sample.active_intent),
            turn_return=float(sample.turn_return),
            turn_advantage=float(sample.turn_advantage),
            chosen_action_future_targets=self._encode_future_summary_targets(
                sample.chosen_action_future_targets or sample.metadata
            ),
            submenu_confirm_target=float(sample.submenu_confirm_target),
            submenu_has_confirm=float(sample.submenu_has_confirm),
        )

    def encode_inference(self, state: BattleState, history: list[HistoryStep], legal_actions: list[LegalAction]) -> EncodedSample:
        return EncodedSample(
            player_numeric=self._encode_player(state),
            static_numeric=self._encode_static_numeric(state),
            static_ids=self._encode_static_ids(state),
            relic_ids=self._encode_id_list(state.context.relics),
            deck_card_ids=self._encode_id_list(state.context.deck_cards),
            potion_ids=self._encode_id_list(state.player.potions),
            draw_pile_ids=self._encode_id_list(state.piles.draw_cards),
            discard_pile_ids=self._encode_id_list(state.piles.discard_cards),
            exhaust_pile_ids=self._encode_id_list(state.piles.exhaust_cards),
            player_buff_ids=self._encode_mapping_ids(state.player.buffs),
            player_buff_values=self._encode_mapping_values(state.player.buffs),
            enemy_numeric=self._encode_enemies(state),
            enemy_ids=[self._hash_id(enemy.enemy_id) for enemy in state.enemies],
            enemy_target_ids=self._encode_enemy_target_ids(state),
            enemy_intent_ids=[self._hash_id(enemy.intent_id) for enemy in state.enemies],
            enemy_buff_ids=[self._encode_mapping_ids(enemy.buffs) for enemy in state.enemies],
            enemy_buff_values=[self._encode_mapping_values(enemy.buffs) for enemy in state.enemies],
            hand_numeric=self._encode_hand(state),
            hand_card_ids=[self._hash_id(card.card_id) for card in state.hand],
            pile_numeric=self._encode_piles(state),
            history_numeric=self._encode_history(history),
            action_numeric=self._encode_actions(legal_actions),
            action_type_ids=[self._hash_id(action.action_type) for action in legal_actions],
            action_card_ids=[self._hash_id(action.card_id or action.potion_id or action.special_id) for action in legal_actions],
            action_card_indices=self._encode_action_card_indices(legal_actions),
            action_target_ids=self._encode_action_target_ids(legal_actions),
            behavior_action_index=0,
            fight_targets=[0.0, 0.0, 0.0],
            delta_targets=[0.0] * (3 + self._config.max_enemies * 2 + 3),
            sample_weight=1.0,
            fight_quality_score=0.0,
            behavior_ce_scale=1.0,
            old_logprob=0.0,
            old_value=0.0,
            old_intent_logprob=0.0,
            old_intent_value=0.0,
            ppo_return=0.0,
            ppo_advantage=0.0,
            turn_id=int((state.context.metadata or {}).get("turn_id", 0) or 0),
            turn_start_mask=1.0,
            active_intent=0,
            turn_return=0.0,
            turn_advantage=0.0,
            chosen_action_future_targets=[0.0] * self._config.future_summary_dim,
            submenu_confirm_target=0.0,
            submenu_has_confirm=1.0 if any(
                str(action.action_type or "").lower() in {"confirm_selection", "combat_confirm_selection"}
                for action in legal_actions
            ) else 0.0,
        )

    def _encode_future_summary_targets(self, raw_source: list[float] | dict[str, object]) -> list[float]:
        if isinstance(raw_source, list):
            values = [float(value) for value in raw_source[: self._config.future_summary_dim]]
            if len(values) < self._config.future_summary_dim:
                values.extend([0.0] * (self._config.future_summary_dim - len(values)))
            return values
        metadata = raw_source
        raw = metadata.get("future_summary_targets")
        if isinstance(raw, list):
            values = [float(value) for value in raw[: self._config.future_summary_dim]]
            if len(values) < self._config.future_summary_dim:
                values.extend([0.0] * (self._config.future_summary_dim - len(values)))
            return values
        return [
            float(metadata.get("future_death_risk_2t", 0.0) or 0.0),
            float(metadata.get("future_next_turn_power", 0.0) or 0.0),
            float(metadata.get("future_setup_value", 0.0) or 0.0),
        ][: self._config.future_summary_dim]

    def _encode_player(self, state: BattleState) -> list[float]:
        hp_ratio = _safe_ratio(state.player.hp, state.player.max_hp)
        return [
            float(state.player.hp),
            float(state.player.max_hp),
            hp_ratio,
            float(state.player.block),
            float(state.player.energy),
            float(len(state.player.potions)),
            *_mapping_to_slots(state.player.buffs, self._config.buff_slots),
            *_mapping_summary(state.player.resources),
            *_player_semantic_features(state.player.buffs),
        ]

    def _encode_static_numeric(self, state: BattleState) -> list[float]:
        metadata = state.context.metadata
        return [
            float(state.context.act),
            float(state.context.floor),
            float(len(state.living_enemies)),
            float(len(state.hand)),
            float(metadata.get("turn_id", metadata.get("round_number_raw", 0.0)) or 0.0),
            float(state.piles.draw_pile_size),
            float(state.piles.discard_pile_size),
            float(state.piles.exhaust_pile_size),
            float(len(state.context.relics)),
            float(len(state.player.buffs)),
            1.0 if str(metadata.get("state_type", "")) in {"hand_select", "card_select"} else 0.0,
            float(metadata.get("submenu_selected_count", 0.0) or 0.0),
            float(metadata.get("submenu_max_select", 0.0) or 0.0),
            float(metadata.get("submenu_remaining_slots", 0.0) or 0.0),
            1.0 if bool(metadata.get("submenu_can_confirm", False)) else 0.0,
            1.0 if bool(metadata.get("submenu_can_cancel", False)) else 0.0,
            float(metadata.get("submenu_selected_engine_count", 0.0) or 0.0),
            float(metadata.get("submenu_selected_payoff_count", 0.0) or 0.0),
            float(metadata.get("submenu_selected_resource_count", 0.0) or 0.0),
        ]

    def _encode_static_ids(self, state: BattleState) -> list[int]:
        return [
            self._hash_id(state.context.character_id),
            self._hash_id(state.context.encounter_class),
            self._hash_id(state.context.encounter_id),
        ]

    def _encode_enemies(self, state: BattleState) -> list[list[float]]:
        rows: list[list[float]] = []
        for enemy in state.enemies[: self._config.max_enemies]:
            rows.append(
                [
                    float(enemy.hp),
                    float(enemy.max_hp),
                    _safe_ratio(enemy.hp, enemy.max_hp),
                    float(enemy.block),
                    1.0 if enemy.alive else 0.0,
                    float(len(enemy.tags)),
                    *_mapping_to_slots(enemy.buffs, self._config.buff_slots),
                ]
            )
        return rows

    def _encode_hand(self, state: BattleState) -> list[list[float]]:
        rows: list[list[float]] = []
        for card in state.hand[: self._config.max_hand_cards]:
            rows.append(
                [
                    float(card.cost_now),
                    float(card.damage_now),
                    float(card.block_now),
                    float(card.magic_now),
                    1.0 if card.is_upgraded else 0.0,
                    1.0 if card.retain else 0.0,
                    1.0 if card.exhaust else 0.0,
                    1.0 if card.ethereal else 0.0,
                    float(len(card.tags)),
                    *_card_semantic_flags(card.card_id, card.tags),
                ]
            )
        return rows

    def _encode_id_list(self, values: Iterable[str]) -> list[int]:
        return [self._hash_id(str(value)) for value in values if str(value)]

    def _encode_value_list(self, values: Iterable[float]) -> list[float]:
        return [float(value) for value in values]

    def _encode_mapping_ids(self, mapping: dict[str, float]) -> list[int]:
        return [self._hash_id(key) for key, _ in sorted(mapping.items())]

    def _encode_mapping_values(self, mapping: dict[str, float]) -> list[float]:
        return [float(value) for _, value in sorted(mapping.items())]

    def _encode_piles(self, state: BattleState) -> list[float]:
        return [
            float(state.piles.draw_pile_size),
            float(state.piles.discard_pile_size),
            float(state.piles.exhaust_pile_size),
            float(state.piles.attack_count),
            float(state.piles.skill_count),
            float(state.piles.power_count),
            float(sum(state.piles.key_card_counts.values())),
            float(sum(state.piles.archetype_stats.values())),
            *_semantic_group_counts(state.piles.key_card_counts),
        ]

    def _encode_history(self, history: list[HistoryStep]) -> list[list[float]]:
        tokens: list[list[float]] = []
        for step in history[-self._config.history_steps :]:
            tokens.append(self._history_token(step))
        return tokens

    def _history_token(self, step: HistoryStep) -> list[float]:
        if step.history_token:
            return list(step.history_token)
        if step.state is None or step.action is None:
            return [0.0] * HISTORY_TOKEN_DIM
        feel_no_pain = float(step.state.player.buffs.get("FEEL_NO_PAIN_POWER", 0.0) or 0.0)
        dark_embrace = float(step.state.player.buffs.get("DARK_EMBRACE_POWER", 0.0) or 0.0)
        pyre_power = float(step.state.player.buffs.get("PYRE_POWER", 0.0) or 0.0)
        engine_total = feel_no_pain + dark_embrace + pyre_power
        action_id = step.action.card_id or step.action.special_id or ""
        action_flags = _action_semantic_flags(step.action)
        total_enemy_hp_ratio = sum(_safe_ratio(enemy.hp, enemy.max_hp) for enemy in step.state.enemies)
        return [
            _safe_ratio(step.state.player.hp, step.state.player.max_hp),
            float(step.state.player.block),
            float(step.state.player.energy),
            total_enemy_hp_ratio,
            float(len(step.state.hand)),
            float(step.state.piles.draw_pile_size),
            float(step.state.piles.discard_pile_size),
            float(step.delta.self_hp),
            float(step.delta.self_block),
            float(step.delta.self_energy),
            float(sum(step.delta.enemy_hp)),
            float(sum(step.delta.enemy_block)),
            float(step.delta.hand_size),
            float(step.delta.draw_pile_size),
            float(step.delta.discard_pile_size),
            float(self._hash_id(step.action.action_type)) / float(self._config.id_hash_buckets),
            float(self._hash_id(action_id)) / float(self._config.id_hash_buckets),
            float(step.state.piles.exhaust_pile_size),
            float(engine_total),
            float(feel_no_pain),
            float(dark_embrace),
            float(pyre_power),
            *action_flags,
        ]

    def encode_history_step_token(self, state: BattleState, action: LegalAction, delta: TransitionDelta) -> list[float]:
        """预编码 history token，供样本池长期存放时减少完整 state/action 挂载。"""
        return self._history_token(HistoryStep(state=state, action=action, delta=delta))

    def _encode_actions(self, actions: list[LegalAction]) -> list[list[float]]:
        rows: list[list[float]] = []
        for action in actions:
            target = action.target_summary
            rows.append(
                [
                    1.0 if action.can_execute else 0.0,
                    float(action.cost_now),
                    float(action.damage_now),
                    float(action.block_now),
                    float(action.magic_now),
                    float(len(action.tags)),
                    1.0 if target is not None else 0.0,
                    float(target.hp if target else 0.0),
                    float(target.max_hp if target else 0.0),
                    _safe_ratio(target.hp, target.max_hp) if target else 0.0,
                    float(target.block if target else 0.0),
                    1.0 if (target.alive if target else False) else 0.0,
                    float(sum((target.buffs if target else {}).values())),
                    *_action_semantic_flags(action),
                ]
            )
        return rows

    def _encode_enemy_target_ids(self, state: BattleState) -> list[int]:
        values: list[int] = []
        for enemy in state.enemies[: self._config.max_enemies]:
            raw_target = str(enemy.target_key or enemy.enemy_id or "")
            values.append(self._hash_id(raw_target) if raw_target else 0)
        return values

    def _encode_action_card_indices(self, actions: list[LegalAction]) -> list[int]:
        indices: list[int] = []
        for action in actions:
            card_index, _ = _parse_action_instance_id(action.action_id)
            indices.append(card_index if card_index is not None else -1)
        return indices

    def _encode_action_target_ids(self, actions: list[LegalAction]) -> list[int]:
        values: list[int] = []
        for action in actions:
            _, parsed_target_id = _parse_action_instance_id(action.action_id)
            raw_target = parsed_target_id or str(action.target_id or "")
            values.append(self._hash_id(raw_target) if raw_target else 0)
        return values

    def _encode_delta(self, delta: TransitionDelta) -> list[float]:
        values = [
            float(delta.self_hp),
            float(delta.self_block),
            float(delta.self_energy),
        ]
        values.extend(_pad(delta.enemy_hp, self._config.max_enemies))
        values.extend(_pad(delta.enemy_block, self._config.max_enemies))
        values.extend(
            [
                float(delta.hand_size),
                float(delta.draw_pile_size),
                float(delta.discard_pile_size),
            ]
        )
        return values

    def _hash_id(self, value: str) -> int:
        raw = value.encode("utf-8", errors="ignore")
        digest = hashlib.blake2b(raw, digest_size=8).digest()
        return int.from_bytes(digest, "little") % self._config.id_hash_buckets


def _mapping_to_slots(values: dict[str, float], width: int) -> list[float]:
    slots = [0.0] * width
    for key, value in values.items():
        raw = key.encode("utf-8", errors="ignore")
        digest = hashlib.blake2b(raw, digest_size=8).digest()
        slot = int.from_bytes(digest, "little") % width
        slots[slot] += float(value)
    return slots


def _mapping_summary(values: dict[str, float]) -> list[float]:
    numbers = list(values.values())
    if not numbers:
        return [0.0, 0.0, 0.0]
    return [float(sum(numbers)), float(max(numbers)), float(min(numbers))]


def _pad(values: Iterable[float], width: int) -> list[float]:
    result = [float(value) for value in values]
    if len(result) >= width:
        return result[:width]
    return result + [0.0] * (width - len(result))


def _safe_ratio(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return float(value) / float(total)


def _normalize_card_id(value: str) -> str:
    return str(value or "").upper().replace("+", "").strip()


def _parse_action_instance_id(action_id: str) -> tuple[int | None, str]:
    text = str(action_id or "").strip()
    if "|" not in text:
        return None, ""
    parts = text.split("|")
    card_index: int | None = None
    if len(parts) >= 3 and parts[2].strip():
        try:
            card_index = int(parts[2])
        except ValueError:
            card_index = None
    target_id = parts[3].strip() if len(parts) >= 4 else ""
    return card_index, target_id


def _tag_set(tags: Iterable[str]) -> set[str]:
    return {str(tag or "").strip().lower() for tag in tags if str(tag or "").strip()}


def _card_semantic_flags(card_id: str, tags: Iterable[str]) -> list[float]:
    normalized_id = _normalize_card_id(card_id)
    tag_set = _tag_set(tags)
    return [
        1.0 if "attack" in tag_set else 0.0,
        1.0 if "skill" in tag_set else 0.0,
        1.0 if "power" in tag_set else 0.0,
        1.0 if "can_play" in tag_set else 0.0,
        1.0 if normalized_id in _ENGINE_POWER_IDS else 0.0,
        1.0 if normalized_id in _EXHAUST_ENABLER_IDS else 0.0,
        1.0 if normalized_id in _EXHAUST_PAYOFF_IDS else 0.0,
        1.0 if normalized_id in _RESOURCE_CARD_IDS else 0.0,
    ]


def _action_semantic_flags(action: LegalAction) -> list[float]:
    normalized_id = _normalize_card_id(action.card_id or action.special_id)
    tag_set = _tag_set(action.tags)
    is_submenu_select = action.action_type in {"select_hand_card", "select_card", "select_card_option", "combat_select_card"}
    is_submenu_confirm = action.action_type in {"confirm_selection", "combat_confirm_selection"}
    return [
        1.0 if action.action_type == "play_card" else 0.0,
        1.0 if action.action_type == "end_turn" else 0.0,
        1.0 if is_submenu_select else 0.0,
        1.0 if is_submenu_confirm else 0.0,
        1.0 if "attack" in tag_set else 0.0,
        1.0 if "skill" in tag_set else 0.0,
        1.0 if "power" in tag_set else 0.0,
        1.0 if normalized_id in _ENGINE_POWER_IDS else 0.0,
        1.0 if normalized_id in _EXHAUST_ENABLER_IDS else 0.0,
        1.0 if normalized_id in _EXHAUST_PAYOFF_IDS else 0.0,
        1.0 if normalized_id in _RESOURCE_CARD_IDS else 0.0,
        1.0 if action.target_summary is not None else 0.0,
        1.0 if float(action.cost_now) <= 0.0 else 0.0,
    ]


def _semantic_group_counts(values: dict[str, int]) -> list[float]:
    if not values:
        return [0.0] * PILE_SEMANTIC_DIM
    engine = 0.0
    enabler = 0.0
    payoff = 0.0
    resource = 0.0
    for raw_id, count in values.items():
        normalized_id = _normalize_card_id(raw_id)
        numeric_count = float(count or 0.0)
        if normalized_id in _ENGINE_POWER_IDS:
            engine += numeric_count
        if normalized_id in _EXHAUST_ENABLER_IDS:
            enabler += numeric_count
        if normalized_id in _EXHAUST_PAYOFF_IDS:
            payoff += numeric_count
        if normalized_id in _RESOURCE_CARD_IDS:
            resource += numeric_count
    return [engine, enabler, payoff, resource]


def _semantic_counts_for_cards(card_ids: Iterable[str]) -> dict[str, float]:
    counts = {"engine": 0.0, "enabler": 0.0, "payoff": 0.0, "resource": 0.0}
    for raw_id in card_ids:
        normalized_id = _normalize_card_id(raw_id)
        if normalized_id in _ENGINE_POWER_IDS:
            counts["engine"] += 1.0
        if normalized_id in _EXHAUST_ENABLER_IDS:
            counts["enabler"] += 1.0
        if normalized_id in _EXHAUST_PAYOFF_IDS:
            counts["payoff"] += 1.0
        if normalized_id in _RESOURCE_CARD_IDS:
            counts["resource"] += 1.0
    return counts


def _player_semantic_features(buffs: dict[str, float]) -> list[float]:
    feel_no_pain = float(buffs.get("FEEL_NO_PAIN_POWER", 0.0) or 0.0)
    dark_embrace = float(buffs.get("DARK_EMBRACE_POWER", 0.0) or 0.0)
    pyre_power = float(buffs.get("PYRE_POWER", 0.0) or 0.0)
    engine_total = feel_no_pain + dark_embrace + pyre_power
    return [engine_total, feel_no_pain, dark_embrace, pyre_power]
