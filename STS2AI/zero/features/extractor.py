from __future__ import annotations

"""Feature extraction from structured domain objects to fixed-width tensors."""

import hashlib
from dataclasses import dataclass
from typing import Iterable

from ..config import EncoderConfig
from ..domain import BattleState, HistoryStep, LegalAction, TrainingSample, TransitionDelta


@dataclass(slots=True)
class EncodedSample:
    player_numeric: list[float]
    static_numeric: list[float]
    static_ids: list[int]
    enemy_numeric: list[list[float]]
    enemy_ids: list[int]
    enemy_intent_ids: list[int]
    hand_numeric: list[list[float]]
    hand_card_ids: list[int]
    pile_numeric: list[float]
    history_numeric: list[list[float]]
    action_numeric: list[list[float]]
    action_type_ids: list[int]
    action_card_ids: list[int]
    behavior_action_index: int
    teacher_policy: list[float] | None
    teacher_best_action_index: int
    fight_targets: list[float]
    delta_targets: list[float]
    uncertainty_target: float
    sample_weight: float


class FeatureExtractor:
    def __init__(self, config: EncoderConfig):
        self._config = config

    def encode_sample(self, sample: TrainingSample) -> EncodedSample:
        return EncodedSample(
            player_numeric=self._encode_player(sample.state),
            static_numeric=self._encode_static_numeric(sample.state),
            static_ids=self._encode_static_ids(sample.state),
            enemy_numeric=self._encode_enemies(sample.state),
            enemy_ids=[self._hash_id(enemy.enemy_id) for enemy in sample.state.enemies],
            enemy_intent_ids=[self._hash_id(enemy.intent_id) for enemy in sample.state.enemies],
            hand_numeric=self._encode_hand(sample.state),
            hand_card_ids=[self._hash_id(card.card_id) for card in sample.state.hand],
            pile_numeric=self._encode_piles(sample.state),
            history_numeric=self._encode_history(sample.history),
            action_numeric=self._encode_actions(sample.legal_actions),
            action_type_ids=[self._hash_id(action.action_type) for action in sample.legal_actions],
            action_card_ids=[self._hash_id(action.card_id or action.potion_id or action.special_id) for action in sample.legal_actions],
            behavior_action_index=sample.behavior_action_index,
            teacher_policy=sample.teacher_label.policy if sample.teacher_label else None,
            teacher_best_action_index=sample.teacher_label.best_action_index if sample.teacher_label else -1,
            fight_targets=[
                sample.fight_label.fight_win,
                sample.fight_label.enemy_hp_fraction_dealt,
                sample.fight_label.self_hp_fraction_remaining,
            ],
            delta_targets=self._encode_delta(sample.delta),
            uncertainty_target=float(sample.metadata.get("uncertainty_target", 0.0) or 0.0),
            sample_weight=max(0.1, sample.fight_label.fight_score + 0.5),
        )

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
        ]

    def _encode_static_numeric(self, state: BattleState) -> list[float]:
        return [
            float(state.context.act),
            float(state.context.floor),
            float(len(state.context.relics)),
            float(len(state.context.fixed_powers)),
            float(len(state.living_enemies)),
            float(len(state.hand)),
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
                ]
            )
        return rows

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
        ]

    def _encode_history(self, history: list[HistoryStep]) -> list[list[float]]:
        tokens: list[list[float]] = []
        for step in history[-self._config.history_steps :]:
            tokens.append(self._history_token(step))
        return tokens

    def _history_token(self, step: HistoryStep) -> list[float]:
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
            float(self._hash_id(step.action.card_id or step.action.special_id)) / float(self._config.id_hash_buckets),
        ]

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
                ]
            )
        return rows

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
