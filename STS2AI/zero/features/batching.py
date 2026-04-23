from __future__ import annotations

"""Batch collation with padding and structural masks."""

from dataclasses import dataclass

import torch

from ..config import EncoderConfig
from .extractor import ACTION_SEMANTIC_DIM, EncodedSample, FeatureExtractor, HAND_SEMANTIC_DIM, HISTORY_TOKEN_DIM, PILE_SEMANTIC_DIM, PLAYER_SEMANTIC_DIM


@dataclass(slots=True)
class TensorBatch:
    player_numeric: torch.Tensor
    static_numeric: torch.Tensor
    static_ids: torch.Tensor
    relic_ids: torch.Tensor
    relic_mask: torch.Tensor
    deck_card_ids: torch.Tensor
    deck_card_mask: torch.Tensor
    potion_ids: torch.Tensor
    potion_mask: torch.Tensor
    draw_pile_ids: torch.Tensor
    draw_pile_mask: torch.Tensor
    discard_pile_ids: torch.Tensor
    discard_pile_mask: torch.Tensor
    exhaust_pile_ids: torch.Tensor
    exhaust_pile_mask: torch.Tensor
    player_buff_ids: torch.Tensor
    player_buff_values: torch.Tensor
    player_buff_mask: torch.Tensor
    enemy_numeric: torch.Tensor
    enemy_ids: torch.Tensor
    enemy_target_ids: torch.Tensor
    enemy_intent_ids: torch.Tensor
    enemy_buff_ids: torch.Tensor
    enemy_buff_values: torch.Tensor
    enemy_buff_mask: torch.Tensor
    enemy_mask: torch.Tensor
    hand_numeric: torch.Tensor
    hand_card_ids: torch.Tensor
    hand_mask: torch.Tensor
    pile_numeric: torch.Tensor
    history_numeric: torch.Tensor
    history_mask: torch.Tensor
    action_numeric: torch.Tensor
    action_type_ids: torch.Tensor
    action_card_ids: torch.Tensor
    action_card_indices: torch.Tensor
    action_target_ids: torch.Tensor
    action_mask: torch.Tensor
    behavior_action_index: torch.Tensor
    fight_targets: torch.Tensor
    delta_targets: torch.Tensor
    sample_weight: torch.Tensor
    fight_quality_score: torch.Tensor
    behavior_ce_scale: torch.Tensor
    old_logprob: torch.Tensor
    old_value: torch.Tensor
    old_intent_logprob: torch.Tensor
    old_intent_value: torch.Tensor
    ppo_return: torch.Tensor
    ppo_advantage: torch.Tensor
    turn_id: torch.Tensor
    turn_start_mask: torch.Tensor
    active_intent: torch.Tensor
    turn_return: torch.Tensor
    turn_advantage: torch.Tensor
    chosen_action_future_targets: torch.Tensor
    submenu_confirm_target: torch.Tensor
    submenu_has_confirm: torch.Tensor

    def to(self, device: torch.device | str) -> "TensorBatch":
        return TensorBatch(**{field: getattr(self, field).to(device) for field in self.__dataclass_fields__})


class BatchCollator:
    def __init__(self, config: EncoderConfig):
        self._config = config
        self._extractor = FeatureExtractor(config)

    def collate(self, samples: list) -> TensorBatch:
        encoded = [self._extractor.encode_sample(sample) for sample in samples]
        return self._collate_encoded(encoded)

    def collate_inference(self, state, history, legal_actions) -> TensorBatch:
        encoded = [self._extractor.encode_inference(state, history, legal_actions)]
        return self._collate_encoded(encoded)

    def collate_inference_batch(self, requests: list[tuple]) -> TensorBatch:
        encoded = [
            self._extractor.encode_inference(state, history, legal_actions)
            for state, history, legal_actions in requests
        ]
        return self._collate_encoded(encoded)

    def _collate_encoded(self, encoded: list[EncodedSample]) -> TensorBatch:
        action_width = max(1, max(len(item.action_numeric) for item in encoded))
        relic_width = max(1, max(len(item.relic_ids) for item in encoded))
        deck_width = max(1, max(len(item.deck_card_ids) for item in encoded))
        potion_width = max(1, max(len(item.potion_ids) for item in encoded))
        pile_width = max(
            1,
            max(
                max(len(item.draw_pile_ids), len(item.discard_pile_ids), len(item.exhaust_pile_ids))
                for item in encoded
            ),
        )
        player_buff_width = max(1, max(len(item.player_buff_ids) for item in encoded))
        enemy_buff_width = max(
            1,
            max(
                max((len(buff_ids) for buff_ids in item.enemy_buff_ids), default=0)
                for item in encoded
            ),
        )
        enemy_feature_dim = _infer_feature_dim([item.enemy_numeric for item in encoded], default=6 + self._config.buff_slots)
        hand_feature_dim = _infer_feature_dim([item.hand_numeric for item in encoded], default=9 + HAND_SEMANTIC_DIM)
        history_feature_dim = _infer_feature_dim([item.history_numeric for item in encoded], default=HISTORY_TOKEN_DIM)
        action_feature_dim = _infer_feature_dim([item.action_numeric for item in encoded], default=13 + ACTION_SEMANTIC_DIM)
        return TensorBatch(
            player_numeric=_to_tensor_2d([item.player_numeric for item in encoded]),
            static_numeric=_to_tensor_2d([item.static_numeric for item in encoded]),
            static_ids=torch.tensor([item.static_ids for item in encoded], dtype=torch.long),
            relic_ids=_to_tensor_2d_int([item.relic_ids for item in encoded], relic_width),
            relic_mask=_mask_from_lengths([len(item.relic_ids) for item in encoded], relic_width),
            deck_card_ids=_to_tensor_2d_int([item.deck_card_ids for item in encoded], deck_width),
            deck_card_mask=_mask_from_lengths([len(item.deck_card_ids) for item in encoded], deck_width),
            potion_ids=_to_tensor_2d_int([item.potion_ids for item in encoded], potion_width),
            potion_mask=_mask_from_lengths([len(item.potion_ids) for item in encoded], potion_width),
            draw_pile_ids=_to_tensor_2d_int([item.draw_pile_ids for item in encoded], pile_width),
            draw_pile_mask=_mask_from_lengths([len(item.draw_pile_ids) for item in encoded], pile_width),
            discard_pile_ids=_to_tensor_2d_int([item.discard_pile_ids for item in encoded], pile_width),
            discard_pile_mask=_mask_from_lengths([len(item.discard_pile_ids) for item in encoded], pile_width),
            exhaust_pile_ids=_to_tensor_2d_int([item.exhaust_pile_ids for item in encoded], pile_width),
            exhaust_pile_mask=_mask_from_lengths([len(item.exhaust_pile_ids) for item in encoded], pile_width),
            player_buff_ids=_to_tensor_2d_int([item.player_buff_ids for item in encoded], player_buff_width),
            player_buff_values=_to_tensor_2d_float([item.player_buff_values for item in encoded], player_buff_width),
            player_buff_mask=_mask_from_lengths([len(item.player_buff_ids) for item in encoded], player_buff_width),
            enemy_numeric=_to_tensor_3d(
                [item.enemy_numeric for item in encoded],
                self._config.max_enemies,
                enemy_feature_dim,
            ),
            enemy_ids=_to_tensor_2d_int([item.enemy_ids for item in encoded], self._config.max_enemies),
            enemy_target_ids=_to_tensor_2d_int([item.enemy_target_ids for item in encoded], self._config.max_enemies),
            enemy_intent_ids=_to_tensor_2d_int([item.enemy_intent_ids for item in encoded], self._config.max_enemies),
            enemy_buff_ids=_to_tensor_3d_int([item.enemy_buff_ids for item in encoded], self._config.max_enemies, enemy_buff_width),
            enemy_buff_values=_to_tensor_3d_float([item.enemy_buff_values for item in encoded], self._config.max_enemies, enemy_buff_width),
            enemy_buff_mask=_mask_from_nested_lengths([item.enemy_buff_ids for item in encoded], self._config.max_enemies, enemy_buff_width),
            enemy_mask=_enemy_mask_from_rows([item.enemy_numeric for item in encoded], self._config.max_enemies),
            hand_numeric=_to_tensor_3d(
                [item.hand_numeric for item in encoded],
                self._config.max_hand_cards,
                hand_feature_dim,
            ),
            hand_card_ids=_to_tensor_2d_int([item.hand_card_ids for item in encoded], self._config.max_hand_cards),
            hand_mask=_mask_from_lengths([len(item.hand_numeric) for item in encoded], self._config.max_hand_cards),
            pile_numeric=_to_tensor_2d([item.pile_numeric for item in encoded]),
            history_numeric=_to_tensor_3d(
                [item.history_numeric for item in encoded],
                self._config.history_steps,
                history_feature_dim,
            ),
            history_mask=_mask_from_lengths([len(item.history_numeric) for item in encoded], self._config.history_steps),
            action_numeric=_to_tensor_3d(
                [item.action_numeric for item in encoded],
                action_width,
                action_feature_dim,
            ),
            action_type_ids=_to_tensor_2d_int(
                [item.action_type_ids for item in encoded],
                action_width,
            ),
            action_card_ids=_to_tensor_2d_int(
                [item.action_card_ids for item in encoded],
                action_width,
            ),
            action_card_indices=_to_tensor_2d_int_fill(
                [item.action_card_indices for item in encoded],
                action_width,
                fill_value=-1,
            ),
            action_target_ids=_to_tensor_2d_int(
                [item.action_target_ids for item in encoded],
                action_width,
            ),
            action_mask=_mask_from_lengths(
                [len(item.action_numeric) for item in encoded],
                action_width,
            ),
            behavior_action_index=torch.tensor([item.behavior_action_index for item in encoded], dtype=torch.long),
            fight_targets=_to_tensor_2d([item.fight_targets for item in encoded]),
            delta_targets=_to_tensor_2d([item.delta_targets for item in encoded]),
            sample_weight=torch.tensor([item.sample_weight for item in encoded], dtype=torch.float32),
            fight_quality_score=torch.tensor([item.fight_quality_score for item in encoded], dtype=torch.float32),
            behavior_ce_scale=torch.tensor([item.behavior_ce_scale for item in encoded], dtype=torch.float32),
            old_logprob=torch.tensor([item.old_logprob for item in encoded], dtype=torch.float32),
            old_value=torch.tensor([item.old_value for item in encoded], dtype=torch.float32),
            old_intent_logprob=torch.tensor([item.old_intent_logprob for item in encoded], dtype=torch.float32),
            old_intent_value=torch.tensor([item.old_intent_value for item in encoded], dtype=torch.float32),
            ppo_return=torch.tensor([item.ppo_return for item in encoded], dtype=torch.float32),
            ppo_advantage=torch.tensor([item.ppo_advantage for item in encoded], dtype=torch.float32),
            turn_id=torch.tensor([item.turn_id for item in encoded], dtype=torch.long),
            turn_start_mask=torch.tensor([item.turn_start_mask for item in encoded], dtype=torch.float32),
            active_intent=torch.tensor([item.active_intent for item in encoded], dtype=torch.long),
            turn_return=torch.tensor([item.turn_return for item in encoded], dtype=torch.float32),
            turn_advantage=torch.tensor([item.turn_advantage for item in encoded], dtype=torch.float32),
            chosen_action_future_targets=_to_tensor_2d([item.chosen_action_future_targets for item in encoded]),
            submenu_confirm_target=torch.tensor([item.submenu_confirm_target for item in encoded], dtype=torch.float32),
            submenu_has_confirm=torch.tensor([item.submenu_has_confirm for item in encoded], dtype=torch.float32),
        )


def _to_tensor_2d(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def _to_tensor_2d_int(rows: list[list[int]], width: int) -> torch.Tensor:
    padded = []
    for row in rows:
        values = list(row[:width])
        if len(values) < width:
            values.extend([0] * (width - len(values)))
        padded.append(values)
    return torch.tensor(padded, dtype=torch.long)


def _to_tensor_2d_int_fill(rows: list[list[int]], width: int, *, fill_value: int) -> torch.Tensor:
    padded = []
    for row in rows:
        values = list(row[:width])
        if len(values) < width:
            values.extend([fill_value] * (width - len(values)))
        padded.append(values)
    return torch.tensor(padded, dtype=torch.long)


def _to_tensor_2d_float(rows: list[list[float]], width: int) -> torch.Tensor:
    padded = []
    for row in rows:
        values = [float(value) for value in row[:width]]
        if len(values) < width:
            values.extend([0.0] * (width - len(values)))
        padded.append(values)
    return torch.tensor(padded, dtype=torch.float32)


def _to_tensor_3d(rows: list[list[list[float]]], width: int, feature_dim: int) -> torch.Tensor:
    tensor = torch.zeros((len(rows), width, feature_dim), dtype=torch.float32)
    for batch_index, row in enumerate(rows):
        for item_index, item in enumerate(row[:width]):
            limit = min(feature_dim, len(item))
            if limit <= 0:
                continue
            tensor[batch_index, item_index, :limit] = torch.tensor(item[:limit], dtype=torch.float32)
    return tensor


def _to_tensor_3d_int(rows: list[list[list[int]]], width: int, feature_dim: int) -> torch.Tensor:
    tensor = torch.zeros((len(rows), width, feature_dim), dtype=torch.long)
    for batch_index, row in enumerate(rows):
        for item_index, item in enumerate(row[:width]):
            limit = min(feature_dim, len(item))
            if limit <= 0:
                continue
            tensor[batch_index, item_index, :limit] = torch.tensor(item[:limit], dtype=torch.long)
    return tensor


def _to_tensor_3d_float(rows: list[list[list[float]]], width: int, feature_dim: int) -> torch.Tensor:
    tensor = torch.zeros((len(rows), width, feature_dim), dtype=torch.float32)
    for batch_index, row in enumerate(rows):
        for item_index, item in enumerate(row[:width]):
            limit = min(feature_dim, len(item))
            if limit <= 0:
                continue
            tensor[batch_index, item_index, :limit] = torch.tensor(item[:limit], dtype=torch.float32)
    return tensor


def _mask_from_lengths(lengths: list[int], width: int) -> torch.Tensor:
    rows = []
    for length in lengths:
        rows.append([1.0] * min(length, width) + [0.0] * max(0, width - min(length, width)))
    return torch.tensor(rows, dtype=torch.float32)


def _enemy_mask_from_rows(rows: list[list[list[float]]], width: int) -> torch.Tensor:
    mask_rows = []
    for enemies in rows:
        row = []
        for enemy in enemies[:width]:
            alive = enemy[4] if len(enemy) > 4 else 1.0
            row.append(1.0 if alive > 0.5 else 0.0)
        while len(row) < width:
            row.append(0.0)
        mask_rows.append(row)
    return torch.tensor(mask_rows, dtype=torch.float32)


def _mask_from_nested_lengths(rows: list[list[list[int]]], outer_width: int, inner_width: int) -> torch.Tensor:
    tensor = torch.zeros((len(rows), outer_width, inner_width), dtype=torch.float32)
    for batch_index, row in enumerate(rows):
        for item_index, item in enumerate(row[:outer_width]):
            limit = min(inner_width, len(item))
            if limit > 0:
                tensor[batch_index, item_index, :limit] = 1.0
    return tensor


def _infer_feature_dim(rows: list[list[list[float]]], *, default: int) -> int:
    feature_dim = 0
    for row in rows:
        for item in row:
            feature_dim = max(feature_dim, len(item))
    return feature_dim or default
