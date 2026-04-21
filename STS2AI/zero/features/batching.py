from __future__ import annotations

"""Batch collation with padding and structural masks."""

from dataclasses import dataclass

import torch

from ..config import EncoderConfig
from .extractor import EncodedSample, FeatureExtractor


@dataclass(slots=True)
class TensorBatch:
    player_numeric: torch.Tensor
    static_numeric: torch.Tensor
    static_ids: torch.Tensor
    enemy_numeric: torch.Tensor
    enemy_ids: torch.Tensor
    enemy_intent_ids: torch.Tensor
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
    action_mask: torch.Tensor
    behavior_action_index: torch.Tensor
    search_policy: torch.Tensor
    search_policy_mask: torch.Tensor
    search_best_action_index: torch.Tensor
    search_ranking_margin: torch.Tensor
    fight_targets: torch.Tensor
    delta_targets: torch.Tensor
    uncertainty_target: torch.Tensor
    sample_weight: torch.Tensor
    fight_quality_score: torch.Tensor
    behavior_ce_scale: torch.Tensor

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

    def _collate_encoded(self, encoded: list[EncodedSample]) -> TensorBatch:
        action_width = max(1, max(len(item.action_numeric) for item in encoded))
        return TensorBatch(
            player_numeric=_to_tensor_2d([item.player_numeric for item in encoded]),
            static_numeric=_to_tensor_2d([item.static_numeric for item in encoded]),
            static_ids=torch.tensor([item.static_ids for item in encoded], dtype=torch.long),
            enemy_numeric=_to_tensor_3d(
                [item.enemy_numeric for item in encoded],
                self._config.max_enemies,
                len(encoded[0].enemy_numeric[0]) if encoded and encoded[0].enemy_numeric else 6 + self._config.buff_slots,
            ),
            enemy_ids=_to_tensor_2d_int([item.enemy_ids for item in encoded], self._config.max_enemies),
            enemy_intent_ids=_to_tensor_2d_int([item.enemy_intent_ids for item in encoded], self._config.max_enemies),
            enemy_mask=_enemy_mask_from_rows([item.enemy_numeric for item in encoded], self._config.max_enemies),
            hand_numeric=_to_tensor_3d(
                [item.hand_numeric for item in encoded],
                self._config.max_hand_cards,
                len(encoded[0].hand_numeric[0]) if encoded and encoded[0].hand_numeric else 9,
            ),
            hand_card_ids=_to_tensor_2d_int([item.hand_card_ids for item in encoded], self._config.max_hand_cards),
            hand_mask=_mask_from_lengths([len(item.hand_numeric) for item in encoded], self._config.max_hand_cards),
            pile_numeric=_to_tensor_2d([item.pile_numeric for item in encoded]),
            history_numeric=_to_tensor_3d(
                [item.history_numeric for item in encoded],
                self._config.history_steps,
                len(encoded[0].history_numeric[0]) if encoded and encoded[0].history_numeric else 17,
            ),
            history_mask=_mask_from_lengths([len(item.history_numeric) for item in encoded], self._config.history_steps),
            action_numeric=_to_tensor_3d(
                [item.action_numeric for item in encoded],
                action_width,
                len(encoded[0].action_numeric[0]) if encoded and encoded[0].action_numeric else 13,
            ),
            action_type_ids=_to_tensor_2d_int(
                [item.action_type_ids for item in encoded],
                action_width,
            ),
            action_card_ids=_to_tensor_2d_int(
                [item.action_card_ids for item in encoded],
                action_width,
            ),
            action_mask=_mask_from_lengths(
                [len(item.action_numeric) for item in encoded],
                action_width,
            ),
            behavior_action_index=torch.tensor([item.behavior_action_index for item in encoded], dtype=torch.long),
            search_policy=_build_search_policy(encoded),
            search_policy_mask=torch.tensor(
                [1.0 if item.search_policy is not None else 0.0 for item in encoded], dtype=torch.float32
            ),
            search_best_action_index=torch.tensor([item.search_best_action_index for item in encoded], dtype=torch.long),
            search_ranking_margin=torch.tensor([item.search_ranking_margin for item in encoded], dtype=torch.float32),
            fight_targets=_to_tensor_2d([item.fight_targets for item in encoded]),
            delta_targets=_to_tensor_2d([item.delta_targets for item in encoded]),
            uncertainty_target=torch.tensor([item.uncertainty_target for item in encoded], dtype=torch.float32),
            sample_weight=torch.tensor([item.sample_weight for item in encoded], dtype=torch.float32),
            fight_quality_score=torch.tensor([item.fight_quality_score for item in encoded], dtype=torch.float32),
            behavior_ce_scale=torch.tensor([item.behavior_ce_scale for item in encoded], dtype=torch.float32),
        )


def _build_search_policy(encoded: list[EncodedSample]) -> torch.Tensor:
    width = max(len(item.action_numeric) for item in encoded)
    result = []
    for item in encoded:
        if item.search_policy is None:
            result.append([0.0] * width)
            continue
        values = list(item.search_policy[:width])
        if len(values) < width:
            values.extend([0.0] * (width - len(values)))
        result.append(values)
    return torch.tensor(result, dtype=torch.float32)


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


def _to_tensor_3d(rows: list[list[list[float]]], width: int, feature_dim: int) -> torch.Tensor:
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
