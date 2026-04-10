from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from combat_nn import _get_power_amount
from rl_encoder_v2 import (
    CARD_AUX_DIM,
    ENEMY_AUX_DIM,
    Vocab,
    _cached_card_encoding,
    _cached_monster_idx,
    _enemy_aux_features,
    _extract_player,
    _safe_float,
    _safe_int,
    load_vocab,
)

LEAF_DATASET_SCHEMA_VERSION = 2
LEAF_SCORE_V1_COEFFICIENTS = {
    "win_prob": 2.0,
    "win_offset": -1.0,
    "hp_loss_ratio": -0.3,
    "boss_damage_ratio": 0.2,
}
LEAF_SCORE_SOFTCLIP_TEMPERATURE = 1.0
LEAF_SCORE_TARGETS = (
    "score_v1_clipped",
    "score_v1_raw",
    "search_value_softclip",
)
DEFAULT_LEAF_SCORE_TARGET = "score_v1_raw"
BOSS_BUCKET_FEATURES = (
    "boss_bucket_unknown",
    "boss_bucket_ceremonial_beast",
    "boss_bucket_the_kin",
    "boss_bucket_vantom",
    "boss_bucket_other",
)
TOKEN_TYPE_TO_IDX = {
    "cls": 0,
    "player": 1,
    "enemy": 2,
    "hand": 3,
    "draw": 4,
    "discard": 5,
    "exhaust": 6,
}
MAX_LEAF_TOKENS = 96
SIGNATURE_FEATURE_KEYS = [
    "round_number",
    "player_hp_frac",
    "player_block_frac",
    "player_energy",
    "hand_size",
    "draw_size",
    "discard_size",
    "exhaust_size",
    "enemy_count",
    "enemy_hp_frac",
    "enemy_max_hp_total_log",
    "potion_count_total",
    "buff_count_total",
    "debuff_count_total",
    "incoming_intent_damage",
]
logger = logging.getLogger("boss_leaf_evaluator")


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _boss_bucket_index(token: Any) -> int:
    normalized = _lower(token)
    if not normalized:
        return 0
    if normalized == "ceremonial_beast":
        return 1
    if normalized == "the_kin":
        return 2
    if normalized == "vantom":
        return 3
    return 4


def _boss_bucket_vector(token: Any) -> list[float]:
    values = [0.0] * len(BOSS_BUCKET_FEATURES)
    values[_boss_bucket_index(token)] = 1.0
    return values


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def vocab_snapshot_checksum(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json_dumps(snapshot).encode("utf-8")).hexdigest()


def _power_count(units: Iterable[dict[str, Any]] | None) -> int:
    total = 0
    for unit in units or []:
        if not isinstance(unit, dict):
            continue
        values = unit.get("status") or unit.get("powers") or unit.get("buffs") or []
        if isinstance(values, list):
            total += len(values)
    return total


def _alive_enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    alive: list[dict[str, Any]] = []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        if hp > 0:
            alive.append(enemy)
    return alive


def _incoming_intent_damage(state: dict[str, Any]) -> float:
    total = 0.0
    for enemy in _alive_enemies(state):
        intents = enemy.get("intents") if isinstance(enemy.get("intents"), list) else []
        if intents and isinstance(intents[0], dict):
            for intent in intents:
                if not isinstance(intent, dict):
                    continue
                damage = _safe_float(intent.get("damage", intent.get("total_damage", 0)))
                hits = max(1.0, _safe_float(intent.get("hits", intent.get("multiplier", 1)), 1.0))
                total += damage * hits
        else:
            total += _safe_float(enemy.get("intent_damage", 0))
    return total


def _boss_token_from_state(state: dict[str, Any]) -> str:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    token = run.get("boss_entry_token") or run.get("next_boss_token") or run.get("boss_token")
    if token:
        normalized = _lower(token)
        if normalized:
            return normalized
    enemy_ids = [
        _lower(enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id") or "")
        for enemy in _alive_enemies(state)
    ]
    if any("kin_" in enemy_id or enemy_id == "the_kin" for enemy_id in enemy_ids):
        return "the_kin"
    if any("ceremonial_beast" in enemy_id for enemy_id in enemy_ids):
        return "ceremonial_beast"
    if any("vantom" in enemy_id for enemy_id in enemy_ids):
        return "vantom"
    for enemy_id in enemy_ids:
        if enemy_id:
            return enemy_id
    return ""


def _combat_round_number(state: dict[str, Any]) -> int:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    return _safe_int(
        battle.get("round_number", battle.get("round", state.get("round_number", state.get("round", 0)))),
        0,
    )


def build_leaf_state_signature(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = _extract_player(state)
    enemies = _alive_enemies(state)
    powers = player.get("status") or player.get("powers") or player.get("buffs") or []
    debuff_total = sum(int(round(_get_power_amount(powers, name))) for name in ("vulnerable", "weak", "frail"))
    max_hp = max(1, _safe_int(player.get("max_hp", 1), 1))
    enemy_max_hp_total = sum(max(1, _safe_int(enemy.get("max_hp", 1), 1)) for enemy in enemies)
    return {
        "state_type": _lower(state.get("state_type")),
        "round_number": _combat_round_number(state),
        "player_hp": _safe_int(player.get("hp", player.get("current_hp", 0)), 0),
        "player_max_hp": max_hp,
        "player_block": _safe_int(player.get("block", 0), 0),
        "player_energy": _safe_int(battle.get("energy", player.get("energy", 0)), 0),
        "hand_size": len(battle.get("hand") or player.get("hand") or []),
        "draw_size": len(battle.get("draw_pile_cards") or battle.get("draw_pile") or player.get("draw_pile") or []),
        "discard_size": len(battle.get("discard_pile_cards") or battle.get("discard_pile") or player.get("discard_pile") or []),
        "exhaust_size": len(battle.get("exhaust_pile_cards") or battle.get("exhaust_pile") or player.get("exhaust_pile") or []),
        "enemy_count": len(enemies),
        "enemy_hp_total": sum(_safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0) for enemy in enemies),
        "enemy_max_hp_total": enemy_max_hp_total,
        "potion_count_total": len(player.get("potions") or []),
        "buff_count_total": _power_count([player]) + _power_count(enemies),
        "debuff_count_total": debuff_total,
        "incoming_intent_damage": round(_incoming_intent_damage(state), 4),
        "boss_token": _boss_token_from_state(state),
    }


def build_leaf_player_features(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = _extract_player(state)
    max_hp = max(1.0, _safe_float(player.get("max_hp", 1), 1.0))
    max_energy = max(1.0, _safe_float(battle.get("max_energy", player.get("max_energy", 3)), 3.0))
    powers = player.get("status") or player.get("powers") or player.get("buffs") or []
    return {
        "hp_frac": round(_safe_float(player.get("hp", player.get("current_hp", 0)), 0.0) / max_hp, 6),
        "block_frac": round(_safe_float(player.get("block", 0), 0.0) / max_hp, 6),
        "energy": _safe_int(battle.get("energy", player.get("energy", 0)), 0),
        "max_energy": int(max_energy),
        "strength": round(_get_power_amount(powers, "strength"), 4),
        "dexterity": round(_get_power_amount(powers, "dexterity"), 4),
        "vulnerable": round(_get_power_amount(powers, "vulnerable"), 4),
        "weak": round(_get_power_amount(powers, "weak"), 4),
        "frail": round(_get_power_amount(powers, "frail"), 4),
        "metallicize": round(_get_power_amount(powers, "metallicize"), 4),
        "regen": round(_get_power_amount(powers, "regen"), 4),
        "artifact": round(_get_power_amount(powers, "artifact"), 4),
        "potion_count": len(player.get("potions") or []),
    }


def _card_block(card: dict[str, Any], vocab: Vocab) -> dict[str, Any]:
    card_idx, card_aux = _cached_card_encoding(card, vocab)
    return {
        "card_idx": int(card_idx),
        "card_id": str(card.get("id") or ""),
        "aux": [float(value) for value in np.asarray(card_aux, dtype=np.float32).tolist()],
    }


def build_leaf_state_features(state: dict[str, Any], vocab: Vocab) -> dict[str, Any]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = _extract_player(state)
    enemies = _alive_enemies(state)
    floor = _safe_int(((state.get("run") or {}) if isinstance(state.get("run"), dict) else {}).get("floor", 0), 0)
    round_number = _combat_round_number(state)
    enemy_rows: list[dict[str, Any]] = []
    for enemy in enemies:
        aux = _enemy_aux_features(enemy)
        enemy_rows.append(
            {
                "enemy_idx": int(_cached_monster_idx(vocab, enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id") or "")),
                "entity_id": str(enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id") or ""),
                "aux": [float(value) for value in np.asarray(aux, dtype=np.float32).tolist()],
            }
        )
    return {
        "player": build_leaf_player_features(state),
        "enemies": enemy_rows,
        "hand": [_card_block(card, vocab) for card in (battle.get("hand") or player.get("hand") or []) if isinstance(card, dict)],
        "draw": [_card_block(card, vocab) for card in (battle.get("draw_pile_cards") or battle.get("draw_pile") or player.get("draw_pile") or []) if isinstance(card, dict)],
        "discard": [_card_block(card, vocab) for card in (battle.get("discard_pile_cards") or battle.get("discard_pile") or player.get("discard_pile") or []) if isinstance(card, dict)],
        "exhaust": [_card_block(card, vocab) for card in (battle.get("exhaust_pile_cards") or battle.get("exhaust_pile") or player.get("exhaust_pile") or []) if isinstance(card, dict)],
        "floor": floor,
        "round_number": round_number,
        "boss_token": _boss_token_from_state(state),
    }


def normalize_score_target(score_target: str | None) -> str:
    normalized = _lower(score_target)
    aliases = {
        "score_v1": "score_v1_clipped",
        "clipped": "score_v1_clipped",
        "score_v1_clipped": "score_v1_clipped",
        "raw": "score_v1_raw",
        "score_v1_raw": "score_v1_raw",
        "softclip": "search_value_softclip",
        "search_value_softclip": "search_value_softclip",
    }
    return aliases.get(normalized, DEFAULT_LEAF_SCORE_TARGET)


def leaf_score_raw_from_labels(win_prob: float, boss_damage_ratio: float, hp_loss_ratio: float = 0.0) -> float:
    return float(
        LEAF_SCORE_V1_COEFFICIENTS["win_prob"] * float(win_prob)
        + LEAF_SCORE_V1_COEFFICIENTS["win_offset"]
        + LEAF_SCORE_V1_COEFFICIENTS["hp_loss_ratio"] * float(hp_loss_ratio)
        + LEAF_SCORE_V1_COEFFICIENTS["boss_damage_ratio"] * float(boss_damage_ratio)
    )


def leaf_score_target_from_raw(
    raw_score: float,
    *,
    score_target: str = DEFAULT_LEAF_SCORE_TARGET,
    temperature: float = LEAF_SCORE_SOFTCLIP_TEMPERATURE,
) -> float:
    target = normalize_score_target(score_target)
    raw_value = float(raw_score)
    if target == "score_v1_raw":
        return raw_value
    if target == "search_value_softclip":
        safe_temperature = max(1e-6, float(temperature))
        return float(np.tanh(raw_value / safe_temperature))
    return float(np.clip(raw_value, -1.0, 1.0))


def build_score_targets(
    *,
    win_prob: float,
    boss_damage_ratio: float,
    hp_loss_ratio: float = 0.0,
    temperature: float = LEAF_SCORE_SOFTCLIP_TEMPERATURE,
) -> dict[str, Any]:
    raw_score = leaf_score_raw_from_labels(win_prob, boss_damage_ratio, hp_loss_ratio)
    clipped_score = leaf_score_target_from_raw(raw_score, score_target="score_v1_clipped")
    softclip_score = leaf_score_target_from_raw(
        raw_score,
        score_target="search_value_softclip",
        temperature=temperature,
    )
    return {
        "score_v1_raw": raw_score,
        "score_v1_clipped": clipped_score,
        "search_value_softclip": softclip_score,
        "clip_saturated": bool(abs(raw_score - clipped_score) > 1e-6),
    }


def score_targets_from_labels(
    labels: dict[str, Any] | None,
    *,
    temperature: float = LEAF_SCORE_SOFTCLIP_TEMPERATURE,
) -> dict[str, Any]:
    labels = labels or {}
    return build_score_targets(
        win_prob=float(labels.get("win_prob", 0.0) or 0.0),
        boss_damage_ratio=float(labels.get("boss_damage_ratio", 0.0) or 0.0),
        hp_loss_ratio=float(labels.get("hp_loss_ratio", 0.0) or 0.0),
        temperature=temperature,
    )


def score_targets_from_row(
    row: dict[str, Any] | None,
    *,
    temperature: float = LEAF_SCORE_SOFTCLIP_TEMPERATURE,
) -> dict[str, Any]:
    row = row or {}
    computed = score_targets_from_labels(
        row.get("labels") if isinstance(row.get("labels"), dict) else {},
        temperature=temperature,
    )
    existing = row.get("score_targets") if isinstance(row.get("score_targets"), dict) else {}
    merged = dict(computed)
    for key in ("score_v1_raw", "score_v1_clipped", "search_value_softclip"):
        if key in existing:
            try:
                merged[key] = float(existing.get(key))
            except Exception:
                merged[key] = computed[key]
    if "clip_saturated" in existing:
        merged["clip_saturated"] = bool(existing.get("clip_saturated"))
    return merged


def leaf_score_from_labels(
    win_prob: float,
    boss_damage_ratio: float,
    hp_loss_ratio: float = 0.0,
    *,
    score_target: str = DEFAULT_LEAF_SCORE_TARGET,
    temperature: float = LEAF_SCORE_SOFTCLIP_TEMPERATURE,
) -> float:
    raw_score = leaf_score_raw_from_labels(win_prob, boss_damage_ratio, hp_loss_ratio)
    return leaf_score_target_from_raw(raw_score, score_target=score_target, temperature=temperature)


def leaf_value_from_score(
    score: float,
    *,
    score_target: str = DEFAULT_LEAF_SCORE_TARGET,
    temperature: float = LEAF_SCORE_SOFTCLIP_TEMPERATURE,
) -> float:
    return leaf_score_target_from_raw(score, score_target=score_target, temperature=temperature)


def signature_to_feature_vector(signature: dict[str, Any]) -> np.ndarray:
    player_max = max(1.0, float(signature.get("player_max_hp", 1) or 1))
    enemy_max_total = max(1.0, float(signature.get("enemy_max_hp_total", 1) or 1))
    incoming = float(signature.get("incoming_intent_damage", 0.0) or 0.0)
    return np.array(
        [
            float(signature.get("round_number", 0) or 0) / 10.0,
            float(signature.get("player_hp", 0) or 0) / player_max,
            float(signature.get("player_block", 0) or 0) / player_max,
            float(signature.get("player_energy", 0) or 0) / 5.0,
            float(signature.get("hand_size", 0) or 0) / 12.0,
            float(signature.get("draw_size", 0) or 0) / 40.0,
            float(signature.get("discard_size", 0) or 0) / 40.0,
            float(signature.get("exhaust_size", 0) or 0) / 20.0,
            float(signature.get("enemy_count", 0) or 0) / 4.0,
            float(signature.get("enemy_hp_total", 0) or 0) / enemy_max_total,
            math.log1p(enemy_max_total) / 10.0,
            float(signature.get("potion_count_total", 0) or 0) / 3.0,
            float(signature.get("buff_count_total", 0) or 0) / 10.0,
            float(signature.get("debuff_count_total", 0) or 0) / 10.0,
            incoming / player_max,
        ],
        dtype=np.float32,
    )


def heuristic_leaf_value_from_signature(signature: dict[str, Any]) -> float:
    player_max = max(1.0, float(signature.get("player_max_hp", 1) or 1))
    enemy_max_total = max(1.0, float(signature.get("enemy_max_hp_total", 1) or 1))
    player_hp_frac = float(signature.get("player_hp", 0) or 0) / player_max
    player_block_frac = float(signature.get("player_block", 0) or 0) / player_max
    enemy_hp_frac = float(signature.get("enemy_hp_total", 0) or 0) / enemy_max_total
    enemy_count = max(0.0, float(signature.get("enemy_count", 0) or 0))
    incoming = max(0.0, float(signature.get("incoming_intent_damage", 0.0) or 0.0) - float(signature.get("player_block", 0) or 0))
    incoming_frac = incoming / player_max
    score = (
        + 1.0 * player_hp_frac
        - 1.0 * enemy_hp_frac
        + 0.10 * player_block_frac
        - 0.80 * incoming_frac
        + 0.20 * (1.0 - min(1.0, enemy_count / 3.0))
    )
    return float(max(-1.5, min(1.5, score)))


@dataclass
class LeafSample:
    row: dict[str, Any]

    @property
    def parent_id(self) -> str:
        return str(self.row.get("parent_id") or "")

    @property
    def labels(self) -> dict[str, Any]:
        return self.row.get("labels") or {}

    @property
    def state_signature(self) -> dict[str, Any]:
        return self.row.get("state_signature") or {}

    @property
    def state_features(self) -> dict[str, Any]:
        return self.row.get("state_features") or {}

    @property
    def score_targets(self) -> dict[str, Any]:
        return score_targets_from_row(self.row)

    def score_target(self, score_target: str = DEFAULT_LEAF_SCORE_TARGET) -> float:
        target = normalize_score_target(score_target)
        value = self.score_targets.get(target)
        try:
            return float(value)
        except Exception:
            fallback = score_targets_from_labels(self.labels)
            return float(fallback[target])

    @property
    def target_score(self) -> float:
        return self.score_target("score_v1_clipped")

    @property
    def target_win(self) -> float:
        return float(self.labels.get("win_prob", 0.0) or 0.0)

    @property
    def target_damage(self) -> float:
        return float(self.labels.get("boss_damage_ratio", 0.0) or 0.0)

    @property
    def target_hp_loss(self) -> float:
        return float(self.labels.get("hp_loss_ratio", 0.0) or 0.0)

    @property
    def clip_saturated(self) -> bool:
        return bool(self.score_targets.get("clip_saturated"))

    @property
    def boss_bucket(self) -> str:
        token = str(self.row.get("boss_token") or self.state_features.get("boss_token") or "").strip().lower()
        if token:
            return token
        encounter_kind = str(self.row.get("encounter_kind") or "").strip().lower()
        return encounter_kind or "unknown"


class LeafDataset:
    def __init__(self, samples: list[LeafSample]):
        self.samples = samples

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LeafDataset":
        samples: list[LeafSample] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                samples.append(LeafSample(json.loads(raw_line)))
        return cls(samples)


class MlpLeafEvaluator(nn.Module):
    def __init__(self, input_dim: int = len(SIGNATURE_FEATURE_KEYS), hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(x)
        return {
            "win_prob": torch.sigmoid(raw[..., 0]),
            "boss_damage_ratio": torch.sigmoid(raw[..., 1]),
            "hp_loss_ratio": torch.sigmoid(raw[..., 2]),
        }


class BossLeafEvaluator(nn.Module):
    def __init__(
        self,
        *,
        card_vocab_size: int,
        monster_vocab_size: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_tokens: int = MAX_LEAF_TOKENS,
        card_aux_dim: int = CARD_AUX_DIM,
        enemy_aux_dim: int = ENEMY_AUX_DIM,
        player_aux_dim: int = 21,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.card_vocab_size = int(card_vocab_size)
        self.monster_vocab_size = int(monster_vocab_size)
        self.card_aux_dim = int(card_aux_dim)
        self.enemy_aux_dim = int(enemy_aux_dim)
        self.player_aux_dim = int(player_aux_dim)

        self.token_type_embed = nn.Embedding(len(TOKEN_TYPE_TO_IDX), hidden_dim)
        self.position_embed = nn.Embedding(max_tokens, hidden_dim)
        self.card_embed = nn.Embedding(card_vocab_size, hidden_dim, padding_idx=0)
        self.enemy_embed = nn.Embedding(monster_vocab_size, hidden_dim, padding_idx=0)
        self.player_aux_proj = nn.Linear(player_aux_dim, hidden_dim)
        self.enemy_aux_proj = nn.Linear(enemy_aux_dim, hidden_dim)
        self.card_aux_proj = nn.Linear(card_aux_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.win_head = nn.Linear(hidden_dim, 1)
        self.damage_head = nn.Linear(hidden_dim, 1)
        self.hp_loss_head = nn.Linear(hidden_dim, 1)

    def encode(
        self,
        token_types: torch.Tensor,
        card_ids: torch.Tensor,
        enemy_ids: torch.Tensor,
        aux: torch.Tensor,
        aux_kind: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = token_types.shape
        positions = torch.arange(seq_len, device=token_types.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_type_embed(token_types) + self.position_embed(positions)
        card_mask = (aux_kind == 1) | (aux_kind == 2)
        enemy_mask = aux_kind == 3
        player_mask = aux_kind == 4
        hidden = hidden + self.card_embed(card_ids) * card_mask.unsqueeze(-1)
        hidden = hidden + self.enemy_embed(enemy_ids) * enemy_mask.unsqueeze(-1)
        hidden = hidden + self.card_aux_proj(aux[..., : self.card_aux_dim]) * card_mask.unsqueeze(-1)
        hidden = hidden + self.enemy_aux_proj(aux[..., : self.enemy_aux_dim]) * enemy_mask.unsqueeze(-1)
        hidden = hidden + self.player_aux_proj(aux[..., : self.player_aux_dim]) * player_mask.unsqueeze(-1)
        encoded = self.encoder(self.input_norm(hidden), src_key_padding_mask=~attention_mask)
        return self.head_norm(encoded[:, 0])

    def forward(
        self,
        token_types: torch.Tensor,
        card_ids: torch.Tensor,
        enemy_ids: torch.Tensor,
        aux: torch.Tensor,
        aux_kind: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pooled = self.encode(token_types, card_ids, enemy_ids, aux, aux_kind, attention_mask)
        return {
            "win_prob": torch.sigmoid(self.win_head(pooled).squeeze(-1)),
            "boss_damage_ratio": torch.sigmoid(self.damage_head(pooled).squeeze(-1)),
            "hp_loss_ratio": torch.sigmoid(self.hp_loss_head(pooled).squeeze(-1)),
        }


def player_aux_from_features(state_features: dict[str, Any]) -> np.ndarray:
    player = state_features.get("player") or {}
    boss_token = state_features.get("boss_token") or ""
    return np.array(
        [
            float(player.get("hp_frac", 0.0) or 0.0),
            float(player.get("block_frac", 0.0) or 0.0),
            float(player.get("energy", 0.0) or 0.0) / 5.0,
            float(player.get("max_energy", 0.0) or 0.0) / 5.0,
            float(player.get("strength", 0.0) or 0.0) / 10.0,
            float(player.get("dexterity", 0.0) or 0.0) / 10.0,
            float(player.get("vulnerable", 0.0) or 0.0) / 5.0,
            float(player.get("weak", 0.0) or 0.0) / 5.0,
            float(player.get("frail", 0.0) or 0.0) / 5.0,
            float(player.get("metallicize", 0.0) or 0.0) / 10.0,
            float(player.get("regen", 0.0) or 0.0) / 10.0,
            float(player.get("artifact", 0.0) or 0.0) / 3.0,
            float(state_features.get("floor", 0) or 0) / 20.0,
            float(state_features.get("round_number", 0) or 0) / 20.0,
            float(player.get("potion_count", 0) or 0) / 3.0,
            len(state_features.get("hand") or []) / 12.0,
            *_boss_bucket_vector(boss_token),
        ],
        dtype=np.float32,
    )


def tokenize_leaf_state(
    state_features: dict[str, Any],
    *,
    max_tokens: int = MAX_LEAF_TOKENS,
    player_aux_dim: int = 21,
    card_aux_dim: int = CARD_AUX_DIM,
    enemy_aux_dim: int = ENEMY_AUX_DIM,
) -> dict[str, np.ndarray]:
    aux_dim = max(player_aux_dim, card_aux_dim, enemy_aux_dim)
    token_types = np.zeros(max_tokens, dtype=np.int64)
    card_ids = np.zeros(max_tokens, dtype=np.int64)
    enemy_ids = np.zeros(max_tokens, dtype=np.int64)
    aux = np.zeros((max_tokens, aux_dim), dtype=np.float32)
    aux_kind = np.zeros(max_tokens, dtype=np.int64)
    attention_mask = np.zeros(max_tokens, dtype=bool)

    def add_token(
        token_type: str,
        *,
        card_idx: int = 0,
        enemy_idx: int = 0,
        aux_values: list[float] | np.ndarray | None = None,
        kind: int = 0,
    ) -> bool:
        index = int(attention_mask.sum())
        if index >= max_tokens:
            return False
        token_types[index] = TOKEN_TYPE_TO_IDX[token_type]
        card_ids[index] = int(card_idx)
        enemy_ids[index] = int(enemy_idx)
        if aux_values is not None:
            values = np.asarray(aux_values, dtype=np.float32)
            aux[index, : min(aux_dim, values.shape[0])] = values[:aux_dim]
        aux_kind[index] = kind
        attention_mask[index] = True
        return True

    add_token("cls")
    add_token("player", aux_values=player_aux_from_features(state_features), kind=4)
    for enemy in state_features.get("enemies") or []:
        add_token("enemy", enemy_idx=int(enemy.get("enemy_idx", 0) or 0), aux_values=enemy.get("aux") or [], kind=3)
    for pile_name, token_name in (("hand", "hand"), ("draw", "draw"), ("discard", "discard"), ("exhaust", "exhaust")):
        for card in state_features.get(pile_name) or []:
            if not add_token(
                token_name,
                card_idx=int(card.get("card_idx", 0) or 0),
                aux_values=card.get("aux") or [],
                kind=1 if pile_name == "hand" else 2,
            ):
                break

    return {
        "token_types": token_types,
        "card_ids": card_ids,
        "enemy_ids": enemy_ids,
        "aux": aux,
        "aux_kind": aux_kind,
        "attention_mask": attention_mask,
    }


def collate_leaf_batch(
    samples: list[LeafSample],
    *,
    max_tokens: int = MAX_LEAF_TOKENS,
    score_target: str = DEFAULT_LEAF_SCORE_TARGET,
) -> dict[str, torch.Tensor]:
    tokenized = [tokenize_leaf_state(sample.state_features, max_tokens=max_tokens) for sample in samples]
    return {
        "token_types": torch.tensor(np.stack([item["token_types"] for item in tokenized], axis=0)).long(),
        "card_ids": torch.tensor(np.stack([item["card_ids"] for item in tokenized], axis=0)).long(),
        "enemy_ids": torch.tensor(np.stack([item["enemy_ids"] for item in tokenized], axis=0)).long(),
        "aux": torch.tensor(np.stack([item["aux"] for item in tokenized], axis=0)).float(),
        "aux_kind": torch.tensor(np.stack([item["aux_kind"] for item in tokenized], axis=0)).long(),
        "attention_mask": torch.tensor(np.stack([item["attention_mask"] for item in tokenized], axis=0)).bool(),
        "target_win": torch.tensor([sample.target_win for sample in samples]).float(),
        "target_damage": torch.tensor([sample.target_damage for sample in samples]).float(),
        "target_hp_loss": torch.tensor([sample.target_hp_loss for sample in samples]).float(),
        "target_score": torch.tensor([sample.score_target(score_target) for sample in samples]).float(),
    }


def collate_signature_batch(
    samples: list[LeafSample],
    *,
    score_target: str = DEFAULT_LEAF_SCORE_TARGET,
) -> dict[str, torch.Tensor]:
    features = np.stack([signature_to_feature_vector(sample.state_signature) for sample in samples], axis=0)
    return {
        "features": torch.tensor(features).float(),
        "target_win": torch.tensor([sample.target_win for sample in samples]).float(),
        "target_damage": torch.tensor([sample.target_damage for sample in samples]).float(),
        "target_hp_loss": torch.tensor([sample.target_hp_loss for sample in samples]).float(),
        "target_score": torch.tensor([sample.score_target(score_target) for sample in samples]).float(),
    }


def outputs_to_score(
    outputs: dict[str, torch.Tensor],
    *,
    score_target: str = DEFAULT_LEAF_SCORE_TARGET,
    temperature: float = LEAF_SCORE_SOFTCLIP_TEMPERATURE,
) -> torch.Tensor:
    raw_score = (
        LEAF_SCORE_V1_COEFFICIENTS["win_prob"] * outputs["win_prob"]
        + LEAF_SCORE_V1_COEFFICIENTS["win_offset"]
        + LEAF_SCORE_V1_COEFFICIENTS["hp_loss_ratio"] * outputs["hp_loss_ratio"]
        + LEAF_SCORE_V1_COEFFICIENTS["boss_damage_ratio"] * outputs["boss_damage_ratio"]
    )
    target = normalize_score_target(score_target)
    if target == "score_v1_raw":
        return raw_score
    if target == "search_value_softclip":
        safe_temperature = max(1e-6, float(temperature))
        return torch.tanh(raw_score / safe_temperature)
    return torch.clamp(raw_score, -1.0, 1.0)


def pairwise_group_accuracy(parent_ids: list[str], predicted: np.ndarray, target: np.ndarray, *, min_gap: float = 1e-6) -> float:
    correct = 0
    total = 0
    grouped: dict[str, list[int]] = {}
    for index, parent_id in enumerate(parent_ids):
        grouped.setdefault(parent_id, []).append(index)
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                gap = float(target[left] - target[right])
                if abs(gap) <= min_gap:
                    continue
                pred_gap = float(predicted[left] - predicted[right])
                if (gap > 0 and pred_gap > 0) or (gap < 0 and pred_gap < 0):
                    correct += 1
                total += 1
    return float(correct / total) if total else 0.0


def compute_near_win_recall(predicted_score: np.ndarray, target_score: np.ndarray, *, threshold: float = 0.75) -> float:
    positives = target_score >= threshold
    if positives.sum() == 0:
        return 0.0
    predicted_positive = predicted_score >= threshold
    hits = np.logical_and(positives, predicted_positive).sum()
    return float(hits / positives.sum())


def expected_calibration_error(predicted_prob: np.ndarray, labels: np.ndarray, *, bins: int = 10) -> float:
    if predicted_prob.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        if end == 1.0:
            mask = (predicted_prob >= start) & (predicted_prob <= end)
        else:
            mask = (predicted_prob >= start) & (predicted_prob < end)
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float(predicted_prob[mask].mean()) - float(labels[mask].mean()))
    return total


def _safe_corrcoef(predicted: np.ndarray, target: np.ndarray) -> float:
    if predicted.size > 1 and float(np.std(predicted)) > 1e-8 and float(np.std(target)) > 1e-8:
        return float(np.corrcoef(predicted, target)[0, 1])
    return 0.0


def metric_bundle(
    samples: list[LeafSample],
    *,
    predicted_score: np.ndarray,
    predicted_damage: np.ndarray,
    predicted_win: np.ndarray,
    target_score: np.ndarray | None = None,
    target_damage: np.ndarray | None = None,
    target_win: np.ndarray | None = None,
) -> dict[str, Any]:
    target_score = target_score if target_score is not None else np.array([sample.target_score for sample in samples], dtype=np.float32)
    target_damage = target_damage if target_damage is not None else np.array([sample.target_damage for sample in samples], dtype=np.float32)
    target_win = target_win if target_win is not None else np.array([sample.target_win for sample in samples], dtype=np.float32)
    return {
        "count": len(samples),
        "score_corr": _safe_corrcoef(predicted_score, target_score),
        "damage_corr": _safe_corrcoef(predicted_damage, target_damage),
        "pair_acc": pairwise_group_accuracy([sample.parent_id for sample in samples], predicted_score, target_score),
        "near_win_recall": compute_near_win_recall(predicted_score, target_score),
        "win_ece": expected_calibration_error(predicted_win, target_win),
    }


def bucketed_metric_rows(
    samples: list[LeafSample],
    predicted_score: np.ndarray,
    *,
    predicted_damage: np.ndarray | None = None,
    predicted_win: np.ndarray | None = None,
    target_score: np.ndarray | None = None,
    target_damage: np.ndarray | None = None,
    target_win: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    target_score = target_score if target_score is not None else np.array([sample.target_score for sample in samples], dtype=np.float32)
    target_damage = target_damage if target_damage is not None else np.array([sample.target_damage for sample in samples], dtype=np.float32)
    target_win = target_win if target_win is not None else np.array([sample.target_win for sample in samples], dtype=np.float32)
    predicted_damage = predicted_damage if predicted_damage is not None else np.zeros_like(target_damage)
    predicted_win = predicted_win if predicted_win is not None else np.zeros_like(target_win)
    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(sample.boss_bucket, []).append(index)
    rows: list[dict[str, Any]] = []
    for bucket, indices in sorted(grouped.items()):
        bundle = metric_bundle(
            [samples[idx] for idx in indices],
            predicted_score=predicted_score[indices],
            predicted_damage=predicted_damage[indices],
            predicted_win=predicted_win[indices],
            target_score=target_score[indices],
            target_damage=target_damage[indices],
            target_win=target_win[indices],
        )
        rows.append(
            {
                "bucket": bucket,
                **bundle,
                "sibling_pair_acc": bundle["pair_acc"],
            }
        )
    return rows


class BossLeafEvaluatorRuntime:
    def __init__(self, payload: dict[str, Any], *, device: torch.device | None = None):
        self.payload = payload
        self.device = device or torch.device("cpu")
        self.vocab = self._load_vocab(payload)
        self.training_score_target = normalize_score_target(
            payload.get("score_target") or payload.get("label_used_for_training") or DEFAULT_LEAF_SCORE_TARGET
        )
        self.runtime_leaf_value_target = normalize_score_target(
            payload.get("runtime_leaf_value_target")
            or ("search_value_softclip" if self.training_score_target == "score_v1_raw" else "score_v1_clipped")
        )
        self.score_softclip_temperature = float(
            payload.get("score_softclip_temperature", LEAF_SCORE_SOFTCLIP_TEMPERATURE) or LEAF_SCORE_SOFTCLIP_TEMPERATURE
        )
        model_type = str(payload.get("model_type") or "transformer").strip().lower()
        self.model_type = model_type
        if model_type == "mlp":
            self.model = MlpLeafEvaluator(input_dim=int(payload.get("input_dim", len(SIGNATURE_FEATURE_KEYS))), hidden_dim=int(payload.get("hidden_dim", 64)))
        else:
            self.model = BossLeafEvaluator(
                card_vocab_size=int(payload.get("card_vocab_size", self.vocab.card_vocab_size)),
                monster_vocab_size=int(payload.get("monster_vocab_size", self.vocab.monster_vocab_size)),
                hidden_dim=int(payload.get("hidden_dim", 128)),
                n_heads=int(payload.get("n_heads", 4)),
                n_layers=int(payload.get("n_layers", 3)),
                max_tokens=int(payload.get("max_tokens", MAX_LEAF_TOKENS)),
                card_aux_dim=int(payload.get("card_aux_dim", CARD_AUX_DIM)),
                enemy_aux_dim=int(payload.get("enemy_aux_dim", ENEMY_AUX_DIM)),
                player_aux_dim=int(payload.get("player_aux_dim", 21)),
            )
        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("Boss leaf evaluator checkpoint is missing model_state_dict")
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        if missing or unexpected:
            if missing:
                logger.error("Boss leaf evaluator checkpoint missing keys: %s", missing)
            if unexpected:
                logger.error("Boss leaf evaluator checkpoint unexpected keys: %s", unexpected)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise ValueError("Boss leaf evaluator checkpoint mismatch: " + "; ".join(details))
        self.model.to(self.device).eval()

    @staticmethod
    def _load_vocab(payload: dict[str, Any]) -> Vocab:
        snapshot = payload.get("vocab_snapshot")
        if isinstance(snapshot, dict):
            data = snapshot.get("data") if isinstance(snapshot.get("data"), dict) else snapshot
            expected_checksum = str(payload.get("vocab_snapshot_checksum") or snapshot.get("checksum") or "").strip().lower()
            actual_checksum = vocab_snapshot_checksum(data)
            if expected_checksum and actual_checksum != expected_checksum:
                raise ValueError(
                    "Boss leaf evaluator vocab snapshot checksum mismatch: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )
            try:
                return Vocab.from_dict(data)
            except Exception as exc:
                raise ValueError(f"Invalid boss leaf evaluator vocab_snapshot: {exc}") from exc
        logger.warning("Boss leaf evaluator checkpoint has no vocab_snapshot; falling back to workspace vocab.json")
        return load_vocab()

    def predict_features(self, state_features: dict[str, Any], state_signature: dict[str, Any]) -> dict[str, float]:
        with torch.no_grad():
            if self.model_type == "mlp":
                features = torch.tensor(signature_to_feature_vector(state_signature)).unsqueeze(0).to(self.device)
                outputs = self.model(features)
            else:
                tokenized = tokenize_leaf_state(state_features, max_tokens=int(self.payload.get("max_tokens", MAX_LEAF_TOKENS)))
                outputs = self.model(
                    torch.tensor(tokenized["token_types"]).unsqueeze(0).long().to(self.device),
                    torch.tensor(tokenized["card_ids"]).unsqueeze(0).long().to(self.device),
                    torch.tensor(tokenized["enemy_ids"]).unsqueeze(0).long().to(self.device),
                    torch.tensor(tokenized["aux"]).unsqueeze(0).float().to(self.device),
                    torch.tensor(tokenized["aux_kind"]).unsqueeze(0).long().to(self.device),
                    torch.tensor(tokenized["attention_mask"]).unsqueeze(0).bool().to(self.device),
                )
        win_prob = float(outputs["win_prob"].squeeze(0).item())
        boss_damage_ratio = float(outputs["boss_damage_ratio"].squeeze(0).item())
        hp_loss_ratio = float(outputs["hp_loss_ratio"].squeeze(0).item())
        score_raw = leaf_score_from_labels(
            win_prob,
            boss_damage_ratio,
            hp_loss_ratio,
            score_target="score_v1_raw",
        )
        score_clipped = leaf_score_target_from_raw(score_raw, score_target="score_v1_clipped")
        score_softclip = leaf_score_target_from_raw(
            score_raw,
            score_target="search_value_softclip",
            temperature=self.score_softclip_temperature,
        )
        leaf_value = leaf_value_from_score(
            score_raw,
            score_target=self.runtime_leaf_value_target,
            temperature=self.score_softclip_temperature,
        )
        return {
            "win_prob": win_prob,
            "boss_damage_ratio": boss_damage_ratio,
            "hp_loss_ratio": hp_loss_ratio,
            "leaf_score_raw": score_raw,
            "leaf_score": score_clipped,
            "leaf_score_softclip": score_softclip,
            "leaf_value": leaf_value,
        }

    def predict_state(self, state: dict[str, Any]) -> dict[str, float]:
        signature = build_leaf_state_signature(state)
        features = build_leaf_state_features(state, self.vocab)
        return self.predict_features(features, signature)


def load_boss_leaf_evaluator_runtime(path: str | Path | None, *, device: torch.device | None = None) -> BossLeafEvaluatorRuntime | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    return BossLeafEvaluatorRuntime(payload, device=device)
