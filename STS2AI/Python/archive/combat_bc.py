from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from archive.combat_actions import action_to_key, enumerate_legal_actions, normalize_action


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _is_status_like(card: dict[str, Any]) -> bool:
    card_id = _lower(card.get("id"))
    title = _lower(card.get("title"))
    return any(token in card_id or token in title for token in ("status", "curse", "wound", "burn", "dazed", "slime"))


def _estimate_attack_damage(card: dict[str, Any]) -> int:
    card_id = _lower(card.get("id"))
    explicit_damage = {
        "strike_silent": 6,
        "strike_ironclad": 6,
        "neutralize": 3,
        "bash": 8,
    }
    if card_id in explicit_damage:
        return explicit_damage[card_id]
    if "strike" in card_id:
        return 6
    if "bash" in card_id:
        return 8
    if "neutralize" in card_id:
        return 3
    return 0


def _estimate_block(card: dict[str, Any]) -> int:
    card_id = _lower(card.get("id"))
    explicit_block = {
        "defend_silent": 5,
        "defend_ironclad": 5,
        "survivor": 8,
    }
    if card_id in explicit_block:
        return explicit_block[card_id]
    if "defend" in card_id:
        return 5
    if "survivor" in card_id:
        return 8
    return 0


def _enemy_incoming_damage(enemy: dict[str, Any]) -> int:
    total = 0
    for intent in enemy.get("intents", []):
        if _lower(intent.get("intent_type")) in ("attack", "deathblow"):
            total += int(intent.get("total_damage") or 0)
    return total


def _find_enemy(state: dict[str, Any], combat_id: int | None) -> dict[str, Any] | None:
    if combat_id is None:
        return None
    for enemy in state.get("enemies", []):
        if int(enemy.get("combat_id", -1)) == int(combat_id):
            return enemy
    return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _power_amount(creature: dict[str, Any], power_id: str) -> int:
    for power in creature.get("powers", []):
        if _lower(power.get("id")) == _lower(power_id):
            return int(power.get("amount", 0))
    return 0


def _find_potion(state: dict[str, Any], slot: int | None) -> dict[str, Any] | None:
    if slot is None:
        return None
    for potion in state.get("potions", []):
        if int(potion.get("slot", -1)) == int(slot):
            return potion
    player = state.get("player") or {}
    for potion in player.get("potions", []):
        if int(potion.get("slot", -1)) == int(slot):
            return potion
    return None


def _find_card_selection_option(state: dict[str, Any], option_index: int | None) -> dict[str, Any] | None:
    if option_index is None:
        return None
    card_selection = state.get("card_selection")
    if not isinstance(card_selection, dict):
        return None
    for key in ("selectable_options", "selectable_cards", "options", "cards", "selected_options", "selected_cards"):
        options = card_selection.get(key)
        if not isinstance(options, list):
            continue
        for option in options:
            if not isinstance(option, dict):
                continue
            candidate_index = option.get("option_index", option.get("index", option.get("hand_index", option.get("card_index"))))
            try:
                if candidate_index is not None and int(candidate_index) == int(option_index):
                    return option
            except (TypeError, ValueError):
                continue
    return None


def _potion_text(potion: dict[str, Any] | None) -> str:
    if not isinstance(potion, dict):
        return ""
    return " ".join(
        _lower(value)
        for value in (
            potion.get("id"),
            potion.get("title"),
            potion.get("name"),
            potion.get("description"),
        )
        if value
    )


def _potion_is_type(potion: dict[str, Any] | None, *tokens: str) -> float:
    text = _potion_text(potion)
    return 1.0 if any(token in text for token in tokens) else 0.0


STATE_FEATURE_NAMES = [
    "bias",
    "round_number",
    "player_hp_ratio",
    "player_block",
    "player_energy_ratio",
    "draw_pile_ratio",
    "discard_pile_ratio",
    "exhaust_pile_ratio",
    "alive_enemy_count",
    "total_enemy_hp_ratio",
    "lowest_enemy_hp_ratio",
    "incoming_damage_ratio",
    "enemy_0_hp_ratio",
    "enemy_0_block",
    "enemy_0_attack_intent",
    "enemy_0_incoming_damage_ratio",
    "enemy_0_thorns",
    "enemy_1_hp_ratio",
    "enemy_1_block",
    "enemy_1_attack_intent",
    "enemy_1_incoming_damage_ratio",
    "enemy_1_thorns",
    "hand_size_ratio",
    "playable_count_ratio",
    "attack_card_count_ratio",
    "block_card_count_ratio",
    "status_card_count_ratio",
    "zero_cost_count_ratio",
    "one_cost_count_ratio",
    "two_plus_cost_count_ratio",
]


ACTION_FEATURE_NAMES = [
    "action_end_turn",
    "action_play_attack",
    "action_play_block",
    "action_play_other",
    "action_select_card",
    "action_confirm",
    "action_cancel",
    "action_hand_index_ratio",
    "action_select_card_option",
    "action_option_index_ratio",
    "action_target_hp_ratio",
    "action_target_block",
    "action_target_intends_attack",
    "action_target_incoming_damage_ratio",
    "action_is_lethal",
    "action_card_cost_ratio",
    "action_card_damage_ratio",
    "action_card_block_ratio",
    "action_card_is_status",
    "action_card_requires_target",
    "action_use_potion",
    "action_potion_slot_ratio",
    "action_potion_requires_target",
    "action_potion_is_dexterity",
    "action_potion_is_strength",
    "action_potion_is_attack",
    "action_potion_is_heal",
    "action_potion_is_draw",
]


FEATURE_NAMES = STATE_FEATURE_NAMES + ACTION_FEATURE_NAMES


def featurize_state_action(state: dict[str, Any], action: dict[str, Any]) -> np.ndarray:
    return np.asarray(_state_features(state) + _action_features(state, action), dtype=np.float64)


def _align_feature_matrix(feature_matrix: np.ndarray, weight_count: int) -> np.ndarray:
    if feature_matrix.shape[1] == weight_count:
        return feature_matrix
    if feature_matrix.shape[1] > weight_count:
        return feature_matrix[:, :weight_count]
    padding = np.zeros((feature_matrix.shape[0], weight_count - feature_matrix.shape[1]), dtype=feature_matrix.dtype)
    return np.hstack([feature_matrix, padding])


def _state_features(state: dict[str, Any]) -> list[float]:
    player = state.get("player") or {}
    piles = state.get("piles") or {}
    hand_cards = list(state.get("hand_cards", []))
    alive_enemies = [enemy for enemy in state.get("enemies", []) if enemy.get("is_alive")]
    total_enemy_hp = sum(int(enemy.get("current_hp", 0)) for enemy in alive_enemies)
    lowest_enemy_hp = min((int(enemy.get("current_hp", 0)) for enemy in alive_enemies), default=0)
    incoming_damage = sum(_enemy_incoming_damage(enemy) for enemy in alive_enemies)
    playable_cards = [card for card in hand_cards if card.get("can_play")]
    attack_cards = [card for card in hand_cards if _estimate_attack_damage(card) > 0]
    block_cards = [card for card in hand_cards if _estimate_block(card) > 0]
    status_cards = [card for card in hand_cards if _is_status_like(card)]
    zero_cost_cards = [card for card in hand_cards if int(card.get("energy_cost", 0)) == 0]
    one_cost_cards = [card for card in hand_cards if int(card.get("energy_cost", 0)) == 1]
    two_plus_cost_cards = [card for card in hand_cards if int(card.get("energy_cost", 0)) >= 2]

    features: list[float] = [
        1.0,
        _safe_ratio(float(state.get("round_number", 0)), 10.0),
        _safe_ratio(float(player.get("current_hp", 0)), float(player.get("max_hp", 1))),
        _safe_ratio(float(player.get("block", 0)), 20.0),
        _safe_ratio(float(player.get("energy", 0)), float(max(1, int(player.get("max_energy", 1))))),
        _safe_ratio(float(piles.get("draw", 0)), 20.0),
        _safe_ratio(float(piles.get("discard", 0)), 20.0),
        _safe_ratio(float(piles.get("exhaust", 0)), 10.0),
        _safe_ratio(float(len(alive_enemies)), 3.0),
        _safe_ratio(float(total_enemy_hp), 100.0),
        _safe_ratio(float(lowest_enemy_hp), 50.0),
        _safe_ratio(float(incoming_damage), 30.0),
    ]

    for enemy_index in range(2):
        if enemy_index < len(alive_enemies):
            enemy = alive_enemies[enemy_index]
            features.extend(
                [
                    _safe_ratio(float(enemy.get("current_hp", 0)), float(max(1, int(enemy.get("max_hp", 1))))),
                    _safe_ratio(float(enemy.get("block", 0)), 20.0),
                    1.0 if enemy.get("intends_to_attack") else 0.0,
                    _safe_ratio(float(_enemy_incoming_damage(enemy)), 20.0),
                    _safe_ratio(float(_power_amount(enemy, "THORNS_POWER")), 5.0),
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    features.extend(
        [
            _safe_ratio(float(len(hand_cards)), 10.0),
            _safe_ratio(float(len(playable_cards)), 10.0),
            _safe_ratio(float(len(attack_cards)), 10.0),
            _safe_ratio(float(len(block_cards)), 10.0),
            _safe_ratio(float(len(status_cards)), 10.0),
            _safe_ratio(float(len(zero_cost_cards)), 10.0),
            _safe_ratio(float(len(one_cost_cards)), 10.0),
            _safe_ratio(float(len(two_plus_cost_cards)), 10.0),
        ]
    )
    return features


def _action_features(state: dict[str, Any], action: dict[str, Any]) -> list[float]:
    normalized = normalize_action(action)
    action_type = normalized.get("type")
    hand_index = normalized.get("hand_index")
    option_index = normalized.get("option_index")
    selection_index = option_index if option_index is not None else hand_index
    slot = normalized.get("slot")
    target_id = normalized.get("target_id")
    card = None
    if hand_index is not None:
        for hand_card in state.get("hand_cards", []):
            if int(hand_card.get("hand_index", -1)) == int(hand_index):
                card = hand_card
                break
    if card is None and option_index is not None:
        card = _find_card_selection_option(state, option_index)
    potion = _find_potion(state, slot)
    target = _find_enemy(state, target_id)

    attack_damage = _estimate_attack_damage(card or {})
    block_amount = _estimate_block(card or {})
    is_status = 1.0 if card and _is_status_like(card) else 0.0
    requires_target = 1.0 if card and card.get("requires_target") else 0.0
    potion_requires_target = 1.0 if potion and potion.get("requires_target") else 0.0
    is_lethal = 0.0
    if target is not None and attack_damage > 0 and int(target.get("current_hp", 0)) <= attack_damage:
        is_lethal = 1.0

    return [
        1.0 if action_type == "end_turn" else 0.0,
        1.0 if action_type == "play_card" and attack_damage > 0 else 0.0,
        1.0 if action_type == "play_card" and attack_damage <= 0 and block_amount > 0 else 0.0,
        1.0 if action_type == "play_card" and attack_damage <= 0 and block_amount <= 0 else 0.0,
        1.0 if action_type == "select_hand_card" else 0.0,
        1.0 if action_type == "select_card_option" else 0.0,
        1.0 if action_type == "confirm_selection" else 0.0,
        1.0 if action_type == "cancel_selection" else 0.0,
        _safe_ratio(float(selection_index or 0), 10.0),
        _safe_ratio(float(option_index or 0), 10.0),
        _safe_ratio(float(target.get("current_hp", 0) if target else 0), float(max(1, int(target.get("max_hp", 1) if target else 1)))),
        _safe_ratio(float(target.get("block", 0) if target else 0), 20.0),
        1.0 if target and target.get("intends_to_attack") else 0.0,
        _safe_ratio(float(_enemy_incoming_damage(target) if target else 0), 20.0),
        is_lethal,
        _safe_ratio(float(card.get("energy_cost", 0) if card else 0), 3.0),
        _safe_ratio(float(attack_damage), 20.0),
        _safe_ratio(float(block_amount), 20.0),
        is_status,
        requires_target,
        1.0 if action_type == "use_potion" else 0.0,
        _safe_ratio(float(slot or 0), 3.0),
        potion_requires_target,
        _potion_is_type(potion, "dexterity_potion", "dexterity"),
        _potion_is_type(potion, "strength_potion", "flex_potion", "strength", "flex"),
        _potion_is_type(potion, "fire_potion", "attack_potion", "explosive_potion", "attack", "explosive"),
        _potion_is_type(potion, "blood_potion", "fruit_juice", "heal", "healing"),
        _potion_is_type(potion, "swift_potion", "skill_potion", "power_potion", "draw"),
    ]


@dataclass(slots=True)
class BehaviorCloningLinearPolicy:
    weights: np.ndarray
    learning_rate: float = 0.05
    l2: float = 1e-4

    def score_actions(self, state: dict[str, Any], actions: list[dict[str, Any]]) -> np.ndarray:
        if not actions:
            return np.empty((0,), dtype=np.float64)
        feature_matrix = np.vstack([featurize_state_action(state, action) for action in actions])
        feature_matrix = _align_feature_matrix(feature_matrix, self.weights.shape[0])
        return feature_matrix @ self.weights

    def choose_action(self, state: dict[str, Any]) -> dict[str, Any] | None:
        actions = enumerate_legal_actions(state)
        if not actions:
            return None
        scores = self.score_actions(state, actions)
        return actions[int(np.argmax(scores))]

    def train_step(self, state: dict[str, Any], chosen_action: dict[str, Any]) -> tuple[float, bool]:
        actions = enumerate_legal_actions(state)
        if not actions:
            return 0.0, False

        normalized_choice = action_to_key(chosen_action)
        choice_index = None
        for index, action in enumerate(actions):
            if action_to_key(action) == normalized_choice:
                choice_index = index
                break
        if choice_index is None:
            return 0.0, False

        feature_matrix = np.vstack([featurize_state_action(state, action) for action in actions])
        feature_matrix = _align_feature_matrix(feature_matrix, self.weights.shape[0])
        logits = feature_matrix @ self.weights
        probs = _softmax(logits)
        grad = feature_matrix.T @ probs - feature_matrix[choice_index]
        grad += self.l2 * self.weights
        self.weights -= self.learning_rate * grad
        loss = -np.log(max(probs[choice_index], 1e-12))
        return float(loss), True

    def evaluate_dataset(self, samples: list[dict[str, Any]]) -> dict[str, float]:
        total = 0
        correct = 0
        losses: list[float] = []
        for sample in samples:
            state = sample["state"]
            chosen_action = sample["action"]
            actions = enumerate_legal_actions(state)
            if not actions:
                continue
            normalized_choice = action_to_key(chosen_action)
            choice_index = None
            for index, action in enumerate(actions):
                if action_to_key(action) == normalized_choice:
                    choice_index = index
                    break
            if choice_index is None:
                continue
            feature_matrix = np.vstack([featurize_state_action(state, action) for action in actions])
            feature_matrix = _align_feature_matrix(feature_matrix, self.weights.shape[0])
            logits = feature_matrix @ self.weights
            probs = _softmax(logits)
            prediction_index = int(np.argmax(logits))
            total += 1
            correct += int(prediction_index == choice_index)
            losses.append(float(-np.log(max(probs[choice_index], 1e-12))))

        return {
            "samples": float(total),
            "accuracy": float(correct / total) if total else 0.0,
            "avg_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "model_type": "linear_softmax_state_action",
            "feature_names": FEATURE_NAMES,
            "weights": self.weights.tolist(),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BehaviorCloningLinearPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        weights = np.asarray(payload["weights"], dtype=np.float64)
        return cls(weights=weights)


def initialize_policy(seed: int = 0, learning_rate: float = 0.05, l2: float = 1e-4) -> BehaviorCloningLinearPolicy:
    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=0.01, size=(len(FEATURE_NAMES),))
    return BehaviorCloningLinearPolicy(weights=weights, learning_rate=learning_rate, l2=l2)


def load_demonstrations(paths: list[str | Path], limit: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                state = record.get("state")
                action = record.get("action")
                info = record.get("info") or {}
                if not isinstance(state, dict) or not isinstance(action, dict):
                    continue
                if info.get("accepted") is False:
                    continue
                samples.append(
                    {
                        "state": state,
                        "action": normalize_action(action),
                        "episode_number": record.get("episode_number"),
                        "encounter_id": record.get("encounter_id"),
                        "character_id": record.get("character_id"),
                    }
                )
                if limit is not None and len(samples) >= limit:
                    return samples
    return samples


def split_samples(
    samples: list[dict[str, Any]],
    validation_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not samples:
        return [], []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    validation_size = int(len(samples) * validation_ratio)
    if validation_ratio > 0 and validation_size == 0 and len(samples) > 1:
        validation_size = 1
    validation_indices = set(int(index) for index in indices[:validation_size])
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in validation_indices]
    validation_samples = [sample for idx, sample in enumerate(samples) if idx in validation_indices]
    if not train_samples:
        train_samples = validation_samples
        validation_samples = []
    return train_samples, validation_samples


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)
