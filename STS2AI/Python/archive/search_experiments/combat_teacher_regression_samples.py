from __future__ import annotations

from typing import Any

from combat_teacher_common import canonical_public_state_hash, stable_sample_id
from combat_teacher_dataset import CombatTeacherSample, stable_split


def _make_combat_state(
    hand_cards: list[dict[str, Any]],
    *,
    enemy_hp: int = 30,
    block: int = 0,
    potions: list[dict[str, Any]] | None = None,
    intent_damage: int = 8,
) -> dict[str, Any]:
    enemy_entity = "cultist-0"
    return {
        "state_type": "monster",
        "run": {
            "floor": 7,
            "act": 1,
        },
        "battle": {
            "round": 1,
            "turn": "player",
            "is_play_phase": True,
            "player": {
                "character": "IRONCLAD",
                "hp": 55,
                "max_hp": 80,
                "block": block,
                "energy": 3,
                "max_energy": 3,
                "gold": 99,
                "draw_pile_count": 5,
                "discard_pile_count": 0,
                "exhaust_pile_count": 0,
                "potions": list(potions or []),
                "hand": hand_cards,
            },
            "enemies": [
                {
                    "entity_id": enemy_entity,
                    "combat_id": 0,
                    "name": "Cultist",
                    "hp": enemy_hp,
                    "max_hp": enemy_hp,
                    "block": 0,
                    "intents": [{"type": "attack", "label": str(intent_damage), "total_damage": intent_damage}],
                    "status": [],
                }
            ],
        },
        "player": {
            "hp": 55,
            "max_hp": 80,
            "gold": 99,
            "potions": list(potions or []),
        },
    }


def _sample(
    *,
    sample_key: str,
    split: str,
    motif_labels: list[str],
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    baseline_best_action_index: int,
    best_action_index: int,
    per_action_score: list[float],
    per_action_regret: list[float],
    continuation_targets: dict[str, float],
    best_full_turn_line: list[dict[str, Any]] | None = None,
) -> CombatTeacherSample:
    sample_id = stable_sample_id(sample_key, state, legal_actions)
    return CombatTeacherSample(
        schema_version="combat_teacher_dataset.v1",
        sample_id=sample_id,
        split=split or stable_split(sample_id),
        source_bucket="motif_regression",
        source_seed=sample_key,
        source_checkpoint="regression",
        state_hash=canonical_public_state_hash(state),
        motif_labels=list(motif_labels),
        state=state,
        legal_actions=legal_actions,
        baseline_logits=[0.0 for _ in legal_actions],
        baseline_probs=[1.0 / max(1, len(legal_actions)) for _ in legal_actions],
        baseline_best_action_index=int(baseline_best_action_index),
        best_action_index=int(best_action_index),
        best_full_turn_line=[dict(item) for item in (best_full_turn_line or [legal_actions[best_action_index]])],
        per_action_score=[float(v) for v in per_action_score],
        per_action_regret=[float(v) for v in per_action_regret],
        root_value=float(max(per_action_score)),
        leaf_breakdown={"total": float(max(per_action_score))},
        continuation_targets={str(k): float(v) for k, v in continuation_targets.items()},
    )


def build_regression_motif_samples() -> list[CombatTeacherSample]:
    enemy_entity = "cultist-0"
    samples: list[CombatTeacherSample] = []

    # Bash before Strike
    for split, sample_key, enemy_hp, intent_damage, extra_cards, scores in (
        ("train", "reg-bash-train-01", 20, 8, [], [1.0, 0.55, 0.15, -0.25]),
        ("train", "reg-bash-train-02", 18, 11, [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}], [1.05, 0.45, 0.2, -0.1, -0.35]),
        ("train", "reg-bash-train-03", 22, 12, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.1, 0.6, 0.35, 0.1, -0.4]),
        ("train", "reg-bash-train-04", 16, 14, [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}, {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.15, 0.55, 0.45, 0.2, 0.05, -0.45]),
        ("holdout", "reg-bash-holdout-01", 24, 10, [], [1.0, 0.5, 0.1, -0.3]),
        ("holdout", "reg-bash-holdout-02", 17, 12, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.1, 0.5, 0.25, 0.05, -0.4]),
    ):
        bash_state = _make_combat_state(
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                *extra_cards,
            ],
            enemy_hp=enemy_hp,
            intent_damage=intent_damage,
        )
        bash_actions = [
            {"action": "play_card", "card_index": 0, "label": "BASH", "card_id": "BASH", "target_id": enemy_entity, "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "STRIKE_IRONCLAD", "card_id": "STRIKE_IRONCLAD", "target_id": enemy_entity, "is_enabled": True},
            {"action": "play_card", "card_index": 2, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": enemy_entity, "is_enabled": True},
            *[
                {
                    "action": "play_card",
                    "card_index": 3 + idx,
                    "label": card["id"],
                    "card_id": card["id"],
                    "target_id": enemy_entity,
                    "is_enabled": True,
                }
                for idx, card in enumerate(extra_cards)
            ],
            {"action": "end_turn", "is_enabled": True},
        ]
        samples.append(
            _sample(
                sample_key=sample_key,
                split=split,
                motif_labels=["bash_before_strike", "bad_end_turn"],
                state=bash_state,
                legal_actions=bash_actions,
                baseline_best_action_index=1,
                best_action_index=0,
                per_action_score=scores,
                per_action_regret=[max(scores) - value for value in scores],
                continuation_targets={"win_prob": 0.7, "expected_hp_loss": 4.0, "expected_potion_cost": 0.0},
            )
        )

    # Direct lethal first action
    for split, sample_key, enemy_hp, hand_cards, scores in (
        ("train", "reg-lethal-train-01", 6, [], [1.0, -0.5]),
        ("train", "reg-lethal-train-02", 8, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.0, 0.2, -0.6]),
        ("train", "reg-lethal-train-03", 5, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.05, 0.15, -0.7]),
        ("train", "reg-lethal-train-04", 6, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}, {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.1, 0.25, 0.2, -0.8]),
        ("train", "reg-lethal-train-05", 7, [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}], [1.08, 0.5, -0.65]),
        ("train", "reg-lethal-train-06", 9, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}, {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}], [1.12, 0.35, 0.3, -0.75]),
        (
            "train",
            "reg-lethal-train-07",
            8,
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
            ],
            [1.12, 0.2, -0.75],
        ),
        (
            "train",
            "reg-lethal-train-08",
            9,
            [
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": True},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
            ],
            [1.1, 0.18, -0.7],
        ),
        (
            "train",
            "reg-lethal-train-09",
            6,
            [
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
            ],
            [1.14, 0.28, 0.05, -0.78],
        ),
        (
            "train",
            "reg-lethal-train-10",
            8,
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
            ],
            [1.15, 0.35, 0.2, -0.8],
        ),
        ("holdout", "reg-lethal-holdout-01", 7, [], [1.0, -0.5]),
        ("holdout", "reg-lethal-holdout-02", 9, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.0, 0.1, -0.7]),
        ("holdout", "reg-lethal-holdout-03", 6, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.05, 0.2, -0.75]),
        ("holdout", "reg-lethal-holdout-04", 8, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}, {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.1, 0.3, 0.25, -0.85]),
        (
            "holdout",
            "reg-lethal-holdout-05",
            8,
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
            ],
            [1.08, 0.16, -0.76],
        ),
        (
            "holdout",
            "reg-lethal-holdout-06",
            9,
            [
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": True},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
            ],
            [1.07, 0.12, -0.74],
        ),
    ):
        lethal_state = _make_combat_state(
            hand_cards or [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}],
            enemy_hp=enemy_hp,
        )
        lethal_actions = [
            {
                "action": "play_card",
                "card_index": 0,
                "label": (hand_cards[0]["id"] if hand_cards else "STRIKE_IRONCLAD"),
                "card_id": (hand_cards[0]["id"] if hand_cards else "STRIKE_IRONCLAD"),
                "target_id": enemy_entity,
                "is_enabled": True,
            },
            *[
                {
                    "action": "play_card",
                    "card_index": 1 + idx,
                    "label": card["id"],
                    "card_id": card["id"],
                    "target_id": enemy_entity,
                    "is_enabled": True,
                }
                for idx, card in enumerate((hand_cards or [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}])[1:])
            ],
            {"action": "end_turn", "is_enabled": True},
        ]
        samples.append(
            _sample(
                sample_key=sample_key,
                split=split,
                motif_labels=["missed_lethal", "direct_lethal_first_action", "bad_end_turn"],
                state=lethal_state,
                legal_actions=lethal_actions,
                baseline_best_action_index=min(1, len(lethal_actions) - 1),
                best_action_index=0,
                per_action_score=scores,
                per_action_regret=[max(scores) - value for value in scores],
                continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
            )
        )

    # Turn lethal without ending turn
    for split, sample_key, enemy_hp, scores in (
        ("train", "reg-turnlethal-train-01", 12, [1.0, 1.0, 0.2, -0.7]),
        ("train", "reg-turnlethal-train-02", 11, [1.05, 1.05, 0.25, -0.75]),
        ("train", "reg-turnlethal-train-03", 13, [1.1, 1.1, 0.35, -0.8]),
        ("train", "reg-turnlethal-train-04", 12, [1.08, 1.08, 0.3, -0.78]),
        ("holdout", "reg-turnlethal-holdout-01", 12, [1.0, 1.0, 0.15, -0.8]),
        ("holdout", "reg-turnlethal-holdout-02", 10, [1.05, 1.05, 0.2, -0.85]),
    ):
        cards = [
            {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
        ]
        lethal_turn_state = _make_combat_state(cards, enemy_hp=enemy_hp)
        lethal_turn_actions = [
            {"action": "play_card", "card_index": 0, "label": "STRIKE_IRONCLAD", "card_id": "STRIKE_IRONCLAD", "target_id": enemy_entity, "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "STRIKE_IRONCLAD", "card_id": "STRIKE_IRONCLAD", "target_id": enemy_entity, "is_enabled": True},
            {"action": "play_card", "card_index": 2, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": enemy_entity, "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        samples.append(
            _sample(
                sample_key=sample_key,
                split=split,
                motif_labels=["missed_lethal", "turn_lethal_no_end_turn", "bad_end_turn"],
                state=lethal_turn_state,
                legal_actions=lethal_turn_actions,
                baseline_best_action_index=2,
                best_action_index=0,
                per_action_score=scores,
                per_action_regret=[max(scores) - value for value in scores],
                continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
                best_full_turn_line=[dict(lethal_turn_actions[0]), dict(lethal_turn_actions[1])],
            )
        )

    # Body Slam before block
    for split, sample_key, enemy_hp, intent_damage, extra_cards, scores in (
        ("train", "reg-bodyslam-train-01", 18, 8, [], [1.0, 0.1, -0.2]),
        ("train", "reg-bodyslam-train-02", 16, 12, [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}], [1.05, 0.15, 0.2, -0.25]),
        ("holdout", "reg-bodyslam-holdout-01", 22, 10, [], [1.0, 0.1, -0.2]),
        ("holdout", "reg-bodyslam-holdout-02", 20, 13, [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}], [1.0, 0.05, 0.25, -0.3]),
    ):
        body_slam_state = _make_combat_state(
            [
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                {"id": "BODY_SLAM", "name": "Body Slam", "cost": 1, "is_upgraded": False},
                *extra_cards,
            ],
            enemy_hp=enemy_hp,
            block=0,
            intent_damage=intent_damage,
        )
        body_slam_actions = [
            {"action": "play_card", "card_index": 0, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": enemy_entity, "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "BODY_SLAM", "card_id": "BODY_SLAM", "target_id": enemy_entity, "is_enabled": True},
            *[
                {
                    "action": "play_card",
                    "card_index": 2 + idx,
                    "label": card["id"],
                    "card_id": card["id"],
                    "target_id": enemy_entity,
                    "is_enabled": True,
                }
                for idx, card in enumerate(extra_cards)
            ],
            {"action": "end_turn", "is_enabled": True},
        ]
        samples.append(
            _sample(
                sample_key=sample_key,
                split=split,
                motif_labels=["bodyslam_before_block", "bad_end_turn"],
                state=body_slam_state,
                legal_actions=body_slam_actions,
                baseline_best_action_index=1,
                best_action_index=0,
                per_action_score=scores,
                per_action_regret=[max(scores) - value for value in scores],
                continuation_targets={"win_prob": 0.8, "expected_hp_loss": 3.0, "expected_potion_cost": 0.0},
            )
        )

    # Potion misuse: prefer defend over wasting potion when incoming pressure is low.
    for split, sample_key, potion_id, hand_cards, scores in (
        (
            "train",
            "reg-potion-train-01",
            "WEAK_POTION",
            [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}],
            [0.8, -0.1, -0.3],
        ),
        (
            "train",
            "reg-potion-train-02",
            "STRENGTH_POTION",
            [
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ],
            [0.85, 0.4, -0.15, -0.35],
        ),
        (
            "holdout",
            "reg-potion-holdout-01",
            "FIRE_POTION",
            [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}],
            [0.8, -0.1, -0.3],
        ),
        (
            "holdout",
            "reg-potion-holdout-02",
            "EXPLOSIVE_POTION",
            [
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ],
            [0.9, 0.45, -0.2, -0.45],
        ),
    ):
        potion_state = _make_combat_state(
            hand_cards,
            enemy_hp=30,
            block=0,
            potions=[{"slot": 0, "id": potion_id, "name": potion_id, "can_use_in_combat": True}],
            intent_damage=0,
        )
        potion_actions = [
            *[
                {
                    "action": "play_card",
                    "card_index": idx,
                    "label": card["id"],
                    "card_id": card["id"],
                    "target_id": enemy_entity,
                    "is_enabled": True,
                }
                for idx, card in enumerate(hand_cards)
            ],
            {"action": "use_potion", "slot": 0, "label": potion_id, "target_id": enemy_entity, "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        samples.append(
            _sample(
                sample_key=sample_key,
                split=split,
                motif_labels=["potion_misuse", "bad_end_turn"],
                state=potion_state,
                legal_actions=potion_actions,
                baseline_best_action_index=len(hand_cards),
                best_action_index=0,
                per_action_score=scores,
                per_action_regret=[max(scores) - value for value in scores],
                continuation_targets={"win_prob": 0.55, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
            )
        )
    return samples
