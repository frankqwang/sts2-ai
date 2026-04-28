from __future__ import annotations

from llm.inference.hybrid_gate import choose_simple_action, choose_survival_action


def test_choose_only_end_turn() -> None:
    decision = choose_simple_action({}, [{"action": "end_turn"}])

    assert decision is not None
    assert decision.action_index == 0
    assert decision.route == "heuristic_forced"
    assert "end_turn" in decision.reason


def test_choose_only_play_card() -> None:
    decision = choose_simple_action({}, [{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}])

    assert decision is not None
    assert decision.action_index == 0
    assert "STRIKE_IRONCLAD" in decision.reason


def test_multi_target_choice_stays_with_llm() -> None:
    decision = choose_simple_action(
        {},
        [
            {"action": "play_card", "card_id": "STRIKE_IRONCLAD", "target_id": 1},
            {"action": "play_card", "card_id": "STRIKE_IRONCLAD", "target_id": 2},
            {"action": "end_turn"},
        ],
    )

    assert decision is None


def test_claim_reward_is_forced_before_proceed() -> None:
    decision = choose_simple_action(
        {},
        [
            {"action": "claim_reward", "label": "11 Gold"},
            {"action": "claim_reward", "label": "Add a card to your deck."},
            {"action": "proceed"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 0
    assert "claim visible reward" in decision.reason


def test_unclaimable_potion_reward_is_skipped_before_proceed() -> None:
    decision = choose_simple_action(
        {
            "player": {"open_potion_slots": 0},
            "rewards": {
                "items": [
                    {"index": 0, "type": "potion", "label": "Vulnerable Potion", "claimable": False},
                ],
                "can_proceed": True,
            },
        },
        [
            {"action": "claim_reward", "index": 0, "label": "Vulnerable Potion"},
            {"action": "proceed"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1
    assert "unclaimable" in decision.reason


def test_claimable_card_reward_is_selected_when_potion_slots_are_full() -> None:
    decision = choose_simple_action(
        {
            "player": {"open_potion_slots": 0},
            "rewards": {
                "items": [
                    {"index": 0, "type": "potion", "label": "Vulnerable Potion", "claimable": False},
                    {"index": 1, "type": "card", "label": "Add a card to your deck.", "claimable": True},
                ],
                "can_proceed": True,
            },
        },
        [
            {"action": "claim_reward", "index": 0, "label": "Vulnerable Potion"},
            {"action": "claim_reward", "index": 1, "label": "Add a card to your deck."},
            {"action": "proceed"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1
    assert "Add a card" in decision.reason


def test_claim_reward_label_resolves_compacted_reward_index() -> None:
    decision = choose_simple_action(
        {
            "player": {"open_potion_slots": 0},
            "rewards": {
                "items": [
                    {"index": 0, "type": "potion", "label": "Vulnerable Potion", "claimable": False},
                    {"index": 1, "type": "card", "label": "Add a card to your deck.", "claimable": True},
                ],
                "can_proceed": True,
            },
        },
        [
            {"action": "claim_reward", "index": 0, "label": "Add a card to your deck."},
            {"action": "proceed"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 0
    assert "Add a card" in decision.reason


def test_optional_potion_only_choice_prefers_end_turn() -> None:
    decision = choose_simple_action(
        {},
        [
            {"action": "use_potion", "slot": 0, "label": "POWER_POTION"},
            {"action": "use_potion", "slot": 1, "label": "REGEN_POTION"},
            {"action": "end_turn"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 2
    assert "avoid optional potion" in decision.reason


def test_optional_potion_only_choice_uses_potion_when_end_turn_is_lethal() -> None:
    state = {
        "player": {"hp": 6},
        "battle": {
            "player": {"block": 10, "powers": [{"id": "CONSTRICT_POWER", "amount": 6}]},
            "enemies": [{"intent_type": "Attack", "intent_damage": 12}],
        },
    }
    decision = choose_simple_action(
        state,
        [
            {"action": "use_potion", "slot": 0, "label": "FORTIFIER"},
            {"action": "end_turn"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 0
    assert "urgent end_turn" in decision.reason


def test_optional_non_defensive_potion_does_not_override_end_turn() -> None:
    state = {
        "player": {"hp": 6, "potions": [{"id": "ASHWATER"}]},
        "battle": {
            "player": {"block": 0},
            "enemies": [{"intent_type": "Attack", "intent_damage": 8}],
        },
    }
    decision = choose_simple_action(
        state,
        [
            {"action": "use_potion", "slot": 0, "label": "ASHWATER"},
            {"action": "end_turn"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1


def test_card_removal_prefers_basic_strike_over_bash() -> None:
    decision = choose_simple_action(
        {"state_type": "card_select"},
        [
            {"action": "select_card", "index": 0, "card_index": 0, "card_id": "STRIKE_IRONCLAD", "label": "remove_card STRIKE_IRONCLAD"},
            {"action": "select_card", "index": 1, "card_index": 1, "card_id": "DEFEND_IRONCLAD", "label": "remove_card DEFEND_IRONCLAD"},
            {"action": "select_card", "index": 2, "card_index": 2, "card_id": "BASH", "label": "remove_card BASH"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 0
    assert "STRIKE_IRONCLAD" in decision.reason


def test_card_removal_confirms_good_selection() -> None:
    decision = choose_simple_action(
        {"card_select": {"selected_cards": [{"id": "STRIKE_IRONCLAD"}]}},
        [
            {"action": "confirm_selection", "label": "confirm remove_card"},
            {"action": "cancel_selection", "label": "cancel remove_card"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 0


def test_card_removal_cancels_protected_bash_selection() -> None:
    decision = choose_simple_action(
        {"card_select": {"selected_cards": [{"id": "BASH"}]}},
        [
            {"action": "confirm_selection", "label": "confirm remove_card"},
            {"action": "cancel_selection", "label": "cancel remove_card"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1
    assert "protected" in decision.reason


def test_disabled_actions_do_not_make_complex_choice() -> None:
    decision = choose_simple_action(
        {},
        [
            {"action": "play_card", "card_id": "DEFEND_IRONCLAD", "is_enabled": False},
            {"action": "end_turn"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1


def test_combat_card_selection_confirms_when_available() -> None:
    decision = choose_simple_action(
        {},
        [
            {"action": "combat_select_card", "card_index": 0},
            {"action": "combat_select_card", "card_index": 1},
            {"action": "combat_confirm_selection", "label": "Confirm"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 2
    assert "confirm" in decision.reason


def test_shop_gate_leaves_instead_of_repeated_purchase() -> None:
    decision = choose_simple_action(
        {"state_type": "shop"},
        [
            {"action": "shop_purchase", "label": "Body Slam"},
            {"action": "shop_purchase", "label": "Anger"},
            {"action": "proceed"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 2
    assert "leave shop" in decision.reason


def test_rest_gate_heals_when_hp_is_low() -> None:
    decision = choose_simple_action(
        {"state_type": "rest_site", "player": {"hp": 25, "max_hp": 80}},
        [
            {"action": "choose_rest_option", "label": "heal"},
            {"action": "choose_rest_option", "label": "smith"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 0
    assert "recover" in decision.reason


def test_tablet_of_truth_safety_gate_stops_repeated_deciphering() -> None:
    decision = choose_simple_action(
        {
            "state_type": "event",
            "player": {"hp": 75, "max_hp": 77},
            "event": {"event_id": "EVENT.TABLET_OF_TRUTH"},
        },
        [
            {"action": "choose_event_option", "label": "Continue Deciphering"},
            {"action": "choose_event_option", "label": "Give Up"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1
    assert "Tablet of Truth" in decision.reason


def test_upgrade_selection_gate_picks_high_impact_card() -> None:
    decision = choose_simple_action(
        {"state_type": "card_select"},
        [
            {"action": "select_card", "purpose": "upgrade_card", "card_id": "STRIKE_IRONCLAD"},
            {"action": "select_card", "purpose": "upgrade_card", "card_id": "BASH"},
            {"action": "cancel_selection", "purpose": "upgrade_card"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1
    assert "BASH" in decision.reason


def test_survival_gate_chooses_block_at_low_hp() -> None:
    state = {
        "player": {"hp": 16, "max_hp": 80},
        "battle": {
            "energy": 3,
            "player": {"block": 0},
            "hand": [
                {"id": "STRIKE_IRONCLAD", "preview_damage_per_target": {"1": 6}},
                {"id": "DEFEND_IRONCLAD", "preview_block": 5},
            ],
        },
        "enemies": [
            {"target_id": 1, "hp": 30, "block": 0, "intent_type": "Attack", "intent_damage": 10, "intent_hits": 1},
        ],
    }

    decision = choose_survival_action(
        state,
        [
            {"action": "play_card", "card_index": 0, "target_id": 1},
            {"action": "play_card", "card_index": 1, "target_id": -1},
            {"action": "end_turn"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1
    assert decision.route == "heuristic_survival"


def test_survival_gate_chooses_weak_when_no_block_is_visible() -> None:
    state = {
        "player": {"hp": 15, "max_hp": 80},
        "battle": {
            "energy": 3,
            "player": {"block": 0},
            "hand": [
                {"id": "MOLTEN_FIST", "description": "Deal 14 damage.", "preview_damage_per_target": {"1": 14}},
                {
                    "id": "UPPERCUT",
                    "description": "Deal 13 damage. Apply 1 Weak. Apply 1 Vulnerable.",
                    "preview_damage_per_target": {"1": 13},
                },
            ],
        },
        "enemies": [
            {"target_id": 1, "hp": 72, "block": 0, "intent_type": "Attack", "intent_damage": 4, "intent_hits": 2},
        ],
    }

    decision = choose_survival_action(
        state,
        [
            {"action": "play_card", "card_index": 0, "target_id": 1},
            {"action": "play_card", "card_index": 1, "target_id": 1},
            {"action": "end_turn"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1
    assert "UPPERCUT" in decision.reason


def test_survival_gate_does_not_override_safe_high_hp_turn() -> None:
    state = {
        "player": {"hp": 70, "max_hp": 80},
        "battle": {
            "energy": 3,
            "player": {"block": 0},
            "hand": [{"id": "DEFEND_IRONCLAD", "preview_block": 5}],
        },
        "enemies": [
            {"target_id": 1, "hp": 30, "block": 0, "intent_type": "Attack", "intent_damage": 10, "intent_hits": 1},
        ],
    }

    decision = choose_survival_action(
        state,
        [
            {"action": "play_card", "card_index": 0, "target_id": -1},
            {"action": "end_turn"},
        ],
    )

    assert decision is None


def test_survival_gate_blocks_significant_hp_loss_before_low_hp() -> None:
    state = {
        "player": {"hp": 59, "max_hp": 80},
        "battle": {
            "energy": 3,
            "player": {"block": 0},
            "hand": [
                {"id": "STRIKE_IRONCLAD", "preview_damage_per_target": {"1": 6}},
                {"id": "DEFEND_IRONCLAD", "preview_block": 5},
            ],
        },
        "enemies": [
            {"target_id": 1, "hp": 30, "block": 0, "intent_type": "Attack", "intent_damage": 19, "intent_hits": 1},
        ],
    }

    decision = choose_survival_action(
        state,
        [
            {"action": "play_card", "card_index": 0, "target_id": 1},
            {"action": "play_card", "card_index": 1, "target_id": -1},
            {"action": "end_turn"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 1
    assert "reduce immediate hp loss" in decision.reason


def test_survival_gate_uses_legal_action_block_when_card_preview_is_missing() -> None:
    state = {
        "player": {"hp": 64, "max_hp": 80},
        "battle": {
            "energy": 3,
            "player": {"block": 0},
            "hand": [
                {"id": "DEFEND_IRONCLAD"},
                {"id": "HELLRAISER"},
            ],
        },
        "enemies": [
            {"target_id": 1, "hp": 26, "block": 0, "intent_type": "Attack", "intent_damage": 14, "intent_hits": 1},
        ],
    }

    decision = choose_survival_action(
        state,
        [
            {"action": "play_card", "card_index": 0, "target_id": -1, "block": 5},
            {"action": "play_card", "card_index": 1, "target_id": -1},
            {"action": "end_turn"},
        ],
    )

    assert decision is not None
    assert decision.action_index == 0


def test_survival_gate_never_chooses_self_lethal_block() -> None:
    state = {
        "player": {"hp": 1, "max_hp": 87},
        "battle": {
            "energy": 3,
            "player": {"block": 0},
            "hand": [
                {"id": "STRIKE_IRONCLAD", "preview_damage_per_target": {"1": 12}},
                {"id": "DEFEND_IRONCLAD", "preview_block": 5},
                {"id": "DEFEND_IRONCLAD", "preview_block": 5},
                {"id": "BLOOD_WALL", "description": "Lose 2 HP. Gain 16 Block.", "preview_block": 16},
            ],
        },
        "enemies": [
            {"target_id": 1, "hp": 31, "block": 0, "intent_type": "Attack", "intent_damage": 25, "intent_hits": 1},
        ],
    }

    decision = choose_survival_action(
        state,
        [
            {"action": "play_card", "card_index": 0, "target_id": 1, "damage": 12},
            {"action": "play_card", "card_index": 1, "target_id": -1, "block": 5},
            {"action": "play_card", "card_index": 2, "target_id": -1, "block": 5},
            {"action": "play_card", "card_index": 3, "target_id": -1, "block": 16, "self_hp_loss": 2},
            {"action": "end_turn"},
        ],
    )

    assert decision is not None
    assert decision.action_index in {1, 2}
    assert "BLOOD_WALL" not in decision.reason
