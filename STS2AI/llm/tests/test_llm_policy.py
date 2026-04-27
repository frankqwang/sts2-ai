from __future__ import annotations

from llm.data_pipeline.action_decoder import DecodedAction
from llm.inference.llm_policy import LlmExternalPolicyAdapter


def test_llm_policy_records_reason_arithmetic_contradiction_without_fallback() -> None:
    state = {
        "battle": {
            "energy": 1,
            "player": {"block": 0},
            "hand": [
                {
                    "id": "STRIKE_IRONCLAD",
                    "preview_damage_per_target": {"1": 6},
                }
            ],
        },
        "enemies": [
            {
                "target_id": 1,
                "hp": 9,
                "block": 0,
                "intent_type": "Attack",
                "intent_damage": 6,
                "intent_hits": 1,
                "is_alive": True,
            }
        ],
    }
    legal = [{"action": "play_card", "card_index": 0, "target_id": 1}]
    adapter = LlmExternalPolicyAdapter(parse_retries=0)

    decoded = DecodedAction(
        action_index=0,
        reason="lethal CULTIST(6>=9)",
        used_fallback=False,
    )
    flags = adapter._reason_quality_flags(
        state,
        legal,
        decoded,
    )

    assert decoded.used_fallback is False
    assert "reason_math_contradiction" in flags
    assert "reason_claims_lethal_but_action_not_lethal" in flags


def test_llm_policy_records_reason_consistency_without_blocking_action(monkeypatch) -> None:
    state = {
        "battle": {
            "energy": 1,
            "player": {"block": 0},
            "hand": [
                {
                    "id": "STRIKE_IRONCLAD",
                    "preview_damage_per_target": {"1": 6},
                }
            ],
        },
        "enemies": [
            {
                "target_id": 1,
                "hp": 9,
                "block": 0,
                "intent_type": "Attack",
                "intent_damage": 6,
                "intent_hits": 1,
                "is_alive": True,
            }
        ],
    }
    legal = [{"action": "play_card", "card_index": 0, "target_id": 1}]
    adapter = LlmExternalPolicyAdapter(parse_retries=0)
    adapter._simple_gate_enabled = False

    monkeypatch.setattr(
        adapter,
        "_render_prompt",
        lambda *_args, **_kwargs: "<|im_start|>user\nstate<|im_end|>",
    )
    monkeypatch.setattr(adapter, "_activate_adapter", lambda _name: 0.0)
    monkeypatch.setattr(
        adapter,
        "_generate",
        lambda _prompt: '{"action_index":0,"reason":"take lethal: damage=6 target_hp=9"}',
    )
    monkeypatch.setattr(adapter, "_prompt_token_count", lambda _prompt: 128)

    chosen = adapter.select_action(state, legal)

    assert chosen is not None
    assert chosen["action"] == "play_card"
    assert chosen["_policy_route"] == "llm"
    assert "reason_claims_lethal_but_action_not_lethal" in chosen["_policy_quality_flags"]
    assert adapter.stats["invalid_outputs"] == 0
    assert adapter.stats["explanation_recoveries"] == 0


def test_llm_policy_records_action_score_consistency_without_blocking_action(monkeypatch) -> None:
    state = {
        "battle": {
            "energy": 1,
            "player": {"block": 0},
            "hand": [
                {
                    "id": "STRIKE_IRONCLAD",
                    "preview_damage_per_target": {"1": 6},
                }
            ],
        },
        "enemies": [
            {
                "target_id": 1,
                "hp": 9,
                "block": 0,
                "intent_type": "Attack",
                "intent_damage": 6,
                "intent_hits": 1,
                "is_alive": True,
            }
        ],
    }
    legal = [{"action": "play_card", "card_index": 0, "target_id": 1}]
    adapter = LlmExternalPolicyAdapter(parse_retries=0)
    adapter._simple_gate_enabled = False

    monkeypatch.setattr(
        adapter,
        "_render_prompt",
        lambda *_args, **_kwargs: "<|im_start|>user\nstate<|im_end|>",
    )
    monkeypatch.setattr(adapter, "_activate_adapter", lambda _name: 0.0)
    monkeypatch.setattr(
        adapter,
        "_generate",
        lambda _prompt: (
            '{"action_index":0,"reason":"deal damage",'
            '"action_scores":[{"action_index":0,"score":1,"note":"lethal enemy1: damage=6 target_hp=9"}]}'
        ),
    )
    monkeypatch.setattr(adapter, "_prompt_token_count", lambda _prompt: 128)

    chosen = adapter.select_action(state, legal)

    assert chosen is not None
    assert chosen["action"] == "play_card"
    assert chosen["_policy_route"] == "llm"
    assert "action_score_lethal_math_contradiction" in chosen["_policy_quality_flags"]
    assert adapter.stats["invalid_outputs"] == 0
    assert adapter.stats["explanation_recoveries"] == 0


def test_llm_policy_recovers_dangerous_end_turn_with_last_resort(monkeypatch) -> None:
    state = {
        "player": {"hp": 6},
        "battle": {
            "energy": 1,
            "player": {"block": 0},
            "hand": [{"id": "DEFEND_IRONCLAD", "preview_block": 5}],
        },
        "enemies": [
            {
                "target_id": 1,
                "hp": 20,
                "block": 0,
                "intent_type": "Attack",
                "intent_damage": 8,
                "intent_hits": 1,
                "is_alive": True,
            }
        ],
    }
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": -1},
        {"action": "end_turn"},
    ]
    adapter = LlmExternalPolicyAdapter(parse_retries=0)
    adapter._simple_gate_enabled = False

    monkeypatch.setattr(
        adapter,
        "_render_prompt",
        lambda *_args, **_kwargs: "<|im_start|>user\nstate<|im_end|>",
    )
    monkeypatch.setattr(adapter, "_activate_adapter", lambda _name: 0.0)
    monkeypatch.setattr(
        adapter,
        "_generate",
        lambda _prompt: '{"action_index":1,"reason":"no playable cards"}',
    )
    monkeypatch.setattr(adapter, "_prompt_token_count", lambda _prompt: 128)

    chosen = adapter.select_action(state, legal)

    assert chosen is not None
    assert chosen["action"] == "play_card"
    assert chosen["card_index"] == 0
    assert chosen["_policy_route"] == "heuristic_last_resort"
    assert adapter.stats["invalid_outputs"] == 1
    assert adapter.stats["safety_rejections"] == 1


def test_llm_policy_uses_survival_gate_before_generation(monkeypatch) -> None:
    state = {
        "state_type": "monster",
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
            {
                "target_id": 1,
                "hp": 72,
                "block": 0,
                "intent_type": "Attack",
                "intent_damage": 4,
                "intent_hits": 2,
                "is_alive": True,
            }
        ],
    }
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": 1},
        {"action": "play_card", "card_index": 1, "target_id": 1},
        {"action": "end_turn"},
    ]
    adapter = LlmExternalPolicyAdapter(parse_retries=0)
    adapter._simple_gate_enabled = False

    def fail_generate(_prompt: str) -> str:
        raise AssertionError("survival gate should skip LLM generation")

    monkeypatch.setattr(adapter, "_generate", fail_generate)

    chosen = adapter.select_action(state, legal)

    assert chosen is not None
    assert chosen["card_index"] == 1
    assert chosen["_policy_route"] == "heuristic_survival"
    assert adapter.stats["survival_gate_calls"] == 1
    assert adapter.stats["llm_calls"] == 0


def test_llm_policy_stops_dead_overlay_loop() -> None:
    adapter = LlmExternalPolicyAdapter(parse_retries=0)
    state = {"player": {"hp": 0, "max_hp": 0}}
    legal = [{"action": "overlay_press", "label": "Continue"}]

    chosen = adapter.select_action(state, legal)

    assert chosen is None
    assert adapter.stats["terminal_overlay_stops"] == 1
    assert adapter.stats["llm_calls"] == 0


def test_llm_policy_uses_last_resort_instead_of_none_after_safety_rejection(monkeypatch) -> None:
    adapter = LlmExternalPolicyAdapter(parse_retries=0)
    adapter._simple_gate_enabled = False
    adapter._survival_gate_enabled = False
    state = {
        "player": {"hp": 16, "max_hp": 80},
        "battle": {
            "energy": 1,
            "player": {"block": 0},
            "hand": [
                {"id": "BLOODLETTING", "description": "Lose 3 HP. Gain 2 Energy."},
            ],
        },
        "enemies": [
            {"target_id": 1, "hp": 100, "block": 0, "intent_type": "Attack", "intent_damage": 21, "intent_hits": 1},
        ],
    }
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": -1},
        {"action": "end_turn"},
    ]

    monkeypatch.setattr(
        adapter,
        "_render_prompt",
        lambda *_args, **_kwargs: "<|im_start|>user\nstate<|im_end|>",
    )
    monkeypatch.setattr(adapter, "_activate_adapter", lambda _name: 0.0)
    monkeypatch.setattr(
        adapter,
        "_generate",
        lambda _prompt: '{"action_index":1,"confidence":1.0,"action_scores":[],"reason":"end_turn"}',
    )
    monkeypatch.setattr(adapter, "_prompt_token_count", lambda _prompt: 128)

    chosen = adapter.select_action(state, legal)

    assert chosen is not None
    assert chosen["_policy_route"] == "heuristic_last_resort"
    assert adapter.stats["invalid_outputs"] == 1


def test_llm_policy_retries_non_strict_json_before_accepting(monkeypatch) -> None:
    adapter = LlmExternalPolicyAdapter(parse_retries=1)
    adapter._simple_gate_enabled = False
    adapter._survival_gate_enabled = False
    state = {"state_type": "map"}
    legal = [{"action": "choose_map_node", "node_id": "A"}]
    generations = iter([
        '{action_index:0,confidence:0.4,action_scores:[],reason:"json-like"}',
        '{"action_index":0,"confidence":0.5,"action_scores":[],"reason":"valid json"}',
    ])

    monkeypatch.setattr(
        adapter,
        "_render_prompt",
        lambda *_args, **_kwargs: "<|im_start|>user\nstate<|im_end|>",
    )
    monkeypatch.setattr(
        adapter,
        "_render_retry_prompt",
        lambda *_args, **_kwargs: "<|im_start|>user\nretry<|im_end|>",
    )
    monkeypatch.setattr(adapter, "_activate_adapter", lambda _name: 0.0)
    monkeypatch.setattr(adapter, "_generate", lambda _prompt: next(generations))
    monkeypatch.setattr(adapter, "_prompt_token_count", lambda _prompt: 128)

    chosen = adapter.select_action(state, legal)

    assert chosen is not None
    assert chosen["action"] == "choose_map_node"
    assert chosen["_policy_route"] == "llm"
    assert chosen["_policy_reason"] == "valid json"
    assert adapter.stats["strict_json_failures"] == 1
    assert adapter.stats["strict_json_rejections"] == 1
    assert adapter.stats["strict_json_ok"] == 1
    assert adapter.stats["retry_attempts"] == 1
    assert adapter.stats["retry_recovered"] == 1


def test_llm_policy_routes_combat_and_non_combat_adapters(tmp_path) -> None:
    combat = tmp_path / "combat"
    non_combat = tmp_path / "non_combat"
    combat.mkdir()
    non_combat.mkdir()

    adapter = LlmExternalPolicyAdapter(
        adapter_dir=None,
        combat_adapter_dir=str(combat),
        non_combat_adapter_dir=str(non_combat),
    )

    assert adapter._adapter_key_for_state({"state_type": "monster"}) == "combat"
    assert adapter._adapter_key_for_state({"state_type": "map"}) == "non_combat"
    assert adapter._adapter_key_for_state({"state_type": "card_reward"}) == "non_combat"
    assert adapter._adapter_key_for_state({"state_type": "shop"}) == "non_combat"
    assert adapter._adapter_key_for_state({"state_type": "monster"}, {"decision_type": "map_choice"}) == "non_combat"


def test_llm_policy_hot_switches_loaded_adapter_without_reload() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def set_adapter(self, name: str) -> None:
            self.calls.append(name)

    adapter = LlmExternalPolicyAdapter(adapter_dir=None)
    fake = FakeModel()
    adapter._model = fake
    adapter._tokenizer = object()

    first_ms = adapter._activate_adapter("combat")
    second_ms = adapter._activate_adapter("combat")
    third_ms = adapter._activate_adapter("non_combat")

    assert first_ms >= 0.0
    assert second_ms == 0.0
    assert third_ms >= 0.0
    assert fake.calls == ["combat", "non_combat"]
    assert adapter.stats["adapter_switches"] == 2


def test_llm_policy_suppresses_optional_potion_actions(monkeypatch) -> None:
    monkeypatch.delenv("STS2_LLM_ALLOW_POTIONS", raising=False)
    adapter = LlmExternalPolicyAdapter(adapter_dir=None)

    enabled = adapter._filter_optional_potion_actions([
        {"action": "use_potion", "slot": 0},
        {"action": "play_card", "card_index": 1},
        {"action": "end_turn"},
    ])

    assert [action["action"] for action in enabled] == ["play_card", "end_turn"]
    assert adapter.stats["potion_actions_suppressed"] == 1


def test_llm_policy_can_allow_potion_actions(monkeypatch) -> None:
    monkeypatch.setenv("STS2_LLM_ALLOW_POTIONS", "1")
    adapter = LlmExternalPolicyAdapter(adapter_dir=None)

    enabled = adapter._filter_optional_potion_actions([
        {"action": "use_potion", "slot": 0},
        {"action": "end_turn"},
    ])

    assert [action["action"] for action in enabled] == ["use_potion", "end_turn"]


def test_llm_policy_keeps_potions_for_urgent_lethal_threat(monkeypatch) -> None:
    monkeypatch.delenv("STS2_LLM_ALLOW_POTIONS", raising=False)
    adapter = LlmExternalPolicyAdapter(adapter_dir=None)
    state = {
        "player": {"hp": 6},
        "battle": {
            "player": {"block": 10, "powers": [{"id": "CONSTRICT_POWER", "amount": 6}]},
            "enemies": [{"intent_type": "Attack", "intent_damage": 12}],
        },
    }

    enabled = adapter._filter_optional_potion_actions(
        [
            {"action": "use_potion", "slot": 0, "label": "FORTIFIER"},
            {"action": "end_turn"},
        ],
        state,
    )

    assert [action["action"] for action in enabled] == ["use_potion", "end_turn"]
    assert adapter.stats["potion_actions_suppressed"] == 0


def test_llm_policy_suppresses_non_urgent_potions_during_urgent_threat(monkeypatch) -> None:
    monkeypatch.delenv("STS2_LLM_ALLOW_POTIONS", raising=False)
    adapter = LlmExternalPolicyAdapter(adapter_dir=None)
    state = {
        "player": {"hp": 6, "potions": [{"id": "ASHWATER"}, {"id": "FORTIFIER"}]},
        "battle": {
            "player": {"block": 0},
            "enemies": [{"intent_type": "Attack", "intent_damage": 8}],
        },
    }

    enabled = adapter._filter_optional_potion_actions(
        [
            {"action": "use_potion", "slot": 0, "label": "ASHWATER"},
            {"action": "use_potion", "slot": 1, "label": "FORTIFIER"},
            {"action": "end_turn"},
        ],
        state,
    )

    assert [(action["action"], action.get("label")) for action in enabled] == [
        ("use_potion", "FORTIFIER"),
        ("end_turn", None),
    ]
    assert adapter.stats["potion_actions_suppressed"] == 1
