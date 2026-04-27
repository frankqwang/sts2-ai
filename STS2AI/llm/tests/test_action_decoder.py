from __future__ import annotations

from llm.data_pipeline.action_decoder import (
    action_score_margin,
    decode_action,
    decode_structured_action,
    format_structured_action_json,
)


def test_decode_action_accepts_json_like_model_output() -> None:
    legal = [{"action": "play_card"}, {"action": "end_turn"}]

    decoded = decode_action('{action_index: 1, reason: "done"}\n<tool_call>', legal)

    assert decoded.action_index == 1
    assert decoded.reason == "done"
    assert decoded.used_fallback is False


def test_decode_action_extracts_confidence_and_scores() -> None:
    legal = [{"action": "play_card"}, {"action": "end_turn"}]

    decoded = decode_action(
        '{"action_index":0,"confidence":0.82,'
        '"action_scores":[{"action_index":0,"score":9.5},{"action_index":1,"score":8.0}],'
        '"reason":"close"}',
        legal,
    )

    assert decoded.action_index == 0
    assert decoded.confidence == 0.82
    assert decoded.action_scores == (
        {"action_index": 0, "score": 9.5},
        {"action_index": 1, "score": 8.0},
    )
    assert action_score_margin(decoded.action_scores) == 1.5


def test_decode_action_rejects_out_of_range_lax_index() -> None:
    legal = [{"action": "play_card"}, {"action": "end_turn"}]

    decoded = decode_action("{action_index: 9, reason: 'bad'}", legal, fallback_index=1)

    assert decoded.action_index == 1
    assert decoded.reason == "bad"
    assert decoded.used_fallback is True
    assert decoded.fallback_reason == "action_index_out_of_range"


def test_decode_structured_action_maps_card_and_target_to_legal_index() -> None:
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": 1},
        {"action": "play_card", "card_index": 0, "target_id": 2},
        {"action": "end_turn"},
    ]

    decoded = decode_structured_action(
        '{"action":"play_card","hand_index":0,"target_id":2,"reason":"kill"}',
        legal,
    )

    assert decoded.action_index == 1
    assert decoded.reason == "kill"
    assert decoded.used_fallback is False


def test_decode_structured_action_rejects_ambiguous_missing_target() -> None:
    legal = [
        {"action": "play_card", "card_index": 0, "target_id": 1},
        {"action": "play_card", "card_index": 0, "target_id": 2},
        {"action": "end_turn"},
    ]

    decoded = decode_structured_action('{"action":"play_card","hand_index":0}', legal)

    assert decoded.used_fallback is True
    assert decoded.fallback_reason == "ambiguous_action"


def test_decode_structured_action_maps_end_turn_without_fixed_index() -> None:
    legal = [
        {"action": "play_card", "card_index": 1, "target_id": 1},
        {"action": "end_turn"},
    ]

    decoded = decode_structured_action('{"action":"end_turn","reason":"done"}', legal)

    assert decoded.action_index == 1
    assert decoded.used_fallback is False


def test_format_structured_action_json_omits_target_for_self_card() -> None:
    rendered = format_structured_action_json(
        {"action": "play_card", "card_index": 2, "target_id": -1},
        "block",
    )

    assert rendered == '{"action": "play_card", "hand_index": 2, "reason": "block"}'
