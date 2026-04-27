from __future__ import annotations

from llm.scripts.extract_uncertainty_cases import bucket_cases


def test_bucket_cases_splits_low_and_high_margin() -> None:
    rows = [
        {
            "decoded": {
                "action_index": 0,
                "confidence": 0.4,
                "action_scores": [
                    {"action_index": 0, "score": 10.0},
                    {"action_index": 1, "score": 9.5},
                ],
            },
        },
        {
            "decoded": {
                "action_index": 2,
                "confidence": 0.9,
                "action_scores": [
                    {"action_index": 2, "score": 20.0},
                    {"action_index": 0, "score": 5.0},
                ],
            },
            "quality_flags": ["missed_visible_lethal"],
        },
    ]

    buckets = bucket_cases(rows, low_margin=1.0, high_margin=5.0, low_confidence=0.55)

    assert len(buckets["low_margin"]) == 1
    assert len(buckets["low_confidence"]) == 1
    assert len(buckets["high_margin"]) == 1
    assert len(buckets["high_margin_with_flags"]) == 1


def test_bucket_cases_accepts_mechanism_eval_rows() -> None:
    rows = [
        {
            "row_index": 3,
            "parse_status": "json_parse_failed",
            "generated_action_index": 4,
            "target_action_index": 2,
            "action_valid": True,
            "action_exact": False,
            "confidence": 0.8,
            "action_scores": [
                {"action_index": 4, "score": 7.0},
                {"action_index": 2, "score": 1.0},
            ],
        }
    ]

    buckets = bucket_cases(rows, low_margin=1.0, high_margin=5.0, low_confidence=0.55)

    assert len(buckets["invalid_output"]) == 1
    assert len(buckets["high_margin"]) == 1
    assert buckets["invalid_output"][0]["action_index"] == 4
    assert buckets["invalid_output"][0]["target_action_index"] == 2


def test_bucket_cases_flags_confidence_margin_mismatch() -> None:
    rows = [
        {
            "decoded": {
                "action_index": 0,
                "confidence": 0.95,
                "action_scores": [
                    {"action_index": 0, "score": 10.0},
                    {"action_index": 1, "score": 10.0},
                ],
            },
        },
        {
            "decoded": {
                "action_index": 2,
                "confidence": 0.5,
                "action_scores": [
                    {"action_index": 2, "score": 20.0},
                    {"action_index": 0, "score": 1.0},
                ],
            },
        },
    ]

    buckets = bucket_cases(rows, low_margin=1.0, high_margin=5.0, low_confidence=0.55)

    assert len(buckets["high_confidence_low_margin"]) == 1
    assert len(buckets["low_confidence_high_margin"]) == 1


def test_bucket_cases_can_recompute_stale_quality_flags() -> None:
    rows = [
        {
            "decoded": {
                "action_index": 0,
                "confidence": 0.9,
                "action_scores": [
                    {"action_index": 0, "score": 10.0},
                    {"action_index": 1, "score": 1.0},
                ],
            },
            "quality_flags": ["dangerous_end_turn"],
            "state": {
                "battle": {"energy": 0, "player": {"block": 0}},
                "enemies": [{"intent_type": "Attack", "intent_damage": 8, "is_alive": True}],
            },
            "legal_actions": [{"action": "end_turn"}],
        }
    ]

    stale = bucket_cases(rows, low_margin=1.0, high_margin=5.0, low_confidence=0.55)
    recomputed = bucket_cases(
        rows,
        low_margin=1.0,
        high_margin=5.0,
        low_confidence=0.55,
        recompute_quality=True,
    )

    assert len(stale["high_margin_with_flags"]) == 1
    assert len(recomputed["high_margin_with_flags"]) == 0
