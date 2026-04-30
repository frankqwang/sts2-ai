from __future__ import annotations

from llm.scripts.automation.self_iterate import (
    _candidate_passes,
    _strict_json_failure_rate,
)


def test_strict_json_failure_rate_falls_back_when_field_missing():
    """policy_stats 不再写 strict_json_failures 时，分子退到 first_attempt_invalid - retry_recovered。"""
    metrics = {
        "policy_stats": {
            "strict_json_ok": 100,
            "first_attempt_invalid": 5,
            "retry_recovered": 2,
            "generated_outputs": 105,
        }
    }
    rate = _strict_json_failure_rate(metrics)
    # failures = 5 - 2 = 3, denom = max(105, 100+3) = 105
    assert abs(rate - (3 / 105)) < 1e-9


def test_strict_json_failure_rate_uses_generated_outputs_as_denom():
    """分母优先 generated_outputs，避免命名漂移导致 gate 失效。"""
    metrics = {
        "policy_stats": {
            "strict_json_ok": 100,
            "strict_json_failures": 5,
            "generated_outputs": 200,
        }
    }
    rate = _strict_json_failure_rate(metrics)
    assert abs(rate - (5 / 200)) < 1e-9


def test_strict_json_failure_rate_zero_when_no_stats():
    assert _strict_json_failure_rate({}) == 0.0
    assert _strict_json_failure_rate({"policy_stats": "not_a_dict"}) == 0.0


def _base_metrics() -> dict:
    return {
        "win_rate": 1.0,
        "reward": {"avg": 7.0},
        "invalid_output_episode_rate": 0.0,
        "mechanism_score": {"avg": 0.9},
        "action_quality": {
            "reason_math_contradiction": 1,
            "reason_claims_lethal_but_action_not_lethal": 1,
        },
        "by_encounter": {
            "A": {
                "encounter_label": "A",
                "win_rate": 1.0,
                "reward": {"avg": 7.0},
            }
        },
    }


def test_candidate_gate_rejects_reason_consistency_regression() -> None:
    current = _base_metrics()
    candidate = _base_metrics()
    candidate["action_quality"] = {
        "reason_math_contradiction": 2,
        "reason_claims_lethal_but_action_not_lethal": 3,
    }

    passed, reasons = _candidate_passes(
        current=current,
        candidate=candidate,
        min_win_rate_delta=0.0,
        max_reward_regression=0.05,
        max_per_encounter_reward_regression=0.15,
        max_per_encounter_win_rate_regression=0.001,
        max_invalid_output_rate=0.02,
        max_mechanism_score_regression=0.03,
        max_missed_visible_lethal_increase=0,
        max_reason_math_contradiction_increase=0,
        max_reason_lethal_claim_error_increase=0,
        max_action_score_lethal_math_contradiction_increase=0,
        max_strict_json_failure_rate=0.05,
        allow_missing_eval_keys=False,
    )

    assert passed is False
    assert any("reason_math_contradiction increased 1 -> 2" in reason for reason in reasons)
    assert any("reason_claims_lethal_but_action_not_lethal increased 1 -> 3" in reason for reason in reasons)


def test_candidate_gate_rejects_action_score_math_regression() -> None:
    current = _base_metrics()
    candidate = _base_metrics()
    current["action_quality"]["action_score_lethal_math_contradiction"] = 0
    candidate["action_quality"]["action_score_lethal_math_contradiction"] = 1

    passed, reasons = _candidate_passes(
        current=current,
        candidate=candidate,
        min_win_rate_delta=0.0,
        max_reward_regression=0.05,
        max_per_encounter_reward_regression=0.15,
        max_per_encounter_win_rate_regression=0.001,
        max_invalid_output_rate=0.02,
        max_mechanism_score_regression=0.03,
        max_missed_visible_lethal_increase=0,
        max_reason_math_contradiction_increase=0,
        max_reason_lethal_claim_error_increase=0,
        max_action_score_lethal_math_contradiction_increase=0,
        max_strict_json_failure_rate=0.05,
        allow_missing_eval_keys=False,
    )

    assert passed is False
    assert any("action_score_lethal_math_contradiction increased 0 -> 1" in reason for reason in reasons)


def test_candidate_gate_rejects_strict_json_failure_rate() -> None:
    current = _base_metrics()
    candidate = _base_metrics()
    candidate["policy_stats"] = {"strict_json_ok": 90, "strict_json_failures": 10}

    passed, reasons = _candidate_passes(
        current=current,
        candidate=candidate,
        min_win_rate_delta=0.0,
        max_reward_regression=0.05,
        max_per_encounter_reward_regression=0.15,
        max_per_encounter_win_rate_regression=0.001,
        max_invalid_output_rate=0.02,
        max_mechanism_score_regression=0.03,
        max_missed_visible_lethal_increase=0,
        max_reason_math_contradiction_increase=0,
        max_reason_lethal_claim_error_increase=0,
        max_action_score_lethal_math_contradiction_increase=0,
        max_strict_json_failure_rate=0.05,
        allow_missing_eval_keys=False,
    )

    assert passed is False
    assert any("strict JSON failure rate 0.1000 > 0.0500" in reason for reason in reasons)
