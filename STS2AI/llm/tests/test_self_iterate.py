from __future__ import annotations

from llm.scripts.self_iterate import _candidate_passes


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
