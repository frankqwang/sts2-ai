from __future__ import annotations

import json
from collections import Counter

from llm.training.grpo_rollout import (
    EpisodeRecord,
    StepRecord,
    append_episode_trace_files,
    episode_step_trace_rows,
    _filter_optional_potion_actions,
)


def _episode() -> EpisodeRecord:
    step = StepRecord(
        state={"state_type": "combat"},
        legal_actions=[
            {"action": "play_card", "card_id": "BASH", "card_index": 0, "target_id": "enemy1"},
            {"action": "play_card", "card_id": "STRIKE_IRONCLAD", "card_index": 1, "target_id": "enemy1"},
        ],
        chosen_index=0,
        reason="apply vulnerable first",
        raw_generation='{"action_index": 0, "reason": "apply vulnerable first"}',
        attempts=[
            {
                "attempt": 0,
                "gen_ms": 12.5,
                "strict_json_status": "ok",
                "decoded": {
                    "action_index": 0,
                    "reason": "apply vulnerable first",
                    "used_fallback": False,
                    "fallback_reason": "",
                },
            }
        ],
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "legal_actions:\n  [0] BASH\n  [1] STRIKE_IRONCLAD"},
            {"role": "assistant", "content": '{"action_index": 0, "reason": "apply vulnerable first"}'},
        ],
        quality_flags=[],
        quality_report={"mechanism_score": 1.0},
        settlement_events=[
            {
                "type": "damage_received",
                "actor_id": "IRONCLAD",
                "target_id": "CULTIST",
                "source_card_id": "BASH",
                "unblocked_damage": 8,
            },
            {
                "type": "power_received",
                "target_id": "CULTIST",
                "power_id": "VULNERABLE",
                "amount_value": 2.0,
            },
        ],
    )
    return EpisodeRecord(
        encounter_key="CHOMPERS_NORMAL::act1_midrun::seed",
        encounter_id="CHOMPERS_NORMAL",
        encounter_tag="act1_midrun",
        encounter_label="CHOMPERS_NORMAL[act1_midrun:seed]",
        seed="seed-1",
        steps=[step],
        outcome="victory",
        final_state={"state_type": "game_over"},
        reward={"total": 7.5},
        duration_s=1.0,
    )


def test_episode_step_trace_rows_are_replay_trace_compatible() -> None:
    row = episode_step_trace_rows(_episode())[0]

    assert row["step"] == 0
    assert row["episode_step"] == 0
    assert row["encounter_id"] == "CHOMPERS_NORMAL"
    assert row["decoded"]["action_index"] == 0
    assert row["decoded"]["reason"] == "apply vulnerable first"
    assert row["chosen_action"]["card_id"] == "BASH"
    assert row["legal_actions"][0]["card_id"] == "BASH"
    assert row["state"]["state_type"] == "combat"
    assert row["user_message"].startswith("legal_actions:")
    assert row["quality_report"]["mechanism_score"] == 1.0
    assert row["settlement_events"][0]["source_card_id"] == "BASH"


def test_append_episode_trace_files_writes_step_and_episode_jsonl(tmp_path) -> None:
    step_trace = tmp_path / "step_trace.jsonl"
    episode_trace = tmp_path / "episode_trace.jsonl"

    append_episode_trace_files(step_trace_path=step_trace, episode_trace_path=episode_trace, ep=_episode())

    step_rows = [json.loads(line) for line in step_trace.read_text(encoding="utf-8").splitlines()]
    episode_rows = [json.loads(line) for line in episode_trace.read_text(encoding="utf-8").splitlines()]
    assert len(step_rows) == 1
    assert len(episode_rows) == 1
    assert episode_rows[0]["action_sequence"][0]["chosen_action"]["card_id"] == "BASH"
    assert episode_rows[0]["action_sequence"][0]["settlement_events"][1]["power_id"] == "VULNERABLE"


def test_episode_step_trace_rows_preserve_invalid_decision() -> None:
    ep = _episode()
    ep.steps[0].chosen_index = -1
    ep.steps[0].reason = "lethal CULTIST(6>=9)"
    ep.steps[0].invalid_output = True
    ep.steps[0].fallback_reason = "reason_math_contradiction"
    ep.steps[0].trainable = False
    ep.outcome = "invalid_output:reason_math_contradiction"

    row = episode_step_trace_rows(ep)[0]

    assert row["invalid_output"] is True
    assert row["decoded"]["used_fallback"] is True
    assert row["decoded"]["fallback_reason"] == "reason_math_contradiction"
    assert row["chosen_action"] is None


def test_grpo_rollout_filters_optional_potions_by_default(monkeypatch) -> None:
    monkeypatch.delenv("STS2_LLM_ALLOW_POTIONS", raising=False)
    stats: Counter[str] = Counter()

    enabled = _filter_optional_potion_actions(
        [
            {"action": "play_card", "card_id": "DEFEND_IRONCLAD"},
            {"action": "use_potion", "potion_id": "FIRE_POTION"},
            {"action": "end_turn"},
        ],
        stats,
    )

    assert [action["action"] for action in enabled] == ["play_card", "end_turn"]
    assert stats["potion_actions_suppressed"] == 1


def test_grpo_rollout_can_allow_potions(monkeypatch) -> None:
    monkeypatch.setenv("STS2_LLM_ALLOW_POTIONS", "1")

    enabled = _filter_optional_potion_actions(
        [
            {"action": "play_card", "card_id": "DEFEND_IRONCLAD"},
            {"action": "use_potion", "potion_id": "FIRE_POTION"},
        ],
    )

    assert [action["action"] for action in enabled] == ["play_card", "use_potion"]
