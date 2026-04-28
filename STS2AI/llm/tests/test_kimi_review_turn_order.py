from __future__ import annotations

import json

from llm.scripts.kimi_review_turn_order import (
    append_usage_record,
    build_episode_payload,
    build_messages,
    compact_episode_for_prompt,
    count_recorded_api_calls,
    parse_review_json,
    response_content,
    select_episode_rows,
)
from llm.scripts.run_kimi_combat_review_batch import _read_skip_episode_ids


def _row(episode_id: str, step: int, flags: list[str] | None = None) -> dict:
    return {
        "episode_id": episode_id,
        "episode_step": step,
        "encounter_id": "CULTISTS_NORMAL",
        "seed": "seed",
        "outcome": "victory",
        "user_message": (
            "run: char=IRONCLAD round=1\n"
            "player: hp=80/80 block=0 energy=1/3 powers=-\n"
            "enemies:\n"
            "  enemy1: CULTIST hp=3/30 block=0 intent=Attack(6) powers=-\n"
            "hand:\n"
            "  [0] STRIKE_IRONCLAD cost=1 type=attack | Deal 6 damage.\n"
            "legal_actions:\n"
            "  STRIKE_IRONCLAD hand[0]:\n"
            "    [0] target=enemy1 damage=6\n"
            "  [1] end_turn\n"
        ),
        "decoded": {"action_index": 1 if flags else 0, "reason": "test"},
        "chosen_action": {"action": "end_turn" if flags else "play_card", "card_id": "STRIKE_IRONCLAD", "target_id": 1},
        "quality_flags": flags or [],
    }


def test_select_episode_rows_picks_high_signal_episode() -> None:
    rows = [_row("clean", 0), _row("bad", 0, ["missed_visible_lethal"]), _row("bad", 1, ["dangerous_end_turn"])]

    selected = select_episode_rows(rows)

    assert [row["episode_id"] for row in selected] == ["bad", "bad"]


def test_select_episode_rows_accepts_spectate_trace_without_episode_id() -> None:
    rows = [_row("", 2), _row("", 0), _row("", 1)]

    selected = select_episode_rows(rows)

    assert [row["episode_step"] for row in selected] == [0, 1, 2]


def test_read_skip_episode_ids_accepts_json_and_text(tmp_path) -> None:
    json_path = tmp_path / "skip.json"
    txt_path = tmp_path / "skip.txt"
    json_path.write_text(json.dumps({"episode_ids": ["a", "b"]}), encoding="utf-8")
    txt_path.write_text("c\n\n", encoding="utf-8")

    assert _read_skip_episode_ids([str(json_path), str(txt_path)]) == {"a", "b", "c"}


def test_build_episode_payload_and_messages_include_turn_order() -> None:
    payload = build_episode_payload([_row("ep", 0, ["missed_visible_lethal"])], max_decision_state_chars=5000)
    messages = build_messages(payload)
    user = messages[1]["content"]

    assert payload["turns"][0]["actions_played_in_order"] == ["[1] end_turn"]
    assert "visible lethal" in user
    assert "usable_training_labels" in user


def test_milestone_focus_keeps_first_middle_last_and_damage_turns() -> None:
    rows = []
    for step, hp in enumerate([80, 79, 78, 60, 60, 60, 60, 60]):
        row = _row("ep", step)
        row["user_message"] = row["user_message"].replace("round=1", f"round={step + 1}")
        row["user_message"] = row["user_message"].replace("hp=80/80", f"hp={hp}/80")
        rows.append(row)

    payload = build_episode_payload(rows, focus_policy="milestone", damage_turns=1)

    assert payload["focus_policy"] == "milestone"
    assert 3 in payload["focus"]["rounds"]
    assert payload["whole_combat_turn_summary"][2]["observed_hp_loss"] == 18
    assert len(payload["turns"]) < len(payload["whole_combat_turn_summary"])


def test_compact_prompt_removes_repeated_strategy_context() -> None:
    row = _row("ep", 0, ["missed_visible_lethal"])
    row["user_message"] = row["user_message"].replace(
        "player:",
        "strategy_context:\n  memory: repeated\n  plan: repeated\nplayer:",
    )
    payload = build_episode_payload([row], max_decision_state_chars=5000)
    compact = compact_episode_for_prompt(payload)
    messages = build_messages(payload, prompt_style="compact", thinking="enabled")
    user = messages[1]["content"]

    assert "strategy_context:" not in compact["turns"][0]["decisions"][0]["state"]
    assert "<FINAL_JSON>" in user


def test_parse_kimi_response_content() -> None:
    raw = {"choices": [{"message": {"content": json.dumps({"overall_score": 8})}}]}
    parsed, status = parse_review_json(response_content(raw))

    assert status == "ok"
    assert parsed == {"overall_score": 8}


def test_parse_final_json_block() -> None:
    parsed, status = parse_review_json('notes\n<FINAL_JSON>{"overall_score": 7}</FINAL_JSON>')

    assert status == "ok"
    assert parsed == {"overall_score": 7}


def test_kimi_usage_counter_counts_non_dry_run_calls(tmp_path) -> None:
    usage = tmp_path / "usage.jsonl"
    append_usage_record(usage, {"provider": "kimi", "dry_run": False, "call_count": 1})
    append_usage_record(usage, {"provider": "kimi", "dry_run": True, "call_count": 1})
    append_usage_record(usage, {"provider": "other", "dry_run": False, "call_count": 1})

    assert count_recorded_api_calls(usage) == 1
