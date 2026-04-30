from __future__ import annotations

import json

from llm.scripts.analysis.review_step_trace import main as review_main


def test_review_step_trace_writes_reviews_and_lessons(tmp_path, monkeypatch) -> None:
    trace = tmp_path / "step_trace.jsonl"
    row = {
        "episode_id": "ep1",
        "episode_step": 0,
        "step": 0,
        "encounter_id": "CULTISTS_NORMAL",
        "encounter_label": "CULTISTS_NORMAL[test]",
        "seed": "seed",
        "outcome": "victory",
        "episode_reward": {"total": 7.5},
        "user_message": (
            "run: char=IRONCLAD round=1\n"
            "player: hp=50/80 block=0 energy=1/3 powers=-\n"
            "enemies:\n"
            "  enemy1: CULTIST hp=3/30 block=0 intent=Attack(6) powers=-\n"
            "legal_actions:\n"
            "  STRIKE_IRONCLAD hand[0]:\n"
            "    [0] target=enemy1 damage=6\n"
            "  [1] end_turn\n"
        ),
        "decoded": {"action_index": 1, "reason": "end turn"},
        "chosen_action": {"action": "end_turn"},
        "quality_flags": ["missed_visible_lethal", "dangerous_end_turn"],
        "quality_report": {
            "opportunities": {"visible_lethal": 1, "dangerous_turn": 1},
            "misses": {"visible_lethal": 1, "dangerous_turn": 1},
        },
        "raw_generation": '{"action_index": 1, "reason": "end turn"}',
    }
    next_turn = {
        **row,
        "episode_step": 1,
        "step": 1,
        "user_message": (
            "run: char=IRONCLAD floor=2 round=2\n"
            "player: hp=44/80 block=0 energy=3/3 powers=-\n"
            "enemies:\n"
            "  enemy1: CULTIST hp=3/30 block=0 intent=Attack(6) powers=-\n"
            "legal_actions:\n"
            "  [0] end_turn\n"
        ),
        "decoded": {"action_index": 0, "reason": "end turn"},
        "quality_flags": [],
        "quality_report": {"opportunities": {}, "misses": {}, "metrics": {}},
    }
    trace.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(next_turn, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "review"
    exp = tmp_path / "lessons_global.jsonl"

    monkeypatch.setattr(
        "sys.argv",
        [
            "review_step_trace",
            "--trace",
            str(trace),
            "--out-dir",
            str(out_dir),
            "--experience-path",
            str(exp),
            "--append-experience",
        ],
    )

    assert review_main() == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] == 2
    assert summary["turn_reviews"] == 2
    assert summary["combat_reviews"] == 1
    assert summary["hard_cases"] == 1
    assert summary["damage_turn_hard_cases"] == 1
    assert summary["lesson_candidates"] >= 2
    assert summary["experience_appended"] >= 2
    assert (out_dir / "turn_reviews.jsonl").exists()
    assert "observed_hp_loss" in (out_dir / "damage_turn_hard_cases.jsonl").read_text(encoding="utf-8")
    assert "visible lethal" in (out_dir / "lessons.jsonl").read_text(encoding="utf-8")
    assert exp.exists()
