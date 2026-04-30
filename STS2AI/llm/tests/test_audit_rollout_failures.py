from __future__ import annotations

import json

from llm.scripts.analysis.audit_rollout_failures import main as audit_main


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_audit_rollout_failures_records_invalid_abnormal_and_damage_turns(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    episode_trace = dataset / "episode_trace.jsonl"
    step_trace = dataset / "step_trace.jsonl"
    stderr = tmp_path / "stderr.log"
    out_dir = tmp_path / "audit"

    episodes = [
        {
            "episode_id": "ep-invalid",
            "encounter_id": "CULTISTS_NORMAL",
            "encounter_label": "CULTISTS_NORMAL[test]",
            "encounter_tag": "skada_floor_2_normal",
            "outcome": "invalid_output:dangerous_end_turn",
            "invalid_output": True,
            "invalid_reason": "dangerous_end_turn",
            "steps": 1,
            "case_metadata": {"case_id": "case-1", "run_id": 123, "floor": 2},
            "quality_flags": {"dangerous_end_turn": 1},
            "quality_summary": {"hp_lost": 6, "defense_score": 0.2, "mechanism_score": 1.0},
        },
        {
            "episode_id": "ep-left",
            "encounter_id": "SLIMES_NORMAL",
            "encounter_label": "SLIMES_NORMAL[test]",
            "outcome": "left_combat",
            "invalid_output": False,
            "steps": 1,
            "quality_flags": {},
            "quality_summary": {},
        },
    ]
    steps = [
        {
            "episode_id": "ep-invalid",
            "episode_step": 0,
            "user_message": (
                "run: char=IRONCLAD floor=2 round=1\n"
                "player: hp=50/80 block=0 energy=1/3 powers=-\n"
                "legal_actions:\n"
                "  [0] end_turn\n"
            ),
            "decoded": {"action_index": 0, "reason": "end turn", "fallback_reason": "dangerous_end_turn"},
            "invalid_output": True,
            "quality_flags": ["dangerous_end_turn"],
        },
        {
            "episode_id": "ep-invalid",
            "episode_step": 1,
            "user_message": (
                "run: char=IRONCLAD floor=2 round=2\n"
                "player: hp=44/80 block=0 energy=3/3 powers=-\n"
                "legal_actions:\n"
                "  [0] end_turn\n"
            ),
            "decoded": {"action_index": 0, "reason": "end turn"},
            "invalid_output": False,
            "quality_flags": [],
        },
        {
            "episode_id": "ep-left",
            "episode_step": 0,
            "user_message": (
                "run: char=IRONCLAD floor=3 round=1\n"
                "player: hp=70/80 block=0 energy=3/3 powers=-\n"
                "legal_actions:\n"
                "  [0] end_turn\n"
            ),
            "decoded": {"action_index": 0, "reason": "end turn"},
            "invalid_output": False,
            "quality_flags": [],
        },
    ]
    _write_jsonl(episode_trace, episodes)
    _write_jsonl(step_trace, steps)
    stderr.write_text("ok\nException in thread Thread-2\nUnicodeDecodeError: bad byte\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_rollout_failures",
            "--dataset-dir",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--log",
            str(stderr),
        ],
    )

    assert audit_main() == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["episodes"] == 2
    assert summary["invalid_cases"] == 1
    assert summary["abnormal_cases"] == 1
    assert summary["damage_turn_cases"] == 1
    assert summary["log_error_events"] == 2
    assert summary["cause_counts"]["unsafe_end_turn"] == 1
    assert summary["cause_counts"]["left_combat"] == 1
    assert "dangerous_end_turn" in (out_dir / "invalid_cases.jsonl").read_text(encoding="utf-8")
    assert "left_combat" in (out_dir / "abnormal_cases.jsonl").read_text(encoding="utf-8")
    assert "observed_hp_loss" in (out_dir / "damage_turn_cases.jsonl").read_text(encoding="utf-8")
