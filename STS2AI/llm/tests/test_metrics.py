from __future__ import annotations

import json

from llm.metrics import summarize_dataset_dir, summarize_trace


def test_summarize_dataset_dir_counts_action_quality_flags(tmp_path) -> None:
    row = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "run: char=IRONCLAD\n"
                    "legal_actions:\n"
                    "  [0] STRIKE_IRONCLAD hand[0] target=enemy1 damage=6\n"
                    "  [1] end_turn\n"
                ),
            },
            {"role": "assistant", "content": '{"action_index": 1, "reason": "bad"}'},
        ],
        "meta": {
            "action_quality_flags": ["missed_visible_lethal", "dangerous_end_turn"],
            "action_quality_report": {"mechanism_score": 0.25},
        },
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "meta.json").write_text(
        json.dumps({
            "episodes": [
                {
                    "outcome": "victory",
                    "steps": 1,
                    "duration_s": 1.0,
                    "kept_samples": 1,
                    "discarded_samples": 0,
                    "quality_flags": {"missed_visible_lethal": 1},
                    "quality_summary": {
                        "mechanism_score": 0.25,
                        "sequence_score": 0.5,
                        "defense_score": 0.75,
                        "hp_lost": 3,
                        "turns": 2,
                    },
                }
            ],
            "action_quality": {"missed_visible_lethal": 1},
        }),
        encoding="utf-8",
    )

    summary = summarize_dataset_dir(tmp_path)

    assert summary["train"]["action_quality_counts"]["missed_visible_lethal"] == 1
    assert summary["train"]["action_quality_counts"]["dangerous_end_turn"] == 1
    assert summary["train"]["mechanism_score"]["avg"] == 0.25
    assert summary["rollout"]["action_quality"]["missed_visible_lethal"] == 1
    assert summary["rollout"]["mechanism_score"]["avg"] == 0.25
    assert summary["rollout"]["sequence_score"]["avg"] == 0.5
    assert summary["rollout"]["defense_score"]["avg"] == 0.75
    assert summary["rollout"]["hp_lost"]["avg"] == 3


def test_summarize_trace_counts_quality_flags(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({
            "route": "llm",
            "action_mode": "index",
            "enabled_count": 3,
            "attempts": [],
            "decoded": {},
            "chosen_action": {"action": "proceed"},
            "quality_flags": ["proceed_with_unclaimed_rewards"],
            "quality_report": {
                "opportunities": {"unclaimed_reward": 1},
                "misses": {"unclaimed_reward": 1},
            },
        }) + "\n",
        encoding="utf-8",
    )

    summary = summarize_trace(trace)

    assert summary["quality_flags"]["proceed_with_unclaimed_rewards"] == 1
    assert summary["quality_opportunities"]["unclaimed_reward"] == 1
    assert summary["quality_misses"]["unclaimed_reward"] == 1
