from __future__ import annotations

import json

from llm.scripts.analysis.compare_policy_eval import main


def test_compare_policy_eval_writes_rejection(tmp_path, monkeypatch) -> None:
    current = {
        "win_rate": 1.0,
        "reward": {"avg": 7.0},
        "invalid_output_episode_rate": 0.0,
        "mechanism_score": {"avg": 0.9},
        "action_quality": {"reason_math_contradiction": 0},
        "by_encounter": {
            "A": {"encounter_label": "A", "win_rate": 1.0, "reward": {"avg": 7.0}},
        },
    }
    candidate = {
        "win_rate": 1.0,
        "reward": {"avg": 7.0},
        "invalid_output_episode_rate": 0.0,
        "mechanism_score": {"avg": 0.9},
        "action_quality": {"reason_math_contradiction": 1},
        "by_encounter": {
            "A": {"encounter_label": "A", "win_rate": 1.0, "reward": {"avg": 7.0}},
        },
    }
    current_path = tmp_path / "current.json"
    candidate_path = tmp_path / "candidate.json"
    out_path = tmp_path / "promotion.json"
    current_path.write_text(json.dumps(current), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_policy_eval",
            "--current-metrics",
            str(current_path),
            "--candidate-metrics",
            str(candidate_path),
            "--out",
            str(out_path),
        ],
    )

    assert main() == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["candidate"]["reason_math_contradiction"] == 1
