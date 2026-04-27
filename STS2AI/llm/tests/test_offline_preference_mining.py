from __future__ import annotations

import json

from llm.scripts.mine_offline_preferences import (
    _visible_lethal_repair,
    main,
)


def test_visible_lethal_repair_handles_grouped_legal_actions() -> None:
    user = """run: char=IRONCLAD
strategy_context:
  plan: take lethal if available: hand[3] BLUDGEON kills enemy2 with damage=42
legal_actions:
  BLUDGEON hand[3]:
    [4] target=enemy1 damage=42
    [5] target=enemy2 damage=42
  [6] end_turn
"""

    assert _visible_lethal_repair(user, 6) == (5, "take visible lethal on enemy2")


def test_mine_offline_preferences_writes_hard_case_repair_and_pair(tmp_path, monkeypatch, capsys) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    user = """run: char=IRONCLAD
strategy_context:
  plan: take lethal if available: hand[0] STRIKE_IRONCLAD kills enemy1 with damage=6
legal_actions:
  [0] STRIKE_IRONCLAD hand[0] target=enemy1 damage=6
  [1] end_turn
"""
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": '{"action_index": 1, "reason": "bad"}'},
        ],
        "meta": {
            "advantage": -1.0,
            "episode_reward": 1.0,
            "action_quality_flags": ["missed_visible_lethal"],
            "encounter_key": "E",
        },
    }
    (dataset / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (dataset / "eval.jsonl").write_text("", encoding="utf-8")
    out = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "mine_offline_preferences",
            "--dataset-dir",
            str(dataset),
            "--out-dir",
            str(out),
        ],
    )
    assert main() == 0
    _ = capsys.readouterr()

    hard = (out / "hard_cases.jsonl").read_text(encoding="utf-8")
    repairs = [json.loads(line) for line in (out / "repair_sft.jsonl").read_text(encoding="utf-8").splitlines()]
    pairs = [json.loads(line) for line in (out / "preference_pairs.jsonl").read_text(encoding="utf-8").splitlines()]

    assert "missed_visible_lethal" in hard
    assert repairs[0]["messages"][-1]["content"] == '{"action_index": 0, "reason": "take visible lethal on enemy1"}'
    assert pairs[0]["meta"]["pair_source"] == "rule_visible_lethal_repair"
