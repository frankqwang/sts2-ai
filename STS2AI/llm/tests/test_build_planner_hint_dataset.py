from __future__ import annotations

import json

from llm.scripts.datasets.build_planner_hint_dataset import _meta_str, _row_from_pair


def test_meta_str_handles_int_str_bool_list_dict_none():
    """meta 字段 cast：避免 LLM 同字段返回不同类型导致 pyarrow schema 推断失败。"""
    assert _meta_str(None) == ""
    assert _meta_str(5) == "5"
    assert _meta_str("3") == "3"
    assert _meta_str(True) == "true"
    assert _meta_str(False) == "false"
    assert _meta_str(3.14) == "3.14"
    assert _meta_str([1, "x"]) == '[1,"x"]'
    assert _meta_str({"a": 1}) == '{"a":1}'


def test_planner_hint_dataset_strips_action_prompt_blocks(tmp_path) -> None:
    review_path = tmp_path / "turn_order_review.json"
    episode_path = tmp_path / "episode_input.json"
    review_path.write_text(
        json.dumps(
            {
                "planner_hint": {
                    "battle_objective": "kill the scaling enemy first",
                    "kill_order": ["enemy1"],
                }
            }
        ),
        encoding="utf-8",
    )
    episode_path.write_text(
        json.dumps(
            {
                "episode_id": "ep1",
                "turns": [
                    {
                        "decisions": [
                            {
                                "pre_decision_state": (
                                    "run: char=IRONCLAD\n"
                                    "strategy_context:\n"
                                    "  plan: old fallback\n"
                                    "player: hp=40/80\n"
                                    "legal_actions:\n"
                                    "  [0] end_turn\n"
                                    "Return one JSON line: {\"action_index\":0}\n"
                                )
                            }
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row, status = _row_from_pair(review_path, episode_path, system_prompt="planner system")

    assert status == "ok"
    assert row is not None
    user = row["messages"][1]["content"]
    assert "strategy_context:" not in user
    assert "legal_actions:" not in user
    assert "Return one JSON line" not in user
    assert "retrieved_knowledge:" in user
    assert row["messages"][2]["content"] == '{"battle_objective":"kill the scaling enemy first","kill_order":["enemy1"]}'


def test_planner_hint_dataset_rejects_action_fields(tmp_path) -> None:
    review_path = tmp_path / "turn_order_review.json"
    episode_path = tmp_path / "episode_input.json"
    review_path.write_text(
        json.dumps({"planner_hint": {"battle_objective": "ok", "action_index": 0}}),
        encoding="utf-8",
    )
    episode_path.write_text(
        json.dumps({"turns": [{"decisions": [{"pre_decision_state": "run: char=IRONCLAD"}]}]}),
        encoding="utf-8",
    )

    row, status = _row_from_pair(review_path, episode_path, system_prompt="planner system")

    assert row is None
    assert status == "forbidden_action_fields"
