from __future__ import annotations

import json

from llm.scripts.datasets.build_planner_hint_dataset import (
    _coerce_overall_score,
    _meta_str,
    _row_from_pair,
)


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


def test_coerce_overall_score_handles_int_str_slash_garbage():
    assert _coerce_overall_score(8) == 8.0
    assert _coerce_overall_score(8.5) == 8.5
    assert _coerce_overall_score("7") == 7.0
    assert _coerce_overall_score("8/10") == 8.0
    assert _coerce_overall_score("") is None
    assert _coerce_overall_score(None) is None
    assert _coerce_overall_score(True) is None  # bool 不算分数
    assert _coerce_overall_score("not a number") is None


def test_planner_dataset_filters_low_overall_score(tmp_path) -> None:
    """min_overall_score 过滤教师评分低的 review, 让 planner SFT 池只留高质量蒸馏样本."""
    review_path = tmp_path / "turn_order_review.json"
    episode_path = tmp_path / "episode_input.json"
    review_path.write_text(
        json.dumps({
            "planner_hint": {"battle_objective": "ok"},
            "overall_score": 4,  # 教师认为这个 episode 评判信心不高
        }),
        encoding="utf-8",
    )
    episode_path.write_text(
        json.dumps({"turns": [{"decisions": [{"pre_decision_state": "run: char=IRONCLAD"}]}]}),
        encoding="utf-8",
    )

    # 阈值 6: 这个 score=4 应该被过滤
    row, status = _row_from_pair(
        review_path, episode_path,
        system_prompt="planner system",
        min_overall_score=6.0,
    )
    assert row is None
    assert status == "below_min_overall_score"

    # 阈值 0: 不过滤
    row, status = _row_from_pair(
        review_path, episode_path,
        system_prompt="planner system",
        min_overall_score=0.0,
    )
    assert row is not None
    assert status == "ok"


def test_planner_dataset_splices_phase_plan_into_assistant_label(tmp_path) -> None:
    """include_phase_plan=True 时, review 的 phase_plan_zh 拼进 planner_hint 的 assistant
    payload, 让 planner LoRA 学 turn-by-turn 战术节奏 (而不只是 episode-level 目标)."""
    review_path = tmp_path / "turn_order_review.json"
    episode_path = tmp_path / "episode_input.json"
    review_path.write_text(
        json.dumps({
            "planner_hint": {"battle_objective": "kill priest first"},
            "phase_plan_zh": "turn 1-2: 立 Vulnerable; turn 3-4: 集火 priest; turn 5+: 补 block",
        }),
        encoding="utf-8",
    )
    episode_path.write_text(
        json.dumps({"turns": [{"decisions": [{"pre_decision_state": "run: char=IRONCLAD"}]}]}),
        encoding="utf-8",
    )

    row, status = _row_from_pair(
        review_path, episode_path,
        system_prompt="planner system",
        include_phase_plan=True,
    )
    assert status == "ok"
    assert row is not None
    payload = json.loads(row["messages"][2]["content"])
    assert payload["battle_objective"] == "kill priest first"
    assert payload["phase_plan"].startswith("turn 1-2")

    # include_phase_plan=False 时不拼接
    row2, _ = _row_from_pair(
        review_path, episode_path,
        system_prompt="planner system",
        include_phase_plan=False,
    )
    payload2 = json.loads(row2["messages"][2]["content"])
    assert "phase_plan" not in payload2
