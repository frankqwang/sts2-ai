from __future__ import annotations

from llm.scripts.analysis.eval_planner_hint_outputs import _check_trace_row


def test_trace_cache_hit_requires_retrieved_knowledge_when_enabled() -> None:
    row = {
        "planner_hint_status": "cache_hit",
        "planner_hint": "battle_objective: Keep pressure on CULTIST\nkill_order: enemy1",
        "retrieved_knowledge": [],
    }

    assert _check_trace_row(row, require_knowledge=True) == "missing_retrieved_knowledge"


def test_trace_cache_hit_with_retrieved_knowledge_is_valid() -> None:
    row = {
        "planner_hint_status": "cache_hit",
        "planner_hint": "battle_objective: Keep pressure on CULTIST\nkill_order: enemy1",
        "retrieved_knowledge": [{"entry_id": "enemy_cultist_scaling"}],
    }

    assert _check_trace_row(row, require_knowledge=True) == "ok"
