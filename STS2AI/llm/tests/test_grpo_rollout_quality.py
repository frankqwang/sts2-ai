from __future__ import annotations

import json

from llm.data_pipeline.encounter_pool import encounter_key, skada_case_to_spec
from llm.data_pipeline.state_renderer import render_state_text
from llm.data_pipeline.strategy_context import StrategyMemory
from llm.training.grpo_rollout import (
    _classify_tier,
    _inject_spec_context,
    build_eval_metrics,
    compute_enemy_damage_progress,
    compute_episode_reward,
)


def _fake_episode(encounter_id: str, outcome: str, reward_total: float = 8.0, hp_lost: float = 0.0):
    """构造一个最小 EpisodeRecord-like 对象供 build_eval_metrics 测试用。"""
    from types import SimpleNamespace
    return SimpleNamespace(
        encounter_key=f"{encounter_id}::tag::abc",
        encounter_id=encounter_id,
        encounter_tag="tag",
        encounter_label=f"{encounter_id}[tag]",
        seed="seed-1",
        outcome=outcome,
        steps=[],
        duration_s=10.0,
        reward={"total": reward_total},
        invalid_output=outcome.startswith("invalid"),
        invalid_reason="",
        quality_flags={},
        quality_summary={
            "hp_lost": hp_lost,
            "enemy_damage_progress": 1.0 if outcome == "victory" else 0.5,
            "mechanism_score": 1.0,
            "sequence_score": 1.0,
            "defense_score": 1.0,
        },
        case_metadata={},
    )


def test_mask_reason_in_assistant_clears_reason_field():
    from llm.training.grpo_rollout import _maybe_mask_reason_in_assistant
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": '{"action_index":3,"confidence":0.75,"reason":"Deal 6 damage"}'},
    ]
    out = _maybe_mask_reason_in_assistant(msgs, mask=True)
    parsed = json.loads(out[2]["content"])
    assert parsed["reason"] == ""
    assert parsed["action_index"] == 3
    assert parsed["confidence"] == 0.75


def test_mask_reason_in_assistant_off_keeps_reason():
    from llm.training.grpo_rollout import _maybe_mask_reason_in_assistant
    msgs = [{"role": "assistant", "content": '{"action_index":3,"reason":"Deal 6 damage"}'}]
    out = _maybe_mask_reason_in_assistant(msgs, mask=False)
    parsed = json.loads(out[0]["content"])
    assert parsed["reason"] == "Deal 6 damage"


def test_fullrun_aggregate_metrics_basic_shape():
    """fullrun_eval.aggregate_metrics 输出 schema 校验。"""
    from llm.eval.fullrun_eval import FullrunEpisode, aggregate_metrics
    eps = [
        FullrunEpisode(seed="s1", final_act=2, final_floor=18, max_floor=17, outcome="act1_cleared", duration_s=120.0, steps=180, combat_steps=80),
        FullrunEpisode(seed="s2", final_act=1, final_floor=8, max_floor=8, outcome="defeat", duration_s=60.0, steps=90, combat_steps=40, death_floor=8, death_state_type="elite"),
        FullrunEpisode(seed="s3", final_act=1, final_floor=12, max_floor=12, outcome="defeat", duration_s=80.0, steps=120, combat_steps=55, death_floor=12),
    ]
    m = aggregate_metrics(eps)
    assert m["episodes"] == 3
    assert m["act1_cleared"] == 1
    assert abs(m["act1_clear_rate"] - 1/3) < 1e-3  # round 4 位精度
    assert m["floor_reached"]["max"] == 17
    assert m["floor_reached"]["min"] == 8
    assert m["death_floor_counts"] == {8: 1, 12: 1}
    assert m["outcome_counts"]["defeat"] == 2
    assert m["outcome_counts"]["act1_cleared"] == 1


def test_classify_tier_by_encounter_id_keywords():
    assert _classify_tier(_fake_episode("THE_KIN_BOSS", "defeat")) == "boss"
    assert _classify_tier(_fake_episode("PHANTASMAL_GARDENERS_ELITE", "victory")) == "elite"
    assert _classify_tier(_fake_episode("SLIMES_NORMAL", "victory")) == "normal"
    assert _classify_tier(_fake_episode("SEAPUNK_WEAK", "victory")) == "normal"


def test_build_eval_metrics_partitions_by_tier_and_encounter():
    eps = [
        _fake_episode("SEAPUNK_WEAK", "victory", 8.0, 0.0),
        _fake_episode("SEAPUNK_WEAK", "victory", 7.5, 5.0),
        _fake_episode("PHANTASMAL_GARDENERS_ELITE", "defeat", -10.0, 50.0),
        _fake_episode("THE_KIN_BOSS", "defeat", -15.0, 60.0),
        _fake_episode("CEREMONIAL_BEAST_BOSS", "victory", 5.0, 30.0),
    ]
    grouped = {ep.encounter_key: [ep] for ep in eps}
    # 把同 encounter 的两个 SEAPUNK 放一起
    grouped["SEAPUNK_WEAK::tag::abc"] = [eps[0], eps[1]]

    metrics = build_eval_metrics(eps, grouped)
    # 总体
    assert metrics["episodes"] == 5
    assert metrics["victories"] == 3
    assert metrics["win_rate"] == 0.6
    # 分层
    assert "by_tier" in metrics
    assert metrics["by_tier"]["normal"]["episodes"] == 2
    assert metrics["by_tier"]["normal"]["victories"] == 2
    assert metrics["by_tier"]["elite"]["episodes"] == 1
    assert metrics["by_tier"]["elite"]["victories"] == 0
    assert metrics["by_tier"]["boss"]["episodes"] == 2
    assert metrics["by_tier"]["boss"]["victories"] == 1
    assert metrics["by_tier"]["boss"]["win_rate"] == 0.5
    # by_encounter 至少含 SEAPUNK
    assert "SEAPUNK_WEAK::tag::abc" in metrics["by_encounter"]
    assert metrics["by_encounter"]["SEAPUNK_WEAK::tag::abc"]["episodes"] == 2


def test_reward_does_not_give_direct_victory_bonus() -> None:
    victory = compute_episode_reward(
        "victory",
        player_hp_start=70,
        player_hp_end=70,
        player_max_hp=80,
        num_turns=3,
        damage_dealt=80,
        damage_taken=0,
        enemy_damage_progress=1.0,
    )
    unfinished = compute_episode_reward(
        "max_steps",
        player_hp_start=70,
        player_hp_end=70,
        player_max_hp=80,
        num_turns=3,
        damage_dealt=80,
        damage_taken=0,
        enemy_damage_progress=1.0,
    )

    assert victory["total"] == unfinished["total"] == 8.0


def test_enemy_damage_progress_treats_missing_final_enemy_as_dead() -> None:
    initial = {
        "enemies": [
            {"combat_id": 1, "hp": 30},
            {"combat_id": 2, "hp": 20},
        ]
    }
    final = {"enemies": [{"combat_id": 2, "hp": 5}]}

    progress = compute_enemy_damage_progress(initial, final)

    assert progress["enemy_hp_start"] == 50
    assert progress["enemy_hp_end"] == 5
    assert progress["enemy_damage_dealt"] == 45
    assert progress["enemy_damage_progress"] == 0.9


def test_skada_case_prompt_has_stable_run_meta_and_piles() -> None:
    spec = skada_case_to_spec({
        "case_id": "run_1_floor_7_slimes",
        "character_id": "IRONCLAD",
        "floor": 7,
        "encounter_id": "SLIMES_NORMAL",
        "encounter_type": "Normal",
        "won": True,
        "build": {"deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}]},
    })
    state = {
        "run": {"act": 0, "floor": 0},
        "player": {
            "character": "IRONCLAD",
            "hp": 80,
            "max_hp": 80,
            "gold": 99,
            "deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}],
        },
        "battle": {
            "round_number_raw": 1,
            "energy": 3,
            "max_energy": 3,
            "hand": [{"index": 0, "id": "STRIKE_IRONCLAD", "cost": 1, "type": "attack"}],
            "enemies": [{"combat_id": 1, "id": "CULTIST", "hp": 48, "max_hp": 48}],
            "draw_pile_cards": ["BASH"],
            "discard_pile_cards": ["DEFEND_IRONCLAD"],
            "exhaust_pile_cards": [],
        },
    }
    legal = [{"action": "play_card", "card_index": 0, "card_id": "STRIKE_IRONCLAD", "target_id": 1}]

    state = _inject_spec_context(state, spec, seed="unit-seed")
    strategy_context = StrategyMemory().context_text(state, legal)
    prompt = render_state_text(state, legal, strategy_context=strategy_context)

    assert "run: char=IRONCLAD act=1 floor=7 encounter=SLIMES_NORMAL round=1 gold=99" in prompt
    assert encounter_key(spec) + "::unit-seed" not in prompt
    assert "combat_key:" not in prompt
    assert "encounter=?" not in prompt
    assert "act=0 floor=0" not in prompt
    assert "draw_cards: BASH" in prompt
    assert "discard_cards: DEFEND_IRONCLAD" in prompt
