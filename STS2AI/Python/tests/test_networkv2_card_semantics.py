from __future__ import annotations

import sqlite3
import sys
import os
from pathlib import Path

import pytest
import torch

# 确保 STS2AI/Python 在 sys.path 中
_python_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _python_root not in sys.path:
    sys.path.insert(0, _python_root)

from constants import GAME_SEMANTIC_INDEX_DB
from data.build_card_semantic_index import build_card_semantic_index
from networkV2.s1_schema.card_semantic_catalog import CARD_SEMANTICS
from networkV2.s1_schema.entities import PlayerRuntime
from networkV2.s4_compiler.bank_assembler import BankAssembler
from networkV2.s4_compiler.runtime_compiler import RuntimeCompiler
from networkV2.s3_state_tracker.combat_env_wrapper import CombatStateTracker
from networkV2.s5_net.combat_net import CombatNetOutput
from networkV2.s5_net.heads.leaf_evaluator import LeafOutputs
from networkV2.s5_net.heads.value_heads import ValueOutputs
from networkV2.s6_training.losses import CombatLoss, LossConfig
from core.card_tags import FUNCTIONAL_TAG_TO_IDX


def _ensure_default_index() -> None:
    if not GAME_SEMANTIC_INDEX_DB.exists():
        build_card_semantic_index()


def _query_card_ids(where_sql: str, params: tuple = (), limit: int = 1) -> list[str]:
    _ensure_default_index()
    con = sqlite3.connect(str(GAME_SEMANTIC_INDEX_DB))
    try:
        rows = con.execute(
            f"SELECT id FROM cards WHERE {where_sql} LIMIT {int(limit)}",
            params,
        ).fetchall()
    finally:
        con.close()
    return [str(r[0]) for r in rows]


def _make_output(
    *,
    fight_win: float,
    run_value: float,
    turn_damage: float = 0.0,
    turn_block: float = 0.0,
) -> CombatNetOutput:
    return CombatNetOutput(
        logits=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        action_mask=torch.tensor([[True, True]]),
        values=ValueOutputs(
            fight_win=torch.tensor([fight_win], dtype=torch.float32),
            run_value=torch.tensor([run_value], dtype=torch.float32),
            expected_hp_loss=torch.tensor([0.0], dtype=torch.float32),
            survival_2turn=torch.tensor([1.0], dtype=torch.float32),
            tempo=torch.tensor([0.0], dtype=torch.float32),
            turn_damage_lookahead=torch.tensor([turn_damage], dtype=torch.float32),
            turn_block_lookahead=torch.tensor([turn_block], dtype=torch.float32),
        ),
        leaf=LeafOutputs(
            leaf_score=torch.tensor([0.0], dtype=torch.float32),
            transition_risk=torch.tensor([0.0], dtype=torch.float32),
            survival_margin=torch.tensor([1.0], dtype=torch.float32),
            resource_retention=torch.tensor([0.5], dtype=torch.float32),
        ),
    )


def test_build_card_semantic_index_writes_jsonl_and_sqlite(tmp_path: Path):
    db_path = tmp_path / "index.sqlite"
    jsonl_path = tmp_path / "cards.jsonl"
    stats = build_card_semantic_index(db_path=db_path, cards_jsonl_path=jsonl_path)

    assert stats["cards"] > 0
    assert db_path.exists()
    assert jsonl_path.exists()

    con = sqlite3.connect(str(db_path))
    try:
        source = con.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
        cards_count = con.execute("SELECT COUNT(*) FROM cards").fetchone()
    finally:
        con.close()

    assert source is not None and source[0] == "src_card_model_scan"
    assert cards_count is not None and int(cards_count[0]) == stats["cards"]


def test_runtime_compiler_pile_from_string_ids_backfills_counts():
    attack_id = _query_card_ids("card_type='attack'", limit=1)[0]
    skill_id = _query_card_ids("card_type='skill'", limit=1)[0]
    zero_cost_id = _query_card_ids("base_cost=0", limit=1)[0]

    pile = RuntimeCompiler()._pile_from_card_ids("draw", [attack_id, skill_id, zero_cost_id])

    assert pile.size == 3
    assert pile.attack_count >= 1
    assert pile.skill_count >= 1
    assert pile.zero_cost_count >= 1
    assert set(pile.card_ids) == {attack_id, skill_id, zero_cost_id}


def test_draw_horizon_probabilities_are_monotonic_and_post_shuffle_uses_discard():
    draw_tag_id = _query_card_ids("functional_tags_json LIKE ?", ('%"draw"%',), limit=2)
    filler_ids = _query_card_ids(
        "functional_tags_json NOT LIKE ? AND card_type='attack'",
        ('%"draw"%',),
        limit=2,
    )
    assert len(draw_tag_id) >= 2
    assert len(filler_ids) >= 2

    compiler = RuntimeCompiler()
    assembler = BankAssembler()
    draw = compiler._pile_from_card_ids("draw", [draw_tag_id[0], filler_ids[0], filler_ids[1]])
    discard = compiler._pile_from_card_ids("discard", [draw_tag_id[1]])
    ctx = assembler._build_draw_context({"draw": draw, "discard": discard})
    draw_idx = FUNCTIONAL_TAG_TO_IDX["draw"]

    next2 = float(ctx["next2"]["tag_probs"][draw_idx])
    next4 = float(ctx["next4"]["tag_probs"][draw_idx])
    post_shuffle = float(ctx["post_shuffle"]["tag_probs"][draw_idx])

    assert 0.0 < next2 <= next4 <= 1.0
    assert post_shuffle > 0.0

    empty_draw_ctx = assembler._build_draw_context({
        "draw": compiler._pile_from_card_ids("draw", []),
        "discard": compiler._pile_from_card_ids("discard", [draw_tag_id[0]]),
    })
    assert empty_draw_ctx["post_shuffle"]["tag_probs"][draw_idx] > 0.0


def test_build_profile_and_deck_tokens_share_card_semantics():
    draw_card_id = _query_card_ids("functional_tags_json LIKE ?", ('%"draw"%',), limit=1)[0]
    attack_id = _query_card_ids("card_type='attack'", limit=1)[0]
    obs = {
        "run": {"act": 1, "floor": 5},
        "player": {
            "hp": 60,
            "max_hp": 80,
            "gold": 50,
            "relics": [],
            "potions": [],
            "deck": [
                {"id": draw_card_id},
                {"id": attack_id},
            ],
        },
    }

    tracker = CombatStateTracker()
    tracker.on_run_start()
    tracker.refresh_build_profile(obs)

    deck_cards = RuntimeCompiler()._compile_deck(obs)
    shared = BankAssembler()._assemble_shared(
        tracker.run_build_memory,
        PlayerRuntime(hp=60, max_hp=80),
        "monster",
        deck_cards,
        [],
        [],
    )

    draw_tag_idx = FUNCTIONAL_TAG_TO_IDX["draw"]
    draw_token = next(
        tok for tok in shared.build_bank.tokens
        if tok.owner_id == draw_card_id
    )
    assert tracker.run_build_memory.draw > 0.0
    assert draw_token.numeric[11 + draw_tag_idx] == pytest.approx(1.0)


def test_combat_loss_run_value_head_uses_run_value_and_auxiliary_fight_win():
    cfg = LossConfig(
        policy_coef=0.0,
        entropy_coef=0.0,
        value_coef=1.0,
        hp_loss_coef=0.0,
        survival_coef=0.0,
        tempo_coef=0.0,
        leaf_coef=0.0,
        transition_risk_coef=0.0,
        survival_margin_coef=0.0,
        resource_retention_coef=0.0,
        turn_damage_coef=0.0,
        turn_block_coef=0.0,
    )
    loss_fn = CombatLoss(cfg, ppo_value_head="run_value")
    output = _make_output(fight_win=0.9, run_value=0.8)
    loss, metrics = loss_fn(
        output=output,
        action_indices=torch.tensor([0], dtype=torch.long),
        old_log_probs=torch.tensor([0.0], dtype=torch.float32),
        advantages=torch.tensor([0.0], dtype=torch.float32),
        returns=torch.tensor([0.8], dtype=torch.float32),
        fight_win_targets=torch.tensor([0.2], dtype=torch.float32),
        run_win_targets=torch.tensor([0.8], dtype=torch.float32),
        sample_weights=torch.tensor([1.0], dtype=torch.float32),
    )

    assert torch.isfinite(loss)
    assert metrics["vl_run_value"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["vl_fight_win"] > 0.0
