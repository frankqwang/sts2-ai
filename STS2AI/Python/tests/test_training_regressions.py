from __future__ import annotations

import math
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# 确保 STS2AI/Python 在 sys.path 中
_python_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _python_root not in sys.path:
    sys.path.insert(0, _python_root)


def test_load_samples_from_index_ignores_synergy_db_and_passes_balanced_filters(monkeypatch):
    from networkV2.s6_training import skada_offline_loader as loader
    import networkV2.s6_training.skada_index_dataset as index_dataset

    seen: dict[str, object] = {}

    class DummyFetcher:
        def __init__(self, **kwargs):
            seen["init_kwargs"] = kwargs

        def stats(self):
            return {"runs": 0}

        def sample_balanced(self, **kwargs):
            seen["balanced_kwargs"] = kwargs
            return []

        def fetch_records(self, rows):
            return iter(())

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(index_dataset, "SkadaIndexFetcher", DummyFetcher)
    monkeypatch.setattr(loader, "set_path_priors_fetcher", lambda fetcher: None)

    samples = loader.load_samples_from_index(
        Path("dummy.sqlite"),
        priors_db=Path("priors.sqlite"),
        synergy_db=Path("ignored.sqlite"),
        balanced=True,
        n_runs=40,
        characters=["IRONCLAD"],
        asc_bucket="low",
        version_prefix="v0.103.",
    )

    assert samples == []
    assert seen["init_kwargs"] == {
        "index_db": Path("dummy.sqlite"),
        "priors_db": Path("priors.sqlite"),
    }
    assert seen["balanced_kwargs"]["groups"] == [("IRONCLAD", "low")]
    assert seen["balanced_kwargs"]["n_per_group"] == 40
    assert seen["balanced_kwargs"]["version_prefix"] == "v0.103."
    assert seen["closed"] is True


def test_skada_fetcher_balanced_respects_version_prefix_and_filters(tmp_path):
    from networkV2.s6_training.skada_index_dataset import SkadaIndexFetcher

    db = tmp_path / "index.sqlite"
    con = sqlite3.connect(db)
    try:
        con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO metadata(key, value) VALUES ('repo_root', ?)", (str(tmp_path),))
        con.execute(
            """
            CREATE TABLE runs (
                run_id INTEGER PRIMARY KEY,
                character TEXT,
                ascension INTEGER,
                is_victory INTEGER,
                game_version TEXT,
                file_path TEXT,
                line_offset INTEGER,
                line_number INTEGER,
                floor_reached INTEGER,
                duration_sec INTEGER,
                has_map_acts INTEGER,
                has_final_deck INTEGER,
                has_combats INTEGER,
                n_card_choices INTEGER,
                n_relic_choices INTEGER,
                n_campfire INTEGER,
                n_shop INTEGER,
                asc_bucket TEXT,
                is_clean INTEGER
            )
            """
        )
        rows = [
            (1, "IRONCLAD", 1, 1, "v0.103.2", "a.jsonl", 0, 1, 10, 10, 1, 1, 1, 1, 1, 1, 1, "low", 1),
            (2, "IRONCLAD", 1, 1, "v0.102.9", "b.jsonl", 0, 1, 10, 10, 1, 1, 1, 1, 1, 1, 1, "low", 1),
            (3, "SILENT", 1, 1, "v0.103.2", "c.jsonl", 0, 1, 10, 10, 1, 1, 1, 1, 1, 1, 1, "low", 1),
            (4, "IRONCLAD", 1, 1, "v0.103.3", "d.jsonl", 0, 1, 10, 10, 1, 1, 1, 1, 1, 1, 1, "mid", 1),
        ]
        con.executemany(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        con.commit()
    finally:
        con.close()

    fetcher = SkadaIndexFetcher(index_db=db)
    try:
        sampled = fetcher.sample_balanced(
            n_per_group=10,
            characters=["IRONCLAD"],
            asc_bucket="low",
            version_prefix="v0.103.",
        )
    finally:
        fetcher.close()

    assert [row.run_id for row in sampled] == [1]


def test_combat_loss_masks_invalid_actions():
    from networkV2.s6_training.losses import CombatLoss, LossConfig

    output = SimpleNamespace(
        logits=torch.tensor([[0.0, 0.0, 1000.0]], dtype=torch.float32),
        action_mask=torch.tensor([[True, True, False]]),
        values=SimpleNamespace(
            fight_win=torch.tensor([0.5]),
            expected_hp_loss=torch.tensor([0.0]),
            survival_2turn=torch.tensor([0.5]),
            tempo=torch.tensor([0.0]),
            turn_damage_lookahead=torch.tensor([0.0]),
        ),
        leaf=SimpleNamespace(),
    )
    loss_fn = CombatLoss(LossConfig(
        normalize_adv=False,
        entropy_coef=0.0,
        value_coef=0.0,
        leaf_coef=0.0,
        transition_risk_coef=0.0,
        survival_margin_coef=0.0,
        resource_retention_coef=0.0,
    ))
    _, metrics = loss_fn(
        output,
        action_indices=torch.tensor([0]),
        old_log_probs=torch.tensor([math.log(0.5)], dtype=torch.float32),
        advantages=torch.tensor([1.0]),
        returns=torch.tensor([0.5]),
    )
    assert abs(metrics["approx_kl"]) < 1e-6


def test_noncombat_loss_masks_invalid_actions():
    from networkV2.s6_training.losses import NonCombatLoss, NonCombatLossConfig

    output = SimpleNamespace(
        logits=torch.tensor([[0.0, 0.0, 1000.0]], dtype=torch.float32),
        action_mask=torch.tensor([[True, True, False]]),
        run_eval=SimpleNamespace(run_win_prob=torch.tensor([0.5])),
    )
    loss_fn = NonCombatLoss(NonCombatLossConfig(
        normalize_adv=False,
        entropy_coef=0.0,
    ))
    _, metrics = loss_fn(
        output,
        action_indices=torch.tensor([1]),
        old_log_probs=torch.tensor([math.log(0.5)], dtype=torch.float32),
        advantages=torch.tensor([1.0]),
        returns=torch.tensor([0.5]),
    )
    assert abs(metrics["nc_approx_kl"]) < 1e-6


def test_runtime_extractor_static_fallbacks_for_hand_keywords_and_piles():
    from networkV2.s4_featurization.runtime_extractor import RuntimeExtractor

    extractor = RuntimeExtractor()
    obs = {
        "player": {
            "hp": 70,
            "max_hp": 80,
            "energy": 3,
            "max_energy": 3,
            "deck": [],
            "relics": [],
            "potions": [],
        },
            "battle": {
                "hand": [
                    {"id": "PANIC_BUTTON", "cost": 2, "can_play": True, "requires_target": False},
                    {"id": "PARSE", "cost": 1, "can_play": True, "requires_target": False},
                ],
            "draw_pile_cards": ["BASH", "PANIC_BUTTON"],
            "discard_pile_cards": ["PANIC_BUTTON"],
            "exhaust_pile_cards": [],
        },
    }

    _, hand, _, piles, _, _, _, _ = extractor.extract(obs)
    assert hand[0].exhaust is True
    assert hand[1].ethereal is True

    draw = next(p for p in piles if p.pile_type == "draw")
    discard = next(p for p in piles if p.pile_type == "discard")
    assert draw.attack_count >= 1
    assert draw.zero_cost_count >= 1
    assert discard.skill_count >= 1


def test_card_reward_option_builder_claim_reward_uses_reward_items():
    from networkV2.s4_featurization.noncombat.card_reward_options import CardRewardOptionBuilder

    builder = CardRewardOptionBuilder()
    obs = {
        "rewards": {
            "items": [
                {"index": 0, "type": "gold", "label": "25 Gold", "id": "gold_25", "claimable": True},
            ],
        },
    }
    candidates = builder.build(
        obs,
        [{"action": "claim_reward", "index": 0, "label": "25 Gold"}],
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.source_card_id == "gold_25"
    assert cand.event_kind == "gain_gold"
    assert cand.roles == ["resource"]
    assert cand.can_afford == 1.0


def test_decision_featurizer_routes_treasure_and_relic_select_to_selection():
    from networkV2.s4_featurization.decision_featurizer import DecisionFeaturizer

    featurizer = DecisionFeaturizer()
    base_player = {
        "hp": 70,
        "max_hp": 80,
        "gold": 100,
        "energy": 3,
        "max_energy": 3,
        "deck": [],
        "relics": [],
        "potions": [],
    }

    treasure_banks = featurizer.featurize(
        {
            "state_type": "treasure",
            "player": dict(base_player),
            "treasure": {"relics": [{"index": 0, "id": "burning_blood"}]},
        },
        [{"action": "claim_treasure_relic", "index": 0, "label": "burning_blood"}],
        room_type="treasure",
    )
    relic_banks = featurizer.featurize(
        {
            "state_type": "relic_select",
            "player": dict(base_player),
            "relic_select": {"relics": [{"index": 0, "id": "burning_blood"}], "can_skip": True},
        },
        [
            {"action": "select_relic", "index": 0, "label": "burning_blood"},
            {"action": "skip_relic_selection", "label": "Skip"},
        ],
        room_type="relic_select",
    )

    assert treasure_banks.decision_domain == "selection"
    assert relic_banks.decision_domain == "selection"
    assert len(treasure_banks.action_bank.tokens) == 1
    assert len(relic_banks.action_bank.tokens) == 2
