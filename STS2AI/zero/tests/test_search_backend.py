from __future__ import annotations

import unittest
from unittest.mock import patch

from zero.config import SearchConfig
from zero.domain import (
    BattleState,
    EnemyState,
    FightLabel,
    LegalAction,
    PileSummary,
    PlayerState,
    StaticContext,
    SearchRequest,
    TrainingSample,
    TransitionDelta,
)
from zero.replay.search_backend import CombatSearchBackend
from zero.replay.skada import SkadaBuild, SkadaCombatCase


def _make_state(*, enemy_hp: float, terminal: bool = False, outcome: str = "") -> BattleState:
    return BattleState(
        player=PlayerState(hp=70.0, max_hp=80.0, block=0.0, energy=3.0),
        enemies=[EnemyState(enemy_id="SLIME", hp=enemy_hp, max_hp=20.0, block=0.0, intent_id="attack")],
        hand=[],
        piles=PileSummary(),
        context=StaticContext(
            character_id="IRONCLAD",
            act=1,
            floor=3,
            encounter_class="normal",
            encounter_id="SLIMES_WEAK",
            metadata={"seed": "seed", "skada_case_id": "case-1"},
        ),
        legal_actions=[
            LegalAction(action_id="play_defend", action_type="play_card", card_id="DEFEND_IRONCLAD", block_now=5.0),
            LegalAction(action_id="play_strike", action_type="play_card", card_id="STRIKE_IRONCLAD", damage_now=6.0),
        ],
        terminal=terminal,
        run_outcome=outcome,
    )


class _FakeReplayRuntime:
    def __init__(self, case, **kwargs) -> None:
        self._root = _make_state(enemy_hp=20.0)
        self._state = self._root

    def reset(self, *, seed: str | None = None):
        self._state = _make_state(enemy_hp=20.0)
        return self._state

    def step(self, action_index: int):
        if action_index == 1:
            self._state = _make_state(enemy_hp=0.0, terminal=True, outcome="victory")
        else:
            self._state = _make_state(enemy_hp=20.0, terminal=True, outcome="defeat")
        return self._state

    def save_state(self) -> str:
        return "s0"

    def load_state(self, state_id: str):
        self._state = _make_state(enemy_hp=20.0)
        return self._state

    def delete_state(self, state_id: str) -> None:
        return None

    def close(self) -> None:
        return None


class SearchBackendTests(unittest.TestCase):
    def test_root_sweep_prefers_higher_quality_action(self) -> None:
        case = SkadaCombatCase(
            source_path="test",
            source_line=1,
            run_id=1,
            seed="seed",
            game_version="v0",
            character_id="IRONCLAD",
            ascension=0,
            player_count=1,
            floor=3,
            encounter_id="SLIMES_WEAK",
            encounter_type="normal",
            won=True,
            build=SkadaBuild(deck=[], relics=[], current_hp=70, max_hp=80),
            metadata={},
            floor_state={},
            card_usage={},
        )
        sample = TrainingSample(
            sample_id="sample",
            run_id="run",
            fight_id="fight",
            step_idx=0,
            state=_make_state(enemy_hp=20.0),
            history=[],
            legal_actions=_make_state(enemy_hp=20.0).legal_actions,
            behavior_action_index=0,
            behavior_action_id="play_defend",
            delta=TransitionDelta(),
            fight_label=FightLabel(fight_win=0.0, enemy_hp_fraction_dealt=0.0, self_hp_fraction_remaining=0.0),
            metadata={"prefix_action_indices": []},
        )
        search_backend = CombatSearchBackend(
            case,
            config=SearchConfig(max_root_actions=2, rollouts_per_action=1, max_branch_steps=4),
            auto_launch=False,
        )
        request = SearchRequest(request_id="req", sample=sample, priority=1.0)
        with patch("zero.replay.search_backend.SkadaReplayRuntime", _FakeReplayRuntime):
            label = search_backend.label_request(request)
        self.assertEqual(label.best_action_index, 1)
        self.assertEqual(len(label.policy), 2)
        self.assertTrue(label.search_trace)
        self.assertEqual(label.metadata["search_backend"], "CombatSearchBackend")

    def test_invalid_prefix_falls_back_to_policy_prior_label(self) -> None:
        case = SkadaCombatCase(
            source_path="test",
            source_line=1,
            run_id=1,
            seed="seed",
            game_version="v0",
            character_id="IRONCLAD",
            ascension=0,
            player_count=1,
            floor=3,
            encounter_id="SLIMES_WEAK",
            encounter_type="normal",
            won=True,
            build=SkadaBuild(deck=[], relics=[], current_hp=70, max_hp=80),
            metadata={},
            floor_state={},
            card_usage={"STRIKE_IRONCLAD": {"plays": 3}},
        )
        sample = TrainingSample(
            sample_id="sample",
            run_id="run",
            fight_id="fight",
            step_idx=0,
            state=_make_state(enemy_hp=20.0),
            history=[],
            legal_actions=_make_state(enemy_hp=20.0).legal_actions,
            behavior_action_index=0,
            behavior_action_id="play_defend",
            delta=TransitionDelta(),
            fight_label=FightLabel(fight_win=0.0, enemy_hp_fraction_dealt=0.0, self_hp_fraction_remaining=0.0),
            metadata={"prefix_action_indices": [99]},
        )
        search_backend = CombatSearchBackend(
            case,
            config=SearchConfig(max_root_actions=2, rollouts_per_action=1, max_branch_steps=4),
            auto_launch=False,
        )
        request = SearchRequest(request_id="req", sample=sample, priority=1.0)
        with patch("zero.replay.search_backend.SkadaReplayRuntime", _FakeReplayRuntime):
            label = search_backend.label_request(request)
        self.assertEqual(label.metadata["search_backend"], "CombatSearchBackendFallback")
        self.assertEqual(len(label.policy), 2)


if __name__ == "__main__":
    unittest.main()
