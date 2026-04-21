from __future__ import annotations

import unittest
from zero.domain import (
    BattleState,
    EnemyState,
    FightLabel,
    HandCardState,
    LegalAction,
    PileSummary,
    PlayerState,
    StaticContext,
    SearchLabel,
    SearchRequest,
    TrainingSample,
    TransitionDelta,
)
from zero.orchestration.search import SearchQueueProcessor


class _BadSearchBackend:
    def __init__(self, label: SearchLabel):
        self._label = label

    def label_request(self, request, runtime_factory=None, seed=None):
        return self._label


def _make_sample() -> TrainingSample:
    state = BattleState(
        player=PlayerState(hp=20.0, max_hp=80.0, block=3.0, energy=3.0),
        enemies=[EnemyState(enemy_id="slime", hp=10.0, max_hp=20.0, block=0.0, intent_id="attack", alive=True)],
        hand=[HandCardState(card_id="strike", cost_now=1.0, damage_now=6.0)],
        piles=PileSummary(draw_pile_size=4, discard_pile_size=1),
        context=StaticContext(character_id="IRONCLAD", act=1, floor=6, encounter_class="normal"),
        legal_actions=[LegalAction(action_id="a0", action_type="play_card", card_id="strike")],
    )
    return TrainingSample(
        sample_id="sample1",
        run_id="run1",
        fight_id="fight1",
        step_idx=0,
        state=state,
        history=[],
        legal_actions=state.legal_actions,
        behavior_action_index=0,
        delta=TransitionDelta(),
        fight_label=FightLabel(fight_win=0.0, enemy_hp_fraction_dealt=0.5, self_hp_fraction_remaining=0.25),
    )


class SearchQueueProcessorTests(unittest.TestCase):
    def test_rejects_search_policy_length_mismatch(self) -> None:
        sample = _make_sample()
        request = SearchRequest(request_id=sample.sample_id, sample=sample, priority=1.0)
        processor = SearchQueueProcessor()
        with self.assertRaises(ValueError):
            processor.label(
                [request],
                _BadSearchBackend(SearchLabel(policy=[0.5, 0.5], topk_indices=[0], best_action_index=0)),
            )

    def test_rejects_search_topk_index_out_of_range(self) -> None:
        sample = _make_sample()
        request = SearchRequest(request_id=sample.sample_id, sample=sample, priority=1.0)
        processor = SearchQueueProcessor()
        with self.assertRaises(ValueError):
            processor.label(
                [request],
                _BadSearchBackend(SearchLabel(policy=[1.0], topk_indices=[1], best_action_index=0)),
            )


if __name__ == "__main__":
    unittest.main()
