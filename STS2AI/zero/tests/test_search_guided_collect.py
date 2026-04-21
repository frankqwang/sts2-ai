from __future__ import annotations

import unittest

from zero.config import SearchConfig
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
)
from zero.orchestration.collector import SearchGuidedActionSelector, TrajectoryCollector
from zero.orchestration.search import SearchQueueBuilder


def _build_state(*, encounter_id: str = "TEST_ENCOUNTER") -> BattleState:
    raw_actions = [
        {"action": "play_card", "type": "play_card", "index": 0, "card_index": 0, "target_id": 1, "label": "STRIKE_IRONCLAD"},
        {"action": "play_card", "type": "play_card", "index": 1, "card_index": 1, "target_id": 0, "label": "DEFEND_IRONCLAD"},
    ]
    return BattleState(
        player=PlayerState(hp=80, max_hp=80, block=0, energy=3),
        enemies=[EnemyState(enemy_id="enemy", hp=20, max_hp=20, block=0, intent_id="Attack")],
        hand=[
            HandCardState(card_id="STRIKE_IRONCLAD", cost_now=1, damage_now=6, tags=["attack"]),
            HandCardState(card_id="DEFEND_IRONCLAD", cost_now=1, block_now=5, tags=["skill"]),
        ],
        piles=PileSummary(),
        context=StaticContext(
            character_id="IRONCLAD",
            act=1,
            floor=3,
            encounter_class="normal",
            encounter_id=encounter_id,
            metadata={"seed": "seed-guided", "skada_case_id": "test_case"},
        ),
        legal_actions=[
            LegalAction(action_id="play_card_0", action_type="play_card", card_id="STRIKE_IRONCLAD", target_id="enemy"),
            LegalAction(action_id="play_card_1", action_type="play_card", card_id="DEFEND_IRONCLAD"),
        ],
        terminal=False,
        run_outcome="",
        raw={"legal_actions": raw_actions},
    )


class _FakeRuntime:
    def __init__(self):
        self._state = _build_state()

    def reset(self, *, seed: str | None = None):
        self._state = _build_state()
        return self._state

    def get_state(self):
        return self._state

    def step(self, action_index: int):
        self._state = BattleState(
            player=PlayerState(hp=80, max_hp=80, block=0, energy=0),
            enemies=[EnemyState(enemy_id="enemy", hp=14, max_hp=20, block=0, intent_id="Attack")],
            hand=[],
            piles=PileSummary(),
            context=self._state.context,
            legal_actions=[],
            terminal=True,
            run_outcome="victory",
        )
        return self._state

    def close(self):
        return None


class _SparseRawRuntime:
    def __init__(self):
        self._state = _build_state()

    def reset(self, *, seed: str | None = None):
        state = _build_state()
        state.raw = {"legal_actions": []}
        self._state = state
        return self._state

    def get_state(self):
        return self._state

    def step(self, action_index: int):
        self._state = BattleState(
            player=PlayerState(hp=80, max_hp=80, block=0, energy=0),
            enemies=[EnemyState(enemy_id="enemy", hp=14, max_hp=20, block=0, intent_id="Attack")],
            hand=[],
            piles=PileSummary(),
            context=self._state.context,
            legal_actions=[],
            terminal=True,
            run_outcome="victory",
            raw={"legal_actions": []},
        )
        return self._state

    def close(self):
        return None


class _GreedyPolicy:
    def reset_episode(self):
        return None

    def infer(self, state):
        return {"scores": [2.0, 1.0], "action_index": 0, "uncertainty": 0.8}

    def observe_transition(self, state, action_index, next_state):
        return None


class _SearchBackend:
    def label_request(self, request, runtime_factory=None, seed: str | None = None):
        return SearchLabel(
            policy=[0.05, 0.95],
            topk_indices=[1, 0],
            best_action_index=1,
            ranking_margin=0.9,
            search_value=0.8,
            search_trace=[{"action_index": 1, "score_avg": 0.8}],
            metadata={"search_backend": "fake"},
        )


class SearchGuidedCollectTests(unittest.TestCase):
    def test_targeted_search_guidance_overrides_policy_greedy_action(self):
        selector = SearchGuidedActionSelector(
            search_backend=_SearchBackend(),
            queue_builder=SearchQueueBuilder(SearchConfig()),
            priority_threshold=99.0,
            max_guided_steps_per_episode=2,
            target_encounters=("TEST_ENCOUNTER",),
        )
        collector = TrajectoryCollector()
        transitions = collector.collect(
            runtime_factory=lambda: _FakeRuntime(),
            policy=_GreedyPolicy(),
            episodes=1,
            max_steps=2,
            search_guidance_factory=lambda _port=None: selector,
        )
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].action_index, 1)
        self.assertTrue(bool(transitions[0].metadata.get("search_guided")))

    def test_search_only_collect_uses_search_policy_and_marks_transition(self):
        collector = TrajectoryCollector()
        from zero.orchestration.collector import SearchSelfPlaySelector

        transitions = collector.collect(
            runtime_factory=lambda: _FakeRuntime(),
            policy=_GreedyPolicy(),
            episodes=1,
            max_steps=2,
            temperature=0.0,
            search_self_play_factory=lambda _port=None: SearchSelfPlaySelector(search_backend=_SearchBackend()),
        )
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].action_index, 1)
        self.assertTrue(bool(transitions[0].metadata.get("search_collected")))
        self.assertEqual(transitions[0].metadata.get("search_policy"), [0.05, 0.95])

    def test_search_only_collect_keeps_search_choice_when_raw_actions_are_sparse(self):
        collector = TrajectoryCollector()
        from zero.orchestration.collector import SearchSelfPlaySelector

        transitions = collector.collect(
            runtime_factory=lambda: _SparseRawRuntime(),
            policy=_GreedyPolicy(),
            episodes=1,
            max_steps=2,
            temperature=0.0,
            search_self_play_factory=lambda _port=None: SearchSelfPlaySelector(search_backend=_SearchBackend()),
        )
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].action_index, 1)
        self.assertTrue(bool(transitions[0].metadata.get("search_collected")))
        self.assertEqual(transitions[0].metadata.get("search_source"), "search_self_play")


if __name__ == "__main__":
    unittest.main()
