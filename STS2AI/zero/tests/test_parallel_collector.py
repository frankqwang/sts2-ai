from __future__ import annotations

import unittest

from zero.domain import BattleState, EnemyState, LegalAction, PileSummary, PlayerState, StaticContext
from zero.orchestration.parallel_collector import ParallelTrajectoryCollector


class _SingleStepRuntime:
    def __init__(self, encounter_id: str) -> None:
        self._encounter_id = encounter_id
        self._state = self.reset()

    def reset(self, *, seed: str | None = None) -> BattleState:
        self._state = BattleState(
            player=PlayerState(hp=50.0, max_hp=50.0, block=0.0, energy=3.0),
            enemies=[EnemyState(enemy_id="enemy", hp=10.0, max_hp=10.0, block=0.0, intent_id="attack")],
            hand=[],
            piles=PileSummary(),
            context=StaticContext(character_id="IRONCLAD", encounter_id=self._encounter_id),
            legal_actions=[LegalAction(action_id="play", action_type="play_card", card_id="strike")],
        )
        return self._state

    def get_state(self) -> BattleState:
        return self._state

    def step(self, action_index: int) -> BattleState:
        self._state = BattleState(
            player=PlayerState(hp=50.0, max_hp=50.0, block=0.0, energy=3.0),
            enemies=[EnemyState(enemy_id="enemy", hp=0.0, max_hp=10.0, block=0.0, intent_id="attack", alive=False)],
            hand=[],
            piles=PileSummary(),
            context=StaticContext(character_id="IRONCLAD", encounter_id=self._encounter_id),
            legal_actions=[],
            terminal=True,
            run_outcome="victory",
        )
        return self._state

    def close(self) -> None:
        return None


class _CyclingFactory:
    def __init__(self, encounters: list[str]) -> None:
        self._encounters = list(encounters)
        self._index = 0

    def __call__(self):
        return _SingleStepRuntime(self._encounters[self._index])

    def on_episode_end(self, event: dict[str, object]) -> None:
        outcome = str(event.get("outcome") or "").lower()
        if outcome in {"victory", "win"}:
            self._index = (self._index + 1) % len(self._encounters)
        else:
            self._index = 0

    def clone_for_port(self, port: int) -> "_CyclingFactory":
        return _CyclingFactory(list(self._encounters))


class _GreedyPolicy:
    def select_action(self, state: BattleState) -> int:
        return 0

    def score_actions(self, state: BattleState) -> list[float]:
        return [1.0 for _ in state.legal_actions]


class ParallelCollectorTests(unittest.TestCase):
    def test_worker_clone_advances_its_own_ordered_runtime_factory(self) -> None:
        collector = ParallelTrajectoryCollector(parallel_envs=2, ports=[19001, 19002])
        transitions = collector.collect(
            runtime_factory=_CyclingFactory(["CASE_A", "CASE_B"]),
            policy=_GreedyPolicy(),
            episodes=4,
            max_steps=1,
            seed=7,
        )

        encounter_ids = [transition.state.context.encounter_id for transition in transitions]
        self.assertIn("CASE_A", encounter_ids)
        self.assertIn("CASE_B", encounter_ids)
        self.assertEqual(encounter_ids.count("CASE_A"), 2)
        self.assertEqual(encounter_ids.count("CASE_B"), 2)


if __name__ == "__main__":
    unittest.main()
