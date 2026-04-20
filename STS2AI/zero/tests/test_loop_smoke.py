from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from zero.buffers import ArtifactStore
from zero.config import CollectConfig, EvalConfig, TrainConfig, ZeroConfig
from zero.domain import BattleState, EnemyState, EvalSummary, HandCardState, LegalAction, PileSummary, PlayerState, StaticContext
from zero.orchestration import LocalCheckpointStore, ZeroLoopRunner
from zero.paths import ZeroPaths


class FakeRuntime:
    def __init__(self) -> None:
        self._step = 0
        self._state = _make_state(step=0)

    def reset(self, *, seed: str | None = None) -> BattleState:
        self._step = 0
        self._state = _make_state(step=0)
        return self._state

    def get_state(self) -> BattleState:
        return self._state

    def step(self, action_index: int) -> BattleState:
        self._step += 1
        if self._step >= 2:
            self._state = _make_state(step=self._step, terminal=True, outcome="victory")
        else:
            self._state = _make_state(step=self._step)
        return self._state

    def close(self) -> None:
        return None


class FakePolicy:
    def select_action(self, state: BattleState) -> int:
        return 0

    def score_actions(self, state: BattleState) -> list[float]:
        return [1.0, 0.1]

    def estimate_uncertainty(self, state: BattleState) -> float:
        return 0.7


class FakeTeacher:
    def label_request(self, request, runtime_factory=None, seed=None):
        from zero.domain import TeacherLabel

        return TeacherLabel(policy=[0.8, 0.2], topk_indices=[0], best_action_index=0, ranking_margin=0.6, teacher_value=0.9)


class FakeEvaluator:
    def evaluate(self, policy) -> list[EvalSummary]:
        return [
            EvalSummary(
                cohort_name="main",
                fight_win_rate=0.6,
                enemy_hp_fraction_dealt=0.8,
                self_hp_fraction_remaining=0.5,
                teacher_agreement_at_1=0.5,
            )
        ]


def _make_state(*, step: int, terminal: bool = False, outcome: str = "") -> BattleState:
    return BattleState(
        player=PlayerState(hp=80.0 - step * 2, max_hp=80.0, block=3.0, energy=3.0),
        enemies=[EnemyState(enemy_id="slime", hp=max(0.0, 20.0 - step * 10), max_hp=20.0, block=0.0, intent_id="attack")],
        hand=[HandCardState(card_id="strike", cost_now=1.0, damage_now=6.0)],
        piles=PileSummary(draw_pile_size=5, discard_pile_size=step),
        context=StaticContext(character_id="IRONCLAD", act=1, floor=6, encounter_class="elite"),
        legal_actions=[
            LegalAction(action_id="play_strike", action_type="play_card", card_id="strike"),
            LegalAction(action_id="end_turn", action_type="end_turn"),
        ],
        terminal=terminal,
        run_outcome=outcome,
    )


class ZeroLoopSmokeTests(unittest.TestCase):
    def test_single_iteration_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = ZeroConfig(
                paths=ZeroPaths(root=root),
                collect=CollectConfig(episodes_per_iteration=2, max_steps_per_episode=2),
                train=TrainConfig(batch_size=2, steps_per_iteration=1),
                evaluation=EvalConfig(episodes_per_cohort=2, promote_min_win_rate_gain=-1.0),
            )
            runner = ZeroLoopRunner(
                config=config,
                artifact_store=ArtifactStore(config.paths),
                checkpoint_store=LocalCheckpointStore(config.paths.checkpoints),
                evaluator=FakeEvaluator(),
            )
            manifest = runner.run_iteration(
                iteration=1,
                runtime_factory=FakeRuntime,
                student_policy=FakePolicy(),
                teacher_oracle=FakeTeacher(),
                baseline_eval=None,
            )
            self.assertTrue(manifest.promotion.promoted)
            self.assertEqual(manifest.collector_version, "FakePolicy")
            self.assertEqual(manifest.sample_counts["teacher_requests"], 4)
            self.assertTrue((config.paths.manifests / "iter_0001.json").exists())
            self.assertEqual(runner.checkpoint_store.read_active_version(), "student_v0001")

            resumed_runner = ZeroLoopRunner(
                config=config,
                artifact_store=ArtifactStore(config.paths),
                checkpoint_store=LocalCheckpointStore(config.paths.checkpoints),
                evaluator=FakeEvaluator(),
            )
            manifest2 = resumed_runner.run_iteration(
                iteration=2,
                runtime_factory=FakeRuntime,
                student_policy=None,
                teacher_oracle=FakeTeacher(),
                baseline_eval=None,
            )
            self.assertTrue((config.paths.checkpoints / "student_v0002.pt").exists())
            self.assertEqual(manifest2.iteration, 2)
            self.assertEqual(manifest2.collector_version, "student_v0001")


if __name__ == "__main__":
    unittest.main()
