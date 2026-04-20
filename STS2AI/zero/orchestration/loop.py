from __future__ import annotations

"""Top-level iterative loop for collect -> label -> train -> evaluate -> promote."""

from dataclasses import dataclass
from typing import Callable

from ..buffers import ArtifactStore, SamplePoolSet
from ..config import ZeroConfig
from ..domain import IterationManifest
from ..features import BatchCollator
from ..model import ZeroNet
from ..ports import BattleRuntime, CheckpointStore, Evaluator, Policy, TeacherOracle
from .admission import SampleAdmissionPlanner
from .collector import TrajectoryCollector
from .promotion import PromotionJudge
from .sample_builder import SampleBuilder
from .teacher import TeacherQueueBuilder, TeacherQueueProcessor
from .trainer import ModelPolicyAdapter, ZeroTrainer


@dataclass
class ZeroLoopRunner:
    config: ZeroConfig
    artifact_store: ArtifactStore
    checkpoint_store: CheckpointStore
    evaluator: Evaluator

    def __post_init__(self) -> None:
        self._pools = SamplePoolSet(self.config.pools)
        self._collector = TrajectoryCollector()
        self._sample_builder = SampleBuilder(self.config.encoder)
        self._admission = SampleAdmissionPlanner()
        self._teacher_queue = TeacherQueueBuilder(self.config.teacher)
        self._teacher_processor = TeacherQueueProcessor()
        self._promotion = PromotionJudge(self.config.evaluation)
        self._collator = BatchCollator(self.config.encoder)
        self._active_version: str | None = None
        self._active_policy: Policy | None = None
        self._baseline_eval: list | None = None

    def bootstrap(self, *, version: str, policy: Policy, baseline_eval: list | None = None) -> None:
        """Bind the iterative loop to an already-promoted student version.

        Use this when the collector should start from an existing checkpoint
        instead of an ad-hoc external policy with no matching model weights.
        """
        self._active_version = version
        self._active_policy = policy
        self._baseline_eval = baseline_eval

    def run_iteration(
        self,
        *,
        iteration: int,
        runtime_factory: Callable[[], BattleRuntime],
        student_policy: Policy | None = None,
        teacher_oracle: TeacherOracle,
        baseline_eval: list | None = None,
    ) -> IterationManifest:
        collector_policy = self._active_policy or student_policy
        if collector_policy is None:
            raise ValueError("run_iteration 需要 student_policy，或已有晋级后的 active policy。")
        # Snapshot the collector lineage before training/promotion mutates active state.
        collector_version = self._active_version or type(collector_policy).__name__
        transitions = self._collector.collect(
            runtime_factory=runtime_factory,
            policy=collector_policy,
            episodes=self.config.evaluation.episodes_per_cohort,
        )
        samples = self._sample_builder.build(transitions)
        online_entries = self._admission.build_online_entries(samples)
        teacher_requests = self._teacher_queue.select(samples)
        labeled_samples = self._teacher_processor.label(teacher_requests, teacher_oracle, runtime_factory=runtime_factory)
        teacher_entries = self._admission.build_teacher_entries(labeled_samples)
        self._pools.add_many(online_entries)
        self._pools.add_many(teacher_entries)

        model = ZeroNet(self.config.encoder)
        if self._active_version is not None:
            model.load_state_dict(self.checkpoint_store.load(self._active_version), strict=False)
        trainer = ZeroTrainer(model, self.config.train, self.config.losses, self._collator)
        training_summary = trainer.train_iteration(self._pools)
        candidate_policy = ModelPolicyAdapter(model, self._collator, self.config.encoder.history_steps)

        version = f"student_v{iteration:04d}"
        self.checkpoint_store.save(version, model.state_dict())
        evaluations = self.evaluator.evaluate(candidate_policy)
        baseline = baseline_eval if baseline_eval is not None else self._baseline_eval
        promotion = self._promotion.decide(candidate_version=version, current=evaluations, baseline=baseline)
        if promotion.promoted:
            self._active_version = version
            self._active_policy = candidate_policy
            self._baseline_eval = evaluations

        manifest = IterationManifest(
            iteration=iteration,
            collector_version=collector_version,
            teacher_version=type(teacher_oracle).__name__,
            sample_counts={
                "transitions": len(transitions),
                "samples": len(samples),
                "online_entries": len(online_entries),
                "teacher_requests": len(teacher_requests),
                "labeled_samples": len(labeled_samples),
                "teacher_entries": len(teacher_entries),
            },
            pool_sizes=self._pools.size_by_pool(),
            training=training_summary,
            evaluations=evaluations,
            promotion=promotion,
        )

        self.artifact_store.write_raw_runs(iteration, [transition.to_dict() for transition in transitions])
        self.artifact_store.write_teacher_labels(iteration, teacher_requests)
        self.artifact_store.write_dataset_shard(iteration, online_entries + teacher_entries)
        self.artifact_store.write_manifest(manifest)
        return manifest
