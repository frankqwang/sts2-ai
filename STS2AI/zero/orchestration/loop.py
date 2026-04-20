from __future__ import annotations

"""Top-level iterative loop for collect -> label -> train -> evaluate -> promote."""

import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch

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
        self._seed_everything(self.config.seed)
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
        self._active_training_state: dict[str, object] | None = None
        self._baseline_eval: list | None = None
        self._try_resume()

    def bootstrap(self, *, version: str, policy: Policy, baseline_eval: list | None = None) -> None:
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
        self.artifact_store.reset_iteration_outputs(iteration)
        collector_policy = self._active_policy or student_policy
        if collector_policy is None:
            raise ValueError("run_iteration 需要 student_policy，或已有晋级后的 active policy。")

        collector_version = self._active_version or type(collector_policy).__name__
        iteration_started_at = time.perf_counter()
        self._write_progress(
            iteration,
            phase="iteration",
            status="started",
            collector_version=collector_version,
            teacher_version=type(teacher_oracle).__name__,
        )
        self._write_status(
            iteration,
            phase="started",
            collector_version=collector_version,
            teacher_version=type(teacher_oracle).__name__,
        )
        collect_started_at = time.perf_counter()
        transitions = self._collector.collect(
            runtime_factory=runtime_factory,
            policy=collector_policy,
            episodes=self.config.collect.episodes_per_iteration,
            max_steps=self.config.collect.max_steps_per_episode,
            epsilon_greedy=self.config.collect.epsilon_greedy,
            temperature=self.config.collect.temperature,
            seed=self.config.seed + iteration,
            on_episode_start=lambda event: self._write_progress(iteration, phase="collect_episode", status="started", **event),
            on_transition=lambda transition: self.artifact_store.append_raw_run_row(iteration, transition.to_dict()),
            on_episode_end=lambda event: self._write_progress(iteration, phase="collect_episode", status="completed", **event),
        )
        self._write_progress(
            iteration,
            phase="collect",
            status="completed",
            duration_s=round(time.perf_counter() - collect_started_at, 6),
            transitions=len(transitions),
        )
        self._write_status(
            iteration,
            phase="collect_completed",
            transitions=len(transitions),
            elapsed_s=round(time.perf_counter() - iteration_started_at, 6),
        )

        build_started_at = time.perf_counter()
        samples = self._sample_builder.build(transitions)
        online_entries = self._admission.build_online_entries(samples)
        teacher_requests = self._teacher_queue.select(samples)
        labeled_samples = self._teacher_processor.label(teacher_requests, teacher_oracle, runtime_factory=runtime_factory)
        teacher_entries = self._admission.build_teacher_entries(labeled_samples)
        self._pools.add_many(online_entries)
        self._pools.add_many(teacher_entries)
        self._write_progress(
            iteration,
            phase="build_and_label",
            status="completed",
            duration_s=round(time.perf_counter() - build_started_at, 6),
            samples=len(samples),
            online_entries=len(online_entries),
            teacher_requests=len(teacher_requests),
            labeled_samples=len(labeled_samples),
            teacher_entries=len(teacher_entries),
        )
        self._write_status(
            iteration,
            phase="build_and_label_completed",
            samples=len(samples),
            teacher_requests=len(teacher_requests),
            elapsed_s=round(time.perf_counter() - iteration_started_at, 6),
        )

        trainer = self._build_trainer()
        train_started_at = time.perf_counter()
        training_summary = trainer.train_iteration(self._pools)
        self._write_progress(
            iteration,
            phase="train",
            status="completed",
            duration_s=round(time.perf_counter() - train_started_at, 6),
            steps=training_summary.steps,
            total_loss=training_summary.total_loss,
            learning_rate=training_summary.learning_rate,
            skipped_non_finite_steps=training_summary.skipped_non_finite_steps,
        )
        self._write_status(
            iteration,
            phase="train_completed",
            train_steps=training_summary.steps,
            total_loss=training_summary.total_loss,
            elapsed_s=round(time.perf_counter() - iteration_started_at, 6),
        )
        candidate_policy = ModelPolicyAdapter(trainer.model, self._collator, self.config.encoder.history_steps)
        candidate_payload = {
            "model_state": trainer.model.state_dict(),
            "training_state": trainer.state_dict(),
            "seed": self.config.seed,
        }
        version = f"student_v{iteration:04d}"
        candidate_path = self.checkpoint_store.save_candidate(version, candidate_payload)
        self._set_evaluator_trace_context(iteration=iteration, phase="candidate_eval")
        eval_started_at = time.perf_counter()
        evaluations = self.evaluator.evaluate(candidate_policy)
        self._write_progress(
            iteration,
            phase="candidate_eval",
            status="completed",
            duration_s=round(time.perf_counter() - eval_started_at, 6),
            cohorts=len(evaluations),
        )
        self._write_status(
            iteration,
            phase="candidate_eval_completed",
            cohorts=len(evaluations),
            elapsed_s=round(time.perf_counter() - iteration_started_at, 6),
        )
        baseline = baseline_eval if baseline_eval is not None else self._baseline_eval
        promotion = self._promotion.decide(candidate_version=version, current=evaluations, baseline=baseline)

        if promotion.promoted:
            promoted_payload = {
                **candidate_payload,
                "baseline_eval": [asdict(item) for item in evaluations],
            }
            self.checkpoint_store.save(version, promoted_payload)
            self.checkpoint_store.write_active_version(version)
            self.checkpoint_store.discard(candidate_path)
            self._active_version = version
            self._active_policy = candidate_policy
            self._active_training_state = trainer.state_dict()
            self._baseline_eval = evaluations
        elif not self.config.checkpoints.keep_rejected_checkpoints:
            self.checkpoint_store.discard(candidate_path)

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
        self._write_progress(
            iteration,
            phase="iteration",
            status="completed",
            duration_s=round(time.perf_counter() - iteration_started_at, 6),
            promoted=promotion.promoted,
            promotion_reason=promotion.reason,
        )
        self._write_status(
            iteration,
            phase="completed",
            promoted=promotion.promoted,
            promotion_reason=promotion.reason,
            elapsed_s=round(time.perf_counter() - iteration_started_at, 6),
        )
        return manifest

    def _build_trainer(self) -> ZeroTrainer:
        model = ZeroNet(self.config.encoder)
        trainer = ZeroTrainer(model, self.config.train, self.config.losses, self._collator)
        if self._active_version is not None:
            payload = self.checkpoint_store.load(self._active_version)
            model.load_state_dict(payload.get("model_state", payload), strict=False)
            trainer.load_state_dict(payload.get("training_state", self._active_training_state))
        elif self._active_training_state is not None:
            trainer.load_state_dict(self._active_training_state)
        return trainer

    def _try_resume(self) -> None:
        read_active = getattr(self.checkpoint_store, "read_active_version", None)
        if not callable(read_active):
            return
        version = read_active()
        if not version:
            return
        payload = self.checkpoint_store.load(version)
        model = ZeroNet(self.config.encoder)
        model.load_state_dict(payload.get("model_state", payload), strict=False)
        self._active_version = version
        self._active_policy = ModelPolicyAdapter(model, self._collator, self.config.encoder.history_steps)
        self._active_training_state = payload.get("training_state")
        baseline_rows = payload.get("baseline_eval")
        if baseline_rows:
            from ..domain import EvalSummary

            self._baseline_eval = [EvalSummary(**row) for row in baseline_rows]

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _write_progress(self, iteration: int, *, phase: str, status: str, **payload: object) -> None:
        self.artifact_store.write_progress_event(
            iteration,
            {
                "phase": phase,
                "status": status,
                **payload,
            },
        )

    def _write_status(self, iteration: int, *, phase: str, **payload: object) -> None:
        self.artifact_store.write_status(
            iteration,
            {
                "phase": phase,
                **payload,
            },
        )

    def _set_evaluator_trace_context(self, *, iteration: int, phase: str) -> None:
        configure = getattr(self.evaluator, "set_trace_context", None)
        if callable(configure):
            configure(iteration=iteration, phase=phase)
