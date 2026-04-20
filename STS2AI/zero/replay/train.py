from __future__ import annotations

"""基于 skada replay case 索引跑多 case combat 训练。"""

import argparse
import json
import random
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path

ZERO_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(ZERO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ZERO_PACKAGE_ROOT))

from zero import ZeroConfig, ZeroLoopRunner
from zero.analysis import generate_training_analysis
from zero.buffers import ArtifactStore
from zero.config import CollectConfig, EvalConfig, TeacherConfig, TrainConfig, ZERO_RUNTIME_DEFAULTS
from zero.orchestration.trainer import LocalCheckpointStore
from zero.paths import STS2AI_ROOT, ZeroPaths
from zero.replay import (
    FixedSkadaCaseEvaluator,
    MultiCaseAggregateTeacher,
    OrderedRunCaseEvaluator,
    OrderedRunRuntimeFactory,
    SkadaReplayRuntime,
    load_case_index,
)
from zero.replay.naming import dated_artifact_dir_name
from zero.replay.shared_sim import launch_shared_proto_sim


class RandomPolicy:
    def reset_episode(self) -> None:
        return None

    def select_action(self, state) -> int:
        if not state.legal_actions:
            return 0
        return random.randrange(len(state.legal_actions))

    def score_actions(self, state) -> list[float]:
        return [1.0 for _ in state.legal_actions]

    def estimate_uncertainty(self, state) -> float:
        return 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-index",
        type=Path,
        default=STS2AI_ROOT / "Assets" / "datasets" / "zero_skada_replay_cases" / "v0_103_2_a0_single_combat_v1" / "cases.jsonl",
    )
    parser.add_argument("--port", type=int, default=ZERO_RUNTIME_DEFAULTS.default_port)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--train-case-limit", type=int, default=8)
    parser.add_argument("--eval-case-limit", type=int, default=4)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--ordered-run", action="store_true")
    parser.add_argument("--max-run-combats", type=int, default=0)
    parser.add_argument("--collect-episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=STS2AI_ROOT / "Artifacts" / "zero",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cases = load_case_index(args.case_index)
    if not cases:
        raise ValueError(f"case index 为空: {args.case_index}")

    curriculum_mode = "random_cases"
    if args.run_id:
        run_cases = [case for case in cases if int(case.run_id) == int(args.run_id)]
        if not run_cases:
            raise ValueError(f"case index 中未找到 run_id={args.run_id}")
        ordered_cases = sorted(run_cases, key=lambda case: (int(case.floor), str(case.encounter_id)))
        if args.max_run_combats > 0:
            ordered_cases = ordered_cases[: args.max_run_combats]
        train_cases = ordered_cases
        eval_cases = ordered_cases
        curriculum_mode = "ordered_run" if args.ordered_run or args.run_id else "single_run"
    else:
        shuffled = list(cases)
        random.Random(args.seed).shuffle(shuffled)
        eval_cases = shuffled[: max(1, min(args.eval_case_limit, len(shuffled)))]
        remaining = shuffled[len(eval_cases) :]
        train_pool = remaining if remaining else shuffled
        train_cases = train_pool[: max(1, min(args.train_case_limit, len(train_pool)))]
        if args.ordered_run:
            curriculum_mode = "ordered_cases"

    run_name = (
        f"{curriculum_mode}_cases_{len(train_cases)}_eval_{len(eval_cases)}"
        f"_iters_{args.iterations}_seed_{args.seed}"
    )
    output_root = args.output_root / dated_artifact_dir_name("skada-replay-train") / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    config = ZeroConfig(
        paths=ZeroPaths(root=output_root),
        collect=CollectConfig(episodes_per_iteration=args.collect_episodes),
        teacher=TeacherConfig(
            top2_gap_threshold=1.0,
            uncertainty_threshold=0.0,
            near_lethal_hp_ratio=1.0,
            max_requests_per_iteration=4096,
        ),
        train=TrainConfig(
            batch_size=16,
            steps_per_iteration=args.train_steps,
            learning_rate=3e-4,
            weight_decay=1e-4,
            grad_clip_norm=1.0,
        ),
        evaluation=EvalConfig(
            episodes_per_cohort=args.eval_episodes,
            promote_min_win_rate_gain=-1.0,
            allow_hp_remaining_drop=1.0,
        ),
    )

    artifact_store = ArtifactStore(config.paths)
    checkpoint_store = LocalCheckpointStore(config.paths.checkpoints)
    with launch_shared_proto_sim(port=args.port, connect_timeout_s=45.0) as sim_info:
        if curriculum_mode == "ordered_run":
            evaluator = OrderedRunCaseEvaluator(
                eval_cases,
                port=args.port,
                auto_launch=False,
                connect_timeout_s=45.0,
                episodes_per_case=config.evaluation.episodes_per_cohort,
                artifact_store=artifact_store,
            )
            runtime_factory = OrderedRunRuntimeFactory(
                train_cases,
                port=args.port,
                auto_launch=False,
                connect_timeout_s=45.0,
            )
        else:
            evaluator = FixedSkadaCaseEvaluator(
                eval_cases,
                port=args.port,
                auto_launch=False,
                connect_timeout_s=45.0,
                episodes_per_case=config.evaluation.episodes_per_cohort,
                artifact_store=artifact_store,
            )
            train_case_cycle = list(train_cases)
            selection_rng = random.Random(args.seed)
            ordered_cycle = deque(train_case_cycle)

            def runtime_factory():
                if curriculum_mode == "ordered_cases":
                    case = ordered_cycle[0]
                    ordered_cycle.rotate(-1)
                else:
                    case = selection_rng.choice(train_case_cycle)
                return SkadaReplayRuntime(case, port=args.port, auto_launch=False, connect_timeout_s=45.0)

        runner = ZeroLoopRunner(
            config=config,
            artifact_store=artifact_store,
            checkpoint_store=checkpoint_store,
            evaluator=evaluator,
        )
        teacher = MultiCaseAggregateTeacher(train_cases)

        baseline_policy = RandomPolicy()
        set_trace_context = getattr(evaluator, "set_trace_context", None)
        if callable(set_trace_context):
            set_trace_context(iteration=0, phase="baseline_eval")
        baseline = evaluator.evaluate(baseline_policy)
        manifests = []
        student_policy = RandomPolicy()
        for iteration in range(1, args.iterations + 1):
            manifest = runner.run_iteration(
                iteration=iteration,
                runtime_factory=runtime_factory,
                student_policy=student_policy,
                teacher_oracle=teacher,
                baseline_eval=baseline if iteration == 1 else None,
            )
            manifests.append(manifest.to_dict())

    metadata = {
        "case_index": str(args.case_index),
        "seed": args.seed,
        "curriculum_mode": curriculum_mode,
        "shared_sim": sim_info,
        "run_id": args.run_id or None,
        "train_cases": [case.to_dict() for case in train_cases],
        "eval_cases": [case.to_dict() for case in eval_cases],
        "baseline": [asdict(item) for item in baseline],
        "manifests": manifests,
    }
    metrics_path = output_root / "run_metrics.json"
    metrics_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    analysis_dir = generate_training_analysis(run_root=output_root, run_metrics_path=metrics_path)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "metrics_path": str(metrics_path),
                "analysis_dir": str(analysis_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
