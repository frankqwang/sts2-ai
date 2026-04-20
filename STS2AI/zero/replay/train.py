from __future__ import annotations

"""基于 skada replay case 索引跑多 case combat 训练。"""

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ZERO_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(ZERO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ZERO_PACKAGE_ROOT))

from zero import ZeroConfig, ZeroLoopRunner
from zero.buffers import ArtifactStore
from zero.config import EvalConfig, TeacherConfig, TrainConfig, ZERO_RUNTIME_DEFAULTS
from zero.orchestration.trainer import LocalCheckpointStore
from zero.paths import ZeroPaths
from zero.replay import FixedSkadaCaseEvaluator, MultiCaseAggregateTeacher, SkadaReplayRuntime, load_case_index


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
        default=Path("Assets/datasets/zero_skada_replay_cases/v0_103_2_a0_single_combat_v1/cases.jsonl"),
    )
    parser.add_argument("--port", type=int, default=ZERO_RUNTIME_DEFAULTS.default_port)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--train-case-limit", type=int, default=8)
    parser.add_argument("--eval-case-limit", type=int, default=4)
    parser.add_argument("--collect-episodes", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("STS2AI/Artifacts/zero/skada_replay_train_2026-04-20"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cases = load_case_index(args.case_index)
    if not cases:
        raise ValueError(f"case index 为空: {args.case_index}")

    shuffled = list(cases)
    random.Random(args.seed).shuffle(shuffled)
    eval_cases = shuffled[: max(1, min(args.eval_case_limit, len(shuffled)))]
    remaining = shuffled[len(eval_cases) :]
    train_pool = remaining if remaining else shuffled
    train_cases = train_pool[: max(1, min(args.train_case_limit, len(train_pool)))]

    run_name = (
        f"cases_{len(train_cases)}_eval_{len(eval_cases)}"
        f"_iters_{args.iterations}_seed_{args.seed}"
    )
    output_root = args.output_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    config = ZeroConfig(
        paths=ZeroPaths(root=output_root),
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
            episodes_per_cohort=args.collect_episodes,
            promote_min_win_rate_gain=-1.0,
            allow_hp_remaining_drop=1.0,
        ),
    )

    artifact_store = ArtifactStore(config.paths)
    checkpoint_store = LocalCheckpointStore(config.paths.checkpoints)
    evaluator = FixedSkadaCaseEvaluator(eval_cases, port=args.port, auto_launch=True, connect_timeout_s=45.0)
    runner = ZeroLoopRunner(
        config=config,
        artifact_store=artifact_store,
        checkpoint_store=checkpoint_store,
        evaluator=evaluator,
    )
    teacher = MultiCaseAggregateTeacher(train_cases)

    train_case_cycle = list(train_cases)
    selection_rng = random.Random(args.seed)

    def runtime_factory():
        case = selection_rng.choice(train_case_cycle)
        return SkadaReplayRuntime(case, port=args.port, auto_launch=True, connect_timeout_s=45.0)

    baseline_policy = RandomPolicy()
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
        "train_cases": [case.to_dict() for case in train_cases],
        "eval_cases": [case.to_dict() for case in eval_cases],
        "baseline": [asdict(item) for item in baseline],
        "manifests": manifests,
    }
    metrics_path = output_root / "run_metrics.json"
    metrics_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "metrics_path": str(metrics_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
