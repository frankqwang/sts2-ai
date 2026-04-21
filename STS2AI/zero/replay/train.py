from __future__ import annotations

"""基于 skada replay case 索引跑多 case combat 训练。

这是当前 zero combat 训练的主入口，常见用途有两类：
1. 单 case / 少量 case 的 smoke 与过拟合验证
2. ordered-run curriculum 训练：按同一条 run 的战斗顺序推进，
   中途失败就从第一场重新开始，尽量贴近真实 run 的战斗阶段学习

脚本职责：
- 读取已经清洗好的 `cases.jsonl`
- 选择训练 / 评估用的 case 集合
- 启动 shared sim
- 驱动 `ZeroLoopRunner` 完成 collect -> search -> train -> eval -> promote
- 把 run_metrics / analysis / checkpoints 落到本次产物目录

注意：
- 这里默认依赖已经清洗好的 replay case 索引，不直接读取 `runs_full_detail`
- 想重建索引，请用 `zero.replay.build_case_index`
- collect 支持探索；eval 始终保持贪心，便于稳定比较版本差异
"""

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

from contextlib import ExitStack

from zero import ZeroConfig, ZeroLoopRunner
from zero.analysis import generate_training_analysis
from zero.buffers import ArtifactStore
from zero.config import CollectConfig, EvalConfig, LossWeights, SearchConfig, TrainConfig, ZERO_RUNTIME_DEFAULTS
from zero.features import BatchCollator
from zero.model import ZeroNet
from zero.orchestration import ModelPolicyAdapter, ParallelTrajectoryCollector
from zero.orchestration.trainer import LocalCheckpointStore
from zero.paths import STS2AI_ROOT, ZeroPaths
from zero.replay import (
    FixedSkadaCaseEvaluator,
    MultiCaseAggregateSearchBackend,
    MultiCaseSearchBackend,
    OrderedRunCaseEvaluator,
    OrderedRunRuntimeFactory,
    SkadaReplayRuntime,
    close_shared_replay_runtimes,
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

    def clone_for_rollout(self):
        return RandomPolicy()


def build_fresh_policy(config: ZeroConfig) -> ModelPolicyAdapter:
    """首轮 bootstrap：随机初始化的 ZeroNet，而不是 RandomPolicy。"""

    model = ZeroNet(config.encoder)
    collator = BatchCollator(config.encoder)
    return ModelPolicyAdapter(model, collator, config.encoder.history_steps)


def _build_eval_config(*, episodes_per_cohort: int, strict_promotion: bool) -> EvalConfig:
    """构造 EvalConfig，strict 模式走 dataclass 默认（生产晋级阈值）。

    非 strict 模式下把 `promote_min_win_rate_gain` / `allow_hp_quality_drop`
    放开，让 V1 smoke 跑不至于因为随机 eval 噪声永远无法晋级；此行为仅用于
    训练链路早期 debug，真正产出 checkpoint 时应加 `--strict-promotion`。
    """

    if strict_promotion:
        return EvalConfig(episodes_per_cohort=episodes_per_cohort)
    return EvalConfig(
        episodes_per_cohort=episodes_per_cohort,
        promote_min_win_rate_gain=-1.0,
        allow_hp_quality_drop=1.0,
    )


def config_losses_for_search_mode(search_mode: str) -> LossWeights:
    """按 search 强度给一版保守但可用的损失权重。"""
    weights = LossWeights()
    if search_mode == "weak":
        return weights
    weights.ranking = 0.9
    weights.delta = 0.08
    weights.uncertainty = 0.04
    weights.policy_search_kl_weight = 1.8
    weights.policy_behavior_ce_weight = 0.0
    weights.policy_bad_rollout_ce_scale = 0.0
    return weights


def normalize_search_mode(search_mode: str) -> str:
    """统一搜索模式命名。"""
    return search_mode


def build_search_backend(
    cases,
    *,
    search_mode: str,
    config: SearchConfig,
    port: int,
    auto_launch: bool,
    connect_timeout_s: float,
):
    normalized_mode = normalize_search_mode(search_mode)
    if normalized_mode == "weak":
        return MultiCaseAggregateSearchBackend(cases)
    return MultiCaseSearchBackend(
        cases,
        config=config,
        port=port,
        auto_launch=auto_launch,
        connect_timeout_s=connect_timeout_s,
    )


def build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--target-encounter", type=str, default="")
    parser.add_argument("--target-case-id", type=str, default="")
    parser.add_argument(
        "--target-source",
        choices=["", "encounter", "case", "run-segment"],
        default="",
    )
    parser.add_argument("--collect-episodes", type=int, default=8)
    parser.add_argument("--parallel-envs", type=int, default=1)
    parser.add_argument(
        "--collect-mode",
        choices=["search_only_collect", "policy_only_collect", "search_guided_collect"],
        default="search_only_collect",
        help="collect 动作来源；默认直接按搜索分布自博弈。",
    )
    parser.add_argument(
        "--collect-epsilon-greedy",
        type=float,
        default=0.0,
        help="仅作用于 collect rollout 的 epsilon-greedy 探索概率；评估仍保持贪心。",
    )
    parser.add_argument(
        "--collect-temperature",
        type=float,
        default=0.0,
        help="仅作用于 collect rollout 的 softmax 温度；0 表示关闭温度采样。",
    )
    parser.add_argument("--search-guidance-priority-threshold", type=float, default=1.2)
    parser.add_argument("--search-guidance-max-steps-per-episode", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument(
        "--search-mode",
        dest="search_mode",
        choices=["weak", "search_root_sweep", "search_branching"],
        default="search_root_sweep",
        help="搜索后端模式；默认走 root MCTS 自博弈。",
    )
    parser.add_argument("--search-max-root-actions", type=int, default=8)
    parser.add_argument("--search-rollouts-per-action", type=int, default=2)
    parser.add_argument("--search-max-branch-steps", type=int, default=24)
    parser.add_argument("--search-rollout-policy", choices=["aggregate_search_prior"], default="aggregate_search_prior")
    parser.add_argument(
        "--host-path",
        type=Path,
        default=STS2AI_ROOT / "Artifacts" / "tmp" / "headlesssim_build_dynamic_pool" / "HeadlessSim.dll",
    )
    parser.add_argument(
        "--strict-promotion",
        action="store_true",
        help="启用 EvalConfig 默认晋级阈值（生产训练用）；不加此 flag 时保持 V1 smoke "
        "放开的行为，便于早期迭代观察闭环数据。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=STS2AI_ROOT / "Artifacts",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="显式新开一套训练产物目录；默认优先复用同名 run 目录续训。",
    )
    return parser


def _legacy_run_candidates(base_output_root: Path, run_name: str) -> list[Path]:
    if not base_output_root.exists():
        return []
    candidates: list[Path] = []
    for child in base_output_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.endswith(run_name):
            candidates.append(child)
        nested = child / run_name
        if nested.is_dir():
            candidates.append(nested)
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_run_output_root(
    *,
    base_output_root: Path,
    run_name: str,
    from_scratch: bool,
) -> Path:
    """默认复用同名 run 目录；fresh run 直接落到 `Artifacts/<时间+名字>`。"""

    if from_scratch:
        return base_output_root / dated_artifact_dir_name(run_name)

    stable_root = base_output_root / run_name
    if stable_root.exists():
        return stable_root

    legacy_candidates = _legacy_run_candidates(base_output_root, run_name)
    if legacy_candidates:
        return legacy_candidates[0]
    return stable_root


def _apply_target_filters(cases, *, target_encounter: str, target_case_id: str):
    if target_case_id:
        filtered = [case for case in cases if str(case.case_id) == str(target_case_id)]
        if not filtered:
            raise ValueError(f"未找到 target_case_id={target_case_id}")
        return filtered
    if target_encounter:
        normalized = target_encounter.strip().upper()
        filtered = [case for case in cases if str(case.encounter_id).upper() == normalized]
        if not filtered:
            raise ValueError(f"未找到 target_encounter={target_encounter}")
        return filtered
    return list(cases)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.search_mode = normalize_search_mode(args.search_mode)

    random.seed(args.seed)
    cases = load_case_index(args.case_index)
    if not cases:
        raise ValueError(f"case index 为空: {args.case_index}")

    # 训练集 / 评估集的选择逻辑：
    # - 指定 run_id 时，优先走“同一条 run 的多场战斗课程”
    # - 未指定 run_id 时，按清洗后的 case 索引随机切 train / eval
    curriculum_mode = "random_cases"
    if args.run_id:
        run_cases = [case for case in cases if int(case.run_id) == int(args.run_id)]
        if not run_cases:
            raise ValueError(f"case index 中未找到 run_id={args.run_id}")
        ordered_cases = sorted(run_cases, key=lambda case: (int(case.floor), str(case.encounter_id)))
        ordered_cases = _apply_target_filters(
            ordered_cases,
            target_encounter=args.target_encounter,
            target_case_id=args.target_case_id,
        )
        if args.max_run_combats > 0:
            ordered_cases = ordered_cases[: args.max_run_combats]
        train_cases = ordered_cases
        eval_cases = ordered_cases
        curriculum_mode = "ordered_run" if args.ordered_run or args.run_id else "single_run"
        if args.target_source == "run-segment" and not args.ordered_run:
            curriculum_mode = "targeted_run_segment"
    else:
        filtered_cases = _apply_target_filters(
            cases,
            target_encounter=args.target_encounter,
            target_case_id=args.target_case_id,
        )
        shuffled = list(filtered_cases)
        random.Random(args.seed).shuffle(shuffled)
        eval_cases = shuffled[: max(1, min(args.eval_case_limit, len(shuffled)))]
        remaining = shuffled[len(eval_cases) :]
        train_pool = remaining if remaining else shuffled
        train_cases = train_pool[: max(1, min(args.train_case_limit, len(train_pool)))]
        if args.ordered_run:
            curriculum_mode = "ordered_cases"
        elif args.target_encounter or args.target_case_id:
            curriculum_mode = "targeted_cases"

    # 产物目录结构：
    # - fresh run: STS2AI/Artifacts/MM-DD-HH-MM-<run_name>
    # - resume run: STS2AI/Artifacts/<run_name>
    # run_name 里保留课程模式、case 数、iter 数和 seed，方便之后直接比较。
    run_name = (
        f"{curriculum_mode}_cases_{len(train_cases)}_eval_{len(eval_cases)}"
        f"_iters_{args.iterations}_seed_{args.seed}"
    )
    output_root = resolve_run_output_root(
        base_output_root=args.output_root,
        run_name=run_name,
        from_scratch=bool(args.from_scratch),
    )
    output_root.mkdir(parents=True, exist_ok=True)

    config = ZeroConfig(
        paths=ZeroPaths(root=output_root),
        collect=CollectConfig(
            episodes_per_iteration=args.collect_episodes,
            epsilon_greedy=args.collect_epsilon_greedy,
            temperature=args.collect_temperature,
            mode=args.collect_mode,
            search_guidance_priority_threshold=args.search_guidance_priority_threshold,
            search_guidance_max_steps_per_episode=args.search_guidance_max_steps_per_episode,
            search_guidance_target_encounters=tuple(
                value for value in [args.target_encounter.strip().upper()] if value
            ),
        ),
        search=SearchConfig(
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
        losses=config_losses_for_search_mode(args.search_mode),
        evaluation=_build_eval_config(
            episodes_per_cohort=args.eval_episodes,
            strict_promotion=args.strict_promotion,
        ),
    )
    config.search.mode = args.search_mode
    config.search.max_root_actions = args.search_max_root_actions
    config.search.rollouts_per_action = args.search_rollouts_per_action
    config.search.max_branch_steps = args.search_max_branch_steps
    config.search.allow_branching = args.search_mode == "search_branching"
    config.search.rollout_policy = args.search_rollout_policy

    artifact_store = ArtifactStore(config.paths)
    checkpoint_store = LocalCheckpointStore(config.paths.checkpoints)
    with ExitStack() as stack:
        sim_info = stack.enter_context(
            launch_shared_proto_sim(port=args.port, connect_timeout_s=45.0, host_path=args.host_path)
        )
        collect_ports = [args.port]
        guidance_ports = []
        if args.collect_mode in {"search_guided_collect", "search_only_collect"}:
            guidance_ports.append(args.port + config.collect.search_guidance_port_offset)
        if args.parallel_envs > 1:
            if curriculum_mode != "ordered_run":
                raise ValueError("并发 collect 目前仅支持 ordered_run 模式。")
            collect_ports = [args.port + 1 + offset for offset in range(args.parallel_envs)]
            for collect_port in collect_ports:
                stack.enter_context(
                    launch_shared_proto_sim(
                        port=collect_port,
                        connect_timeout_s=45.0,
                        host_path=args.host_path,
                    )
                )
            if args.collect_mode in {"search_guided_collect", "search_only_collect"}:
                guidance_ports = [port + config.collect.search_guidance_port_offset for port in collect_ports]
        for guidance_port in guidance_ports:
            stack.enter_context(
                launch_shared_proto_sim(
                    port=guidance_port,
                    connect_timeout_s=45.0,
                    host_path=args.host_path,
                )
            )

        try:
            if curriculum_mode == "ordered_run":
                # ordered-run 评估会从第一场开始，失败后直接停止，不再继续评后续战斗。
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
            if args.parallel_envs > 1:
                # 并发 collect 目前只并发 rollout，不并发 trainer / evaluator；
                # 这样能提吞吐，同时避免评估语义和晋级判定变复杂。
                runner._collector = ParallelTrajectoryCollector(
                    parallel_envs=args.parallel_envs,
                    ports=collect_ports,
                )
            search_backend = build_search_backend(
                train_cases,
                search_mode=args.search_mode,
                config=config.search,
                port=args.port,
                auto_launch=False,
                connect_timeout_s=45.0,
            )

            baseline_policy = RandomPolicy()
            set_trace_context = getattr(evaluator, "set_trace_context", None)
            if callable(set_trace_context):
                set_trace_context(iteration=0, phase="baseline_eval")
            baseline = evaluator.evaluate(baseline_policy)
            manifests = []
            policy = build_fresh_policy(config)
            for iteration in range(1, args.iterations + 1):
                manifest = runner.run_iteration(
                    iteration=iteration,
                    runtime_factory=runtime_factory,
                    policy=policy,
                    search_backend=search_backend,
                    baseline_eval=baseline if iteration == 1 else None,
                )
                manifests.append(manifest.to_dict())
        finally:
            close_shared_replay_runtimes()

    metadata = {
        "case_index": str(args.case_index),
        "seed": args.seed,
        "curriculum_mode": curriculum_mode,
        "shared_sim": sim_info,
        "run_id": args.run_id or None,
        "target_encounter": args.target_encounter or None,
        "target_case_id": args.target_case_id or None,
        "from_scratch": bool(args.from_scratch),
        "resolved_output_root": str(output_root),
        "search_mode": args.search_mode,
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
