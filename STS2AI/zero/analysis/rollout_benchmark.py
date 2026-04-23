from __future__ import annotations

"""并发 rollout 吞吐基准。

目标：
- 固定同一批 replay case、同一总 episode 数
- 对比 1 env / 4 env 的 wall time 与 step/s
- 显式区分“总体 step/s”与“单 env 内核 step/s”

说明：
- 这里只量化并发 collect 能否提速，不直接改训练主循环
- worker 进程各自拉起独立 sim，避免共享单一 port 互相阻塞
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

ZERO_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(ZERO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ZERO_PACKAGE_ROOT))

from zero.orchestration.collector import TrajectoryCollector
from zero.paths import STS2AI_ROOT
from zero.replay import OrderedRunRuntimeFactory, SkadaReplayRuntime, load_case_index
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


@dataclass(slots=True)
class WorkerResult:
    worker_id: int
    env_count: int
    port: int
    sim_pid: int
    sim_log_path: str
    episodes: int
    transitions: int
    duration_s: float
    wall_step_throughput: float
    avg_episode_duration_s: float
    avg_episode_steps: float
    avg_step_throughput: float
    avg_core_step_throughput: float
    timeouts: int


def _run_worker(
    *,
    worker_id: int,
    env_count: int,
    port: int,
    episodes: int,
    seed: int,
    ordered_run: bool,
    host_path: str,
    case_payloads: list[dict[str, object]],
) -> dict[str, object]:
    random.seed(seed)
    from zero.replay import OrderedRunRuntimeFactory, SkadaCombatCase, SkadaReplayRuntime

    cases = [SkadaCombatCase.from_dict(payload) for payload in case_payloads]
    collector = TrajectoryCollector()
    episode_events: list[dict[str, object]] = []

    with launch_shared_proto_sim(port=port, connect_timeout_s=45.0, host_path=host_path) as sim_info:
        if ordered_run:
            runtime_factory = OrderedRunRuntimeFactory(
                cases,
                port=port,
                auto_launch=False,
                connect_timeout_s=45.0,
            )
        else:
            chooser = random.Random(seed)

            def runtime_factory():
                case = chooser.choice(cases)
                return SkadaReplayRuntime(case, port=port, auto_launch=False, connect_timeout_s=45.0)

        started_at = time.perf_counter()
        transitions = collector.collect(
            runtime_factory=runtime_factory,
            policy=RandomPolicy(),
            episodes=episodes,
            max_steps=200,
            seed=seed,
            on_episode_end=lambda event: episode_events.append(dict(event)),
        )
        duration_s = time.perf_counter() - started_at

    total_steps = sum(int(event.get("steps", 0) or 0) for event in episode_events)
    avg_episode_duration_s = (
        sum(float(event.get("duration_s", 0.0) or 0.0) for event in episode_events) / max(len(episode_events), 1)
    )
    avg_episode_steps = total_steps / max(len(episode_events), 1)
    avg_step_throughput = (
        sum(float(event.get("step_throughput", 0.0) or 0.0) for event in episode_events) / max(len(episode_events), 1)
    )
    avg_core_step_throughput = (
        sum(float(event.get("core_step_throughput", 0.0) or 0.0) for event in episode_events) / max(len(episode_events), 1)
    )
    timeouts = sum(1 for event in episode_events if bool(event.get("truncated", False)))

    return asdict(
        WorkerResult(
            worker_id=worker_id,
            env_count=env_count,
            port=port,
            sim_pid=int(sim_info["pid"]),
            sim_log_path=str(sim_info.get("log_path", "")),
            episodes=episodes,
            transitions=len(transitions),
            duration_s=duration_s,
            wall_step_throughput=total_steps / max(duration_s, 1e-6),
            avg_episode_duration_s=avg_episode_duration_s,
            avg_episode_steps=avg_episode_steps,
            avg_step_throughput=avg_step_throughput,
            avg_core_step_throughput=avg_core_step_throughput,
            timeouts=timeouts,
        )
    )


def _split_episodes(total_episodes: int, envs: int) -> list[int]:
    base = total_episodes // envs
    remainder = total_episodes % envs
    return [base + (1 if idx < remainder else 0) for idx in range(envs)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-index",
        type=Path,
        default=STS2AI_ROOT / "Assets" / "datasets" / "zero_skada_replay_cases" / "v0_103_2_a0_single_combat_v1" / "cases.jsonl",
    )
    parser.add_argument("--run-id", type=int, default=1312734)
    parser.add_argument("--ordered-run", action="store_true", default=True)
    parser.add_argument("--max-run-combats", type=int, default=3)
    parser.add_argument("--total-episodes", type=int, default=64)
    parser.add_argument("--env-options", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--port-base", type=int, default=18150)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--host-path",
        type=Path,
        default=STS2AI_ROOT / "Artifacts" / "tmp" / "headlesssim_build_dynamic_pool" / "HeadlessSim.dll",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=STS2AI_ROOT / "Artifacts" / "zero",
    )
    args = parser.parse_args()

    cases = [case for case in load_case_index(args.case_index) if int(case.run_id) == int(args.run_id)]
    if not cases:
        raise ValueError(f"case index 中未找到 run_id={args.run_id}")
    cases = sorted(cases, key=lambda case: (int(case.floor), str(case.encounter_id)))
    if args.max_run_combats > 0:
        cases = cases[: args.max_run_combats]
    case_payloads = [case.to_dict() for case in cases]

    output_root = (
        args.output_root
        / dated_artifact_dir_name("rollout-benchmark")
        / f"run_{args.run_id}_combats_{len(cases)}_episodes_{args.total_episodes}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    overall_started_at = time.perf_counter()
    for env_count in args.env_options:
        worker_episodes = _split_episodes(args.total_episodes, env_count)
        env_started_at = time.perf_counter()
        with ProcessPoolExecutor(max_workers=env_count, mp_context=mp.get_context("spawn")) as executor:
            futures = []
            for worker_id, episodes in enumerate(worker_episodes):
                if episodes <= 0:
                    continue
                futures.append(
                    executor.submit(
                        _run_worker,
                        worker_id=worker_id,
                        env_count=env_count,
                        port=args.port_base + worker_id,
                        episodes=episodes,
                        seed=args.seed + worker_id,
                        ordered_run=args.ordered_run,
                        host_path=str(args.host_path),
                        case_payloads=case_payloads,
                    )
                )
            worker_results = [future.result() for future in futures]
        wall_time_s = time.perf_counter() - env_started_at
        total_transitions = sum(int(item["transitions"]) for item in worker_results)
        total_steps = sum(int(item["episodes"]) * float(item["avg_episode_steps"]) for item in worker_results)
        total_timeouts = sum(int(item["timeouts"]) for item in worker_results)
        summary_rows.append(
            {
                "env_count": env_count,
                "total_episodes": args.total_episodes,
                "wall_time_s": round(wall_time_s, 6),
                "transitions": total_transitions,
                "estimated_steps": round(total_steps, 3),
                "wall_step_throughput": round(total_steps / max(wall_time_s, 1e-6), 6),
                "avg_worker_step_throughput": round(
                    sum(float(item["wall_step_throughput"]) for item in worker_results) / max(len(worker_results), 1),
                    6,
                ),
                "timeouts": total_timeouts,
            }
        )
        detail_rows.extend(worker_results)

    payload = {
        "case_index": str(args.case_index),
        "run_id": args.run_id,
        "ordered_run": bool(args.ordered_run),
        "max_run_combats": len(cases),
        "total_episodes": args.total_episodes,
        "host_path": str(args.host_path),
        "elapsed_s": round(time.perf_counter() - overall_started_at, 6),
        "summary": summary_rows,
        "workers": detail_rows,
    }
    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "summary_path": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    mp.freeze_support()
    main()
