#!/usr/bin/env python3
"""Rollout throughput benchmark for eager vs CUDA graph.

Usage:
    cd STS2AI/Python
    python -m tools.cuda_graph_rollout_bench --workers 2,4 --episodes 24 --warmup-episodes 8

Notes:
  - Reuses combat cotrainer rollout paths to keep benchmark close to training reality.
  - Bypasses curriculum / GAME_CATALOG to avoid unrelated catalog noise.
  - Before timed measurement, each case warms every actor at least once.
  - Async runtime stats are reset after warmup so warmup does not pollute measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import torch

import networkV2.s4_compiler.feature_compiler as feature_compiler_mod
from networkV2.s2_config.mechanism_registry import MechanismRegistry
from networkV2.s5_net.network_config import from_preset
from networkV2.s5_net.unified_net import UnifiedNet
from networkV2.s6_training.combat_cotrainer import CombatClientPool, _worker_collect
from networkV2.s6_training.deck_eval import ironclad_starter_deck
from networkV2.s6_training.rollout_async_engine import (
    ActorSampleEnvelope,
    PersistentActorRuntime,
    RolloutEngineConfig,
)
from networkV2.s6_training.rollout_workers import cotrainer_actor_entry


logger = logging.getLogger(__name__)

ENCOUNTER_ID = "bowlbugs_weak"
ROOM_TYPE = "monster"


def _disable_registry_autoload() -> None:
    """Benchmark should not spend time on mechanism registry autoload."""
    feature_compiler_mod.get_registry = lambda: MechanismRegistry()


def _configure_graph(net: UnifiedNet, enabled: bool, parity_every: int) -> None:
    if not enabled:
        if hasattr(net, "_cuda_graph_cfg"):
            delattr(net, "_cuda_graph_cfg")
        return

    from networkV2.s5_net.graph_runner import patch_dropout_for_graph_safety
    import torch.nn as nn

    patch_dropout_for_graph_safety()
    for module in net.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
        elif isinstance(module, nn.Dropout):
            module.p = 0.0
    net._cuda_graph_cfg = {
        "parity_check_every": int(parity_every),
        "atol": 1e-3,
        "rtol": 1e-3,
        "startup_parity_n": 8,
        "startup_parity_noise": 0.0,
        "strict": True,
    }


def _make_tasks(num_workers: int, episodes: int, seed_prefix: str) -> list[list[tuple]]:
    deck = ironclad_starter_deck()
    tasks_per_worker: list[list[tuple]] = [[] for _ in range(num_workers)]
    for i in range(episodes):
        tasks_per_worker[i % num_workers].append(
            (ENCOUNTER_ID, ROOM_TYPE, deck, f"{seed_prefix}-{i}", False)
        )
    return tasks_per_worker


def _make_async_tasks(episodes: int, seed_prefix: str) -> list[tuple]:
    deck = ironclad_starter_deck()
    return [
        (ENCOUNTER_ID, ROOM_TYPE, deck, f"{seed_prefix}-{i}", False)
        for i in range(episodes)
    ]


def _resolve_warmup_episodes(
    *,
    num_workers: int,
    warmup_episodes: int,
    warmup_each_actor: bool,
) -> int:
    if warmup_each_actor:
        return max(int(warmup_episodes), int(num_workers))
    return max(int(warmup_episodes), 0)


def _aggregate_results(result_q: queue.Queue) -> dict[str, Any]:
    samples = []
    infos: list[dict[str, Any]] = []
    while not result_q.empty():
        item = result_q.get()
        samples.extend(item.get("samples") or [])
        infos.extend(item.get("infos") or [])

    errors = [info for info in infos if info.get("outcome") == "error"]
    prof_agg = {"compile_ms": 0.0, "forward_ms": 0.0, "step_ms": 0.0, "post_ms": 0.0}
    prof_steps = 0
    for info in infos:
        prof = info.get("_prof")
        steps = int(info.get("steps", 0) or 0)
        if not isinstance(prof, dict) or steps <= 0:
            continue
        for key in prof_agg:
            prof_agg[key] += float(prof.get(key, 0.0)) * steps
        prof_steps += steps

    avg_prof = {
        key: (value / prof_steps if prof_steps > 0 else 0.0)
        for key, value in prof_agg.items()
    }
    avg_prof["total_ms"] = sum(avg_prof.values())
    return {
        "steps": len(samples),
        "combats": len(infos),
        "errors": len(errors),
        "error_examples": [str(info.get("error", ""))[:160] for info in errors[:3]],
        "avg_prof_ms": avg_prof,
    }


def _aggregate_actor_results(envelopes: list[ActorSampleEnvelope]) -> dict[str, Any]:
    samples = []
    infos: list[dict[str, Any]] = []
    for env in envelopes:
        samples.extend(env.samples)
        infos.extend(env.infos)

    errors = [info for info in infos if info.get("outcome") == "error"]
    prof_agg = {"compile_ms": 0.0, "forward_ms": 0.0, "step_ms": 0.0, "post_ms": 0.0}
    prof_steps = 0
    for info in infos:
        prof = info.get("_prof")
        steps = int(info.get("steps", 0) or 0)
        if not isinstance(prof, dict) or steps <= 0:
            continue
        for key in prof_agg:
            prof_agg[key] += float(prof.get(key, 0.0)) * steps
        prof_steps += steps

    avg_prof = {
        key: (value / prof_steps if prof_steps > 0 else 0.0)
        for key, value in prof_agg.items()
    }
    avg_prof["total_ms"] = sum(avg_prof.values())
    return {
        "steps": len(samples),
        "combats": len(infos),
        "errors": len(errors),
        "error_examples": [str(info.get("error", ""))[:160] for info in errors[:3]],
        "avg_prof_ms": avg_prof,
    }


def _run_round(
    *,
    pool: CombatClientPool,
    net: UnifiedNet,
    num_workers: int,
    episodes: int,
    max_steps: int,
    seed_prefix: str,
    graph_holders: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    tasks_per_worker = _make_tasks(
        num_workers=num_workers,
        episodes=episodes,
        seed_prefix=seed_prefix,
    )
    result_q: queue.Queue = queue.Queue()
    threads: list[threading.Thread] = []
    t0 = time.perf_counter()
    for worker_id in range(num_workers):
        worker_tasks = tasks_per_worker[worker_id]
        if not worker_tasks:
            continue
        holder = None if graph_holders is None else graph_holders.setdefault(
            worker_id,
            {"worker_id": worker_id, "runner": None, "init_attempted": False},
        )
        thread = threading.Thread(
            target=_worker_collect,
            args=(worker_id, pool, net, worker_tasks, max_steps, result_q, holder),
            daemon=False,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - t0
    summary = _aggregate_results(result_q)
    summary["wall_s"] = wall
    summary["steps_per_s"] = summary["steps"] / wall if wall > 0 else 0.0
    summary["steps_per_s_per_env"] = summary["steps_per_s"] / max(num_workers, 1)
    return summary


def _run_async_round(
    *,
    runtime: PersistentActorRuntime,
    num_workers: int,
    episodes: int,
    seed_prefix: str,
) -> tuple[dict[str, Any], list[ActorSampleEnvelope]]:
    tasks = _make_async_tasks(episodes=episodes, seed_prefix=seed_prefix)
    t0 = time.perf_counter()
    task_ids = runtime.submit_tasks(tasks)
    envelopes = runtime.gather_results(len(task_ids))
    wall = time.perf_counter() - t0
    summary = _aggregate_actor_results(envelopes)
    summary["wall_s"] = wall
    summary["steps_per_s"] = summary["steps"] / wall if wall > 0 else 0.0
    summary["steps_per_s_per_env"] = summary["steps_per_s"] / max(num_workers, 1)
    summary["rollout_stats"] = runtime.stats()
    return summary, envelopes


def _warmup_async_runtime(
    *,
    runtime: PersistentActorRuntime,
    num_workers: int,
    warmup_episodes: int,
    warmup_each_actor: bool,
    seed_prefix: str,
) -> dict[str, Any]:
    target_episodes = _resolve_warmup_episodes(
        num_workers=num_workers,
        warmup_episodes=warmup_episodes,
        warmup_each_actor=warmup_each_actor,
    )
    if target_episodes <= 0:
        runtime.reset_stats()
        return {
            "episodes": 0,
            "rounds": 0,
            "actor_ids": [],
            "actor_coverage_complete": True,
        }

    seen_actor_ids: set[int] = set()
    submitted = 0
    rounds = 0
    max_rounds = max(8, num_workers * 2)
    while submitted < target_episodes or (
        warmup_each_actor and len(seen_actor_ids) < num_workers
    ):
        remaining = max(target_episodes - submitted, 0)
        min_batch = num_workers if warmup_each_actor and len(seen_actor_ids) < num_workers else 1
        batch_episodes = max(remaining, min_batch)
        _, envelopes = _run_async_round(
            runtime=runtime,
            num_workers=num_workers,
            episodes=batch_episodes,
            seed_prefix=f"{seed_prefix}-r{rounds}",
        )
        submitted += batch_episodes
        seen_actor_ids.update(int(env.actor_id) for env in envelopes)
        rounds += 1
        if rounds >= max_rounds:
            break

    runtime.reset_stats()
    return {
        "episodes": submitted,
        "rounds": rounds,
        "actor_ids": sorted(seen_actor_ids),
        "actor_coverage_complete": len(seen_actor_ids) >= num_workers,
    }


def _run_case(
    *,
    num_workers: int,
    episodes: int,
    warmup_episodes: int,
    warmup_each_actor: bool,
    max_steps: int,
    base_port: int,
    use_cuda_graph: bool,
    preset: str,
    parity_every: int,
    engine: str,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UnifiedNet(config=from_preset(preset)).to(device).eval()
    _configure_graph(net, enabled=use_cuda_graph, parity_every=parity_every)

    if engine == "async":
        runtime = PersistentActorRuntime(
            actor_entry=cotrainer_actor_entry,
            actor_init_payload={
                "base_port": int(base_port),
                "max_steps": int(max_steps),
                "connect_timeout_s": 30.0,
                "reply_timeout_s": 120.0,
            },
            net=net,
            config=RolloutEngineConfig(
                rollout_num_actors=num_workers,
                rollout_infer_batch_size=max(2, num_workers),
                rollout_infer_max_wait_ms=1.0,
                rollout_queue_depth=max(64, episodes * 4),
                rollout_graph_enabled=use_cuda_graph,
                max_numeric_dim=int(getattr(net.tokenizer, "max_numeric_dim", 58)),
            ),
            model_kind="unified",
        )
        runtime.start()
        try:
            warmup_info = _warmup_async_runtime(
                runtime=runtime,
                num_workers=num_workers,
                warmup_episodes=warmup_episodes,
                warmup_each_actor=warmup_each_actor,
                seed_prefix=f"warm-{engine}-{num_workers}-{'graph' if use_cuda_graph else 'eager'}",
            )
            measured, _ = _run_async_round(
                runtime=runtime,
                num_workers=num_workers,
                episodes=episodes,
                seed_prefix=f"bench-{engine}-{num_workers}-{'graph' if use_cuda_graph else 'eager'}",
            )
        finally:
            runtime.shutdown()
    else:
        pool = CombatClientPool(base_port=base_port, n_clients=num_workers)
        graph_holders = None if not use_cuda_graph else {
            worker_id: {"worker_id": worker_id, "runner": None, "init_attempted": False}
            for worker_id in range(num_workers)
        }
        try:
            legacy_warmup_episodes = _resolve_warmup_episodes(
                num_workers=num_workers,
                warmup_episodes=warmup_episodes,
                warmup_each_actor=warmup_each_actor,
            )
            warmup_info = {
                "episodes": legacy_warmup_episodes,
                "rounds": 1 if legacy_warmup_episodes > 0 else 0,
                "actor_ids": list(range(num_workers)) if legacy_warmup_episodes > 0 else [],
                "actor_coverage_complete": (not warmup_each_actor) or legacy_warmup_episodes >= num_workers,
            }
            if legacy_warmup_episodes > 0:
                _ = _run_round(
                    pool=pool,
                    net=net,
                    num_workers=num_workers,
                    episodes=legacy_warmup_episodes,
                    max_steps=max_steps,
                    seed_prefix=f"warm-{engine}-{num_workers}-{'graph' if use_cuda_graph else 'eager'}",
                    graph_holders=graph_holders,
                )
            measured = _run_round(
                pool=pool,
                net=net,
                num_workers=num_workers,
                episodes=episodes,
                max_steps=max_steps,
                seed_prefix=f"bench-{engine}-{num_workers}-{'graph' if use_cuda_graph else 'eager'}",
                graph_holders=graph_holders,
            )
        finally:
            pool.close_all()

    measured["workers"] = num_workers
    measured["mode"] = "graph" if use_cuda_graph else "eager"
    measured["engine"] = engine
    measured["base_port"] = base_port
    measured["warmup"] = warmup_info
    return measured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", type=str, default="legacy,async")
    parser.add_argument("--workers", type=str, default="2,4")
    parser.add_argument("--modes", type=str, default="eager,graph")
    parser.add_argument("--episodes", type=int, default=24, help="episodes per measured case")
    parser.add_argument("--warmup-episodes", type=int, default=8, help="warmup episodes per case")
    parser.add_argument(
        "--warmup-each-actor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ensure each actor has actually run before timed measurement",
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--base-port", type=int, default=19400)
    parser.add_argument("--preset", type=str, default="slim")
    parser.add_argument("--graph-parity-every", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    _disable_registry_autoload()

    engine_list = [item.strip().lower() for item in args.engines.split(",") if item.strip()]
    worker_list = [int(item) for item in args.workers.split(",") if item.strip()]
    mode_list = [item.strip().lower() for item in args.modes.split(",") if item.strip()]
    results: list[dict[str, Any]] = []

    print(
        f"{'engine':>8} {'mode':>6} {'envs':>4} {'steps':>8} {'wall_s':>8} "
        f"{'step/s':>10} {'step/s/env':>12} {'fwd_ms':>9} {'step_ms':>9} {'err':>4}"
    )

    case_index = 0
    for engine in engine_list:
        for mode in mode_list:
            use_cuda_graph = mode == "graph"
            for workers in worker_list:
                base_port = args.base_port + case_index * 20
                case_index += 1
                result = _run_case(
                    num_workers=workers,
                    episodes=args.episodes,
                    warmup_episodes=args.warmup_episodes,
                    warmup_each_actor=bool(args.warmup_each_actor),
                    max_steps=args.max_steps,
                    base_port=base_port,
                    use_cuda_graph=use_cuda_graph,
                    preset=args.preset,
                    parity_every=args.graph_parity_every,
                    engine=engine,
                )
                results.append(result)
                prof = result["avg_prof_ms"]
                print(
                    f"{result['engine']:>8} {result['mode']:>6} {result['workers']:>4d} {result['steps']:>8d} {result['wall_s']:>8.2f} "
                    f"{result['steps_per_s']:>10.1f} {result['steps_per_s_per_env']:>12.1f} "
                    f"{prof['forward_ms']:>9.3f} {prof['step_ms']:>9.3f} {result['errors']:>4d}"
                )
                if result["error_examples"]:
                    for example in result["error_examples"]:
                        print(f"       ERR: {example}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "encounter_id": ENCOUNTER_ID,
                    "room_type": ROOM_TYPE,
                    "episodes": args.episodes,
                    "warmup_episodes": args.warmup_episodes,
                    "warmup_each_actor": bool(args.warmup_each_actor),
                    "max_steps": args.max_steps,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s", args.output_json)


if __name__ == "__main__":
    main()
