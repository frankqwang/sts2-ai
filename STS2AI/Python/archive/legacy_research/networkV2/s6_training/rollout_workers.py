"""统一的 async rollout worker/runtime 适配层。

上层 trainer 只生成任务并消费结果，不再各自维护：
  - actor 进程主循环
  - QueueInferenceClient 初始化
  - per-actor sim/client/featurizer 生命周期
  - runtime 启停样板
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from networkV2.s0_bridge.combat_training_env import PipeBackedCombatTrainingClient
from networkV2.s0_bridge.combat_training_env import PipeBackedCombatTrainingClient as JsonCombatCatalogClient
from networkV2.s0_bridge.full_run_env import BinaryBackedFullRunClient
from networkV2.s0_bridge.headless_sim_runner import DEFAULT_DLL_PATH, DEFAULT_REPO_ROOT
from networkV2.s0_bridge.combat_session import CombatSession as ProtoCombatSession
from networkV2.s3_temporal_state.combat_state_tracker import CombatStateTracker
from networkV2.s4_featurization.decision_featurizer import DecisionFeaturizer
from networkV2.s6_training.rollout_async_engine import (
    ActorSampleEnvelope,
    PersistentActorRuntime,
    QueueInferenceClient,
    RolloutEngineConfig,
)


def _run_async_actor_loop(
    actor_id: int,
    task_queue,
    result_queue,
    request_queue,
    reply_queue,
    shared_slot,
    actor_init_payload: dict[str, Any],
    *,
    worker_factory: Callable[[int, dict[str, Any]], Any],
    task_runner: Callable[[Any, Any, QueueInferenceClient, dict[str, Any], int], tuple[list[Any], list[dict[str, Any]], dict[str, float]]],
    worker_closer: Callable[[Any], None] | None = None,
) -> None:
    worker_state = worker_factory(actor_id, actor_init_payload)
    inference_client = QueueInferenceClient(
        actor_id=actor_id,
        request_queue=request_queue,
        reply_queue=reply_queue,
        slot=shared_slot,
        reply_timeout_s=float(actor_init_payload.get("reply_timeout_s", 120.0)),
    )
    try:
        while True:
            task_env = task_queue.get()
            if task_env is None:
                break
            shared_slot.active_task_id[0] = int(task_env.task_id)
            samples_out: list[Any] = []
            infos: list[dict[str, Any]] = []
            actor_metrics: dict[str, float] = {}
            try:
                samples_out, infos, actor_metrics = task_runner(
                    worker_state,
                    task_env.payload,
                    inference_client,
                    actor_init_payload,
                    int(task_env.task_id),
                )
            except Exception as e:
                infos = [{"error": str(e), "steps": 0}]
            finally:
                result_queue.put(ActorSampleEnvelope(
                    actor_id=int(actor_id),
                    task_id=int(task_env.task_id),
                    samples=samples_out,
                    infos=infos,
                    actor_metrics=actor_metrics,
                ))
                shared_slot.active_task_id[0] = -1
    finally:
        if worker_closer is not None:
            try:
                worker_closer(worker_state)
            except Exception:
                pass


def _close_client_worker(worker_state: dict[str, Any]) -> None:
    client = worker_state.get("client")
    if client is not None and hasattr(client, "close"):
        client.close()


def _make_cotrainer_worker_state(actor_id: int, actor_init_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "featurizer": DecisionFeaturizer(),
        "client": ProtoCombatSession(
            port=int(actor_init_payload["base_port"]) + int(actor_id),
            auto_launch=True,
            connect_timeout_s=float(actor_init_payload.get("connect_timeout_s", 30.0)),
        ),
    }


def _run_cotrainer_task(
    worker_state: dict[str, Any],
    task: Any,
    inference_client: QueueInferenceClient,
    actor_init_payload: dict[str, Any],
    task_id: int,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, float]]:
    from networkV2.s6_training.combat_cotrainer import (
        chained_combat_rollout,
        combat_rollout,
        skada_chain_combat_rollout,
    )

    client = worker_state["client"]
    featurizer = worker_state["featurizer"]
    max_steps = int(actor_init_payload["max_steps"])
    samples_out: list[Any] = []
    infos: list[dict[str, Any]] = []

    if task and task[0] == "chain":
        _tag, sequence, chain_deck, seed_prefix, record_traj = task
        samples, sub_infos = chained_combat_rollout(
            client,
            None,
            featurizer,
            sequence,
            chain_deck,
            max_steps_per_combat=max_steps,
            seed_prefix=seed_prefix,
            record_trajectory=record_traj,
            graph_runner_holder=None,
            inference_client=inference_client,
            task_id=task_id,
        )
        samples_out.extend(samples)
        infos.extend(sub_infos)
    elif task and task[0] == "skada_chain":
        _tag, task_chain, seed_prefix, record_traj = task
        samples, sub_infos = skada_chain_combat_rollout(
            client,
            None,
            featurizer,
            task_chain,
            max_steps_per_combat=max_steps,
            seed_prefix=seed_prefix,
            record_trajectory=record_traj,
            graph_runner_holder=None,
            inference_client=inference_client,
            task_id=task_id,
        )
        samples_out.extend(samples)
        infos.extend(sub_infos)
    else:
        enc_id, rt, deck, seed, record_traj = task
        samples, info = combat_rollout(
            client,
            None,
            featurizer,
            enc_id,
            rt,
            deck,
            max_steps=max_steps,
            seed=seed,
            record_trajectory=record_traj,
            graph_runner_holder=None,
            inference_client=inference_client,
            task_id=task_id,
        )
        samples_out.extend(samples)
        infos.append(info)
    return samples_out, infos, {}


def cotrainer_actor_entry(
    actor_id: int,
    task_queue,
    result_queue,
    request_queue,
    reply_queue,
    shared_slot,
    actor_init_payload: dict[str, Any],
) -> None:
    _run_async_actor_loop(
        actor_id,
        task_queue,
        result_queue,
        request_queue,
        reply_queue,
        shared_slot,
        actor_init_payload,
        worker_factory=_make_cotrainer_worker_state,
        task_runner=_run_cotrainer_task,
        worker_closer=_close_client_worker,
    )


def create_cotrainer_runtime(
    *,
    args,
    net: torch.nn.Module,
    rollout_cfg: RolloutEngineConfig,
) -> PersistentActorRuntime:
    runtime = PersistentActorRuntime(
        actor_entry=cotrainer_actor_entry,
        actor_init_payload={
            "base_port": int(args.base_port),
            "max_steps": int(args.max_steps),
            "connect_timeout_s": 30.0,
            "reply_timeout_s": float(rollout_cfg.actor_reply_timeout_s),
        },
        net=net,
        config=rollout_cfg,
        model_kind="unified",
    )
    runtime.start()
    return runtime


def open_cotrainer_catalog_client(port: int) -> JsonCombatCatalogClient:
    return JsonCombatCatalogClient(
        port=int(port),
        auto_launch=True,
        connect_timeout_s=30.0,
    )


def _make_fullrun_worker_state(actor_id: int, actor_init_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "featurizer": DecisionFeaturizer(),
        "client": BinaryBackedFullRunClient(
            port=int(actor_init_payload["base_port"]) + int(actor_id),
            auto_launch=True,
            connect_timeout_s=float(actor_init_payload.get("connect_timeout_s", 30.0)),
            protocol="proto",
            repo_root=str(actor_init_payload.get("repo_root") or DEFAULT_REPO_ROOT),
            dll_path=str(actor_init_payload.get("dll_path") or DEFAULT_DLL_PATH),
        ),
    }


def _run_fullrun_task(
    worker_state: dict[str, Any],
    payload: Any,
    inference_client: QueueInferenceClient,
    actor_init_payload: dict[str, Any],
    task_id: int,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, float]]:
    from networkV2.s6_training.train_full_run_v2 import run_full_episode

    samples, info = run_full_episode(
        worker_state["client"],
        None,
        worker_state["featurizer"],
        seed=str((payload or {}).get("seed", "")),
        max_steps=int((payload or {}).get("max_steps", actor_init_payload.get("max_steps", 800))),
        greedy=bool((payload or {}).get("greedy", False)),
        record_trajectory=bool((payload or {}).get("record_trajectory", False)),
        inference_client=inference_client,
        task_id=task_id,
        capture_root=str((payload or {}).get("capture_root", "")),
    )
    return samples, [info], {}


def fullrun_actor_entry(
    actor_id: int,
    task_queue,
    result_queue,
    request_queue,
    reply_queue,
    shared_slot,
    actor_init_payload: dict[str, Any],
) -> None:
    _run_async_actor_loop(
        actor_id,
        task_queue,
        result_queue,
        request_queue,
        reply_queue,
        shared_slot,
        actor_init_payload,
        worker_factory=_make_fullrun_worker_state,
        task_runner=_run_fullrun_task,
        worker_closer=_close_client_worker,
    )


def create_fullrun_runtime(
    *,
    args,
    net: torch.nn.Module,
    rollout_cfg: RolloutEngineConfig,
) -> PersistentActorRuntime:
    runtime = PersistentActorRuntime(
        actor_entry=fullrun_actor_entry,
        actor_init_payload={
            "base_port": int(args.port),
            "max_steps": int(args.max_steps),
            "connect_timeout_s": 30.0,
            "reply_timeout_s": float(rollout_cfg.actor_reply_timeout_s),
            "repo_root": str(DEFAULT_REPO_ROOT),
            "dll_path": str(DEFAULT_DLL_PATH),
        },
        net=net,
        config=rollout_cfg,
        model_kind="unified",
    )
    runtime.start()
    return runtime


def open_fullrun_catalog_client(port: int) -> BinaryBackedFullRunClient:
    return BinaryBackedFullRunClient(
        port=int(port),
        auto_launch=True,
        connect_timeout_s=30.0,
        protocol="json",
        repo_root=str(DEFAULT_REPO_ROOT),
        dll_path=str(DEFAULT_DLL_PATH),
    )


def _make_combat_v2_worker_state(actor_id: int, actor_init_payload: dict[str, Any]) -> dict[str, Any]:
    from networkV2.s0_bridge.combat_training_env import PipeBackedCombatTrainingClient

    return {
        "featurizer": DecisionFeaturizer(),
        "client": PipeBackedCombatTrainingClient(
            port=int(actor_init_payload["base_port"]) + int(actor_id),
            auto_launch=True,
            connect_timeout_s=float(actor_init_payload.get("connect_timeout_s", 30.0)),
        ),
    }


def _run_combat_v2_task(
    worker_state: dict[str, Any],
    payload: Any,
    inference_client: QueueInferenceClient,
    actor_init_payload: dict[str, Any],
    task_id: int,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, float]]:
    from networkV2.s6_training.train_combat_v2 import collect_combat_rollout

    tracker = CombatStateTracker()
    samples, info = collect_combat_rollout(
        worker_state["client"],
        None,
        worker_state["featurizer"],
        tracker,
        encounter_id=str(payload["encounter_id"]),
        room_type=str(payload["room_type"]),
        build_spec=payload.get("build_spec"),
        max_steps=int(payload.get("max_steps", actor_init_payload.get("max_episode_steps", 200))),
        greedy=bool(payload.get("greedy", False)),
        inference_client=inference_client,
        task_id=task_id,
    )
    return samples, [info], {}


def combat_v2_actor_entry(
    actor_id: int,
    task_queue,
    result_queue,
    request_queue,
    reply_queue,
    shared_slot,
    actor_init_payload: dict[str, Any],
) -> None:
    _run_async_actor_loop(
        actor_id,
        task_queue,
        result_queue,
        request_queue,
        reply_queue,
        shared_slot,
        actor_init_payload,
        worker_factory=_make_combat_v2_worker_state,
        task_runner=_run_combat_v2_task,
        worker_closer=_close_client_worker,
    )


def create_combat_v2_runtime(
    *,
    args,
    net: torch.nn.Module,
    rollout_cfg: RolloutEngineConfig,
) -> PersistentActorRuntime:
    runtime = PersistentActorRuntime(
        actor_entry=combat_v2_actor_entry,
        actor_init_payload={
            "base_port": int(args.port),
            "max_episode_steps": int(args.max_episode_steps),
            "connect_timeout_s": 30.0,
            "reply_timeout_s": float(rollout_cfg.actor_reply_timeout_s),
        },
        net=net,
        config=rollout_cfg,
        model_kind="combat_v2",
    )
    runtime.start()
    return runtime


def open_combat_v2_catalog_client(port: int) -> PipeBackedCombatTrainingClient:
    return PipeBackedCombatTrainingClient(
        port=int(port),
        auto_launch=True,
        connect_timeout_s=30.0,
    )


__all__ = [
    "combat_v2_actor_entry",
    "create_combat_v2_runtime",
    "create_cotrainer_runtime",
    "create_fullrun_runtime",
    "cotrainer_actor_entry",
    "fullrun_actor_entry",
    "open_cotrainer_catalog_client",
    "open_fullrun_catalog_client",
]
