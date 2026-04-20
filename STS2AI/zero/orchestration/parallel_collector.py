from __future__ import annotations

"""仅并发 rollout collect 的轻量 collector。

设计边界：
- 只并发采样，不并发训练和评估。
- 目前优先服务 ordered-run replay 训练：runtime_factory 需要提供 `clone_for_port(...)`。
- 为了避免跨进程传大对象，这里先用线程并发；底层 env.step 主要在外部 sim 进程中执行，
  因此仍然能把多 env 的 rollout 重叠起来。
"""

import copy
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..domain import RawTransition
from ..ports import BattleRuntime, Policy
from .collector import TrajectoryCollector


class ParallelTrajectoryCollector:
    def __init__(self, *, parallel_envs: int, ports: list[int]):
        self._parallel_envs = max(1, int(parallel_envs))
        self._ports = list(ports)
        if len(self._ports) < self._parallel_envs:
            raise ValueError("ports 数量不足，无法覆盖所有并发 env。")

    def collect(
        self,
        *,
        runtime_factory: Callable[[], BattleRuntime],
        policy: Policy,
        episodes: int,
        max_steps: int = 200,
        epsilon_greedy: float = 0.0,
        temperature: float = 0.0,
        seed: int | None = None,
        on_episode_start: Callable[[dict[str, object]], None] | None = None,
        on_transition: Callable[[RawTransition], None] | None = None,
        on_episode_end: Callable[[dict[str, object]], None] | None = None,
    ) -> list[RawTransition]:
        if self._parallel_envs <= 1 or episodes <= 1:
            return TrajectoryCollector().collect(
                runtime_factory=runtime_factory,
                policy=policy,
                episodes=episodes,
                max_steps=max_steps,
                epsilon_greedy=epsilon_greedy,
                temperature=temperature,
                seed=seed,
                on_episode_start=on_episode_start,
                on_transition=on_transition,
                on_episode_end=on_episode_end,
            )

        clone_factory = getattr(runtime_factory, "clone_for_port", None)
        if not callable(clone_factory):
            raise ValueError("并发 collect 目前要求 runtime_factory 提供 clone_for_port(port)。")

        episode_counts = _split_episodes(episodes, self._parallel_envs)
        callback_lock = threading.Lock()

        def _wrap_callback(callback, payload):
            if callback is None:
                return
            with callback_lock:
                callback(payload)

        def _run_worker(worker_id: int, worker_episodes: int) -> list[RawTransition]:
            worker_factory = clone_factory(self._ports[worker_id])
            worker_policy = _clone_policy(policy)
            return TrajectoryCollector().collect(
                runtime_factory=worker_factory,
                policy=worker_policy,
                episodes=worker_episodes,
                max_steps=max_steps,
                epsilon_greedy=epsilon_greedy,
                temperature=temperature,
                seed=(seed + worker_id) if seed is not None else None,
                on_episode_start=(lambda event: _wrap_callback(on_episode_start, {"worker_id": worker_id, **event})),
                on_transition=(lambda transition: _wrap_callback(on_transition, transition)),
                on_episode_end=(lambda event: _wrap_callback(on_episode_end, {"worker_id": worker_id, **event})),
            )

        futures = []
        with ThreadPoolExecutor(max_workers=self._parallel_envs) as executor:
            for worker_id, worker_episodes in enumerate(episode_counts):
                if worker_episodes <= 0:
                    continue
                futures.append(executor.submit(_run_worker, worker_id, worker_episodes))

        transitions: list[RawTransition] = []
        for future in futures:
            transitions.extend(future.result())
        transitions.sort(key=lambda item: (item.run_id, item.fight_id, item.step_idx))
        return transitions


def _split_episodes(total_episodes: int, envs: int) -> list[int]:
    base = total_episodes // envs
    remainder = total_episodes % envs
    return [base + (1 if idx < remainder else 0) for idx in range(envs)]


def _clone_policy(policy: Policy) -> Policy:
    clone_hook = getattr(policy, "clone_for_rollout", None)
    if callable(clone_hook):
        return clone_hook()
    return copy.deepcopy(policy)
