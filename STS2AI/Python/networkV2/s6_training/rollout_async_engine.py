"""高吞吐 rollout 异步引擎：多进程 actor + 集中 batched inference。"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import logging
import queue
import threading
import time
from typing import Any, Callable

import torch
import torch.multiprocessing as mp
from torch.distributions import Categorical

from networkV2.s1_schema.token_banks import DECISION_DOMAINS, TokenBank, UnifiedTokenBanks
from networkV2.s5_net.bank_max_spec import BankMaxSpec, DEFAULT_MAX_SPEC, BankOverflowError
from networkV2.s5_net.graph_runner import GraphBatchBucketCache
from networkV2.s6_training.batch import PaddedBank


logger = logging.getLogger(__name__)

DOMAIN_TO_CODE = {name: idx for idx, name in enumerate(DECISION_DOMAINS)}
CODE_TO_DOMAIN = {idx: name for name, idx in DOMAIN_TO_CODE.items()}


def _bank_names(max_spec: BankMaxSpec) -> tuple[str, ...]:
    return tuple(
        name for name in getattr(max_spec, "__annotations__", {})
        if name != "numeric_dim" and isinstance(getattr(max_spec, name, None), int)
    )


@dataclass(slots=True)
class RolloutEngineConfig:
    rollout_num_actors: int = 4
    rollout_infer_batch_size: int = 8
    rollout_infer_max_wait_ms: float = 1.0
    rollout_queue_depth: int = 256
    rollout_graph_batch_buckets: tuple[int, ...] = (1, 2, 4, 8, 16)
    rollout_graph_enabled: bool = True
    max_spec: BankMaxSpec = field(default_factory=lambda: DEFAULT_MAX_SPEC)
    max_numeric_dim: int = 58
    actor_reply_timeout_s: float = 120.0
    result_poll_timeout_s: float = 0.25
    max_actor_restarts: int = 3
    use_legacy_thread_rollout: bool = False


def parse_rollout_graph_buckets(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, (tuple, list)):
        values = [int(v) for v in raw]
    else:
        text = str(raw or "").strip()
        values = [int(part) for part in text.split(",") if part.strip()]
    buckets = tuple(sorted({value for value in values if value > 0}))
    return buckets or (1, 2, 4, 8, 16)


def resolve_rollout_num_actors(args: argparse.Namespace) -> int:
    value = getattr(args, "rollout_num_actors", None)
    if value is None or int(value) <= 0:
        value = getattr(args, "num_workers", None)
    return max(int(value or 1), 1)


def build_rollout_engine_config(
    args: argparse.Namespace,
    *,
    max_numeric_dim: int,
    max_spec: BankMaxSpec | None = None,
) -> RolloutEngineConfig:
    return RolloutEngineConfig(
        rollout_num_actors=resolve_rollout_num_actors(args),
        rollout_infer_batch_size=max(1, int(getattr(args, "rollout_infer_batch_size", 8))),
        rollout_infer_max_wait_ms=max(
            0.0,
            float(getattr(args, "rollout_infer_max_wait_ms", 1.0)),
        ),
        rollout_queue_depth=max(8, int(getattr(args, "rollout_queue_depth", 256))),
        rollout_graph_batch_buckets=parse_rollout_graph_buckets(
            getattr(args, "rollout_graph_batch_buckets", (1, 2, 4, 8, 16)),
        ),
        rollout_graph_enabled=bool(getattr(args, "rollout_graph_enabled", True)),
        max_spec=max_spec or DEFAULT_MAX_SPEC,
        max_numeric_dim=max(1, int(max_numeric_dim)),
        actor_reply_timeout_s=max(
            1.0,
            float(getattr(args, "rollout_actor_reply_timeout_s", 120.0)),
        ),
        result_poll_timeout_s=max(
            0.01,
            float(getattr(args, "rollout_result_poll_timeout_s", 0.25)),
        ),
        max_actor_restarts=max(0, int(getattr(args, "rollout_max_actor_restarts", 3))),
        use_legacy_thread_rollout=bool(getattr(args, "legacy_thread_rollout", False)),
    )


def add_rollout_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rollout-num-actors",
        type=int,
        default=None,
        help="异步 rollout actor 进程数；默认继承 --num-workers。",
    )
    parser.add_argument(
        "--rollout-infer-batch-size",
        type=int,
        default=8,
        help="集中推理 batch 上限。",
    )
    parser.add_argument(
        "--rollout-infer-max-wait-ms",
        type=float,
        default=1.0,
        help="集中推理凑 batch 的最长等待时间(ms)。",
    )
    parser.add_argument(
        "--rollout-queue-depth",
        type=int,
        default=256,
        help="actor/request/result 队列容量。",
    )
    parser.add_argument(
        "--rollout-graph-batch-buckets",
        type=str,
        default="1,2,4,8,16",
        help="CUDA graph batch bucket，逗号分隔。",
    )
    parser.add_argument(
        "--rollout-graph-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="异步 rollout 的集中推理是否启用 batch-bucket CUDA graph。",
    )
    parser.add_argument(
        "--legacy-thread-rollout",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rollout-actor-reply-timeout-s",
        type=float,
        default=120.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rollout-result-poll-timeout-s",
        type=float,
        default=0.25,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rollout-max-actor-restarts",
        type=int,
        default=3,
        help=argparse.SUPPRESS,
    )


def runtime_stats_to_metrics(prefix: str, stats: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("requests", "batches", "graph_hits", "eager_fallbacks", "queue_wait_ms_avg", "infer_ms_avg"):
        value = stats.get(key)
        if value is None:
            continue
        out[f"{prefix}_{key}"] = float(value)
    batch_hist = stats.get("batch_hist") or {}
    for batch_size, count in batch_hist.items():
        out[f"{prefix}_batch_hist_{batch_size}"] = float(count)
    return out


@dataclass(slots=True)
class RolloutTaskEnvelope:
    task_id: int
    payload: Any


@dataclass(slots=True)
class InferenceRequestMeta:
    actor_id: int
    task_id: int
    request_id: int
    decision_domain: str
    legal_len: int
    greedy: bool
    submitted_at_ns: int


@dataclass(slots=True)
class InferenceReply:
    task_id: int
    request_id: int
    chosen_action_index: int
    old_log_prob: float
    value_estimate: float
    batch_id: int
    queue_wait_ms: float
    infer_ms: float


@dataclass(slots=True)
class ActorSampleEnvelope:
    actor_id: int
    task_id: int
    samples: list[Any]
    infos: list[dict[str, Any]]
    actor_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SharedBankTensors:
    numeric: torch.Tensor
    type_ids: torch.Tensor
    ts_ids: torch.Tensor
    mask: torch.Tensor


@dataclass
class SharedRolloutSlot:
    bank_names: tuple[str, ...]
    max_spec: BankMaxSpec
    max_numeric_dim: int
    banks: dict[str, SharedBankTensors]
    decision_domain_code: torch.Tensor
    encounter_idx: torch.Tensor
    legal_len: torch.Tensor
    greedy: torch.Tensor
    request_id: torch.Tensor
    active_task_id: torch.Tensor


def create_shared_rollout_slot(
    *,
    max_spec: BankMaxSpec | None = None,
    max_numeric_dim: int = 58,
) -> SharedRolloutSlot:
    spec = max_spec or DEFAULT_MAX_SPEC
    names = _bank_names(spec)
    banks: dict[str, SharedBankTensors] = {}
    for name in names:
        max_len = spec.get(name)
        banks[name] = SharedBankTensors(
            numeric=torch.zeros(max_len, max_numeric_dim, dtype=torch.float32).share_memory_(),
            type_ids=torch.zeros(max_len, dtype=torch.long).share_memory_(),
            ts_ids=torch.zeros(max_len, dtype=torch.long).share_memory_(),
            mask=torch.zeros(max_len, dtype=torch.bool).share_memory_(),
        )
    return SharedRolloutSlot(
        bank_names=names,
        max_spec=spec,
        max_numeric_dim=max_numeric_dim,
        banks=banks,
        decision_domain_code=torch.zeros(1, dtype=torch.long).share_memory_(),
        encounter_idx=torch.zeros(1, dtype=torch.long).share_memory_(),
        legal_len=torch.zeros(1, dtype=torch.long).share_memory_(),
        greedy=torch.zeros(1, dtype=torch.bool).share_memory_(),
        request_id=torch.full((1,), -1, dtype=torch.long).share_memory_(),
        active_task_id=torch.full((1,), -1, dtype=torch.long).share_memory_(),
    )


def _write_token_bank(
    dst: SharedBankTensors,
    *,
    bank: TokenBank | None,
    max_len: int,
    max_numeric_dim: int,
) -> None:
    dst.numeric.zero_()
    dst.type_ids.zero_()
    dst.ts_ids.zero_()
    dst.mask.zero_()
    if bank is None or bank.is_empty:
        return
    tokens = list(bank.tokens)
    if len(tokens) > max_len:
        raise BankOverflowError(
            f"bank '{bank.bank_name}' has {len(tokens)} tokens > max_len {max_len}",
        )
    for idx, tok in enumerate(tokens):
        if tok.token_type == "pad":
            break
        n = min(len(tok.numeric), max_numeric_dim)
        if n > 0:
            dst.numeric[idx, :n] = torch.tensor(tok.numeric[:n], dtype=torch.float32)
        dst.type_ids[idx] = tok.type_idx
        dst.ts_ids[idx] = tok.time_scale_idx
        dst.mask[idx] = True


def write_unified_banks_to_slot(
    slot: SharedRolloutSlot,
    banks: UnifiedTokenBanks,
    *,
    encounter_idx: int,
    legal_len: int,
    greedy: bool,
    request_id: int,
) -> None:
    bank_map = {bank.bank_name: bank for bank in banks.all_banks()}
    for name in slot.bank_names:
        _write_token_bank(
            slot.banks[name],
            bank=bank_map.get(name),
            max_len=slot.max_spec.get(name),
            max_numeric_dim=slot.max_numeric_dim,
        )
    slot.decision_domain_code[0] = DOMAIN_TO_CODE.get(banks.decision_domain, 0)
    slot.encounter_idx[0] = int(encounter_idx)
    slot.legal_len[0] = int(legal_len)
    slot.greedy[0] = bool(greedy)
    slot.request_id[0] = int(request_id)


def read_batched_banks_from_slots(
    requests: list[InferenceRequestMeta],
    slots: list[SharedRolloutSlot],
) -> tuple[dict[str, PaddedBank], torch.Tensor]:
    if not requests:
        return {}, torch.zeros(0, dtype=torch.long)
    names = slots[0].bank_names
    batched: dict[str, PaddedBank] = {}
    for name in names:
        numeric = torch.stack([slots[req.actor_id].banks[name].numeric for req in requests], dim=0)
        type_ids = torch.stack([slots[req.actor_id].banks[name].type_ids for req in requests], dim=0)
        ts_ids = torch.stack([slots[req.actor_id].banks[name].ts_ids for req in requests], dim=0)
        mask = torch.stack([slots[req.actor_id].banks[name].mask for req in requests], dim=0)
        batched[name] = PaddedBank(
            numeric=numeric,
            type_ids=type_ids,
            ts_ids=ts_ids,
            mask=mask,
            bank_name=name,
        )
    encounter_idx = torch.tensor(
        [int(slots[req.actor_id].encounter_idx[0].item()) for req in requests],
        dtype=torch.long,
    )
    return batched, encounter_idx


class QueueInferenceClient:
    """actor 进程侧 inference client。"""

    def __init__(
        self,
        *,
        actor_id: int,
        request_queue: mp.Queue,
        reply_queue: mp.Queue,
        slot: SharedRolloutSlot,
        reply_timeout_s: float = 120.0,
    ):
        self.actor_id = int(actor_id)
        self.request_queue = request_queue
        self.reply_queue = reply_queue
        self.slot = slot
        self.reply_timeout_s = float(reply_timeout_s)
        self._next_request_id = 1

    def infer(
        self,
        banks: UnifiedTokenBanks,
        *,
        encounter_idx: int,
        legal_len: int,
        greedy: bool,
        task_id: int,
    ) -> InferenceReply:
        request_id = self._next_request_id
        self._next_request_id += 1
        write_unified_banks_to_slot(
            self.slot,
            banks,
            encounter_idx=encounter_idx,
            legal_len=legal_len,
            greedy=greedy,
            request_id=request_id,
        )
        meta = InferenceRequestMeta(
            actor_id=self.actor_id,
            task_id=int(task_id),
            request_id=request_id,
            decision_domain=banks.decision_domain,
            legal_len=int(legal_len),
            greedy=bool(greedy),
            submitted_at_ns=time.perf_counter_ns(),
        )
        self.request_queue.put(meta)
        while True:
            reply = self.reply_queue.get(timeout=self.reply_timeout_s)
            if reply.request_id == request_id and reply.task_id == int(task_id):
                return reply


class InferenceCoordinator:
    """主进程常驻 batcher。"""

    def __init__(
        self,
        *,
        net: torch.nn.Module,
        config: RolloutEngineConfig,
        request_queue: mp.Queue,
        reply_queues: list[mp.Queue],
        slots: list[SharedRolloutSlot],
        model_kind: str = "unified",
    ):
        self.net = net
        self.config = config
        self.request_queue = request_queue
        self.reply_queues = reply_queues
        self.slots = slots
        self.model_kind = model_kind
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: deque[InferenceRequestMeta] = deque()
        self._batch_id = 0
        self._stats_lock = threading.Lock()
        self._stats = {
            "requests": 0,
            "batches": 0,
            "graph_hits": 0,
            "eager_fallbacks": 0,
            "queue_wait_ms_sum": 0.0,
            "infer_ms_sum": 0.0,
            "batch_hist": Counter(),
        }
        self._graph_caches: dict[str, GraphBatchBucketCache] = {}
        if self.config.rollout_graph_enabled and self.model_kind == "unified":
            for domain in ("combat", "card_reward", "shop", "route", "rest", "event", "selection"):
                self._graph_caches[domain] = GraphBatchBucketCache(
                    self.net,
                    decision_domain=domain,
                    buckets=self.config.rollout_graph_batch_buckets,
                    max_spec=self.config.max_spec,
                    device=next(self.net.parameters()).device,
                    enabled=self.config.rollout_graph_enabled,
                )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "requests": self._stats["requests"],
                "batches": self._stats["batches"],
                "graph_hits": self._stats["graph_hits"],
                "eager_fallbacks": self._stats["eager_fallbacks"],
                "queue_wait_ms_avg": (
                    self._stats["queue_wait_ms_sum"] / max(self._stats["requests"], 1)
                ),
                "infer_ms_avg": (
                    self._stats["infer_ms_sum"] / max(self._stats["batches"], 1)
                ),
                "batch_hist": dict(self._stats["batch_hist"]),
            }

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._stats = {
                "requests": 0,
                "batches": 0,
                "graph_hits": 0,
                "eager_fallbacks": 0,
                "queue_wait_ms_sum": 0.0,
                "infer_ms_sum": 0.0,
                "batch_hist": Counter(),
            }

    def _record_batch(self, batch_size: int, *, infer_ms: float, queue_wait_ms_sum: float, graph_hit: bool) -> None:
        with self._stats_lock:
            self._stats["requests"] += int(batch_size)
            self._stats["batches"] += 1
            self._stats["queue_wait_ms_sum"] += float(queue_wait_ms_sum)
            self._stats["infer_ms_sum"] += float(infer_ms)
            self._stats["batch_hist"][int(batch_size)] += 1
            if graph_hit:
                self._stats["graph_hits"] += 1
            else:
                self._stats["eager_fallbacks"] += 1

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            batch = self._next_batch()
            if not batch:
                continue
            try:
                self._process_batch(batch)
            except Exception as e:
                logger.exception("InferenceCoordinator batch failed: %s", e)
                for req in batch:
                    self.reply_queues[req.actor_id].put(
                        InferenceReply(
                            task_id=req.task_id,
                            request_id=req.request_id,
                            chosen_action_index=0,
                            old_log_prob=0.0,
                            value_estimate=0.5,
                            batch_id=-1,
                            queue_wait_ms=0.0,
                            infer_ms=0.0,
                        )
                    )

    def _next_batch(self) -> list[InferenceRequestMeta]:
        if not self._pending:
            try:
                first = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                return []
            self._pending.append(first)
        first = self._pending.popleft()
        batch = [first]
        target_domain = first.decision_domain
        deadline = time.perf_counter() + self.config.rollout_infer_max_wait_ms / 1000.0

        while len(batch) < self.config.rollout_infer_batch_size:
            picked = None
            for idx, meta in enumerate(self._pending):
                if meta.decision_domain == target_domain:
                    picked = self._pending[idx]
                    del self._pending[idx]
                    break
            if picked is not None:
                batch.append(picked)
                continue
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                meta = self.request_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if meta.decision_domain == target_domain:
                batch.append(meta)
            else:
                self._pending.append(meta)
        return batch

    def _forward_batch(
        self,
        *,
        decision_domain: str,
        batched_banks: dict[str, PaddedBank],
        encounter_idx: torch.Tensor,
    ) -> tuple[Any, bool]:
        if self.model_kind == "unified":
            cache = self._graph_caches.get(decision_domain)
            if cache is not None:
                out, bucket = cache.run(batched_banks, encounter_idx)
                return out, bucket is not None
            with torch.no_grad():
                return (
                    self.net(
                        batched_banks=batched_banks,
                        decision_domain=decision_domain,
                        encounter_idx=encounter_idx.to(next(self.net.parameters()).device),
                    ),
                    False,
                )

        with torch.no_grad():
            return (
                self.net(
                    batched_banks=batched_banks,
                    encounter_idx=encounter_idx.to(next(self.net.parameters()).device),
                ),
                False,
            )

    def _value_at(self, output: Any, row: int, decision_domain: str) -> float:
        if self.model_kind == "combat_v2":
            return float(output.values.fight_win[row].item())
        if decision_domain == "combat" and output.values is not None:
            return float(output.values.fight_win[row].item())
        if output.run_eval is not None:
            return float(output.run_eval.run_win_prob[row].item())
        return 0.5

    def _process_batch(self, batch: list[InferenceRequestMeta]) -> None:
        decision_domain = batch[0].decision_domain
        batched_banks, encounter_idx = read_batched_banks_from_slots(batch, self.slots)
        t0 = time.perf_counter()
        output, graph_hit = self._forward_batch(
            decision_domain=decision_domain,
            batched_banks=batched_banks,
            encounter_idx=encounter_idx,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self._batch_id += 1
        batch_id = self._batch_id
        queue_wait_ms_sum = 0.0

        for row, req in enumerate(batch):
            legal_len = max(int(req.legal_len), 1)
            logits = output.logits[row, :legal_len]
            mask = output.action_mask[row, :legal_len]
            masked_logits = torch.nan_to_num(
                logits.masked_fill(~mask, float("-inf")),
                nan=0.0,
            )
            dist = Categorical(logits=masked_logits)
            idx_t = masked_logits.argmax() if req.greedy else dist.sample()
            log_prob_t = dist.log_prob(idx_t)
            queue_wait_ms = max(0.0, (time.perf_counter_ns() - req.submitted_at_ns) / 1_000_000.0)
            queue_wait_ms_sum += queue_wait_ms
            reply = InferenceReply(
                task_id=req.task_id,
                request_id=req.request_id,
                chosen_action_index=int(idx_t.item()),
                old_log_prob=float(log_prob_t.item()),
                value_estimate=self._value_at(output, row, decision_domain),
                batch_id=batch_id,
                queue_wait_ms=queue_wait_ms,
                infer_ms=infer_ms,
            )
            self.reply_queues[req.actor_id].put(reply)

        self._record_batch(
            len(batch),
            infer_ms=infer_ms,
            queue_wait_ms_sum=queue_wait_ms_sum,
            graph_hit=graph_hit,
        )


class PersistentActorRuntime:
    """actor 进程池 + 推理 coordinator。"""

    def __init__(
        self,
        *,
        actor_entry: Callable[..., None],
        actor_init_payload: dict[str, Any],
        net: torch.nn.Module,
        config: RolloutEngineConfig,
        model_kind: str = "unified",
    ):
        self.actor_entry = actor_entry
        self.actor_init_payload = dict(actor_init_payload)
        self.net = net
        self.config = config
        self.model_kind = model_kind
        self.ctx = mp.get_context("spawn")
        self.request_queue: mp.Queue = self.ctx.Queue(maxsize=self.config.rollout_queue_depth)
        self.result_queue: mp.Queue = self.ctx.Queue(maxsize=self.config.rollout_queue_depth)
        self.task_queue: mp.Queue = self.ctx.Queue(maxsize=self.config.rollout_queue_depth)
        self.reply_queues: list[mp.Queue] = [
            self.ctx.Queue(maxsize=self.config.rollout_queue_depth)
            for _ in range(self.config.rollout_num_actors)
        ]
        self.slots: list[SharedRolloutSlot] = [
            create_shared_rollout_slot(
                max_spec=self.config.max_spec,
                max_numeric_dim=self.config.max_numeric_dim,
            )
            for _ in range(self.config.rollout_num_actors)
        ]
        self.coordinator = InferenceCoordinator(
            net=self.net,
            config=self.config,
            request_queue=self.request_queue,
            reply_queues=self.reply_queues,
            slots=self.slots,
            model_kind=self.model_kind,
        )
        self.processes: dict[int, mp.Process] = {}
        self.restart_counts: Counter[int] = Counter()
        self._next_task_id = 1
        self._inflight: dict[int, RolloutTaskEnvelope] = {}
        self._inflight_started_ns: dict[int, int] = {}

    def start(self) -> None:
        self.coordinator.start()
        for actor_id in range(self.config.rollout_num_actors):
            self._spawn_actor(actor_id)

    def _spawn_actor(self, actor_id: int) -> None:
        proc = self.ctx.Process(
            target=self.actor_entry,
            args=(
                actor_id,
                self.task_queue,
                self.result_queue,
                self.request_queue,
                self.reply_queues[actor_id],
                self.slots[actor_id],
                self.actor_init_payload,
            ),
            daemon=False,
        )
        proc.start()
        self.processes[actor_id] = proc

    def submit_tasks(self, tasks: list[Any]) -> list[int]:
        ids: list[int] = []
        for payload in tasks:
            task = RolloutTaskEnvelope(task_id=self._next_task_id, payload=payload)
            self._next_task_id += 1
            self._inflight[task.task_id] = task
            self._inflight_started_ns[task.task_id] = time.perf_counter_ns()
            self.task_queue.put(task)
            ids.append(task.task_id)
        return ids

    def gather_results(self, expected_tasks: int) -> list[ActorSampleEnvelope]:
        results: list[ActorSampleEnvelope] = []
        while len(results) < expected_tasks:
            try:
                item = self.result_queue.get(timeout=self.config.result_poll_timeout_s)
            except queue.Empty:
                self._restart_dead_actors()
                self._restart_stuck_actors()
                continue
            results.append(item)
            self._inflight.pop(int(item.task_id), None)
            self._inflight_started_ns.pop(int(item.task_id), None)
        return results

    def _restart_actor(self, actor_id: int, *, active_task_id: int, reason: str) -> None:
        proc = self.processes.get(actor_id)
        if proc is None:
            return
        if self.restart_counts[actor_id] >= self.config.max_actor_restarts:
            return
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        if active_task_id >= 0 and active_task_id in self._inflight:
            self.task_queue.put(self._inflight[active_task_id])
            self._inflight_started_ns[active_task_id] = time.perf_counter_ns()
        self.slots[actor_id].active_task_id[0] = -1
        self.restart_counts[actor_id] += 1
        logger.warning(
            "[rollout_async] respawning actor %s (%s, restart=%s)",
            actor_id,
            reason,
            self.restart_counts[actor_id],
        )
        self._spawn_actor(actor_id)

    def _restart_dead_actors(self) -> None:
        for actor_id, proc in list(self.processes.items()):
            if proc.is_alive():
                continue
            active_task_id = int(self.slots[actor_id].active_task_id[0].item())
            self._restart_actor(
                actor_id,
                active_task_id=active_task_id,
                reason="dead",
            )

    def _restart_stuck_actors(self) -> None:
        deadline_ns = int(self.config.actor_reply_timeout_s * 1_000_000_000)
        now_ns = time.perf_counter_ns()
        for actor_id, proc in list(self.processes.items()):
            if not proc.is_alive():
                continue
            active_task_id = int(self.slots[actor_id].active_task_id[0].item())
            if active_task_id < 0:
                continue
            started_ns = self._inflight_started_ns.get(active_task_id)
            if started_ns is None:
                continue
            if now_ns - started_ns <= deadline_ns:
                continue
            self._restart_actor(
                actor_id,
                active_task_id=active_task_id,
                reason="timeout",
            )

    def stats(self) -> dict[str, Any]:
        data = self.coordinator.stats()
        data["actor_restarts"] = dict(self.restart_counts)
        return data

    def reset_stats(self) -> None:
        self.coordinator.reset_stats()
        self.restart_counts.clear()

    def shutdown(self) -> None:
        for _ in range(self.config.rollout_num_actors):
            self.task_queue.put(None)
        for proc in self.processes.values():
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
        self.processes.clear()
        self.coordinator.stop()
