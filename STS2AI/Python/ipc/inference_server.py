"""Batch inference server for multi-process NN evaluation.

Central GPU inference server that batches requests from multiple worker
processes, doing a single forward pass per batch instead of one per request.

Architecture:
    Main Process (GPU)          Worker Processes (CPU + pipe IPC)
    ┌─────────────────┐         ┌──────────────────────────┐
    │ InferenceServer  │◄──req──│ InferenceClient          │
    │  combat_net      │──res──►│   .combat_inference()    │
    │  ppo_net         │        │   .ppo_inference()       │
    │  batch+forward   │        │   (blocks until result)  │
    └─────────────────┘         └──────────────────────────┘

Usage:
    # Main process
    server = InferenceServer(ppo_net, mcts_net, device, num_workers=8)
    server.start()

    # Worker process
    client = InferenceClient(worker_id, server.request_queue, server.get_result_queue(worker_id))
    logits, value = client.combat_inference(state_features, action_features)
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Request types
REQ_COMBAT = "combat"
REQ_PPO = "ppo"
REQ_SHUTDOWN = "shutdown"


@dataclass
class InferenceRequest:
    worker_id: int
    req_type: str  # REQ_COMBAT or REQ_PPO
    state_features: dict[str, np.ndarray]
    action_features: dict[str, np.ndarray]
    request_id: int = 0


@dataclass
class CombatResult:
    logits: np.ndarray       # (MAX_ACTIONS,)
    value: float


@dataclass
class PPOResult:
    action_idx: int
    log_prob: float
    entropy: float
    value: float


class InferenceServer:
    """Batched GPU inference server running in a daemon thread.

    Collects requests from worker processes via mp.Queue, groups by network
    type, batches tensors, runs a single forward pass, and sends results back.
    """

    def __init__(
        self,
        ppo_net: torch.nn.Module,
        combat_net: torch.nn.Module,
        device: torch.device,
        num_workers: int = 8,
        max_batch: int = 32,
        timeout_s: float = 0.005,
    ):
        self.ppo_net = ppo_net
        self.combat_net = combat_net
        self.device = device
        self.max_batch = max_batch
        self.timeout_s = timeout_s

        # Shared request queue (all workers → server)
        self.request_queue: mp.Queue = mp.Queue()
        # Per-worker result queues (server → each worker)
        self.result_queues: dict[int, mp.Queue] = {}
        for i in range(num_workers):
            self.result_queues[i] = mp.Queue()

        self._thread: threading.Thread | None = None
        self._running = False
        self._total_requests = 0
        self._total_batches = 0

        # NOTE: CUDAGraph capture fails on this model (MultiheadAttention's masked_fill
        # + dynamic ops are incompatible with graph capture on Windows PyTorch 2.x).
        # GPU forward is ~1.7ms/call regardless of batch size (kernel launch dominated).
        # Future: ONNX export or torch.compile (when Triton supports Windows) could help.

    def get_result_queue(self, worker_id: int) -> mp.Queue:
        return self.result_queues[worker_id]

    def start(self) -> None:
        """Start the inference server daemon thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="InferenceServer")
        self._thread.start()
        logger.info("InferenceServer started (device=%s, max_batch=%d, timeout=%.0fms)",
                     self.device, self.max_batch, self.timeout_s * 1000)

    def stop(self) -> None:
        self._running = False
        # Send shutdown sentinel
        try:
            self.request_queue.put_nowait(InferenceRequest(
                worker_id=-1, req_type=REQ_SHUTDOWN,
                state_features={}, action_features={}))
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("InferenceServer stopped. Total: %d requests in %d batches",
                     self._total_requests, self._total_batches)

    def _run(self) -> None:
        """Main server loop: collect requests, batch, forward, dispatch."""
        while self._running:
            batch = self._collect_batch()
            if not batch:
                continue

            # Split by type
            combat_reqs = [r for r in batch if r.req_type == REQ_COMBAT]
            ppo_reqs = [r for r in batch if r.req_type == REQ_PPO]

            if combat_reqs:
                self._batch_combat(combat_reqs)
            if ppo_reqs:
                self._batch_ppo(ppo_reqs)

    def _collect_batch(self) -> list[InferenceRequest]:
        """Collect up to max_batch requests, waiting up to timeout_s.

        Strategy: after receiving the first request, wait a short grace
        period for more workers to submit, improving GPU batch utilization.
        """
        batch: list[InferenceRequest] = []
        deadline = time.monotonic() + self.timeout_s

        # Block on first request
        try:
            req = self.request_queue.get(timeout=self.timeout_s)
            if req.req_type == REQ_SHUTDOWN:
                return []
            batch.append(req)
        except queue.Empty:
            return []

        # Grace period: wait briefly for more workers to arrive
        # This dramatically improves batch fill rate (8→6-8 per batch instead of 1-3)
        grace_deadline = time.monotonic() + min(self.timeout_s, 0.002)  # 2ms grace
        while len(batch) < self.max_batch and time.monotonic() < grace_deadline:
            try:
                req = self.request_queue.get(timeout=0.0005)  # 0.5ms poll
                if req.req_type == REQ_SHUTDOWN:
                    continue
                batch.append(req)
            except queue.Empty:
                continue

        # Drain any remaining without blocking
        while len(batch) < self.max_batch and time.monotonic() < deadline:
            try:
                req = self.request_queue.get_nowait()
                if req.req_type == REQ_SHUTDOWN:
                    continue
                batch.append(req)
            except queue.Empty:
                break

        self._total_requests += len(batch)
        self._total_batches += 1
        return batch

    def _batch_combat(self, reqs: list[InferenceRequest]) -> None:
        """Batch combat inference and dispatch results."""
        B = len(reqs)
        try:
            sf_batch = self._stack_features([r.state_features for r in reqs])
            af_batch = self._stack_features([r.action_features for r in reqs])

            with torch.no_grad():
                logits, values = self.combat_net(sf_batch, af_batch)

            logits_np = logits.cpu().numpy()
            values_np = values.cpu().numpy()

            for i, req in enumerate(reqs):
                result = CombatResult(
                    logits=logits_np[i],
                    value=float(values_np[i]),
                )
                self.result_queues[req.worker_id].put(result)

        except Exception as e:
            logger.error("Combat batch inference failed: %s", e)
            for req in reqs:
                self.result_queues[req.worker_id].put(
                    CombatResult(logits=np.zeros(400, dtype=np.float32), value=0.0))

    def _batch_ppo(self, reqs: list[InferenceRequest]) -> None:
        """Batch PPO inference and dispatch results."""
        B = len(reqs)
        try:
            sf_batch = self._stack_features([r.state_features for r in reqs])
            af_batch = self._stack_features([r.action_features for r in reqs])

            with torch.no_grad():
                action_idx, log_prob, entropy, values = self.ppo_net.get_action_and_value(
                    sf_batch, af_batch)

            action_np = action_idx.cpu().numpy()
            logprob_np = log_prob.cpu().numpy()
            entropy_np = entropy.cpu().numpy()
            values_np = values.cpu().numpy()

            for i, req in enumerate(reqs):
                result = PPOResult(
                    action_idx=int(action_np[i]),
                    log_prob=float(logprob_np[i]),
                    entropy=float(entropy_np[i]),
                    value=float(values_np[i]),
                )
                self.result_queues[req.worker_id].put(result)

        except Exception as e:
            logger.error("PPO batch inference failed: %s", e)
            for req in reqs:
                self.result_queues[req.worker_id].put(
                    PPOResult(action_idx=0, log_prob=-1.0, entropy=0.0, value=0.0))

    def _stack_features(self, feature_dicts: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        """Stack list of numpy feature dicts into batched tensors on device."""
        keys = feature_dicts[0].keys()
        result = {}
        for k in keys:
            arrays = [fd[k] for fd in feature_dicts]
            stacked = np.stack(arrays, axis=0)  # (B, ...)
            t = torch.from_numpy(stacked)
            if stacked.dtype in (np.int64, np.int32):
                t = t.long()
            elif stacked.dtype == bool:
                t = t.bool()
            else:
                t = t.float()
            result[k] = t.to(self.device)
        return result


class InferenceClient:
    """Client-side proxy for worker processes to request inference.

    Sends numpy feature dicts to the server via mp.Queue and blocks
    until the result arrives.
    """

    def __init__(self, worker_id: int, request_queue: mp.Queue, result_queue: mp.Queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self._req_counter = 0

    def combat_inference(
        self,
        state_features: dict[str, np.ndarray],
        action_features: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """Request combat NN inference. Returns (logits, value)."""
        self._req_counter += 1
        req = InferenceRequest(
            worker_id=self.worker_id,
            req_type=REQ_COMBAT,
            state_features=state_features,
            action_features=action_features,
            request_id=self._req_counter,
        )
        self.request_queue.put(req)
        result: CombatResult = self.result_queue.get(timeout=30.0)
        return result.logits, result.value

    def ppo_inference(
        self,
        state_features: dict[str, np.ndarray],
        action_features: dict[str, np.ndarray],
    ) -> tuple[int, float, float, float]:
        """Request PPO NN inference. Returns (action_idx, log_prob, entropy, value)."""
        self._req_counter += 1
        req = InferenceRequest(
            worker_id=self.worker_id,
            req_type=REQ_PPO,
            state_features=state_features,
            action_features=action_features,
            request_id=self._req_counter,
        )
        self.request_queue.put(req)
        result: PPOResult = self.result_queue.get(timeout=30.0)
        return result.action_idx, result.log_prob, result.entropy, result.value
