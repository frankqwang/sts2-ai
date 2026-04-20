from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from networkV2.s1_schema.token_banks import (
    CombatBanks,
    SharedWorldBanks,
    Token,
    TokenBank,
    UnifiedTokenBanks,
)
from networkV2.s6_training.rollout_async_engine import (
    DOMAIN_TO_CODE,
    InferenceCoordinator,
    InferenceRequestMeta,
    RolloutEngineConfig,
    RolloutTaskEnvelope,
    PersistentActorRuntime,
    create_shared_rollout_slot,
    read_batched_banks_from_slots,
    write_unified_banks_to_slot,
)


def _make_banks(decision_domain: str = "combat") -> UnifiedTokenBanks:
    shared = SharedWorldBanks(
        build_bank=TokenBank(
            bank_name="build",
            tokens=[Token(numeric=[1.0, 2.0], token_type="deck_card")],
        ),
    )
    combat = None
    if decision_domain == "combat":
        combat = CombatBanks(
            board_bank=TokenBank(
                bank_name="board",
                tokens=[Token(numeric=[3.0], token_type="player")],
            ),
        )
    action_bank = TokenBank(
        bank_name="action",
        tokens=[
            Token(numeric=[0.1], token_type="action_candidate"),
            Token(numeric=[0.2], token_type="action_candidate"),
        ],
    )
    return UnifiedTokenBanks(
        shared=shared,
        combat=combat,
        action_bank=action_bank,
        decision_domain=decision_domain,
    )


class _FakeUnifiedNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *, batched_banks=None, decision_domain="combat", encounter_idx=None):
        batch = int(next(iter(batched_banks.values())).numeric.shape[0])
        logits = torch.zeros(batch, 2, dtype=torch.float32)
        mask = torch.ones(batch, 2, dtype=torch.bool)
        if batch >= 1:
            logits[0] = torch.tensor([0.1, 1.5])
        if batch >= 2:
            logits[1] = torch.tensor([2.0, 0.2])
        values = SimpleNamespace(fight_win=torch.tensor([0.7, 0.3][:batch], dtype=torch.float32))
        return SimpleNamespace(
            logits=logits,
            action_mask=mask,
            values=values,
            run_eval=None,
        )


class _DeadProcess:
    def is_alive(self) -> bool:
        return False


class _AliveProcess:
    def __init__(self):
        self.terminated = False

    def is_alive(self) -> bool:
        return not self.terminated

    def terminate(self) -> None:
        self.terminated = True

    def join(self, timeout: float | None = None) -> None:
        return None


def test_shared_slot_roundtrip_preserves_domain_and_masks():
    slot = create_shared_rollout_slot(max_numeric_dim=8)
    banks = _make_banks("combat")
    write_unified_banks_to_slot(
        slot,
        banks,
        encounter_idx=7,
        legal_len=2,
        greedy=True,
        request_id=99,
    )
    req = InferenceRequestMeta(
        actor_id=0,
        task_id=11,
        request_id=99,
        decision_domain="combat",
        legal_len=2,
        greedy=True,
        submitted_at_ns=time.perf_counter_ns(),
    )
    batched, encounter_idx = read_batched_banks_from_slots([req], [slot])

    assert encounter_idx.tolist() == [7]
    assert slot.decision_domain_code[0].item() == DOMAIN_TO_CODE["combat"]
    assert slot.legal_len[0].item() == 2
    assert bool(batched["build"].mask[0, 0].item()) is True
    assert batched["action"].mask[0, :2].tolist() == [True, True]


def test_inference_coordinator_processes_batched_requests():
    ctx = torch.multiprocessing.get_context("spawn")
    slots = [create_shared_rollout_slot(max_numeric_dim=8) for _ in range(2)]
    request_queue = ctx.Queue()
    reply_queues = [ctx.Queue(), ctx.Queue()]
    config = RolloutEngineConfig(
        rollout_num_actors=2,
        rollout_infer_batch_size=4,
        rollout_graph_enabled=False,
        max_numeric_dim=8,
    )
    net = _FakeUnifiedNet()
    coordinator = InferenceCoordinator(
        net=net,
        config=config,
        request_queue=request_queue,
        reply_queues=reply_queues,
        slots=slots,
        model_kind="unified",
    )

    for actor_id in range(2):
        write_unified_banks_to_slot(
            slots[actor_id],
            _make_banks("combat"),
            encounter_idx=actor_id + 1,
            legal_len=2,
            greedy=True,
            request_id=actor_id + 1,
        )
    batch = [
        InferenceRequestMeta(
            actor_id=0,
            task_id=101,
            request_id=1,
            decision_domain="combat",
            legal_len=2,
            greedy=True,
            submitted_at_ns=time.perf_counter_ns() - 1_000_000,
        ),
        InferenceRequestMeta(
            actor_id=1,
            task_id=102,
            request_id=2,
            decision_domain="combat",
            legal_len=2,
            greedy=True,
            submitted_at_ns=time.perf_counter_ns() - 1_000_000,
        ),
    ]

    coordinator._process_batch(batch)
    reply0 = reply_queues[0].get(timeout=1.0)
    reply1 = reply_queues[1].get(timeout=1.0)
    stats = coordinator.stats()

    assert reply0.chosen_action_index == 1
    assert reply1.chosen_action_index == 0
    assert reply0.task_id == 101
    assert reply1.task_id == 102
    assert stats["requests"] == 2
    assert stats["batches"] == 1
    assert stats["eager_fallbacks"] == 1
    assert stats["batch_hist"][2] == 1


def test_inference_coordinator_reset_stats_clears_warmup_counters():
    ctx = torch.multiprocessing.get_context("spawn")
    slots = [create_shared_rollout_slot(max_numeric_dim=8) for _ in range(1)]
    request_queue = ctx.Queue()
    reply_queues = [ctx.Queue()]
    config = RolloutEngineConfig(
        rollout_num_actors=1,
        rollout_infer_batch_size=2,
        rollout_graph_enabled=False,
        max_numeric_dim=8,
    )
    coordinator = InferenceCoordinator(
        net=_FakeUnifiedNet(),
        config=config,
        request_queue=request_queue,
        reply_queues=reply_queues,
        slots=slots,
        model_kind="unified",
    )

    write_unified_banks_to_slot(
        slots[0],
        _make_banks("combat"),
        encounter_idx=1,
        legal_len=2,
        greedy=True,
        request_id=1,
    )
    coordinator._process_batch([
        InferenceRequestMeta(
            actor_id=0,
            task_id=201,
            request_id=1,
            decision_domain="combat",
            legal_len=2,
            greedy=True,
            submitted_at_ns=time.perf_counter_ns() - 1_000_000,
        ),
    ])

    assert coordinator.stats()["requests"] == 1
    coordinator.reset_stats()
    reset = coordinator.stats()

    assert reset["requests"] == 0
    assert reset["batches"] == 0
    assert reset["graph_hits"] == 0
    assert reset["eager_fallbacks"] == 0
    assert reset["batch_hist"] == {}


def test_restart_dead_actor_requeues_inflight_task():
    runtime = PersistentActorRuntime(
        actor_entry=lambda *args, **kwargs: None,
        actor_init_payload={},
        net=_FakeUnifiedNet(),
        config=RolloutEngineConfig(
            rollout_num_actors=1,
            rollout_graph_enabled=False,
            max_numeric_dim=8,
        ),
        model_kind="unified",
    )
    respawned: list[int] = []
    runtime.processes[0] = _DeadProcess()
    runtime._inflight[11] = RolloutTaskEnvelope(task_id=11, payload={"seed": "x"})
    runtime.slots[0].active_task_id[0] = 11
    runtime._spawn_actor = lambda actor_id: respawned.append(actor_id)  # type: ignore[method-assign]

    runtime._restart_dead_actors()
    requeued = runtime.task_queue.get(timeout=1.0)

    assert requeued.task_id == 11
    assert runtime.restart_counts[0] == 1
    assert respawned == [0]
    assert runtime.slots[0].active_task_id[0].item() == -1


def test_restart_stuck_actor_terminates_and_requeues():
    runtime = PersistentActorRuntime(
        actor_entry=lambda *args, **kwargs: None,
        actor_init_payload={},
        net=_FakeUnifiedNet(),
        config=RolloutEngineConfig(
            rollout_num_actors=1,
            rollout_graph_enabled=False,
            max_numeric_dim=8,
            actor_reply_timeout_s=0.01,
        ),
        model_kind="unified",
    )
    respawned: list[int] = []
    proc = _AliveProcess()
    runtime.processes[0] = proc
    runtime._inflight[21] = RolloutTaskEnvelope(task_id=21, payload={"seed": "slow"})
    runtime._inflight_started_ns[21] = time.perf_counter_ns() - 50_000_000
    runtime.slots[0].active_task_id[0] = 21
    runtime._spawn_actor = lambda actor_id: respawned.append(actor_id)  # type: ignore[method-assign]

    runtime._restart_stuck_actors()
    requeued = runtime.task_queue.get(timeout=1.0)

    assert proc.terminated is True
    assert requeued.task_id == 21
    assert runtime.restart_counts[0] == 1
    assert respawned == [0]
