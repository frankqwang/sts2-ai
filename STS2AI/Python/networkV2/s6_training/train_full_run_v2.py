"""Full Run V2 训练脚本。

用 UnifiedNet 跑完整一局（地图+选牌+商店+战斗+事件），
收集所有 domain 的 rollout 数据，统一 PPO 更新。

用法:
  python -m networkV2.s6_training.train_full_run_v2 \
    --d-model 384 --n-heads 8 --episodes-per-iter 10 --max-iterations 500
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from networkV2.s3_state_tracker.combat_env_wrapper import CombatStateTracker
from networkV2.s4_compiler.feature_compiler import CombatFeatureCompiler
from networkV2.s5_net.unified_net import UnifiedNet, UnifiedNetOutput
from networkV2.s5_net.network_config import from_preset, NetworkConfig
from networkV2.s6_training.batch import TrainingSample
from networkV2.s6_training.ppo import UnifiedPPOTrainer, PPOConfig
from networkV2.s7_diagnostics.rollout_dumper import RolloutDumper

from core.rl_reward_shaping import combat_step_reward, combat_local_tactical_reward
from env.full_run_env import BinaryBackedFullRunClient

logger = logging.getLogger(__name__)

COMBAT_TYPES = {"monster", "elite", "boss", "hand_select", "card_select"}

# 非战斗 reward: 活着走一步就是正向，后续可以接更精细的 shaping
NONCOMBAT_STEP_REWARD = 0.01
# 终局 reward
WIN_REWARD = 1.0
LOSE_REWARD = -1.0


# ---------------------------------------------------------------------------
# 单局 rollout
# ---------------------------------------------------------------------------

def run_full_episode(
    client: BinaryBackedFullRunClient,
    net: UnifiedNet,
    compiler: CombatFeatureCompiler,
    *,
    seed: str = "",
    max_steps: int = 800,
    greedy: bool = False,
) -> tuple[list[TrainingSample], dict[str, Any]]:
    """跑完整一局，收集所有 step 的 TrainingSample。"""

    state = client.reset(character_id="IRONCLAD", seed=seed)
    tracker = CombatStateTracker()
    tracker.on_run_start()
    in_combat = False

    samples: list[TrainingSample] = []
    combat_samples_start = 0  # 当前战斗的 sample 起始 index
    prev_state = state
    hp_at_combat_start = 0
    step_count = 0
    combat_count = 0

    for _ in range(max_steps):
        st = str(state.get("state_type", "")).lower()
        terminal = state.get("terminal", False)
        outcome = state.get("run_outcome")

        if terminal or outcome:
            # 终局 reward
            won = str(outcome or "").lower() == "victory"
            final_reward = WIN_REWARD if won else LOSE_REWARD
            # 给最后一个 sample 加终局 reward + 显式硬标签
            if samples:
                samples[-1].reward += final_reward
                samples[-1].fight_win_target = 1.0 if won else 0.0  # >= 0 → 显式监督
            if in_combat:
                tracker.on_combat_end(state)
            break

        legal = state.get("legal_actions", [])
        if not legal:
            break

        # ---- 编译 + 推理 ----
        is_combat = st in COMBAT_TYPES
        if is_combat:
            if not in_combat:
                eid = str(state.get("encounter_id", "")).lower()
                rt = "boss" if "boss" in st else "elite" if "elite" in st else "monster"
                tracker.on_combat_start(state, eid, rt)
                in_combat = True
                combat_count += 1
                hp_at_combat_start = int(
                    (state.get("player") or {}).get("hp", 0) or 0)
                combat_samples_start = len(samples)
            # 注意：不在此处调 tracker.on_step(state) —— 那样 prev_state 就丢了，
            # 动作效果差分无法计算。改到 act 之后调 on_step(next_state, chosen, prev_state=state)。

        elif in_combat:
            # 从战斗切出（战斗结束 → 奖励/选牌等）
            tracker.on_combat_end(state)
            in_combat = False

        # 非战斗房间登记到 RunBuildMemory 的 room/event 历史（combat 由 on_combat_start 登记）
        if not is_combat and st in ("shop", "rest_site", "map", "event", "card_reward", "combat_rewards"):
            room_kind = {"rest_site": "rest", "combat_rewards": "card_reward"}.get(st, st)
            rbm = tracker.run_build_memory
            if not rbm.room_type_history or rbm.room_type_history[-1] != room_kind:
                rbm.register_room(room_kind)
                if st == "event":
                    eid = str(state.get("event_id", state.get("encounter_id", "")) or "")
                    rbm.register_event(eid)

        banks = compiler.compile(
            state, legal,
            combat_memory=tracker.combat_memory if is_combat else None,
            turn_prefix=tracker.turn_prefix if is_combat else None,
            run_build_memory=tracker.run_build_memory,
            encounter_id=tracker.encounter_id if is_combat else "",
            room_type=tracker.room_type if is_combat else "monster",
        )

        with torch.no_grad():
            out = net(banks=banks)
        logits = out.logits[0, :len(legal)]
        mask = out.action_mask[0, :len(legal)]
        logits = torch.nan_to_num(logits.masked_fill(~mask, float("-inf")), nan=0.0)
        dist = Categorical(logits=logits)
        idx = logits.argmax().item() if greedy else dist.sample().item()
        lp = dist.log_prob(torch.tensor(idx, device=logits.device)).item()

        # Value 估计
        if is_combat and out.values is not None:
            value = out.values.fight_win.item()
        elif out.run_eval is not None:
            value = out.run_eval.run_win_prob.item()
        else:
            value = 0.5

        chosen = legal[idx]

        # ---- Step ----
        # act 失败处理策略：
        #   1) 不把这个无效动作写入 samples（避免污染 PPO 训练）
        #   2) 尝试 get_state 继续 rollout（可能拿到服务端恢复后的状态）
        #   3) get_state 也失败 → break
        act_succeeded = True
        try:
            next_state = client.act(chosen)
        except Exception as e:
            act_succeeded = False
            logger.warning(f"act failed ({chosen.get('action','?')}): {e}")
            try:
                next_state = client.get_state()
            except Exception:
                break

        if not act_succeeded:
            # 跳过本步：不加 sample，让下次循环用当前 state 重新决策
            # 防止死循环：累计失败多次就 break
            act_fail_count = getattr(run_full_episode, "_afc", 0) + 1
            run_full_episode._afc = act_fail_count
            if act_fail_count >= 5:
                logger.warning("act_failed 连续 5 次，放弃 episode")
                run_full_episode._afc = 0
                break
            state = next_state
            legal = state.get("legal_actions", []) or []
            continue
        # 成功时重置失败计数
        if hasattr(run_full_episode, "_afc"):
            run_full_episode._afc = 0

        # ---- tracker.on_step：必须在 act 之后调用，才能做 prev/next 差分算效果量 ----
        if is_combat:
            tracker.on_step(next_state, chosen, prev_state=state)

        # ---- Reward ----
        next_outcome = next_state.get("run_outcome")
        next_terminal = next_state.get("terminal", False)

        if is_combat:
            # 战斗 step reward
            combat_won = None
            if next_terminal or next_outcome:
                combat_won = str(next_outcome or "").lower() == "victory"
            elif str(next_state.get("state_type", "")).lower() not in COMBAT_TYPES:
                combat_won = True  # 从战斗切出 = 战斗胜利
            reward = combat_step_reward(
                prev_state, next_state,
                combat_won=combat_won,
                hp_at_combat_start=hp_at_combat_start,
            )
            reward += combat_local_tactical_reward(state, chosen, legal)
        else:
            reward = NONCOMBAT_STEP_REWARD

        # ---- HP / survival targets ----
        player = (next_state.get("player") or
                  (next_state.get("battle") or {}).get("player") or {})
        cur_hp = int(player.get("hp") or player.get("current_hp") or 0)
        max_hp = max(int(player.get("max_hp") or 1), 1)

        sw = 1.0
        if is_combat:
            sw = {"boss": 1.5, "elite": 1.2}.get(tracker.room_type, 1.0)

        # ---- 辅助 head targets（启发式，小权重监督，消除饥饿 head）----
        cm = tracker.combat_memory
        rbm = tracker.run_build_memory
        hp_ratio = cur_hp / max_hp
        # 敌方行为切换频率（当前战斗，归一化到 [0,1]）
        transition_risk_t = min(cm.transition_count / max(cm.turn_index, 1), 1.0) if is_combat else 0.0
        # 资源保留度：HP 健康 × 药水余量
        resource_retention_t = max(0.0, min(1.0, hp_ratio * (1.0 - min(rbm.potions_used_total / 10.0, 1.0))))
        # RunEvaluator 辅助 target
        boss_readiness_t = max(0.0, min(1.0, (rbm.frontload + rbm.block + rbm.scaling) * hp_ratio))
        resource_health_t = max(0.0, min(1.0, hp_ratio * (1.0 - rbm.resource_priority)))
        deck_quality_t = max(-1.0, min(1.0, rbm.consistency - rbm.curse_density))

        samples.append(TrainingSample(
            banks=banks,
            action_index=idx,
            old_log_prob=lp,
            reward=reward,
            advantage=0.0,
            value_target=0.0,
            value_estimate=value,          # GAE bootstrap，非监督
            fight_win_target=-1.0,          # 哨值：loss 用 returns；终局会被 0/1 覆盖
            hp_loss_target=float(max(hp_at_combat_start - cur_hp, 0)) if is_combat else 0.0,
            survival_target=hp_ratio,
            # leaf_target 在 _compute_gae 后用 value_target 映射到 [-1,1]
            leaf_target=0.0,
            transition_risk_target=transition_risk_t,
            resource_retention_target=resource_retention_t,
            boss_readiness_target=boss_readiness_t,
            resource_health_target=resource_health_t,
            deck_quality_target=deck_quality_t,
            sample_weight=sw,
            encounter_id=tracker.encounter_id if is_combat else "",
            room_type=tracker.room_type if is_combat else st,
        ))

        step_count += 1
        prev_state = next_state
        state = next_state

    # GAE
    _compute_gae(samples)

    info = {
        "steps": step_count,
        "combats": combat_count,
        "outcome": str(state.get("run_outcome", "unknown")),
        "act": (state.get("run") or {}).get("act", 0),
        "floor": (state.get("run") or {}).get("floor", 0),
        "final_hp": int((state.get("player") or {}).get("hp", 0) or 0),
        "max_hp": int((state.get("player") or {}).get("max_hp", 0) or 0),
    }
    return samples, info


def _compute_gae(
    samples: list[TrainingSample], gamma: float = 0.99, lam: float = 0.95,
) -> None:
    """GAE。Bootstrap 用 rollout 时网络 value_estimate；终局硬标签（fight_win_target>=0）优先。"""
    n = len(samples)
    if n == 0:
        return

    def _bootstrap(sample: TrainingSample) -> float:
        if sample.fight_win_target >= 0.0:
            return float(sample.fight_win_target)
        return float(sample.value_estimate)

    adv = [0.0] * n
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = _bootstrap(samples[t + 1]) if t < n - 1 else 0.0
        cur_val = _bootstrap(samples[t])
        delta = samples[t].reward + gamma * next_val - cur_val
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    for t in range(n):
        cur_val = _bootstrap(samples[t])
        samples[t].advantage = adv[t]
        samples[t].value_target = max(0.0, min(1.0, adv[t] + cur_val))
        # leaf_score 是 tanh 输出 ∈ [-1,1]，用 2*value_target - 1 映射
        samples[t].leaf_target = 2.0 * samples[t].value_target - 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_client(port: int) -> BinaryBackedFullRunClient:
    try:
        from constants import REPO_ROOT, SIM_HOST_EXE
        repo, dll = str(REPO_ROOT), str(SIM_HOST_EXE)
    except ImportError:
        repo, dll = "", ""
    client = BinaryBackedFullRunClient(
        port=port, protocol="bin", auto_launch=True,
        repo_root=repo, dll_path=dll)
    client.connect_timeout_s = 30
    return client


class SimClientPool:
    """预热的 simulator client 池。

    训练开始时一次性起 N 个 sim 并预热（dummy reset 触发 JIT），
    整个训练过程复用这些 client，避免每轮都冷启动。

    典型启动时间：
      - 不池化：每轮重新起 N 个 sim ≈ 60s/轮
      - 池化 + 预热：首次 ~30s，后续轮 ~0s

    使用：
        pool = SimClientPool(base_port=15527, size=8)
        pool.warmup()  # 预热：启动所有 sim 并触发 JIT

        # worker 拿一个 client 用（基于 port 固定分配，单线程专属）
        client = pool.get(worker_id=0)
        ...使用...

        pool.close_all()  # 训练结束时统一关闭
    """

    def __init__(self, base_port: int = 15527, size: int = 8):
        self.base_port = base_port
        self.size = size
        self.clients: list[BinaryBackedFullRunClient] = []

    def warmup(self, character_id: str = "IRONCLAD") -> None:
        """启动所有 sim 并发送 dummy reset 触发 JIT 编译。"""
        import threading as _th

        logger.info(f"Warming up {self.size} simulators (base_port={self.base_port})...")
        t0 = time.time()

        # 并发启动所有 client（每个会 auto_launch 自己的 sim 进程）
        self.clients = [_make_client(self.base_port + i) for i in range(self.size)]

        def _warmup_one(idx: int) -> None:
            c = self.clients[idx]
            try:
                # 触发 connect + JIT（reset 会跑到第一个决策点）
                c.reset(character_id=character_id, seed=f"warmup-{idx}")
            except Exception as e:
                logger.warning(f"  warmup sim[{idx}] failed: {e}")

        threads = [_th.Thread(target=_warmup_one, args=(i,)) for i in range(self.size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        logger.info(f"Warmup done in {time.time() - t0:.1f}s")

    def get(self, worker_id: int) -> BinaryBackedFullRunClient:
        """按 worker_id 固定返回一个 client（不同 worker_id 必须在不同线程）。"""
        return self.clients[worker_id % len(self.clients)]

    def close_all(self) -> None:
        for c in self.clients:
            try:
                c.close()
            except Exception:
                pass
        self.clients.clear()


def _collect_worker(
    worker_id: int,
    pool: SimClientPool,
    net: UnifiedNet,
    seeds: list[str],
    max_steps: int,
    result_q: "queue.Queue",
) -> None:
    """单个收集 worker（在独立线程中运行）。

    从 pool 拿一个 client 复用（不 close），避免每轮冷启动。
    """
    compiler = CombatFeatureCompiler()
    client = pool.get(worker_id)
    samples_out: list[TrainingSample] = []
    infos: list[dict] = []
    for seed in seeds:
        try:
            samples, info = run_full_episode(client, net, compiler, seed=seed, max_steps=max_steps)
            samples_out.extend(samples)
            infos.append(info)
        except Exception as e:
            infos.append({"outcome": "error", "floor": 0, "steps": 0, "error": str(e)})
    # 注意：不 close client，留给 pool 统一管理
    result_q.put({"samples": samples_out, "infos": infos})


def train_full_run(args: argparse.Namespace) -> None:
    import queue
    import threading

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 优先用 --preset（slim/full/tiny），否则用散参数（旧方式）
    if args.preset:
        cfg = from_preset(args.preset)
        # 允许散参数覆盖预设（只有显式指定才生效）
        if args.d_model > 0: cfg.d_model = args.d_model
        if args.n_heads > 0: cfg.n_heads = args.n_heads
        net = UnifiedNet(config=cfg).to(device)
        logger.info(f"Using preset '{args.preset}': {cfg}")
    else:
        net = UnifiedNet(
            d_model=args.d_model or 384, n_heads=args.n_heads or 8,
            n_build_slots=args.n_build_slots,
            max_numeric_dim=args.max_numeric_dim,
            dropout=args.dropout,
        ).to(device)
    params = sum(p.numel() for p in net.parameters())
    logger.info(f"UnifiedNet: {params:,} params ({params/1e6:.1f}M) on {device}")

    if args.checkpoint and Path(args.checkpoint).exists():
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    trainer = UnifiedPPOTrainer(net, PPOConfig(
        lr=args.lr, ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        max_numeric_dim=args.max_numeric_dim,
        value_warmup_iters=args.value_warmup_iters,
        target_kl=args.target_kl,
    ))

    n_workers = args.num_workers
    base_port = args.port
    rng = random.Random(args.seed)
    total_wins = 0
    total_runs = 0

    # ---- 一次性启动并预热 simulator 池（复用整个训练过程）----
    pool = SimClientPool(base_port=base_port, size=n_workers)
    pool.warmup()

    # ---- Rollout + metrics dumper (diagnostic) ----
    dumper: RolloutDumper | None = None
    if args.dump_dir:
        dumper = RolloutDumper(args.dump_dir)
        dumper.write_meta({
            "preset": args.preset,
            "d_model": args.d_model, "n_heads": args.n_heads,
            "lr": args.lr, "clip_eps_loss_default": 0.15,
            "ppo_epochs": args.ppo_epochs, "mini_batch_size": args.mini_batch_size,
            "value_warmup_iters": args.value_warmup_iters,
            "target_kl": args.target_kl,
            "num_workers": n_workers,
            "episodes_per_iter": args.episodes_per_iter,
            "net_params": sum(p.numel() for p in net.parameters()),
        })
        logger.info(f"Dumping rollout data to: {args.dump_dir}")

    print(f"\nConfig: d_model={args.d_model} n_heads={args.n_heads} lr={args.lr} "
          f"eps/iter={args.episodes_per_iter} workers={n_workers} ppo_epochs={args.ppo_epochs}")
    print()
    print("Iter | Eps | Steps | W/L | Cum%  | AvgFlr | Losses                                            | Time")
    print("-----|-----|-------|-----|-------|--------|---------------------------------------------------|------")

    try:
      for iteration in range(1, args.max_iterations + 1):
        t0 = time.time()

        # 分配 seeds 给各 worker
        eps_total = args.episodes_per_iter
        seeds_per_worker: list[list[str]] = [[] for _ in range(n_workers)]
        for i in range(eps_total):
            seed = f"fr-{iteration}-{i}-{rng.getrandbits(32):08x}"
            seeds_per_worker[i % n_workers].append(seed)

        # 并发收集：每个 worker 复用 pool 里的固定 client
        net.eval()
        result_q: queue.Queue = queue.Queue()
        threads = []
        for w_idx in range(n_workers):
            if not seeds_per_worker[w_idx]:
                continue
            t = threading.Thread(
                target=_collect_worker,
                args=(w_idx, pool, net, seeds_per_worker[w_idx], args.max_steps, result_q),
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=300)

        # 汇总
        iter_samples: list[TrainingSample] = []
        iter_episode_infos: list[dict] = []
        w, l = 0, 0
        floors = []
        while not result_q.empty():
            r = result_q.get()
            iter_samples.extend(r["samples"])
            for info in r["infos"]:
                iter_episode_infos.append(info)
                total_runs += 1
                floors.append(info.get("floor", 0))
                if info.get("outcome") == "victory":
                    w += 1
                    total_wins += 1
                elif info.get("outcome") != "error":
                    l += 1

        # PPO update
        net.train()
        metrics = {}
        if len(iter_samples) >= args.min_update_samples:
            metrics = trainer.train_step(iter_samples)

        # ---- Dump diagnostic data ----
        if dumper is not None:
            try:
                dumper.dump_iteration(
                    iteration=iteration,
                    samples=iter_samples,
                    metrics=metrics,
                    episode_infos=iter_episode_infos,
                    extra={
                        "wall_time_s": time.time() - t0,
                        "total_runs": total_runs,
                        "total_wins": total_wins,
                    },
                )
            except Exception as e:
                logger.warning(f"dump failed: {e}")

        elapsed = time.time() - t0
        cum = total_wins / max(total_runs, 1) * 100
        avg_floor = sum(floors) / max(len(floors), 1)
        # 用 .6f 精度：PPO 早期 value 同质化时 loss 可能是 1e-4 ~ 1e-5 级别
        pl = metrics.get("policy_loss", 0)
        vl = metrics.get("value_loss", 0)
        hp = metrics.get("vl_hp_loss", 0)
        kl = metrics.get("approx_kl", 0)
        ep = int(metrics.get("epochs_done", 0))
        wm = "W" if metrics.get("warmup", 0) > 0.5 else " "
        print(f" {iteration:3d}{wm} | {w+l:3d} | {len(iter_samples):5d} | {w}/{l} | {cum:4.1f}% | "
              f"{avg_floor:6.2f} | pl={pl:.5f} vl={vl:.3f} hp={hp:.3f} kl={kl:.4f} ep={ep} | {elapsed:5.1f}s")

        if iteration % args.save_every == 0:
            p = Path(args.output_dir) / f"unified_v2_iter{iteration}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), p)
    finally:
      # 训练结束（或异常退出）时统一关闭所有 sim 进程
      pool.close_all()

    p = Path(args.output_dir) / "unified_v2_final.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), p)
    logger.info(f"Done: {total_wins}W / {total_runs} runs = {total_wins/max(total_runs,1)*100:.1f}%")


def main():
    p = argparse.ArgumentParser(description="Full Run V2 Training")
    # Network preset (推荐)：slim (训练快) / full (能力强) / tiny (调试)
    # 详见 networkV2/s5_net/network_config.py
    p.add_argument("--preset", type=str, default="",
                   help="Network preset: slim / full / tiny. 空字符串 = 用散参数")
    # Network 散参数（旧接口，不推荐直接改，用 --preset 代替）
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-build-slots", type=int, default=8)
    p.add_argument("--max-numeric-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--lr", type=float, default=1e-4)  # 降自 3e-4
    p.add_argument("--ppo-epochs", type=int, default=4)
    # Value warmup: 前 N 轮只训 value head（policy_coef=0），让 value 先分化
    # 避免 PPO 冷启动时 advantages 同质化导致 policy_loss 卡 0
    p.add_argument("--value-warmup-iters", type=int, default=3)
    # KL 早停：一个 epoch 内平均 approx_kl 超阈值就终止剩余 epoch
    # 防止策略更新过大导致 catastrophic forgetting；0 = 禁用
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--mini-batch-size", type=int, default=64)
    p.add_argument("--max-iterations", type=int, default=500)
    p.add_argument("--episodes-per-iter", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--min-update-samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=15527)
    p.add_argument("--num-workers", type=int, default=4)
    # IO
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--output-dir", type=str, default="checkpoints/unified_v2")
    p.add_argument("--save-every", type=int, default=50)
    # 诊断：每 iter 把 samples/metrics/advantages 写到 dump_dir 下
    # 事后用 analyze_rollout.py 分析异常
    p.add_argument("--dump-dir", type=str, default="",
                   help="If set, dump rollout/metrics to this dir for diagnosis")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")
    train_full_run(args)


if __name__ == "__main__":
    main()
