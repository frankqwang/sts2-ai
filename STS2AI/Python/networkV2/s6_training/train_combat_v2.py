"""CombatNet V2 战斗训练脚本。

使用 networkV2 全套流水线：
  s0_bridge (ProtoPipeClient 或 PipeBackedCombatTrainingClient)
  → s3_state_tracker → s4_compiler → s5_net → s6_training PPO

用法:
  python -m networkV2.s6_training.train_combat_v2 \
    --builds ../Assets/builds/combat_sandbox_builds.json \
    --d-model 128 --n-heads 4 --episodes-per-iter 20 --max-iterations 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from networkV2.s3_state_tracker.combat_env_wrapper import CombatStateTracker
from networkV2.s4_compiler.feature_compiler import CombatFeatureCompiler
from networkV2.s5_net.combat_net import CombatNetV2, CombatNetOutput
from networkV2.s6_training.batch import TrainingSample
from networkV2.s6_training.ppo import CombatPPOTrainerV2, PPOConfig

from core.rl_reward_shaping import combat_step_reward, combat_local_tactical_reward
from env.combat_training_env import (
    PipeBackedCombatTrainingClient,
    adapt_combat_snapshot,
    build_combat_legal_actions,
)
from env.run_outcome_vocab import is_victory_outcome, is_failure_outcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome helper
# ---------------------------------------------------------------------------

def detect_combat_outcome(state: dict[str, Any], done: bool) -> bool | None:
    """统一的战斗结果检测。True=胜, False=败, None=未结束。

    P3 修复：改用 env.run_outcome_vocab 的 helper，避免本地字符串字面比较和上游
    normalize（"loss"/"lose"/"dead"/"defeat" 等多种拼写）漂移。
    """
    if not done:
        return None
    raw = state.get("run_outcome")
    if is_victory_outcome(raw):
        return True
    if is_failure_outcome(raw):
        return False
    if state.get("terminal"):
        enemies = (state.get("battle") or {}).get("enemies") or state.get("enemies") or []
        alive = [e for e in enemies if isinstance(e, dict) and e.get("is_alive", True)]
        return len(alive) == 0
    return None


# ---------------------------------------------------------------------------
# Multi-head target 生成
# ---------------------------------------------------------------------------

def compute_step_targets(
    prev_state: dict[str, Any],
    next_state: dict[str, Any],
    combat_won: bool | None,
    hp_at_start: int,
) -> dict[str, float]:
    """从两帧状态计算所有 value head 的 target。

    fight_win_target 只在终局（combat_won 为 True/False）时给出 0/1 硬标签；
    非终局时返回哨值 -1.0，表示"loss 改用 GAE returns 监督"，避免 win head 自蒸馏。
    """
    player = next_state.get("player") or (next_state.get("battle") or {}).get("player") or {}
    current_hp = int(player.get("hp") or player.get("current_hp") or 0)
    max_hp = max(int(player.get("max_hp") or 1), 1)

    if combat_won is True:
        fight_win = 1.0
    elif combat_won is False:
        fight_win = 0.0
    else:
        fight_win = -1.0  # 哨值：loss 里会 fallback 到 returns

    # hp_loss_target: 从战斗开始到现在总共掉了多少血
    hp_loss = max(0, hp_at_start - current_hp)

    # survival_target: 当前 HP ratio 作为粗略生存概率
    survival = current_hp / max_hp

    return {
        "fight_win_target": fight_win,
        "hp_loss_target": float(hp_loss),
        "survival_target": survival,
    }


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

def sample_action_v2(
    net: CombatNetV2,
    banks,
    num_actions: int,
    *,
    greedy: bool = False,
    encounter_id: str = "",
) -> tuple[int, float, float]:
    # rollout 必须传 encounter_idx，否则 conditioning 下 rollout/train 策略不一致
    # （PPO 训练时 collate_training_samples 会给每个 sample 注入 encounter_idx）
    enc_idx = None
    if getattr(net, "enable_encounter_conditioning", False):
        from networkV2.s1_schema.encounter_vocab import encounter_to_index
        device = next(net.parameters()).device
        enc_idx = torch.tensor(
            [encounter_to_index(encounter_id)], dtype=torch.long, device=device,
        )

    with torch.no_grad():
        output: CombatNetOutput = net(banks=banks, encounter_idx=enc_idx)

    logits = output.logits[0, :num_actions]
    mask = output.action_mask[0, :num_actions]
    logits = torch.nan_to_num(logits.masked_fill(~mask, float("-inf")), nan=0.0)

    dist = Categorical(logits=logits)
    action_idx = logits.argmax().item() if greedy else dist.sample().item()
    log_prob = dist.log_prob(torch.tensor(action_idx)).item()
    value = output.values.fight_win.item()
    return action_idx, log_prob, value


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_combat_rollout(
    client: PipeBackedCombatTrainingClient,
    net: CombatNetV2,
    compiler: CombatFeatureCompiler,
    tracker: CombatStateTracker,
    *,
    encounter_id: str,
    room_type: str,
    build_spec: dict[str, Any] | None = None,
    max_steps: int = 200,
    greedy: bool = False,
) -> tuple[list[TrainingSample], dict[str, Any]]:
    state = client.reset(encounter_id=encounter_id, build=build_spec)
    legal_actions = build_combat_legal_actions(state)
    tracker.on_combat_start(state, encounter_id, room_type)
    hp_at_start = int((state.get("player") or {}).get("hp", 0) or 0)

    samples: list[TrainingSample] = []
    prev_state = state
    steps = 0
    combat_won = None

    for _ in range(max_steps):
        if not legal_actions:
            break

        banks = compiler.compile(
            state, legal_actions,
            combat_memory=tracker.combat_memory,
            turn_prefix=tracker.turn_prefix,
            run_build_memory=tracker.run_build_memory,
            encounter_id=encounter_id,
            room_type=room_type,
        )

        action_idx, log_prob, value = sample_action_v2(
            net, banks, len(legal_actions), greedy=greedy,
            encounter_id=encounter_id,
        )
        chosen = legal_actions[action_idx]

        try:
            next_state, _, done, _ = client.step(chosen)
        except Exception as e:
            logger.warning(f"Step error: {e}")
            break

        combat_won = detect_combat_outcome(next_state, done)

        reward = combat_step_reward(
            prev_state, next_state,
            combat_won=combat_won,
            hp_at_combat_start=hp_at_start,
        )
        reward += combat_local_tactical_reward(state, chosen, legal_actions)

        tracker.on_step(next_state, chosen, prev_state=prev_state)

        # 生成真实的 multi-head targets
        targets = compute_step_targets(prev_state, next_state, combat_won, hp_at_start)

        # sample weight: boss/elite 权重更高
        sw = {"boss": 1.25, "elite": 1.1}.get(room_type, 1.0)

        # 辅助 leaf head target（启发式）
        cm = tracker.combat_memory
        transition_risk_t = min(cm.transition_count / max(cm.turn_index, 1), 1.0)
        resource_retention_t = max(0.0, min(1.0, targets["survival_target"]))  # 战斗中近似等价于 HP ratio

        samples.append(TrainingSample(
            banks=banks,
            action_index=action_idx,
            old_log_prob=log_prob,
            reward=reward,
            advantage=0.0,      # GAE 后算
            value_target=0.0,   # GAE 后算
            value_estimate=value,  # 仅用于 GAE bootstrap，不作监督
            fight_win_target=targets["fight_win_target"],
            hp_loss_target=targets["hp_loss_target"],
            survival_target=targets["survival_target"],
            leaf_target=0.0,    # _compute_gae 后用 value_target 映射填上
            transition_risk_target=transition_risk_t,
            resource_retention_target=resource_retention_t,
            sample_weight=sw,
            encounter_id=encounter_id,
            room_type=room_type,
        ))
        steps += 1

        if done:
            tracker.on_combat_end(next_state)
            break

        prev_state = next_state
        state = next_state
        legal_actions = build_combat_legal_actions(state)

    _compute_gae(samples)
    return samples, {"steps": steps, "combat_won": combat_won,
                     "encounter_id": encounter_id, "room_type": room_type}


def _compute_gae(
    samples: list[TrainingSample],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> None:
    """GAE 计算。

    Bootstrap 使用 rollout 时网络的 value_estimate（冻结旧值）；
    若样本带有终局硬标签 (fight_win_target >= 0)，则 bootstrap 用该硬标签，
    以让最后一步正确收敛到 0/1。
    """
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
        samples[t].leaf_target = 2.0 * samples[t].value_target - 1.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_v2(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    net = CombatNetV2(
        d_model=args.d_model, n_heads=args.n_heads,
        n_build_slots=args.n_build_slots,
        max_numeric_dim=args.max_numeric_dim,
        dropout=args.dropout,
        contextualizer_mode=args.contextualizer_mode,
        enable_encounter_conditioning=args.encounter_conditioning,
    ).to(device)
    logger.info(f"CombatNetV2: {sum(p.numel() for p in net.parameters()):,} params")

    if args.checkpoint and Path(args.checkpoint).exists():
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))

    trainer = CombatPPOTrainerV2(net, PPOConfig(
        lr=args.lr, ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        max_numeric_dim=args.max_numeric_dim,
    ))
    compiler = CombatFeatureCompiler()

    # Builds
    builds = []
    if args.builds and Path(args.builds).exists():
        raw = json.loads(Path(args.builds).read_text(encoding="utf-8"))
        builds = raw if isinstance(raw, list) else raw.get("builds", [])
        logger.info(f"Loaded {len(builds)} builds")

    # Client + encounter catalog
    client = PipeBackedCombatTrainingClient(auto_launch=True, connect_timeout_s=30)
    catalog = client.combat_catalog()
    encounter_pools: dict[str, list[str]] = {}
    for e in catalog.get("encounters", []):
        rt = str(e.get("room_type", "monster")).lower()
        eid = str(e.get("encounter_id", "")).upper()
        if eid:
            encounter_pools.setdefault(rt, []).append(eid)
    logger.info(f"Encounter pools: {' | '.join(f'{k}:{len(v)}' for k, v in encounter_pools.items())}")

    room_weights = {
        "monster": args.monster_weight,
        "elite": args.elite_weight,
        "boss": args.boss_weight,
    }
    # 过滤掉没有 encounter 的 room type
    room_weights = {k: v for k, v in room_weights.items() if k in encounter_pools and v > 0}
    room_types = list(room_weights.keys())
    room_w = [room_weights[r] for r in room_types]

    rng = random.Random(args.seed)
    total_w, total_l = 0, 0

    for iteration in range(1, args.max_iterations + 1):
        t0 = time.time()
        iter_samples: list[TrainingSample] = []
        w, l, err = 0, 0, 0

        for _ in range(args.episodes_per_iter):
            build = rng.choice(builds) if builds else {"build": {}, "source": {"character": "IRONCLAD"}}
            room_type = rng.choices(room_types, weights=room_w, k=1)[0]
            encounter_id = rng.choice(encounter_pools[room_type])
            character = str((build.get("source") or {}).get("character", "IRONCLAD")).upper()

            tracker = CombatStateTracker()
            try:
                samples, info = collect_combat_rollout(
                    client, net, compiler, tracker,
                    encounter_id=encounter_id,
                    room_type=room_type,
                    build_spec=build.get("build"),
                    max_steps=args.max_episode_steps,
                )
            except Exception as e:
                logger.warning(f"Episode failed ({encounter_id}): {e}")
                err += 1
                continue

            iter_samples.extend(samples)
            if info.get("combat_won") is True:
                w += 1
            elif info.get("combat_won") is False:
                l += 1

        total_w += w
        total_l += l

        metrics = trainer.train_step(iter_samples) if len(iter_samples) >= args.min_update_samples else {}
        elapsed = time.time() - t0
        wr = w / max(w + l, 1) * 100
        cum = total_w / max(total_w + total_l, 1) * 100

        logger.info(
            f"Iter {iteration:4d} | eps={w+l+err:3d} steps={len(iter_samples):5d} "
            f"W/L/E={w}/{l}/{err} wr={wr:.0f}%/{cum:.0f}% | "
            f"pl={metrics.get('policy_loss',0):.4f} vl={metrics.get('value_loss',0):.4f} "
            f"hp={metrics.get('vl_hp_loss',0):.4f} surv={metrics.get('vl_survival',0):.4f} "
            f"ent={metrics.get('entropy',0):.4f} | {elapsed:.1f}s"
        )

        if iteration % args.save_every == 0:
            p = Path(args.output_dir) / f"combat_v2_iter{iteration}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), p)

    client.close()
    p = Path(args.output_dir) / "combat_v2_final.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), p)
    logger.info(f"Done: {total_w}W {total_l}L = {total_w/max(total_w+total_l,1)*100:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-build-slots", type=int, default=8)
    p.add_argument("--max-numeric-dim", type=int, default=58)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--contextualizer-mode", type=str, default="full",
                   choices=["full", "merged", "minimal"],
                   help="ActionContextualizer 模式，和 UnifiedNet 的 preset 对齐")
    p.add_argument("--encounter-conditioning", action="store_true",
                   help="启用 encounter-conditioning embedding（UNKNOWN=0 保持 neutral）")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--mini-batch-size", type=int, default=32)
    p.add_argument("--max-iterations", type=int, default=1000)
    p.add_argument("--episodes-per-iter", type=int, default=20)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--min-update-samples", type=int, default=64)
    p.add_argument("--monster-weight", type=float, default=6.0)
    p.add_argument("--elite-weight", type=float, default=3.0)
    p.add_argument("--boss-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--builds", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--output-dir", type=str, default="checkpoints/combat_v2")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")
    train_v2(args)


if __name__ == "__main__":
    main()
