"""Full Run V2 训练脚本。

用 UnifiedNet 跑完整一局（地图+选牌+商店+战斗+事件），
收集所有 domain 的 rollout 数据，统一 PPO 更新。

用法:
  python -m networkV2.s6_training.train_full_run_v2 \
    --d-model 384 --n-heads 8 --episodes-per-iter 10 --max-iterations 500
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
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
from networkV2.s6_training.combat_teacher_v1 import (
    OfflineCombatTeacherConfig,
    generate_branch_rollout_dataset,
    load_offline_combat_teacher_entries,
    run_offline_combat_teacher_updates,
    write_critical_step_queue,
)
from networkV2.s6_training.critical_step_pipeline import (
    annotate_critical_steps,
    rebalance_training_samples,
    sort_capture_records,
)
from networkV2.s6_training.head_targets import (
    compute_boss_readiness_target,
    compute_deck_quality_target,
    compute_resource_health_target,
    compute_resource_retention_target,
)
from networkV2.s6_training.ppo import UnifiedPPOTrainer, PPOConfig
from networkV2.s6_training.rollout_async_engine import (
    add_rollout_engine_args,
    build_rollout_engine_config,
    runtime_stats_to_metrics,
)
from networkV2.s6_training.rollout_workers import (
    create_fullrun_runtime,
    open_fullrun_catalog_client,
)
from networkV2.s7_diagnostics.rollout_dumper import RolloutDumper

from networkV2.s6_training.rewards import (
    combat_step_reward, combat_local_tactical_reward,
    shaped_reward, terminal_reward,
)
from env.full_run_env import BinaryBackedFullRunClient
from env.headless_sim_runner import DEFAULT_DLL_PATH, DEFAULT_REPO_ROOT
from env.run_outcome_vocab import (
    is_victory_outcome, is_failure_outcome, normalize_run_outcome,
)

logger = logging.getLogger(__name__)

COMBAT_TYPES = {"monster", "elite", "boss", "hand_select", "card_select"}

# 非战斗 reward 基线：活着走一步给极小正反馈
# 注：接入 shaped_reward 后此常量仅供 entropy_nudge 和历史注释参考
NONCOMBAT_STEP_REWARD = 0.01

# 终局 reward 改用 rl_reward_shaping.terminal_reward（按楼层 scale）；
# 原 WIN_REWARD=+1.0 / LOSE_REWARD=-1.0 常数已弃用。

# Non-combat shaping 鼓励主动 build（long1 监测到 card_reward/shop 100% skip 的
# entropy collapse；选择 = 短期负反馈、不选 = baseline 0 → policy 学到"啥都不做")
# 对"非 skip" 动作给小额额外奖励，抵消短期负反馈。
NONCOMBAT_PICK_BONUS = 0.05      # card_reward 选了一张牌
NONCOMBAT_BUY_BONUS = 0.03       # shop 买了东西
NONCOMBAT_EVENT_CHOOSE_BONUS = 0.02  # event 做出非 skip 选择

# Tier1 P3: episode 内 shaping 累计 cap —— 防 PBRS + milestones 盖过终局 ±1
# 一局典型 400+ step，PBRS 每步 ±0.01~0.05 + milestones 单次 +0.15~0.95 可累积到 ~2~3。
# 不 cap 的话 shaping 主导 advantage → agent 学"最大化 shaping"而非"赢"。
#
# 取值 1.5 的理由：terminal_reward 定义下 act 2+ 死 = 0（原作者设计，进 act 2 视为
# "已脱离失败"）。若 cap 过大（如 2.5），"act 2 死 + 满 shaping" 与 "win + 满 shaping"
# 差值只有 1 (终局部分)，相对 shaping 噪声不稳定。cap=1.5 时：
#   win          = +1 + 1.5  = +2.5
#   act2 死      =  0 + 1.5  = +1.5    # 梯度 1.0 对 win
#   act1 f15 死  = -0.12 + ~1.0 = ~0.9
#   act1 f3 死   = -0.82 + ~0   = -0.82
# 单调梯度保留、终局信号仍有 40% 权重、milestone 最大值（+0.95 boss_entry）仍能显著反映。
EPISODE_SHAPING_CAP = 1.5
SKIP_ACTION_NAMES = {
    "skip", "skip_card_reward", "skip_relic_selection",
    "proceed", "leave_shop", "cancel_selection",
}


def _enemies_total_hp(state: dict) -> int:
    """敌人总 HP（含 block 不算，只看真实血）。用于算 step damage。"""
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    return sum(int(e.get("hp", 0) or 0) for e in enemies if e.get("is_alive", True))


def _player_block_total(state: dict) -> int:
    battle = state.get("battle") or {}
    player = battle.get("player") or state.get("player") or {}
    return int(player.get("block", 0) or 0)


def _backfill_turn_damage(
    samples: list[TrainingSample],
    turn_start_idx: int,
    turn_step_damages: list[float],
) -> None:
    """回合结束时，为 [turn_start_idx, turn_start_idx + len(turn_step_damages)) 范围内
    的 combat samples 写入 turn_damage_target = sum(damages from this step onwards)。
    """
    n = len(turn_step_damages)
    if n == 0:
        return
    # 后缀和：从后往前累加
    suffix = 0.0
    for k in range(n - 1, -1, -1):
        suffix += turn_step_damages[k]
        sample_idx = turn_start_idx + k
        if 0 <= sample_idx < len(samples):
            samples[sample_idx].turn_damage_target = suffix


def _backfill_turn_block(
    samples: list[TrainingSample],
    turn_start_idx: int,
    turn_step_blocks: list[float],
) -> None:
    n = len(turn_step_blocks)
    if n == 0:
        return
    suffix = 0.0
    for k in range(n - 1, -1, -1):
        suffix += turn_step_blocks[k]
        sample_idx = turn_start_idx + k
        if 0 <= sample_idx < len(samples):
            samples[sample_idx].turn_block_target = suffix


def _backfill_combat_outcome(
    samples: list[TrainingSample],
    start_idx: int,
    won: bool,
    end_idx: int | None = None,
) -> None:
    end = len(samples) if end_idx is None else min(end_idx, len(samples))
    value = 1.0 if won else 0.0
    for sample_idx in range(max(start_idx, 0), max(end, 0)):
        samples[sample_idx].fight_win_target = value


def _backfill_combat_future_targets(
    samples: list[TrainingSample],
    trace: list[tuple[int, int]],
    final_hp: int,
    max_hp: int,
    combat_won: bool | None,
) -> None:
    """R1.1 + R1.2: 战斗结束时把 hp_loss_target / survival_target 从"过去累计" 覆盖成
    "未来到战斗末"信号（真正的 future-looking target）。

    - hp_loss_target[t] = max(hp_after_sample_t - hp_at_combat_end, 0)
      即"从本步之后到战斗结束还会再掉多少血"。当前累计掉血是已知量，这里换成未来掉血。
    - survival_target[t] = hp_at_combat_end / max_hp if won else 0.0
      即"战斗末存活状态下的 hp_ratio"。当前 hp_ratio 是已知量，换成战斗末存活度。

    trace: [(sample_idx, hp_after_this_step), ...]
    combat_won: True=胜 / False=败 / None=不确定（rollout 截断等）→ 走 proxy fallback
    """
    if not trace:
        return
    final_hp = max(int(final_hp), 0)
    max_hp = max(int(max_hp), 1)
    for sample_idx, hp_then in trace:
        if not (0 <= sample_idx < len(samples)):
            continue
        # hp_loss_target: 未来累计掉血
        future_loss = max(int(hp_then) - final_hp, 0)
        samples[sample_idx].hp_loss_target = float(future_loss)
        # survival_target: 战斗末存活状态
        if combat_won is True:
            samples[sample_idx].survival_target = float(final_hp) / max_hp
        elif combat_won is False:
            samples[sample_idx].survival_target = 0.0
        # combat_won is None: 保留原 proxy（已在 sample 构造时赋值）


def noncombat_entropy_nudge(chosen: dict, room_type: str) -> float:
    """非战斗"非 skip"动作的小额 bonus，专门对冲 entropy collapse。

    long1 监测到 card_reward/shop 100% skip 的 collapse（选择 = 短期负反馈、
    skip = baseline → policy 学到"啥都不做"）。本函数只返回"非 skip 额外量"；
    baseline（每 step +0.01）和 PBRS / milestones 由 shaped_reward 统一给。

    原 noncombat_step_reward 函数被废弃，其 baseline+bonus 已经被拆成两部分：
      - baseline 和非战斗全部主 reward → shaped_reward (PBRS + milestones)
      - 非 skip 冷启动 bonus → 本函数
    """
    action_name = str(chosen.get("action", "")).strip().lower()
    if action_name in SKIP_ACTION_NAMES:
        return 0.0
    rt = (room_type or "").lower()
    if rt in ("card_reward", "combat_rewards"):
        return NONCOMBAT_PICK_BONUS
    if rt == "shop":
        return NONCOMBAT_BUY_BONUS
    if rt == "event":
        return NONCOMBAT_EVENT_CHOOSE_BONUS
    return 0.0


# ---------------------------------------------------------------------------
# 单局 rollout
# ---------------------------------------------------------------------------

def run_full_episode(
    client: BinaryBackedFullRunClient,
    net: UnifiedNet | None,
    compiler: CombatFeatureCompiler,
    *,
    seed: str = "",
    max_steps: int = 800,
    greedy: bool = False,
    record_trajectory: bool = False,
    inference_client: Any | None = None,
    task_id: int = 0,
    capture_root: str = "",
) -> tuple[list[TrainingSample], dict[str, Any]]:
    """跑完整一局，收集所有 step 的 TrainingSample。

    record_trajectory=True 时额外返回 info["trajectory"]：每步轻量摘要
    （room_type / floor / hp / chosen_action / reward / value），用于诊断 stuck loop。
    """

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
    # P2-2 修复：act 失败计数改局部变量（原先挂在 run_full_episode._afc 函数对象上，
    # 被所有 worker thread 共享无锁，一个 flaky sim 会让无关 rollout 提前 abort）
    act_fail_count = 0
    trajectory: list[dict[str, Any]] = [] if record_trajectory else []
    # 1-turn lookahead 用：记录当前回合每步对敌人造成的伤害，
    # end_turn / 战斗结束时回填 turn_damage_target 给本回合所有 combat samples。
    turn_start_sample_idx = 0
    turn_step_damages: list[float] = []  # 与当前回合 samples 一一对应
    turn_step_blocks: list[float] = []   # 与 turn_step_damages 对齐，记录本步净挡伤
    # R1.1 / R1.2: 战斗全程 trace，记录每个 combat sample 的 (sample_idx, hp_after_step)。
    # 战斗结束时回填 hp_loss_target (未来累计掉血) + survival_target (战斗末 hp_ratio)，
    # 把原先的"过去累计"proxy 替换成真正 future-looking target。
    combat_sample_trace: list[tuple[int, int]] = []
    hp_max_at_combat: int = 0
    capture_records: list[dict[str, Any]] = []

    for _ in range(max_steps):
        st = str(state.get("state_type", "")).lower()
        terminal = state.get("terminal", False)
        outcome = state.get("run_outcome")

        if terminal or outcome:
            # 终局 reward：Tier1 升级——用 rl_reward_shaping.terminal_reward 按楼层 scale
            # (原 WIN_REWARD/LOSE_REWARD 常数 ±1 不区分"死 floor 3"和"死 floor 50"，
            #  近胜近败没有梯度。terminal_reward: 胜=+1；败=-1+floor_bonus，
            #  floor 越高惩罚越轻，near-win 更温和)
            won = is_victory_outcome(outcome)  # P3 修复：统一用 run_outcome_vocab helper
            final_reward = terminal_reward(state, won)
            # R1.1 / R1.2: 如果 run 死在战斗里，也要 backfill 这一战的 future targets
            # combat_won 用整 run 的 won：死在战斗里 → 必败；如果是战斗后期死（一般不可能）
            # 也按整 run 结果给。未来可改更细粒度的 per-combat win detection。
            if in_combat and combat_sample_trace:
                _backfill_turn_damage(samples, turn_start_sample_idx, turn_step_damages)
                _backfill_turn_block(samples, turn_start_sample_idx, turn_step_blocks)
                _backfill_combat_outcome(samples, combat_samples_start, won)
                _final_hp = int((state.get("player") or {}).get("hp", 0) or 0)
                _final_max_hp = int(
                    (state.get("player") or {}).get("max_hp", 0) or 0) or hp_max_at_combat
                _backfill_combat_future_targets(
                    samples, combat_sample_trace,
                    final_hp=_final_hp, max_hp=_final_max_hp,
                    combat_won=won,
                )
                combat_sample_trace = []
                turn_step_damages = []
                turn_step_blocks = []
            # 给最后一个 sample 加终局 reward + 显式硬标签
            if samples:
                # Tier1 P3: 加 terminal reward 前先 cap shaping 累积量级
                shaping_sum = sum(s.reward for s in samples)
                if abs(shaping_sum) > EPISODE_SHAPING_CAP:
                    scale = EPISODE_SHAPING_CAP / abs(shaping_sum)
                    for s in samples:
                        s.reward *= scale
                samples[-1].reward += final_reward
                samples[-1].run_win_target = 1.0 if won else 0.0
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
                # P1-1 修复：sim 不返回 encounter_id 时，fallback 要用 registry 注册的
                # 正式 encounter_id 格式（如 "frog_knight_normal"），否则 feature_compiler.py
                # 查不到 mechanism config → mechanism/modifier bank 永远为空。
                # 原 fallback 拼的是 monster-id 小写（"frog_knight" / "jaw_worm,fungi_beast"），
                # 和 registry 的 "{monster}_{room_type}" 格式对不上。
                rt = "boss" if "boss" in st else "elite" if "elite" in st else "monster"
                eid_explicit = str(state.get("encounter_id", "") or "").lower()
                if not eid_explicit:
                    battle = state.get("battle") or {}
                    enemies = battle.get("enemies") or state.get("enemies") or []
                    monster_ids = [
                        str(e.get("monster_id") or e.get("enemy_id") or e.get("id") or "").strip()
                        for e in enemies if isinstance(e, dict)
                    ]
                    monster_ids = [m for m in monster_ids if m]
                    # 从 registry 反查正式 encounter_id
                    from networkV2.s2_config.mechanism_registry import get_registry
                    eid_explicit = get_registry().find_encounter_id(monster_ids, rt) or ""
                    # 兜底：若 registry 也查不到（encounter 没注册机制），保留
                    # sorted-monster-id 拼接作为诊断字符串（tracker 仍可当唯一标识用）
                    if not eid_explicit:
                        eid_explicit = ",".join(sorted(m.lower() for m in monster_ids))
                eid = eid_explicit
                tracker.on_combat_start(state, eid, rt)
                in_combat = True
                combat_count += 1
                hp_at_combat_start = int(
                    (state.get("player") or {}).get("hp", 0) or 0)
                hp_max_at_combat = int(
                    (state.get("player") or {}).get("max_hp", 0) or 0) or hp_at_combat_start
                combat_samples_start = len(samples)
                combat_sample_trace = []  # R1：新战斗开始，重置 trace
            # 注意：不在此处调 tracker.on_step(state) —— 那样 prev_state 就丢了，
            # 动作效果差分无法计算。改到 act 之后调 on_step(next_state, chosen, prev_state=state)。

        elif in_combat:
            # 从战斗切出（战斗结束 → 奖励/选牌等 → 胜利）
            # R1.1 / R1.2: 在 on_combat_end 之前 backfill hp_loss / survival
            # 用当前 state 的 hp（即战斗末 hp）
            _backfill_turn_damage(samples, turn_start_sample_idx, turn_step_damages)
            _backfill_turn_block(samples, turn_start_sample_idx, turn_step_blocks)
            _backfill_combat_outcome(samples, combat_samples_start, True)
            _final_hp = int((state.get("player") or {}).get("hp", 0) or 0)
            _final_max_hp = int((state.get("player") or {}).get("max_hp", 0) or 0) or hp_max_at_combat
            _backfill_combat_future_targets(
                samples, combat_sample_trace,
                final_hp=_final_hp, max_hp=_final_max_hp,
                combat_won=True,  # 切出战斗通常 = 胜利
            )
            combat_sample_trace = []
            turn_step_damages = []
            turn_step_blocks = []
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
            # P1① 修复：非战斗 step 也要刷 build profile —— shop/card_reward/rest/event
            # 可能改动 gold/deck/relic/potion/act/floor 以及派生的 frontload/scaling/...
            # 之前这些字段只在 on_combat_start 里刷，非战斗决策会读到上一次进战前的旧值。
            tracker.refresh_build_profile(state)

        cur_encounter_id = tracker.encounter_id if is_combat else ""
        banks = compiler.compile(
            state, legal,
            combat_memory=tracker.combat_memory if is_combat else None,
            turn_prefix=tracker.turn_prefix if is_combat else None,
            run_build_memory=tracker.run_build_memory,
            encounter_id=cur_encounter_id,
            room_type=tracker.room_type if is_combat else "monster",
        )

        # Encounter conditioning（方案 A: Conditional Policy）
        from networkV2.s1_schema.encounter_vocab import encounter_to_index
        enc_idx_value = int(encounter_to_index(cur_encounter_id))
        if inference_client is not None:
            reply = inference_client.infer(
                banks,
                encounter_idx=enc_idx_value,
                legal_len=len(legal),
                greedy=greedy,
                task_id=task_id,
            )
            idx = int(reply.chosen_action_index)
            lp = float(reply.old_log_prob)
            value = float(reply.value_estimate)
        else:
            if net is None:
                raise ValueError("run_full_episode requires `net` when inference_client is absent")
            _dev = next(net.parameters()).device
            enc_idx_tensor = torch.tensor(
                [encounter_to_index(cur_encounter_id)], dtype=torch.long, device=_dev,
            )
            with torch.no_grad():
                out = net(banks=banks, encounter_idx=enc_idx_tensor)
            logits = out.logits[0, :len(legal)]
            mask = out.action_mask[0, :len(legal)]
            logits = torch.nan_to_num(logits.masked_fill(~mask, float("-inf")), nan=0.0)
            dist = Categorical(logits=logits)
            idx = logits.argmax().item() if greedy else dist.sample().item()
            lp = dist.log_prob(torch.tensor(idx, device=logits.device)).item()

            # Value 估计
            if is_combat and out.values is not None:
                value = out.values.run_value.item()
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
            # 防止死循环：累计失败多次就 break。act_fail_count 是 episode-local 变量，
            # 线程安全（每个 run_full_episode 调用有自己的栈帧）。
            act_fail_count += 1
            if act_fail_count >= 5:
                logger.warning("act_failed 连续 5 次，放弃 episode")
                break
            state = next_state
            legal = state.get("legal_actions", []) or []
            continue
        # 成功时重置失败计数
        act_fail_count = 0

        # ---- 1-turn lookahead damage 跟踪 ----
        if is_combat:
            step_damage = max(0, _enemies_total_hp(state) - _enemies_total_hp(next_state))
            step_block = max(0, _player_block_total(next_state) - _player_block_total(state))
        else:
            step_damage = 0
            step_block = 0

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
                combat_won = is_victory_outcome(next_outcome)  # P3 修复：统一 helper
            elif str(next_state.get("state_type", "")).lower() not in COMBAT_TYPES:
                combat_won = True  # 从战斗切出 = 战斗胜利
            reward = combat_step_reward(
                prev_state, next_state,
                combat_won=combat_won,
                hp_at_combat_start=hp_at_combat_start,
            )
            reward += combat_local_tactical_reward(state, chosen, legal)
            # Anti-"do nothing" penalty: long2 iter 10 监测到 agent 学会
            # "end_turn × 8 直接死" 局部最优。加 end_turn-with-resources penalty 打破：
            # 还有能量 + 还有可打牌时 end_turn → -0.10 强反馈。
            chosen_action_name = str(chosen.get("action", "")).lower()
            if chosen_action_name in ("end_turn", "end"):
                player = (state.get("player") or {})
                battle = state.get("battle") or {}
                prev_energy = int(battle.get("energy", player.get("energy", 0)) or 0)
                hand = battle.get("hand") or player.get("hand") or []
                playable = sum(1 for c in hand if isinstance(c, dict) and c.get("can_play", False))
                if prev_energy > 0 and playable > 0:
                    reward -= 0.10
        else:
            # Tier1 reward 升级：主信号走 rl_reward_shaping.shaped_reward
            # （PBRS Φ(s) + milestones：楼层推进/boss 门槛/boss_entry_quality/幕通关
            #  + early-damage-potion penalty）；之前的 0.01 baseline + 分桶 bonus
            # 只是占位实现，shaped_reward 才是真正设计好的。
            # entropy collapse 保险：对非 skip 动作叠加小额 nudge（原 long1 修法）
            reward = shaped_reward(
                state, next_state,
                raw_terminal_reward=0.0, done=False,
                action=chosen,
            )
            reward += noncombat_entropy_nudge(chosen, st)

        # ---- HP / survival targets ----
        player = (next_state.get("player") or
                  (next_state.get("battle") or {}).get("player") or {})
        cur_hp = int(player.get("hp") or player.get("current_hp") or 0)
        max_hp = max(int(player.get("max_hp") or 1), 1)

        sw = 1.0
        if is_combat:
            sw = {"boss": 1.5, "elite": 1.2}.get(tracker.room_type, 1.0)

        # ---- 辅助 head targets ----
        # 历史问题：原 target 是 RunBuildMemory 字段的线性组合，等于让网络学输出=输入，
        # 没有监督价值（白训 heads）。现改用 head_targets.py 提供的多信号 + 非线性 +
        # 未来导向（floor_pressure）函数，target 真正含信息量。
        cm = tracker.combat_memory
        rbm = tracker.run_build_memory
        hp_ratio = cur_hp / max_hp
        # 敌方行为切换频率（当前战斗，归一化到 [0,1]）—— 这个本就 OK，保留
        transition_risk_t = min(cm.transition_count / max(cm.turn_index, 1), 1.0) if is_combat else 0.0
        # 4 个 RunEvaluator 辅助 target 全部用 head_targets 计算
        deck_quality_t = compute_deck_quality_target(rbm, hp_ratio)
        boss_readiness_t = compute_boss_readiness_target(rbm, hp_ratio)
        resource_health_t = compute_resource_health_target(rbm, hp_ratio)
        resource_retention_t = compute_resource_retention_target(rbm, hp_ratio)

        samples.append(TrainingSample(
            banks=banks,
            action_index=idx,
            old_log_prob=lp,
            reward=reward,
            advantage=0.0,
            value_target=0.0,
            value_estimate=value,          # GAE bootstrap，非监督
            fight_win_target=-1.0,          # 哨值：loss 用 returns；终局会被 0/1 覆盖
            run_win_target=-1.0,
            # R1.1 proxy 初始值：若战斗正常结束，_backfill_combat_future_targets 会覆盖成
            # "未来累计掉血"；若 rollout 截断（max_steps / 异常）保留 proxy 作 fallback。
            hp_loss_target=float(max(hp_at_combat_start - cur_hp, 0)) if is_combat else 0.0,
            # R1.2 proxy 同上：战斗结束后覆盖成"战斗末 hp_ratio"
            survival_target=hp_ratio,
            # leaf_target 在 _compute_gae 后用 n-step return（R1.3）覆盖
            leaf_target=0.0,
            transition_risk_target=transition_risk_t,
            resource_retention_target=resource_retention_t,
            boss_readiness_target=boss_readiness_t,
            resource_health_target=resource_health_t,
            deck_quality_target=deck_quality_t,
            turn_damage_target=-1.0,  # 待 backfill；非战斗保持 -1（loss 跳过）
            turn_block_target=-1.0,   # 待 backfill；非战斗保持 -1（loss 跳过）
            sample_weight=sw,
            base_sample_weight=sw,
            encounter_id=tracker.encounter_id if is_combat else "",
            room_type=tracker.room_type if is_combat else st,
            floor=int((state.get("run") or {}).get("floor", 0) or 0),
            action_name=str(chosen.get("action") or ""),
        ))

        if capture_root and is_combat:
            sample_index = len(samples) - 1
            snapshot_dir = Path(capture_root)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"{seed}_sample{sample_index:05d}.json"
            try:
                written_snapshot = client.export_state(str(snapshot_path))
                capture_records.append({
                    "seed": seed,
                    "episode_id": seed,
                    "sample_index": int(sample_index),
                    "floor": int((state.get("run") or {}).get("floor", 0) or 0),
                    "encounter_id": tracker.encounter_id if is_combat else "",
                    "room_type": tracker.room_type if is_combat else st,
                    "action_name": str(chosen.get("action") or ""),
                    "legal_actions": copy.deepcopy(legal),
                    "snapshot_path": str(written_snapshot),
                    "root_state": copy.deepcopy(state),
                })
            except Exception as exc:
                logger.warning(f"critical-step snapshot export failed: {exc}")

        # R1: combat sample trace，记录 sample_idx + 本步结束时的 hp
        # 战斗结束（切出 / terminal）时 _backfill_combat_future_targets 用这个 trace
        # 把 hp_loss_target / survival_target 覆盖成 future-looking
        if is_combat:
            combat_sample_trace.append((len(samples) - 1, cur_hp))

        # ---- 1-turn lookahead damage 后处理 ----
        # 战斗中：append step_damage；end_turn 或战斗切出时 backfill 整个回合
        if is_combat:
            turn_step_damages.append(float(step_damage))
            turn_step_blocks.append(float(step_block))
            chosen_action = str(chosen.get("action", "")).lower()
            next_st = str(next_state.get("state_type", "")).lower()
            turn_ended = (
                chosen_action in ("end_turn", "end")
                or next_st not in COMBAT_TYPES  # 战斗结束
                or next_terminal or next_outcome
            )
            if turn_ended:
                _backfill_turn_damage(samples, turn_start_sample_idx, turn_step_damages)
                _backfill_turn_block(samples, turn_start_sample_idx, turn_step_blocks)
                turn_start_sample_idx = len(samples)  # 下一回合从此开始
                turn_step_damages = []
                turn_step_blocks = []
        else:
            # 非战斗 sample：reset turn tracking（防止跨房间状态泄漏）
            turn_start_sample_idx = len(samples)
            turn_step_damages = []
            turn_step_blocks = []

        # Trajectory record（轻量 per-step 快照，仅 record_trajectory=True 时启用）
        if record_trajectory:
            run_info = state.get("run") or {}
            trajectory.append({
                "step": step_count,
                "state_type": st,
                "room_type": tracker.room_type if is_combat else st,
                "floor": int(run_info.get("floor", 0) or 0),
                "act": int(run_info.get("act", 0) or 0),
                "hp": int((state.get("player") or {}).get("hp", 0) or 0),
                "max_hp": int((state.get("player") or {}).get("max_hp", 1) or 1),
                "n_legal": len(legal),
                "chosen_idx": idx,
                "chosen_action": str(chosen.get("action", ""))[:32],
                "value_est": round(value, 4),
                "reward": round(float(reward), 4),
                "act_succeeded": act_succeeded,
            })

        step_count += 1
        prev_state = next_state
        state = next_state

    # GAE
    _compute_gae(samples)

    info = {
        "steps": step_count,
        "combats": combat_count,
        "outcome": normalize_run_outcome(state.get("run_outcome"), default="unknown"),
        "act": (state.get("run") or {}).get("act", 0),
        "floor": (state.get("run") or {}).get("floor", 0),
        "final_hp": int((state.get("player") or {}).get("hp", 0) or 0),
        "max_hp": int((state.get("player") or {}).get("max_hp", 0) or 0),
    }
    if record_trajectory:
        info["trajectory"] = trajectory
    if capture_records:
        info["critical_captures"] = capture_records
    return samples, info


def _compute_gae(
    samples: list[TrainingSample], gamma: float = 0.99, lam: float = 0.95,
    leaf_horizon: int = 3,
) -> None:
    """GAE。Bootstrap 用 rollout 时网络 value_estimate；终局硬标签（run_win_target>=0）优先。

    R1.3: leaf_target 从"2*value_target - 1"（和 fight_win 同源线性重复）改为
    n-step return（horizon=leaf_horizon，默认 3 步）的 tanh 压缩。这样 leaf_score 的
    时间 horizon 比 fight_win（全 GAE return，≈ 整 run 胜率）短、比 tempo（1 步 advantage）长，
    三者在 time horizon 上分化，不再冗余。
    """
    n = len(samples)
    if n == 0:
        return

    def _bootstrap(sample: TrainingSample) -> float:
        if sample.run_win_target >= 0.0:
            return float(sample.run_win_target)
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

        # R1.3: n-step return bootstrap —— 和 fight_win 的全 horizon GAE 拉开
        acc = 0.0
        disc = 1.0
        for k in range(leaf_horizon):
            if t + k >= n:
                break
            acc += disc * float(samples[t + k].reward)
            disc *= gamma
        # 若 horizon 越界，bootstrap 用 cur_val；否则用 value_estimate at t+h
        if t + leaf_horizon < n:
            bootstrap = float(samples[t + leaf_horizon].value_estimate)
        else:
            bootstrap = cur_val
        n_step_return = acc + disc * bootstrap
        # leaf_score 是 tanh 输出 ∈ [-1,1]，用 tanh(scale * n_step_return) 映射。
        # scale=2 让 typical return ~0.5 映射到 tanh(1)≈0.76，占用有效动态范围。
        # 不直接 2*return-1，因为 n-step return 量级不固定（reward 有 shaping + terminal cap）。
        samples[t].leaf_target = math.tanh(2.0 * n_step_return)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_client(port: int, protocol: str = "proto") -> BinaryBackedFullRunClient:
    """创建 sim client。默认 protocol='proto'（2026-04-17 从 bin 迁移）。

    Proto 编码 state 用 protobuf schema，schema 比手写 binary 更稳定；
    性能和 bin 相当，未来扩字段也方便。bin 保留作 legacy（如需回退传 protocol='bin'）。
    """
    try:
        from constants import REPO_ROOT, SIM_HOST_EXE
        repo, dll = str(REPO_ROOT), str(SIM_HOST_EXE)
    except ImportError:
        repo, dll = "", ""
    client = BinaryBackedFullRunClient(
        port=port, protocol=protocol, auto_launch=True,
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
    record_trajectory_every: int = 0,
    greedy: bool = False,
    capture_root: str = "",
) -> None:
    """单个收集 worker（在独立线程中运行）。

    record_trajectory_every: 每 N 局记录一次完整 trajectory；0 = 不记录。
    长 trajectory 占空间，建议每 worker 只记 1-2 局/iter（采样代表）。
    greedy: eval 时传 True（argmax action），训练 rollout 保持 False（sample）。
    """
    compiler = CombatFeatureCompiler()
    client = pool.get(worker_id)
    samples_out: list[TrainingSample] = []
    infos: list[dict] = []
    for ep_i, seed in enumerate(seeds):
        try:
            rec_traj = bool(record_trajectory_every and ep_i % record_trajectory_every == 0)
            sample_offset = len(samples_out)
            samples, info = run_full_episode(
                client, net, compiler, seed=seed,
                max_steps=max_steps,
                record_trajectory=rec_traj,
                greedy=greedy,
                capture_root=capture_root,
            )
            for capture in info.get("critical_captures") or []:
                capture["global_sample_index"] = sample_offset + int(capture.get("sample_index") or 0)
            samples_out.extend(samples)
            infos.append(info)
        except Exception as e:
            infos.append({"outcome": "error", "floor": 0, "steps": 0, "error": str(e)})
    # 注意：不 close client，留给 pool 统一管理
    result_q.put({"samples": samples_out, "infos": infos})


def _compute_fight_win_calibration(
    samples: list[TrainingSample],
    n_bins: int = 10,
) -> dict:
    """对 iter_samples 算 fight_win head 的 calibration。

    把 samples 按 `value_estimate`（rollout 时网络的 fight_win 预测）分 n_bins 个桶，
    每桶统计实际 GAE return（value_target 在 [0,1]，代表"这个 state 的真实赢面"）。
    理想情况：pred=0.3 的样本实际平均 return ≈ 0.3。

    返回 metrics: {cal_bin_i_pred_mid, cal_bin_i_actual_mean, cal_bin_i_count, cal_ece}
    其中 ECE = Expected Calibration Error = Σ (bin_weight × |pred - actual|)。
    ECE 越小，value head 的预测越接近真实胜率。
    """
    if not samples:
        return {}
    # 只对有意义的样本算（value_estimate 在 [0,1]，value_target 已被 GAE 填好）
    preds: list[float] = []
    actuals: list[float] = []
    for s in samples:
        pv = float(s.value_estimate)
        tv = float(s.value_target)
        # value_estimate 是 rollout 时的 fight_win 预测（sigmoid 输出 ∈ [0,1]）
        if 0.0 <= pv <= 1.0 and 0.0 <= tv <= 1.0:
            preds.append(pv)
            actuals.append(tv)
    if not preds:
        return {}
    bin_width = 1.0 / n_bins
    total = len(preds)
    ece = 0.0
    out: dict = {}
    for b in range(n_bins):
        lo = b * bin_width
        hi = (b + 1) * bin_width if b < n_bins - 1 else 1.0 + 1e-6
        bin_samples = [(p, a) for p, a in zip(preds, actuals) if lo <= p < hi]
        if not bin_samples:
            continue
        mean_pred = sum(p for p, _ in bin_samples) / len(bin_samples)
        mean_actual = sum(a for _, a in bin_samples) / len(bin_samples)
        weight = len(bin_samples) / total
        ece += weight * abs(mean_pred - mean_actual)
        out[f"cal_bin{b}_count"] = len(bin_samples)
        out[f"cal_bin{b}_pred_mean"] = round(mean_pred, 4)
        out[f"cal_bin{b}_actual_mean"] = round(mean_actual, 4)
    out["cal_ece"] = round(ece, 4)
    out["cal_n_samples"] = total
    return out


def _run_eval(
    iteration: int,
    net: UnifiedNet,
    pool: "SimClientPool | None",
    eval_episodes: int,
    eval_seed_base: str,
    max_steps: int,
    n_workers: int,
    dumper: RolloutDumper | None,
    runtime=None,
) -> dict:
    """跑一批固定 seed 的 greedy eval；单独 dump；不入 PPO。

    固定 seeds 让 eval 之间可比 —— 消除"agent 真变强 vs 抽到好运气"的混淆。
    greedy=True 用 argmax action 让 policy 输出尽可能稳定。
    """
    import statistics
    import queue
    import threading

    # 固定 seeds（不随训练 rng 变化）
    seeds = [f"{eval_seed_base}-{i}" for i in range(eval_episodes)]
    seeds_per_worker: list[list[str]] = [[] for _ in range(n_workers)]
    for i, s in enumerate(seeds):
        seeds_per_worker[i % n_workers].append(s)

    net.eval()
    infos: list[dict] = []
    if runtime is not None:
        eval_tasks = [
            {
                "seed": seed,
                "record_trajectory": False,
                "greedy": True,
                "max_steps": max_steps,
            }
            for seed in seeds
        ]
        task_ids = runtime.submit_tasks(eval_tasks)
        for env in runtime.gather_results(len(task_ids)):
            infos.extend(env.infos)
    else:
        result_q: "queue.Queue" = queue.Queue()
        threads: list = []
        for w_idx in range(n_workers):
            if not seeds_per_worker[w_idx]:
                continue
            t = threading.Thread(
                target=_collect_worker,
                args=(w_idx, pool, net, seeds_per_worker[w_idx], max_steps, result_q),
                kwargs={"greedy": True},  # eval 用 argmax
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=600)

        while not result_q.empty():
            r = result_q.get()
            infos.extend(r["infos"])

    n_total = len(infos)
    n_wins = sum(1 for info in infos if is_victory_outcome(info.get("outcome", "")))
    floors = [int(info.get("floor", 0) or 0) for info in infos]
    final_hps = [int(info.get("final_hp", 0) or 0) for info in infos]

    metrics = {
        "eval_iteration": iteration,
        "eval_episodes": n_total,
        "eval_wins": n_wins,
        "eval_win_rate": n_wins / max(n_total, 1),
        "eval_avg_floor": sum(floors) / max(n_total, 1),
        "eval_median_floor": statistics.median(floors) if floors else 0,
        "eval_max_floor": max(floors) if floors else 0,
        "eval_min_floor": min(floors) if floors else 0,
        "eval_avg_final_hp": sum(final_hps) / max(n_total, 1),
    }

    if dumper is not None:
        eval_dir = Path(getattr(dumper, "root", Path(".")))
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / f"eval_iter{iteration:04d}_episodes.jsonl", "w", encoding="utf-8") as f:
            for info in infos:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")
        with open(eval_dir / f"eval_iter{iteration:04d}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def _check_artifacts_path(path_str: str, label: str) -> None:
    """DIAGNOSTICS_CONVENTION.md: 训练产物必须落 STS2AI/Artifacts/ 下。"""
    if not path_str:
        return
    resolved = Path(path_str).resolve()
    artifacts_root = (Path(__file__).resolve().parents[3] / "Artifacts").resolve()
    try:
        resolved.relative_to(artifacts_root)
        return
    except ValueError:
        pass
    logger.warning(
        f"[convention] --{label}={path_str} 不在 STS2AI/Artifacts/ 下。"
        f" 规范路径见 docs/design/DIAGNOSTICS_CONVENTION.md。"
    )


def train_full_run(args: argparse.Namespace) -> None:
    import queue
    import threading

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 优先用 --preset（slim/full/tiny），否则用散参数（旧方式）
    if args.preset:
        cfg = from_preset(args.preset)
        # U3 修复：只有显式传入（非 None）才覆盖 preset。之前 `args.d_model > 0` 永远为
        # True（默认 384），tiny 的 128 设计值被强制覆盖成 384，preset 形同虚设。
        if args.d_model is not None: cfg.d_model = args.d_model
        if args.n_heads is not None: cfg.n_heads = args.n_heads
        if args.n_build_slots is not None: cfg.n_build_slots = args.n_build_slots
        if args.dropout is not None: cfg.dropout = args.dropout
        if args.max_numeric_dim is not None: cfg.max_numeric_dim = args.max_numeric_dim
        net = UnifiedNet(config=cfg).to(device)
        logger.info(f"Using preset '{args.preset}': {cfg}")
    else:
        # 散参数路径（没传 --preset）：None fallback 到历史默认值
        net = UnifiedNet(
            d_model=args.d_model or 384,
            n_heads=args.n_heads or 8,
            n_build_slots=args.n_build_slots or 8,
            max_numeric_dim=args.max_numeric_dim,
            dropout=args.dropout if args.dropout is not None else 0.1,
        ).to(device)
    params = sum(p.numel() for p in net.parameters())
    logger.info(f"UnifiedNet: {params:,} params ({params/1e6:.1f}M) on {device}")

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # 兼容多种 checkpoint 格式 (和 combat_cotrainer 对齐):
        #   {"net": ...}             (cotrainer 自己保存的老格式)
        #   {"model_state": ..., ...}(BC / train_noncombat_offline 保存)
        #   裸 state_dict
        if isinstance(state, dict):
            if "net" in state:
                state = state["net"]
            elif "model_state" in state:
                state = state["model_state"]
        try:
            net.load_state_dict(state)
            logger.info(f"Loaded checkpoint: {args.checkpoint}")
        except Exception as e:
            logger.warning(f"Full load failed: {e}; falling back to load_compatible_params")
            report = net.load_compatible_params(state, strict_shapes=True)
            logger.info(
                f"Partial load: loaded={report['loaded']} skipped_shape={report['skipped_shape']} "
                f"missing={report['missing']}"
            )

    trainer = UnifiedPPOTrainer(net, PPOConfig(
        lr=args.lr, ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        max_numeric_dim=getattr(net.config, "max_numeric_dim", args.max_numeric_dim or 58),
        value_warmup_iters=args.value_warmup_iters,
        target_kl=args.target_kl,
    ))

    rollout_cfg = build_rollout_engine_config(
        args,
        max_numeric_dim=int(getattr(net.tokenizer, "max_numeric_dim", 58)),
    )
    n_workers = rollout_cfg.rollout_num_actors
    base_port = args.port
    rng = random.Random(args.seed)
    total_wins = 0
    total_runs = 0

    pool: SimClientPool | None = None
    runtime = None
    catalog_client = None
    branch_client: BinaryBackedFullRunClient | None = None
    branch_compiler = CombatFeatureCompiler()
    artifacts_root = (Path(__file__).resolve().parents[3] / "Artifacts").resolve()
    run_name = Path(args.output_dir).resolve().name
    combat_teacher_root = artifacts_root / "combat_teacher" / run_name
    if rollout_cfg.use_legacy_thread_rollout:
        pool = SimClientPool(base_port=base_port, size=n_workers)
        pool.warmup()
    else:
        runtime = create_fullrun_runtime(args=args, net=net, rollout_cfg=rollout_cfg)
        helper_port = int(base_port) + n_workers + 100
        try:
            catalog_client = open_fullrun_catalog_client(helper_port)
        except Exception as e:
            logger.warning(f"Async helper client startup failed: {e}")
    if args.critical_step_capture:
        try:
            branch_client = _make_client(int(base_port) + n_workers + 200)
        except Exception as e:
            logger.warning(f"critical-step branch client startup failed: {e}")

    # ---- Attach sim 到 GAME_CATALOG：bank_assembler 的 power token 会用
    # game_catalog.powers[].base_classes / is_debuff_hint 精确判定 semantic group
    # 和 debuff，覆盖率 ~60% → ~95%+（fallback 到本地 heuristic）。
    try:
        from networkV2.s1_schema.sim_catalog import GAME_CATALOG
        if pool is not None:
            GAME_CATALOG.attach_sim(pool.clients[0])
        elif catalog_client is not None:
            GAME_CATALOG.attach_sim(catalog_client)
        else:
            raise RuntimeError("no sim client available for GAME_CATALOG.attach_sim")
        n_powers = len(getattr(GAME_CATALOG, "_power_metadata_by_class", {}))
        logger.info(f"Attached sim to GAME_CATALOG: {n_powers} power classes with metadata")
    except Exception as e:
        logger.warning(f"GAME_CATALOG.attach_sim failed: {e}（fallback 到 heuristic）")

    # ---- 路径规范校验（DIAGNOSTICS_CONVENTION.md）----
    _check_artifacts_path(args.dump_dir, "dump-dir")
    _check_artifacts_path(args.output_dir, "output-dir")

    # ---- Rollout + metrics dumper (diagnostic) ----
    dumper: RolloutDumper | None = None
    if args.dump_dir:
        dumper = RolloutDumper(args.dump_dir)
        # U3: 记录实际网络参数（args.d_model 可能是 None）
        _meta_d = getattr(net.config, "d_model", args.d_model) if hasattr(net, "config") else args.d_model
        _meta_nh = getattr(net.config, "n_heads", args.n_heads) if hasattr(net, "config") else args.n_heads
        dumper.write_meta({
            "preset": args.preset,
            "d_model": _meta_d, "n_heads": _meta_nh,
            "lr": args.lr, "clip_eps_loss_default": 0.15,
            "ppo_epochs": args.ppo_epochs, "mini_batch_size": args.mini_batch_size,
            "value_warmup_iters": args.value_warmup_iters,
            "target_kl": args.target_kl,
            "num_workers": n_workers,
            "rollout_async_default": float(not rollout_cfg.use_legacy_thread_rollout),
            "episodes_per_iter": args.episodes_per_iter,
            "net_params": sum(p.numel() for p in net.parameters()),
        })
        logger.info(f"Dumping rollout data to: {args.dump_dir}")

    # U3: args.d_model / args.n_heads 可能是 None（用户没传，走 preset 默认）
    # 打印时用实际网络配置（net.config.d_model / net.config.n_heads）而非 argparse 值
    actual_d = getattr(net.config, "d_model", args.d_model) if hasattr(net, "config") else args.d_model
    actual_nh = getattr(net.config, "n_heads", args.n_heads) if hasattr(net, "config") else args.n_heads
    print(f"\nConfig: d_model={actual_d} n_heads={actual_nh} lr={args.lr} "
          f"eps/iter={args.episodes_per_iter} actors={n_workers} "
          f"async={not rollout_cfg.use_legacy_thread_rollout} ppo_epochs={args.ppo_epochs}")
    print()
    print("Iter | Eps | Steps | W/L | Cum%  | AvgFlr | Losses                                            | Time")
    print("-----|-----|-------|-----|-------|--------|---------------------------------------------------|------")

    try:
      for iteration in range(1, args.max_iterations + 1):
        t0 = time.time()

        # 分配 seeds 给各 worker
        eps_total = args.episodes_per_iter
        episode_tasks: list[dict[str, Any]] = []
        capture_root = ""
        if args.critical_step_capture:
            capture_root = str(combat_teacher_root / "snapshots" / f"iter_{iteration:04d}")
        seeds_per_worker: list[list[str]] = [[] for _ in range(n_workers)]
        for i in range(eps_total):
            seed = f"fr-{iteration}-{i}-{rng.getrandbits(32):08x}"
            if rollout_cfg.use_legacy_thread_rollout:
                seeds_per_worker[i % n_workers].append(seed)
            episode_tasks.append({
                "seed": seed,
                "record_trajectory": False,
                "greedy": False,
                "max_steps": args.max_steps,
                "capture_root": capture_root,
            })

        # 并发收集：每个 worker 复用 pool 里的固定 client
        net.eval()
        # 汇总
        iter_samples: list[TrainingSample] = []
        iter_episode_infos: list[dict] = []
        w, l = 0, 0
        floors = []
        rollout_stats: dict[str, Any] = {}
        if rollout_cfg.use_legacy_thread_rollout:
            result_q: queue.Queue = queue.Queue()
            threads = []
            for w_idx in range(n_workers):
                if not seeds_per_worker[w_idx]:
                    continue
                # 每 worker 第 1 局记录完整 trajectory（用于诊断 stuck loop / 决策可读性）
                t = threading.Thread(
                    target=_collect_worker,
                    args=(w_idx, pool, net, seeds_per_worker[w_idx], args.max_steps, result_q),
                    kwargs={
                        "record_trajectory_every": max(1, len(seeds_per_worker[w_idx])),
                        "capture_root": capture_root,
                    },
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join(timeout=300)

            while not result_q.empty():
                r = result_q.get()
                sample_offset = len(iter_samples)
                iter_samples.extend(r["samples"])
                for info in r["infos"]:
                    for capture in info.get("critical_captures") or []:
                        capture["global_sample_index"] = sample_offset + int(capture.get("global_sample_index") or 0)
                    iter_episode_infos.append(info)
        else:
            if runtime is None:
                raise RuntimeError("async rollout runtime was not initialized")
            # 每 iter 取首个样本保留 trajectory，便于诊断 async 路径。
            if episode_tasks:
                episode_tasks[0]["record_trajectory"] = True
            task_ids = runtime.submit_tasks(episode_tasks)
            for env in runtime.gather_results(len(task_ids)):
                sample_offset = len(iter_samples)
                iter_samples.extend(env.samples)
                for info in env.infos:
                    for capture in info.get("critical_captures") or []:
                        capture["global_sample_index"] = sample_offset + int(capture.get("sample_index") or 0)
                    iter_episode_infos.append(info)
            rollout_stats = runtime.stats()

        for info in iter_episode_infos:
            total_runs += 1
            floors.append(info.get("floor", 0))
            outcome_str = info.get("outcome", "")
            if is_victory_outcome(outcome_str):
                w += 1
                total_wins += 1
            elif outcome_str != "error":
                l += 1

        # PPO update
        net.train()
        metrics: dict[str, Any] = {}
        need_critical_annotation = bool(
            args.critical_step_rebalance
            or args.critical_step_capture
            or args.offline_combat_teacher_data
        )
        if need_critical_annotation:
            metrics.update(annotate_critical_steps(iter_samples))
        train_samples = iter_samples
        if args.critical_step_rebalance:
            train_samples, rebalance_metrics = rebalance_training_samples(iter_samples, rng=rng)
            metrics.update(rebalance_metrics)
        if len(iter_samples) >= args.min_update_samples:
            metrics.update(trainer.train_step(train_samples))
        if rollout_stats:
            metrics.update(runtime_stats_to_metrics("rollout", rollout_stats))

        generated_teacher_path = combat_teacher_root / "critical_step_teacher_v1.jsonl"
        if args.critical_step_capture:
            capture_records: list[dict[str, Any]] = []
            for info in iter_episode_infos:
                for capture in info.get("critical_captures") or []:
                    sample_idx = int(capture.get("global_sample_index") or -1)
                    if not (0 <= sample_idx < len(iter_samples)):
                        continue
                    sample = iter_samples[sample_idx]
                    if float(sample.critical_score) < 0.8:
                        continue
                    capture_records.append({
                        "seed": str(capture.get("seed") or ""),
                        "episode_id": str(capture.get("episode_id") or ""),
                        "sample_index": int(capture.get("sample_index") or 0),
                        "floor": int(capture.get("floor") or sample.floor or 0),
                        "encounter_id": str(capture.get("encounter_id") or sample.encounter_id or ""),
                        "room_type": str(capture.get("room_type") or sample.room_type or ""),
                        "action_name": str(capture.get("action_name") or sample.action_name or ""),
                        "critical_tags": list(sample.critical_tags),
                        "critical_score": float(sample.critical_score),
                        "advantage": float(sample.advantage),
                        "legal_actions": capture.get("legal_actions") or [],
                        "snapshot_path": str(capture.get("snapshot_path") or ""),
                        "root_state": capture.get("root_state") or {},
                    })
            sorted_captures = sort_capture_records(capture_records)
            metrics["critical_capture_candidates"] = float(len(sorted_captures))
            if sorted_captures:
                combat_teacher_root.mkdir(parents=True, exist_ok=True)
                queue_path = combat_teacher_root / "critical_step_queue.jsonl"
                selected_queue_records = write_critical_step_queue(
                    sorted_captures,
                    output_path=queue_path,
                    top_k=int(args.critical_step_queue_topk),
                )
                metrics["critical_queue_size"] = float(len(selected_queue_records))
                if branch_client is not None:
                    try:
                        _branch_records, teacher_records, _raw_path = generate_branch_rollout_dataset(
                            selected_queue_records,
                            output_dir=combat_teacher_root,
                            client=branch_client,
                            net=net,
                            compiler=branch_compiler,
                        )
                        metrics["critical_teacher_records"] = float(len(teacher_records))
                    except Exception as exc:
                        logger.warning(f"critical-step branch generation failed: {exc}")
                else:
                    logger.warning("critical-step capture enabled but branch client is unavailable")

        teacher_data_path = str(args.offline_combat_teacher_data or "")
        if not teacher_data_path and generated_teacher_path.exists():
            teacher_data_path = str(generated_teacher_path)
        if teacher_data_path:
            try:
                teacher_entries = load_offline_combat_teacher_entries(
                    teacher_data_path,
                    compiler=CombatFeatureCompiler(),
                )
                metrics["combat_teacher_loaded"] = float(len(teacher_entries))
                teacher_metrics = run_offline_combat_teacher_updates(
                    net=net,
                    optimizer=trainer.optimizer,
                    entries=teacher_entries,
                    config=OfflineCombatTeacherConfig(
                        updates_per_iter=int(args.offline_combat_teacher_updates_per_iter),
                        batch_size=int(args.offline_combat_teacher_batch_size),
                        rank_weight=1.0,
                        cont_weight=1.0,
                        ce_weight=0.0,
                    ),
                    rng=rng,
                    max_numeric_dim=int(getattr(net.tokenizer, "max_numeric_dim", 58)),
                )
                metrics.update(teacher_metrics)
            except Exception as exc:
                logger.warning(f"offline combat teacher update failed: {exc}")

        # Fight-win head calibration：预测胜率 vs 真实 GAE return 的分桶对齐度
        # ECE（Expected Calibration Error）越小越好。0.1 以上说明 value head 系统性偏差大。
        try:
            cal_metrics = _compute_fight_win_calibration(iter_samples)
            metrics.update(cal_metrics)
        except Exception as e:
            logger.warning(f"calibration failed: {e}")

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
              f"{avg_floor:6.2f} | pl={pl:.5f} vl={vl:.3f} hp={hp:.3f} kl={kl:.4f} ep={ep} "
              f"cc={int(metrics.get('critical_combat_count', 0))} tq={int(metrics.get('critical_queue_size', 0))} "
              f"tu={int(metrics.get('combat_teacher_updates', 0))} | {elapsed:5.1f}s")

        if iteration % args.save_every == 0:
            p = Path(args.output_dir) / f"unified_v2_iter{iteration}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), p)

        # ---- Eval set: 固定 seeds greedy eval，判 agent 真实水平 ----
        if args.eval_every > 0 and iteration % args.eval_every == 0 and args.eval_episodes > 0:
            try:
                em = _run_eval(
                    iteration=iteration, net=net, pool=pool,
                    eval_episodes=args.eval_episodes,
                    eval_seed_base=args.eval_seed_base,
                    max_steps=args.max_steps, n_workers=n_workers, dumper=dumper,
                    runtime=runtime if not rollout_cfg.use_legacy_thread_rollout else None,
                )
                logger.info(
                    f"[Eval iter {iteration}] wr={em['eval_win_rate']*100:.1f}% "
                    f"({em['eval_wins']}/{em['eval_episodes']})  "
                    f"avg_floor={em['eval_avg_floor']:.2f}  "
                    f"median={em['eval_median_floor']:.0f}  "
                    f"max={em['eval_max_floor']}  "
                    f"avg_final_hp={em['eval_avg_final_hp']:.1f}"
                )
            except Exception as e:
                logger.warning(f"Eval failed: {e}")
    finally:
      # 训练结束（或异常退出）时统一关闭所有 sim 进程
      if runtime is not None:
          runtime.shutdown()
      if pool is not None:
          pool.close_all()
      if catalog_client is not None:
          try:
              catalog_client.close()
          except Exception:
              pass
      if branch_client is not None:
          try:
              branch_client.close()
          except Exception:
              pass

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
    # U3 修复：默认 None 让"未显式传入"和"传了默认值"可区分——之前默认 384/8 会无条件
    # 覆盖 preset 的 d_model/n_heads（比如 --preset tiny d_model 设计值 128 永远生效不了）。
    # 现在只有显式 --d-model / --n-heads 才覆盖 preset；不传就用 preset 定义的值。
    p.add_argument("--d-model", type=int, default=None,
                   help="覆盖 preset 的 d_model；不传 = 用 preset 默认 (tiny=128, slim/full=384)")
    p.add_argument("--n-heads", type=int, default=None,
                   help="覆盖 preset 的 n_heads；不传 = 用 preset 默认 (tiny=4, slim/full=8)")
    p.add_argument("--n-build-slots", type=int, default=None,
                   help="覆盖 preset 的 n_build_slots；不传 = 用 preset 默认")
    p.add_argument("--max-numeric-dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None,
                   help="覆盖 preset 的 dropout；不传 = 用 preset 默认 0.1")
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
    # 默认 20（从 10 调高）：高方差环境（STS2 抽牌/intent/事件随机性大）需要更多
    # sample 才能稳定评估胜率。4 workers × 5 ep ≈ 单 iter ~10k transitions，比旧 ~4k 更稳。
    p.add_argument("--episodes-per-iter", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--min-update-samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=15527)
    p.add_argument("--num-workers", type=int, default=4,
                   help="兼容旧参数；默认映射到 --rollout-num-actors。")
    # Eval set：每 N iter 用固定 seeds 跑一批 greedy eval，单独 dump 不入 PPO。
    # 判断 agent 真实水平 vs 训练 rollout 运气波动的关键信号。
    p.add_argument("--eval-every", type=int, default=10,
                   help="每 N iter 跑一次 eval；0 = 禁用")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="每次 eval 跑多少局（固定 seeds 保证可比）")
    p.add_argument("--eval-seed-base", type=str, default="eval-2026-04-17",
                   help="eval seeds 前缀；改变此值可切换到一组新的 eval 场景")
    # IO —— 路径规范（DIAGNOSTICS_CONVENTION.md）：统一落 STS2AI/Artifacts/ 下
    _artifacts_root = Path(__file__).resolve().parents[3] / "Artifacts"
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--output-dir", type=str,
                   default=str(_artifacts_root / "checkpoints" / "unified_v2"),
                   help="checkpoint 输出目录。规范路径：Artifacts/checkpoints/<exp>/")
    p.add_argument("--save-every", type=int, default=50)
    # 诊断：每 iter 把 samples/metrics/advantages 写到 dump_dir 下
    # 事后用 analyze_rollout.py 分析异常
    p.add_argument("--dump-dir", type=str, default="",
                   help="If set, dump rollout/metrics to this dir. 规范路径：Artifacts/runs/<exp>/")
    p.add_argument("--critical-step-rebalance", action="store_true",
                   help="开启关键 combat step 打标后的重采样配额（35/45/20）。")
    p.add_argument("--critical-step-capture", action="store_true",
                   help="导出关键 combat step snapshot，并生成 branch teacher 数据。")
    p.add_argument("--critical-step-queue-topk", type=int, default=64,
                   help="每 iter 导出的 critical-step queue 上限。")
    p.add_argument("--offline-combat-teacher-data", type=str, default="",
                   help="外部 critical_step_teacher_v1.jsonl 路径；为空时可复用本 iter 生成数据。")
    p.add_argument("--offline-combat-teacher-updates-per-iter", type=int, default=4,
                   help="每 iter 执行多少次 offline combat teacher update；需配合 capture 或 teacher 数据使用。")
    p.add_argument("--offline-combat-teacher-batch-size", type=int, default=64,
                   help="offline combat teacher update 的 batch size。")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    add_rollout_engine_args(p)
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")
    train_full_run(args)


if __name__ == "__main__":
    main()
