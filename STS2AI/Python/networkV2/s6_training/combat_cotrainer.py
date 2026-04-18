"""Combat-only Co-Trainer：针对硬战斗（elite/boss）的专项训练。

背景：
  full-run rollout (train_full_run_v2) 有 sparse reward 问题 —— 大部分 rollout 时间
  在打 CULTISTS 级别的简单怪，elite/boss 样本极少。200 iter × 50 ep 后 deck_eval
  显示只会打 CULTISTS，elite/boss 全 0%。

本 trainer：
  - 绕过 full-run，直接用 PipeBackedCombatTrainingClient
  - 每个 rollout 是 1 场指定 (encounter × deck) 的战斗
  - 8 worker 并发，大量生产 combat samples
  - 用 UnifiedNet 的 combat branch + UnifiedPPOTrainer 训练
  - Curriculum：简单怪 → 精英 → boss
  - 定期 deck_eval 评估

用法（所有产物必须落 STS2AI/Artifacts/ 下，见 DIAGNOSTICS_CONVENTION.md）：
  python -m networkV2.s6_training.combat_cotrainer \\
    --preset slim --checkpoint ../Artifacts/checkpoints/co8/cotrainer_iter120.pt \\
    --num-workers 8 --max-iterations 200 \\
    --base-port 15700 \\
    --dump-dir ../Artifacts/runs/co13 \\
    --output-dir ../Artifacts/checkpoints/co13
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import random
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from networkV2.s3_state_tracker.combat_env_wrapper import CombatStateTracker
from networkV2.s4_compiler.feature_compiler import CombatFeatureCompiler
from networkV2.s5_net.unified_net import UnifiedNet
from networkV2.s5_net.network_config import from_preset
from networkV2.s6_training.batch import TrainingSample
from networkV2.s6_training.deck_eval import (
    evaluate_deck, baseline_act1_set, ironclad_starter_deck,
)
from networkV2.s6_training.head_targets import (
    compute_boss_readiness_target, compute_deck_quality_target,
    compute_resource_health_target, compute_resource_retention_target,
)
from networkV2.s6_training.ppo import UnifiedPPOTrainer, PPOConfig
from networkV2.s6_training.rewards import (
    combat_step_reward,
    turn_end_reward,
    kill_overkill_reward,
    combat_local_tactical_reward,
    dense_combat_shaping,
    co_trainer_boss_damage_bonus,
    co_trainer_boss_debuff_bonus,
    co_trainer_final_reward,
    boss_damage_ratio,
    WIN_REWARD,
    LOSE_REWARD,
)
from networkV2.s6_training.train_full_run_v2 import (
    _backfill_turn_damage, _enemies_total_hp,
)
from networkV2.s7_diagnostics.rollout_dumper import RolloutDumper

from env.combat_training_env import PipeBackedCombatTrainingClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chain Episode 配置（1 ep = 多场串连，HP 跨场保留，场间可定量回血）
# ---------------------------------------------------------------------------

# 每 ep 的战斗序列：(room_type, count)。总长度 = sum(count)
# 默认：3 monster → 1 elite → 1 boss，共 5 场
CHAIN_STRUCTURE: list[tuple[str, int]] = [
    ("monster", 3),
    ("elite", 1),
    ("boss", 1),
]

# Chain 内部回血：sub_combat_index (0-based) → 战斗结束后回血比例 (of max_hp)
# 索引 i 表示第 i+1 场打完后回血；最后一场（boss）打完无意义不设
# 默认规则：3 monster 打完（index 2）回 20%；elite 打完（index 3）回 30%
CHAIN_HEAL_AFTER_COMBAT: dict[int, float] = {
    2: 0.20,  # 3 场 monster 完成后
    3: 0.30,  # elite 完成后
}

# Chain 开始满血（无论 deck 里 current_hp 是多少）
CHAIN_START_FULL_HP: bool = True

# 任一场 defeat 后中止整个 chain（玩家死了）
CHAIN_ABORT_ON_DEFEAT: bool = True


# ---------------------------------------------------------------------------
# Curriculum：按游戏数据派生的难度（data-driven）
# ---------------------------------------------------------------------------

# 规范（SCHEMA_CONVENTION.md）：curriculum pool 从 GAME_CATALOG 运行时派生，
# 不手写 encounter_id 列表。分级规则基于真实 monster powers（block 机制、scaling 等）。

from networkV2.s1_schema.sim_catalog import GAME_CATALOG


# 已知 sim 侧有 NullReferenceException bug 的 encounter（训练时排除）
# 校验命令：对每个 encounter 做 client.reset + 1 step，会 CRASH 的标记在此。
# 目前发现：KAISER_CRAB_BOSS step 1 必 NullReference crash（sim C# bug）
_SIM_BROKEN_ENCOUNTERS: frozenset[str] = frozenset({
    "KAISER_CRAB_BOSS",
})


@lru_cache(maxsize=1)
def _derive_curriculum_pools() -> dict[str, list[tuple[str, str]]]:
    """从 GAME_CATALOG 派生 curriculum pools。一次启动 cache 一次。

    分级逻辑（纯基于 game data 的 power class 存在性）：
      - starter_friendly: monster + 无 block 机制（has_hard_scaling 也视为可学）
      - block_heavy:      monster + 有 block 机制（PlatingPower/IntangiblePower 等）
      - elite:            room_type=elite
      - boss:             room_type=boss
    """
    pools: dict[str, list[tuple[str, str]]] = {
        "starter_friendly": [],
        "block_heavy": [],
        "elite": [],
        "boss": [],
    }
    # Act 过滤：只训 act 0 (STS2 Act 1 = Overgrowth)。
    # ModelDb.Acts 顺序（C# ModelDb.cs:175）：
    #   Acts[0]=Overgrowth  Acts[1]=Hive  Acts[2]=Glory  Acts[3]=Underdocks
    TARGET_ACTS = frozenset({0})   # only STS2 Act 1 for now
    missing_act_idx: list[str] = []  # 收集 act_idx=-1 的 encounter 以便 warn

    for enc in GAME_CATALOG.encounters():
        eid = enc["encounter_id"].upper()
        rt = enc["room_type"]
        act_idx = enc.get("act_index", -1)
        # 过滤 sim crash encounter
        if eid in _SIM_BROKEN_ENCOUNTERS:
            continue
        # 过滤 event encounter（如 ARCHITECT_EVENT_ENCOUNTER 9999HP 假怪，永 timeout）
        if "EVENT_ENCOUNTER" in eid:
            continue
        # 严格过滤非目标 act：act_idx 不在 TARGET_ACTS（含 -1/缺失情形）都 reject
        # 历史 co11/co12 bug：条件写成 `act_idx >= 0 and act_idx not in TARGET_ACTS`，
        # act_idx=-1 时第一分支 False 导致漏过滤 → 训了全 4 个 act 的 boss (77 pool)
        if act_idx not in TARGET_ACTS:
            if act_idx == -1:
                missing_act_idx.append(eid)
            continue
        if rt == "boss":
            pools["boss"].append((eid, "boss"))
            continue
        if rt == "elite":
            pools["elite"].append((eid, "elite"))
            continue
        # monster - 只看 block 机制（scaling 可通过 dense shaping partial 学）
        sig = GAME_CATALOG.encounter_difficulty_signals(enc["encounter_id"])
        if sig.get("is_starter_blocker"):
            pools["block_heavy"].append((eid, "monster"))
        else:
            pools["starter_friendly"].append((eid, "monster"))

    if missing_act_idx and not _pool_log_state["warned"]:
        logger.warning(
            f"[curriculum] {len(missing_act_idx)} encounters have act_index=-1 "
            f"(被当作非目标 act 过滤)。前 10 个: {missing_act_idx[:10]}"
        )
        _pool_log_state["warned"] = True
    return pools


# 注意：不在 import 时 call _derive_curriculum_pools()！
# GAME_CATALOG 需要先 attach_sim 才能拿到 act_index 等完整数据；
# 如果 eager call 会用 sqlite fallback（无 act_index）→ filter 失效 → co11/co12 bug。
# 所有使用都通过 _derive_curriculum_pools() 即时调用（cache 生效后仅首次计算）。
_pool_log_state: dict[str, Any] = {"warned": False, "printed_pool": False}


def curriculum_at_iter(iteration: int) -> list[tuple[str, str]]:
    """随 iter 推进难度（**boss 攻坚模式**：boss 提前入池，用真实 deck 训练）：
      iter 1-10:   starter_friendly only（让 agent 暖身）
      iter 11-20:  + elite + boss（boss 用真实 boss-ready deck，攻坚开始）
      iter 21+:    + block_heavy（补全覆盖）
    """
    pools = _derive_curriculum_pools()
    if not _pool_log_state["printed_pool"]:
        logger.info(
            f"[curriculum] TARGET_ACTS filter result: "
            f"starter_friendly={len(pools['starter_friendly'])} "
            f"block_heavy={len(pools['block_heavy'])} "
            f"elite={len(pools['elite'])} "
            f"boss={len(pools['boss'])}"
        )
        logger.info(f"[curriculum] boss pool: {[e[0] for e in pools['boss']]}")
        logger.info(f"[curriculum] elite pool: {[e[0] for e in pools['elite']]}")
        _pool_log_state["printed_pool"] = True
    pool = list(pools["starter_friendly"])
    if not pool:
        for k in ("block_heavy", "elite", "boss"):
            if pools.get(k):
                pool = list(pools[k])
                break
    if iteration >= 11:
        pool += pools["elite"]
        pool += pools["boss"]    # 提前加 boss（用真实 deck）
    if iteration >= 21:
        pool += pools["block_heavy"]
    return pool


# ---------------------------------------------------------------------------
# Deck variants：block-heavy / elite / boss 用强化 deck 训练
# ---------------------------------------------------------------------------

def buffed_ironclad_deck() -> dict[str, Any]:
    """给 block-heavy 战斗用的升级 deck。

    规范（SCHEMA_CONVENTION）：所有卡 ID 必须 GAME_CATALOG 验证存在。
    历史坑：`CLEAVE` 是 STS1 卡名，STS2 没有 → combat_reset 全失败（co7 iter30+
    39% 战斗 error）。

    现在 deck 构建时校验每张卡存在；不存在的跳过 + warn。
    """
    from networkV2.s1_schema.sim_catalog import GAME_CATALOG
    candidates = [
        "STRIKE_IRONCLAD",     # 1 cost, 6 damage
        "DEFEND_IRONCLAD",     # 1 cost, 5 block
        "BASH",                # 2 cost, 8 damage + vulnerable
        "POMMEL_STRIKE",       # 1 cost, 9 damage + draw
        "IRON_WAVE",           # 1 cost, 5 damage + 5 block（攻防兼备）
        "THUNDERCLAP",         # 1 cost, AoE 4 damage + weak
        "ANGER",               # 0 cost, 6 damage, copy to discard
        "BODY_SLAM",           # 1 cost, damage = block
    ]
    valid = [c for c in candidates if GAME_CATALOG.card_exists(c)]
    missing = set(candidates) - set(valid)
    if missing:
        logger.warning(f"buffed_deck 跳过不存在的 STS2 卡: {missing}")

    # 用 valid 卡构造 deck（多复制基础卡确保容量）
    deck_list: list[str] = []
    for c in valid:
        # 核心 strike/defend 给 4 和 3 张，其他 1 张
        if c == "STRIKE_IRONCLAD":
            deck_list += [c] * 4
        elif c == "DEFEND_IRONCLAD":
            deck_list += [c] * 3
        else:
            deck_list += [c]

    return {
        "deck": deck_list,
        "max_hp": 80,
        "current_hp": 80,
        "gold": 99,
        "max_energy": 3,
    }


@lru_cache(maxsize=1)
def _load_real_boss_decks_cached() -> list[dict]:
    """Cache 真实 boss deck 列表（启动时加载一次）。"""
    try:
        from networkV2.s6_training.real_boss_decks import load_real_boss_decks
        return load_real_boss_decks()
    except Exception as e:
        logger.warning(f"Failed loading real boss decks: {e}")
        return []


def deck_for_encounter(encounter_id: str, rng: random.Random | None = None) -> dict[str, Any]:
    """根据 encounter 难度派 deck（data-driven）：
      - starter_friendly: starter deck
      - block_heavy/elite: buffed synthetic deck
      - boss:              随机抽一个真实 AI 打 boss 时的 deck snapshot（14-22 张 + 2-3 relics）

    真实 deck 来自 `combat_teacher/tactical_v1_replay_*` artifacts，由历史 solver
    成功走到 boss 的 state。比 synthetic deck 更符合真实中后期分布，破解 boss 训练
    zero-positive-advantage trap。
    """
    from networkV2.s6_training.deck_eval import ironclad_starter_deck
    pools = _derive_curriculum_pools()
    eid_up = encounter_id.upper()
    boss_ids = {e[0] for e in pools["boss"]}
    elite_block_ids = (
        {e[0] for e in pools["block_heavy"]}
        | {e[0] for e in pools["elite"]}
    )
    # Boss: 优先用真实 deck
    if eid_up in boss_ids:
        real_decks = _load_real_boss_decks_cached()
        if real_decks:
            chosen = (rng or random).choice(real_decks)
            # 返回 clean spec（去掉 metadata _floor / _state_type）
            return {
                "deck": chosen["deck"],
                "relics": chosen.get("relics", []),
                "max_hp": chosen.get("max_hp", 80),
                "current_hp": chosen.get("current_hp", chosen.get("max_hp", 80)),
                "gold": chosen.get("gold", 99),
                "max_energy": chosen.get("max_energy", 3),
            }
        # Fallback：无 real decks 则用 buffed
        return buffed_ironclad_deck()
    if eid_up in elite_block_ids:
        return buffed_ironclad_deck()
    return ironclad_starter_deck()


# 战斗 reward 实现已移至 networkV2.s6_training.rewards：
#   - dense_combat_shaping (co 独有 attack+block shaping)
#   - co_trainer_boss_debuff_bonus (boss 战 Vuln/Weak setup +0.02)
#   - co_trainer_final_reward (boss win ×2.0, boss near-loss -0.3)
#   - boss_damage_ratio (boss 败局的 near-win 判定)


# ---------------------------------------------------------------------------
# 单场战斗 rollout（PipeBackedCombatTrainingClient）
# ---------------------------------------------------------------------------

def combat_rollout(
    client: PipeBackedCombatTrainingClient,
    net: UnifiedNet,
    compiler: CombatFeatureCompiler,
    encounter_id: str,
    room_type: str,
    deck: dict[str, Any] | None,
    max_steps: int = 300,
    seed: str = "",
    greedy: bool = False,
    record_trajectory: bool = False,
) -> tuple[list[TrainingSample], dict[str, Any]]:
    """跑一场 combat，收 TrainingSample。

    record_trajectory=True 时 info["trajectory"] 含每步摘要（step/turn/hp/block/
    enemy_hp/chosen_action/chosen_card/reward/value），用于事后 debug 决策序列。
    """
    try:
        state = client.reset(encounter_id=encounter_id, build=deck, seed=seed or None)
    except Exception as e:
        return [], {"outcome": "error", "error": str(e),
                    "encounter_id": encounter_id, "steps": 0}

    tracker = CombatStateTracker()
    tracker.on_run_start()
    tracker.on_combat_start(state, encounter_id.lower(), room_type)

    samples: list[TrainingSample] = []
    prev_state = state
    steps = 0
    turn_start_sample_idx = 0
    turn_step_damages: list[float] = []
    hp_at_start = int((state.get("player") or {}).get("hp", 0) or 0)
    # Gap 1:per-turn HP/enemy_max 跟踪,给 turn_end_reward 算 hp_loss_this_turn
    hp_at_turn_start = hp_at_start
    enemy_max_hp_at_turn_start = sum(
        int(e.get("max_hp", 0) or 0)
        for e in ((state.get("battle") or {}).get("enemies") or state.get("enemies") or [])
        if isinstance(e, dict)
    ) or 1
    # boss 战开始时的 enemy 总 max_hp，用于 terminal 时算 near-win ratio
    enemy_max_hp_at_start = sum(
        int(e.get("max_hp", 0) or 0)
        for e in ((state.get("battle") or {}).get("enemies") or state.get("enemies") or [])
        if isinstance(e, dict)
    )
    device = next(net.parameters()).device
    trajectory: list[dict[str, Any]] = [] if record_trajectory else []
    # Encounter conditioning：boss/encounter id → embedding index，每步 forward 传入
    from networkV2.s1_schema.encounter_vocab import encounter_to_index
    enc_idx_tensor = torch.tensor(
        [encounter_to_index(encounter_id)], dtype=torch.long, device=device,
    )

    for _ in range(max_steps):
        legal = state.get("legal_actions", [])
        if not legal:
            break

        banks = compiler.compile(
            state, legal,
            combat_memory=tracker.combat_memory,
            turn_prefix=tracker.turn_prefix,
            run_build_memory=tracker.run_build_memory,
            encounter_id=encounter_id.lower(),
            room_type=room_type,
        )
        with torch.no_grad():
            out = net(banks=banks, encounter_idx=enc_idx_tensor)
        logits = out.logits[0, :len(legal)]
        mask = out.action_mask[0, :len(legal)]
        logits = torch.nan_to_num(logits.masked_fill(~mask, float("-inf")), nan=0.0)
        dist = Categorical(logits=logits)
        idx = logits.argmax().item() if greedy else dist.sample().item()
        lp = dist.log_prob(torch.tensor(idx, device=logits.device)).item()
        value = out.values.fight_win.item() if out.values is not None else 0.5
        chosen = legal[idx]

        # step
        try:
            next_state, _r, done, _info = client.step(chosen)
        except Exception as e:
            logger.warning(f"[co] step failed: {e}")
            break

        # per-step damage
        step_damage = max(0, _enemies_total_hp(state) - _enemies_total_hp(next_state))
        turn_step_damages.append(float(step_damage))

        tracker.on_step(next_state, chosen, prev_state=state)

        # reward: PBRS + tactical + dense shaping + boss-aware
        next_outcome = next_state.get("run_outcome") if next_state.get("terminal") else None
        combat_won = None
        terminal_boss_damage_ratio = 0.0
        if done:
            combat_won = str(next_outcome or "").lower() == "victory"
            # Boss 败局 near-win ratio：让第一层 terminal reward 也有 near-win 渐变
            if combat_won is False and room_type == "boss" and enemy_max_hp_at_start > 0:
                terminal_boss_damage_ratio = boss_damage_ratio(next_state, enemy_max_hp_at_start)
        reward = combat_step_reward(
            prev_state, next_state,
            combat_won=combat_won,
            hp_at_combat_start=hp_at_start,
            boss_damage_ratio=terminal_boss_damage_ratio,
        )
        reward += combat_local_tactical_reward(state, chosen, legal)
        # Dense damage + block balanced shaping（解 zero-positive-adv trap）
        _player_max_hp = max(int((state.get("player") or {}).get("max_hp", 1) or 1), 1)
        reward += dense_combat_shaping(state, next_state, _player_max_hp)
        # Boss 战：每 step damage 按掉血百分比额外奖励（boss HP 高，damage 珍贵）
        reward += co_trainer_boss_damage_bonus(state, next_state, room_type)
        # Boss 战：Vuln/Weak 套 boss 身上额外 +0.02（放大 debuff setup 价值）
        reward += co_trainer_boss_debuff_bonus(state, chosen, room_type)

        # Gap 2:kill bonus(+0.05/enemy)+ overkill penalty(-0.02 大牌打残血)
        reward += kill_overkill_reward(state, next_state, chosen)

        chosen_action_name = str(chosen.get("action", "")).lower()
        if chosen_action_name in ("end_turn", "end"):
            player = (state.get("player") or {})
            battle = state.get("battle") or {}
            prev_energy = int(battle.get("energy", player.get("energy", 0)) or 0)
            hand = battle.get("hand") or player.get("hand") or []
            playable = sum(1 for c in hand if isinstance(c, dict) and c.get("can_play", False))
            if prev_energy > 0 and playable > 0:
                reward -= 0.10

        # head targets
        cm = tracker.combat_memory
        rbm = tracker.run_build_memory
        player = (next_state.get("player") or {})
        cur_hp = int(player.get("hp", 0) or 0)
        max_hp = max(int(player.get("max_hp", 1) or 1), 1)
        hp_ratio = cur_hp / max_hp
        sw = {"boss": 1.5, "elite": 1.2}.get(room_type, 1.0)
        transition_risk_t = min(cm.transition_count / max(cm.turn_index, 1), 1.0)
        deck_quality_t = compute_deck_quality_target(rbm, hp_ratio)
        boss_readiness_t = compute_boss_readiness_target(rbm, hp_ratio)
        resource_health_t = compute_resource_health_target(rbm, hp_ratio)
        resource_retention_t = compute_resource_retention_target(rbm, hp_ratio)

        samples.append(TrainingSample(
            banks=banks,
            action_index=idx,
            old_log_prob=lp,
            reward=reward,
            advantage=0.0, value_target=0.0,
            value_estimate=value,
            fight_win_target=-1.0,
            hp_loss_target=float(max(hp_at_start - cur_hp, 0)),
            survival_target=hp_ratio,
            leaf_target=0.0,
            transition_risk_target=transition_risk_t,
            resource_retention_target=resource_retention_t,
            boss_readiness_target=boss_readiness_t,
            resource_health_target=resource_health_t,
            deck_quality_target=deck_quality_t,
            turn_damage_target=-1.0,
            sample_weight=sw,
            encounter_id=encounter_id.lower(),
            room_type=room_type,
        ))

        # Trajectory record（轻量 per-step 快照，仅 record_trajectory=True 时启用）
        if record_trajectory:
            battle = state.get("battle") or {}
            enemies_prev = battle.get("enemies") or state.get("enemies") or []
            enemies_next = (next_state.get("battle") or {}).get("enemies") or next_state.get("enemies") or []
            # chosen_card 多路径提取（不同 env 暴露的 legal action 格式不同）
            _card_id = ""
            if isinstance(chosen, dict):
                for _k in ("card_id", "target_card_id"):
                    _v = chosen.get(_k)
                    if isinstance(_v, str) and _v:
                        _card_id = _v; break
                if not _card_id:
                    _c = chosen.get("card")
                    if isinstance(_c, dict):
                        _card_id = _c.get("id") or _c.get("card_id") or ""
                    elif isinstance(_c, str):
                        _card_id = _c
                if not _card_id:
                    # hand_index fallback: 从 state.battle.hand 查
                    _hi = chosen.get("hand_index", chosen.get("hand_idx", chosen.get("card_index")))
                    if _hi is not None:
                        _hand = battle.get("hand") or (state.get("player") or {}).get("hand") or []
                        if isinstance(_hi, int) and 0 <= _hi < len(_hand):
                            _card = _hand[_hi]
                            if isinstance(_card, dict):
                                _card_id = _card.get("id") or _card.get("card_id") or _card.get("name") or ""
            trajectory.append({
                "step": steps,
                "turn": int(cm.turn_index),
                "hp": cur_hp,
                "max_hp": max_hp,
                "block": int(battle.get("block", (state.get("player") or {}).get("block", 0)) or 0),
                "energy": int(battle.get("energy", (state.get("player") or {}).get("energy", 0)) or 0),
                "enemy_hp_total": sum(int(e.get("hp", 0) or 0) for e in enemies_prev if isinstance(e, dict)),
                "enemy_hp_after": sum(int(e.get("hp", 0) or 0) for e in enemies_next if isinstance(e, dict)),
                "n_legal": len(legal),
                "chosen_idx": idx,
                "chosen_action": str(chosen.get("action", ""))[:32],
                "chosen_card": str(_card_id)[:32],
                "chosen_raw_keys": sorted(list(chosen.keys()))[:10] if isinstance(chosen, dict) else [],
                "value_est": round(float(value), 4),
                "reward": round(float(reward), 4),
                "done": bool(done),
                "combat_won": combat_won,
            })

        # turn boundary → backfill turn_damage + Gap 1 turn_end_reward
        next_st = str(next_state.get("state_type", "")).lower()
        turn_ended = (
            chosen_action_name in ("end_turn", "end")
            or next_st not in ("monster", "elite", "boss")
            or done
        )
        if turn_ended:
            _backfill_turn_damage(samples, turn_start_sample_idx, turn_step_damages)
            # Gap 1:回合末给 combo / 防守即时 reward,加到本 step(end_turn 这一步)的 reward 上
            hp_after_turn = int((next_state.get("player") or {}).get("hp", 0) or 0)
            ter = turn_end_reward(
                turn_total_damage=sum(turn_step_damages),
                enemy_max_hp_at_turn_start=enemy_max_hp_at_turn_start,
                hp_at_turn_start=hp_at_turn_start,
                hp_after_turn=hp_after_turn,
                this_step_is_end_turn=True,
            )
            if ter != 0.0 and samples:
                # 加到本 step 产出的最后一个 sample 的 reward 上
                samples[-1].reward = float(samples[-1].reward) + ter

            # 重置 turn 跟踪
            turn_start_sample_idx = len(samples)
            turn_step_damages = []
            hp_at_turn_start = hp_after_turn
            enemy_max_hp_at_turn_start = sum(
                int(e.get("max_hp", 0) or 0)
                for e in ((next_state.get("battle") or {}).get("enemies") or [])
                if isinstance(e, dict) and e.get("is_alive", True)
            ) or 1

        steps += 1
        prev_state = state
        state = next_state
        if done:
            break

    # 终局 reward + hard label（boss-aware：win ×2.0，boss 近胜败 -0.3 而非 -1.0）
    final_hp = int((state.get("player") or {}).get("hp", 0) or 0)
    outcome = state.get("run_outcome")
    won = str(outcome or "").lower() == "victory"
    if samples:
        final_boss_ratio = (
            boss_damage_ratio(state, enemy_max_hp_at_start)
            if (not won and room_type == "boss" and enemy_max_hp_at_start > 0)
            else 0.0
        )
        final_r = co_trainer_final_reward(
            won=won, room_type=room_type,
            boss_damage_ratio=final_boss_ratio,
        )
        samples[-1].reward += final_r
        samples[-1].fight_win_target = 1.0 if won else 0.0

    # GAE
    _compute_gae_combat(samples)

    info = {
        "encounter_id": encounter_id,
        "room_type": room_type,
        "outcome": "victory" if won else "defeat",
        "steps": steps,
        "final_hp": final_hp,
        "max_hp": (state.get("player") or {}).get("max_hp", 0) or 0,
        "hp_loss": max(hp_at_start - final_hp, 0),
    }
    if record_trajectory:
        info["trajectory"] = trajectory
    return samples, info


# ---------------------------------------------------------------------------
# Chained Episode：1 ep = 多场串连（HP 跨场保留，场间可定量回血）
# ---------------------------------------------------------------------------

def build_chain_sequence(
    rng: random.Random,
    structure: list[tuple[str, int]] = CHAIN_STRUCTURE,
) -> list[tuple[str, str]]:
    """按 structure 从 curriculum pools 随机抽出 encounter 序列。

    返回: [(encounter_id, room_type), ...] 长度 = sum(count)
    """
    pools = _derive_curriculum_pools()
    type_to_pool = {
        "monster": pools["starter_friendly"] + pools.get("block_heavy", []),
        "elite": pools["elite"],
        "boss": pools["boss"],
    }
    sequence: list[tuple[str, str]] = []
    for rt, count in structure:
        pool = type_to_pool.get(rt, [])
        if not pool:
            # Fallback：缺失时用 monster 填（iter 早期 catalog 未 attach 等情况）
            pool = type_to_pool["monster"]
            if not pool:
                continue
        for _ in range(count):
            sequence.append(rng.choice(pool))
    return sequence


def chained_combat_rollout(
    client: PipeBackedCombatTrainingClient,
    net: UnifiedNet,
    compiler: CombatFeatureCompiler,
    encounter_sequence: list[tuple[str, str]],
    chain_deck: dict[str, Any],
    max_steps_per_combat: int,
    seed_prefix: str,
    record_trajectory: bool = False,
    heal_after_combat: dict[int, float] = CHAIN_HEAL_AFTER_COMBAT,
    start_full_hp: bool = CHAIN_START_FULL_HP,
    abort_on_defeat: bool = CHAIN_ABORT_ON_DEFEAT,
) -> tuple[list[TrainingSample], list[dict[str, Any]]]:
    """顺序跑 N 场战斗，HP 跨场保留，场间按 heal_after_combat 回血。

    返回: (samples_all, sub_combat_infos)
      - samples_all: 所有场的 TrainingSample 扁平列表（GAE 每场独立已算完）
      - sub_combat_infos: 每场一条 info（兼容老 episodes.jsonl 格式）
    """
    max_hp = int(chain_deck.get("max_hp", 80) or 80)
    cur_hp = max_hp if start_full_hp else int(chain_deck.get("current_hp", max_hp) or max_hp)

    all_samples: list[TrainingSample] = []
    sub_infos: list[dict[str, Any]] = []

    for i, (eid, rt) in enumerate(encounter_sequence):
        combat_deck = dict(chain_deck)
        combat_deck["current_hp"] = cur_hp
        combat_deck["max_hp"] = max_hp

        samples, info = combat_rollout(
            client, net, compiler, eid, rt, combat_deck,
            max_steps=max_steps_per_combat,
            seed=f"{seed_prefix}-c{i}",
            record_trajectory=record_trajectory,
        )
        # 给 sub_info 加 chain 上下文
        info["chain_index"] = i
        info["chain_total"] = len(encounter_sequence)
        info["hp_enter"] = cur_hp
        sub_infos.append(info)
        all_samples.extend(samples)

        # 失败则终止 chain
        if info.get("outcome") != "victory":
            if abort_on_defeat:
                break
            # 不终止时，死了也重置血量（避免 0 HP 进下一场）
            cur_hp = max_hp
        else:
            cur_hp = int(info.get("final_hp", cur_hp))
            # 场间回血
            heal_frac = heal_after_combat.get(i, 0.0)
            if heal_frac > 0:
                heal_amount = int(round(max_hp * heal_frac))
                cur_hp = min(max_hp, cur_hp + heal_amount)

    return all_samples, sub_infos


def build_chain_deck(rng: random.Random) -> dict[str, Any]:
    """整个 chain 共用的 deck。默认用真实 boss-ready deck（模拟玩家走到 boss 前状态）。

    Fallback：若 real_boss_decks 加载失败 → buffed_ironclad_deck。
    """
    real_decks = _load_real_boss_decks_cached()
    if real_decks:
        chosen = rng.choice(real_decks)
        return {
            "deck": list(chosen["deck"]),
            "relics": list(chosen.get("relics", [])),
            "max_hp": chosen.get("max_hp", 80),
            "current_hp": chosen.get("current_hp", chosen.get("max_hp", 80)),
            "gold": chosen.get("gold", 99),
            "max_energy": chosen.get("max_energy", 3),
        }
    return buffed_ironclad_deck()


def _compute_gae_combat(samples: list[TrainingSample], gamma: float = 0.99, lam: float = 0.95) -> None:
    """GAE for combat-only samples。和 train_full_run_v2._compute_gae 语义对齐：
    - `cur_val` 和 `next_val` 都走 bootstrap-aware lookup（终局硬标签优先于 rollout
      value_estimate）。原实现只对 next_val 做 terminal override、cur_val 固定用
      value_estimate，导致终局 delta 有系统性偏差，GAE 误差又往前传一整段。
    - value_target 显式 clamp 到 [0,1]，避免超界值进 loss 前还要靠 CombatLoss 二次
      clamp 隐式修正。
    """
    n = len(samples)
    if n == 0:
        return

    def _bootstrap(sample: TrainingSample) -> float:
        if sample.fight_win_target >= 0.0:
            return float(sample.fight_win_target)
        return float(sample.value_estimate)

    advantages = [0.0] * n
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = _bootstrap(samples[t + 1]) if t + 1 < n else 0.0
        cur_value = _bootstrap(samples[t])
        if samples[t].fight_win_target >= 0.0:
            # terminal：未来没有 step，next_value 强制 0（即便 t+1 是新战斗起点也一样）
            next_value = 0.0
        delta = samples[t].reward + gamma * next_value - cur_value
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    for t in range(n):
        cur_val = _bootstrap(samples[t])
        samples[t].advantage = advantages[t]
        samples[t].value_target = max(0.0, min(1.0, advantages[t] + cur_val))
        samples[t].leaf_target = max(-1.0, min(1.0, 2 * samples[t].value_target - 1))


# ---------------------------------------------------------------------------
# Sim Client Pool
# ---------------------------------------------------------------------------

class CombatClientPool:
    def __init__(self, base_port: int, n_clients: int):
        self.clients: list[PipeBackedCombatTrainingClient] = []
        for i in range(n_clients):
            self.clients.append(PipeBackedCombatTrainingClient(
                port=base_port + i, auto_launch=True,
            ))

    def get(self, worker_id: int) -> PipeBackedCombatTrainingClient:
        return self.clients[worker_id % len(self.clients)]

    def close_all(self) -> None:
        for c in self.clients:
            try:
                c.close()
            except Exception:
                pass
        self.clients.clear()


def _worker_collect(
    worker_id: int,
    pool: CombatClientPool,
    net: UnifiedNet,
    tasks: list,  # 两种形态：
                  #   single-combat: (enc_id, rt, deck, seed, record_traj)
                  #   chained:       ("chain", sequence, chain_deck, seed_prefix, record_traj)
    max_steps: int,
    result_q: queue.Queue,
) -> None:
    compiler = CombatFeatureCompiler()
    client = pool.get(worker_id)
    samples_out: list[TrainingSample] = []
    infos: list[dict] = []
    for task in tasks:
        try:
            if task and task[0] == "chain":
                _tag, sequence, chain_deck, seed_prefix, record_traj = task
                samples, sub_infos = chained_combat_rollout(
                    client, net, compiler, sequence, chain_deck,
                    max_steps_per_combat=max_steps,
                    seed_prefix=seed_prefix,
                    record_trajectory=record_traj,
                )
                samples_out.extend(samples)
                infos.extend(sub_infos)
            else:
                enc_id, rt, deck, seed, record_traj = task
                samples, info = combat_rollout(
                    client, net, compiler, enc_id, rt, deck,
                    max_steps=max_steps, seed=seed,
                    record_trajectory=record_traj,
                )
                samples_out.extend(samples)
                infos.append(info)
        except Exception as e:
            infos.append({"outcome": "error", "error": str(e),
                          "encounter_id": str(task[1] if task and task[0] == "chain" else task[0] if task else ""),
                          "steps": 0})
    result_q.put({"samples": samples_out, "infos": infos})


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _warn_if_outside_artifacts(path_str: str, label: str) -> None:
    """DIAGNOSTICS_CONVENTION.md: 训练产物必须落 STS2AI/Artifacts/ 下。"""
    if not path_str:
        return
    resolved = Path(path_str).resolve()
    artifacts_root = (Path(__file__).resolve().parents[3] / "Artifacts").resolve()
    try:
        resolved.relative_to(artifacts_root)
        return  # 在 Artifacts 下，OK
    except ValueError:
        pass
    logger.warning(
        f"[convention] --{label}={path_str} 不在 STS2AI/Artifacts/ 下。"
        f" 规范路径见 docs/design/DIAGNOSTICS_CONVENTION.md。"
    )


def run_cotrainer(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    cfg = from_preset(args.preset)
    net = UnifiedNet(config=cfg).to(device)
    params = sum(p.numel() for p in net.parameters())
    logger.info(f"UnifiedNet: {params:,} params ({params/1e6:.1f}M)")

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        if isinstance(state, dict) and "net" in state:
            state = state["net"]
        try:
            net.load_state_dict(state)
            logger.info(f"Loaded checkpoint: {args.checkpoint}")
        except Exception as e:
            logger.warning(f"Full load failed: {e}")
            report = net.load_compatible_params(state, strict_shapes=True)
            logger.info(
                f"Partial load: loaded={report['loaded']} skipped_shape={report['skipped_shape']} "
                f"missing={report['missing']}"
            )
            if report.get("skipped_sample"):
                logger.info(f"  skipped (sample): {report['skipped_sample']}")

        # 强制 re-init encounter conditioning（load 会覆盖 init，这里重置）
        # 原因：co17/co18 诊断发现 gate≈0.108 / embed_norm≈0.39 一直没动，conditioning
        # 机制处于"死锁"——注入量级太小 → PPO 梯度几乎 0 → 不更新 → 继续太小。
        # Load 把继承 checkpoint 的旧小值又拷回来，新 init（gate=1.0, embed std=0.3）被覆盖。
        # 强制 reset 让 conditioning 从"强注入"起步打破死锁。
        if args.checkpoint and getattr(net, "enable_encounter_conditioning", False) \
           and getattr(args, "reset_encounter_conditioning", True):
            import torch.nn as _nn
            _nn.init.normal_(net.encounter_embed.weight, mean=0.0, std=0.3)
            with torch.no_grad():
                # 保持 UnifiedNet __init__ 的不变量：UNKNOWN slot（index 0）=0。
                # 配合 _apply_encounter_conditioning 的 per-sample mask，这个 slot
                # 在 forward 里永远不会被读到，设置为 0 只是让 state_dict 检查时一目了然。
                net.encounter_embed.weight[0].zero_()
            net.encounter_gate.data.fill_(1.0)
            logger.info(
                "[conditioning] Forced re-init: gate=1.0, embed std=0.3, embed[0]=0 "
                "(破解 conditioning 死锁；UNKNOWN slot 保持 neutral)"
            )

    trainer = UnifiedPPOTrainer(net, PPOConfig(
        lr=args.lr, ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        value_warmup_iters=args.value_warmup_iters,
        target_kl=args.target_kl,
    ))

    pool = CombatClientPool(args.base_port, args.num_workers)
    # 把第一个 client 挂到 GAME_CATALOG，让所有后续特征查询走 sim API
    # （game_catalog 预取一次 cache，不干扰 worker rollouts）
    try:
        from networkV2.s1_schema.sim_catalog import GAME_CATALOG
        GAME_CATALOG.attach_sim(pool.clients[0])
        logger.info("Attached sim API to GAME_CATALOG (data-driven schema active)")
    except Exception as e:
        logger.warning(f"Failed to attach sim API, fallback to sqlite: {e}")

    rng = random.Random(args.seed)
    # 每个 encounter 用对应难度的 deck：starter-friendly 用 starter，
    # block-heavy / elite / boss 用 buffed deck（见 deck_for_encounter）

    _warn_if_outside_artifacts(args.dump_dir, "dump-dir")
    _warn_if_outside_artifacts(args.output_dir, "output-dir")

    dumper = RolloutDumper(args.dump_dir) if args.dump_dir else None
    if dumper:
        dumper.write_meta({
            "trainer": "combat_cotrainer",
            "preset": args.preset,
            "checkpoint": args.checkpoint,
            "num_workers": args.num_workers,
            "base_port": args.base_port,
        })

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    use_chained = not bool(getattr(args, "no_chained_episodes", False))
    chain_len = sum(c for _, c in CHAIN_STRUCTURE)

    print()
    print(f"Config: preset={args.preset} lr={args.lr} eps/iter={args.episodes_per_iter} workers={args.num_workers}")
    if use_chained:
        struct_str = " → ".join(f"{n}x{rt}" for rt, n in CHAIN_STRUCTURE)
        heal_str = ", ".join(f"after_c{i}+{int(100*f)}%" for i, f in sorted(CHAIN_HEAL_AFTER_COMBAT.items()))
        print(f"Chained episodes ENABLED: 1 ep = {chain_len} combats ({struct_str})")
        print(f"  heal schedule: {heal_str} | full_hp_start={CHAIN_START_FULL_HP}")
        print(f"  NOTE: chain fixed-structure from iter 1 (boss 战从 iter 1 就开始，不走 legacy curriculum)")
    else:
        print(f"Chained episodes DISABLED (legacy single-combat per ep)")
        print(f"  Curriculum: iter 1-10 easy, 11-30 +elite+boss, 31+ +block_heavy")
    print()
    print(f"Iter | Combats | Steps | W/L | Easy% / Med% / Hard% | Losses                           | Time")
    print(f"-----|-----|-------|-----|----------------------|----------------------------------|-----")

    for iteration in range(1, args.max_iterations + 1):
        t0 = time.time()
        eps_total = args.episodes_per_iter

        # 分配 tasks 给 workers
        # record_trajectory 逻辑（优先级从高到低）：
        #   1. --record-trajectory → 全量记录每 ep（~1-3 MB/iter）
        #   2. --record-trajectory-every N → 采样 N 条（优先 boss/elite）
        #   3. 默认都不记
        record_all = bool(getattr(args, "record_trajectory", False))
        n_record = max(0, int(getattr(args, "record_trajectory_every", 0) or 0))
        tasks_per_worker: list[list] = [[] for _ in range(args.num_workers)]

        if use_chained:
            # Chained 模式：每 ep 一个 chain（3m+1e+1b），整个 chain 用同一 deck
            for i in range(eps_total):
                seq = build_chain_sequence(rng)
                chain_deck = build_chain_deck(rng)
                seed_prefix = f"co-{iteration}-{i}-{rng.getrandbits(32):08x}"
                # 全量记录 → chain 内所有 sub_combat 都记；否则都不记（采样模式对 chain 不适用）
                record_traj = record_all
                tasks_per_worker[i % args.num_workers].append(
                    ("chain", seq, chain_deck, seed_prefix, record_traj))
        else:
            # 老模式：单 ep = 单 combat，由 curriculum_at_iter 控制难度池
            pool_encs = curriculum_at_iter(iteration)
            n_recorded = 0
            for i in range(eps_total):
                enc_id, rt = rng.choice(pool_encs)
                seed = f"co-{iteration}-{i}-{rng.getrandbits(32):08x}"
                ep_deck = deck_for_encounter(enc_id, rng=rng)
                if record_all:
                    record_traj = True
                elif n_record > 0 and n_recorded < n_record and rt in ("boss", "elite"):
                    record_traj = True; n_recorded += 1
                elif n_record > 0 and n_recorded < n_record and i >= eps_total - (n_record - n_recorded):
                    record_traj = True; n_recorded += 1
                else:
                    record_traj = False
                tasks_per_worker[i % args.num_workers].append((enc_id, rt, ep_deck, seed, record_traj))

        # 并发收集
        net.eval()
        result_q: queue.Queue = queue.Queue()
        threads = []
        for w_idx in range(args.num_workers):
            if not tasks_per_worker[w_idx]:
                continue
            t = threading.Thread(
                target=_worker_collect,
                args=(w_idx, pool, net, tasks_per_worker[w_idx], args.max_steps, result_q),
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=300)

        # 汇总
        iter_samples: list[TrainingSample] = []
        iter_infos: list[dict] = []
        while not result_q.empty():
            r = result_q.get()
            iter_samples.extend(r["samples"])
            iter_infos.extend(r["infos"])

        # 按难度分类胜率
        wins_by_rt = {"monster": 0, "elite": 0, "boss": 0}
        total_by_rt = {"monster": 0, "elite": 0, "boss": 0}
        for info in iter_infos:
            rt = info.get("room_type", "")
            if rt in total_by_rt:
                total_by_rt[rt] += 1
                if info.get("outcome") == "victory":
                    wins_by_rt[rt] += 1

        def _wr(rt): return wins_by_rt[rt] / max(total_by_rt[rt], 1)

        # 训练
        if len(iter_samples) >= args.min_update_samples:
            net.train()
            metrics = trainer.train_step(iter_samples)
        else:
            metrics = {"policy_loss": 0.0, "value_loss": 0.0}

        wall = time.time() - t0
        total_wins = sum(wins_by_rt.values())
        total_runs = sum(total_by_rt.values())

        line = (
            f"{iteration:5d} | {total_runs:3d} | {len(iter_samples):5d} | {total_wins}/{total_runs} | "
            f"{100*_wr('monster'):5.1f}% / {100*_wr('elite'):5.1f}% / {100*_wr('boss'):5.1f}% | "
            f"pl={metrics.get('policy_loss',0):.4f} vl={metrics.get('value_loss',0):.3f} "
            f"kl={metrics.get('approx_kl',0):.4f} ep={int(metrics.get('epochs_done',0))} | "
            f"{wall:5.1f}s"
        )
        print(line)

        if dumper:
            dumper.dump_iteration(
                iteration=iteration,
                samples=iter_samples,
                metrics=metrics,
                episode_infos=iter_infos,
                extra={
                    "wall_time_s": wall,
                    "total_runs": total_runs,
                    "total_wins": total_wins,
                    "wins_by_room_type": wins_by_rt,
                    "total_by_room_type": total_by_rt,
                    "curriculum_pool_size": (
                        len(pool_encs) if not use_chained
                        else sum(c for _, c in CHAIN_STRUCTURE)
                    ),
                    "chained_episodes": use_chained,
                },
            )

        # save
        if iteration % args.save_every == 0:
            torch.save(net.state_dict(), out_dir / f"cotrainer_iter{iteration}.pt")

    pool.close_all()
    torch.save(net.state_dict(), out_dir / "cotrainer_final.pt")
    print(f"Done. Saved {out_dir/'cotrainer_final.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="slim")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--base-port", type=int, default=15700)
    ap.add_argument("--max-iterations", type=int, default=200)
    ap.add_argument("--episodes-per-iter", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--min-update-samples", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)  # 比 full-run 低（专项训练更集中）
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--mini-batch-size", type=int, default=64)
    ap.add_argument("--value-warmup-iters", type=int, default=2)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--no-reset-encounter-conditioning", dest="reset_encounter_conditioning",
                    action="store_false", default=True,
                    help="load checkpoint 时不强制重置 encounter embed/gate（默认会重置以破解死锁）。")
    # 路径规范（DIAGNOSTICS_CONVENTION.md）：训练产物统一落 STS2AI/Artifacts/ 下
    _artifacts_root = Path(__file__).resolve().parents[3] / "Artifacts"
    ap.add_argument("--dump-dir", type=str, default="",
                    help="rollout dump 目录；空字符串=不 dump。规范路径：Artifacts/runs/<exp>/")
    ap.add_argument("--output-dir", type=str,
                    default=str(_artifacts_root / "checkpoints" / "co_default"),
                    help="checkpoint 输出目录。规范路径：Artifacts/checkpoints/<exp>/")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--no-chained-episodes", action="store_true",
                    help="关闭 chained episode（回到 legacy 单场/ep 模式）。"
                         "默认开启：1 ep = 5 场串连（3 monster → 20%回血 → 1 elite → 30%回血 → 1 boss）"
                         "HP 跨场保留，模拟玩家到 boss 前的 pressure。")
    ap.add_argument("--record-trajectory", action="store_true",
                    help="全量记录每 episode 的 per-step trajectory（80 ep/iter 约 1-3 MB/iter）。"
                         "写到 runs/<exp>/iter*_trajectories.jsonl。chained 模式下每个 sub_combat 独立记录。")
    ap.add_argument("--record-trajectory-every", type=int, default=0,
                    help="采样模式：每 iter 记录 N 条 episode（优先 boss/elite）。"
                         "0=关；与 --record-trajectory 互斥（全量优先）。")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run_cotrainer(args)


if __name__ == "__main__":
    main()
