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

from networkV2.s3_temporal_state.combat_state_tracker import CombatStateTracker
from networkV2.s4_featurization.decision_featurizer import DecisionFeaturizer
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
from networkV2.s6_training.rollout_async_engine import (
    add_rollout_engine_args,
    build_rollout_engine_config,
    runtime_stats_to_metrics,
)
from networkV2.s6_training.rollout_workers import (
    create_cotrainer_runtime,
    open_cotrainer_catalog_client,
)
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
    _backfill_turn_block, _backfill_turn_damage, _enemies_total_hp, _player_block_total,
)
from networkV2.s7_diagnostics.rollout_dumper import RolloutDumper

# 2026-04-18: combat rollout 通道切到 proto 直连。`CombatSession` 是
# `networkV2.s0_bridge.transport.PipeConnection + ProtoCodec` 的高层封装,
# sim 侧 legal_actions 是权威字段,Python 不再自己推断。
from networkV2.s0_bridge.combat_session import CombatSession as PipeBackedCombatTrainingClient

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

    encounters = list(GAME_CATALOG.encounters())
    if not encounters:
        sim_client = getattr(GAME_CATALOG, "_sim_client", None)
        if sim_client is not None and hasattr(sim_client, "combat_catalog"):
            try:
                direct_cat = sim_client.combat_catalog()
                encounters = list((direct_cat or {}).get("encounters") or [])
                if encounters:
                    logger.warning(
                        "[curriculum] GAME_CATALOG.encounters() 为空，已退化到 direct combat_catalog。"
                    )
            except Exception as e:
                logger.warning(f"[curriculum] direct combat_catalog fallback failed: {e}")

    def _fill_pools(*, relax_missing_act: bool) -> tuple[dict[str, list[tuple[str, str]]], list[str]]:
        out: dict[str, list[tuple[str, str]]] = {
            "starter_friendly": [],
            "block_heavy": [],
            "elite": [],
            "boss": [],
        }
        missing: list[str] = []
        for enc in encounters:
            eid = enc["encounter_id"].upper()
            rt = enc["room_type"]
            act_idx = enc.get("act_index", -1)
            if eid in _SIM_BROKEN_ENCOUNTERS:
                continue
            if "EVENT_ENCOUNTER" in eid:
                continue
            # sqlite fallback 没有 act_index 时，全部 encounter 会是 -1。此时不要把 pool 清空；
            # 退化成“放宽 act 过滤但保留 room_type/难度分级”。
            if act_idx not in TARGET_ACTS:
                if act_idx == -1:
                    missing.append(eid)
                    if relax_missing_act:
                        act_idx = 0
                if act_idx not in TARGET_ACTS:
                    continue
            if rt == "boss":
                out["boss"].append((eid, "boss"))
                continue
            if rt == "elite":
                out["elite"].append((eid, "elite"))
                continue
            sig = GAME_CATALOG.encounter_difficulty_signals(enc["encounter_id"])
            if sig.get("is_starter_blocker"):
                out["block_heavy"].append((eid, "monster"))
            else:
                out["starter_friendly"].append((eid, "monster"))
        return out, missing

    pools, missing_act_idx = _fill_pools(relax_missing_act=False)
    if not any(pools.values()) and encounters:
        pools, missing_act_idx = _fill_pools(relax_missing_act=True)
        logger.warning(
            "[curriculum] act_index 全缺失，已放宽 TARGET_ACTS 过滤并退化到 room_type/difficulty 分级。"
        )

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
def _load_skada_victory_decks_cached(character: str | None = None) -> list[dict]:
    """加载 skada 真实玩家 final_deck(18K unique decks,按 character 过滤)。

    优先级最高:比老 Artifacts/combat_teacher/ 的 teacher deck 多 100x,真实玩家分布。
    """
    try:
        from networkV2.s6_training.skada_victory_decks import load_skada_victory_decks
        decks = load_skada_victory_decks(character=character)
        return decks or []
    except Exception as e:
        logger.debug(f"skada_victory_decks load failed: {e}")
        return []


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
    net: UnifiedNet | None,
    featurizer: DecisionFeaturizer,
    encounter_id: str,
    room_type: str,
    deck: dict[str, Any] | None,
    max_steps: int = 300,
    seed: str = "",
    greedy: bool = False,
    record_trajectory: bool = False,
    graph_runner_holder: dict | None = None,
    reward_profile: str = "stochastic_stable",
    defer_gae: bool = False,
    inference_client: Any | None = None,
    task_id: int = 0,
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
    turn_step_blocks: list[float] = []
    # Step-level profiling(性能瓶颈定位)。每 phase 累积 ms,combat 结束时写 info["_prof"]
    _prof_compile = 0.0
    _prof_forward = 0.0
    _prof_step = 0.0
    _prof_post = 0.0
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
    trajectory: list[dict[str, Any]] = [] if record_trajectory else []
    # Encounter conditioning：boss/encounter id → embedding index，每步 forward 传入
    from networkV2.s1_schema.encounter_vocab import encounter_to_index
    enc_idx_value = int(encounter_to_index(encounter_id))
    enc_idx_tensor = None
    if inference_client is None:
        if net is None:
            raise ValueError("combat_rollout requires `net` when inference_client is absent")
        device = next(net.parameters()).device
        enc_idx_tensor = torch.tensor(
            [enc_idx_value], dtype=torch.long, device=device,
        )

    for _ in range(max_steps):
        legal = state.get("legal_actions", [])
        if not legal:
            break

        _t0 = time.perf_counter()
        banks = featurizer.featurize(
            state, legal,
            combat_memory=tracker.combat_memory,
            turn_prefix=tracker.turn_prefix,
            run_build_memory=tracker.run_build_memory,
            encounter_id=encounter_id.lower(),
            room_type=room_type,
        )
        _t1 = time.perf_counter()
        # Lazy init per-worker CUDA graph runner:第一次 step 拿到 banks 样本后 capture
        forward_fn = net
        if (
            inference_client is None
            and graph_runner_holder is not None
            and graph_runner_holder.get("runner") is None
            and not graph_runner_holder.get("init_attempted", False)
        ):
            graph_runner_holder["init_attempted"] = True
            graph_runner_holder["runner"] = _try_init_worker_graph_runner(
                net, banks, enc_idx_tensor,
                graph_runner_holder.get("worker_id", 0),
            )
        if inference_client is None and graph_runner_holder is not None:
            rr = graph_runner_holder.get("runner")
            if rr is not None and rr.enabled:
                forward_fn = rr
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
            with torch.inference_mode():
                out = forward_fn(banks=banks, encounter_idx=enc_idx_tensor)
                logits = out.logits[0, :len(legal)]
                mask = out.action_mask[0, :len(legal)]
                logits = torch.nan_to_num(logits.masked_fill(~mask, float("-inf")), nan=0.0)
                dist = Categorical(logits=logits)
                idx_t = logits.argmax() if greedy else dist.sample()
                lp_t = dist.log_prob(idx_t)
                value_t = out.values.fight_win.squeeze() if out.values is not None else torch.tensor(0.5, device=logits.device)
                # 单次 GPU→CPU sync 取 3 个 scalar,替代原来 3 次 .item() 单独 sync
                idx, lp, value = torch.stack([idx_t.float(), lp_t, value_t]).tolist()
                idx = int(idx)
        chosen = legal[idx]
        _t2 = time.perf_counter()

        # step
        try:
            next_state, _r, done, _info = client.step(chosen)
        except Exception as e:
            logger.warning(f"[co] step failed: {e}")
            break
        _t3 = time.perf_counter()
        _prof_compile += (_t1 - _t0) * 1000.0
        _prof_forward += (_t2 - _t1) * 1000.0
        _prof_step += (_t3 - _t2) * 1000.0

        # per-step damage
        step_damage = max(0, _enemies_total_hp(state) - _enemies_total_hp(next_state))
        step_block = max(0, _player_block_total(next_state) - _player_block_total(state))
        turn_step_damages.append(float(step_damage))
        turn_step_blocks.append(float(step_block))

        tracker.on_step(next_state, chosen, prev_state=state)

        # reward: PBRS + tactical + 通用 dense shaping；boss 专属 bonus 走 profile 开关
        next_outcome = next_state.get("run_outcome") if next_state.get("terminal") else None
        combat_won = None
        terminal_boss_damage_ratio = 0.0
        if done:
            combat_won = str(next_outcome or "").lower() == "victory"
            # Boss 败局 near-win ratio：让第一层 terminal reward 也有 near-win 渐变
            if combat_won is False and room_type == "boss" and enemy_max_hp_at_start > 0:
                terminal_boss_damage_ratio = boss_damage_ratio(next_state, enemy_max_hp_at_start)
        # 分 terminal + shaping:
        #   terminal reward (胜 +1 / 败 -1,带 hp/boss-near-win scale) 保持 1.0 权
        #   shaping reward (PBRS + tactical + dense + boss bonus + kill) 降到
        #   SHAPING_SCALE (0.3) — 避免 v7 发现的 "70 步战斗 shaping 累积 +4 超过 terminal -1"
        #   的 credit assignment bug(参见 handoff-2026-04-19-combat-v7-longrun.md 第 X 节)
        SHAPING_SCALE = 0.3
        terminal_r = combat_step_reward(
            prev_state, next_state,
            combat_won=combat_won,
            hp_at_combat_start=hp_at_start,
            boss_damage_ratio=terminal_boss_damage_ratio,
        )
        shaping_r = 0.0
        if not done:
            # combat_step_reward 在非 terminal 时返回 PBRS 小量,也归到 shaping
            # (让 done 那步的 terminal ±1 完全主导)
            shaping_r += terminal_r
            terminal_r = 0.0
        shaping_r += combat_local_tactical_reward(state, chosen, legal)
        _player_max_hp = max(int((state.get("player") or {}).get("max_hp", 1) or 1), 1)
        shaping_r += dense_combat_shaping(state, next_state, _player_max_hp)
        if reward_profile == "legacy_boss":
            # Boss 战：每 step damage 按掉血百分比额外奖励（boss HP 高，damage 珍贵）
            shaping_r += co_trainer_boss_damage_bonus(state, next_state, room_type)
            # Boss 战：Vuln/Weak 套 boss 身上额外 +0.02（放大 debuff setup 价值）
            shaping_r += co_trainer_boss_debuff_bonus(state, chosen, room_type)
        shaping_r += kill_overkill_reward(state, next_state, chosen)

        chosen_action_name = str(chosen.get("action", "")).lower()
        if chosen_action_name in ("end_turn", "end"):
            player = (state.get("player") or {})
            battle = state.get("battle") or {}
            prev_energy = int(battle.get("energy", player.get("energy", 0)) or 0)
            hand = battle.get("hand") or player.get("hand") or []
            playable = sum(1 for c in hand if isinstance(c, dict) and c.get("can_play", False))
            if prev_energy > 0 and playable > 0:
                shaping_r -= 0.10

        reward = terminal_r + shaping_r * SHAPING_SCALE

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
            run_win_target=-1.0,
            hp_loss_target=float(max(hp_at_start - cur_hp, 0)),
            survival_target=hp_ratio,
            leaf_target=0.0,
            transition_risk_target=transition_risk_t,
            resource_retention_target=resource_retention_t,
            boss_readiness_target=boss_readiness_t,
            resource_health_target=resource_health_t,
            deck_quality_target=deck_quality_t,
            turn_damage_target=-1.0,
            turn_block_target=-1.0,
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
            _backfill_turn_block(samples, turn_start_sample_idx, turn_step_blocks)
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
                # turn_end_reward 也归入 shaping,同样降 scale
                samples[-1].reward = float(samples[-1].reward) + ter * 0.3

            # 重置 turn 跟踪
            turn_start_sample_idx = len(samples)
            turn_step_damages = []
            turn_step_blocks = []
            hp_at_turn_start = hp_after_turn
            enemy_max_hp_at_turn_start = sum(
                int(e.get("max_hp", 0) or 0)
                for e in ((next_state.get("battle") or {}).get("enemies") or [])
                if isinstance(e, dict) and e.get("is_alive", True)
            ) or 1

        steps += 1
        prev_state = state
        state = next_state
        _t4 = time.perf_counter()
        _prof_post += (_t4 - _t3) * 1000.0
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

    # GAE(chain 调用方若 defer_gae=True,会把整个 chain 的 samples 拼起来统一算
    # GAE,这样 chain terminal reward 能真正回传到 early steps)
    if not defer_gae:
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
    # Step-level 性能分解(ms per step, 看瓶颈是 compile/forward/sim_step/post)
    if steps > 0:
        info["_prof"] = {
            "compile_ms": round(_prof_compile / steps, 3),
            "forward_ms": round(_prof_forward / steps, 3),
            "step_ms": round(_prof_step / steps, 3),
            "post_ms": round(_prof_post / steps, 3),
            "total_ms": round((_prof_compile + _prof_forward + _prof_step + _prof_post) / steps, 3),
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
    featurizer: DecisionFeaturizer,
    encounter_sequence: list[tuple[str, str]],
    chain_deck: dict[str, Any],
    max_steps_per_combat: int,
    seed_prefix: str,
    record_trajectory: bool = False,
    heal_after_combat: dict[int, float] = CHAIN_HEAL_AFTER_COMBAT,
    start_full_hp: bool = CHAIN_START_FULL_HP,
    abort_on_defeat: bool = CHAIN_ABORT_ON_DEFEAT,
    graph_runner_holder: dict | None = None,
    inference_client: Any | None = None,
    task_id: int = 0,
    reward_profile: str = "stochastic_stable",
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
            client, net, featurizer, eid, rt, combat_deck,
            max_steps=max_steps_per_combat,
            seed=f"{seed_prefix}-c{i}",
            record_trajectory=record_trajectory,
            graph_runner_holder=graph_runner_holder,
            reward_profile=reward_profile,
            inference_client=inference_client,
            task_id=task_id,
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


def skada_chain_combat_rollout(
    client: PipeBackedCombatTrainingClient,
    net: UnifiedNet,
    featurizer: DecisionFeaturizer,
    task_chain: list[dict[str, Any]],
    max_steps_per_combat: int,
    seed_prefix: str,
    record_trajectory: bool = False,
    abort_on_defeat: bool = False,
    use_skada_hp_each_combat: bool = False,
    graph_runner_holder: dict | None = None,
    inference_client: Any | None = None,
    task_id: int = 0,
    reward_profile: str = "stochastic_stable",
) -> tuple[list[TrainingSample], list[dict[str, Any]]]:
    """按 skada run 的真实 combat 序列顺序跑,每场还原当时的 deck/relics。

    HP 继承策略:
      - 第一场:用 task.build.current_hp(skada 起始 hp)
      - 后续 victory:用 AI 上场残血 或 skada 那时刻 hp(取 min,AI 不能吃白食)
      - 后续 defeat:重置为 skada 真实玩家那时刻的 hp,继续打下一场
        这样确保 AI 一定见过 boss/elite,不会因为前几场挂了就错过训练信号
      - abort_on_defeat=True 时,失败直接中止 chain(更严谨 RL 但 boss 信号稀疏)

    deck/relics:每场都用 task[i] 那时刻的真实 skada 状态
    (玩家可能在战斗间获得新卡/新 relic,AI 应该能用到)。
    """
    if not task_chain:
        return [], []

    all_samples: list[TrainingSample] = []
    sub_infos: list[dict[str, Any]] = []

    cur_hp: int | None = None
    # 记录每场 combat 在 all_samples 里的 (start, end) 范围,用于 chain-level
    # reward 分配 + 统一 GAE
    combat_ranges: list[tuple[int, int]] = []
    for i, task in enumerate(task_chain):
        build = dict(task["build"])
        skada_hp = int(build.get("current_hp", build.get("max_hp", 80)))
        if i == 0 or use_skada_hp_each_combat or cur_hp is None:
            # 第一场或 per-combat HP 模式:用 skada 真实 hp
            build["current_hp"] = skada_hp
        else:
            # 残血继承:AI 真实血 和 skada 当时 hp 取 min
            # (AI 比 skada 强时 hp 不应该超过 skada 真实玩家;AI 比 skada 弱时用残血)
            build["current_hp"] = max(1, min(cur_hp, skada_hp))

        seed = f"{seed_prefix}-c{i}"
        start_idx = len(all_samples)
        samples, info = combat_rollout(
            client, net, featurizer,
            task["encounter_id"], task["room_type"], build,
            max_steps=max_steps_per_combat, seed=seed,
            record_trajectory=record_trajectory,
            graph_runner_holder=graph_runner_holder,
            reward_profile=reward_profile,
            defer_gae=True,  # chain 结束后统一算 GAE
            inference_client=inference_client,
            task_id=task_id,
        )
        info["chain_index"] = i
        info["chain_total"] = len(task_chain)
        info["hp_enter"] = int(build.get("current_hp", 0))
        info["ref_floor"] = task.get("ref_floor", -1)
        info["run_id"] = task.get("run_id", -1)
        sub_infos.append(info)
        all_samples.extend(samples)
        combat_ranges.append((start_idx, len(all_samples)))

        if info.get("outcome") != "victory":
            if abort_on_defeat:
                break
            # 不 abort:下场重置为 skada 那时刻 hp,AI 继续见 boss/elite
            cur_hp = skada_hp
        else:
            cur_hp = int(info.get("final_hp", cur_hp or skada_hp))

    # ---- Chain-level terminal reward(credit assignment 跨 combat) ----
    # 把"AI 在 chain 里打到多深 + 累计省了多少血"折算成 chain-level reward,
    # 按 chain_index 递增权重分配给每场 combat 的最后一步,然后对整个 chain 的
    # all_samples 统一算 GAE。这样:
    #   - AI 在早期 combat 放血 → 后期 combat hp_enter 低 → chain 胜率低
    #     → chain_terminal 扣分 → 回传到早期 step 的 value_target/advantage
    #   - AI 打到越深 chain,越多场的 terminal reward bonus
    if combat_ranges and all_samples:
        chain_total = len(task_chain)
        n_combats_played = len(sub_infos)
        n_wins = sum(1 for info in sub_infos if info.get("outcome") == "victory")
        deepest_floor = max(
            (int(info.get("ref_floor", 0) or 0) for info in sub_infos if info.get("outcome") == "victory"),
            default=0,
        )
        # 以 skada 真实玩家 run 的最深 floor 为 100% baseline
        max_ref_floor = max(
            (int(t.get("ref_floor", 0) or 0) for t in task_chain),
            default=1,
        )
        # Floor bonus:打到越深越好,非线性放大深 floor 价值
        floor_ratio = deepest_floor / max(max_ref_floor, 1)
        floor_bonus = (floor_ratio ** 1.5) * 3.0  # chain 全通 ≈ +3.0
        # Chain completion bonus:全胜整个 chain → 额外 +2.0(稀有奖励)
        chain_complete_bonus = 2.0 if n_wins == chain_total else 0.0
        # HP conservation penalty:累计掉血比 / skada 玩家累计掉血比。
        # 如果 AI 累计比玩家掉血更多 → 扣分
        total_hp_loss = sum(int(info.get("hp_loss", 0) or 0) for info in sub_infos)
        ref_max_hp = int(sub_infos[0].get("max_hp", 75) or 75)
        hp_loss_ratio = total_hp_loss / max(ref_max_hp, 1)
        hp_penalty = -min(hp_loss_ratio, 2.0) * 1.5  # 损 2× max_hp → -3.0

        chain_terminal = floor_bonus + chain_complete_bonus + hp_penalty

        # 分配:按 chain_index 权重递增(后期 combat 拿更多),最后一场拿最多
        # weight_i = (i+1)^2 / sum((k+1)^2)
        weights_sum = sum((k + 1) ** 2 for k in range(len(combat_ranges)))
        for i, (_, end_idx) in enumerate(combat_ranges):
            if end_idx <= 0:
                continue
            w = ((i + 1) ** 2) / max(weights_sum, 1)
            last_sample_idx = end_idx - 1
            if 0 <= last_sample_idx < len(all_samples):
                all_samples[last_sample_idx].reward += chain_terminal * w

    # 统一 GAE:跨整个 chain 当作一个 trajectory,让 chain_terminal 回传到
    # 早期 combat 的 step(真正的跨 combat credit assignment)
    _compute_gae_combat(all_samples)

    return all_samples, sub_infos


def build_chain_deck(rng: random.Random, character: str = "IRONCLAD") -> dict[str, Any]:
    """整个 chain 共用的 deck。Fallback chain:
      1. skada_victory_decks(18K 真实玩家 final_deck,按 character 过滤)
      2. real_boss_decks(Artifacts/combat_teacher,老 AI teacher)
      3. buffed_ironclad_deck(hardcoded fallback)
    """
    # 优先 1:skada 真实玩家 deck(多样性最好)
    skada_decks = _load_skada_victory_decks_cached(character=character)
    if skada_decks:
        chosen = rng.choice(skada_decks)
        return {
            "deck": list(chosen["deck"]),
            "relics": list(chosen.get("relics", [])),
            "max_hp": chosen.get("max_hp", 80),
            "current_hp": chosen.get("current_hp", chosen.get("max_hp", 80)),
            "gold": chosen.get("gold", 99),
            "max_energy": chosen.get("max_energy", 3),
            "_source": "skada_victory",
        }
    # fallback 2:Artifacts/combat_teacher
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
            "_source": "combat_teacher",
        }
    # fallback 3:hardcoded
    d = buffed_ironclad_deck()
    d["_source"] = "buffed_hardcoded"
    return d


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


def _try_init_worker_graph_runner(net, banks_sample, enc_idx_sample, worker_id: int):
    """懒初始化 per-worker CUDA graph runner。

    每 worker 独立 static buffer + graph,避免多 thread race。

    行为依赖 net._cuda_graph_cfg.strict(默认 True):
      - strict=True: capture 失败 → raise (GraphCaptureFailedError/ GraphBankUndeclaredError)
                     让训练立刻挂,强迫修代码而不是静默降级
      - strict=False: 失败降级 eager 并打印 warning(需要显式接受损失 5-10x 加速)

    GraphBankUndeclaredError 永远 raise,无论 strict(因为 fallback 也解决不了漏配 bank)。
    """
    cfg = getattr(net, "_cuda_graph_cfg", None)
    if cfg is None:
        return None
    from networkV2.s5_net.graph_runner import (
        GraphRunner,
        GraphBankUndeclaredError,
        GraphCaptureFailedError,
    )
    strict = bool(cfg.get("strict", True))
    try:
        runner = GraphRunner(
            net, banks_sample, enc_idx_sample,
            parity_check_every=int(cfg.get("parity_check_every", 500)),
            atol=float(cfg.get("atol", 1e-3)),
            rtol=float(cfg.get("rtol", 1e-3)),
            startup_parity_n=int(cfg.get("startup_parity_n", 10)),
            startup_parity_noise=float(cfg.get("startup_parity_noise", 0.0)),
            strict=strict,
        )
    except GraphBankUndeclaredError:
        raise
    except GraphCaptureFailedError:
        raise
    except Exception as e:
        if strict:
            raise
        logger.warning(
            f"[cuda-graph] worker {worker_id} init failed: {type(e).__name__}: "
            f"{str(e)[:150]}. Fallback eager (strict=False)."
        )
        return None

    if runner.enabled:
        logger.info(f"[cuda-graph] worker {worker_id} runner OK")
        return runner
    # enabled=False 且上面没 raise → 唯一可能就是 strict=False 降级
    logger.warning(
        f"[cuda-graph] worker {worker_id} disabled (strict=False fallback). "
        f"训练将以 eager 跑,QPS 会显著降低。"
    )
    return None


def _worker_collect(
    worker_id: int,
    pool: CombatClientPool,
    net: UnifiedNet,
    tasks: list,  # 两种形态：
                  #   single-combat: (enc_id, rt, deck, seed, record_traj)
                  #   chained:       ("chain", sequence, chain_deck, seed_prefix, record_traj)
    max_steps: int,
    result_q: queue.Queue,
    graph_runner_holder: dict | None = None,  # per-worker runner cache(thread-local)
    reward_profile: str = "stochastic_stable",
) -> None:
    featurizer = DecisionFeaturizer()
    client = pool.get(worker_id)
    # Per-worker CUDA graph holder(thread-local);first rollout 时 lazy init
    if graph_runner_holder is None:
        graph_runner_holder = {"worker_id": worker_id, "runner": None, "init_attempted": False}
    samples_out: list[TrainingSample] = []
    infos: list[dict] = []
    for task in tasks:
        try:
            if task and task[0] == "chain":
                _tag, sequence, chain_deck, seed_prefix, record_traj = task
                samples, sub_infos = chained_combat_rollout(
                    client, net, featurizer, sequence, chain_deck,
                    max_steps_per_combat=max_steps,
                    seed_prefix=seed_prefix,
                    record_trajectory=record_traj,
                    graph_runner_holder=graph_runner_holder,
                    reward_profile=reward_profile,
                )
                samples_out.extend(samples)
                infos.extend(sub_infos)
            elif task and task[0] == "skada_chain":
                _tag, task_chain, seed_prefix, record_traj = task
                samples, sub_infos = skada_chain_combat_rollout(
                    client, net, featurizer, task_chain,
                    max_steps_per_combat=max_steps,
                    seed_prefix=seed_prefix,
                    record_trajectory=record_traj,
                    graph_runner_holder=graph_runner_holder,
                    reward_profile=reward_profile,
                )
                samples_out.extend(samples)
                infos.extend(sub_infos)
            else:
                enc_id, rt, deck, seed, record_traj = task
                samples, info = combat_rollout(
                    client, net, featurizer, enc_id, rt, deck,
                    max_steps=max_steps, seed=seed,
                    record_trajectory=record_traj,
                    graph_runner_holder=graph_runner_holder,
                    reward_profile=reward_profile,
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
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # 兼容多种 checkpoint 格式:
        #   {"net": ...} (cotrainer 自己保存的老格式)
        #   {"model_state": ..., "epoch": ..., ...}(BC / train_noncombat_offline 保存)
        #   直接 state_dict(裸)
        if isinstance(state, dict):
            if "net" in state:
                state = state["net"]
            elif "model_state" in state:
                state = state["model_state"]
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

        # BC 从来没训 turn_damage_lookahead head（BC target 里没这个信号）,
        # 加载 BC checkpoint 后 softplus head 里残留 random init,forward 可能输出
        # 1e7 量级 → vl_turn_damage 爆到 1e7,拖爆 total_loss。
        # 对最后一层 Linear 重 init 到 small + bias=0,让 softplus 输出 ~log(2)≈0.7 起步。
        try:
            import torch.nn as _nn
            td_head = net.value_heads.turn_damage
            last_linear = td_head.proj[-1]
            _nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
            _nn.init.zeros_(last_linear.bias)
            logger.info("[turn_damage] Reset turn_damage head last-linear (avoid softplus explosion)")
        except Exception as _e:
            logger.debug(f"turn_damage head reset skipped: {_e}")

    # torch.compile 用于 rollout 加速 (batch=1 eager 22ms → compile 后 3-8ms)
    # 注意:trainer 仍用原 `net`(training 需要 autograd 反传,compile 对 backward 支持有限)
    if getattr(args, "compile_net", False):
        try:
            compiled_net = torch.compile(
                net, mode=args.compile_mode, dynamic=True,
            )
            logger.info(
                f"[compile] torch.compile mode={args.compile_mode} dynamic=True enabled (rollout only)"
            )
        except Exception as e:
            logger.warning(f"[compile] torch.compile failed: {e}; rollout uses eager")
            compiled_net = net
    else:
        compiled_net = net

    # CUDA graph 用于 rollout 加速 (Windows-friendly,不需要 triton)。预期 5-10x forward 加速。
    # GraphRunner 包一层 wrapper,硬检测 shape/parity drift,capture 失败自动 fallback。
    cuda_graph_runner = None
    if getattr(args, "use_cuda_graph", False):
        from networkV2.s5_net.graph_runner import GraphRunner, patch_dropout_for_graph_safety
        import torch.nn as _nn
        # PyTorch issue #99820 防污染:F.dropout p=0 短路 patch (进程全局)
        patch_dropout_for_graph_safety()
        # 双保险:永久关 net dropout (rollout+training 都 0)。RL fine-tune 关 dropout
        # 影响极小;但 cuda graph RNG state 污染训练 100% 挂(issue #99820)。
        _dcount = 0
        for m in net.modules():
            if isinstance(m, _nn.MultiheadAttention):
                m.dropout = 0.0
                _dcount += 1
            elif isinstance(m, _nn.Dropout):
                m.p = 0.0
                _dcount += 1
        logger.info(
            f"[cuda-graph] dropout disabled globally ({_dcount} modules); "
            f"F.dropout(p=0) short-circuited"
        )
        net._cuda_graph_cfg = {
            "parity_check_every": int(args.graph_parity_every),
            "atol": float(args.graph_atol),
            "rtol": float(args.graph_atol),
        }
        logger.info(
            f"[cuda-graph] enabled (parity_every={args.graph_parity_every}, "
            f"atol={args.graph_atol}). 首次 rollout 时 capture,~5s warmup。"
        )

    # 新异步 rollout 引擎默认启用；旧线程模型只保留隐藏 fallback 开关。
    args.rollout_graph_enabled = bool(
        getattr(args, "rollout_graph_enabled", True) and getattr(args, "use_cuda_graph", True)
    )
    rollout_cfg = build_rollout_engine_config(
        args,
        max_numeric_dim=int(getattr(net.tokenizer, "max_numeric_dim", 58)),
    )
    n_collectors = int(rollout_cfg.rollout_num_actors)

    trainer = UnifiedPPOTrainer(net, PPOConfig(
        lr=args.lr, ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        value_warmup_iters=args.value_warmup_iters,
        target_kl=args.target_kl,
    ))

    pool: CombatClientPool | None = None
    runtime = None
    catalog_client = None
    if rollout_cfg.use_legacy_thread_rollout:
        pool = CombatClientPool(args.base_port, n_collectors)
        try:
            from networkV2.s1_schema.sim_catalog import GAME_CATALOG
            GAME_CATALOG.attach_sim(pool.clients[0])
            logger.info("Attached sim API to GAME_CATALOG (legacy thread rollout)")
        except Exception as e:
            logger.warning(f"Failed to attach sim API, fallback to sqlite: {e}")
    else:
        runtime = create_cotrainer_runtime(
            args=args,
            net=compiled_net,
            rollout_cfg=rollout_cfg,
        )
        helper_port = int(args.base_port) + n_collectors + 100
        try:
            catalog_client = open_cotrainer_catalog_client(helper_port)
            from networkV2.s1_schema.sim_catalog import GAME_CATALOG
            GAME_CATALOG.attach_sim(catalog_client)
            logger.info(
                "Attached sim API to GAME_CATALOG (async rollout, helper_port=%s)",
                helper_port,
            )
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
            "rollout_num_actors": n_collectors,
            "rollout_async_default": float(not rollout_cfg.use_legacy_thread_rollout),
            "base_port": args.base_port,
        })

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    use_chained = not bool(getattr(args, "no_chained_episodes", False))
    use_skada_replay = bool(getattr(args, "skada_replay_index_db", None))
    chain_len = sum(c for _, c in CHAIN_STRUCTURE)

    print()
    print(
        f"Config: preset={args.preset} lr={args.lr} eps/iter={args.episodes_per_iter} "
        f"actors={n_collectors} async={not rollout_cfg.use_legacy_thread_rollout} "
        f"reward={args.reward_profile}"
    )
    if use_skada_replay:
        print(f"Skada chain replay ENABLED: 1 ep = 1 victory run 的全部 combat 按 floor 顺序")
        print(f"  每场 reset 还原当时 deck/relics,HP 跨场继承(败则用 skada 当时 hp 重置续跑)")
        print(f"  n_runs={args.skada_replay_n_runs}; avg ~18 combats/run(act1→boss)")
    elif use_chained:
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

    # Sim 健康监控:连续 empty iter 累计,超阈值尝试重启 pool;再失败 abort
    consecutive_empty_iters = 0
    MAX_CONSECUTIVE_EMPTY = 2          # 连续 N iter 没 sample → 重启
    MAX_TOTAL_EMPTY_ABORTS = 3         # 总共重启 N 次还 empty → 彻底 abort
    total_empty_aborts = 0
    # Per-worker CUDA graph holder 跨 iter 持久化(避免每 iter 重 capture 累积 RNG 污染)
    _worker_graph_holders: dict[int, dict] = {}

    for iteration in range(1, args.max_iterations + 1):
        t0 = time.time()
        eps_total = args.episodes_per_iter

        # 分配 rollout tasks
        # record_trajectory 逻辑（优先级从高到低）：
        #   1. --record-trajectory → 全量记录每 ep（~1-3 MB/iter）
        #   2. --record-trajectory-every N → 采样 N 条（优先 boss/elite）
        #   3. 默认都不记
        record_all = bool(getattr(args, "record_trajectory", False))
        n_record = max(0, int(getattr(args, "record_trajectory_every", 0) or 0))
        iter_tasks: list[Any] = []

        if getattr(args, "skada_replay_index_db", None):
            # Skada chain replay 模式:抽 skada victory run,每 ep = 1 run 的全部战斗
            # 按 floor 顺序打,每场战斗 reset 时还原当时真实 deck/relics,HP 跨场继承。
            # 天然解决 deck-encounter 难度匹配 + 真实 build 演化 + curriculum 渐进。
            if "_skada_chains_pool" not in locals():
                from networkV2.s6_training.skada_index_dataset import SkadaIndexFetcher
                from networkV2.s6_training.skada_combat_replay import (
                    sample_combat_chains, chain_stats, load_sim_supported_lists,
                )
                # 从 sim 权威 API 拿支持的 encounter/card/relic 白名单,
                # 过滤 skada 里(多人模式卡 MP_*/老版本 encounter TOADPOLES_NORMAL/
                # mod relic EXTRARELICS-*/错分类 event encounter) 等不兼容数据。
                # 数据源不改,清洗在 cache load 阶段做,训练期永不 hit unsupported。
                if pool is not None:
                    _supported = load_sim_supported_lists(pool.clients[0])
                elif catalog_client is not None:
                    _supported = load_sim_supported_lists(catalog_client)
                else:
                    raise RuntimeError("no sim client available for skada support discovery")
                _sf = SkadaIndexFetcher(
                    index_db=Path(args.skada_replay_index_db),
                )
                _skada_chains_pool = sample_combat_chains(
                    _sf, n_runs=int(args.skada_replay_n_runs or 100),
                    require_map_acts=True, seed=args.seed,
                    supported_encounters=_supported.encounters or None,
                    supported_cards=_supported.cards or None,
                    supported_relics=_supported.relics or None,
                )
                logger.info(f"skada chain pool: {chain_stats(_skada_chains_pool)}")
                _sf.close()
            # 每 ep 抽一个 run chain(整个 run 的全部 combat 顺序打)
            ep_chains = [rng.choice(_skada_chains_pool) for _ in range(eps_total)] \
                if _skada_chains_pool else []
            for i, chain in enumerate(ep_chains):
                seed_prefix = f"co-{iteration}-{i}-{rng.getrandbits(32):08x}"
                iter_tasks.append(("skada_chain", chain, seed_prefix, record_all))
        elif use_chained:
            # Chained 模式：每 ep 一个 chain（3m+1e+1b），整个 chain 用同一 deck
            for i in range(eps_total):
                seq = build_chain_sequence(rng)
                chain_deck = build_chain_deck(rng)
                seed_prefix = f"co-{iteration}-{i}-{rng.getrandbits(32):08x}"
                # 全量记录 → chain 内所有 sub_combat 都记；否则都不记（采样模式对 chain 不适用）
                record_traj = record_all
                iter_tasks.append(("chain", seq, chain_deck, seed_prefix, record_traj))
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
                iter_tasks.append((enc_id, rt, ep_deck, seed, record_traj))

        # 并发收集
        net.eval()
        iter_samples: list[TrainingSample] = []
        iter_infos: list[dict] = []
        rollout_stats: dict[str, Any] = {}
        if rollout_cfg.use_legacy_thread_rollout:
            result_q: queue.Queue = queue.Queue()
            threads = []
            tasks_per_worker: list[list[Any]] = [[] for _ in range(n_collectors)]
            for idx, task in enumerate(iter_tasks):
                tasks_per_worker[idx % n_collectors].append(task)
            for w_idx in range(n_collectors):
                if not tasks_per_worker[w_idx]:
                    continue
                if w_idx not in _worker_graph_holders:
                    _worker_graph_holders[w_idx] = {
                        "worker_id": w_idx, "runner": None, "init_attempted": False,
                    }
                t = threading.Thread(
                    target=_worker_collect,
                    args=(w_idx, pool, compiled_net, tasks_per_worker[w_idx],
                          args.max_steps, result_q,
                          _worker_graph_holders[w_idx], args.reward_profile),
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join(timeout=300)
            while not result_q.empty():
                r = result_q.get()
                iter_samples.extend(r["samples"])
                iter_infos.extend(r["infos"])
        else:
            if runtime is None:
                raise RuntimeError("async rollout runtime was not initialized")
            task_ids = runtime.submit_tasks(iter_tasks)
            envelopes = runtime.gather_results(len(task_ids))
            for env in envelopes:
                iter_samples.extend(env.samples)
                iter_infos.extend(env.infos)
            rollout_stats = runtime.stats()

        # 按难度分类胜率 + sim error 计数
        wins_by_rt = {"monster": 0, "elite": 0, "boss": 0}
        total_by_rt = {"monster": 0, "elite": 0, "boss": 0}
        sim_error_count = 0
        sim_error_types: dict[str, int] = {}
        # 性能 profiling 汇总(步数加权平均各 phase 耗时 ms)
        _prof_agg = {"compile_ms": 0.0, "forward_ms": 0.0, "step_ms": 0.0, "post_ms": 0.0}
        _prof_total_steps = 0
        for info in iter_infos:
            rt = info.get("room_type", "")
            if rt in total_by_rt:
                total_by_rt[rt] += 1
                if info.get("outcome") == "victory":
                    wins_by_rt[rt] += 1
            # sim error 分类统计(WriteFile failed / NPE / invalid action 等)
            if info.get("outcome") == "error":
                sim_error_count += 1
                err_msg = str(info.get("error", ""))[:60]
                sim_error_types[err_msg] = sim_error_types.get(err_msg, 0) + 1
            # 性能汇总
            pr = info.get("_prof")
            nstep = int(info.get("steps", 0) or 0)
            if isinstance(pr, dict) and nstep > 0:
                for k in ("compile_ms", "forward_ms", "step_ms", "post_ms"):
                    _prof_agg[k] += float(pr.get(k, 0.0)) * nstep
                _prof_total_steps += nstep

        def _wr(rt): return wins_by_rt[rt] / max(total_by_rt[rt], 1)

        # Sim 健康检查:如果 iter 整个 0 有效 combat 或 error 爆 → 尝试重启 pool
        n_valid_combats = sum(total_by_rt.values())
        if n_valid_combats == 0 and len(iter_infos) > 0:
            consecutive_empty_iters += 1
            logger.error(
                f"[sim-health] iter {iteration}: 0 valid combats, {sim_error_count} errors. "
                f"top errors: {sorted(sim_error_types.items(), key=lambda x: -x[1])[:3]}"
            )
            if consecutive_empty_iters >= MAX_CONSECUTIVE_EMPTY:
                total_empty_aborts += 1
                if total_empty_aborts >= MAX_TOTAL_EMPTY_ABORTS:
                    logger.error(
                        f"[sim-health] {MAX_TOTAL_EMPTY_ABORTS} recovery attempts failed, ABORTING training"
                    )
                    break
                logger.warning(
                    f"[sim-health] {consecutive_empty_iters} consecutive empty iters, "
                    f"restarting sim pool (attempt {total_empty_aborts}/{MAX_TOTAL_EMPTY_ABORTS})"
                )
                if runtime is not None:
                    runtime.shutdown()
                    time.sleep(2)
                    runtime = create_cotrainer_runtime(
                        args=args,
                        net=compiled_net,
                        rollout_cfg=rollout_cfg,
                    )
                if pool is not None:
                    pool.close_all()
                    time.sleep(3)
                    pool = CombatClientPool(args.base_port, n_collectors)
                    try:
                        from networkV2.s1_schema.sim_catalog import GAME_CATALOG
                        GAME_CATALOG.attach_sim(pool.clients[0])
                    except Exception as _e:
                        logger.warning(f"re-attach GAME_CATALOG failed: {_e}")
                if catalog_client is not None:
                    try:
                        catalog_client.close()
                    except Exception:
                        pass
                    helper_port = int(args.base_port) + n_collectors + 100
                    try:
                        catalog_client = open_cotrainer_catalog_client(helper_port)
                        from networkV2.s1_schema.sim_catalog import GAME_CATALOG
                        GAME_CATALOG.attach_sim(catalog_client)
                    except Exception as _e:
                        logger.warning(f"re-attach helper catalog client failed: {_e}")
                consecutive_empty_iters = 0
                # 跳过本 iter 训练
                metrics = {"policy_loss": 0.0, "value_loss": 0.0}
                wall = time.time() - t0
                print(f" {iteration:4d} | sim-recover | {len(iter_infos)} errors | restart-pool | {wall:.1f}s")
                continue
        else:
            consecutive_empty_iters = 0

        # 训练
        if len(iter_samples) >= args.min_update_samples:
            net.train()
            metrics = trainer.train_step(iter_samples)
        else:
            metrics = {"policy_loss": 0.0, "value_loss": 0.0}
        # 把 sim 错误信息进 metrics
        metrics["sim_error_count"] = float(sim_error_count)
        metrics["sim_error_rate"] = float(sim_error_count) / max(len(iter_infos), 1)
        if rollout_stats:
            metrics.update(runtime_stats_to_metrics("rollout", rollout_stats))
        # 把性能 profile 进 metrics(步数加权平均每 step 各 phase 的 ms)
        if _prof_total_steps > 0:
            for k, v in _prof_agg.items():
                metrics[f"prof_{k}"] = v / _prof_total_steps
            metrics["prof_total_ms"] = sum(_prof_agg.values()) / _prof_total_steps

        wall = time.time() - t0
        total_wins = sum(wins_by_rt.values())
        total_runs = sum(total_by_rt.values())

        err_flag = f" ERR={sim_error_count}" if sim_error_count > 0 else ""
        line = (
            f"{iteration:5d} | {total_runs:3d} | {len(iter_samples):5d} | {total_wins}/{total_runs} | "
            f"{100*_wr('monster'):5.1f}% / {100*_wr('elite'):5.1f}% / {100*_wr('boss'):5.1f}% | "
            f"pl={metrics.get('policy_loss',0):.4f} vl={metrics.get('value_loss',0):.3f} "
            f"kl={metrics.get('approx_kl',0):.4f} ep={int(metrics.get('epochs_done',0))}{err_flag} | "
            f"{wall:5.1f}s"
        )
        print(line)
        # ERR 分类:每次 iter 有 ERR 都打印 top 3,便于定位 sim crash 根因
        if sim_error_count > 0:
            top = sorted(sim_error_types.items(), key=lambda x: -x[1])[:3]
            top_str = "; ".join(f"[{cnt}x] {msg}" for msg, cnt in top)
            logger.warning(f"[sim-errors] iter {iteration}: {top_str}")
        if rollout_stats:
            logger.info(
                "[rollout_async] iter=%s req=%s batches=%s graph_hits=%s eager=%s "
                "queue_wait=%.3fms infer=%.3fms batch_hist=%s",
                iteration,
                int(rollout_stats.get("requests", 0)),
                int(rollout_stats.get("batches", 0)),
                int(rollout_stats.get("graph_hits", 0)),
                int(rollout_stats.get("eager_fallbacks", 0)),
                float(rollout_stats.get("queue_wait_ms_avg", 0.0)),
                float(rollout_stats.get("infer_ms_avg", 0.0)),
                rollout_stats.get("batch_hist", {}),
            )

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
                        len(locals().get("_skada_chains_pool", [])) if getattr(args, "skada_replay_index_db", None)
                        else (len(pool_encs) if (not use_chained and "pool_encs" in locals())
                              else sum(c for _, c in CHAIN_STRUCTURE))
                    ),
                    "chained_episodes": use_chained,
                },
            )

        # save
        if iteration % args.save_every == 0:
            torch.save(net.state_dict(), out_dir / f"cotrainer_iter{iteration}.pt")

    if runtime is not None:
        runtime.shutdown()
    if pool is not None:
        pool.close_all()
    if catalog_client is not None:
        try:
            catalog_client.close()
        except Exception:
            pass
    torch.save(net.state_dict(), out_dir / "cotrainer_final.pt")
    print(f"Done. Saved {out_dir/'cotrainer_final.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="slim")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="兼容旧参数；默认映射到 --rollout-num-actors。")
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
    ap.add_argument("--skada-replay-index-db", type=str, default="",
                    help="启用 skada replay 模式:路径指 skada_runs.sqlite 索引。"
                         "开启后 rollout 用 skada victory runs 里的真实 combat (encounter, build) 作 task,"
                         "自动 curriculum + deck 难度匹配。和 chain/curriculum 模式互斥。")
    ap.add_argument("--skada-replay-n-runs", type=int, default=100,
                    help="skada replay 抽多少 run(每 run 产 ~15-20 combat task)。100 runs → ~2K task pool。")
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
    ap.add_argument("--reward-profile", type=str, default="stochastic_stable",
                    choices=["stochastic_stable", "legacy_boss"],
                    help="combat reward profile。stochastic_stable=通用 dense shaping；"
                         "legacy_boss=额外启用 boss damage/debuff bonus。")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--compile-net", action="store_true",
                    help="torch.compile(net) for inference — 消除 batch=1 launch overhead,"
                         " 首次 forward 会 trace/编译 10-30s,之后 3-5x 加速 (需要 PyTorch 2.0+)")
    ap.add_argument("--compile-mode", type=str, default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode; reduce-overhead 对 RL inference 最合适")
    ap.add_argument("--use-cuda-graph",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="rollout forward 用 CUDA graph (Windows-friendly,不需要 triton)。"
                         " 默认启用;关掉用 --no-use-cuda-graph。"
                         " 首次 capture ~1s/worker(进程全局锁串行),稳态 replay 预期"
                         " 3-10x forward 加速,多 worker 并发 replay 不持锁。"
                         " 有硬检测:shape/periodic parity,drift 会抛异常。"
                         " capture 失败自动 fallback 到 eager (不阻塞训练)。")
    ap.add_argument("--graph-parity-every", type=int, default=500,
                    help="CUDA graph 周期性 parity check 间隔(0=禁用)")
    ap.add_argument("--graph-atol", type=float, default=1e-3,
                    help="CUDA graph vs eager logits 允许的绝对误差")
    add_rollout_engine_args(ap)
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run_cotrainer(args)


if __name__ == "__main__":
    main()
