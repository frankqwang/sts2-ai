"""Skada offline loader:JSONL/SQLite → list[TrainingSample]。

输入:skada run record(jsonl/sqlite)
输出:networkV2 TrainingSample 列表,供 train_noncombat_offline.py 直接用。

产出 sample 类型:
- card_reward:     was_picked 硬标签,监督 policy_head
- relic_reward:    was_picked,监督 policy_head(relic-specific)
- ancient_choice:  source_kind='ancient_choices',同上
- campfire:        campfire_choice 硬标签
- shop:            shop_actions(简化版:一次聚合 sample,不拆 purchase/remove)
- map_route:       从 visited_coords[i] → visited_coords[i+1] 选 child
- value_only:      每个 floor 一个,仅监督 run_win/deck_quality/boss_readiness/resource_health

所有样本共享同一个 RunBuildMemory 计算器,decision_domain 分别为
"card_reward" / "event" / "rest" / "shop" / "route" / "noncombat"。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from networkV2.s1_schema.actions import ActionCandidate
from networkV2.s1_schema.token_banks import UnifiedTokenBanks
from networkV2.s4_featurization.token_bank_builder import TokenBankBuilder
from networkV2.s6_training.batch import TrainingSample
from networkV2.s6_training.skada_state_rebuilder import (
    SkadaRunState, BanksContext, iter_timeline_with_state,
)
from networkV2.s6_training.skada_id_mapping import (
    normalize_card_id, normalize_relic_id,
    is_known_card, is_known_relic,
    mask_map_with_visibility, MAP_VISIBILITY_DEPTH, UNKNOWN_NODE_TYPE,
    room_letter_to_domain,
)

# Skada priors (singleton):查 cards.win_rate_delta / pick_rate / synergy 等给 option 做 inductive bias
from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=1)
def _get_skada_priors():
    """懒加载 + 单例 skada priors(419 MB sqlite,只读一次)。

    路径:显式指向 STS2AI/Assets/datasets/skada/skada_analytics.sqlite。
    V1 skada_priors.py 的 _DEFAULT_DB 用 parents[2] 指到 Python/Assets/,
    但实际 DB 在 STS2AI/Assets/ 下,差一层。这里显式构造正确 path。
    """
    try:
        from data.skada.skada_priors import SkadaPriors
        # __file__ = networkV2/s6_training/skada_offline_loader.py
        # parents[0]=s6_training, [1]=networkV2, [2]=Python, [3]=STS2AI
        db_path = Path(__file__).resolve().parents[3] / "Assets" / "datasets" / "skada" / "skada_analytics.sqlite"
        sp = SkadaPriors(db_path=db_path)
        if not sp.loaded:
            logger.warning(f"SkadaPriors DB not loaded from {db_path}, option prior fields will be 0")
        else:
            logger.info(f"SkadaPriors loaded: {sp.num_cards} cards, {sp.num_relics} relics, {sp.num_synergies} synergies")
        return sp
    except Exception as e:
        logger.warning(f"skada priors unavailable: {e}")
        return None


logger = logging.getLogger(__name__)

_bank_builder = TokenBankBuilder()


# ---------------------------------------------------------------------------
# Sample weight 策略
# ---------------------------------------------------------------------------

def _sample_weight(run_state: SkadaRunState) -> float:
    """根据 run 的 is_victory × ascension 计算 sample weight。

    - 赢的 run:基础 0.5,asc 加权最高到 1.0
    - 输的 run:固定 0.3,仍然学但信号弱
    策略可后续迭代(AWR hard filter 等)。
    """
    if run_state.is_victory:
        return 0.5 + 0.5 / (1.0 + math.exp(-(run_state.ascension - 10) / 3.0))
    return 0.3


def _run_win_target(run_state: SkadaRunState) -> float:
    """run_win_prob 监督信号:输=0,赢=1。"""
    return 1.0 if run_state.is_victory else 0.0


def _deck_quality_target(run_state: SkadaRunState) -> float:
    """deck_quality ∈ [-1,1]。

    MVP 简易版:
    - 基础 = tanh((deck_size_diff_from_15) × -0.1)(越接近 15 张越好,太大扣分)
    - is_victory 加 bonus
    - curse_density 扣分
    TODO: 用 skada cards.win_rate_delta 聚合。
    """
    deck_size = max(len(run_state.deck), 1)
    # id 都是 lower snake,checks 也统一 lower
    curse = sum(
        1 for c in run_state.deck
        if "curse" in c.lower() or c.lower() == "ascenders_bane"
    ) / deck_size
    base = math.tanh(-(deck_size - 15) * 0.08)
    vic_bonus = 0.3 if run_state.is_victory else -0.1
    return max(-1.0, min(1.0, base + vic_bonus - 0.5 * curse))


def _boss_readiness_target(run_state: SkadaRunState) -> float:
    """boss_readiness ∈ [0,1]。

    MVP:hp_ratio × (is_victory ? 1.0 : 0.4),floor 深度加成。
    TODO: 接入真实 boss 战胜负。
    """
    hp_ratio = run_state.hp / max(run_state.max_hp, 1)
    floor_weight = min(run_state.floor / 50.0, 1.0)
    base = 0.5 * hp_ratio + 0.3 * floor_weight
    if run_state.is_victory:
        base += 0.2
    return max(0.0, min(1.0, base))


def _resource_health_target(run_state: SkadaRunState) -> float:
    """resource_health ∈ [0,1]。"""
    hp_ratio = run_state.hp / max(run_state.max_hp, 1)
    gold_ratio = min(run_state.gold / 300.0, 1.0)
    potion_ratio = min(len(run_state.potions) / 3.0, 1.0)
    return max(0.0, min(1.0, 0.5 * hp_ratio + 0.3 * gold_ratio + 0.2 * potion_ratio))


# ---------------------------------------------------------------------------
# Common: banks 组装 + TrainingSample 包装
# ---------------------------------------------------------------------------

def _compute_boss_checkpoints(rec: dict[str, Any]) -> list[tuple[int, float]]:
    """预扫整条 run,返回 [(boss_floor, deck_quality_at_that_floor), ...]。

    用于档 2:给每个 sample 填 future_dq_target = 选完后下个 boss 入口的 dq。
    """
    checkpoints: list[tuple[int, float]] = []
    state = SkadaRunState.from_run_record(rec)
    for fd in rec.get("floor_timeline") or []:
        state.apply_floor_pre(fd)
        state.apply_floor_post(fd)
        cmb = fd.get("combat") or {}
        # enc_type='B' 是显式 boss;同时兼容 floor ∈ {17, 34, 48} 启发式
        is_boss = (str(cmb.get("enc_type", "")).upper() == "B") or (state.floor in (17, 34, 48))
        if is_boss:
            checkpoints.append((state.floor, _deck_quality_target(state)))
    return checkpoints


def _next_boss_dq(current_floor: int, checkpoints: list[tuple[int, float]]) -> float:
    """找 current_floor 之后最近一个 boss 的 dq。没有未来 boss → -2(无效,loss skip)。"""
    for bf, dq in checkpoints:
        if bf > current_floor:
            return dq
    return -2.0


def _make_training_sample(
    run_state: SkadaRunState,
    candidates: list[ActionCandidate],
    chosen_index: int,
    decision_domain: str,
    room_type: str,
    *,
    future_dq: float = -2.0,
) -> TrainingSample | None:
    """产出 TrainingSample。chosen_index < 0 → skip。

    所有 value target 都用当前 run_state 计算(当层快照),
    value head 因此学的是 "这个 state 未来会怎样"。
    """
    if not candidates or chosen_index < 0 or chosen_index >= len(candidates):
        return None
    ctx = run_state.snapshot_banks_context()
    banks = _bank_builder.build(
        player_rt=ctx.player_rt,
        hand_cards_rt=[],
        enemies_rt=[],
        piles_rt=[],
        deck_cards=ctx.deck_cards,
        relics=ctx.relics,
        potions=ctx.potions,
        action_candidates=candidates,
        run_build_memory=ctx.rbm,
        room_type=room_type,
        is_combat=False,
        decision_domain=decision_domain,
    )
    return TrainingSample(
        banks=banks,
        action_index=chosen_index,
        # PPO 字段留 default(BC loss 不用)
        value_target=_run_win_target(run_state),
        advantage=0.0,
        old_log_prob=0.0,
        value_estimate=0.0,
        # 字段复用:BC loss 把下面 3 个字段当作 run_evaluator.expected_* head 的 target
        # (非战斗训练不走战斗 value_heads / leaf,不会误学)
        fight_win_target=_run_win_target(run_state),               # → run_win_prob target
        run_win_target=_run_win_target(run_state),
        hp_loss_target=run_state.run_avg_dmg_taken_per_combat,     # → expected_hp_loss target
        turn_damage_target=run_state.run_avg_dmg_dealt_per_combat, # → expected_dmg_output target
        survival_target=run_state.run_floor_clear_prob,            # → floor_clear_prob target
        leaf_target=0.0,
        transition_risk_target=0.0,
        resource_retention_target=_resource_health_target(run_state),
        boss_readiness_target=_boss_readiness_target(run_state),
        resource_health_target=_resource_health_target(run_state),
        deck_quality_target=_deck_quality_target(run_state),
        future_dq_target=future_dq,       # 档 2:走到下个 boss 时 dq 的真值
        sample_weight=_sample_weight(run_state),
        encounter_id="",
        room_type=room_type,
    )


# ---------------------------------------------------------------------------
# Decision-point builders
# ---------------------------------------------------------------------------

_RARITY_WEIGHT = {
    "basic":    0.0,
    "common":   0.25,
    "uncommon": 0.5,
    "rare":     1.0,
    "special":  0.5,
    "curse":   -0.3,
    "status":  -0.2,
}


def _rarity_weight(rarity: str) -> float:
    return _RARITY_WEIGHT.get(str(rarity or "").strip().lower(), 0.0)


def _fill_card_priors(
    cand: ActionCandidate,
    card_slug: str,
    floor: int,
    deck: list[str],
    character: str = "",
) -> None:
    """给 ActionCandidate 填 skada priors(pick_rate / win_rate_delta / deck_win_rate / synergy)。

    Synergy 计算优先走新 SkadaIndexFetcher(13175 pair,build_card_synergy_matrix 产出);
    fallback V1 SkadaPriors 的 18 pair + sum-based boost。
    """
    sp = _get_skada_priors()
    if sp is None or not card_slug:
        return
    cp = sp.card(card_slug)
    if cp is None:
        return
    cand.pick_rate_prior = float(cp.pick_rate)
    cand.win_rate_delta_prior = float(cp.win_rate_delta)
    # deck_win_rate 取"拥有该卡的 run 胜率"(floor-conditioned 中的 early/mid/late 加权平均)
    cand.deck_win_rate_prior = float(
        cp.floor_early if floor <= 6 else
        cp.floor_mid if floor <= 33 else
        cp.floor_late
    )
    # synergy:优先查新 13175-pair synergy matrix(lift + PMI 合并,char-conditional)
    syn = 0.0
    fetcher = _PATH_PRIORS_FETCHER
    if fetcher is not None and character:
        try:
            syn = fetcher.deck_card_synergy(character, card_slug, deck)
        except Exception:
            syn = 0.0
    if syn == 0.0:
        # fallback:V1 18-pair 的 raw sum(旧版行为)
        syn = sp.deck_synergy_boost(card_slug, deck)
    cand.synergy_prior = max(-1.0, min(1.0, syn))


def _fill_relic_priors(cand: ActionCandidate, relic_slug: str) -> None:
    """给遗物 option 填 priors(pick_rate / win_rate_delta)。"""
    sp = _get_skada_priors()
    if sp is None or not relic_slug:
        return
    rp = sp.relic(relic_slug)
    if rp is None:
        return
    cand.pick_rate_prior = float(rp.pick_rate)
    cand.win_rate_delta_prior = float(rp.win_rate_delta)
    cand.deck_win_rate_prior = float(rp.win_rate_owned)
    cand.synergy_prior = 0.0  # relic 同 deck synergy 数据不足,留 0


def _build_card_reward_sample(
    run_state: SkadaRunState,
    card_choices: list[dict[str, Any]],
    *,
    include_skip: bool = True,
    strict: bool = False,
    future_dq: float = -2.0,
) -> TrainingSample | None:
    """对 floor_timeline[i].card_choices 产一个 sample。

    如果全部 was_picked=False,视为 skip(当 include_skip=True 时加 skip candidate)。
    strict=True 时,候选中出现未知 card_id 会 return None(跳过该 sample)。
    """
    if not card_choices:
        return None
    candidates: list[ActionCandidate] = []
    chosen = -1
    n_unknown = 0
    for i, c in enumerate(card_choices):
        base, _upg = normalize_card_id(c.get("card_id", ""))
        if base and not is_known_card(base):
            n_unknown += 1
        cand = ActionCandidate(
            action_type="select_card_reward",
            action_index=i,
            label=base,
            family="card_reward",
            source_card_id=base,
            rarity_weight=_rarity_weight(c.get("rarity", "")),
            target_scope="none",
        )
        # Skada priors(pick_rate / win_rate_delta / synergy / floor-conditioned)
        _fill_card_priors(cand, base, run_state.floor, run_state.deck, character=run_state.character)
        candidates.append(cand)
        if c.get("was_picked"):
            chosen = i
    if strict and n_unknown > 0:
        logger.debug(f"strict: skip card_reward sample with {n_unknown} unknown ids")
        return None
    if chosen < 0 and include_skip:
        candidates.append(ActionCandidate(
            action_type="skip",
            action_index=len(candidates),
            label="skip",
            family="card_reward",
            roles=["terminal"],
            target_scope="none",
            ends_turn=True,
        ))
        chosen = len(candidates) - 1
    return _make_training_sample(
        run_state, candidates, chosen,
        decision_domain="card_reward", room_type="card_reward",
        future_dq=future_dq,
    )


def _build_relic_choice_sample(
    run_state: SkadaRunState,
    relic_choices: list[dict[str, Any]],
    *,
    ancient: bool = False,
    include_skip: bool = True,
    strict: bool = False,
    future_dq: float = -2.0,
) -> TrainingSample | None:
    """relic_choices / ancient_choices 都走这里,domain 归为 event。"""
    if not relic_choices:
        return None
    candidates: list[ActionCandidate] = []
    chosen = -1
    n_unknown = 0
    for i, c in enumerate(relic_choices):
        rid = normalize_relic_id(c.get("relic_id", ""))
        if rid and not is_known_relic(rid):
            n_unknown += 1
        cand = ActionCandidate(
            action_type="select_relic",
            action_index=i,
            label=rid,
            family="reward",
            source_card_id=rid,   # 复用字段存 relic id
            event_kind="gain_relic",
            target_scope="none",
        )
        _fill_relic_priors(cand, rid)
        candidates.append(cand)
        if c.get("was_picked"):
            chosen = i
    if strict and n_unknown > 0:
        return None
    if chosen < 0 and include_skip:
        candidates.append(ActionCandidate(
            action_type="skip",
            action_index=len(candidates),
            label="skip",
            family="reward",
            roles=["terminal"],
            target_scope="none",
            ends_turn=True,
        ))
        chosen = len(candidates) - 1
    return _make_training_sample(
        run_state, candidates, chosen,
        decision_domain="event",
        room_type="ancient" if ancient else "relic_reward",
        future_dq=future_dq,
    )


def _build_campfire_sample(
    run_state: SkadaRunState,
    floor_data: dict[str, Any],
    *,
    future_dq: float = -2.0,
) -> TrainingSample | None:
    """campfire 决策 — 简化为 SMITH vs HEAL 二分类。

    业务现实:
      - SMITH(升级)和 HEAL(回血)是**默认两个可用选项**(绝大多数 run)
      - MEND / CLONE / COOK / HATCH / LIFT / DIG 等**需要特定遗物解锁才会出现**
        (skada 里这些合计 <5% 样本)
      - 不加 skada-wide 的 option 池当 candidate 会污染 token(因为这些选项
        在玩家当前 run 根本不会出现,当成 "可选" 让网络学反而误导)

    处理:
      - choice ∈ {SMITH, HEAL}  → 产二分类 sample(candidates 含 HEAL + SMITH)
      - choice 其他值 → 跳过(需要 relic-unlock 上下文才能正确建模,后续扩展)

    同时每个 option 加 event_kind 区分(SMITH=upgrade_card / HEAL=gain_hp),
    让 token_bank_builder 的 9 维 event_kind one-hot 把两个 option token 区分开。
    """
    choice = str(floor_data.get("campfire_choice", "") or "").upper()
    # 只产 SMITH/HEAL 二分类 sample
    if choice not in ("SMITH", "HEAL"):
        return None

    options = [
        ("HEAL",  "gain_hp"),       # 回血
        ("SMITH", "upgrade_card"),  # 升级
    ]
    candidates = [
        ActionCandidate(
            action_type="campfire_" + opt.lower(),
            action_index=i,
            label=opt,
            family="rest",
            target_scope="none",
            event_kind=kind,   # 关键:给每个 option 独立语义 → token 可区分
        )
        for i, (opt, kind) in enumerate(options)
    ]
    chosen = 0 if choice == "HEAL" else 1
    return _make_training_sample(
        run_state, candidates, chosen,
        decision_domain="rest", room_type="rest",
        future_dq=future_dq,
    )


def _build_shop_sample(
    run_state: SkadaRunState,
    shop_actions: list[dict[str, Any]],
) -> list[TrainingSample]:
    """shop 每个 action 一个 sample(purchase_card/purchase_relic/purchase_potion/remove/leave)。

    简化:不包含 "不买什么" 作为反例。真正的 shop 决策需要当时商店存货,
    skada 只存 actions(买了什么),无法完全还原未买的 offered items。
    MVP 阶段:每个 action 是一个单 candidate sample(退化成 "必选此 action"),
    只用来提升 run-level value head 的监督,不太有 policy 学习价值。
    后续:接入 shop 全存货数据时,每个 step 展开成 "buy item X vs leave"。
    """
    samples: list[TrainingSample] = []
    if not shop_actions:
        return samples
    # 现阶段策略:聚合为一个"花了多少钱,买了几个物品"的抽象 sample,不做 per-action。
    # TODO: 接真实 shop offered items 后展开 per-action policy sample。
    return samples


# 全局可选:fetcher 传进来用于查 path priors(训练时由 train 脚本注入)
_PATH_PRIORS_FETCHER = None


def set_path_priors_fetcher(fetcher) -> None:
    """让训练脚本注入 SkadaIndexFetcher,loader 内部查 path priors 时用。"""
    global _PATH_PRIORS_FETCHER
    _PATH_PRIORS_FETCHER = fetcher


def _compute_path_fingerprint_key(
    start_coord: tuple,
    nodes_by_coord: dict[tuple, dict],
    max_hops: int = 20,
) -> str:
    """和 build_path_priors.py 的 _compute_fingerprint 同逻辑,BFS 下游算 fingerprint key。"""
    from networkV2.s6_training.build_path_priors import _compute_fingerprint

    visited: set[tuple] = set()
    frontier = {tuple(start_coord)}
    types: list[str] = []
    bfs_depth = 0
    while frontier and bfs_depth < max_hops:
        next_frontier: set[tuple] = set()
        for c in frontier:
            if c in visited:
                continue
            visited.add(c)
            node = nodes_by_coord.get(c)
            if node is None:
                continue
            t = str(node.get("type", "") or "")
            if t:
                types.append(t)
            for child in (node.get("children") or []):
                ct = tuple(child)
                if ct not in visited:
                    next_frontier.add(ct)
        frontier = next_frontier
        bfs_depth += 1
    fp = _compute_fingerprint(types)
    return fp.key()


def _build_map_route_samples(
    run_state: SkadaRunState,
    map_act: dict[str, Any],
    *,
    visibility_depth: int = MAP_VISIBILITY_DEPTH,
    future_dq: float = -2.0,
) -> list[TrainingSample]:
    """从 map_acts[i] 展开路线选择 sample。

    每一步 visited_coords[j] → visited_coords[j+1]:
      - 当前 coord 的 children 是候选
      - visited_coords[j+1] 对应那个 child 是标签
      - **子节点 type** 用 mask_map_with_visibility 限制(默认只暴露 +1 层)

    这是为了避免"上帝视角信息泄漏":
    skada 存的 map 节点是 run 结束后全部已揭示;但玩家在 UI 上只能看到
    当前层 + 下 1 层的节点类型(特定遗物可揭示更深)。
    如果不 mask,网络会学到"远处有 rest 所以走这条"这种游戏中拿不到的信号。

    注:这里 run_state 仍然共享一个快照(简化),精确做法是按 floor snapshot。
    """
    samples: list[TrainingSample] = []
    visited = map_act.get("visited_coords") or []
    raw_nodes = list(map_act.get("nodes") or [])
    if len(visited) < 2 or not raw_nodes:
        return samples

    for j in range(len(visited) - 1):
        cur_coord = tuple(visited[j])
        next_coord = tuple(visited[j + 1])

        # 每步重新 mask 一次,基于当前所在节点
        masked_nodes = mask_map_with_visibility(raw_nodes, cur_coord, visibility_depth)
        nodes_by_coord = {tuple(n["coord"]): n for n in masked_nodes if "coord" in n}

        node = nodes_by_coord.get(cur_coord)
        if node is None:
            # 起始节点(y=0)常常不在 nodes 里 → 跳过第一步
            continue
        children = [tuple(c) for c in (node.get("children") or [])]
        if next_coord not in children or len(children) < 2:
            continue
        candidates: list[ActionCandidate] = []
        chosen = -1
        for k, child in enumerate(children):
            child_node = nodes_by_coord.get(child) or {}
            child_type = str(child_node.get("type", "") or "").upper()
            # child 在可见层(+1)所以类型应该都知道,兜底 masked 情况
            if child_type == UNKNOWN_NODE_TYPE:
                risk, value = 0.1, 0.2
                stats = _PathGlobalStats()
            else:
                # 自身 type 粗 risk/value
                risk = _route_risk(child_type)
                value = _route_value(child_type)
                # 全局路径 stat:从此 child 到 boss 所有可达路径的 type 分布 +
                # 最优子路径 rest 数 + 路径长度 → 让网络看到"下游所有选择"
                stats = _path_global_stats(child, nodes_by_coord)
            # Skada data-driven path priors(freq + efficiency,查 sqlite)
            prior_freq, prior_eff = 0.0, 0.5
            if _PATH_PRIORS_FETCHER is not None and child_type != UNKNOWN_NODE_TYPE:
                try:
                    fp_key = _compute_path_fingerprint_key(child, nodes_by_coord)
                    # asc_bucket 从 ascension 算
                    asc = int(run_state.ascension or 0)
                    if asc < 5: asc_bucket = "low"
                    elif asc < 15: asc_bucket = "mid"
                    elif asc < 20: asc_bucket = "high"
                    else: asc_bucket = "max"
                    result = _PATH_PRIORS_FETCHER.lookup_path_prior(
                        run_state.character, asc_bucket, fp_key,
                    )
                    if result is not None:
                        prior_freq, prior_eff = result
                except Exception:
                    pass

            cand = ActionCandidate(
                action_type="select_map_node",
                action_index=k,
                label=f"{child}:{child_type}",
                family="map",
                target_scope="map",
                route_risk=risk,
                route_value=value,
                route_path_rest_rate=stats.rest_rate,
                route_path_shop_rate=stats.shop_rate,
                route_path_elite_rate=stats.elite_rate,
                route_path_treasure_rate=stats.treasure_rate,
                route_path_event_rate=stats.event_rate,
                route_path_monster_rate=stats.monster_rate,
                route_best_rest_count=stats.best_rest_count_norm,
                route_path_length_norm=stats.path_length_norm,
                route_prior_frequency=prior_freq,
                route_prior_efficiency=prior_eff,
            )
            candidates.append(cand)
            if child == next_coord:
                chosen = k
        if chosen < 0:
            continue
        sample = _make_training_sample(
            run_state, candidates, chosen,
            decision_domain="route", room_type="map",
            future_dq=future_dq,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def _route_risk(node_type: str) -> float:
    return {
        "M": 0.3, "E": 0.7, "B": 1.0, "S": 0.0, "R": 0.0, "V": 0.2, "T": 0.0,
    }.get(str(node_type or "").upper(), 0.1)


def _route_value(node_type: str) -> float:
    return {
        "M": 0.5, "E": 0.7, "B": 1.0, "S": 0.6, "R": 0.7, "V": 0.4, "T": 0.9,
    }.get(str(node_type or "").upper(), 0.2)


def _path_lookahead(
    start_coord: tuple,
    nodes_by_coord: dict[tuple, dict],
    max_depth: int = 5,
) -> dict[str, int]:
    """从 start_coord 沿 children BFS 至 max_depth,统计**所有经过节点**的 type 计数。

    (用于 legacy risk/value 聚合;路径规划更复杂的版本见 _path_global_stats)
    """
    counts: dict[str, int] = {}
    visited: set[tuple] = {tuple(start_coord)}
    frontier: set[tuple] = {tuple(start_coord)}
    for _ in range(max_depth):
        next_frontier: set[tuple] = set()
        for c in frontier:
            node = nodes_by_coord.get(c)
            if node is None:
                continue
            for child in (node.get("children") or []):
                ct = tuple(child)
                if ct in visited:
                    continue
                visited.add(ct)
                next_frontier.add(ct)
                child_node = nodes_by_coord.get(ct)
                if child_node is not None:
                    t = str(child_node.get("type", "") or "").upper()
                    counts[t] = counts.get(t, 0) + 1
        if not next_frontier:
            break
        frontier = next_frontier
    return counts


@dataclass
class _PathGlobalStats:
    """从 child 到 boss 的全局路径 stat。让网络看到"下游所有可能路线"的聚合信号。

    字段都是归一化到 [0,1] 的 rate 或 normalized count。
    """
    rest_rate: float = 0.0
    shop_rate: float = 0.0
    elite_rate: float = 0.0
    treasure_rate: float = 0.0
    event_rate: float = 0.0
    monster_rate: float = 0.0
    best_rest_count_norm: float = 0.0   # 所有可达路径中 rest 最多那条的 rest 数 / 5
    path_length_norm: float = 0.0       # 到 boss 的最短路径长度 / 17


def _path_global_stats(
    start_coord: tuple,
    nodes_by_coord: dict[tuple, dict],
    boss_coord: tuple | None = None,
    max_hops: int = 20,
) -> _PathGlobalStats:
    """从 start_coord 做完整下游展开,算到 boss 的全局路径 stat。

    做 2 件事:
      1. 全下游 BFS:统计可达节点总的 type 分布(occurrence rate = type 计数 / 总节点数)
         → 近似 "走这条下去,未来每一步遇到某 type 的概率"
      2. DP 求最优子路径的 rest 数:
         max_rest[node] = node 自身 rest? + max(max_rest[child] for child in children)
         表达"如果规划得当,走这条最多能拿几个 rest"

    boss_coord=None 时自动检测(拓扑里没有 children 的节点 = boss 或叶子;
    没 children 的节点深度作为 path_length)。
    """
    start = tuple(start_coord)
    if start not in nodes_by_coord:
        return _PathGlobalStats()

    # ---- 1. 全下游 BFS ----
    type_counts: dict[str, int] = {}
    total_nodes = 0
    visited: set[tuple] = set()
    frontier = {start}
    bfs_depth = 0
    max_depth_seen = 0
    while frontier and bfs_depth < max_hops:
        next_frontier: set[tuple] = set()
        for c in frontier:
            if c in visited:
                continue
            visited.add(c)
            node = nodes_by_coord.get(c)
            if node is None:
                continue
            t = str(node.get("type", "") or "").upper()
            if t:
                type_counts[t] = type_counts.get(t, 0) + 1
                total_nodes += 1
            max_depth_seen = max(max_depth_seen, bfs_depth)
            for child in (node.get("children") or []):
                ct = tuple(child)
                if ct not in visited:
                    next_frontier.add(ct)
        frontier = next_frontier
        bfs_depth += 1

    rates: dict[str, float] = {}
    if total_nodes > 0:
        for t, c in type_counts.items():
            rates[t] = c / total_nodes

    # ---- 2. DP: 最优子路径 rest 计数 ----
    # 拓扑是 DAG(y 递增),按 y 倒序 DP
    # max_rest_from[node] = (node.type == R ? 1 : 0) + max(max_rest_from[child])
    sorted_coords: list[tuple] = sorted(
        visited,
        key=lambda c: c[1] if len(c) >= 2 else 0,
        reverse=True,   # 远端(y 大)先算
    )
    max_rest_from: dict[tuple, int] = {}
    for c in sorted_coords:
        node = nodes_by_coord.get(c)
        if node is None:
            max_rest_from[c] = 0
            continue
        self_count = 1 if str(node.get("type", "") or "").upper() == "R" else 0
        children = [tuple(ch) for ch in (node.get("children") or [])]
        if not children:
            max_rest_from[c] = self_count
        else:
            best_child = max((max_rest_from.get(ch, 0) for ch in children), default=0)
            max_rest_from[c] = self_count + best_child
    best_rest = max_rest_from.get(start, 0)

    return _PathGlobalStats(
        rest_rate=rates.get("R", 0.0),
        shop_rate=rates.get("S", 0.0),
        elite_rate=rates.get("E", 0.0),
        treasure_rate=rates.get("T", 0.0),
        event_rate=rates.get("V", 0.0),
        monster_rate=rates.get("M", 0.0),
        best_rest_count_norm=min(best_rest / 5.0, 1.0),
        path_length_norm=min(max_depth_seen / 17.0, 1.0),
    )


def _route_features_with_lookahead(
    child_coord: tuple,
    child_type: str,
    nodes_by_coord: dict[tuple, dict],
    lookahead_depth: int = 5,
) -> tuple[float, float]:
    """已废弃:保留只为其他地方兼容 import。
    新代码应走 _path_global_stats 把 8 维特征喂给 ActionCandidate。
    """
    base_risk = _route_risk(child_type)
    base_value = _route_value(child_type)
    la = _path_lookahead(child_coord, nodes_by_coord, max_depth=lookahead_depth)
    rest_ahead = la.get("R", 0)
    shop_ahead = la.get("S", 0)
    treasure_ahead = la.get("T", 0)
    elite_ahead = la.get("E", 0)
    event_ahead = la.get("V", 0)
    value = base_value + 0.1 * rest_ahead + 0.05 * shop_ahead + 0.08 * treasure_ahead + 0.02 * event_ahead
    risk = base_risk + 0.1 * elite_ahead
    return max(0.0, min(1.0, risk)), max(0.0, min(1.0, value))


def _build_value_only_sample(
    run_state: SkadaRunState,
    room_type: str,
    *,
    future_dq: float = -2.0,
) -> TrainingSample | None:
    """每个 floor 产一个 "只监督 value head" 的 sample。

    动作只有 1 个 dummy(proceed),policy loss 被设置为小权重/跳过。
    主用途:给 run_evaluator 4 个 head 提供密集监督。
    """
    candidates = [
        ActionCandidate(
            action_type="proceed",
            action_index=0,
            label="Proceed",
            family="proceed",
            roles=["terminal"],
            target_scope="none",
            ends_turn=True,
        ),
    ]
    sample = _make_training_sample(
        run_state, candidates, 0,
        decision_domain="event", room_type=f"value_only_{room_type}",
        future_dq=future_dq,
    )
    if sample is not None:
        # 设置小 sample_weight 避免 policy 学 "永远选 proceed"(退化)
        sample.sample_weight *= 0.1
    return sample


# ---------------------------------------------------------------------------
# 主入口:一条 run → list[TrainingSample]
# ---------------------------------------------------------------------------

def load_run_samples(
    rec: dict[str, Any],
    *,
    include_value_only: bool = True,
    include_map_routes: bool = True,
    strict_ids: bool = False,
    map_visibility_depth: int = MAP_VISIBILITY_DEPTH,
    skip_unknown_character: bool = True,
) -> list[TrainingSample]:
    """从一条 skada run 产出所有 TrainingSample。

    参数:
      strict_ids: True → 候选含未知 card/relic id 时跳过该 sample。
                  False(默认)→ tolerant,产 sample 但 card semantics 为空(ID 仍被送进 bank)。
      map_visibility_depth: map route sample 可见层深,默认 1(只暴露 +1 层 children type)
      skip_unknown_character: 遇到 KNOWN_CHARACTERS 外的 character 直接跳过整条 run
    """
    samples: list[TrainingSample] = []

    # 校验 character
    character = str(rec.get("run", {}).get("character", "") or "").upper()
    from networkV2.s6_training.skada_id_mapping import KNOWN_CHARACTERS as _KC
    if skip_unknown_character and character and character not in _KC:
        logger.info(f"skip unknown character {character!r} run_id={rec.get('run', {}).get('run_id')}")
        return []

    # 档 2:预扫 run 的 boss 层 deck_quality 检查点(供 future_dq_target 回填)
    boss_checkpoints = _compute_boss_checkpoints(rec)

    for floor_data, pre_state, post_state in iter_timeline_with_state(rec):
        future_dq = _next_boss_dq(pre_state.floor, boss_checkpoints)
        # card_reward
        if floor_data.get("card_choices"):
            s = _build_card_reward_sample(pre_state, floor_data["card_choices"], strict=strict_ids,
                                           future_dq=future_dq)
            if s is not None:
                samples.append(s)

        if floor_data.get("relic_choices"):
            s = _build_relic_choice_sample(pre_state, floor_data["relic_choices"], ancient=False,
                                           strict=strict_ids, future_dq=future_dq)
            if s is not None:
                samples.append(s)

        if floor_data.get("ancient_choices"):
            s = _build_relic_choice_sample(pre_state, floor_data["ancient_choices"], ancient=True,
                                           strict=strict_ids, future_dq=future_dq)
            if s is not None:
                samples.append(s)

        if floor_data.get("campfire_choice"):
            s = _build_campfire_sample(pre_state, floor_data, future_dq=future_dq)
            if s is not None:
                samples.append(s)

        # shop:未来接 offered items 后展开 per-action
        # if floor_data.get("shop_actions"):
        #     samples.extend(_build_shop_sample(pre_state, floor_data["shop_actions"]))

        if include_value_only:
            room_type = str(floor_data.get("room_type", "") or "").upper()
            s = _build_value_only_sample(pre_state, room_type, future_dq=future_dq)
            if s is not None:
                samples.append(s)

    # ---- map route samples ----
    if include_map_routes:
        # 用 floor_timeline 最末的 state 近似(简化;真正精确需要 per-floor snapshot)
        final_state = SkadaRunState.from_run_record(rec)
        for floor_data, pre_state, post_state in iter_timeline_with_state(rec):
            final_state = post_state
        for map_act in rec.get("map_acts", []) or []:
            # map route 用 act 起始 floor 算 future_dq
            act_idx = int(map_act.get("act", 0) or 0)
            ref_floor = act_idx * 17
            act_future_dq = _next_boss_dq(ref_floor, boss_checkpoints)
            samples.extend(_build_map_route_samples(
                final_state, map_act, visibility_depth=map_visibility_depth,
                future_dq=act_future_dq,
            ))

    return samples


# ---------------------------------------------------------------------------
# 文件/目录级入口
# ---------------------------------------------------------------------------

def iter_records_from_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """兼容两种 jsonl 格式:每行一条 record,或 pretty-printed 整文件一条。"""
    text = Path(path).read_text(encoding="utf-8")
    # 先尝试整文件当一条 parse
    try:
        rec = json.loads(text)
        if isinstance(rec, dict):
            yield rec
            return
    except json.JSONDecodeError:
        pass
    # 退回逐行
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"skip invalid jsonl line in {path}: {e}")


def load_samples_from_jsonl_dir(
    dir_path: Path,
    *,
    max_runs: int | None = None,
    skip_expired: bool = True,
    **kwargs: Any,
) -> list[TrainingSample]:
    """扫目录下所有 jsonl 文件,累积 TrainingSample。"""
    all_samples: list[TrainingSample] = []
    runs_loaded = 0
    for jf in sorted(Path(dir_path).glob("*.jsonl")):
        for rec in iter_records_from_jsonl(jf):
            if skip_expired and rec.get("detail_expired"):
                continue
            # 跳过没有 floor_timeline.card_choices 等字段的空骨架
            if not rec.get("floor_timeline"):
                continue
            try:
                samples = load_run_samples(rec, **kwargs)
            except Exception as e:
                logger.warning(f"load_run_samples failed run_id={rec.get('run', {}).get('run_id')}: {e}")
                continue
            all_samples.extend(samples)
            runs_loaded += 1
            if max_runs is not None and runs_loaded >= max_runs:
                return all_samples
    logger.info(f"loaded {len(all_samples)} samples from {runs_loaded} runs in {dir_path}")
    return all_samples


# ---------------------------------------------------------------------------
# SQLite loader(生产级入口)
# ---------------------------------------------------------------------------

def iter_records_from_sqlite(
    db_path: Path,
    *,
    max_runs: int | None = None,
    only_ok: bool = True,
    only_victory: bool = False,
    min_ascension: int = 0,
    characters: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """从 skada_analytics.sqlite 的 `run_details.raw_json` 取完整 record。

    `run_details.raw_json` 存了整条 jsonl 原文(和 newSample.jsonl 一致的格式),
    直接 json.loads 即可喂给 load_run_samples,完美复用 jsonl pipeline。

    参数:
      only_ok:        仅 run_details.status='ok' 的 record
      only_victory:   仅 runs.is_victory=1
      min_ascension:  runs.ascension >= N(0 = 无过滤)
      characters:     白名单(如 ['IRONCLAD', 'REGENT']);None = 全部
    """
    import sqlite3
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    where: list[str] = []
    params: list[Any] = []
    if only_ok:
        where.append("d.status = 'ok'")
    if only_victory:
        where.append("r.is_victory = 1")
    if min_ascension > 0:
        where.append("r.ascension >= ?")
        params.append(int(min_ascension))
    if characters:
        placeholders = ",".join("?" for _ in characters)
        where.append(f"upper(r.character) IN ({placeholders})")
        params.extend([ch.upper() for ch in characters])
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""

    q = (
        "SELECT r.run_id, r.character, r.ascension, r.is_victory, r.game_version, "
        "       r.floor_reached, d.raw_json "
        "FROM runs r JOIN run_details d ON d.run_id = r.run_id"
        + where_clause + " ORDER BY r.run_id"
    )
    if max_runs is not None:
        q += f" LIMIT {int(max_runs)}"

    try:
        for row in con.execute(q, params):
            rj = row["raw_json"]
            if not rj:
                continue
            try:
                rec = json.loads(rj)
            except json.JSONDecodeError as e:
                logger.warning(f"skip invalid raw_json run_id={row['run_id']}: {e}")
                continue
            # 补上 meta 字段(raw_json 里的 run.* 一般已齐全,但兜底)
            rec.setdefault("run", {})
            rec["run"].setdefault("run_id", row["run_id"])
            rec["run"].setdefault("character", row["character"])
            rec["run"].setdefault("ascension", row["ascension"])
            rec["run"].setdefault("is_victory", bool(row["is_victory"]))
            rec["run"].setdefault("game_version", row["game_version"])
            rec["run"].setdefault("floor_reached", row["floor_reached"])
            yield rec
    finally:
        con.close()


def load_samples_from_index(
    index_db: Path,
    *,
    priors_db: Path | None = None,
    synergy_db: Path | None = None,
    n_runs: int = 2000,
    balanced: bool = True,
    n_per_group: int | None = None,
    require_map_acts: bool = False,
    characters: list[str] | None = None,
    asc_bucket: str | None = None,
    seed: int = 0,
    **loader_kwargs: Any,
) -> list[TrainingSample]:
    """基于 skada_runs.sqlite 索引 + 原始 jsonl seek 读取产 TrainingSample。

    4.7 GB jsonl 不全 load,按 index 分层采样 → 只拉选中的 runs。
    priors_db 提供时会自动 attach 到 fetcher,candidate 填 data-driven prior feature。
    """
    from networkV2.s6_training.skada_index_dataset import SkadaIndexFetcher

    fetcher = SkadaIndexFetcher(index_db=index_db, priors_db=priors_db, synergy_db=synergy_db)
    set_path_priors_fetcher(fetcher)

    try:
        stats = fetcher.stats()
        logger.info(f"index stats: {stats}")

        if balanced:
            per_group = n_per_group or max(n_runs // 20, 10)
            rows = fetcher.sample_balanced(
                n_per_group=per_group, require_map_acts=require_map_acts, seed=seed,
            )
        else:
            rows = fetcher.sample_clean_runs(
                n=n_runs,
                character=characters[0] if characters and len(characters) == 1 else None,
                asc_bucket=asc_bucket,
                require_map_acts=require_map_acts,
                seed=seed,
            )

        logger.info(f"selected {len(rows)} runs for loading")
        all_samples: list[TrainingSample] = []
        for i, record in enumerate(fetcher.fetch_records(rows)):
            if not record.get("floor_timeline"):
                continue
            try:
                samples = load_run_samples(record, **loader_kwargs)
            except Exception as e:
                logger.warning(f"load_run_samples failed: {e}")
                continue
            all_samples.extend(samples)
            if (i + 1) % 500 == 0:
                logger.info(f"loaded {i+1} runs → {len(all_samples)} samples")
        logger.info(f"done: {len(all_samples)} samples from {len(rows)} runs")
        return all_samples
    finally:
        fetcher.close()
        set_path_priors_fetcher(None)


def load_samples_from_sqlite(
    db_path: Path,
    *,
    max_runs: int | None = None,
    only_victory: bool = False,
    min_ascension: int = 0,
    characters: list[str] | None = None,
    **loader_kwargs: Any,
) -> list[TrainingSample]:
    """生产入口:从 skada_analytics.sqlite 批量产 TrainingSample。

    `loader_kwargs` 透传给 `load_run_samples`:
      - include_value_only, include_map_routes, strict_ids,
        map_visibility_depth, skip_unknown_character

    典型用法(训练脚本里):
        samples = load_samples_from_sqlite(
            db_path=Path("../Assets/datasets/skada/skada_analytics.sqlite"),
            max_runs=2000,
            only_victory=False,
            min_ascension=0,
            characters=None,  # 全 character
        )
    """
    all_samples: list[TrainingSample] = []
    runs_used = 0
    for rec in iter_records_from_sqlite(
        db_path, max_runs=max_runs,
        only_victory=only_victory, min_ascension=min_ascension,
        characters=characters,
    ):
        if not rec.get("floor_timeline"):
            continue
        try:
            samples = load_run_samples(rec, **loader_kwargs)
        except Exception as e:
            logger.warning(f"load_run_samples failed run_id={rec.get('run', {}).get('run_id')}: {e}")
            continue
        all_samples.extend(samples)
        runs_used += 1
    logger.info(f"loaded {len(all_samples)} samples from {runs_used} sqlite runs ({db_path})")
    return all_samples
