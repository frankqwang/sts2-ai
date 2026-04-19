"""Skada → networkV2 state 翻译层。

沿 run.floor_timeline 累计重建一个 `SkadaRunState`,
包括当前 deck / relics / potions / hp / gold / floor / combats_seen
/ elites_seen / bosses_seen / total_hp_lost / room_history 等。

在每个决策点(card_reward / relic_choice / campfire / shop / map),
直接用 `to_banks_context(...)` 产出 `TokenBankBuilder.build(...)` 期望的
PlayerRuntime / CardSemantics / RelicSemantics / PotionSemantics / RunBuildMemory,
供 loader 进一步构造 TrainingSample。

约束:
- 不依赖 sim,不依赖 obs。只读 skada record。
- 所有游戏 id 都走 GAME_CATALOG / source_knowledge 查(SCHEMA_CONVENTION)。
- starter deck 按 character 查 STARTER_DECKS(暂硬编码,后续可迁到 sqlite)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

from networkV2.s1_schema.entities import (
    PlayerRuntime, CardSemantics, RelicSemantics, PotionSemantics,
)
from networkV2.s1_schema.memory import RunBuildMemory
from networkV2.s1_schema.card_tags import card_feature_vector, NUM_FUNCTIONAL_TAGS
from networkV2.s6_training.skada_id_mapping import (
    normalize_card_id, normalize_relic_id, normalize_potion_id,
    is_known_card, is_known_relic,
    character_starter_deck, character_starter_relic,
    KNOWN_CHARACTERS,
)


logger = logging.getLogger(__name__)


# 基础 max_hp:STS2 各角色起手血量。skada 记录里 floor_timeline[0].hp_after 即真值,
# 但 floor 0 (A 房)可能触发事件掉/加血,所以用 hardcode 作为兜底,实际 apply_floor_pre
# 会 override 到 hp_before 读到的值。
_BASE_MAX_HP: dict[str, int] = {
    "IRONCLAD":    80,
    "REGENT":      70,
    "SILENT":      70,
    "DEFECT":      75,
    "NECROBINDER": 68,   # 待 sim 校验
}


# ---------------------------------------------------------------------------
# Card / Relic / Potion Semantics 构造(数据驱动,走 skada_id_mapping)
# ---------------------------------------------------------------------------

# source_knowledge.cards 的 card_type/rarity 查询缓存
from functools import lru_cache
import sqlite3
from pathlib import Path as _Path

_SRC_DB_PATH = _Path(__file__).resolve().parents[1] / "data" / "source_knowledge.sqlite"


@lru_cache(maxsize=1)
def _load_card_meta() -> dict[str, tuple[str, str, int]]:
    """card_id (lower) → (card_type, rarity, cost)。"""
    if not _SRC_DB_PATH.exists():
        return {}
    con = sqlite3.connect(str(_SRC_DB_PATH))
    try:
        rows = con.execute("SELECT id, card_type, rarity, cost FROM cards").fetchall()
    finally:
        con.close()
    return {
        str(r[0]).lower(): (
            str(r[1] or "").lower(),
            str(r[2] or "").lower(),
            int(r[3] or 0),
        )
        for r in rows
    }


# 向后兼容:保留名字供别处导入;新代码建议用 normalize_card_id 返回的 base_id
def _strip_upgrade_suffix(card_id: str) -> str:
    base, _ = normalize_card_id(card_id)
    # 向前保留 "IRONCLAD_BURN" 这种 UPPER 风格 —— 给老 loader 用
    # 但新逻辑(_card_semantics_from_id)内部会再 normalize
    return base.upper()


def _card_semantics_from_id(card_id: str) -> CardSemantics:
    """从 card_id 查 CardSemantics(数据驱动,走 source_knowledge)。

    输入可以是任意大小写 / 带 "+" 后缀。输出 entity_id 保留原大小写 + "+"
    (compatible with token_bank_builder 的 id 比较)。
    """
    raw = str(card_id or "")
    base_lower, upg_count = normalize_card_id(raw)
    is_upg = upg_count > 0

    meta = _load_card_meta().get(base_lower)
    if meta is None:
        # 未知 card_id(skada 旧版本 / 打错?):fallback 到空语义,但不崩
        if base_lower:
            logger.debug(f"unknown card_id {base_lower!r}, using empty semantics")
        return CardSemantics(
            entity_id=raw,
            card_type="",
            rarity="",
            base_cost=0,
            is_upgraded=is_upg,
            tags=[],
            keywords=[],
        )
    card_type, rarity, cost = meta
    return CardSemantics(
        entity_id=raw,
        card_type=card_type,
        rarity=rarity,
        base_cost=cost,
        is_upgraded=is_upg,
        tags=[],
        keywords=[],
    )


def _relic_semantics_from_id(relic_id: str) -> RelicSemantics:
    return RelicSemantics(
        entity_id=str(relic_id or ""),
        relic_tags=[],
        functional_signals={},
    )


def _potion_semantics_from_id(potion_id: str) -> PotionSemantics:
    return PotionSemantics(
        entity_id=str(potion_id or ""),
        potion_type="",
        tags=[],
    )


# ---------------------------------------------------------------------------
# Deck aggregate → RunBuildMemory 构筑画像
# ---------------------------------------------------------------------------

def _aggregate_deck_profile(deck: list[str]) -> dict[str, float]:
    """基于 card_tags 聚合构筑画像(frontload/block/draw/scaling/aoe/heal)。

    目前实现简易:依赖 card_feature_vector 的 34 维向量,
    把 tag 维度粗糙映射到 6 个 archetype 维度。
    TODO: 和 build_profile 模块对齐精确的 tag → archetype 映射。
    """
    if not deck:
        return {
            "frontload": 0.0, "block": 0.0, "draw": 0.0,
            "scaling": 0.0, "aoe": 0.0, "heal": 0.0,
            "curse_density": 0.0, "high_cost_density": 0.0,
            "zero_cost_density": 0.0, "x_cost_density": 0.0,
            "consistency": 0.0,
        }

    n = len(deck)
    sums = [0.0] * NUM_FUNCTIONAL_TAGS
    curse = 0
    for cid in deck:
        base = _strip_upgrade_suffix(cid).upper()
        try:
            fv = card_feature_vector(base)
        except Exception:
            fv = [0.0] * NUM_FUNCTIONAL_TAGS
        for i, v in enumerate(fv):
            sums[i] += v
        if "CURSE" in base or base == "ASCENDERS_BANE":
            curse += 1
    avg = [s / n for s in sums]

    # index 约定(见 card_tags.py):
    #   0=attack  1=skill  2=power  3=curse  4=status
    # 后续维度不稳定,先粗糙使用 attack 近似 frontload、skill 近似 block、power 近似 scaling
    return {
        "frontload": min(avg[0] if len(avg) > 0 else 0.0, 1.0),
        "block":     min(avg[1] if len(avg) > 1 else 0.0, 1.0),
        "draw":      0.0,     # TODO: 需要 draw 标签索引
        "scaling":   min(avg[2] if len(avg) > 2 else 0.0, 1.0),
        "aoe":       0.0,     # TODO: 需要 aoe 标签索引
        "heal":      0.0,     # TODO: 需要 heal 标签索引
        "curse_density": curse / n,
        "high_cost_density": 0.0,
        "zero_cost_density": 0.0,
        "x_cost_density": 0.0,
        "consistency": 1.0 - min(curse / n, 1.0),  # 粗略反比诅咒比例
    }


# ---------------------------------------------------------------------------
# SkadaRunState - 沿 timeline 累计状态
# ---------------------------------------------------------------------------

@dataclass
class BanksContext:
    """供 TokenBankBuilder.build() 使用的打包上下文。"""
    player_rt: PlayerRuntime
    deck_cards: list[CardSemantics]
    relics: list[RelicSemantics]
    potions: list[PotionSemantics]
    rbm: RunBuildMemory


@dataclass
class SkadaRunState:
    """一条 skada run 在某一时刻的累计状态。

    每次进入新 floor(_apply_floor_pre)和离开 floor(_apply_floor_post)都会更新,
    使得在任意决策点都能 snapshot 出当前 deck/hp/gold/counters。
    """
    # ---- run meta ----
    run_id: int = 0
    character: str = "IRONCLAD"
    ascension: int = 0
    is_victory: bool = False
    floor_reached: int = 0

    # ---- 当前状态(决策点时读它) ----
    floor: int = 0
    hp: int = 0
    max_hp: int = 80
    gold: int = 99
    deck: list[str] = field(default_factory=list)       # 卡 id 列表(含重复)
    relics: list[str] = field(default_factory=list)     # 遗物 id
    potions: list[str] = field(default_factory=list)    # 药水 id

    # ---- 累计 counter ----
    combats_seen: int = 0
    elites_seen: int = 0
    bosses_seen: int = 0
    total_hp_lost: int = 0
    room_type_history: list[str] = field(default_factory=list)
    event_history: list[str] = field(default_factory=list)
    enemy_types_seen: dict[str, int] = field(default_factory=dict)

    # ---- 诊断用 ----
    last_floor_seen: int = 0

    # ---- Run-level 聚合真值(供 run_evaluator.expected_* head 做监督)----
    # 从整条 rec 一次性算出:每战平均掉血 / 每战平均输出 / 通关度
    run_avg_dmg_taken_per_combat: float = 0.0
    run_avg_dmg_dealt_per_combat: float = 0.0
    run_floor_clear_prob: float = 0.0   # [0,1],is_victory=1 → 1.0,否则 floor_reached/55

    @staticmethod
    def from_run_record(rec: dict[str, Any]) -> "SkadaRunState":
        """根据一条 skada run_record 初始化(起手 deck/relic/hp)。

        所有 id 通过 skada_id_mapping 规范化成 lower snake(和 source_knowledge 一致),
        方便下游 card_feature_vector 直接查。

        不 apply floor_timeline,调用方负责 iterate。
        """
        run = rec.get("run", {})
        character = str(run.get("character", "IRONCLAD") or "IRONCLAD").upper()
        if character not in KNOWN_CHARACTERS:
            logger.warning(f"unknown character {character!r} in run {run.get('run_id')}, falling back to IRONCLAD")
        ascension = int(run.get("ascension", 0) or 0)
        max_hp = _BASE_MAX_HP.get(character, 80)

        # 数据驱动的起手 deck(从 source_knowledge 查)
        deck = character_starter_deck(character, ascension=ascension)
        starter_relic = character_starter_relic(character)
        relics = [starter_relic] if starter_relic else []

        # ---- Run-level combat aggregate(供 run_evaluator.expected_* head 监督)----
        combats = rec.get("combats") or []
        n_combats = len(combats)
        total_dmg_taken = sum(
            int(c.get("total_dmg_taken", 0) or 0) for c in combats
        )
        total_dmg_dealt = sum(
            int(c.get("total_dmg_dealt", 0) or 0) for c in combats
        )
        # 老数据可能 total_dmg_taken 缺失,fallback 到 combat_stats 聚合
        if n_combats > 0 and total_dmg_taken == 0 and total_dmg_dealt == 0:
            for c in combats:
                for stat in (c.get("combat_stats") or []):
                    total_dmg_taken += int(stat.get("dmg_taken", 0) or 0)
                    total_dmg_dealt += int(stat.get("dmg_dealt", 0) or 0)

        avg_taken = total_dmg_taken / max(n_combats, 1)
        avg_dealt = total_dmg_dealt / max(n_combats, 1)
        is_vic = bool(run.get("is_victory", False))
        floor_reached = int(run.get("floor_reached", 0) or 0)
        # 通关度:胜=1,败=floor/55(55 楼通关近似)
        floor_clear = 1.0 if is_vic else min(max(floor_reached / 55.0, 0.0), 1.0)

        return SkadaRunState(
            run_id=int(run.get("run_id", 0) or 0),
            character=character,
            ascension=ascension,
            is_victory=is_vic,
            floor_reached=floor_reached,
            floor=0,
            hp=max_hp,
            max_hp=max_hp,
            gold=99,
            deck=deck,
            relics=relics,
            potions=[],
            run_avg_dmg_taken_per_combat=avg_taken,
            run_avg_dmg_dealt_per_combat=avg_dealt,
            run_floor_clear_prob=floor_clear,
        )

    # ------------------------------------------------------------------
    # 每层推进
    # ------------------------------------------------------------------

    def apply_floor_pre(self, floor_data: dict[str, Any]) -> None:
        """进入本 floor 前:刷新 floor/hp/gold 到 "before" 快照。

        调用方应在**处理本 floor 的所有决策点之前**调用。
        """
        self.floor = int(floor_data.get("floor", self.floor) or self.floor)
        self.hp = int(floor_data.get("hp_before", self.hp) or self.hp)
        self.gold = int(floor_data.get("gold_before", self.gold) or self.gold)

    def apply_floor_post(self, floor_data: dict[str, Any]) -> None:
        """离开 floor 后:把本层的 choices 结果写进状态。

        处理顺序(决策点):card_choices / relic_choices / ancient_choices /
        campfire(card_upgrades) / shop_actions / event_text
        → 最后用 hp_after/gold_after 对齐状态。
        """
        room_type = str(floor_data.get("room_type", "") or "").upper()
        room_key = _map_room_type(room_type)
        if room_key:
            self.room_type_history.append(room_key)

        # --- card_choices: was_picked=True 的加入 deck ---
        for c in floor_data.get("card_choices", []) or []:
            if c.get("was_picked"):
                base, _upg = normalize_card_id(c.get("card_id", ""))
                if base:
                    self.deck.append(base)

        # --- relic_choices (含 ancient_choices) ---
        # 2026-04-19:skada 数据里 ancient_choices 偶尔把 character_id (NECROBINDER /
        # IRONCLAD / REGENT / SILENT) 写进 relic_id 字段(爬虫 bug)。还有 mod relic
        # (EXTRARELICS-*) / skada 自制 relic (NEOWS_*)。用 `is_known_relic`(走
        # source_knowledge.sqlite)过滤,确保只有 sim 认识的 id 进 state。
        for c in floor_data.get("relic_choices", []) or []:
            if c.get("was_picked"):
                rid = normalize_relic_id(c.get("relic_id", ""))
                if rid and is_known_relic(rid):
                    self.relics.append(rid)
        for c in floor_data.get("ancient_choices", []) or []:
            if c.get("was_picked"):
                rid = normalize_relic_id(c.get("relic_id", ""))
                if rid and is_known_relic(rid) and rid not in self.relics:
                    self.relics.append(rid)

        # --- campfire card_upgrades ---
        for up in floor_data.get("card_upgrades", []) or []:
            upg_base, _u = normalize_card_id(up.get("card_id", ""))
            if upg_base:
                # 升级第一张匹配的 base card → 卡 id + "+"
                # normalize_card_id 已剥掉 "+" 后缀,所以 compare base↔base 即可
                for i, deck_cid in enumerate(self.deck):
                    base_i, u_i = normalize_card_id(deck_cid)
                    if base_i == upg_base and u_i == 0:
                        self.deck[i] = base_i + "+"
                        break

        # --- shop_actions: buy_card / buy_relic / buy_potion / remove ---
        # 实际 skada 字段约定(与校验报告一致):buy_* 不是 purchase_*
        for act in floor_data.get("shop_actions", []) or []:
            atype = str(act.get("action_type", "") or "").lower()
            iid_raw = act.get("item_id", "")
            if not iid_raw:
                continue
            if atype in ("buy_card", "purchase_card"):  # 兼容老字段名
                base, _u = normalize_card_id(iid_raw)
                if base:
                    self.deck.append(base)
            elif atype in ("buy_relic", "purchase_relic"):
                rid = normalize_relic_id(iid_raw)
                if rid and is_known_relic(rid):
                    self.relics.append(rid)
            elif atype in ("buy_potion", "purchase_potion"):
                pid = normalize_potion_id(iid_raw)
                if pid:
                    self.potions.append(pid)
            elif atype == "remove":
                base, _u = normalize_card_id(iid_raw)
                for i, deck_cid in enumerate(self.deck):
                    base_i, _ui = normalize_card_id(deck_cid)
                    if base_i == base:
                        self.deck.pop(i)
                        break

        # --- combat counter ---
        combat = floor_data.get("combat")
        if isinstance(combat, dict):
            self.combats_seen += 1
            enc_type = str(combat.get("enc_type", "") or "").upper()
            if enc_type == "E":
                self.elites_seen += 1
            elif enc_type == "B":
                self.bosses_seen += 1
            eid = str(combat.get("encounter", "") or "").upper()
            if eid:
                self.enemy_types_seen[eid] = self.enemy_types_seen.get(eid, 0) + 1

        # --- event ---
        ev = str(floor_data.get("event_text", "") or "")
        if ev:
            self.event_history.append(ev)

        # --- hp/gold post-snapshot ---
        self.hp = int(floor_data.get("hp_after", self.hp) or self.hp)
        self.gold = int(floor_data.get("gold_after", self.gold) or self.gold)
        self.total_hp_lost += max(0, int(floor_data.get("hp_before", 0) or 0) - int(floor_data.get("hp_after", 0) or 0))

        self.last_floor_seen = self.floor

    # ------------------------------------------------------------------
    # Snapshot → BanksContext
    # ------------------------------------------------------------------

    def snapshot_banks_context(self) -> BanksContext:
        """把当前状态打包成 TokenBankBuilder.build() 需要的所有 Python 对象。"""
        player_rt = PlayerRuntime(
            hp=self.hp,
            max_hp=max(self.max_hp, 1),
            block=0,
            energy=3,
            max_energy=3,
            powers={},
        )
        deck_cards = [_card_semantics_from_id(cid) for cid in self.deck]
        relics = [_relic_semantics_from_id(rid) for rid in self.relics]
        potions = [_potion_semantics_from_id(pid) for pid in self.potions]

        profile = _aggregate_deck_profile(self.deck)
        rbm = RunBuildMemory(
            build_identity="",        # TODO: detect_archetype
            deck_size=len(self.deck),
            frontload=profile["frontload"],
            block=profile["block"],
            draw=profile["draw"],
            scaling=profile["scaling"],
            aoe=profile["aoe"],
            heal=profile["heal"],
            curse_density=profile["curse_density"],
            high_cost_density=profile["high_cost_density"],
            zero_cost_density=profile["zero_cost_density"],
            x_cost_density=profile["x_cost_density"],
            consistency=profile["consistency"],
            act=_act_from_floor(self.floor),
            floor=self.floor,
            gold=self.gold,
            relic_count=len(self.relics),
            potion_count=len(self.potions),
            combats_seen=self.combats_seen,
            elites_seen=self.elites_seen,
            bosses_seen=self.bosses_seen,
            total_hp_lost=self.total_hp_lost,
            potions_used_total=0,
            enemy_types_seen=dict(self.enemy_types_seen),
            room_type_history=list(self.room_type_history),
            event_history=list(self.event_history),
        )

        return BanksContext(
            player_rt=player_rt,
            deck_cards=deck_cards,
            relics=relics,
            potions=potions,
            rbm=rbm,
        )


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

_ROOM_TYPE_MAP = {
    "M": "monster",
    "E": "elite",
    "B": "boss",
    "S": "shop",
    "R": "rest",
    "V": "event",
    "T": "event",    # treasure 走 event 桶(networkV2 内暂无独立 domain)
    "A": "event",    # ancient(古代神坛)
    "?": "event",
}


def _map_room_type(room_type_raw: str) -> str:
    return _ROOM_TYPE_MAP.get(str(room_type_raw or "").upper().strip(), "")


def _act_from_floor(floor: int) -> int:
    if floor <= 0:
        return 1
    if floor <= 17:
        return 1
    if floor <= 33:
        return 2
    if floor <= 48:
        return 3
    return 4


# ---------------------------------------------------------------------------
# 迭代器:按 timeline 顺序 yield (floor_data, state_snapshot_pre, state_snapshot_post)
# ---------------------------------------------------------------------------

def iter_timeline_with_state(rec: dict[str, Any]) -> Iterator[tuple[dict[str, Any], SkadaRunState, SkadaRunState]]:
    """遍历 floor_timeline,每层 yield (floor_data, pre_state, post_state)。

    pre_state:进入 floor 前的累计状态(已 apply hp_before/gold_before/floor)
    post_state:离开 floor 后的累计状态(已 apply 所有 choices + hp_after/gold_after)

    调用方在 pre_state 阶段做决策点 sample 产出(因为当时 deck/hp 还是"决策前"状态)。
    """
    import copy as _copy
    state = SkadaRunState.from_run_record(rec)
    for floor_data in rec.get("floor_timeline", []) or []:
        state.apply_floor_pre(floor_data)
        pre = _copy.deepcopy(state)
        state.apply_floor_post(floor_data)
        post = _copy.deepcopy(state)
        yield floor_data, pre, post
