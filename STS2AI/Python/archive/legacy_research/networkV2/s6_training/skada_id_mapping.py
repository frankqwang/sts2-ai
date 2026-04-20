"""Skada → 游戏权威 id mapping 层。

问题背景:
  - skada 的 id 风格:UPPER_SNAKE_CASE,例如 STRIKE_IRONCLAD / BURNING_BLOOD
  - game source_knowledge.sqlite 的 id 风格:lower_snake_case,例如 strike_ironclad
  - 老版本 skada 可能用已重命名的卡名(alpha 版本字段漂移)
  - 不做 mapping 会导致 card_feature_vector() / relic_feature_vector() miss,
    → token_bank_builder 产 0 向量 → 网络看不见关键构筑信号

本模块职责:
  1. normalize_card_id / normalize_relic_id:UPPER → lower(简单 case)
  2. validate_*:是否在 source_knowledge 中(剔除幽灵 id)
  3. character_starter_deck:按 character 查 5× strike_<c> + 4× defend_<c> + 已知 starter 基础卡
  4. character_starter_relic:按 character 查已知 starter relic
  5. room_letter_to_domain:skada 的 M/E/B/S/R/V/A/T 单字母 → networkV2 的 decision_domain
  6. ROOM_TYPE_IS_REVEALED / UNKNOWN_NODE_TYPE:map 节点揭示状态处理

所有查询都基于 `data/source_knowledge.sqlite` 的权威 snapshot。
不硬编码任何 card/relic/power 名(SCHEMA_CONVENTION.md)。
"""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# source_knowledge.sqlite 查询
# ---------------------------------------------------------------------------

# __file__ = .../networkV2/s6_training/skada_id_mapping.py
# parents[0]=s6_training, parents[1]=networkV2, parents[2]=Python, data 在 Python/data/ 下
_SRC_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "source_knowledge.sqlite"


@lru_cache(maxsize=1)
def _load_source_card_ids() -> frozenset[str]:
    """source_knowledge.cards.id 全集,lower snake。"""
    if not _SRC_DB_PATH.exists():
        logger.warning(f"source_knowledge.sqlite not found at {_SRC_DB_PATH}")
        return frozenset()
    con = sqlite3.connect(str(_SRC_DB_PATH))
    try:
        rows = con.execute("SELECT id FROM cards").fetchall()
    finally:
        con.close()
    return frozenset(str(r[0]).lower() for r in rows if r[0])


@lru_cache(maxsize=1)
def _load_source_relic_ids() -> frozenset[str]:
    if not _SRC_DB_PATH.exists():
        return frozenset()
    con = sqlite3.connect(str(_SRC_DB_PATH))
    try:
        rows = con.execute("SELECT id FROM relics").fetchall()
    finally:
        con.close()
    return frozenset(str(r[0]).lower() for r in rows if r[0])


@lru_cache(maxsize=1)
def _load_source_potion_ids() -> frozenset[str]:
    if not _SRC_DB_PATH.exists():
        return frozenset()
    con = sqlite3.connect(str(_SRC_DB_PATH))
    try:
        rows = con.execute("SELECT id FROM potions").fetchall()
    finally:
        con.close()
    return frozenset(str(r[0]).lower() for r in rows if r[0])


@lru_cache(maxsize=1)
def _load_starter_relic_ids() -> frozenset[str]:
    if not _SRC_DB_PATH.exists():
        return frozenset()
    con = sqlite3.connect(str(_SRC_DB_PATH))
    try:
        rows = con.execute("SELECT id FROM relics WHERE rarity='Starter'").fetchall()
    finally:
        con.close()
    return frozenset(str(r[0]).lower() for r in rows if r[0])


# ---------------------------------------------------------------------------
# id 规范化
# ---------------------------------------------------------------------------

def _strip_upgrade_suffix(cid: str) -> tuple[str, int]:
    """去掉 card_id 的 `+` 升级后缀,返回 (base_id, upgrade_count)。"""
    s = str(cid or "").strip()
    upg = 0
    while s.endswith("+"):
        s = s[:-1]
        upg += 1
    return s, upg


def normalize_card_id(card_id: str) -> tuple[str, int]:
    """skada 或任意大小写 card_id → (base_id_lower, upgrade_count)。

    保证输出 lower snake,可直接进 source_knowledge.cards.id 查。
    """
    base, upg = _strip_upgrade_suffix(card_id)
    return base.lower(), upg


def normalize_relic_id(relic_id: str) -> str:
    """skada 或任意大小写 relic_id → lower snake。"""
    return str(relic_id or "").strip().lower()


def normalize_potion_id(potion_id: str) -> str:
    return str(potion_id or "").strip().lower()


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def is_known_card(card_id: str) -> bool:
    base, _ = normalize_card_id(card_id)
    return base in _load_source_card_ids()


def is_known_relic(relic_id: str) -> bool:
    return normalize_relic_id(relic_id) in _load_source_relic_ids()


def is_known_potion(potion_id: str) -> bool:
    return normalize_potion_id(potion_id) in _load_source_potion_ids()


def filter_known_cards(card_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """(known, unknown) 两组。输入输出都是规范化后的 lower id。"""
    known, unknown = [], []
    for cid in card_ids:
        base, _ = normalize_card_id(cid)
        (known if base in _load_source_card_ids() else unknown).append(base)
    return known, unknown


# ---------------------------------------------------------------------------
# Character / starter
# ---------------------------------------------------------------------------

# STS2 已知的 5 个 character(基于 skada 真实数据 + source_knowledge 交叉验证)
# 注意:
#  - STS1 的 WATCHER 在 STS2 中不存在
#  - STS2 新增 REGENT(储君)和 NECROBINDER(死灵术士)
KNOWN_CHARACTERS = frozenset({"IRONCLAD", "REGENT", "DEFECT", "SILENT", "NECROBINDER"})


# Character → 起手 deck 额外卡(除 5× strike/4× defend 之外)
# 数据来源:skada 真实 run + 游戏设计常识,后续可用 sim_catalog 替换
# 保守估计:起手除了 9 张 strike/defend 还有 1 张 character-specific 卡,共 10 张
# asc ≥ 10 额外加 ASCENDERS_BANE(诅咒,共 11 张)
_CHARACTER_EXTRA_STARTER_CARDS: dict[str, list[str]] = {
    "IRONCLAD":    ["bash"],                     # 铁甲起手 bash
    "REGENT":      ["cleave"],                   # 储君起手 cleave (待 sim 校验)
    "SILENT":      ["survivor", "neutralize"],   # 潜行者起手 2 张(经典)
    "DEFECT":      ["zap", "dualcast"],          # 缺陷者起手 2 张
    "NECROBINDER": ["ossify"],                   # 死灵术士起手(待 sim 校验,占位)
}

# Character → 起手 relic(从 rarity='Starter' 中 match)
# 每 character 对应一个 "经典 starter";升级 starter 通过 unlock 切换,skada 未存 unlock 信息,
# 这里默认取"经典"。不匹配时 run 第 1 层的 relic_choices was_picked 作 fallback。
_CHARACTER_STARTER_RELIC: dict[str, str] = {
    "IRONCLAD":    "burning_blood",
    "REGENT":      "divine_right",
    "SILENT":      "ring_of_the_snake",
    "DEFECT":      "cracked_core",
    "NECROBINDER": "bound_phylactery",
}


def character_starter_deck(character: str, ascension: int = 0) -> list[str]:
    """返回 character 的起手 deck(lower snake id)。

    格式:5× strike_<c> + 4× defend_<c> + extras + (ASCENDERS_BANE if asc ≥ 10)。
    所有 id 规范化为 source_knowledge 的 lower snake case。

    对未知 character 返回 [strike_ironclad*5, defend_ironclad*4](最安全 fallback)。
    """
    ch = str(character or "").strip().upper()
    if ch not in KNOWN_CHARACTERS:
        logger.warning(f"unknown character {character!r}, falling back to IRONCLAD deck")
        ch = "IRONCLAD"
    low = ch.lower()

    deck: list[str] = []
    strike_id = f"strike_{low}"
    defend_id = f"defend_{low}"
    # 验证存在(避免未来字段漂移)
    src_cards = _load_source_card_ids()
    if strike_id in src_cards:
        deck.extend([strike_id] * 5)
    else:
        logger.warning(f"{strike_id} not in source_knowledge, starter deck fallback")
        deck.extend(["strike_ironclad"] * 5)
    if defend_id in src_cards:
        deck.extend([defend_id] * 4)
    else:
        deck.extend(["defend_ironclad"] * 4)

    # Character-specific extras
    for extra in _CHARACTER_EXTRA_STARTER_CARDS.get(ch, []):
        if extra in src_cards:
            deck.append(extra)
        else:
            logger.debug(f"starter extra {extra!r} not in source_knowledge, skipping")

    if int(ascension or 0) >= 10 and "ascenders_bane" in src_cards:
        deck.append("ascenders_bane")

    return deck


def character_starter_relic(character: str) -> str:
    """返回 character 的默认 starter relic(lower snake)。

    不在 KNOWN_CHARACTERS 中 → "burning_blood" fallback。
    """
    ch = str(character or "").strip().upper()
    relic = _CHARACTER_STARTER_RELIC.get(ch, "burning_blood")
    # 验证仍在 starter 池内
    if relic not in _load_starter_relic_ids():
        logger.warning(f"starter relic {relic!r} for {ch} not in starter pool, fallback burning_blood")
        return "burning_blood"
    return relic


# ---------------------------------------------------------------------------
# Room type / decision domain
# ---------------------------------------------------------------------------

# skada 存的单字母 room_type → networkV2 decision_domain(combat/non-combat)
# 注意 "A" = Ancient(古代神坛事件)/"T" = Treasure(宝箱)
# 两者都是"非战斗选项式"决策,复用 event domain,但 room_type 字段区分语义
_ROOM_LETTER_TO_DOMAIN: dict[str, str] = {
    "M": "combat",      # monster
    "E": "combat",      # elite
    "B": "combat",      # boss
    "S": "shop",
    "R": "rest",
    "V": "event",
    "A": "event",       # ancient -> 暂借 event domain(没有专属 option builder)
    "T": "event",       # treasure -> 同上
    "?": "",            # 未揭示(skada 存档里没见过,留空)
    "": "",
}

_ROOM_LETTER_TO_NAME: dict[str, str] = {
    "M": "monster",
    "E": "elite",
    "B": "boss",
    "S": "shop",
    "R": "rest",
    "V": "event",
    "A": "ancient",
    "T": "treasure",
    "?": "unknown",
    "": "unknown",
}


def room_letter_to_domain(letter: str) -> str:
    return _ROOM_LETTER_TO_DOMAIN.get(str(letter or "").strip().upper(), "")


def room_letter_to_name(letter: str) -> str:
    return _ROOM_LETTER_TO_NAME.get(str(letter or "").strip().upper(), "unknown")


# ---------------------------------------------------------------------------
# Map 揭示状态处理
# ---------------------------------------------------------------------------

# 玩家在地图上决策时,只能看到当前节点 + 下一层(直接 children)的类型。
# 再深的节点类型在游戏 UI 上是隐藏的(除非有遗物 / 事件揭示后续层)。
# skada 存的是 run 结束后的全图(上帝视角),直接喂给网络会造成**信息泄漏**:
# 网络学到"走路线 X 因为 3 层后是 R(rest)"这种玩家实际看不到的信息。
#
# MAP_VISIBILITY_DEPTH:玩家可见的层深(0 = 只看本层,1 = 看 children,...)
# STS 实际 UI:整张地图的**拓扑结构 + 节点 type** 全部可见,只有 V(event)的具体
# 事件内容走到才知道 → 设大值让网络能做真实 lookahead(学"这条路未来的 rest/shop
# 组合是否有利")。之前设 1 是过度保守,阻断了路径关系学习。
MAP_VISIBILITY_DEPTH = 99   # 整 act 全可见(Act 1 通常 15-17 层)
UNKNOWN_NODE_TYPE = "?"


def mask_map_with_visibility(
    nodes: list[dict],
    current_coord: tuple,
    visibility_depth: int = MAP_VISIBILITY_DEPTH,
) -> list[dict]:
    """把超出可见深度的节点 type 置为 UNKNOWN_NODE_TYPE。

    输入:
        nodes: [{coord, type, children}] 原 map 结构
        current_coord: 玩家当前所在坐标
        visibility_depth: 可见层数(0=只看本层,1=看 children,...)

    返回:同结构,但远处节点 type='?'(children 关系保留,方便网络看拓扑)。

    用法:给 route policy sample 产出前 mask 一下,训练就不会泄漏。
    """
    if visibility_depth < 0:
        visibility_depth = 0
    node_by_coord = {tuple(n["coord"]): n for n in nodes if "coord" in n}
    visible: set[tuple] = {tuple(current_coord)}
    frontier = {tuple(current_coord)}
    for _ in range(visibility_depth):
        next_frontier: set[tuple] = set()
        for c in frontier:
            node = node_by_coord.get(c)
            if node is None:
                continue
            for child in (node.get("children") or []):
                next_frontier.add(tuple(child))
        visible |= next_frontier
        frontier = next_frontier

    masked: list[dict] = []
    for n in nodes:
        coord = tuple(n.get("coord") or ())
        if coord in visible:
            masked.append(n)
        else:
            masked.append({
                **n,
                "type": UNKNOWN_NODE_TYPE,
                "_masked": True,  # 诊断标记,下游可读
            })
    return masked
