"""从 data/source_knowledge.sqlite 提取的真实游戏 vocab。

**规范**：见 docs/design/SCHEMA_CONVENTION.md —— 所有涉及游戏命名的 schema
必须来自真实数据，不得手写。本模块是唯一的 power/card/relic 名字来源。

每次 sqlite 重建后，本模块的 vocab 自动更新。

运行时：
  from networkV2.s1_schema.game_vocab import (
      MONSTER_POWER_VOCAB,        # 按频次排序的 monster power class 列表
      PLAYER_POWER_VOCAB,         # 按频次排序的 player（来自 cards + relics）
      symbol_index,               # 查询：name → index in global vocab
  )
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path


_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "source_knowledge.sqlite"


@lru_cache(maxsize=1)
def _load_vocabs() -> dict:
    """一次性加载所有 vocab 表，按频次排序返回。"""
    if not _DB_PATH.exists():
        # fallback：sqlite 不存在时返回空（测试环境）
        return {
            "monster_powers": [],
            "monster_powers_freq": {},
            "card_powers": [],
            "card_tags": [],
            "card_keywords": [],
            "relic_powers": [],
        }
    conn = sqlite3.connect(str(_DB_PATH))
    cur = conn.cursor()

    def _count_json_column(table: str, col: str) -> Counter:
        cnt = Counter()
        try:
            for (raw,) in cur.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL"):
                try:
                    for s in json.loads(raw):
                        cnt[str(s)] += 1
                except Exception:
                    pass
        except Exception:
            pass
        return cnt

    m_powers = _count_json_column("monsters", "powers_json")
    c_powers = _count_json_column("cards", "powers_json")
    c_tags = _count_json_column("cards", "tags_json")
    c_keys = _count_json_column("cards", "keywords_json")
    r_powers = _count_json_column("relics", "powers_json")

    conn.close()

    # player powers = card + relic 的 power 合并，按 **合并后频次** 排序。
    # P2-1 修复：原先写成 sorted(set(CARD_POWER_VOCAB) | set(RELIC_POWER_VOCAB))，
    # 是字母序 → bank_assembler._player_token 取前 17 个当 top-N 时选到的都是 A 开头
    # 的低频 power (Accelerant/Arsenal/Afterimage…)，而 Strength / Vulnerable /
    # Poison / Focus 这些高频的落不进 vocab slot。
    p_powers = Counter()
    p_powers.update(c_powers)
    p_powers.update(r_powers)

    return {
        "monster_powers": [name for name, _ in m_powers.most_common()],
        "monster_powers_freq": dict(m_powers),
        "card_powers": [name for name, _ in c_powers.most_common()],
        "card_tags": [name for name, _ in c_tags.most_common()],
        "card_keywords": [name for name, _ in c_keys.most_common()],
        "relic_powers": [name for name, _ in r_powers.most_common()],
        "player_powers": [name for name, _ in p_powers.most_common()],
    }


# 公开接口：按频次排序的 vocab（高频在前）

MONSTER_POWER_VOCAB: list[str] = _load_vocabs()["monster_powers"]
"""monster 使用的 power class（按全游戏 monster 出现频次排序）。
示例：['StrengthPower', 'FrailPower', 'WeakPower', 'VulnerablePower', ...]"""

CARD_POWER_VOCAB: list[str] = _load_vocabs()["card_powers"]
"""card 引用的 power class。"""

CARD_TAG_VOCAB: list[str] = _load_vocabs()["card_tags"]
CARD_KEYWORD_VOCAB: list[str] = _load_vocabs()["card_keywords"]

RELIC_POWER_VOCAB: list[str] = _load_vocabs()["relic_powers"]
"""relic 引用的 power class（玩家 power 的一部分来自这些）。"""

# Player power vocab = card + relic powers_json 合并后按真实频次排序的 class 列表。
# bank_assembler._player_token 会取前 N 个作为"top-N 高频 player power"的 vocab slot，
# 所以这里必须是频次序（见 _load_vocabs 里构造 `player_powers`）。
PLAYER_POWER_VOCAB: list[str] = _load_vocabs()["player_powers"]


def normalize_power_name(raw: str) -> str:
    """sim 返回的 power ID 归一化成 sqlite class name。

    sim 用 `FROG_KNIGHT_POWER` 格式，sqlite 用 `FrogKnightPower` 格式。
    两者之间需要映射。

    注意：runtime_compiler._normalize_power_id 做了 lowercase + 去后缀，
    但我们需要保持与 sqlite class name 格式一致。
    """
    s = raw.strip()
    if not s:
        return ""
    # 已经是 CamelCase + Power suffix → 直接返回
    if s[0].isupper() and s.endswith("Power"):
        return s
    # SCREAMING_CASE → CamelCase
    if "_" in s:
        parts = s.split("_")
        return "".join(p.capitalize() for p in parts)
    # lower_case → Capitalize (runtime_compiler 已做 lower + 去后缀)
    lower = s.lower()
    # 常见 lowercase → class name 映射（由 _compute_lowercase_map 动态派生）
    lower_to_class = _lowercase_to_class_map()
    if lower in lower_to_class:
        return lower_to_class[lower]
    return s  # fallback 原样


@lru_cache(maxsize=1)
def _lowercase_to_class_map() -> dict[str, str]:
    """把 sqlite vocab 的 class name 映射到 lowercase stem。
    使 runtime_compiler._normalize_power_id 输出（lowercase, 去 _Power）
    能查到 sqlite class。"""
    mapping: dict[str, str] = {}
    for cls_name in MONSTER_POWER_VOCAB + PLAYER_POWER_VOCAB:
        # class: StrengthPower → stem: strength
        stem = cls_name
        for suffix in ("Power", "power"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        stem = stem.lower()
        if stem not in mapping:
            mapping[stem] = cls_name
        # 也处理 SCREAMING_CASE 形式
        screaming_stem = stem  # 已 lower 过
        if screaming_stem != stem.lower():
            mapping[screaming_stem] = cls_name
    return mapping


def monster_power_slot(stem_lower: str) -> int:
    """给定 lowercase stem（如 'plating'），返回其在 MONSTER_POWER_VOCAB
    中的 index（按频次排名）。-1 = 未知 power。"""
    m = _lowercase_to_class_map()
    cls = m.get(stem_lower, "")
    if cls and cls in MONSTER_POWER_VOCAB:
        return MONSTER_POWER_VOCAB.index(cls)
    return -1


# ---------------------------------------------------------------------------
# Power 语义分组（power_bank token 的 semantic group one-hot）
# ---------------------------------------------------------------------------
#
# 设计：STS2 的 power class 通常继承自某个 base class（TimedPower / TriggerOnAttackedPower
# / DamageReductionPower / ...），base_classes 本身就是语义分组。优先用 game_catalog
# API 拿 base_classes；fallback 用 class 名 heuristic。
#
# 分组顺序固定，bank_assembler 的 power_instance_token numeric slot 会按此对齐。

POWER_SEMANTIC_GROUPS: list[str] = [
    "timed",                # 有时限（TimedPower / DamageNextTurn 等回合末过期）
    "trigger_on_hit",       # 受击触发（TriggerOnAttackedPower）
    "trigger_on_play",      # 出牌触发（TriggerOnCardPlayedPower）
    "trigger_on_turn",      # 回合末触发（TriggerEndOfTurnPower / RitualPower）
    "damage_reduction",     # 伤害 cap 类（Intangible/Slippery/HardToKill/Flight）
    "minion_spawn",         # 召唤类（MinionPower / SummonPower）
    "phase_transition",     # 阶段切换（PhaseTransitionPower / Transform）
    "other",                # 兜底
]
N_POWER_SEMANTIC_GROUPS = len(POWER_SEMANTIC_GROUPS)
POWER_GROUP_TO_IDX = {g: i for i, g in enumerate(POWER_SEMANTIC_GROUPS)}


# base_class → semantic group 映射（优先）
_BASE_CLASS_TO_GROUP = {
    "TimedPower": "timed",
    "TemporaryPower": "timed",
    "TriggerOnAttackedPower": "trigger_on_hit",
    "TriggerOnHitPower": "trigger_on_hit",
    "TriggerOnCardPlayedPower": "trigger_on_play",
    "TriggerOnCardPlayPower": "trigger_on_play",
    "TriggerEndOfTurnPower": "trigger_on_turn",
    "TriggerStartOfTurnPower": "trigger_on_turn",
    "DamageReductionPower": "damage_reduction",
    "MinionSpawnerPower": "minion_spawn",
    "PhaseTransitionPower": "phase_transition",
}

# class_name 模式 → group（heuristic fallback，用于没有清晰 base class 的 power）
_CLASS_NAME_PATTERNS = [
    # 伤害 cap
    ("intangible", "damage_reduction"),
    ("slippery", "damage_reduction"),
    ("hardtokill", "damage_reduction"),
    ("hard_to_kill", "damage_reduction"),
    ("flight", "damage_reduction"),
    ("illusion", "damage_reduction"),
    # 触发类
    ("thorns", "trigger_on_hit"),
    ("curlup", "trigger_on_hit"),
    ("curl_up", "trigger_on_hit"),
    ("angry", "trigger_on_hit"),
    ("modeshift", "trigger_on_hit"),
    ("mode_shift", "trigger_on_hit"),
    ("spore", "trigger_on_hit"),
    ("plated", "trigger_on_hit"),
    ("enrage", "trigger_on_play"),
    ("ritual", "trigger_on_turn"),
    ("plating", "trigger_on_turn"),
    # 召唤
    ("minion", "minion_spawn"),
    ("summon", "minion_spawn"),
    ("spawn", "minion_spawn"),
    # 阶段
    ("phase", "phase_transition"),
    ("transform", "phase_transition"),
    ("split", "phase_transition"),
    ("revival", "phase_transition"),
    # 时限
    ("nextturn", "timed"),
    ("next_turn", "timed"),
    ("timelimit", "timed"),
    ("battleworn", "timed"),
]


# cache：class_name → group idx
_POWER_GROUP_CACHE: dict[str, int] = {}


def _upgrade_to_class_name(class_name: str) -> str:
    """stem ("thorns") → ClassName ("ThornsPower")；传 ClassName 直接返回。

    runtime_compiler._normalize_power_id 把 sim 返回的 power id 转成了 lowercase stem
    （STRENGTH_POWER → strength / HardenedShell → hardenedshell），但 VOCAB 和基类表
    是 ClassName 格式。helper 内部做一次 upgrade，让 caller 传 stem / ClassName 都通。
    """
    if not class_name:
        return ""
    # 如果已经是 ClassName 格式（首字母大写）且存在于任一 vocab → 直接返回
    if class_name and class_name[0].isupper():
        if class_name in MONSTER_POWER_VOCAB or class_name in PLAYER_POWER_VOCAB:
            return class_name
    # 否则按 lowercase stem 处理
    stem = class_name.lower().strip().strip("_")
    return _lowercase_to_class_map().get(stem, class_name)


def _resolve_power_group(class_name: str, base_classes: list[str] | None = None) -> int:
    """返回 power class 的 semantic group idx。

    优先级：
      1. base_classes 里的直接 parent（如 "TriggerOnAttackedPower"）→ 映射到 group
      2. class_name lowercase 的 pattern 匹配
      3. "other"
    """
    if not class_name:
        return POWER_GROUP_TO_IDX["other"]
    cache_key = class_name
    if cache_key in _POWER_GROUP_CACHE:
        return _POWER_GROUP_CACHE[cache_key]
    group = None
    # 1) base_classes 查表
    if base_classes:
        for base in base_classes:
            g = _BASE_CLASS_TO_GROUP.get(base)
            if g:
                group = g
                break
    # 2) class name heuristic（stem 和 ClassName 都能命中 pattern，因为 "thorns" in
    #    "thornspower" 也命中）
    if group is None:
        lower = class_name.lower()
        for pattern, g in _CLASS_NAME_PATTERNS:
            if pattern in lower:
                group = g
                break
    # 3) fallback
    if group is None:
        group = "other"
    idx = POWER_GROUP_TO_IDX[group]
    _POWER_GROUP_CACHE[cache_key] = idx
    return idx


def power_semantic_group_onehot(
    class_name: str,
    base_classes: list[str] | None = None,
) -> list[float]:
    """返回 N_POWER_SEMANTIC_GROUPS 维 one-hot，用于 power_instance_token 的 numeric。"""
    idx = _resolve_power_group(class_name, base_classes)
    out = [0.0] * N_POWER_SEMANTIC_GROUPS
    out[idx] = 1.0
    return out


def power_class_idx_normalized(class_name: str, is_player: bool) -> float:
    """返回 class_name 在对应 vocab 里的归一化 index（[0, 1]），-1 归一化到 0。

    - is_player=True → 在 PLAYER_POWER_VOCAB 里查
    - is_player=False → 在 MONSTER_POWER_VOCAB 里查
    未知 class 返回 0.0（同 index 0 —— 歧义但至少稳定）。

    自动 upgrade：runtime_compiler 传来的 stem ("thorns") 会先转成
    ClassName ("ThornsPower") 再查 vocab，避免 "对齐 bug"。
    """
    vocab = PLAYER_POWER_VOCAB if is_player else MONSTER_POWER_VOCAB
    if not vocab or not class_name:
        return 0.0
    upgraded = _upgrade_to_class_name(class_name)
    try:
        idx = vocab.index(upgraded)
        return float(idx) / max(len(vocab), 1)
    except ValueError:
        # fallback 再试原值（可能原本就是 ClassName 但不在此 vocab）
        try:
            idx = vocab.index(class_name)
            return float(idx) / max(len(vocab), 1)
        except ValueError:
            return 0.0


# is_debuff heuristic（fallback；优先用 game_catalog 的 is_debuff_hint）
_DEBUFF_CLASS_STEMS = frozenset({
    "weak", "vulnerable", "frail", "poison", "lockon", "entangled",
    "confusion", "slow", "suck", "stock",
})


def is_debuff_heuristic(class_name: str) -> bool:
    """当 game_catalog 的 is_debuff_hint 不可用时，按 class 名 stem 推断 buff/debuff。

    接受 stem ("weak") 或 ClassName ("WeakPower") —— 都正确命中。
    """
    if not class_name:
        return False
    lower = class_name.lower()
    # 去 power 后缀（ClassName 和 stem 都能命中）
    for suf in ("power",):
        if lower.endswith(suf):
            lower = lower[: -len(suf)]
            break
    lower = lower.strip("_")
    return lower in _DEBUFF_CLASS_STEMS or "debuff" in lower
