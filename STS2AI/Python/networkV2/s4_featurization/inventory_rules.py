"""Relic / Potion 静态规则表。

为什么需要：bridge 只发 relic id，token_bank_builder 原本用 [1.0] 占位，网络完全看不到
"这个 relic 做什么"。此表提供离线规则，让 token 能编码 relic 的功能语义。

维度设计（和 token numeric 对齐）：
  - trigger tags (6 维)：on_combat_start / on_turn_start / on_play_card /
                        on_enemy_hit / on_turn_end / on_combat_end
  - effect tags (6 维)：damage / block / heal / energy / draw / scaling
  - strength hints (2 维)：base_value / conditional
共 14 维，扩展 relic_token 和 potion_token 时直接拼到 numeric 后面。

覆盖度：Ironclad 核心 relic + Act-1 boss/elite/shop relic + 通用 relic。
未覆盖的 relic 默认全 0（和占位态一致，不变差）。
"""
from __future__ import annotations


# --- 标签词典（与 token numeric 顺序对齐）---
TRIGGER_TAGS = ("combat_start", "turn_start", "on_play", "on_hit", "turn_end", "combat_end")
EFFECT_TAGS = ("damage", "block", "heal", "energy", "draw", "scaling")


def _upper(s: str) -> str:
    return (s or "").strip().upper().replace(" ", "_")


# --- Relic 规则：id → (triggers, effects, base_value, conditional) ---
# base_value: 该 relic 的主要数值效果（归一化到 0-1 用 /10）
# conditional: 是否有条件触发（比如 Dead Branch 只在 exhaust 时触发）
RELIC_RULES: dict[str, tuple[frozenset[str], frozenset[str], float, bool]] = {
    # --- 起始 relic ---
    "BURNING_BLOOD":        (frozenset({"combat_end"}),               frozenset({"heal"}),              6, False),
    "RING_OF_THE_SNAKE":    (frozenset({"combat_start"}),             frozenset({"draw"}),              2, False),
    "CRACKED_CORE":         (frozenset({"combat_start"}),             frozenset({"damage"}),            1, False),
    "PURE_WATER":           (frozenset({"combat_start"}),             frozenset({"draw"}),              1, False),

    # --- Act-1 Boss relic（常见爆发性）---
    "BLACK_STAR":           (frozenset({"combat_end"}),               frozenset({}),                    0, True),
    "BUSTED_CROWN":         (frozenset({"combat_start"}),             frozenset({"energy"}),            1, True),
    "COFFEE_DRIPPER":       (frozenset({"combat_start"}),             frozenset({"energy"}),            1, True),
    "CURSED_KEY":           (frozenset({"combat_start"}),             frozenset({"energy"}),            1, True),
    "ECTOPLASM":            (frozenset({"combat_start"}),             frozenset({"energy"}),            1, True),
    "EMPTY_CAGE":           (frozenset({}),                           frozenset({}),                    0, True),
    "FUSION_HAMMER":        (frozenset({"combat_start"}),             frozenset({"energy"}),            1, True),
    "PANDORAS_BOX":         (frozenset({}),                           frozenset({}),                    0, True),
    "PHILOSOPHERS_STONE":   (frozenset({"combat_start"}),             frozenset({"energy"}),            1, True),
    "RUNIC_DOME":           (frozenset({"combat_start"}),             frozenset({"energy"}),            1, True),
    "SOZU":                 (frozenset({"combat_start"}),             frozenset({"energy"}),            1, True),
    "VELVET_CHOKER":        (frozenset({"combat_start"}),             frozenset({"energy"}),            1, True),

    # --- 常用战斗类 ---
    "ANCHOR":               (frozenset({"combat_start"}),             frozenset({"block"}),             10, False),
    "AKABEKO":              (frozenset({"combat_start"}),             frozenset({"damage"}),            8, False),
    "VAJRA":                (frozenset({"combat_start"}),             frozenset({"damage", "scaling"}), 1, False),
    "OODOO_STRING":         (frozenset({"on_play"}),                  frozenset({"damage"}),            1, True),
    "BRONZE_SCALES":        (frozenset({"on_hit"}),                   frozenset({"damage"}),            3, False),
    "ORICHALCUM":           (frozenset({"turn_end"}),                 frozenset({"block"}),             6, True),
    "CALIPERS":             (frozenset({"turn_start"}),               frozenset({"block"}),             0, True),
    "HORN_CLEAT":           (frozenset({"turn_start"}),               frozenset({"block"}),             14, True),
    "CAPTAINS_WHEEL":       (frozenset({"turn_start"}),               frozenset({"block"}),             18, True),

    # --- 能量/抽牌类 ---
    "RUNIC_PYRAMID":        (frozenset({"turn_end"}),                 frozenset({"draw"}),              0, False),
    "BAG_OF_MARBLES":       (frozenset({"combat_start"}),             frozenset({"scaling"}),           1, False),
    "BAG_OF_PREPARATION":   (frozenset({"combat_start"}),             frozenset({"draw"}),              2, False),
    "COFFEE":               (frozenset({"combat_start"}),             frozenset({"energy"}),            1, False),
    "DATA_DISK":            (frozenset({"combat_start"}),             frozenset({"scaling"}),           1, False),
    "HAPPY_FLOWER":         (frozenset({"turn_start"}),               frozenset({"energy"}),            1, True),
    "MERCURY_HOURGLASS":    (frozenset({"turn_start"}),               frozenset({"damage"}),            3, False),
    "POCKETWATCH":          (frozenset({"turn_end"}),                 frozenset({"draw"}),              3, True),
    "TURNIP":               (frozenset({"on_hit"}),                   frozenset({}),                    0, False),

    # --- 治疗/生存 ---
    "BLOODY_IDOL":          (frozenset({"on_hit"}),                   frozenset({"heal"}),              5, True),
    "BLUE_CANDLE":          (frozenset({"on_play"}),                  frozenset({}),                    0, True),
    "MEAT_ON_THE_BONE":     (frozenset({"combat_end"}),               frozenset({"heal"}),              12, True),
    "MAGIC_FLOWER":         (frozenset({}),                           frozenset({"heal"}),              0, False),

    # --- 经济类 ---
    "GOLDEN_IDOL":          (frozenset({}),                           frozenset({}),                    0, False),
    "OLD_COIN":             (frozenset({}),                           frozenset({}),                    0, False),
    "SMILING_MASK":         (frozenset({}),                           frozenset({}),                    0, False),
}


def relic_feature_vector(relic_id: str) -> list[float]:
    """返回 14 维特征：6 trigger + 6 effect + base_value + conditional。

    未知 relic 返回全 0（和占位态等价，不会比原先差）。
    """
    rule = RELIC_RULES.get(_upper(relic_id))
    if rule is None:
        return [0.0] * 14
    triggers, effects, base_value, conditional = rule
    out: list[float] = []
    out.extend(float(t in triggers) for t in TRIGGER_TAGS)
    out.extend(float(t in effects) for t in EFFECT_TAGS)
    out.append(min(base_value / 10.0, 1.0))
    out.append(float(conditional))
    return out


# --- Potion 规则 ---
# potion_type 本身 runtime_extractor 可能已经填了；这里提供效果估计
POTION_RULES: dict[str, tuple[str, float, bool]] = {
    # id → (type, magnitude, combat_only)
    "ATTACK_POTION":        ("damage",  20, True),
    "BLOCK_POTION":         ("block",   12, True),
    "BLOOD_POTION":         ("heal",    25, False),   # 25% max HP
    "DEXTERITY_POTION":     ("buff",     2, True),
    "ENERGY_POTION":        ("energy",   2, True),
    "ESSENCE_OF_STEEL":     ("block",    4, True),
    "EXPLOSIVE_POTION":     ("damage",  10, True),
    "FAIRY_IN_A_BOTTLE":    ("heal",    30, False),   # 30% on fatal
    "FEAR_POTION":          ("debuff",   3, True),
    "FIRE_POTION":          ("damage",  20, True),
    "FLEX_POTION":          ("buff",     5, True),
    "FRUIT_JUICE":          ("heal",     5, False),   # +5 max HP
    "GAMBLERS_BREW":        ("draw",     0, True),
    "GHOST_IN_A_JAR":       ("buff",     1, True),    # intangible 1
    "HEROIC_POTION":        ("buff",     1, True),
    "LIQUID_BRONZE":        ("buff",     3, True),
    "LIQUID_MEMORIES":      ("draw",     1, True),
    "POTION_OF_CAPACITY":   ("buff",     2, False),
    "POWER_POTION":         ("draw",     1, True),
    "REGEN_POTION":         ("heal",     5, True),
    "SKILL_POTION":         ("draw",     1, True),
    "SMOKE_BOMB":           ("escape",   0, True),
    "SNECKO_OIL":           ("draw",     5, True),
    "SPEED_POTION":         ("block",   20, True),
    "STANCE_POTION":        ("buff",     1, True),
    "STRENGTH_POTION":      ("buff",     2, True),
    "SWIFT_POTION":         ("draw",     3, True),
    "WEAK_POTION":          ("debuff",   3, True),
}


def potion_feature_vector(potion_id: str) -> list[float]:
    """返回 8 维特征：6 effect one-hot + magnitude + combat_only。"""
    rule = POTION_RULES.get(_upper(potion_id))
    effect_onehot = [0.0] * 6  # damage/block/heal/energy/draw/scaling
    if rule is None:
        return effect_onehot + [0.0, 0.0]
    ptype, magnitude, combat_only = rule
    effect_map = {
        "damage": 0, "block": 1, "heal": 2,
        "energy": 3, "draw": 4,
        "buff": 5, "debuff": 5, "escape": 5,  # scaling 槽放 buff/debuff/escape
    }
    idx = effect_map.get(ptype)
    if idx is not None:
        effect_onehot[idx] = 1.0
    return effect_onehot + [min(magnitude / 25.0, 1.0), float(combat_only)]
