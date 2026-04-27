"""Ironclad 常用卡牌的静态效果表。

sim 的 state 里手牌只给 `id / cost / type / can_play`，不给每张卡的
damage_now / block_now（实时值）。要启发式打分只能查表 + 简化估算。

这里只覆盖 Ironclad starter + 一些中期常见卡。覆盖不到的卡 fallback 到
`type` 字段（attack / skill / power）+ 粗糙估算。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class CardEffect:
    base_damage: float = 0.0
    upg_damage: float = 0.0
    base_block: float = 0.0
    upg_block: float = 0.0
    multi_hits: int = 1
    applies_vulnerable: int = 0
    applies_weak: int = 0
    is_power: bool = False
    is_aoe: bool = False


# 数值按 STS 常识填；STS2 具体数值可能略有差异，但对启发式打分足够
_IRONCLAD_CARDS: dict[str, CardEffect] = {
    "STRIKE_IRONCLAD": CardEffect(base_damage=6, upg_damage=9),
    "DEFEND_IRONCLAD": CardEffect(base_block=5, upg_block=8),
    "BASH": CardEffect(base_damage=8, upg_damage=10, applies_vulnerable=2),
    "BLUDGEON": CardEffect(base_damage=32, upg_damage=42),
    "ANGER": CardEffect(base_damage=6, upg_damage=8),
    "BODY_SLAM": CardEffect(base_damage=0),  # 伤害 = 当前 block
    "CLASH": CardEffect(base_damage=14, upg_damage=18),
    "CLEAVE": CardEffect(base_damage=8, upg_damage=11, is_aoe=True),
    "CLOTHESLINE": CardEffect(base_damage=12, upg_damage=14, applies_weak=2),
    "FLEX": CardEffect(is_power=False),  # 短暂 Strength
    "HAVOC": CardEffect(),
    "HEADBUTT": CardEffect(base_damage=9, upg_damage=12),
    "HEAVY_BLADE": CardEffect(base_damage=14, upg_damage=14),  # 被 Strength 加成
    "IRON_WAVE": CardEffect(base_damage=5, upg_damage=7, base_block=5, upg_block=7),
    "PERFECTED_STRIKE": CardEffect(base_damage=6, upg_damage=6),
    "POMMEL_STRIKE": CardEffect(base_damage=9, upg_damage=10),
    "SHRUG_IT_OFF": CardEffect(base_block=8, upg_block=11),
    "SWORD_BOOMERANG": CardEffect(base_damage=3, upg_damage=3, multi_hits=3),
    "THUNDERCLAP": CardEffect(base_damage=4, upg_damage=7, applies_vulnerable=1, is_aoe=True),
    "TRUE_GRIT": CardEffect(base_block=7, upg_block=9),
    "TWIN_STRIKE": CardEffect(base_damage=5, upg_damage=7, multi_hits=2),
    "UPPERCUT": CardEffect(base_damage=13, upg_damage=14, applies_weak=1),
    "WILD_STRIKE": CardEffect(base_damage=12, upg_damage=17),
    "INFLAME": CardEffect(is_power=True),
    "DEMON_FORM": CardEffect(is_power=True),
    "METALLICIZE": CardEffect(is_power=True),
    "FEEL_NO_PAIN": CardEffect(is_power=True),
    "DARK_EMBRACE": CardEffect(is_power=True),
    "JUGGERNAUT": CardEffect(is_power=True),
    # STS2 新卡 / smoke build 里出现过的
    "CINDER": CardEffect(base_damage=8, upg_damage=11),
    "FORGOTTEN_RITUAL": CardEffect(),
    "SETUP_STRIKE": CardEffect(base_damage=6, upg_damage=8),
    "HEAVY_FORGE": CardEffect(base_damage=10, upg_damage=14),
}


def lookup_effect(card_id: str, is_upgraded: bool = False) -> CardEffect:
    cid = (card_id or "").upper()
    eff = _IRONCLAD_CARDS.get(cid)
    if eff is not None:
        return eff
    # fallback：只靠 id 粗猜
    if "DEFEND" in cid:
        return CardEffect(base_block=5, upg_block=8)
    if "STRIKE" in cid:
        return CardEffect(base_damage=6, upg_damage=9)
    return CardEffect()


def effective_damage(card_id: str, is_upgraded: bool = False) -> float:
    eff = lookup_effect(card_id, is_upgraded)
    base = eff.upg_damage if is_upgraded and eff.upg_damage else eff.base_damage
    return float(base) * max(1, eff.multi_hits)


def effective_block(card_id: str, is_upgraded: bool = False) -> float:
    eff = lookup_effect(card_id, is_upgraded)
    base = eff.upg_block if is_upgraded and eff.upg_block else eff.base_block
    return float(base)


def card_is_attack(card_id: str) -> bool:
    return effective_damage(card_id) > 0


def card_is_defend(card_id: str) -> bool:
    return effective_block(card_id) > 0 and effective_damage(card_id) == 0


def card_is_debuff(card_id: str) -> bool:
    eff = lookup_effect(card_id)
    return bool(eff.applies_vulnerable or eff.applies_weak)


def card_is_power(card_id: str) -> bool:
    return lookup_effect(card_id).is_power


def card_effect_hint(card_id: str, is_upgraded: bool = False) -> str:
    """渲染用：一行紧凑的 effect 提示。"""
    eff = lookup_effect(card_id, is_upgraded)
    parts: list[str] = []
    dmg = eff.upg_damage if is_upgraded and eff.upg_damage else eff.base_damage
    blk = eff.upg_block if is_upgraded and eff.upg_block else eff.base_block
    if dmg:
        if eff.multi_hits > 1:
            parts.append(f"dmg={dmg}x{eff.multi_hits}")
        elif eff.is_aoe:
            parts.append(f"dmg={dmg}(AOE)")
        else:
            parts.append(f"dmg={dmg}")
    if blk:
        parts.append(f"blk={blk}")
    if eff.applies_vulnerable:
        parts.append(f"vuln={eff.applies_vulnerable}")
    if eff.applies_weak:
        parts.append(f"weak={eff.applies_weak}")
    if eff.is_power:
        parts.append("power")
    return " ".join(parts) if parts else "-"


__all__ = [
    "CardEffect",
    "card_effect_hint",
    "card_is_attack",
    "card_is_debuff",
    "card_is_defend",
    "card_is_power",
    "effective_block",
    "effective_damage",
    "lookup_effect",
]
