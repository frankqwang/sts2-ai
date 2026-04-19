#!/usr/bin/env python3
"""把 sim_logs/*.log 里的 raw C# log 清洗成人类可读的中文战斗轨迹。

用途:排查 AI 为什么打得好/差 — 看一场完整 combat 的每步决策。

输入:Artifacts/sim_logs/sim_port{PORT}_{TS}.log(sim 的 stdout/stderr)
输出:控制台 / 文件,每场 combat 一个段落,step 按行分开

用法::

    # 列出 log 里所有 combat 起始点
    python -m diagnostics.clean_combat_trajectory Artifacts/sim_logs/sim_port19600_X.log --list

    # 清洗特定 encounter 的所有战斗
    python -m diagnostics.clean_combat_trajectory <log> --encounter CEREMONIAL_BEAST

    # 清洗第 N 场 combat(按出现顺序)
    python -m diagnostics.clean_combat_trajectory <log> --combat-index 3

    # 筛选胜利 / 失败 / 关键战斗(有 boss 怪物出现的)
    python -m diagnostics.clean_combat_trajectory <log> --bosses-only
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Card / Monster / Move 中文翻译表 (基础,覆盖常见 id)
# ---------------------------------------------------------------------------

CARD_ZH: dict[str, str] = {
    # Ironclad
    "STRIKE_IRONCLAD": "打击(铁甲)", "DEFEND_IRONCLAD": "防御(铁甲)", "BASH": "重击",
    "ANGER": "愤怒", "BODY_SLAM": "猛砸", "CLEAVE": "劈砍", "CLOTHESLINE": "臂铠",
    "HEAVY_BLADE": "重剑", "IRON_WAVE": "铁浪", "PERFECTED_STRIKE": "完美打击",
    "POMMEL_STRIKE": "柄击", "SHRUG_IT_OFF": "抖落", "SWORD_BOOMERANG": "剑回旋",
    "THUNDERCLAP": "霹雳", "TWIN_STRIKE": "双重打击", "WILD_STRIKE": "狂乱挥击",
    "BLOODLETTING": "放血", "BLOOD_FOR_BLOOD": "血债血偿", "BURNING_PACT": "焚烧契约",
    "DARK_SHACKLES": "黑色镣铐", "DUALCAST": "双重施法", "DROPKICK": "飞踢",
    "FEEL_NO_PAIN": "无痛", "FLAME_BARRIER": "烈焰护盾", "GHOSTLY_ARMOR": "幽灵护甲",
    "HEMOKINESIS": "操控血液", "INFLAME": "助燃", "INTIMIDATE": "威吓",
    "METALLICIZE": "金属化", "POWER_THROUGH": "破势", "PUMMEL": "重击",
    "RAGE": "暴怒", "RAMPAGE": "狂暴", "RECKLESS_CHARGE": "鲁莽冲锋",
    "SEARING_BLOW": "灼热之击", "SECOND_WIND": "二次呼吸", "SEEING_RED": "杀红眼",
    "SENTINEL": "哨兵", "SEVER_SOUL": "斩魂", "SHOCKWAVE": "冲击波",
    "SPOT_WEAKNESS": "寻机", "TRUE_GRIT": "真正的毅力", "WHIRLWIND": "旋风",
    # Silent
    "STRIKE_SILENT": "打击(静默)", "DEFEND_SILENT": "防御(静默)",
    "NEUTRALIZE": "中和", "SURVIVOR": "幸存者", "DEADLY_POISON": "致命毒药",
    "DAGGER_SPRAY": "飞刀乱射", "DAGGER_THROW": "飞刀",
    # Defect
    "STRIKE_DEFECT": "打击(故障)", "DEFEND_DEFECT": "防御(故障)", "DUALCAST": "双重施法",
    "ZAP": "电击", "COOLHEADED": "冷静", "GLACIER": "冰川",
    "COLD_SNAP": "急冻", "COMPILE_DRIVER": "编译驱动", "HOLOGRAM": "全息图",
    "GO_FOR_THE_EYES": "戳眼", "STEAM_BARRIER": "蒸汽屏障",
    # Regent
    "STRIKE_REGENT": "打击(摄政)", "DEFEND_REGENT": "防御(摄政)", "RADIATE": "辐射",
    "TYRANNY": "暴政", "CHILD_OF_THE_STARS": "群星之子",
    # Necrobinder
    "STRIKE_NECROBINDER": "打击(缚骸者)", "DEFEND_NECROBINDER": "防御(缚骸者)",
    "FEAR": "恐惧", "SCOURGE": "鞭笞",
    # Status / Curse
    "WOUND": "负伤", "SLIMED": "黏液", "DAZED": "眩晕", "BURN": "燃烧",
    "VOID": "虚空", "ASCENDERS_BANE": "登顶之祸", "SHAME": "耻辱",
    "NECRONOMICURSE": "死灵诅咒",
    # Common
    "END_TURN": "结束回合",
}

MONSTER_ZH: dict[str, str] = {
    "JAW_WORM": "颚虫", "SPIKER": "尖刺怪", "SHRINKER_BEETLE": "缩小甲虫",
    "BOWLBUG_ROCK": "碗虫(岩石)", "BOWLBUG_NECTAR": "碗虫(花蜜)",
    "BOWLBUG_SILK": "碗虫(丝绸)",
    "EXOSKELETON": "外骨骼", "MITE": "螨虫", "TORCH_HEAD": "火把头",
    "CEREMONIAL_BEAST": "祭祀之兽", "KAISER_CRAB": "凯撒蟹",
    "TEST_SUBJECT": "实验体", "KNOWLEDGE_DEMON": "知识恶魔",
    "THE_INSATIABLE": "无尽饥渴", "LAGAVULIN_MATRIARCH": "拉格维林女族长",
    "VANTOM": "幻灵", "DOORMAKER": "造门者", "QUEEN": "女王",
    "WATERFALL_GIANT": "瀑布巨人", "THE_KIN": "血亲",
    "SOUL_FYSH": "灵魂鱼", "TERROR_EEL": "恐怖鳗",
    "FROG_KNIGHT": "青蛙骑士", "SCROLL_OF_BITING": "啃咬卷轴",
    "DEVOTED_SCULPTOR": "虔诚雕塑家", "TURRET_OPERATOR": "炮台操作员",
    "CHOMPER": "咬噬者", "ENTOMANCER": "蠕虫师",
    "THIEVING_HOPPER": "偷盗跳蚤", "TUNNELER": "钻地者",
    "MECHA_KNIGHT": "机械骑士", "THE_OBSCURA": "晦暗者",
    "LIVING_SHIELD": "活化之盾", "TOUGH_EGG": "坚韧卵",
    "HATCHLING": "幼体", "OVICOPTER": "卵舰",
    "CORPSE_SLUG": "尸体蛞蝓", "AXEBOT": "斧机",
    "SOUL_NEXUS": "灵魂枢纽", "MYTE": "迈特",
}

MOVE_ZH: dict[str, str] = {
    "STAMP_MOVE": "践踏", "PLOW_MOVE": "犁耕", "CHARGE_MOVE": "冲锋",
    "STRIKE_MOVE": "打击", "BITE_MOVE": "咬", "CHOMP_MOVE": "啃",
    "BASH_MOVE": "重击", "SMASH_MOVE": "猛砸", "SLAM_MOVE": "撞击",
    "HEADBUTT_MOVE": "头槌", "THRASH_MOVE": "猛抽", "TAIL_WHIP": "尾扫",
    "BUFF_MOVE": "自增益", "DEFEND_MOVE": "防御", "GROW_MOVE": "成长",
    "ESCAPE_MOVE": "逃跑", "SUMMON_MOVE": "召唤", "SLEEP_MOVE": "沉睡",
    "STUNNED": "眩晕", "WAIT_MOVE": "等待", "DEBUFF_MOVE": "减益",
    "POISON_MOVE": "下毒", "WEAK_MOVE": "削弱",
    "VULNERABLE_MOVE": "施加易伤", "FRAIL_MOVE": "施加虚弱",
    "SKITTER_MOVE": "疾行", "MANDIBLE_MOVE": "颚咬", "ENRAGE_MOVE": "激怒",
    "GLOMP_MOVE": "吞咽", "WHIP_SLAP_MOVE": "鞭甩", "GOOP_MOVE": "粘液",
    "LAY_EGGS_MOVE": "产卵", "TENDERIZER_MOVE": "嫩肉器",
    "HATCH_MOVE": "孵化", "NIBBLE_MOVE": "啃咬",
    "BEES_MOVE": "放蜂", "SPEAR_MOVE": "长矛", "PHEROMONE_SPIT_MOVE": "信息素吐沫",
    "MORE_TEETH": "长牙", "CHOMP": "啃咬", "CHEW": "嚼",
    "TONGUE_LASH_MOVE": "舌鞭", "CRASH_MOVE": "坠毁",
    "UNLOAD_MOVE_1": "卸载(1)", "UNLOAD_MOVE_2": "卸载(2)", "RELOAD_MOVE": "装填",
    "SHIELD_SLAM_MOVE": "盾击", "PONDER_MOVE": "沉思",
    "SLAP_MOVE": "拍打", "CURSE_OF_KNOWLEDGE_MOVE": "知识诅咒",
    "KNOWLEDGE_OVERWHELMING_MOVE": "知识压倒",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_CARD_LINE_RE = re.compile(r"Player \d+ playing card (\S+?)\s*(?:\(targeting (\S+)\.name \(index (\d+)\)\))?\s*(?:\(no target\))?$")
_CHOSE_LINE_RE = re.compile(r"Player \d+ chose cards \[(.*?)\]")
_MONSTER_MOVE_RE = re.compile(r"Monster (\S+) performing move (\S+)")
_COMBAT_RESET_RE = re.compile(r"combat_reset|Combat started")
_ENCOUNTER_HINT_RE = re.compile(r"encounter[_=:]\s*(\S+)", re.IGNORECASE)


@dataclass
class Step:
    kind: str                    # "play" / "chose" / "monster" / "marker"
    content: str                 # raw
    card_id: str = ""
    target: str = ""
    target_idx: int | None = None
    monster_id: str = ""
    move_id: str = ""
    raw_line_no: int = 0

    def format_zh(self) -> str:
        if self.kind == "play":
            card = CARD_ZH.get(self.card_id, self.card_id)
            if self.target:
                tgt = MONSTER_ZH.get(self.target, self.target)
                return f"    ▶ 出牌: {card}  →  {tgt}#{self.target_idx}"
            return f"    ▶ 出牌: {card}  (无目标)"
        if self.kind == "chose":
            picks = [CARD_ZH.get(c.strip(), c.strip()) for c in self.content.split(",") if c.strip()]
            return f"    ☑ 选牌: [{', '.join(picks) or '空'}]"
        if self.kind == "monster":
            m = MONSTER_ZH.get(self.monster_id, self.monster_id)
            mv = MOVE_ZH.get(self.move_id, self.move_id)
            return f"    ⚔ 敌方: {m}  执行  {mv}"
        if self.kind == "marker":
            return f"    ★ {self.content}"
        return f"    · {self.content}"


@dataclass
class Combat:
    start_line: int
    steps: list[Step] = field(default_factory=list)
    encounter_hint: str = ""  # 从 monster 名字推断

    def encounter_guess(self) -> str:
        # 取出现最多且 unique 的 monster 作 encounter 代表
        counts: dict[str, int] = {}
        for s in self.steps:
            if s.monster_id:
                counts[s.monster_id] = counts.get(s.monster_id, 0) + 1
            if s.target:
                counts[s.target] = counts.get(s.target, 0) + 1
        if not counts:
            return "UNKNOWN"
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def summary(self) -> str:
        n_play = sum(1 for s in self.steps if s.kind == "play")
        n_move = sum(1 for s in self.steps if s.kind == "monster")
        return (f"start_line={self.start_line} "
                f"主要敌人={self.encounter_guess()} "
                f"步数={n_play} 出牌 / {n_move} 敌方行动")


def split_into_combats(lines: list[str]) -> list[Combat]:
    """按 'CardSelectCmd.Reset' WARN 作 combat 分界(每场 combat 结束时会触发)。

    实际 sim log 没有显式 "combat_start" 标记;近似做法:
      - 连续的 Player/Monster 动作视为一场
      - 中间遇 "CardSelectCmd.Reset" + monster 身份突变 = 新 combat
    """
    combats: list[Combat] = []
    current: Combat | None = None
    last_monsters: set[str] = set()

    for lno, line in enumerate(lines, 1):
        line_s = line.strip()

        # 纯状态行,skip
        if "[INFO]" not in line_s and "[WARN]" not in line_s:
            continue

        play_m = _CARD_LINE_RE.search(line_s)
        chose_m = _CHOSE_LINE_RE.search(line_s)
        monster_m = _MONSTER_MOVE_RE.search(line_s)

        if play_m:
            step = Step(kind="play", content=line_s,
                        card_id=play_m.group(1),
                        target=play_m.group(2) or "",
                        target_idx=int(play_m.group(3)) if play_m.group(3) else None,
                        raw_line_no=lno)
            if current is None:
                current = Combat(start_line=lno)
                combats.append(current)
            current.steps.append(step)
            if step.target:
                last_monsters.add(step.target)
        elif chose_m:
            step = Step(kind="chose", content=chose_m.group(1), raw_line_no=lno)
            if current is not None:
                current.steps.append(step)
        elif monster_m:
            m_id = monster_m.group(1)
            mv_id = monster_m.group(2)
            # 怪物突变启发式:遇新 monster + current combat 超 10 步
            if current is not None and current.steps and len(current.steps) > 10 and m_id not in last_monsters:
                # 新 combat
                current = Combat(start_line=lno)
                combats.append(current)
                last_monsters = {m_id}
            else:
                last_monsters.add(m_id)
                if current is None:
                    current = Combat(start_line=lno)
                    combats.append(current)
            current.steps.append(Step(kind="monster", content=line_s,
                                       monster_id=m_id, move_id=mv_id, raw_line_no=lno))
        elif "CardSelectCmd.Reset" in line_s:
            if current is not None:
                current.steps.append(Step(kind="marker", content="[进入选牌阶段]", raw_line_no=lno))

    return combats


def format_combat(c: Combat, idx: int) -> str:
    out = [f"\n{'='*70}",
           f"战斗 #{idx}  (log 行 {c.start_line})",
           f"  摘要: {c.summary()}",
           "-"*70]
    turn = 1
    for s in c.steps:
        out.append(s.format_zh())
        # 粗略 turn 分隔:遇 end_turn 或大段 monster 动作视为回合结束
    out.append("="*70)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="sim C# 日志 → 人类可读的中文战斗轨迹")
    ap.add_argument("log", type=Path, help="sim_port*.log 路径")
    ap.add_argument("--list", action="store_true", help="只列出战斗清单,不展开细节")
    ap.add_argument("--encounter", type=str, default=None,
                    help="只输出主要敌人匹配该 ID 的战斗(子串匹配,大小写不敏感)")
    ap.add_argument("--combat-index", type=int, default=None,
                    help="只输出第 N 场(按 log 顺序,1-based)")
    ap.add_argument("--bosses-only", action="store_true",
                    help="只输出 encounter_guess 含 BOSS 字样的战斗(粗略)")
    ap.add_argument("--output", type=Path, default=None,
                    help="输出路径(默认 stdout)")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1

    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    combats = split_into_combats(lines)

    filtered: list[tuple[int, Combat]] = []
    for i, c in enumerate(combats, 1):
        enc = c.encounter_guess()
        if args.encounter and args.encounter.upper() not in enc.upper():
            continue
        if args.bosses_only and "BOSS" not in enc.upper() and enc.upper() not in (
            # Boss-like monsters(无 _BOSS 后缀但本身是 boss)
            "CEREMONIAL_BEAST", "KAISER_CRAB", "TEST_SUBJECT", "KNOWLEDGE_DEMON",
            "THE_INSATIABLE", "LAGAVULIN_MATRIARCH", "VANTOM", "DOORMAKER",
            "QUEEN", "WATERFALL_GIANT", "THE_KIN", "SOUL_FYSH",
        ):
            continue
        if args.combat_index is not None and i != args.combat_index:
            continue
        filtered.append((i, c))

    if args.list:
        out_lines = [f"{'#':<4}  {'line':<6}  {'主要敌人':<30}  {'步数':<8}"]
        out_lines.append("-" * 65)
        for i, c in filtered:
            n_play = sum(1 for s in c.steps if s.kind == "play")
            n_move = sum(1 for s in c.steps if s.kind == "monster")
            out_lines.append(f"{i:<4}  {c.start_line:<6}  {c.encounter_guess():<30}  "
                             f"{n_play} 出牌 / {n_move} 敌动")
        out = "\n".join(out_lines)
    else:
        out = "\n".join(format_combat(c, i) for i, c in filtered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out + "\n", encoding="utf-8")
        print(f"wrote {args.output}  ({len(filtered)} 场战斗)")
    else:
        print(out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
