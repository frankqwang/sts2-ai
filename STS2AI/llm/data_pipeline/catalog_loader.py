"""统一的 catalog 加载器：卡牌 / 遗物 / powers 的中文描述。

卡牌：从 `STS2AI/data/game_wiki/game_catalog.sqlite` 的 `cards` 表读（有 description_zh）。
遗物：DB 里没描述，手写常见 50-80 条（覆盖 Act1-2 高频）。
Powers：同上，手写 30-50 条。

设计原则：
- 模糊大小写匹配（sim 给的 id 是 STRIKE_IRONCLAD，DB 存 strike_ironclad）
- 描述里 `[gold]...[/gold]` / `[yellow]...[/yellow]` 等 markup **去掉**
- 模板占位符 `{XxxPower:diff()}` / `{Energy:energyIcons()}` 在渲染时替换成数值；
  主要数值从 catalog 记录的 C# source_path 解析 `CanonicalVars` / `UpgradeValueBy`
"""
from __future__ import annotations

import re
import sqlite3
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_STS2AI_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _STS2AI_ROOT.parent
_CATALOG_DB = _STS2AI_ROOT / "data" / "game_wiki" / "game_catalog.sqlite"
_ZHS_CARDS_JSON = _REPO_ROOT / "localization" / "zhs" / "cards.json"


_MARKUP_TAG_RE = re.compile(r"\[/?(gold|yellow|red|blue|green|cyan|magenta|white|orange|purple)\]")
_IMG_TAG_RE = re.compile(r"\[img\].*?\[/img\]")
_WHITESPACE_RE = re.compile(r"\s+")
_PLACEHOLDER_RE = re.compile(r"\{([^{}]*)\}")
_DYNAMIC_VAR_CTOR_RE = re.compile(
    r"new\s+(?P<class>[A-Za-z_]\w*Var|DynamicVar)"
    r"(?:<(?P<generic>[A-Za-z_]\w*)>)?"
    r"\((?P<args>[^()]*)\)",
    re.S,
)
_UPGRADE_DOT_RE = re.compile(
    r"DynamicVars\.(?P<name>[A-Za-z_]\w*)\.UpgradeValueBy\((?P<amount>[-+]?\d+(?:\.\d+)?m?)\)"
)
_UPGRADE_INDEX_RE = re.compile(
    r"DynamicVars\[\s*\"(?P<name>[^\"]+)\"\s*\]\.UpgradeValueBy\((?P<amount>[-+]?\d+(?:\.\d+)?m?)\)"
)

_DEFAULT_VAR_NAMES: dict[str, str] = {
    "BlockVar": "Block",
    "CalculatedBlockVar": "CalculatedBlock",
    "CalculatedDamageVar": "CalculatedDamage",
    "CalculationBaseVar": "CalculationBase",
    "CalculationExtraVar": "CalculationExtra",
    "CardsVar": "Cards",
    "DamageVar": "Damage",
    "EnergyVar": "Energy",
    "ExtraDamageVar": "ExtraDamage",
    "ForgeVar": "Forge",
    "GoldVar": "Gold",
    "HealVar": "Heal",
    "HpLossVar": "HpLoss",
    "IfUpgradedVar": "IfUpgraded",
    "MaxHpVar": "MaxHp",
    "OstyDamageVar": "OstyDamage",
    "RepeatVar": "Repeat",
    "StarsVar": "Stars",
    "SummonVar": "Summon",
}

_DYNAMIC_VAR_ALIASES: dict[str, str] = {
    "Block": "Block",
    "CalculatedBlock": "CalculatedBlock",
    "CalculatedDamage": "CalculatedDamage",
    "CalculationBase": "CalculationBase",
    "CalculationExtra": "CalculationExtra",
    "Cards": "Cards",
    "Damage": "Damage",
    "Dexterity": "DexterityPower",
    "Doom": "DoomPower",
    "Energy": "Energy",
    "ExtraDamage": "ExtraDamage",
    "Forge": "Forge",
    "Gold": "Gold",
    "Heal": "Heal",
    "HpLoss": "HpLoss",
    "MaxHp": "MaxHp",
    "OstyDamage": "OstyDamage",
    "Poison": "PoisonPower",
    "Repeat": "Repeat",
    "Stars": "Stars",
    "Strength": "StrengthPower",
    "Summon": "Summon",
    "Vulnerable": "VulnerablePower",
    "Weak": "WeakPower",
}


def _clean_text(text: str) -> str:
    if not text:
        return ""
    def _replace_img(match: re.Match[str]) -> str:
        tag = match.group(0).lower()
        if "star_icon" in tag:
            return "星"
        if "energy_icon" in tag:
            return "能量"
        return ""

    text = _IMG_TAG_RE.sub(_replace_img, text)
    text = _MARKUP_TAG_RE.sub("", text)
    text = text.replace("\\n", "\n").replace("\n", "；")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=1)
def _load_repo_card_texts() -> dict[str, dict[str, str]]:
    if not _ZHS_CARDS_JSON.exists():
        return {}
    try:
        rows = json.loads(_ZHS_CARDS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[str, str]] = {}
    for raw_key, raw_value in rows.items():
        key = str(raw_key or "")
        value = str(raw_value or "")
        if key.endswith(".description"):
            card_id = key.removesuffix(".description").upper()
            out.setdefault(card_id, {})["description"] = value
        elif key.endswith(".title"):
            card_id = key.removesuffix(".title").upper()
            out.setdefault(card_id, {})["name"] = value
    return out


def _split_args(text: str) -> list[str]:
    args: list[str] = []
    start = 0
    in_string = False
    escape = False
    depth = 0
    for index, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        args.append(tail)
    return args


def _parse_string_literal(text: str) -> str | None:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return stripped[1:-1]
    return None


def _parse_number(text: str) -> float | int | None:
    stripped = text.strip().rstrip("mMfFdD")
    if stripped.lower() == "true":
        return 1
    if stripped.lower() == "false":
        return 0
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped):
        return None
    value = float(stripped)
    return int(value) if value.is_integer() else value


def _format_value(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _var_from_constructor(var_class: str, generic: str | None, args_text: str) -> tuple[str, float | int] | None:
    args = _split_args(args_text)
    if not args:
        return None

    explicit_name = _parse_string_literal(args[0])
    value_arg_index = 1 if explicit_name is not None else 0
    if var_class == "PowerVar":
        name = explicit_name or str(generic or "")
    elif var_class in ("DynamicVar", "IntVar", "BoolVar", "StringVar"):
        if explicit_name is None:
            return None
        name = explicit_name
    else:
        name = explicit_name or _DEFAULT_VAR_NAMES.get(var_class, "")
    if not name:
        return None

    if value_arg_index >= len(args):
        value: float | int = 0
    else:
        parsed = _parse_number(args[value_arg_index])
        value = parsed if parsed is not None else 0
    return name, value


def _source_path_for_card(card_id: str) -> Path | None:
    info = lookup_card(card_id)
    source_path = str(info.get("source_path") or "")
    if not source_path:
        return None
    path = _REPO_ROOT / source_path
    return path if path.exists() else None


@lru_cache(maxsize=1024)
def _load_card_source_vars(card_id: str, is_upgraded: bool) -> dict[str, Any]:
    path = _source_path_for_card(card_id)
    if path is None:
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    values: dict[str, float | int] = {}
    for match in _DYNAMIC_VAR_CTOR_RE.finditer(text):
        parsed = _var_from_constructor(match.group("class"), match.group("generic"), match.group("args"))
        if parsed is None:
            continue
        name, value = parsed
        values.setdefault(name, value)

    if is_upgraded:
        for match in _UPGRADE_DOT_RE.finditer(text):
            name = _DYNAMIC_VAR_ALIASES.get(match.group("name"), match.group("name"))
            amount = _parse_number(match.group("amount"))
            if amount is None:
                continue
            values[name] = values.get(name, 0) + amount
        for match in _UPGRADE_INDEX_RE.finditer(text):
            name = match.group("name")
            amount = _parse_number(match.group("amount"))
            if amount is None:
                continue
            values[name] = values.get(name, 0) + amount

    # CalculatedVar 在无目标/无战斗上下文时会退回 CalculationBase。
    if values.get("CalculationBase") is not None:
        values.setdefault("CalculatedDamage", values["CalculationBase"])
        values.setdefault("CalculatedBlock", values["CalculationBase"])
    return dict(values)


def _runtime_template_values(runtime_values: dict[str, Any] | None) -> dict[str, Any]:
    if not runtime_values:
        return {}
    out: dict[str, Any] = {}
    for key, value in runtime_values.items():
        if value in (None, ""):
            continue
        normalized = {
            "damage": "Damage",
            "damage_now": "Damage",
            "effective_damage": "Damage",
            "block": "Block",
            "block_now": "Block",
            "effective_block": "Block",
            "magic": "MagicNumber",
            "magic_now": "MagicNumber",
        }.get(str(key), str(key))
        if normalized in ("preview_damage_per_target", "preview_block"):
            continue
        out[normalized] = value

    preview_damage = runtime_values.get("preview_damage_per_target")
    if isinstance(preview_damage, dict):
        damages = [value for value in preview_damage.values() if value not in (None, "", 0)]
        if damages:
            unique = list(dict.fromkeys(damages))
            out["Damage"] = unique[0] if len(unique) == 1 else "各目标不同，见dmg(actual)"
            out.setdefault("CalculatedDamage", out["Damage"])
    preview_block = runtime_values.get("preview_block")
    if preview_block not in (None, "", 0):
        out["Block"] = preview_block
        out.setdefault("CalculatedBlock", preview_block)
    return out


def _split_format_options(text: str) -> list[str]:
    return [part for part in text.split("|")]


def _format_energy(value: Any) -> str:
    return f"{_format_value(value)}能量"


def _format_stars(value: Any) -> str:
    return f"{_format_value(value)}星"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "none", "no")
    return bool(value)


def _eval_placeholder(expr: str, values: dict[str, Any], *, is_upgraded: bool) -> str:
    expr = expr.strip()
    if not expr:
        return ""
    if ":" not in expr:
        return _format_value(values.get(expr, "?"))

    key, rest = expr.split(":", 1)
    key = key.strip()
    rest = rest.strip()
    value = values.get(key)

    if rest.startswith("diff") or rest.startswith("diffInverse"):
        return _format_value(value if value is not None else "?")
    if rest.startswith("energyIcons"):
        if key == "energyPrefix":
            return "能量"
        if value is None:
            match = re.search(r"energyIcons\(([-+]?\d+)\)", rest)
            value = int(match.group(1)) if match else 1
        return _format_energy(value)
    if rest.startswith("starIcons"):
        return _format_stars(value if value is not None else 1)
    if rest.startswith("show:"):
        options = _split_format_options(rest.removeprefix("show:"))
        true_text = options[0] if options else ""
        false_text = options[1] if len(options) > 1 else ""
        return true_text if is_upgraded else false_text
    if rest.startswith("choose("):
        match = re.match(r"choose\(([^)]*)\):(.*)", rest, re.S)
        if match:
            expected = match.group(1).strip()
            options = _split_format_options(match.group(2))
            actual = _format_value(value if value is not None else "")
            chosen = options[0] if actual == expected and options else (options[1] if len(options) > 1 else actual)
            return chosen.replace("__CURRENT_VALUE__", actual)
    if rest.startswith("cond:"):
        cond = rest.removeprefix("cond:")
        match = re.match(r"([><=!]+)\s*(-?\d+(?:\.\d+)?)\?(.*)", cond, re.S)
        if match:
            op, rhs_raw, body = match.groups()
            options = _split_format_options(body)
            lhs = float(value or 0)
            rhs = float(rhs_raw)
            passed = {
                ">": lhs > rhs,
                ">=": lhs >= rhs,
                "<": lhs < rhs,
                "<=": lhs <= rhs,
                "==": lhs == rhs,
                "!=": lhs != rhs,
            }.get(op, False)
            return (options[0] if passed else (options[1] if len(options) > 1 else "")).replace("{}", _format_value(value))
    if rest.startswith("plural:"):
        options = _split_format_options(rest.removeprefix("plural:"))
        chosen = options[0] if value == 1 and options else (options[1] if len(options) > 1 else "")
        return chosen.replace("{}", _format_value(value))
    if "|" in rest:
        options = _split_format_options(rest)
        return options[0] if _truthy(value) else (options[1] if len(options) > 1 else "")

    return _format_value(value if value is not None else "?")


def _render_template(text: str, values: dict[str, Any], *, is_upgraded: bool) -> str:
    if not text:
        return ""
    text = text.replace("{:diff()}", "__CURRENT_VALUE__").replace("{}", "__CURRENT_VALUE__")
    for _ in range(8):
        new_text = _PLACEHOLDER_RE.sub(lambda m: _eval_placeholder(m.group(1), values, is_upgraded=is_upgraded), text)
        if new_text == text:
            break
        text = new_text
    text = text.replace("__CURRENT_VALUE__", "?")
    return _clean_text(text)


def render_card_description(
    card_id: str,
    description: str | None = None,
    *,
    is_upgraded: bool = False,
    runtime_values: dict[str, Any] | None = None,
) -> str:
    """把卡牌描述模板渲染成给 LLM 看的纯文本。

    `runtime_values` 可传 sim hand card dict；其中 preview_damage / preview_block
    会覆盖静态 source 解析值。
    """
    info = lookup_card(card_id)
    raw_desc = description if description not in (None, "") else str(info.get("description") or "")
    if not raw_desc:
        return ""
    values: dict[str, Any] = dict(_load_card_source_vars(card_id, is_upgraded))
    values.update(
        {
            "IfUpgraded": 1 if is_upgraded else 0,
            "InCombat": True,
            "OnTable": True,
            "IsTargeting": False,
            "TargetType": str(info.get("target_type") or ""),
            "energyPrefix": "能量",
            "singleStarIcon": "星",
        }
    )
    values.update(_runtime_template_values(runtime_values))
    return _render_template(_clean_text(raw_desc), values, is_upgraded=is_upgraded)


@lru_cache(maxsize=1)
def _load_card_catalog() -> dict[str, dict[str, Any]]:
    if not _CATALOG_DB.exists():
        return {}
    conn = sqlite3.connect(str(_CATALOG_DB))
    try:
        c = conn.cursor()
        c.execute(
            "SELECT id, name_zh, description_zh, upgrade_preview_zh, card_type, base_cost, target_type, source_path "
            "FROM cards"
        )
        rows = c.fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, Any]] = {}
    repo_texts = _load_repo_card_texts()
    for cid, name_zh, desc_zh, upg_zh, ctype, cost, target, source_path in rows:
        if not cid:
            continue
        key = str(cid).upper()
        repo_text = repo_texts.get(key, {})
        name = repo_text.get("name") or name_zh or cid
        desc = repo_text.get("description") or desc_zh or ""
        out[key] = {
            "id": key,
            "name": name,
            "description": _clean_text(desc),
            "upgrade_preview": _clean_text(upg_zh or ""),
            "card_type": (ctype or "").lower(),
            "base_cost": cost if cost is not None else "?",
            "target_type": (target or "").lower(),
            "source_path": source_path or "",
        }
    return out


# 手写的遗物描述（Act1-2 Ironclad 高频 + 通用），50+ 条
_RELIC_DESCRIPTIONS: dict[str, str] = {
    # Ironclad 起始
    "BURNING_BLOOD": "赤红熔血：战斗结束时回复 6 HP",
    # 通用常见
    "AKABEKO": "赤红牛：每场战斗首张攻击牌额外造成 8 伤害",
    "ANCHOR": "锚：首回合获得 10 格挡",
    "ANCIENT_TEA_SET": "古老茶具：进入战斗回合开始时获得 2 能量（仅下一场）",
    "ART_OF_WAR": "战争艺术：上回合没打攻击牌，这回合开始+1 能量",
    "BAG_OF_MARBLES": "弹珠袋：战斗开始对所有敌人施加 1 脆弱",
    "BAG_OF_PREPARATION": "筹备之袋：战斗开始多抽 2 张",
    "BLOOD_VIAL": "鲜血之瓶：每场战斗开始回 2 HP",
    "BRONZE_SCALES": "青铜鳞片：荆棘 3（受到攻击反伤）",
    "CENTENNIAL_PUZZLE": "百年谜题：每场战斗首次失血时抽 3 张",
    "CERAMIC_FISH": "陶瓷鱼：获得卡牌时+9 金币",
    "DREAM_CATCHER": "集梦者：休息时可选领一张卡",
    "HAPPY_FLOWER": "快乐花：每 3 回合+1 能量",
    "JUZU_BRACELET": "念珠：地图不再出现普通战斗房",
    "LANTERN": "灯笼：首回合+1 能量",
    "MAW_BANK": "血口银行：踩到层数时+6 金币（买东西后失效）",
    "MEAT_ON_THE_BONE": "带骨之肉：战斗结束 HP ≤50% 时回 12 HP",
    "MEDICAL_KIT": "医疗包：Status 牌可打出（消耗）",
    "MEMBERSHIP_CARD": "会员卡：商店价格 -50%",
    "NLOTHS_GIFT": "恩洛斯的礼物：下个稀有卡奖提升品质（可能带诅咒）",
    "ODD_MUSHROOM": "怪菇：易伤状态下多受 25% 伤害变多受 50%",
    "OMAMORI": "御守：前两次诅咒免疫",
    "ORICHALCUM": "山铜：回合结束无格挡时获得 6 格挡",
    "PEN_NIB": "笔尖：每 10 张攻击牌一张伤害翻倍",
    "PHILOSOPHERS_STONE": "贤者之石：+1 最大能量但所有敌人+1 力量",
    "RED_SKULL": "赤红头骨：HP ≤50% 时+3 力量",
    "REGAL_PILLOW": "华丽枕头：休息回 15 HP（原 30% HP）",
    "RUNIC_CUBE": "符文方块：每次失血抽 1 张",
    "RUNIC_DOME": "符文穹顶：看不到敌人意图，但+1 能量",
    "RUNIC_PYRAMID": "符文金字塔：回合结束不弃手牌",
    "SHOVEL": "铲子：休息时可选挖宝（得 1 遗物）",
    "SNECKO_EYE": "玄蛇之眼：+2 手牌上限，所有卡 cost 随机 0-3",
    "STRAWBERRY": "草莓：+7 最大 HP",
    "TINY_CHEST": "小宝箱：每 4 个未知房升级成稀有事件",
    "TORII": "鸟居：受到 5 点及以下伤害时减到 1",
    "WARPED_TONGS": "扭曲钳子：每回合随机升级一张手牌（仅该回合）",
    "WHITE_BEAST_STATUE": "白兽雕像：战斗后药水掉率 100%",
    "HAND_DRILL": "手钻：对破盾的敌人施加 2 易伤",
    "MINIATURE_CANNON": "迷你加农炮：战斗开始对所有敌人造成 7 伤害",
    "SILVER_CRUCIBLE": "银坩埚：战斗结束 1/2 概率回 4 HP",
    # 通用升级类
    "PRAYER_WHEEL": "祈祷轮：战斗后卡奖多 1 张（可选）",
    "POTION_BELT": "药水腰带：+2 药水槽位",
    "SLING_OF_COURAGE": "勇气投石：精英战开始+2 力量",
    "STRANGE_SPOON": "奇异勺子：消耗牌 50% 概率不消耗",
    "MATRYOSHKA": "套娃：下 2 次开箱多 1 个 common 遗物",
}


# 手写的 powers 描述
_POWER_DESCRIPTIONS: dict[str, str] = {
    # 玩家 + 敌人通用
    "STRENGTH_POWER": "力量：每点+1 攻击伤害",
    "DEXTERITY_POWER": "敏捷：每点+1 格挡",
    "FOCUS_POWER": "专注：每点+1 球伤害/效果",
    "ARTIFACT_POWER": "神器：免疫下次减益，每次-1",
    "PLATED_ARMOR": "镀甲：每回合-1，等量格挡",
    "METALLICIZE_POWER": "金属化：每回合结束+X 格挡",
    "RITUAL_POWER": "仪式：回合结束+X 力量",
    "BARRICADE_POWER": "街垒：回合结束格挡不衰减",
    "DEMON_FORM_POWER": "恶魔形态：回合开始+X 力量",
    "BRUTALITY_POWER": "残暴：回合开始-1 HP +1 抽卡",
    "COMBUST_POWER": "燃烧：回合结束-X HP 对所有敌人 X 伤",
    "DARK_EMBRACE_POWER": "黑暗拥抱：每次消耗牌抽 1 张",
    "FEEL_NO_PAIN_POWER": "麻木：每次消耗牌+X 格挡",
    "EVOLVE_POWER": "进化：抽到 Status 牌时多抽 X 张",
    "JUGGERNAUT_POWER": "碎碎念：获得格挡时随机造成 X 伤害",
    "FIRE_BREATHING_POWER": "喷火：抽到 Status/诅咒时+X 伤害对所有敌人",
    "RUPTURE_POWER": "破裂：每次自残+X 力量",
    "BERSERK_POWER": "狂暴：每回合+1 能量",
    # debuff
    "VULNERABLE_POWER": "易伤：受攻击伤害 +50%（每回合-1）",
    "WEAK_POWER": "虚弱：造成攻击伤害 -25%（每回合-1）",
    "FRAIL_POWER": "脆弱：获得格挡 -25%（每回合-1）",
    "POISON_POWER": "中毒：回合开始受 X HP，每回合-1",
    "ENTANGLED_POWER": "缠绕：本回合不能打攻击牌",
    "NO_DRAW_POWER": "禁抽：本回合不再抽牌",
    # 敌人/特殊
    "SURROUNDED_POWER": "包围：从侧/后被击时受额外 50% 伤害",
    "INTANGIBLE_POWER": "虚无：所有伤害减为 1（每回合-1）",
    "REGEN_POWER": "再生：回合结束+X HP",
    "MODE_SHIFT_POWER": "形态转换：受击累计 X 后切换形态",
    "CURIOSITY_POWER": "好奇：打出 power 牌时+X 力量",
    "SPORE_CLOUD_POWER": "孢子云：被击杀时施加 X 易伤给玩家",
    "CHOKED_POWER": "窒息：下张打出的牌后失 X HP",
    "THIEVERY_POWER": "盗窃：攻击时偷 X 金币",
    "SURPRISE_POWER": "惊吓：（敌人特定技能）",
}


def lookup_card(card_id: str) -> dict[str, Any]:
    catalog = _load_card_catalog()
    key = (card_id or "").upper().strip()
    return catalog.get(key, {})


def lookup_relic(relic_id: str) -> str:
    key = (relic_id or "").upper().strip()
    return _RELIC_DESCRIPTIONS.get(key, "")


def lookup_power(power_id: str) -> str:
    key = (power_id or "").upper().strip()
    return _POWER_DESCRIPTIONS.get(key, "")


def card_short(card_id: str, *, is_upgraded: bool = False) -> str:
    """给渲染用：一行紧凑的卡牌描述。

    例：`STRIKE_IRONCLAD cost=1 [attack] 造成 6 伤害`
    """
    info = lookup_card(card_id)
    if not info:
        return ""
    name = info.get("name") or card_id
    ctype = info.get("card_type") or "?"
    cost = info.get("base_cost", "?")
    desc = render_card_description(card_id, is_upgraded=is_upgraded)
    if is_upgraded and info.get("upgrade_preview"):
        upgrade_preview = render_card_description(
            card_id,
            str(info["upgrade_preview"]),
            is_upgraded=True,
        )
        if upgrade_preview:
            desc += f"（升级后：{upgrade_preview}）"
    return f"{name} cost={cost} [{ctype}] {desc}"


def relic_short(relic_id: str) -> str:
    desc = lookup_relic(relic_id)
    if desc:
        return desc
    return f"{relic_id}（未识别）"


def power_short(power_id: str, amount: Any = None) -> str:
    desc = lookup_power(power_id)
    if desc and amount is not None:
        return f"{desc} (×{amount})"
    if desc:
        return desc
    return f"{power_id}" + (f"×{amount}" if amount else "")


__all__ = [
    "card_short", "relic_short", "power_short", "render_card_description",
    "lookup_card", "lookup_relic", "lookup_power",
]
