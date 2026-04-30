"""STS2 官方游戏 tooltip / description 加载器。

数据源：仓库根目录下的 ``localization/eng/*.json``（游戏官方英文 localization 文件，
随 STS2 仓库一起发布，包含全部 relic / power / card / intent / keyword / potion 的描述）。

之前 ``state_renderer.py`` 用手写的 ``_RELIC_GLOSSARY`` / ``_POWER_GLOSSARY``
（4-7 条核心条目）作为 prompt 里的 tooltip 来源，导致：

- 大量 relic（NEW_LEAF / WAR_PAINT / WINGED_BOOTS …）在 prompt 里裸 ID
  显示，AI 完全不知道 relic 效果，下游 GRPO 训练时模型只能瞎猜。
- power 类似（CONSTRICT_POWER / SKITTISH_POWER 等没描述）。
- 任何修改 / 新增 relic 都需要手动维护字典。

正确的做法是直接读取游戏官方 localization 文件，确保：
  1. 覆盖率：游戏支持的所有 relic / power 自带 tooltip。
  2. 准确性：完全是游戏内显示的文案，没有人为脑补。
  3. 维护成本：游戏更新时同步 ``localization/`` 目录即可，无需改代码。

本模块只暴露 lookup 接口，调用方（``state_renderer``）负责拼装到 prompt。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

# 仓库根目录： STS2AI/llm/data_pipeline/localization_loader.py
#                ^^^^^^ parents[0]
#                       ^^^^^^^^^^^ parents[1]
#                                 ^^^^^^^^^^^^^^ parents[2]
# parents[3] 即仓库根。
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCALIZATION_ENG = _REPO_ROOT / "localization" / "eng"

# ---------------------------------------------------------------------------
# 文本清理：去掉游戏内 markup / 格式化标签，保留占位符（LLM 能理解）。
# ---------------------------------------------------------------------------

_COLOR_TAG_RE = re.compile(
    r"\[/?(?:gold|yellow|red|blue|green|cyan|magenta|white|orange|purple|black|gray|brown)\]",
    re.IGNORECASE,
)
_FONT_TAG_RE = re.compile(r"\[/?font_size=\d+\]", re.IGNORECASE)
_IMG_TAG_RE = re.compile(r"\[img\].*?\[/img\]", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: Any) -> str:
    """剥离 STS2 文案里的 [gold]X[/gold] / [img]…[/img] / [font_size=18]X[/font_size] 等装饰，
    保留占位符（``{Block}``、``{Damage}`` 等）让下游进一步替换或让 LLM 自己理解。
    """
    if not text:
        return ""
    cleaned = str(text)
    cleaned = _IMG_TAG_RE.sub("", cleaned)
    cleaned = _COLOR_TAG_RE.sub("", cleaned)
    cleaned = _FONT_TAG_RE.sub("", cleaned)
    cleaned = cleaned.replace("\n", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# JSON 加载：每个文件加载一次后缓存为 dict[str, str]（id -> 描述）。
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_localization(file_name: str) -> dict[str, str]:
    """加载 localization/eng/<file_name>，返回 raw key->value（不预处理 markup）。

    key 形如 ``"AKABEKO.description"`` / ``"AKABEKO.title"`` /
    ``"ARTIFACT_POWER.smartDescription"``，调用方按需 lookup。
    """
    path = _LOCALIZATION_ENG / file_name
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, (str, int, float))}


def _lookup_field(file_name: str, entity_id: str, *suffixes: str) -> str:
    """按 entity_id + suffix 查找；suffix 顺序就是优先级。"""
    if not entity_id:
        return ""
    table = _load_localization(file_name)
    key_id = entity_id.upper().strip()
    for suffix in suffixes:
        value = table.get(f"{key_id}.{suffix}")
        if value:
            return _clean_text(value)
    return ""


# ---------------------------------------------------------------------------
# 公开 lookup API
# ---------------------------------------------------------------------------

def lookup_relic(relic_id: str) -> str:
    """返回 relic 的 description 文本（已去除 markup）。"""
    return _lookup_field("relics.json", relic_id, "description")


def lookup_relic_title(relic_id: str) -> str:
    return _lookup_field("relics.json", relic_id, "title")


def lookup_power(power_id: str) -> str:
    """power 优先 smartDescription（带 {Amount} 占位符，更精确），fallback description。"""
    return _lookup_field("powers.json", power_id, "smartDescription", "description")


def lookup_power_title(power_id: str) -> str:
    return _lookup_field("powers.json", power_id, "title")


def lookup_card(card_id: str) -> str:
    return _lookup_field("cards.json", card_id, "description")


def lookup_card_title(card_id: str) -> str:
    return _lookup_field("cards.json", card_id, "title")


def lookup_intent(intent_type: str) -> str:
    """intent_type 例如 "ATTACK" / "BUFF" / "DEFEND" / "DEBUFF_STRONG"。"""
    return _lookup_field("intents.json", intent_type, "description")


def lookup_intent_title(intent_type: str) -> str:
    return _lookup_field("intents.json", intent_type, "title")


def lookup_keyword(keyword: str) -> str:
    """关键词 tooltip。

    游戏内的"keyword"散落在多个 localization 文件：

    - ``card_keywords.json``：ETERNAL/ETHEREAL/EXHAUST/INNATE/RETAIN/SLY/UNPLAYABLE
    - ``static_hover_tips.json``：BLOCK/CHANNELING/COOK 等机制 tooltip
    - ``powers.json``：VULNERABLE_POWER/WEAK_POWER/STRENGTH_POWER 等（也常被当 keyword 提）

    本函数按优先级依次查；找到即返回。
    """
    if not keyword:
        return ""
    direct = _lookup_field("card_keywords.json", keyword, "description")
    if direct:
        return direct
    hover = _lookup_field("static_hover_tips.json", keyword, "description")
    if hover:
        return hover
    # power 形式：VULNERABLE/WEAK 自动尝试 _POWER 后缀
    upper = keyword.upper().strip()
    power_id = upper if upper.endswith("_POWER") else f"{upper}_POWER"
    return _lookup_field("powers.json", power_id, "smartDescription", "description")


def lookup_potion(potion_id: str) -> str:
    return _lookup_field("potions.json", potion_id, "description")


def lookup_potion_title(potion_id: str) -> str:
    return _lookup_field("potions.json", potion_id, "title")


def lookup_monster_name(monster_id: str) -> str:
    return _lookup_field("monsters.json", monster_id, "name")


__all__ = [
    "lookup_relic",
    "lookup_relic_title",
    "lookup_power",
    "lookup_power_title",
    "lookup_card",
    "lookup_card_title",
    "lookup_intent",
    "lookup_intent_title",
    "lookup_keyword",
    "lookup_potion",
    "lookup_potion_title",
    "lookup_monster_name",
]
