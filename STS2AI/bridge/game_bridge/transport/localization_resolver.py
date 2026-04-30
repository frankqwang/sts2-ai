"""Bridge 端 localization tooltip 解析器。

背景
====
sim 当前在 ``RelicInfo.description`` / ``PotionInfo.description`` / ``Power.description``
里发的是 LocString 占位符（``"MegaCrit.Sts2.Core.Localization.LocString"``），未在 sim
端 resolve 出真实的 i18n 文案，因此 ``proto_state_converter`` 看到的 ``description`` 字段要么是空、
要么是占位符字符串。

为了让训练 / inference 端拿到真实游戏 tooltip（而不是裸 ID 或假描述），bridge 在转 dict 时
回退到本仓库附带的官方 localization 文件 ``localization/eng/*.json``：

- ``relics.json`` -> ``<RELIC_ID>.description``
- ``powers.json`` -> ``<POWER_ID>.smartDescription`` 优先，``<POWER_ID>.description`` 兜底
- ``potions.json`` -> ``<POTION_ID>.description``
- ``intents.json`` -> ``<INTENT_TYPE>.description``

文本里的颜色 markup（``[gold]X[/gold]``）/字体标签 / 图标占位符 (``[img]…[/img]``) 都会
去掉，但保留游戏数值占位符（``{Damage}`` / ``{Amount}``）让上层 LLM 渲染可以理解或进一步替换。

设计目标
========
- bridge 不依赖 llm/ 模块（layering）。
- 数据源只有 localization JSON，无脑补 / 无手写 fallback。
- 缺数据时返回空字符串，让调用方决定是否兜底；不会把"未识别"塞回去。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# 仓库根：STS2AI/bridge/game_bridge/transport/localization_resolver.py
#         ^^^^^^ parents[0]
#                ^^^^^^^^^^^ parents[1]
#                            ^^^^^^^^^^^^^^^ parents[2]
#                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ parents[3]
# parents[4] 即仓库根。
_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOCALIZATION_ENG = _REPO_ROOT / "localization" / "eng"

_COLOR_TAG_RE = re.compile(
    r"\[/?(?:gold|yellow|red|blue|green|cyan|magenta|white|orange|purple|black|gray|brown)\]",
    re.IGNORECASE,
)
_FONT_TAG_RE = re.compile(r"\[/?font_size=\d+\]", re.IGNORECASE)
_IMG_TAG_RE = re.compile(r"\[img\].*?\[/img\]", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")

# sim 当前在 description / name 字段里塞的 LocString 占位符长这样：
#   "MegaCrit.Sts2.Core.Localization.LocString"
# 我们需要把这种"假 description"识别出来，回退到 localization 文件。
_LOCSTRING_SENTINEL = "MegaCrit.Sts2.Core.Localization.LocString"


def _is_real_description(value: str | None) -> bool:
    """判断 sim 给的 description 是不是真实文案。"""
    if not value:
        return False
    text = value.strip()
    if not text:
        return False
    if _LOCSTRING_SENTINEL in text:
        return False
    return True


def _clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _IMG_TAG_RE.sub("", text)
    cleaned = _COLOR_TAG_RE.sub("", cleaned)
    cleaned = _FONT_TAG_RE.sub("", cleaned)
    cleaned = cleaned.replace("\n", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


@lru_cache(maxsize=None)
def _load_localization(file_name: str) -> dict[str, str]:
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


def _lookup(file_name: str, entity_id: str, *suffixes: str) -> str:
    if not entity_id:
        return ""
    table = _load_localization(file_name)
    key = entity_id.upper().strip()
    for suffix in suffixes:
        value = table.get(f"{key}.{suffix}")
        if value:
            return _clean_text(value)
    return ""


def resolve_relic_description(relic_id: str, sim_description: str | None = None) -> str:
    """sim 给真文案就用 sim 的；否则查 localization。"""
    if _is_real_description(sim_description):
        return _clean_text(sim_description or "")
    return _lookup("relics.json", relic_id, "description")


def resolve_potion_description(potion_id: str, sim_description: str | None = None) -> str:
    if _is_real_description(sim_description):
        return _clean_text(sim_description or "")
    return _lookup("potions.json", potion_id, "description")


def resolve_power_description(power_id: str, sim_description: str | None = None) -> str:
    """power 优先 smartDescription（含 {Amount} 占位符更精准）。"""
    if _is_real_description(sim_description):
        return _clean_text(sim_description or "")
    return _lookup("powers.json", power_id, "smartDescription", "description")


def resolve_intent_description(intent_type: str, sim_description: str | None = None) -> str:
    if _is_real_description(sim_description):
        return _clean_text(sim_description or "")
    return _lookup("intents.json", intent_type, "description")


def resolve_keyword_description(keyword: str) -> str:
    """关键词 tooltip 多源：card_keywords -> static_hover_tips -> powers。"""
    if not keyword:
        return ""
    direct = _lookup("card_keywords.json", keyword, "description")
    if direct:
        return direct
    hover = _lookup("static_hover_tips.json", keyword, "description")
    if hover:
        return hover
    upper = keyword.upper().strip()
    power_id = upper if upper.endswith("_POWER") else f"{upper}_POWER"
    return _lookup("powers.json", power_id, "smartDescription", "description")


def resolve_relic_title(relic_id: str, sim_name: str | None = None) -> str:
    if _is_real_description(sim_name):
        return _clean_text(sim_name or "")
    return _lookup("relics.json", relic_id, "title")


def resolve_potion_title(potion_id: str, sim_name: str | None = None) -> str:
    if _is_real_description(sim_name):
        return _clean_text(sim_name or "")
    return _lookup("potions.json", potion_id, "title")


def resolve_power_title(power_id: str, sim_name: str | None = None) -> str:
    if _is_real_description(sim_name):
        return _clean_text(sim_name or "")
    return _lookup("powers.json", power_id, "title")


__all__ = [
    "resolve_relic_description",
    "resolve_relic_title",
    "resolve_potion_description",
    "resolve_potion_title",
    "resolve_power_description",
    "resolve_power_title",
    "resolve_intent_description",
]
