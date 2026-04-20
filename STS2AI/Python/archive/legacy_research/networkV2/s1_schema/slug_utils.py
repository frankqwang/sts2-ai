"""名称 slugify 工具：CamelCase → UPPER_SNAKE_CASE。

复制 C# StringHelper.Slugify 的规则，把卡牌/遗物/怪物的 class name 标准化成 id，
供 card_tags / card_semantic_catalog / build_card_semantic_index 使用。
"""

from __future__ import annotations

import re


def _slugify(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"\s+", "_", s.upper())
    s = re.sub(r"[^A-Z0-9_]", "", s)
    return s
