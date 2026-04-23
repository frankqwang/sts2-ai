from __future__ import annotations

"""Artifacts 目录命名约定。

规范：
- `STS2AI/Artifacts` 下的直接子目录统一使用 `MMDD-HHMM-name` 形式
- 这样按名称排序时，就能直接定位最近一次输出
"""

from datetime import datetime
import re


def dated_artifact_dir_name(name: str, *, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now()).strftime("%m%d-%H%M")
    slug = _slugify(name)
    return f"{timestamp}-{slug}" if slug else timestamp


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip("-._")
    return normalized.lower()
