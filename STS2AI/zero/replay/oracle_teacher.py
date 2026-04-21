from __future__ import annotations

"""兼容旧模块名的薄封装。

主实现已经迁移到 `search_teacher.py`，这里仅保留导出，避免旧脚本立即断掉。
新代码请优先从 `zero.replay.search_teacher` 导入。
"""

from .search_teacher import *  # noqa: F401,F403
