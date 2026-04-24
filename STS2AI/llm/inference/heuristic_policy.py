"""把启发式老师包成 `game_bridge.spectate.ExternalPolicy` 兼容 handler。

观战 ps1 脚本里通过 `--external-policy llm.inference.heuristic_policy:select_action`
把这里的 `select_action` 注入到 SpectatorController。

和 `game_bridge.spectate.zero_external_policy:select_action` 完全同构，
只不过底层决策是规则而不是神经网络。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# spectate cli 的 cwd 是 STS2AI/bridge，需要把 STS2AI 加进 sys.path
_STS2AI_ROOT = Path(__file__).resolve().parents[2]
_path = str(_STS2AI_ROOT)
if _path not in sys.path:
    sys.path.insert(0, _path)

from llm.data_pipeline.heuristic_teacher import pick_action


def select_action(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    _context: Any = None,
) -> dict[str, Any] | None:
    """SpectatorController 每步调这个函数。"""
    enabled = [
        action for action in (legal_actions or [])
        if isinstance(action, dict) and action.get("is_enabled") is not False
    ]
    if not enabled:
        return None

    decision = pick_action(state, enabled)
    idx = max(0, min(decision.action_index, len(enabled) - 1))
    chosen = dict(enabled[idx])
    # 把决策信息 attach 到 action（overlay 可读，不影响 sim）
    chosen.setdefault("_teacher_reason", decision.reason)
    chosen.setdefault("_teacher_score", round(decision.score, 3))
    return chosen


__all__ = ["select_action"]
