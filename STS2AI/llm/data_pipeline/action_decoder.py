"""把 LLM 的生成文本解析回一个合法 action_index。

约定见 `STS2AI/llm/prompts/system_prompt.md`：模型必须输出一行
`{"action_index": <int>, "reason": "..."}`。

解析时要求严格：不允许模型"编造"动作字段，只允许挑 index。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_JSON_RE = re.compile(r"\{[^{}]*\"action_index\"[^{}]*\}")


@dataclass(slots=True, frozen=True)
class DecodedAction:
    action_index: int
    reason: str
    used_fallback: bool
    fallback_reason: str = ""


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    try:
        blob = json.loads(stripped)
        if isinstance(blob, dict):
            return blob
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(stripped)
    if not match:
        return None
    try:
        blob = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return blob if isinstance(blob, dict) else None


def decode_action(
    raw_text: str,
    legal_actions: list[dict[str, Any]],
    *,
    fallback_index: int = 0,
) -> DecodedAction:
    """从模型原始输出抽出 action_index，并做合法性校验。

    失败时回退到 `fallback_index`（默认 0）。调用方也可以自己指定
    end_turn 的 index 作为 fallback。
    """
    if not legal_actions:
        return DecodedAction(
            action_index=-1,
            reason="",
            used_fallback=True,
            fallback_reason="no_legal_actions",
        )

    blob = _extract_json_blob(raw_text)
    if blob is None:
        return DecodedAction(
            action_index=fallback_index,
            reason="",
            used_fallback=True,
            fallback_reason="parse_failed",
        )

    raw_index = blob.get("action_index")
    if not isinstance(raw_index, int) or isinstance(raw_index, bool):
        try:
            raw_index = int(raw_index)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return DecodedAction(
                action_index=fallback_index,
                reason=str(blob.get("reason") or ""),
                used_fallback=True,
                fallback_reason="action_index_not_int",
            )

    if raw_index < 0 or raw_index >= len(legal_actions):
        return DecodedAction(
            action_index=fallback_index,
            reason=str(blob.get("reason") or ""),
            used_fallback=True,
            fallback_reason="action_index_out_of_range",
        )

    return DecodedAction(
        action_index=int(raw_index),
        reason=str(blob.get("reason") or ""),
        used_fallback=False,
    )


__all__ = ["DecodedAction", "decode_action"]
