"""把 LLM 的生成文本解析回合法动作。

约定见 `STS2AI/llm/prompts/system_prompt.md`：模型必须输出一行
`{"action_index": <int>, "reason": "..."}`。

解析时只信任 action_index，并要求它落在当前 legal_actions 范围内。
模型偶尔会输出 `{action_index: 3, reason: "..."}` 这种 JSON-like
文本；这里容忍这种格式，但仍不接受编造动作字段。

实验模式还支持让模型自己组合动作：
`{"action":"play_card","hand_index":1,"target_id":2,"reason":"..."}`
或 `{"action":"end_turn","reason":"..."}`。解析后仍会映射回当前
legal_actions 的真实下标；映射失败只返回 invalid，不会偷偷执行兜底动作。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_JSON_RE = re.compile(r"\{[^{}]*\"action_index\"[^{}]*\}")
_ANY_JSON_RE = re.compile(r"\{[^{}]*\}")
_LAX_ACTION_RE = re.compile(
    r"\{\s*action_index\s*:\s*(?P<index>-?\d+)"
    r"(?:\s*,\s*reason\s*:\s*(?P<reason>\"[^\"]*\"|'[^']*'|[^}\n]*))?",
    re.IGNORECASE,
)
_ENEMY_TARGET_RE = re.compile(r"^enemy\s*#?\s*(?P<id>-?\d+)$", re.IGNORECASE)
# Qwen3 thinking block: strip everything before the last </think> or within <think>...</think>
_THINK_END_RE = re.compile(r"</think>\s*", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _clean_lax_reason(raw: str | None) -> str:
    if raw is None:
        return ""
    reason = raw.strip().rstrip(",").strip()
    if len(reason) >= 2 and reason[0] == reason[-1] and reason[0] in {'"', "'"}:
        return reason[1:-1]
    return reason


@dataclass(slots=True, frozen=True)
class DecodedAction:
    action_index: int
    reason: str
    used_fallback: bool
    fallback_reason: str = ""
    confidence: float | None = None
    action_scores: tuple[dict[str, Any], ...] = ()


def _strip_thinking(text: str) -> str:
    """Remove thinking blocks and keep only the text after the last </think>."""
    if not text:
        return text
    # If there's a closing think tag, take everything after the last one
    if _THINK_END_RE.search(text):
        parts = _THINK_END_RE.split(text)
        return parts[-1].strip()
    # Otherwise remove any <think>...</think> inline
    text = _THINK_BLOCK_RE.sub("", text)
    return text.strip()


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    # First strip thinking content
    stripped = _strip_thinking(text).strip()
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


def _extract_lax_blob(text: str) -> dict[str, Any] | None:
    match = _LAX_ACTION_RE.search(_strip_thinking(text).strip())
    if not match:
        return None
    return {
        "action_index": match.group("index"),
        "reason": _clean_lax_reason(match.group("reason")),
    }


def _coerce_confidence(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def _normalize_action_scores(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    out: list[dict[str, Any]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        ok_index, action_index = _coerce_int(item.get("action_index", item.get("index")))
        if not ok_index or action_index is None:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        entry: dict[str, Any] = {
            "action_index": int(action_index),
            "score": round(score, 4),
        }
        note = str(item.get("note") or item.get("reason") or "").strip()
        if note:
            entry["note"] = note[:80]
        out.append(entry)
    return tuple(out)


def action_score_margin(action_scores: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> float | None:
    """Return top1-top2 score gap when at least two scored actions are present."""

    scores: list[float] = []
    for item in action_scores or []:
        if not isinstance(item, dict):
            continue
        try:
            scores.append(float(item.get("score")))
        except (TypeError, ValueError):
            continue
    if len(scores) < 2:
        return None
    scores.sort(reverse=True)
    return round(scores[0] - scores[1], 4)


def _extract_any_json_blob(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = _strip_thinking(text).strip()
    try:
        blob = json.loads(stripped)
        if isinstance(blob, dict):
            return blob
    except json.JSONDecodeError:
        pass
    for match in _ANY_JSON_RE.finditer(stripped):
        try:
            blob = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(blob, dict):
            return blob
    return None


def _normalize_action_name(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"play", "playcard", "play_card", "card"}:
        return "play_card"
    if raw in {"end", "endturn", "end_turn", "pass", "done"}:
        return "end_turn"
    return raw


def _coerce_int(value: Any) -> tuple[bool, int | None]:
    if value is None or value == "":
        return True, None
    if isinstance(value, bool):
        return False, None
    if isinstance(value, int):
        return True, value
    try:
        return True, int(str(value).strip())
    except (TypeError, ValueError):
        return False, None


def _target_to_int(value: Any) -> tuple[bool, int | None]:
    if value is None:
        return True, None
    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in {"", "self", "player", "none", "no_target", "no-target"}:
            return True, None
        match = _ENEMY_TARGET_RE.match(raw)
        if match:
            return True, int(match.group("id"))
    ok, parsed = _coerce_int(value)
    if not ok:
        return False, None
    if parsed in (None, -1, 0):
        return True, None
    return True, parsed


def _legal_action_name(action: dict[str, Any]) -> str:
    return _normalize_action_name(
        action.get("action")
        or action.get("action_type")
        or action.get("type")
    )


def _legal_card_index(action: dict[str, Any]) -> int | None:
    ok, parsed = _coerce_int(action.get("card_index", action.get("hand_index")))
    return parsed if ok else None


def _legal_target_id(action: dict[str, Any]) -> int | None:
    ok, parsed = _target_to_int(action.get("target_id", action.get("target")))
    return parsed if ok else None


def _fallback(
    index: int,
    reason: str,
    fallback_reason: str,
    *,
    confidence: float | None = None,
    action_scores: tuple[dict[str, Any], ...] = (),
) -> DecodedAction:
    return DecodedAction(
        action_index=index,
        reason=reason,
        used_fallback=True,
        fallback_reason=fallback_reason,
        confidence=confidence,
        action_scores=action_scores,
    )


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

    blob = _extract_json_blob(raw_text) or _extract_lax_blob(raw_text)
    if blob is None:
        return DecodedAction(
            action_index=fallback_index,
            reason="",
            used_fallback=True,
            fallback_reason="parse_failed",
        )

    confidence = _coerce_confidence(blob.get("confidence"))
    action_scores = _normalize_action_scores(blob.get("action_scores", blob.get("scores")))
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
                confidence=confidence,
                action_scores=action_scores,
            )

    if raw_index < 0 or raw_index >= len(legal_actions):
        return DecodedAction(
            action_index=fallback_index,
            reason=str(blob.get("reason") or ""),
            used_fallback=True,
            fallback_reason="action_index_out_of_range",
            confidence=confidence,
            action_scores=action_scores,
        )

    return DecodedAction(
        action_index=int(raw_index),
        reason=str(blob.get("reason") or ""),
        used_fallback=False,
        confidence=confidence,
        action_scores=action_scores,
    )


def decode_structured_action(
    raw_text: str,
    legal_actions: list[dict[str, Any]],
    *,
    fallback_index: int = 0,
) -> DecodedAction:
    """解析结构化动作 JSON，并映射回当前 legal_actions 的下标。

    这个模式故意不接受纯 `action_index` 输出，避免实验时模型继续走旧捷径。
    调用方看到 `used_fallback=True` 时应当重试或丢弃该样本，不应执行 fallback。
    """
    if not legal_actions:
        return _fallback(-1, "", "no_legal_actions")

    blob = _extract_any_json_blob(raw_text)
    if blob is None:
        return _fallback(fallback_index, "", "parse_failed")

    confidence = _coerce_confidence(blob.get("confidence"))
    action_scores = _normalize_action_scores(blob.get("action_scores", blob.get("scores")))
    reason = str(blob.get("reason") or "")
    action_name = _normalize_action_name(
        blob.get("action")
        or blob.get("command")
        or blob.get("type")
    )
    if not action_name:
        return _fallback(fallback_index, reason, "action_missing", confidence=confidence, action_scores=action_scores)

    if action_name == "end_turn":
        matches = [
            index for index, action in enumerate(legal_actions)
            if isinstance(action, dict) and _legal_action_name(action) == "end_turn"
        ]
        if len(matches) == 1:
            return DecodedAction(matches[0], reason, used_fallback=False, confidence=confidence, action_scores=action_scores)
        return _fallback(
            fallback_index,
            reason,
            "no_matching_legal_action" if not matches else "ambiguous_action",
            confidence=confidence,
            action_scores=action_scores,
        )

    if action_name != "play_card":
        return _fallback(
            fallback_index,
            reason,
            "unsupported_structured_action",
            confidence=confidence,
            action_scores=action_scores,
        )

    ok, hand_index = _coerce_int(
        blob.get("hand_index", blob.get("card_index", blob.get("hand_idx")))
    )
    if not ok:
        return _fallback(fallback_index, reason, "hand_index_not_int", confidence=confidence, action_scores=action_scores)
    if hand_index is None or hand_index < 0:
        return _fallback(fallback_index, reason, "hand_index_missing", confidence=confidence, action_scores=action_scores)

    target_key = "target_id" if "target_id" in blob else "target"
    ok, requested_target = _target_to_int(blob.get(target_key))
    if not ok:
        return _fallback(fallback_index, reason, "target_id_not_int", confidence=confidence, action_scores=action_scores)

    candidates: list[tuple[int, int | None]] = []
    for index, action in enumerate(legal_actions):
        if not isinstance(action, dict):
            continue
        if _legal_action_name(action) != "play_card":
            continue
        if _legal_card_index(action) != hand_index:
            continue
        candidates.append((index, _legal_target_id(action)))

    if requested_target is not None:
        candidates = [
            (index, target) for index, target in candidates
            if target == requested_target
        ]
    else:
        no_target = [(index, target) for index, target in candidates if target is None]
        if no_target:
            candidates = no_target

    if len(candidates) == 1:
        return DecodedAction(candidates[0][0], reason, used_fallback=False, confidence=confidence, action_scores=action_scores)
    return _fallback(
        fallback_index,
        reason,
        "no_matching_legal_action" if not candidates else "ambiguous_action",
        confidence=confidence,
        action_scores=action_scores,
    )


def format_structured_action_json(action: dict[str, Any], reason: str) -> str:
    """把一个真实 legal action 格式化成结构化训练标签。"""
    action_name = _legal_action_name(action)
    if action_name == "end_turn":
        payload: dict[str, Any] = {
            "action": "end_turn",
            "reason": reason[:80],
        }
    elif action_name == "play_card":
        card_index = _legal_card_index(action)
        payload = {
            "action": "play_card",
            "hand_index": int(card_index) if card_index is not None else -1,
        }
        target_id = _legal_target_id(action)
        if target_id is not None:
            payload["target_id"] = int(target_id)
        payload["reason"] = reason[:80]
    else:
        payload = {
            "action": action_name or "unknown",
            "reason": reason[:80],
        }
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "DecodedAction",
    "action_score_margin",
    "decode_action",
    "decode_structured_action",
    "format_structured_action_json",
]
