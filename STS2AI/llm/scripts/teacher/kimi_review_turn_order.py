"""Deprecated alias for ``teacher_review_turn_order``.

The teacher review pipeline is no longer Kimi-specific (default provider
is now DeepSeek-V4-Pro; Kimi / claude_cli still supported as alternates).
The implementation moved to ``teacher_review_turn_order`` to match its
real scope. This module re-exports everything verbatim so legacy import
sites keep working; remove once all callers have migrated.
"""

from __future__ import annotations

import warnings as _warnings

from llm.scripts.teacher.teacher_review_turn_order import *  # noqa: F401,F403
from llm.scripts.teacher.teacher_review_turn_order import (  # noqa: F401
    DEFAULT_BASE_URL,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_DEEPSEEK_API_KEY_ENV,
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MODEL,
    DEFAULT_TEACHER_PROVIDER,
    append_usage_record,
    build_episode_payload,
    build_messages,
    call_claude_cli,
    call_kimi,  # legacy alias preserved for callers that import the old name
    call_openai_chat,
    compact_episode_for_prompt,
    count_recorded_api_calls,
    main as _main,
    normalize_provider,
    parse_review_json,
    resolve_provider_api_key_env,
    resolve_provider_base_url,
    resolve_provider_model,
    response_content,
    select_episode_rows,
)

_warnings.warn(
    "llm.scripts.teacher.kimi_review_turn_order is deprecated; "
    "import from llm.scripts.teacher.teacher_review_turn_order instead.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":  # pragma: no cover - keep CLI entry working
    raise SystemExit(_main())
