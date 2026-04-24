"""共享 session 异常类型。

旧 singleplayer HTTP DTO client 已下线；spectator HTTP 现在走 protobuf JSON RPC。
"""

from __future__ import annotations


class SingleplayerAutomationError(RuntimeError):
    pass


class SingleplayerApiError(SingleplayerAutomationError):
    pass


class SingleplayerTimeoutError(SingleplayerAutomationError):
    pass


__all__ = [
    "SingleplayerApiError",
    "SingleplayerAutomationError",
    "SingleplayerTimeoutError",
]
