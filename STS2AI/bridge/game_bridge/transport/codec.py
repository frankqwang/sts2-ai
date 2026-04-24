"""协议 codec - 把 proto 的 encode/decode 抽成统一接口。

PipeConnection 本身协议无关,通过注入 `ProtocolCodec` 决定 wire format。
combat / full-run / 未来的 RPC 都复用 PipeConnection + 对应 codec,禁止
业务模块自己写字节编码。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProtocolCodec(ABC):
    """把 `method + params` 编成 pipe frame 字节;把响应字节解成 dict。"""

    @abstractmethod
    def encode_request(self, method: str, params: dict[str, Any] | None) -> bytes:
        """返回发给 sim 的 payload bytes (不含 4 字节长度前缀)。"""
        raise NotImplementedError

    @abstractmethod
    def decode_response(self, payload: bytes) -> dict[str, Any]:
        """把 sim 发来的 payload bytes 解成业务可消费的 dict。

        要求 dict 里:
          - 正常响应: 业务字段
          - 错误响应: {"error": "...", "error_code": optional}
        """
        raise NotImplementedError

    @abstractmethod
    def read_handshake(self, payload: bytes) -> dict[str, Any]:
        """解析 server 握手帧。失败时返 {"error": ...}。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """codec 名,用于日志/pipe 命名"""


__all__ = ["ProtocolCodec"]
