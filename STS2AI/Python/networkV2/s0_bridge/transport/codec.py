"""协议 codec - 把 JSON / binary / proto 的 encode/decode 抽成统一接口。

PipeConnection 本身协议无关,通过注入 `ProtocolCodec` 决定 wire format。
combat / full-run / 未来的 RPC 都复用 PipeConnection + 对应 codec,禁止
业务模块自己写字节编码。
"""
from __future__ import annotations

import json
import struct
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


# ---------------------------------------------------------------------------
# JSON codec - {"method": "...", "params": {...}} + 4-byte length frame
# ---------------------------------------------------------------------------

class JsonCodec(ProtocolCodec):
    """JSON 协议 (combat_training_env 历史默认,pipe 名 sts2_mcts_{port})。"""

    name = "json"

    def encode_request(self, method: str, params: dict[str, Any] | None) -> bytes:
        req: dict[str, Any] = {"method": method}
        if params:
            req["params"] = params
        return json.dumps(req).encode("utf-8")

    def decode_response(self, payload: bytes) -> dict[str, Any]:
        return json.loads(payload.decode("utf-8"))

    def read_handshake(self, payload: bytes) -> dict[str, Any]:
        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            return {}


# ---------------------------------------------------------------------------
# Binary opcode codec - 给老 binary 协议 / 未来 proto 提供脚手架
# 现阶段只用在 JSON 旁补占位;具体 proto encode/decode 放
# `networkV2.s0_bridge.proto_pipe_client` 逻辑后迁过来(下一 PR)。
# ---------------------------------------------------------------------------

class BinaryOpcodeCodec(ProtocolCodec):
    """Binary RPC 协议骨架:[status 1][opcode 1][payload bytes]。

    这是 `proto_pipe_client.py` 的 wire format。具体 opcode dispatch + proto
    decode 逻辑本身不在这里(那涉及 protobuf 生成代码和对齐旧 binary decoder),
    留作 **codec.BinaryOpcodeCodec 子类** 或注入 `payload_decoder` 回调。
    当前只保证"有占位,不造新轮子"。
    """

    name = "binary"

    def __init__(
        self,
        *,
        request_encoder,                  # Callable[[str, dict|None], bytes]
        response_decoder,                 # Callable[[int, int, bytes], dict]
        handshake_decoder,                # Callable[[bytes], dict]
    ):
        self._encode = request_encoder
        self._decode = response_decoder
        self._handshake = handshake_decoder

    def encode_request(self, method: str, params: dict[str, Any] | None) -> bytes:
        return self._encode(method, params or {})

    def decode_response(self, payload: bytes) -> dict[str, Any]:
        # binary wire: [status][opcode][payload...]
        if len(payload) < 2:
            return {"error": f"binary payload too short: {len(payload)} bytes"}
        status = payload[0]
        opcode = payload[1]
        inner = bytes(payload[2:])
        return self._decode(status, opcode, inner)

    def read_handshake(self, payload: bytes) -> dict[str, Any]:
        return self._handshake(payload)


__all__ = ["ProtocolCodec", "JsonCodec", "BinaryOpcodeCodec"]
