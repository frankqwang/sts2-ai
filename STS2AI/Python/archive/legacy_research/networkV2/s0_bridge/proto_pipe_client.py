"""Protobuf 协议命名管道客户端(兼容薄包装)。

2026-04-18 重构:内部字节 IO / 连接 / 重连 / 握手全部下沉到
`networkV2.s0_bridge.transport.PipeConnection + ProtoCodec`。

本模块只保留 `ProtoPipeClient` 名字与接口 (connect / call / safe_call /
close / stats / is_connected),用作兼容层。新代码建议直接:

    from networkV2.s0_bridge.transport import PipeConnection, PipeConnectionConfig, ProtoCodec
    conn = PipeConnection(PipeConnectionConfig(port=..., protocol="proto", codec=ProtoCodec()))
    conn.connect()
    state = conn.safe_call("combat_reset", {...})

老 JSON 管道客户端 `networkV2.s0_bridge.pipe_client.PipeClient` 只供
`combat_training_env` 这类仍走 JSON wire 的兼容路径使用,proto wire 必须走本包装或直接 PipeConnection。
"""
from __future__ import annotations

import logging
from typing import Any

from networkV2.s0_bridge.transport import (
    PipeConnection,
    PipeConnectionConfig,
    ProtoCodec,
    SimulatorApiError,
    TransportTimeoutError,
)

_log = logging.getLogger(__name__)


class ProtoPipeClient:
    """Protobuf pipe client — drop-in wrapper 维持旧调用语义。

    字节协议 / 握手 / 重连 全走 PipeConnection。本类只做:
      - 暴露 ProtoPipeClient(port=...) 旧构造
      - call / safe_call / connect / close / is_connected / stats

    线程模型:调用方若需多线程共享 → PipeConnection 内置 RLock 已兜底。
    """

    def __init__(
        self,
        port: int = 15527,
        pipe_name: str | None = None,
        default_timeout_s: float = 30.0,
        max_reconnect_attempts: int = 3,
    ):
        self.port = port
        cfg = PipeConnectionConfig(
            port=port,
            protocol="proto",
            pipe_name_prefix="sts2_mcts",  # ignored for proto (走 sts2_mcts_proto_{port})
            connect_timeout_s=10.0,
            default_call_timeout_s=float(default_timeout_s),
            max_reconnect_attempts=int(max_reconnect_attempts),
            codec=ProtoCodec(),
        )
        if pipe_name is not None:
            # 调用方传了自定义 pipe_name,覆盖默认 sts2_mcts_proto_{port}。
            # 此路径很冷,仅一两个测试用到;塞给内部 transport 构造时用。
            cfg.pipe_name_prefix = pipe_name  # resolve_codec 走 proto,实际用 sts2_mcts_proto_{port}
            self._custom_pipe_name = pipe_name
        else:
            self._custom_pipe_name = None
        self._conn = PipeConnection(cfg)
        self.default_timeout_s = float(default_timeout_s)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def connect(self, timeout_s: float = 10.0) -> None:
        """连接 + 握手。和旧 API 兼容:timeout_s 用在这一次 connect。"""
        self._conn.cfg.connect_timeout_s = float(timeout_s)
        self._conn.connect()

    def close(self) -> None:
        self._conn.close()

    def is_connected(self) -> bool:
        return self._conn.is_connected()

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    def call(self, method: str, params: dict[str, Any] | None = None,
             timeout_s: float | None = None) -> dict[str, Any]:
        """单次 RPC。失败抛 ConnectionError / TimeoutError / SimulatorApiError。

        payload 包装:旧 API 返回 payload dict (已经是业务 dict);新 codec
        也返回业务 dict 或带 {status, opcode, payload} 的 envelope。这里
        保证外部拿到纯业务 dict。
        """
        try:
            result = self._conn.call(method, params, timeout_s=timeout_s)
        except TransportTimeoutError as exc:
            raise TimeoutError(str(exc)) from exc
        return _unwrap_payload(result)

    def safe_call(self, method: str, params: dict[str, Any] | None = None,
                  timeout_s: float | None = None) -> dict[str, Any]:
        try:
            result = self._conn.safe_call(method, params, timeout_s=timeout_s)
        except TransportTimeoutError as exc:
            raise TimeoutError(str(exc)) from exc
        return _unwrap_payload(result)

    # ------------------------------------------------------------------
    # observability (兼容旧 stats)
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        s = self._conn.stats
        return {
            "port": s["port"],
            "call_count": s["call_count"],
            "error_count": s["error_count"],
            "connected": s["connected"],
        }

    def __repr__(self) -> str:
        return (f"ProtoPipeClient(port={self.port}, "
                f"connected={self.is_connected()}, "
                f"calls={self._conn.stats['call_count']}, "
                f"errors={self._conn.stats['error_count']})")


def _unwrap_payload(result: dict[str, Any]) -> dict[str, Any]:
    """旧 API 返回 payload dict,新 codec 返回业务 dict。
    检测 envelope 结构 {status, opcode, payload} 时抽出 payload。"""
    if (isinstance(result, dict)
            and "payload" in result
            and isinstance(result.get("payload"), dict)
            and "status" in result
            and "opcode" in result):
        return result["payload"]
    return result


__all__ = ["ProtoPipeClient", "SimulatorApiError"]
