"""PipeConnection — 高层 pipe 连接管理。

唯一的 combat/full-run 连接入口。封装所有通用设施:
  - Lifecycle: connect / close / is_connected
  - Thread safety: 内置 RLock,所有 call 串行化(安全让多 thread 共享一个 conn)
  - 自动重连: call 遇 ConnectionError/TimeoutError → 重连 + fallback 重启 sim
  - 错误分类: ConnectionError / TransportTimeoutError / SimulatorApiError
  - 协议编码: 4 字节长度前缀 + JSON payload (proto wire 以后再切,接口保持不变)

禁止任何上层模块自己重写连接/重连/锁/heartbeat 逻辑。heartbeat 走
`transport.heartbeat.HealthMonitor`(用独立 PipeConnection 保活,不和业务共享)。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from game_bridge.transport.codec import JsonCodec, ProtocolCodec
from game_bridge.transport.pipe_transport import (
    PipeTransport,
    TransportClosedError,
    TransportTimeoutError,
)

logger = logging.getLogger(__name__)


class SimulatorApiError(RuntimeError):
    """sim 返回 {"error": "..."} 的逻辑错误 (非 pipe 故障)。"""
    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


@dataclass
class PipeConnectionConfig:
    """连接参数。"""
    port: int
    pipe_name_prefix: str = "sts2_mcts"     # 最终 pipe 名 = prefix_{port}
    protocol: str = "json"                  # "json" / "bin" / "proto"
    connect_timeout_s: float = 10.0
    default_call_timeout_s: float = 30.0
    write_timeout_ms: int = 10000
    max_reconnect_attempts: int = 3
    reconnect_backoff_s: float = 0.5
    auto_launch: bool = False
    sim_launcher: Callable[[int], Any] | None = None  # 返回 sim 进程句柄;用于 auto_launch
    sim_stopper: Callable[[Any], None] | None = None  # 停 sim 进程(reconnect 失败时重启 sim)
    # 协议 codec:控制 encode_request / decode_response / handshake 字节语义。
    # 留 None 时,按 `protocol` 字段自动选:json→JsonCodec,bin/proto→需调用方显式传。
    codec: ProtocolCodec | None = None

    @property
    def pipe_name(self) -> str:
        # combat_training_env 以前用 sts2_mcts_{port} (JSON);保持兼容
        p = self.protocol.lower()
        if p == "bin":
            return f"sts2_mcts_bin_{self.port}"
        if p == "proto":
            return f"sts2_mcts_proto_{self.port}"
        return f"{self.pipe_name_prefix}_{self.port}"

    def resolve_codec(self) -> ProtocolCodec:
        """选中有效 codec:显式传 > protocol 字段推断 > JsonCodec 兜底。"""
        if self.codec is not None:
            return self.codec
        return JsonCodec()


class PipeConnection:
    """高层 pipe 连接。

    使用 (最常见 json 协议):
        cfg = PipeConnectionConfig(port=17000, auto_launch=True, sim_launcher=...)
        conn = PipeConnection(cfg)
        conn.connect()
        result = conn.safe_call("combat_reset", {...})
        conn.close()

    线程安全: 多 thread 可以并发 call();内部用 RLock 串行化。
    不要把同一个 conn 实例的底层 transport 暴露给外部直接 IO。
    """

    def __init__(self, config: PipeConnectionConfig):
        self.cfg = config
        self._codec: ProtocolCodec = config.resolve_codec()
        self._lock = threading.RLock()
        self._transport: PipeTransport | None = None
        self._sim_proc: Any | None = None
        self._call_count = 0
        self._error_count = 0
        self._last_activity = time.monotonic()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """建立连接。若 auto_launch 且 pipe 未就绪,先拉起 sim。"""
        with self._lock:
            if self._transport is not None and self._transport.is_connected():
                return
            self._transport = PipeTransport(self.cfg.pipe_name)
            try:
                self._transport.connect(timeout_s=self.cfg.connect_timeout_s)
            except ConnectionError:
                if not self.cfg.auto_launch or self.cfg.sim_launcher is None:
                    raise
                logger.info(f"[pipe {self.cfg.port}] auto-launch sim and retry connect")
                self._sim_proc = self.cfg.sim_launcher(self.cfg.port)
                self._transport = PipeTransport(self.cfg.pipe_name)
                self._transport.connect(
                    timeout_s=max(15.0, self.cfg.connect_timeout_s),
                )
            # 握手(json 协议:server 发 {"ok": true} 欢迎帧)
            self._handshake()
            self._last_activity = time.monotonic()

    def close(self) -> None:
        with self._lock:
            if self._transport is not None:
                try:
                    self._transport.close()
                except Exception:
                    pass
                self._transport = None
            if self._sim_proc is not None and self.cfg.sim_stopper is not None:
                try:
                    self.cfg.sim_stopper(self._sim_proc)
                except Exception:
                    pass
                self._sim_proc = None

    def is_connected(self) -> bool:
        return self._transport is not None and self._transport.is_connected()

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    def call(self, method: str, params: dict[str, Any] | None = None,
             *, timeout_s: float | None = None) -> dict[str, Any]:
        """同步 RPC (JSON 协议);不带重连。失败抛 ConnectionError / TimeoutError / SimulatorApiError。"""
        if timeout_s is None:
            timeout_s = self.cfg.default_call_timeout_s
        timeout_ms = int(timeout_s * 1000)

        payload = self._encode_request(method, params or {})
        with self._lock:
            if self._transport is None or not self._transport.is_connected():
                raise TransportClosedError(f"pipe {self.cfg.pipe_name} not connected")
            self._transport.write_frame(payload, timeout_ms=self.cfg.write_timeout_ms)
            resp_bytes = self._transport.read_frame(timeout_ms=timeout_ms)
            self._call_count += 1
            self._last_activity = time.monotonic()

        result = self._decode_response(resp_bytes)
        if isinstance(result, dict) and result.get("error"):
            raise SimulatorApiError(
                str(result["error"]),
                error_code=result.get("error_code"),
            )
        return result

    def safe_call(self, method: str, params: dict[str, Any] | None = None,
                  *, timeout_s: float | None = None) -> dict[str, Any]:
        """带自动重连的 call。

        重试策略:
          attempt 0: 直接调
          attempt 1..(N-1): pipe 重连,sim 进程保留
          attempt N: 重启 sim 进程 + 重连

        SimulatorApiError (sim 逻辑错) 不重试,直接 raise (调用方该 handle)。
        """
        last_exc: Exception | None = None
        max_attempts = max(1, self.cfg.max_reconnect_attempts)
        for attempt in range(max_attempts + 1):
            try:
                return self.call(method, params, timeout_s=timeout_s)
            except SimulatorApiError:
                raise  # sim logic error,不重试
            except (ConnectionError, TransportClosedError, TransportTimeoutError) as e:
                last_exc = e
                self._error_count += 1
                if attempt == max_attempts:
                    break
                restart_sim = (attempt == max_attempts - 1)  # 最后一次 retry 前重启 sim
                logger.warning(
                    f"[pipe {self.cfg.port}] call '{method}' failed (attempt {attempt+1}): "
                    f"{type(e).__name__}: {str(e)[:120]}. reconnecting (restart_sim={restart_sim})"
                )
                try:
                    self._reconnect(restart_sim=restart_sim)
                except Exception as rec_exc:
                    last_exc = rec_exc
                    time.sleep(self.cfg.reconnect_backoff_s * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def ping(self, method: str = "state") -> bool:
        """健康检查(heartbeat 用)。返回 True 表示 pipe 活。任何失败返 False。"""
        try:
            self.call(method, {}, timeout_s=5.0)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handshake(self) -> None:
        """读 server 欢迎帧(可能为空或含版本信息)。

        握手字节语义由 codec 决定:JsonCodec 期望 {"ok": true},proto/bin
        codec 期望 opcode=0x00 payload。任意错误在 codec 侧以 dict['error']
        上报,此处转换为 SimulatorApiError。
        """
        try:
            hello_bytes = self._transport.read_frame(timeout_ms=5000)
        except TransportTimeoutError:
            # 一些 server 不发欢迎帧,容忍
            return
        hello = self._codec.read_handshake(hello_bytes)
        if isinstance(hello, dict) and hello.get("error"):
            raise SimulatorApiError(
                str(hello["error"]),
                error_code=hello.get("error_code"),
            )

    def _encode_request(self, method: str, params: dict[str, Any]) -> bytes:
        return self._codec.encode_request(method, params)

    def _decode_response(self, data: bytes) -> dict[str, Any]:
        return self._codec.decode_response(data)

    def _reconnect(self, *, restart_sim: bool = False) -> None:
        with self._lock:
            try:
                if self._transport is not None:
                    self._transport.close()
            except Exception:
                pass
            self._transport = None
            if restart_sim and self._sim_proc is not None and self.cfg.sim_stopper is not None:
                try:
                    self.cfg.sim_stopper(self._sim_proc)
                except Exception:
                    pass
                self._sim_proc = None
            self.connect()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "port": self.cfg.port,
            "pipe": self.cfg.pipe_name,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "connected": self.is_connected(),
            "idle_s": round(time.monotonic() - self._last_activity, 2),
        }

    @property
    def last_activity(self) -> float:
        return self._last_activity

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
