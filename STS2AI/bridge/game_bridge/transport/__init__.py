"""game_bridge.transport — 统一 pipe 传输层。

**唯一**的 pipe 连接管理设施。所有 session (combat / full-run) 和
ProtoPipeClient / PipeBackedCombatTrainingClient 等 client 都走这里,
禁止任何模块自己造 pipe/reconnect/heartbeat/lock 轮子。

模块分层:
  - pipe_transport.py — 纯 Win32 named pipe 字节 IO (read/write/connect/close)
  - connection.py     — 高层 PipeConnection: RLock + safe_call + 自动重连
  - heartbeat.py      — 独立 pipe 连接做保活的 HealthMonitor

使用范例:
    from game_bridge.transport import PipeConnection

    conn = PipeConnection(port=17000, auto_launch=True, heartbeat_interval_s=5.0)
    conn.connect()
    try:
        result = conn.safe_call("combat_reset", {...})  # 自动重连 + 线程安全
    finally:
        conn.close()
"""
from game_bridge.transport.pipe_transport import (
    PipeTransport,
    TransportClosedError,
    TransportTimeoutError,
)
from game_bridge.transport.codec import JsonCodec, ProtocolCodec
from game_bridge.transport.connection import (
    PipeConnection,
    PipeConnectionConfig,
    SimulatorApiError,
)
from game_bridge.transport.heartbeat import HealthMonitor
from game_bridge.transport.proto_codec import ProtoCodec

__all__ = [
    "PipeTransport",
    "PipeConnection",
    "PipeConnectionConfig",
    "HealthMonitor",
    "SimulatorApiError",
    "TransportClosedError",
    "TransportTimeoutError",
    "ProtocolCodec",
    "JsonCodec",
    "ProtoCodec",
]
