"""Protobuf 协议命名管道客户端。

替代 env.binary_pipe_client.BinaryPipeClient:
- 传输层不变（Named Pipe + overlapped I/O + 长度前缀帧）
- 请求编码不变（opcode + 二进制参数，够简单不需要 proto）
- 响应的 state payload 从手写二进制改为 protobuf
- 不再需要符号表 (_symbol_table) 和玩家静态缓存 (_player_static_cache)

接口与 BinaryPipeClient 完全一致，call() 返回相同格式的 dict。

并发安全 & 资源管理:
- 单个 ProtoPipeClient 实例 **不是线程安全的**（和 BinaryPipeClient 一样）
- 多环境并发请每个 worker 创建独立实例（不同 port）
- 连接断开时自动清理句柄，支持 reconnect
- protobuf 解析后立即释放中间 message 对象，避免大量 state 堆积
"""
from __future__ import annotations

import json
import logging
import struct
import sys
import time
from typing import Any

_log = logging.getLogger(__name__)

from env.pipe_client import (
    FILE_FLAG_OVERLAPPED,
    GENERIC_READ,
    GENERIC_WRITE,
    INVALID_HANDLE_VALUE,
    OPEN_EXISTING,
    PipeClient,
    _kernel32,
)
from env.simulator_api_error import SimulatorApiError

from networkV2.s0_bridge.generated import game_state_pb2 as pb
from networkV2.s0_bridge.proto_state_converter import game_state_to_dict
from networkV2.s0_bridge.constants import (
    ACTION_CODES,
    OP_BATCH_STEP,
    OP_DELETE_STATE,
    OP_EXPORT_STATE,
    OP_HANDSHAKE,
    OP_IMPORT_STATE,
    OP_LOAD_ORT_MODEL,
    OP_LOAD_STATE,
    OP_PERF_STATS,
    OP_RESET,
    OP_RESET_PERF_STATS,
    OP_RUN_COMBAT_LOCAL,
    OP_SAVE_STATE,
    OP_SEARCH_COMBAT_MCTS,
    OP_SKIP_COMBAT,
    OP_STATE,
    OP_STEP,
    OP_STEP_LOCAL_POLICY,
    PROTO_PROTOCOL_VERSION,
    PROTO_SCHEMA_ID,
    STATUS_OK,
    STATUS_PROTOCOL_ERROR,
    STATUS_SIMULATOR_ERROR,
)

import ctypes


class ProtoPipeClient(PipeClient):
    """Protobuf named-pipe client — drop-in replacement for BinaryPipeClient.

    调用方只需把::

        from env.binary_pipe_client import BinaryPipeClient
        client = BinaryPipeClient(port=15527)

    改为::

        from networkV2.s0_bridge.proto_pipe_client import ProtoPipeClient
        client = ProtoPipeClient(port=15527)

    其他代码（connect / call / close）零修改。

    并发使用:
        每个 worker / env 实例应创建独立的 ProtoPipeClient，使用不同端口::

            clients = [ProtoPipeClient(port=15527 + i) for i in range(num_envs)]
            for c in clients:
                c.connect()

        单实例 **不可** 跨线程共享。
    """

    def __init__(
        self,
        port: int = 15527,
        pipe_name: str | None = None,
        default_timeout_s: float = 30.0,
        max_reconnect_attempts: int = 3,
    ):
        super().__init__(
            port=port,
            pipe_name=pipe_name or f"sts2_mcts_proto_{port}",
            default_timeout_s=default_timeout_s,
        )
        self.port = port
        self._max_reconnect_attempts = max_reconnect_attempts
        self._protocol_version: int | None = None
        self._server_build_git_sha: str | None = None
        self._server_schema_id: str | None = None
        self._call_count: int = 0
        self._error_count: int = 0

    # ------------------------------------------------------------------
    # 连接 & 握手
    # ------------------------------------------------------------------

    def connect(self, timeout_s: float = 10.0) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Named pipes are only supported on Windows")

        pipe_path = f"\\\\.\\pipe\\{self.pipe_name}"
        deadline = time.monotonic() + timeout_s
        last_err = None

        while time.monotonic() < deadline:
            try:
                if not _kernel32.WaitNamedPipeW(pipe_path, 200):
                    last_err = f"Pipe {pipe_path} not ready"
                    time.sleep(0.1)
                    continue

                handle = _kernel32.CreateFileW(
                    pipe_path,
                    GENERIC_READ | GENERIC_WRITE,
                    0,
                    None,
                    OPEN_EXISTING,
                    FILE_FLAG_OVERLAPPED,
                    None,
                )
                if handle == INVALID_HANDLE_VALUE:
                    last_err = f"CreateFileW failed: winerror={ctypes.GetLastError()}"
                    time.sleep(0.1)
                    continue

                self._handle = handle
                self._event = _kernel32.CreateEventW(None, True, False, None)

                hello = self._read_raw_response(timeout_s=timeout_s, expect_handshake=True)
                if hello.get("status") != STATUS_OK or hello.get("opcode") != OP_HANDSHAKE:
                    self.close()
                    raise ConnectionError(str(hello.get("error") or f"Unexpected handshake: {hello!r}"))

                self._protocol_version = int(hello.get("version") or 0)
                if self._protocol_version != PROTO_PROTOCOL_VERSION:
                    self.close()
                    raise ConnectionError(
                        f"Proto protocol version mismatch: "
                        f"expected {PROTO_PROTOCOL_VERSION}, got {self._protocol_version}"
                    )
                self._server_build_git_sha = str(hello.get("build_git_sha") or "").strip() or None
                self._server_schema_id = str(hello.get("schema_id") or "").strip() or None
                if self._server_schema_id != PROTO_SCHEMA_ID:
                    self.close()
                    raise ConnectionError(
                        f"Proto schema mismatch: "
                        f"expected {PROTO_SCHEMA_ID}, got {self._server_schema_id or '<missing>'}"
                    )
                return

            except (SimulatorApiError, ConnectionError):
                raise
            except OSError as exc:
                last_err = f"Pipe open failed: {exc}"
                time.sleep(0.1)

        raise ConnectionError(f"Failed to connect after {timeout_s}s: {last_err}")

    # ------------------------------------------------------------------
    # 主调用接口
    # ------------------------------------------------------------------

    def call(self, method: str, params: dict[str, Any] | None = None,
             timeout_s: float | None = None) -> dict[str, Any]:
        if self._handle is None:
            raise ConnectionError("Not connected. Call connect() first.")
        if timeout_s is None:
            timeout_s = self.default_timeout_s

        request = self._encode_request(method, params or {})
        try:
            self._write_bytes(struct.pack("<I", len(request)) + request)
            result = self._read_raw_response(timeout_s=timeout_s)
        except (ConnectionError, TimeoutError, OSError) as exc:
            self._error_count += 1
            _log.warning("pipe call %s failed (port=%d, errors=%d): %s",
                         method, self.port, self._error_count, exc)
            raise
        finally:
            self._call_count += 1

        status = int(result.get("status", STATUS_SIMULATOR_ERROR))
        if status in {STATUS_PROTOCOL_ERROR, STATUS_SIMULATOR_ERROR}:
            self._error_count += 1
            raise SimulatorApiError(
                str(result.get("error") or "Proto pipe error"),
                error_code=result.get("error_code"),
            )
        return result["payload"]

    def safe_call(self, method: str, params: dict[str, Any] | None = None,
                  timeout_s: float | None = None) -> dict[str, Any]:
        """带自动重连的 call — 连接断开时尝试重建。

        适用于长时间运行的训练/评估循环，避免因偶发管道断开中断整个流程。
        """
        last_exc: Exception | None = None
        for attempt in range(1 + self._max_reconnect_attempts):
            try:
                return self.call(method, params, timeout_s)
            except (ConnectionError, OSError) as exc:
                last_exc = exc
                if attempt < self._max_reconnect_attempts:
                    _log.warning("reconnecting (attempt %d/%d, port=%d): %s",
                                 attempt + 1, self._max_reconnect_attempts,
                                 self.port, exc)
                    self.close()
                    time.sleep(0.5 * (attempt + 1))
                    try:
                        self.connect(timeout_s=10.0)
                    except Exception as conn_exc:
                        _log.error("reconnect failed: %s", conn_exc)
                        last_exc = conn_exc
        raise ConnectionError(
            f"All {self._max_reconnect_attempts} reconnect attempts failed"
        ) from last_exc

    @property
    def stats(self) -> dict[str, int]:
        """调用/错误统计，用于多环境监控。"""
        return {
            "port": self.port,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "connected": self.is_connected(),
        }

    def __repr__(self) -> str:
        return (f"ProtoPipeClient(port={self.port}, "
                f"connected={self.is_connected()}, "
                f"calls={self._call_count}, errors={self._error_count})")

    # ------------------------------------------------------------------
    # 响应解码
    # ------------------------------------------------------------------

    def _read_raw_response(self, *, timeout_s: float,
                           expect_handshake: bool = False) -> dict[str, Any]:
        """读取一帧响应：[4字节长度][status][opcode][payload]。"""
        timeout_ms = int(timeout_s * 1000)
        length = struct.unpack("<I", self._read_bytes(4, timeout_ms))[0]
        body = self._read_bytes(length, timeout_ms)

        status = body[0]
        opcode = body[1]
        payload_bytes = body[2:]

        if expect_handshake:
            return self._decode_handshake(status, opcode, payload_bytes)

        if status in {STATUS_PROTOCOL_ERROR, STATUS_SIMULATOR_ERROR}:
            return self._decode_error(status, opcode, payload_bytes)

        payload = self._decode_payload(opcode, payload_bytes)
        return {"status": status, "opcode": opcode, "payload": payload}

    def _decode_handshake(self, status: int, opcode: int,
                          data: bytes) -> dict[str, Any]:
        """握手响应：[u16 version][string build_git_sha][string schema_id]。"""
        if status == STATUS_OK:
            off = 0
            version = struct.unpack_from("<H", data, off)[0]; off += 2
            sha, off = _read_string(data, off)
            schema_id, off = _read_string(data, off)
            return {
                "status": status,
                "opcode": opcode,
                "version": version,
                "build_git_sha": sha,
                "schema_id": schema_id,
            }
        # 错误握手
        off = 0
        error_code, off = _read_string(data, off)
        error_msg, off = _read_string(data, off)
        return {"status": status, "opcode": opcode,
                "error_code": error_code, "error": error_msg}

    @staticmethod
    def _decode_error(status: int, opcode: int,
                      data: bytes) -> dict[str, Any]:
        off = 0
        error_code, off = _read_string(data, off)
        error_msg, off = _read_string(data, off)
        return {
            "status": status,
            "opcode": opcode,
            "error_code": error_code,
            "error": error_msg,
            "payload": {},
        }

    def _decode_payload(self, opcode: int, data: bytes) -> dict[str, Any]:
        """按 opcode 解析 payload。状态类 payload 用 protobuf 解码。"""
        if opcode in {OP_RESET, OP_STATE, OP_LOAD_STATE, OP_IMPORT_STATE}:
            return self._parse_proto_state(data)

        if opcode == OP_STEP:
            accepted = bool(data[0])
            off = 1
            error, off = _read_optional_string(data, off)
            state = self._parse_proto_state(data[off:])
            return {
                "accepted": accepted,
                "error": error,
                "state": state,
                "reward": _terminal_reward(state),
                "done": bool(state.get("terminal")),
                "info": {
                    "state_type": state.get("state_type"),
                    "run_outcome": state.get("run_outcome"),
                },
            }

        if opcode == OP_BATCH_STEP:
            accepted = bool(data[0])
            steps_executed = struct.unpack_from("<H", data, 1)[0]
            off = 3
            error, off = _read_optional_string(data, off)
            state = self._parse_proto_state(data[off:])
            return {
                "accepted": accepted,
                "steps_executed": steps_executed,
                "error": error,
                "state": state,
            }

        if opcode == OP_SAVE_STATE:
            off = 0
            state_id, off = _read_string(data, off)
            cache_size = struct.unpack_from("<i", data, off)[0]
            return {"state_id": state_id, "cache_size": cache_size}

        if opcode == OP_EXPORT_STATE:
            off = 0
            path, off = _read_string(data, off)
            cache_size = struct.unpack_from("<i", data, off)[0]
            return {"path": path, "cache_size": cache_size}

        if opcode == OP_DELETE_STATE:
            return {
                "deleted": bool(data[0]),
                "cache_size": struct.unpack_from("<i", data, 1)[0],
            }

        if opcode == OP_PERF_STATS:
            off = 0
            json_str, off = _read_string(data, off)
            return json.loads(json_str or "{}")

        if opcode == OP_RESET_PERF_STATS:
            return {"reset": bool(data[0])}

        if opcode == OP_LOAD_ORT_MODEL:
            payload: dict[str, Any] = {"loaded": bool(data[0])}
            off = 1
            if off + 4 <= len(data):
                payload["has_value"] = bool(data[off]); off += 1
                payload["has_deck_inputs"] = bool(data[off]); off += 1
                payload["has_continuation_output"] = bool(data[off]); off += 1
                payload["has_extra_scalars_input"] = bool(data[off]); off += 1
            if off < len(data):
                payload["execution_provider"], off = _read_string(data, off)
            if off < len(data):
                payload["requested_device"], off = _read_string(data, off)
            if off < len(data):
                payload["fell_back_to_cpu"] = bool(data[off])
            return payload

        if opcode == OP_SKIP_COMBAT:
            accepted = bool(data[0])
            off = 1
            error, off = _read_optional_string(data, off)
            state = self._parse_proto_state(data[off:])
            return {"accepted": accepted, "error": error, "state": state, "skipped": True}

        if opcode == OP_RUN_COMBAT_LOCAL:
            combat_steps = struct.unpack_from("<H", data, 0)[0]
            elapsed_ms = struct.unpack_from("<f", data, 2)[0]
            off = 6
            timing: dict[str, float] = {}
            try:
                timing["get_snapshot_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
                timing["ort_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
                timing["step_async_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
                timing["wait_async_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
                timing["max_step_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
                timing["max_wait_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
            except Exception:
                pass
            state = self._parse_proto_state(data[off:])
            return {
                "combat_steps": combat_steps,
                "elapsed_ms": elapsed_ms,
                "timing": timing,
                "state": state,
            }

        if opcode == OP_SEARCH_COMBAT_MCTS:
            return self._decode_mcts_payload(data)

        raise RuntimeError(f"Unsupported proto response opcode: {opcode}")

    def _parse_proto_state(self, data: bytes) -> dict[str, Any]:
        """protobuf bytes → GameState proto → 兼容 dict。

        解析后立即转为 dict，不持有 proto message 引用，
        避免大量 GameState 对象堆积导致内存增长。
        """
        gs = pb.GameState()
        gs.ParseFromString(data)
        result = game_state_to_dict(gs)
        gs.Clear()  # 显式释放 proto 内部缓冲区
        return result

    @staticmethod
    def _decode_mcts_payload(data: bytes) -> dict[str, Any]:
        off = 0
        action_index = struct.unpack_from("<h", data, off)[0]; off += 2
        count = struct.unpack_from("<H", data, off)[0]; off += 2
        visit_counts = []
        for _ in range(count):
            visit_counts.append(struct.unpack_from("<i", data, off)[0]); off += 4
        visit_probs = []
        for _ in range(count):
            visit_probs.append(struct.unpack_from("<f", data, off)[0]); off += 4
        q_values = []
        for _ in range(count):
            q_values.append(struct.unpack_from("<f", data, off)[0]); off += 4
        priors = []
        for _ in range(count):
            priors.append(struct.unpack_from("<f", data, off)[0]); off += 4
        payload: dict[str, Any] = {
            "action_index": action_index,
            "visit_counts": visit_counts,
            "visit_probs": visit_probs,
            "q_values": q_values,
            "priors": priors,
            "root_value": struct.unpack_from("<f", data, off)[0],
            "search_ms": struct.unpack_from("<f", data, off + 4)[0],
            "restored_ok": bool(data[off + 8]),
            "snapshot_count": struct.unpack_from("<i", data, off + 9)[0],
        }
        off += 13
        if off + (11 * 4) + (8 * 4) <= len(data):
            payload["breakdown"] = {
                "simulation_count": struct.unpack_from("<i", data, off)[0],
                "save_state_count": struct.unpack_from("<i", data, off + 4)[0],
                "load_state_count": struct.unpack_from("<i", data, off + 8)[0],
                "delete_state_count": struct.unpack_from("<i", data, off + 12)[0],
                "step_count": struct.unpack_from("<i", data, off + 16)[0],
                "advance_state_count": struct.unpack_from("<i", data, off + 20)[0],
                "eval_call_count": struct.unpack_from("<i", data, off + 24)[0],
                "eval_batch_count": struct.unpack_from("<i", data, off + 28)[0],
                "eval_state_count": struct.unpack_from("<i", data, off + 32)[0],
                "select_child_count": struct.unpack_from("<i", data, off + 36)[0],
                "backprop_count": struct.unpack_from("<i", data, off + 40)[0],
                "save_state_ms": struct.unpack_from("<f", data, off + 44)[0],
                "load_state_ms": struct.unpack_from("<f", data, off + 48)[0],
                "delete_state_ms": struct.unpack_from("<f", data, off + 52)[0],
                "step_ms": struct.unpack_from("<f", data, off + 56)[0],
                "advance_state_ms": struct.unpack_from("<f", data, off + 60)[0],
                "eval_ms": struct.unpack_from("<f", data, off + 64)[0],
                "selection_ms": struct.unpack_from("<f", data, off + 68)[0],
                "backprop_ms": struct.unpack_from("<f", data, off + 72)[0],
            }
            off += 76
        if off < len(data):
            has_trace = data[off]; off += 1
            if has_trace:
                trace_len = struct.unpack_from("<H", data, off)[0]; off += 2
                trace_json = data[off:off + trace_len].decode("utf-8")
                try:
                    payload["debug_trace"] = json.loads(trace_json)
                except Exception:
                    payload["debug_trace_json"] = trace_json
        return payload

    # ------------------------------------------------------------------
    # 请求编码（与 BinaryPipeClient 完全一致）
    # ------------------------------------------------------------------

    def _encode_request(self, method: str, params: dict[str, Any]) -> bytes:
        method = str(method).strip().lower()
        body = bytearray()

        if method == "reset":
            body.append(OP_RESET)
            _write_optional_string(body, params.get("character_id") or params.get("character"))
            _write_optional_string(body, params.get("seed"))
            body.extend(struct.pack("<i", int(params.get("ascension_level", params.get("ascension", 0)) or 0)))
            build = params.get("build")
            build_json = None if build is None else json.dumps(build, ensure_ascii=False, separators=(",", ":"))
            _write_optional_string(body, build_json)
            return bytes(body)

        if method in {"state", "get_state", "legal_actions"}:
            return bytes([OP_STATE])

        if method == "step":
            body.append(OP_STEP)
            _write_action(body, params)
            return bytes(body)

        if method == "batch_step":
            actions = list(params.get("actions") or [])
            body.append(OP_BATCH_STEP)
            body.extend(struct.pack("<H", len(actions)))
            for action in actions:
                _write_action(body, action or {})
            return bytes(body)

        if method == "save_state":
            return bytes([OP_SAVE_STATE])

        if method == "export_state":
            body.append(OP_EXPORT_STATE)
            _write_string(body, str(params["path"]))
            _write_optional_string(body, params.get("state_id"))
            return bytes(body)

        if method == "import_state":
            body.append(OP_IMPORT_STATE)
            _write_string(body, str(params["path"]))
            return bytes(body)

        if method == "load_state":
            body.append(OP_LOAD_STATE)
            _write_string(body, str(params["state_id"]))
            return bytes(body)

        if method in {"delete_state", "clear_state_cache"}:
            clear_all = bool(params.get("clear_all")) or method == "clear_state_cache"
            body.append(OP_DELETE_STATE)
            body.append(1 if clear_all else 0)
            if not clear_all:
                _write_string(body, str(params["state_id"]))
            return bytes(body)

        if method == "perf_stats":
            return bytes([OP_PERF_STATS])

        if method == "reset_perf_stats":
            return bytes([OP_RESET_PERF_STATS])

        if method == "step_local_policy":
            return bytes([OP_STEP_LOCAL_POLICY])

        if method == "skip_combat":
            return bytes([OP_SKIP_COMBAT])

        if method == "run_combat_local":
            body.append(OP_RUN_COMBAT_LOCAL)
            body.extend(struct.pack("<H", int(params.get("max_steps", 600))))
            return bytes(body)

        if method == "load_ort_model":
            body.append(OP_LOAD_ORT_MODEL)
            path_bytes = str(params.get("path", "")).encode("utf-8")
            body.extend(struct.pack("<H", len(path_bytes)))
            body.extend(path_bytes)
            return bytes(body)

        if method == "search_combat_mcts":
            body.append(OP_SEARCH_COMBAT_MCTS)
            body.extend(struct.pack("<H", int(params.get("num_simulations", 0))))
            body.extend(struct.pack("<f", float(params.get("c_puct", 1.5))))
            body.extend(struct.pack("<f", float(params.get("dirichlet_alpha", 0.0))))
            body.extend(struct.pack("<f", float(params.get("dirichlet_fraction", 0.0))))
            body.extend(struct.pack("<H", int(params.get("max_step_budget", 200))))
            mode = str(params.get("final_action_mode", "visit")).strip().lower()
            body.append(1 if mode == "visit_q_blend" else 0)
            body.extend(struct.pack("<H", int(params.get("final_action_top_k", 3))))
            body.extend(struct.pack("<f", float(params.get("final_action_q_weight", 0.35))))
            body.append(1 if bool(params.get("use_continuation_value", False)) else 0)
            body.append(1 if bool(params.get("debug_trace", False)) else 0)
            return bytes(body)

        raise ValueError(f"Unsupported proto pipe method: {method}")


# ======================================================================
# 模块级辅助函数（二进制读写）
# ======================================================================

def _read_string(data: bytes, off: int) -> tuple[str, int]:
    length = struct.unpack_from("<H", data, off)[0]
    off += 2
    value = data[off:off + length].decode("utf-8")
    return value, off + length


def _read_optional_string(data: bytes, off: int) -> tuple[str | None, int]:
    if data[off]:
        off += 1
        return _read_string(data, off)
    return None, off + 1


def _write_string(body: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    body.extend(struct.pack("<H", len(encoded)))
    body.extend(encoded)


def _write_optional_string(body: bytearray, value: Any) -> None:
    if value is None or str(value).strip() == "":
        body.append(0)
        return
    body.append(1)
    _write_string(body, str(value))


def _write_action(body: bytearray, action: dict[str, Any]) -> None:
    action_name = str(action.get("action") or action.get("type") or "other").strip().lower()
    body.append(ACTION_CODES.get(action_name, 255))
    body.extend(struct.pack("<h", _optional_short(action.get("index"))))
    body.extend(struct.pack("<h", _optional_short(action.get("card_index"))))
    body.extend(struct.pack("<h", _optional_short(action.get("target_id"))))
    body.extend(struct.pack("<b", _optional_sbyte(action.get("col"))))
    body.extend(struct.pack("<b", _optional_sbyte(action.get("row"))))
    body.extend(struct.pack("<b", _optional_sbyte(action.get("slot"))))


def _optional_short(value: Any) -> int:
    if value is None:
        return -1
    return max(min(int(value), 32767), -32768)


def _optional_sbyte(value: Any) -> int:
    if value is None:
        return -1
    return max(min(int(value), 127), -128)


def _terminal_reward(state: dict[str, Any]) -> float:
    if not bool(state.get("terminal")):
        return 0.0
    outcome = str(state.get("run_outcome") or "").strip().lower()
    if outcome in {"victory", "win"}:
        return 1.0
    if outcome in {"defeat", "loss", "death"}:
        return -1.0
    return 0.0
