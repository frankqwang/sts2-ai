"""底层 Win32 Named Pipe 字节 IO。

职责:**只**提供 connect / read / write / close 的字节流接口。
**不做**:协议解析、JSON/proto、重连、heartbeat、锁、错误分类。
(上面这些都在 `connection.py` 的 PipeConnection 做)。

设计准则:
  - Stateless: 只 handle pipe handle 本身,不 cache 业务状态
  - Win32 overlapped I/O: 支持正确的读超时
  - 线程不安全:调用方必须串行化 (PipeConnection 内置 RLock 负责)
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import struct
import sys
import time


class TransportClosedError(RuntimeError):
    """pipe 未连接或已关闭时的 IO 尝试。"""


class TransportTimeoutError(TimeoutError):
    """pipe read/write 超时。"""


# Windows constants
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_OPEN_EXISTING = 3
_FILE_FLAG_OVERLAPPED = 0x40000000
_INVALID_HANDLE_VALUE = -1
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 0x102
_ERROR_IO_PENDING = 997


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.POINTER(ctypes.c_ulong)),
        ("InternalHigh", ctypes.POINTER(ctypes.c_ulong)),
        ("Offset", ctypes.wintypes.DWORD),
        ("OffsetHigh", ctypes.wintypes.DWORD),
        ("hEvent", ctypes.wintypes.HANDLE),
    ]


_kernel32 = ctypes.windll.kernel32 if sys.platform == "win32" else None


class PipeTransport:
    """Windows Named Pipe 字节流(opened via CreateFile + overlapped I/O)。

    不保留 protocol 知识 — 上层协议 (JSON / proto) 在 PipeConnection 里。
    """

    def __init__(self, pipe_name: str):
        self.pipe_name = pipe_name
        self._handle = None
        self._event = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, timeout_s: float = 10.0) -> None:
        """打开 pipe,等待 server 可连接。"""
        if sys.platform != "win32":
            raise RuntimeError("Named pipes only supported on Windows")

        pipe_path = f"\\\\.\\pipe\\{self.pipe_name}"
        deadline = time.monotonic() + timeout_s
        last_err = None

        while time.monotonic() < deadline:
            if not _kernel32.WaitNamedPipeW(pipe_path, 200):
                last_err = f"Pipe {pipe_path} not ready"
                time.sleep(0.1)
                continue

            handle = _kernel32.CreateFileW(
                pipe_path,
                _GENERIC_READ | _GENERIC_WRITE,
                0, None, _OPEN_EXISTING, _FILE_FLAG_OVERLAPPED, None,
            )
            if handle == _INVALID_HANDLE_VALUE:
                err = ctypes.GetLastError()
                last_err = f"CreateFileW failed: winerror={err}"
                time.sleep(0.1)
                continue

            self._handle = handle
            self._event = _kernel32.CreateEventW(None, True, False, None)
            return
        raise ConnectionError(f"Failed to connect after {timeout_s}s: {last_err}")

    def close(self) -> None:
        if self._event is not None:
            _kernel32.CloseHandle(self._event)
            self._event = None
        if self._handle is not None:
            _kernel32.CloseHandle(self._handle)
            self._handle = None

    def is_connected(self) -> bool:
        return self._handle is not None

    # ------------------------------------------------------------------
    # 字节 IO
    # ------------------------------------------------------------------

    def write_bytes(self, data: bytes, timeout_ms: int = 10000) -> None:
        if self._handle is None:
            raise TransportClosedError(f"Not connected to {self.pipe_name}")
        ovl = _OVERLAPPED()
        ovl.hEvent = self._event
        _kernel32.ResetEvent(self._event)

        written = ctypes.wintypes.DWORD(0)
        ok = _kernel32.WriteFile(
            self._handle, data, len(data),
            ctypes.byref(written), ctypes.byref(ovl),
        )
        if not ok:
            err = ctypes.GetLastError()
            if err == _ERROR_IO_PENDING:
                _kernel32.WaitForSingleObject(self._event, timeout_ms)
                _kernel32.GetOverlappedResult(
                    self._handle, ctypes.byref(ovl), ctypes.byref(written), False,
                )
            else:
                raise ConnectionError(f"WriteFile failed: winerror={err}")

    def read_bytes(self, n: int, timeout_ms: int = 30000) -> bytes:
        """精确读取 n 字节。超时 → TransportTimeoutError;pipe 断 → ConnectionError。"""
        if self._handle is None:
            raise TransportClosedError(f"Not connected to {self.pipe_name}")

        buf = ctypes.create_string_buffer(n)
        total_read = 0

        while total_read < n:
            ovl = _OVERLAPPED()
            ovl.hEvent = self._event
            _kernel32.ResetEvent(self._event)

            bytes_read = ctypes.wintypes.DWORD(0)
            remaining = n - total_read
            ok = _kernel32.ReadFile(
                self._handle,
                ctypes.cast(ctypes.addressof(buf) + total_read, ctypes.c_void_p),
                remaining,
                ctypes.byref(bytes_read),
                ctypes.byref(ovl),
            )
            if ok:
                total_read += bytes_read.value
                if bytes_read.value == 0:
                    raise ConnectionError("Pipe closed by server (0-byte read)")
                continue

            err = ctypes.GetLastError()
            if err != _ERROR_IO_PENDING:
                raise ConnectionError(f"ReadFile failed: winerror={err}")

            wait_result = _kernel32.WaitForSingleObject(self._event, timeout_ms)
            if wait_result == _WAIT_TIMEOUT:
                _kernel32.CancelIo(self._handle)
                raise TransportTimeoutError(
                    f"Pipe read timed out after {timeout_ms}ms "
                    f"(read {total_read}/{n} bytes)"
                )
            if wait_result != _WAIT_OBJECT_0:
                _kernel32.CancelIo(self._handle)
                raise ConnectionError(f"WaitForSingleObject: {wait_result}")

            _kernel32.GetOverlappedResult(
                self._handle, ctypes.byref(ovl), ctypes.byref(bytes_read), False,
            )
            if bytes_read.value == 0:
                raise ConnectionError("Pipe closed by server (0-byte overlapped)")
            total_read += bytes_read.value

        return buf.raw[:n]

    # ------------------------------------------------------------------
    # Length-prefixed frame helpers
    # ------------------------------------------------------------------

    def write_frame(self, payload: bytes, timeout_ms: int = 10000) -> None:
        """4 字节小端长度前缀 + payload。"""
        self.write_bytes(struct.pack("<I", len(payload)) + payload, timeout_ms=timeout_ms)

    def read_frame(self, timeout_ms: int = 30000, max_size: int = 10_000_000) -> bytes:
        """读一帧:先 4 字节长度,再 payload。"""
        hdr = self.read_bytes(4, timeout_ms=timeout_ms)
        msg_len = struct.unpack("<I", hdr)[0]
        if msg_len > max_size:
            raise RuntimeError(f"Frame too large: {msg_len} bytes")
        return self.read_bytes(msg_len, timeout_ms=timeout_ms)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
