"""Named pipe client for high-speed IPC with STS2 simulator.

~50x faster than HTTP for small JSON messages (no TCP handshake/headers).
Protocol: 4-byte little-endian length prefix + UTF-8 JSON payload.

Uses Windows overlapped I/O for reads with proper timeout support.
Previous implementation used os.fdopen().read() which holds the GIL
and cannot be interrupted by threading timeout.

Usage:
    from pipe_client import PipeClient

    pipe = PipeClient(port=15527)
    pipe.connect()
    result = pipe.call("reset", {"seed": "TEST1", "character_id": "IRONCLAD"})
    result = pipe.call("step", {"action": "choose_map_node", "index": 0})
    state_id = pipe.call("save_state")["state_id"]
    pipe.call("load_state", {"state_id": state_id})
    pipe.close()
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
import struct
import sys
import time
from typing import Any

try:
    from .simulator_api_error import SimulatorApiError
except ImportError:
    from simulator_api_error import SimulatorApiError


# Windows constants
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_FLAG_OVERLAPPED = 0x40000000
INVALID_HANDLE_VALUE = -1
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102
ERROR_IO_PENDING = 997


class OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.POINTER(ctypes.c_ulong)),
        ("InternalHigh", ctypes.POINTER(ctypes.c_ulong)),
        ("Offset", ctypes.wintypes.DWORD),
        ("OffsetHigh", ctypes.wintypes.DWORD),
        ("hEvent", ctypes.wintypes.HANDLE),
    ]


_kernel32 = ctypes.windll.kernel32 if sys.platform == "win32" else None


class PipeClient:
    """Named pipe client using Windows overlapped I/O for timeout support."""

    def __init__(self, port: int = 15527, pipe_name: str | None = None,
                 default_timeout_s: float = 30.0):
        self.pipe_name = pipe_name or f"sts2_mcts_{port}"
        self.default_timeout_s = default_timeout_s
        self._handle = None
        self._event = None

    def connect(self, timeout_s: float = 10.0) -> None:
        """Connect to the named pipe. Retries until timeout."""
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
                    FILE_FLAG_OVERLAPPED,  # Enable overlapped I/O
                    None,
                )
                if handle == INVALID_HANDLE_VALUE:
                    err = ctypes.GetLastError()
                    last_err = f"CreateFileW failed: winerror={err}"
                    time.sleep(0.1)
                    continue

                self._handle = handle
                self._event = _kernel32.CreateEventW(None, True, False, None)

                # Read handshake message from server
                hello = self._read_message(timeout_s=timeout_s)
                error = hello.get("error")
                if error:
                    self.close()
                    raise SimulatorApiError(
                        error,
                        error_code=hello.get("error_code"),
                    )
                if not hello.get("ok"):
                    self.close()
                    raise ConnectionError(f"Unexpected handshake: {hello!r}")
                return
            except (SimulatorApiError, ConnectionError):
                raise
            except OSError as exc:
                last_err = f"Pipe open failed: {exc}"
                time.sleep(0.1)

        raise ConnectionError(f"Failed to connect after {timeout_s}s: {last_err}")

    def close(self) -> None:
        """Close the pipe connection."""
        if self._event is not None:
            _kernel32.CloseHandle(self._event)
            self._event = None
        if self._handle is not None:
            _kernel32.CloseHandle(self._handle)
            self._handle = None

    def is_connected(self) -> bool:
        return self._handle is not None

    def call(self, method: str, params: dict[str, Any] | None = None,
             timeout_s: float | None = None) -> dict:
        """Send a request and return the response.

        Args:
            method: API method name
            params: Optional parameters dict
            timeout_s: Max seconds to wait (None = use default_timeout_s)

        Returns:
            Parsed JSON response dict

        Raises:
            TimeoutError: if response not received within timeout
            ConnectionError: if pipe is broken
            SimulatorApiError: if server returned an error
        """
        if self._handle is None:
            raise ConnectionError("Not connected. Call connect() first.")

        if timeout_s is None:
            timeout_s = self.default_timeout_s

        request = {"method": method}
        if params:
            request["params"] = params

        payload = json.dumps(request).encode("utf-8")

        # Send: 4-byte length + payload
        self._write_bytes(struct.pack("<I", len(payload)) + payload)

        # Read response with timeout
        result = self._read_message(timeout_s=timeout_s)

        if isinstance(result, dict) and "error" in result and result["error"]:
            raise SimulatorApiError(
                result["error"],
                error_code=result.get("error_code"),
            )

        return result

    def _write_bytes(self, data: bytes) -> None:
        """Write bytes to pipe (synchronous, writes are fast)."""
        ovl = OVERLAPPED()
        ovl.hEvent = self._event
        _kernel32.ResetEvent(self._event)

        written = ctypes.wintypes.DWORD(0)
        ok = _kernel32.WriteFile(
            self._handle,
            data,
            len(data),
            ctypes.byref(written),
            ctypes.byref(ovl),
        )
        if not ok:
            err = ctypes.GetLastError()
            if err == ERROR_IO_PENDING:
                _kernel32.WaitForSingleObject(self._event, 10000)  # 10s write timeout
                _kernel32.GetOverlappedResult(
                    self._handle, ctypes.byref(ovl), ctypes.byref(written), False)
            else:
                raise ConnectionError(f"WriteFile failed: winerror={err}")

    def _read_bytes(self, n: int, timeout_ms: int) -> bytes:
        """Read exactly n bytes with timeout using overlapped I/O."""
        buf = ctypes.create_string_buffer(n)
        total_read = 0

        while total_read < n:
            ovl = OVERLAPPED()
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
                # Completed synchronously
                total_read += bytes_read.value
                if bytes_read.value == 0:
                    raise ConnectionError("Pipe closed by server")
                continue

            err = ctypes.GetLastError()
            if err != ERROR_IO_PENDING:
                raise ConnectionError(f"ReadFile failed: winerror={err}")

            # Wait for overlapped read with timeout
            wait_result = _kernel32.WaitForSingleObject(self._event, timeout_ms)

            if wait_result == WAIT_TIMEOUT:
                _kernel32.CancelIo(self._handle)
                raise TimeoutError(
                    f"Pipe read timed out after {timeout_ms}ms "
                    f"(read {total_read}/{n} bytes)")

            if wait_result != WAIT_OBJECT_0:
                _kernel32.CancelIo(self._handle)
                raise ConnectionError(f"WaitForSingleObject failed: {wait_result}")

            # Get actual bytes read
            _kernel32.GetOverlappedResult(
                self._handle, ctypes.byref(ovl), ctypes.byref(bytes_read), False)
            if bytes_read.value == 0:
                raise ConnectionError("Pipe closed by server")
            total_read += bytes_read.value

        return buf.raw[:n]

    def _read_message(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Read one length-prefixed JSON message."""
        timeout_ms = int(timeout_s * 1000)

        # Read 4-byte length header
        len_buf = self._read_bytes(4, timeout_ms)
        msg_len = struct.unpack("<I", len_buf)[0]

        if msg_len > 10_000_000:  # 10MB safety limit
            raise RuntimeError(f"Response too large: {msg_len} bytes")

        # Read message body
        msg_buf = self._read_bytes(msg_len, timeout_ms)
        return json.loads(msg_buf.decode("utf-8"))

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


class PipeBackedMctsEnv:
    """MCTS environment using named pipe for high-speed IPC."""

    def __init__(self, port: int = 15527):
        self.pipe = PipeClient(port=port)
        self.pipe.connect()
        self._state: dict | None = None

    def reset(self, character_id: str = "IRONCLAD",
              ascension_level: int = 0,
              seed: str | None = None) -> dict:
        params: dict[str, Any] = {
            "character_id": character_id,
            "ascension_level": ascension_level,
        }
        if seed:
            params["seed"] = seed
        self._state = self.pipe.call("reset", params)
        return self._state

    def step(self, action: dict) -> dict:
        result = self.pipe.call("step", action)
        if "state" in result and isinstance(result["state"], dict):
            self._state = result["state"]
        else:
            self._state = result
        return self._state

    def get_state(self) -> dict:
        self._state = self.pipe.call("state")
        return self._state

    def save(self) -> str:
        result = self.pipe.call("save_state")
        return result["state_id"]

    def load(self, state_id: str) -> dict:
        self._state = self.pipe.call("load_state", {"state_id": state_id})
        return self._state

    def delete(self, state_id: str) -> bool:
        result = self.pipe.call("delete_state", {"state_id": state_id})
        return result.get("deleted", False)

    def clear_cache(self) -> None:
        self.pipe.call("delete_state", {"clear_all": True})

    def legal_actions(self) -> list[dict]:
        result = self.pipe.call("legal_actions")
        actions = result.get("legal_actions") or []
        return [a for a in actions
                if isinstance(a, dict) and a.get("is_enabled") is not False]

    def batch_step(self, actions: list[dict]) -> dict:
        return self.pipe.call("batch_step", {"actions": actions})

    @property
    def state_type(self) -> str:
        if self._state is None:
            return ""
        return (self._state.get("state_type") or "").lower()

    @property
    def is_terminal(self) -> bool:
        if self._state is None:
            return False
        return bool(self._state.get("terminal"))

    def close(self):
        self.pipe.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--stress", type=int, default=0,
                        help="Run N random steps as stress test")
    args = parser.parse_args()

    pipe = PipeClient(port=args.port)
    pipe.connect(timeout_s=10)
    result = pipe.call("state")
    print(f"Connected! state_type={result.get('state_type')}")

    if args.stress > 0:
        import random
        state = pipe.call("reset", {"character_id": "IRONCLAD"})
        t0 = time.monotonic()
        steps = 0
        errors = 0
        for i in range(args.stress):
            st = (state.get("state_type") or "").lower()
            if st == "game_over" or state.get("terminal"):
                state = pipe.call("reset", {"character_id": "IRONCLAD"})
                continue
            legal = [a for a in state.get("legal_actions", [])
                     if isinstance(a, dict) and a.get("is_enabled") is not False]
            if not legal:
                try:
                    result = pipe.call("step", {"action": "wait"})
                except Exception:
                    state = pipe.call("state")
                    continue
            else:
                action = random.choice(legal)
                clean = {k: v for k, v in action.items()
                         if k in ("action", "index", "card_index", "hand_index",
                                  "slot", "target_id", "target", "value")}
                try:
                    result = pipe.call("step", clean)
                except SimulatorApiError:
                    errors += 1
                    state = pipe.call("state")
                    continue
            if isinstance(result, dict):
                if "state" in result and isinstance(result["state"], dict):
                    state = result["state"]
                elif "state_type" in result:
                    state = result
            steps += 1
        elapsed = time.monotonic() - t0
        print(f"{steps} steps, {errors} errors in {elapsed:.1f}s "
              f"({elapsed / max(1, steps) * 1000:.1f}ms/step)")

    pipe.close()
