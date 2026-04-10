from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    python_root = Path(__file__).resolve().parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

from binary_pipe_client import BinaryPipeClient
from pipe_client import PipeClient
from sts2ai_paths import REPO_ROOT, SIM_HOST_EXE, SIM_LEGACY_DLL


DEFAULT_REPO_ROOT = REPO_ROOT
DEFAULT_DLL_PATH = SIM_HOST_EXE if SIM_HOST_EXE.exists() else SIM_LEGACY_DLL


def _build_launch_command(host_path: Path, protocol: str, port: int) -> list[str]:
    normalized_protocol = "bin" if protocol in {"bin", "binary"} else "json"
    host_args = ["--port", str(port), "--protocol", normalized_protocol]
    if host_path.suffix.lower() == ".dll":
        return ["dotnet", str(host_path), *host_args]
    return [str(host_path), *host_args]


def start_headless_sim(
    *,
    port: int,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
    dll_path: str | Path = DEFAULT_DLL_PATH,
    connect_timeout_s: float = 15.0,
    protocol: str = "json",
) -> subprocess.Popen:
    repo_root = Path(repo_root)
    dll_path = Path(dll_path)
    protocol = str(protocol).strip().lower()
    launch_cmd = _build_launch_command(dll_path, protocol, port)
    proc = subprocess.Popen(
        launch_cmd,
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    _wait_until_ready(port=port, timeout_s=connect_timeout_s, protocol=protocol)
    return proc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def _wait_until_ready(*, port: int, timeout_s: float, protocol: str = "json") -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    protocol = str(protocol).strip().lower()
    while time.monotonic() < deadline:
        try:
            client = BinaryPipeClient(port=port) if protocol in {"bin", "binary"} else PipeClient(port=port)
            client.connect(timeout_s=1.0)
            client.close()
            # The standalone host allows only one active pipe owner at a time.
            # A single successful handshake is sufficient to prove readiness;
            # avoid an immediate second connect so benchmarks and training
            # workers do not race the launcher for ownership.
            time.sleep(0.25)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"HeadlessSim on port {port} did not become ready within {timeout_s:.1f}s: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch standalone HeadlessSim and wait for pipe readiness.")
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--dll-path", type=Path, default=DEFAULT_DLL_PATH)
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    parser.add_argument("--protocol", choices=["json", "bin"], default="json")
    args = parser.parse_args()

    proc = start_headless_sim(
        port=args.port,
        repo_root=args.repo_root,
        dll_path=args.dll_path,
        connect_timeout_s=args.ready_timeout,
        protocol=args.protocol,
    )
    print(f"HeadlessSim ready on port {args.port} (pid={proc.pid})")
    try:
        proc.wait()
    except KeyboardInterrupt:
        stop_process(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
