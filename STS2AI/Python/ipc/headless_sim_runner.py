from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    python_root = Path(__file__).resolve().parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

from binary_pipe_client import BinaryPipeClient
from pipe_client import PipeClient
from sts2ai_paths import REPO_ROOT, SIM_HOST_EXE, SIM_LEGACY_DLL


DEFAULT_REPO_ROOT = REPO_ROOT
DEFAULT_DLL_PATH = SIM_HOST_EXE if SIM_HOST_EXE.exists() else SIM_LEGACY_DLL


def _source_roots(repo_root: Path) -> Iterable[Path]:
    candidates = (
        repo_root / "src",
        repo_root / "STS2AI" / "ENV" / "Sim" / "Overlay",
        repo_root / "STS2AI" / "ENV" / "Sim" / "Runtime",
    )
    for root in candidates:
        if root.exists():
            yield root


def _newest_source_file(repo_root: Path) -> tuple[Path | None, float]:
    newest_path: Path | None = None
    newest_mtime = 0.0
    for root in _source_roots(repo_root):
        for path in root.rglob("*.cs"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_path = path
    return newest_path, newest_mtime


def ensure_host_binary_is_fresh(*, repo_root: Path, dll_path: Path) -> None:
    if not dll_path.exists():
        raise FileNotFoundError(f"HeadlessSim host binary does not exist: {dll_path}")
    newest_source, newest_source_mtime = _newest_source_file(repo_root)
    if newest_source is None:
        return
    try:
        host_mtime = dll_path.stat().st_mtime
    except OSError as exc:
        raise RuntimeError(f"Unable to stat HeadlessSim host binary: {dll_path}") from exc
    # Small tolerance to avoid false positives on coarse timestamp resolutions.
    if host_mtime + 1.0 < newest_source_mtime:
        raise RuntimeError(
            "HeadlessSim host binary is stale: "
            f"{dll_path} is older than source {newest_source}. "
            "Rebuild STS2AI/ENV/Sim/Host/headless_sim_host_0991.csproj before auto-launch."
        )


def _kill_stale_headless_processes(*, port: int, dll_path: Path) -> None:
    if sys.platform != "win32":
        return
    resolved = dll_path.resolve()
    escaped_path = str(resolved).replace("'", "''")
    escaped_name = resolved.name.replace("'", "''")
    port_arg = f"--port {port}"
    command = (
        f"$targetPath = '{escaped_path}'; "
        f"$targetName = '{escaped_name}'; "
        f"$portArg = '{port_arg}'; "
        "$procs = Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -and $_.CommandLine -like ('*' + $portArg + '*') -and ("
        "($_.ExecutablePath -eq $targetPath) -or "
        "($_.Name -eq 'dotnet.exe' -and $_.CommandLine -like ('*' + $targetName + '*'))"
        ") }; "
        "foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass


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
    ensure_host_binary_is_fresh(repo_root=repo_root, dll_path=dll_path)
    _kill_stale_headless_processes(port=port, dll_path=dll_path)
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
