from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    python_root = Path(__file__).resolve().parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

from ipc.binary_pipe_client import BinaryPipeClient
from ipc.pipe_client import PipeClient
from constants import REPO_ROOT, SIM_HOST_EXE, SIM_LEGACY_DLL


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


def _host_project_path(repo_root: Path) -> Path:
    return repo_root / "STS2AI" / "ENV" / "Sim" / "Host" / "headless_sim_host_0991.csproj"


def _resolve_msbuild_path(raw_value: str, *, repo_root: Path, host_project: Path) -> Path | None:
    value = raw_value.strip()
    if not value:
        return None
    value = value.replace("$(MSBuildThisFileDirectory)", str(host_project.parent) + os.sep)
    if "$(" in value:
        return None
    expanded = Path(os.path.expandvars(value))
    if expanded.is_absolute():
        return expanded
    return (host_project.parent / expanded).resolve()


def _host_upstream_root(repo_root: Path) -> Path | None:
    host_project = _host_project_path(repo_root)
    if not host_project.exists():
        return None
    try:
        root = ET.parse(host_project).getroot()
    except ET.ParseError:
        return None
    for prop_group in root.findall("PropertyGroup"):
        upstream_node = prop_group.find("UpstreamRoot")
        if upstream_node is None or upstream_node.text is None:
            continue
        resolved = _resolve_msbuild_path(upstream_node.text, repo_root=repo_root, host_project=host_project)
        if resolved is not None and resolved.exists():
            return resolved
    return None


def _newest_source_file(repo_root: Path) -> tuple[Path | None, float]:
    newest_path: Path | None = None
    newest_mtime = 0.0
    candidate_roots = list(_source_roots(repo_root))
    upstream_root = _host_upstream_root(repo_root)
    if upstream_root is not None:
        candidate_roots.extend(
            path for path in (upstream_root / "src", upstream_root / "System", upstream_root / "Properties") if path.exists()
        )
    for root in candidate_roots:
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
    # Windows + dotnet build can leave source and output mtimes within a couple
    # of seconds of each other even when the rebuild succeeded. Keep a slightly
    # larger tolerance so the freshness guard catches real stale hosts without
    # tripping immediately after a successful local build.
    if host_mtime + 3.0 < newest_source_mtime:
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


def _pipe_name_for_port(*, port: int, protocol: str) -> str:
    normalized_protocol = "bin" if protocol in {"bin", "binary"} else "json"
    pipe_suffix = f"sts2_mcts_bin_{port}" if normalized_protocol == "bin" else f"sts2_mcts_{port}"
    return rf"\\.\pipe\{pipe_suffix}"


def _wait_for_pipe_slot_windows(*, port: int, timeout_s: float, protocol: str) -> None:
    pipe_name = _pipe_name_for_port(port=port, protocol=protocol)
    kernel32 = ctypes.windll.kernel32
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        ok = kernel32.WaitNamedPipeW(pipe_name, remaining_ms)
        if ok:
            # Avoid handing the same single-owner session from the probe to the
            # training client in the same scheduler tick.
            time.sleep(0.15)
            return
        time.sleep(0.05)
    raise RuntimeError(f"HeadlessSim pipe {pipe_name} did not become ready within {timeout_s:.1f}s")


def start_headless_sim(
    *,
    port: int,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
    dll_path: str | Path = DEFAULT_DLL_PATH,
    connect_timeout_s: float = 15.0,
    protocol: str = "json",
    extra_host_args: Iterable[str] | None = None,
) -> subprocess.Popen:
    repo_root = Path(repo_root)
    dll_path = Path(dll_path)
    protocol = str(protocol).strip().lower()
    ensure_host_binary_is_fresh(repo_root=repo_root, dll_path=dll_path)
    _kill_stale_headless_processes(port=port, dll_path=dll_path)
    launch_cmd = _build_launch_command(dll_path, protocol, port)
    if extra_host_args:
        launch_cmd.extend(str(arg) for arg in extra_host_args)
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
    if sys.platform == "win32":
        try:
            _wait_for_pipe_slot_windows(port=port, timeout_s=timeout_s, protocol=protocol)
            return
        except Exception as exc:
            last_error = exc
        else:
            last_error = None
    else:
        last_error = None

    deadline = time.monotonic() + timeout_s
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
