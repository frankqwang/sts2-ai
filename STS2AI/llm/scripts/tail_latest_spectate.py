"""自动找 `Artifacts/llm/spectate_llm/` 最新一次 run，tail 它的 step_trace.jsonl。

用法：

    python llm\\scripts\\tail_latest_spectate.py [--full] [--raw-only]

默认盯 `spectate_llm`；想看启发式 run 加 `--heuristic`（但启发式 run 不落 trace）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--heuristic", action="store_true",
                   help="盯 spectate_heuristic 目录（默认盯 spectate_llm）")
    p.add_argument("--full", action="store_true", help="展开完整 user message")
    p.add_argument("--raw-only", action="store_true", help="只打 LLM 原始输出")
    return p.parse_args()


def find_latest_trace(root: Path, wait_seconds: int = 120) -> Path:
    """在指定目录下找最新 step_trace.jsonl；没有就等。"""
    deadline = time.time() + wait_seconds
    last_logged = ""
    while time.time() < deadline:
        if root.exists():
            runs = sorted(
                [p for p in root.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for run in runs:
                trace = run / "step_trace.jsonl"
                if trace.exists():
                    return trace
        msg = f"waiting for run under {root} ..."
        if msg != last_logged:
            print(msg)
            last_logged = msg
        time.sleep(1.5)
    raise SystemExit(f"timed out waiting for trace under {root}")


def main() -> int:
    args = parse_args()
    sts2ai_root = Path(__file__).resolve().parents[2]
    kind = "spectate_heuristic" if args.heuristic else "spectate_llm"
    root = sts2ai_root / "Artifacts" / "llm" / kind
    trace = find_latest_trace(root)
    print(f"tailing: {trace}")

    replay_script = Path(__file__).parent / "replay_trace.py"
    cmd = [sys.executable, str(replay_script), "--trace", str(trace), "--follow"]
    if args.full:
        cmd.append("--full")
    if args.raw_only:
        cmd.append("--raw-only")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
