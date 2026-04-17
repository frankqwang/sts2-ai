#!/usr/bin/env python3
"""Skada runs 抓取守护脚本。

职责：
- 监督 pages-only / details-only 两个子抓取任务
- 子任务退出后按本地指数退避等待后重启，避免长时间空等
- details 任务若仍在运行则不重复拉起；异常退出则自动重启
- 直到 pages 抓完且 details 数量补齐为止
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
RUNNER = THIS_DIR / "crawl_skada_runs_raw.py"
DEFAULT_OUT_DIR = THIS_DIR / "runs"
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "Artifacts" / "skada"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_page_files(out_dir: Path) -> int:
    return len(list((out_dir / "pages").glob("runs_page_*.json")))


def _load_existing_detail_ids(details_dir: Path) -> set[int]:
    done: set[int] = set()
    for path in sorted(details_dir.glob("run_details_*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    run = record.get("run") or {}
                    run_id = run.get("run_id")
                    if run_id is not None:
                        done.add(int(run_id))
        except Exception:
            continue
    return done


def _collect_run_ids_from_pages(pages_dir: Path) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for path in sorted(pages_dir.glob("runs_page_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            run_id = item.get("run_id")
            if run_id is None:
                continue
            run_id = int(run_id)
            if run_id in seen:
                continue
            seen.add(run_id)
            ordered.append(run_id)
    return ordered


@dataclass
class ChildSpec:
    name: str
    args: list[str]
    stdout_log: Path
    stderr_log: Path
    proc: subprocess.Popen[str] | None = None
    next_restart_ts: float = 0.0
    restart_streak: int = 0


def _start_child(spec: ChildSpec, cwd: Path) -> None:
    spec.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    spec.stderr_log.parent.mkdir(parents=True, exist_ok=True)
    stdout = spec.stdout_log.open("a", encoding="utf-8", buffering=1)
    stderr = spec.stderr_log.open("a", encoding="utf-8", buffering=1)
    cmd = [sys.executable, "-u", str(RUNNER), *spec.args]
    spec.proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    spec.next_restart_ts = 0.0
    print(f"[daemon] started {spec.name} pid={spec.proc.pid}", flush=True)


def _compute_progress(out_dir: Path) -> dict[str, Any]:
    manifest = _load_json(out_dir / "meta" / "runs_manifest.json") or {}
    total_pages = int(manifest.get("total_pages") or 0)
    pages_done = _count_page_files(out_dir)
    run_ids = _collect_run_ids_from_pages(out_dir / "pages")
    details_done = _load_existing_detail_ids(out_dir / "details")
    return {
        "total_pages": total_pages,
        "pages_done": pages_done,
        "total_run_ids": len(run_ids),
        "details_done": len(details_done),
        "pages_complete": bool(total_pages and pages_done >= total_pages),
        "details_complete": bool(run_ids and len(details_done) >= len(run_ids)),
    }


def _child_finished(spec: ChildSpec) -> bool:
    return spec.proc is not None and spec.proc.poll() is not None


def _maybe_schedule_restart(spec: ChildSpec, *, base_sleep: float, max_sleep: float) -> None:
    if spec.proc is None:
        return
    rc = spec.proc.poll()
    if rc is None:
        return
    spec.restart_streak += 1
    backoff_s = min(max(max_sleep, 1.0), max(base_sleep, 0.1) * (2 ** max(spec.restart_streak - 1, 0)))
    spec.next_restart_ts = time.time() + backoff_s
    wake_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(spec.next_restart_ts))
    print(
        f"[daemon] {spec.name} exited rc={rc}, restart_streak={spec.restart_streak} "
        f"backoff_s={backoff_s:.1f} restart_at={wake_at}",
        flush=True,
    )
    spec.proc = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skada runs 抓取守护脚本")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="原始数据输出目录")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="守护与子任务日志目录")
    parser.add_argument("--page-size", type=int, default=20, help="列表页 page_size，默认 20")
    parser.add_argument("--sort-by", type=str, default="run_id")
    parser.add_argument("--sort-dir", type=str, default="asc", choices=["asc", "desc"])
    parser.add_argument("--page-workers", type=int, default=8)
    parser.add_argument("--detail-workers", type=int, default=3)
    parser.add_argument("--page-qps", type=float, default=0.6)
    parser.add_argument("--detail-qps", type=float, default=0.6)
    parser.add_argument("--detail-shard-lines", type=int, default=2000)
    parser.add_argument("--rate-limit-base-sleep", type=float, default=2.0, help="子任务退出后的本地指数退避起始秒数")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=60.0, help="子任务退出后的本地指数退避上限秒数")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="守护轮询间隔")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    page_spec = ChildSpec(
        name="pages",
        args=[
            "--out-dir", str(out_dir),
            "--page-size", str(args.page_size),
            "--sort-by", args.sort_by,
            "--sort-dir", args.sort_dir,
            "--page-workers", str(args.page_workers),
            "--page-qps", str(args.page_qps),
            "--rate-limit-base-sleep", str(args.rate_limit_base_sleep),
            "--rate-limit-max-sleep", str(args.rate_limit_max_sleep),
            "--ignore-server-retry-after",
            "--pages-only",
        ],
        stdout_log=log_dir / "crawl_skada_daemon_pages_stdout.log",
        stderr_log=log_dir / "crawl_skada_daemon_pages_stderr.log",
    )
    detail_spec = ChildSpec(
        name="details",
        args=[
            "--out-dir", str(out_dir),
            "--details-only",
            "--follow-pages",
            "--poll-seconds", "10",
            "--detail-workers", str(args.detail_workers),
            "--detail-qps", str(args.detail_qps),
            "--detail-shard-lines", str(args.detail_shard_lines),
            "--rate-limit-base-sleep", str(args.rate_limit_base_sleep),
            "--rate-limit-max-sleep", str(args.rate_limit_max_sleep),
            "--ignore-server-retry-after",
        ],
        stdout_log=log_dir / "crawl_skada_daemon_details_stdout.log",
        stderr_log=log_dir / "crawl_skada_daemon_details_stderr.log",
    )

    print(
        json.dumps(
            {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "out_dir": str(out_dir),
                "log_dir": str(log_dir),
                "page_size": args.page_size,
                "sort_by": args.sort_by,
                "sort_dir": args.sort_dir,
                "page_workers": args.page_workers,
                "detail_workers": args.detail_workers,
                "page_qps": args.page_qps,
                "detail_qps": args.detail_qps,
                "rate_limit_base_sleep": args.rate_limit_base_sleep,
                "rate_limit_max_sleep": args.rate_limit_max_sleep,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    _start_child(page_spec, cwd=out_dir.parents[3])
    _start_child(detail_spec, cwd=out_dir.parents[3])

    try:
        while True:
            progress = _compute_progress(out_dir)
            print(
                f"[daemon] pages={progress['pages_done']}/{progress['total_pages']} "
                f"details={progress['details_done']}/{progress['total_run_ids']}",
                flush=True,
            )

            if progress["pages_complete"] and progress["details_complete"]:
                print("[daemon] crawl complete", flush=True)
                break

            for spec in (page_spec, detail_spec):
                if _child_finished(spec):
                    _maybe_schedule_restart(
                        spec,
                        base_sleep=args.rate_limit_base_sleep,
                        max_sleep=args.rate_limit_max_sleep,
                    )

            now = time.time()
            if page_spec.proc is None and not progress["pages_complete"] and now >= page_spec.next_restart_ts:
                _start_child(page_spec, cwd=out_dir.parents[3])
            if detail_spec.proc is None and not progress["details_complete"] and now >= detail_spec.next_restart_ts:
                _start_child(detail_spec, cwd=out_dir.parents[3])

            time.sleep(max(args.poll_seconds, 5.0))
    finally:
        for spec in (page_spec, detail_spec):
            if spec.proc is not None and spec.proc.poll() is None:
                spec.proc.terminate()


if __name__ == "__main__":
    main()
