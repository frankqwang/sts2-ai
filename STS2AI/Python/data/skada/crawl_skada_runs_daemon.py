#!/usr/bin/env python3
"""Skada runs 抓取守护脚本。

职责：
- 监督 pages-only / details-only 两个子抓取任务
- 子任务退出后按本地指数退避等待后重启，避免长时间空等
- details 任务若仍在运行则不重复拉起；异常退出则自动重启
- 直到 pages 抓完且 details 数量补齐为止

---- 常用命令速查（以后直接 copy 改参数即可）----

前置：
- 需要 set 快代理 TPS 账密（或 DPS）环境变量
- Clash 不需要（TPS 是国内隧道，直连即可）

# ① 增量爬 victory（只拉最近 3 天新增的）
#   sort=created_at desc + stop-on-known，daemon 自动在扫到全已知 run_id 的页时退出
cd STS2AI/Python/data/skada
KDL_TPS_HOST=sXXX.kdltps.com KDL_TPS_PORT=15818 \
KDL_TPS_USERNAME=... KDL_TPS_PASSWORD=... KDL_TPS_BACKUP_HOST=sXXX.kdltps.com \
python crawl_skada_runs_daemon.py \
  --out-dir runs_victory \
  --use-proxy-pool --proxy-product tps \
  --outcome victory \
  --sort-by created_at --sort-dir desc \
  --stop-on-known \
  --page-workers 2 --page-qps 1 \
  --detail-workers 12 --detail-qps 9 \
  --detail-shard-lines 2000 --stall-kill-seconds 180

# ② 增量爬 failure（有意义的失败 run，本地 filter）
KDL_TPS_HOST=... KDL_TPS_USERNAME=... KDL_TPS_PASSWORD=... \
python crawl_skada_runs_daemon.py \
  --out-dir runs_failure \
  --use-proxy-pool --proxy-product tps \
  --outcome defeat --filter-preset failure \
  --sort-by created_at --sort-dir desc \
  --stop-on-known \
  --page-workers 2 --page-qps 1 \
  --detail-workers 12 --detail-qps 9 \
  --detail-shard-lines 2000 --stall-kill-seconds 180

# ③ 首次全量爬（sort_by run_id asc，配 --target-detail-count 或让其跑完）
#   和 ①② 一样，改 --sort-by run_id --sort-dir asc，去掉 --stop-on-known
#   可选 --target-detail-count 50000 抓到目标数量就停

# ④ 爬完后提取新结构（map_acts/final_deck 齐全的 run）到 runs_full_detail
python build_runs_full_detail.py

---- 注意 ----
- sts2log detail 窗口只有 3 天；3 天前创建的 run 即使抓详情也只返回 detail_expired=true，没 map_acts
- 所以 victory/failure 里"完整详情"只存在于最近 3 天的 run，用 desc + stop-on-known 最精准
- sts2log list API 不支持 date filter 参数（date_from/since/... 全被忽略）
- 精确 version filter 支持（`version=v0.103.2` → total 只返回该版本），但不支持版本范围
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
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "derived" / "crawl_logs"
NON_RETRIABLE_EXIT_CODES = {2}

_FAILURE_MIN_VERSION_DAEMON = (0, 99, 0)


def _parse_version_daemon(v: str | None) -> tuple[int, ...]:
    if not v:
        return (0,)
    s = v.lstrip("vV")
    out: list[int] = []
    for p in s.split("."):
        try:
            out.append(int(p))
        except ValueError:
            out.append(0)
    return tuple(out) if out else (0,)


def _filter_failure_daemon(r: dict[str, Any]) -> bool:
    if r.get("is_victory") is True:
        return False
    if r.get("abandoned") is True:
        return False
    if not r.get("death_cause"):
        return False
    try:
        dur = float(r.get("duration_sec") or 0)
    except (TypeError, ValueError):
        dur = 0.0
    if dur < 180:
        return False
    try:
        floor = int(r.get("floor_reached") or 0)
        asc = int(r.get("ascension") or 0)
    except (TypeError, ValueError):
        return False
    if floor < 5:
        return False
    if asc < 10 and floor < 8:
        return False
    if asc >= 10 and floor < 12:
        return False
    if _parse_version_daemon(r.get("game_version")) < _FAILURE_MIN_VERSION_DAEMON:
        return False
    return True


FILTER_PRESETS_DAEMON = {
    "failure": _filter_failure_daemon,
}


def _make_failure_filter_daemon(min_created_at: str | None = None) -> Any:
    base = _filter_failure_daemon
    if not min_created_at:
        return base
    cutoff = min_created_at

    def filter_fn(r: dict[str, Any]) -> bool:
        if not base(r):
            return False
        ca = r.get("created_at") or ""
        if ca < cutoff:
            return False
        return True

    return filter_fn


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


def _collect_run_ids_from_pages(pages_dir: Path, filter_fn: Any = None) -> list[int]:
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
            if filter_fn is not None and not filter_fn(item):
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
    last_progress_value: int = -1
    last_progress_ts: float = 0.0


def _start_child(spec: ChildSpec, cwd: Path) -> None:
    spec.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    spec.stderr_log.parent.mkdir(parents=True, exist_ok=True)
    stdout = spec.stdout_log.open("a", encoding="utf-8", buffering=1)
    stderr = spec.stderr_log.open("a", encoding="utf-8", buffering=1)
    try:
        cmd = [sys.executable, "-u", str(RUNNER), *spec.args]
        spec.proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    finally:
        stdout.close()
        stderr.close()
    spec.next_restart_ts = 0.0
    print(f"[daemon] started {spec.name} pid={spec.proc.pid}", flush=True)


def _load_permanently_failed_ids(errors_path: Path) -> set[int]:
    done: set[int] = set()
    if not errors_path.exists():
        return done
    try:
        with errors_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("run_id") is not None:
                        done.add(int(rec["run_id"]))
                except Exception:
                    continue
    except Exception:
        pass
    return done


def _compute_progress(
    out_dir: Path,
    filter_fn: Any = None,
    target_detail_count: int | None = None,
    stop_on_known: bool = False,
) -> dict[str, Any]:
    manifest = _load_json(out_dir / "meta" / "runs_manifest.json") or {}
    total_pages = int(manifest.get("total_pages") or 0)
    pages_done = _count_page_files(out_dir)
    run_ids = _collect_run_ids_from_pages(out_dir / "pages", filter_fn=filter_fn)
    details_done = _load_existing_detail_ids(out_dir / "details")
    failed_done = _load_permanently_failed_ids(out_dir / "state" / "detail_errors.jsonl")
    finished = details_done | failed_done
    details_complete = len(finished) >= len(run_ids)
    if not details_complete and target_detail_count and len(details_done) >= int(target_detail_count):
        details_complete = True
    stop_hit = manifest.get("stop_on_known_hit_page")
    # pages_complete 判定:
    # - stop-on-known 启用: 必须等 manifest 里写入 stop_on_known_hit_page 才算完成
    #   (防止 daemon 首次 poll 就凭"老 pages 的 run_id 全 processed"过早退出,
    #   实际 pages 子进程还在扫新 page 里的增量 run_id)
    # - 否则: pages_done >= total_pages (完整覆盖) 或 老 stop_hit 记录都算完成
    if stop_on_known:
        pages_complete = stop_hit is not None
    else:
        pages_complete = bool(
            (total_pages and pages_done >= total_pages) or stop_hit is not None
        )
    return {
        "total_pages": total_pages,
        "pages_done": pages_done,
        "total_run_ids": len(run_ids),
        "details_done": len(details_done),
        "details_failed": len(failed_done),
        "pages_complete": pages_complete,
        "details_complete": details_complete,
        "stop_on_known_hit_page": stop_hit,
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
    parser.add_argument("--outcome", type=str, default=None, choices=[None, "victory", "defeat"], help="仅抓指定结局的 runs")
    parser.add_argument("--stop-on-known", action="store_true", help="pages 子任务遇到全已知 run_id 页后停止（增量模式）")
    parser.add_argument("--filter-preset", type=str, default=None, choices=[None, *FILTER_PRESETS_DAEMON.keys()], help="对 page list 返回的 run 记录做 filter（failure 同 raw.py）")
    parser.add_argument("--target-detail-count", type=int, default=0, help="抓到这么多 success detail 后即视为 details 完成并退出（0=不限）")
    parser.add_argument("--min-created-at", type=str, default=None, help="filter: 仅保留 created_at >= 该值的 run（格式 YYYY-MM-DD，字典序比较）")
    parser.add_argument("--page-workers", type=int, default=8)
    parser.add_argument("--detail-workers", type=int, default=3)
    parser.add_argument("--page-qps", type=float, default=0.6)
    parser.add_argument("--detail-qps", type=float, default=0.6)
    parser.add_argument("--detail-shard-lines", type=int, default=2000)
    parser.add_argument("--rate-limit-base-sleep", type=float, default=2.0, help="子任务退出后的本地指数退避起始秒数")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=60.0, help="子任务退出后的本地指数退避上限秒数")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="守护轮询间隔")
    parser.add_argument("--stall-kill-seconds", type=float, default=180.0, help="子进程进度连续多少秒不推进就强杀重启（默认 180s）")
    parser.add_argument("--use-proxy-pool", action="store_true", help="子任务启用快代理代理（产品由 --proxy-product 决定）")
    parser.add_argument("--proxy-product", type=str, default="dps", choices=["dps", "fps", "tps"], help="dps=私密代理池 / fps=海外住宅隧道 / tps=国内隧道每请求换 IP")
    parser.add_argument("--proxy-pool-size", type=int, default=20, help="代理池大小（建议 >= 3×detail-workers）")
    parser.add_argument("--proxy-ip-lifetime", type=float, default=300.0, help="代理 IP 本地预设有效期（秒），与订单配置一致")
    parser.add_argument("--proxy-bad-cooldown", type=float, default=60.0, help="IP 标 bad 后的冷却秒数")
    parser.add_argument("--proxy-max-uses-per-ip", type=int, default=0, help="每 IP 最多使用次数（>0 自动淘汰）")
    parser.add_argument("--proxy-area", type=str, default=None, help="可选：按地区筛选代理")
    parser.add_argument("--proxy-protocol", type=str, default="http", choices=["http", "https"], help="代理协议")
    parser.add_argument("--proxy-verbose", action="store_true", help="打印代理池拉取/标 bad 日志")
    return parser.parse_args()


def _build_proxy_args(args: argparse.Namespace) -> list[str]:
    if not args.use_proxy_pool:
        return []
    extra = [
        "--use-proxy-pool",
        "--proxy-product", args.proxy_product,
        "--proxy-pool-size", str(args.proxy_pool_size),
        "--proxy-ip-lifetime", str(args.proxy_ip_lifetime),
        "--proxy-bad-cooldown", str(args.proxy_bad_cooldown),
        "--proxy-max-uses-per-ip", str(args.proxy_max_uses_per_ip),
        "--proxy-protocol", args.proxy_protocol,
    ]
    if args.proxy_area:
        extra += ["--proxy-area", args.proxy_area]
    if args.proxy_verbose:
        extra.append("--proxy-verbose")
    return extra


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    log_dir = args.log_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    proxy_args = _build_proxy_args(args)
    outcome_args = ["--outcome", args.outcome] if args.outcome else []
    stop_on_known_args = ["--stop-on-known"] if args.stop_on_known else []
    filter_preset_args = ["--filter-preset", args.filter_preset] if args.filter_preset else []
    if args.min_created_at:
        filter_preset_args += ["--min-created-at", args.min_created_at]
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
            *outcome_args,
            *stop_on_known_args,
            *proxy_args,
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
            *filter_preset_args,
            *proxy_args,
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
                "use_proxy_pool": bool(args.use_proxy_pool),
                "proxy_pool_size": args.proxy_pool_size if args.use_proxy_pool else None,
                "proxy_ip_lifetime": args.proxy_ip_lifetime if args.use_proxy_pool else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # 清除 manifest 里残留的 stop_on_known_hit_page,避免 daemon 第一轮 poll
    # 看到旧标记立刻判"pages 已完成"过早退出(pages 子进程还没刷新 manifest)
    manifest_path = out_dir / "meta" / "runs_manifest.json"
    if manifest_path.exists():
        try:
            _manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "stop_on_known_hit_page" in _manifest:
                _manifest.pop("stop_on_known_hit_page", None)
                manifest_path.write_text(
                    json.dumps(_manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print("[daemon] 清除残留 stop_on_known_hit_page,本次重新判定", flush=True)
        except Exception as e:
            print(f"[daemon] 清除 stop_on_known_hit_page 失败(忽略): {e}", flush=True)

    _start_child(page_spec, cwd=out_dir.parents[3])
    _start_child(detail_spec, cwd=out_dir.parents[3])

    stall_kill_sec = max(float(args.stall_kill_seconds), 60.0)
    if args.filter_preset == "failure":
        filter_fn = _make_failure_filter_daemon(min_created_at=args.min_created_at)
    else:
        filter_fn = FILTER_PRESETS_DAEMON.get(args.filter_preset) if args.filter_preset else None
    target_count = int(args.target_detail_count) if args.target_detail_count else None
    try:
        while True:
            progress = _compute_progress(
                out_dir, filter_fn=filter_fn, target_detail_count=target_count,
                stop_on_known=bool(args.stop_on_known),
            )
            now_ts = time.time()
            print(
                f"[daemon] pages={progress['pages_done']}/{progress['total_pages']} "
                f"details={progress['details_done']}/{progress['total_run_ids']}",
                flush=True,
            )

            if progress["pages_complete"] and progress["details_complete"]:
                print("[daemon] crawl complete", flush=True)
                break

            # watchdog: 每个子进程监控自己关注的进度指标，连续 stall_kill_sec 不推进就强杀
            for spec, value_key, display_key in (
                (page_spec, "pages_done", "pages"),
                (detail_spec, "details_done", "details"),
            ):
                if spec.proc is None or spec.proc.poll() is not None:
                    continue
                current = int(progress.get(value_key, 0))
                if current > spec.last_progress_value:
                    spec.last_progress_value = current
                    spec.last_progress_ts = now_ts
                elif spec.last_progress_ts <= 0:
                    spec.last_progress_ts = now_ts
                elif now_ts - spec.last_progress_ts >= stall_kill_sec:
                    stall = now_ts - spec.last_progress_ts
                    print(
                        f"[daemon] WATCHDOG: {spec.name} {display_key} 停滞 {stall:.0f}s "
                        f"at {current} → 强杀 pid={spec.proc.pid}",
                        flush=True,
                    )
                    try:
                        spec.proc.terminate()
                    except Exception:
                        pass
                    spec.last_progress_ts = now_ts

            for spec in (page_spec, detail_spec):
                if _child_finished(spec):
                    rc = spec.proc.poll() if spec.proc is not None else None
                    if rc in NON_RETRIABLE_EXIT_CODES:
                        print(
                            f"[daemon] {spec.name} exited rc={rc}; non-retriable, stop restarting. "
                            f"stdout={spec.stdout_log} stderr={spec.stderr_log}",
                            flush=True,
                        )
                        raise SystemExit(rc)
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
