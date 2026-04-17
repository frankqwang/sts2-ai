#!/usr/bin/env python3
"""抓取 sts2log.com 原始 runs 数据。

目标：
1. 保存游戏版本与 quick-guide 原始 JSON
2. 以 page_size=20 抓取 runs 列表，每页一个 JSON list 文件
3. 抓取每个 run 的详情，按 JSONL 分片落盘

说明：
- 站点 API 需要前端同款签名（X-Skada-T / X-Skada-S）
- 默认使用 `sort_by=run_id&sort_dir=asc` 做全量回填，避免 desc 翻页时被新数据顶动
- 支持断点续爬：已有页文件 / 已有详情 JSONL 会自动跳过
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

import requests


BASE_URL = "https://sts2log.com"
API_BASE = f"{BASE_URL}/api"
PAGE_SIZE = 20
DEFAULT_TIMEOUT = 30.0
DEFAULT_PAGE_WORKERS = 8
DEFAULT_DETAIL_WORKERS = 12
DEFAULT_PAGE_QPS = 3.0
DEFAULT_DETAIL_QPS = 6.0
DEFAULT_DETAIL_SHARD_LINES = 2000
SIGNING_SECRET = "xK7m2pQ9dR4wF1jN8sL3vB6hY0tG5cA"

_THREAD_LOCAL = threading.local()


class ApiRateLimitError(RuntimeError):
    def __init__(
        self,
        *,
        url: str,
        retry_after: float | None = None,
        error_code: str = "",
        message: str = "",
    ):
        super().__init__(f"429 rate limited: code={error_code or 'unknown'} retry_after={retry_after} url={url}")
        self.url = url
        self.retry_after = retry_after
        self.error_code = error_code
        self.message = message


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    tmp.replace(path)


def iter_jsonl_records(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class RateLimiter:
    def __init__(self, qps: float):
        self._interval = 0.0 if qps <= 0 else 1.0 / qps
        self._lock = threading.Lock()
        self._next_ready = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        wait_s = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_ready:
                wait_s = self._next_ready - now
                self._next_ready += self._interval
            else:
                self._next_ready = now + self._interval
        if wait_s > 0:
            time.sleep(wait_s)


class SkadaApiClient:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def _session(self) -> requests.Session:
        session = getattr(_THREAD_LOCAL, "skada_session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Referer": f"{BASE_URL}/",
                "Origin": BASE_URL,
            })
            _THREAD_LOCAL.skada_session = session
        return session

    @staticmethod
    def _sign(url: str) -> dict[str, str]:
        parsed = urlparse(url)
        ts = str(int(time.time()))
        items = sorted(parse_qsl(parsed.query, keep_blank_values=True), key=lambda kv: (kv[0], kv[1]))
        canonical_query = "&".join(f"{k}={v}" for k, v in items)
        message = f"{ts}:{parsed.path}?{canonical_query}" if canonical_query else f"{ts}:{parsed.path}"
        sig = hmac.new(
            SIGNING_SECRET.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {"X-Skada-T": ts, "X-Skada-S": sig}

    def get_json(
        self,
        url: str,
        *,
        limiter: RateLimiter | None = None,
        max_retries: int = 6,
    ) -> Any:
        session = self._session()
        for attempt in range(max_retries):
            if limiter is not None:
                limiter.acquire()
            headers = self._sign(url)
            try:
                resp = session.get(url, headers=headers, timeout=self.timeout)
                if resp.status_code == 429:
                    retry_after_header = resp.headers.get("Retry-After")
                    retry_after = float(retry_after_header) if retry_after_header else None
                    error_code = ""
                    error_message = ""
                    try:
                        body = resp.json()
                        error_code = str(body.get("error") or "")
                        error_message = str(body.get("message") or "")
                        if retry_after is None and body.get("retry_after") is not None:
                            retry_after = float(body["retry_after"])
                    except Exception:
                        body = None
                    print(
                        f"[rate_limit] attempt={attempt + 1}/{max_retries} "
                        f"code={error_code or 'unknown'} retry_after={retry_after} url={url}"
                    )
                    if error_code != "daily_limit_exceeded" and attempt < max_retries - 1:
                        backoff = retry_after if retry_after is not None else min(60.0, 2 ** attempt)
                        time.sleep(max(backoff, 1.0))
                        continue
                    raise ApiRateLimitError(
                        url=url,
                        retry_after=retry_after,
                        error_code=error_code,
                        message=error_message,
                    )
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError):
                if attempt >= max_retries - 1:
                    raise
                time.sleep(min(60.0, 2 ** attempt))
        raise RuntimeError(f"Unreachable retry loop for {url}")


class DetailShardWriter:
    def __init__(self, details_dir: Path, shard_lines: int):
        self.details_dir = details_dir
        self.shard_lines = max(int(shard_lines), 1)
        self.details_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = None
        self._count_in_shard = 0
        self._shard_index = self._discover_next_shard_index()

    def _discover_next_shard_index(self) -> int:
        existing = sorted(self.details_dir.glob("run_details_*.jsonl"))
        if not existing:
            return 1
        last = existing[-1].stem.split("_")[-1]
        try:
            return int(last) + 1
        except ValueError:
            return len(existing) + 1

    def _rotate_if_needed(self) -> None:
        if self._fh is not None and self._count_in_shard < self.shard_lines:
            return
        if self._fh is not None:
            self._fh.close()
        path = self.details_dir / f"run_details_{self._shard_index:06d}.jsonl"
        self._fh = path.open("a", encoding="utf-8", newline="\n")
        self._count_in_shard = 0
        self._shard_index += 1

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._rotate_if_needed()
            assert self._fh is not None
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._count_in_shard += 1

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class ErrorWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_runs_url(*, page: int, page_size: int, sort_by: str, sort_dir: str) -> str:
    query = urlencode({
        "page": page,
        "page_size": page_size,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
    })
    return f"{API_BASE}/runs?{query}"


def build_run_detail_url(run_id: int) -> str:
    return f"{API_BASE}/runs/{int(run_id)}"


def load_existing_detail_ids(details_dir: Path) -> set[int]:
    done: set[int] = set()
    for path in sorted(details_dir.glob("run_details_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    run = record.get("run") or {}
                    run_id = run.get("run_id")
                    if run_id is not None:
                        done.add(int(run_id))
                except Exception:
                    continue
    return done


def load_runs_manifest(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "meta" / "runs_manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_run_ids_from_pages(pages_dir: Path) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for path in sorted(pages_dir.glob("runs_page_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
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


def fetch_and_save_page(
    client: SkadaApiClient,
    limiter: RateLimiter,
    page: int,
    *,
    page_size: int,
    sort_by: str,
    sort_dir: str,
    pages_dir: Path,
) -> tuple[int, int]:
    url = build_runs_url(page=page, page_size=page_size, sort_by=sort_by, sort_dir=sort_dir)
    payload = client.get_json(url, limiter=limiter)
    runs = payload.get("runs") or []
    if not isinstance(runs, list):
        raise RuntimeError(f"runs payload malformed for page={page}")
    save_json(pages_dir / f"runs_page_{page:06d}.json", runs)
    return page, len(runs)


def crawl_page_lists(
    client: SkadaApiClient,
    out_dir: Path,
    *,
    page_size: int,
    sort_by: str,
    sort_dir: str,
    page_workers: int,
    page_qps: float,
    max_pages: int | None,
) -> dict[str, Any]:
    meta_dir = out_dir / "meta"
    pages_dir = out_dir / "pages"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    versions = client.get_json(f"{API_BASE}/versions")
    quick_guide = client.get_json(f"{API_BASE}/quick-guide")
    save_json(meta_dir / "versions.json", versions)
    save_json(meta_dir / "quick_guide.json", quick_guide)

    first_payload = client.get_json(
        build_runs_url(page=1, page_size=page_size, sort_by=sort_by, sort_dir=sort_dir)
    )
    first_runs = first_payload.get("runs") or []
    pagination = first_payload.get("pagination") or {}
    total_pages = int(pagination.get("total_pages") or 1)
    total_runs = int(pagination.get("total") or len(first_runs))
    if max_pages is not None:
        total_pages = min(total_pages, int(max_pages))
    save_json(pages_dir / "runs_page_000001.json", first_runs)

    manifest = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": BASE_URL,
        "api_base": API_BASE,
        "page_size": page_size,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
        "total_runs": total_runs,
        "total_pages": total_pages,
        "notice": first_payload.get("_notice"),
        "available_versions": first_payload.get("available_versions"),
        "game_mode_filter": first_payload.get("game_mode_filter"),
    }
    save_json(meta_dir / "runs_manifest.json", manifest)

    existing_pages = {
        int(path.stem.split("_")[-1])
        for path in pages_dir.glob("runs_page_*.json")
        if path.stem.split("_")[-1].isdigit()
    }
    pending_pages = [page for page in range(2, total_pages + 1) if page not in existing_pages]

    limiter = RateLimiter(page_qps)
    if pending_pages:
        completed = 1
        with ThreadPoolExecutor(max_workers=max(1, int(page_workers))) as pool:
            pending_futures = {
                pool.submit(
                    fetch_and_save_page,
                    client,
                    limiter,
                    page,
                    page_size=page_size,
                    sort_by=sort_by,
                    sort_dir=sort_dir,
                    pages_dir=pages_dir,
                ): page
                for page in pending_pages
            }
            while pending_futures:
                done, _ = wait(list(pending_futures.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    pending_futures.pop(future, None)
                    page, run_count = future.result()
                    completed += 1
                    if completed % 100 == 0 or completed == total_pages:
                        print(f"[pages] {completed}/{total_pages} page={page} runs={run_count}")
    else:
        print("[pages] no pending pages")

    return manifest


def crawl_run_details(
    client: SkadaApiClient,
    out_dir: Path,
    *,
    detail_workers: int,
    detail_qps: float,
    detail_shard_lines: int,
    max_details: int | None,
    follow_pages: bool = False,
    poll_seconds: float = 10.0,
    rate_limit_base_sleep: float = 2.0,
    rate_limit_max_sleep: float = 60.0,
    ignore_server_retry_after: bool = False,
) -> dict[str, Any]:
    pages_dir = out_dir / "pages"
    details_dir = out_dir / "details"
    state_dir = out_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    writer = DetailShardWriter(details_dir, shard_lines=detail_shard_lines)
    error_writer = ErrorWriter(state_dir / "detail_errors.jsonl")
    limiter = RateLimiter(detail_qps)
    manifest = load_runs_manifest(out_dir)
    total_pages_expected = int(manifest.get("total_pages") or 0) if manifest else 0
    existing_ids = load_existing_detail_ids(details_dir)
    processed_ids = set(existing_ids)
    success = 0
    failed = 0
    lock = threading.Lock()
    consecutive_rate_limits = 0
    cooldown_until = 0.0

    def _fetch(run_id: int) -> tuple[int, dict[str, Any]]:
        payload = client.get_json(build_run_detail_url(run_id), limiter=limiter, max_retries=1)
        return run_id, payload

    try:
        while True:
            now = time.time()
            if cooldown_until > now:
                sleep_s = max(cooldown_until - now, 0.0)
                wake_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cooldown_until))
                print(f"[details] cooldown active for {sleep_s:.1f}s, resume_at={wake_at}")
                time.sleep(sleep_s)

            run_ids = collect_run_ids_from_pages(pages_dir)
            pending_ids = [run_id for run_id in run_ids if run_id not in processed_ids]
            if max_details is not None:
                remaining = max(int(max_details) - success - failed, 0)
                pending_ids = pending_ids[:remaining]

            total_this_round = len(pending_ids)
            if total_this_round > 0:
                rate_limit_exc: ApiRateLimitError | None = None
                pool = ThreadPoolExecutor(max_workers=max(1, int(detail_workers)))
                try:
                    pending_futures = {}
                    cursor = 0
                    while cursor < total_this_round or pending_futures:
                        while cursor < total_this_round and len(pending_futures) < max(1, int(detail_workers)):
                            run_id = pending_ids[cursor]
                            future = pool.submit(_fetch, run_id)
                            pending_futures[future] = run_id
                            cursor += 1

                        done, _ = wait(list(pending_futures.keys()), return_when=FIRST_COMPLETED)
                        for future in done:
                            run_id = pending_futures.pop(future)
                            try:
                                _, payload = future.result()
                                writer.write(payload)
                                with lock:
                                    success += 1
                                    processed_ids.add(run_id)
                                    done_count = success + failed
                                    consecutive_rate_limits = 0
                                if done_count % 100 == 0:
                                    print(f"[details] done={done_count} ok={success} fail={failed}")
                            except ApiRateLimitError as exc:
                                consecutive_rate_limits += 1
                                rate_limit_exc = exc
                                retry_after = exc.retry_after if exc.retry_after is not None else 0.0
                                print(
                                    f"[details] rate-limited run_id={run_id} "
                                    f"code={exc.error_code or 'unknown'} retry_after={retry_after} "
                                    f"consecutive={consecutive_rate_limits}"
                                )
                            except Exception as exc:
                                error_writer.write({
                                    "run_id": run_id,
                                    "error": repr(exc),
                                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                })
                                with lock:
                                    failed += 1
                                    done_count = success + failed
                                if done_count % 50 == 0:
                                    print(f"[details] done={done_count} ok={success} fail={failed}")
                        if rate_limit_exc is not None:
                            break
                finally:
                    if rate_limit_exc is not None:
                        for future in pending_futures:
                            future.cancel()
                        pool.shutdown(wait=False, cancel_futures=True)
                    else:
                        pool.shutdown(wait=True, cancel_futures=False)

                if rate_limit_exc is not None:
                    retry_after = rate_limit_exc.retry_after if rate_limit_exc.retry_after is not None else 0.0
                    local_backoff = min(
                        max(float(rate_limit_max_sleep), 1.0),
                        max(float(rate_limit_base_sleep), 0.1) * (2 ** max(consecutive_rate_limits - 1, 0)),
                    )
                    if ignore_server_retry_after:
                        cooldown_s = local_backoff
                    else:
                        cooldown_s = max(retry_after, local_backoff)
                    cooldown_until = time.time() + cooldown_s
                    wake_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cooldown_until))
                    print(
                        f"[details] enter cooldown {cooldown_s:.1f}s "
                        f"until={wake_at} code={rate_limit_exc.error_code or 'unknown'} "
                        f"(local_backoff={local_backoff:.1f}s, retry_after={retry_after})"
                    )
                    continue
            elif not follow_pages:
                print("[details] no pending run details")
                break

            if max_details is not None and (success + failed) >= int(max_details):
                break

            if not follow_pages:
                break

            current_page_count = len(list(pages_dir.glob("runs_page_*.json")))
            pages_complete = bool(total_pages_expected and current_page_count >= total_pages_expected)
            if pages_complete:
                run_ids = collect_run_ids_from_pages(pages_dir)
                pending_after = [run_id for run_id in run_ids if run_id not in processed_ids]
                if not pending_after:
                    break

            time.sleep(max(float(poll_seconds), 1.0))
    finally:
        writer.close()

    return {
        "total_run_ids": len(collect_run_ids_from_pages(pages_dir)),
        "existing_detail_ids": len(existing_ids),
        "fetched_details": success,
        "failed_details": failed,
        "pending_after": len([run_id for run_id in collect_run_ids_from_pages(pages_dir) if run_id not in processed_ids]),
        "follow_pages": follow_pages,
    }


def parse_args() -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent / "runs"
    parser = argparse.ArgumentParser(description="抓取 sts2log 原始 runs / detail 数据")
    parser.add_argument("--out-dir", type=Path, default=default_out, help="输出目录")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE, help="runs 列表页大小，默认 20")
    parser.add_argument("--sort-by", type=str, default="run_id", help="分页排序字段，默认 run_id")
    parser.add_argument("--sort-dir", type=str, default="asc", choices=["asc", "desc"], help="排序方向")
    parser.add_argument("--page-workers", type=int, default=DEFAULT_PAGE_WORKERS, help="列表页并发数")
    parser.add_argument("--detail-workers", type=int, default=DEFAULT_DETAIL_WORKERS, help="详情并发数")
    parser.add_argument("--page-qps", type=float, default=DEFAULT_PAGE_QPS, help="列表页总 QPS 上限")
    parser.add_argument("--detail-qps", type=float, default=DEFAULT_DETAIL_QPS, help="详情总 QPS 上限")
    parser.add_argument("--detail-shard-lines", type=int, default=DEFAULT_DETAIL_SHARD_LINES, help="每个详情 JSONL 分片行数")
    parser.add_argument("--max-pages", type=int, default=None, help="最多抓多少页（调试用）")
    parser.add_argument("--max-details", type=int, default=None, help="最多抓多少详情（调试用）")
    parser.add_argument("--pages-only", action="store_true", help="只抓列表页")
    parser.add_argument("--details-only", action="store_true", help="只抓详情（基于已有 pages）")
    parser.add_argument("--follow-pages", action="store_true", help="详情模式持续轮询 pages，边出页边抓详情")
    parser.add_argument("--poll-seconds", type=float, default=10.0, help="follow-pages 时轮询间隔秒数")
    parser.add_argument("--rate-limit-base-sleep", type=float, default=2.0, help="429 时本地指数退避起始秒数")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=60.0, help="429 时本地指数退避上限秒数")
    parser.add_argument("--ignore-server-retry-after", action="store_true", help="忽略服务端 retry_after，改用本地指数退避")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="单请求超时秒数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pages_only and args.details_only:
        raise SystemExit("--pages-only 和 --details-only 不能同时使用")
    if args.page_size != 20:
        print(f"[warn] 用户要求 page_size=20，当前传入 {args.page_size}")

    client = SkadaApiClient(timeout=args.timeout)
    started_at = time.time()
    summary: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": out_dir,
        "page_size": args.page_size,
        "sort_by": args.sort_by,
        "sort_dir": args.sort_dir,
        "page_workers": args.page_workers,
        "detail_workers": args.detail_workers,
        "page_qps": args.page_qps,
        "detail_qps": args.detail_qps,
        "detail_shard_lines": args.detail_shard_lines,
        "max_pages": args.max_pages,
        "max_details": args.max_details,
        "rate_limit_base_sleep": args.rate_limit_base_sleep,
        "rate_limit_max_sleep": args.rate_limit_max_sleep,
        "ignore_server_retry_after": bool(args.ignore_server_retry_after),
    }

    try:
        if not args.details_only:
            summary["pages"] = crawl_page_lists(
                client,
                out_dir,
                page_size=args.page_size,
                sort_by=args.sort_by,
                sort_dir=args.sort_dir,
                page_workers=args.page_workers,
                page_qps=args.page_qps,
                max_pages=args.max_pages,
            )

        if not args.pages_only:
            summary["details"] = crawl_run_details(
                client,
                out_dir,
                detail_workers=args.detail_workers,
                detail_qps=args.detail_qps,
                detail_shard_lines=args.detail_shard_lines,
                max_details=args.max_details,
                follow_pages=args.follow_pages,
                poll_seconds=args.poll_seconds,
                rate_limit_base_sleep=args.rate_limit_base_sleep,
                rate_limit_max_sleep=args.rate_limit_max_sleep,
                ignore_server_retry_after=args.ignore_server_retry_after,
            )
    except ApiRateLimitError as exc:
        summary["rate_limited"] = {
            "url": exc.url,
            "error_code": exc.error_code,
            "message": exc.message,
            "retry_after": exc.retry_after,
            "limited_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        print(
            f"[main] rate limited: code={exc.error_code or 'unknown'} "
            f"retry_after={exc.retry_after} url={exc.url}"
        )

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["elapsed_sec"] = round(time.time() - started_at, 2)
    save_json(out_dir / "meta" / "crawl_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
