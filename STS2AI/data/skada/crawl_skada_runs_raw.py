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

from kuaidaili_proxy_pool import (
    KuaidailiFPSTunnel,
    KuaidailiProxyPool,
    KuaidailiTPSTunnel,
    ProxyTicket,
)


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
DEFAULT_PROXY_POOL_SIZE = 20
DEFAULT_PROXY_IP_LIFETIME = 300.0
DEFAULT_PROXY_BAD_COOLDOWN = 60.0
DEFAULT_PROXY_MAX_RETRIES = 12

_FAILURE_MIN_VERSION = (0, 99, 0)


def _parse_version(v: str | None) -> tuple[int, ...]:
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


def _filter_failure_record(r: dict[str, Any]) -> bool:
    """有意义的 failure run: abandoned=false + 有死因 + 有足够时长/进度 + 新版本。"""
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
    if _parse_version(r.get("game_version")) < _FAILURE_MIN_VERSION:
        return False
    return True


FILTER_PRESETS: dict[str, Any] = {
    "failure": _filter_failure_record,
}


def make_failure_filter(min_created_at: str | None = None) -> Any:
    """构造带日期下界的 failure filter。min_created_at 形如 '2026-04-15'。"""
    base = _filter_failure_record
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


class ProxyAuthConfigError(RuntimeError):
    """代理鉴权配置错误（比如机器 IP 没在代理白名单，或账密鉴权没设）。

    这种错误所有代理都会返回同样的 407，继续换 IP 无意义，
    需要用户在快代理后台修复白名单 / 设置 KDL_USERNAME / KDL_PASSWORD 后再跑。
    """


class ProxyPoolUnavailableError(RuntimeError):
    """代理池当前拿不到可用代理。

    这是临时性故障：可能是池子空、全部 bad、或后台补拉失败。
    调用方应退避后重试，而不是静默直连宿主机公网 IP。
    """


def _is_proxy_auth_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "407" in msg
        or "Proxy Authentication Required" in msg
        or "456" in msg
        or "CN Client Forbidden" in msg
    )


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
    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        proxy_pool: "KuaidailiProxyPool | KuaidailiFPSTunnel | KuaidailiTPSTunnel | None" = None,
    ):
        self.timeout = timeout
        self.proxy_pool = proxy_pool

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
                "Accept-Encoding": "gzip, deflate",
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

    def _acquire_ticket(self) -> ProxyTicket | None:
        if self.proxy_pool is None:
            return None
        return self.proxy_pool.acquire()

    def get_json(
        self,
        url: str,
        *,
        limiter: RateLimiter | None = None,
        max_retries: int = 6,
    ) -> Any:
        session = self._session()
        use_pool = self.proxy_pool is not None
        effective_retries = max(max_retries, DEFAULT_PROXY_MAX_RETRIES) if use_pool else max_retries
        for attempt in range(effective_retries):
            headers = self._sign(url)
            ticket = self._acquire_ticket()
            if use_pool and ticket is None:
                if attempt >= effective_retries - 1:
                    raise ProxyPoolUnavailableError(
                        f"代理池当前没有可用代理，已重试 {effective_retries} 次: {url}"
                    )
                if attempt == 0 or (attempt + 1) % 5 == 0:
                    print(
                        f"[proxy_pool_empty] attempt={attempt + 1}/{effective_retries} url={url}"
                    )
                time.sleep(0.5)
                continue
            if limiter is not None:
                limiter.acquire()
            proxies = ticket.proxies if ticket is not None else None
            proxy_label = ticket.ip_port if ticket is not None else "-"
            try:
                resp = session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    proxies=proxies,
                )
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
                    if ticket is not None:
                        self.proxy_pool.mark_bad(ticket.ip_port)
                    print(
                        f"[rate_limit] attempt={attempt + 1}/{effective_retries} "
                        f"code={error_code or 'unknown'} retry_after={retry_after} "
                        f"proxy={proxy_label} url={url}"
                    )
                    if error_code == "daily_limit_exceeded":
                        raise ApiRateLimitError(
                            url=url,
                            retry_after=retry_after,
                            error_code=error_code,
                            message=error_message,
                        )
                    if attempt < effective_retries - 1:
                        if use_pool:
                            time.sleep(0.2)
                        else:
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
                payload = resp.json()
                if ticket is not None:
                    self.proxy_pool.release(ticket, succeeded=True)
                return payload
            except (requests.RequestException, ValueError) as exc:
                if use_pool and _is_proxy_auth_error(exc):
                    err_msg = str(exc)
                    if "456" in err_msg or "CN Client Forbidden" in err_msg:
                        raise ProxyAuthConfigError(
                            "代理返回 456 CN Client Forbidden — "
                            "FPS（海外代理）不允许从中国大陆客户端连接。\n"
                            "修复方式：\n"
                            "  A) 联系快代理客服开通 CN 客户端访问权限\n"
                            "  B) 把脚本部署到海外 VPS 跑\n"
                            "  C) 临时改用 --proxy-product dps 回到国内私密代理池"
                        ) from exc
                    raise ProxyAuthConfigError(
                        "代理返回 407 Proxy Authentication Required — "
                        "当前机器 IP 不在快代理【代理白名单】里；"
                        "或订单是账密鉴权但未设 KDL_USERNAME / KDL_PASSWORD。\n"
                        "修复方式：\n"
                        "  A) 后台【IP 白名单】添加本机公网 IP（curl ifconfig.me 可查），或\n"
                        "  B) export KDL_USERNAME=... KDL_PASSWORD=... 改走账密鉴权"
                    ) from exc
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    sc = exc.response.status_code
                    if 400 <= sc < 500 and sc not in (407, 408, 429):
                        raise
                if ticket is not None:
                    self.proxy_pool.mark_bad(ticket.ip_port)
                if attempt >= effective_retries - 1:
                    raise
                if use_pool:
                    time.sleep(0.2)
                else:
                    time.sleep(min(60.0, 2 ** attempt))
                if use_pool and isinstance(exc, requests.RequestException):
                    print(
                        f"[proxy_err] attempt={attempt + 1}/{effective_retries} "
                        f"proxy={proxy_label} err={exc.__class__.__name__} url={url}"
                    )
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


def build_runs_url(
    *,
    page: int,
    page_size: int,
    sort_by: str,
    sort_dir: str,
    outcome: str | None = None,
) -> str:
    params = {
        "page": page,
        "page_size": page_size,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
    }
    if outcome:
        params["outcome"] = outcome
    return f"{API_BASE}/runs?{urlencode(params)}"


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


def load_permanently_failed_ids(errors_path: Path) -> set[int]:
    """从 detail_errors.jsonl 读取永久失败的 run_id（404 等不可重试错误），
    用于 merge 进 processed_ids，避免每次 daemon 重启都重新抓 404。"""
    done: set[int] = set()
    if not errors_path.exists():
        return done
    with errors_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                run_id = rec.get("run_id")
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


def collect_run_ids_from_pages(
    pages_dir: Path,
    filter_fn: Any = None,
) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for path in sorted(pages_dir.glob("runs_page_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
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


def fetch_and_save_page(
    client: SkadaApiClient,
    limiter: RateLimiter,
    page: int,
    *,
    page_size: int,
    sort_by: str,
    sort_dir: str,
    pages_dir: Path,
    outcome: str | None = None,
) -> tuple[int, int, list[int]]:
    url = build_runs_url(
        page=page, page_size=page_size, sort_by=sort_by, sort_dir=sort_dir, outcome=outcome
    )
    payload = client.get_json(url, limiter=limiter)
    runs = payload.get("runs") or []
    if not isinstance(runs, list):
        raise RuntimeError(f"runs payload malformed for page={page}")
    save_json(pages_dir / f"runs_page_{page:06d}.json", runs)
    run_ids = [int(r["run_id"]) for r in runs if isinstance(r, dict) and r.get("run_id") is not None]
    return page, len(runs), run_ids


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
    outcome: str | None = None,
    stop_on_known: bool = False,
    known_run_ids: set[int] | None = None,
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
        build_runs_url(
            page=1, page_size=page_size, sort_by=sort_by, sort_dir=sort_dir, outcome=outcome
        )
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
        "outcome_filter": outcome,
        "total_runs": total_runs,
        "total_pages": total_pages,
        "notice": first_payload.get("_notice"),
        "available_versions": first_payload.get("available_versions"),
        "game_mode_filter": first_payload.get("game_mode_filter"),
    }
    save_json(meta_dir / "runs_manifest.json", manifest)

    existing_page_files = sorted(
        [p for p in pages_dir.glob("runs_page_*.json") if p.stem.split("_")[-1].isdigit()],
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if existing_page_files:
        last_path = existing_page_files[-1]
        last_page_num = int(last_path.stem.split("_")[-1])
        if 1 < last_page_num <= total_pages:
            try:
                last_path.unlink()
                print(f"[pages] resume: 删除最后一页 {last_path.name} 强制重抓（tail-page 可能有增量）")
            except OSError:
                pass
    if stop_on_known:
        # 增量模式（desc + stop-on-known）：sts2log 新增 run 会把老 run 挤到
        # 后面 page,老 page 2..N 的内容会变。必须重抓所有 pages,靠 stop-on-known
        # 自动在"扫到整页全已知"时停止,否则 6 小时新增的 ~3500 run_id 会全漏掉
        # (老 page 文件存在 → skip exist → 读不到新填充内容)。
        pending_pages = list(range(2, total_pages + 1))
    else:
        existing_pages = {
            int(path.stem.split("_")[-1])
            for path in pages_dir.glob("runs_page_*.json")
            if path.stem.split("_")[-1].isdigit()
        }
        pending_pages = [page for page in range(2, total_pages + 1) if page not in existing_pages]

    limiter = RateLimiter(page_qps)
    known_ids: set[int] = set(known_run_ids or ())
    early_stop = threading.Event()
    stop_hit_page: int | None = None
    if pending_pages:
        completed = 1
        with ThreadPoolExecutor(max_workers=max(1, int(page_workers))) as pool:
            pending_futures: dict[Any, int] = {}
            cursor = 0
            total = len(pending_pages)
            while cursor < total or pending_futures:
                while (
                    cursor < total
                    and len(pending_futures) < max(1, int(page_workers))
                    and not early_stop.is_set()
                ):
                    page = pending_pages[cursor]
                    future = pool.submit(
                        fetch_and_save_page,
                        client,
                        limiter,
                        page,
                        page_size=page_size,
                        sort_by=sort_by,
                        sort_dir=sort_dir,
                        pages_dir=pages_dir,
                        outcome=outcome,
                    )
                    pending_futures[future] = page
                    cursor += 1
                if not pending_futures:
                    break
                done, _ = wait(list(pending_futures.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    page = pending_futures.pop(future, None)
                    try:
                        page_done, run_count, run_ids = future.result()
                    except Exception as exc:
                        if early_stop.is_set():
                            continue
                        raise
                    completed += 1
                    if stop_on_known and run_ids and known_ids:
                        page_ids_set = set(run_ids)
                        if page_ids_set and page_ids_set.issubset(known_ids):
                            if stop_hit_page is None or page_done < stop_hit_page:
                                stop_hit_page = page_done
                            print(
                                f"[stop-on-known] page={page_done} 全部 {len(page_ids_set)} 个 run_id "
                                f"已在 known({len(known_ids)}) 中 → 停止抓后续 pages"
                            )
                            early_stop.set()
                    if completed % 100 == 0 or completed == total_pages:
                        print(f"[pages] {completed}/{total_pages} page={page_done} runs={run_count}")
                if early_stop.is_set():
                    for f in list(pending_futures.keys()):
                        f.cancel()
    else:
        print("[pages] no pending pages")

    if stop_hit_page is not None:
        manifest["stop_on_known_hit_page"] = stop_hit_page
        save_json(meta_dir / "runs_manifest.json", manifest)
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
    filter_fn: Any = None,
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
    perm_failed_ids = load_permanently_failed_ids(state_dir / "detail_errors.jsonl")
    processed_ids = set(existing_ids) | perm_failed_ids
    success = 0
    failed = 0
    lock = threading.Lock()
    consecutive_transient_errors = 0
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

            run_ids = collect_run_ids_from_pages(pages_dir, filter_fn=filter_fn)
            pending_ids = [run_id for run_id in run_ids if run_id not in processed_ids]
            if max_details is not None:
                remaining = max(int(max_details) - success - failed, 0)
                pending_ids = pending_ids[:remaining]

            total_this_round = len(pending_ids)
            if total_this_round > 0:
                transient_exc: ApiRateLimitError | ProxyPoolUnavailableError | None = None
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
                                    consecutive_transient_errors = 0
                                if done_count % 100 == 0:
                                    print(f"[details] done={done_count} ok={success} fail={failed}")
                            except ApiRateLimitError as exc:
                                consecutive_transient_errors += 1
                                transient_exc = exc
                                retry_after = exc.retry_after if exc.retry_after is not None else 0.0
                                print(
                                    f"[details] rate-limited run_id={run_id} "
                                    f"code={exc.error_code or 'unknown'} retry_after={retry_after} "
                                    f"consecutive={consecutive_transient_errors}"
                                )
                            except ProxyPoolUnavailableError as exc:
                                consecutive_transient_errors += 1
                                transient_exc = exc
                                print(
                                    f"[details] proxy unavailable run_id={run_id} "
                                    f"consecutive={consecutive_transient_errors}: {exc}"
                                )
                            except Exception as exc:
                                error_writer.write({
                                    "run_id": run_id,
                                    "error": repr(exc),
                                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                })
                                with lock:
                                    failed += 1
                                    processed_ids.add(run_id)
                                    done_count = success + failed
                                if done_count % 50 == 0:
                                    print(f"[details] done={done_count} ok={success} fail={failed}")
                        if transient_exc is not None:
                            break
                finally:
                    if transient_exc is not None:
                        for future in pending_futures:
                            future.cancel()
                        pool.shutdown(wait=False, cancel_futures=True)
                    else:
                        pool.shutdown(wait=True, cancel_futures=False)

                if transient_exc is not None:
                    retry_after = 0.0
                    if isinstance(transient_exc, ApiRateLimitError) and transient_exc.retry_after is not None:
                        retry_after = transient_exc.retry_after
                    local_backoff = min(
                        max(float(rate_limit_max_sleep), 1.0),
                        max(float(rate_limit_base_sleep), 0.1) * (2 ** max(consecutive_transient_errors - 1, 0)),
                    )
                    if ignore_server_retry_after or not isinstance(transient_exc, ApiRateLimitError):
                        cooldown_s = local_backoff
                    else:
                        cooldown_s = max(retry_after, local_backoff)
                    cooldown_until = time.time() + cooldown_s
                    wake_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cooldown_until))
                    error_code = (
                        transient_exc.error_code
                        if isinstance(transient_exc, ApiRateLimitError)
                        else "proxy_pool_unavailable"
                    )
                    print(
                        f"[details] enter cooldown {cooldown_s:.1f}s "
                        f"until={wake_at} code={error_code or 'unknown'} "
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
                run_ids = collect_run_ids_from_pages(pages_dir, filter_fn=filter_fn)
                pending_after = [run_id for run_id in run_ids if run_id not in processed_ids]
                if not pending_after:
                    break

            time.sleep(max(float(poll_seconds), 1.0))
    finally:
        writer.close()

    return {
        "total_run_ids": len(collect_run_ids_from_pages(pages_dir, filter_fn=filter_fn)),
        "existing_detail_ids": len(existing_ids),
        "fetched_details": success,
        "failed_details": failed,
        "pending_after": len([run_id for run_id in collect_run_ids_from_pages(pages_dir, filter_fn=filter_fn) if run_id not in processed_ids]),
        "follow_pages": follow_pages,
    }


def parse_args() -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent / "runs"
    parser = argparse.ArgumentParser(description="抓取 sts2log 原始 runs / detail 数据")
    parser.add_argument("--out-dir", type=Path, default=default_out, help="输出目录")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE, help="runs 列表页大小，默认 20")
    parser.add_argument("--sort-by", type=str, default="run_id", help="分页排序字段，默认 run_id")
    parser.add_argument("--sort-dir", type=str, default="asc", choices=["asc", "desc"], help="排序方向")
    parser.add_argument("--outcome", type=str, default=None, choices=[None, "victory", "defeat"], help="仅抓指定结局的 runs（默认抓全部）")
    parser.add_argument("--stop-on-known", action="store_true", help="增量模式：某页 run_id 全部已存在于 details 目录时停止抓后续 pages（配合 sort=created_at desc 做纯增量爬取）")
    parser.add_argument("--filter-preset", type=str, default=None, choices=[None, *FILTER_PRESETS.keys()], help="对 page list 返回的 run 记录做 filter（failure: abandoned=false + death_cause + dur>=180 + floor/asc 门槛 + ver>=0.99）")
    parser.add_argument("--min-created-at", type=str, default=None, help="filter: 仅保留 created_at >= 该值的 run（格式 YYYY-MM-DD 或完整时间戳，字典序比较）")
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
    parser.add_argument("--use-proxy-pool", action="store_true", help="启用快代理代理（DPS 池 或 FPS 隧道，视 --proxy-product 而定）")
    parser.add_argument("--proxy-product", type=str, default="dps", choices=["dps", "fps", "tps"], help="dps=动态私密代理（按 IP 配额）；fps=海外住宅隧道；tps=国内隧道（每请求换 IP）")
    parser.add_argument("--proxy-pool-size", type=int, default=DEFAULT_PROXY_POOL_SIZE, help="[DPS 专用] 代理池大小（建议 >= 3×detail-workers）")
    parser.add_argument("--proxy-ip-lifetime", type=float, default=DEFAULT_PROXY_IP_LIFETIME, help="单个代理 IP 本地预设有效期（秒），与快代理订单配置一致")
    parser.add_argument("--proxy-bad-cooldown", type=float, default=DEFAULT_PROXY_BAD_COOLDOWN, help="IP 被标为 bad 后的冷却秒数")
    parser.add_argument("--proxy-max-uses-per-ip", type=int, default=0, help="每个 IP 最多使用次数（>0 后自动淘汰，用于 per-IP 限流对抗，推荐 15-30）")
    parser.add_argument("--proxy-area", type=str, default=None, help="可选：按地区筛选代理（例：'北京,上海'）")
    parser.add_argument("--proxy-protocol", type=str, default="http", choices=["http", "https"], help="代理协议，默认 http")
    parser.add_argument("--proxy-verbose", action="store_true", help="打印代理池拉取/标 bad 日志")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pages_only and args.details_only:
        raise SystemExit("--pages-only 和 --details-only 不能同时使用")
    if args.page_size != 20:
        print(f"[warn] 用户要求 page_size=20，当前传入 {args.page_size}")

    proxy_pool = None
    if args.use_proxy_pool:
        if args.proxy_product == "tps":
            proxy_pool = KuaidailiTPSTunnel.from_env(
                protocol=args.proxy_protocol,
                verbose=args.proxy_verbose,
            )
            if proxy_pool is None:
                raise SystemExit(
                    "--proxy-product tps 需要 KDL_TPS_HOST + KDL_TPS_PORT 环境变量；"
                    "账密鉴权再加 KDL_TPS_USERNAME / KDL_TPS_PASSWORD；"
                    "可选 KDL_TPS_BACKUP_HOST 做主备切换"
                )
            snap = proxy_pool.snapshot()
            print(f"[proxy] kdl TPS tunnel: {snap['tunnel_endpoint']} auth={snap['auth_mode']} backup={snap['has_backup']}")
        elif args.proxy_product == "fps":
            proxy_pool = KuaidailiFPSTunnel.from_env(
                protocol=args.proxy_protocol,
                verbose=args.proxy_verbose,
            )
            if proxy_pool is None:
                raise SystemExit(
                    "--proxy-product fps 需要 KDL_FPS_HOST + KDL_FPS_PORT 环境变量；"
                    "账密鉴权再加 KDL_FPS_USERNAME / KDL_FPS_PASSWORD"
                )
            snap = proxy_pool.snapshot()
            print(f"[proxy] kdl FPS tunnel: {snap['tunnel_endpoint']} auth={snap['auth_mode']}")
        else:
            proxy_pool = KuaidailiProxyPool.from_env(
                pool_size=args.proxy_pool_size,
                ip_lifetime_sec=args.proxy_ip_lifetime,
                bad_cooldown_sec=args.proxy_bad_cooldown,
                max_uses_per_ip=args.proxy_max_uses_per_ip,
                area=args.proxy_area,
                protocol=args.proxy_protocol,
                verbose=args.proxy_verbose,
            )
            if proxy_pool is None:
                raise SystemExit(
                    "--proxy-product dps 需要 KDL_SECRET_ID / KDL_SIGNATURE 环境变量\n"
                    "（账密鉴权需同时设置 KDL_USERNAME / KDL_PASSWORD；白名单鉴权不需要）"
                )
            balance = proxy_pool.get_balance()
            print(f"[proxy] kdl DPS pool: today_balance={balance}, pool_size={args.proxy_pool_size}")

    client = SkadaApiClient(timeout=args.timeout, proxy_pool=proxy_pool)
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
        "use_proxy_pool": bool(args.use_proxy_pool),
        "proxy_product": args.proxy_product if args.use_proxy_pool else None,
        "proxy_pool_size": args.proxy_pool_size if args.use_proxy_pool and args.proxy_product == "dps" else None,
        "proxy_ip_lifetime": args.proxy_ip_lifetime if args.use_proxy_pool and args.proxy_product == "dps" else None,
    }

    try:
        if not args.details_only:
            known_ids_for_stop: set[int] = set()
            if args.stop_on_known:
                known_ids_for_stop = load_existing_detail_ids(out_dir / "details")
                known_ids_for_stop |= load_permanently_failed_ids(out_dir / "state" / "detail_errors.jsonl")
                # 把 pages 里已经见过的所有 run_id 也算 known（含 filter 不通过的），
                # 否则 filter-out 的 run_id 永远"unknown"，stop-on-known 永远不触发
                known_ids_for_stop |= set(collect_run_ids_from_pages(out_dir / "pages"))
                # 支持任意 pages_*_backup/ 目录作为历史 run_id 源（sort 切换时 mv 的备份）
                for backup_dir in out_dir.glob("pages_*_backup"):
                    if backup_dir.is_dir():
                        before = len(known_ids_for_stop)
                        known_ids_for_stop |= set(collect_run_ids_from_pages(backup_dir))
                        print(f"[stop-on-known] 合并备份 {backup_dir.name}: +{len(known_ids_for_stop) - before}")
                print(f"[stop-on-known] 加载 {len(known_ids_for_stop)} 个已知 run_id")
            summary["pages"] = crawl_page_lists(
                client,
                out_dir,
                page_size=args.page_size,
                sort_by=args.sort_by,
                sort_dir=args.sort_dir,
                page_workers=args.page_workers,
                page_qps=args.page_qps,
                max_pages=args.max_pages,
                outcome=args.outcome,
                stop_on_known=bool(args.stop_on_known),
                known_run_ids=known_ids_for_stop,
            )

        if not args.pages_only:
            if args.filter_preset == "failure":
                filter_fn = make_failure_filter(min_created_at=args.min_created_at)
            else:
                filter_fn = FILTER_PRESETS.get(args.filter_preset) if args.filter_preset else None
            if filter_fn is not None:
                print(f"[filter] 启用 preset={args.filter_preset} min_created_at={args.min_created_at}")
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
                filter_fn=filter_fn,
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
    except ProxyAuthConfigError as exc:
        summary["proxy_auth_error"] = str(exc)
        print(f"[main] proxy auth config error:\n{exc}")
        summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        summary["elapsed_sec"] = round(time.time() - started_at, 2)
        if proxy_pool is not None:
            summary["proxy_pool_metrics"] = proxy_pool.snapshot()
        save_json(out_dir / "meta" / "crawl_summary.json", summary)
        raise SystemExit(2)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["elapsed_sec"] = round(time.time() - started_at, 2)
    if proxy_pool is not None:
        summary["proxy_pool_metrics"] = proxy_pool.snapshot()
    save_json(out_dir / "meta" / "crawl_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
