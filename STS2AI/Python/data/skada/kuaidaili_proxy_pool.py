#!/usr/bin/env python3
"""快代理（Kuaidaili）私密代理（DPS）代理池。

职责：
- 通过 getdps 接口批量拉取代理 IP
- 为并发线程分发代理（轮询 + 失败标记 + 过期淘汰 + 自动续拉）
- 用 getdpsvalidtime 校正精确剩余时长（可选）
- 用 checkdpsvalid 复活 bad 列表（可选）
- 用 getipbalance 暴露今日剩余配额到 metrics（可选）

两套概念互不相干，常搞混：

(A) 取 IP 的 API 鉴权（getdps / checkdpsvalid / ...）— 两种签名方式：
  - token: signature 参数 = 订单后台生成的「API 签名」(即 secret_token)，直接传
  - hmacsha1: 用 SecretKey 对请求做 HMAC-SHA1，每次请求重新计算

(B) 访问业务网站时对代理 IP 的鉴权 — 两种方式（互斥）：
  - 白名单: 出口机器 IP 加入快代理白名单，取出来的 IP 直接 http://ip:port
  - 账密: 订单内的用户名/密码，取出来的 IP 用 http://user:pwd@ip:port

环境变量：
- KDL_SECRET_ID    订单 SecretId（必填）
- KDL_SIGN_TYPE    token | hmacsha1（默认 token）
- KDL_SIGNATURE    token 模式下是订单 API 签名；hmacsha1 模式留空
- KDL_SECRET_KEY   hmacsha1 模式的 SecretKey（hmacsha1 必填）
- KDL_USERNAME     代理账密鉴权的用户名（可选）
- KDL_PASSWORD     代理账密鉴权的密码（可选）

参考：
- https://www.kuaidaili.com/doc/product/dps/
- https://www.kuaidaili.com/doc/product/api/getdps/
- https://www.kuaidaili.com/doc/product/api/checkdpsvalid/
- https://www.kuaidaili.com/doc/product/api/getdpsvalidtime/
- https://www.kuaidaili.com/doc/product/api/getipbalance/
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests


DEFAULT_BASE_URL = "https://dps.kdlapi.com/api"
DEFAULT_POOL_SIZE = 20
DEFAULT_IP_LIFETIME_SEC = 300.0
DEFAULT_FETCH_MIN_INTERVAL_SEC = 5.0
DEFAULT_BAD_COOLDOWN_SEC = 60.0
DEFAULT_MAX_USES_PER_IP = 0  # 0 = 不限；>0 则用到该次数后自动淘汰
EXPIRE_SAFETY_MARGIN_SEC = 5.0
DEFAULT_FETCH_TIMEOUT = 10.0


@dataclass
class ProxyEntry:
    ip_port: str
    fetched_at: float
    lifetime_sec: float
    last_used_at: float = 0.0
    used_count: int = 0
    fail_count: int = 0

    @property
    def expire_at(self) -> float:
        return self.fetched_at + self.lifetime_sec

    def is_expired(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return now >= (self.expire_at - EXPIRE_SAFETY_MARGIN_SEC)

    def remaining_sec(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, self.expire_at - now)


class KuaidailiProxyPool:
    """线程安全的快代理 DPS 代理池。

    典型用法：

        pool = KuaidailiProxyPool.from_env(pool_size=30, ip_lifetime_sec=300)
        ticket = pool.acquire()
        if ticket is not None:
            try:
                resp = requests.get(url, proxies=ticket.proxies, timeout=10)
            except Exception:
                pool.mark_bad(ticket.ip_port)
                raise
            else:
                pool.release(ticket)
    """

    def __init__(
        self,
        *,
        secret_id: str,
        signature: str | None = None,
        secret_key: str | None = None,
        sign_type: str = "token",
        username: str | None = None,
        password: str | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        ip_lifetime_sec: float = DEFAULT_IP_LIFETIME_SEC,
        base_url: str = DEFAULT_BASE_URL,
        fetch_min_interval_sec: float = DEFAULT_FETCH_MIN_INTERVAL_SEC,
        bad_cooldown_sec: float = DEFAULT_BAD_COOLDOWN_SEC,
        fetch_timeout: float = DEFAULT_FETCH_TIMEOUT,
        max_uses_per_ip: int = DEFAULT_MAX_USES_PER_IP,
        protocol: str = "http",
        area: str | None = None,
        dedup: bool = True,
        extra_params: dict[str, Any] | None = None,
        verbose: bool = False,
    ):
        sign_type = (sign_type or "token").lower()
        if sign_type not in {"token", "hmacsha1"}:
            raise ValueError(f"sign_type 只支持 token / hmacsha1，收到 {sign_type!r}")
        if not secret_id:
            raise ValueError("快代理 secret_id 必填（见环境变量 KDL_SECRET_ID）")
        if sign_type == "token" and not signature:
            raise ValueError("sign_type=token 时必须提供 signature（见环境变量 KDL_SIGNATURE，值即订单 API 签名/secret_token）")
        if sign_type == "hmacsha1" and not secret_key:
            raise ValueError("sign_type=hmacsha1 时必须提供 secret_key（见环境变量 KDL_SECRET_KEY）")
        if (username and not password) or (password and not username):
            raise ValueError("username / password 必须同时提供或同时为空")

        self.secret_id = secret_id
        self.signature = signature
        self.secret_key = secret_key
        self.sign_type = sign_type
        self.username = username
        self.password = password
        self.pool_size = max(int(pool_size), 1)
        self.ip_lifetime_sec = float(ip_lifetime_sec)
        self.base_url = base_url.rstrip("/")
        self.fetch_min_interval_sec = float(fetch_min_interval_sec)
        self.bad_cooldown_sec = float(bad_cooldown_sec)
        self.fetch_timeout = float(fetch_timeout)
        self.max_uses_per_ip = max(int(max_uses_per_ip), 0)
        self.protocol = protocol
        self.area = area
        self.dedup = bool(dedup)
        self.extra_params = dict(extra_params or {})
        self.verbose = bool(verbose)

        self._lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._entries: list[ProxyEntry] = []
        self._index: dict[str, ProxyEntry] = {}
        self._cursor = 0
        self._bad: dict[str, float] = {}
        self._last_fetch_ts = 0.0

        self._admin_session = requests.Session()

        self.metrics = {
            "fetches_total": 0,
            "fetch_errors": 0,
            "ips_minted": 0,
            "ips_marked_bad": 0,
            "ips_expired": 0,
            "ips_revived": 0,
            "order_left_count": -1,
            "daily_balance": -1,
        }

    @classmethod
    def from_env(cls, **overrides: Any) -> "KuaidailiProxyPool | None":
        """从环境变量构造；未配置必备 key 则返回 None。

        必备：KDL_SECRET_ID
        token 模式 (默认)：再加 KDL_SIGNATURE
        hmacsha1 模式：KDL_SIGN_TYPE=hmacsha1 + KDL_SECRET_KEY
        """
        secret_id = os.environ.get("KDL_SECRET_ID", "").strip()
        if not secret_id:
            return None
        sign_type = os.environ.get("KDL_SIGN_TYPE", "token").strip().lower() or "token"
        signature = os.environ.get("KDL_SIGNATURE", "").strip() or None
        secret_key = os.environ.get("KDL_SECRET_KEY", "").strip() or None
        if sign_type == "token" and not signature:
            return None
        if sign_type == "hmacsha1" and not secret_key:
            return None
        user = os.environ.get("KDL_USERNAME", "").strip() or None
        pwd = os.environ.get("KDL_PASSWORD", "").strip() or None
        kwargs: dict[str, Any] = {
            "secret_id": secret_id,
            "signature": signature,
            "secret_key": secret_key,
            "sign_type": sign_type,
            "username": user,
            "password": pwd,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    # ---------- 对外：取/还/标记 ----------

    def acquire(self) -> "ProxyTicket | None":
        """取一个代理。若池子空/全部标 bad，触发一次补拉；仍无则返回 None。"""
        self._refill_if_needed()
        now = time.time()
        with self._lock:
            self._purge_locked(now)
            if not self._entries:
                return None
            n = len(self._entries)
            for _ in range(n):
                entry = self._entries[self._cursor % n]
                self._cursor += 1
                if entry.ip_port in self._bad:
                    continue
                if self.max_uses_per_ip > 0 and entry.used_count >= self.max_uses_per_ip:
                    self._bad[entry.ip_port] = now + self.bad_cooldown_sec
                    self.metrics["ips_marked_bad"] += 1
                    continue
                entry.last_used_at = now
                entry.used_count += 1
                return ProxyTicket(
                    ip_port=entry.ip_port,
                    proxies=self._proxies_for(entry.ip_port),
                )
            return None

    def release(self, ticket: "ProxyTicket", *, succeeded: bool = True) -> None:
        """归还代理（成功则清理 fail_count）。"""
        if ticket is None:
            return
        with self._lock:
            entry = self._index.get(ticket.ip_port)
            if entry is None:
                return
            if succeeded:
                entry.fail_count = 0

    def mark_bad(self, ip_port: str, *, cool_down_sec: float | None = None) -> None:
        """把 IP 放入冷却名单，指定时长内不会被 acquire 返回。"""
        if not ip_port:
            return
        cd = float(cool_down_sec if cool_down_sec is not None else self.bad_cooldown_sec)
        with self._lock:
            self._bad[ip_port] = time.time() + cd
            self.metrics["ips_marked_bad"] += 1
            entry = self._index.get(ip_port)
            if entry is not None:
                entry.fail_count += 1
        if self.verbose:
            print(f"[kdl_pool] mark_bad {ip_port} cooldown={cd:.0f}s")

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            alive = sum(
                1
                for e in self._entries
                if not e.is_expired(now) and e.ip_port not in self._bad
            )
            return {
                "alive": alive,
                "total_entries": len(self._entries),
                "bad_count": len(self._bad),
                "last_fetch_at": self._last_fetch_ts,
                **self.metrics,
            }

    # ---------- 运维辅助：检测有效性 / 余额 / 精确时长 ----------

    def check_valid(self, ip_ports: list[str]) -> dict[str, bool]:
        """调 checkdpsvalid 检测给定 IP 是否还活着。"""
        if not ip_ports:
            return {}
        resp = self._admin_get("/checkdpsvalid", {"proxy": ",".join(ip_ports)})
        data = resp.get("data") or {}
        return {str(k): bool(v) for k, v in data.items()}

    def check_valid_time(self, ip_ports: list[str]) -> dict[str, int]:
        """调 getdpsvalidtime 查询每个 IP 剩余秒数。"""
        if not ip_ports:
            return {}
        resp = self._admin_get("/getdpsvalidtime", {"proxy": ",".join(ip_ports)})
        data = resp.get("data") or {}
        out: dict[str, int] = {}
        for k, v in data.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    def refresh_lifetimes(self) -> int:
        """用 getdpsvalidtime 校正池子里每个 IP 的剩余时间；返回更新条数。"""
        with self._lock:
            ips = [e.ip_port for e in self._entries]
        if not ips:
            return 0
        mapping = self.check_valid_time(ips)
        updated = 0
        now = time.time()
        with self._lock:
            for e in self._entries:
                sec = mapping.get(e.ip_port)
                if sec is None:
                    continue
                e.fetched_at = now
                e.lifetime_sec = float(sec)
                updated += 1
        return updated

    def revive_bad(self) -> int:
        """调 checkdpsvalid 把 bad 列表里仍然有效的 IP 放回来；返回复活条数。"""
        with self._lock:
            candidates = list(self._bad.keys())
        if not candidates:
            return 0
        valid = self.check_valid(candidates)
        revived = 0
        with self._lock:
            for ip_port, ok in valid.items():
                if ok and ip_port in self._bad:
                    del self._bad[ip_port]
                    revived += 1
        self.metrics["ips_revived"] += revived
        return revived

    def get_balance(self) -> int:
        """调 getipbalance 查询今日/订单剩余 IP 配额；失败返回 -1。"""
        try:
            resp = self._admin_get("/getipbalance", {})
        except Exception as exc:
            if self.verbose:
                print(f"[kdl_pool] get_balance failed: {exc!r}")
            return -1
        data = resp.get("data") or {}
        try:
            bal = int(data.get("balance", -1))
        except (TypeError, ValueError):
            bal = -1
        self.metrics["daily_balance"] = bal
        return bal

    # ---------- 内部 ----------

    def _proxies_for(self, ip_port: str) -> dict[str, str]:
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        url = f"{self.protocol}://{auth}{ip_port}"
        return {"http": url, "https": url}

    def _purge_locked(self, now: float) -> None:
        if not self._entries:
            return
        alive: list[ProxyEntry] = []
        expired = 0
        for entry in self._entries:
            if entry.is_expired(now):
                self._index.pop(entry.ip_port, None)
                expired += 1
                continue
            alive.append(entry)
        if expired:
            self.metrics["ips_expired"] += expired
        self._entries = alive
        self._bad = {k: v for k, v in self._bad.items() if v > now}

    def _refill_if_needed(self) -> None:
        now = time.time()
        with self._lock:
            alive = [
                e for e in self._entries
                if not e.is_expired(now) and e.ip_port not in self._bad
            ]
            need = max(0, self.pool_size - len(alive))
            if need == 0:
                return
            if self._entries and (now - self._last_fetch_ts) < self.fetch_min_interval_sec:
                return
        if not self._fetch_lock.acquire(blocking=False):
            return
        try:
            now = time.time()
            with self._lock:
                alive = [
                    e for e in self._entries
                    if not e.is_expired(now) and e.ip_port not in self._bad
                ]
                need = max(0, self.pool_size - len(alive))
                if need == 0:
                    return
                if self._entries and (now - self._last_fetch_ts) < self.fetch_min_interval_sec:
                    return
                self._last_fetch_ts = now
            try:
                fresh = self._fetch_new_ips(need)
            except Exception as exc:
                self.metrics["fetch_errors"] += 1
                print(f"[kdl_pool] fetch failed: {exc!r}")
                return
            if not fresh:
                return
            now2 = time.time()
            with self._lock:
                for ip_port in fresh:
                    if ip_port in self._index:
                        entry = self._index[ip_port]
                        entry.fetched_at = now2
                        entry.lifetime_sec = self.ip_lifetime_sec
                        continue
                    entry = ProxyEntry(
                        ip_port=ip_port,
                        fetched_at=now2,
                        lifetime_sec=self.ip_lifetime_sec,
                    )
                    self._entries.append(entry)
                    self._index[ip_port] = entry
                    self.metrics["ips_minted"] += 1
                random.shuffle(self._entries)
            if self.verbose:
                print(f"[kdl_pool] minted {len(fresh)} new ips (pool alive={len(alive) + len(fresh)})")
        finally:
            self._fetch_lock.release()

    def _fetch_new_ips(self, count: int) -> list[str]:
        params: dict[str, Any] = {
            "num": int(count),
            "format": "json",
            "sep": 1,
        }
        if self.dedup:
            params["dedup"] = 1
        if self.area:
            params["area"] = self.area
        params.update(self.extra_params)
        payload = self._admin_get("/getdps", params)
        data = payload.get("data") or {}
        try:
            self.metrics["order_left_count"] = int(data.get("order_left_count", -1))
        except (TypeError, ValueError):
            pass
        proxies = data.get("proxy_list") or []
        out: list[str] = []
        for p in proxies:
            if isinstance(p, dict):
                ip = p.get("ip") or p.get("proxy")
                port = p.get("port")
                if ip and port is not None:
                    out.append(f"{ip}:{port}")
            else:
                s = str(p).strip()
                if s:
                    out.append(s)
        return out

    def _path_for(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        base = self.base_url
        from urllib.parse import urlparse
        base_path = urlparse(base).path or ""
        return base_path + path

    def _sign_params(self, method: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        out = {"secret_id": self.secret_id}
        for k, v in params.items():
            if k in ("signature", "secret_id"):
                continue
            out[k] = v
        if self.sign_type == "token":
            out["signature"] = self.signature or ""
            return out
        # hmacsha1
        out["timestamp"] = str(int(time.time()))
        out["sign_type"] = "hmacsha1"
        full_path = self._path_for(path)
        sorted_keys = sorted(out.keys())
        canonical = "&".join(f"{k}={out[k]}" for k in sorted_keys)
        message = f"{method.upper()}{full_path}?{canonical}"
        digest = hmac.new(
            (self.secret_key or "").encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        out["signature"] = base64.b64encode(digest).decode("utf-8")
        return out

    def _admin_get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        signed = self._sign_params("GET", path, params)
        resp = self._admin_session.get(url, params=signed, timeout=self.fetch_timeout)
        resp.raise_for_status()
        payload = resp.json()
        self.metrics["fetches_total"] += 1
        code = payload.get("code")
        if code is None or int(code) != 0:
            raise RuntimeError(
                f"kdl api {path} error: code={code} msg={payload.get('msg')!r}"
            )
        return payload


@dataclass
class ProxyTicket:
    ip_port: str
    proxies: dict[str, str] = field(default_factory=dict)


class KuaidailiTPSTunnel:
    """快代理 TPS（国内隧道代理，每请求换 IP）。

    相比 FPS：TPS 是**国内 IP 段**，访问国内站延迟低，不需要 Clash；
    按并发请求数计费（默认 10 QPS 上限）。

    环境变量：
    - KDL_TPS_HOST     必填（订单分配的 sXXX.kdltps.com）
    - KDL_TPS_PORT     必填（订单控制台给出的端口）
    - KDL_TPS_USERNAME 账密鉴权用户名
    - KDL_TPS_PASSWORD 账密鉴权密码
    - KDL_TPS_BACKUP_HOST  备用隧道 host（订单分配的备用 sXXX.kdltps.com），主挂了自动切
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str | None = None,
        password: str | None = None,
        backup_host: str | None = None,
        protocol: str = "http",
        verbose: bool = False,
    ):
        if not host or not port:
            raise ValueError("TPS tunnel host / port 必填")
        if (username and not password) or (password and not username):
            raise ValueError("TPS username / password 必须同时提供或同时为空")
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = password
        self.backup_host = backup_host
        self.protocol = protocol
        self.verbose = bool(verbose)

        self._active_host = host
        self._proxies_cached = self._build_proxies(host)

        self.metrics = {
            "requests_via_tunnel": 0,
            "mark_bad_noops": 0,
            "failovers_to_backup": 0,
        }
        self._metrics_lock = threading.Lock()

    @classmethod
    def from_env(cls, **overrides: Any) -> "KuaidailiTPSTunnel | None":
        host = os.environ.get("KDL_TPS_HOST", "").strip()
        port_s = os.environ.get("KDL_TPS_PORT", "").strip()
        if not host or not port_s:
            return None
        try:
            port = int(port_s)
        except ValueError:
            return None
        user = os.environ.get("KDL_TPS_USERNAME", "").strip() or None
        pwd = os.environ.get("KDL_TPS_PASSWORD", "").strip() or None
        backup = os.environ.get("KDL_TPS_BACKUP_HOST", "").strip() or None
        protocol = os.environ.get("KDL_TPS_PROTOCOL", "http").strip() or "http"
        kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "username": user,
            "password": pwd,
            "backup_host": backup,
            "protocol": protocol,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def acquire(self) -> "ProxyTicket | None":
        with self._metrics_lock:
            self.metrics["requests_via_tunnel"] += 1
        return ProxyTicket(ip_port=f"{self._active_host}:{self.port}", proxies=self._proxies_cached)

    def release(self, ticket: "ProxyTicket", *, succeeded: bool = True) -> None:
        pass

    def mark_bad(self, ip_port: str, *, cool_down_sec: float | None = None) -> None:
        with self._metrics_lock:
            self.metrics["mark_bad_noops"] += 1
        if self.backup_host and self._active_host == self.host:
            self._active_host = self.backup_host
            self._proxies_cached = self._build_proxies(self.backup_host)
            with self._metrics_lock:
                self.metrics["failovers_to_backup"] += 1
            print(f"[kdl_tps] failover: {self.host} -> {self.backup_host}")

    def snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                "tunnel_endpoint": f"{self._active_host}:{self.port}",
                "auth_mode": "user_pwd" if self.username else "whitelist",
                "protocol": self.protocol,
                "has_backup": bool(self.backup_host),
                **self.metrics,
            }

    def get_balance(self) -> int:
        return -1

    def _build_proxies(self, host: str) -> dict[str, str]:
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        url = f"{self.protocol}://{auth}{host}:{self.port}"
        return {"http": url, "https": url}


class KuaidailiFPSTunnel:
    """快代理 FPS（海外动态住宅）隧道代理。

    与 DPS 的关键差异：
    - 无需 API 提取 IP：一个固定 host:port 即可，后端自动为每次请求分配新 IP
    - 不消耗"每日 IP 配额"（按流量计费），也不需要池子/淘汰/补拉等概念
    - 每请求换 IP，对"per-IP 滑动窗口"限流几乎免疫

    环境变量（与 DPS 的 KDL_ 区隔，避免串配置）：
    - KDL_FPS_HOST     必填（订单分配的 as.eXXX.kdlfps.com 亚洲区 host）
    - KDL_FPS_PORT     默认 18866
    - KDL_FPS_USERNAME 账密鉴权用户名
    - KDL_FPS_PASSWORD 账密鉴权密码
    - KDL_FPS_PROTOCOL http | socks5，默认 http
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str | None = None,
        password: str | None = None,
        protocol: str = "http",
        verbose: bool = False,
    ):
        if not host:
            raise ValueError("FPS tunnel host 必填")
        if not port:
            raise ValueError("FPS tunnel port 必填")
        if (username and not password) or (password and not username):
            raise ValueError("FPS username / password 必须同时提供或同时为空")
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = password
        self.protocol = protocol
        self.verbose = bool(verbose)

        self._tunnel_endpoint = f"{host}:{self.port}"
        self._proxies_cached = self._build_proxies()

        self.metrics = {
            "requests_via_tunnel": 0,
            "mark_bad_noops": 0,
        }
        self._metrics_lock = threading.Lock()

    @classmethod
    def from_env(cls, **overrides: Any) -> "KuaidailiFPSTunnel | None":
        host = os.environ.get("KDL_FPS_HOST", "").strip()
        port_s = os.environ.get("KDL_FPS_PORT", "18866").strip()
        user = os.environ.get("KDL_FPS_USERNAME", "").strip() or None
        pwd = os.environ.get("KDL_FPS_PASSWORD", "").strip() or None
        protocol = os.environ.get("KDL_FPS_PROTOCOL", "http").strip() or "http"
        if not host or not port_s:
            return None
        try:
            port = int(port_s)
        except ValueError:
            return None
        kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "username": user,
            "password": pwd,
            "protocol": protocol,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def acquire(self) -> "ProxyTicket | None":
        with self._metrics_lock:
            self.metrics["requests_via_tunnel"] += 1
        return ProxyTicket(ip_port=self._tunnel_endpoint, proxies=self._proxies_cached)

    def release(self, ticket: "ProxyTicket", *, succeeded: bool = True) -> None:
        pass

    def mark_bad(self, ip_port: str, *, cool_down_sec: float | None = None) -> None:
        with self._metrics_lock:
            self.metrics["mark_bad_noops"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                "tunnel_endpoint": self._tunnel_endpoint,
                "auth_mode": "user_pwd" if self.username else "whitelist",
                "protocol": self.protocol,
                **self.metrics,
            }

    def get_balance(self) -> int:
        return -1

    def _build_proxies(self) -> dict[str, str]:
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        url = f"{self.protocol}://{auth}{self._tunnel_endpoint}"
        return {"http": url, "https": url}


__all__ = [
    "KuaidailiProxyPool",
    "KuaidailiFPSTunnel",
    "KuaidailiTPSTunnel",
    "ProxyTicket",
    "ProxyEntry",
]
