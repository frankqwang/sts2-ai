"""HealthMonitor — 用独立 PipeConnection 做后台保活。

**为什么独立连接**:
  PipeClient 底层 Win32 overlapped I/O 的 event/handle 不能跨线程共享
  (两个线程并发 ReadFile/WriteFile 会 silent 数据错乱)。以前的实现让
  heartbeat 和业务 thread 共享同一个 PipeClient + RLock,但每 iter 新 thread
  带来 thread-id 不一致问题。

  正解:**heartbeat 用自己独立的 PipeConnection 连到同一个 port**。
  Windows Named Pipe server 允许多个 client 连(MAX_INSTANCES),sim 侧
  完全支持。两个连接互不干扰,heartbeat 挂 business 不受影响。

## 使用

    from networkV2.s0_bridge.transport import PipeConnection, HealthMonitor

    # 业务连接
    biz_conn = PipeConnection(cfg)
    biz_conn.connect()

    # 独立 heartbeat
    monitor = HealthMonitor(cfg, interval_s=5.0, idle_threshold_s=4.0)
    monitor.start()
    # ...业务用 biz_conn...
    monitor.stop()
    biz_conn.close()
"""
from __future__ import annotations

import logging
import threading
import time

from networkV2.s0_bridge.transport.connection import PipeConnection, PipeConnectionConfig

logger = logging.getLogger(__name__)


class HealthMonitor:
    """独立连接的 pipe 后台保活线程。

    连接策略: 启动时 connect 自己独立 PipeConnection;stop() 时 close。
    失败策略: heartbeat 失败只记 log + count,不 propagate(保持后台运行)。
    业务 PipeConnection 和 HealthMonitor 的 PipeConnection **独立**,互不干扰。
    """

    def __init__(
        self,
        config: PipeConnectionConfig,
        *,
        interval_s: float = 5.0,
        idle_threshold_s: float = 4.0,
        ping_method: str = "state",
    ):
        self.cfg = config
        self.interval_s = float(interval_s)
        self.idle_threshold_s = float(idle_threshold_s)
        self.ping_method = ping_method
        self._conn: PipeConnection | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_ping_ok = True
        self._fail_count = 0
        self._ping_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"heartbeat-{self.cfg.port}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=join_timeout_s)
            except Exception:
                pass
            self._thread = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    @property
    def stats(self) -> dict:
        return {
            "ping_count": self._ping_count,
            "fail_count": self._fail_count,
            "last_ok": self._last_ping_ok,
        }

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> None:
        if self._conn is not None and self._conn.is_connected():
            return
        # heartbeat 自己拉不动 sim(业务已经拉了),这里只 connect 不 auto_launch
        cfg = PipeConnectionConfig(
            port=self.cfg.port,
            pipe_name_prefix=self.cfg.pipe_name_prefix,
            protocol=self.cfg.protocol,
            connect_timeout_s=3.0,
            default_call_timeout_s=3.0,
            auto_launch=False,
        )
        self._conn = PipeConnection(cfg)
        self._conn.connect()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._ensure_conn()
                if self._conn is not None and self._conn.ping(self.ping_method):
                    self._ping_count += 1
                    self._last_ping_ok = True
                else:
                    self._fail_count += 1
                    self._last_ping_ok = False
                    # ping 失败下次循环重连
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
            except Exception as e:
                self._fail_count += 1
                self._last_ping_ok = False
                if self._fail_count == 1 or self._fail_count % 20 == 0:
                    logger.warning(
                        f"[heartbeat port={self.cfg.port}] failed "
                        f"(count={self._fail_count}): {type(e).__name__}: {str(e)[:80]}"
                    )
                # 失败时 drop conn,下轮重连
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
