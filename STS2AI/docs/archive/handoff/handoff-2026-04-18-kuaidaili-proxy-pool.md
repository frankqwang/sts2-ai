# handoff 2026-04-18 — skada 爬虫接入快代理 DPS 代理池

## 背景

`STS2AI/Python/data/skada/crawl_skada_runs_raw.py` 与 `crawl_skada_runs_daemon.py`
原本所有请求都走本机直连到 `sts2log.com`，一旦遇到 429 就整个任务进入全局
`cooldown_until` 睡一段才能恢复，并发度越高越容易被封出口 IP。

目标：接入快代理私密代理（DPS）IP 池，让 429 只是「换个 IP」的小事件，
而不是整个任务停摆。

参考官方文档：
- 产品主页：https://www.kuaidaili.com/doc/product/dps/
- 开发手册：https://www.kuaidaili.com/doc/product/dev/dps/
- 获取 IP：https://www.kuaidaili.com/doc/product/api/getdps/
- 检测有效性：https://www.kuaidaili.com/doc/product/api/checkdpsvalid/
- 剩余时长：https://www.kuaidaili.com/doc/product/api/getdpsvalidtime/
- 订单余额：https://www.kuaidaili.com/doc/product/api/getipbalance/

## 两套鉴权概念（容易混淆）

| 概念 | 作用对象 | 方式 |
|------|---------|------|
| **API 鉴权** | 调快代理接口（如 getdps）| `token`（固定 secret_token）或 `hmacsha1`（用 SecretKey 对每次请求签名）|
| **代理鉴权** | 拿到 IP 后访问目标站时，代理对你的鉴权 | 白名单（出口 IP 预加白，免密码）或账密（`user:pwd@ip:port`）|

两组互不相干，都要配。其中：
- **API 鉴权** 要么用订单后台直接复制的「API 签名（secret_token）」走 token 模式，
  要么用「SecretKey」走 hmacsha1 模式。SecretKey 是算签名的密钥，不能直接当
  signature 传。
- **代理鉴权** 账密和白名单互斥，不要同时配（官方明确说明会共享白名单配额）。

## 环境变量（密钥只能走 env，不进 CLI 参数）

```bash
# 必填
export KDL_SECRET_ID=xxxxxxxxxxxxxxxxxxx

# 二选一：
# A) token 模式（推荐简单上手）
export KDL_SIGN_TYPE=token            # 默认值，可省
export KDL_SIGNATURE=oxf0n0g59h7wcdyvz2uo68ph2s

# B) hmacsha1 模式（每次请求重算签名，防篡改）
export KDL_SIGN_TYPE=hmacsha1
export KDL_SECRET_KEY=u5auqjz4yr756f8akwbbok05e4sr5vy2

# 代理鉴权二选一：
# A) 白名单：不设置用户名密码，把当前机器出口 IP 加入快代理后台白名单
# B) 账密：
export KDL_USERNAME=xxx
export KDL_PASSWORD=yyy
```

**你手上那串 `u5auqjz4yr756f8akwbbok05e4sr5vy2` 应该是 SecretKey 还是 API 签名？**
在订单页面对照：
- 如果订单页标的是「**SecretKey**」→ 走 hmacsha1，设 `KDL_SECRET_KEY`
- 如果订单页标的是「**API 签名 / signature / secret_token**」→ 走 token，设 `KDL_SIGNATURE`

> 今后不要再把真实密钥直接贴到聊天里，写到 env 或 `.env` 文件（别 commit）即可。

## 启用代理池（爬虫命令）

```bash
cd STS2AI/Python/data/skada

# 单进程模式（带代理池）
python crawl_skada_runs_raw.py \
  --out-dir runs \
  --detail-workers 12 \
  --detail-qps 6 \
  --use-proxy-pool \
  --proxy-pool-size 40 \
  --proxy-ip-lifetime 300 \
  --proxy-bad-cooldown 60 \
  --proxy-verbose

# 守护进程模式（推荐，pages / details 两个子任务各挂一份）
python crawl_skada_runs_daemon.py \
  --detail-workers 12 \
  --detail-qps 6 \
  --use-proxy-pool \
  --proxy-pool-size 40 \
  --proxy-ip-lifetime 300
```

> `--proxy-pool-size` 建议 ≥ 3 × `--detail-workers`。单 IP 官方建议 ≤ 1 QPS，
> 池子越大，单 IP 压力越小、被目标站封的概率越低。

## 行为变化

未启用代理池（默认）：维持原行为，429 会全局 cooldown。

启用 `--use-proxy-pool` 后：
- `SkadaApiClient.get_json` 每次请求从池子 `acquire()` 一个 IP，proxies 传给
  `requests.get`（不污染 session.proxies，thread-local session 仍复用连接）。
- 收到 429 或连接异常 → `pool.mark_bad(ip)` 后**立即换 IP 重试**，而不是全局 cooldown。
  重试上限提高到 12 次（见 `DEFAULT_PROXY_MAX_RETRIES`）。
- 池子空了/不够，会调 `getdps` 拉一批（5s 最小间隔节流，避免撞 API 的 10QPS 限制）。
- 代理池指标（已拉/已 bad/已复活/今日余额）会打到 `meta/crawl_summary.json`
  的 `proxy_pool_metrics` 字段。

## 模块文件一览

| 文件 | 作用 |
|------|------|
| `STS2AI/Python/data/skada/kuaidaili_proxy_pool.py` | 代理池核心：`KuaidailiProxyPool` + `ProxyTicket`，线程安全，支持 token/hmacsha1 两种签名 |
| `STS2AI/Python/data/skada/crawl_skada_runs_raw.py` | 原爬虫，新增 `--use-proxy-pool` 等 CLI；`SkadaApiClient` 接 pool |
| `STS2AI/Python/data/skada/crawl_skada_runs_daemon.py` | 守护脚本，新增同名 CLI 透传给两个子任务 |

## 代理池 API 摘要（脚本外也能直接用）

```python
from kuaidaili_proxy_pool import KuaidailiProxyPool

pool = KuaidailiProxyPool.from_env(pool_size=40, ip_lifetime_sec=300)

ticket = pool.acquire()
if ticket is None:
    # 池子拉不到 IP，退路：直连或抛错
    ...
else:
    try:
        resp = requests.get(url, proxies=ticket.proxies, timeout=10)
        resp.raise_for_status()
    except Exception:
        pool.mark_bad(ticket.ip_port)   # 冷却这个 IP，下次 acquire 不再返回
        raise
    else:
        pool.release(ticket, succeeded=True)

# 运维方法：
pool.get_balance()          # 今日剩余 IP 提取配额
pool.refresh_lifetimes()    # 调 getdpsvalidtime 校正精确剩余时间
pool.revive_bad()           # 调 checkdpsvalid 把 bad 列表里仍有效的 IP 放回来
pool.snapshot()             # 全量 metrics
```

## 常见问题排查

1. **启动时报 `--use-proxy-pool 需要先设置 KDL_SECRET_ID / KDL_SIGNATURE`**
   → env 没设置或 sign_type 选错。hmacsha1 模式需要 `KDL_SECRET_KEY`，不能光配 `KDL_SIGNATURE`。

2. **`[kdl_pool] fetch failed: kdl api /getdps error: code=-6 ...`**
   → 触发 IP 白名单限制。解决：把当前出口 IP 加入快代理后台白名单，或改走账密鉴权
   （设 `KDL_USERNAME` / `KDL_PASSWORD`，同时清空白名单配置）。

3. **`code=1` 今日提取余额已用尽**
   → 增加 `--proxy-ip-lifetime`（延长单 IP 使用时间）或缩小 `--proxy-pool-size`。

4. **`code=-15` 订单过期** → 续订。

5. **大量 `[proxy_err] attempt=...`** → 可能是代理质量差或目标站封得快。
   排查步骤：
   - 先跑 `pool.get_balance()` 确认还有配额
   - 跑 `pool.refresh_lifetimes()` 看 IP 是否已经过期
   - 降低 `--detail-qps` 或增大 `--proxy-pool-size`

6. **守护脚本启动时不看见 `[proxy] kdl pool enabled`**
   → 说明 `--use-proxy-pool` flag 没传到子任务。检查 `crawl_skada_daemon_pages_stdout.log`
   / `..._details_stdout.log` 头一行 JSON 里 `use_proxy_pool` 字段。

## 未接入的可选增强（留给后续）

- 拉 IP 时带上 `f_et=1` 让服务端直接返回可用秒数，用真实值替代本地预设 lifetime
- `--proxy-auto-revive`：周期性跑 `pool.revive_bad()`
- 对 `refresh_lifetimes()` 的后台线程化（当前是同步调用）
