"""Live training monitor：定期扫 dump 目录给出滚动趋势 + 异常告警。

用法：
  python -m networkV2.s7_diagnostics.live_monitor runs/long1 [--interval 30]

输出（每 interval 秒刷新）：
  - Iter / Time / Eps / W/L / AvgFlr 最近 N 轮趋势
  - KL / policy_loss / value_loss / nan_skip 趋势
  - 房间类型分布 + 动作分布
  - 异常告警（KL > 0.5 / nan_skip > 20% / win_rate 长期 0）

不打断训练，只读 dump 目录文件。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def _read_metrics(root: Path) -> list[dict]:
    """读所有 iterNNNN_metrics.json，按 iter 升序。"""
    out = []
    for p in sorted(root.glob("iter*_metrics.json")):
        try:
            with p.open(encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            pass
    return out


def _read_episodes(root: Path, iteration: int) -> list[dict]:
    p = root / f"iter{iteration:04d}_episodes.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _read_samples(root: Path, iteration: int) -> list[dict]:
    p = root / f"iter{iteration:04d}_samples.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _fmt_trend(values: list[float], width: int = 30) -> str:
    """简易 ASCII sparkline（Windows 兼容字符）。"""
    if not values:
        return ""
    chars = ".-=+*#@%$"  # 9 levels, all ASCII
    lo, hi = min(values), max(values)
    rng = hi - lo if hi > lo else 1.0
    pts = values[-width:]
    return "".join(chars[min(8, max(0, int((v - lo) / rng * 8)))] for v in pts)


def _summary_line(m: dict) -> str:
    it = m["iteration"]
    mt = m["metrics"]
    n = m["n_samples"]
    extra = m.get("extra", {})
    wins = extra.get("total_wins", 0)
    runs = extra.get("total_runs", 0)
    wt = extra.get("wall_time_s", 0.0)
    pl = mt.get("policy_loss", 0.0)
    ncpl = mt.get("nc_policy_loss", 0.0)
    kl = mt.get("approx_kl", 0.0)
    nckl = mt.get("nc_approx_kl", 0.0)
    nan = int(mt.get("nan_skip_count", 0))
    n_mb = max(n // 64, 1)
    return (
        f"  iter {it:3d} | n={n:5d} | runs {wins}/{runs} | {wt:5.1f}s | "
        f"pl={pl:.4f} ncpl={ncpl:.4f} | kl={kl:.4f} nckl={nckl:.4f} | "
        f"nan={nan}/{n_mb} ({100*nan/n_mb:.0f}%)"
    )


def _aggregate_episodes(root: Path, iters: list[int]) -> dict:
    """汇总最近 N 轮 episodes：胜率 / 平均楼层 / outcome 分布。"""
    all_eps = []
    for it in iters:
        all_eps.extend(_read_episodes(root, it))
    if not all_eps:
        return {}
    n = len(all_eps)
    wins = sum(1 for e in all_eps if e.get("outcome") == "victory")
    avg_floor = sum(e.get("floor", 0) for e in all_eps) / n
    avg_steps = sum(e.get("steps", 0) for e in all_eps) / n
    avg_combats = sum(e.get("combats", 0) for e in all_eps) / n
    avg_final_hp = sum(e.get("final_hp", 0) for e in all_eps) / n
    outcomes = Counter(e.get("outcome", "?") for e in all_eps)
    return {
        "n": n, "wins": wins, "win_rate": wins / n,
        "avg_floor": avg_floor, "avg_steps": avg_steps, "avg_combats": avg_combats,
        "avg_final_hp": avg_final_hp, "outcomes": dict(outcomes),
    }


def _room_distribution(root: Path, iteration: int) -> dict:
    samples = _read_samples(root, iteration)
    if not samples:
        return {}
    rooms = Counter(s.get("room_type", "?") for s in samples)
    domains = Counter(s.get("decision_domain", "?") for s in samples)
    return {"rooms": dict(rooms), "domains": dict(domains), "n": len(samples)}


def _alerts(metrics: list[dict]) -> list[str]:
    """生成异常告警列表。"""
    alerts = []
    if not metrics:
        return alerts
    recent = metrics[-5:]
    # KL 持续高
    kls = [m["metrics"].get("approx_kl", 0.0) for m in recent]
    if kls and sum(kls) / len(kls) > 0.5:
        alerts.append(f"WARN: 近 5 iter avg KL={sum(kls)/len(kls):.3f} > 0.5（policy 抖动大）")
    # NaN 比例高
    nan_rates = []
    for m in recent:
        n = m["n_samples"]
        nan = m["metrics"].get("nan_skip_count", 0)
        if n > 0:
            nan_rates.append(nan / max(n // 64, 1))
    if nan_rates and sum(nan_rates) / len(nan_rates) > 0.2:
        alerts.append(f"WARN: 近 5 iter avg NaN 比例 {100*sum(nan_rates)/len(nan_rates):.1f}% > 20%")
    # win_rate 长期 0
    if len(metrics) >= 20:
        recent_runs = sum(m.get("extra", {}).get("total_runs", 0) for m in metrics[-20:])
        recent_wins = sum(m.get("extra", {}).get("total_wins", 0) for m in metrics[-20:])
        if recent_runs >= 200 and recent_wins == 0:
            alerts.append(f"WARN: 近 20 iter ({recent_runs} runs) 0 wins —— policy 可能学不动")
    return alerts


def render_dashboard(root: Path) -> str:
    metrics = _read_metrics(root)
    if not metrics:
        return f"[{time.strftime('%H:%M:%S')}] {root}: no data yet"

    n_iter = len(metrics)
    last_iter = metrics[-1]["iteration"]
    show_n = min(10, n_iter)
    recent_iters = [m["iteration"] for m in metrics[-show_n:]]

    lines = []
    lines.append(f"=== {time.strftime('%H:%M:%S')}  {root}  iter {last_iter} ===")

    # 趋势 sparklines（最近 30 iter）
    win = 30
    trend = metrics[-win:]
    pls = [m["metrics"].get("policy_loss", 0.0) for m in trend]
    kls = [m["metrics"].get("approx_kl", 0.0) for m in trend]
    vls = [m["metrics"].get("value_loss", 0.0) for m in trend]
    floors = [m.get("extra", {}).get("total_runs", 0) and
              sum(_read_episodes(root, m["iteration"])[0].get("floor", 0) for _ in [0])
              for m in trend]  # placeholder
    # 重新计算 avg floor per iter
    floors = []
    for m in trend:
        eps = _read_episodes(root, m["iteration"])
        if eps:
            floors.append(sum(e.get("floor", 0) for e in eps) / len(eps))
        else:
            floors.append(0.0)

    lines.append(f"  policy_loss [{min(pls):.3f}..{max(pls):.3f}]: {_fmt_trend(pls)}")
    lines.append(f"  approx_kl   [{min(kls):.3f}..{max(kls):.3f}]: {_fmt_trend(kls)}")
    lines.append(f"  value_loss  [{min(vls):.3f}..{max(vls):.3f}]: {_fmt_trend(vls)}")
    lines.append(f"  avg_floor   [{min(floors):.1f}..{max(floors):.1f}]: {_fmt_trend(floors)}")

    lines.append("")
    lines.append(f"  Recent {show_n} iter:")
    for m in metrics[-show_n:]:
        lines.append(_summary_line(m))

    # Room distribution (latest iter)
    rd = _room_distribution(root, last_iter)
    if rd:
        rooms_str = ", ".join(f"{k}={v}" for k, v in sorted(rd["rooms"].items(), key=lambda x: -x[1]))
        domains_str = ", ".join(f"{k}={v}" for k, v in sorted(rd["domains"].items(), key=lambda x: -x[1]))
        lines.append("")
        lines.append(f"  iter {last_iter} rooms  : {rooms_str}")
        lines.append(f"  iter {last_iter} domains: {domains_str}")

    # Episode aggregate
    agg20 = _aggregate_episodes(root, recent_iters)
    if agg20:
        lines.append("")
        lines.append(f"  Recent {show_n} iter ({agg20['n']} episodes):")
        lines.append(f"    win_rate: {agg20['wins']}/{agg20['n']} = {100*agg20['win_rate']:.1f}%")
        lines.append(f"    avg_floor={agg20['avg_floor']:.2f}  avg_steps={agg20['avg_steps']:.1f}  "
                     f"avg_combats={agg20['avg_combats']:.2f}  avg_final_hp={agg20['avg_final_hp']:.1f}")
        lines.append(f"    outcomes: {agg20['outcomes']}")

    # Alerts
    alerts = _alerts(metrics)
    if alerts:
        lines.append("")
        lines.append("  ALERTS:")
        for a in alerts:
            lines.append(f"    ! {a}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--interval", type=int, default=30, help="刷新秒数（默认 30）")
    ap.add_argument("--once", action="store_true", help="只输出一次")
    args = ap.parse_args()

    if not args.dump_dir.exists():
        print(f"目录不存在: {args.dump_dir}")
        return

    while True:
        try:
            print("\033[2J\033[H", end="")  # clear screen
            print(render_dashboard(args.dump_dir))
        except Exception as e:
            print(f"render error: {e}")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
