"""训练日志可视化：从 train_full_run_v2 的 stdout 日志提取关键指标并画图。

用法:
    python -m networkV2.s6_training.plot_training /tmp/slim8w_fixed.log

输出:
    /tmp/train_curves.png  (6 子图：AvgFloor / Steps / Policy Loss / Value Loss / HP Loss / WinRate)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

# 日志每行格式示例:
#  2W |  80 |  1685 | 0/80 |  0.0% |   2.08 | pl=0.135920 vl=15.7219 hp=30.6829 ncp=0.017737 |  21.9s
# 字段：Iter | Eps | Steps | W/L | Cum% | AvgFlr | Losses | Time
_LINE_RE = re.compile(
    r"^\s*(\d+)([W ])\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)/(\d+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|"
    r"\s*pl=([-\d.e]+)\s+vl=([-\d.e]+)\s+hp=([-\d.e]+)"
    r"(?:\s+ncp=([-\d.e]+))?(?:\s+kl=([-\d.e]+))?(?:\s+ep=(\d+))?"
    r"\s*\|\s*([\d.]+)s"
)


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        it, wm, eps, steps, w, l, cum, flr, pl, vl, hp, ncp, kl, ep, t = m.groups()
        rows.append({
            "iter": int(it),
            "warmup": wm == "W",
            "eps": int(eps),
            "steps": int(steps),
            "wins": int(w),
            "losses": int(l),
            "cum_winrate": float(cum),
            "avg_floor": float(flr),
            "policy_loss": float(pl),
            "value_loss": float(vl),
            "hp_loss": float(hp),
            "nc_policy_loss": float(ncp) if ncp else 0.0,
            "approx_kl": float(kl) if kl else 0.0,
            "epochs_done": int(ep) if ep else 0,
            "time_s": float(t),
        })
    return rows


def describe(rows: list[dict]) -> None:
    """文字描述训练趋势。"""
    if not rows:
        print("No data.")
        return

    first, last = rows[0], rows[-1]
    print(f"=== Training Summary ({len(rows)} iterations) ===")
    print()
    print(f"  Episodes total:   {sum(r['eps'] for r in rows)}")
    print(f"  Steps total:      {sum(r['steps'] for r in rows):,}")
    print(f"  Wall time total:  {sum(r['time_s'] for r in rows):.1f}s")
    print(f"  Avg time/iter:    {sum(r['time_s'] for r in rows)/len(rows):.1f}s")
    print()
    print(f"  AvgFloor:  {first['avg_floor']:.2f} -> {last['avg_floor']:.2f} "
          f"(delta: +{last['avg_floor']-first['avg_floor']:.2f})")
    print(f"  Steps/ep:  {first['steps']/max(first['eps'],1):.1f} -> {last['steps']/max(last['eps'],1):.1f}")
    print(f"  WinRate:   {first['cum_winrate']:.1f}% -> {last['cum_winrate']:.1f}%")
    print()
    print("  Latest losses:")
    print(f"    policy={last['policy_loss']:.6f} value={last['value_loss']:.4f} "
          f"hp={last['hp_loss']:.4f} nc_policy={last['nc_policy_loss']:.6f}")

    # Throughput
    total_eps = sum(r["eps"] for r in rows)
    total_time = sum(r["time_s"] for r in rows)
    total_steps = sum(r["steps"] for r in rows)
    print()
    print(f"  Throughput:")
    print(f"    {total_eps/total_time:.2f} ep/s")
    print(f"    {total_steps/total_time:.0f} steps/s")

    # Warmup 分界
    last_warmup = max((r["iter"] for r in rows if r["warmup"]), default=0)
    if last_warmup > 0:
        print()
        print(f"  Value warmup: iter 1-{last_warmup} (policy_coef=0)")


def plot(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        return

    if not rows:
        return

    iters = [r["iter"] for r in rows]
    warmup_until = max((r["iter"] for r in rows if r["warmup"]), default=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Subplot 1: AvgFloor
    ax = axes[0, 0]
    ax.plot(iters, [r["avg_floor"] for r in rows], "o-", color="tab:blue")
    if warmup_until > 0:
        ax.axvspan(0.5, warmup_until + 0.5, alpha=0.15, color="orange", label=f"warmup (iter 1-{warmup_until})")
        ax.legend(fontsize=8)
    ax.set_title("Avg Floor Reached")
    ax.set_xlabel("iteration"); ax.set_ylabel("floor")
    ax.grid(alpha=0.3)

    # Subplot 2: Steps per episode
    ax = axes[0, 1]
    steps_per_ep = [r["steps"] / max(r["eps"], 1) for r in rows]
    ax.plot(iters, steps_per_ep, "o-", color="tab:green")
    if warmup_until > 0:
        ax.axvspan(0.5, warmup_until + 0.5, alpha=0.15, color="orange")
    ax.set_title("Steps per Episode")
    ax.set_xlabel("iteration"); ax.set_ylabel("steps/ep")
    ax.grid(alpha=0.3)

    # Subplot 3: Win rate (cumulative)
    ax = axes[0, 2]
    ax.plot(iters, [r["cum_winrate"] for r in rows], "o-", color="tab:red")
    ax.set_title("Cumulative Win Rate")
    ax.set_xlabel("iteration"); ax.set_ylabel("%")
    ax.grid(alpha=0.3)

    # Subplot 4: Policy loss
    ax = axes[1, 0]
    ax.plot(iters, [r["policy_loss"] for r in rows], "o-", color="tab:blue", label="combat policy")
    ax.plot(iters, [r["nc_policy_loss"] for r in rows], "s-", color="tab:purple", label="nc policy", alpha=0.7)
    if warmup_until > 0:
        ax.axvspan(0.5, warmup_until + 0.5, alpha=0.15, color="orange")
    ax.set_title("Policy Loss")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Subplot 5: Value loss (log scale)
    ax = axes[1, 1]
    vl = [max(r["value_loss"], 1e-6) for r in rows]
    ax.semilogy(iters, vl, "o-", color="tab:orange")
    if warmup_until > 0:
        ax.axvspan(0.5, warmup_until + 0.5, alpha=0.15, color="orange")
    ax.set_title("Value Loss (log)")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3)

    # Subplot 6: HP loss (log scale)
    ax = axes[1, 2]
    hp = [max(r["hp_loss"], 1e-6) for r in rows]
    ax.semilogy(iters, hp, "o-", color="tab:olive")
    if warmup_until > 0:
        ax.axvspan(0.5, warmup_until + 0.5, alpha=0.15, color="orange")
    ax.set_title("HP Loss (log)")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print(f"\nSaved plot to: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", type=str, help="Path to train log")
    p.add_argument("--out", type=str, default="/tmp/train_curves.png")
    args = p.parse_args()

    rows = parse_log(Path(args.log))
    describe(rows)
    plot(rows, Path(args.out))


if __name__ == "__main__":
    main()
