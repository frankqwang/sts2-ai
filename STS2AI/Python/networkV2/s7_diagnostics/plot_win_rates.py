"""Plot win rates per encounter × iter, plus room_type aggregates.

用法：
  python -m networkV2.s7_diagnostics.plot_win_rates runs/co6 --out /tmp/co6_wr.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gather(dump_dir: Path):
    """Returns:
        iters: sorted list of iter numbers
        per_enc: {encounter_id: {iter: (wins, total, avg_hp_loss)}}
        per_rt:  {room_type: {iter: (wins, total)}}
    """
    per_enc: dict[str, dict[int, tuple]] = defaultdict(dict)
    per_rt: dict[str, dict[int, tuple]] = defaultdict(dict)
    iters = []
    for p in sorted(dump_dir.glob("iter*_episodes.jsonl")):
        it = int(p.stem[4:8])
        iters.append(it)
        enc_stats = defaultdict(lambda: [0, 0, 0])  # [wins, total, hp_loss_sum]
        rt_stats = defaultdict(lambda: [0, 0])
        for line in p.open(encoding="utf-8"):
            try:
                e = json.loads(line)
            except Exception:
                continue
            enc = e.get("encounter_id", "")
            rt = e.get("room_type", "")
            won = e.get("outcome") == "victory"
            hp_loss = e.get("hp_loss", 0)
            if enc:
                enc_stats[enc][0] += int(won)
                enc_stats[enc][1] += 1
                enc_stats[enc][2] += hp_loss
            if rt:
                rt_stats[rt][0] += int(won)
                rt_stats[rt][1] += 1
        for enc, (w, t, h) in enc_stats.items():
            per_enc[enc][it] = (w, t, h / max(t, 1))
        for rt, (w, t) in rt_stats.items():
            per_rt[rt][it] = (w, t)
    return sorted(iters), per_enc, per_rt


def plot_all(dump_dir: Path, out: Path) -> None:
    iters, per_enc, per_rt = gather(dump_dir)
    if not iters:
        print(f"no data in {dump_dir}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Win Rate Analysis: {dump_dir}", fontsize=14, fontweight="bold")

    # 1) Per-encounter win rate over iters
    ax = axes[0, 0]
    for enc in sorted(per_enc):
        xs, ys = [], []
        for it in iters:
            if it in per_enc[enc]:
                w, t, _ = per_enc[enc][it]
                xs.append(it)
                ys.append(100 * w / max(t, 1))
        ax.plot(xs, ys, marker="o", label=enc, markersize=4, alpha=0.8)
    ax.set_xlabel("Iter")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win rate per encounter over time")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(-5, 105)

    # 2) Per-encounter AVG HP loss
    ax = axes[0, 1]
    for enc in sorted(per_enc):
        xs, ys = [], []
        for it in iters:
            if it in per_enc[enc]:
                _, _, hpl = per_enc[enc][it]
                xs.append(it)
                ys.append(hpl)
        ax.plot(xs, ys, marker="o", label=enc, markersize=4, alpha=0.8)
    ax.set_xlabel("Iter")
    ax.set_ylabel("Avg HP Loss")
    ax.set_title("Damage taken per encounter")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # 3) Per-room_type win rate
    ax = axes[1, 0]
    colors = {"monster": "#3498db", "elite": "#f39c12", "boss": "#e74c3c"}
    for rt in ["monster", "elite", "boss"]:
        if rt not in per_rt:
            continue
        xs, ys = [], []
        for it in iters:
            if it in per_rt[rt]:
                w, t = per_rt[rt][it]
                xs.append(it)
                ys.append(100 * w / max(t, 1))
        ax.plot(xs, ys, marker="o", label=rt, color=colors.get(rt, "gray"),
                linewidth=2, markersize=5)
    ax.set_xlabel("Iter")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win rate by room type")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 105)

    # 4) Rolling-mean aggregate win rate + episode counts per encounter
    ax = axes[1, 1]
    # Overall + rolling avg
    overall_wr = []
    for it in iters:
        total_w, total_t = 0, 0
        for stats in per_rt.values():
            if it in stats:
                w, t = stats[it]
                total_w += w
                total_t += t
        overall_wr.append((it, 100 * total_w / max(total_t, 1), total_t))

    xs = [x[0] for x in overall_wr]
    ys = [x[1] for x in overall_wr]
    ax.plot(xs, ys, marker="o", label="overall wr%", color="black", linewidth=2)
    if len(ys) >= 5:
        win = 5
        smooth = []
        for i in range(len(ys)):
            lo = max(0, i - win + 1)
            smooth.append(sum(ys[lo:i+1]) / (i - lo + 1))
        ax.plot(xs, smooth, linestyle="--", label=f"{win}-iter avg", color="red", alpha=0.7)
    ax.set_xlabel("Iter")
    ax.set_ylabel("Overall Win Rate (%)")
    ax.set_title("Overall win rate + smoothed")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 105)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved: {out}")

    # Print summary table
    print()
    print(f"=== Per-encounter final stats (last iter {iters[-1]}) ===")
    print(f"{'encounter':<28} {'iters':>7} {'last_wr':>8} {'best_wr':>8}")
    for enc in sorted(per_enc):
        stats = per_enc[enc]
        last = stats.get(iters[-1])
        last_wr = 100 * last[0] / max(last[1], 1) if last else 0
        best_wr = max(100 * w/max(t,1) for it,(w,t,_) in stats.items())
        print(f"  {enc:<26} {len(stats):>7} {last_wr:>7.1f}% {best_wr:>7.1f}%")

    print()
    print(f"=== Per-room_type aggregate (last iter {iters[-1]}) ===")
    print(f"{'room_type':<10} {'iters':>7} {'last_wr':>8} {'best_wr':>8}")
    for rt in sorted(per_rt):
        stats = per_rt[rt]
        last = stats.get(iters[-1])
        last_wr = 100 * last[0] / max(last[1], 1) if last else 0
        best_wr = max(100 * w/max(t,1) for it,(w,t) in stats.items())
        print(f"  {rt:<8} {len(stats):>7} {last_wr:>7.1f}% {best_wr:>7.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="输出图路径；默认写到 <dump_dir>/analysis/win_rate_curves.png "
                         "（遵循 docs/design/DIAGNOSTICS_CONVENTION.md）")
    args = ap.parse_args()
    if args.out is None:
        args.out = args.dump_dir / "analysis" / "win_rate_curves.png"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_all(args.dump_dir, args.out)


if __name__ == "__main__":
    main()
