"""对比多个 run 的关键训练指标。

用法：
  python -m networkV2.s7_diagnostics.compare_runs \\
      ../Artifacts/runs/co15 \\
      --compare ../Artifacts/runs/co14_baseline_r1 ../Artifacts/runs/co14_no_traj_r1
      [--out custom.png]

默认输出 `<primary>/analysis/compare_runs.png`（按 DIAGNOSTICS_CONVENTION）。

子图：
  (A) aggregate boss 胜率（smooth）—— 主图，对比 conditional policy 增量
  (B) aggregate elite / monster 胜率
  (C) PPO 健康度：KL & policy_loss
  (D) per-boss 胜率（仅 primary run）—— 3 条 boss 曲线
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networkV2.s7_diagnostics.plot_win_rates import gather


def gather_metrics(dump_dir: Path) -> dict[int, dict]:
    """读 iter*_metrics.json 取 losses / KL / entropy。"""
    out: dict[int, dict] = {}
    for p in sorted(dump_dir.glob("iter*_metrics.json")):
        it = int(p.stem[4:8])
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            m = d.get("metrics", {})
            out[it] = {
                "policy_loss": float(m.get("policy_loss", 0.0) or 0.0),
                "value_loss": float(m.get("value_loss", 0.0) or 0.0),
                "approx_kl": float(m.get("approx_kl", 0.0) or 0.0),
                "entropy": float(m.get("entropy", 0.0) or 0.0),
            }
        except Exception:
            continue
    return out


def smooth(x: list[float], window: int = 5) -> list[float]:
    """简单移动平均（对数据稀疏的胜率曲线，降低视觉噪声）。"""
    if len(x) < 2:
        return list(x)
    out = []
    for i in range(len(x)):
        lo = max(0, i - window // 2)
        hi = min(len(x), i + window // 2 + 1)
        seg = x[lo:hi]
        out.append(sum(seg) / max(len(seg), 1))
    return out


def wr_series(per_rt: dict, rt: str) -> tuple[list[int], list[float]]:
    """从 per_rt 提取 room_type 的胜率曲线（未见过的 iter 跳过）。"""
    if rt not in per_rt:
        return [], []
    data = per_rt[rt]
    iters = sorted(data.keys())
    wr = [data[i][0] / max(data[i][1], 1) for i in iters]
    return iters, wr


def plot_compare(primary: Path, compares: list[Path], out: Path) -> None:
    all_runs = [primary] + compares
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Multi-run comparison (primary: {primary.name})",
                 fontsize=13, fontweight="bold")

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    # ---- (A) Boss win rate ----
    ax = axes[0, 0]
    for i, run in enumerate(all_runs):
        _, _, per_rt = gather(run)
        iters, wr = wr_series(per_rt, "boss")
        if not iters:
            continue
        wr_s = smooth(wr, window=5)
        ax.plot(iters, [100 * w for w in wr_s], color=colors[i % len(colors)],
                label=f"{run.name} (n_iter={len(iters)})", linewidth=2.0,
                marker="o" if i == 0 else None, markersize=3)
        # 原始点（浅色）
        ax.scatter(iters, [100 * w for w in wr], color=colors[i % len(colors)],
                   alpha=0.2, s=8)
    ax.axhline(20, color="gray", linestyle="--", alpha=0.5, label="target 20%")
    ax.set_title("Boss win rate (smooth window=5)")
    ax.set_xlabel("iter"); ax.set_ylabel("win rate %")
    ax.set_ylim(-2, 60)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # ---- (B) Elite + Monster ----
    ax = axes[0, 1]
    for i, run in enumerate(all_runs):
        _, _, per_rt = gather(run)
        for rt, ls in [("elite", "-"), ("monster", ":")]:
            iters, wr = wr_series(per_rt, rt)
            if not iters:
                continue
            wr_s = smooth(wr, window=5)
            ax.plot(iters, [100 * w for w in wr_s], linestyle=ls,
                    color=colors[i % len(colors)], alpha=0.8 if ls == "-" else 0.5,
                    label=f"{run.name} {rt}", linewidth=1.5)
    ax.set_title("Elite (solid) / Monster (dotted) win rate")
    ax.set_xlabel("iter"); ax.set_ylabel("win rate %")
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # ---- (C) PPO health: KL & policy_loss ----
    ax = axes[1, 0]
    for i, run in enumerate(all_runs):
        metrics = gather_metrics(run)
        if not metrics:
            continue
        iters = sorted(metrics.keys())
        kls = [metrics[it]["approx_kl"] for it in iters]
        pls = [metrics[it]["policy_loss"] for it in iters]
        ax.plot(iters, kls, color=colors[i % len(colors)], linewidth=1.8,
                label=f"{run.name} KL")
        ax.plot(iters, pls, color=colors[i % len(colors)], linewidth=1.0,
                linestyle="--", alpha=0.5, label=f"{run.name} policy_loss")
    ax.set_title("PPO health: KL (solid) / policy_loss (dashed)")
    ax.set_xlabel("iter"); ax.set_ylabel("value")
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # ---- (D) Per-boss breakdown (primary only) ----
    ax = axes[1, 1]
    _, per_enc, _ = gather(primary)
    boss_ids = sorted([e for e in per_enc if "BOSS" in e.upper()])
    for i, bid in enumerate(boss_ids):
        data = per_enc[bid]
        iters = sorted(data.keys())
        wr = [data[it][0] / max(data[it][1], 1) for it in iters]
        if not iters:
            continue
        wr_s = smooth(wr, window=5)
        ax.plot(iters, [100 * w for w in wr_s], color=colors[i % len(colors)],
                linewidth=2.0, label=bid.replace("_BOSS", ""), marker="o",
                markersize=3)
        ax.scatter(iters, [100 * w for w in wr], color=colors[i % len(colors)],
                   alpha=0.2, s=8)
    ax.axhline(20, color="gray", linestyle="--", alpha=0.5, label="target 20%")
    ax.set_title(f"Per-boss win rate ({primary.name})")
    ax.set_xlabel("iter"); ax.set_ylabel("win rate %")
    ax.set_ylim(-2, 60)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_runs] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primary", type=Path, help="主要分析的 run 目录（输出也落这里）")
    ap.add_argument("--compare", type=Path, nargs="*", default=[],
                    help="对比的 run 目录（可多个）")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出图路径；默认 <primary>/analysis/compare_runs.png")
    args = ap.parse_args()
    if args.out is None:
        args.out = args.primary / "analysis" / "compare_runs.png"
    plot_compare(args.primary, args.compare, args.out)


if __name__ == "__main__":
    main()
