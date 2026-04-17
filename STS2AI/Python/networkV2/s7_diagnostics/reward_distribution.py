"""Per-iter reward / advantage 分布可视化。

用法：
  python -m networkV2.s7_diagnostics.reward_distribution ../Artifacts/runs/co15

产出：<dump>/analysis/reward_distribution.png

子图：
  (A) reward mean/std 随 iter 变化（看 shaping 量级是否稳定）
  (B) advantage 正负样本比例（zero-positive-advantage trap 预警）
  (C) reward 按 room_type 分组 mean（monster/elite/boss 各自 reward 水平）
  (D) value_estimate vs value_target 散点（value head 学习质量）
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gather_samples(dump_dir: Path) -> dict[int, dict]:
    """每个 iter 汇总 samples 的 reward / advantage / value 统计。"""
    stats: dict[int, dict] = {}
    for p in sorted(dump_dir.glob("iter*_samples.jsonl")):
        it = int(p.stem[4:8])
        rewards: list[float] = []
        advs: list[float] = []
        values: list[float] = []
        targets: list[float] = []
        by_rt: dict[str, list[float]] = defaultdict(list)
        for line in p.open(encoding="utf-8"):
            try:
                s = json.loads(line)
            except Exception:
                continue
            r = float(s.get("reward", 0.0))
            rewards.append(r)
            advs.append(float(s.get("advantage", 0.0)))
            values.append(float(s.get("value_estimate", 0.0)))
            targets.append(float(s.get("value_target", 0.0)))
            rt = s.get("room_type", "")
            if rt:
                by_rt[rt].append(r)
        if not rewards:
            continue
        stats[it] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_pos_rate": float(np.mean([r > 0 for r in rewards])),
            "advantage_pos_rate": float(np.mean([a > 0 for a in advs])),
            "advantage_mean": float(np.mean(advs)),
            "advantage_std": float(np.std(advs)),
            "value_est": np.array(values, dtype=np.float32),
            "value_tgt": np.array(targets, dtype=np.float32),
            "reward_by_rt": {rt: float(np.mean(rs)) for rt, rs in by_rt.items()},
        }
    return stats


def plot(dump_dir: Path, out: Path) -> None:
    stats = gather_samples(dump_dir)
    if not stats:
        print(f"no samples in {dump_dir}")
        return
    iters = sorted(stats.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Reward / Advantage distribution — {dump_dir.name}",
                 fontsize=13, fontweight="bold")

    # (A) reward mean ± std
    ax = axes[0, 0]
    mean = np.array([stats[i]["reward_mean"] for i in iters])
    std = np.array([stats[i]["reward_std"] for i in iters])
    ax.plot(iters, mean, color="#1f77b4", linewidth=1.8, label="reward mean")
    ax.fill_between(iters, mean - std, mean + std, color="#1f77b4", alpha=0.15, label="±1σ")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Reward mean ± std per iter")
    ax.set_xlabel("iter"); ax.set_ylabel("reward")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (B) advantage pos rate (zero-positive trap 预警)
    ax = axes[0, 1]
    adv_pos = [stats[i]["advantage_pos_rate"] * 100 for i in iters]
    rwd_pos = [stats[i]["reward_pos_rate"] * 100 for i in iters]
    ax.plot(iters, adv_pos, color="#2ca02c", linewidth=1.8, label="advantage > 0 %")
    ax.plot(iters, rwd_pos, color="#ff7f0e", linewidth=1.5, linestyle="--",
            label="reward > 0 %")
    ax.axhline(50, color="gray", linewidth=0.5, linestyle=":", label="50% (ideal center)")
    ax.set_title("Positive-advantage sample rate (learning signal)")
    ax.set_xlabel("iter"); ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (C) reward mean by room_type
    ax = axes[1, 0]
    colors = {"monster": "#9467bd", "elite": "#e377c2", "boss": "#d62728"}
    for rt in ("monster", "elite", "boss"):
        xs = [i for i in iters if rt in stats[i]["reward_by_rt"]]
        ys = [stats[i]["reward_by_rt"][rt] for i in xs]
        if xs:
            ax.plot(xs, ys, color=colors.get(rt, "gray"), linewidth=1.8, label=rt)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Reward mean by room_type")
    ax.set_xlabel("iter"); ax.set_ylabel("reward mean")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (D) value_est vs value_target（最后一个 iter 的散点）
    ax = axes[1, 1]
    last_it = iters[-1]
    v_est = stats[last_it]["value_est"]
    v_tgt = stats[last_it]["value_tgt"]
    ax.scatter(v_tgt, v_est, alpha=0.3, s=8, color="#17becf")
    lo = min(float(v_tgt.min()), float(v_est.min()))
    hi = max(float(v_tgt.max()), float(v_est.max()))
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1.0, linestyle="--",
            label="y=x (perfect)")
    ax.set_title(f"Value learning quality (iter {last_it}, n={len(v_est)})")
    ax.set_xlabel("value_target (return)"); ax.set_ylabel("value_estimate")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[reward_distribution] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = args.dump_dir / "analysis" / "reward_distribution.png"
    plot(args.dump_dir, args.out)


if __name__ == "__main__":
    main()
