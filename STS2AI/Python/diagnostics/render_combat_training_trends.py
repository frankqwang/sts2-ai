"""Render matplotlib charts of combat-teacher training metrics over iterations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _configure_cjk_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Microsoft JhengHei",
        "Arial Unicode MS",
    ]
    for name in candidates:
        try:
            matplotlib.font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def _load_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not metrics_path.exists():
        return rows
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _rolling(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _get(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def _room_win_rate(rows: list[dict[str, Any]], room_type: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        bucket = ((row.get("room_breakdown") or {}).get(room_type) or {})
        episodes = float(bucket.get("episodes", 0) or 0.0)
        wins = float(bucket.get("wins", 0) or 0.0)
        out.append(0.0 if episodes <= 0 else wins / episodes)
    return out


def render(training_dir: Path, rows: list[dict[str, Any]], output_path: Path, *, roll_window: int) -> None:
    iterations = _get(rows, "iteration")
    win_rate = [value * 100.0 for value in _get(rows, "win_rate")]
    avg_reward = _get(rows, "avg_reward")
    avg_steps = _get(rows, "avg_steps")
    invalid_actions = _get(rows, "invalid_actions")

    monster_win = [value * 100.0 for value in _room_win_rate(rows, "monster")]
    elite_win = [value * 100.0 for value in _room_win_rate(rows, "elite")]
    boss_win = [value * 100.0 for value in _room_win_rate(rows, "boss")]

    ppo_ploss = [float((row.get("update") or {}).get("combat_ppo_ploss", 0.0) or 0.0) for row in rows]
    ppo_vloss = [float((row.get("update") or {}).get("combat_ppo_vloss", 0.0) or 0.0) for row in rows]
    entropy = [float((row.get("update") or {}).get("combat_entropy", 0.0) or 0.0) for row in rows]
    approx_kl = [float((row.get("update") or {}).get("combat_ppo_approx_kl", 0.0) or 0.0) for row in rows]
    clip_fraction = [float((row.get("update") or {}).get("combat_ppo_clip_fraction", 0.0) or 0.0) for row in rows]

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    fig.suptitle(f"Combat-Only 训练趋势: {training_dir.name}", fontsize=16)

    ax = axes[0, 0]
    ax.plot(iterations, win_rate, marker="o", alpha=0.45, label="overall win %")
    ax.plot(iterations, _rolling(win_rate, roll_window), linewidth=2.0, label=f"roll{roll_window}")
    ax.set_title("整体战斗胜率")
    ax.set_xlabel("iteration")
    ax.set_ylabel("%")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(iterations, monster_win, marker="o", label="monster win %")
    ax.plot(iterations, elite_win, marker="o", label="elite win %")
    ax.plot(iterations, boss_win, marker="o", label="boss win %")
    ax.set_title("分房型胜率")
    ax.set_xlabel("iteration")
    ax.set_ylabel("%")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(iterations, avg_reward, marker="o", label="avg_reward")
    ax.plot(iterations, avg_steps, marker="o", label="avg_steps")
    ax.set_title("样本质量")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(iterations, invalid_actions, marker="o", color="#d62728", label="invalid_actions")
    ax.plot(iterations, _rolling(invalid_actions, roll_window), color="#8c1d18", linewidth=2.0, label=f"roll{roll_window}")
    ax.set_title("非法动作拒绝次数")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(iterations, ppo_ploss, marker="o", label="ppo_ploss")
    ax.plot(iterations, ppo_vloss, marker="o", label="ppo_vloss")
    ax.plot(iterations, entropy, marker="o", label="entropy")
    ax.set_title("PPO 核心量")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(iterations, approx_kl, marker="o", label="approx_kl")
    ax.plot(iterations, clip_fraction, marker="o", label="clip_fraction")
    ax.set_title("PPO 稳定性")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    _configure_cjk_font()
    parser = argparse.ArgumentParser(description="渲染 combat-only 训练趋势图。")
    parser.add_argument("training_dir", help="训练输出目录")
    parser.add_argument("--roll-window", type=int, default=3, help="滚动均值窗口")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    rows = _load_rows(training_dir / "metrics.jsonl")
    if not rows:
        raise SystemExit(f"未加载到任何 metrics 行: {training_dir / 'metrics.jsonl'}")

    output_path = Path(args.output) if args.output else training_dir / "analysis" / "combat_training_trends.png"
    render(training_dir, rows, output_path, roll_window=max(1, args.roll_window))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
