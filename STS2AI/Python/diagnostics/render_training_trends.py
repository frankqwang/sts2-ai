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


def _merge_rows(training_dirs: list[Path], iter_start: int | None, iter_end: int | None) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for training_dir in training_dirs:
        for row in _load_rows(training_dir / "metrics.jsonl"):
            iteration = int(row.get("iteration", -1))
            if iter_start is not None and iteration < iter_start:
                continue
            if iter_end is not None and iteration > iter_end:
                continue
            merged[iteration] = row
    return [merged[key] for key in sorted(merged)]


def _get(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def _rate(rows: list[dict[str, Any]], num_key: str, den_key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        den = float(row.get(den_key, 0.0) or 0.0)
        num = float(row.get(num_key, 0.0) or 0.0)
        out.append(0.0 if den <= 0 else num / den)
    return out


def _rolling(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def render(training_dirs: list[Path], rows: list[dict[str, Any]], output_path: Path, *, roll_window: int) -> None:
    iterations = _get(rows, "iteration")
    avg_floor = _get(rows, "avg_floor")
    boss = [value * 100.0 for value in _get(rows, "boss_reach_rate")]
    act1 = [value * 100.0 for value in _get(rows, "act1_clear_rate")]
    boss_hp = [value * 100.0 for value in _get(rows, "boss_hp_fraction_dealt_mean")]
    deck = _get(rows, "deck_size_at_boss_mean")
    ppo_vloss = _get(rows, "ppo_vloss")
    skip = [value * 100.0 for value in _get(rows, "card_reward_skip_rate")]
    wait_steps = _get(rows, "combat_pending_wait_steps")
    refresh_steps = _get(rows, "combat_pending_refresh_steps")
    stall = _get(rows, "combat_pending_stall_count")
    potion = [value * 100.0 for value in _rate(rows, "hard_state_potion_steps", "combat_ppo_steps")]
    premature = [value * 100.0 for value in _rate(rows, "hard_state_premature_end_turn_steps", "combat_ppo_steps")]
    repeat = [value * 100.0 for value in _rate(rows, "hard_state_repeat_loop_steps", "combat_ppo_steps")]

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    fig.suptitle(
        "训练趋势总览\n" + " + ".join(training_dir.name for training_dir in training_dirs),
        fontsize=16,
    )

    ax = axes[0, 0]
    ax.plot(iterations, avg_floor, marker="o", alpha=0.45, label="avg_floor")
    ax.plot(iterations, _rolling(avg_floor, roll_window), linewidth=2.0, label=f"roll{roll_window}")
    ax.set_title("平均层数")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(iterations, boss, marker="o", label="boss_reach %")
    ax.plot(iterations, act1, marker="o", label="act1_clear %")
    ax.plot(iterations, _rolling(boss, roll_window), linewidth=2.0, alpha=0.9)
    ax.plot(iterations, _rolling(act1, roll_window), linewidth=2.0, alpha=0.9)
    ax.set_title("推进率")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(iterations, boss_hp, marker="o", label="boss_hp_dealt %")
    ax.plot(iterations, deck, marker="o", label="deck@boss")
    ax.set_title("Boss 就绪度")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(iterations, ppo_vloss, marker="o", label="ppo_vloss")
    ax.plot(iterations, skip, marker="o", label="card_skip %")
    ax.set_title("非战斗信号")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(iterations, wait_steps, marker="o", label="pending_wait")
    ax.plot(iterations, refresh_steps, marker="o", label="pending_refresh")
    ax.plot(iterations, stall, marker="o", label="pending_stall")
    ax.set_title("combat_pending 过渡")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(iterations, potion, marker="o", label="use_potion %")
    ax.plot(iterations, premature, marker="o", label="premature_end_turn %")
    ax.plot(iterations, repeat, marker="o", label="repeat_loop %")
    ax.set_title("战斗坏模式")
    ax.grid(alpha=0.25)
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel("iteration")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    _configure_cjk_font()
    parser = argparse.ArgumentParser(description="把多个训练目录拼成连续趋势图。")
    parser.add_argument("training_dirs", nargs="+", help="一个或多个训练输出目录")
    parser.add_argument("--iter-start", type=int, default=None, help="起始 iteration")
    parser.add_argument("--iter-end", type=int, default=None, help="结束 iteration")
    parser.add_argument("--roll-window", type=int, default=3, help="滚动均值窗口")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    args = parser.parse_args()

    training_dirs = [Path(item) for item in args.training_dirs]
    rows = _merge_rows(training_dirs, args.iter_start, args.iter_end)
    if not rows:
        raise SystemExit("未加载到任何 metrics 行。")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = training_dirs[-1] / "analysis" / "training_trends.png"
    render(training_dirs, rows, output_path, roll_window=args.roll_window)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
