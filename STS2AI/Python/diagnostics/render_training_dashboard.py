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


def _load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _rolling(values: list[float], window: int = 5) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _get(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def _rate(rows: list[dict[str, Any]], num_key: str, den_key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        den = float(row.get(den_key, 0.0) or 0.0)
        num = float(row.get(num_key, 0.0) or 0.0)
        out.append(0.0 if den <= 0 else num / den)
    return out


def render_metrics_dashboard(training_dir: Path, rows: list[dict[str, Any]], output_path: Path) -> None:
    iterations = _get(rows, "iteration")
    avg_floor = _get(rows, "avg_floor")
    boss = [value * 100.0 for value in _get(rows, "boss_reach_rate")]
    act1 = [value * 100.0 for value in _get(rows, "act1_clear_rate")]
    boss_hp = [value * 100.0 for value in _get(rows, "boss_hp_fraction_dealt_mean")]
    deck_boss = _get(rows, "deck_size_at_boss_mean")
    ppo_v = _get(rows, "ppo_vloss")
    combat_v = _get(rows, "combat_ppo_vloss")
    ppo_ent = _get(rows, "ppo_entropy")
    combat_ent = _get(rows, "combat_entropy")
    ppo_kl = _get(rows, "ppo_approx_kl")
    combat_kl = _get(rows, "combat_ppo_approx_kl")
    potion = [value * 100.0 for value in _rate(rows, "hard_state_potion_steps", "combat_ppo_steps")]
    premature = [value * 100.0 for value in _rate(rows, "hard_state_premature_end_turn_steps", "combat_ppo_steps")]
    repeat = [value * 100.0 for value in _rate(rows, "hard_state_repeat_loop_steps", "combat_ppo_steps")]
    card_skip = [value * 100.0 for value in _get(rows, "card_reward_skip_rate")]
    collect_time = _get(rows, "collect_time_s")
    update_time = _get(rows, "update_time_s")
    iter_time = _get(rows, "iter_time_s")
    nc_forward = _get(rows, "nc_forward_time_s")
    inference = _get(rows, "inference_time_s")
    gates = _get(rows, "combat_teacher_action_context_gate")

    fig, axes = plt.subplots(3, 3, figsize=(20, 14), constrained_layout=True)
    fig.suptitle(f"Training Dashboard: {training_dir.name}", fontsize=16)

    ax = axes[0, 0]
    ax.plot(iterations, avg_floor, color="#1f77b4", alpha=0.35, label="avg_floor")
    ax.plot(iterations, _rolling(avg_floor, 5), color="#1f77b4", linewidth=2.0, label="avg_floor(roll5)")
    ax.set_title("Avg Floor")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(iterations, boss, label="boss_reach_rate %", color="#d62728")
    ax.plot(iterations, act1, label="act1_clear_rate %", color="#2ca02c")
    ax.plot(iterations, _rolling(boss, 5), color="#d62728", linewidth=2.0, alpha=0.9)
    ax.plot(iterations, _rolling(act1, 5), color="#2ca02c", linewidth=2.0, alpha=0.9)
    ax.set_title("Progress Rates")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(iterations, boss_hp, label="boss_hp_dealt %", color="#9467bd")
    ax.plot(iterations, deck_boss, label="deck@boss", color="#8c564b")
    ax.set_title("Boss Readiness")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(iterations, ppo_v, label="ppo_vloss", color="#ff7f0e")
    ax.plot(iterations, combat_v, label="combat_vloss", color="#17becf")
    ax.set_title("Value Loss")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(iterations, ppo_ent, label="ppo_entropy", color="#7f7f7f")
    ax.plot(iterations, combat_ent, label="combat_entropy", color="#bcbd22")
    ax.plot(iterations, ppo_kl, label="ppo_kl", color="#e377c2")
    ax.plot(iterations, combat_kl, label="combat_kl", color="#8c564b")
    ax.set_title("Entropy / KL")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 2]
    ax.plot(iterations, potion, label="use_potion_rate %", color="#d62728")
    ax.plot(iterations, premature, label="premature_end_turn_rate %", color="#1f77b4")
    ax.plot(iterations, repeat, label="repeat_loop_rate %", color="#2ca02c")
    ax.set_title("Hard-State Rates")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(iterations, card_skip, label="card_reward_skip %", color="#ff9896")
    ax.plot(iterations, _rolling(card_skip, 5), color="#ff9896", linewidth=2.0)
    ax.set_title("Card Reward Skip")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(iterations, collect_time, label="collect_time_s", color="#1f77b4")
    ax.plot(iterations, update_time, label="update_time_s", color="#ff7f0e")
    ax.plot(iterations, iter_time, label="iter_time_s", color="#2ca02c")
    ax.set_title("Time Cost")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 2]
    ax.plot(iterations, nc_forward, label="nc_forward_time_s", color="#9467bd")
    ax.plot(iterations, inference, label="inference_time_s", color="#8c564b")
    ax.plot(iterations, gates, label="combat_teacher_gate", color="#17becf")
    ax.set_title("Inference / Gate")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _barh(ax: Any, items: list[dict[str, Any]], title: str, *, max_items: int = 10) -> None:
    picked = items[:max_items]
    labels = [str(item.get("display") or item.get("name") or "") for item in picked][::-1]
    values = [float(item.get("count", 0)) for item in picked][::-1]
    if not labels:
        labels = ["无"]
        values = [0.0]
    ax.barh(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)


def render_iteration_dashboard(report: dict[str, Any], output_path: Path) -> None:
    floors = report.get("floors") or {}
    death = report.get("death") or {}
    route = report.get("route") or {}
    rewards = report.get("rewards") or {}
    shop = report.get("shop") or {}
    combat = report.get("combat") or {}
    bins = floors.get("bins") or {}

    fig, axes = plt.subplots(3, 2, figsize=(18, 15), constrained_layout=True)
    fig.suptitle(f"Iteration {report.get('metrics', {}).get('iteration', 'N/A')} Replay Dashboard", fontsize=16)

    ax = axes[0, 0]
    labels = list(bins.keys()) or ["无"]
    values = [float(bins[key]) for key in labels] or [0.0]
    ax.bar(labels, values, color="#72b7b2")
    ax.set_title("Floor Bins")
    ax.grid(axis="y", alpha=0.25)

    _barh(axes[0, 1], death.get("terminal_enemy_top") or [], "Death Enemy Top")
    _barh(axes[1, 0], route.get("early_map_top") or [], "Early Map Choice Top")
    _barh(axes[1, 1], rewards.get("card_pick_top") or [], "Card Pick Top")
    _barh(axes[2, 0], shop.get("shop_action_top") or [], "Shop Action Top")
    _barh(axes[2, 1], combat.get("potion_fight_top") or [], "Potion Fight Top")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    _configure_cjk_font()
    parser = argparse.ArgumentParser(description="渲染训练趋势与 iteration replay 可视化。")
    parser.add_argument("training_dir", help="训练输出目录")
    parser.add_argument("--iteration", type=int, help="可选：生成对应 iteration 的 replay 可视化")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    metrics_path = training_dir / "metrics.jsonl"
    analysis_dir = training_dir / "analysis"
    rows = _load_metrics(metrics_path)
    if not rows:
        raise SystemExit(f"metrics 为空: {metrics_path}")

    render_metrics_dashboard(training_dir, rows, analysis_dir / "training_dashboard.png")
    print(f"Saved: {analysis_dir / 'training_dashboard.png'}")

    if args.iteration is not None:
        report_path = analysis_dir / f"iter_{args.iteration:05d}_replay_report.json"
        if not report_path.exists():
            raise SystemExit(f"缺少 replay 报告: {report_path}")
        report = json.loads(report_path.read_text(encoding='utf-8'))
        render_iteration_dashboard(report, analysis_dir / f"iter_{args.iteration:05d}_replay_dashboard.png")
        print(f"Saved: {analysis_dir / f'iter_{args.iteration:05d}_replay_dashboard.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
