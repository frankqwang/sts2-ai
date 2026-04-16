"""Generate comparison report and charts for combat-teacher experiment results."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _summary(eval_payload: dict[str, Any]) -> dict[str, Any]:
    summaries = eval_payload.get("summaries")
    if isinstance(summaries, dict) and isinstance(summaries.get("nn"), dict):
        return summaries["nn"]
    summary = eval_payload.get("summary")
    return summary if isinstance(summary, dict) else {}


def _metric(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(summary.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _teacher_breakdown_means(teacher_eval: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    breakdown = teacher_eval.get("score_breakdown")
    if not isinstance(breakdown, dict):
        return out
    for key, stats in breakdown.items():
        if not isinstance(stats, dict):
            continue
        try:
            out[str(key)] = float(stats.get("mean", 0.0))
        except (TypeError, ValueError):
            continue
    return out


def _plot_training(metrics: list[dict[str, Any]], out_dir: Path) -> list[str]:
    if not metrics:
        return []
    x = [int(row.get("iteration", idx)) for idx, row in enumerate(metrics)]
    groups = {
        "teacher_loss": ["combat_teacher_loss", "combat_teacher_ce", "combat_teacher_rank"],
        "run_progress": ["avg_floor", "boss_reach_rate", "act1_clear_rate"],
        "hard_states": ["hard_state_premature_end_turn_steps", "hard_state_order_sensitive_steps", "hard_state_potion_steps"],
    }
    paths: list[str] = []
    for name, keys in groups.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for key in keys:
            values: list[float] = []
            has_key = False
            for row in metrics:
                if key in row:
                    has_key = True
                try:
                    values.append(float(row.get(key, 0.0)))
                except (TypeError, ValueError):
                    values.append(0.0)
            if has_key:
                ax.plot(x, values, marker="o", label=key)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(name)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _plot_eval_comparison(
    baseline_eval: dict[str, Any],
    trained_eval: dict[str, Any],
    out_dir: Path,
) -> str | None:
    baseline = _summary(baseline_eval)
    trained = _summary(trained_eval)
    if not baseline or not trained:
        return None
    keys = [
        "win_rate",
        "boss_reach_rate",
        "act1_clear_rate",
        "avg_floor",
        "avg_boss_hp_fraction_dealt",
        "avg_combats_won",
    ]
    labels = ["baseline", "trained"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, key in zip(axes.flatten(), keys):
        values = [_metric(baseline, key), _metric(trained, key)]
        ax.bar(labels, values, color=["#4e79a7", "#f28e2b"])
        ax.set_title(key)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "eval_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_teacher_eval(teacher_eval: dict[str, Any], out_dir: Path) -> list[str]:
    paths: list[str] = []
    means = _teacher_breakdown_means(teacher_eval)
    selected = [
        "total",
        "tactical_score",
        "mechanism_score",
        "continuation_score",
        "rule_bonus",
        "enemy_damage_progress",
        "expected_hp_loss_ratio",
        "mechanism_safety_gate",
    ]
    values = [means.get(key, 0.0) for key in selected]
    if any(abs(v) > 1e-9 for v in values):
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(selected, values, color="#59a14f")
        ax.set_title("teacher score breakdown mean")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = out_dir / "teacher_score_breakdown.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    motif_counts = teacher_eval.get("motif_counts")
    if isinstance(motif_counts, dict) and motif_counts:
        top = sorted(((str(k), int(v)) for k, v in motif_counts.items()), key=lambda item: item[1], reverse=True)[:18]
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
        labels = [item[0] for item in top]
        counts = [item[1] for item in top]
        ax.barh(labels[::-1], counts[::-1], color="#e15759")
        ax.set_title("teacher motif coverage")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path = out_dir / "teacher_motif_coverage.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _build_report(
    *,
    teacher_eval: dict[str, Any],
    metrics: list[dict[str, Any]],
    baseline_eval: dict[str, Any],
    trained_eval: dict[str, Any],
    figure_paths: list[str],
) -> dict[str, Any]:
    baseline = _summary(baseline_eval)
    trained = _summary(trained_eval)
    last_metrics = metrics[-1] if metrics else {}
    first_metrics = metrics[0] if metrics else {}
    return {
        "teacher": {
            "sample_count": teacher_eval.get("sample_count", 0),
            "root_sample_count": teacher_eval.get("root_sample_count", 0),
            "prefix_sample_count": teacher_eval.get("prefix_sample_count", 0),
            "solver_support_rate": teacher_eval.get("solver_support_rate", 0.0),
            "baseline_teacher_disagreement_rate": teacher_eval.get("baseline_teacher_disagreement_rate", 0.0),
            "baseline_regret_mean": (teacher_eval.get("baseline_regret") or {}).get("mean", 0.0),
            "line_length_mean": (teacher_eval.get("line_length") or {}).get("mean", 0.0),
            "search_nodes_mean": (teacher_eval.get("search_nodes") or {}).get("mean", 0.0),
            "motif_counts": teacher_eval.get("motif_counts", {}),
        },
        "training": {
            "iterations": len(metrics),
            "first": first_metrics,
            "last": last_metrics,
            "combat_teacher_ce_delta": _metric(last_metrics, "combat_teacher_ce") - _metric(first_metrics, "combat_teacher_ce"),
            "combat_teacher_loss_delta": _metric(last_metrics, "combat_teacher_loss") - _metric(first_metrics, "combat_teacher_loss"),
        },
        "eval": {
            "baseline": baseline,
            "trained": trained,
            "delta": {
                "win_rate": _metric(trained, "win_rate") - _metric(baseline, "win_rate"),
                "boss_reach_rate": _metric(trained, "boss_reach_rate") - _metric(baseline, "boss_reach_rate"),
                "act1_clear_rate": _metric(trained, "act1_clear_rate") - _metric(baseline, "act1_clear_rate"),
                "avg_floor": _metric(trained, "avg_floor") - _metric(baseline, "avg_floor"),
                "avg_boss_hp_fraction_dealt": _metric(trained, "avg_boss_hp_fraction_dealt") - _metric(baseline, "avg_boss_hp_fraction_dealt"),
                "avg_combats_won": _metric(trained, "avg_combats_won") - _metric(baseline, "avg_combats_won"),
            },
        },
        "figures": figure_paths,
    }


def _write_markdown(report: dict[str, Any], out_path: Path) -> None:
    teacher = report["teacher"]
    training = report["training"]
    eval_block = report["eval"]
    lines = [
        "# Combat Teacher Experiment Report",
        "",
        "## Teacher Data",
        "",
        f"- samples: {teacher['sample_count']}",
        f"- root/prefix: {teacher['root_sample_count']} / {teacher['prefix_sample_count']}",
        f"- solver support rate: {teacher['solver_support_rate']:.3f}",
        f"- baseline-teacher disagreement: {teacher['baseline_teacher_disagreement_rate']:.3f}",
        f"- baseline regret mean: {teacher['baseline_regret_mean']:.4f}",
        f"- line length mean: {teacher['line_length_mean']:.3f}",
        f"- search nodes mean: {teacher['search_nodes_mean']:.1f}",
        "",
        "## Training",
        "",
        f"- iterations: {training['iterations']}",
        f"- last combat_teacher_ce: {_metric(training['last'], 'combat_teacher_ce'):.6f}",
        f"- last combat_teacher_loss: {_metric(training['last'], 'combat_teacher_loss'):.6f}",
        f"- combat_teacher_ce delta: {training['combat_teacher_ce_delta']:.6f}",
        "",
        "## Fixed Seed Eval",
        "",
    ]
    baseline = eval_block["baseline"]
    trained = eval_block["trained"]
    delta = eval_block["delta"]
    for key in ["win_rate", "boss_reach_rate", "act1_clear_rate", "avg_floor", "avg_boss_hp_fraction_dealt", "avg_combats_won"]:
        lines.append(
            f"- {key}: baseline={_metric(baseline, key):.4f}, "
            f"trained={_metric(trained, key):.4f}, delta={delta[key]:+.4f}"
        )
    lines.extend(["", "## Figures", ""])
    for figure in report.get("figures", []):
        lines.append(f"- {figure}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate combat teacher experiment metrics and plots.")
    parser.add_argument("--teacher-eval", default="")
    parser.add_argument("--train-metrics", default="")
    parser.add_argument("--baseline-eval", default="")
    parser.add_argument("--trained-eval", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_eval = _load_json(args.teacher_eval)
    metrics = _load_jsonl(args.train_metrics)
    baseline_eval = _load_json(args.baseline_eval)
    trained_eval = _load_json(args.trained_eval)

    figure_paths: list[str] = []
    figure_paths.extend(_plot_teacher_eval(teacher_eval, out_dir))
    figure_paths.extend(_plot_training(metrics, out_dir))
    eval_plot = _plot_eval_comparison(baseline_eval, trained_eval, out_dir)
    if eval_plot:
        figure_paths.append(eval_plot)

    report = _build_report(
        teacher_eval=teacher_eval,
        metrics=metrics,
        baseline_eval=baseline_eval,
        trained_eval=trained_eval,
        figure_paths=figure_paths,
    )
    (out_dir / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, out_dir / "experiment_report.md")
    print(json.dumps({"output_dir": str(out_dir), "figures": figure_paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
