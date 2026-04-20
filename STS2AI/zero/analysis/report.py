from __future__ import annotations

"""训练产物分析与可视化。

职责：
- 读取单次训练 run 目录里的 `run_metrics / manifests / raw_runs / eval / logs`
- 生成面向排查的摘要表和 PNG 图
- 输出统一放到当前 run 根目录下的 `analysis/`
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_training_analysis(*, run_root: Path, run_metrics_path: Path) -> Path:
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
    manifests = _extract_manifests(metrics)

    training_df = _build_training_dataframe(manifests)
    eval_df = _build_eval_dataframe(manifests)
    sampling_df = _build_sampling_dataframe(manifests)
    rollout_df = _build_rollout_dataframe(run_root / "raw_runs")
    episode_df = _build_episode_event_dataframe(run_root / "logs")

    summary = {
        "iterations": len(manifests),
        "train_cases": len(metrics.get("train_cases") or ([metrics["selected_case"]] if "selected_case" in metrics else [])),
        "eval_cases": len(metrics.get("eval_cases") or ([metrics["selected_case"]] if "selected_case" in metrics else [])),
        "curriculum_mode": metrics.get("curriculum_mode", "smoke"),
        "run_id": metrics.get("run_id", (metrics.get("selected_case") or {}).get("run_id")),
        "has_rollout_rows": int(len(rollout_df)),
        "has_eval_rows": int(len(eval_df)),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_dataframe(training_df, analysis_dir / "training_metrics.csv")
    _write_dataframe(eval_df, analysis_dir / "evaluation_metrics.csv")
    _write_dataframe(sampling_df, analysis_dir / "sampling_metrics.csv")
    _write_dataframe(rollout_df, analysis_dir / "rollout_metrics.csv")
    _write_dataframe(episode_df, analysis_dir / "episode_metrics.csv")

    _plot_training_metrics(training_df, analysis_dir / "training_metrics.png")
    _plot_sampling_metrics(sampling_df, episode_df, analysis_dir / "sampling_metrics.png")
    _plot_pool_diagnostics(sampling_df, analysis_dir / "pool_diagnostics.png")
    _plot_eval_metrics(eval_df, analysis_dir / "evaluation_metrics.png")
    _plot_rollout_behavior(rollout_df, analysis_dir / "rollout_behavior.png")
    _plot_cohort_heatmap(eval_df, analysis_dir / "cohort_overview.png")
    return analysis_dir


def _build_training_dataframe(manifests: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        training = dict(manifest.get("training") or {})
        row = {
            "iteration": int(manifest.get("iteration") or 0),
            "collector_version": manifest.get("collector_version", ""),
            "promoted": bool((manifest.get("promotion") or {}).get("promoted", False)),
            "promotion_reason": (manifest.get("promotion") or {}).get("reason", ""),
        }
        row.update(training)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_eval_dataframe(manifests: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        iteration = int(manifest.get("iteration") or 0)
        for item in manifest.get("evaluations") or []:
            metadata = dict(item.get("metadata") or {})
            rows.append(
                {
                    "iteration": iteration,
                    "cohort_name": item.get("cohort_name", ""),
                    "fight_win_rate": float(item.get("fight_win_rate") or 0.0),
                    "enemy_hp_fraction_dealt": float(item.get("enemy_hp_fraction_dealt") or 0.0),
                    "self_hp_fraction_remaining": float(item.get("self_hp_fraction_remaining") or 0.0),
                    "teacher_agreement_at_1": float(item.get("teacher_agreement_at_1") or 0.0),
                    "teacher_topk_overlap": float(item.get("teacher_topk_overlap") or 0.0),
                    "timeout_rate": float(metadata.get("timeout_rate", 0.0) or 0.0),
                    "avg_no_progress_ratio": float(metadata.get("avg_no_progress_ratio", 0.0) or 0.0),
                    "avg_max_no_progress_streak": float(metadata.get("avg_max_no_progress_streak", 0.0) or 0.0),
                    "eval_bucket": str(metadata.get("eval_bucket", metadata.get("encounter_type", "default")) or "default"),
                    "encounter_id": str(metadata.get("encounter_id", "") or ""),
                }
            )
    return pd.DataFrame(rows)


def _build_sampling_dataframe(manifests: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        row = {
            "iteration": int(manifest.get("iteration") or 0),
        }
        row.update({f"sample_{key}": value for key, value in (manifest.get("sample_counts") or {}).items()})
        admission = dict(manifest.get("admission_stats") or {})
        row.update({f"admission_{key}": value for key, value in admission.items() if key != "pool_mutation_counters"})
        for pool_name, counters in dict(admission.get("pool_mutation_counters") or {}).items():
            for key, value in dict(counters or {}).items():
                row[f"pool_counter_{pool_name}_{key}"] = value
        row.update({f"pool_{key}": value for key, value in (manifest.get("pool_sizes") or {}).items()})
        row.update({f"pool_capacity_{key}": value for key, value in (manifest.get("pool_capacities") or {}).items()})
        for pool_name, stats in dict(manifest.get("pool_stats") or {}).items():
            stats_dict = dict(stats or {})
            for key in ("keep_score_min", "keep_score_avg", "keep_score_max", "sample_weight_avg", "bucket_count"):
                if key in stats_dict:
                    row[f"pool_stat_{pool_name}_{key}"] = stats_dict[key]
        rows.append(row)
    return pd.DataFrame(rows)


def _build_rollout_dataframe(raw_runs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_runs_dir.glob("iter_*.jsonl")):
        iteration = _extract_iteration(path.name)
        action_type_counter: Counter[str] = Counter()
        outcome_counter: Counter[str] = Counter()
        progress_counter: Counter[str] = Counter()
        encounter_counter: Counter[str] = Counter()
        total_rows = 0
        for line in path.open("r", encoding="utf-8"):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            total_rows += 1
            action = row.get("action") or {}
            state = row.get("state") or {}
            context = (state.get("context") or {}) if isinstance(state, dict) else {}
            metadata = row.get("metadata") or {}
            action_type_counter[str(action.get("action_type") or "unknown")] += 1
            outcome_counter[str(row.get("fight_outcome") or "none")] += 1
            progress_counter["progress" if bool(metadata.get("made_progress", False)) else "no_progress"] += 1
            encounter_counter[str(context.get("encounter_id") or "unknown")] += 1
        rows.append(
            {
                "iteration": iteration,
                "transition_rows": total_rows,
                "top_action_type": action_type_counter.most_common(1)[0][0] if action_type_counter else "",
                "top_action_count": action_type_counter.most_common(1)[0][1] if action_type_counter else 0,
                "progress_ratio": (
                    progress_counter["progress"] / max(progress_counter["progress"] + progress_counter["no_progress"], 1)
                ),
                "top_encounter": encounter_counter.most_common(1)[0][0] if encounter_counter else "",
                "top_encounter_count": encounter_counter.most_common(1)[0][1] if encounter_counter else 0,
                "victory_rows": outcome_counter.get("victory", 0),
                "timeout_rows": outcome_counter.get("timeout", 0),
            }
        )
    return pd.DataFrame(rows)


def _build_episode_event_dataframe(logs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(logs_dir.glob("iter_*.events.jsonl")):
        iteration = _extract_iteration(path.name)
        episode_rows: list[dict[str, Any]] = []
        for line in path.open("r", encoding="utf-8"):
            text = line.strip()
            if not text:
                continue
            event = json.loads(text)
            if event.get("phase") != "collect_episode" or event.get("status") != "completed":
                continue
            episode_rows.append(event)
        if not episode_rows:
            continue
        df = pd.DataFrame(episode_rows)
        rows.append(
            {
                "iteration": iteration,
                "episodes": int(len(df)),
                "avg_episode_duration_s": float(df["duration_s"].mean()),
                "max_episode_duration_s": float(df["duration_s"].max()),
                "avg_episode_steps": float(df["steps"].mean()),
                "avg_step_throughput": float(df.get("step_throughput", pd.Series(dtype=float)).mean() if "step_throughput" in df else 0.0),
                "avg_core_step_throughput": float(df.get("core_step_throughput", pd.Series(dtype=float)).mean() if "core_step_throughput" in df else 0.0),
                "avg_reset_duration_s": float(df.get("reset_duration_s", pd.Series(dtype=float)).mean() if "reset_duration_s" in df else 0.0),
                "avg_policy_infer_duration_s": float(df.get("policy_infer_duration_s", pd.Series(dtype=float)).mean() if "policy_infer_duration_s" in df else 0.0),
                "avg_env_step_duration_s": float(df.get("env_step_duration_s", pd.Series(dtype=float)).mean() if "env_step_duration_s" in df else 0.0),
                "avg_observe_duration_s": float(df.get("observe_duration_s", pd.Series(dtype=float)).mean() if "observe_duration_s" in df else 0.0),
                "avg_emit_duration_s": float(df.get("emit_duration_s", pd.Series(dtype=float)).mean() if "emit_duration_s" in df else 0.0),
                "avg_overhead_duration_s": float(df.get("overhead_duration_s", pd.Series(dtype=float)).mean() if "overhead_duration_s" in df else 0.0),
                "avg_no_progress_ratio": float(df.get("no_progress_ratio", pd.Series(dtype=float)).mean() if "no_progress_ratio" in df else 0.0),
                "max_no_progress_streak": float(df.get("max_no_progress_streak", pd.Series(dtype=float)).max() if "max_no_progress_streak" in df else 0.0),
                "timeouts": int((df.get("truncated", False) == True).sum()) if "truncated" in df else 0,
            }
        )
    return pd.DataFrame(rows)


def _plot_training_metrics(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = df["iteration"]

    axes[0, 0].plot(x, df["total_loss"], marker="o", label="total")
    for column in ("policy_loss", "value_loss", "ranking_loss", "delta_loss", "uncertainty_loss"):
        if column in df:
            axes[0, 0].plot(x, df[column], marker="o", label=column.replace("_loss", ""))
    axes[0, 0].set_title("Training Losses")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, df["learning_rate"], marker="o", label="lr")
    axes[0, 1].plot(x, df["grad_norm"], marker="o", label="grad_norm")
    axes[0, 1].set_title("LR / Grad Norm")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(x, df["teacher_sample_ratio"], label="teacher_ratio")
    if "skipped_non_finite_steps" in df:
        axes[1, 0].plot(x, df["skipped_non_finite_steps"], marker="o", color="crimson", label="skipped_non_finite")
    axes[1, 0].set_title("Teacher Ratio / Stability")
    axes[1, 0].legend(fontsize=8)

    promoted = df["promoted"].astype(int) if "promoted" in df else pd.Series([0] * len(df))
    axes[1, 1].bar(x, promoted, color="#4c78a8")
    axes[1, 1].set_title("Promotion (1=yes)")
    axes[1, 1].set_ylim(0, 1.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_sampling_metrics(sampling_df: pd.DataFrame, episode_df: pd.DataFrame, output_path: Path) -> None:
    if sampling_df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = sampling_df["iteration"]

    sample_columns = [col for col in sampling_df.columns if col.startswith("sample_")]
    for column in sample_columns:
        axes[0, 0].plot(x, sampling_df[column], marker="o", label=column.replace("sample_", ""))
    axes[0, 0].set_title("Sample Counts")
    axes[0, 0].legend(fontsize=8)

    pool_columns = [col for col in sampling_df.columns if col.startswith("pool_")]
    for column in pool_columns:
        if column.startswith("pool_capacity_"):
            continue
        axes[0, 1].plot(x, sampling_df[column], marker="o", label=column.replace("pool_", ""))
    capacity_columns = [col for col in sampling_df.columns if col.startswith("pool_capacity_")]
    for column in capacity_columns:
        axes[0, 1].plot(x, sampling_df[column], linestyle="--", alpha=0.6, label=column.replace("pool_capacity_", "") + "_cap")
    axes[0, 1].set_title("Pool Sizes")
    axes[0, 1].legend(fontsize=8)

    if not episode_df.empty:
        ex = episode_df["iteration"]
        axes[1, 0].plot(ex, episode_df["avg_episode_duration_s"], marker="o", label="avg_duration_s")
        axes[1, 0].plot(ex, episode_df["avg_step_throughput"], marker="o", label="avg_step_throughput")
        axes[1, 0].plot(ex, episode_df["avg_core_step_throughput"], marker="o", label="core_step_throughput")
        axes[1, 0].set_title("Collect Throughput")
        axes[1, 0].legend(fontsize=8)

        for column in (
            "avg_reset_duration_s",
            "avg_policy_infer_duration_s",
            "avg_env_step_duration_s",
            "avg_emit_duration_s",
            "avg_overhead_duration_s",
        ):
            if column in episode_df:
                axes[1, 1].plot(ex, episode_df[column], marker="o", label=column.replace("avg_", ""))
        axes[1, 1].set_title("Collect Timing Breakdown")
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 0].axis("off")
        counter_columns = [col for col in sampling_df.columns if col.startswith("pool_counter_")]
        if counter_columns:
            for column in counter_columns:
                if column.endswith("_accepted_adds") or column.endswith("_rejected_adds"):
                    axes[1, 1].plot(x, sampling_df[column], marker="o", label=column.replace("pool_counter_", ""))
            axes[1, 1].set_title("Pool Admission / Rejection")
            axes[1, 1].legend(fontsize=8)
        else:
            axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_eval_metrics(eval_df: pd.DataFrame, output_path: Path) -> None:
    if eval_df.empty:
        return
    grouped = eval_df.groupby("iteration", as_index=False).agg(
        fight_win_rate=("fight_win_rate", "mean"),
        enemy_hp_fraction_dealt=("enemy_hp_fraction_dealt", "mean"),
        self_hp_fraction_remaining=("self_hp_fraction_remaining", "mean"),
        teacher_agreement_at_1=("teacher_agreement_at_1", "mean"),
        timeout_rate=("timeout_rate", "mean"),
        avg_no_progress_ratio=("avg_no_progress_ratio", "mean"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    x = grouped["iteration"]
    for column in ("fight_win_rate", "enemy_hp_fraction_dealt", "self_hp_fraction_remaining", "teacher_agreement_at_1"):
        axes[0].plot(x, grouped[column], marker="o", label=column)
    axes[0].set_title("Evaluation Quality")
    axes[0].legend(fontsize=8)

    axes[1].plot(x, grouped["timeout_rate"], marker="o", label="timeout_rate")
    axes[1].plot(x, grouped["avg_no_progress_ratio"], marker="o", label="avg_no_progress_ratio")
    axes[1].set_title("Evaluation Failure Signals")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_pool_diagnostics(sampling_df: pd.DataFrame, output_path: Path) -> None:
    if sampling_df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    x = sampling_df["iteration"]

    counter_columns = [col for col in sampling_df.columns if col.startswith("pool_counter_")]
    plotted = False
    for column in counter_columns:
        if column.endswith("_accepted_adds") or column.endswith("_rejected_adds") or column.endswith("_evicted_items"):
            axes[0].plot(x, sampling_df[column], marker="o", label=column.replace("pool_counter_", ""))
            plotted = True
    axes[0].set_title("Pool Admission / Rejection / Eviction")
    if plotted:
        axes[0].legend(fontsize=8)
    else:
        axes[0].axis("off")

    stat_columns = [col for col in sampling_df.columns if col.startswith("pool_stat_")]
    plotted = False
    for column in stat_columns:
        if column.endswith("_keep_score_avg") or column.endswith("_sample_weight_avg"):
            axes[1].plot(x, sampling_df[column], marker="o", label=column.replace("pool_stat_", ""))
            plotted = True
    axes[1].set_title("Pool Quality Signals")
    if plotted:
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_rollout_behavior(rollout_df: pd.DataFrame, output_path: Path) -> None:
    if rollout_df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    x = rollout_df["iteration"]
    axes[0].bar(x, rollout_df["transition_rows"], label="transition_rows")
    axes[0].plot(x, rollout_df["progress_ratio"], marker="o", color="darkgreen", label="progress_ratio")
    axes[0].set_title("Rollout Volume / Progress")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, rollout_df["victory_rows"], label="victory_rows", color="#4c78a8")
    axes[1].bar(x, rollout_df["timeout_rows"], label="timeout_rows", color="#e45756")
    axes[1].set_title("Rollout Outcome Rows")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_cohort_heatmap(eval_df: pd.DataFrame, output_path: Path) -> None:
    if eval_df.empty:
        return
    metric_columns = [
        ("fight_win_rate", "Cohort Win Rate"),
        ("timeout_rate", "Cohort Timeout Rate"),
    ]
    fig, axes = plt.subplots(1, len(metric_columns), figsize=(14, 6))
    if len(metric_columns) == 1:
        axes = [axes]
    for axis, (metric, title) in zip(axes, metric_columns, strict=False):
        pivot = eval_df.pivot(index="cohort_name", columns="iteration", values=metric).fillna(0.0)
        image = axis.imshow(pivot.values, aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Cohort")
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels([str(col) for col in pivot.columns])
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels(list(pivot.index), fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    df.to_csv(path, index=False, encoding="utf-8")


def _extract_iteration(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return 0
    return int(digits[:4])


def _extract_manifests(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    manifests = list(metrics.get("manifests") or [])
    if manifests:
        return manifests
    manifest = metrics.get("manifest")
    if isinstance(manifest, dict):
        return [manifest]
    return []
