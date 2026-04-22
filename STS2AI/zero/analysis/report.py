from __future__ import annotations

"""训练产物分析与可视化。

职责：
- 读取单次训练 run 目录里的 `run_metrics / manifests / raw_runs / eval / logs`
- 生成面向排查的摘要表和 PNG 图
- 输出统一放到当前 run 根目录下的 `analysis/`

这是训练后的离线分析层，不参与训练决策本身。
如果之后要扩展图表，请优先保持：
- 训练指标
- 采样/样本池指标
- 评估指标
- rollout 行为
这四块结构稳定，方便跨 run 对比。
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ENGINE_CARDS = {"FEEL_NO_PAIN", "DARK_EMBRACE"}
PAYOFF_CARDS = {"PACTS_END", "TRUE_GRIT", "PURITY", "PYRE", "SECOND_WIND"}
SUBMENU_ACTION_TYPES = {"select_hand_card", "select_card_option", "confirm_selection", "cancel_selection"}


def generate_training_analysis(*, run_root: Path, run_metrics_path: Path) -> Path:
    # analysis 目录是“单次 run 的可读侧产物”，不作为训练输入回灌。
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
    manifests = _extract_manifests(metrics)

    training_df = _build_training_dataframe(manifests)
    eval_df = _build_eval_dataframe(manifests)
    sampling_df = _build_sampling_dataframe(manifests)
    rollout_df = _build_rollout_dataframe(run_root / "raw_runs")
    episode_df = _build_episode_event_dataframe(run_root / "logs")
    encounter_coverage_df = _build_encounter_coverage_dataframe(run_root)
    encounter_pool_df = _build_encounter_pool_dataframe(run_root)
    turn_order_df, turn_metrics_df = _build_turn_order_dashboard_dataframes(
        raw_runs_dir=run_root / "raw_runs",
        logs_dir=run_root / "logs",
    )

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
    _write_dataframe(encounter_coverage_df, analysis_dir / "encounter_coverage.csv")
    _write_dataframe(encounter_pool_df, analysis_dir / "encounter_pool_stats.csv")
    _write_dataframe(turn_order_df, analysis_dir / "turn_order_dashboard.csv")
    _write_dataframe(turn_metrics_df, analysis_dir / "turn_metrics.csv")

    _plot_training_metrics(training_df, analysis_dir / "training_metrics.png")
    _plot_sampling_metrics(sampling_df, episode_df, analysis_dir / "sampling_metrics.png")
    _plot_pool_diagnostics(sampling_df, analysis_dir / "pool_diagnostics.png")
    _plot_eval_metrics(eval_df, analysis_dir / "evaluation_metrics.png")
    _plot_rollout_behavior(rollout_df, analysis_dir / "rollout_behavior.png")
    _plot_cohort_heatmap(eval_df, analysis_dir / "cohort_overview.png")
    _plot_encounter_coverage(encounter_coverage_df, analysis_dir / "encounter_coverage.png")
    _plot_turn_order_dashboard(turn_order_df, analysis_dir / "turn_order_dashboard.png")
    _write_turn_order_markdown(turn_order_df, analysis_dir / "turn_order_dashboard.md")
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
                    "fight_quality_score": float(metadata.get("fight_quality_score", 0.0) or 0.0),
                    "hp_quality_score": float(metadata.get("hp_quality_score", 0.0) or 0.0),
                    "avg_step_count": float(metadata.get("avg_step_count", 0.0) or 0.0),
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


def _build_encounter_coverage_dataframe(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_root = run_root / "raw_runs"
    if not raw_root.exists():
        return pd.DataFrame()
    for path in sorted(raw_root.glob("iter_*.jsonl")):
        iteration = _extract_iteration(path.name)
        grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
        fight_seen: set[tuple[str, str, int, str]] = set()
        for line in path.open("r", encoding="utf-8"):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            state = row.get("state") or {}
            context = state.get("context") or {}
            metadata = row.get("metadata") or {}
            encounter_id = str(context.get("encounter_id") or "unknown")
            floor = int(context.get("metadata", {}).get("skada_floor", context.get("floor", 0)) or 0)
            encounter_class = str(context.get("encounter_class") or "default")
            key = (encounter_id, floor, encounter_class)
            item = grouped.setdefault(
                key,
                {
                    "iteration": iteration,
                    "encounter_id": encounter_id,
                    "floor": floor,
                    "encounter_class": encounter_class,
                    "collect_episodes": 0,
                    "transition_count": 0,
                    "progress_steps": 0,
                    "no_progress_steps": 0,
                    "victory_rows": 0,
                    "timeout_rows": 0,
                },
            )
            fight_key = (row.get("fight_id", ""), encounter_id, floor, encounter_class)
            if fight_key not in fight_seen:
                fight_seen.add(fight_key)
                item["collect_episodes"] += 1
            item["transition_count"] += 1
            item["progress_steps"] += 1 if bool(metadata.get("made_progress", False)) else 0
            item["no_progress_steps"] += 0 if bool(metadata.get("made_progress", False)) else 1
            outcome = str(row.get("fight_outcome") or "")
            if outcome.lower() in {"victory", "win"}:
                item["victory_rows"] += 1
            if outcome.lower() == "timeout":
                item["timeout_rows"] += 1
        for item in grouped.values():
            total = max(int(item["transition_count"]), 1)
            item["avg_no_progress_ratio"] = float(item["no_progress_steps"]) / total
            item["progress_ratio"] = float(item["progress_steps"]) / total
            rows.append(item)
    return pd.DataFrame(rows)


def _build_encounter_pool_dataframe(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    shard_root = run_root / "dataset_shards"
    if not shard_root.exists():
        return pd.DataFrame()
    for path in sorted(shard_root.glob("iter_*.jsonl")):
        iteration = _extract_iteration(path.name)
        grouped: dict[tuple[str, int, str, str], dict[str, Any]] = {}
        for line in path.open("r", encoding="utf-8"):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            state = row.get("state") or {}
            context = state.get("context") or {}
            metadata = row.get("metadata") or {}
            encounter_id = str(context.get("encounter_id") or "unknown")
            floor = int(context.get("metadata", {}).get("skada_floor", context.get("floor", 0)) or 0)
            encounter_class = str(context.get("encounter_class") or "default")
            pool_name = str(row.get("pool_name") or "unknown")
            key = (encounter_id, floor, encounter_class, pool_name)
            item = grouped.setdefault(
                key,
                {
                    "iteration": iteration,
                    "encounter_id": encounter_id,
                    "floor": floor,
                    "encounter_class": encounter_class,
                    "pool_name": pool_name,
                    "pool_entries": 0,
                    "avg_sample_weight": 0.0,
                    "avg_keep_score": 0.0,
                },
            )
            item["pool_entries"] += 1
            item["avg_sample_weight"] += float(row.get("sample_weight") or metadata.get("sample_weight") or 0.0)
            item["avg_keep_score"] += float(row.get("keep_score") or 0.0)
        for item in grouped.values():
            denom = max(int(item["pool_entries"]), 1)
            item["avg_sample_weight"] /= denom
            item["avg_keep_score"] /= denom
            rows.append(item)
    return pd.DataFrame(rows)


def _build_turn_order_dashboard_dataframes(*, raw_runs_dir: Path, logs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not raw_runs_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    episode_events = _load_episode_events_by_iteration_and_run(logs_dir)
    dashboard_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []

    for path in sorted(raw_runs_dir.glob("iter_*.jsonl")):
        iteration = _extract_iteration(path.name)
        fights: dict[str, list[dict[str, Any]]] = {}
        for line in path.open("r", encoding="utf-8"):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            run_id = str(row.get("run_id") or "")
            fights.setdefault(run_id, []).append(row)

        if not fights:
            continue

        wins = 0
        hp_loss_on_win: list[float] = []
        damage_taken_all: list[float] = []
        victory_end_hp: list[float] = []
        turn_steps: list[int] = []
        submenu_turns = 0
        no_progress_turns = 0
        end_turn_with_options_turns = 0
        long_turns = 0
        threat_damage_total = 0.0
        threat_damage_to_attack = 0.0
        attack_focus_checks = 0
        attack_focus_hits = 0
        lethal_checks = 0
        lethal_hits = 0
        engine_checks = 0
        engine_hits = 0
        quota_checks = 0
        quota_hits = 0
        submenu_sequences = 0
        submenu_overruns = 0
        submenu_latencies: list[int] = []
        max_no_progress_streak_values: list[float] = []

        for run_id, rows in fights.items():
            rows.sort(key=lambda item: int(item.get("step_idx") or 0))
            event = episode_events.get(iteration, {}).get(run_id)
            if isinstance(event, dict) and str(event.get("outcome") or "") == "victory":
                wins += 1

            first = rows[0]
            last = rows[-1]
            start_hp = float(
                ((first.get("state") or {}).get("context") or {}).get("metadata", {}).get(
                    "combat_start_hp",
                    ((first.get("state") or {}).get("player") or {}).get("hp", 0.0),
                )
                or 0.0
            )
            terminal_hp_obs = float((((last.get("next_state") or {}).get("player") or {}).get("hp") or 0.0))
            relics = set(((first.get("state") or {}).get("context") or {}).get("relics") or [])
            heal_after_victory = 6.0 if isinstance(event, dict) and str(event.get("outcome") or "") == "victory" and "BURNING_BLOOD" in relics else 0.0
            est_end_hp = terminal_hp_obs - heal_after_victory
            est_damage_taken = start_hp - est_end_hp
            damage_taken_all.append(est_damage_taken)
            if isinstance(event, dict):
                max_no_progress_streak_values.append(float(event.get("max_no_progress_streak") or 0.0))
                if str(event.get("outcome") or "") == "victory":
                    hp_loss_on_win.append(est_damage_taken)
                    victory_end_hp.append(est_end_hp)

            turns: dict[int, list[dict[str, Any]]] = {}
            for row in rows:
                state = row.get("state") or {}
                turn = ((state.get("context") or {}).get("metadata") or {}).get("round_number_raw")
                if isinstance(turn, int):
                    turns.setdefault(turn, []).append(row)

            for turn, turn_entries in turns.items():
                turn_entries.sort(key=lambda item: int(item.get("step_idx") or 0))
                turn_step_count = len(turn_entries)
                turn_steps.append(turn_step_count)
                if turn_step_count >= 8:
                    long_turns += 1

                has_submenu = False
                any_progress = False
                repeat_streak_max = 0
                repeat_last_action_id = ""
                repeat_count = 0
                first_targeted_attack_checked = False
                engine_order: list[str] = []
                quota_reached = False
                submenu_latency = 0

                for row in turn_entries:
                    state = row.get("state") or {}
                    next_state = row.get("next_state") or {}
                    legal_actions = state.get("legal_actions") or []
                    legal_action_types = {str(item.get("action_type") or "") for item in legal_actions}
                    if ((state.get("context") or {}).get("metadata") or {}).get("state_type") in {"hand_select", "card_select"} or (legal_action_types & SUBMENU_ACTION_TYPES):
                        has_submenu = True

                    metadata = row.get("metadata") or {}
                    if bool(metadata.get("made_progress")) or float(metadata.get("enemy_hp_delta") or 0.0) > 0 or float(metadata.get("enemy_count_delta") or 0.0) > 0:
                        any_progress = True

                    action = row.get("action") or {}
                    action_type = str(action.get("action_type") or "")
                    action_id = str(action.get("action_id") or "")
                    if action_id == repeat_last_action_id:
                        repeat_count += 1
                    else:
                        repeat_last_action_id = action_id
                        repeat_count = 1
                    repeat_streak_max = max(repeat_streak_max, repeat_count)

                    enemies = (state.get("enemies") or [])
                    alive_enemies = [(str(index + 1), enemy) for index, enemy in enumerate(enemies) if bool(enemy.get("alive", True))]
                    attack_enemy_ids = [enemy_id for enemy_id, enemy in alive_enemies if str(enemy.get("intent_id") or "") == "Attack"]
                    mixed_intent = bool(attack_enemy_ids) and any(str(enemy.get("intent_id") or "") != "Attack" for _, enemy in alive_enemies)
                    if mixed_intent and action_type == "play_card" and action.get("target_id"):
                        damage_delta = max(0.0, float(metadata.get("enemy_hp_delta") or 0.0))
                        threat_damage_total += damage_delta
                        if str(action.get("target_id")) in attack_enemy_ids:
                            threat_damage_to_attack += damage_delta
                        if not first_targeted_attack_checked:
                            attack_focus_checks += 1
                            if str(action.get("target_id")) in attack_enemy_ids:
                                attack_focus_hits += 1
                            first_targeted_attack_checked = True

                    lethal_target_ids: set[str] = set()
                    for legal_action in legal_actions:
                        if str(legal_action.get("action_type") or "") != "play_card":
                            continue
                        target_id = str(legal_action.get("target_id") or "")
                        if not target_id:
                            continue
                        try:
                            enemy_index = int(target_id) - 1
                        except ValueError:
                            continue
                        if enemy_index < 0 or enemy_index >= len(enemies):
                            continue
                        enemy = enemies[enemy_index]
                        if not bool(enemy.get("alive", True)):
                            continue
                        hp_plus_block = float(enemy.get("hp") or 0.0) + float(enemy.get("block") or 0.0)
                        if float(legal_action.get("damage_now") or 0.0) >= hp_plus_block and hp_plus_block > 0:
                            lethal_target_ids.add(target_id)
                    if lethal_target_ids:
                        lethal_checks += 1
                        alive_before = sum(1 for enemy in enemies if bool(enemy.get("alive", True)))
                        alive_after = sum(1 for enemy in ((next_state.get("enemies") or [])) if bool(enemy.get("alive", True)))
                        if action_type == "play_card" and str(action.get("target_id") or "") in lethal_target_ids and alive_after < alive_before:
                            lethal_hits += 1

                    if action_type == "play_card":
                        card_id = str(action.get("card_id") or "")
                        if card_id in ENGINE_CARDS:
                            engine_order.append("engine")
                        elif card_id in PAYOFF_CARDS:
                            engine_order.append("payoff")

                    if action_type in {"select_hand_card", "select_card_option"}:
                        next_action_types = [str(item.get("action_type") or "") for item in (next_state.get("legal_actions") or [])]
                        has_confirm = "confirm_selection" in next_action_types
                        has_more_select = bool({"select_hand_card", "select_card_option"} & set(next_action_types))
                        if has_confirm and not has_more_select and not quota_reached:
                            submenu_sequences += 1
                            quota_reached = True
                            submenu_latency = 0
                            continue
                    if quota_reached:
                        if action_type == "confirm_selection":
                            submenu_latencies.append(submenu_latency)
                            quota_checks += 1
                            quota_hits += 1
                            if submenu_latency > 0:
                                submenu_overruns += 1
                            quota_reached = False
                            submenu_latency = 0
                        elif ((state.get("context") or {}).get("metadata") or {}).get("state_type") in {"hand_select", "card_select"} or (legal_action_types & SUBMENU_ACTION_TYPES):
                            submenu_latency += 1
                        else:
                            submenu_latencies.append(submenu_latency)
                            quota_checks += 1
                            if submenu_latency > 0:
                                submenu_overruns += 1
                            quota_reached = False
                            submenu_latency = 0

                if has_submenu:
                    submenu_turns += 1
                if not any_progress:
                    no_progress_turns += 1
                if engine_order.count("engine") > 0 and engine_order.count("payoff") > 0:
                    engine_checks += 1
                    if engine_order.index("engine") < engine_order.index("payoff"):
                        engine_hits += 1

                last_row = turn_entries[-1]
                last_legal_action_types = {
                    str(item.get("action_type") or "")
                    for item in (((last_row.get("state") or {}).get("legal_actions") or []))
                }
                ended_with_options = False
                if str((last_row.get("action") or {}).get("action_type") or "") == "end_turn":
                    meaningful_options = last_legal_action_types - {"end_turn", "confirm_selection", "cancel_selection"}
                    ended_with_options = bool(meaningful_options)
                    if ended_with_options:
                        end_turn_with_options_turns += 1

                turn_rows.append(
                    {
                        "iteration": iteration,
                        "run_id": run_id,
                        "fight_id": str(turn_entries[0].get("fight_id") or ""),
                        "turn": turn,
                        "turn_ref": f"{path.name} | run={run_id} | fight={turn_entries[0].get('fight_id', '')} | turn={turn}",
                        "turn_steps": turn_step_count,
                        "has_submenu": int(has_submenu),
                        "no_progress_turn": int(not any_progress),
                        "ended_with_options": int(ended_with_options),
                        "repeat_streak_max": repeat_streak_max,
                        "source_file": path.name,
                    }
                )

        total_turns = len(turn_steps)
        dashboard_rows.append(
            {
                "iteration": iteration,
                "episodes": len(fights),
                "win_rate": wins / max(len(fights), 1),
                "avg_hp_loss_on_win": _safe_mean(hp_loss_on_win),
                "avg_est_damage_taken": _safe_mean(damage_taken_all),
                "avg_victory_est_end_hp": _safe_mean(victory_end_hp),
                "mean_turn_steps": _safe_mean(turn_steps),
                "p95_turn_steps": _percentile(turn_steps, 0.95),
                "max_turn_steps": max(turn_steps) if turn_steps else 0,
                "submenu_turn_rate": submenu_turns / max(total_turns, 1),
                "submenu_overrun_rate": submenu_overruns / max(submenu_sequences, 1) if submenu_sequences else 0.0,
                "submenu_confirm_latency": _safe_mean(submenu_latencies),
                "no_progress_turn_rate": no_progress_turns / max(total_turns, 1),
                "end_turn_with_options_rate": end_turn_with_options_turns / max(total_turns, 1),
                "threat_damage_share": threat_damage_to_attack / max(threat_damage_total, 1e-9) if threat_damage_total > 0 else 0.0,
                "lethal_conversion_rate": lethal_hits / max(lethal_checks, 1) if lethal_checks else 0.0,
                "attack_focus_rate": attack_focus_hits / max(attack_focus_checks, 1) if attack_focus_checks else 0.0,
                "engine_before_payoff_rate": engine_hits / max(engine_checks, 1) if engine_checks else 0.0,
                "quota_confirm_rate": quota_hits / max(quota_checks, 1) if quota_checks else 0.0,
                "p95_max_no_progress_streak": _percentile(max_no_progress_streak_values, 0.95),
            }
        )

    return pd.DataFrame(dashboard_rows), pd.DataFrame(turn_rows)


def _plot_training_metrics(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = df["iteration"]

    axes[0, 0].plot(x, df["total_loss"], marker="o", label="total")
    for column in ("policy_loss", "value_loss", "delta_loss", "uncertainty_loss"):
        if column in df:
            axes[0, 0].plot(x, df[column], marker="o", label=column.replace("_loss", ""))
    axes[0, 0].set_title("Training Losses")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, df["learning_rate"], marker="o", label="lr")
    axes[0, 1].plot(x, df["grad_norm"], marker="o", label="grad_norm")
    axes[0, 1].set_title("LR / Grad Norm")
    axes[0, 1].legend(fontsize=8)

    if "skipped_non_finite_steps" in df:
        axes[1, 0].plot(x, df["skipped_non_finite_steps"], marker="o", color="crimson", label="skipped_non_finite")
    axes[1, 0].set_title("Stability")
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
        timeout_rate=("timeout_rate", "mean"),
        avg_no_progress_ratio=("avg_no_progress_ratio", "mean"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    x = grouped["iteration"]
    for column in ("fight_win_rate", "enemy_hp_fraction_dealt", "self_hp_fraction_remaining"):
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


def _plot_encounter_coverage(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    grouped = (
        df.groupby(["iteration", "encounter_id"], as_index=False)
        .agg(
            collect_episodes=("collect_episodes", "sum"),
            transition_count=("transition_count", "sum"),
            avg_no_progress_ratio=("avg_no_progress_ratio", "mean"),
        )
    )
    top_encounters = (
        grouped.groupby("encounter_id")["collect_episodes"].sum().sort_values(ascending=False).head(8).index.tolist()
    )
    plotted = grouped[grouped["encounter_id"].isin(top_encounters)]
    if plotted.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for encounter_id, frame in plotted.groupby("encounter_id"):
        axes[0].plot(frame["iteration"], frame["collect_episodes"], marker="o", label=encounter_id)
        axes[1].plot(frame["iteration"], frame["avg_no_progress_ratio"], marker="o", label=encounter_id)
    axes[0].set_title("Per-Encounter Collect Episodes")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Per-Encounter Avg No-Progress Ratio")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_turn_order_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return

    df = df.sort_values("iteration")
    x = df["iteration"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(x, df["win_rate"], marker="o", label="win_rate")
    if "avg_hp_loss_on_win" in df:
        axes[0, 0].plot(x, df["avg_hp_loss_on_win"], marker="o", label="avg_hp_loss_on_win")
    axes[0, 0].set_title("Outcome")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, df["mean_turn_steps"], marker="o", label="mean_turn_steps")
    axes[0, 1].plot(x, df["p95_turn_steps"], marker="o", label="p95_turn_steps")
    axes[0, 1].plot(x, df["p95_max_no_progress_streak"], marker="o", label="p95_no_progress_streak")
    axes[0, 1].set_title("Turn Flow")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x, df["submenu_turn_rate"], marker="o", label="submenu_turn_rate")
    axes[1, 0].plot(x, df["submenu_overrun_rate"], marker="o", label="submenu_overrun_rate")
    axes[1, 0].plot(x, df["quota_confirm_rate"], marker="o", label="quota_confirm_rate")
    axes[1, 0].set_title("Submenu")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, df["threat_damage_share"], marker="o", label="threat_damage_share")
    axes[1, 1].plot(x, df["lethal_conversion_rate"], marker="o", label="lethal_conversion_rate")
    axes[1, 1].plot(x, df["engine_before_payoff_rate"], marker="o", label="engine_before_payoff_rate")
    axes[1, 1].set_title("Decision Quality")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_turn_order_markdown(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    df = df.sort_values("iteration").copy()
    latest = df.iloc[-1]
    lines = [
        "# Turn Order Dashboard",
        "",
        "这份摘要面向单个训练 run，不依赖 old/new 对比。",
        "",
        "## 指标怎么理解",
        "",
        "- `mean_turn_steps / p95_turn_steps / max_turn_steps`：看每回合决策长度有没有收敛。",
        "- `submenu_turn_rate / submenu_overrun_rate / quota_confirm_rate`：看 submenu 是否还在犹豫、改选、拖步数。",
        "- `no_progress_turn_rate / end_turn_with_options_rate / p95_max_no_progress_streak`：看是否经常空转或过早结束回合。",
        "- `threat_damage_share / lethal_conversion_rate / attack_focus_rate`：看是否会处理当前威胁、抓斩杀窗口。",
        "- `engine_before_payoff_rate`：看是否学会先铺引擎、再兑现。",
        "",
        "## 末轮快照",
        "",
        f"- `iteration={int(latest['iteration'])}`",
        f"- `win_rate={_fmt_metric(latest.get('win_rate'))}`",
        f"- `avg_hp_loss_on_win={_fmt_metric(latest.get('avg_hp_loss_on_win'))}`",
        f"- `mean_turn_steps={_fmt_metric(latest.get('mean_turn_steps'))}`",
        f"- `p95_turn_steps={_fmt_metric(latest.get('p95_turn_steps'))}`",
        f"- `submenu_overrun_rate={_fmt_metric(latest.get('submenu_overrun_rate'))}`",
        f"- `quota_confirm_rate={_fmt_metric(latest.get('quota_confirm_rate'))}`",
        f"- `no_progress_turn_rate={_fmt_metric(latest.get('no_progress_turn_rate'))}`",
        f"- `end_turn_with_options_rate={_fmt_metric(latest.get('end_turn_with_options_rate'))}`",
        f"- `threat_damage_share={_fmt_metric(latest.get('threat_damage_share'))}`",
        f"- `lethal_conversion_rate={_fmt_metric(latest.get('lethal_conversion_rate'))}`",
        f"- `engine_before_payoff_rate={_fmt_metric(latest.get('engine_before_payoff_rate'))}`",
        "",
        "## 文件",
        "",
        "- `turn_order_dashboard.csv`：iter 级主表",
        "- `turn_order_dashboard.png`：iter 趋势图",
        "- `turn_metrics.csv`：turn 级附录，必要时回原始轨迹定位",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    df.to_csv(path, index=False, encoding="utf-8")


def _load_episode_events_by_iteration_and_run(logs_dir: Path) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    if not logs_dir.exists():
        return result
    for path in sorted(logs_dir.glob("iter_*.events.jsonl")):
        iteration = _extract_iteration(path.name)
        items: dict[str, dict[str, Any]] = {}
        for line in path.open("r", encoding="utf-8"):
            text = line.strip()
            if not text:
                continue
            event = json.loads(text)
            if event.get("phase") != "collect_episode" or event.get("status") != "completed":
                continue
            run_id = str(event.get("run_id") or "")
            if run_id:
                items[run_id] = event
        if items:
            result[iteration] = items
    return result


def _safe_mean(values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(pd.Series(values, dtype=float).mean())


def _percentile(values: list[Any], quantile: float) -> float:
    if not values:
        return 0.0
    return float(pd.Series(values, dtype=float).quantile(quantile))


def _fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "0.0000"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate single-run training analysis.")
    parser.add_argument("--run-root", required=True, help="Path to one training run directory that contains run_metrics.json")
    parser.add_argument(
        "--run-metrics-path",
        default="",
        help="Optional explicit path to run_metrics.json; defaults to <run-root>/run_metrics.json",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    run_metrics_path = Path(args.run_metrics_path).resolve() if args.run_metrics_path else (run_root / "run_metrics.json")
    analysis_dir = generate_training_analysis(run_root=run_root, run_metrics_path=run_metrics_path)
    print(json.dumps({"analysis_dir": str(analysis_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
