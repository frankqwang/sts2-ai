"""Fixed-seed policy evaluation for LLM adapters."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.data_pipeline.encounter_pool import ACT1_WINNABLE_POOL, filter_encounter_pool
from llm.metrics import write_json
from llm.paths import BASE_MODEL_ID, EVALS_ROOT, ensure_dirs
from llm.training.grpo_rollout import _RolloutPolicy, append_episode_trace_files, rollout_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=str, default="", help="要评测的 LoRA adapter；为空则评测 base model")
    parser.add_argument("--base-model-id", type=str, default=BASE_MODEL_ID)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--episodes-per-encounter", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--port-base", type=int, default=15940)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--encounter-filter", type=str, default="")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    return parser.parse_args()


def _number_stats(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "avg": round(float(sum(ordered)) / len(ordered), 4),
        "max": ordered[-1],
    }


def _episode_summary(ep: Any) -> dict[str, Any]:
    return {
        "encounter_key": ep.encounter_key,
        "encounter_id": ep.encounter_id,
        "encounter_tag": ep.encounter_tag,
        "encounter_label": ep.encounter_label,
        "seed": ep.seed,
        "outcome": ep.outcome,
        "steps": len(ep.steps),
        "duration_s": ep.duration_s,
        "reward": ep.reward,
        "invalid_output": ep.invalid_output,
        "invalid_reason": ep.invalid_reason,
        "quality_flags": ep.quality_flags,
        "quality_summary": ep.quality_summary,
        "final_player_hp": (
            (ep.final_state.get("player") or {}).get("hp")
            or ((ep.final_state.get("battle") or {}).get("player") or {}).get("hp")
        ),
        "final_state_type": ep.final_state.get("state_type"),
    }


def _group_payload(episodes: list[Any]) -> dict[str, Any]:
    outcomes = Counter(str(ep.outcome) for ep in episodes)
    rewards = [float(ep.reward.get("total") or 0.0) for ep in episodes]
    steps = [len(ep.steps) for ep in episodes]
    invalid = sum(1 for ep in episodes if ep.invalid_output)
    quality = Counter()
    mechanism_scores: list[float] = []
    hp_lost: list[float] = []
    sequence_scores: list[float] = []
    defense_scores: list[float] = []
    turns: list[float] = []
    steps_per_turn: list[float] = []
    visible_damage_per_step: list[float] = []
    for ep in episodes:
        quality.update(getattr(ep, "quality_flags", {}) or {})
        summary = getattr(ep, "quality_summary", {}) or {}
        if isinstance(summary.get("mechanism_score"), (int, float)):
            mechanism_scores.append(float(summary["mechanism_score"]))
        if isinstance(summary.get("hp_lost"), (int, float)):
            hp_lost.append(float(summary["hp_lost"]))
        if isinstance(summary.get("sequence_score"), (int, float)):
            sequence_scores.append(float(summary["sequence_score"]))
        if isinstance(summary.get("defense_score"), (int, float)):
            defense_scores.append(float(summary["defense_score"]))
        if isinstance(summary.get("turns"), (int, float)):
            turns.append(float(summary["turns"]))
        if isinstance(summary.get("steps_per_turn"), (int, float)):
            steps_per_turn.append(float(summary["steps_per_turn"]))
        if isinstance(summary.get("visible_damage_per_step"), (int, float)):
            visible_damage_per_step.append(float(summary["visible_damage_per_step"]))
    total = len(episodes)
    victories = outcomes.get("victory", 0)
    return {
        "episodes": total,
        "victories": victories,
        "win_rate": round(victories / total, 4) if total else None,
        "invalid_output_episodes": invalid,
        "invalid_output_episode_rate": round(invalid / total, 4) if total else None,
        "reward": _number_stats(rewards),
        "steps": _number_stats(steps),
        "outcome_counts": {key: int(value) for key, value in outcomes.most_common()},
        "action_quality": {key: int(value) for key, value in quality.most_common()},
        "mechanism_score": _number_stats(mechanism_scores),
        "sequence_score": _number_stats(sequence_scores),
        "defense_score": _number_stats(defense_scores),
        "hp_lost": _number_stats(hp_lost),
        "turns": _number_stats(turns),
        "steps_per_turn": _number_stats(steps_per_turn),
        "visible_damage_per_step": _number_stats(visible_damage_per_step),
    }


def _hard_cases(by_encounter: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, payload in by_encounter.items():
        reward_avg = float((payload.get("reward") or {}).get("avg") or 0.0)
        win_rate = float(payload.get("win_rate") or 0.0)
        invalid_rate = float(payload.get("invalid_output_episode_rate") or 0.0)
        hp_lost_avg = float((payload.get("hp_lost") or {}).get("avg") or 0.0)
        hp_lost_max = float((payload.get("hp_lost") or {}).get("max") or 0.0)
        defense_score = float((payload.get("defense_score") or {}).get("avg") or 1.0)
        rows.append({
            "encounter_key": key,
            "encounter_label": payload.get("encounter_label"),
            "win_rate": win_rate,
            "reward_avg": reward_avg,
            "invalid_output_rate": invalid_rate,
            "hp_lost_avg": hp_lost_avg,
            "hp_lost_max": hp_lost_max,
            "defense_score": defense_score,
            "reason": (
                "invalid_output" if invalid_rate > 0.0 else
                "not_perfect_win" if win_rate < 1.0 else
                "high_hp_loss" if hp_lost_avg >= 8.0 or hp_lost_max >= 12.0 else
                "weak_defense" if defense_score < 0.9 else
                "lowest_reward"
            ),
        })
    primary = [
        row
        for row in rows
        if row["invalid_output_rate"] > 0.0
        or row["win_rate"] < 1.0
        or row["hp_lost_avg"] >= 8.0
        or row["hp_lost_max"] >= 12.0
        or row["defense_score"] < 0.9
    ]
    if primary:
        return sorted(
            primary,
            key=lambda row: (
                row["win_rate"],
                -row["hp_lost_avg"],
                -row["hp_lost_max"],
                row["defense_score"],
                row["reward_avg"],
            ),
        )[:8]
    return sorted(
        rows,
        key=lambda row: (
            row["reward_avg"],
            -row["hp_lost_avg"],
            row["defense_score"],
        ),
    )[: max(1, min(8, len(rows) // 4 or 1))]


def _summary_payload(*, args: argparse.Namespace, run_root: Path, episodes: list[Any], policy_stats: dict[str, int]) -> dict[str, Any]:
    outcomes = Counter(str(ep.outcome) for ep in episodes)
    rewards = [float(ep.reward.get("total") or 0.0) for ep in episodes]
    steps = [len(ep.steps) for ep in episodes]
    durations = [float(ep.duration_s or 0.0) for ep in episodes]
    invalid = sum(1 for ep in episodes if ep.invalid_output)
    quality = Counter()
    mechanism_scores: list[float] = []
    hp_lost: list[float] = []
    sequence_scores: list[float] = []
    defense_scores: list[float] = []
    turns: list[float] = []
    steps_per_turn: list[float] = []
    visible_damage_per_step: list[float] = []
    for ep in episodes:
        quality.update(getattr(ep, "quality_flags", {}) or {})
        summary = getattr(ep, "quality_summary", {}) or {}
        if isinstance(summary.get("mechanism_score"), (int, float)):
            mechanism_scores.append(float(summary["mechanism_score"]))
        if isinstance(summary.get("hp_lost"), (int, float)):
            hp_lost.append(float(summary["hp_lost"]))
        if isinstance(summary.get("sequence_score"), (int, float)):
            sequence_scores.append(float(summary["sequence_score"]))
        if isinstance(summary.get("defense_score"), (int, float)):
            defense_scores.append(float(summary["defense_score"]))
        if isinstance(summary.get("turns"), (int, float)):
            turns.append(float(summary["turns"]))
        if isinstance(summary.get("steps_per_turn"), (int, float)):
            steps_per_turn.append(float(summary["steps_per_turn"]))
        if isinstance(summary.get("visible_damage_per_step"), (int, float)):
            visible_damage_per_step.append(float(summary["visible_damage_per_step"]))
    total = len(episodes)
    victories = outcomes.get("victory", 0)
    grouped: dict[str, list[Any]] = {}
    labels: dict[str, str] = {}
    for ep in episodes:
        grouped.setdefault(ep.encounter_key, []).append(ep)
        labels[ep.encounter_key] = ep.encounter_label
    by_encounter = {key: _group_payload(value) for key, value in grouped.items()}
    for key, payload in by_encounter.items():
        payload["encounter_label"] = labels.get(key, key)

    return {
        "kind": "policy_eval",
        "run_root": str(run_root),
        "step_trace_path": str(run_root / "step_trace.jsonl"),
        "episode_trace_path": str(run_root / "episode_trace.jsonl"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "adapter_dir": args.adapter_dir or None,
        "base_model": args.base_model_id,
        "episodes": total,
        "victories": victories,
        "win_rate": round(victories / total, 4) if total else None,
        "invalid_output_episodes": invalid,
        "invalid_output_episode_rate": round(invalid / total, 4) if total else None,
        "outcome_counts": {key: int(value) for key, value in outcomes.most_common()},
        "reward": _number_stats(rewards),
        "steps": _number_stats(steps),
        "duration_s": _number_stats(durations),
        "policy_stats": policy_stats,
        "action_quality": {key: int(value) for key, value in quality.most_common()},
        "mechanism_score": _number_stats(mechanism_scores),
        "sequence_score": _number_stats(sequence_scores),
        "defense_score": _number_stats(defense_scores),
        "hp_lost": _number_stats(hp_lost),
        "turns": _number_stats(turns),
        "steps_per_turn": _number_stats(steps_per_turn),
        "visible_damage_per_step": _number_stats(visible_damage_per_step),
        "by_encounter": by_encounter,
        "hard_cases": _hard_cases(by_encounter),
        "args": vars(args),
    }


def main() -> None:
    args = parse_args()
    ensure_dirs()
    run_name = args.run_name or f"policy_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root = EVALS_ROOT / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    step_trace_path = run_root / "step_trace.jsonl"
    episode_trace_path = run_root / "episode_trace.jsonl"
    step_trace_path.write_text("", encoding="utf-8")
    episode_trace_path.write_text("", encoding="utf-8")

    pool = ACT1_WINNABLE_POOL
    pool = filter_encounter_pool(pool, args.encounter_filter)
    if not pool:
        raise SystemExit("no encounters matched filter")

    policy = _RolloutPolicy(
        adapter_dir=args.adapter_dir or None,
        base_model_id=args.base_model_id,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        enable_thinking=args.enable_thinking,
        parse_retries=args.parse_retries,
    )

    rng = random.Random(args.seed)
    episodes: list[Any] = []
    t0 = time.monotonic()
    with (run_root / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for enc_idx, spec in enumerate(pool):
            port = args.port_base + enc_idx
            for ep_idx in range(args.episodes_per_encounter):
                seed = f"eval-{args.seed}-{enc_idx}-{ep_idx}-{rng.randint(0, 10**9)}"
                print(f"[policy-eval] {spec.encounter_id} ep={ep_idx} seed={seed}")
                ep = rollout_episode(policy, spec, seed=seed, max_steps=args.max_steps, port=port)
                episodes.append(ep)
                append_episode_trace_files(
                    step_trace_path=step_trace_path,
                    episode_trace_path=episode_trace_path,
                    ep=ep,
                )
                summary = _episode_summary(ep)
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
                print(
                    f"  -> outcome={ep.outcome} reward={ep.reward['total']:.2f} "
                    f"steps={len(ep.steps)} invalid={ep.invalid_output}"
                )

    metrics = _summary_payload(args=args, run_root=run_root, episodes=episodes, policy_stats=dict(policy.stats))
    metrics["runtime_s"] = round(time.monotonic() - t0, 2)
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "hard_cases.json", {"hard_cases": metrics.get("hard_cases", [])})
    write_json(run_root / "manifest.json", {
        "metrics": str(run_root / "metrics.json"),
        "episodes": str(run_root / "episodes.jsonl"),
        "hard_cases": str(run_root / "hard_cases.json"),
    })
    print(f"[policy-eval] metrics -> {run_root / 'metrics.json'}")


if __name__ == "__main__":
    main()
