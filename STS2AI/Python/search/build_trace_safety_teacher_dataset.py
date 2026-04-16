from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PYTHON_ROOT = _THIS_FILE.parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import _path_init  # noqa: F401

import argparse
import json
from collections import Counter
from typing import Any

from training.combat_safety import (
    _alive_enemies,
    _enemy_attack_damage,
    _estimate_block_for_action,
    _incoming_damage,
    _is_block_action,
    _is_self_damage_action,
    _is_setup_action,
    _player,
    _target_enemy,
    rerank_combat_logits_with_safety,
)
from combat_teacher_common import (
    COMBAT_STATE_TYPES,
    canonical_public_state_hash,
    detect_motif_labels,
    estimate_line_continuation_targets,
    load_baseline_combat_policy,
    sanitize_action,
    stable_sample_id,
)
from combat_teacher_dataset import (
    CombatTeacherSample,
    dedupe_samples_by_id,
    stable_split,
    write_combat_teacher_samples,
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _enabled_legal_actions(record: dict[str, Any]) -> list[dict[str, Any]]:
    legal = record.get("candidate_actions")
    if not isinstance(legal, list):
        raw_state = record.get("raw_state") or {}
        legal = raw_state.get("legal_actions")
    if not isinstance(legal, list):
        return []
    return [dict(action) for action in legal if isinstance(action, dict) and action.get("is_enabled") is not False]


def _severe_danger(state: dict[str, Any]) -> tuple[bool, int, int]:
    player = _player(state)
    hp = _safe_int(player.get("current_hp", player.get("hp", 0)), 0)
    max_hp = max(1, _safe_int(player.get("max_hp", 1), 1))
    block = _safe_int(player.get("block", 0), 0)
    incoming = _incoming_damage(state)
    net_incoming = max(0, incoming - block)
    severe = net_incoming >= max(10, int(max_hp * 0.2)) or hp <= max(18, int(max_hp * 0.35))
    return severe, hp, net_incoming


def _lowest_hp_attacker(state: dict[str, Any]) -> dict[str, Any] | None:
    attackers = [enemy for enemy in _alive_enemies(state) if _enemy_attack_damage(enemy) > 0]
    if not attackers:
        return None
    return min(attackers, key=lambda enemy: _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0))


def _teacher_motifs(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    baseline_index: int,
    teacher_index: int,
) -> list[str]:
    if not (0 <= teacher_index < len(legal)) or not (0 <= baseline_index < len(legal)):
        return []

    baseline_action = legal[baseline_index]
    teacher_action = legal[teacher_index]
    severe_danger, _hp, net_incoming = _severe_danger(state)
    has_block_option = any(_estimate_block_for_action(state, action) > 0 for action in legal)

    motifs = set(detect_motif_labels(state, legal))
    if teacher_index == baseline_index:
        return sorted(motifs)

    if severe_danger and has_block_option and _is_block_action(state, teacher_action) and not _is_block_action(state, baseline_action):
        motifs.add("danger_blocking")
    if severe_danger and (_is_self_damage_action(state, baseline_action) or _is_setup_action(state, baseline_action)):
        if _is_block_action(state, teacher_action):
            motifs.add("avoid_greedy_setup")

    low_attacker = _lowest_hp_attacker(state)
    if low_attacker is not None:
        low_hp = _safe_int(low_attacker.get("hp", low_attacker.get("current_hp", 0)), 0)
        teacher_target = _target_enemy(state, teacher_action)
        baseline_target = _target_enemy(state, baseline_action)
        if (
            teacher_target is not None
            and str(teacher_target.get("combat_id")) == str(low_attacker.get("combat_id"))
            and _estimate_block_for_action(state, teacher_action) == 0
        ):
            baseline_target_id = str(baseline_target.get("combat_id")) if baseline_target is not None else ""
            if baseline_target_id != str(low_attacker.get("combat_id")) and low_hp <= 8:
                motifs.add("focus_fire_attacker")

    if severe_danger and net_incoming >= 12 and _lower(baseline_action.get("action")) == "end_turn":
        motifs.add("panic_end_turn")

    if _lower(baseline_action.get("action")) != "end_turn":
        motifs.discard("bad_end_turn")

    return sorted(motifs)


def _build_sample(
    record: dict[str, Any],
    *,
    baseline_policy,
    source_checkpoint: str,
) -> CombatTeacherSample | None:
    state = record.get("raw_state") or {}
    state_type = _lower(state.get("state_type"))
    if state_type not in COMBAT_STATE_TYPES:
        return None
    battle = state.get("battle") or {}
    if _lower(battle.get("turn")) not in {"player", ""}:
        return None

    legal = _enabled_legal_actions(record)
    if len(legal) <= 1:
        return None

    baseline = baseline_policy.score(state, legal)
    base_logits = baseline.get("logits")
    if base_logits is None or len(base_logits) != len(legal):
        return None
    reranked, adjustments = rerank_combat_logits_with_safety(state, legal, base_logits)
    baseline_best = int(baseline.get("best_index", 0))
    teacher_best = int(max(range(len(legal)), key=lambda idx: float(reranked[idx])))
    if teacher_best == baseline_best:
        return None

    motifs = _teacher_motifs(state, legal, baseline_best, teacher_best)
    if not motifs:
        return None

    regrets = [float(max(reranked[teacher_best] - float(score), 0.0)) for score in reranked]
    score_list = [float(score) for score in reranked]
    next_state = record.get("next_state") if isinstance(record.get("next_state"), dict) else state
    continuation = estimate_line_continuation_targets(
        terminal_state=next_state,
        baseline_value=float(baseline.get("value", 0.0)),
        total_potions_used=0,
    )

    seed = str(record.get("seed") or "")
    sample_id = stable_sample_id(seed, state, legal)
    return CombatTeacherSample(
        schema_version="combat_teacher_dataset.v1",
        sample_id=sample_id,
        split=stable_split(sample_id),
        source_bucket="trace_safety",
        source_seed=seed,
        source_checkpoint=source_checkpoint,
        state_hash=canonical_public_state_hash(state),
        motif_labels=motifs,
        state=state,
        legal_actions=legal,
        baseline_logits=[float(value) for value in base_logits],
        baseline_probs=[float(value) for value in baseline.get("probs", [])],
        baseline_best_action_index=baseline_best,
        best_action_index=teacher_best,
        best_full_turn_line=[sanitize_action(legal[teacher_best]) or dict(legal[teacher_best])],
        per_action_score=score_list,
        per_action_regret=regrets,
        root_value=float(baseline.get("value", 0.0)),
        leaf_breakdown={
            "combat_net_value": float(baseline.get("value", 0.0)),
            "teacher_adjustment": float(adjustments[teacher_best]) if teacher_best < len(adjustments) else 0.0,
            "baseline_adjustment": float(adjustments[baseline_best]) if baseline_best < len(adjustments) else 0.0,
            "total": float(score_list[teacher_best]),
        },
        continuation_targets=continuation,
    )


def _iter_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build focused combat_teacher samples from trajectory traces.")
    parser.add_argument(
        "--trajectory-glob",
        default="STS2AI/Artifacts/combat_trace/nn_trajectory/*_trajectory.jsonl",
        help="Glob for trajectory JSONL files.",
    )
    parser.add_argument(
        "--combat-checkpoint",
        default="STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt",
        help="Combat checkpoint used to score baseline logits.",
    )
    parser.add_argument(
        "--output",
        default="STS2AI/Artifacts/combat_teacher/trace_safety_teacher_dataset.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on emitted sample count (0 = unlimited).",
    )
    args = parser.parse_args()

    trajectory_paths = sorted(Path().glob(args.trajectory_glob))
    if not trajectory_paths:
        raise FileNotFoundError(f"No trajectory files matched: {args.trajectory_glob}")

    baseline_policy = load_baseline_combat_policy(args.combat_checkpoint)
    samples: list[CombatTeacherSample] = []
    motif_counts: Counter[str] = Counter()
    per_seed_counts: Counter[str] = Counter()

    for record in _iter_records(trajectory_paths):
        sample = _build_sample(
            record,
            baseline_policy=baseline_policy,
            source_checkpoint=str(args.combat_checkpoint),
        )
        if sample is None:
            continue
        samples.append(sample)
        motif_counts.update(sample.motif_labels)
        per_seed_counts.update([sample.source_seed])
        if args.max_samples > 0 and len(samples) >= int(args.max_samples):
            break

    samples = dedupe_samples_by_id(samples)
    metadata = {
        "trajectory_files": [str(path) for path in trajectory_paths],
        "combat_checkpoint": str(args.combat_checkpoint),
        "motif_counts": dict(sorted(motif_counts.items())),
        "per_seed_counts": dict(sorted(per_seed_counts.items())),
        "train_count": sum(1 for sample in samples if sample.split != "holdout"),
        "holdout_count": sum(1 for sample in samples if sample.split == "holdout"),
    }
    write_combat_teacher_samples(args.output, samples, metadata=metadata)
    print(json.dumps({"output": str(args.output), "sample_count": len(samples), "metadata": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
