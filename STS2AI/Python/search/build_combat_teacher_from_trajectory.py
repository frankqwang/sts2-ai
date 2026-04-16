"""Build combat-teacher samples by replaying evaluate_ai trajectory JSONL.

This bypasses the in-builder full-run progression (which reliably stalls on
most seeds at floor 2–6 when driven purely by FullRunPolicyNetworkV2).

Inputs are `full_run_trajectory.v1` JSONL files produced by evaluate_ai's
`--save-trajectory-dir` + `--trajectory-seeds` flags. For each trajectory we:

1. Reset the sim with the recorded seed.
2. Replay every recorded chosen_action in order against the live sim.
3. When we reach a combat/elite/boss state at `floor >= min_sample_floor`,
   run the full-turn solver on the live state and emit teacher samples
   exactly like `build_act1_combat_teacher_v2_dataset.py`.
4. Dedupe by sample_id and write out JSONL + manifest + teacher_eval.

The replay is deterministic against the sim's seed, so solver samples come
from the true states the evaluate_ai run visited, including diverse bosses /
elites that the live-progression builder cannot reach.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PYTHON_ROOT = _THIS_FILE.parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import _path_init  # noqa: F401

import argparse
import atexit
import json
from collections import Counter
from typing import Any

import torch

from search.combat_teacher_common import (
    canonical_public_state_hash,
    is_supported_solver_state,
    load_baseline_combat_policy,
    sanitize_action,
    stable_sample_id,
)
from search.combat_teacher_dataset import (
    CombatTeacherSample,
    dedupe_samples_by_id,
    write_combat_teacher_samples,
)
from search.combat_turn_solver import CombatTurnSolver
from search.combat_turn_teacher_config import load_combat_turn_teacher_config
from ipc.full_run_env import create_full_run_client
from ipc.headless_sim_runner import DEFAULT_DLL_PATH, DEFAULT_REPO_ROOT, start_headless_sim, stop_process
from constants import MAINLINE_CHECKPOINT
from core.vocab import load_vocab

# Re-use helpers from the live builder so the sample shape stays identical.
from search.build_act1_combat_teacher_v2_dataset import (
    _build_samples,
    _enabled_legal_actions,
    _sampling_reasons,
    _write_teacher_eval_report,
)


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _match_action_index(legal_actions: list[dict[str, Any]], chosen: dict[str, Any]) -> int:
    clean = sanitize_action(chosen) or {}
    for idx, action in enumerate(legal_actions):
        if sanitize_action(action) == clean:
            return idx
    # Fallback: match by action type + index if card_index differs after sanitize.
    clean_type = _lower(clean.get("action") or clean.get("type"))
    for idx, action in enumerate(legal_actions):
        if _lower(action.get("action") or action.get("type")) != clean_type:
            continue
        if action.get("index") == clean.get("index") and action.get("target_id") == clean.get("target_id"):
            return idx
    return -1


def _load_trajectory(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    records.sort(key=lambda r: int(r.get("step_index") or 0))
    return records


def _seed_from_trajectory(records: list[dict[str, Any]]) -> str:
    for r in records:
        seed = str(r.get("seed") or "").strip()
        if seed:
            return seed
    return ""


def _process_trajectory(
    *,
    client,
    trajectory_path: Path,
    baseline_policy,
    solver: CombatTurnSolver,
    source_checkpoint: str,
    min_sample_floor: int,
    max_samples_per_floor_per_seed: int,
    max_samples_per_seed: int,
    uncertainty_margin_threshold: float,
    uncertainty_entropy_threshold: float,
    low_hp_attacker_threshold: int,
    danger_net_incoming_threshold: int,
    include_baseline_matches: bool,
    emit_prefix_samples: bool,
    rerun_solver_per_prefix: bool,
) -> tuple[list[CombatTeacherSample], dict[str, Any]]:
    records = _load_trajectory(trajectory_path)
    seed = _seed_from_trajectory(records) or trajectory_path.stem.replace("_trajectory", "")

    if not records:
        return [], {"seed": seed, "trajectory_path": str(trajectory_path), "records": 0}

    state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed, timeout_s=30.0)
    samples: list[CombatTeacherSample] = []
    sampled_reason_counts: Counter[str] = Counter()
    solver_reason_counts: Counter[str] = Counter()
    per_seed_floor_counts: Counter[tuple[str, int]] = Counter()
    considered_states = 0
    supported_states = 0
    replay_errors = 0
    sampled_steps = 0
    seed_sample_count = 0

    for step_idx, record in enumerate(records):
        if max_samples_per_seed > 0 and seed_sample_count >= max_samples_per_seed:
            break

        state_type = _lower(state.get("state_type"))
        if state_type == "game_over" or state.get("terminal"):
            break

        legal_actions = _enabled_legal_actions(state)
        if not legal_actions:
            state = client.get_state()
            legal_actions = _enabled_legal_actions(state)
            if not legal_actions:
                break

        floor = _safe_int((state.get("run") or {}).get("floor") or state.get("floor"), 0)

        if (
            state_type in {"combat", "monster", "elite", "boss"}
            and floor >= int(min_sample_floor)
            and is_supported_solver_state(state)
        ):
            considered_states += 1
            supported_states += 1
            baseline = baseline_policy.score(state, legal_actions)
            sample_reasons = _sampling_reasons(
                state,
                legal_actions,
                baseline["probs"],
                uncertainty_margin_threshold=uncertainty_margin_threshold,
                uncertainty_entropy_threshold=uncertainty_entropy_threshold,
                low_hp_attacker_threshold=low_hp_attacker_threshold,
                danger_net_incoming_threshold=danger_net_incoming_threshold,
            )
            floor_cap_ok = (
                max_samples_per_floor_per_seed <= 0
                or per_seed_floor_counts[(seed, floor)] < int(max_samples_per_floor_per_seed)
            )
            if floor_cap_ok:
                new_samples, build_stats = _build_samples(
                    client=client,
                    state=state,
                    legal_actions=legal_actions,
                    baseline_policy=baseline_policy,
                    solver=solver,
                    seed=seed,
                    source_checkpoint=source_checkpoint,
                    sample_reasons=sample_reasons,
                    include_baseline_matches=include_baseline_matches,
                    emit_prefix_samples=emit_prefix_samples,
                    rerun_solver_per_prefix=rerun_solver_per_prefix,
                )
                for sample in new_samples:
                    samples.append(sample)
                    sampled_reason_counts.update(sample_reasons)
                    solver_reason_counts.update(sample.motif_labels)
                    per_seed_floor_counts[(seed, floor)] += 1
                seed_sample_count += len(new_samples)
                sampled_steps += 1
                if max_samples_per_seed > 0 and seed_sample_count >= max_samples_per_seed:
                    break
                _ = build_stats  # unused for now but available for future stats
        # Replay the original chosen action.
        chosen_action = record.get("chosen_action") or {}
        idx = _match_action_index(legal_actions, chosen_action)
        if idx < 0:
            # Match failure: fall back to action[0] (usually proceed/first_legal) to
            # keep replay marching forward instead of bailing out.
            replay_errors += 1
            idx = 0
        try:
            state = client.act(legal_actions[idx])
        except Exception:
            replay_errors += 1
            try:
                state = client.get_state()
            except Exception:
                break
            if not state or _lower(state.get("state_type")) == "game_over":
                break

    meta = {
        "seed": seed,
        "trajectory_path": str(trajectory_path),
        "records": len(records),
        "considered_states": considered_states,
        "supported_states": supported_states,
        "sampled_steps": sampled_steps,
        "replay_errors": replay_errors,
        "per_seed_floor_counts": {
            f"{seed}:floor_{floor}": count
            for (seed_name, floor), count in per_seed_floor_counts.items()
            if seed_name == seed
            for _ in [0]
        },
        "sampled_reason_counts": dict(sorted(sampled_reason_counts.items())),
        "solver_reason_counts": dict(sorted(solver_reason_counts.items())),
        "seed_sample_count": seed_sample_count,
    }
    return samples, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combat-teacher samples by replaying evaluate_ai trajectories.")
    parser.add_argument("--trajectory-dir", required=True, help="Directory containing *_trajectory.jsonl files.")
    parser.add_argument("--trajectory-glob", default="*_trajectory.jsonl", help="Glob within trajectory-dir.")
    parser.add_argument("--combat-checkpoint", default=str(MAINLINE_CHECKPOINT), help="Combat checkpoint for baseline policy + teacher init.")
    parser.add_argument("--transport", choices=["pipe", "pipe-binary"], default="pipe-binary")
    parser.add_argument("--port", type=int, default=15800)
    parser.add_argument("--auto-launch", action="store_true", default=False)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_DLL_PATH)
    parser.add_argument("--teacher-config", default="STS2AI/Python/configs/combat_turn_teacher_tactical_v1.toml")
    parser.add_argument("--min-sample-floor", type=int, default=14)
    parser.add_argument("--max-samples-per-floor-per-seed", type=int, default=10)
    parser.add_argument("--max-samples-per-seed", type=int, default=40)
    parser.add_argument("--max-player-actions", type=int, default=12)
    parser.add_argument("--uncertainty-margin-threshold", type=float, default=0.16)
    parser.add_argument("--uncertainty-entropy-threshold", type=float, default=0.78)
    parser.add_argument("--low-hp-attacker-threshold", type=int, default=12)
    parser.add_argument("--danger-net-incoming-threshold", type=int, default=10)
    parser.add_argument("--include-baseline-matches", action="store_true", default=False)
    parser.add_argument("--emit-prefix-samples", dest="emit_prefix_samples", action="store_true", default=None)
    parser.add_argument("--no-emit-prefix-samples", dest="emit_prefix_samples", action="store_false")
    parser.add_argument("--rerun-solver-per-prefix", dest="rerun_solver_per_prefix", action="store_true", default=None)
    parser.add_argument("--no-rerun-solver-per-prefix", dest="rerun_solver_per_prefix", action="store_false")
    parser.add_argument("--output", required=True, help="Output JSONL.")
    parser.add_argument("--eval-output-dir", default="", help="Optional eval report directory.")
    args = parser.parse_args()

    teacher_config = load_combat_turn_teacher_config(args.teacher_config).with_overrides(
        emit_prefix_samples=args.emit_prefix_samples,
        rerun_solver_per_prefix=args.rerun_solver_per_prefix,
    )

    traj_dir = Path(args.trajectory_dir)
    trajectories = sorted(traj_dir.glob(args.trajectory_glob))
    if not trajectories:
        raise SystemExit(f"No trajectory JSONL found in {traj_dir} (glob={args.trajectory_glob})")

    spawned_sim_proc = None
    if args.auto_launch:
        protocol = "binary" if args.transport == "pipe-binary" else "json"
        spawned_sim_proc = start_headless_sim(
            port=int(args.port),
            repo_root=args.repo_root,
            dll_path=args.headless_dll,
            connect_timeout_s=20.0,
            protocol=protocol,
        )
        atexit.register(lambda: stop_process(spawned_sim_proc))

    load_vocab()  # warms caches used by solver helpers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_policy = load_baseline_combat_policy(args.combat_checkpoint, vocab=load_vocab(), device=device)

    client = None
    solver = None
    all_samples: list[CombatTeacherSample] = []
    per_trajectory_meta: list[dict[str, Any]] = []
    try:
        client = create_full_run_client(
            use_pipe=True,
            transport=args.transport,
            port=int(args.port),
            ready_timeout_s=20.0,
            request_timeout_s=30.0,
        )
        solver = CombatTurnSolver(
            client,
            baseline_policy,
            max_player_actions=int(args.max_player_actions),
            teacher_config=teacher_config,
        )

        for traj_path in trajectories:
            samples, meta = _process_trajectory(
                client=client,
                trajectory_path=traj_path,
                baseline_policy=baseline_policy,
                solver=solver,
                source_checkpoint=str(args.combat_checkpoint),
                min_sample_floor=int(args.min_sample_floor),
                max_samples_per_floor_per_seed=int(args.max_samples_per_floor_per_seed),
                max_samples_per_seed=int(args.max_samples_per_seed),
                uncertainty_margin_threshold=float(args.uncertainty_margin_threshold),
                uncertainty_entropy_threshold=float(args.uncertainty_entropy_threshold),
                low_hp_attacker_threshold=int(args.low_hp_attacker_threshold),
                danger_net_incoming_threshold=int(args.danger_net_incoming_threshold),
                include_baseline_matches=bool(args.include_baseline_matches),
                emit_prefix_samples=bool(teacher_config.emit_prefix_samples),
                rerun_solver_per_prefix=bool(teacher_config.rerun_solver_per_prefix),
            )
            all_samples.extend(samples)
            per_trajectory_meta.append(meta)
            print(f"{meta['seed']}: samples={len(samples)} considered={meta['considered_states']} errors={meta['replay_errors']}")
    finally:
        if solver is not None:
            try:
                solver.cleanup()
            except Exception:
                pass
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        stop_process(spawned_sim_proc)

    candidate_count = len(all_samples)
    all_samples = dedupe_samples_by_id(all_samples)
    deduped_count = len(all_samples)

    floor_counts: Counter[int] = Counter()
    state_type_counts: Counter[str] = Counter()
    motif_counts: Counter[str] = Counter()
    for sample in all_samples:
        state = sample.state if isinstance(sample.state, dict) else {}
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        floor = _safe_int(run.get("floor") or state.get("floor"), 0)
        floor_counts[floor] += 1
        state_type_counts[_lower(state.get("state_type") or run.get("room_type"))] += 1
        for motif in sample.motif_labels or []:
            motif_counts[motif] += 1

    metadata = {
        "trajectory_dir": str(traj_dir),
        "trajectory_count": len(trajectories),
        "combat_checkpoint": str(args.combat_checkpoint),
        "teacher_config": teacher_config.to_metadata(),
        "emit_prefix_samples": bool(teacher_config.emit_prefix_samples),
        "rerun_solver_per_prefix": bool(teacher_config.rerun_solver_per_prefix),
        "min_sample_floor": int(args.min_sample_floor),
        "max_samples_per_floor_per_seed": int(args.max_samples_per_floor_per_seed),
        "max_samples_per_seed": int(args.max_samples_per_seed),
        "include_baseline_matches": bool(args.include_baseline_matches),
        "candidate_count": int(candidate_count),
        "deduped_count": int(deduped_count),
        "train_count": sum(1 for s in all_samples if s.split != "holdout"),
        "holdout_count": sum(1 for s in all_samples if s.split == "holdout"),
        "per_trajectory": per_trajectory_meta,
        "floor_counts": dict(sorted(floor_counts.items())),
        "state_type_counts": dict(sorted(state_type_counts.items())),
        "motif_counts": dict(sorted(motif_counts.items())),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_combat_teacher_samples(args.output, all_samples, metadata=metadata)
    _write_teacher_eval_report(
        args.output,
        all_samples,
        metadata=metadata,
        teacher_config=teacher_config,
        eval_output_dir=args.eval_output_dir or None,
    )

    summary = {
        "output": str(args.output),
        "sample_count": len(all_samples),
        "floor_counts": dict(floor_counts),
        "state_type_counts": dict(state_type_counts),
        "per_trajectory": per_trajectory_meta,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
