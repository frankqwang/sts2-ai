from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PYTHON_ROOT = _THIS_FILE.parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import _path_init  # noqa: F401  (adds STS2AI/Python library dirs to sys.path)

import argparse
import atexit
import json
import math
from collections import Counter
from typing import Any

import numpy as np
import torch

from combat_safety import (
    _alive_enemies,
    _enemy_attack_damage,
    _estimate_block_for_action,
    _incoming_damage,
    _is_self_damage_action,
    _is_setup_action,
    _player,
    _target_enemy,
)
from combat_teacher_common import (
    COMBAT_TEACHER_SCHEMA_VERSION,
    canonical_public_state_hash,
    detect_motif_labels,
    is_supported_solver_state,
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
from combat_turn_solver import CombatTurnSolver
from full_run_env import create_full_run_client
from headless_sim_runner import DEFAULT_DLL_PATH, DEFAULT_REPO_ROOT, start_headless_sim, stop_process
from rl_encoder_v2 import build_structured_actions, build_structured_state
from rl_policy_v2 import FullRunPolicyNetworkV2, _structured_actions_to_numpy_dict, _structured_state_to_numpy_dict
from sts2ai_paths import MAINLINE_CHECKPOINT
from vocab import Vocab, load_vocab


BALANCE_MOTIF_TARGET_RATIOS: dict[str, float] = {
    "direct_lethal_first_action": 0.10,
    "turn_lethal_no_end_turn": 0.05,
    "danger_blocking": 0.16,
    "focus_fire_window": 0.22,
    "greedy_setup_window": 0.06,
    "bash_before_strike": 0.10,
    "bodyslam_before_block": 0.02,
    "high_leverage_room": 0.06,
}

SAMPLE_PRIORITY_WEIGHTS: dict[str, float] = {
    "direct_lethal_first_action": 8.0,
    "turn_lethal_no_end_turn": 6.0,
    "missed_lethal": 4.0,
    "danger_blocking": 3.5,
    "focus_fire_window": 3.0,
    "greedy_setup_window": 2.5,
    "bash_before_strike": 2.0,
    "bodyslam_before_block": 1.5,
    "high_leverage_room": 1.2,
    "fragile_hp_window": 1.0,
    "uncertain_policy": 0.25,
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _load_seeds(path: str | Path | None, limit: int | None = None) -> list[str]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    seeds: list[str] = []
    if isinstance(payload, list):
        seeds = [str(item).strip() for item in payload if str(item).strip()]
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        seed = str(item.get("seed") or "").strip()
                    else:
                        seed = str(item or "").strip()
                    if seed:
                        seeds.append(seed)
    if limit is not None and limit > 0:
        seeds = seeds[:limit]
    return seeds


def _safe_load_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None)
    }
    model.load_state_dict(filtered, strict=False)


def _infer_ppo_embed_dim(state_dict: dict[str, Any] | None, fallback: int = 32) -> int:
    if isinstance(state_dict, dict):
        weight = state_dict.get("entity_emb.card_embed.weight")
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[1])
    return int(fallback)


def load_noncombat_policy(
    checkpoint_path: str | Path,
    *,
    vocab: Vocab,
    device: torch.device,
) -> FullRunPolicyNetworkV2:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ppo_state = checkpoint.get("ppo_model") or checkpoint.get("model_state_dict")
    if not isinstance(ppo_state, dict):
        raise ValueError(f"Hybrid checkpoint missing ppo_model/model_state_dict: {checkpoint_path}")
    embed_dim = _infer_ppo_embed_dim(ppo_state, 32)
    network = FullRunPolicyNetworkV2(vocab=vocab, embed_dim=embed_dim)
    _safe_load_state_dict(network, ppo_state)
    network.to(device).eval()
    return network


def _build_ppo_tensors(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    vocab: Vocab,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    structured_state = build_structured_state(state, vocab)
    structured_actions = build_structured_actions(state, legal_actions, vocab)
    state_t: dict[str, torch.Tensor] = {}
    for key, value in _structured_state_to_numpy_dict(structured_state).items():
        tensor = torch.tensor(value).unsqueeze(0) if isinstance(value, np.ndarray) else torch.tensor([value])
        if "ids" in key or "idx" in key or "types" in key or "count" in key:
            tensor = tensor.long()
        elif "mask" in key:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        state_t[key] = tensor.to(device)

    action_t: dict[str, torch.Tensor] = {}
    for key, value in _structured_actions_to_numpy_dict(structured_actions).items():
        tensor = torch.tensor(value).unsqueeze(0) if isinstance(value, np.ndarray) else torch.tensor([value])
        if "ids" in key or "types" in key or "indices" in key:
            tensor = tensor.long()
        elif "mask" in key:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        action_t[key] = tensor.to(device)
    return state_t, action_t


def _select_noncombat_action(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    ppo_net: FullRunPolicyNetworkV2,
    *,
    vocab: Vocab,
    device: torch.device,
) -> dict[str, Any]:
    state_t, action_t = _build_ppo_tensors(state, legal_actions, vocab, device)
    with torch.no_grad():
        logits, _value, _deck_q, _boss_ready, _action_adv = ppo_net(state_t, action_t)
    idx = int(logits.squeeze(0)[:len(legal_actions)].argmax().item()) if legal_actions else 0
    return legal_actions[idx] if 0 <= idx < len(legal_actions) else legal_actions[0]


def _enabled_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(action)
        for action in state.get("legal_actions") or []
        if isinstance(action, dict) and action.get("is_enabled") is not False
    ]


def _match_action_index(legal_actions: list[dict[str, Any]], chosen_action: dict[str, Any] | None) -> int:
    clean = sanitize_action(chosen_action) or {}
    for idx, action in enumerate(legal_actions):
        if sanitize_action(action) == clean:
            return idx
    return 0


def _normalized_entropy(probs: list[float] | np.ndarray) -> float:
    arr = np.asarray(list(probs), dtype=np.float32)
    arr = arr[arr > 0]
    if arr.size <= 1:
        return 0.0
    arr = arr / max(float(arr.sum()), 1e-8)
    entropy = float(-(arr * np.log(arr + 1e-8)).sum())
    return entropy / max(math.log(float(arr.size)), 1e-8)


def _best_margin(probs: list[float] | np.ndarray) -> float:
    arr = sorted((float(item) for item in probs), reverse=True)
    if len(arr) < 2:
        return 1.0
    return arr[0] - arr[1]


def _lowest_hp_attacker(state: dict[str, Any]) -> dict[str, Any] | None:
    attackers = [enemy for enemy in _alive_enemies(state) if _enemy_attack_damage(enemy) > 0]
    if not attackers:
        return None
    return min(attackers, key=lambda enemy: _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0))


def _has_action_targeting_enemy(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    enemy: dict[str, Any] | None,
) -> bool:
    if not isinstance(enemy, dict):
        return False
    target_id = str(enemy.get("combat_id") or "")
    if not target_id:
        return False
    for action in legal_actions:
        target = _target_enemy(state, action)
        if target is not None and str(target.get("combat_id") or "") == target_id:
            return True
    return False


def _sampling_reasons(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    baseline_probs: list[float] | np.ndarray,
    *,
    uncertainty_margin_threshold: float,
    uncertainty_entropy_threshold: float,
    low_hp_attacker_threshold: int,
    danger_net_incoming_threshold: int,
) -> list[str]:
    player = _player(state)
    hp = _safe_int(player.get("current_hp", player.get("hp", 0)), 0)
    max_hp = max(1, _safe_int(player.get("max_hp", 1), 1))
    block = _safe_int(player.get("block", 0), 0)
    incoming = _safe_int(_incoming_damage(state), 0)
    net_incoming = max(0, incoming - block)
    hp_frac = hp / max_hp
    has_block_option = any(_estimate_block_for_action(state, action) > 0 for action in legal_actions)
    has_setup_option = any(_is_setup_action(state, action) for action in legal_actions)
    has_self_damage_option = any(_is_self_damage_action(state, action) for action in legal_actions)

    reasons: list[str] = []
    if net_incoming >= max(danger_net_incoming_threshold, int(max_hp * 0.18)) and has_block_option:
        reasons.append("danger_blocking")
    if hp_frac <= 0.45 and net_incoming >= 6 and has_block_option:
        reasons.append("fragile_hp_window")
    if (has_setup_option or has_self_damage_option) and net_incoming >= 8:
        reasons.append("greedy_setup_window")

    low_attacker = _lowest_hp_attacker(state)
    if low_attacker is not None:
        low_hp = _safe_int(low_attacker.get("hp", low_attacker.get("current_hp", 0)), 0)
        if low_hp <= low_hp_attacker_threshold and _has_action_targeting_enemy(state, legal_actions, low_attacker):
            reasons.append("focus_fire_window")

    margin = _best_margin(baseline_probs)
    entropy = _normalized_entropy(baseline_probs)
    if margin <= uncertainty_margin_threshold or entropy >= uncertainty_entropy_threshold:
        reasons.append("uncertain_policy")

    room_type = _lower((state.get("run") or {}).get("room_type") or state.get("state_type"))
    if room_type in {"elite", "boss"}:
        reasons.append("high_leverage_room")

    return sorted(set(reasons))


def _solver_motifs(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    sample_reasons: list[str],
    solution,
) -> list[str]:
    motifs = set(detect_motif_labels(state, legal_actions))
    motifs.update(sample_reasons)
    if float(solution.leaf_breakdown.get("lethal_bonus", 0.0)) > 0.0:
        motifs.add("missed_lethal")
        if len(solution.best_full_turn_line or []) <= 1:
            motifs.add("direct_lethal_first_action")
        else:
            motifs.add("turn_lethal_no_end_turn")
    return sorted(motifs)


def _build_sample(
    *,
    client,
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    baseline_policy,
    solver: CombatTurnSolver,
    seed: str,
    source_checkpoint: str,
    sample_reasons: list[str],
    include_baseline_matches: bool,
) -> CombatTeacherSample | None:
    root_state_id = client.save_state()
    try:
        solution = solver.solve(state, root_state_id=root_state_id)
    finally:
        try:
            client.delete_state(root_state_id)
        except Exception:
            pass

    if not solution.supported or not solution.best_first_action:
        return None

    baseline = baseline_policy.score(state, legal_actions)
    best_action_index = _match_action_index(legal_actions, solution.best_first_action)
    baseline_best_index = int(baseline.get("best_index", 0))
    if not include_baseline_matches and best_action_index == baseline_best_index:
        return None

    per_action_score = [
        float(item.get("score", float("-inf")))
        for item in solution.per_action_score
    ]
    per_action_regret = [
        float(item.get("regret", 1e9) if item.get("regret", float("inf")) != float("inf") else 1e9)
        for item in solution.per_action_regret
    ]
    if best_action_index >= len(legal_actions):
        return None

    sample_id = stable_sample_id(seed, state, legal_actions)
    return CombatTeacherSample(
        schema_version=COMBAT_TEACHER_SCHEMA_VERSION,
        sample_id=sample_id,
        split=stable_split(sample_id),
        source_bucket="act1_solver_v2",
        source_seed=seed,
        source_checkpoint=str(source_checkpoint),
        state_hash=canonical_public_state_hash(state),
        motif_labels=_solver_motifs(state, legal_actions, sample_reasons, solution),
        state=state,
        legal_actions=legal_actions,
        baseline_logits=[float(item) for item in baseline["logits"].tolist()],
        baseline_probs=[float(item) for item in baseline["probs"].tolist()],
        baseline_best_action_index=baseline_best_index,
        best_action_index=best_action_index,
        best_full_turn_line=[dict(item) for item in solution.best_full_turn_line],
        per_action_score=per_action_score,
        per_action_regret=per_action_regret,
        root_value=float(solution.root_value),
        leaf_breakdown={str(key): float(value) for key, value in solution.leaf_breakdown.items()},
        continuation_targets={str(key): float(value) for key, value in solution.continuation_targets.items()},
    )


def _combat_room_type(state: dict[str, Any]) -> str:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    room = _lower(run.get("room_type"))
    if room:
        return room
    return _lower(state.get("state_type"))


def _baseline_regret(sample: CombatTeacherSample) -> float:
    idx = int(sample.baseline_best_action_index)
    if 0 <= idx < len(sample.per_action_regret):
        try:
            return float(sample.per_action_regret[idx])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _sample_priority(sample: CombatTeacherSample) -> float:
    score = 0.0
    labels = set(sample.motif_labels or [])
    for label in labels:
        score += float(SAMPLE_PRIORITY_WEIGHTS.get(label, 0.0))
    score += min(max(_baseline_regret(sample), 0.0), 6.0) * 0.35
    score += min(len(labels), 6) * 0.15
    return float(score)


def _take_balanced_samples(
    samples: list[CombatTeacherSample],
    *,
    target_samples: int,
    per_seed_cap: int,
) -> list[CombatTeacherSample]:
    if target_samples <= 0 or len(samples) <= target_samples:
        return list(samples)

    ranked = sorted(
        samples,
        key=lambda sample: (
            -_sample_priority(sample),
            -_baseline_regret(sample),
            str(sample.sample_id or ""),
        ),
    )
    seed_cap = max(1, int(per_seed_cap))
    selected: list[CombatTeacherSample] = []
    selected_ids: set[str] = set()
    selected_hashes: set[str] = set()
    per_seed_counts: Counter[str] = Counter()

    def try_take(sample: CombatTeacherSample) -> bool:
        sample_id = str(sample.sample_id or "")
        if sample_id in selected_ids:
            return False
        state_hash = str(sample.state_hash or "")
        if state_hash and state_hash in selected_hashes:
            return False
        seed = str(sample.source_seed or "")
        if seed and per_seed_counts[seed] >= seed_cap:
            return False
        selected.append(sample)
        selected_ids.add(sample_id)
        if state_hash:
            selected_hashes.add(state_hash)
        if seed:
            per_seed_counts[seed] += 1
        return True

    quota_targets = {
        motif: max(1, int(round(float(target_samples) * ratio)))
        for motif, ratio in BALANCE_MOTIF_TARGET_RATIOS.items()
    }

    for motif, _quota in quota_targets.items():
        candidates = [sample for sample in ranked if motif in set(sample.motif_labels or [])]
        taken = 0
        for sample in candidates:
            if len(selected) >= target_samples or taken >= _quota:
                break
            if try_take(sample):
                taken += 1

    for sample in ranked:
        if len(selected) >= target_samples:
            break
        try_take(sample)

    return selected[:target_samples]


def _rollout_and_collect(
    *,
    client,
    seeds: list[str],
    noncombat_policy: FullRunPolicyNetworkV2,
    baseline_policy,
    solver: CombatTurnSolver,
    source_checkpoint: str,
    vocab: Vocab,
    device: torch.device,
    floor_limit: int,
    sample_every_combat_step: int,
    max_episode_steps: int,
    max_samples: int,
    max_samples_per_seed: int,
    uncertainty_margin_threshold: float,
    uncertainty_entropy_threshold: float,
    low_hp_attacker_threshold: int,
    danger_net_incoming_threshold: int,
    include_baseline_matches: bool,
) -> tuple[list[CombatTeacherSample], dict[str, Any]]:
    samples: list[CombatTeacherSample] = []
    sampled_reason_counts: Counter[str] = Counter()
    solver_reason_counts: Counter[str] = Counter()
    per_seed_counts: Counter[str] = Counter()
    considered_states = 0
    supported_states = 0

    for seed in seeds:
        if max_samples > 0 and len(samples) >= max_samples:
            break
        seed_sample_count = 0
        state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed, timeout_s=30.0)
        combat_step = 0
        for _ in range(max_episode_steps):
            state_type = _lower(state.get("state_type"))
            if state_type == "game_over" or state.get("terminal"):
                break

            legal_actions = _enabled_legal_actions(state)
            if not legal_actions:
                state = client.get_state()
                continue

            floor = _safe_int((state.get("run") or {}).get("floor") or state.get("floor"), 0)
            act = _safe_int((state.get("run") or {}).get("act") or state.get("act"), 0)

            if state_type in {"combat", "monster", "elite", "boss"} and act <= 1 and floor <= floor_limit:
                considered_states += 1
                if is_supported_solver_state(state):
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
                    structural_reasons = [
                        reason for reason in sample_reasons
                        if reason not in {"uncertain_policy", "high_leverage_room"}
                    ]
                    should_sample = (bool(structural_reasons) or _combat_room_type(state) in {"elite", "boss"}) and (
                        combat_step % max(1, sample_every_combat_step) == 0
                        or _combat_room_type(state) in {"elite", "boss"}
                    )
                    if should_sample and (max_samples_per_seed <= 0 or seed_sample_count < max_samples_per_seed):
                        sample = _build_sample(
                            client=client,
                            state=state,
                            legal_actions=legal_actions,
                            baseline_policy=baseline_policy,
                            solver=solver,
                            seed=seed,
                            source_checkpoint=source_checkpoint,
                            sample_reasons=sample_reasons,
                            include_baseline_matches=include_baseline_matches,
                        )
                        if sample is not None:
                            samples.append(sample)
                            seed_sample_count += 1
                            sampled_reason_counts.update(sample_reasons)
                            solver_reason_counts.update(sample.motif_labels)
                            per_seed_counts.update([seed])
                            if max_samples > 0 and len(samples) >= max_samples:
                                break
                baseline_choice = int(baseline_policy.score(state, legal_actions)["best_index"])
                action = legal_actions[baseline_choice] if 0 <= baseline_choice < len(legal_actions) else legal_actions[0]
                combat_step += 1
            else:
                action = _select_noncombat_action(state, legal_actions, noncombat_policy, vocab=vocab, device=device)

            state = client.act(action)

            next_floor = _safe_int((state.get("run") or {}).get("floor") or state.get("floor"), 0)
            next_act = _safe_int((state.get("run") or {}).get("act") or state.get("act"), 0)
            if next_act > 1 or next_floor > floor_limit + 1:
                break

    metadata = {
        "considered_states": considered_states,
        "supported_states": supported_states,
        "sampled_reason_counts": dict(sorted(sampled_reason_counts.items())),
        "solver_reason_counts": dict(sorted(solver_reason_counts.items())),
        "per_seed_counts": dict(sorted(per_seed_counts.items())),
    }
    return samples, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IRONCLAD Act1 combat teacher v2 dataset from live solver rollouts.")
    parser.add_argument("--hybrid-checkpoint", default=str(MAINLINE_CHECKPOINT), help="Hybrid checkpoint for non-combat progression.")
    parser.add_argument("--combat-checkpoint", default=str(MAINLINE_CHECKPOINT), help="Combat checkpoint used for baseline policy and teacher init.")
    parser.add_argument("--seed-file", default="", help="Optional JSON seed file.")
    parser.add_argument("--seed", action="append", default=[], help="Explicit seed(s). Can be repeated.")
    parser.add_argument("--num-seeds", type=int, default=12, help="Seed count limit when using --seed-file or generated defaults.")
    parser.add_argument("--transport", choices=["pipe", "pipe-binary"], default="pipe-binary")
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--auto-launch", action="store_true", default=False)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_DLL_PATH)
    parser.add_argument("--floor-limit", type=int, default=17)
    parser.add_argument("--sample-every-combat-step", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--max-player-actions", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=120, help="Candidate pool cap before balancing.")
    parser.add_argument("--target-samples", type=int, default=0, help="Optional final balanced dataset size (0 keeps all collected samples).")
    parser.add_argument("--max-samples-per-seed", type=int, default=10)
    parser.add_argument("--balanced-seed-cap", type=int, default=12, help="Max kept samples per seed after balanced selection.")
    parser.add_argument("--uncertainty-margin-threshold", type=float, default=0.16)
    parser.add_argument("--uncertainty-entropy-threshold", type=float, default=0.78)
    parser.add_argument("--low-hp-attacker-threshold", type=int, default=12)
    parser.add_argument("--danger-net-incoming-threshold", type=int, default=10)
    parser.add_argument("--include-baseline-matches", action="store_true", default=False)
    parser.add_argument("--output", default="artifacts/combat_teacher/ironclad_act1_solver_v2_dataset.jsonl")
    args = parser.parse_args()

    seeds = [str(seed).strip() for seed in args.seed if str(seed).strip()]
    if args.seed_file:
        seeds.extend(_load_seeds(args.seed_file, limit=args.num_seeds))
    if not seeds:
        seeds = [f"EVAL_{idx:03d}" for idx in range(1, int(args.num_seeds) + 1)]
    seeds = list(dict.fromkeys(seeds))
    if args.num_seeds > 0:
        seeds = seeds[: int(args.num_seeds)]

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

    vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noncombat_policy = load_noncombat_policy(args.hybrid_checkpoint, vocab=vocab, device=device)
    baseline_policy = load_baseline_combat_policy(args.combat_checkpoint, vocab=vocab, device=device)
    solver = None
    client = None
    try:
        client = create_full_run_client(
            use_pipe=True,
            transport=args.transport,
            port=int(args.port),
            ready_timeout_s=20.0,
            request_timeout_s=30.0,
        )
        solver = CombatTurnSolver(client, baseline_policy, max_player_actions=int(args.max_player_actions))
        samples, stats = _rollout_and_collect(
            client=client,
            seeds=seeds,
            noncombat_policy=noncombat_policy,
            baseline_policy=baseline_policy,
            solver=solver,
            source_checkpoint=str(args.combat_checkpoint),
            vocab=vocab,
            device=device,
            floor_limit=int(args.floor_limit),
            sample_every_combat_step=int(args.sample_every_combat_step),
            max_episode_steps=int(args.max_episode_steps),
            max_samples=int(args.max_samples),
            max_samples_per_seed=int(args.max_samples_per_seed),
            uncertainty_margin_threshold=float(args.uncertainty_margin_threshold),
            uncertainty_entropy_threshold=float(args.uncertainty_entropy_threshold),
            low_hp_attacker_threshold=int(args.low_hp_attacker_threshold),
            danger_net_incoming_threshold=int(args.danger_net_incoming_threshold),
            include_baseline_matches=bool(args.include_baseline_matches),
        )
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

    candidate_count = len(samples)
    samples = dedupe_samples_by_id(samples)
    deduped_count = len(samples)
    if int(args.target_samples) > 0:
        samples = _take_balanced_samples(
            samples,
            target_samples=int(args.target_samples),
            per_seed_cap=int(args.balanced_seed_cap),
        )
    metadata = {
        "hybrid_checkpoint": str(args.hybrid_checkpoint),
        "combat_checkpoint": str(args.combat_checkpoint),
        "seeds": seeds,
        "transport": args.transport,
        "floor_limit": int(args.floor_limit),
        "sample_every_combat_step": int(args.sample_every_combat_step),
        "max_episode_steps": int(args.max_episode_steps),
        "max_player_actions": int(args.max_player_actions),
        "max_samples_per_seed": int(args.max_samples_per_seed),
        "balanced_seed_cap": int(args.balanced_seed_cap),
        "candidate_count": int(candidate_count),
        "deduped_count": int(deduped_count),
        "include_baseline_matches": bool(args.include_baseline_matches),
        "train_count": sum(1 for sample in samples if sample.split != "holdout"),
        "holdout_count": sum(1 for sample in samples if sample.split == "holdout"),
        **stats,
    }
    write_combat_teacher_samples(args.output, samples, metadata=metadata)
    print(json.dumps({"output": str(args.output), "sample_count": len(samples), "metadata": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
