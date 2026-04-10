from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from combat_teacher_common import (
    COMBAT_TEACHER_SCHEMA_VERSION,
    BaselineCombatPolicy,
    canonical_public_state_hash,
    detect_motif_labels,
    is_supported_solver_state,
    load_baseline_combat_policy,
    sanitize_action,
    stable_sample_id,
)
from combat_teacher_dataset import (
    CombatTeacherSample,
    load_combat_teacher_samples,
    sample_metric_applicable,
    stable_split,
    write_combat_teacher_samples,
)
from combat_teacher_regression_samples import build_regression_motif_samples
from combat_turn_solver import CombatTurnSolver
from full_run_env import create_full_run_client
from rl_encoder_v2 import build_structured_actions, build_structured_state
from rl_policy_v2 import FullRunPolicyNetworkV2, _structured_actions_to_numpy_dict, _structured_state_to_numpy_dict
from vocab import Vocab, load_vocab

REAL_SOURCE_FRACTIONS: dict[str, float] = {
    "on_policy": 0.80,
    "historical": 0.20,
}

CORE_ANCHOR_COUNTS: dict[str, int] = {
    # Keep anchors small relative to the real on-policy pool, but allocate
    # enough coverage for the brittle tactical motifs that regressed under the
    # pure real-first mix.
    "direct_lethal_first_action": 10,
    "turn_lethal_no_end_turn": 4,
    "bash_before_strike": 6,
    "bodyslam_before_block": 2,
    "potion_misuse": 2,
}

MIN_HOLDOUT_ANCHOR_COUNTS: dict[str, int] = {
    # Keep a minimal, stable holdout for the brittle tactical motifs so
    # broader real/historical mining does not starve evaluation coverage.
    "direct_lethal_first_action": 2,
    "turn_lethal_no_end_turn": 1,
    "bash_before_strike": 2,
    "bodyslam_before_block": 1,
    "potion_misuse": 1,
}

MOTIF_ORDER = (
    "direct_lethal_first_action",
    "turn_lethal_no_end_turn",
    "bash_before_strike",
    "bodyslam_before_block",
    "potion_misuse",
    "bad_end_turn",
)

TARGETED_REAL_HARD_MOTIFS = frozenset({
    "direct_lethal_first_action",
    "bash_before_strike",
})

ON_POLICY_HOLDOUT_PRIORITY_MOTIFS = frozenset({
    "direct_lethal_first_action",
})


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


def _select_combat_action(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    baseline_policy: BaselineCombatPolicy,
) -> dict[str, Any]:
    scored = baseline_policy.score(state, legal_actions)
    idx = int(scored["best_index"]) if legal_actions else 0
    return legal_actions[idx] if 0 <= idx < len(legal_actions) else legal_actions[0]


def _match_action_index(legal_actions: list[dict[str, Any]], chosen_action: dict[str, Any] | None) -> int:
    clean = sanitize_action(chosen_action) or {}
    for idx, action in enumerate(legal_actions):
        if sanitize_action(action) == clean:
            return idx
    if not legal_actions:
        return 0
    return 0


def _build_sample(
    *,
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    seed: str,
    source_bucket: str,
    source_checkpoint: str,
    baseline_policy: BaselineCombatPolicy,
    solver: CombatTurnSolver,
    client,
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
    if not (0 <= best_action_index < len(legal_actions)):
        return None

    motif_labels = detect_motif_labels(state, legal_actions)
    if float(solution.leaf_breakdown.get("lethal_bonus", 0.0)) > 0.0:
        lethal_labels = ["missed_lethal"]
        if len(solution.best_full_turn_line or []) <= 1:
            lethal_labels.append("direct_lethal_first_action")
        else:
            lethal_labels.append("turn_lethal_no_end_turn")
        motif_labels = sorted(set(motif_labels + lethal_labels))

    sample_id = stable_sample_id(seed, state, legal_actions)
    per_action_score = []
    for item in solution.per_action_score:
        per_action_score.append(float(item.get("score", float("-inf"))))
    per_action_regret = []
    for item in solution.per_action_regret:
        regret = item.get("regret", float("inf"))
        per_action_regret.append(float(regret if regret != float("inf") else 1e9))

    return CombatTeacherSample(
        schema_version=COMBAT_TEACHER_SCHEMA_VERSION,
        sample_id=sample_id,
        split=stable_split(sample_id),
        source_bucket=source_bucket,
        source_seed=seed,
        source_checkpoint=str(source_checkpoint),
        state_hash=canonical_public_state_hash(state),
        motif_labels=motif_labels,
        state=state,
        legal_actions=legal_actions,
        baseline_logits=[float(item) for item in baseline["logits"].tolist()],
        baseline_probs=[float(item) for item in baseline["probs"].tolist()],
        baseline_best_action_index=int(baseline["best_index"]),
        best_action_index=best_action_index,
        best_full_turn_line=[dict(item) for item in solution.best_full_turn_line],
        per_action_score=per_action_score,
        per_action_regret=per_action_regret,
        root_value=float(solution.root_value),
        leaf_breakdown={str(key): float(value) for key, value in solution.leaf_breakdown.items()},
        continuation_targets={str(key): float(value) for key, value in solution.continuation_targets.items()},
    )


def _action_card_id(action: dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return ""
    for key in ("card_id", "id", "label"):
        value = action.get(key)
        if value is not None:
            text = str(value).strip().upper()
            if text:
                return text
    return str(action.get("action") or "").strip().upper()


def _is_end_turn_action(action: dict[str, Any]) -> bool:
    return str((action or {}).get("action") or "").strip().lower() == "end_turn"


def _is_use_potion_action(action: dict[str, Any]) -> bool:
    return str((action or {}).get("action") or "").strip().lower() == "use_potion"


def _sample_has_card(sample: CombatTeacherSample, card_token: str) -> bool:
    token = str(card_token).strip().upper()
    for action in sample.legal_actions:
        if str(action.get("action") or "").strip().lower() != "play_card":
            continue
        if _action_card_id(action) == token:
            return True
    return False


def _sample_matches_motif(sample: CombatTeacherSample, motif: str) -> bool:
    return sample_metric_applicable(sample, motif)


def _strict_unique_count(
    samples: list[CombatTeacherSample],
    motif: str,
    *,
    split: str | None = None,
) -> int:
    sample_ids = {
        str(sample.sample_id or "")
        for sample in samples
        if (split is None or str(sample.split or "") == split) and _sample_matches_motif(sample, motif)
    }
    return len(sample_ids)


def _take_samples(
    pool: list[CombatTeacherSample],
    target_count: int,
    *,
    rng: random.Random,
    allow_repeat: bool = True,
) -> list[CombatTeacherSample]:
    if target_count <= 0 or not pool:
        return []
    if len(pool) >= target_count:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled[:target_count]
    if not allow_repeat:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled
    return [pool[rng.randrange(len(pool))] for _ in range(target_count)]


def _dedupe_samples_by_id(samples: list[CombatTeacherSample]) -> list[CombatTeacherSample]:
    seen: set[str] = set()
    unique: list[CombatTeacherSample] = []
    for sample in samples:
        sample_id = str(sample.sample_id or "")
        if sample_id in seen:
            continue
        seen.add(sample_id)
        unique.append(sample)
    return unique


def _sample_baseline_regret(sample: CombatTeacherSample) -> float:
    idx = int(sample.baseline_best_action_index)
    if 0 <= idx < len(sample.per_action_regret):
        return float(sample.per_action_regret[idx])
    return 0.0


def _source_bucket_priority(sample: CombatTeacherSample) -> int:
    bucket = str(sample.source_bucket or "").strip().lower()
    if bucket == "on_policy":
        return 0
    if bucket == "historical":
        return 1
    if bucket == "motif_regression":
        return 3
    return 2


def _split_priority(sample: CombatTeacherSample) -> int:
    split = str(sample.split or "").strip().lower()
    if split == "train":
        return 0
    if split == "holdout":
        return 1
    return 2


def _take_prioritized_samples(
    pool: list[CombatTeacherSample],
    target_count: int,
    *,
    selected_ids: set[str],
    motif_name: str | None = None,
    prefer_diverse_state_hash: bool = False,
) -> list[CombatTeacherSample]:
    if target_count <= 0 or not pool:
        return []
    available = [sample for sample in pool if str(sample.sample_id or "") not in selected_ids]
    motif_token = str(motif_name or "").strip().lower()
    if motif_token in TARGETED_REAL_HARD_MOTIFS:
        ranked = sorted(
            available,
            key=lambda sample: (
                _split_priority(sample),
                -_sample_baseline_regret(sample),
                _source_bucket_priority(sample),
                str(sample.state_hash or ""),
                str(sample.sample_id or ""),
            ),
        )
    else:
        ranked = sorted(
            available,
            key=lambda sample: (
                _split_priority(sample),
                _source_bucket_priority(sample),
                -_sample_baseline_regret(sample),
                str(sample.sample_id or ""),
            ),
        )
    if not prefer_diverse_state_hash:
        return ranked[:target_count]

    taken: list[CombatTeacherSample] = []
    taken_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for sample in ranked:
        if len(taken) >= target_count:
            break
        state_hash = str(sample.state_hash or "")
        if state_hash and state_hash in seen_hashes:
            continue
        taken.append(sample)
        taken_ids.add(str(sample.sample_id or ""))
        if state_hash:
            seen_hashes.add(state_hash)
    if len(taken) >= target_count:
        return taken[:target_count]
    for sample in ranked:
        if len(taken) >= target_count:
            break
        sample_id = str(sample.sample_id or "")
        if sample_id in taken_ids:
            continue
        taken.append(sample)
        taken_ids.add(sample_id)
    return taken[:target_count]


def _collect_samples_for_checkpoint(
    *,
    client,
    seeds: list[str],
    hybrid_policy: FullRunPolicyNetworkV2,
    combat_policy: BaselineCombatPolicy,
    solver: CombatTurnSolver,
    vocab: Vocab,
    device: torch.device,
    source_bucket: str,
    source_checkpoint: str,
    sample_every_combat_step: int,
    max_episode_steps: int,
) -> list[CombatTeacherSample]:
    samples: list[CombatTeacherSample] = []
    for seed in seeds:
        state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed, timeout_s=30.0)
        combat_step = 0
        for _ in range(max_episode_steps):
            state_type = str(state.get("state_type") or "").strip().lower()
            if state_type == "game_over" or state.get("terminal"):
                break
            legal_actions = [
                action for action in state.get("legal_actions") or []
                if isinstance(action, dict) and action.get("is_enabled") is not False
            ]
            if not legal_actions:
                state = client.get_state()
                continue

            if state_type in {"combat", "monster", "elite", "boss"}:
                if is_supported_solver_state(state):
                    if combat_step % max(1, sample_every_combat_step) == 0:
                        sample = _build_sample(
                            state=state,
                            legal_actions=legal_actions,
                            seed=seed,
                            source_bucket=source_bucket,
                            source_checkpoint=source_checkpoint,
                            baseline_policy=combat_policy,
                            solver=solver,
                            client=client,
                        )
                        if sample is not None:
                            samples.append(sample)
                action = _select_combat_action(state, legal_actions, combat_policy)
                combat_step += 1
            else:
                action = _select_noncombat_action(state, legal_actions, hybrid_policy, vocab=vocab, device=device)
            state = client.act(action)
    return samples


def _load_seeds(path: str | Path, limit: int | None = None) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    seeds: list[str] = []
    if isinstance(payload, list):
        seeds = [str(item) for item in payload]
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        seed = str(item.get("seed") or "").strip()
                        if seed:
                            seeds.append(seed)
                    elif item:
                        seeds.append(str(item))
    if limit is not None:
        seeds = seeds[:limit]
    return seeds


def _load_manual_motif_samples(path: str | Path) -> list[CombatTeacherSample]:
    raw_path = Path(path)
    if raw_path.suffix.lower() == ".jsonl":
        return load_combat_teacher_samples(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [CombatTeacherSample.from_dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return [CombatTeacherSample.from_dict(item) for item in payload["samples"] if isinstance(item, dict)]
    return []


def _assemble_dataset(
    on_policy_samples: list[CombatTeacherSample],
    historical_samples: list[CombatTeacherSample],
    motif_samples: list[CombatTeacherSample],
    *,
    target_samples: int,
    rng_seed: int,
    historical_final_fraction: float = REAL_SOURCE_FRACTIONS["historical"],
) -> list[CombatTeacherSample]:
    rng = random.Random(rng_seed)
    on_policy = _dedupe_samples_by_id(on_policy_samples)
    historical = _dedupe_samples_by_id(historical_samples)
    motif = _dedupe_samples_by_id(motif_samples)
    rng.shuffle(on_policy)
    rng.shuffle(historical)
    rng.shuffle(motif)

    if target_samples <= 0:
        return on_policy + historical + motif

    selected: list[CombatTeacherSample] = []
    selected_ids: set[str] = set()
    motif_pools = {
        motif_name: [sample for sample in motif if _sample_matches_motif(sample, motif_name)]
        for motif_name in MOTIF_ORDER
    }
    real_anchor_pool = list(on_policy) + list(historical)
    holdout_anchor_selected_count: dict[str, int] = {}
    holdout_anchor_regression_fallback_count: dict[str, int] = {}
    for motif_name, target_for_motif in MIN_HOLDOUT_ANCHOR_COUNTS.items():
        holdout_real_candidates = [
            sample
            for sample in real_anchor_pool
            if str(sample.split or "").strip().lower() == "holdout" and _sample_matches_motif(sample, motif_name)
        ]
        if motif_name in ON_POLICY_HOLDOUT_PRIORITY_MOTIFS:
            holdout_on_policy_candidates = [
                sample
                for sample in holdout_real_candidates
                if str(sample.source_bucket or "").strip().lower() == "on_policy"
            ]
            taken = _take_prioritized_samples(
                holdout_on_policy_candidates,
                int(max(0, target_for_motif)),
                selected_ids=selected_ids,
                motif_name=motif_name,
                prefer_diverse_state_hash=motif_name in TARGETED_REAL_HARD_MOTIFS,
            )
            if len(taken) < target_for_motif:
                taken.extend(
                    _take_prioritized_samples(
                        holdout_real_candidates,
                        int(max(0, target_for_motif - len(taken))),
                        selected_ids=selected_ids.union(str(sample.sample_id or "") for sample in taken),
                        motif_name=motif_name,
                        prefer_diverse_state_hash=motif_name in TARGETED_REAL_HARD_MOTIFS,
                    )
                )
        else:
            taken = _take_prioritized_samples(
                holdout_real_candidates,
                int(max(0, target_for_motif)),
                selected_ids=selected_ids,
                motif_name=motif_name,
                prefer_diverse_state_hash=motif_name in TARGETED_REAL_HARD_MOTIFS,
            )
        if len(taken) < target_for_motif:
            fallback_needed = target_for_motif - len(taken)
            holdout_regression_fallback_pool = [
                sample
                for sample in motif_pools.get(motif_name, [])
                if str(sample.split or "").strip().lower() == "holdout"
                and str(sample.source_bucket or "").strip().lower() == "motif_regression"
            ]
            fallback_taken = _take_prioritized_samples(
                holdout_regression_fallback_pool,
                fallback_needed,
                selected_ids=selected_ids,
                motif_name=motif_name,
                prefer_diverse_state_hash=motif_name in TARGETED_REAL_HARD_MOTIFS,
            )
            if len(fallback_taken) < fallback_needed:
                generic_holdout_pool = [
                    sample
                    for sample in motif_pools.get(motif_name, [])
                    if str(sample.split or "").strip().lower() == "holdout"
                ]
                fallback_taken.extend(
                    _take_prioritized_samples(
                        generic_holdout_pool,
                        fallback_needed - len(fallback_taken),
                        selected_ids=selected_ids,
                        motif_name=motif_name,
                        prefer_diverse_state_hash=motif_name in TARGETED_REAL_HARD_MOTIFS,
                    )
                )
            if len(fallback_taken) < fallback_needed:
                generic_holdout_pool = [
                    sample
                    for sample in motif_pools.get(motif_name, [])
                    if str(sample.split or "").strip().lower() == "holdout"
                ]
                fallback_taken.extend(
                    _take_samples(
                        generic_holdout_pool,
                        fallback_needed - len(fallback_taken),
                        rng=rng,
                        allow_repeat=True,
                    )
                )
            taken.extend(fallback_taken)
            holdout_anchor_regression_fallback_count[motif_name] = sum(
                1 for sample in fallback_taken if str(sample.source_bucket or "").strip().lower() == "motif_regression"
            )
        else:
            holdout_anchor_regression_fallback_count[motif_name] = 0
        selected.extend(taken)
        selected_ids.update(str(sample.sample_id or "") for sample in taken)
        holdout_anchor_selected_count[motif_name] = len(taken)

    anchor_selected_count: dict[str, int] = {}
    anchor_real_selected_count: dict[str, int] = {}
    anchor_regression_fallback_count: dict[str, int] = {}
    for motif_name, target_for_motif in CORE_ANCHOR_COUNTS.items():
        real_candidates = [sample for sample in real_anchor_pool if _sample_matches_motif(sample, motif_name)]
        taken = _take_prioritized_samples(
            real_candidates,
            int(max(0, target_for_motif)),
            selected_ids=selected_ids,
            motif_name=motif_name,
            prefer_diverse_state_hash=motif_name in TARGETED_REAL_HARD_MOTIFS,
        )
        if len(taken) < target_for_motif:
            fallback_needed = target_for_motif - len(taken)
            regression_fallback_pool = [
                sample
                for sample in motif_pools.get(motif_name, [])
                if str(sample.source_bucket or "").strip().lower() == "motif_regression"
            ]
            fallback_taken = _take_prioritized_samples(
                regression_fallback_pool,
                fallback_needed,
                selected_ids=selected_ids,
                motif_name=motif_name,
                prefer_diverse_state_hash=motif_name in TARGETED_REAL_HARD_MOTIFS,
            )
            if len(fallback_taken) < fallback_needed:
                generic_fallback_pool = list(motif_pools.get(motif_name, []))
                fallback_taken.extend(
                    _take_prioritized_samples(
                        generic_fallback_pool,
                        fallback_needed - len(fallback_taken),
                        selected_ids=selected_ids,
                        motif_name=motif_name,
                        prefer_diverse_state_hash=motif_name in TARGETED_REAL_HARD_MOTIFS,
                    )
                )
            if len(fallback_taken) < fallback_needed:
                generic_fallback_pool = list(motif_pools.get(motif_name, []))
                fallback_taken.extend(_take_samples(generic_fallback_pool, fallback_needed - len(fallback_taken), rng=rng, allow_repeat=True))
            taken.extend(fallback_taken)
            anchor_regression_fallback_count[motif_name] = sum(
                1 for sample in fallback_taken if str(sample.source_bucket or "").strip().lower() == "motif_regression"
            )
        else:
            anchor_regression_fallback_count[motif_name] = 0
        selected.extend(taken)
        selected_ids.update(str(sample.sample_id or "") for sample in taken)
        anchor_selected_count[motif_name] = len(taken)
        anchor_real_selected_count[motif_name] = sum(
            1 for sample in taken if str(sample.source_bucket or "").strip().lower() in {"on_policy", "historical"}
        )

    motif_selected_count: dict[str, int] = {
        motif_name: sum(1 for sample in selected if _sample_matches_motif(sample, motif_name))
        for motif_name in MOTIF_ORDER
    }

    remaining = max(0, target_samples - len(selected))
    if remaining > 0:
        on_policy_pool = [sample for sample in on_policy if str(sample.sample_id or "") not in selected_ids]
        historical_pool = [sample for sample in historical if str(sample.sample_id or "") not in selected_ids]
        historical_fraction = float(max(0.0, min(1.0, historical_final_fraction)))
        historical_target = min(len(historical_pool), int(round(remaining * historical_fraction))) if historical_pool else 0
        on_policy_target = remaining - historical_target
        primary_on = _take_samples(on_policy_pool, on_policy_target, rng=rng, allow_repeat=False)
        primary_hist = _take_samples(historical_pool, historical_target, rng=rng, allow_repeat=False)
        selected.extend(primary_on)
        selected.extend(primary_hist)
        selected_ids.update(str(sample.sample_id or "") for sample in primary_on + primary_hist)

    if len(selected) < target_samples:
        real_fallback_pool = [
            sample
            for sample in (list(on_policy) + list(historical))
            if str(sample.sample_id or "") not in selected_ids
        ]
        selected.extend(_take_samples(real_fallback_pool, target_samples - len(selected), rng=rng, allow_repeat=False))
        selected_ids.update(str(sample.sample_id or "") for sample in selected)

    if len(selected) < target_samples:
        fallback_pool = list(on_policy) + list(historical) + list(motif)
        selected.extend(_take_samples(fallback_pool, target_samples - len(selected), rng=rng, allow_repeat=True))

    rng.shuffle(selected)
    return selected[:target_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combat_teacher_dataset.v1 from pureSim rollouts")
    parser.add_argument("--hybrid-checkpoint", required=True, help="Hybrid checkpoint for non-combat progression")
    parser.add_argument("--combat-checkpoint", required=True, help="Combat baseline checkpoint")
    parser.add_argument("--historical-combat-checkpoint", action="append", default=[], help="Optional extra combat checkpoints for historical state buckets")
    parser.add_argument("--seed-file", required=True, help="Seed file used to collect rollout states")
    parser.add_argument("--transport", default="pipe-binary", help="PureSim transport")
    parser.add_argument("--port", type=int, default=15527, help="PureSim port")
    parser.add_argument("--num-seeds", type=int, default=12, help="Number of seeds to roll for each source bucket")
    parser.add_argument("--sample-every-combat-step", type=int, default=2, help="Sample every N supported combat states")
    parser.add_argument("--max-episode-steps", type=int, default=500, help="Max steps per rollout episode")
    parser.add_argument("--target-samples", type=int, default=200, help="Final target dataset size after bucket assembly")
    parser.add_argument("--historical-final-fraction", type=float, default=REAL_SOURCE_FRACTIONS["historical"], help="Fraction of non-anchor final dataset budget filled from historical buckets")
    parser.add_argument("--rng-seed", type=int, default=0, help="Deterministic dataset assembly seed")
    parser.add_argument("--motif-sample-file", default=None, help="Optional extra motif samples (combat_teacher_dataset.v1 JSONL)")
    parser.add_argument("--include-regression-motifs", action="store_true", help="Append built-in deterministic motif regression samples")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid_policy = load_noncombat_policy(args.hybrid_checkpoint, vocab=vocab, device=device)
    on_policy_combat = load_baseline_combat_policy(args.combat_checkpoint, vocab=vocab, device=device)
    client = create_full_run_client(
        use_pipe=True,
        transport=args.transport,
        port=args.port,
        ready_timeout_s=30.0,
        request_timeout_s=30.0,
    )
    seeds = _load_seeds(args.seed_file, limit=args.num_seeds)

    on_policy_solver = CombatTurnSolver(client, on_policy_combat)
    on_policy_samples = _collect_samples_for_checkpoint(
        client=client,
        seeds=seeds,
        hybrid_policy=hybrid_policy,
        combat_policy=on_policy_combat,
        solver=on_policy_solver,
        vocab=vocab,
        device=device,
        source_bucket="on_policy",
        source_checkpoint=str(args.combat_checkpoint),
        sample_every_combat_step=args.sample_every_combat_step,
        max_episode_steps=args.max_episode_steps,
    )

    historical_samples: list[CombatTeacherSample] = []
    for checkpoint_path in args.historical_combat_checkpoint:
        historical_combat = load_baseline_combat_policy(checkpoint_path, vocab=vocab, device=device)
        historical_solver = CombatTurnSolver(client, historical_combat)
        historical_samples.extend(
            _collect_samples_for_checkpoint(
                client=client,
                seeds=seeds,
                hybrid_policy=hybrid_policy,
                combat_policy=historical_combat,
                solver=historical_solver,
                vocab=vocab,
                device=device,
                source_bucket="historical",
                source_checkpoint=str(checkpoint_path),
                sample_every_combat_step=args.sample_every_combat_step,
                max_episode_steps=args.max_episode_steps,
            )
        )

    motif_samples = [sample for sample in on_policy_samples + historical_samples if sample.motif_labels]
    if args.motif_sample_file:
        motif_samples.extend(_load_manual_motif_samples(args.motif_sample_file))
    if args.include_regression_motifs:
        motif_samples.extend(build_regression_motif_samples())

    motif_pool_counts = {
        motif_name: sum(1 for sample in motif_samples if _sample_matches_motif(sample, motif_name))
        for motif_name in MOTIF_ORDER
    }
    final_samples = _assemble_dataset(
        on_policy_samples,
        historical_samples,
        motif_samples,
        target_samples=args.target_samples,
        rng_seed=args.rng_seed,
        historical_final_fraction=args.historical_final_fraction,
    )
    metadata = {
        "hybrid_checkpoint": args.hybrid_checkpoint,
        "combat_checkpoint": args.combat_checkpoint,
        "historical_combat_checkpoints": args.historical_combat_checkpoint,
        "seed_file": args.seed_file,
        "num_seeds": args.num_seeds,
        "sample_every_combat_step": args.sample_every_combat_step,
        "max_episode_steps": args.max_episode_steps,
        "historical_final_fraction": args.historical_final_fraction,
        "bucket_counts": {
            "on_policy": len(on_policy_samples),
            "historical": len(historical_samples),
            "motif": len(motif_samples),
            "final": len(final_samples),
        },
        "pure_motif_pool_counts": motif_pool_counts,
        "motif_counts": {
            motif: sum(1 for sample in final_samples if motif in sample.motif_labels)
            for motif in ("missed_lethal", "bash_before_strike", "bodyslam_before_block", "bad_end_turn", "potion_misuse")
        },
        "lethal_submotif_counts": {
            motif: sum(1 for sample in final_samples if motif in sample.motif_labels)
            for motif in ("direct_lethal_first_action", "turn_lethal_no_end_turn")
        },
        "pure_motif_selected_counts": {
            motif: sum(1 for sample in final_samples if _sample_matches_motif(sample, motif))
            for motif in MOTIF_ORDER
        },
        "core_anchor_selected_counts": {
            motif: sum(1 for sample in final_samples if _sample_matches_motif(sample, motif))
            for motif in CORE_ANCHOR_COUNTS
        },
        "holdout_anchor_selected_counts": {
            motif: sum(
                1
                for sample in final_samples
                if _sample_matches_motif(sample, motif) and str(sample.split or "").strip().lower() == "holdout"
            )
            for motif in MIN_HOLDOUT_ANCHOR_COUNTS
        },
        "holdout_anchor_on_policy_selected_counts": {
            motif: sum(
                1
                for sample in final_samples
                if _sample_matches_motif(sample, motif)
                and str(sample.split or "").strip().lower() == "holdout"
                and str(sample.source_bucket or "").strip().lower() == "on_policy"
            )
            for motif in MIN_HOLDOUT_ANCHOR_COUNTS
        },
        "core_anchor_real_selected_counts": {
            motif: sum(
                1
                for sample in final_samples
                if _sample_matches_motif(sample, motif)
                and str(sample.source_bucket or "").strip().lower() in {"on_policy", "historical"}
            )
            for motif in CORE_ANCHOR_COUNTS
        },
        "core_anchor_regression_fallback_counts": {
            motif: sum(
                1
                for sample in final_samples
                if _sample_matches_motif(sample, motif)
                and str(sample.source_bucket or "").strip().lower() == "motif_regression"
            )
            for motif in CORE_ANCHOR_COUNTS
        },
        "holdout_anchor_regression_fallback_counts": {
            motif: sum(
                1
                for sample in final_samples
                if _sample_matches_motif(sample, motif)
                and str(sample.split or "").strip().lower() == "holdout"
                and str(sample.source_bucket or "").strip().lower() == "motif_regression"
            )
            for motif in MIN_HOLDOUT_ANCHOR_COUNTS
        },
        "strict_applicable_unique_counts": {
            motif: _strict_unique_count(final_samples, motif)
            for motif in ("direct_lethal_first_action", "turn_lethal_no_end_turn", "missed_lethal", "bash_before_strike", "bodyslam_before_block", "bad_end_turn", "potion_misuse")
        },
        "strict_applicable_unique_counts_by_split": {
            split: {
                motif: _strict_unique_count(final_samples, motif, split=split)
                for motif in ("direct_lethal_first_action", "turn_lethal_no_end_turn", "missed_lethal", "bash_before_strike", "bodyslam_before_block", "bad_end_turn", "potion_misuse")
            }
            for split in ("train", "holdout")
        },
        "final_source_bucket_counts": {
            bucket: sum(1 for sample in final_samples if str(sample.source_bucket or "").strip().lower() == bucket)
            for bucket in ("on_policy", "historical", "motif_regression", "motif")
        },
    }
    write_combat_teacher_samples(args.output, final_samples, metadata=metadata)
    try:
        client.clear_state_cache()
    except Exception:
        pass
    client.close()


if __name__ == "__main__":
    main()
