from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from combat_nn import CombatPolicyValueNetwork
from evaluate_ai import (
    COMBAT_SCREENS,
    _build_combat_tensors,
    _build_ppo_tensors,
    _infer_combat_dims,
    _infer_ppo_embed_dim,
    _safe_load_state_dict,
)
from rl_policy_v2 import FullRunPolicyNetworkV2
from sim_semantic_audit_common import (
    DEFAULT_GODOT_EXE,
    DEFAULT_HEADLESS_DLL,
    DEFAULT_PORT,
    DEFAULT_REPO_ROOT,
    backend_client,
    build_seed_list,
)
from test_simulator_consistency import normalize_legal_action, state_summary
from vocab import Vocab, load_vocab


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary_hash(summary: dict[str, Any]) -> str:
    payload = json.dumps(_json_ready(summary), sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _normalize_action_key(action: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if not isinstance(action, dict):
        return None
    return normalize_legal_action(action)


def _topk_actions(
    legal: list[dict[str, Any]],
    logits: np.ndarray,
    *,
    k: int = 3,
) -> list[dict[str, Any]]:
    if len(legal) == 0 or logits.size == 0:
        return []
    order = np.argsort(-logits)[: max(1, min(k, len(legal)))]
    clipped = logits[: len(legal)]
    shifted = clipped - np.max(clipped)
    probs = np.exp(shifted)
    probs = probs / max(1e-9, probs.sum())
    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        action = legal[int(idx)]
        results.append(
            {
                "rank": rank,
                "index": int(idx),
                "action": _json_ready(action),
                "prob": round(float(probs[int(idx)]), 6),
                "logit": round(float(clipped[int(idx)]), 6),
            }
        )
    return results


def _tensor_diff_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }

    if left.dtype == torch.bool or right.dtype == torch.bool:
        mismatch = int((left.bool() != right.bool()).sum().item())
        return {
            "shape_match": True,
            "dtype": "bool",
            "mismatch_count": mismatch,
            "allclose": mismatch == 0,
        }

    if left.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) or right.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        mismatch = int((left.long() != right.long()).sum().item())
        return {
            "shape_match": True,
            "dtype": "int",
            "mismatch_count": mismatch,
            "allclose": mismatch == 0,
        }

    diff = (left.float() - right.float()).abs()
    return {
        "shape_match": True,
        "dtype": "float",
        "max_abs_diff": float(diff.max().item()) if diff.numel() > 0 else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() > 0 else 0.0,
        "allclose": bool(torch.allclose(left.float(), right.float(), atol=1e-6, rtol=1e-5)),
    }


def _compare_tensor_dicts(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> dict[str, Any]:
    keys = sorted(set(left.keys()) | set(right.keys()))
    per_key: dict[str, Any] = {}
    mismatch_keys: list[str] = []
    max_float_diff = 0.0
    for key in keys:
        if key not in left or key not in right:
            per_key[key] = {
                "present_in_left": key in left,
                "present_in_right": key in right,
            }
            mismatch_keys.append(key)
            continue
        summary = _tensor_diff_summary(left[key].detach().cpu(), right[key].detach().cpu())
        per_key[key] = summary
        if not summary.get("allclose", False):
            mismatch_keys.append(key)
        max_float_diff = max(max_float_diff, _safe_float(summary.get("max_abs_diff"), 0.0))
    return {
        "mismatch_count": len(mismatch_keys),
        "mismatch_keys": mismatch_keys,
        "max_float_diff": max_float_diff,
        "per_key": per_key,
    }


def _find_matching_action_index(
    action: dict[str, Any] | None,
    legal: list[dict[str, Any]],
) -> int | None:
    target_key = _normalize_action_key(action)
    if target_key is None:
        return None
    for idx, candidate in enumerate(legal):
        if _normalize_action_key(candidate) == target_key:
            return idx
    return None


def _load_models(
    *,
    checkpoint: str,
    combat_checkpoint: str | None,
    vocab: Vocab,
    device: torch.device,
) -> tuple[FullRunPolicyNetworkV2, CombatPolicyValueNetwork]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ppo_state = ckpt.get("ppo_model") or ckpt.get("model_state_dict")
    if not isinstance(ppo_state, dict):
        raise ValueError(f"Checkpoint has no PPO weights: {checkpoint}")
    ppo_embed_dim = _infer_ppo_embed_dim(ppo_state, fallback=48)
    ppo_net = FullRunPolicyNetworkV2(vocab=vocab, embed_dim=ppo_embed_dim)
    _safe_load_state_dict(ppo_net, ppo_state, "PPO")

    combat_state = ckpt.get("mcts_model") or ckpt.get("combat_model") or ckpt.get("model_state_dict")
    if combat_checkpoint:
        override_ckpt = torch.load(combat_checkpoint, map_location="cpu", weights_only=False)
        combat_state = override_ckpt.get("mcts_model") or override_ckpt.get("model_state_dict")
    if not isinstance(combat_state, dict):
        raise ValueError(f"Checkpoint has no combat weights: {combat_checkpoint or checkpoint}")
    combat_embed_dim, combat_hidden_dim = _infer_combat_dims(
        combat_state,
        fallback_embed_dim=48,
        fallback_hidden_dim=192,
    )
    combat_net = CombatPolicyValueNetwork(
        vocab=vocab,
        embed_dim=combat_embed_dim,
        hidden_dim=combat_hidden_dim,
    )
    _safe_load_state_dict(combat_net, combat_state, "combat")

    ppo_net.to(device).eval()
    combat_net.to(device).eval()
    return ppo_net, combat_net


def _decision_snapshot(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    ppo_net: FullRunPolicyNetworkV2,
    combat_net: CombatPolicyValueNetwork,
    vocab: Vocab,
    device: torch.device,
) -> dict[str, Any]:
    state_type = _lower(state.get("state_type"))
    if state_type in COMBAT_SCREENS:
        state_t, action_t = _build_combat_tensors(state, legal, vocab, device)
        with torch.no_grad():
            logits_t, value_t = combat_net(state_t, action_t)
        masked = logits_t.squeeze(0) + (1.0 - action_t["action_mask"].float().squeeze(0)) * (-1e9)
        logits = masked[: len(legal)].detach().cpu().numpy()
        value = float(value_t.squeeze().item())
        policy_family = "combat"
    else:
        state_t, action_t = _build_ppo_tensors(state, legal, vocab, device)
        with torch.no_grad():
            logits_t, values_t, _dq, _boss_ready, _adv = ppo_net(state_t, action_t)
        logits = logits_t.squeeze(0)[: len(legal)].detach().cpu().numpy()
        value = float(values_t.squeeze().item())
        policy_family = "ppo"

    top_idx = int(np.argmax(logits)) if len(legal) > 0 else 0
    top_action = legal[top_idx] if top_idx < len(legal) else (legal[0] if legal else {})
    return {
        "policy_family": policy_family,
        "state_tensors": state_t,
        "action_tensors": action_t,
        "logits": logits,
        "value": value,
        "top1_index": top_idx,
        "top1_action": _json_ready(top_action),
        "top1_key": _json_ready(_normalize_action_key(top_action)),
        "top3": _topk_actions(legal, logits, k=3),
    }


def _compare_logits(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    if left.size == 0:
        return {
            "shape_match": True,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "argmax_match": True,
        }
    diff = np.abs(left - right)
    return {
        "shape_match": True,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "argmax_match": bool(int(np.argmax(left)) == int(np.argmax(right))),
    }


def _compare_step(
    *,
    step: int,
    baseline_state: dict[str, Any],
    candidate_state: dict[str, Any],
    baseline_legal: list[dict[str, Any]],
    candidate_legal: list[dict[str, Any]],
    baseline_decision: dict[str, Any],
    candidate_decision: dict[str, Any],
) -> dict[str, Any]:
    baseline_summary = state_summary(baseline_state)
    candidate_summary = state_summary(candidate_state)
    state_equal = baseline_summary == candidate_summary
    legal_equal = [
        normalize_legal_action(action) for action in baseline_legal
    ] == [
        normalize_legal_action(action) for action in candidate_legal
    ]
    state_tensor_diff = _compare_tensor_dicts(
        baseline_decision["state_tensors"],
        candidate_decision["state_tensors"],
    )
    action_tensor_diff = _compare_tensor_dicts(
        baseline_decision["action_tensors"],
        candidate_decision["action_tensors"],
    )
    logits_diff = _compare_logits(
        np.asarray(baseline_decision["logits"], dtype=np.float64),
        np.asarray(candidate_decision["logits"], dtype=np.float64),
    )
    value_diff = abs(
        _safe_float(baseline_decision.get("value"))
        - _safe_float(candidate_decision.get("value"))
    )
    top1_match = baseline_decision.get("top1_key") == candidate_decision.get("top1_key")
    mismatch_reasons: list[str] = []
    if not state_equal:
        mismatch_reasons.append("state_summary")
    if not legal_equal:
        mismatch_reasons.append("legal_actions")
    if state_tensor_diff["mismatch_count"] > 0:
        mismatch_reasons.append("state_tensors")
    if action_tensor_diff["mismatch_count"] > 0:
        mismatch_reasons.append("action_tensors")
    if not logits_diff.get("argmax_match", True):
        mismatch_reasons.append("top1_action")
    if _safe_float(logits_diff.get("max_abs_diff")) > 1e-5:
        mismatch_reasons.append("logits")
    if value_diff > 1e-5:
        mismatch_reasons.append("value")
    result = {
        "step": step,
        "state_type": _lower(baseline_state.get("state_type")),
        "floor": _safe_int((baseline_state.get("run") or {}).get("floor")),
        "state_summary_equal": state_equal,
        "legal_actions_equal": legal_equal,
        "top1_match": top1_match,
        "value_diff": value_diff,
        "logits_diff": logits_diff,
        "state_tensor_diff": state_tensor_diff,
        "action_tensor_diff": action_tensor_diff,
        "baseline_top1": baseline_decision["top1_action"],
        "candidate_top1": candidate_decision["top1_action"],
        "baseline_top3": baseline_decision["top3"],
        "candidate_top3": candidate_decision["top3"],
        "mismatch_reasons": mismatch_reasons,
    }
    result["baseline_state_hash"] = _summary_hash(baseline_summary)
    result["candidate_state_hash"] = _summary_hash(candidate_summary)
    if not state_equal:
        result["baseline_state_summary"] = _json_ready(baseline_summary)
        result["candidate_state_summary"] = _json_ready(candidate_summary)
    return result


def run_parity_audit(
    *,
    checkpoint: str,
    combat_checkpoint: str | None,
    baseline_backend: str,
    candidate_backend: str,
    baseline_port: int,
    candidate_port: int,
    auto_launch: bool,
    repo_root: Path,
    godot_exe: Path,
    headless_dll: Path,
    seeds: list[str],
    max_steps: int,
    stop_on_first_mismatch: bool,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab()
    ppo_net, combat_net = _load_models(
        checkpoint=checkpoint,
        combat_checkpoint=combat_checkpoint,
        vocab=vocab,
        device=device,
    )

    results: list[dict[str, Any]] = []
    mismatch_counter: Counter[str] = Counter()

    with backend_client(
        backend=baseline_backend,
        port=baseline_port,
        auto_launch=auto_launch,
        repo_root=repo_root,
        godot_exe=godot_exe,
        headless_dll=headless_dll,
    ) as baseline_client, backend_client(
        backend=candidate_backend,
        port=candidate_port,
        auto_launch=auto_launch,
        repo_root=repo_root,
        godot_exe=godot_exe,
        headless_dll=headless_dll,
    ) as candidate_client:
        for seed in seeds:
            baseline_state = baseline_client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
            candidate_state = candidate_client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
            steps: list[dict[str, Any]] = []
            seed_status = "pass"

            for step in range(max_steps):
                baseline_legal = baseline_state.get("legal_actions") or []
                candidate_legal = candidate_state.get("legal_actions") or []
                if bool(baseline_state.get("terminal")) or _lower(baseline_state.get("state_type")) == "game_over":
                    break
                if bool(candidate_state.get("terminal")) or _lower(candidate_state.get("state_type")) == "game_over":
                    break
                if not baseline_legal or not candidate_legal:
                    if not baseline_legal and not candidate_legal:
                        baseline_state = baseline_client.act({"action": "wait"})
                        candidate_state = candidate_client.act({"action": "wait"})
                        steps.append(
                            {
                                "step": step,
                                "state_type": _lower(baseline_state.get("state_type")),
                                "floor": _safe_int((baseline_state.get("run") or {}).get("floor")),
                                "mismatch_reasons": [],
                                "diagnostic_note": "both_backends_waited_for_state_progress",
                            }
                        )
                        continue
                    steps.append(
                        {
                            "step": step,
                            "state_type": _lower(baseline_state.get("state_type")),
                            "floor": _safe_int((baseline_state.get("run") or {}).get("floor")),
                            "mismatch_reasons": ["missing_legal_actions"],
                            "baseline_legal_count": len(baseline_legal),
                            "candidate_legal_count": len(candidate_legal),
                        }
                    )
                    seed_status = "mismatch"
                    mismatch_counter["missing_legal_actions"] += 1
                    break

                baseline_decision = _decision_snapshot(
                    state=baseline_state,
                    legal=baseline_legal,
                    ppo_net=ppo_net,
                    combat_net=combat_net,
                    vocab=vocab,
                    device=device,
                )
                candidate_decision = _decision_snapshot(
                    state=candidate_state,
                    legal=candidate_legal,
                    ppo_net=ppo_net,
                    combat_net=combat_net,
                    vocab=vocab,
                    device=device,
                )

                comparison = _compare_step(
                    step=step,
                    baseline_state=baseline_state,
                    candidate_state=candidate_state,
                    baseline_legal=baseline_legal,
                    candidate_legal=candidate_legal,
                    baseline_decision=baseline_decision,
                    candidate_decision=candidate_decision,
                )
                steps.append(comparison)

                effective_reasons = [
                    reason for reason in comparison["mismatch_reasons"]
                    if reason != "state_summary"
                ]
                if effective_reasons:
                    seed_status = "mismatch"
                    comparison["effective_mismatch_reasons"] = effective_reasons
                    mismatch_counter.update(effective_reasons)
                    if stop_on_first_mismatch:
                        break
                elif comparison["mismatch_reasons"]:
                    comparison["diagnostic_only"] = True

                driver_action = baseline_legal[baseline_decision["top1_index"]]
                matched_idx = _find_matching_action_index(driver_action, candidate_legal)
                if matched_idx is None:
                    steps.append(
                        {
                            "step": step,
                            "state_type": _lower(baseline_state.get("state_type")),
                            "floor": _safe_int((baseline_state.get("run") or {}).get("floor")),
                            "mismatch_reasons": ["driver_action_not_found"],
                            "driver_action": _json_ready(driver_action),
                        }
                    )
                    seed_status = "mismatch"
                    mismatch_counter["driver_action_not_found"] += 1
                    break

                baseline_state = baseline_client.act(driver_action)
                candidate_state = candidate_client.act(candidate_legal[matched_idx])

            results.append(
                {
                    "seed": seed,
                    "status": seed_status,
                    "steps": steps,
                    "baseline_terminal": _json_ready(
                        {
                            "state_type": _lower(baseline_state.get("state_type")),
                            "terminal": bool(baseline_state.get("terminal", False)),
                            "run_outcome": baseline_state.get("run_outcome"),
                            "floor": _safe_int((baseline_state.get("run") or {}).get("floor")),
                        }
                    ),
                    "candidate_terminal": _json_ready(
                        {
                            "state_type": _lower(candidate_state.get("state_type")),
                            "terminal": bool(candidate_state.get("terminal", False)),
                            "run_outcome": candidate_state.get("run_outcome"),
                            "floor": _safe_int((candidate_state.get("run") or {}).get("floor")),
                        }
                    ),
                }
            )

    summary = {
        "seed_count": len(seeds),
        "mismatch_seed_count": sum(1 for result in results if result["status"] != "pass"),
        "mismatch_reasons": dict(sorted(mismatch_counter.items())),
    }
    return {
        "checkpoint": checkpoint,
        "combat_checkpoint": combat_checkpoint,
        "baseline_backend": baseline_backend,
        "candidate_backend": candidate_backend,
        "baseline_port": baseline_port,
        "candidate_port": candidate_port,
        "max_steps": max_steps,
        "summary": summary,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NN parity audit across Godot/headless full-run backends.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--combat-checkpoint", default=None)
    parser.add_argument("--baseline-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="godot-http")
    parser.add_argument("--candidate-backend", choices=["godot-http", "headless-pipe", "headless-binary"], default="headless-pipe")
    parser.add_argument("--baseline-port", type=int, default=DEFAULT_PORT + 140)
    parser.add_argument("--candidate-port", type=int, default=DEFAULT_PORT + 141)
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--godot-exe", type=Path, default=DEFAULT_GODOT_EXE)
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_HEADLESS_DLL)
    parser.add_argument("--seed-prefix", default="CONSIST")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--seed", dest="explicit_seeds", action="append", default=None)
    parser.add_argument("--include-default-seeds", action="store_true")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--stop-on-first-mismatch", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.explicit_seeds and not args.include_default_seeds and args.count <= 0:
        args.count = 5
    seeds = build_seed_list(
        explicit_seeds=args.explicit_seeds,
        seed_prefix=args.seed_prefix,
        start_index=args.start_index,
        count=args.count,
        include_default=args.include_default_seeds,
    )
    report = run_parity_audit(
        checkpoint=args.checkpoint,
        combat_checkpoint=args.combat_checkpoint,
        baseline_backend=args.baseline_backend,
        candidate_backend=args.candidate_backend,
        baseline_port=args.baseline_port,
        candidate_port=args.candidate_port,
        auto_launch=args.auto_launch,
        repo_root=args.repo_root,
        godot_exe=args.godot_exe,
        headless_dll=args.headless_dll,
        seeds=seeds,
        max_steps=args.max_steps,
        stop_on_first_mismatch=args.stop_on_first_mismatch,
    )
    payload = json.dumps(_json_ready(report), indent=2, ensure_ascii=False)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if report["summary"]["mismatch_seed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
