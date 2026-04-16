#!/usr/bin/env python3
"""Verify parity between Python root evaluator and C# MCTS evaluator on shared combat states."""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PYTHON_ROOT = _THIS_DIR.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))


import argparse
import json
import math
import os
import time
from typing import Any

import numpy as np
import torch

from network.combat_network import (
    CombatPolicyValueNetwork,
    build_combat_action_features,
    build_combat_features,
    _tensorize_features,
)
from evaluate_ai import (
    _infer_combat_dims,
    _infer_ppo_embed_dim,
    _infer_retrieval_proj_dim,
    _safe_load_state_dict,
)
from env.full_run_env import PipeBackedFullRunClient, create_full_run_client
from env.headless_sim_runner import DEFAULT_DLL_PATH, start_headless_sim, stop_process
from network.fullrun_policy import FullRunPolicyNetworkV2
from verify_save_load import COMBAT_TYPES, drive_to_state
from core.vocab import load_vocab
from core.checkpoint_compat import get_combat_model_config, get_combat_model_state


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits.astype(np.float32)
    shifted = logits - float(np.max(logits))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.zeros_like(logits, dtype=np.float32)
    return (weights / total).astype(np.float32)


def _wait_for_file(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for dump file: {path}")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _load_models(checkpoint_path: Path, device: torch.device):
    vocab = load_vocab()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    ppo_state = ckpt.get("ppo_model") or ckpt.get("model_state_dict")
    ppo_config = ckpt.get("ppo_config", {})
    ppo_embed_dim = _infer_ppo_embed_dim(ppo_state, ppo_config.get("embed_dim", 32))
    retrieval_proj_dim = _infer_retrieval_proj_dim(ppo_state or {})
    use_retrieval = retrieval_proj_dim > 0

    ppo_net = FullRunPolicyNetworkV2(
        vocab=vocab,
        embed_dim=ppo_embed_dim,
        use_symbolic_features=use_retrieval,
        symbolic_proj_dim=retrieval_proj_dim if use_retrieval else 16,
    )
    if isinstance(ppo_state, dict):
        _safe_load_state_dict(ppo_net, ppo_state, "PPO")
    ppo_net.to(device).eval()

    combat_state = get_combat_model_state(ckpt, allow_standalone=False) or {}
    combat_model_config = get_combat_model_config(ckpt)
    combat_embed_dim, combat_hidden_dim = _infer_combat_dims(
        combat_state,
        combat_model_config.get("embed_dim", ppo_embed_dim),
        combat_model_config.get("hidden_dim", 128),
    )
    deck_repr_dim = 0
    for key, value in combat_state.items():
        if key == "deck_encoder.norm.weight" and isinstance(value, torch.Tensor):
            deck_repr_dim = int(value.shape[0])
            break

    combat_net = CombatPolicyValueNetwork(
        vocab=vocab,
        embed_dim=combat_embed_dim,
        hidden_dim=combat_hidden_dim,
        entity_embeddings=ppo_net.entity_emb,
        deck_repr_dim=deck_repr_dim,
        symbolic_head=ppo_net.symbolic_head,
    )
    if isinstance(combat_state, dict):
        _safe_load_state_dict(combat_net, combat_state, "combat")
    combat_net.to(device).eval()
    return vocab, ppo_net, combat_net


def _compute_python_eval(
    *,
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    vocab,
    ppo_net: FullRunPolicyNetworkV2,
    combat_net: CombatPolicyValueNetwork,
    device: torch.device,
    use_continuation_value: bool,
) -> dict[str, Any]:
    sf = build_combat_features(state, vocab)
    af = build_combat_action_features(state, legal_actions, vocab)

    try:
        from network.state_features import build_structured_state

        ss = build_structured_state(state, vocab)
        deck_t = {
            "deck_ids": torch.tensor(ss.deck_ids).unsqueeze(0).to(device),
            "deck_aux": torch.tensor(ss.deck_aux).unsqueeze(0).float().to(device),
            "deck_mask": torch.tensor(ss.deck_mask).unsqueeze(0).bool().to(device),
        }
        with torch.no_grad():
            sf["deck_repr"] = ppo_net.compute_deck_repr(deck_t).squeeze(0).detach().cpu().numpy()
    except Exception:
        pass

    state_t = _tensorize_features(sf, device)
    action_t = _tensorize_features(af, device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if use_continuation_value:
                    logits_t, _value_t, _scores_t, continuation_t = combat_net.forward_teacher(state_t, action_t)
                    value_scalar = float(continuation_t[0, 0].detach().cpu().float().item() * 2.0 - 1.0)
                else:
                    logits_t, value_t = combat_net.forward(state_t, action_t)
                    value_scalar = float(value_t[0].detach().cpu().float().item())
        else:
            if use_continuation_value:
                logits_t, _value_t, _scores_t, continuation_t = combat_net.forward_teacher(state_t, action_t)
                value_scalar = float(continuation_t[0, 0].detach().cpu().float().item() * 2.0 - 1.0)
            else:
                logits_t, value_t = combat_net.forward(state_t, action_t)
                value_scalar = float(value_t[0].detach().cpu().float().item())

    n = len(legal_actions)
    raw_logits = logits_t[0, :n].detach().cpu().float().numpy()
    policy = _softmax(raw_logits)

    return {
        "policy_logits": raw_logits.astype(np.float32),
        "policy_probs": policy,
        "value": value_scalar,
        "encoded": {
            "scalars": sf.get("scalars"),
            "extra_scalars": sf.get("extra_scalars"),
            "hand_ids": sf.get("hand_ids"),
            "hand_aux": sf.get("hand_aux"),
            "hand_mask": sf.get("hand_mask"),
            "enemy_ids": sf.get("enemy_ids"),
            "enemy_aux": sf.get("enemy_aux"),
            "enemy_mask": sf.get("enemy_mask"),
            "deck_ids": sf.get("deck_ids"),
            "deck_aux": sf.get("deck_aux"),
            "deck_mask": sf.get("deck_mask"),
            "draw_pile_ids": sf.get("draw_pile_ids"),
            "draw_pile_aux": sf.get("draw_pile_aux"),
            "draw_pile_mask": sf.get("draw_pile_mask"),
            "discard_pile_ids": sf.get("discard_pile_ids"),
            "discard_pile_aux": sf.get("discard_pile_aux"),
            "discard_pile_mask": sf.get("discard_pile_mask"),
            "exhaust_pile_ids": sf.get("exhaust_pile_ids"),
            "exhaust_pile_aux": sf.get("exhaust_pile_aux"),
            "exhaust_pile_mask": sf.get("exhaust_pile_mask"),
            "action_type_ids": af.get("action_type_ids"),
            "target_card_ids": af.get("target_card_ids"),
            "target_enemy_ids": af.get("target_enemy_ids"),
            "action_mask": af.get("action_mask"),
        },
    }


def _array_compare(py_value: Any, csharp_value: Any) -> dict[str, Any]:
    if py_value is None and csharp_value is None:
        return {"present": False}
    if py_value is None or csharp_value is None:
        return {
            "present": True,
            "one_side_missing": True,
            "python_present": py_value is not None,
            "csharp_present": csharp_value is not None,
        }
    py_arr = np.asarray(py_value, dtype=np.float32)
    cs_arr = np.asarray(csharp_value, dtype=np.float32)
    shape_match = py_arr.shape == cs_arr.shape
    if not shape_match:
        return {
            "present": True,
            "shape_match": False,
            "python_shape": list(py_arr.shape),
            "csharp_shape": list(cs_arr.shape),
        }
    diff = np.abs(py_arr - cs_arr)
    return {
        "present": True,
        "shape_match": True,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "python_shape": list(py_arr.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Python combat evaluator vs C# ORT evaluator on the same root state.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ort-model-path", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--port", type=int, default=15557)
    parser.add_argument("--headless-dll", default=str(DEFAULT_DLL_PATH))
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--use-continuation-value", action="store_true", default=False)
    parser.add_argument("--ort-device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    onnx_path = Path(args.ort_model_path).resolve()
    output_path = Path(args.output).resolve()
    dump_path = output_path.with_suffix(".csharp_dump.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dump_path.exists():
        dump_path.unlink()

    old_dump_env = os.environ.get("STS2AI_DEBUG_COMBAT_FEATURES_PATH")
    old_ort_device = os.environ.get("STS2AI_ORT_DEVICE")
    os.environ["STS2AI_DEBUG_COMBAT_FEATURES_PATH"] = str(dump_path)
    os.environ["STS2AI_ORT_DEVICE"] = args.ort_device

    proc = None
    client = None
    try:
        proc = start_headless_sim(
            port=int(args.port),
            repo_root=Path(args.repo_root).resolve(),
            dll_path=Path(args.headless_dll).resolve(),
            connect_timeout_s=15.0,
            protocol="bin",
        )
        client = create_full_run_client(
            port=int(args.port),
            use_pipe=True,
            transport="pipe-binary",
            ready_timeout_s=15.0,
        )
        assert isinstance(client, PipeBackedFullRunClient)
        client._ensure_connected()
        load_info = client.load_ort_model(str(onnx_path))

        root_state = drive_to_state(client, args.seed, COMBAT_TYPES)
        legal_actions = list(root_state.get("legal_actions") or [])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab, ppo_net, combat_net = _load_models(checkpoint, device)
        python_eval = _compute_python_eval(
            state=root_state,
            legal_actions=legal_actions,
            vocab=vocab,
            ppo_net=ppo_net,
            combat_net=combat_net,
            device=device,
            use_continuation_value=bool(args.use_continuation_value),
        )

        search_result = client.search_combat_mcts(
            num_simulations=1,
            c_puct=1.5,
            dirichlet_alpha=0.0,
            dirichlet_fraction=0.0,
            max_step_budget=200,
            final_action_mode="visit",
            final_action_top_k=3,
            final_action_q_weight=0.35,
            use_continuation_value=bool(args.use_continuation_value),
            debug_trace=False,
        )

        _wait_for_file(dump_path, timeout_s=10.0)
        csharp_dump = json.loads(dump_path.read_text(encoding="utf-8"))
        csharp_logits = np.asarray(csharp_dump["output"]["policy_logits"][: len(legal_actions)], dtype=np.float32)
        csharp_policy = _softmax(csharp_logits)
        csharp_value = float(csharp_dump["output"]["value"])

        action_rows = []
        py_logits = np.asarray(python_eval["policy_logits"], dtype=np.float32)
        py_policy = np.asarray(python_eval["policy_probs"], dtype=np.float32)
        priors_from_search = list(search_result.get("priors") or [])
        for idx, action in enumerate(legal_actions):
            action_rows.append(
                {
                    "index": idx,
                    "label": action.get("label"),
                    "action": action.get("action"),
                    "card_index": action.get("card_index"),
                    "target_id": action.get("target_id"),
                    "python_logit": float(py_logits[idx]),
                    "csharp_logit": float(csharp_logits[idx]),
                    "logit_abs_diff": float(abs(py_logits[idx] - csharp_logits[idx])),
                    "python_policy": float(py_policy[idx]),
                    "csharp_policy": float(csharp_policy[idx]),
                    "search_prior": float(priors_from_search[idx]) if idx < len(priors_from_search) else None,
                    "policy_abs_diff": float(abs(py_policy[idx] - csharp_policy[idx])),
                }
            )

        payload = {
            "checkpoint": str(checkpoint),
            "ort_model_path": str(onnx_path),
            "seed": args.seed,
            "port": int(args.port),
            "load_ort_model": load_info,
            "root_state_type": root_state.get("state_type"),
            "root_floor": ((root_state.get("run") or {}).get("floor")),
            "legal_action_count": len(legal_actions),
            "legal_actions": [
                {
                    "index": idx,
                    "label": action.get("label"),
                    "action": action.get("action"),
                    "card_index": action.get("card_index"),
                    "target_id": action.get("target_id"),
                    "slot": action.get("slot"),
                }
                for idx, action in enumerate(legal_actions)
            ],
            "python": {
                "device": str(device),
                "value": float(python_eval["value"]),
                "policy_logits": py_logits.tolist(),
                "policy_probs": py_policy.tolist(),
            },
            "csharp": {
                "value": csharp_value,
                "policy_logits": csharp_logits.tolist(),
                "policy_probs": csharp_policy.tolist(),
                "search_priors": priors_from_search,
            },
            "diff": {
                "value_abs_diff": float(abs(float(python_eval["value"]) - csharp_value)),
                "max_abs_logit_diff": float(np.max(np.abs(py_logits - csharp_logits))) if len(legal_actions) else 0.0,
                "mean_abs_logit_diff": float(np.mean(np.abs(py_logits - csharp_logits))) if len(legal_actions) else 0.0,
                "max_abs_policy_diff": float(np.max(np.abs(py_policy - csharp_policy))) if len(legal_actions) else 0.0,
                "mean_abs_policy_diff": float(np.mean(np.abs(py_policy - csharp_policy))) if len(legal_actions) else 0.0,
                "feature_compare": {
                    "scalars": _array_compare(python_eval["encoded"].get("scalars"), csharp_dump["encoded"].get("scalars")),
                    "extra_scalars": _array_compare(python_eval["encoded"].get("extra_scalars"), csharp_dump["encoded"].get("extra_scalars")),
                    "hand_ids": _array_compare(python_eval["encoded"].get("hand_ids"), csharp_dump["encoded"].get("hand_ids")),
                    "enemy_ids": _array_compare(python_eval["encoded"].get("enemy_ids"), csharp_dump["encoded"].get("enemy_ids")),
                    "action_type_ids": _array_compare(python_eval["encoded"].get("action_type_ids"), csharp_dump["encoded"].get("action_type_ids")),
                    "target_card_ids": _array_compare(python_eval["encoded"].get("target_card_ids"), csharp_dump["encoded"].get("target_card_ids")),
                    "target_enemy_ids": _array_compare(python_eval["encoded"].get("target_enemy_ids"), csharp_dump["encoded"].get("target_enemy_ids")),
                    "action_mask": _array_compare(python_eval["encoded"].get("action_mask"), csharp_dump["encoded"].get("action_mask")),
                },
            },
            "per_action": action_rows,
            "csharp_dump_path": str(dump_path),
        }

        output_path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote evaluator parity report to {output_path}")
        return 0
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        stop_process(proc)
        if old_dump_env is None:
            os.environ.pop("STS2AI_DEBUG_COMBAT_FEATURES_PATH", None)
        else:
            os.environ["STS2AI_DEBUG_COMBAT_FEATURES_PATH"] = old_dump_env
        if old_ort_device is None:
            os.environ.pop("STS2AI_ORT_DEVICE", None)
        else:
            os.environ["STS2AI_ORT_DEVICE"] = old_ort_device


if __name__ == "__main__":
    raise SystemExit(main())
