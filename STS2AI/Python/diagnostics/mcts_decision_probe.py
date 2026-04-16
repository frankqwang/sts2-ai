#!/usr/bin/env python3
"""Interactive MCTS decision probe — run a single combat state through MCTS and dump tree statistics."""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PYTHON_ROOT = _THIS_DIR.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))


import argparse
import atexit
import json
from typing import Any

import numpy as np
import torch

from search.combat_mcts_agent import CombatMCTSAgent, PipeCombatForwardModel, _reconcile_action
from network.combat_network import CombatNNEvaluator, CombatPolicyValueNetwork
from evaluate_ai import (
    _build_combat_tensors,
    _infer_combat_dims,
    _infer_ppo_embed_dim,
    _infer_retrieval_proj_dim,
    _safe_load_state_dict,
    _summarize_mcts_root,
)
from env.headless_sim_runner import DEFAULT_DLL_PATH, start_headless_sim, stop_process
from env.full_run_env import PipeBackedFullRunClient, create_full_run_client
from search.mcts_core import MCTSConfig
from network.fullrun_policy import FullRunPolicyNetworkV2
from verify_save_load import COMBAT_TYPES, drive_to_state
from core.vocab import load_vocab
from core.checkpoint_compat import get_combat_model_config, get_combat_model_state


def _load_models(checkpoint_path: Path) -> tuple[Any, FullRunPolicyNetworkV2, CombatPolicyValueNetwork]:
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
    ppo_net.eval()

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
    combat_net.eval()
    return vocab, ppo_net, combat_net


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") if isinstance(battle.get("player"), dict) else (state.get("player") or {})
    legal = []
    for entry in state.get("legal_actions") or []:
        if isinstance(entry, dict):
            legal.append(
                {
                    "action": entry.get("action"),
                    "label": entry.get("label"),
                    "target_id": entry.get("target_id"),
                    "card_index": entry.get("card_index"),
                }
            )
    enemies = []
    for enemy in state.get("enemies") or battle.get("enemies") or []:
        if not isinstance(enemy, dict):
            continue
        enemies.append(
            {
                "id": enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id"),
                "hp": enemy.get("hp"),
                "block": enemy.get("block"),
                "intent": [
                    intent.get("type")
                    for intent in enemy.get("intents") or []
                    if isinstance(intent, dict)
                ],
            }
        )
    return {
        "state_type": state.get("state_type"),
        "floor": run.get("floor"),
        "round": state.get("round_number") or battle.get("round_number"),
        "hp": player.get("hp"),
        "block": player.get("block"),
        "energy": battle.get("energy") or player.get("energy"),
        "legal_action_count": len(state.get("legal_actions") or []),
        "legal_actions": legal[:12],
        "enemies": enemies,
    }


def _top_plain_actions(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    vocab: Any,
    device: torch.device,
    ppo_net: FullRunPolicyNetworkV2,
    combat_net: CombatPolicyValueNetwork,
    top_k: int,
) -> dict[str, Any]:
    sf_t, af_t = _build_combat_tensors(state, legal, vocab, device, ppo_net=ppo_net)
    with torch.no_grad():
        logits, value = combat_net(sf_t, af_t)
    logits_np = logits.squeeze(0)[: len(legal)].detach().cpu().float().numpy()
    probs = np.exp(logits_np - np.max(logits_np))
    probs = probs / max(1e-8, float(np.sum(probs)))
    order = np.argsort(-logits_np)[: max(1, int(top_k))]
    top = []
    for idx in order:
        action = legal[int(idx)]
        top.append(
            {
                "index": int(idx),
                "logit": round(float(logits_np[int(idx)]), 4),
                "prob": round(float(probs[int(idx)]), 4),
                "action": {
                    "action": action.get("action"),
                    "label": action.get("label"),
                    "target_id": action.get("target_id"),
                    "card_index": action.get("card_index"),
                },
            }
        )
    return {
        "value": round(float(value.squeeze(0).detach().cpu().item()), 4),
        "top_actions": top,
    }


def _replay_to_probe_state(
    client: PipeBackedFullRunClient,
    *,
    seed: str,
    trace_path: Path,
    combat_step_index: int,
) -> dict[str, Any]:
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_steps = trace_payload.get("trace") or []
    combat_actions = [
        step.get("chosen_action")
        for step in trace_steps
        if step.get("state_type") in COMBAT_TYPES
    ]
    if combat_step_index < 0 or combat_step_index > len(combat_actions):
        raise ValueError(f"combat_step_index {combat_step_index} out of range for {len(combat_actions)} combat actions")

    state = drive_to_state(client, seed, COMBAT_TYPES)
    for action in combat_actions[:combat_step_index]:
        legal = state.get("legal_actions") or []
        state = client.act(_reconcile_action(action, legal))
    return state


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    trace_path = Path(args.trace_json).resolve()
    vocab, ppo_net, combat_net = _load_models(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_net.to(device).eval()
    combat_net.to(device).eval()

    proc = start_headless_sim(
        port=int(args.port),
        repo_root=Path(args.repo_root).resolve(),
        dll_path=Path(args.dll_path).resolve(),
        protocol="binary",
    )
    atexit.register(stop_process, proc)
    try:
        client = create_full_run_client(
            port=int(args.port),
            use_pipe=True,
            transport="pipe-binary",
            ready_timeout_s=15.0,
        )
        assert isinstance(client, PipeBackedFullRunClient)
        client._ensure_connected()
        if args.ort_model_path:
            client.load_ort_model(str(Path(args.ort_model_path).resolve()))

        state = _replay_to_probe_state(
            client,
            seed=args.seed,
            trace_path=trace_path,
            combat_step_index=int(args.combat_step_index),
        )
        legal = state.get("legal_actions") or []
        evaluator = CombatNNEvaluator(
            combat_net,
            vocab,
            device=device,
            ppo_net=ppo_net,
            use_continuation_value=bool(getattr(args, "combat_mcts_continuation_value", False)),
        )
        eval_policy, eval_value = evaluator.evaluate(state, legal)
        batch_result = evaluator.evaluate_batch([state], [legal])[0]
        batch_policy, batch_value = batch_result

        plain = _top_plain_actions(
            state=state,
            legal=legal,
            vocab=vocab,
            device=device,
            ppo_net=ppo_net,
            combat_net=combat_net,
            top_k=int(args.top_k),
        )

        mcts_roots: list[dict[str, Any]] = []
        for sims in [int(part.strip()) for part in str(args.sims).split(",") if part.strip()]:
            agent = CombatMCTSAgent(
                network=combat_net,
                vocab=vocab,
                config=MCTSConfig(
                    num_simulations=max(1, sims),
                    c_puct=float(args.c_puct),
                    temperature=0.0,
                    dirichlet_alpha=0.0,
                    dirichlet_fraction=0.0,
                    num_determinizations=1,
                ),
                training=False,
                device=device,
                ppo_net=ppo_net,
                backend=str(getattr(args, "combat_mcts_backend", "python")),
                use_continuation_value=bool(getattr(args, "combat_mcts_continuation_value", False)),
            )
            fm = PipeCombatForwardModel.from_current_state(
                client._pipe,
                max_step_budget=max(1, int(args.step_budget)),
            )
            try:
                chosen_action, root = agent.choose_action(fm)
                top_actions, root_value = _summarize_mcts_root(root, k=int(args.top_k))
            finally:
                restored = fm.cleanup_and_restore()
                if restored is None:
                    fm.cleanup()
            mcts_roots.append(
                {
                    "sims": sims,
                    "chosen_action": chosen_action,
                    "root_value": root_value,
                    "top_actions": top_actions,
                }
            )

        top_eval = np.argsort(-eval_policy)[: max(1, int(args.top_k))]
        top_batch = np.argsort(-batch_policy)[: max(1, int(args.top_k))]
        payload = {
            "seed": args.seed,
            "combat_step_index": int(args.combat_step_index),
            "state_summary": _state_summary(state),
            "plain_policy": plain,
            "evaluator": {
                "value": round(float(eval_value), 4),
                "top_actions": [
                    {
                        "index": int(idx),
                        "prob": round(float(eval_policy[int(idx)]), 4),
                        "action": {
                            "action": legal[int(idx)].get("action"),
                            "label": legal[int(idx)].get("label"),
                            "target_id": legal[int(idx)].get("target_id"),
                            "card_index": legal[int(idx)].get("card_index"),
                        },
                    }
                    for idx in top_eval
                ],
            },
            "evaluator_batch": {
                "value": round(float(batch_value), 4),
                "top_actions": [
                    {
                        "index": int(idx),
                        "prob": round(float(batch_policy[int(idx)]), 4),
                        "action": {
                            "action": legal[int(idx)].get("action"),
                            "label": legal[int(idx)].get("label"),
                            "target_id": legal[int(idx)].get("target_id"),
                            "card_index": legal[int(idx)].get("card_index"),
                        },
                    }
                    for idx in top_batch
                ],
            },
            "mcts": mcts_roots,
        }
        return payload
    finally:
        stop_process(proc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--combat-step-index", type=int, default=0)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dll-path", default=str(DEFAULT_DLL_PATH))
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--step-budget", type=int, default=200)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--sims", default="1,50")
    parser.add_argument("--combat-mcts-backend", choices=["python", "csharp"], default="python")
    parser.add_argument("--combat-mcts-continuation-value", action="store_true", default=False)
    parser.add_argument("--ort-model-path", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    payload = run_probe(args)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
