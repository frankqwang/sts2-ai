#!/usr/bin/env python3
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
import logging
from typing import Any

import torch

from search.combat_mcts_agent import CombatMCTSAgent, PipeCombatForwardModel, _reconcile_action
from network.combat_network import CombatPolicyValueNetwork
from evaluate_ai import (
    _infer_combat_dims,
    _infer_ppo_embed_dim,
    _infer_retrieval_proj_dim,
    _safe_load_state_dict,
    _summarize_mcts_root,
)
from ipc.headless_sim_runner import DEFAULT_DLL_PATH, start_headless_sim, stop_process
from ipc.full_run_env import PipeBackedFullRunClient, create_full_run_client
from search.mcts_core import MCTSConfig
from tools.public_state_trace import build_trace_entry
from network.fullrun_policy import FullRunPolicyNetworkV2
from verify_save_load import COMBAT_TYPES, drive_to_state
from core.vocab import load_vocab
from core.checkpoint_compat import get_combat_model_config, get_combat_model_state


logger = logging.getLogger("mcts_pipe_audit")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _state_hash(state: dict[str, Any]) -> str:
    return build_trace_entry(state, step=0, action=None).public_state_hash


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") if isinstance(battle.get("player"), dict) else (state.get("player") or {})
    enemies = []
    for enemy in state.get("enemies") or battle.get("enemies") or []:
        if not isinstance(enemy, dict):
            continue
        enemies.append(
            {
                "id": enemy.get("entity_id") or enemy.get("id") or enemy.get("monster_id"),
                "hp": enemy.get("hp"),
                "max_hp": enemy.get("max_hp"),
                "block": enemy.get("block"),
                "alive": bool(enemy.get("is_alive", True)),
                "intent": [
                    intent.get("type")
                    for intent in enemy.get("intents") or []
                    if isinstance(intent, dict)
                ],
            }
        )
    legal = state.get("legal_actions") or []
    return {
        "hash": _state_hash(state),
        "state_type": state.get("state_type"),
        "floor": run.get("floor"),
        "act": run.get("act"),
        "round": state.get("round_number") or battle.get("round_number"),
        "terminal": bool(state.get("terminal", False)),
        "hp": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "block": player.get("block"),
        "energy": battle.get("energy") or player.get("energy"),
        "legal_action_count": len(legal),
        "legal_action_labels": [
            {
                "action": entry.get("action"),
                "label": entry.get("label"),
                "target_id": entry.get("target_id"),
                "card_index": entry.get("card_index"),
            }
            for entry in legal[:12]
            if isinstance(entry, dict)
        ],
        "enemies": enemies,
    }


def _load_models(checkpoint_path: Path):
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
    return vocab, combat_net


def _root_save_load_audit(client: PipeBackedFullRunClient, root_state: dict[str, Any]) -> dict[str, Any]:
    state_id = client.save_state()
    try:
        restored = client.load_state(state_id)
        return {
            "before": _state_summary(root_state),
            "after": _state_summary(restored),
            "matches": _state_hash(root_state) == _state_hash(restored),
        }
    finally:
        client.delete_state(state_id)


def _child_actual_transition(
    client: PipeBackedFullRunClient,
    action: dict[str, Any],
) -> dict[str, Any]:
    state_id = client.save_state()
    try:
        next_state = client.act(action)
        return _state_summary(next_state)
    finally:
        client.load_state(state_id)
        client.delete_state(state_id)


def _restore_root_state(
    client: PipeBackedFullRunClient,
    root_state_id: str,
) -> None:
    client.load_state(root_state_id)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    vocab, combat_net = _load_models(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combat_net.to(device).eval()
    agent = CombatMCTSAgent(
        network=combat_net,
        vocab=vocab,
        config=MCTSConfig(
            num_simulations=max(1, int(args.combat_mcts_sims)),
            c_puct=float(args.combat_mcts_c_puct),
            temperature=0.0,
            dirichlet_alpha=0.0,
            dirichlet_fraction=0.0,
            num_determinizations=1,
        ),
        training=False,
        device=device,
        backend=str(getattr(args, "combat_mcts_backend", "python")),
        use_continuation_value=bool(getattr(args, "combat_mcts_continuation_value", False)),
    )
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

    root_state = drive_to_state(client, args.seed, COMBAT_TYPES)
    root_summary = _state_summary(root_state)
    root_save_load = _root_save_load_audit(client, root_state)
    root_state_id = client.save_state()

    fm = PipeCombatForwardModel.from_current_state(
        client._pipe,
        max_step_budget=max(1, int(args.combat_mcts_step_budget)),
    )
    try:
        chosen_action, root = agent.choose_action(fm)
        root_top_actions, root_value = _summarize_mcts_root(root, k=max(1, int(args.top_k)))

        child_reports: list[dict[str, Any]] = []
        for rank, item in enumerate(root_top_actions, start=1):
            raw_action = item.get("action") if isinstance(item, dict) else {}
            if not isinstance(raw_action, dict):
                continue
            action = _reconcile_action(raw_action, root_state.get("legal_actions") or [])
            _restore_root_state(client, root_state_id)
            child_model = fm.clone()
            child_model.step(action)
            predicted_summary = _state_summary(child_model.get_state_dict())
            predicted_is_terminal = bool(child_model.is_terminal)
            predicted_player_won = bool(child_model.player_won)
            predicted_legal = child_model.get_legal_actions()
            if predicted_is_terminal or not predicted_legal:
                predicted_value = 1.0 if predicted_player_won else -1.0
            else:
                _policy, predicted_value = agent.evaluator.evaluate(
                    child_model.get_state_dict(),
                    predicted_legal,
                )

            _restore_root_state(client, root_state_id)
            actual_summary = _child_actual_transition(client, action)
            child_reports.append(
                {
                    "rank": rank,
                    "action": _json_safe(action),
                    "root_stats": _json_safe(item),
                    "predicted_child": predicted_summary,
                    "actual_child": actual_summary,
                    "child_hash_matches": predicted_summary["hash"] == actual_summary["hash"],
                    "predicted_is_terminal": predicted_is_terminal,
                    "predicted_player_won": predicted_player_won,
                    "predicted_leaf_value": float(predicted_value),
                }
            )

        restored = fm.cleanup_and_restore()
        restore_summary = _state_summary(restored or client.get_state())
        restore_matches = restore_summary["hash"] == root_summary["hash"]

        return {
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "device": str(device),
            "combat_mcts_sims": int(args.combat_mcts_sims),
            "combat_mcts_c_puct": float(args.combat_mcts_c_puct),
            "combat_mcts_step_budget": int(args.combat_mcts_step_budget),
            "root_state": root_summary,
            "root_save_load": root_save_load,
            "search_restore": {
                "after": restore_summary,
                "matches_root": restore_matches,
            },
            "root_choice": {
                "chosen_action": _json_safe(chosen_action),
                "root_value": float(root_value),
                "top_actions": _json_safe(root_top_actions),
            },
            "children": child_reports,
        }
    finally:
        try:
            fm.cleanup()
        except Exception:
            pass
        try:
            client.delete_state(root_state_id)
        except Exception:
            pass
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pipe-backed combat MCTS root/child semantics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--combat-mcts-sims", type=int, default=50)
    parser.add_argument("--combat-mcts-c-puct", type=float, default=1.5)
    parser.add_argument("--combat-mcts-step-budget", type=int, default=200)
    parser.add_argument("--combat-mcts-backend", choices=["python", "csharp"], default="python")
    parser.add_argument("--combat-mcts-continuation-value", action="store_true", default=False)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ort-model-path", default=None)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--headless-dll", default=str(DEFAULT_DLL_PATH))
    parser.add_argument("--auto-launch", action="store_true", default=False)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    proc = None
    if args.auto_launch:
        proc = start_headless_sim(
            port=int(args.port),
            repo_root=Path(args.repo_root).resolve(),
            dll_path=Path(args.headless_dll).resolve(),
            protocol="bin",
        )
        atexit.register(lambda: stop_process(proc))
    report = run_audit(args)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote MCTS pipe audit to %s", output_path)
    if proc is not None:
        stop_process(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
