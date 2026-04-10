"""Export combat actor model to ONNX for C# ORT CPU inference.

Usage:
    python export_actor_onnx.py --checkpoint checkpoints/act1/wizardly_baseline_iter1000.pt --output actor_combat.onnx
    python export_actor_onnx.py --checkpoint checkpoints/act1/wizardly_baseline_iter1000.pt --output actor_combat.onnx --dump-fixtures fixtures/

The exported model takes flat tensor inputs (not dict) and returns (logits, value).
Batch dimension is dynamic (default batch=1 for C# per-env inference).
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from combat_nn import (
    CARD_AUX_DIM,
    CombatPolicyValueNetwork,
    build_combat_action_features,
    build_combat_features,
)
from vocab import Vocab, load_vocab


# Combat feature constants — keep in sync with rl_encoder_v2.py / combat_nn.py
MAX_HAND_SIZE = 12
MAX_ENEMIES = 5
MAX_ACTIONS = 30
COMBAT_SCALAR_DIM = 18
COMBAT_EXTRA_SCALAR_DIM = 14  # v2 player powers (appended at END of state_input)
# 2026-04-08 PM (wizardly merge): bumped 32 → 40 for the P0+QG union
# slot table. See `rl_encoder_v2.ENEMY_AUX_DIM` slot reservation table.
ENEMY_AUX_DIM = 40


MAX_DECK_SIZE = 50  # from rl_encoder_v2


class CombatActorONNXWrapper(nn.Module):
    """Wrapper that takes flat tensors instead of dicts for ONNX export.

    Supports optional deck inputs for build_plan_z models (deck_repr_dim > 0).
    """

    def __init__(self, network: CombatPolicyValueNetwork, has_deck: bool = False):
        super().__init__()
        self.network = network
        self.has_deck = has_deck

    def forward(
        self,
        scalars: torch.Tensor,       # (B, 18)
        hand_ids: torch.Tensor,       # (B, 12) int64
        hand_aux: torch.Tensor,       # (B, 12, 51) float32
        hand_mask: torch.Tensor,      # (B, 12) float32 (1/0)
        enemy_ids: torch.Tensor,      # (B, 5) int64
        enemy_aux: torch.Tensor,      # (B, 5, 40) float32 (v3 wizardly union, was 32)
        enemy_mask: torch.Tensor,     # (B, 5) float32 (1/0)
        action_type_ids: torch.Tensor,   # (B, 30) int64
        target_card_ids: torch.Tensor,   # (B, 30) int64
        target_enemy_ids: torch.Tensor,  # (B, 30) int64
        action_mask: torch.Tensor,       # (B, 30) float32 (1/0)
        extra_scalars: torch.Tensor | None = None,  # (B, 14) float32 (v2 player powers)
        deck_ids: torch.Tensor | None = None,     # (B, 50) int64
        deck_aux: torch.Tensor | None = None,     # (B, 50, 51) float32
        deck_mask: torch.Tensor | None = None,    # (B, 50) float32 (1/0)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_features = {
            "scalars": scalars,
            "hand_ids": hand_ids,
            "hand_aux": hand_aux,
            "hand_mask": hand_mask.bool(),
            "enemy_ids": enemy_ids,
            "enemy_aux": enemy_aux,
            "enemy_mask": enemy_mask.bool(),
        }
        if extra_scalars is not None:
            state_features["extra_scalars"] = extra_scalars
        else:
            state_features["extra_scalars"] = torch.zeros(
                scalars.shape[0], COMBAT_EXTRA_SCALAR_DIM,
                dtype=scalars.dtype, device=scalars.device,
            )
        if self.has_deck and deck_ids is not None:
            state_features["deck_ids"] = deck_ids
            state_features["deck_aux"] = deck_aux
            state_features["deck_mask"] = deck_mask.bool()
        action_features = {
            "action_type_ids": action_type_ids,
            "target_card_ids": target_card_ids,
            "target_enemy_ids": target_enemy_ids,
            "action_mask": action_mask.bool(),
        }
        logits, value = self.network(state_features, action_features)
        return logits, value


def load_combat_network(checkpoint_path: str, vocab: Vocab, device: str = "cpu") -> CombatPolicyValueNetwork:
    """Load combat network from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("mcts_model") or ckpt.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"No combat model found in {checkpoint_path}")

    # Auto-detect dimensions
    card_w = state_dict.get("entity_emb.card_embed.weight")
    action_w = state_dict.get("action_proj.weight")
    embed_dim = int(card_w.shape[1]) if card_w is not None else 32
    hidden_dim = int(action_w.shape[0]) if action_w is not None else 128

    # Check for optional components
    has_deck = any("deck_encoder" in k for k in state_dict)
    deck_repr_dim = 0
    if has_deck:
        deck_w = state_dict.get("deck_encoder.proj.weight")
        if deck_w is not None:
            deck_repr_dim = int(deck_w.shape[0])

    has_adapter = any("delta_logits_head" in k for k in state_dict)

    # Check if pile encoders exist in checkpoint — if not, don't enable deck_repr_dim
    # because current code auto-creates pile encoders when deck_repr_dim > 0,
    # which changes state_encoder input dim
    has_pile_encoders = any("draw_pile_encoder" in k for k in state_dict)
    effective_deck_dim = deck_repr_dim if has_pile_encoders or deck_repr_dim == 0 else deck_repr_dim
    # If checkpoint has deck but not pile encoders, we need to handle it:
    # The checkpoint was trained when pile encoders didn't exist yet.
    # We must match the exact architecture it was trained with.
    if deck_repr_dim > 0 and not has_pile_encoders:
        # Temporarily set deck_repr_dim=0 for network creation, then manually
        # add just the deck_encoder and adjust state_encoder
        se_weight = state_dict.get("state_encoder.0.weight")
        if se_weight is not None:
            actual_input_dim = se_weight.shape[1]
            # If the state encoder expects scalar+hand+enemy+deck (no pile), we need
            # to create network that matches. The simplest way: set deck_repr_dim=0
            # and let safe_load handle the deck_encoder separately.
            # But this won't work because state_encoder won't have right input dim.
            # Better: just use deck_repr_dim=0 and skip deck for now.
            pass

    # For old checkpoints (pre-pile): deck_repr_dim is detected from state_encoder
    # but pile encoders don't exist. Current CombatPolicyValueNetwork auto-creates
    # pile encoders when deck_repr_dim > 0, changing state_encoder input dim.
    # Solution: if checkpoint has deck but not pile encoders, set deck_repr_dim=0
    # for network creation (lose deck info in ORT, but get correct weights).
    # For live training snapshots: the network is already correctly configured.
    has_pile_encoders = any("draw_pile_encoder" in k for k in state_dict)

    network = CombatPolicyValueNetwork(
        vocab=vocab,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        deck_repr_dim=deck_repr_dim,
        residual_adapter=has_adapter,
        pile_specific=has_pile_encoders,  # False for pre-pile checkpoints
    )

    # Safe load (skip mismatched shapes)
    current = network.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in current and current[k].shape == v.shape}
    network.load_state_dict(filtered, strict=False)
    network.to(device).eval()
    return network


def make_dummy_inputs(batch: int = 1, device: str = "cpu", has_deck: bool = False) -> tuple[list[torch.Tensor], list[str]]:
    """Create dummy inputs for ONNX tracing."""
    inputs = [
        torch.zeros(batch, COMBAT_SCALAR_DIM, dtype=torch.float32, device=device),
        torch.zeros(batch, MAX_HAND_SIZE, dtype=torch.int64, device=device),
        torch.zeros(batch, MAX_HAND_SIZE, CARD_AUX_DIM, dtype=torch.float32, device=device),
        torch.ones(batch, MAX_HAND_SIZE, dtype=torch.float32, device=device),
        torch.zeros(batch, MAX_ENEMIES, dtype=torch.int64, device=device),
        torch.zeros(batch, MAX_ENEMIES, ENEMY_AUX_DIM, dtype=torch.float32, device=device),
        torch.ones(batch, MAX_ENEMIES, dtype=torch.float32, device=device),
        torch.zeros(batch, MAX_ACTIONS, dtype=torch.int64, device=device),
        torch.zeros(batch, MAX_ACTIONS, dtype=torch.int64, device=device),
        torch.zeros(batch, MAX_ACTIONS, dtype=torch.int64, device=device),
        torch.ones(batch, MAX_ACTIONS, dtype=torch.float32, device=device),
        torch.zeros(batch, COMBAT_EXTRA_SCALAR_DIM, dtype=torch.float32, device=device),
    ]
    names = [
        "scalars", "hand_ids", "hand_aux", "hand_mask",
        "enemy_ids", "enemy_aux", "enemy_mask",
        "action_type_ids", "target_card_ids", "target_enemy_ids", "action_mask",
        "extra_scalars",
    ]
    if has_deck:
        inputs.extend([
            torch.zeros(batch, MAX_DECK_SIZE, dtype=torch.int64, device=device),
            torch.zeros(batch, MAX_DECK_SIZE, CARD_AUX_DIM, dtype=torch.float32, device=device),
            torch.ones(batch, MAX_DECK_SIZE, dtype=torch.float32, device=device),
        ])
        names.extend(["deck_ids", "deck_aux", "deck_mask"])
    return inputs, names


def export_onnx(
    network: CombatPolicyValueNetwork,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """Export combat actor to ONNX."""
    has_deck = network.deck_repr_dim > 0
    wrapper = CombatActorONNXWrapper(network, has_deck=has_deck)
    wrapper.eval()

    inputs, input_names = make_dummy_inputs(batch=1, device="cpu", has_deck=has_deck)
    wrapper.to("cpu")

    # Dynamic axes for batch dimension
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["policy_logits"] = {0: "batch"}
    dynamic_axes["value"] = {0: "batch"}

    torch.onnx.export(
        wrapper,
        tuple(inputs),
        output_path,
        input_names=input_names,
        output_names=["policy_logits", "value"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Exported ONNX to {output_path} (has_deck={has_deck})")


def export_from_training_snapshot(
    ppo_state_dict: dict,
    mcts_state_dict: dict,
    vocab: Vocab,
    output_path: str,
    policy_version: int = 0,
) -> float:
    """Export ONNX from a training snapshot. Returns export time in ms."""
    import time
    t0 = time.perf_counter()

    # Auto-detect config from state dict
    card_w = mcts_state_dict.get("entity_emb.card_embed.weight")
    action_w = mcts_state_dict.get("action_proj.weight")
    embed_dim = int(card_w.shape[1]) if card_w is not None else 32
    hidden_dim = int(action_w.shape[0]) if action_w is not None else 128
    deck_w = mcts_state_dict.get("deck_encoder.proj.weight")
    deck_repr_dim = int(deck_w.shape[0]) if deck_w is not None else 0
    has_adapter = any("delta_logits_head" in k for k in mcts_state_dict)

    has_pile = any("draw_pile_encoder" in k for k in mcts_state_dict)
    network = CombatPolicyValueNetwork(
        vocab=vocab, embed_dim=embed_dim, hidden_dim=hidden_dim,
        deck_repr_dim=deck_repr_dim, residual_adapter=has_adapter,
        pile_specific=has_pile,
    )
    current = network.state_dict()
    filtered = {k: v for k, v in mcts_state_dict.items() if k in current and current[k].shape == v.shape}
    network.load_state_dict(filtered, strict=False)
    network.cpu().eval()

    export_onnx(network, output_path)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms


def dump_real_fixtures(
    network: CombatPolicyValueNetwork,
    vocab: Vocab,
    output_dir: str,
    port: int = 21580,
    num_samples: int = 10,
) -> None:
    """Collect real combat states from sim and dump features + model outputs as fixtures."""
    from full_run_env import create_full_run_client

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = create_full_run_client(use_pipe=True, transport="pipe-binary", port=port)
    samples = []

    for ep in range(num_samples * 3):  # run enough episodes to get samples
        if len(samples) >= num_samples:
            break
        state = client.reset(character_id="IRONCLAD", ascension_level=0, timeout_s=30)
        for step in range(200):
            st = (state.get("state_type") or "").lower()
            legal = [a for a in (state.get("legal_actions") or []) if isinstance(a, dict)]
            if st == "game_over":
                break
            if not legal:
                state = client.act({"action": "wait"})
                continue
            if st in {"monster", "elite", "boss"} and len(samples) < num_samples:
                # Build features
                sf = build_combat_features(state, vocab)
                af = build_combat_action_features(state, legal, vocab)

                # Get PyTorch outputs
                sf_t = {k: torch.tensor(v).unsqueeze(0).float() if v.dtype not in (np.int64, np.int32) else torch.tensor(v).unsqueeze(0).long() for k, v in sf.items() if isinstance(v, np.ndarray)}
                af_t = {k: torch.tensor(v).unsqueeze(0).float() if v.dtype not in (np.int64, np.int32) else torch.tensor(v).unsqueeze(0).long() for k, v in af.items() if isinstance(v, np.ndarray)}
                # Handle bool
                for k in sf_t:
                    if sf[k].dtype == bool:
                        sf_t[k] = torch.tensor(sf[k]).unsqueeze(0).bool()
                for k in af_t:
                    if af[k].dtype == bool:
                        af_t[k] = torch.tensor(af[k]).unsqueeze(0).bool()

                with torch.no_grad():
                    logits, value = network(sf_t, af_t)

                sample = {
                    "state_features": {k: v.tolist() for k, v in sf.items() if isinstance(v, np.ndarray)},
                    "action_features": {k: v.tolist() for k, v in af.items() if isinstance(v, np.ndarray)},
                    "pytorch_logits": logits[0].cpu().tolist(),
                    "pytorch_value": float(value[0].cpu().item()),
                    "num_legal_actions": len(legal),
                    "state_type": st,
                }
                samples.append(sample)

            state = client.act(legal[0])

    client.close()

    # Save as JSON
    fixture_path = output_path / "combat_actor_fixtures.json"
    with open(fixture_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} fixtures to {fixture_path}")

    # Also save as numpy for easier loading
    for i, sample in enumerate(samples):
        npz_path = output_path / f"sample_{i:03d}.npz"
        arrays = {}
        for k, v in sample["state_features"].items():
            arrays[f"sf_{k}"] = np.array(v)
        for k, v in sample["action_features"].items():
            arrays[f"af_{k}"] = np.array(v)
        arrays["pytorch_logits"] = np.array(sample["pytorch_logits"], dtype=np.float32)
        arrays["pytorch_value"] = np.array(sample["pytorch_value"], dtype=np.float32)
        np.savez(npz_path, **arrays)

    print(f"Saved {len(samples)} NPZ fixtures to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export combat actor to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Combat checkpoint path")
    parser.add_argument("--output", default="actor_combat.onnx", help="ONNX output path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dump-fixtures", default=None, help="Directory to dump real feature fixtures")
    parser.add_argument("--fixture-port", type=int, default=21580, help="HeadlessSim port for fixtures")
    parser.add_argument("--fixture-samples", type=int, default=10, help="Number of fixture samples")
    args = parser.parse_args()

    vocab = load_vocab()
    network = load_combat_network(args.checkpoint, vocab)
    print(f"Loaded combat network: embed_dim={network.entity_emb.card_embed.embedding_dim}, "
          f"hidden_dim={network.state_encoder[0].in_features}")

    export_onnx(network, args.output, opset_version=args.opset)

    if args.dump_fixtures:
        dump_real_fixtures(network, vocab, args.dump_fixtures,
                          port=args.fixture_port, num_samples=args.fixture_samples)


if __name__ == "__main__":
    main()
