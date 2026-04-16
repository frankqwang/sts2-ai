"""Forward hook collector for extracting NN internals without modifying training code.

Registers PyTorch forward hooks on combat_nn and ppo_net model instances
to capture attention weights, intermediate representations, and auxiliary
predictions for real-time visualization.

Usage:
    collector = NNInternalsCollector(combat_net, ppo_net)
    # ... run inference ...
    internals = collector.get_and_clear()
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Any


class AttentionCaptureWrapper:
    """Wraps nn.MultiheadAttention.forward to capture attention weights.

    PyTorch's MHA discards attention weights when need_weights=False (default
    in some code paths). This wrapper monkey-patches the module's forward to
    always request weights and stores them.
    """

    def __init__(self, mha_module: nn.MultiheadAttention, name: str):
        self.name = name
        self.weights: torch.Tensor | None = None
        self._module = mha_module
        self._original_forward = mha_module.forward

        # Replace forward with our capturing version
        mha_module.forward = self._capturing_forward

    def _capturing_forward(self, *args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False  # get per-head weights
        output = self._original_forward(*args, **kwargs)
        if isinstance(output, tuple) and len(output) >= 2:
            self.weights = output[1]  # (B, num_heads, tgt_len, src_len)
        return output

    def get_and_clear(self) -> np.ndarray | None:
        if self.weights is None:
            return None
        w = self.weights.detach().cpu().numpy()
        self.weights = None
        return w

    def restore(self):
        """Restore original forward method."""
        self._module.forward = self._original_forward


class RepresentationHook:
    """Captures output of a module via forward hook."""

    def __init__(self, module: nn.Module, name: str):
        self.name = name
        self.output: torch.Tensor | None = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        if isinstance(output, tuple):
            self.output = output[0].detach()
        else:
            self.output = output.detach()

    def get_and_clear(self) -> np.ndarray | None:
        if self.output is None:
            return None
        o = self.output.cpu().numpy()
        self.output = None
        return o

    def remove(self):
        self._handle.remove()


class NNInternalsCollector:
    """Collects neural network internals from combat_nn and ppo_net during inference.

    Captures:
    - Attention weights from SetEncoder and ScreenHead modules
    - Intermediate representations (hand, enemy, deck, trunk, screen context)
    - Auxiliary predictions (deck_quality, boss_readiness)
    """

    def __init__(self, combat_net=None, ppo_net=None):
        self._attn_wrappers: list[AttentionCaptureWrapper] = []
        self._repr_hooks: list[RepresentationHook] = []

        if combat_net is not None:
            self._setup_combat_hooks(combat_net)
        if ppo_net is not None:
            self._setup_ppo_hooks(ppo_net)

    def _setup_combat_hooks(self, net):
        """Hook into combat NN's attention and encoder layers."""
        # Attention weights from hand and enemy encoders
        if hasattr(net, 'hand_encoder') and hasattr(net.hand_encoder, 'attn'):
            self._attn_wrappers.append(
                AttentionCaptureWrapper(net.hand_encoder.attn, "combat_hand_attn"))
        if hasattr(net, 'enemy_encoder') and hasattr(net.enemy_encoder, 'attn'):
            self._attn_wrappers.append(
                AttentionCaptureWrapper(net.enemy_encoder.attn, "combat_enemy_attn"))

        # Representations
        if hasattr(net, 'hand_encoder'):
            self._repr_hooks.append(
                RepresentationHook(net.hand_encoder, "combat_hand_repr"))
        if hasattr(net, 'enemy_encoder'):
            self._repr_hooks.append(
                RepresentationHook(net.enemy_encoder, "combat_enemy_repr"))
        if hasattr(net, 'state_encoder'):
            self._repr_hooks.append(
                RepresentationHook(net.state_encoder, "combat_state_repr"))

    def _setup_ppo_hooks(self, net):
        """Hook into PPO network's attention and encoder layers."""
        # Set encoder attention weights
        for name in ['deck_encoder', 'relic_encoder', 'potion_encoder',
                      'hand_encoder', 'enemy_encoder']:
            encoder = getattr(net, name, None)
            if encoder is not None and hasattr(encoder, 'attn'):
                self._attn_wrappers.append(
                    AttentionCaptureWrapper(encoder.attn, f"ppo_{name}_attn"))

        # Screen head cross-attention weights
        for name in ['map_head', 'card_reward_head', 'shop_head']:
            head = getattr(net, name, None)
            if head is not None and hasattr(head, 'cross_attn'):
                self._attn_wrappers.append(
                    AttentionCaptureWrapper(head.cross_attn, f"ppo_{name}_cross_attn"))

        # Trunk representation
        if hasattr(net, 'trunk'):
            self._repr_hooks.append(
                RepresentationHook(net.trunk, "ppo_trunk_repr"))

        # Set encoder representations
        for name in ['deck_encoder', 'relic_encoder', 'potion_encoder']:
            encoder = getattr(net, name, None)
            if encoder is not None:
                self._repr_hooks.append(
                    RepresentationHook(encoder, f"ppo_{name}_repr"))

    def get_and_clear(self) -> dict[str, Any]:
        """Retrieve all captured internals and clear buffers.

        Returns dict with keys like:
            "combat_hand_attn": ndarray (B, heads, L, L) or None
            "combat_hand_repr": ndarray (B, dim) or None
            "ppo_deck_encoder_attn": ndarray or None
            etc.
        """
        result = {}
        for wrapper in self._attn_wrappers:
            data = wrapper.get_and_clear()
            if data is not None:
                result[wrapper.name] = data
        for hook in self._repr_hooks:
            data = hook.get_and_clear()
            if data is not None:
                result[hook.name] = data
        return result

    def cleanup(self):
        """Remove all hooks and restore original forwards."""
        for wrapper in self._attn_wrappers:
            wrapper.restore()
        for hook in self._repr_hooks:
            hook.remove()
        self._attn_wrappers.clear()
        self._repr_hooks.clear()


def format_internals_for_broadcast(
    internals: dict[str, Any],
    hand_names: list[str] | None = None,
    enemy_names: list[str] | None = None,
    deck_quality: float | None = None,
    boss_readiness: float | None = None,
    action_advantages: np.ndarray | None = None,
) -> dict:
    """Format raw internals into JSON-serializable dict for WebSocket broadcast.

    Converts numpy arrays to lists, applies sensible truncation,
    and adds semantic labels where possible.
    """
    out: dict[str, Any] = {}

    # Combat hand attention: (1, heads, L, L) → average over heads → (L,) per-card importance
    if "combat_hand_attn" in internals:
        attn = internals["combat_hand_attn"]  # (1, heads, L, L)
        # Average attention received by each card (column-wise mean of head-averaged matrix)
        avg_attn = attn[0].mean(axis=0)  # (L, L) averaged across heads
        card_importance = avg_attn.mean(axis=0)  # (L,) - how much attention each card receives
        # Per-head attention (first row = query-to-all-keys for first position)
        per_head = attn[0, :, :, :].tolist()  # heads × L × L
        out["hand_attention"] = {
            "importance": card_importance.tolist(),
            "per_head": per_head,
            "labels": hand_names or [],
        }

    # Combat enemy attention
    if "combat_enemy_attn" in internals:
        attn = internals["combat_enemy_attn"]
        avg_attn = attn[0].mean(axis=0)
        enemy_importance = avg_attn.mean(axis=0)
        per_head = attn[0, :, :, :].tolist()
        out["enemy_attention"] = {
            "importance": enemy_importance.tolist(),
            "per_head": per_head,
            "labels": enemy_names or [],
        }

    # PPO deck encoder attention
    if "ppo_deck_encoder_attn" in internals:
        attn = internals["ppo_deck_encoder_attn"]
        avg_attn = attn[0].mean(axis=0)
        card_importance = avg_attn.mean(axis=0)
        out["deck_attention"] = {
            "importance": card_importance.tolist(),
        }

    # Screen head cross-attention (map, card_reward, shop)
    for prefix, label in [("ppo_map_head_cross_attn", "map_attention"),
                           ("ppo_card_reward_head_cross_attn", "card_reward_attention"),
                           ("ppo_shop_head_cross_attn", "shop_attention")]:
        if prefix in internals:
            attn = internals[prefix]  # (1, heads, 1, L)
            # Average across heads: (1, L) → (L,)
            avg = attn[0].mean(axis=0).squeeze(0)
            out[label] = avg.tolist()

    # Auxiliary predictions
    if deck_quality is not None:
        out["deck_quality"] = round(float(deck_quality), 3)
    if boss_readiness is not None:
        out["boss_readiness"] = round(float(boss_readiness), 3)

    # Action advantages
    if action_advantages is not None:
        out["action_advantages"] = [round(float(a), 3) for a in action_advantages]

    return out
