"""Combat MCTS agent — wires together MCTS search + Combat NN + Forward Model.

This is the combat decision-maker. Given a combat state + legal actions,
it runs MCTS search guided by the neural network and returns the best action.

Usage:
    agent = CombatMCTSAgent.from_checkpoint("combat_best.pt")
    action = agent.choose_action(state, legal_actions, forward_model)
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import json
import logging
from pathlib import Path
from typing import Any

import torch

from mcts_core import (
    CombatForwardModel,
    MCTSConfig,
    MCTSNode,
    UniformEvaluator,
    mcts_search,
    mcts_search_with_determinization,
)
from combat_nn import (
    CombatNNEvaluator,
    CombatPolicyValueNetwork,
)
from vocab import Vocab, load_vocab

logger = logging.getLogger(__name__)


def _safe_load_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None)
    }
    model.load_state_dict(filtered, strict=False)


class CombatMCTSAgent:
    """MCTS-based combat agent with neural network guidance."""

    def __init__(
        self,
        network: CombatPolicyValueNetwork,
        vocab: Vocab,
        config: MCTSConfig | None = None,
        training: bool = False,
        device: torch.device | None = None,
    ):
        self.network = network
        self.vocab = vocab
        self.config = config or MCTSConfig()
        self.training = training
        self.evaluator = CombatNNEvaluator(network, vocab, device=device)

    def choose_action(
        self,
        forward_model: CombatForwardModel,
    ) -> tuple[dict[str, Any], MCTSNode]:
        """Run MCTS and choose an action.

        Args:
            forward_model: Combat simulator at current decision point.

        Returns:
            action: chosen action dict
            root: MCTS root node (for extracting training targets)
        """
        if self.config.num_determinizations > 1:
            root = mcts_search_with_determinization(
                forward_model, self.evaluator, self.config)
        else:
            root = mcts_search(forward_model, self.evaluator, self.config)

        temperature = self.config.temperature if self.training else 0.0
        action = root.best_action(temperature=temperature)

        return action, root

    def choose_action_from_state(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        forward_model: CombatForwardModel,
    ) -> dict[str, Any]:
        """Convenience method matching the non-combat brain interface."""
        action, _ = self.choose_action(forward_model)
        return action

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        vocab: Vocab | None = None,
        config: MCTSConfig | None = None,
        training: bool = False,
    ) -> CombatMCTSAgent:
        """Load from a saved checkpoint."""
        path = Path(path)
        if vocab is None:
            vocab = load_vocab()

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        net_config = checkpoint.get("config", {})
        network = CombatPolicyValueNetwork(
            vocab=vocab,
            embed_dim=net_config.get("embed_dim", 32),
            hidden_dim=net_config.get("hidden_dim", 128),
        )
        _safe_load_state_dict(network, checkpoint["model_state_dict"])
        network.eval()
        return cls(network=network, vocab=vocab, config=config, training=training)

    @classmethod
    def with_random_policy(
        cls,
        vocab: Vocab | None = None,
        config: MCTSConfig | None = None,
    ) -> CombatMCTSAgent:
        """Create agent with untrained network (for initial self-play)."""
        if vocab is None:
            vocab = load_vocab()
        network = CombatPolicyValueNetwork(vocab=vocab)
        return cls(network=network, vocab=vocab, config=config, training=True)


# ---------------------------------------------------------------------------
# HTTP-based forward model adapter (SLOW — prototype only)
# ---------------------------------------------------------------------------

class HttpCombatForwardModel:
    """Adapts the existing HTTP simulator API as a CombatForwardModel.

    This is a SLOW forward model (~24ms/step) — usable for prototype
    MCTS with small simulation budgets (10-50 sims), but not for
    production-grade 200+ sim search.

    When the pure C# combat simulator is ready, replace with a
    faster implementation (pythonnet or named pipe).
    """

    def __init__(self, state: dict[str, Any], client=None, base_url: str | None = None):
        self._state = state
        self._client = client
        self._base_url = base_url
        self._is_terminal = False
        self._player_won = False

        # Check if already terminal
        st = (state.get("state_type") or "").lower()
        if st == "game_over" or state.get("terminal"):
            self._is_terminal = True
            go = state.get("game_over") or {}
            outcome = (go.get("run_outcome") or go.get("outcome") or "").lower()
            self._player_won = "victory" in outcome or outcome == "win"

    def clone(self) -> HttpCombatForwardModel:
        """WARNING: HTTP model cannot truly clone server state.

        Returns a copy of the Python-side state dict. This means MCTS
        branching won't work correctly with HTTP — multiple clones will
        share the same server state.

        Use this only for single-path rollouts (no branching MCTS).
        For real MCTS, use the pure C# forward model.
        """
        import copy
        return HttpCombatForwardModel(
            state=copy.deepcopy(self._state),
            client=self._client,
            base_url=self._base_url,
        )

    def get_legal_actions(self) -> list[dict[str, Any]]:
        legal = self._state.get("legal_actions", [])
        if isinstance(legal, list):
            return legal
        return []

    def step(self, action: dict[str, Any]) -> None:
        if self._client is None:
            # Dummy step — just mark as terminal
            self._is_terminal = True
            return
        try:
            self._state = self._client.act(action)
            st = (self._state.get("state_type") or "").lower()
            if st == "game_over" or self._state.get("terminal"):
                self._is_terminal = True
                go = self._state.get("game_over") or {}
                outcome = (go.get("run_outcome") or go.get("outcome") or "").lower()
                self._player_won = "victory" in outcome or outcome == "win"
        except Exception:
            self._is_terminal = True

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @property
    def player_won(self) -> bool:
        return self._player_won

    def get_state_dict(self) -> dict[str, Any]:
        return self._state


# ---------------------------------------------------------------------------
# Pipe-based forward model (FAST — production MCTS)
# ---------------------------------------------------------------------------

# Screen types where combat is still active
_COMBAT_ACTIVE_STATES = {"combat", "monster", "elite", "boss", "hand_select", "card_select"}


def _check_terminal(state: dict[str, Any]) -> tuple[bool, bool]:
    """Check if state is terminal for MCTS purposes.

    Returns (is_terminal, player_won).

    Terminal means combat is over: either the player died (game_over),
    or combat ended and the state transitioned to post-combat
    (combat_rewards, card_reward, etc.).
    """
    st = (state.get("state_type") or "").lower()

    # Explicit game over
    if st == "game_over" or state.get("terminal"):
        go = state.get("game_over") or {}
        outcome = (go.get("run_outcome") or go.get("outcome") or "").lower()
        won = "victory" in outcome or outcome == "win"
        return True, won

    # Combat ended → moved to post-combat screen
    if st not in _COMBAT_ACTIVE_STATES:
        # Player survived combat (moved to rewards, map, etc.)
        return True, True

    return False, False


def _reconcile_action(action: dict[str, Any],
                      server_legal: list[dict[str, Any]]) -> dict[str, Any]:
    """Match MCTS tree action to server's current legal_actions.

    After save/load, card_index may map to a DIFFERENT card because hand
    order is reconstructed (draw pile re-shuffled). NEVER trust card_index
    across save/load boundaries. Use label (card name) as identity instead.

    Matching priority:
    1. Exact: (action, label, target_id)
    2. By label: (action, label) — use server's card_index/target_id
    3. By action type + slot (potions): (action, slot)
    4. By action type alone (end_turn, proceed, etc.)
    5. Fallback: return cleaned original
    """
    act_type = action.get("action", "")
    target = action.get("target_id")
    label = action.get("label", "")
    slot = action.get("slot")

    _FIELDS = ("action", "index", "card_index", "hand_index",
               "slot", "target_id", "target", "col", "row", "value", "label")

    if not server_legal:
        return {k: v for k, v in action.items() if k in _FIELDS}

    # 1. Exact match by label + target
    if label:
        for la in server_legal:
            if (la.get("action") == act_type
                    and la.get("label") == label
                    and la.get("target_id") == target):
                return {k: v for k, v in la.items() if k in _FIELDS}

    # 2. Match by label only (server provides correct card_index/target)
    if label:
        for la in server_legal:
            if la.get("action") == act_type and la.get("label") == label:
                return {k: v for k, v in la.items() if k in _FIELDS}

    # 3. Match by slot (for potions — slot is reliable across save/load)
    if slot is not None:
        for la in server_legal:
            if la.get("action") == act_type and la.get("slot") == slot:
                return {k: v for k, v in la.items() if k in _FIELDS}

    # 4. Match on action type alone (end_turn, proceed, etc.)
    for la in server_legal:
        if la.get("action") == act_type:
            return {k: v for k, v in la.items() if k in _FIELDS}

    # 5. Fallback
    return {k: v for k, v in action.items() if k in _FIELDS}


class PipeCombatForwardModel:
    """High-speed forward model using named pipe IPC + save/load state.

    Uses the C# simulator's state snapshot system for MCTS branching:
    - clone() = save server state → get state_id, mark clone as needing restore
    - step() = restore to snapshot (if needed), then execute action
    - ~0.5ms/step instead of ~24ms/step (HTTP), enabling 1000+ sims/decision

    Memory management:
    - Each clone holds a state_id (server-side snapshot)
    - Caller must call cleanup() on the root model after MCTS completes
      to delete all accumulated snapshots, OR use the context manager
    - Alternatively, call clear_all_snapshots() for bulk cleanup

    Typical usage with mcts_core.mcts_search():
        pipe = PipeClient(port=15527)
        pipe.connect()
        state = pipe.call("state")
        fm = PipeCombatForwardModel.from_current_state(pipe)
        root = mcts_search(fm, evaluator, config)
        fm.cleanup()  # delete all snapshots created during search
        action = root.best_action(temperature=0)
    """

    def __init__(
        self,
        pipe,  # PipeClient instance OR callable returning PipeClient
        state_id: str,
        state: dict[str, Any],
        *,
        needs_restore: bool = False,
        snapshot_registry: list[str] | None = None,
    ):
        # Support both a direct PipeClient and a callable that returns one.
        # When a callable (pipe_getter) is used, every access to self.pipe
        # fetches the latest reference — surviving reconnections.
        if callable(pipe) and not hasattr(pipe, "call"):
            self._pipe_getter = pipe
            self._pipe_direct = None
        else:
            self._pipe_getter = None
            self._pipe_direct = pipe
        self._state_id = state_id
        self._state = state
        self._needs_restore = needs_restore
        self._step_count = 0

        is_term, p_won = _check_terminal(state)
        self._is_terminal = is_term
        self._player_won = p_won

        # Shared list across all clones from the same root — tracks all
        # state_ids created so they can be bulk-deleted after search.
        self._snapshot_registry = snapshot_registry if snapshot_registry is not None else [state_id]

    @property
    def _pipe(self):
        """Always return the latest PipeClient reference."""
        if self._pipe_getter is not None:
            return self._pipe_getter()
        return self._pipe_direct

    @classmethod
    def from_current_state(cls, pipe, max_step_budget: int = 500) -> PipeCombatForwardModel:
        """Create a forward model from the pipe's current server state.

        Args:
            pipe: PipeClient instance OR callable returning PipeClient.
                  When a callable is passed, it's stored so all clones
                  always get the latest pipe reference (survives reconnect).

        Polls until state has legal_actions (handles async combat init),
        then saves a snapshot so clone()/load can work.
        """
        import time
        # Resolve pipe for immediate calls; keep original for storage
        raw_pipe = pipe() if (callable(pipe) and not hasattr(pipe, "call")) else pipe
        state = raw_pipe.call("state")
        # Poll until actionable (combat init may need frames)
        for _ in range(50):
            legal = state.get("legal_actions", [])
            if isinstance(legal, list) and any(
                isinstance(a, dict) and a.get("is_enabled") is not False for a in legal
            ):
                break
            if state.get("terminal") or (state.get("state_type") or "").lower() == "game_over":
                break
            time.sleep(0.01)
            raw_pipe = pipe() if (callable(pipe) and not hasattr(pipe, "call")) else pipe
            state = raw_pipe.call("state")

        raw_pipe = pipe() if (callable(pipe) and not hasattr(pipe, "call")) else pipe
        result = raw_pipe.call("save_state")
        state_id = result["state_id"]
        fm = cls(pipe=pipe, state_id=state_id, state=state)  # pass original (getter or direct)
        fm._max_step_budget = max_step_budget
        return fm

    def clone(self) -> PipeCombatForwardModel:
        """Clone by reusing our snapshot ID.

        The clone remembers our state_id and will load_state() before its
        first step().  No new save_state call is needed because:
        - MCTS only clones the root model
        - All clones restore to the same root state before stepping
        - The root's snapshot persists until cleanup()

        This avoids one save_state round-trip per MCTS simulation (200x
        fewer pipe calls than saving per clone).
        """
        # Pass the pipe_getter (or direct pipe) so clones also get fresh refs
        pipe_arg = self._pipe_getter if self._pipe_getter is not None else self._pipe_direct
        child = PipeCombatForwardModel(
            pipe=pipe_arg,
            state_id=self._state_id,  # reuse our snapshot
            state=self._state,         # cached state (replaced on step)
            needs_restore=True,        # will load_state before first step
            snapshot_registry=self._snapshot_registry,
        )
        child._max_step_budget = getattr(self, "_max_step_budget", 500)
        return child

    def get_legal_actions(self) -> list[dict[str, Any]]:
        legal = self._state.get("legal_actions", [])
        if isinstance(legal, list):
            return [a for a in legal
                    if isinstance(a, dict) and a.get("is_enabled") is not False]
        return []

    def step(self, action: dict[str, Any]) -> None:
        if self._is_terminal:
            return

        # Safety: prevent runaway simulations
        budget = getattr(self, "_max_step_budget", 500)
        self._step_count += 1
        if self._step_count > budget:
            logger.warning("PipeCombatForwardModel exceeded step budget (%d), forcing terminal", budget)
            self._is_terminal = True
            return

        # Restore to our snapshot if needed (first step after clone)
        if self._needs_restore:
            resp = self._pipe.call("load_state", {"state_id": self._state_id})
            if isinstance(resp, dict) and "state_type" in resp:
                self._state = resp
            self._needs_restore = False

        # Reconcile action with server's current legal_actions.
        # After load_state, the server's legal_actions are the ground truth.
        # The MCTS tree's stored action may have stale target_id (e.g., DEFEND
        # stored with target_id from a different branch, or STRIKE missing target).
        clean = _reconcile_action(action, self.get_legal_actions())

        try:
            result = self._pipe.call("step", clean)
            # step returns envelope with state inside, or flat state
            if "state" in result and isinstance(result["state"], dict):
                self._state = result["state"]
            elif "state_type" in result:
                self._state = result
            else:
                logger.warning("Unexpected step response format: %s", list(result.keys())[:5])
                self._is_terminal = True
                return

            # Poll until state is actionable (combat init may need frames)
            import time
            for _ in range(30):
                legal = self._state.get("legal_actions", [])
                if isinstance(legal, list) and any(
                    isinstance(a, dict) and a.get("is_enabled") is not False for a in legal
                ):
                    break
                st = (self._state.get("state_type") or "").lower()
                if st == "game_over" or self._state.get("terminal") or st not in _COMBAT_ACTIVE_STATES:
                    break
                time.sleep(0.005)
                self._state = self._pipe.call("state")
        except Exception as e:
            logger.warning("Pipe step failed: %s", e)
            self._is_terminal = True
            return

        self._is_terminal, self._player_won = _check_terminal(self._state)

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @property
    def player_won(self) -> bool:
        return self._player_won

    def get_state_dict(self) -> dict[str, Any]:
        return self._state

    def cleanup_and_restore(self) -> dict[str, Any] | None:
        """Restore server to root state, delete snapshots, return restored state.

        After MCTS search (N simulations), the server is at whatever leaf
        state the last simulation reached. This restores to the root snapshot,
        executes via pipe to get the correct state, then deletes snapshots.

        Returns the restored state dict, or None if restore failed.
        """
        if not self._snapshot_registry:
            return None
        root_id = self._snapshot_registry[0]
        restored = None
        try:
            resp = self._pipe.call("load_state", {"state_id": root_id})
            if isinstance(resp, dict) and "state_type" in resp:
                restored = resp
            else:
                restored = self._pipe.call("state")
        except Exception:
            pass
        for sid in self._snapshot_registry:
            try:
                self._pipe.call("delete_state", {"state_id": sid})
            except Exception:
                pass
        self._snapshot_registry.clear()
        return restored

    def cleanup(self) -> None:
        """Delete all snapshots. Does NOT restore server state.

        Use cleanup_and_restore() if you need the server back at root.
        """
        if not self._snapshot_registry:
            return
        for sid in self._snapshot_registry:
            try:
                self._pipe.call("delete_state", {"state_id": sid})
            except Exception:
                pass
        self._snapshot_registry.clear()

    def clear_all_snapshots(self) -> None:
        """Bulk-delete ALL snapshots on the server (nuclear option)."""
        try:
            self._pipe.call("delete_state", {"clear_all": True})
        except Exception:
            pass
        self._snapshot_registry.clear()

    @property
    def snapshot_count(self) -> int:
        """Number of snapshots tracked by this model tree."""
        return len(self._snapshot_registry)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
