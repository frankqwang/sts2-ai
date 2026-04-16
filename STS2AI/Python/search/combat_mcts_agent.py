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

import numpy as np
import torch

from search.mcts_core import (
    CombatForwardModel,
    MCTSConfig,
    MCTSNode,
    UniformEvaluator,
    action_key,
    mcts_search,
    mcts_search_with_determinization,
)
from network.combat_network import (
    CombatNNEvaluator,
    CombatPolicyValueNetwork,
)
from core.vocab import Vocab, load_vocab

logger = logging.getLogger(__name__)
_PIPE_STEP_FAIL_DIAG_PATH = Path.cwd() / "STS2AI" / "Artifacts" / "tmp" / "mcts_pipe_step_fail_latest.json"


def _combat_diag_snapshot(state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
    hand = player.get("hand") if isinstance(player.get("hand"), list) else []
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
    return {
        "state_type": state.get("state_type"),
        "terminal": state.get("terminal"),
        "run_floor": ((state.get("run") or {}).get("floor")),
        "player_hp": player.get("hp", player.get("current_hp")),
        "player_block": player.get("block"),
        "player_energy": player.get("energy", player.get("current_energy")),
        "hand": [
            {
                "index": card.get("index"),
                "id": card.get("id"),
                "label": card.get("label"),
                "cost": card.get("cost"),
            }
            for card in hand if isinstance(card, dict)
        ],
        "enemies": [
            {
                "id": enemy.get("id") or enemy.get("name"),
                "hp": enemy.get("hp", enemy.get("current_hp")),
                "block": enemy.get("block"),
            }
            for enemy in enemies if isinstance(enemy, dict)
        ],
        "legal_actions": [
            {
                "action": action.get("action"),
                "label": action.get("label"),
                "card_index": action.get("card_index"),
                "target_id": action.get("target_id"),
                "slot": action.get("slot"),
                "cost": action.get("cost"),
                "card_id": action.get("card_id") or action.get("id"),
            }
            for action in (state.get("legal_actions") or [])
            if isinstance(action, dict)
        ],
    }


def _write_pipe_step_fail_diag(*, error: Exception, action: dict[str, Any], clean: dict[str, Any], state: dict[str, Any], legal: list[dict[str, Any]]) -> None:
    try:
        _PIPE_STEP_FAIL_DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "error": str(error),
            "action": action,
            "clean_action": clean,
            "legal_before_step": legal,
            "state_snapshot": _combat_diag_snapshot(state),
        }
        _PIPE_STEP_FAIL_DIAG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _with_required_target(action: dict[str, Any], source: dict[str, Any], preferred_target: Any) -> dict[str, Any] | None:
    valid_targets = source.get("valid_target_ids") if isinstance(source.get("valid_target_ids"), list) else []
    requires_target = bool(source.get("requires_target"))
    if not requires_target:
        return dict(action)
    if preferred_target is not None and preferred_target in valid_targets:
        patched = dict(action)
        patched["target_id"] = preferred_target
        return patched
    if len(valid_targets) == 1:
        patched = dict(action)
        patched["target_id"] = valid_targets[0]
        return patched
    return None


def _safe_load_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None)
    }
    model.load_state_dict(filtered, strict=False)


class _CSharpMCTSChild:
    def __init__(self, *, action: dict[str, Any], visit_count: int, prior: float, q_value: float):
        self.action = action
        self.visit_count = int(visit_count)
        self.prior = float(prior)
        self._q_value = float(q_value)
        self.total_value = self._q_value * self.visit_count

    @property
    def q_value(self) -> float:
        return self._q_value


class CSharpMCTSRoot:
    def __init__(
        self,
        *,
        legal_actions: list[dict[str, Any]],
        action_index: int,
        visit_counts: list[int],
        visit_probs: list[float],
        q_values: list[float],
        priors: list[float],
        root_value: float,
    ):
        self._legal_actions = [dict(action) for action in legal_actions]
        self._action_index = int(action_index)
        self._visit_probs = np.array(visit_probs, dtype=np.float32)
        self._children_ordered: list[_CSharpMCTSChild] = []
        self.children: dict[tuple, _CSharpMCTSChild] = {}
        for idx, action in enumerate(self._legal_actions):
            child = _CSharpMCTSChild(
                action=action,
                visit_count=visit_counts[idx] if idx < len(visit_counts) else 0,
                prior=priors[idx] if idx < len(priors) else 0.0,
                q_value=q_values[idx] if idx < len(q_values) else 0.0,
            )
            self._children_ordered.append(child)
            self.children[action_key(action)] = child
        self.visit_count = int(sum(child.visit_count for child in self._children_ordered))
        self.total_value = float(root_value) * self.visit_count
        self._root_value = float(root_value)

    @property
    def q_value(self) -> float:
        return self._root_value

    def visit_distribution(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        return [dict(action) for action in self._legal_actions], self._visit_probs.copy()

    def best_action(
        self,
        temperature: float = 0.0,
        *,
        mode: str = "visit",
        top_k: int = 3,
        q_weight: float = 0.35,
    ) -> dict[str, Any]:
        _ = temperature, mode, top_k, q_weight
        if not self._legal_actions:
            raise ValueError("No children to select from")
        idx = min(max(self._action_index, 0), len(self._legal_actions) - 1)
        return dict(self._legal_actions[idx])


class CombatMCTSAgent:
    """MCTS-based combat agent with neural network guidance."""

    def __init__(
        self,
        network: CombatPolicyValueNetwork,
        vocab: Vocab,
        config: MCTSConfig | None = None,
        training: bool = False,
        device: torch.device | None = None,
        ppo_net: Any | None = None,
        *,
        backend: str = "python",
        use_continuation_value: bool = False,
    ):
        self.network = network
        self.vocab = vocab
        self.config = config or MCTSConfig()
        self.training = training
        self.backend = str(backend or "python").strip().lower()
        self.use_continuation_value = bool(use_continuation_value)
        self.evaluator = CombatNNEvaluator(
            network,
            vocab,
            device=device,
            use_continuation_value=use_continuation_value,
            ppo_net=ppo_net,
        )

    def choose_action(
        self,
        forward_model: CombatForwardModel,
    ) -> tuple[dict[str, Any], MCTSNode | CSharpMCTSRoot]:
        """Run MCTS and choose an action.

        Args:
            forward_model: Combat simulator at current decision point.

        Returns:
            action: chosen action dict
            root: MCTS root node (for extracting training targets)
        """
        if self.backend == "csharp":
            root = self._choose_action_csharp(forward_model)
            return root.best_action(), root

        if self.config.num_determinizations > 1:
            root = mcts_search_with_determinization(
                forward_model, self.evaluator, self.config)
        else:
            root = mcts_search(forward_model, self.evaluator, self.config)

        temperature = self.config.temperature if self.training else 0.0
        action = root.best_action(
            temperature=temperature,
            mode=getattr(self.config, "final_action_mode", "visit"),
            top_k=getattr(self.config, "final_action_top_k", 3),
            q_weight=getattr(self.config, "final_action_q_weight", 0.35),
        )

        return action, root

    def _choose_action_csharp(self, forward_model: CombatForwardModel) -> CSharpMCTSRoot:
        if not isinstance(forward_model, PipeCombatForwardModel):
            raise TypeError("C# combat MCTS backend requires PipeCombatForwardModel.")
        pipe = forward_model._pipe
        params = {
            "num_simulations": int(getattr(self.config, "num_simulations", 0)),
            "c_puct": float(getattr(self.config, "c_puct", 1.5)),
            "dirichlet_alpha": float(getattr(self.config, "dirichlet_alpha", 0.0)),
            "dirichlet_fraction": float(getattr(self.config, "dirichlet_fraction", 0.0)),
            "max_step_budget": int(getattr(forward_model, "_max_step_budget", 200)),
            "final_action_mode": str(getattr(self.config, "final_action_mode", "visit")),
            "final_action_top_k": int(getattr(self.config, "final_action_top_k", 3)),
            "final_action_q_weight": float(getattr(self.config, "final_action_q_weight", 0.35)),
            "use_continuation_value": bool(self.use_continuation_value),
        }
        result = pipe.call("search_combat_mcts", params)
        legal_actions = forward_model.get_legal_actions()
        if not isinstance(result, dict):
            raise RuntimeError("search_combat_mcts did not return a payload dict.")
        return CSharpMCTSRoot(
            legal_actions=legal_actions,
            action_index=int(result.get("action_index", 0)),
            visit_counts=list(result.get("visit_counts") or []),
            visit_probs=list(result.get("visit_probs") or []),
            q_values=list(result.get("q_values") or []),
            priors=list(result.get("priors") or []),
            root_value=float(result.get("root_value", 0.0)),
        )

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

# Screen types where combat is still active for tree search purposes.
# Startup pending is still inside the combat lifecycle; post-end pending is not.
_COMBAT_ACTIVE_STATES = {
    "combat",
    "monster",
    "elite",
    "boss",
    "hand_select",
    "card_select",
    "combat_pending",
    "combat_start_pending",
}


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
                      server_legal: list[dict[str, Any]]) -> dict[str, Any] | None:
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
    card_id = action.get("card_id") or action.get("id")
    cost = action.get("cost")

    _FIELDS = ("action", "index", "card_index", "hand_index",
               "slot", "target_id", "target", "col", "row", "value", "label", "card_id", "id", "cost")

    if not server_legal:
        return {k: v for k, v in action.items() if k in _FIELDS}

    def _match_play_like(la: dict[str, Any]) -> bool:
        if la.get("action") != act_type:
            return False
        if card_id and (la.get("card_id") or la.get("id")) != card_id:
            return False
        if label and la.get("label") != label:
            return False
        if cost is not None and la.get("cost") != cost:
            return False
        return True

    # Card-like actions are the main restore boundary hazard. After save/load,
    # hand order and transient legal payloads can shift even when the visible
    # card name stays the same, so we try to reconstruct the action from stable
    # card metadata first and only keep server-provided indices/targets.
    # 1. Exact match by card metadata when available.
    if card_id and cost is not None:
        for la in server_legal:
            if (
                la.get("action") == act_type
                and (la.get("card_id") or la.get("id")) == card_id
                and la.get("cost") == cost
                and la.get("target_id") == target
            ):
                candidate = {k: v for k, v in la.items() if k in _FIELDS}
                if act_type == "play_card" and candidate.get("target_id") is None:
                    return _with_required_target(candidate, la, target)
                return candidate

    if label and cost is not None:
        for la in server_legal:
            if (
                la.get("action") == act_type
                and la.get("label") == label
                and la.get("cost") == cost
                and la.get("target_id") == target
            ):
                candidate = {k: v for k, v in la.items() if k in _FIELDS}
                if act_type == "play_card" and candidate.get("target_id") is None:
                    return _with_required_target(candidate, la, target)
                return candidate

    # 2. Exact match by label + target
    if label:
        for la in server_legal:
            if (la.get("action") == act_type
                    and la.get("label") == label
                    and la.get("target_id") == target):
                candidate = {k: v for k, v in la.items() if k in _FIELDS}
                if act_type == "play_card" and candidate.get("target_id") is None:
                    return _with_required_target(candidate, la, target)
                return candidate

    # 2. Match by label only (server provides correct card_index/target)
    if label and cost is not None:
        for la in server_legal:
            if la.get("action") == act_type and la.get("label") == label and la.get("cost") == cost:
                candidate = {k: v for k, v in la.items() if k in _FIELDS}
                if act_type == "play_card" and candidate.get("target_id") is None:
                    return _with_required_target(candidate, la, target)
                return candidate

    if label:
        for la in server_legal:
            if la.get("action") == act_type and la.get("label") == label:
                candidate = {k: v for k, v in la.items() if k in _FIELDS}
                if act_type == "play_card" and candidate.get("target_id") is None:
                    return _with_required_target(candidate, la, target)
                return candidate

    # 3. Match by slot (for potions — slot is reliable across save/load)
    if act_type in {"play_card", "combat_select_card", "combat_confirm_selection"}:
        if target is not None:
            for la in server_legal:
                if _match_play_like(la) and la.get("target_id") == target:
                    return {k: v for k, v in la.items() if k in _FIELDS}
        for la in server_legal:
            if _match_play_like(la):
                candidate = {k: v for k, v in la.items() if k in _FIELDS}
                if candidate.get("target_id") is None:
                    return _with_required_target(candidate, la, target)
                return candidate
        return None

    if slot is not None:
        for la in server_legal:
            if la.get("action") == act_type and la.get("slot") == slot:
                return {k: v for k, v in la.items() if k in _FIELDS}

    # 4. Match on action type alone (end_turn, proceed, etc.)
    if act_type not in {"play_card", "use_potion", "combat_select_card", "combat_confirm_selection", "select_card_option"}:
        for la in server_legal:
            if la.get("action") == act_type:
                return {k: v for k, v in la.items() if k in _FIELDS}

    # Card-like actions should not fall back to a stale action payload if the
    # current state no longer exposes them as legal.
    return None


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
            battle = self._state.get("battle") if isinstance(self._state.get("battle"), dict) else {}
            player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
            hand = player.get("hand") if isinstance(player.get("hand"), list) else []
            enriched: list[dict[str, Any]] = []
            for action in legal:
                if not isinstance(action, dict) or action.get("is_enabled") is False:
                    continue
                action_copy = dict(action)
                # Reset card metadata every loop iteration. This used to leak the
                # previous card's id/cost/targetability into the next legal action,
                # which made MCTS reconcile stale actions and produced bogus
                # EnergyCostTooHigh / requires-target failures.
                card: dict[str, Any] | None = None
                card_index = action_copy.get("card_index")
                if isinstance(card_index, int) and 0 <= card_index < len(hand):
                    card = hand[card_index]
                if isinstance(card, dict):
                    card_id = card.get("id") or card.get("label")
                    if card_id:
                        action_copy.setdefault("card_id", card_id)
                        action_copy.setdefault("id", card_id)
                        action_copy.setdefault("label", card.get("label") or card_id)
                    if card.get("cost") is not None:
                        action_copy.setdefault("cost", card.get("cost"))
                    if card.get("requires_target") is not None:
                        action_copy.setdefault("requires_target", bool(card.get("requires_target")))
                    if isinstance(card.get("valid_target_ids"), list):
                        action_copy.setdefault("valid_target_ids", list(card.get("valid_target_ids")))
                enriched.append(action_copy)
            return enriched
        return []

    def _refresh_until_actionable(self, max_polls: int = 30, sleep_s: float = 0.005) -> None:
        import time
        for _ in range(max(1, int(max_polls))):
            legal = self._state.get("legal_actions", [])
            if isinstance(legal, list) and any(
                isinstance(a, dict) and a.get("is_enabled") is not False for a in legal
            ):
                return
            st = (self._state.get("state_type") or "").lower()
            if st == "game_over" or self._state.get("terminal") or st not in _COMBAT_ACTIVE_STATES:
                return
            time.sleep(sleep_s)
            self._state = self._pipe.call("state")

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
            self._refresh_until_actionable(max_polls=40, sleep_s=0.005)
            self._needs_restore = False

        # Reconcile action with server's current legal_actions.
        # After load_state, the server's legal_actions are the ground truth.
        # The MCTS tree's stored action may have stale target_id (e.g., DEFEND
        # stored with target_id from a different branch, or STRIKE missing target).
        legal_before_step = self.get_legal_actions()
        clean = _reconcile_action(action, legal_before_step)
        if clean is None:
            _write_pipe_step_fail_diag(
                error=RuntimeError("MCTS branch action is no longer legal in current server state"),
                action=dict(action) if isinstance(action, dict) else {},
                clean={},
                state=dict(self._state) if isinstance(self._state, dict) else {},
                legal=legal_before_step,
            )
            self._is_terminal = True
            self._player_won = False
            return

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
            self._refresh_until_actionable(max_polls=30, sleep_s=0.005)
        except Exception as e:
            _write_pipe_step_fail_diag(
                error=e,
                action=dict(action) if isinstance(action, dict) else {},
                clean=dict(clean) if isinstance(clean, dict) else {},
                state=dict(self._state) if isinstance(self._state, dict) else {},
                legal=legal_before_step,
            )
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
