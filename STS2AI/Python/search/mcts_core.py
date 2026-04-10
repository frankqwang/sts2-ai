"""MCTS core algorithm for STS2 combat.

Implements Monte Carlo Tree Search with:
- PUCT selection (AlphaZero-style)
- Neural network policy prior + value estimation
- Determinization for hidden information (draw pile order)
- Dirichlet noise at root for exploration during training

The forward model is pluggable — accepts any object implementing
the CombatForwardModel protocol.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Forward model protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CombatForwardModel(Protocol):
    """Interface that the C# combat simulator must implement."""

    def clone(self) -> CombatForwardModel:
        """Deep-copy the entire combat state for branching."""
        ...

    def get_legal_actions(self) -> list[dict[str, Any]]:
        """Return list of legal action dicts (same format as simulator API)."""
        ...

    def step(self, action: dict[str, Any]) -> None:
        """Execute one action, mutating state in-place."""
        ...

    @property
    def is_terminal(self) -> bool:
        """True if combat is over (victory or defeat)."""
        ...

    @property
    def player_won(self) -> bool:
        """True if player won the combat."""
        ...

    def get_state_dict(self) -> dict[str, Any]:
        """Return state dict for NN evaluation (same format as simulator state)."""
        ...


# ---------------------------------------------------------------------------
# NN evaluator protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class NNEvaluator(Protocol):
    """Interface for the neural network that guides MCTS."""

    def evaluate(self, state: dict[str, Any], legal_actions: list[dict[str, Any]],
                 ) -> tuple[np.ndarray, float]:
        """Evaluate a combat state.

        Returns:
            policy: np.ndarray of shape (num_legal_actions,) — prior probabilities
            value: float in [-1, 1] — estimated win probability
        """
        ...

    def evaluate_batch(self, states: list[dict[str, Any]],
                        legal_actions_list: list[list[dict[str, Any]]],
                        ) -> list[tuple[np.ndarray, float]]:
        """Batch evaluate multiple states in one forward pass."""
        ...


# ---------------------------------------------------------------------------
# Action key
# ---------------------------------------------------------------------------

def action_key(action: dict[str, Any]) -> tuple:
    """Convert action dict to a hashable key for tree indexing."""
    return (
        action.get("action", ""),
        action.get("index"),
        action.get("card_index"),
        action.get("target"),
        action.get("target_id"),
        action.get("slot"),
    )


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = (
        "parent", "action", "children",
        "visit_count", "total_value", "prior",
        "is_expanded", "is_terminal", "terminal_value",
    )

    def __init__(
        self,
        parent: MCTSNode | None = None,
        action: dict[str, Any] | None = None,
        prior: float = 0.0,
    ):
        self.parent = parent
        self.action = action
        self.children: dict[tuple, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior: float = prior
        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, parent_visits: int, c_puct: float) -> float:
        """PUCT selection score (higher = more promising to explore)."""
        exploitation = self.q_value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return exploitation + exploration

    def expand(self, legal_actions: list[dict], priors: np.ndarray) -> None:
        """Expand node with legal actions and their NN prior probabilities."""
        assert len(legal_actions) == len(priors)
        for action, prior in zip(legal_actions, priors):
            key = action_key(action)
            if key not in self.children:
                self.children[key] = MCTSNode(parent=self, action=action, prior=prior)
        self.is_expanded = True

    def select_child(self, c_puct: float) -> tuple[tuple, MCTSNode]:
        """Select child with highest PUCT score."""
        best_key = None
        best_score = -float("inf")
        parent_visits = self.visit_count

        for key, child in self.children.items():
            score = child.puct_score(parent_visits, c_puct)
            if score > best_score:
                best_score = score
                best_key = key

        assert best_key is not None
        return best_key, self.children[best_key]

    def best_action(self, temperature: float = 0.0) -> dict[str, Any]:
        """Select action based on visit counts.

        temperature=0: argmax (deterministic)
        temperature>0: sample proportional to visit_count^(1/T)
        """
        if not self.children:
            raise ValueError("No children to select from")

        if temperature < 1e-6:
            # Deterministic: highest visit count
            best = max(self.children.values(), key=lambda c: c.visit_count)
            return best.action
        else:
            # Stochastic: sample proportional to visit_count^(1/T)
            actions = []
            weights = []
            for child in self.children.values():
                actions.append(child.action)
                weights.append(child.visit_count ** (1.0 / temperature))
            total = sum(weights)
            probs = [w / total for w in weights]
            idx = random.choices(range(len(actions)), weights=probs, k=1)[0]
            return actions[idx]

    def visit_distribution(self) -> tuple[list[dict], np.ndarray]:
        """Return actions and their visit count distribution (training target)."""
        actions = []
        counts = []
        for child in self.children.values():
            actions.append(child.action)
            counts.append(child.visit_count)
        total = sum(counts)
        if total == 0:
            probs = np.ones(len(counts)) / max(1, len(counts))
        else:
            probs = np.array(counts, dtype=np.float32) / total
        return actions, probs


# ---------------------------------------------------------------------------
# Uniform random evaluator (for testing without NN)
# ---------------------------------------------------------------------------

class UniformEvaluator:
    """Dummy evaluator: uniform policy, zero value."""

    def evaluate(self, state, legal_actions):
        n = len(legal_actions)
        policy = np.ones(n, dtype=np.float32) / max(1, n)
        value = 0.0
        return policy, value

    def evaluate_batch(self, states, legal_actions_list):
        return [self.evaluate(s, la) for s, la in zip(states, legal_actions_list)]


# ---------------------------------------------------------------------------
# MCTS Search
# ---------------------------------------------------------------------------

@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""
    num_simulations: int = 200
    c_puct: float = 1.5
    temperature: float = 1.0        # action selection temperature (training)
    dirichlet_alpha: float = 0.3    # root noise
    dirichlet_fraction: float = 0.25
    num_determinizations: int = 1   # >1 for information set MCTS


_EVAL_BATCH_SIZE = 8  # leaves to accumulate before batch NN forward pass


def mcts_search(
    forward_model: CombatForwardModel,
    evaluator: NNEvaluator,
    config: MCTSConfig | None = None,
) -> MCTSNode:
    """Run MCTS search from current state.

    Uses batched leaf evaluation: accumulates up to _EVAL_BATCH_SIZE
    unexpanded leaves, then evaluates them in one NN forward pass.

    Args:
        forward_model: Combat simulator at current state (will be cloned internally)
        evaluator: Neural network for policy prior + value estimation
        config: MCTS hyperparameters

    Returns:
        root: MCTSNode with visit statistics.
              Use root.best_action(temperature) to select action.
              Use root.visit_distribution() for training targets.
    """
    if config is None:
        config = MCTSConfig()

    root = MCTSNode()

    # Get initial state and legal actions
    root_state = forward_model.get_state_dict()
    legal_actions = forward_model.get_legal_actions()

    if not legal_actions:
        root.is_terminal = True
        root.terminal_value = 1.0 if forward_model.player_won else -1.0
        return root

    # Evaluate root with NN
    policy, value = evaluator.evaluate(root_state, legal_actions)

    # Add Dirichlet noise at root for exploration
    if config.dirichlet_alpha > 0 and config.dirichlet_fraction > 0:
        noise = np.random.dirichlet([config.dirichlet_alpha] * len(legal_actions))
        policy = (1 - config.dirichlet_fraction) * policy + config.dirichlet_fraction * noise

    root.expand(legal_actions, policy)

    # Check if evaluator supports batching
    has_batch = hasattr(evaluator, "evaluate_batch") and callable(getattr(evaluator, "evaluate_batch", None))

    # Run simulations with batched leaf evaluation
    pending_leaves: list[tuple[MCTSNode, dict, list[dict]]] = []  # (node, state, legal_actions)
    pending_terminal: list[tuple[MCTSNode, float]] = []  # (node, value)
    sims_done = 0

    while sims_done < config.num_simulations:
        node = root
        sim_model = forward_model.clone()

        # 1. SELECTION: traverse tree using PUCT
        while node.is_expanded and not node.is_terminal:
            _key, child = node.select_child(config.c_puct)
            sim_model.step(child.action)
            node = child

        # 2. EXPANSION: classify leaf
        if node.is_terminal:
            pending_terminal.append((node, node.terminal_value))
        elif sim_model.is_terminal:
            leaf_value = 1.0 if sim_model.player_won else -1.0
            node.is_terminal = True
            node.terminal_value = leaf_value
            pending_terminal.append((node, leaf_value))
        else:
            sim_legal = sim_model.get_legal_actions()
            if not sim_legal:
                leaf_value = 1.0 if sim_model.player_won else -1.0
                node.is_terminal = True
                node.terminal_value = leaf_value
                pending_terminal.append((node, leaf_value))
            else:
                sim_state = sim_model.get_state_dict()
                if has_batch:
                    # Add virtual loss to prevent re-selecting same path
                    node.visit_count += 1
                    node.total_value -= 1.0  # pessimistic virtual loss
                    pending_leaves.append((node, sim_state, sim_legal))
                else:
                    # Fallback: evaluate one at a time
                    sim_policy, leaf_value = evaluator.evaluate(sim_state, sim_legal)
                    node.expand(sim_legal, sim_policy)
                    pending_terminal.append((node, leaf_value))

        sims_done += 1

        # 3. FLUSH: batch evaluate when enough pending or last sim
        should_flush = (
            pending_leaves
            and (len(pending_leaves) >= _EVAL_BATCH_SIZE or sims_done >= config.num_simulations)
        )
        if should_flush:
            states = [s for _, s, _ in pending_leaves]
            legals = [la for _, _, la in pending_leaves]
            results = evaluator.evaluate_batch(states, legals)

            for (leaf_node, _, sim_legal), (sim_policy, leaf_value) in zip(pending_leaves, results):
                # Remove virtual loss
                leaf_node.visit_count -= 1
                leaf_node.total_value += 1.0
                # Expand + backprop
                leaf_node.expand(sim_legal, sim_policy)
                pending_terminal.append((leaf_node, leaf_value))

            pending_leaves.clear()

        # 4. BACKPROPAGATION for all resolved leaves
        for leaf_node, leaf_value in pending_terminal:
            n = leaf_node
            while n is not None:
                n.visit_count += 1
                n.total_value += leaf_value
                n = n.parent
        pending_terminal.clear()

    return root


def mcts_search_with_determinization(
    forward_model: CombatForwardModel,
    evaluator: NNEvaluator,
    config: MCTSConfig | None = None,
) -> MCTSNode:
    """MCTS with multiple determinizations for hidden information.

    Runs K independent MCTS trees (one per draw pile determinization),
    then aggregates visit counts into a single root node.

    For STS2, this handles the unknown draw pile order.
    """
    if config is None:
        config = MCTSConfig()

    if config.num_determinizations <= 1:
        return mcts_search(forward_model, evaluator, config)

    # Run K independent searches
    aggregated_visits: dict[tuple, tuple[dict, int]] = {}

    per_det_sims = max(1, config.num_simulations // config.num_determinizations)
    det_config = MCTSConfig(
        num_simulations=per_det_sims,
        c_puct=config.c_puct,
        temperature=config.temperature,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_fraction=config.dirichlet_fraction,
        num_determinizations=1,
    )

    for _det in range(config.num_determinizations):
        det_model = forward_model.clone()
        # TODO: shuffle draw pile in det_model to create determinization
        # For now, just use the same state (works when draw pile is known)

        root = mcts_search(det_model, evaluator, det_config)

        # Aggregate visit counts
        for key, child in root.children.items():
            if key in aggregated_visits:
                action, prev_count = aggregated_visits[key]
                aggregated_visits[key] = (action, prev_count + child.visit_count)
            else:
                aggregated_visits[key] = (child.action, child.visit_count)

    # Build aggregated root
    agg_root = MCTSNode()
    legal_actions = []
    visit_counts = []
    for key, (action, count) in aggregated_visits.items():
        legal_actions.append(action)
        visit_counts.append(count)

    total = sum(visit_counts) or 1
    priors = np.array(visit_counts, dtype=np.float32) / total
    agg_root.expand(legal_actions, priors)

    # Set visit counts on children
    for (key, (action, count)), (_, child) in zip(
        aggregated_visits.items(), agg_root.children.items()
    ):
        child.visit_count = count

    agg_root.visit_count = total
    agg_root.is_expanded = True

    return agg_root
