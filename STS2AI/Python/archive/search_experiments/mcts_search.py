"""MCTS (Monte Carlo Tree Search) for STS2.

Implements UCT-based tree search using the simulator's save/load state API.
Each tree node corresponds to a game state snapshot. The search:
1. SELECT: traverse tree using UCB1 to find a promising leaf
2. EXPAND: try an untried action from the leaf
3. ROLLOUT: play random actions until terminal
4. BACKPROPAGATE: update visit counts and values up the tree

Usage:
    # With HTTP backend (slower, ~30ms/step):
    python mcts_search.py --port 15527 --seed TEST1 --simulations 100

    # With pipe backend (faster, ~0.5ms/step):
    python mcts_search.py --port 15527 --pipe --seed TEST1 --simulations 100

    # Search from a specific game state (after N random steps):
    python mcts_search.py --port 15527 --seed TEST1 --warmup-steps 10 --simulations 200
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
try:
    from .simulator_api_error import SimulatorApiError
except ImportError:
    from simulator_api_error import SimulatorApiError


# ---------------------------------------------------------------------------
# MCTS Tree Node
# ---------------------------------------------------------------------------

@dataclass
class MctsNode:
    """A node in the MCTS search tree."""
    state_id: str | None       # saved state snapshot ID
    action: dict | None        # action that led to this node (None for root)
    parent: MctsNode | None = None
    children: list[MctsNode] = field(default_factory=list)
    untried_actions: list[dict] = field(default_factory=list)
    snapshot_supported: bool = True

    visits: int = 0
    total_value: float = 0.0
    state_type: str = ""
    is_terminal: bool = False

    @property
    def avg_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def ucb1(self, exploration: float = 1.41) -> float:
        """Upper Confidence Bound for tree selection."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.avg_value
        exploit = self.total_value / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


# ---------------------------------------------------------------------------
# MCTS Search
# ---------------------------------------------------------------------------

COMBAT_STATES = {"monster", "elite", "boss"}


def _pick_rollout_action(legal_actions: list[dict], rng: random.Random,
                          state_type: str) -> dict:
    """Heuristic action selection for rollouts (faster than uniform random)."""
    if not legal_actions:
        return {"action": "wait"}

    # Combat: prefer play_card > use_potion > end_turn
    if state_type in COMBAT_STATES:
        play = [a for a in legal_actions if (a.get("action") or "").lower() == "play_card"]
        if play:
            return rng.choice(play)
        end = [a for a in legal_actions if (a.get("action") or "").lower() == "end_turn"]
        if end:
            return end[0]

    # Card select: prefer confirm/cancel
    if state_type == "card_select":
        escape = [a for a in legal_actions
                  if (a.get("action") or "").lower() in ("confirm_selection", "cancel_selection")]
        if escape:
            return escape[0]

    return rng.choice(legal_actions)


def mcts_search(env, num_simulations: int = 100,
                max_rollout_steps: int = 200,
                exploration: float = 1.41,
                rng_seed: int = 42,
                verbose: bool = False) -> dict | None:
    """Run MCTS from the current game state.

    Args:
        env: MctsEnv or PipeBackedMctsEnv instance (must be at a non-terminal state)
        num_simulations: number of MCTS iterations
        max_rollout_steps: max steps per rollout
        exploration: UCB1 exploration constant
        rng_seed: seed for rollout randomness
        verbose: print per-iteration stats

    Returns:
        Best action dict, or None if no actions available
    """
    rng = random.Random(rng_seed)

    state = env.get_state()
    legal = _get_legal(state)

    if not legal:
        return None

    root_state_id = None
    root: MctsNode | None = None
    try:
        root_state_id = env.save()
    except Exception as exc:
        if _is_combat_snapshot_not_supported(exc):
            return _pick_rollout_action(
                legal,
                rng,
                (state.get("state_type") or "").lower(),
            )
        raise

    root = MctsNode(
        state_id=root_state_id,
        action=None,
        untried_actions=list(legal),
        state_type=(state.get("state_type") or "").lower(),
        is_terminal=bool(state.get("terminal")),
    )

    sim_times = []
    best_action = None
    search_error: Exception | None = None
    try:
        for sim_i in range(num_simulations):
            t0 = time.monotonic()

            node = root
            env.load(root.state_id)

            while node.is_fully_expanded and node.children and not node.is_terminal:
                replayable_children = [child for child in node.children if child.snapshot_supported]
                if not replayable_children:
                    break
                node = max(replayable_children, key=lambda c: c.ucb1(exploration))
                env.load(node.state_id)

            if node.untried_actions and not node.is_terminal:
                action = node.untried_actions.pop(0)
                state = env.step(action)
                child_legal = _get_legal(state)
                child = MctsNode(
                    state_id=None,
                    action=action,
                    parent=node,
                    untried_actions=[],
                    state_type=(state.get("state_type") or "").lower(),
                    is_terminal=bool(state.get("terminal")),
                    snapshot_supported=False,
                )
                try:
                    child.state_id = env.save()
                    child.untried_actions = list(child_legal)
                    child.snapshot_supported = True
                except Exception as exc:
                    if not _is_combat_snapshot_not_supported(exc):
                        raise
                node.children.append(child)
                node = child

            reward = _rollout(env, max_rollout_steps, rng)

            while node is not None:
                node.visits += 1
                node.total_value += reward
                node = node.parent

            elapsed = time.monotonic() - t0
            sim_times.append(elapsed)

            if verbose and (sim_i + 1) % 10 == 0:
                best = max(root.children, key=lambda c: c.visits) if root.children else None
                best_action_name = (best.action.get("action", "?") if best and best.action else "?")
                print(f"  Sim {sim_i+1:4d}: {elapsed*1000:.0f}ms, "
                      f"best={best_action_name}(v={best.avg_value:.2f}, n={best.visits})")

        if not root.children:
            return None

        best_child = max(root.children, key=lambda c: c.visits)
        best_action = best_child.action

        if verbose:
            print(f"\n  MCTS Result ({num_simulations} sims, "
                  f"avg {sum(sim_times)/len(sim_times)*1000:.0f}ms/sim):")
            for child in sorted(root.children, key=lambda c: -c.visits)[:5]:
                a = child.action.get("action", "?") if child.action else "?"
                idx = child.action.get("index", "") if child.action else ""
                suffix = "" if child.snapshot_supported else " (no snapshot)"
                print(f"    {a}[{idx}]: visits={child.visits}, "
                      f"avg_value={child.avg_value:.3f}{suffix}")

        return best_action
    except Exception as exc:
        search_error = exc
        raise
    finally:
        restore_error = None
        if root_state_id is not None:
            try:
                env.load(root_state_id)
            except Exception as exc:
                restore_error = exc
        if root is not None:
            _cleanup_tree(env, root)
        if restore_error is not None and search_error is None:
            raise restore_error


def _get_legal(state: dict) -> list[dict]:
    """Get legal actions, filtering out unsupported ones."""
    actions = state.get("legal_actions") or []
    return [a for a in actions
            if isinstance(a, dict) and a.get("is_enabled") is not False]


def _rollout(env, max_steps: int, rng: random.Random) -> float:
    """Random rollout from current state. Returns reward in [-1, 1]."""
    for _ in range(max_steps):
        state = env.get_state()
        if state.get("terminal") or (state.get("state_type") or "").lower() == "game_over":
            outcome = state.get("run_outcome") or ""
            if "victory" in outcome.lower() or outcome.lower() == "win":
                return 1.0
            return -1.0

        legal = _get_legal(state)
        if not legal:
            # Try wait during enemy turn
            try:
                env.step({"action": "wait"})
            except Exception:
                return 0.0
            continue

        st = (state.get("state_type") or "").lower()
        action = _pick_rollout_action(legal, rng, st)
        try:
            env.step(action)
        except Exception:
            return 0.0

    # Didn't reach terminal — use heuristic based on HP
    state = env.get_state()
    player = state.get("player") or {}
    hp = player.get("hp", player.get("current_hp", 0)) or 0
    max_hp = player.get("max_hp", 80) or 80
    run = state.get("run") or {}
    floor_val = run.get("floor", 0) or 0

    # Heuristic: floor progress + HP ratio
    return min(floor_val / 20.0, 0.5) + (hp / max_hp) * 0.3 - 0.3


def _cleanup_tree(env, root: MctsNode) -> None:
    """Delete all saved state snapshots in the tree using an iterative walk."""
    seen_state_ids: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if not node.state_id or node.state_id in seen_state_ids:
            continue
        seen_state_ids.add(node.state_id)
        try:
            env.delete(node.state_id)
        except Exception:
            pass


def _is_combat_snapshot_not_supported(exc: Exception) -> bool:
    if isinstance(exc, SimulatorApiError):
        return exc.error_code == "combat_snapshot_not_supported"
    message = str(exc).lower()
    return "combat_snapshot_not_supported" in message


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCTS search for STS2")
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--pipe", action="store_true", help="Use named pipe instead of HTTP")
    parser.add_argument("--seed", default="MCTS_TEST", help="Game seed")
    parser.add_argument("--character", default="IRONCLAD")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--max-rollout", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Random steps before starting MCTS")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num-decisions", type=int, default=5,
                        help="How many MCTS decisions to make before stopping")
    args = parser.parse_args()

    if args.pipe:
        try:
            from .pipe_client import PipeBackedMctsEnv
        except ImportError:
            from pipe_client import PipeBackedMctsEnv
        env = PipeBackedMctsEnv(port=args.port)
    else:
        try:
            from .mcts_env import MctsEnv
        except ImportError:
            from mcts_env import MctsEnv
        env = MctsEnv(base_url=f"http://127.0.0.1:{args.port}")

    print(f"=== MCTS Search ===")
    print(f"Seed: {args.seed}, Sims: {args.simulations}, Backend: {'pipe' if args.pipe else 'http'}")
    print()

    # Reset
    state = env.reset(character_id=args.character, seed=args.seed)
    print(f"Reset: state_type={state.get('state_type')}")

    # Warmup: play random steps
    rng = random.Random(42)
    for i in range(args.warmup_steps):
        if state.get("terminal"):
            break
        legal = _get_legal(state)
        if not legal:
            state = env.step({"action": "wait"})
            continue
        state = env.step(rng.choice(legal))
    if args.warmup_steps:
        run = state.get("run") or {}
        print(f"After {args.warmup_steps} warmup steps: "
              f"state_type={state.get('state_type')}, floor={run.get('floor')}")

    # MCTS loop
    for decision in range(args.num_decisions):
        state = env.get_state()
        if state.get("terminal"):
            print(f"\nGame over: {state.get('run_outcome')}")
            break

        legal = _get_legal(state)
        if not legal:
            env.step({"action": "wait"})
            continue

        st = (state.get("state_type") or "").lower()
        run = state.get("run") or {}
        print(f"\n--- Decision {decision+1} ---")
        print(f"State: {st}, Floor: {run.get('floor')}, "
              f"Legal: {len(legal)} actions")

        t0 = time.monotonic()
        best_action = mcts_search(
            env,
            num_simulations=args.simulations,
            max_rollout_steps=args.max_rollout,
            verbose=args.verbose,
        )
        elapsed = time.monotonic() - t0

        if best_action is None:
            print("No action found!")
            break

        action_str = best_action.get("action", "?")
        if "index" in best_action:
            action_str += f"[{best_action['index']}]"
        print(f"Best action: {action_str} (search took {elapsed:.1f}s)")

        # Execute the chosen action
        state = env.step(best_action)
        run = state.get("run") or {}
        player = state.get("player") or {}
        print(f"Result: state_type={state.get('state_type')}, "
              f"floor={run.get('floor')}, "
              f"hp={player.get('hp', player.get('current_hp'))}/{player.get('max_hp')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
