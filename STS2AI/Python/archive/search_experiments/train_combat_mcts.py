#!/usr/bin/env python3
"""Self-play training for combat MCTS neural network.

AlphaZero-style training loop:
1. Generate self-play data using MCTS + current NN
2. Store (state, MCTS policy, game outcome) tuples
3. Train NN to predict MCTS policy (cross-entropy) and outcome (MSE)
4. Repeat with updated NN

Usage:
    # Train with pipe backend (FAST — recommended, ~0.5ms/step):
    python train_combat_mcts.py --pipe --port 15527 --mcts-sims 200

    # Train with HTTP backend (SLOW — prototype, ~24ms/step):
    python train_combat_mcts.py --base-url http://127.0.0.1:15527 --mcts-sims 50

    # Resume from checkpoint:
    python train_combat_mcts.py --pipe --resume artifacts/combat_mcts/.../combat_best.pt
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import json
import logging
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import load_vocab, Vocab
from combat_nn import (
    CombatPolicyValueNetwork,
    CombatNNEvaluator,
    build_combat_features,
    build_combat_action_features,
    MAX_ACTIONS,
)
from mcts_core import (
    MCTSConfig,
    mcts_search,
    action_key,
)
from combat_mcts_agent import CombatMCTSAgent, HttpCombatForwardModel, PipeCombatForwardModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _safe_load_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None)
    }
    model.load_state_dict(filtered, strict=False)

# Graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested (signal %d), finishing current iteration...", signum)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    """One training example from self-play."""
    state_features: dict[str, np.ndarray]
    action_features: dict[str, np.ndarray]
    mcts_policy: np.ndarray   # (MAX_ACTIONS,) visit count distribution
    outcome: float            # +1 victory, -1 death


class ReplayBuffer:
    """Fixed-size replay buffer for self-play data."""

    def __init__(self, max_size: int = 50000):
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, example: TrainingExample) -> None:
        self.buffer.append(example)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)),
                                   replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------------

class HealthMonitor:
    """Track training health metrics and detect anomalies."""

    def __init__(self, max_consecutive_failures: int = 5,
                 episode_timeout_s: float = 300.0,
                 decision_timeout_s: float = 30.0,
                 max_snapshots_warn: int = 500):
        self.max_consecutive_failures = max_consecutive_failures
        self.episode_timeout_s = episode_timeout_s
        self.decision_timeout_s = decision_timeout_s
        self.max_snapshots_warn = max_snapshots_warn

        self.consecutive_failures = 0
        self.total_episodes = 0
        self.total_failures = 0
        self.total_decisions = 0
        self.slow_decisions = 0  # decisions exceeding 2x median
        self.decision_times: deque[float] = deque(maxlen=200)
        self.snapshot_high_water = 0

    def record_decision(self, elapsed_s: float, snapshot_count: int = 0):
        self.total_decisions += 1
        self.decision_times.append(elapsed_s)
        self.snapshot_high_water = max(self.snapshot_high_water, snapshot_count)

        if snapshot_count > self.max_snapshots_warn:
            logger.warning("Snapshot count high: %d (potential memory leak)", snapshot_count)

        # Detect slow decisions (>2x median)
        if len(self.decision_times) > 10:
            median = sorted(self.decision_times)[len(self.decision_times) // 2]
            if elapsed_s > max(median * 3, 5.0):
                self.slow_decisions += 1
                logger.warning("Slow MCTS decision: %.1fs (median %.1fs)", elapsed_s, median)

    def record_episode_success(self):
        self.total_episodes += 1
        self.consecutive_failures = 0

    def record_episode_failure(self, reason: str):
        self.total_episodes += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        logger.warning("Episode failed (%d consecutive): %s", self.consecutive_failures, reason)

    def should_halt(self) -> str | None:
        """Returns halt reason if training should stop, else None."""
        if self.consecutive_failures >= self.max_consecutive_failures:
            return f"Too many consecutive failures ({self.consecutive_failures})"
        return None

    def summary(self) -> dict:
        times = list(self.decision_times)
        return {
            "total_episodes": self.total_episodes,
            "total_failures": self.total_failures,
            "total_decisions": self.total_decisions,
            "slow_decisions": self.slow_decisions,
            "snapshot_high_water": self.snapshot_high_water,
            "decision_time_median_ms": sorted(times)[len(times) // 2] * 1000 if times else 0,
            "decision_time_p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000 if len(times) > 5 else 0,
        }


# ---------------------------------------------------------------------------
# Self-play episode collection (pipe backend)
# ---------------------------------------------------------------------------

COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}


def collect_combat_episode_pipe(
    agent: CombatMCTSAgent,
    pipe,
    vocab: Vocab,
    health: HealthMonitor,
    character_id: str = "IRONCLAD",
    ascension_level: int = 0,
    max_steps: int = 600,
    http_base_url: str = "http://127.0.0.1:15527",
) -> tuple[list[TrainingExample], dict]:
    """Play one full run collecting MCTS training data from combats.

    Architecture: HTTP API drives the episode (handles room transitions
    that need Godot main-thread frame processing), while the pipe is used
    only for MCTS tree search within combat (save/load/step are fast
    because combat actions are synchronous in pure-sim mode).

    Returns:
        examples: list of TrainingExample from combat steps
        stats: dict with episode statistics
    """
    from full_run_env import ApiBackedFullRunClient

    client = ApiBackedFullRunClient(base_url=http_base_url, poll_interval_s=0.005,
                                    request_timeout_s=90.0)
    episode_start = time.monotonic()
    stats: dict[str, Any] = {"floors": 0, "combats": 0, "steps": 0,
                              "mcts_decisions": 0, "outcome": None, "error": None}

    try:
        state = client.reset(character_id=character_id, ascension_level=ascension_level)
    except Exception as e:
        stats["error"] = f"reset: {e}"
        health.record_episode_failure(stats["error"])
        return [], stats

    examples: list[TrainingExample] = []
    combat_examples_pending: list[dict] = []
    in_combat = False
    stall_count = 0
    last_sig = ""

    for step_i in range(max_steps):
        elapsed = time.monotonic() - episode_start
        if elapsed > health.episode_timeout_s:
            stats["error"] = f"episode timeout ({elapsed:.0f}s)"
            logger.warning("Episode timed out after %.0fs at step %d", elapsed, step_i)
            break

        st = (state.get("state_type") or "").lower()

        # Stall detection
        sig = f"{st}:{len(state.get('legal_actions', []))}"
        if sig == last_sig:
            stall_count += 1
            if stall_count > 20:
                stats["error"] = f"stall detected ({stall_count} identical states)"
                logger.warning("Stall at step %d: %s", step_i, sig)
                break
        else:
            stall_count = 0
            last_sig = sig

        # Terminal
        if st == "game_over" or state.get("terminal"):
            go = state.get("game_over") or {}
            outcome = (go.get("run_outcome") or go.get("outcome") or "").lower()
            game_value = 1.0 if ("victory" in outcome or outcome == "win") else -1.0
            stats["outcome"] = "victory" if game_value > 0 else "death"
            for ex_data in combat_examples_pending:
                ex_data["outcome"] = game_value
                examples.append(TrainingExample(**ex_data))
            combat_examples_pending.clear()
            break

        run = state.get("run") or {}
        stats["floors"] = max(stats["floors"], int(run.get("floor", 0)))

        if st in COMBAT_SCREENS:
            if not in_combat:
                in_combat = True
                stats["combats"] += 1

            legal = state.get("legal_actions", [])
            legal = [a for a in legal if isinstance(a, dict) and a.get("is_enabled") is not False]
            if not legal:
                try:
                    state = client.act({"action": "wait"})
                except Exception:
                    break
                continue

            # Run MCTS via pipe (fast save/load for tree search)
            decision_start = time.monotonic()
            fm = None
            try:
                fm = PipeCombatForwardModel.from_current_state(pipe)
                action, root = agent.choose_action(fm)
                decision_elapsed = time.monotonic() - decision_start
                health.record_decision(decision_elapsed, fm.snapshot_count)
            except Exception as e:
                logger.warning("MCTS decision failed at step %d: %s", step_i, e)
                action = legal[0]
                root = None
                decision_elapsed = time.monotonic() - decision_start
            finally:
                if fm is not None:
                    try:
                        fm.cleanup()
                    except Exception:
                        pass

            # Store training data
            if root is not None:
                try:
                    sf = build_combat_features(state, vocab)
                    af = build_combat_action_features(state, legal, vocab)
                    # Skip examples with empty action_mask (would cause nan in loss)
                    if af["action_mask"].any():
                        _, mcts_policy = root.visit_distribution()
                        padded_policy = np.zeros(MAX_ACTIONS, dtype=np.float32)
                        padded_policy[:len(mcts_policy)] = mcts_policy
                        combat_examples_pending.append({
                            "state_features": sf,
                            "action_features": af,
                            "mcts_policy": padded_policy,
                            "outcome": 0.0,
                        })
                        stats["mcts_decisions"] += 1
                except Exception as e:
                    logger.warning("Feature extraction failed: %s", e)

            # Execute via HTTP (handles state transitions correctly)
            try:
                state = client.act(action)
            except Exception:
                # MCTS action may have stale target; retry without target fields
                try:
                    fallback = {k: v for k, v in action.items()
                                if k not in ("target_id", "slot", "target")}
                    state = client.act(fallback)
                except Exception:
                    # Last resort: just end turn
                    try:
                        state = client.act({"action": "end_turn"})
                    except Exception as e2:
                        stats["error"] = f"step: {e2}"
                        break
            stats["steps"] += 1

        else:
            in_combat = False
            if combat_examples_pending:
                player = state.get("player") or {}
                hp = float(player.get("hp", player.get("current_hp", 0)))
                max_hp = max(1, float(player.get("max_hp", 1)))
                combat_value = hp / max_hp
                for ex_data in combat_examples_pending:
                    ex_data["outcome"] = combat_value
                    examples.append(TrainingExample(**ex_data))
                combat_examples_pending.clear()

            # Non-combat: take first action via HTTP
            legal = state.get("legal_actions", [])
            legal = [a for a in legal if isinstance(a, dict) and a.get("is_enabled") is not False]
            action = legal[0] if legal else {"action": "proceed"}
            try:
                state = client.act(action)
            except Exception:
                break

    if combat_examples_pending:
        for ex_data in combat_examples_pending:
            ex_data["outcome"] = 0.0
            examples.append(TrainingExample(**ex_data))
        combat_examples_pending.clear()

    stats["episode_time_s"] = time.monotonic() - episode_start

    if stats["error"]:
        health.record_episode_failure(stats["error"])
    else:
        health.record_episode_success()

    return examples, stats


# ---------------------------------------------------------------------------
# Self-play episode collection (HTTP backend — original, slower)
# ---------------------------------------------------------------------------

def collect_combat_episode_http(
    agent: CombatMCTSAgent,
    base_url: str,
    vocab: Vocab,
    health: HealthMonitor,
    character_id: str = "IRONCLAD",
    ascension_level: int = 0,
) -> tuple[list[TrainingExample], dict]:
    """Play one full run via HTTP, collecting MCTS training data from combats."""
    from full_run_env import ApiBackedFullRunClient

    client = ApiBackedFullRunClient(base_url=base_url, poll_interval_s=0.005,
                                    request_timeout_s=90.0)
    stats: dict[str, Any] = {"floors": 0, "combats": 0, "steps": 0,
                              "mcts_decisions": 0, "outcome": None, "error": None}

    try:
        state = client.reset(character_id=character_id, ascension_level=ascension_level)
    except Exception as e:
        stats["error"] = f"reset: {e}"
        health.record_episode_failure(stats["error"])
        return [], stats

    examples: list[TrainingExample] = []
    combat_examples_pending: list[dict] = []
    in_combat = False
    episode_start = time.monotonic()

    for step in range(600):
        if time.monotonic() - episode_start > health.episode_timeout_s:
            stats["error"] = "episode timeout"
            break

        st = (state.get("state_type") or "").lower()

        if st == "game_over" or state.get("terminal"):
            go = state.get("game_over") or {}
            outcome = (go.get("run_outcome") or go.get("outcome") or "").lower()
            game_value = 1.0 if ("victory" in outcome or outcome == "win") else -1.0
            stats["outcome"] = "victory" if game_value > 0 else "death"
            for ex_data in combat_examples_pending:
                ex_data["outcome"] = game_value
                examples.append(TrainingExample(**ex_data))
            combat_examples_pending.clear()
            break

        run = state.get("run") or {}
        stats["floors"] = max(stats["floors"], int(run.get("floor", 0)))

        if st in COMBAT_SCREENS:
            if not in_combat:
                in_combat = True
                stats["combats"] += 1

            legal = state.get("legal_actions", [])
            if not legal:
                try:
                    state = client.act({"action": "wait"})
                except Exception:
                    break
                continue

            fm = HttpCombatForwardModel(state, client=client, base_url=base_url)
            try:
                action, root = agent.choose_action(fm)
            except Exception as e:
                logger.warning("MCTS decision failed: %s", e)
                action = legal[0]
                root = None

            if root is not None:
                try:
                    sf = build_combat_features(state, vocab)
                    af = build_combat_action_features(state, legal, vocab)
                    _, mcts_policy = root.visit_distribution()
                    padded_policy = np.zeros(MAX_ACTIONS, dtype=np.float32)
                    padded_policy[:len(mcts_policy)] = mcts_policy
                    combat_examples_pending.append({
                        "state_features": sf, "action_features": af,
                        "mcts_policy": padded_policy, "outcome": 0.0,
                    })
                    stats["mcts_decisions"] += 1
                except Exception:
                    pass

            try:
                state = client.act(action)
            except Exception:
                break
            stats["steps"] += 1

        else:
            in_combat = False
            if combat_examples_pending:
                player = state.get("player") or {}
                hp = float(player.get("hp", player.get("current_hp", 0)))
                max_hp = max(1, float(player.get("max_hp", 1)))
                for ex_data in combat_examples_pending:
                    ex_data["outcome"] = hp / max_hp
                    examples.append(TrainingExample(**ex_data))
                combat_examples_pending.clear()

            legal = state.get("legal_actions", [])
            action = legal[0] if legal else {"action": "proceed"}
            try:
                state = client.act(action)
            except Exception:
                break

    stats["episode_time_s"] = time.monotonic() - episode_start
    if stats["error"]:
        health.record_episode_failure(stats["error"])
    else:
        health.record_episode_success()
    return examples, stats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_step(
    network: CombatPolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[TrainingExample],
) -> dict[str, float]:
    """One training step on a batch of examples."""
    state_tensors = {}
    action_tensors = {}
    keys_s = batch[0].state_features.keys()
    keys_a = batch[0].action_features.keys()

    for k in keys_s:
        arrays = [ex.state_features[k] for ex in batch]
        arr = np.stack(arrays)
        if arr.dtype in (np.int64, np.int32):
            state_tensors[k] = torch.tensor(arr, dtype=torch.long)
        elif arr.dtype == bool:
            state_tensors[k] = torch.tensor(arr, dtype=torch.bool)
        else:
            state_tensors[k] = torch.tensor(arr, dtype=torch.float32)

    for k in keys_a:
        arrays = [ex.action_features[k] for ex in batch]
        arr = np.stack(arrays)
        if arr.dtype in (np.int64, np.int32):
            action_tensors[k] = torch.tensor(arr, dtype=torch.long)
        elif arr.dtype == bool:
            action_tensors[k] = torch.tensor(arr, dtype=torch.bool)
        else:
            action_tensors[k] = torch.tensor(arr, dtype=torch.float32)

    target_policy = torch.tensor(np.stack([ex.mcts_policy for ex in batch]),
                                  dtype=torch.float32)
    target_value = torch.tensor([ex.outcome for ex in batch], dtype=torch.float32)

    logits, value = network.forward(state_tensors, action_tensors)

    # Clamp logits to prevent -inf from all-zero masks propagating NaN
    logits_safe = logits.clamp(min=-30.0)
    log_probs = F.log_softmax(logits_safe, dim=-1)
    mask = action_tensors["action_mask"].float()
    # Zero out log_probs first so nan doesn't propagate via 0*nan
    policy_loss = -(target_policy * (log_probs * mask)).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, target_value)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "total_loss": loss.item(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Combat MCTS self-play training")
    parser.add_argument("--base-url", default=None,
                        help="Single HTTP URL (legacy). Use --num-envs + --start-port instead.")
    parser.add_argument("--pipe", action="store_true",
                        help="Use named pipe instead of HTTP for MCTS search (50x faster)")
    parser.add_argument("--port", type=int, default=None,
                        help="Single port (legacy). Use --start-port instead.")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel Godot instances (ports start-port .. start-port+N-1)")
    parser.add_argument("--start-port", type=int, default=15527,
                        help="First port for Godot instances")
    parser.add_argument("--character-id", default="IRONCLAD")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--episodes-per-iter", type=int, default=10)
    parser.add_argument("--mcts-sims", type=int, default=200,
                        help="MCTS simulations per move (200 for pipe, 50 for HTTP)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-steps-per-iter", type=int, default=20)
    parser.add_argument("--replay-buffer-size", type=int, default=50000)
    parser.add_argument("--output-dir", default="artifacts/combat_mcts")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--episode-timeout", type=float, default=60.0,
                        help="Max seconds per episode before timeout")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="Halt after this many consecutive episode failures")
    parser.add_argument("--repo-root", type=str, default=None,
                        help="Repo root for auto-starting Godot instances")
    parser.add_argument("--godot-exe", type=str, default=None,
                        help="Path to Godot executable for auto-start")
    args = parser.parse_args()

    # Build env list from --num-envs + --start-port (or legacy --base-url/--port)
    if args.base_url and args.num_envs == 1 and args.port is None:
        # Legacy single-env mode
        env_ports = [int(args.base_url.rsplit(":", 1)[-1])]
    elif args.port is not None and args.num_envs == 1:
        env_ports = [args.port]
    else:
        env_ports = [args.start_port + i for i in range(args.num_envs)]

    env_urls = [f"http://127.0.0.1:{p}" for p in env_ports]

    vocab = load_vocab()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backend_tag = "pipe" if args.pipe else "http"
    output_dir = Path(args.output_dir) / f"combat_{backend_tag}_{len(env_ports)}env_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Network
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        net_config = checkpoint.get("config", {})
        network = CombatPolicyValueNetwork(vocab=vocab,
                                            embed_dim=net_config.get("embed_dim", 32),
                                            hidden_dim=net_config.get("hidden_dim", 128))
        _safe_load_state_dict(network, checkpoint["model_state_dict"])
        start_iter = checkpoint.get("iteration", 0) + 1
    else:
        network = CombatPolicyValueNetwork(vocab=vocab)
        start_iter = 0

    logger.info("Combat NN params: %d", network.param_count())
    logger.info("Backend: %s | Sims/decision: %d | Envs: %d (%s)",
                backend_tag, args.mcts_sims, len(env_ports),
                [f":{p}" for p in env_ports])

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = ReplayBuffer(max_size=args.replay_buffer_size)

    mcts_config = MCTSConfig(
        num_simulations=args.mcts_sims,
        c_puct=1.5,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
    )

    agent = CombatMCTSAgent(network=network, vocab=vocab, config=mcts_config, training=True)
    health = HealthMonitor(
        max_consecutive_failures=args.max_consecutive_failures,
        episode_timeout_s=args.episode_timeout,
    )
    metrics_log = output_dir / "metrics.jsonl"

    # Connect pipes (one per env) if needed
    pipe_clients: dict[int, Any] = {}
    if args.pipe:
        from pipe_client import PipeClient
        for port in env_ports:
            pc = PipeClient(port=port)
            try:
                pc.connect(timeout_s=15.0)
                pipe_clients[port] = pc
                logger.info("Connected pipe sts2_mcts_%d", port)
            except Exception as e:
                logger.warning("Failed to connect pipe on port %d: %s", port, e)
        if not pipe_clients:
            logger.error("No pipe connections established!")
            return 1

    logger.info("Starting combat MCTS training from iter %d (output: %s)", start_iter, output_dir)

    def _collect_one_episode(env_idx: int) -> tuple[list[TrainingExample], dict]:
        """Collect one episode from env at env_idx."""
        port = env_ports[env_idx % len(env_ports)]
        url = env_urls[env_idx % len(env_urls)]
        pipe = pipe_clients.get(port)

        if args.pipe and pipe is not None:
            return collect_combat_episode_pipe(
                agent, pipe, vocab, health,
                character_id=args.character_id,
                http_base_url=url,
            )
        else:
            return collect_combat_episode_http(
                agent, url, vocab, health,
                character_id=args.character_id,
            )

    try:
        for iteration in range(start_iter, args.max_iterations):
            if _shutdown_requested:
                logger.info("Shutdown requested, saving and exiting...")
                break

            iter_start = time.monotonic()

            # --- Self-play (parallel across envs) ---
            network.eval()
            new_examples = 0
            total_combats = 0
            total_floors = 0
            victories = 0
            ep_times = []

            num_workers = min(len(env_ports), args.episodes_per_iter)
            episodes_remaining = args.episodes_per_iter

            if num_workers > 1:
                # Parallel collection
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    env_idx = 0
                    futures = {}
                    # Submit initial batch
                    for _ in range(min(num_workers, episodes_remaining)):
                        if _shutdown_requested:
                            break
                        futures[executor.submit(_collect_one_episode, env_idx)] = env_idx
                        env_idx += 1
                        episodes_remaining -= 1

                    while futures:
                        if _shutdown_requested:
                            break
                        for future in as_completed(futures, timeout=args.episode_timeout + 30):
                            fidx = futures.pop(future)
                            try:
                                examples, stats = future.result()
                            except Exception as e:
                                logger.warning("Episode on env %d failed: %s", fidx, e)
                                health.record_episode_failure(str(e))
                                examples, stats = [], {"error": str(e)}

                            for ex in examples:
                                replay.add(ex)
                            new_examples += len(examples)
                            total_combats += stats.get("combats", 0)
                            total_floors += stats.get("floors", 0)
                            ep_times.append(stats.get("episode_time_s", 0))
                            if stats.get("outcome") == "victory":
                                victories += 1

                            # Submit next episode if any remaining
                            if episodes_remaining > 0 and not _shutdown_requested:
                                futures[executor.submit(_collect_one_episode, env_idx)] = env_idx
                                env_idx += 1
                                episodes_remaining -= 1

                            break  # process one at a time to refill
            else:
                # Single env — serial collection
                for ep in range(args.episodes_per_iter):
                    if _shutdown_requested:
                        break
                    examples, stats = _collect_one_episode(ep)
                    for ex in examples:
                        replay.add(ex)
                    new_examples += len(examples)
                    total_combats += stats.get("combats", 0)
                    total_floors += stats.get("floors", 0)
                    ep_times.append(stats.get("episode_time_s", 0))
                    if stats.get("outcome") == "victory":
                        victories += 1

            # Check health
            halt_reason = health.should_halt()
            if halt_reason:
                logger.error("HALTING: %s", halt_reason)
                torch.save({
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": iteration,
                    "config": {"embed_dim": 32, "hidden_dim": 128},
                    "halt_reason": halt_reason,
                }, output_dir / "combat_emergency.pt")
                return 2

            # --- Training ---
            network.train()
            total_ploss = 0.0
            total_vloss = 0.0
            train_steps = 0

            if len(replay) >= args.batch_size:
                for _ts in range(args.train_steps_per_iter):
                    batch = replay.sample(args.batch_size)
                    metrics = train_step(network, optimizer, batch)
                    total_ploss += metrics["policy_loss"]
                    total_vloss += metrics["value_loss"]
                    train_steps += 1

            iter_time = time.monotonic() - iter_start
            avg_ploss = total_ploss / max(1, train_steps)
            avg_vloss = total_vloss / max(1, train_steps)
            avg_floors = total_floors / max(1, args.episodes_per_iter)
            avg_ep_time = sum(ep_times) / max(1, len(ep_times))

            entry = {
                "iteration": iteration,
                "new_examples": new_examples,
                "replay_size": len(replay),
                "combats": total_combats,
                "avg_floor": avg_floors,
                "victories": victories,
                "episodes": args.episodes_per_iter,
                "policy_loss": avg_ploss,
                "value_loss": avg_vloss,
                "iter_time_s": iter_time,
                "avg_episode_time_s": avg_ep_time,
                "backend": backend_tag,
                "mcts_sims": args.mcts_sims,
            }
            with open(metrics_log, "a") as f:
                f.write(json.dumps(entry) + "\n")

            logger.info("Iter %3d | floor %.1f | vic %d/%d | ex %d (buf %d) | "
                         "ploss %.4f vloss %.4f | ep %.1fs | iter %.1fs",
                         iteration, avg_floors, victories, args.episodes_per_iter,
                         new_examples, len(replay), avg_ploss, avg_vloss,
                         avg_ep_time, iter_time)

            # Save checkpoint
            if iteration % args.save_interval == 0:
                ckpt_path = output_dir / f"combat_{iteration:05d}.pt"
                torch.save({
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": iteration,
                    "config": {"embed_dim": 32, "hidden_dim": 128},
                }, ckpt_path)

    except Exception as e:
        logger.error("Unhandled exception: %s\n%s", e, traceback.format_exc())
        # Save crash checkpoint
        torch.save({
            "model_state_dict": network.state_dict(),
            "iteration": iteration if "iteration" in dir() else -1,
            "config": {"embed_dim": 32, "hidden_dim": 128},
            "crash": str(e),
        }, output_dir / "combat_crash.pt")
        raise
    finally:
        # Cleanup all pipe connections
        for port, pc in pipe_clients.items():
            try:
                pc.call("delete_state", {"clear_all": True})
            except Exception:
                pass
            pc.close()

    # Save final
    final_path = output_dir / "combat_final.pt"
    torch.save({
        "model_state_dict": network.state_dict(),
        "iteration": args.max_iterations - 1,
        "config": {"embed_dim": 32, "hidden_dim": 128},
    }, final_path)

    # Log health summary
    hs = health.summary()
    logger.info("Training complete. Health: %s", json.dumps(hs))
    (output_dir / "health_summary.json").write_text(json.dumps(hs, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
