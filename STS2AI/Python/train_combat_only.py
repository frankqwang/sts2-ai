from __future__ import annotations

import _path_init  # noqa: F401

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from network.combat_network import CombatPolicyValueNetwork, build_combat_action_features, build_combat_features
from core.checkpoint_compat import get_combat_model_state
from core.rl_reward_shaping import combat_local_tactical_reward, combat_step_reward
from core.vocab import Vocab, load_vocab
from ipc.combat_training_env import PipeBackedCombatTrainingClient, sample_weighted_room_type
from ipc.headless_sim_runner import DEFAULT_DLL_PATH, DEFAULT_REPO_ROOT
from ipc.simulator_api_error import SimulatorApiError
from train_hybrid import (
    CombatPPOTrainer,
    CombatRolloutBuffer,
    _combat_action_label_zh,
    _combat_enemy_intent_summary,
    _combat_enemy_intent_summary_zh,
    _combat_hand_summary,
    _combat_hand_summary_zh,
    _combat_result_summary_zh,
    _combat_step_structured_summary,
    _combat_target_label_zh,
    _topk_action_summary,
    _topk_action_summary_zh,
)


logger = logging.getLogger(__name__)

_DEFAULT_BUILDS_PATH = Path(__file__).resolve().parents[1] / "Artifacts" / "skada" / "human_victory_builds_2026-04-14.json"
_VALID_ROOM_TYPES = {"monster", "elite", "boss"}


def _parse_room_types(raw: str | None) -> set[str]:
    if not raw:
        return set(_VALID_ROOM_TYPES)
    return {
        token.strip().lower()
        for token in str(raw).split(",")
        if token.strip().lower() in _VALID_ROOM_TYPES
    }


def _should_train_combat(
    floor: int,
    room_type: str,
    min_floor: int,
    allowed_room_types: set[str],
) -> bool:
    return int(floor) >= int(min_floor) and str(room_type).strip().lower() in allowed_room_types


def _load_seeds_from_file(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        for key in ("benchmark", "smoke"):
            value = payload.get(key)
            if not isinstance(value, list):
                continue
            seeds = [str(item.get("seed") if isinstance(item, dict) else item) for item in value]
            seeds = [seed for seed in seeds if seed.strip()]
            if seeds:
                return seeds
    raise ValueError(f"Unsupported seed file format: {path}")


@dataclass
class CyclicSeedSource:
    seeds: list[str]
    _index: int = 0

    def __post_init__(self) -> None:
        self.seeds = [str(seed) for seed in self.seeds if str(seed).strip()]
        if not self.seeds:
            raise ValueError("CyclicSeedSource requires at least one seed.")

    def next_seed(self) -> str:
        seed = self.seeds[self._index % len(self.seeds)]
        self._index += 1
        return seed


@dataclass(slots=True)
class BuildEntry:
    name: str
    notes: str
    source: dict[str, Any]
    build: dict[str, Any]

    @property
    def character_id(self) -> str:
        return str(self.source.get("character") or "IRONCLAD").strip().upper()

    @property
    def ascension_level(self) -> int:
        return int(self.source.get("ascension") or 0)


def _load_build_entries(path: Path) -> list[BuildEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Build file must be a JSON array: {path}")
    entries: list[BuildEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        build = item.get("build")
        if not isinstance(build, dict):
            continue
        entries.append(
            BuildEntry(
                name=str(item.get("name") or f"build_{len(entries)}"),
                notes=str(item.get("notes") or ""),
                source=dict(item.get("source") or {}),
                build=build,
            )
        )
    if not entries:
        raise ValueError(f"No usable build entries in {path}")
    return entries


def _build_encounter_pools(catalog: dict[str, Any], allowed_room_types: set[str]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {room_type: [] for room_type in _VALID_ROOM_TYPES}
    for item in catalog.get("encounters") or []:
        if not isinstance(item, dict):
            continue
        encounter_id = str(item.get("encounter_id") or "").strip().upper()
        room_type = str(item.get("room_type") or "").strip().lower()
        if encounter_id and room_type in allowed_room_types and room_type in pools:
            pools[room_type].append(encounter_id)
    return {room_type: sorted(values) for room_type, values in pools.items() if values}


def _load_or_init_network(
    *,
    checkpoint_path: Path | None,
    vocab: Vocab,
    device: torch.device,
    embed_dim: int,
    hidden_dim: int,
) -> tuple[CombatPolicyValueNetwork, dict[str, Any]]:
    metadata: dict[str, Any] = {"loaded_from": None}
    if checkpoint_path is None:
        network = CombatPolicyValueNetwork(vocab=vocab, embed_dim=embed_dim, hidden_dim=hidden_dim)
        return network.to(device), metadata

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    combat_state = get_combat_model_state(checkpoint, allow_standalone=True)
    if not isinstance(combat_state, dict):
        raise ValueError(f"Checkpoint has no combat state dict: {checkpoint_path}")

    embed_weight = combat_state.get("entity_emb.card_embed.weight")
    action_proj = combat_state.get("action_proj.weight")
    inferred_embed = int(embed_weight.shape[1]) if isinstance(embed_weight, torch.Tensor) and embed_weight.ndim == 2 else embed_dim
    inferred_hidden = int(action_proj.shape[0]) if isinstance(action_proj, torch.Tensor) and action_proj.ndim == 2 else hidden_dim

    network = CombatPolicyValueNetwork(vocab=vocab, embed_dim=inferred_embed, hidden_dim=inferred_hidden)
    current = network.state_dict()
    filtered: dict[str, Any] = {}
    for key, value in combat_state.items():
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None):
            filtered[key] = value
    network.load_state_dict(filtered, strict=False)
    metadata["loaded_from"] = str(checkpoint_path)
    metadata["checkpoint_keys"] = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    return network.to(device), metadata


def _np_features_to_tensors(features: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in features.items():
        tensor = torch.tensor(value).unsqueeze(0)
        if value.dtype in (np.int64, np.int32):
            tensor = tensor.long()
        elif value.dtype == bool:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        result[key] = tensor.to(device)
    return result


def _estimate_boss_damage_ratio(state: dict[str, Any]) -> float:
    enemies = ((state.get("battle") or {}).get("enemies") or state.get("enemies") or [])
    total_hp = 0.0
    total_max_hp = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        total_hp += max(0.0, float(enemy.get("hp", enemy.get("current_hp", 0)) or 0.0))
        total_max_hp += max(1.0, float(enemy.get("max_hp", 1) or 1.0))
    if total_max_hp <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (total_hp / total_max_hp)))


def _sample_action(
    network: CombatPolicyValueNetwork,
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    vocab: Vocab,
    device: torch.device,
    greedy: bool,
) -> tuple[int, float, float, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    state_features = build_combat_features(state, vocab)
    action_features = build_combat_action_features(state, legal_actions, vocab)
    state_tensors = _np_features_to_tensors(state_features, device)
    action_tensors = _np_features_to_tensors(action_features, device)

    with torch.no_grad():
        logits, value_t = network(state_tensors, action_tensors)
    mask = action_tensors["action_mask"].float()
    masked_logits = logits + (1.0 - mask) * (-1e9)
    dist = torch.distributions.Categorical(logits=masked_logits)
    if greedy:
        action_idx = int(torch.argmax(masked_logits[0, : len(legal_actions)]).item())
    else:
        action_idx = int(dist.sample().item())
    log_prob = float(dist.log_prob(torch.tensor([action_idx], device=device)).item())
    value = float(value_t.squeeze(0).item())
    return action_idx, log_prob, value, state_features, action_features, masked_logits.squeeze(0).detach().float().cpu().numpy()


def _save_checkpoint(
    path: Path,
    *,
    network: CombatPolicyValueNetwork,
    trainer: CombatPPOTrainer,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _create_client(args: argparse.Namespace) -> PipeBackedCombatTrainingClient:
    return PipeBackedCombatTrainingClient(
        port=args.port,
        connect_timeout_s=args.connect_timeout_s,
        auto_launch=args.auto_launch,
        repo_root=args.repo_root,
        dll_path=args.headless_dll,
    )


def run_training(args: argparse.Namespace) -> Path:
    rng = random.Random(args.global_seed)
    np.random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    builds = _load_build_entries(args.builds_path)
    seed_source = CyclicSeedSource(_load_seeds_from_file(str(args.seed_file))) if args.seed_file else None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    vocab = load_vocab()
    network, checkpoint_meta = _load_or_init_network(
        checkpoint_path=args.checkpoint,
        vocab=vocab,
        device=device,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    trainer = CombatPPOTrainer(
        network=network,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
    )

    client = _create_client(args)
    metrics_path = output_dir / "metrics.jsonl"
    trace_path = output_dir / "combat_episode_traces.jsonl"
    try:
        catalog = client.combat_catalog()
        allowed_room_types = _parse_room_types(args.room_types)
        encounter_pools = _build_encounter_pools(catalog, allowed_room_types)
        missing = [room_type for room_type in allowed_room_types if room_type not in encounter_pools]
        if missing:
            raise ValueError(f"Combat catalog has no encounters for room types: {missing}")

        run_manifest = {
            "builds_path": str(args.builds_path),
            "build_names": [entry.name for entry in builds],
            "room_types": sorted(allowed_room_types),
            "room_weights": {
                "monster": args.monster_weight,
                "elite": args.elite_weight,
                "boss": args.boss_weight,
            },
            "checkpoint": checkpoint_meta,
            "encounter_pool_sizes": {key: len(value) for key, value in encounter_pools.items()},
            "combat_episode_trace_path": str(trace_path),
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        buffer = CombatRolloutBuffer()
        total_episodes = 0
        metrics_file = metrics_path.open("w", encoding="utf-8")
        trace_file = trace_path.open("w", encoding="utf-8")
        try:
            for iteration in range(1, args.max_iterations + 1):
                iter_stats: list[dict[str, Any]] = []
                iter_reset_failures = 0
                for _episode in range(args.episodes_per_iter):
                    build_entry = rng.choice(builds)
                    room_type = sample_weighted_room_type(
                        rng,
                        monster_weight=args.monster_weight if "monster" in encounter_pools else 0,
                        elite_weight=args.elite_weight if "elite" in encounter_pools else 0,
                        boss_weight=args.boss_weight if "boss" in encounter_pools else 0,
                    )
                    if room_type not in encounter_pools:
                        room_type = next(iter(encounter_pools.keys()))
                    encounter_id = rng.choice(encounter_pools[room_type])
                    episode_seed = seed_source.next_seed() if seed_source is not None else f"combat-train-{rng.getrandbits(48):012x}"
                    ascension_level = args.ascension_level if args.ascension_level is not None else build_entry.ascension_level
                    episode_trace: dict[str, Any] = {
                        "iteration": iteration,
                        "episode": total_episodes + 1,
                        "build_name": build_entry.name,
                        "build_notes": build_entry.notes,
                        "build_source": build_entry.source,
                        "room_type": room_type,
                        "encounter_id": encounter_id,
                        "seed": episode_seed,
                        "ascension_level": ascension_level,
                        "steps": [],
                    }
                    state: dict[str, Any] | None = None
                    last_reset_error: Exception | None = None
                    for reset_attempt in range(max(1, int(args.reset_retries))):
                        try:
                            state = client.reset(
                                character_id=build_entry.character_id,
                                encounter_id=encounter_id,
                                ascension_level=ascension_level,
                                seed=episode_seed,
                                build=build_entry.build,
                            )
                            last_reset_error = None
                            break
                        except Exception as exc:
                            last_reset_error = exc
                            iter_reset_failures += 1
                            logger.warning(
                                "combat reset failed iter=%d episode=%d attempt=%d/%d room=%s encounter=%s build=%s error=%s",
                                iteration,
                                total_episodes + 1,
                                reset_attempt + 1,
                                args.reset_retries,
                                room_type,
                                encounter_id,
                                build_entry.name,
                                exc,
                            )
                            try:
                                client.close()
                            except Exception:
                                pass
                            if reset_attempt + 1 < int(args.reset_retries):
                                client = _create_client(args)
                    if state is None:
                        total_episodes += 1
                        iter_stats.append(
                            {
                                "iteration": iteration,
                                "episode": total_episodes,
                                "build_name": build_entry.name,
                                "room_type": room_type,
                                "encounter_id": encounter_id,
                                "seed": episode_seed,
                                "outcome": "reset_failed",
                                "steps": 0,
                                "invalid_actions": 0,
                                "reward": 0.0,
                                "reset_error": str(last_reset_error) if last_reset_error is not None else None,
                            }
                        )
                        episode_trace["outcome"] = "reset_failed"
                        episode_trace["reset_error"] = str(last_reset_error) if last_reset_error is not None else None
                        trace_file.write(json.dumps(episode_trace, ensure_ascii=False) + "\n")
                        trace_file.flush()
                        client = _create_client(args)
                        continue

                    hp_at_combat_start = int((state.get("player") or {}).get("hp", 0))
                    episode_trace["combat_start_hp"] = hp_at_combat_start
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_invalid_actions = 0
                    outcome = "timeout"
                    while episode_steps < args.max_episode_steps:
                        legal_actions = [action for action in (state.get("legal_actions") or []) if isinstance(action, dict)]
                        if not legal_actions:
                            break
                        chosen_transition: tuple[int, float, float, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any], bool, dict[str, Any]] | None = None
                        candidate_actions = list(legal_actions)
                        while candidate_actions:
                            action_idx, log_prob, value, sf, af, masked_logits = _sample_action(
                                network,
                                state,
                                candidate_actions,
                                vocab=vocab,
                                device=device,
                                greedy=args.greedy,
                            )
                            action = candidate_actions[action_idx]
                            try:
                                next_state, _terminal_reward, done, info = client.step(action)
                                chosen_transition = (action_idx, log_prob, value, sf, af, action, next_state, done, info, masked_logits)
                                break
                            except SimulatorApiError:
                                episode_invalid_actions += 1
                                candidate_actions = [
                                    candidate
                                    for candidate_index, candidate in enumerate(candidate_actions)
                                    if candidate_index != action_idx
                                ]
                        if chosen_transition is None:
                            outcome = "invalid_action_reject"
                            break

                        action_idx, log_prob, value, sf, af, action, next_state, done, info, masked_logits = chosen_transition
                        if done:
                            won = info.get("run_outcome") == "victory"
                            boss_damage_ratio = _estimate_boss_damage_ratio(next_state) if room_type == "boss" and not won else 0.0
                            reward = combat_step_reward(
                                state,
                                next_state,
                                combat_won=won,
                                hp_at_combat_start=hp_at_combat_start,
                                boss_damage_ratio=boss_damage_ratio,
                            )
                            outcome = "victory" if won else "defeat"
                        else:
                            reward = combat_step_reward(state, next_state)
                        reward += combat_local_tactical_reward(state, action, legal_actions)
                        episode_trace["steps"].append(
                            {
                                "step": episode_steps + 1,
                                "state_type": state.get("state_type"),
                                "hand": _combat_hand_summary(state),
                                "hand_zh": _combat_hand_summary_zh(state),
                                "intent": _combat_enemy_intent_summary(state),
                                "intent_zh": _combat_enemy_intent_summary_zh(state),
                                "action": action,
                                "action_label_zh": _combat_action_label_zh(action, state),
                                "action_target_zh": _combat_target_label_zh(action, state),
                                "topk": _topk_action_summary(candidate_actions, masked_logits, k=5),
                                "topk_zh": _topk_action_summary_zh(candidate_actions, masked_logits, k=5),
                                "log_prob": round(float(log_prob), 6),
                                "value": round(float(value), 6),
                                "reward": round(float(reward), 6),
                                "done": bool(done),
                                "result_zh": _combat_result_summary_zh(state, next_state, action),
                                "result_structured": _combat_step_structured_summary(state, next_state, action),
                            }
                        )
                        buffer.add(
                            sf=sf,
                            af=af,
                            action_idx=action_idx,
                            log_prob=log_prob,
                            reward=reward,
                            value=value,
                            done=done,
                            screen_type=room_type,
                            sample_weight=1.25 if room_type == "boss" else 1.1 if room_type == "elite" else 1.0,
                        )
                        episode_reward += reward
                        episode_steps += 1
                        state = next_state
                        if done:
                            break

                    total_episodes += 1
                    episode_trace["episode"] = total_episodes
                    episode_trace["outcome"] = outcome
                    episode_trace["steps_taken"] = episode_steps
                    episode_trace["invalid_actions"] = episode_invalid_actions
                    episode_trace["episode_reward"] = round(float(episode_reward), 6)
                    episode_trace["final_state_type"] = state.get("state_type") if isinstance(state, dict) else None
                    episode_trace["run_outcome"] = state.get("run_outcome") if isinstance(state, dict) else None
                    trace_file.write(json.dumps(episode_trace, ensure_ascii=False) + "\n")
                    trace_file.flush()
                    iter_stats.append(
                        {
                            "iteration": iteration,
                            "episode": total_episodes,
                            "build_name": build_entry.name,
                            "room_type": room_type,
                            "encounter_id": encounter_id,
                            "seed": episode_seed,
                            "outcome": outcome,
                            "steps": episode_steps,
                            "invalid_actions": episode_invalid_actions,
                            "reward": round(float(episode_reward), 4),
                        }
                    )

                update_metrics = {"combat_ppo_ploss": 0.0, "combat_ppo_vloss": 0.0, "combat_entropy": 0.0}
                if len(buffer) >= args.update_batch_size:
                    update_metrics = trainer.update(buffer)
                    buffer.clear()

                wins = sum(1 for item in iter_stats if item["outcome"] == "victory")
                room_breakdown: dict[str, dict[str, int]] = {}
                for item in iter_stats:
                    bucket = room_breakdown.setdefault(item["room_type"], {"episodes": 0, "wins": 0})
                    bucket["episodes"] += 1
                    bucket["wins"] += int(item["outcome"] == "victory")

                record = {
                    "iteration": iteration,
                    "episodes": len(iter_stats),
                    "wins": wins,
                    "win_rate": round(wins / max(1, len(iter_stats)), 4),
                    "avg_steps": round(sum(item["steps"] for item in iter_stats) / max(1, len(iter_stats)), 2),
                    "avg_reward": round(sum(item["reward"] for item in iter_stats) / max(1, len(iter_stats)), 4),
                    "invalid_actions": sum(int(item.get("invalid_actions", 0)) for item in iter_stats),
                    "reset_failures": iter_reset_failures,
                    "buffer_size": len(buffer),
                    "room_breakdown": room_breakdown,
                    "update": {key: round(float(value), 6) for key, value in update_metrics.items()},
                }
                metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_file.flush()
                logger.info(
                    "iter=%d win_rate=%.3f episodes=%d buffer=%d cppo_ploss=%.4f cppo_vloss=%.4f",
                    iteration,
                    record["win_rate"],
                    record["episodes"],
                    record["buffer_size"],
                    update_metrics.get("combat_ppo_ploss", 0.0),
                    update_metrics.get("combat_ppo_vloss", 0.0),
                )

                if args.save_every > 0 and iteration % args.save_every == 0:
                    _save_checkpoint(
                        output_dir / f"combat_only_iter_{iteration:04d}.pt",
                        network=network,
                        trainer=trainer,
                        metadata={"iteration": iteration, "episodes": total_episodes},
                    )
        finally:
            metrics_file.close()
            trace_file.close()
    finally:
        client.close()

    _save_checkpoint(
        output_dir / "combat_only_last.pt",
        network=network,
        trainer=trainer,
        metadata={"iterations": args.max_iterations, "global_seed": args.global_seed},
    )
    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combat-only simulator training with build pools and weighted encounter curriculum.")
    parser.add_argument("--builds-path", type=Path, default=_DEFAULT_BUILDS_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional combat or hybrid checkpoint to warm-start from.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to Artifacts/combat_training/<timestamp>.")
    parser.add_argument("--room-types", type=str, default="monster,elite,boss")
    parser.add_argument("--monster-weight", type=int, default=7)
    parser.add_argument("--elite-weight", type=int, default=2)
    parser.add_argument("--boss-weight", type=int, default=1)
    parser.add_argument("--episodes-per-iter", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--update-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.05)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--target-kl", type=float, default=0.0)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--global-seed", type=int, default=1337)
    parser.add_argument("--seed-file", type=Path, default=None, help="Optional seed list/manifest; cycles deterministically.")
    parser.add_argument("--ascension-level", type=int, default=None, help="Override the ascension in build metadata.")
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--connect-timeout-s", type=float, default=15.0)
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_DLL_PATH)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--reset-retries", type=int, default=3)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.builds_path.exists():
        raise FileNotFoundError(f"Build pool not found: {args.builds_path}")
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(__file__).resolve().parents[1] / "Artifacts" / "combat_training" / f"sim_build_curriculum_{timestamp}"
    output_dir = run_training(args)
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
