"""networkV2 原生 critical-step combat teacher 管线。"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.raw.branch_schema import make_raw_branch_rollout_record
from data.raw.raw_dataset_writer import write_jsonl_records, write_raw_branch_exports
from env.full_run_env import BinaryBackedFullRunClient
from networkV2.s1_schema.encounter_vocab import encounter_to_index
from networkV2.s4_compiler.feature_compiler import CombatFeatureCompiler
from networkV2.s5_net.unified_net import UnifiedNet
from networkV2.s6_training.batch import TrainingSample, collate_training_samples


logger = logging.getLogger(__name__)

_COMBAT_STATE_TYPES = {"monster", "elite", "boss", "combat", "hand_select", "card_select"}


def _move_batched_to_device(batched, device: torch.device):
    for bank in batched.banks.values():
        bank.numeric = bank.numeric.to(device)
        bank.type_ids = bank.type_ids.to(device)
        bank.ts_ids = bank.ts_ids.to(device)
        bank.mask = bank.mask.to(device)
    for attr in (
        "action_indices", "old_log_probs", "advantages", "returns",
        "fight_win_targets", "run_win_targets", "hp_loss_targets", "survival_targets",
        "turn_damage_targets", "turn_block_targets", "leaf_targets", "transition_risk_targets",
        "resource_retention_targets", "boss_readiness_targets",
        "resource_health_targets", "deck_quality_targets",
        "future_dq_targets", "sample_weights",
    ):
        value = getattr(batched, attr, None)
        if value is not None:
            setattr(batched, attr, value.to(device))
    if batched.encounter_indices is not None:
        batched.encounter_indices = batched.encounter_indices.to(device)
    return batched


def _state_floor(state: dict[str, Any]) -> int:
    return int(((state.get("run") or {}).get("floor")) or 0)


def _player_hp_ratio(state: dict[str, Any]) -> float:
    player = (state.get("player") or (state.get("battle") or {}).get("player") or {})
    hp = float(player.get("hp") or player.get("current_hp") or 0.0)
    max_hp = max(float(player.get("max_hp") or 1.0), 1.0)
    return hp / max_hp


def _enemies_total_hp(state: dict[str, Any]) -> float:
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    return float(sum(float(enemy.get("hp", 0.0) or 0.0) for enemy in enemies if enemy.get("is_alive", True)))


def _state_room_type(state: dict[str, Any], fallback: str = "monster") -> str:
    room_type = str(state.get("state_type") or fallback or "monster").strip().lower()
    if room_type not in _COMBAT_STATE_TYPES:
        return fallback
    if room_type == "combat":
        return fallback
    return room_type


def _terminal_summary(state: dict[str, Any], *, score: float) -> dict[str, Any]:
    player = (state.get("player") or (state.get("battle") or {}).get("player") or {})
    return {
        "state_type": str(state.get("state_type") or "").strip().lower(),
        "run_outcome": str(state.get("run_outcome") or "").strip().lower(),
        "terminal": bool(state.get("terminal", False)),
        "floor": _state_floor(state),
        "player_hp": int(player.get("hp") or player.get("current_hp") or 0),
        "player_max_hp": int(player.get("max_hp") or 0),
        "enemy_hp": _enemies_total_hp(state),
        "score": float(score),
    }


def _score_combat_state(state: dict[str, Any], *, root_enemy_hp: float, root_hp_ratio: float) -> float:
    hp_ratio = _player_hp_ratio(state)
    enemy_hp = _enemies_total_hp(state)
    enemy_progress = 1.0 - enemy_hp / max(root_enemy_hp, 1.0)
    score = enemy_progress * 2.0 + hp_ratio
    score += (hp_ratio - root_hp_ratio) * 0.5
    state_type = str(state.get("state_type") or "").strip().lower()
    outcome = str(state.get("run_outcome") or "").strip().lower()
    if state.get("terminal") or outcome:
        score += 2.0 if outcome == "victory" else -2.0
    elif state_type not in _COMBAT_STATE_TYPES:
        score += 1.5
    return float(score)


def _greedy_combat_action(
    net: UnifiedNet,
    compiler: CombatFeatureCompiler,
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    *,
    encounter_id: str,
    room_type: str,
) -> tuple[int, float]:
    banks = compiler.compile(
        state,
        legal_actions,
        encounter_id=encounter_id,
        room_type=room_type,
    )
    device = next(net.parameters()).device
    enc_idx_tensor = torch.tensor([encounter_to_index(encounter_id)], dtype=torch.long, device=device)
    with torch.no_grad():
        output = net(banks=banks, encounter_idx=enc_idx_tensor)
    logits = output.logits[0, :len(legal_actions)]
    mask = output.action_mask[0, :len(legal_actions)]
    logits = torch.nan_to_num(logits.masked_fill(~mask, float("-inf")), nan=0.0)
    idx = int(torch.argmax(logits).item())
    value = float(output.values.run_value.item()) if output.values is not None else 0.0
    return idx, value


def write_critical_step_queue(
    records: list[dict[str, Any]],
    *,
    output_path: str | Path,
    top_k: int,
) -> list[dict[str, Any]]:
    selected = list(records[: max(int(top_k), 0)])
    path = write_jsonl_records(output_path, selected)
    logger.info(f"Wrote {len(selected)} critical-step queue records to {path}")
    return selected


def generate_branch_rollout_dataset(
    queue_records: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    client: BinaryBackedFullRunClient,
    net: UnifiedNet,
    compiler: CombatFeatureCompiler,
    branch_horizon: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    output_dir = Path(output_dir)
    branch_records: list[dict[str, Any]] = []
    teacher_records: list[dict[str, Any]] = []
    sample_type_counts: dict[str, int] = {}

    for queue_item in queue_records:
        snapshot_path = str(queue_item.get("snapshot_path") or "").strip()
        if not snapshot_path:
            continue
        imported_state = client.import_state(snapshot_path)
        root_state = dict(imported_state if isinstance(imported_state, dict) else {})
        legal_actions = queue_item.get("legal_actions") or root_state.get("legal_actions") or []
        if not legal_actions:
            continue
        root_state["legal_actions"] = legal_actions

        root_enemy_hp = _enemies_total_hp(root_state)
        root_hp_ratio = _player_hp_ratio(root_state)
        scores: list[float] = []
        combat_outcomes: dict[str, Any] = {}
        option_traces: dict[str, list[dict[str, Any]]] = {}
        options: list[dict[str, Any]] = []

        room_type = str(queue_item.get("room_type") or _state_room_type(root_state)).strip().lower() or "monster"
        encounter_id = str(queue_item.get("encounter_id") or root_state.get("encounter_id") or "").strip().lower()
        for option_index, legal_action in enumerate(legal_actions):
            client.import_state(snapshot_path)
            trace: list[dict[str, Any]] = []
            current_state = root_state
            current_score = float("-inf")
            try:
                current_state = client.act(legal_action)
                current_score = _score_combat_state(
                    current_state,
                    root_enemy_hp=root_enemy_hp,
                    root_hp_ratio=root_hp_ratio,
                )
                trace.append({
                    "step": 0,
                    "action": legal_action,
                    "score": current_score,
                    "state_type": str(current_state.get("state_type") or "").strip().lower(),
                })
                for step_idx in range(1, max(int(branch_horizon), 1)):
                    state_type = str(current_state.get("state_type") or "").strip().lower()
                    if current_state.get("terminal") or current_state.get("run_outcome") or state_type not in _COMBAT_STATE_TYPES:
                        break
                    legal_next = current_state.get("legal_actions") or []
                    if not legal_next:
                        break
                    greedy_idx, value_estimate = _greedy_combat_action(
                        net,
                        compiler,
                        current_state,
                        legal_next,
                        encounter_id=encounter_id,
                        room_type=room_type,
                    )
                    chosen = legal_next[greedy_idx]
                    current_state = client.act(chosen)
                    current_score = _score_combat_state(
                        current_state,
                        root_enemy_hp=root_enemy_hp,
                        root_hp_ratio=root_hp_ratio,
                    ) + value_estimate * 0.1
                    trace.append({
                        "step": step_idx,
                        "action": chosen,
                        "score": current_score,
                        "state_type": str(current_state.get("state_type") or "").strip().lower(),
                    })
            except Exception as exc:
                trace.append({"step": len(trace), "error": str(exc)})
                current_score = -5.0
            scores.append(float(current_score))
            options.append({
                "action": legal_action,
                "label": str(legal_action.get("label") or legal_action.get("action") or option_index),
            })
            combat_outcomes[str(option_index)] = _terminal_summary(current_state, score=current_score)
            option_traces[str(option_index)] = trace

        if not scores:
            continue
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        branch_record = make_raw_branch_rollout_record(
            episode_id=str(queue_item.get("episode_id") or queue_item.get("seed") or "critical-step"),
            seed=str(queue_item.get("seed") or ""),
            sample_index=int(queue_item.get("sample_index") or 0),
            sample_type="critical_combat",
            label_source="branch_search",
            root_state=root_state,
            options=options,
            scores=scores,
            best_idx=best_idx,
            combat_outcomes=combat_outcomes,
            option_traces=option_traces,
            tree_summary={"branch_horizon": int(branch_horizon), "critical_score": float(queue_item.get("critical_score", 0.0))},
            option_tree_values=None,
            port=getattr(client, "port", None),
            transport="proto",
            backend_kind="full_run_v2",
            checkpoint_path=None,
            checkpoint_sha256=None,
            combat_checkpoint_path=None,
            combat_checkpoint_sha256=None,
            generator_config={"branch_horizon": int(branch_horizon)},
        )
        branch_records.append(branch_record)
        teacher_records.append({
            "root_state": root_state,
            "legal_actions": legal_actions,
            "best_idx": int(best_idx),
            "scores": [float(score) for score in scores],
            "encounter_id": encounter_id,
            "room_type": room_type,
            "critical_tags": list(queue_item.get("critical_tags") or []),
            "critical_score": float(queue_item.get("critical_score", 0.0)),
            "label_source": "branch_search",
            "terminal_summary": combat_outcomes.get(str(best_idx)),
        })
        sample_type_counts["critical_combat"] = sample_type_counts.get("critical_combat", 0) + 1

    metadata = {
        "sample_type_counts": sample_type_counts,
        "branch_horizon": int(branch_horizon),
    }
    raw_path, _manifest_path = write_raw_branch_exports(
        output_dir=output_dir,
        branch_records=branch_records,
        metadata=metadata,
        partial=False,
    )
    write_jsonl_records(output_dir / "critical_step_teacher_v1.jsonl", teacher_records)
    return branch_records, teacher_records, raw_path


@dataclass(slots=True)
class OfflineCombatTeacherEntry:
    sample: TrainingSample
    scores: list[float]


@dataclass(slots=True)
class OfflineCombatTeacherConfig:
    updates_per_iter: int = 4
    batch_size: int = 64
    rank_weight: float = 1.0
    cont_weight: float = 1.0
    ce_weight: float = 0.0
    pairwise_margin: float = 0.2


def load_offline_combat_teacher_entries(
    path: str | Path,
    *,
    compiler: CombatFeatureCompiler | None = None,
) -> list[OfflineCombatTeacherEntry]:
    records_path = Path(path)
    if not records_path.exists():
        return []
    compiler = compiler or CombatFeatureCompiler()
    entries: list[OfflineCombatTeacherEntry] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            root_state = record.get("root_state")
            legal_actions = record.get("legal_actions") or []
            scores = [float(score) for score in (record.get("scores") or [])]
            if not isinstance(root_state, dict) or not legal_actions or not scores:
                continue
            best_idx = int(record.get("best_idx") or 0)
            if best_idx < 0 or best_idx >= len(scores):
                continue
            room_type = str(record.get("room_type") or _state_room_type(root_state)).strip().lower() or "monster"
            encounter_id = str(record.get("encounter_id") or root_state.get("encounter_id") or "").strip().lower()
            banks = compiler.compile(
                root_state,
                legal_actions,
                encounter_id=encounter_id,
                room_type=room_type,
            )
            terminal = record.get("terminal_summary") if isinstance(record.get("terminal_summary"), dict) else {}
            fight_win_target = -1.0
            turn_damage_target = -1.0
            turn_block_target = -1.0
            if terminal:
                outcome = str(terminal.get("run_outcome") or "").strip().lower()
                if outcome in {"victory", "death", "defeat"}:
                    fight_win_target = 1.0 if outcome == "victory" else 0.0
                if terminal.get("score") is not None:
                    turn_damage_target = max(float(terminal.get("score") or 0.0), 0.0)
            others = [scores[idx] for idx in range(len(scores)) if idx != best_idx]
            other_mean = sum(others) / len(others) if others else scores[best_idx]
            sample_weight = min(max(float(scores[best_idx] - other_mean), 1.0), 3.0)
            entries.append(OfflineCombatTeacherEntry(
                sample=TrainingSample(
                    banks=banks,
                    action_index=best_idx,
                    old_log_prob=0.0,
                    advantage=0.0,
                    value_target=0.0,
                    value_estimate=0.0,
                    fight_win_target=fight_win_target,
                    turn_damage_target=turn_damage_target,
                    turn_block_target=turn_block_target,
                    sample_weight=sample_weight,
                    base_sample_weight=sample_weight,
                    encounter_id=encounter_id,
                    room_type=room_type,
                    floor=_state_floor(root_state),
                    action_name=str((legal_actions[best_idx] or {}).get("action") or ""),
                    critical_tags=tuple(record.get("critical_tags") or ()),
                    critical_score=float(record.get("critical_score") or 0.0),
                ),
                scores=scores,
            ))
    return entries


def _score_tensor(entries: list[OfflineCombatTeacherEntry], max_actions: int, device: torch.device) -> torch.Tensor:
    tensor = torch.full((len(entries), max_actions), float("nan"), dtype=torch.float32, device=device)
    for row_idx, entry in enumerate(entries):
        row_scores = entry.scores[:max_actions]
        if row_scores:
            tensor[row_idx, :len(row_scores)] = torch.tensor(row_scores, dtype=torch.float32, device=device)
    return tensor


def run_offline_combat_teacher_updates(
    *,
    net: UnifiedNet,
    optimizer: torch.optim.Optimizer,
    entries: list[OfflineCombatTeacherEntry],
    config: OfflineCombatTeacherConfig,
    rng: random.Random,
    max_numeric_dim: int,
) -> dict[str, float]:
    if not entries or config.updates_per_iter <= 0 or config.batch_size <= 0:
        return {}
    net.train()
    device = next(net.parameters()).device
    metrics_acc: dict[str, float] = {
        "combat_teacher_rank_loss": 0.0,
        "combat_teacher_cont_loss": 0.0,
        "combat_teacher_ce_loss": 0.0,
        "combat_teacher_total_loss": 0.0,
    }
    completed = 0

    for _ in range(int(config.updates_per_iter)):
        batch_entries = list(rng.choices(entries, k=min(int(config.batch_size), len(entries))))
        batch_samples = [entry.sample for entry in batch_entries]
        batched = collate_training_samples(batch_samples, max_numeric_dim=max_numeric_dim)
        batched = _move_batched_to_device(batched, device)
        enc_idx = batched.encounter_indices if batched.encounter_indices is not None else None
        output = net(batched_banks=batched.banks, decision_domain="combat", encounter_idx=enc_idx)
        logits = torch.nan_to_num(output.logits, nan=0.0)
        score_t = _score_tensor(batch_entries, logits.size(1), device)
        weights = batched.sample_weights.to(device)
        weights = weights / weights.sum().clamp(min=1e-8)
        action_indices = batched.action_indices.to(device)

        chosen_scores = score_t.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        chosen_logits = logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        score_gap = chosen_scores.unsqueeze(1) - score_t
        valid_pairs = torch.isfinite(score_t) & (score_gap > 1e-6)
        pairwise = F.relu(logits - chosen_logits.unsqueeze(1) + config.pairwise_margin)
        per_sample_rank = []
        for row_idx in range(pairwise.size(0)):
            mask = valid_pairs[row_idx]
            if bool(mask.any()):
                per_sample_rank.append(pairwise[row_idx][mask].mean())
            else:
                per_sample_rank.append(torch.tensor(0.0, device=device))
        rank_loss = (torch.stack(per_sample_rank) * weights).sum()

        ce_loss = torch.tensor(0.0, device=device)
        if config.ce_weight > 0.0:
            ce_loss = (F.cross_entropy(logits, action_indices, reduction="none") * weights).sum()

        cont_terms: list[torch.Tensor] = []
        if output.values is not None:
            fw_t = batched.fight_win_targets.to(device)
            fw_mask = (fw_t >= 0.0).float()
            if fw_mask.sum() > 0:
                cont_terms.append((((output.values.fight_win - fw_t.clamp(0, 1)).pow(2)) * weights * fw_mask).sum() / fw_mask.sum().clamp(min=1.0))
            td_t = batched.turn_damage_targets.to(device)
            td_mask = (td_t >= 0.0).float()
            if td_mask.sum() > 0:
                cont_terms.append((F.smooth_l1_loss(output.values.turn_damage_lookahead, td_t.clamp(min=0.0), reduction="none") * weights * td_mask).sum() / td_mask.sum().clamp(min=1.0))
            tb_t = batched.turn_block_targets.to(device)
            tb_mask = (tb_t >= 0.0).float()
            if tb_mask.sum() > 0:
                cont_terms.append((F.smooth_l1_loss(output.values.turn_block_lookahead, tb_t.clamp(min=0.0), reduction="none") * weights * tb_mask).sum() / tb_mask.sum().clamp(min=1.0))
        cont_loss = torch.stack(cont_terms).mean() if cont_terms else torch.tensor(0.0, device=device)

        total_loss = config.rank_weight * rank_loss + config.cont_weight * cont_loss + config.ce_weight * ce_loss
        if not torch.isfinite(total_loss):
            continue
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

        completed += 1
        metrics_acc["combat_teacher_rank_loss"] += float(rank_loss.item())
        metrics_acc["combat_teacher_cont_loss"] += float(cont_loss.item())
        metrics_acc["combat_teacher_ce_loss"] += float(ce_loss.item())
        metrics_acc["combat_teacher_total_loss"] += float(total_loss.item())

    if completed <= 0:
        return {}
    metrics_acc["combat_teacher_updates"] = float(completed)
    for key in ("combat_teacher_rank_loss", "combat_teacher_cont_loss", "combat_teacher_ce_loss", "combat_teacher_total_loss"):
        metrics_acc[key] /= completed
    return metrics_acc
