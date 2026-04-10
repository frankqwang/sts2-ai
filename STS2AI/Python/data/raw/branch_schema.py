from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


RAW_BRANCH_ROLLOUT_SCHEMA_VERSION = "raw_branch_rollout.v1"


def _utc_now() -> str:
    return datetime.now(UTC).replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _boss_token(state: dict[str, Any]) -> str | None:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    for key in ("boss_entry_token", "next_boss_token", "boss_token"):
        value = run.get(key)
        if value:
            return str(value)
    return None


def _room_type(state: dict[str, Any]) -> str | None:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    for key in ("room_type", "current_room_type", "screen_room_type"):
        value = run.get(key)
        if value:
            return str(value)
    return None


def _action_mask_signature(legal_actions: list[dict[str, Any]]) -> str:
    names = sorted(str(action.get("action") or "") for action in legal_actions if isinstance(action, dict))
    digest = hashlib.sha256(json.dumps(names, ensure_ascii=True).encode("utf-8")).hexdigest()
    return digest[:16]


def make_raw_branch_rollout_record(
    *,
    episode_id: str,
    seed: str,
    sample_index: int,
    sample_type: str,
    label_source: str,
    root_state: dict[str, Any],
    options: list[dict[str, Any]],
    scores: list[float],
    best_idx: int,
    combat_outcomes: dict[str, Any],
    option_traces: dict[str, list[dict[str, Any]]],
    tree_summary: dict[str, Any] | None,
    option_tree_values: list[dict[str, Any]] | None,
    port: int | None,
    transport: str,
    backend_kind: str,
    checkpoint_path: str | None,
    checkpoint_sha256: str | None,
    combat_checkpoint_path: str | None,
    combat_checkpoint_sha256: str | None,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    root_legal = root_state.get("legal_actions") if isinstance(root_state.get("legal_actions"), list) else []
    floor = int(((root_state.get("run") or {}).get("floor")) or 0)
    act = int(((root_state.get("run") or {}).get("act")) or 0)
    root_decision_id = f"{seed}:{sample_type}:f{floor:02d}:s{sample_index:05d}"
    branch_rollouts: list[dict[str, Any]] = []
    for option_index, option in enumerate(options):
        action_payload = option.get("action") if isinstance(option, dict) else None
        branch_rollouts.append(
            {
                "branch_id": f"{root_decision_id}:opt{option_index:02d}",
                "parent_branch_id": None,
                "branch_depth": 0,
                "root_option_index": int(option_index),
                "branch_label": str(
                    option.get("card_name")
                    or option.get("card_id")
                    or option.get("label")
                    or option.get("type")
                    or option_index
                ),
                "option_metadata": option,
                "applied_action": action_payload,
                "score": float(scores[option_index]) if option_index < len(scores) else 0.0,
                "terminal_summary": combat_outcomes.get(str(option_index)) or combat_outcomes.get(option_index),
                "per_step_trace": option_traces.get(str(option_index)) or option_traces.get(option_index) or [],
            }
        )
    return {
        "schema_version": RAW_BRANCH_ROLLOUT_SCHEMA_VERSION,
        "recorded_at_utc": _utc_now(),
        "root_decision_id": root_decision_id,
        "episode_id": episode_id,
        "run_id": episode_id,
        "seed": seed,
        "sample_index": int(sample_index),
        "sample_type": sample_type,
        "label_source": label_source,
        "backend_kind": backend_kind,
        "transport": transport,
        "port": port,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "combat_checkpoint_path": combat_checkpoint_path,
        "combat_checkpoint_sha256": combat_checkpoint_sha256,
        "generator_config": generator_config,
        "state_type": str(root_state.get("state_type") or "").strip().lower(),
        "act": act,
        "floor": floor,
        "boss_token": _boss_token(root_state),
        "room_type": _room_type(root_state),
        "raw_state": root_state,
        "legal_actions": root_legal,
        "action_mask_signature": _action_mask_signature(root_legal),
        "options": options,
        "scores": scores,
        "best_idx": int(best_idx),
        "tree_summary": tree_summary,
        "option_tree_values": option_tree_values,
        "branch_rollouts": branch_rollouts,
    }
