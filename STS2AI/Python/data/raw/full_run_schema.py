from __future__ import annotations

import hashlib
import json
from typing import Any


RAW_FULL_RUN_STEP_SCHEMA_VERSION = "raw_full_run_step.v1"


def _state_type(state: dict[str, Any] | None) -> str:
    return str((state or {}).get("state_type") or "").strip().lower()


def _run_value(state: dict[str, Any], key: str, default: Any = None) -> Any:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    return run.get(key, default)


def _boss_token(state: dict[str, Any]) -> str | None:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    for key in ("boss_entry_token", "next_boss_token", "boss_token"):
        value = run.get(key)
        if value:
            return str(value)
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        enemy_id = enemy.get("id") or enemy.get("monster_id") or enemy.get("name")
        if enemy_id:
            return str(enemy_id)
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


def trajectory_record_to_raw_step(
    record: dict[str, Any],
    *,
    episode_id: str | None = None,
    backend_kind: str | None = None,
    port: int | None = None,
    transport: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    raw_state = record.get("raw_state") if isinstance(record.get("raw_state"), dict) else {}
    legal_actions = record.get("candidate_actions") if isinstance(record.get("candidate_actions"), list) else []
    next_state = record.get("next_state") if isinstance(record.get("next_state"), dict) else {}
    seed = str(record.get("seed") or "")
    run_id = str(record.get("run_id") or "")
    step_index = int(record.get("step_index") or 0)
    return {
        "schema_version": RAW_FULL_RUN_STEP_SCHEMA_VERSION,
        "source_schema_version": str(record.get("schema_version") or ""),
        "run_id": run_id,
        "episode_id": episode_id or run_id or seed,
        "seed": seed,
        "step_index": step_index,
        "timestamp_utc": record.get("timestamp_utc"),
        "backend_kind": backend_kind or record.get("backend_kind") or record.get("env_api_mode") or "unknown",
        "transport": transport,
        "port": port,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "state_type": _state_type(raw_state) or str(record.get("state_type") or "").strip().lower(),
        "act": int(record.get("act") or _run_value(raw_state, "act", 0) or 0),
        "floor": int(record.get("floor") or _run_value(raw_state, "floor", 0) or 0),
        "boss_token": _boss_token(raw_state),
        "room_type": _room_type(raw_state),
        "raw_state": raw_state,
        "legal_actions": legal_actions,
        "action_mask_signature": _action_mask_signature(legal_actions),
        "chosen_action": record.get("chosen_action"),
        "action_source": record.get("action_source"),
        "next_state": next_state,
        "delta": record.get("delta"),
        "terminal": bool(record.get("terminal")),
        "run_outcome": record.get("run_outcome"),
    }
