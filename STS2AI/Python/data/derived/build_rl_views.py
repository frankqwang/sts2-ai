from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from data.manifest import build_manifest, write_manifest


RANKING_SAMPLE_SCHEMA_VERSION = "ranking_sample.v1"
RL_TRANSITION_SCHEMA_VERSION = "rl_transition.v1"


def _encode_option_tensors(
    state: dict[str, Any],
    option_actions: list[dict[str, Any]],
) -> dict[str, np.ndarray] | None:
    if not option_actions:
        return None
    try:
        from rl_encoder_v2 import build_structured_actions, build_structured_state
        from vocab import load_vocab as _load_vocab

        vocab = _load_vocab()
        ss = build_structured_state(state, vocab)
        sa = build_structured_actions(state, option_actions, vocab)
        encoded: dict[str, np.ndarray] = {}
        for fname in [
            "scalars", "deck_ids", "deck_aux", "deck_mask",
            "relic_ids", "relic_aux", "relic_mask",
            "potion_ids", "potion_mask",
            "hand_ids", "hand_aux", "hand_mask",
            "enemy_ids", "enemy_aux", "enemy_mask",
            "map_node_types", "map_node_mask",
            "reward_card_ids", "reward_card_aux", "reward_card_mask",
            "shop_card_ids", "shop_relic_ids", "shop_potion_ids",
            "shop_prices", "shop_mask",
            "rest_option_ids", "rest_option_mask",
        ]:
            val = getattr(ss, fname, None)
            if val is not None:
                encoded[f"state_{fname}"] = np.asarray(val)
        encoded["state_screen_type_idx"] = np.array(ss.screen_type_idx, dtype=np.int64)
        encoded["state_next_boss_idx"] = np.array(ss.next_boss_idx, dtype=np.int64)
        encoded["state_event_option_count"] = np.array(ss.event_option_count, dtype=np.int64)

        for fname in [
            "action_type_ids", "target_card_ids", "target_enemy_ids",
            "target_node_types", "target_indices", "action_mask",
        ]:
            val = getattr(sa, fname, None)
            if val is not None:
                encoded[f"action_{fname}"] = np.asarray(val)
        return encoded
    except Exception:
        return None


def build_ranking_view(
    *,
    raw_branch_records: list[dict[str, Any]],
    output_dir: str | Path,
    compatibility_root: str | Path,
    partial: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_dir = Path(output_dir)
    compatibility_dir = Path(compatibility_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir = compatibility_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = compatibility_dir / "card_ranking.jsonl"
    derived_ranking_path = out_dir / "ranking_sample.jsonl"

    derived_samples: list[dict[str, Any]] = []
    sample_type_counts: Counter[str] = Counter()
    usable = 0
    tensor_count = 0

    with ranking_path.open("w", encoding="utf-8") as compat_handle, derived_ranking_path.open("w", encoding="utf-8") as derived_handle:
        for sample_index, raw in enumerate(raw_branch_records):
            options = raw.get("options") or []
            scores = raw.get("scores") or []
            root_state = raw.get("raw_state") or {}
            if len(options) < 2 or len(scores) < 2:
                continue
            encoded_tensors = _encode_option_tensors(
                root_state,
                [opt.get("action") for opt in options if isinstance(opt, dict) and isinstance(opt.get("action"), dict)],
            )
            tensor_rel_path = f"tensors/sample_{sample_index:05d}.npz" if encoded_tensors else None
            if encoded_tensors:
                np.savez_compressed(str(compatibility_dir / tensor_rel_path), **encoded_tensors)
                tensor_count += 1
            record = {
                "schema_version": RANKING_SAMPLE_SCHEMA_VERSION,
                "source_schema_version": raw.get("schema_version"),
                "source_raw_id": raw.get("root_decision_id"),
                "deck_ids": [c.get("id") or c.get("label") or "?" for c in ((root_state.get("player") or {}).get("deck") or [])],
                "relic_ids": [r.get("id") or r.get("name") or "?" for r in ((root_state.get("player") or {}).get("relics") or [])],
                "floor": int(raw.get("floor") or 0),
                "act": int(raw.get("act") or 0),
                "sample_type": raw.get("sample_type") or "unknown",
                "label_source": raw.get("label_source") or "single_step",
                "options": [{k: v for k, v in opt.items() if k != "action"} for opt in options if isinstance(opt, dict)],
                "scores": scores,
                "best_idx": int(raw.get("best_idx") or 0),
                "combat_outcomes": {
                    str(branch.get("root_option_index")): branch.get("terminal_summary")
                    for branch in (raw.get("branch_rollouts") or [])
                    if isinstance(branch, dict)
                },
                "tree_summary": raw.get("tree_summary"),
                "option_tree_values": raw.get("option_tree_values"),
                "state_tensors_path": tensor_rel_path,
            }
            derived_samples.append(record)
            serialized = json.dumps(record, ensure_ascii=False)
            compat_handle.write(serialized)
            compat_handle.write("\n")
            derived_handle.write(serialized)
            derived_handle.write("\n")
            sample_type_counts[str(record["sample_type"])] += 1
            if max(scores) - min(scores) >= 0.01:
                usable += 1

    manifest = build_manifest(
        dataset_kind="ranking_sample",
        schema_version=RANKING_SAMPLE_SCHEMA_VERSION,
        output_dir=str(compatibility_dir),
        status="partial" if partial else "complete",
        source_raw_paths=[str(Path(compatibility_root) / "raw" / "raw_branch_rollout.jsonl")],
        derived_from=["raw_branch_rollout.v1"],
        summary={
            "num_samples": len(derived_samples),
            "usable_samples": usable,
            "tensor_samples": tensor_count,
            "sample_type_counts": dict(sample_type_counts),
        },
        extra={
            "ranking_jsonl_path": str(ranking_path),
            "derived_ranking_jsonl_path": str(derived_ranking_path),
            "tensors_dir": str(tensors_dir),
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return derived_samples, manifest["summary"]


def build_transition_view(
    *,
    raw_full_run_records: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = out_dir.parent.parent / "raw" / "raw_full_run_steps.jsonl"
    output_path = out_dir / "rl_transition.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in raw_full_run_records:
            payload = {
                "schema_version": RL_TRANSITION_SCHEMA_VERSION,
                "source_schema_version": record.get("schema_version"),
                "run_id": record.get("run_id"),
                "episode_id": record.get("episode_id"),
                "seed": record.get("seed"),
                "step_index": int(record.get("step_index") or 0),
                "state_type": record.get("state_type"),
                "act": int(record.get("act") or 0),
                "floor": int(record.get("floor") or 0),
                "state": record.get("raw_state"),
                "candidate_actions": record.get("legal_actions"),
                "action": record.get("chosen_action"),
                "next_state": record.get("next_state"),
                "delta": record.get("delta"),
                "terminal": bool(record.get("terminal")),
                "run_outcome": record.get("run_outcome"),
                "action_source": record.get("action_source"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
    manifest = build_manifest(
        dataset_kind="rl_transition",
        schema_version=RL_TRANSITION_SCHEMA_VERSION,
        output_dir=str(out_dir),
        status="complete",
        source_raw_paths=[str(raw_root)],
        derived_from=["raw_full_run_step.v1"],
        summary={"num_records": len(raw_full_run_records)},
        extra={"transition_jsonl_path": str(output_path)},
    )
    manifest_path = write_manifest(out_dir / "manifest.json", manifest)
    return output_path, manifest_path
