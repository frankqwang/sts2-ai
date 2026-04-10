from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.manifest import build_manifest, write_manifest


SFT_DIALOGUE_SCHEMA_VERSION = "sft_dialogue.v1"
PREFERENCE_PAIR_SCHEMA_VERSION = "preference_pair.v1"


def build_sft_dialogue_view(
    *,
    raw_full_run_records: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = out_dir.parent.parent / "raw" / "raw_full_run_steps.jsonl"
    output_path = out_dir / "sft_dialogue.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in raw_full_run_records:
            prompt_state = {
                "state_type": record.get("state_type"),
                "act": record.get("act"),
                "floor": record.get("floor"),
                "boss_token": record.get("boss_token"),
                "room_type": record.get("room_type"),
                "raw_state": record.get("raw_state"),
                "candidate_actions": record.get("legal_actions"),
            }
            payload = {
                "schema_version": SFT_DIALOGUE_SCHEMA_VERSION,
                "source_schema_version": record.get("schema_version"),
                "source_raw_id": f"{record.get('run_id')}:{record.get('step_index')}",
                "run_id": record.get("run_id"),
                "episode_id": record.get("episode_id"),
                "seed": record.get("seed"),
                "step_index": int(record.get("step_index") or 0),
                "prompt_state": json.dumps(prompt_state, ensure_ascii=False),
                "candidate_actions": record.get("legal_actions"),
                "target_action": record.get("chosen_action"),
                "outcome_summary": {
                    "terminal": bool(record.get("terminal")),
                    "run_outcome": record.get("run_outcome"),
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
    manifest = build_manifest(
        dataset_kind="sft_dialogue",
        schema_version=SFT_DIALOGUE_SCHEMA_VERSION,
        output_dir=str(out_dir),
        status="complete",
        source_raw_paths=[str(raw_root)],
        derived_from=["raw_full_run_step.v1"],
        summary={"num_records": len(raw_full_run_records)},
        extra={"sft_dialogue_path": str(output_path)},
    )
    manifest_path = write_manifest(out_dir / "manifest.json", manifest)
    return output_path, manifest_path


def build_preference_pair_view(
    *,
    raw_branch_records: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = out_dir.parent.parent / "raw" / "raw_branch_rollout.jsonl"
    output_path = out_dir / "preference_pair.jsonl"
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in raw_branch_records:
            scores = record.get("scores") or []
            options = record.get("options") or []
            if len(scores) < 2 or len(options) < 2:
                continue
            prompt_state = {
                "state_type": record.get("state_type"),
                "act": record.get("act"),
                "floor": record.get("floor"),
                "sample_type": record.get("sample_type"),
                "boss_token": record.get("boss_token"),
                "room_type": record.get("room_type"),
                "raw_state": record.get("raw_state"),
            }
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    if float(scores[i]) == float(scores[j]):
                        continue
                    preferred = "a" if float(scores[i]) > float(scores[j]) else "b"
                    payload = {
                        "schema_version": PREFERENCE_PAIR_SCHEMA_VERSION,
                        "source_schema_version": record.get("schema_version"),
                        "source_raw_id": record.get("root_decision_id"),
                        "prompt_state": json.dumps(prompt_state, ensure_ascii=False),
                        "candidate_a": options[i],
                        "candidate_b": options[j],
                        "preferred": preferred,
                        "preference_source": record.get("label_source") or "counterfactual_score",
                        "score_a": float(scores[i]),
                        "score_b": float(scores[j]),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False))
                    handle.write("\n")
                    written += 1
    manifest = build_manifest(
        dataset_kind="preference_pair",
        schema_version=PREFERENCE_PAIR_SCHEMA_VERSION,
        output_dir=str(out_dir),
        status="complete",
        source_raw_paths=[str(raw_root)],
        derived_from=["raw_branch_rollout.v1"],
        summary={"num_pairs": written, "num_roots": len(raw_branch_records)},
        extra={"preference_pair_path": str(output_path)},
    )
    manifest_path = write_manifest(out_dir / "manifest.json", manifest)
    return output_path, manifest_path


def _ranking_prompt_state(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_type": record.get("sample_type"),
        "act": record.get("act"),
        "floor": record.get("floor"),
        "sample_type": record.get("sample_type"),
        "deck_ids": record.get("deck_ids"),
        "relic_ids": record.get("relic_ids"),
        "label_source": record.get("label_source"),
    }


def build_sft_dialogue_from_ranking_records(
    *,
    ranking_records: list[dict[str, Any]],
    output_dir: str | Path,
    source_schema_version: str = "ranking_sample.v1",
    source_path: str | Path | None = None,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "sft_dialogue.jsonl"
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(ranking_records):
            options = record.get("options") or []
            best_idx = record.get("best_idx")
            if not options or best_idx is None:
                continue
            try:
                best_idx = int(best_idx)
            except Exception:
                continue
            if best_idx < 0 or best_idx >= len(options):
                continue
            payload = {
                "schema_version": SFT_DIALOGUE_SCHEMA_VERSION,
                "source_schema_version": source_schema_version,
                "source_raw_id": record.get("source_raw_id") or f"ranking:{idx}",
                "run_id": record.get("run_id"),
                "episode_id": record.get("episode_id"),
                "seed": record.get("seed"),
                "step_index": int(record.get("floor") or 0),
                "prompt_state": json.dumps(_ranking_prompt_state(record), ensure_ascii=False),
                "candidate_actions": options,
                "target_action": options[best_idx],
                "outcome_summary": {
                    "best_idx": best_idx,
                    "best_score": float((record.get("scores") or [0.0])[best_idx]),
                    "score_spread": float(max(record.get("scores") or [0.0]) - min(record.get("scores") or [0.0])),
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
            written += 1
    manifest = build_manifest(
        dataset_kind="sft_dialogue",
        schema_version=SFT_DIALOGUE_SCHEMA_VERSION,
        output_dir=str(out_dir),
        status="complete",
        source_raw_paths=[str(source_path)] if source_path else [],
        derived_from=[source_schema_version],
        summary={"num_records": written},
        extra={"sft_dialogue_path": str(output_path)},
    )
    manifest_path = write_manifest(out_dir / "manifest.json", manifest)
    return output_path, manifest_path


def build_preference_pair_from_ranking_records(
    *,
    ranking_records: list[dict[str, Any]],
    output_dir: str | Path,
    source_schema_version: str = "ranking_sample.v1",
    source_path: str | Path | None = None,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "preference_pair.jsonl"
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(ranking_records):
            scores = record.get("scores") or []
            options = record.get("options") or []
            if len(scores) < 2 or len(options) < 2:
                continue
            prompt_state = _ranking_prompt_state(record)
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    if float(scores[i]) == float(scores[j]):
                        continue
                    preferred = "a" if float(scores[i]) > float(scores[j]) else "b"
                    payload = {
                        "schema_version": PREFERENCE_PAIR_SCHEMA_VERSION,
                        "source_schema_version": source_schema_version,
                        "source_raw_id": record.get("source_raw_id") or f"ranking:{idx}",
                        "prompt_state": json.dumps(prompt_state, ensure_ascii=False),
                        "candidate_a": options[i],
                        "candidate_b": options[j],
                        "preferred": preferred,
                        "preference_source": record.get("label_source") or "counterfactual_score",
                        "score_a": float(scores[i]),
                        "score_b": float(scores[j]),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False))
                    handle.write("\n")
                    written += 1
    manifest = build_manifest(
        dataset_kind="preference_pair",
        schema_version=PREFERENCE_PAIR_SCHEMA_VERSION,
        output_dir=str(out_dir),
        status="complete",
        source_raw_paths=[str(source_path)] if source_path else [],
        derived_from=[source_schema_version],
        summary={"num_pairs": written, "num_roots": len(ranking_records)},
        extra={"preference_pair_path": str(output_path)},
    )
    manifest_path = write_manifest(out_dir / "manifest.json", manifest)
    return output_path, manifest_path
