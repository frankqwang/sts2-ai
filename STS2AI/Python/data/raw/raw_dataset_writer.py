from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.manifest import build_manifest, write_manifest
from data.raw.full_run_schema import RAW_FULL_RUN_STEP_SCHEMA_VERSION, trajectory_record_to_raw_step
from data.raw.branch_schema import RAW_BRANCH_ROLLOUT_SCHEMA_VERSION


def write_jsonl_records(path: str | Path, records: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return output_path


def write_raw_full_run_exports(
    *,
    output_dir: str | Path,
    trajectory_payloads: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    checkpoint_path: str | None = None,
    checkpoint_sha256: str | None = None,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    raw_dir = out_dir / "raw"
    records: list[dict[str, Any]] = []
    for seed, steps in sorted(trajectory_payloads.items()):
        for record in steps:
            records.append(
                trajectory_record_to_raw_step(
                    record,
                    episode_id=str(record.get("run_id") or seed),
                    backend_kind=str(metadata.get("backend_kind") or metadata.get("env_api_mode") or "unknown"),
                    transport=metadata.get("transport"),
                    port=metadata.get("port"),
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha256,
                )
            )
    raw_path = write_jsonl_records(raw_dir / "raw_full_run_steps.jsonl", records)
    manifest = build_manifest(
        dataset_kind="raw_full_run",
        schema_version=RAW_FULL_RUN_STEP_SCHEMA_VERSION,
        output_dir=str(out_dir),
        status="complete",
        summary={
            "num_records": len(records),
            "num_runs": len(trajectory_payloads),
            "captured_seeds": sorted(trajectory_payloads.keys()),
        },
        generator_config=metadata,
        extra={
            "raw_path": str(raw_path),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
        },
    )
    manifest_path = write_manifest(raw_dir / "raw_manifest.json", manifest)
    return raw_path, manifest_path


def write_raw_branch_exports(
    *,
    output_dir: str | Path,
    branch_records: list[dict[str, Any]],
    metadata: dict[str, Any],
    partial: bool,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    raw_dir = out_dir / "raw"
    raw_path = write_jsonl_records(raw_dir / "raw_branch_rollout.jsonl", branch_records)
    manifest = build_manifest(
        dataset_kind="raw_branch_rollout",
        schema_version=RAW_BRANCH_ROLLOUT_SCHEMA_VERSION,
        output_dir=str(out_dir),
        status="partial" if partial else "complete",
        summary={
            "num_records": len(branch_records),
            "sample_type_counts": metadata.get("sample_type_counts", {}),
        },
        generator_config=metadata,
        extra={"raw_path": str(raw_path)},
    )
    manifest_path = write_manifest(raw_dir / "raw_manifest.json", manifest)
    return raw_path, manifest_path
