from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def build_manifest(
    *,
    dataset_kind: str,
    schema_version: str,
    output_dir: str,
    status: str,
    summary: dict[str, Any] | None = None,
    source_raw_paths: list[str] | None = None,
    derived_from: list[str] | None = None,
    generator_config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "dataset_kind": dataset_kind,
        "schema_version": schema_version,
        "generated_at_utc": utc_now(),
        "output_dir": output_dir,
        "status": status,
        "source_raw_paths": source_raw_paths or [],
        "derived_from": derived_from or [],
        "generator_config": generator_config or {},
        "summary": summary or {},
    }
    if extra:
        payload.update(extra)
    return payload


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
