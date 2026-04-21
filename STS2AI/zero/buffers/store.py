from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..domain import IterationManifest, SearchRequest, TrainingSample
from ..paths import ZeroPaths


class ArtifactStore:
    def __init__(self, paths: ZeroPaths):
        self._paths = paths
        self._paths.ensure()

    def write_dataset_shard(self, iteration: int, samples: Iterable[TrainingSample]) -> Path:
        path = self._paths.dataset_shards / f"iter_{iteration:04d}.jsonl"
        self._write_jsonl(path, (sample.to_dict() for sample in samples))
        return path

    def write_search_labels(self, iteration: int, requests: Iterable[SearchRequest]) -> Path:
        path = self._paths.search_labels / f"iter_{iteration:04d}.jsonl"
        self._write_jsonl(path, (asdict(request) for request in requests))
        return path

    def write_raw_runs(self, iteration: int, rows: Iterable[dict[str, object]]) -> Path:
        path = self._paths.raw_runs / f"iter_{iteration:04d}.jsonl"
        self._write_jsonl(path, rows)
        return path

    def append_raw_run_row(self, iteration: int, row: dict[str, object]) -> Path:
        path = self._paths.raw_runs / f"iter_{iteration:04d}.jsonl"
        self._append_jsonl_row(path, row)
        return path

    def write_manifest(self, manifest: IterationManifest) -> Path:
        path = self._paths.manifests / f"iter_{manifest.iteration:04d}.json"
        self._atomic_write_text(path, json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
        return path

    def write_eval_trace(self, name: str, rows: Iterable[dict[str, object]]) -> Path:
        safe_name = name.replace(" ", "_")
        path = self._paths.evaluations / f"{safe_name}.jsonl"
        self._write_jsonl(path, rows)
        return path

    def append_eval_trace_row(self, name: str, row: dict[str, object]) -> Path:
        safe_name = name.replace(" ", "_")
        path = self._paths.evaluations / f"{safe_name}.jsonl"
        self._append_jsonl_row(path, row)
        return path

    def write_progress_event(self, iteration: int, event: dict[str, object]) -> Path:
        path = self._paths.logs / f"iter_{iteration:04d}.events.jsonl"
        self._append_jsonl_row(path, event)
        return path

    def write_status(self, iteration: int, payload: dict[str, object]) -> Path:
        path = self._paths.logs / f"iter_{iteration:04d}.status.json"
        self._atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
        return path

    def reset_iteration_outputs(self, iteration: int) -> None:
        for path in (
            self._paths.raw_runs / f"iter_{iteration:04d}.jsonl",
            self._paths.logs / f"iter_{iteration:04d}.events.jsonl",
            self._paths.logs / f"iter_{iteration:04d}.status.json",
        ):
            path.unlink(missing_ok=True)

    def _write_jsonl(self, path: Path, rows: Iterable[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False))
                    handle.write("\n")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _append_jsonl_row(self, path: Path, row: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
