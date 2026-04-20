from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..domain import IterationManifest, TeacherRequest, TrainingSample
from ..paths import ZeroPaths


class ArtifactStore:
    def __init__(self, paths: ZeroPaths):
        self._paths = paths
        self._paths.ensure()

    def write_dataset_shard(self, iteration: int, samples: Iterable[TrainingSample]) -> Path:
        path = self._paths.dataset_shards / f"iter_{iteration:04d}.jsonl"
        self._write_jsonl(path, (sample.to_dict() for sample in samples))
        return path

    def write_teacher_labels(self, iteration: int, requests: Iterable[TeacherRequest]) -> Path:
        path = self._paths.teacher_labels / f"iter_{iteration:04d}.jsonl"
        self._write_jsonl(path, (asdict(request) for request in requests))
        return path

    def write_raw_runs(self, iteration: int, rows: Iterable[dict[str, object]]) -> Path:
        path = self._paths.raw_runs / f"iter_{iteration:04d}.jsonl"
        self._write_jsonl(path, rows)
        return path

    def write_manifest(self, manifest: IterationManifest) -> Path:
        path = self._paths.manifests / f"iter_{manifest.iteration:04d}.json"
        path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _write_jsonl(self, path: Path, rows: Iterable[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
