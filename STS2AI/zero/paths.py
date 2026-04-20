from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ZERO_ROOT = Path(__file__).resolve().parent
STS2AI_ROOT = ZERO_ROOT.parent
REPO_ROOT = STS2AI_ROOT.parent


@dataclass(slots=True)
class ZeroPaths:
    root: Path = ZERO_ROOT

    @property
    def raw_runs(self) -> Path:
        return self.root / "raw_runs"

    @property
    def teacher_labels(self) -> Path:
        return self.root / "teacher_labels"

    @property
    def dataset_shards(self) -> Path:
        return self.root / "dataset_shards"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evaluations(self) -> Path:
        return self.root / "eval"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def ensure(self) -> None:
        for path in (
            self.raw_runs,
            self.teacher_labels,
            self.dataset_shards,
            self.checkpoints,
            self.evaluations,
            self.logs,
            self.manifests,
        ):
            path.mkdir(parents=True, exist_ok=True)
