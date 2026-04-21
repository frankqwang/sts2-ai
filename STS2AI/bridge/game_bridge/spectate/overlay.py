"""观战 overlay 输出。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OverlayWriter:
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, data: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
