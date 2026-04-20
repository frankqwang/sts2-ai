"""策略无关观战 policy 协议与内置实现。"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from game_bridge.types import PolicyContext


class PolicyAdapter(Protocol):
    def select_action(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        context: PolicyContext,
    ) -> dict[str, Any] | None: ...


class NullPolicy:
    def select_action(self, state, legal_actions, context):
        return None


@dataclass
class ReplayPolicy:
    actions: list[dict[str, Any]] = field(default_factory=list)
    _cursor: int = 0

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ReplayPolicy":
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(actions=records)

    def select_action(self, state, legal_actions, context):
        if self._cursor >= len(self.actions):
            return None
        action = self.actions[self._cursor]
        self._cursor += 1
        return dict(action)


class ManualPolicy:
    def select_action(self, state, legal_actions, context):
        if not legal_actions:
            return None
        print(f"\n[step {context.step_index}] state_type={state.get('state_type')} legal={len(legal_actions)}")
        for idx, action in enumerate(legal_actions):
            label = action.get("label") or action.get("action") or idx
            print(f"  [{idx}] {label} -> {action}")
        raw = input("Choose action index (empty to stop): ").strip()
        if raw == "":
            return None
        return dict(legal_actions[int(raw)])


@dataclass
class ExternalPolicy:
    handler: Callable[[dict[str, Any], list[dict[str, Any]], PolicyContext], dict[str, Any] | None]

    def select_action(self, state, legal_actions, context):
        return self.handler(state, legal_actions, context)

    @classmethod
    def from_import_path(cls, import_path: str) -> "ExternalPolicy":
        module_name, attr_name = import_path.split(":", 1)
        module = importlib.import_module(module_name)
        handler = getattr(module, attr_name)
        if not callable(handler):
            raise TypeError(f"external policy target is not callable: {import_path}")
        return cls(handler=handler)
