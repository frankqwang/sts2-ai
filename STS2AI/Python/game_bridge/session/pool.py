"""通用 session pool。"""

from __future__ import annotations

from dataclasses import dataclass, field

from game_bridge.session.base import SessionFactory


@dataclass
class SessionPool:
    factory: SessionFactory
    size: int = 1
    _sessions: list[object] = field(default_factory=list, init=False, repr=False)

    def warmup(self) -> None:
        if self._sessions:
            return
        self._sessions = [self.factory.create() for _ in range(max(int(self.size), 1))]

    def get(self, worker_id: int):
        if not self._sessions:
            self.warmup()
        return self._sessions[int(worker_id) % len(self._sessions)]

    def close_all(self) -> None:
        for session in self._sessions:
            try:
                session.close()
            except Exception:
                pass
        self._sessions.clear()

    def __enter__(self):
        self.warmup()
        return self

    def __exit__(self, *_args):
        self.close_all()
