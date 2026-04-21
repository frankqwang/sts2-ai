"""运行时 session 入口。"""

from game_bridge.session.base import SessionFactory
from game_bridge.session.combat import CombatSession
from game_bridge.session.full_run import (
    ApiBackedFullRunClient,
    BinaryBackedFullRunClient,
    PipeBackedFullRunClient,
    create_full_run_client,
)
from game_bridge.session.pool import SessionPool

FullRunSession = PipeBackedFullRunClient | ApiBackedFullRunClient | BinaryBackedFullRunClient


def create_combat_session(**kwargs) -> CombatSession:
    return CombatSession(**kwargs)


def create_full_run_session(**kwargs) -> FullRunSession:
    return create_full_run_client(**kwargs)


__all__ = [
    "CombatSession",
    "FullRunSession",
    "SessionFactory",
    "SessionPool",
    "create_combat_session",
    "create_full_run_session",
]
