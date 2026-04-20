"""STS2AI 运行时平台主入口。"""

from game_bridge.catalog import GAME_CATALOG, GameCatalog
from game_bridge.session import (
    CombatSession,
    FullRunSession,
    SessionFactory,
    SessionPool,
    create_combat_session,
    create_full_run_session,
)
from game_bridge.sim import launch_headless_sim
from game_bridge.spectate import PolicyAdapter, SpectatorController

__all__ = [
    "CombatSession",
    "FullRunSession",
    "SessionFactory",
    "SessionPool",
    "create_combat_session",
    "create_full_run_session",
    "launch_headless_sim",
    "SpectatorController",
    "PolicyAdapter",
    "GameCatalog",
    "GAME_CATALOG",
]
