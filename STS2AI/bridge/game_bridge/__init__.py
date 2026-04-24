"""STS2AI 运行时平台主入口。"""

from game_bridge.catalog import GAME_CATALOG, GameCatalog
from game_bridge.session import (
    FullRunSession,
    GameSession,
    HttpProtoJsonTransport,
    PipeProtoTransport,
    RpcTransport,
    SessionFactory,
    SessionPool,
    SettlePolicy,
    create_game_session,
)
from game_bridge.sim import launch_headless_sim
from game_bridge.spectate import PolicyAdapter, SpectatorController

__all__ = [
    "FullRunSession",
    "GameSession",
    "HttpProtoJsonTransport",
    "PipeProtoTransport",
    "RpcTransport",
    "SessionFactory",
    "SessionPool",
    "SettlePolicy",
    "create_game_session",
    "launch_headless_sim",
    "SpectatorController",
    "PolicyAdapter",
    "GameCatalog",
    "GAME_CATALOG",
]
