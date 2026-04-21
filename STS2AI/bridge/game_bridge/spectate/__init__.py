"""策略无关观战层。"""

from game_bridge.spectate.controller import SpectatorController
from game_bridge.spectate.overlay import OverlayWriter
from game_bridge.spectate.policy import ExternalPolicy, ManualPolicy, NullPolicy, PolicyAdapter, ReplayPolicy

__all__ = [
    "ExternalPolicy",
    "ManualPolicy",
    "NullPolicy",
    "OverlayWriter",
    "PolicyAdapter",
    "ReplayPolicy",
    "SpectatorController",
]
