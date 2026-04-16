"""Backward-compat shim — canonical code moved to network/combat_network.py and core/combat_features.py.

All existing ``from combat_nn import X`` / ``from core.combat_nn import X``
statements continue to work. New code should import from:
  - network.combat_network — CombatPolicyValueNetwork, CombatNNEvaluator
  - core.combat_features   — build_combat_features, build_combat_action_features, constants
"""

from network.combat_network import *  # noqa: F401,F403
from core.combat_features import *  # noqa: F401,F403

# Explicit re-exports for transitive symbols callers depend on
from network.shared_encoders import CARD_AUX_DIM, MAX_ACTIONS  # noqa: F401
