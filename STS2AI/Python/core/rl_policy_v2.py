"""Backward-compat shim — canonical code moved to network/fullrun_policy.py.

All existing ``from core.rl_policy_v2 import X`` statements continue to work.
New code should import from:
  - network.fullrun_policy — FullRunPolicyNetworkV2, PPOTrainerV2, etc.
"""

from network.fullrun_policy import *  # noqa: F401,F403
from network.fullrun_policy import (  # noqa: F401  — explicit for underscore names
    _structured_state_to_numpy_dict,
    _structured_actions_to_numpy_dict,
)
