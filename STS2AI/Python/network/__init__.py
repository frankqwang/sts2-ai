"""STS2 AI network architecture — all neural network definitions.

This package contains the complete neural network architecture for the STS2 AI:

  shared_encoders.py   — EntityEmbeddings, SetEncoder, BilinearActionScorer, etc.
                         Shared building blocks used by both combat and full-run networks.

  combat_network.py    — CombatPolicyValueNetwork + CombatNNEvaluator
                         Policy+Value network for MCTS-guided combat decisions.

  fullrun_policy.py    — FullRunPolicyNetworkV2 + PPOTrainerV2
                         Full-run policy network for non-combat screens (map, shop, etc.)

Feature engineering lives in core/ (combat_features.py, state_features.py).
"""

from network.shared_encoders import (
    EntityEmbeddings,
    SetEncoder,
    SharedTrunk,
    ScreenHead,
    SimpleScreenHead,
    BilinearActionScorer,
)
from network.combat_network import CombatPolicyValueNetwork, CombatNNEvaluator
from network.fullrun_policy import FullRunPolicyNetworkV2
