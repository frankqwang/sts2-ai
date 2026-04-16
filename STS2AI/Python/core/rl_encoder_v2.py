"""Backward-compat shim — canonical code moved to network/shared_encoders.py and core/state_features.py.

All existing ``from core.rl_encoder_v2 import X`` statements continue to work.
New code should import from:
  - network.shared_encoders — EntityEmbeddings, SetEncoder, BilinearActionScorer, etc.
  - core.state_features    — build_structured_state, constants, feature functions
"""

# Re-export everything from state_features (public + private names)
from core.state_features import *  # noqa: F401,F403
from core.state_features import (  # noqa: F401  — explicit for underscore names
    _card_aux_features, _cached_card_encoding, _cached_card_idx,
    _cached_relic_idx, _cached_potion_idx, _cached_monster_idx,
    _enemy_aux_features, _lower, _safe_float, _safe_int,
    _extract_player, _compute_route_features, _extract_map_paths,
    _card_reward_cards_from_state_or_actions,
    _text_token_id, _get_card_tags, _get_relic_tags,
    _bounded_cache_put, _get_enemy_power, _enemy_is_minion,
    _player_richness,
)

# Re-export NN modules
from network.shared_encoders import (  # noqa: F401
    EntityEmbeddings,
    SetEncoder,
    SharedTrunk,
    ScreenHead,
    SimpleScreenHead,
    BilinearActionScorer,
)
