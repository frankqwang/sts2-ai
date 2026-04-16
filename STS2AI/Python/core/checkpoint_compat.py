"""检查点键兼容辅助函数。

混合检查点过去将战斗策略/价值网络存储在旧键 ``mcts_model`` 下。新检查点
应写入 ``combat_model`` 和 ``combat_model_config``；加载器继续接受旧键，
使现有制品保持可用。
"""

from __future__ import annotations

from typing import Any

COMBAT_MODEL_KEY = "combat_model"
COMBAT_MODEL_CONFIG_KEY = "combat_model_config"
LEGACY_MCTS_MODEL_KEY = "mcts_model"
LEGACY_MCTS_CONFIG_KEY = "mcts_config"


def get_combat_model_state(
    checkpoint: dict[str, Any],
    *,
    allow_standalone: bool = True,
) -> dict[str, Any] | None:
    """Return the combat network state dict from supported checkpoint layouts."""

    if not isinstance(checkpoint, dict):
        return None
    for key in (COMBAT_MODEL_KEY, LEGACY_MCTS_MODEL_KEY, "combat_state_dict", "mcts_state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    if allow_standalone:
        state = checkpoint.get("model_state_dict")
        if isinstance(state, dict):
            return state
    return None


def get_combat_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return combat model architecture metadata from new or legacy keys."""

    if not isinstance(checkpoint, dict):
        return {}
    config = checkpoint.get(COMBAT_MODEL_CONFIG_KEY)
    if isinstance(config, dict):
        return config
    config = checkpoint.get(LEGACY_MCTS_CONFIG_KEY)
    if isinstance(config, dict):
        return config
    return {}


def make_hybrid_checkpoint_payload(
    *,
    ppo_model: dict[str, Any],
    combat_model: dict[str, Any],
    ppo_config: dict[str, Any],
    combat_model_config: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    """Build the canonical hybrid checkpoint payload for new saves."""

    payload = {
        "checkpoint_schema_version": "hybrid.v2",
        "ppo_model": ppo_model,
        COMBAT_MODEL_KEY: combat_model,
        "ppo_config": ppo_config,
        COMBAT_MODEL_CONFIG_KEY: combat_model_config,
    }
    payload.update(extra)
    return payload
