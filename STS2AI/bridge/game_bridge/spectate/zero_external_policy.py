"""把 zero checkpoint 挂到 game_bridge.spectate external policy 上。"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


def _ensure_repo_roots() -> Path:
    sts2ai_root = Path(__file__).resolve().parents[3]
    for root in (sts2ai_root, sts2ai_root / "bridge"):
        path = str(root)
        if path not in sys.path:
            sys.path.insert(0, path)
    return sts2ai_root


_STS2AI_ROOT = _ensure_repo_roots()

from zero.adapters.game_bridge import convert_game_bridge_state
from zero.config import EncoderConfig
from zero.features import BatchCollator
from zero.model import ZeroNet
from zero.orchestration import ModelPolicyAdapter


def _read_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}")
    return value


class ZeroExternalPolicyAdapter:
    def __init__(self) -> None:
        checkpoint_path = Path(_read_required_env("STS2_ZERO_CHECKPOINT"))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"zero checkpoint 不存在: {checkpoint_path}")

        model_variant = os.environ.get("STS2_ZERO_MODEL_VARIANT", "stateless").strip() or "stateless"
        encoder = EncoderConfig()
        encoder.model_variant = model_variant

        model = ZeroNet(encoder)
        payload = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(payload["model_state"])
        model.eval()

        self._policy = ModelPolicyAdapter(model, BatchCollator(encoder), encoder.history_steps)
        self._previous_state = None
        self._previous_action_index: int | None = None

    def reset_episode(self) -> None:
        self._policy.reset_episode()
        self._previous_state = None
        self._previous_action_index = None

    def select_action(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        _context: Any,
    ) -> dict[str, Any] | None:
        enabled_legal = [
            action
            for action in legal_actions
            if isinstance(action, dict) and action.get("is_enabled") is not False
        ]
        if not enabled_legal:
            self.reset_episode()
            return None

        state_type = str(state.get("state_type") or "")
        if state_type != "combat":
            self.reset_episode()
            return dict(enabled_legal[0])

        battle_state = self._convert_enabled_combat_state(state, enabled_legal)
        if self._previous_state is not None and self._previous_action_index is not None:
            try:
                self._policy.observe_transition(self._previous_state, self._previous_action_index, battle_state)
            except Exception:
                self.reset_episode()

        action_index = int(self._policy.select_action(battle_state))
        if action_index < 0 or action_index >= len(enabled_legal):
            action_index = 0

        self._previous_state = battle_state
        self._previous_action_index = action_index
        return dict(enabled_legal[action_index])

    def _convert_enabled_combat_state(
        self,
        raw_state: dict[str, Any],
        enabled_legal: list[dict[str, Any]],
    ):
        battle_state = convert_game_bridge_state(raw_state)
        enabled_ids = {self._build_action_instance_id(action) for action in enabled_legal}
        filtered_legal = [
            action
            for action in battle_state.legal_actions
            if action.action_id in enabled_ids and action.can_execute
        ]
        if not filtered_legal:
            filtered_legal = [action for action in battle_state.legal_actions if action.can_execute]
        if not filtered_legal:
            filtered_legal = list(battle_state.legal_actions)
        return replace(
            battle_state,
            legal_actions=filtered_legal,
            raw={**battle_state.raw, "legal_actions": enabled_legal},
        )

    @staticmethod
    def _build_action_instance_id(raw: dict[str, Any]) -> str:
        return "|".join(
            [
                str(raw.get("action", "")),
                str(raw.get("index", "")),
                str(raw.get("card_index", "")),
                str(raw.get("target_id", "")),
                str(raw.get("col", "")),
                str(raw.get("row", "")),
                str(raw.get("slot", "")),
            ]
        )


_ADAPTER: ZeroExternalPolicyAdapter | None = None


def _adapter() -> ZeroExternalPolicyAdapter:
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = ZeroExternalPolicyAdapter()
    return _ADAPTER


def select_action(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    context: Any,
) -> dict[str, Any] | None:
    return _adapter().select_action(state, legal_actions, context)
