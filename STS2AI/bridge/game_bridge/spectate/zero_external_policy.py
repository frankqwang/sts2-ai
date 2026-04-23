"""把 zero checkpoint 挂到 game_bridge.spectate external policy 上。"""

from __future__ import annotations

import json
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
from game_bridge.session.state_semantics import is_actionable_combat_state


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

        if not is_actionable_combat_state(state):
            self.reset_episode()
            return dict(enabled_legal[0])

        battle_state = self._convert_enabled_combat_state(state, enabled_legal)
        if self._previous_state is not None and self._previous_action_index is not None:
            try:
                self._policy.observe_transition(self._previous_state, self._previous_action_index, battle_state)
            except Exception:
                self.reset_episode()

        action_index = int(self._policy.select_action(battle_state))
        if action_index < 0 or action_index >= len(battle_state.legal_actions):
            action_index = 0

        selected_raw_action = self._resolve_enabled_action(battle_state, enabled_legal, action_index)
        self._previous_state = battle_state
        self._previous_action_index = action_index
        self._log_selection(battle_state, enabled_legal, action_index, selected_raw_action)
        return dict(selected_raw_action)

    def _convert_enabled_combat_state(
        self,
        raw_state: dict[str, Any],
        enabled_legal: list[dict[str, Any]],
    ):
        battle_state = convert_game_bridge_state(raw_state)
        legal_by_id: dict[str, list[Any]] = {}
        for action in battle_state.legal_actions:
            if not action.can_execute:
                continue
            legal_by_id.setdefault(action.action_id, []).append(action)
        filtered_legal: list[Any] = []
        for raw_action in enabled_legal:
            action_id = self._build_action_instance_id(raw_action)
            matched_actions = legal_by_id.get(action_id) or []
            if matched_actions:
                filtered_legal.append(matched_actions.pop(0))
        if not filtered_legal:
            filtered_legal = [action for action in battle_state.legal_actions if action.can_execute]
        if not filtered_legal:
            filtered_legal = list(battle_state.legal_actions)
        return replace(
            battle_state,
            legal_actions=filtered_legal,
            raw={**battle_state.raw, "legal_actions": enabled_legal},
        )

    def _resolve_enabled_action(
        self,
        battle_state,
        enabled_legal: list[dict[str, Any]],
        action_index: int,
    ) -> dict[str, Any]:
        if not enabled_legal:
            raise ValueError("enabled_legal 不能为空")
        if 0 <= action_index < len(battle_state.legal_actions):
            selected_action = battle_state.legal_actions[action_index]
            selected_id = str(getattr(selected_action, "action_id", "") or "")
            for raw_action in enabled_legal:
                if self._build_action_instance_id(raw_action) == selected_id:
                    return raw_action
        safe_index = min(max(int(action_index), 0), len(enabled_legal) - 1)
        return enabled_legal[safe_index]

    def _log_selection(
        self,
        battle_state,
        enabled_legal: list[dict[str, Any]],
        action_index: int,
        selected_raw_action: dict[str, Any],
    ) -> None:
        try:
            metadata = getattr(getattr(battle_state, "context", None), "metadata", {}) or {}
            hand_cards = [str(getattr(card, "card_id", "") or "") for card in getattr(battle_state, "hand", [])]
            legal_action_ids = [
                str(getattr(action, "action_id", "") or "")
                for action in getattr(battle_state, "legal_actions", [])
            ]
            selected_action_id = (
                str(getattr(battle_state.legal_actions[action_index], "action_id", "") or "")
                if 0 <= action_index < len(getattr(battle_state, "legal_actions", []))
                else ""
            )
            payload = {
                "event": "visible_policy_step",
                "round_number_raw": metadata.get("round_number_raw"),
                "state_type": metadata.get("state_type"),
                "player_hp": float(getattr(getattr(battle_state, "player", None), "hp", 0.0) or 0.0),
                "enemy_hp": [float(getattr(enemy, "hp", 0.0) or 0.0) for enemy in getattr(battle_state, "enemies", [])],
                "hand": hand_cards,
                "legal_action_ids": legal_action_ids,
                "selected_action_index": int(action_index),
                "selected_action_id": selected_action_id,
                "returned_action_id": self._build_action_instance_id(selected_raw_action),
                "returned_action": selected_raw_action,
                "enabled_count": len(enabled_legal),
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        except Exception:
            return None

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
