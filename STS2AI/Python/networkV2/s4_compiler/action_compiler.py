"""ActionCandidates 编译器：从 legal_actions 提取结构化动作候选。"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.actions import ActionCandidate
from networkV2.s1_schema.entities import HandCardRuntime, EnemyRuntime


def _pick(raw: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if not isinstance(raw, dict):
        return default
    for key in keys:
        if key in raw:
            return raw[key]
    return default


class ActionCompiler:
    """从 legal_actions 编译 ActionCandidate 列表。"""

    def compile(
        self,
        legal_actions: list[dict[str, Any]],
        hand_cards: list[HandCardRuntime],
        enemies: list[EnemyRuntime],
    ) -> list[ActionCandidate]:
        # 建索引
        hand_by_index = {c.hand_index: c for c in hand_cards}
        enemy_by_combat_id = {e.combat_id: e for e in enemies}

        candidates: list[ActionCandidate] = []
        for i, raw_action in enumerate(legal_actions):
            if not isinstance(raw_action, dict):
                continue
            candidates.append(
                self._compile_one(raw_action, i, hand_by_index, enemy_by_combat_id)
            )
        return candidates

    def _compile_one(
        self,
        raw: dict[str, Any],
        index: int,
        hand_by_index: dict[int, HandCardRuntime],
        enemy_by_combat_id: dict[int, EnemyRuntime],
    ) -> ActionCandidate:
        action_type = str(_pick(raw, "action", default="other") or "other").lower()
        label = str(_pick(raw, "label", default="") or "")

        candidate = ActionCandidate(
            action_type=action_type,
            action_index=index,
            label=label,
        )

        if action_type == "play_card":
            self._fill_play_card(candidate, raw, hand_by_index, enemy_by_combat_id)
        elif action_type == "end_turn":
            candidate.family = "end_turn"
            candidate.ends_turn = True
        elif action_type in ("use_potion", "drink_potion"):
            candidate.family = "use_potion"
            candidate.source_potion_id = str(_pick(raw, "potion_id", "id", default="") or "")
        elif action_type == "select_hand_card":
            candidate.family = "card_selection"
            # 不用 `or -1` 短路，否则 hand_index == 0 会被错误降级到 -1
            hand_pick = _pick(raw, "hand_index", "index", default=-1)
            candidate.hand_index = int(hand_pick if hand_pick is not None else -1)
            candidate.source_card_id = str(_pick(raw, "card_id", default="") or "").lower()
        elif action_type == "select_card_option":
            candidate.family = "card_selection"
            candidate.target_card_id = str(_pick(raw, "card_id", default="") or "").lower()
        elif action_type in ("confirm_selection", "confirm"):
            candidate.family = "confirm"
        elif action_type in ("cancel_selection", "cancel"):
            candidate.family = "cancel"
        else:
            candidate.family = "other"

        return candidate

    def _fill_play_card(
        self,
        candidate: ActionCandidate,
        raw: dict[str, Any],
        hand_by_index: dict[int, HandCardRuntime],
        enemy_by_combat_id: dict[int, EnemyRuntime],
    ) -> None:
        candidate.family = "play_card"
        # 不用 `or -1` 短路，否则 hand_index == 0 会被错误降级到 -1，
        # 导致 hand_by_index 查不到卡 → damage_est/block_est/draw_est 全是 0
        hand_pick = _pick(raw, "hand_index", "card_index", "index", default=-1)
        hand_idx = int(hand_pick if hand_pick is not None else -1)
        candidate.hand_index = hand_idx
        candidate.source_card_id = str(_pick(raw, "card_id", default="") or "").lower()

        # 从 hand_cards 补充卡牌信息
        card = hand_by_index.get(hand_idx)
        if card is not None:
            candidate.source_card_type = card.card_type
            candidate.cost = card.current_cost
            candidate.is_zero_cost = card.current_cost == 0
            candidate.exhausts = card.exhaust
            candidate.retains = card.retain
            # preview 数值（原本恒 0，被 bank_assembler 写进 token numeric）
            candidate.damage_est = card.damage_est
            candidate.block_est = card.block_est
            candidate.draw_est = card.draw_est
            if not candidate.source_card_id:
                candidate.source_card_id = card.card_id
            # 语义角色推断
            roles: list[str] = []
            ct = card.card_type.lower()
            if ct == "attack":
                roles.append("attack")
            elif ct == "skill":
                roles.append("block")
            elif ct == "power":
                roles.append("buff")
            if card.exhaust:
                roles.append("exhaust")
            if card.retain:
                roles.append("retain")
            if card.ethereal:
                roles.append("ethereal")
            candidate.roles = roles

        # 目标信息
        target_id = _pick(raw, "target_id", "target_combat_id")
        if target_id is not None:
            target_combat_id = int(target_id)
            candidate.target_combat_id = target_combat_id
            candidate.target_scope = "single_enemy"
            target_enemy = enemy_by_combat_id.get(target_combat_id)
            if target_enemy is not None:
                candidate.target_enemy_id = target_enemy.entity_id
        elif card is not None and not card.requires_target:
            # 无需目标的卡：可能是 AOE 或 self
            if "aoe" in (card.keywords or []):
                candidate.target_scope = "all_enemies"
                candidate.roles.append("aoe")
            else:
                candidate.target_scope = "self"
