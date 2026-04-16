"""战斗诊断和追踪函数 —— 日志、摘要和 MCTS 分析。

从 train_hybrid.py 提取。这些函数纯粹用于诊断：它们产出日志输出、
中英文摘要和 MCTS 可疑分析，但不影响训练逻辑或动作选择。

由 train_hybrid.py 的 collect_unified_episode() 用于追踪记录。
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path

from network.state_features import _lower, _safe_float, _safe_int, _extract_player
from network.state_features import COMBAT_SCREENS
from search.mcts_core import action_key

logger = logging.getLogger(__name__)


def _lower_text(value) -> str:
    """Normalize to lower-cased string (duplicated from train_hybrid for decoupling)."""
    return str(value or "").strip().lower()


_COMBAT_HARD_STATE_WEIGHTS = {
    "boss_low_hp": 3.0,
    "lethal_risk": 2.5,
    "elite": 2.0,
    "multi_enemy": 1.5,
    "high_round": 1.2,
}

# Module-level cache for trace name maps (loaded lazily)
_TRACE_NAME_MAPS: dict[str, dict] | None = None



def _load_trace_name_maps() -> dict[str, dict[str, str]]:
    global _TRACE_NAME_MAPS
    if _TRACE_NAME_MAPS is not None:
        return _TRACE_NAME_MAPS
    maps: dict[str, dict[str, str]] = {
        "cards": {},
        "encounters": {},
        "relics": {},
        "campfire": {},
    }
    db_path = Path(__file__).resolve().parents[1] / "Assets" / "datasets" / "skada" / "skada_analytics.sqlite"
    if not db_path.exists():
        _TRACE_NAME_MAPS = maps
        return maps
    try:
        conn = sqlite3.connect(db_path)
        try:
            for card_id, name_zh in conn.execute("SELECT card_id, name_zh FROM cards"):
                key = str(card_id or "").strip().upper()
                val = str(name_zh or "").strip()
                if key and val:
                    maps["cards"][key] = val
            for encounter, name_zh in conn.execute("SELECT encounter, name_zh FROM encounters"):
                key = str(encounter or "").strip().upper()
                val = str(name_zh or "").strip()
                if key and val:
                    maps["encounters"][key] = val
            for relic_id, name_zh in conn.execute("SELECT relic_id, name_zh FROM relics"):
                key = str(relic_id or "").strip().upper()
                val = str(name_zh or "").strip()
                if key and val:
                    maps["relics"][key] = val
            for action, name_zh in conn.execute("SELECT action, name_zh FROM campfire_decisions"):
                key = str(action or "").strip().upper()
                val = str(name_zh or "").strip()
                if key and val:
                    maps["campfire"][key] = val
        finally:
            conn.close()
    except Exception:
        pass
    _TRACE_NAME_MAPS = maps
    return maps



def _trace_pretty_token(token: Any) -> str:
    text = str(token or "").strip()
    if not text:
        return "未知"
    if all(ch.isupper() or ch.isdigit() or ch == "_" for ch in text):
        return " ".join(part.capitalize() for part in text.split("_") if part)
    return text



def _trace_resolve_name(token: Any, *, category: str = "generic") -> str:
    text = str(token or "").strip()
    if not text:
        return "未知"
    key = text.upper()
    maps = _load_trace_name_maps()
    if category == "card":
        return maps["cards"].get(key) or _trace_pretty_token(text)
    if category == "encounter":
        return maps["encounters"].get(key) or _trace_pretty_token(text)
    if category == "relic":
        return maps["relics"].get(key) or _trace_pretty_token(text)
    if category == "campfire":
        return maps["campfire"].get(key) or _trace_pretty_token(text)
    generic_known = {
        "monster": "普通战斗",
        "unknown": "事件",
        "rest_site": "火堆",
        "treasure": "宝箱",
        "shop": "商店",
        "elite": "精英",
        "boss": "Boss",
        "skip_card_reward": "跳过卡奖",
        "rest": "休息",
        "smith": "锻造",
        "proceed": "离开/继续",
        "remove_card": "删牌",
        "play_card": "出牌",
        "use_potion": "使用药水",
        "end_turn": "结束回合",
    }
    return (
        maps["cards"].get(key)
        or maps["encounters"].get(key)
        or maps["relics"].get(key)
        or generic_known.get(text.lower())
        or _trace_pretty_token(text)
    )



def _combat_player_view(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle")
    if isinstance(battle, dict):
        player = battle.get("player")
        if isinstance(player, dict):
            return player
    player = state.get("player")
    return player if isinstance(player, dict) else {}



def _combat_hand_summary(state: dict[str, Any], max_cards: int = 6) -> str:
    battle = state.get("battle") or {}
    hand = battle.get("hand") or _combat_player_view(state).get("hand") or []
    if not isinstance(hand, list) or not hand:
        return "-"
    parts: list[str] = []
    for card in hand[:max_cards]:
        if not isinstance(card, dict):
            continue
        name = str(card.get("name") or card.get("label") or card.get("id") or "?")
        cost = card.get("cost")
        parts.append(f"{name}({cost})")
    more = "" if len(hand) <= max_cards else f"+{len(hand) - max_cards}"
    return ",".join(parts) + more



def _combat_enemy_intent_summary(state: dict[str, Any], max_enemies: int = 3) -> str:
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    if not isinstance(enemies, list) or not enemies:
        return "-"
    parts: list[str] = []
    for enemy in enemies[:max_enemies]:
        if not isinstance(enemy, dict):
            continue
        name = str(enemy.get("name") or enemy.get("id") or "?")
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        block = _safe_int(enemy.get("block", 0), 0)
        intents = enemy.get("intents") or []
        if isinstance(intents, list) and intents:
            intent0 = intents[0] if isinstance(intents[0], dict) else {}
        else:
            intent0 = {}
        intent_type = str(intent0.get("type") or intent0.get("label") or "?")
        dmg = _safe_int(intent0.get("damage", 0), 0)
        hits = max(1, _safe_int(intent0.get("hits", 1), 1))
        dmg_str = f"{dmg}x{hits}" if dmg > 0 else intent_type
        parts.append(f"{name}[{hp}/{block}:{dmg_str}]")
    more = "" if len(enemies) <= max_enemies else f"+{len(enemies) - max_enemies}"
    return ",".join(parts) + more



def _combat_enemy_group_key(state: dict[str, Any]) -> str:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    names: list[str] = []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        name = str(enemy.get("id") or enemy.get("name") or "").strip().upper()
        if name:
            names.append(name)
    return "+".join(names) if names else "UNKNOWN"



def _combat_enemy_map(state: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    out: dict[Any, dict[str, Any]] = {}
    for idx, enemy in enumerate(enemies):
        if not isinstance(enemy, dict):
            continue
        key = enemy.get("combat_id", enemy.get("target_id", enemy.get("entity_id", idx)))
        out[key] = enemy
    return out



def _combat_enemy_change_items(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    action: dict[str, Any] | None,
    *,
    max_items: int = 6,
) -> list[dict[str, Any]]:
    pre_map = _combat_enemy_map(pre_state)
    post_map = _combat_enemy_map(post_state)
    target_key = None if not isinstance(action, dict) else action.get("target_id", action.get("enemy_id", action.get("target")))
    items: list[dict[str, Any]] = []
    for key, pre_enemy in pre_map.items():
        post_enemy = post_map.get(key)
        pre_hp = _safe_int(pre_enemy.get("hp", pre_enemy.get("current_hp", 0)), 0)
        pre_blk = _safe_int(pre_enemy.get("block", 0), 0)
        name = _trace_resolve_name((post_enemy or pre_enemy).get("name") or (post_enemy or pre_enemy).get("id"), category="encounter")
        if post_enemy is None:
            items.append({
                "key": key,
                "name": name,
                "pre_hp": pre_hp,
                "post_hp": 0,
                "pre_block": pre_blk,
                "post_block": 0,
                "defeated": True,
                "targeted": key == target_key,
            })
            continue
        post_hp = _safe_int(post_enemy.get("hp", post_enemy.get("current_hp", 0)), 0)
        post_blk = _safe_int(post_enemy.get("block", 0), 0)
        if pre_hp != post_hp or pre_blk != post_blk or key == target_key:
            items.append({
                "key": key,
                "name": name,
                "pre_hp": pre_hp,
                "post_hp": post_hp,
                "pre_block": pre_blk,
                "post_block": post_blk,
                "defeated": False,
                "targeted": key == target_key,
            })
    return items[:max_items]



def _trace_enemy_change_summary(pre_state: dict[str, Any], post_state: dict[str, Any], action: dict[str, Any] | None) -> str:
    items = _combat_enemy_change_items(pre_state, post_state, action, max_items=3)
    parts: list[str] = []
    for item in items:
        if item.get("defeated"):
            parts.append(f"{item['name']} 被击败")
        else:
            parts.append(
                f"{item['name']} 血量 {item['pre_hp']}->{item['post_hp']}，"
                f"格挡 {item['pre_block']}->{item['post_block']}"
            )
    return "；".join(parts) if parts else "敌方数值无明显变化"



def _combat_step_structured_summary(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    action: dict[str, Any] | None,
) -> dict[str, Any]:
    pre_player = _combat_player_view(pre_state)
    post_player = _combat_player_view(post_state)
    target_key = None if not isinstance(action, dict) else action.get("target_id", action.get("enemy_id", action.get("target")))
    target_enemy = _combat_enemy_map(pre_state).get(target_key)
    return {
        "pre_hp": _safe_int(pre_player.get("hp", pre_player.get("current_hp", 0)), 0),
        "post_hp": _safe_int(post_player.get("hp", post_player.get("current_hp", 0)), 0),
        "pre_block": _safe_int(pre_player.get("block", 0), 0),
        "post_block": _safe_int(post_player.get("block", 0), 0),
        "pre_energy": _safe_int(pre_player.get("energy", 0), 0),
        "post_energy": _safe_int(post_player.get("energy", 0), 0),
        "target_key": target_key,
        "target_name": (
            _trace_resolve_name(target_enemy.get("name") or target_enemy.get("id"), category="encounter")
            if isinstance(target_enemy, dict)
            else ""
        ),
        "enemy_changes": _combat_enemy_change_items(pre_state, post_state, action),
        "pre_intent": _combat_enemy_intent_summary(pre_state),
        "pre_intent_zh": _combat_enemy_intent_summary_zh(pre_state),
        "next_intent": _combat_enemy_intent_summary(post_state),
        "next_intent_zh": _combat_enemy_intent_summary_zh(post_state),
        "post_state_type": _lower_text(post_state.get("state_type")),
    }



def _combat_target_label_zh(action: dict[str, Any] | None, state: dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return ""
    target_key = action.get("target_id", action.get("enemy_id", action.get("target")))
    if target_key in (None, ""):
        return ""
    enemy = _combat_enemy_map(state).get(target_key)
    if isinstance(enemy, dict):
        name = _trace_resolve_name(enemy.get("name") or enemy.get("id"), category="encounter")
        return f" -> 目标「{name}」"
    return f" -> 目标 `{target_key}`"



def _combat_result_summary_zh(pre_state: dict[str, Any], post_state: dict[str, Any], action: dict[str, Any] | None) -> str:
    pre_player = _combat_player_view(pre_state)
    post_player = _combat_player_view(post_state)
    pre_hp = _safe_int(pre_player.get("hp", pre_player.get("current_hp", 0)), 0)
    post_hp = _safe_int(post_player.get("hp", post_player.get("current_hp", 0)), 0)
    pre_blk = _safe_int(pre_player.get("block", 0), 0)
    post_blk = _safe_int(post_player.get("block", 0), 0)
    pre_energy = _safe_int(pre_player.get("energy", 0), 0)
    post_energy = _safe_int(post_player.get("energy", 0), 0)
    enemy_delta = _trace_enemy_change_summary(pre_state, post_state, action)
    next_intent = _combat_enemy_intent_summary_zh(post_state)
    if _lower_text(post_state.get("state_type")) not in COMBAT_SCREENS:
        next_intent = "敌人全部击败，战斗结束"
    return (
        f"结果：我方生命 {pre_hp}->{post_hp}，格挡 {pre_blk}->{post_blk}，能量 {pre_energy}->{post_energy}；"
        f"{enemy_delta}；下拍：{next_intent}"
    )



def _action_target_summary(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    for key in ("target", "target_id", "enemy_id", "slot"):
        value = action.get(key)
        if value not in (None, ""):
            return f" target={value}"
    return ""



def _topk_action_summary(
    legal: list[dict[str, Any]],
    logits_or_probs: np.ndarray | list[float] | torch.Tensor,
    k: int = 3,
    already_probs: bool = False,
) -> str:
    if not legal:
        return "-"
    if isinstance(logits_or_probs, torch.Tensor):
        arr = logits_or_probs.detach().float().cpu().numpy()
    else:
        arr = np.asarray(logits_or_probs, dtype=np.float64)
    arr = np.ravel(arr)
    if arr.size == 0:
        return "-"
    arr = arr[:len(legal)]
    if arr.size == 0:
        return "-"
    if already_probs:
        probs = arr
    else:
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        denom = np.sum(exp)
        probs = exp / denom if denom > 0 else np.zeros_like(arr)
    order = np.argsort(-probs)[: min(k, len(legal))]
    parts: list[str] = []
    for idx in order:
        action = legal[int(idx)]
        label = str(action.get("label") or action.get("action") or idx)
        parts.append(f"{label}:{float(probs[int(idx)]):.2f}")
    return " | ".join(parts)



def _combat_action_label(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return "?"
    return str(action.get("label") or action.get("action") or "?")



def _combat_action_name(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("action") or action.get("type") or "").strip().lower()



def _combat_is_end_turn_action(action: dict[str, Any] | None) -> bool:
    return _combat_action_name(action) == "end_turn"



def _combat_is_use_potion_action(action: dict[str, Any] | None) -> bool:
    return _combat_action_name(action) == "use_potion"



def _combat_action_looks_defensive(action: dict[str, Any] | None) -> bool:
    label = _combat_action_label(action).strip().lower()
    if not label or _combat_is_end_turn_action(action) or _combat_is_use_potion_action(action):
        return False
    defense_tokens = (
        "defend",
        "block",
        "barrier",
        "armor",
        "shield",
        "shrug",
        "panic",
        "ghostly",
        "power through",
        "flame barrier",
        "iron wave",
        "entrench",
    )
    return any(token in label for token in defense_tokens)



def _combat_action_looks_attack(action: dict[str, Any] | None) -> bool:
    if _combat_action_name(action) != "play_card":
        return False
    if _combat_action_looks_defensive(action):
        return False
    if any(action.get(key) not in (None, "") for key in ("target", "target_id", "enemy_id")):
        return True
    label = _combat_action_label(action).strip().lower()
    attack_tokens = (
        "strike",
        "bash",
        "anger",
        "boomerang",
        "slash",
        "whirlwind",
        "pummel",
        "headbutt",
        "uppercut",
        "hemokinesis",
        "carnage",
        "bludgeon",
        "perfected",
        "clothesline",
        "dropkick",
        "pommel",
        "twin",
        "sword",
        "sever",
        "thunderclap",
        "wild strike",
        "body slam",
    )
    return any(token in label for token in attack_tokens)



def _combat_played_card_from_action(
    state: dict[str, Any],
    action: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if _combat_action_name(action) != "play_card" or not isinstance(action, dict):
        return None
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = _extract_player(state)
    hand = battle.get("hand") or player.get("hand") or []
    card_idx = _safe_int(
        action.get("card_index", action.get("hand_index", action.get("index", -1))),
        -1,
    )
    if 0 <= card_idx < len(hand) and isinstance(hand[card_idx], dict):
        return dict(hand[card_idx])
    action_label = str(action.get("label") or "").strip().lower()
    if action_label:
        for card in hand:
            if not isinstance(card, dict):
                continue
            card_name = str(card.get("name") or card.get("id") or "").strip().lower()
            if card_name and card_name in action_label:
                return dict(card)
    return None



def _combat_card_type(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get("type") or card.get("card_type") or card.get("cardType") or "").strip().lower()



def _combat_card_effect_summary(card: dict[str, Any] | None) -> tuple[float, float, float, float, float, float, float]:
    if not isinstance(card, dict):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    cost = _to_float(card.get("cost_for_turn", card.get("cost", card.get("energy_cost", 0))))
    damage = _to_float(card.get("damage", card.get("base_damage", card.get("attack_damage", 0))))
    block = _to_float(card.get("block", card.get("base_block", 0)))
    draw = _to_float(card.get("draw", card.get("cards_to_draw", card.get("draw_amount", 0))))
    magic = _to_float(card.get("magic_number", card.get("magic", 0)))
    text = _lower_text(card.get("description") or card.get("raw_description") or card.get("text") or "")
    if draw <= 0 and "draw" in text:
        draw = max(draw, magic)
    discard = 1.0 if any(tok in text for tok in ("discard", "put a card from your hand")) else 0.0
    exhaust = 1.0 if card.get("exhaust") or "exhaust" in text else 0.0
    create = 1.0 if any(tok in text for tok in ("add ", "create ", "shuffle")) else 0.0
    return cost, damage, block, draw, discard, exhaust, create



def _combat_root_topk_summary(root: Any, k: int = 5) -> str:
    children_map = getattr(root, "children", None)
    if not isinstance(children_map, dict) or not children_map:
        return "-"
    children = list(children_map.values())
    total_visits = max(1, sum(max(0, int(getattr(child, "visit_count", 0))) for child in children))
    ranked = sorted(
        children,
        key=lambda child: (
            int(getattr(child, "visit_count", 0)),
            float(getattr(child, "prior", 0.0)),
            float(getattr(child, "q_value", 0.0)),
        ),
        reverse=True,
    )[: min(k, len(children))]
    parts: list[str] = []
    for child in ranked:
        label = _combat_action_label(getattr(child, "action", None))
        visits = max(0, int(getattr(child, "visit_count", 0)))
        visit_frac = visits / total_visits
        q_value = float(getattr(child, "q_value", 0.0))
        prior = float(getattr(child, "prior", 0.0))
        parts.append(f"{label}:n={visits}/{visit_frac:.2f},q={q_value:.2f},p={prior:.2f}")
    return " | ".join(parts)



def _combat_root_action_summary(root: Any, action: dict[str, Any] | None) -> str:
    children_map = getattr(root, "children", None)
    if not isinstance(children_map, dict) or not children_map or not isinstance(action, dict):
        return "chosen[missing]"
    child = children_map.get(action_key(action))
    if child is None:
        for candidate in children_map.values():
            cand_action = getattr(candidate, "action", None)
            if (
                isinstance(cand_action, dict)
                and cand_action.get("action") == action.get("action")
                and cand_action.get("label") == action.get("label")
                and cand_action.get("target") == action.get("target")
            ):
                child = candidate
                break
    if child is None:
        return "chosen[missing]"
    total_visits = max(1, sum(max(0, int(getattr(node, "visit_count", 0))) for node in children_map.values()))
    visits = max(0, int(getattr(child, "visit_count", 0)))
    visit_frac = visits / total_visits
    q_value = float(getattr(child, "q_value", 0.0))
    prior = float(getattr(child, "prior", 0.0))
    return f"chosen[n={visits}/{visit_frac:.2f},q={q_value:.2f},p={prior:.2f}]"



def _combat_mcts_suspect_reasons(
    *,
    action: dict[str, Any] | None,
    legal: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(action, dict):
        return reasons
    player = _combat_player_view(state)
    energy = _safe_int(player.get("energy", 0), 0)
    has_attack_option = any(_combat_action_looks_attack(candidate) for candidate in legal)
    has_defense_option = any(_combat_action_looks_defensive(candidate) for candidate in legal)
    if _combat_is_end_turn_action(action) and energy > 0:
        remaining_plays = [
            candidate for candidate in legal
            if not _combat_is_end_turn_action(candidate)
            and _combat_action_name(candidate) not in {"confirm_selection", "cancel_selection"}
        ]
        if remaining_plays:
            reasons.append("end_turn_with_energy")
        if has_attack_option:
            reasons.append("end_turn_skips_attack")
        if has_defense_option:
            reasons.append("end_turn_skips_block")
    if _combat_is_use_potion_action(action):
        reasons.append("use_potion")
    if _combat_action_looks_defensive(action) and energy > 0 and has_attack_option:
        reasons.append("defense_bias")
    return reasons



def _combat_hard_state_tags(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    action: dict[str, Any] | None,
    repeat_count: int,
    turn_prefix: dict[str, Any] | None,
) -> list[str]:
    tags: list[str] = []
    if any(_combat_is_use_potion_action(candidate) for candidate in legal):
        tags.append("potion_decision")
    suspect_reasons = _combat_mcts_suspect_reasons(action=action, legal=legal, state=state)
    if any(reason.startswith("end_turn_") for reason in suspect_reasons):
        tags.append("premature_end_turn")
    if repeat_count >= 2:
        tags.append("repeat_loop_entry")
    play_card_options = sum(1 for candidate in legal if _combat_action_name(candidate) == "play_card")
    prefix_actions = _safe_int((turn_prefix or {}).get("action_count", 0), 0)
    if (
        _combat_action_name(action) == "play_card"
        and play_card_options >= 2
        and (prefix_actions > 0 or play_card_options >= 3)
    ):
        tags.append("order_sensitive_play")
    return tags



def _combat_hard_state_weight(tags: list[str]) -> float:
    weight = 1.0
    for tag in tags:
        weight = max(weight, float(_COMBAT_HARD_STATE_WEIGHTS.get(tag, 1.0)))
    return weight



def _combat_room_conditioned_continuation_loss(
    continuation_pred: torch.Tensor,
    continuation_target: torch.Tensor,
    room_type_onehot: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train continuation targets with explicit survival/cost semantics.

    Column layout:
    - [:, 0] win_prob
    - [:, 1] expected_hp_loss
    - [:, 2] expected_potion_cost
    """
    if room_type_onehot is None:
        room_type_onehot = torch.zeros(
            continuation_pred.shape[0], 3,
            dtype=continuation_pred.dtype, device=continuation_pred.device,
        )
        room_type_onehot[:, 0] = 1.0
    room_type_onehot = room_type_onehot.to(dtype=continuation_pred.dtype, device=continuation_pred.device)
    hallway = room_type_onehot[:, 0]
    elite = room_type_onehot[:, 1]
    boss = room_type_onehot[:, 2]

    survival_loss = F.binary_cross_entropy(
        continuation_pred[:, 0].clamp(1e-5, 1.0 - 1e-5),
        continuation_target[:, 0].clamp(0.0, 1.0),
        reduction="none",
    )
    hp_loss = F.smooth_l1_loss(
        continuation_pred[:, 1],
        continuation_target[:, 1],
        reduction="none",
    )
    potion_loss = F.smooth_l1_loss(
        continuation_pred[:, 2],
        continuation_target[:, 2],
        reduction="none",
    )

    survival_weight = hallway * 0.9 + elite * 1.0 + boss * 1.25
    hp_weight = hallway * 1.0 + elite * 0.75 + boss * 0.10
    potion_weight = hallway * 1.0 + elite * 0.80 + boss * 0.15

    total = (
        survival_weight * survival_loss
        + hp_weight * hp_loss
        + potion_weight * potion_loss
    ).mean()
    return total, survival_loss.mean(), hp_loss.mean(), potion_loss.mean()



# --- Chinese localization functions (moved from train_hybrid.py) ---



def _combat_hand_summary_zh(state: dict[str, Any], max_cards: int = 6) -> str:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = _combat_player_view(state)
    hand = battle.get("hand") or player.get("hand") or []
    parts: list[str] = []
    for card in hand[:max_cards]:
        if not isinstance(card, dict):
            continue
        name = _trace_resolve_name(card.get("name") or card.get("id"), category="card")
        cost = card.get("cost_for_turn", card.get("cost", "?"))
        parts.append(f"{name}({cost})")
    if not parts:
        return "-"
    more = "" if len(hand) <= max_cards else f" +{len(hand) - max_cards}"
    return "，".join(parts) + more



def _combat_enemy_intent_summary_zh(state: dict[str, Any], max_enemies: int = 3) -> str:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    parts: list[str] = []
    for enemy in enemies[:max_enemies]:
        if not isinstance(enemy, dict):
            continue
        name = _trace_resolve_name(enemy.get("name") or enemy.get("id"), category="encounter")
        hp = _safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0)
        block = _safe_int(enemy.get("block", 0), 0)
        intents = enemy.get("intents") or []
        if isinstance(intents, list) and intents:
            intent0 = intents[0] if isinstance(intents[0], dict) else {}
        else:
            intent0 = {}
        intent_type = str(intent0.get("type") or intent0.get("label") or "?")
        dmg = _safe_int(intent0.get("damage", 0), 0)
        hits = max(1, _safe_int(intent0.get("hits", 1), 1))
        if dmg > 0:
            intent_desc = f"攻击 {dmg}x{hits}"
        else:
            intent_desc = _trace_pretty_token(intent_type)
        parts.append(f"{name} {hp}/{block}，意图 {intent_desc}")
    if not parts:
        return "-"
    more = "" if len(enemies) <= max_enemies else f" +{len(enemies) - max_enemies}"
    return "；".join(parts) + more



def _combat_action_label_zh(action: dict[str, Any] | None, state: dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return "未知动作"
    action_name = _combat_action_name(action)
    label = _combat_action_label(action)
    if action_name == "play_card":
        label = _trace_resolve_name(label, category="card")
        return f"打出「{label}」"
    if action_name == "use_potion":
        player = _combat_player_view(state)
        potions = player.get("potions") or []
        slot = action.get("slot")
        potion_name = None
        for potion in potions:
            if isinstance(potion, dict) and potion.get("slot") == slot:
                potion_name = potion.get("name") or potion.get("id")
                break
        potion_label = _trace_resolve_name(potion_name or label)
        return f"使用药水「{potion_label}」"
    if action_name == "end_turn":
        return "结束回合"
    return _trace_resolve_name(label)



def _topk_action_summary_zh(
    legal: list[dict[str, Any]],
    logits_or_probs: np.ndarray | list[float] | torch.Tensor,
    k: int = 3,
    already_probs: bool = False,
) -> str:
    raw = _topk_action_summary(legal, logits_or_probs, k=k, already_probs=already_probs)
    if raw == "-" or not raw:
        return raw
    parts: list[str] = []
    for chunk in raw.split(" | "):
        if ":" not in chunk:
            parts.append(_trace_resolve_name(chunk))
            continue
        label, prob = chunk.rsplit(":", 1)
        parts.append(f"{_trace_resolve_name(label, category='card')}:{prob}")
    return " | ".join(parts)

