"""GameState protobuf message → 训练端兼容 dict 转换。

输出格式是 V2 训练的规范 state dict。上层 normalize / 网络输入 /
action 选择直接消费 dict。

用法::

    from game_bridge.generated import game_state_pb2 as pb
    from game_bridge.transport.proto_state_converter import game_state_to_dict

    gs = pb.GameState()
    gs.ParseFromString(raw_bytes)
    state_dict = game_state_to_dict(gs)
"""
from __future__ import annotations

from typing import Any

from game_bridge.generated import game_state_pb2 as pb
from game_bridge.session.state_semantics import COMBAT_STATE_TYPES, normalize_run_outcome


# ===================================================================
# 公开 API
# ===================================================================

def game_state_to_dict(gs: pb.GameState) -> dict[str, Any]:
    """顶层转换：GameState proto → 兼容 dict。"""
    state_type = gs.state_type or "other"
    run_outcome = normalize_run_outcome(gs.run_outcome)

    player = _convert_player(gs.player)
    legal_actions = [_convert_legal_action(a) for a in gs.legal_actions]

    state: dict[str, Any] = {
        "state_type": state_type,
        "terminal": gs.terminal,
        "run_outcome": run_outcome,
        "run": {
            "act": gs.run.act if gs.HasField("run") else 0,
            "floor": gs.run.floor if gs.HasField("run") else 0,
        },
        "legal_actions": legal_actions,
    }
    if state_type != "game_over":
        state["player"] = player

    if state_type == "map" and gs.HasField("map"):
        state["map"] = _convert_map(gs.map, player)
    elif state_type == "event" and gs.HasField("event"):
        state["event"] = _convert_event(gs.event, player)
    elif state_type == "rest_site" and gs.HasField("rest_site"):
        state["rest_site"] = _convert_rest_site(gs.rest_site, player)
    elif state_type == "shop" and gs.HasField("shop"):
        state["shop"] = _convert_shop(gs.shop, player)
    elif state_type == "treasure" and gs.HasField("treasure"):
        state["treasure"] = _convert_treasure(gs.treasure, player)
    elif state_type == "treasure":
        # 兼容旧路径：treasure 数据可能在 combat_rewards 里
        state["treasure"] = _convert_treasure_from_rewards(gs.combat_rewards, player)
    elif state_type == "combat_rewards" and gs.HasField("combat_rewards"):
        state["rewards"] = _convert_combat_rewards(gs.combat_rewards, player)
    elif state_type == "card_reward" and gs.HasField("card_reward"):
        state["card_reward"] = _convert_card_reward(gs.card_reward, player)
    elif state_type == "card_select" and gs.HasField("card_select"):
        state["card_select"] = _convert_card_select(gs.card_select, player)
    elif state_type == "relic_select" and gs.HasField("relic_select"):
        state["relic_select"] = _convert_relic_select(gs.relic_select, player)
    elif state_type in COMBAT_STATE_TYPES and gs.HasField("battle"):
        battle = _convert_battle(gs.battle, player)
        state["battle"] = battle
        # combat 场景下，上游通常直接读顶层 player。这里同步 battle.player
        # 的合并结果，确保 deck/relics/powers 等字段口径一致。
        state["player"] = battle.get("player", player)
        state["enemies"] = battle.get("enemies", [])
        state["round_number_raw"] = battle.get("round_number_raw")

    _decorate_action_labels(state)
    return state


# ===================================================================
# Player
# ===================================================================

def _convert_player(p: pb.PlayerState) -> dict[str, Any]:
    """PlayerState proto → 兼容 dict。"""
    hp = p.hp
    deck = []
    for i, c in enumerate(p.deck):
        card_id = c.id
        card_type = c.card_type.lower() if c.card_type else ""
        rarity = c.rarity.lower() if c.rarity else ""
        if card_type == "unknown":
            card_type = ""
        if rarity == "unknown":
            rarity = ""
        deck.append({
            "index": i,  # C# 端按顺序写入，用 enumerate index
            "id": card_id,
            "name": card_id,
            "label": card_id,
            "cost": c.cost,
            "type": card_type,
            "rarity": rarity,
            "is_upgraded": c.is_upgraded,
            "upgrades": 1 if c.is_upgraded else 0,
        })

    relics = [{"index": i, "id": r.id, "name": r.id}
              for i, r in enumerate(p.relics)]
    potions = [{"index": i, "id": pt.id, "name": pt.id}
               for i, pt in enumerate(p.potions)]
    powers = [{"id": pw.id, "amount": pw.amount}
              for pw in p.powers if pw.amount]

    return {
        "hp": hp,
        "current_hp": hp,
        "max_hp": p.max_hp,
        "block": p.block,
        "gold": p.gold,
        "energy": p.energy,
        "max_energy": p.max_energy,
        "draw_pile_count": p.draw_pile_count,
        "discard_pile_count": p.discard_pile_count,
        "exhaust_pile_count": p.exhaust_pile_count,
        "play_pile_count": p.play_pile_count,
        "open_potion_slots": p.open_potion_slots,
        "deck": deck,
        "relics": relics,
        "potions": potions,
        "max_potions": p.max_potions,
        "powers": powers,
    }


# ===================================================================
# Legal Actions
# ===================================================================

def _convert_legal_action(a: pb.LegalAction) -> dict[str, Any]:
    """LegalAction proto → 兼容 dict。

    注意: proto3 int32 默认值为 0。C# 端用 -1 表示 "未设置"，
    所以 >=0 的判断是正确的（0 是合法 index）。
    """
    action_name = a.action or "other"
    action = {
        "action": action_name,
        "type": action_name,
        "index": a.index if a.index >= 0 else None,
        "card_index": a.card_index if a.card_index >= 0 else None,
        "target_id": a.target_id if a.target_id >= 0 else None,
        "col": a.col if a.col >= 0 else None,
        "row": a.row if a.row >= 0 else None,
        "slot": a.slot if a.slot >= 0 else None,
        "is_enabled": True,
    }
    if a.label:
        action["label"] = a.label
    if a.card_id:
        action["card_id"] = a.card_id
    return action


# ===================================================================
# Battle (combat)
# ===================================================================

def _convert_battle(b: pb.BattleState, top_player: dict[str, Any]) -> dict[str, Any]:
    """BattleState proto → 兼容 dict。"""
    # battle.player 覆盖顶层 player 的战斗时数据
    battle_player = dict(top_player)
    if b.HasField("player"):
        bp = b.player
        hp = bp.hp
        battle_player.update({
            "hp": hp,
            "current_hp": hp,
            "max_hp": bp.max_hp,
            "block": bp.block,
            "energy": b.energy,
            "max_energy": b.max_energy,
            "stars": bp.stars,
        })
        battle_player["powers"] = [{"id": pw.id, "amount": pw.amount}
                                   for pw in bp.powers if pw.amount]

    hand = []
    for i, hc in enumerate(b.hand):
        card = _convert_hand_card(hc, i)
        hand.append(card)
    battle_player["hand"] = hand

    enemies = [_convert_enemy(e) for e in b.enemies]

    return {
        "round_number_raw": b.round_number,
        "turn": b.turn_side or "unknown",
        "turn_side": b.turn_side or "unknown",
        "is_play_phase": b.is_play_phase,
        "can_end_turn": b.can_end_turn,
        "player": battle_player,
        "hand": hand,
        "enemies": enemies,
        "energy": b.energy,
        "max_energy": b.max_energy,
        "draw_pile_cards": list(b.draw_pile_cards),
        "discard_pile_cards": list(b.discard_pile_cards),
        "exhaust_pile_cards": list(b.exhaust_pile_cards),
    }


def _convert_hand_card(hc: pb.HandCard, fallback_index: int) -> dict[str, Any]:
    """HandCard proto → 兼容 dict。"""
    card_id = hc.id
    card_type = hc.card_type.lower() if hc.card_type else ""
    rarity = hc.rarity.lower() if hc.rarity else ""
    if card_type == "unknown":
        card_type = ""
    if rarity == "unknown":
        rarity = ""
    # 2026-04-24 新增：sim 直出的动态真实信息
    description = hc.description or ""
    keywords = list(hc.keywords) if hc.keywords else []
    preview_damage = dict(hc.preview_damage_per_target) if hc.preview_damage_per_target else {}
    preview_block = hc.preview_block
    return {
        "index": fallback_index,  # 用调用方传入的序号，避免 proto3 零值歧义
        "id": card_id,
        "name": card_id,
        "label": card_id,
        "cost": hc.cost,
        "type": card_type,
        "rarity": rarity,
        "target_type": hc.target_type or None,
        "is_upgraded": hc.is_upgraded,
        "upgrades": 1 if hc.is_upgraded else 0,
        "can_play": hc.can_play,
        "requires_target": hc.requires_target,
        "valid_target_ids": list(hc.valid_target_ids),
        "description": description,
        "keywords": keywords,
        "preview_damage_per_target": preview_damage,
        "preview_block": preview_block,
    }


def _convert_enemy(e: pb.Enemy) -> dict[str, Any]:
    """Enemy proto → 兼容 dict（含冗余别名字段）。"""
    enemy_id = e.id
    hp = e.hp
    powers = [{"id": pw.id, "amount": pw.amount} for pw in e.powers if pw.amount]
    intents = [_convert_intent(it) for it in e.intents]

    return {
        "id": enemy_id,
        "entity_id": enemy_id,
        "monster_id": enemy_id,
        "name": e.name or enemy_id,
        "combat_id": e.combat_id,
        "target_id": e.combat_id,
        "hp": hp,
        "current_hp": hp,
        "max_hp": e.max_hp,
        "block": e.block,
        "is_alive": e.is_alive,
        "is_hittable": e.is_hittable,
        "intends_to_attack": e.intends_to_attack,
        "next_move_id": e.next_move_id or None,
        "status": powers,
        "powers": powers,
        "buffs": powers,
        "intents": intents,
        "intent_type": intents[0]["type"] if intents else "unknown",
        "intent_damage": intents[0]["damage"] if intents else 0,
        "intent_hits": intents[0]["hits"] if intents else 1,
    }


def _convert_intent(it: pb.Intent) -> dict[str, Any]:
    return {
        "type": it.type or "unknown",
        "label": it.label or it.type or "unknown",
        "damage": it.damage,
        "total_damage": it.total_damage,
        "hits": max(1, it.hits),
    }


# ===================================================================
# Map
# ===================================================================

def _convert_map(m: pb.MapState, player: dict[str, Any]) -> dict[str, Any]:
    options = []
    for opt in m.next_options:
        point_type = opt.point_type or "unknown"
        options.append({
            "index": opt.index,
            "col": opt.col,
            "row": opt.row,
            "point_type": point_type,
            "type": point_type,
            "label": opt.label or point_type,
        })

    nodes = []
    for n in m.nodes:
        children = [[ch.col, ch.row] for ch in n.children]
        nodes.append({
            "col": n.col,
            "row": n.row,
            "type": n.type or "unknown",
            "children": children,
        })

    boss = {"col": m.boss.col, "row": m.boss.row} if m.HasField("boss") else {"col": 0, "row": 0}

    return {
        "next_options": options,
        "nodes": nodes,
        "boss": boss,
        "player": player,
    }


# ===================================================================
# Event
# ===================================================================

def _convert_event(ev: pb.EventState, player: dict[str, Any]) -> dict[str, Any]:
    options = []
    for i, opt in enumerate(ev.options):
        label = opt.label or opt.text or f"option_{i}"
        options.append({
            "index": i,
            "text": opt.text or None,
            "label": label,
            "is_locked": opt.is_locked,
            "is_chosen": opt.is_chosen,
            "is_proceed": opt.is_proceed,
        })
    return {
        "event_id": ev.event_id or "",
        "in_dialogue": ev.in_dialogue,
        "is_finished": ev.is_finished,
        "options": options,
        "player": player,
    }


# ===================================================================
# Rest Site
# ===================================================================

def _convert_rest_site(rs: pb.RestSiteState, player: dict[str, Any]) -> dict[str, Any]:
    options = []
    for i, opt in enumerate(rs.options):
        option_id = opt.id or "other"
        options.append({
            "index": i,
            "id": option_id,
            "name": opt.name or option_id,
            "is_enabled": opt.is_enabled,
        })
    return {"options": options, "can_proceed": rs.can_proceed, "player": player}


# ===================================================================
# Shop
# ===================================================================

def _convert_shop(sh: pb.ShopState, player: dict[str, Any]) -> dict[str, Any]:
    items = []
    for i, it in enumerate(sh.items):
        category = it.category or "unknown"
        item: dict[str, Any] = {
            "index": i,
            "category": category,
            "cost": it.cost,
            "price": it.cost,
            "can_afford": it.can_afford,
            "is_stocked": it.is_stocked,
            "on_sale": it.on_sale,
            "name": it.name or it.id or category,
            "id": it.id or "",
        }
        if category == "card":
            item["card_id"] = it.id
        elif category == "relic":
            item["relic_id"] = it.id
        elif category == "potion":
            item["potion_id"] = it.id
        items.append(item)
    return {"is_open": sh.is_open, "can_proceed": sh.can_proceed, "items": items, "player": player}


# ===================================================================
# Treasure（proto 里暂无独立 message，从 CombatRewardsState 转换）
# ===================================================================

def _convert_treasure_from_rewards(cr: pb.CombatRewardsState, player: dict[str, Any]) -> dict[str, Any]:
    relics = []
    for i, item in enumerate(cr.items):
        if item.type in ("relic", ""):
            relics.append({
                "index": i,
                "id": item.id or "",
                "name": item.id or "",
                "rarity": None,
            })
    return {"can_proceed": cr.can_proceed, "relics": relics, "player": player}


# ===================================================================
# Combat Rewards
# ===================================================================

def _convert_combat_rewards(cr: pb.CombatRewardsState, player: dict[str, Any]) -> dict[str, Any]:
    items = []
    for i, item in enumerate(cr.items):
        items.append({
            "index": i,
            "type": item.type or "unknown",
            "label": item.label or item.type or "unknown",
            "id": None,
            "reward_key": None,
            "reward_source": None,
            "claimable": item.claimable,
            "claim_block_reason": None,
        })
    return {"can_proceed": cr.can_proceed, "items": items, "player": player}


# ===================================================================
# Card Reward
# ===================================================================

def _convert_card_reward(cr: pb.CardRewardState, player: dict[str, Any]) -> dict[str, Any]:
    cards = []
    for i, c in enumerate(cr.cards):
        cards.append(_convert_hand_card(c, i))
    return {"can_skip": cr.can_skip, "cards": cards, "player": player}


# ===================================================================
# Card Select
# ===================================================================

def _convert_card_select(cs: pb.CardSelectState, player: dict[str, Any]) -> dict[str, Any]:
    cards = [_convert_hand_card(c, i) for i, c in enumerate(cs.cards)]
    selected_cards = [_convert_hand_card(c, i) for i, c in enumerate(cs.selected_cards)]
    return {
        "screen_type": cs.screen_type or "card_select",
        "selected_count": cs.selected_count,
        "can_confirm": cs.can_confirm,
        "can_cancel": cs.can_cancel,
        "cards": cards,
        "selected_cards": selected_cards,
        "player": player,
    }


# ===================================================================
# Relic Select
# ===================================================================

def _convert_relic_select(rs: pb.RelicSelectState, player: dict[str, Any]) -> dict[str, Any]:
    relics = [{"index": i, "id": r.id or "", "name": r.id or "", "rarity": None}
              for i, r in enumerate(rs.relics)]
    return {"can_skip": rs.can_skip, "relics": relics, "player": player}


# ===================================================================
# Treasure
# ===================================================================

def _convert_treasure(ts: pb.TreasureState, player: dict[str, Any]) -> dict[str, Any]:
    relics = [{"index": i, "id": r.id or "", "name": r.id or "", "rarity": None}
              for i, r in enumerate(ts.relics)]
    return {"can_proceed": ts.can_proceed, "relics": relics, "player": player}


# ===================================================================
# Action Label 装饰(历史上由 binary_pipe_client._decorate_action_labels 做,已删)
# ===================================================================

def _decorate_action_labels(state: dict[str, Any]) -> None:
    """给 legal_actions 的每个 action 补充人类可读 label。"""
    map_options = (state.get("map") or {}).get("next_options") or []
    reward_items = (state.get("rewards") or {}).get("items") or []
    reward_cards = (state.get("card_reward") or {}).get("cards") or []
    card_select_cards = (state.get("card_select") or {}).get("cards") or []
    relic_select_items = (state.get("relic_select") or {}).get("relics") or []
    treasure_relics = (state.get("treasure") or {}).get("relics") or []
    shop_items = (state.get("shop") or {}).get("items") or []
    rest_items = (state.get("rest_site") or {}).get("options") or []
    event_items = (state.get("event") or {}).get("options") or []
    hand_cards = (state.get("battle") or {}).get("hand") or []

    for action in state.get("legal_actions") or []:
        action_name = str(action.get("action") or "")
        index = action.get("index")
        card_index = action.get("card_index")
        label = action_name

        if action_name == "choose_map_node":
            opt = _get_indexed_item(map_options, index)
            label = str((opt or {}).get("label") or (opt or {}).get("point_type") or action_name)
        elif action_name == "claim_reward":
            reward = _get_indexed_item(reward_items, index)
            label = str((reward or {}).get("label") or (reward or {}).get("type") or action_name)
        elif action_name == "select_card_reward":
            rc = _get_indexed_item(reward_cards, index)
            label = str((rc or {}).get("id") or action_name)
        elif action_name in {"select_card", "combat_select_card", "select_card_option"}:
            label = str(
                (_get_indexed_item(card_select_cards, index)
                 or _get_indexed_item(hand_cards, card_index)
                 or _get_indexed_item(hand_cards, index)
                 or {}).get("id")
                or action_name
            )
        elif action_name == "play_card":
            label = str(
                (_get_indexed_item(hand_cards, card_index)
                 or _get_indexed_item(hand_cards, index)
                 or {}).get("id")
                or action_name
            )
        elif action_name == "select_relic":
            label = str(
                (_get_indexed_item(relic_select_items, index)
                 or _get_indexed_item(treasure_relics, index)
                 or {}).get("id")
                or action_name
            )
        elif action_name == "shop_purchase":
            it = _get_indexed_item(shop_items, index)
            label = str((it or {}).get("name") or (it or {}).get("id")
                        or (it or {}).get("category") or action_name)
        elif action_name == "choose_rest_option":
            opt = _get_indexed_item(rest_items, index)
            label = str((opt or {}).get("id") or (opt or {}).get("name") or action_name)
        elif action_name == "choose_event_option":
            label = str((_get_indexed_item(event_items, index) or {}).get("label") or f"option_{index}")
        elif action_name == "claim_treasure_relic":
            label = str((_get_indexed_item(treasure_relics, index) or {}).get("id") or action_name)

        action["label"] = label


def _get_indexed_item(items: list[dict[str, Any]], index: Any) -> dict[str, Any] | None:
    """按 index 查找列表项（兼容乱序 index）。"""
    if not isinstance(index, int) or index < 0:
        return None
    if index < len(items):
        item = items[index]
        if isinstance(item, dict) and item.get("index", index) == index:
            return item
    for item in items:
        if isinstance(item, dict) and item.get("index") == index:
            return item
    return None
