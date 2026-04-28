"""proto_state_converter 单元测试。

构造 proto GameState 消息 → game_state_to_dict() → 验证输出 dict 格式
是 V2 训练的规范 state 形状(历史上对齐过已删除的 binary_pipe_client 输出)。

运行: cd STS2AI/bridge && python -m pytest tests/test_proto_state_converter.py -v
"""
from __future__ import annotations

import sys
import os

# 确保 STS2AI/bridge 在 sys.path 中
_python_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _python_root not in sys.path:
    sys.path.insert(0, _python_root)

import pytest
from game_bridge.generated import game_state_pb2 as pb
from game_bridge.transport.proto_state_converter import game_state_to_dict


# ======================================================================
# Fixtures — 构造各种 GameState proto 消息
# ======================================================================

def _make_player(**overrides) -> pb.PlayerState:
    defaults = dict(
        hp=50, max_hp=80, block=5, gold=100, energy=3, max_energy=3,
        draw_pile_count=20, discard_pile_count=3, exhaust_pile_count=1,
        play_pile_count=0, open_potion_slots=2, max_potions=3,
    )
    defaults.update(overrides)
    p = pb.PlayerState(**{k: v for k, v in defaults.items()
                          if k not in ("deck", "relics", "potions", "powers")})
    for c in overrides.get("deck", []):
        p.deck.append(pb.CardInfo(**c))
    for r in overrides.get("relics", []):
        p.relics.append(pb.RelicInfo(**r))
    for pt in overrides.get("potions", []):
        p.potions.append(pb.PotionInfo(**pt))
    for pw in overrides.get("powers", []):
        p.powers.append(pb.Power(**pw))
    return p


def _make_game_state(state_type: str = "map", **kwargs) -> pb.GameState:
    gs = pb.GameState()
    gs.state_type = state_type
    gs.terminal = kwargs.get("terminal", False)
    gs.run_outcome = kwargs.get("run_outcome", "")
    gs.run.act = kwargs.get("act", 1)
    gs.run.floor = kwargs.get("floor", 5)
    gs.player.CopyFrom(kwargs.get("player", _make_player()))
    return gs


# ======================================================================
# Tests — 顶层结构
# ======================================================================

class TestTopLevelStructure:
    def test_basic_fields(self):
        gs = _make_game_state("map", act=2, floor=10)
        d = game_state_to_dict(gs)
        assert d["state_type"] == "map"
        assert d["terminal"] is False
        assert d["run_outcome"] is None
        assert d["run"] == {"act": 2, "floor": 10}

    def test_terminal_victory(self):
        gs = _make_game_state("game_over", terminal=True, run_outcome="victory")
        d = game_state_to_dict(gs)
        assert d["terminal"] is True
        assert d["run_outcome"] == "victory"

    def test_empty_run_outcome_is_none(self):
        gs = _make_game_state("map")
        d = game_state_to_dict(gs)
        assert d["run_outcome"] is None

    def test_legal_actions_present(self):
        gs = _make_game_state("map")
        la = gs.legal_actions.add()
        la.action = "choose_map_node"
        la.index = 0
        la.col = 2
        la.row = 1
        d = game_state_to_dict(gs)
        assert len(d["legal_actions"]) == 1
        a = d["legal_actions"][0]
        assert a["action"] == "choose_map_node"
        assert a["type"] == "choose_map_node"
        assert a["index"] == 0
        assert a["col"] == 2
        assert a["is_enabled"] is True


# ======================================================================
# Tests — Player
# ======================================================================

class TestPlayer:
    def test_player_basic_fields(self):
        p = _make_player(hp=50, max_hp=80, gold=200)
        gs = _make_game_state("map", player=p)
        d = game_state_to_dict(gs)
        player = d["player"]
        assert player["hp"] == 50
        assert player["current_hp"] == 50  # 冗余别名
        assert player["max_hp"] == 80
        assert player["gold"] == 200

    def test_player_deck(self):
        p = _make_player(deck=[
            {"index": 0, "id": "Strike", "name": "Strike", "cost": 1,
             "card_type": "ATTACK", "rarity": "basic", "is_upgraded": False, "upgrades": 0},
            {"index": 1, "id": "Defend", "name": "Defend", "cost": 1,
             "card_type": "SKILL", "rarity": "basic", "is_upgraded": True, "upgrades": 1},
        ])
        gs = _make_game_state("map", player=p)
        d = game_state_to_dict(gs)
        deck = d["player"]["deck"]
        assert len(deck) == 2
        assert deck[0]["id"] == "Strike"
        assert deck[0]["type"] == "attack"
        assert deck[0]["is_upgraded"] is False
        assert deck[0]["upgrades"] == 0
        assert deck[1]["is_upgraded"] is True
        assert deck[1]["upgrades"] == 1

    def test_player_relics_potions(self):
        p = _make_player(
            relics=[{"index": 0, "id": "BurningBlood", "name": "BurningBlood"}],
            potions=[{"index": 0, "id": "FirePotion", "name": "FirePotion"}],
            max_potions=3,
        )
        gs = _make_game_state("map", player=p)
        d = game_state_to_dict(gs)
        assert len(d["player"]["relics"]) == 1
        assert d["player"]["relics"][0]["id"] == "BurningBlood"
        assert len(d["player"]["potions"]) == 1
        assert d["player"]["max_potions"] == 3

    def test_player_powers(self):
        p = _make_player(
            powers=[{"id": "STRENGTH_POWER", "amount": 2}],
        )
        gs = _make_game_state("map", player=p)
        d = game_state_to_dict(gs)
        assert d["player"]["powers"] == [{"id": "STRENGTH_POWER", "amount": 2}]


# ======================================================================
# Tests — Battle (combat)
# ======================================================================

class TestBattle:
    def _make_combat_state(self) -> pb.GameState:
        gs = _make_game_state("monster")
        b = gs.battle
        b.round_number = 3
        b.turn_side = "player"
        b.is_play_phase = True
        b.can_end_turn = True
        b.energy = 3
        b.max_energy = 3
        bp = b.player
        bp.hp = 45
        bp.max_hp = 80
        bp.block = 10
        bp.energy = 3
        bp.max_energy = 3
        pw = bp.powers.add()
        pw.id = "STRENGTH_POWER"
        pw.amount = 2
        # hand
        hc = b.hand.add()
        hc.index = 0
        hc.id = "Strike"
        hc.name = "Strike"
        hc.cost = 1
        hc.card_type = "ATTACK"
        hc.rarity = "basic"
        hc.target_type = "AnyEnemy"
        hc.is_upgraded = False
        hc.can_play = True
        hc.requires_target = True
        hc.valid_target_ids.append(0)
        # enemy
        e = b.enemies.add()
        e.id = "JawWorm"
        e.combat_id = 0
        e.name = "Jaw Worm"
        e.hp = 30
        e.max_hp = 44
        e.block = 0
        e.is_alive = True
        e.is_hittable = True
        e.intends_to_attack = True
        e.next_move_id = "Chomp"
        intent = e.intents.add()
        intent.type = "attack"
        intent.label = "attack"
        intent.damage = 11
        intent.total_damage = 11
        intent.hits = 1
        epw = e.powers.add()
        epw.id = "STRENGTH_POWER"
        epw.amount = 3
        # piles
        b.draw_pile_cards.append("Strike")
        b.draw_pile_cards.append("Defend")
        b.discard_pile_cards.append("Bash")
        return gs

    def test_battle_structure(self):
        gs = self._make_combat_state()
        d = game_state_to_dict(gs)
        assert "battle" in d
        battle = d["battle"]
        assert battle["round_number_raw"] == 3
        assert battle["turn"] == "player"
        assert battle["turn_side"] == "player"
        assert battle["is_play_phase"] is True
        assert battle["can_end_turn"] is True
        assert battle["energy"] == 3

    def test_battle_player_powers(self):
        gs = self._make_combat_state()
        d = game_state_to_dict(gs)
        bp = d["battle"]["player"]
        assert bp["hp"] == 45
        assert bp["current_hp"] == 45
        assert len(bp["powers"]) == 1
        assert bp["powers"][0]["id"] == "STRENGTH_POWER"
        assert bp["powers"][0]["amount"] == 2

    def test_combat_top_level_player_uses_merged_battle_player(self):
        gs = self._make_combat_state()
        gs.player.CopyFrom(
            _make_player(
                deck=[{"index": 0, "id": "Strike", "name": "Strike", "cost": 1,
                       "card_type": "ATTACK", "rarity": "basic", "is_upgraded": False, "upgrades": 0}],
                relics=[{"index": 0, "id": "BurningBlood", "name": "BurningBlood"}],
                potions=[{"index": 0, "id": "FirePotion", "name": "FirePotion"}],
            )
        )
        d = game_state_to_dict(gs)
        player = d["player"]
        assert player["deck"][0]["id"] == "Strike"
        assert player["relics"][0]["id"] == "BurningBlood"
        assert player["potions"][0]["id"] == "FirePotion"
        assert player["powers"] == [{"id": "STRENGTH_POWER", "amount": 2}]
        assert player is d["battle"]["player"]

    def test_hand_cards(self):
        gs = self._make_combat_state()
        d = game_state_to_dict(gs)
        hand = d["battle"]["hand"]
        assert len(hand) == 1
        card = hand[0]
        assert card["id"] == "Strike"
        assert card["can_play"] is True
        assert card["requires_target"] is True
        assert card["valid_target_ids"] == [0]
        assert card["type"] == "attack"
        # battle.hand 和 battle.player.hand 应该是同一份
        assert d["battle"]["player"]["hand"] is hand

    def test_enemies(self):
        gs = self._make_combat_state()
        d = game_state_to_dict(gs)
        enemies = d["battle"]["enemies"]
        assert len(enemies) == 1
        e = enemies[0]
        # 冗余别名
        assert e["id"] == "JawWorm"
        assert e["entity_id"] == "JawWorm"
        assert e["monster_id"] == "JawWorm"
        assert e["combat_id"] == 0
        assert e["target_id"] == 0
        assert e["hp"] == 30
        assert e["current_hp"] == 30
        assert e["is_alive"] is True
        assert e["is_hittable"] is True
        assert e["intends_to_attack"] is True
        assert e["next_move_id"] == "Chomp"
        # powers 三个引用
        assert e["status"] is e["powers"]
        assert e["buffs"] is e["powers"]
        # intents
        assert len(e["intents"]) == 1
        assert e["intent_type"] == "attack"
        assert e["intent_damage"] == 11
        assert e["intent_hits"] == 1

    def test_piles(self):
        gs = self._make_combat_state()
        d = game_state_to_dict(gs)
        assert d["battle"]["draw_pile_cards"] == ["Strike", "Defend"]
        assert d["battle"]["discard_pile_cards"] == ["Bash"]
        assert d["battle"]["exhaust_pile_cards"] == []

    def test_top_level_enemies_alias(self):
        gs = self._make_combat_state()
        d = game_state_to_dict(gs)
        assert d["enemies"] is d["battle"]["enemies"]
        assert d["round_number_raw"] == 3


# ======================================================================
# Tests — Map
# ======================================================================

class TestMap:
    def test_map_options(self):
        gs = _make_game_state("map")
        opt = gs.map.next_options.add()
        opt.index = 0
        opt.col = 2
        opt.row = 1
        opt.point_type = "monster"
        opt.label = "monster"
        opt2 = gs.map.next_options.add()
        opt2.index = 1
        opt2.col = 3
        opt2.row = 1
        opt2.point_type = "event"
        d = game_state_to_dict(gs)
        m = d["map"]
        assert len(m["next_options"]) == 2
        assert m["next_options"][0]["point_type"] == "monster"
        assert m["next_options"][0]["type"] == "monster"
        assert m["next_options"][1]["point_type"] == "event"
        assert m["player"] is d["player"]

    def test_map_nodes_and_boss(self):
        gs = _make_game_state("map")
        node = gs.map.nodes.add()
        node.col = 1
        node.row = 0
        node.type = "monster"
        child = node.children.add()
        child.col = 2
        child.row = 1
        gs.map.boss.col = 3
        gs.map.boss.row = 15
        d = game_state_to_dict(gs)
        nodes = d["map"]["nodes"]
        assert len(nodes) == 1
        assert nodes[0]["children"] == [[2, 1]]
        assert d["map"]["boss"] == {"col": 3, "row": 15}


# ======================================================================
# Tests — Event / Rest / Shop / Rewards / CardReward
# ======================================================================

class TestEvent:
    def test_event_state(self):
        gs = _make_game_state("event")
        ev = gs.event
        ev.event_id = "BigFish"
        ev.in_dialogue = True
        ev.is_finished = False
        opt = ev.options.add()
        opt.index = 0
        opt.text = "Eat"
        opt.label = "Eat"
        opt.is_locked = False
        opt.is_chosen = False
        opt.is_proceed = False
        d = game_state_to_dict(gs)
        event = d["event"]
        assert event["event_id"] == "BigFish"
        assert event["in_dialogue"] is True
        assert len(event["options"]) == 1
        assert event["options"][0]["text"] == "Eat"


class TestRestSite:
    def test_rest_site(self):
        gs = _make_game_state("rest_site")
        rs = gs.rest_site
        rs.can_proceed = True
        opt = rs.options.add()
        opt.index = 0
        opt.id = "rest"
        opt.name = "rest"
        opt.is_enabled = True
        opt2 = rs.options.add()
        opt2.index = 1
        opt2.id = "smith"
        opt2.name = "smith"
        opt2.is_enabled = True
        d = game_state_to_dict(gs)
        rest = d["rest_site"]
        assert rest["can_proceed"] is True
        assert len(rest["options"]) == 2
        assert rest["options"][0]["id"] == "rest"


class TestShop:
    def test_shop_items(self):
        gs = _make_game_state("shop")
        sh = gs.shop
        sh.is_open = True
        sh.can_proceed = True
        item = sh.items.add()
        item.index = 0
        item.category = "card"
        item.cost = 75
        item.can_afford = True
        item.is_stocked = True
        item.on_sale = False
        item.id = "Whirlwind"
        item.name = "Whirlwind"
        d = game_state_to_dict(gs)
        shop = d["shop"]
        assert shop["is_open"] is True
        assert len(shop["items"]) == 1
        si = shop["items"][0]
        assert si["category"] == "card"
        assert si["cost"] == 75
        assert si["price"] == 75
        assert si["card_id"] == "Whirlwind"


class TestCombatRewards:
    def test_rewards(self):
        gs = _make_game_state("combat_rewards")
        cr = gs.combat_rewards
        cr.can_proceed = True
        ri = cr.items.add()
        ri.index = 0
        ri.type = "gold"
        ri.label = "25 Gold"
        ri.id = "gold_25"
        ri.claimable = True
        d = game_state_to_dict(gs)
        rewards = d["rewards"]
        assert rewards["can_proceed"] is True
        assert len(rewards["items"]) == 1
        assert rewards["items"][0]["type"] == "gold"
        assert rewards["items"][0]["claimable"] is True


class TestCardReward:
    def test_card_reward(self):
        gs = _make_game_state("card_reward")
        cr = gs.card_reward
        cr.can_skip = True
        c = cr.cards.add()
        c.index = 0
        c.id = "Inflame"
        c.name = "Inflame"
        c.cost = 1
        c.card_type = "POWER"
        c.rarity = "uncommon"
        d = game_state_to_dict(gs)
        card_rw = d["card_reward"]
        assert card_rw["can_skip"] is True
        assert len(card_rw["cards"]) == 1
        assert card_rw["cards"][0]["id"] == "Inflame"
        assert card_rw["cards"][0]["type"] == "power"


# ======================================================================
# Tests — Action Label Decoration
# ======================================================================

class TestActionLabels:
    def test_play_card_label(self):
        gs = _make_game_state("monster")
        b = gs.battle
        b.round_number = 1
        b.turn_side = "player"
        b.energy = 3
        b.max_energy = 3
        hc = b.hand.add()
        hc.index = 0
        hc.id = "Strike"
        hc.name = "Strike"
        hc.cost = 1
        hc.card_type = "ATTACK"
        hc.can_play = True
        hc.requires_target = True
        bp = b.player
        bp.hp = 50
        bp.max_hp = 80
        la = gs.legal_actions.add()
        la.action = "play_card"
        la.card_index = 0
        la.index = 0
        la.card_id = "Strike"
        d = game_state_to_dict(gs)
        assert d["legal_actions"][0]["label"] == "Strike"
        assert d["legal_actions"][0]["card_id"] == "Strike"

    def test_map_node_label(self):
        gs = _make_game_state("map")
        opt = gs.map.next_options.add()
        opt.index = 0
        opt.col = 1
        opt.row = 1
        opt.point_type = "elite"
        opt.label = "elite"
        la = gs.legal_actions.add()
        la.action = "choose_map_node"
        la.index = 0
        d = game_state_to_dict(gs)
        assert d["legal_actions"][0]["label"] == "elite"


# ======================================================================
# Tests — Roundtrip: serialize → parse → convert
# ======================================================================

class TestRoundtrip:
    def test_serialize_parse_roundtrip(self):
        """验证 proto 序列化/反序列化不丢数据。"""
        gs = _make_game_state("monster")
        b = gs.battle
        b.round_number = 5
        b.turn_side = "player"
        b.is_play_phase = True
        b.energy = 2
        b.max_energy = 3
        bp = b.player
        bp.hp = 40
        bp.max_hp = 80
        bp.block = 5
        hc = b.hand.add()
        hc.index = 0
        hc.id = "Bash"
        hc.cost = 2
        hc.card_type = "ATTACK"
        hc.can_play = True
        e = b.enemies.add()
        e.id = "Cultist"
        e.combat_id = 0
        e.hp = 48
        e.max_hp = 48
        e.is_alive = True
        e.is_hittable = True
        intent = e.intents.add()
        intent.type = "attack"
        intent.damage = 6
        intent.total_damage = 6
        intent.hits = 1

        raw = gs.SerializeToString()
        gs2 = pb.GameState()
        gs2.ParseFromString(raw)
        d = game_state_to_dict(gs2)

        assert d["state_type"] == "monster"
        assert d["battle"]["round_number_raw"] == 5
        assert d["battle"]["enemies"][0]["id"] == "Cultist"
        assert d["battle"]["hand"][0]["id"] == "Bash"
        assert d["battle"]["player"]["hp"] == 40


# ======================================================================
# Tests — Proto3 零值边界（BUG1 回归）
# ======================================================================

class TestProto3ZeroValues:
    """验证 proto3 中 int32 默认值 0 不被错误当作 '未设置'。"""

    def test_deck_first_card_index_zero(self):
        """第一张卡 index=0 不能被吞掉。"""
        p = _make_player(deck=[
            {"index": 0, "id": "Strike", "name": "Strike", "cost": 1,
             "card_type": "ATTACK", "rarity": "basic", "is_upgraded": False, "upgrades": 0},
            {"index": 1, "id": "Defend", "name": "Defend", "cost": 1,
             "card_type": "SKILL", "rarity": "basic", "is_upgraded": False, "upgrades": 0},
        ])
        gs = _make_game_state("map", player=p)
        d = game_state_to_dict(gs)
        assert d["player"]["deck"][0]["index"] == 0
        assert d["player"]["deck"][1]["index"] == 1

    def test_relic_index_zero(self):
        p = _make_player(relics=[
            {"index": 0, "id": "BurningBlood", "name": "BurningBlood"},
        ])
        gs = _make_game_state("map", player=p)
        d = game_state_to_dict(gs)
        assert d["player"]["relics"][0]["index"] == 0

    def test_legal_action_index_zero(self):
        """index=0 是合法的动作索引，不能变成 None。"""
        gs = _make_game_state("map")
        la = gs.legal_actions.add()
        la.action = "choose_map_node"
        la.index = 0
        la.card_index = -1
        la.target_id = -1
        la.col = -1
        la.row = -1
        la.slot = -1
        d = game_state_to_dict(gs)
        assert d["legal_actions"][0]["index"] == 0
        assert d["legal_actions"][0]["card_index"] is None

    def test_hand_card_index_zero(self):
        gs = _make_game_state("monster")
        b = gs.battle
        b.turn_side = "player"
        bp = b.player
        bp.hp = 50
        bp.max_hp = 80
        hc = b.hand.add()
        hc.index = 0
        hc.id = "Strike"
        hc.cost = 1
        hc.card_type = "ATTACK"
        hc.can_play = True
        d = game_state_to_dict(gs)
        assert d["battle"]["hand"][0]["index"] == 0

    def test_enemy_combat_id_zero(self):
        gs = _make_game_state("monster")
        b = gs.battle
        b.turn_side = "player"
        bp = b.player
        bp.hp = 50
        bp.max_hp = 80
        e = b.enemies.add()
        e.id = "JawWorm"
        e.combat_id = 0
        e.hp = 44
        e.max_hp = 44
        e.is_alive = True
        e.is_hittable = True
        d = game_state_to_dict(gs)
        assert d["battle"]["enemies"][0]["combat_id"] == 0
        assert d["battle"]["enemies"][0]["target_id"] == 0

    def test_legal_action_no_extra_card_id(self):
        """legal_action 输出不应包含 binary 没有的 card_id 字段。"""
        gs = _make_game_state("map")
        la = gs.legal_actions.add()
        la.action = "end_turn"
        la.index = -1
        la.card_index = -1
        la.target_id = -1
        la.col = -1
        la.row = -1
        la.slot = -1
        d = game_state_to_dict(gs)
        assert "card_id" not in d["legal_actions"][0]

    def test_end_turn_proto_defaults_do_not_become_hand_index(self):
        """proto3 默认 0 不能让 end_turn 看起来绑定了 hand[0]。"""
        gs = _make_game_state("monster")
        la = gs.legal_actions.add()
        la.action = "end_turn"
        d = game_state_to_dict(gs)
        assert d["legal_actions"][0]["card_index"] is None
        assert d["legal_actions"][0]["target_id"] is None


# ======================================================================
# Tests — CardSelect / RelicSelect / Treasure
# ======================================================================

class TestCardSelect:
    def test_card_select(self):
        gs = _make_game_state("card_select")
        cs = gs.card_select
        cs.screen_type = "card_select"
        cs.selected_count = 1
        cs.can_confirm = True
        cs.can_cancel = False
        cs.prompt = "Choose a card to upgrade."
        cs.min_select = 1
        cs.max_select = 1
        c = cs.cards.add()
        c.id = "Strike"
        c.cost = 1
        c.card_type = "ATTACK"
        sel = cs.selected_cards.add()
        sel.id = "Defend"
        sel.cost = 1
        sel.card_type = "SKILL"
        d = game_state_to_dict(gs)
        assert "card_select" in d
        assert d["card_select"]["screen_type"] == "card_select"
        assert d["card_select"]["can_confirm"] is True
        assert d["card_select"]["prompt"] == "Choose a card to upgrade."
        assert d["card_select"]["min_select"] == 1
        assert d["card_select"]["max_select"] == 1
        assert len(d["card_select"]["cards"]) == 1
        assert len(d["card_select"]["selected_cards"]) == 1
        assert d["card_select"]["cards"][0]["id"] == "Strike"

    def test_card_select_preserves_operation_label(self):
        gs = _make_game_state("card_select")
        c = gs.card_select.cards.add()
        c.id = "BASH"
        c.cost = 2
        c.card_type = "ATTACK"
        la = gs.legal_actions.add()
        la.action = "select_card"
        la.index = 0
        la.card_index = 0
        la.target_id = -1
        la.col = -1
        la.row = -1
        la.slot = -1
        la.card_id = "BASH"
        la.label = "remove_card BASH"

        d = game_state_to_dict(gs)

        assert d["legal_actions"][0]["label"] == "remove_card BASH"
        assert d["legal_actions"][0]["card_id"] == "BASH"

    def test_combat_card_select_is_available_on_battle_state(self):
        gs = TestBattle()._make_combat_state()
        cs = gs.card_select
        cs.screen_type = "SimpleSelect"
        cs.prompt = "Choose a card."
        cs.min_select = 0
        cs.max_select = 1
        c = cs.cards.add()
        c.index = 0
        c.id = "ANGER"
        c.name = "Anger"
        c.cost = 0
        c.card_type = "ATTACK"
        la = gs.legal_actions.add()
        la.action = "combat_select_card"
        la.index = 0
        la.card_index = 0
        la.card_id = "ANGER"
        la.label = "Anger"
        la.target_id = -1
        la.col = -1
        la.row = -1
        la.slot = -1

        d = game_state_to_dict(gs)

        assert d["state_type"] == "monster"
        assert d["card_select"]["prompt"] == "Choose a card."
        assert d["battle"]["card_selection"]["selectable_cards"][0]["id"] == "ANGER"
        assert d["legal_actions"][0]["label"] == "Anger"
        assert d["legal_actions"][0]["card_id"] == "ANGER"


class TestRelicSelect:
    def test_relic_select(self):
        gs = _make_game_state("relic_select")
        rs = gs.relic_select
        rs.can_skip = True
        r = rs.relics.add()
        r.id = "BurningBlood"
        r.name = "BurningBlood"
        d = game_state_to_dict(gs)
        assert "relic_select" in d
        assert d["relic_select"]["can_skip"] is True
        assert len(d["relic_select"]["relics"]) == 1
        assert d["relic_select"]["relics"][0]["id"] == "BurningBlood"


class TestTreasure:
    def test_treasure(self):
        gs = _make_game_state("treasure")
        ts = gs.treasure
        ts.can_proceed = False
        r = ts.relics.add()
        r.id = "GoldenIdol"
        r.name = "GoldenIdol"
        d = game_state_to_dict(gs)
        assert "treasure" in d
        assert d["treasure"]["can_proceed"] is False
        assert d["treasure"]["relics"][0]["id"] == "GoldenIdol"
