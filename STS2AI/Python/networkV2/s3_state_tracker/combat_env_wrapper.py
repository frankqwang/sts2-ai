"""CombatStateTracker：管理跨步累积状态。

职责：
  1. 维护 CombatMemory（战斗长程记忆）
  2. 维护 TurnPrefixMemory（本回合历史，新回合清零）
  3. 维护 RunBuildMemory（整局慢变量）
  4. 检测回合切换和战斗结束

HP 追踪策略：
  - 每回合开始时记录 _turn_start_hp
  - 回合结束时用 _turn_start_hp - current_hp 计算本回合掉血
  - 终局时强制 flush 最后一个回合的数据
"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.memory import (
    CombatMemory,
    TurnPrefixMemory,
    RunBuildMemory,
    PlayedAction,
)
from core.card_base_stats import is_aoe as _is_aoe


# 数据驱动规范（CLAUDE.md）：heal / draw 判定走 core.card_tags 的 functional tags，
# 覆盖全职业所有卡 + 升级版自动继承。
#
# U6 修复：原先有 `_FALLBACK_HEAL_CARDS = {"reaper", "bite", "self_repair"}` 兜底集，
# 是 STS1 卡名残留——STS2 里根本没有这些卡（对应 STS2 是 reaper_form / snakebite 等
# 且不一定是 heal 卡）。重跑 card_tags 提取后确认：STS2 只有 spur 这一张带 heal tag
# 的卡，兜底永远不命中。直接删除，完全数据驱动。


def _card_tags_db():
    """Lazy load card_tags.json（由 core/card_tags.py 从 C# 源码提取）。返回
    dict[card_id_lower, list[tag_name]]。"""
    global _CARD_TAGS_CACHE
    if _CARD_TAGS_CACHE is None:
        try:
            from core.card_tags import load_card_tags
            _CARD_TAGS_CACHE = load_card_tags()
        except Exception:
            _CARD_TAGS_CACHE = {}
    return _CARD_TAGS_CACHE


_CARD_TAGS_CACHE: dict[str, list[str]] | None = None


def _strip_upgrade_suffix(cid: str) -> str:
    for suffix in ("_upgraded", "_upgrade", "+1"):
        if cid.endswith(suffix):
            return cid[: -len(suffix)]
    return cid


def _card_has_tag(card_id: str, tag: str) -> bool:
    """查 card 是否有某功能 tag。升级版 "_UPGRADED" 后缀会被剥离后查 base 版。"""
    cid = str(card_id or "").lower().strip()
    if not cid:
        return False
    tags = _card_tags_db().get(cid)
    if tags is None:
        tags = _card_tags_db().get(_strip_upgrade_suffix(cid))
    return tags is not None and tag in tags


def _is_heal_card(card_id: str) -> bool:
    return _card_has_tag(card_id, "heal")


def _is_draw_card(card_id: str) -> bool:
    return _card_has_tag(card_id, "draw")


def _is_x_cost_card(card: dict[str, Any]) -> bool:
    """X-cost 检测：cost == -1（游戏惯例）或 has_energy_cost_x 字段。"""
    cost = card.get("cost", card.get("energy_cost"))
    if cost is not None:
        try:
            if int(cost) == -1:
                return True
        except (TypeError, ValueError):
            pass
    for key in ("is_x_cost", "has_energy_cost_x", "IsXCost", "HasEnergyCostX"):
        if bool(card.get(key, False)):
            return True
    return False


def _extract_hp(obs: dict[str, Any]) -> int:
    """从 obs 中提取玩家当前 HP。兼容 battle.player 和 top-level player。"""
    for source in (obs.get("battle") or {}, obs):
        player = source.get("player")
        if isinstance(player, dict):
            for key in ("hp", "current_hp", "CurrentHp"):
                if key in player:
                    try:
                        return int(player[key])
                    except (TypeError, ValueError):
                        pass
    return 0


def _extract_player_dict(obs: dict[str, Any]) -> dict[str, Any]:
    battle = obs.get("battle") or {}
    return battle.get("player") or obs.get("player") or {}


def _extract_block(obs: dict[str, Any]) -> int:
    p = _extract_player_dict(obs)
    for k in ("block", "Block"):
        if k in p:
            try:
                return int(p[k] or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def _extract_energy(obs: dict[str, Any]) -> int:
    p = _extract_player_dict(obs)
    battle = obs.get("battle") or {}
    for source in (p, battle):
        for k in ("energy", "Energy"):
            if k in source:
                try:
                    return int(source[k] or 0)
                except (TypeError, ValueError):
                    return 0
    return 0


def _extract_enemy_hp_total(obs: dict[str, Any]) -> int:
    """所有敌人当前 HP 之和（只计存活/可伤）。"""
    battle = obs.get("battle") or {}
    enemies = battle.get("enemies") or obs.get("enemies") or obs.get("monsters") or []
    if not isinstance(enemies, list):
        return 0
    total = 0
    for e in enemies:
        if not isinstance(e, dict):
            continue
        for k in ("hp", "current_hp", "CurrentHp"):
            if k in e:
                try:
                    total += max(int(e[k] or 0), 0)
                except (TypeError, ValueError):
                    pass
                break
    return total


def _extract_intent_damage_total(obs: dict[str, Any]) -> int:
    """所有活敌人 intent 总伤害（attack 类 intent 的 damage * repeats 之和）。"""
    battle = obs.get("battle") or {}
    enemies = battle.get("enemies") or obs.get("enemies") or obs.get("monsters") or []
    if not isinstance(enemies, list):
        return 0
    total = 0
    for e in enemies:
        if not isinstance(e, dict):
            continue
        if not e.get("is_alive", True):
            continue
        intents = e.get("intents") or e.get("Intents") or []
        if not isinstance(intents, list):
            continue
        for it in intents:
            if not isinstance(it, dict):
                continue
            itype = str(it.get("intent_type", it.get("IntentType", "")) or "").lower()
            if itype != "attack":
                continue
            try:
                dmg = int(it.get("damage", it.get("Damage", 0)) or 0)
                rep = int(it.get("repeats", it.get("Repeats", 1)) or 1)
                total += max(dmg, 0) * max(rep, 1)
            except (TypeError, ValueError):
                pass
    return total


def _extract_hand_size(obs: dict[str, Any]) -> int:
    battle = obs.get("battle") or {}
    hand = obs.get("hand") or battle.get("hand") or []
    return len(hand) if isinstance(hand, list) else 0


def _extract_pile_counts(obs: dict[str, Any]) -> dict[str, int]:
    """提取 draw / discard / exhaust pile 的卡牌数。"""
    battle = obs.get("battle") or {}
    out = {"draw": 0, "discard": 0, "exhaust": 0}
    configs = [
        ("draw", ("draw_pile_cards",), ("draw_pile_count", "draw_pile_size")),
        ("discard", ("discard_pile_cards",), ("discard_pile_count", "discard_pile_size")),
        ("exhaust", ("exhaust_pile_cards",), ("exhaust_pile_count", "exhaust_pile_size")),
    ]
    for name, list_keys, count_keys in configs:
        for source in (battle, obs):
            found = False
            for k in list_keys:
                v = source.get(k)
                if isinstance(v, list):
                    out[name] = len(v)
                    found = True
                    break
            if found:
                break
            for k in count_keys:
                if k in source:
                    try:
                        out[name] = int(source[k] or 0)
                        found = True
                        break
                    except (TypeError, ValueError):
                        pass
            if found:
                break
    return out


def _find_hand_card(obs: dict[str, Any], hand_index: int) -> dict[str, Any] | None:
    """在 prev_state 的手牌里按 hand_index 找卡，读出 retain/exhaust 等静态属性。"""
    battle = obs.get("battle") or {}
    hand = obs.get("hand") or battle.get("hand") or []
    if not isinstance(hand, list):
        return None
    for card in hand:
        if not isinstance(card, dict):
            continue
        idx = card.get("hand_index", card.get("HandIndex", card.get("index")))
        try:
            if int(idx) == int(hand_index):
                return card
        except (TypeError, ValueError):
            continue
    return None


def _compute_action_effects(
    prev_state: dict[str, Any] | None,
    next_state: dict[str, Any] | None,
    action: dict[str, Any],
    prev_pile_counts: dict[str, int] | None,
    next_pile_counts: dict[str, int] | None,
) -> dict[str, Any]:
    """用两帧状态差分算出动作效果量。

    damage_dealt: sum(prev_enemy_hp - next_enemy_hp)，只取正（敌人血减少 = 造成伤害）
    block_gained: max(next_block - prev_block, 0)
    cards_drawn: next_hand_size - prev_hand_size + (打了一张牌 ? 1 : 0)
    energy_delta: next_energy - prev_energy
    was_exhaust:  next_exhaust_pile > prev_exhaust_pile（execute 后 exhaust 堆变大）
    was_retain:   从 prev_hand 对应 hand_index 读 keywords 里有没有 retain
    """
    out = {
        "damage_dealt": 0.0, "block_gained": 0.0,
        "cards_drawn": 0, "energy_delta": 0,
        "was_exhaust": False, "was_retain": False,
    }
    if prev_state is None or next_state is None:
        return out

    # damage: 敌人总 HP 减少
    d_enemy = _extract_enemy_hp_total(prev_state) - _extract_enemy_hp_total(next_state)
    out["damage_dealt"] = float(max(d_enemy, 0))
    # block: 玩家 block 增量（正向）
    d_block = _extract_block(next_state) - _extract_block(prev_state)
    out["block_gained"] = float(max(d_block, 0))
    # energy: 有正有负（draw 类牌可能 +energy）
    out["energy_delta"] = _extract_energy(next_state) - _extract_energy(prev_state)
    # cards drawn: 如果执行的是 play_card，手牌净变化 = drawn - 1 (打掉那张)
    #              如果是 use_potion/其他，手牌净变化直接就是 drawn（抽药水卡等 rare case）
    d_hand = _extract_hand_size(next_state) - _extract_hand_size(prev_state)
    act_type = str(action.get("action", "") or "").lower()
    if act_type == "play_card":
        out["cards_drawn"] = max(d_hand + 1, 0)
    else:
        out["cards_drawn"] = max(d_hand, 0)

    # exhaust: 看 exhaust pile 是否变大
    if prev_pile_counts is not None and next_pile_counts is not None:
        out["was_exhaust"] = next_pile_counts.get("exhaust", 0) > prev_pile_counts.get("exhaust", 0)
    # retain: 从 prev_hand 的卡片 keywords 读
    if act_type == "play_card":
        hand_idx = action.get("hand_index", action.get("card_index", action.get("index")))
        if hand_idx is not None:
            try:
                card = _find_hand_card(prev_state, int(hand_idx))
            except (TypeError, ValueError):
                card = None
            if card is not None:
                keywords = [str(k).lower() for k in (card.get("keywords") or [])]
                out["was_retain"] = "retain" in keywords
    return out


def _is_reshuffle(prev_pile: dict[str, int], cur_pile: dict[str, int]) -> bool:
    """洗牌事件检测：上一帧 draw<=1 且 discard>0，这一帧 discard 归 0 + draw 明显变多。

    这是确定性状态机判定（不是启发式）：游戏在 draw pile 空了后会把 discard 倒回 draw。
    """
    prev_draw = prev_pile.get("draw", 0)
    prev_disc = prev_pile.get("discard", 0)
    cur_draw = cur_pile.get("draw", 0)
    cur_disc = cur_pile.get("discard", 0)
    return prev_draw <= 1 and prev_disc > 0 and cur_disc == 0 and cur_draw > prev_draw


class CombatStateTracker:
    """跨步状态追踪器。纯追踪，不包装 env.step()。"""

    def __init__(self) -> None:
        self.combat_memory = CombatMemory()
        self.turn_prefix = TurnPrefixMemory()
        self.run_build_memory = RunBuildMemory()
        self.encounter_id: str = ""
        self.room_type: str = "monster"
        self._combat_start_hp: int = 0      # 本场战斗开始时的 HP
        self._turn_start_hp: int = 0        # 本回合开始时的 HP
        self._turn_start_enemy_hp: int = 0  # 本回合开始时的敌方总 HP（用于算敌方掉血）
        self._last_turn: int = 0
        self._in_combat: bool = False
        self._prev_pile_counts: dict[str, int] = {"draw": 0, "discard": 0, "exhaust": 0}
        # 敌人行为追踪：combat_id → 最近看到的 next_move_id
        # 当某敌人的 move_id 变化时触发 combat_memory.on_behavior_change
        self._last_move_ids: dict[int, str] = {}

    def on_combat_start(
        self,
        obs: dict[str, Any],
        encounter_id: str = "",
        room_type: str = "monster",
    ) -> None:
        """新战斗开始时调用。"""
        self.combat_memory.reset()
        self.turn_prefix.reset()
        self.encounter_id = encounter_id
        self.room_type = room_type
        self._in_combat = True

        hp = _extract_hp(obs)
        self._combat_start_hp = hp
        self._turn_start_hp = hp
        self._turn_start_enemy_hp = _extract_enemy_hp_total(obs)
        # 初始回合从 obs 取，避免第一次 on_step 时 _last_turn=0 导致漏 flush
        self._last_turn = self._detect_turn(obs) or 1

        self.run_build_memory.combats_seen += 1
        if room_type == "elite":
            self.run_build_memory.elites_seen += 1
        elif room_type == "boss":
            self.run_build_memory.bosses_seen += 1
        # 登记本场敌人类型 + 房间类型到跨战斗历史
        battle = obs.get("battle") or {}
        enemies = battle.get("enemies") or obs.get("enemies") or obs.get("monsters") or []
        enemy_ids = [str(e.get("id", "")) for e in enemies if isinstance(e, dict)]
        self.run_build_memory.register_combat(enemy_ids, room_type)

        # 从 obs 填充 build profile（每场战斗开始时刷新）
        self.refresh_build_profile(obs)
        # 初始化 pile 状态（用于 reshuffle 检测）
        self._prev_pile_counts = _extract_pile_counts(obs)
        # 初始化敌人行为基线：新战斗开始时清空 + 记录第一帧的 move_id
        self._last_move_ids.clear()
        self._update_enemy_behaviors(obs, record=True)

    def on_step(
        self,
        obs: dict[str, Any],
        action_taken: dict[str, Any] | None = None,
        prev_state: dict[str, Any] | None = None,
    ) -> None:
        """每步调用。

        Args:
            obs: 动作执行后的状态（next_state）
            action_taken: 刚执行的动作 dict
            prev_state: 动作执行前的状态；用于计算效果差分（damage/block/draw/energy/exhaust/retain）
        """
        if not self._in_combat:
            return

        # 检测回合切换
        current_turn = self._detect_turn(obs)
        if current_turn > self._last_turn and self._last_turn > 0:
            self._flush_turn(obs)
        if self._last_turn == 0:
            self._last_turn = current_turn

        # 洗牌检测：prev_pile vs cur_pile
        cur_pile = _extract_pile_counts(obs)
        if _is_reshuffle(self._prev_pile_counts, cur_pile):
            self.combat_memory.reshuffle_count += 1

        # 敌人行为切换检测：每个敌人 next_move_id 变化时记录一次行为事件
        self._update_enemy_behaviors(obs, record=True)

        # 记录动作到 turn prefix（含效果差分）
        if action_taken is not None:
            played = self._extract_played_action(
                action_taken,
                prev_state=prev_state,
                next_state=obs,
                prev_pile=self._prev_pile_counts,
                next_pile=cur_pile,
            )
            if played is not None:
                self.turn_prefix.record_action(played)

        # 更新 pile 基线供下一步差分
        self._prev_pile_counts = cur_pile

    def on_combat_end(self, obs: dict[str, Any] | None = None) -> None:
        """战斗结束时调用。强制 flush 最后一个回合的数据。"""
        if self._in_combat:
            self._flush_turn(obs or {})
            # 累计整场掉血 + 药水使用次数到 run memory
            self.run_build_memory.total_hp_lost += self.combat_memory.cumulative_hp_loss
            self.run_build_memory.potions_used_total += self.combat_memory.potions_used
        self._in_combat = False

    def _flush_turn(self, obs: dict[str, Any]) -> None:
        """将当前回合的数据聚合到 combat_memory，然后重置 turn_prefix。"""
        current_hp = _extract_hp(obs)
        hp_loss_this_turn = max(self._turn_start_hp - current_hp, 0)
        # 敌方时序信号：本回合敌方 HP 减少 + 敌人当前 intent 总伤害
        cur_enemy_hp = _extract_enemy_hp_total(obs)
        enemy_hp_loss_this_turn = max(self._turn_start_enemy_hp - cur_enemy_hp, 0)
        intent_damage_this_turn = _extract_intent_damage_total(obs)

        # 聚合到 combat memory
        self.combat_memory.on_new_turn(
            hp_loss_this_turn,
            enemy_hp_loss_this_turn=enemy_hp_loss_this_turn,
            intent_damage_this_turn=intent_damage_this_turn,
        )
        self.combat_memory.total_damage_dealt += self.turn_prefix.total_damage_dealt
        self.combat_memory.total_block_gained += self.turn_prefix.total_block_gained
        self.combat_memory.max_single_turn_damage = max(
            self.combat_memory.max_single_turn_damage,
            self.turn_prefix.total_damage_dealt,
        )
        self.combat_memory.exhaust_total += self.turn_prefix.total_exhaust
        self.combat_memory.potions_used += self.turn_prefix.potions_used

        # 重置 turn prefix，更新回合起始 HP / 敌方基线
        self.turn_prefix.reset()
        self._turn_start_hp = current_hp
        self._turn_start_enemy_hp = cur_enemy_hp
        self._last_turn = max(self._last_turn, self._detect_turn(obs))

    def _detect_turn(self, obs: dict[str, Any]) -> int:
        """从 obs 中提取当前回合数。游戏字段: battle.round_number_raw"""
        battle = obs.get("battle") or {}
        for source in (battle, obs):
            for key in ("round_number_raw", "round_number", "turn", "round"):
                if key in source:
                    try:
                        val = int(source[key])
                        if val > 0:
                            return val
                    except (TypeError, ValueError):
                        pass
        return self._last_turn

    def _update_enemy_behaviors(self, obs: dict[str, Any], *, record: bool) -> None:
        """扫描 obs.enemies 的 next_move_id，和上一帧对比；出现变化则记录行为切换事件。

        - 首次看到某敌人（combat_id 不在 _last_move_ids）时，若 record=True 则把当前 move 当作"进入初始行为"事件记一条
        - 之后每次 move_id 变化，调 combat_memory.on_behavior_change
        - `record=False` 只刷新基线不记录（用于 on_combat_start 外的特殊场景，暂未使用）
        """
        battle = obs.get("battle") or {}
        enemies = battle.get("enemies") or obs.get("enemies") or obs.get("monsters") or []
        if not isinstance(enemies, list):
            return
        for e in enemies:
            if not isinstance(e, dict):
                continue
            cid = e.get("combat_id", e.get("CombatId"))
            move = e.get("next_move_id") or e.get("NextMoveId")
            if cid is None or not move:
                continue
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            move = str(move)
            eid = str(e.get("id", e.get("Id", "")) or "")
            key = f"{eid}:{move}"   # 带敌人种类前缀，方便后续分析
            last = self._last_move_ids.get(cid)
            if last != move:
                if record:
                    self.combat_memory.on_behavior_change(key)
                self._last_move_ids[cid] = move

    def _extract_played_action(
        self,
        action: dict[str, Any],
        *,
        prev_state: dict[str, Any] | None = None,
        next_state: dict[str, Any] | None = None,
        prev_pile: dict[str, int] | None = None,
        next_pile: dict[str, int] | None = None,
    ) -> PlayedAction | None:
        """从 action dict + 前后帧差分 提取 PlayedAction（含效果量）。"""
        action_type = str(action.get("action", "") or "").lower()
        if action_type not in ("play_card", "use_potion", "drink_potion"):
            return None
        normalized_type = "use_potion" if action_type in ("use_potion", "drink_potion") else action_type

        # ---- 效果差分（核心修复：原来从 action dict 读，永远是 0）----
        eff = _compute_action_effects(prev_state, next_state, action, prev_pile, next_pile)

        # ---- card_type / card_id / cost 回查（legal_actions dict 不稳定带这些字段）----
        # 从 prev_state 的 hand 按 hand_index 查：bridge normalize 的 hand card 保证有 card_type/cost
        card_id = str(action.get("card_id", "") or "").lower()
        card_type = str(action.get("card_type", "") or "").lower()
        cost = int(action.get("cost", 0) or 0)
        if action_type == "play_card" and prev_state is not None:
            hand_idx = action.get("hand_index", action.get("card_index", action.get("index")))
            if hand_idx is not None:
                try:
                    card = _find_hand_card(prev_state, int(hand_idx))
                except (TypeError, ValueError):
                    card = None
                if card is not None:
                    if not card_type:
                        raw_ct = card.get("card_type", card.get("type", ""))
                        card_type = str(raw_ct or "").lower()
                    if not card_id:
                        card_id = str(card.get("id", card.get("card_id", "")) or "").lower()
                    if not cost:
                        try:
                            cost = int(card.get("cost", card.get("energy_cost", 0)) or 0)
                        except (TypeError, ValueError):
                            cost = 0

        return PlayedAction(
            action_type=normalized_type,
            card_id=card_id,
            card_type=card_type,
            target_id=str(action.get("target_id", "") or ""),
            cost=cost,
            damage_dealt=eff["damage_dealt"],
            block_gained=eff["block_gained"],
            cards_drawn=eff["cards_drawn"],
            energy_delta=eff["energy_delta"],
            was_exhaust=eff["was_exhaust"],
            was_retain=eff["was_retain"],
        )

    def on_run_start(self) -> None:
        """新的一局（full run）开始时调用。"""
        self.run_build_memory = RunBuildMemory()
        self.combat_memory.reset()
        self.turn_prefix.reset()
        self._in_combat = False

    def refresh_build_profile(self, obs: dict[str, Any]) -> None:
        """从 obs 的 player 信息填充 RunBuildMemory 的 build profile。

        在 on_combat_start 时内部调用；训练主循环在非战斗 step（shop/card_reward/
        rest/event/map）也应主动调一次，否则 gold/act/floor/deck/relic/potion
        以及派生的 build profile 字段会滞后到上一次进战前的值。
        """
        player = obs.get("player") or {}
        rbm = self.run_build_memory

        # 基础局信息
        run = obs.get("run") or {}
        rbm.act = int(run.get("act", rbm.act) or rbm.act)
        rbm.floor = int(run.get("floor", rbm.floor) or rbm.floor)
        rbm.gold = int(player.get("gold", rbm.gold) or rbm.gold)

        # Deck profile
        deck = player.get("deck") or []
        if isinstance(deck, list) and deck:
            rbm.deck_size = len(deck)
            attacks = skills = powers = curses = zero_cost = high_cost = x_cost = 0
            aoe_count = heal_count = draw_count = 0
            total_cost = 0
            for card in deck:
                if not isinstance(card, dict):
                    continue
                ct = str(card.get("type", card.get("card_type", "")) or "").lower()
                cost = int(card.get("cost", card.get("energy_cost", 0)) or 0)
                cid = str(card.get("id", card.get("card_id", "")) or "")
                if ct in ("attack", "1"):
                    attacks += 1
                elif ct in ("skill", "2"):
                    skills += 1
                elif ct in ("power", "3"):
                    powers += 1
                elif ct in ("curse", "5"):
                    curses += 1
                if cost == 0:
                    zero_cost += 1
                elif cost >= 3:
                    high_cost += 1
                if _is_x_cost_card(card):
                    x_cost += 1
                if _is_aoe(cid):
                    aoe_count += 1
                if _is_heal_card(cid):
                    heal_count += 1
                if _is_draw_card(cid):
                    draw_count += 1
                total_cost += max(cost, 0)

            n = max(len(deck), 1)
            # 粗略的 build 画像
            rbm.frontload = attacks / n
            rbm.block = skills / n
            rbm.draw = draw_count / n   # 从 card_tags 的 "draw" tag 统计
            rbm.scaling = powers / n
            rbm.aoe = aoe_count / n
            rbm.heal = heal_count / n
            rbm.curse_density = curses / n
            rbm.zero_cost_density = zero_cost / n
            rbm.high_cost_density = high_cost / n
            rbm.x_cost_density = x_cost / n
            rbm.consistency = 1.0 - (len(deck) - 8) / 30.0  # 越薄越 consistent

        # Relic / potion count
        relics = player.get("relics") or []
        potions = player.get("potions") or []
        rbm.relic_count = len(relics) if isinstance(relics, list) else 0
        rbm.potion_count = len(potions) if isinstance(potions, list) else 0

        # Objective context（粗略启发式）
        hp = _extract_hp(obs)
        max_hp = max(int(player.get("max_hp", 1) or 1), 1)
        hp_ratio = hp / max_hp
        rbm.survival_priority = max(0.0, 1.0 - hp_ratio)
        rbm.boss_pressure = 1.0 if self.room_type == "boss" else 0.0
        rbm.elite_pressure = 1.0 if self.room_type == "elite" else 0.0
        rbm.preserve_hp_bias = max(0.0, 0.8 - hp_ratio)
        # resource_priority: gold 紧张 + 药水槽快满 → 该开始换/用资源
        # 假设典型 act 上限：gold ~200 足以支撑一次购物；药水槽 ~3
        gold_pressure = max(0.0, 1.0 - min(rbm.gold / 200.0, 1.0))
        potion_pressure = min(rbm.potion_count / 3.0, 1.0)
        rbm.resource_priority = 0.5 * gold_pressure + 0.5 * potion_pressure
