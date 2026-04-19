"""Bank Assembler：将编译好的 schema 对象组装成 UnifiedTokenBanks。

两步组装：
  1. assemble_shared() → SharedWorldBanks (战斗/非战斗共享)
  2. assemble_combat() → CombatBanks + action_bank (仅战斗)
最终合并为 UnifiedTokenBanks。
"""

from __future__ import annotations

import math

from networkV2.s1_schema.entities import (
    PlayerRuntime, HandCardRuntime, EnemyRuntime, PileSummary,
    CardSemantics, RelicSemantics, PotionSemantics,
)
from networkV2.s1_schema.memory import TurnPrefixMemory, CombatMemory, RunBuildMemory
from networkV2.s1_schema.actions import ActionCandidate
from networkV2.s1_schema.primitives import MechanismType, ModifierType, ModifierPrimitive
from networkV2.s1_schema.token_banks import (
    Token, TokenBank, SharedWorldBanks, CombatBanks, UnifiedTokenBanks,
    TK_PLAYER, TK_HAND_CARD, TK_ENEMY_CORE, TK_ENEMY_INTENT,
    TK_PILE_SUMMARY, TK_DRAW_DIST, TK_DRAW_HORIZON, TK_MECHANISM, TK_MODIFIER,
    TK_POWER_INSTANCE,
    TK_PLAYED_ACTION, TK_TURN_SUMMARY, TK_COMBAT_SUMMARY,
    TK_BUILD_PROFILE, TK_DECK_CARD, TK_RELIC, TK_POTION, TK_OBJECTIVE,
    TK_ECONOMY, TK_COMBAT_FORECAST,
    TK_ACTION_CANDIDATE,
)
from networkV2.s1_schema.game_vocab import (
    power_semantic_group_onehot, power_class_idx_normalized,
    is_debuff_heuristic, N_POWER_SEMANTIC_GROUPS,
)
from networkV2.s1_schema.card_semantic_catalog import CARD_SEMANTICS
from networkV2.s1_schema.sim_catalog import GAME_CATALOG
from networkV2.s4_compiler.mechanism_compiler import ActiveMechanism
from networkV2.s4_compiler.memory_compiler import MemoryCompiler
from core.relic_rules import relic_feature_vector, potion_feature_vector
from core.card_tags import FUNCTIONAL_TAGS, NUM_FUNCTIONAL_TAGS


# 归一化常量
_HP = 100.0
_BLK = 50.0
_DMG = 30.0
_NRG = 5.0
_COST = 5.0


def _compute_combo_signals(
    c: HandCardRuntime,
    player: PlayerRuntime | None,
    draw_ctx: dict[str, object] | None,
) -> dict[str, float]:
    """根据手牌 + 玩家 + 牌堆 派生组合/牌序信号。

    规范（SCHEMA_CONVENTION.md）：**不硬编码游戏 power 名**。
    所有信号基于通用特征（card_type / stats / pile 组成），不查特定 power 名。
    power-specific synergy（如 Tactician 触发 discard）通过 player_token 的
    data-driven power vocab 暴露，由网络 attention 自己学习。

    返回 6 个 [0, 1] 信号：
      is_buff_provider          — 打出后给自己加 buff（先打）
      benefits_from_strength    — 伤害随 strength 提升（buff 后再打）
      is_card_draw              — 抽牌（先打扩展选项）
      actions_left_at_cost      — 当前能量还能打几张该 cost 的牌
      is_discard_card           — 牌的 commands 含 Discard（player 有 discard-trigger power 时高价值）
      pile_after_draw_attack_ratio — draw pile 中 attack 占比
    """
    # is_buff_provider
    snap = CARD_SEMANTICS.get(c.card_id)
    is_buff_provider = float(
        c.card_type == "power"
        or (c.card_type == "skill" and c.damage_est == 0
            and c.block_est == 0 and c.draw_est == 0
            and c.rarity not in ("curse", "status"))
    )

    # benefits_from_strength（基于 stat 查询，非硬编码 power 名）
    if c.card_type == "attack" and c.damage_est > 0 and player is not None:
        strength = 0
        for k, v in player.powers.items():
            if str(k).lower() == "strength":
                strength = int(v)
                break
        synergy_scale = 1.0
        if snap.has_functional_tag("multi_hit"):
            synergy_scale += 0.25
        if snap.has_functional_tag("aoe"):
            synergy_scale += 0.10
        if snap.has_functional_tag("x_cost"):
            synergy_scale += 0.15
        base_synergy = min((c.damage_est / _DMG) * synergy_scale, 1.0)
        benefits_from_strength = base_synergy * min(strength / 5.0, 1.0)
        if strength == 0:
            benefits_from_strength = 0.3 * base_synergy
    else:
        benefits_from_strength = 0.0

    is_card_draw = float(c.draw_est > 0 or snap.has_functional_tag("draw"))

    # actions_left_at_cost
    if not c.can_play:
        actions_left = 0.0
    elif c.current_cost <= 0:
        actions_left = 1.0
    elif player is not None:
        actions_left = min(player.energy / max(c.current_cost, 1), 5.0) / 5.0
    else:
        actions_left = 0.0

    is_discard_card = float(snap.has_functional_tag("discard"))

    horizon_key = "next4" if is_card_draw > 0 else "next2"
    horizon = (draw_ctx or {}).get(horizon_key) or {}
    pile_attack_ratio = float(horizon.get("attack_ratio", 0.0))

    return {
        "is_buff_provider": is_buff_provider,
        "benefits_from_strength": benefits_from_strength,
        "is_card_draw": is_card_draw,
        "actions_left_at_cost": actions_left,
        "is_discard_card": is_discard_card,
        "pile_after_draw_attack_ratio": pile_attack_ratio,
    }


class BankAssembler:
    """将编译好的 schema 对象组装成 UnifiedTokenBanks。"""

    def __init__(self) -> None:
        self._mem = MemoryCompiler()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def assemble(
        self,
        player_rt: PlayerRuntime,
        hand_cards_rt: list[HandCardRuntime],
        enemies_rt: list[EnemyRuntime],
        piles_rt: list[PileSummary],
        deck_cards: list[CardSemantics] | None = None,
        relics: list[RelicSemantics] | None = None,
        potions: list[PotionSemantics] | None = None,
        mechanism_states: list[ActiveMechanism] | None = None,
        modifier_states: list[ModifierPrimitive] | None = None,
        turn_prefix: TurnPrefixMemory | None = None,
        combat_memory: CombatMemory | None = None,
        run_build_memory: RunBuildMemory | None = None,
        action_candidates: list[ActionCandidate] | None = None,
        room_type: str = "monster",
        is_combat: bool = True,
        decision_domain: str = "",
        map_state: dict | None = None,
    ) -> UnifiedTokenBanks:
        mechanism_states = mechanism_states or []
        modifier_states = modifier_states or []
        turn_prefix = turn_prefix or TurnPrefixMemory()
        combat_memory = combat_memory or CombatMemory()
        run_build_memory = run_build_memory or RunBuildMemory()
        action_candidates = action_candidates or []

        shared = self._assemble_shared(
            run_build_memory, player_rt, room_type,
            deck_cards or [], relics or [], potions or [],
            map_state=map_state,
        )
        combat = None
        domain = "combat"
        if is_combat:
            combat = self._assemble_combat(
                player_rt, hand_cards_rt, enemies_rt, piles_rt,
                mechanism_states, modifier_states,
                turn_prefix, combat_memory, room_type,
            )
        else:
            domain = decision_domain or "noncombat"
        action_bank = self._assemble_actions(action_candidates)

        return UnifiedTokenBanks(
            shared=shared,
            combat=combat,
            action_bank=action_bank,
            decision_domain=domain,
        )

    # ------------------------------------------------------------------
    # Shared World Banks (6 组)
    # ------------------------------------------------------------------

    def _assemble_shared(
        self,
        rbm: RunBuildMemory,
        player_rt: PlayerRuntime,
        room_type: str,
        deck_cards: list[CardSemantics],
        relics: list[RelicSemantics],
        potions: list[PotionSemantics],
        map_state: dict | None = None,
    ) -> SharedWorldBanks:
        shared = SharedWorldBanks()

        # build_bank: 构筑画像 + deck cards
        shared.build_bank.add(Token(
            numeric=self._mem.compile_run_build_memory(rbm),
            token_type=TK_BUILD_PROFILE,
        ))
        for i, card in enumerate(deck_cards[:50]):  # cap at 50
            # P1② 修复：deck_card token 加语义通道。原来只有 11 维 coarse
            # (cost/type/rarity/upgraded)，同 cost+type+rarity 的牌在 token 层完全
            # 不可区分（如 Strike 和 Pommel Strike）。tokenizer 不读 owner_id，
            # 所以必须把身份编进 numeric。
            # 加 NUM_FUNCTIONAL_TAGS 维 one-hot 功能标签（aoe/draw/strength_scaling/
            # exhaust/innate 等），来自 core/card_tags.py 的离线提取。
            coarse = [
                card.base_cost / _COST,
                float(card.card_type == "attack"),
                float(card.card_type == "skill"),
                float(card.card_type == "power"),
                float(card.card_type == "curse"),
                float(card.card_type == "status"),
                float(card.is_upgraded),
                float(card.rarity == "rare"),
                float(card.rarity == "uncommon"),
                float(card.rarity == "common"),
                float(card.base_cost == 0),
            ]
            semantic = list(CARD_SEMANTICS.get(card.entity_id).functional_tag_vec)
            shared.build_bank.add(Token(
                numeric=coarse + semantic,
                token_type=TK_DECK_CARD,
                owner_id=card.entity_id,
                order=i,
            ))

        # inventory_bank: 遗物 + 药水（带静态规则表编码的语义向量）
        for i, relic in enumerate(relics[:25]):
            shared.inventory_bank.add(Token(
                numeric=[1.0] + relic_feature_vector(relic.entity_id),  # 1 存在 + 14 语义
                token_type=TK_RELIC,
                owner_id=relic.entity_id,
                order=i,
            ))
        for i, potion in enumerate(potions[:5]):
            shared.inventory_bank.add(Token(
                numeric=[1.0] + potion_feature_vector(potion.entity_id),  # 1 存在 + 8 语义
                token_type=TK_POTION,
                owner_id=potion.entity_id,
                order=i,
            ))
        if not relics and not potions:
            # 空 inventory 占位
            shared.inventory_bank.add(Token(
                numeric=[rbm.relic_count / 25.0, rbm.potion_count / 5.0],
                token_type=TK_RELIC,
            ))

        # economy_bank: 经济
        shared.economy_bank.add(Token(
            numeric=[
                rbm.gold / 500.0,
                rbm.floor / 60.0,
                rbm.act / 4.0,
            ],
            token_type=TK_ECONOMY,
        ))

        # objective_bank: 目标
        shared.objective_bank.add(Token(
            numeric=[
                rbm.survival_priority,
                rbm.resource_priority,
                rbm.preserve_hp_bias,
                rbm.boss_pressure,
                rbm.elite_pressure,
                player_rt.hp_ratio,
                float(room_type == "boss"),
                float(room_type == "elite"),
            ],
            token_type=TK_OBJECTIVE,
        ))

        # forecast_bank: 未来战斗压力
        # 当前用 build 画像做粗略估计（后续接精确的 encounter forecast）
        shared.forecast_bank.add(Token(
            numeric=[
                rbm.boss_pressure,
                rbm.elite_pressure,
                rbm.frontload,
                rbm.block,
                rbm.scaling,
                rbm.aoe,
                rbm.consistency,
            ],
            token_type=TK_COMBAT_FORECAST,
        ))

        # route_bank: 完整地图拓扑 + 玩家位置 + boss 位置 + 路径 forecast
        # 历史问题：之前是空占位，agent 只能 1 步看 1 步无法做战略路线规划。
        self._populate_route_bank(shared, map_state, player_rt)

        return shared

    def _populate_route_bank(
        self,
        shared: SharedWorldBanks,
        map_state: dict | None,
        player_rt: PlayerRuntime,
    ) -> None:
        """填充 route_bank：每个未来 map node 一个 token + summary token。"""
        if not map_state:
            # 无 map 信息（战斗中等）→ 空占位
            shared.route_bank.add(Token(
                numeric=[0.0] * 8, token_type=TK_OBJECTIVE, owner_id="map_empty",
            ))
            return

        nodes = map_state.get("nodes") or []
        boss = map_state.get("boss") or {}
        boss_col = int(boss.get("col", -1) or -1)
        boss_row = int(boss.get("row", -1) or -1)
        player_node = map_state.get("player") or {}
        # 取玩家 col/row（不一定有 - act 1 起点 row=0）
        player_col = int(player_node.get("col", -1) or -1)
        player_row = int(player_node.get("row", -1) or -1)

        # 1) summary token: 全局地图统计
        node_types_count = {}
        for n in nodes:
            t = str(n.get("type", "") or "").lower()
            node_types_count[t] = node_types_count.get(t, 0) + 1
        total_nodes = max(len(nodes), 1)
        boss_dist = max(boss_row - player_row, 0) if boss_row >= 0 and player_row >= 0 else 17

        shared.route_bank.add(Token(
            numeric=[
                len(nodes) / 60.0,
                node_types_count.get("monster", 0) / total_nodes,
                node_types_count.get("elite", 0) / total_nodes,
                node_types_count.get("event", 0) / total_nodes,
                node_types_count.get("rest_site", 0) / total_nodes,
                node_types_count.get("shop", 0) / total_nodes,
                node_types_count.get("treasure", 0) / total_nodes,
                boss_dist / 17.0,  # 距离 boss 的步数（act 内）
            ],
            token_type=TK_OBJECTIVE,
            owner_id="map_summary",
            order=0,
        ))

        # 2) per-node tokens（限制最多 30 个，按距离玩家排序优先未来节点）
        candidates = []
        for n in nodes:
            n_col = int(n.get("col", -1) or -1)
            n_row = int(n.get("row", -1) or -1)
            n_type = str(n.get("type", "") or "").lower()
            n_children = n.get("children") or []
            # 只关注未来节点（row >= player_row）
            if player_row >= 0 and n_row < player_row:
                continue
            dist = max(n_row - player_row, 0) if player_row >= 0 else n_row
            # 距离 player 越近越重要
            candidates.append((dist, n_col, n_row, n_type, len(n_children)))

        candidates.sort()  # 按 dist 升序
        for i, (dist, n_col, n_row, n_type, n_children) in enumerate(candidates[:30]):
            shared.route_bank.add(Token(
                numeric=[
                    n_col / 7.0,                                # 列归一化
                    n_row / 17.0,                               # 行归一化
                    dist / 17.0,                                # 距玩家步数
                    float(n_type == "monster"),
                    float(n_type == "elite"),
                    float(n_type == "boss"),
                    float(n_type == "event"),
                    float(n_type == "rest_site"),
                    float(n_type == "shop"),
                    float(n_type == "treasure"),
                    n_children / 4.0,                           # 出度（连通性）
                    float(n_col == boss_col and n_row == boss_row),  # 是不是 boss 节点
                ],
                token_type=TK_OBJECTIVE,
                owner_id=f"node_{n_col}_{n_row}",
                order=i + 1,
            ))

    # ------------------------------------------------------------------
    # Combat Banks (5 组)
    # ------------------------------------------------------------------

    def _assemble_combat(
        self,
        player_rt: PlayerRuntime,
        hand_cards: list[HandCardRuntime],
        enemies: list[EnemyRuntime],
        piles: list[PileSummary],
        mechanisms: list[ActiveMechanism],
        modifiers: list[ModifierPrimitive],
        turn_prefix: TurnPrefixMemory,
        combat_memory: CombatMemory,
        room_type: str,
    ) -> CombatBanks:
        cb = CombatBanks()

        # piles 按 type 索引，便于 _hand_card_token 计算 combo signals
        piles_by_type = {p.pile_type: p for p in piles}
        draw_ctx = self._build_draw_context(piles_by_type)

        # --- board_bank ---
        cb.board_bank.add(self._player_token(player_rt, room_type))
        for i, card in enumerate(hand_cards):
            cb.board_bank.add(self._hand_card_token(card, i, player_rt, draw_ctx))
        for i, enemy in enumerate(enemies):
            cb.board_bank.add(self._enemy_core_token(enemy, i))
            # intent token 也带 owner snapshot，让网络知道"这个 intent 来自哪只怪"
            # （多怪战斗时 attention 能把 intent 关联到正确 enemy_core）
            intent_owner_snap = self._owner_snapshot_enemy(enemy)
            for j, intent in enumerate(enemy.intents):
                cb.board_bank.add(Token(
                    numeric=[
                        float(intent.intent_type == "attack"),
                        float(intent.intent_type == "defend"),
                        float(intent.intent_type == "buff"),
                        float(intent.intent_type == "debuff"),
                        intent.damage / _DMG,
                        intent.total_damage / _DMG,
                        intent.repeats / 5.0,
                        *intent_owner_snap,   # 4 维 owner identity
                    ],
                    token_type=TK_ENEMY_INTENT,
                    owner_id=enemy.entity_id,
                    order=j,
                ))
        for i, pile in enumerate(piles):
            cb.board_bank.add(Token(
                numeric=[
                    pile.size / 30.0,
                    pile.attack_ratio, pile.skill_ratio,
                    pile.zero_cost_density,
                    pile.reshuffle_proximity,
                    float(pile.pile_type == "draw"),
                    float(pile.pile_type == "discard"),
                    float(pile.pile_type == "exhaust"),
                ],
                token_type=TK_PILE_SUMMARY,
                owner_id=pile.pile_type,
                order=i,
            ))
        for order, pile_name in enumerate(("draw", "discard", "exhaust"), start=len(piles)):
            cb.board_bank.add(self._draw_dist_token(pile_name, draw_ctx, order))
        for order, horizon_name in enumerate(("next2", "next4", "post_shuffle"), start=len(piles) + 3):
            cb.board_bank.add(self._draw_horizon_token(horizon_name, draw_ctx, order))

        # --- mechanism_bank ---
        for i, mech in enumerate(mechanisms):
            cb.mechanism_bank.add(self._mechanism_token(mech, i))

        # --- modifier_bank ---
        for i, mod in enumerate(modifiers):
            cb.modifier_bank.add(self._modifier_token(mod, i))

        # --- power_bank (v2) ---
        # 每个 active power（enemy + player）一个 token，覆盖全部 vocab。
        # 解决旧设计 _enemy_core_token / _player_token 的 top-N slot 截断问题：
        # 低频 power（Illusion / HardToKill / CrabRage / ... 230+ 个）之前完全看不见。
        self._append_power_tokens(cb.power_bank, enemies, player_rt)

        # --- turn_prefix_bank ---
        for i, action in enumerate(turn_prefix.played_actions):
            cb.turn_prefix_bank.add(Token(
                numeric=self._mem.compile_played_action(action),
                token_type=TK_PLAYED_ACTION,
                order=i,
            ))
        cb.turn_prefix_bank.add(Token(
            numeric=self._mem.compile_turn_prefix_summary(turn_prefix),
            token_type=TK_TURN_SUMMARY,
        ))

        # --- combat_memory_bank ---
        cb.combat_memory_bank.add(Token(
            numeric=self._mem.compile_combat_memory(combat_memory),
            token_type=TK_COMBAT_SUMMARY,
        ))

        return cb

    # ------------------------------------------------------------------
    # Power Bank (v2)
    # ------------------------------------------------------------------

    # Power stack 归一化：大多数 power 层数在 [0, 10]；StrengthPower 可以到 ±30，
    # 但我们 clip 到 [-1, 1] 量级。
    _POWER_STACK_NORM = 10.0

    def _append_power_tokens(
        self,
        bank: TokenBank,
        enemies: list[EnemyRuntime],
        player_rt: PlayerRuntime,
    ) -> None:
        """为每个 active power（enemy + player）生成一个 token 进入 power_bank。

        Numeric 编码（16 维）：
          [0]       stack_norm          — 层数 / 10，clip [-1.5, 1.5]
          [1]       is_player           — 归属：0=enemy, 1=player
          [2]       is_debuff_hint      — buff / debuff heuristic
          [3:11]    semantic_group      — 8 维 one-hot (见 power_semantic_group_onehot)
          [11]      class_idx_norm      — class_name 在对应 vocab 里的 index / VOCAB_SIZE
          [12:16]   owner snapshot      — 4 维 owner identity (让网络知道 power 在"谁身上")
                    [12] owner_hp_ratio
                    [13] owner_max_hp / 100
                    [14] owner_block / 50
                    [15] owner_is_hittable (player 视为永远 hittable=1)

        owner snapshot 和 enemy_core_token / player_token 的对应 numeric slot 一致，
        网络可以通过 attention 做 owner matching（hp_ratio+max_hp 几乎唯一确定一只怪）。
        """
        order = 0
        for enemy in enemies:
            if not getattr(enemy, "is_alive", True):
                continue
            owner_snap = self._owner_snapshot_enemy(enemy)
            for class_name, stacks in enemy.powers.items():
                try:
                    s = int(stacks)
                except (TypeError, ValueError):
                    s = 0
                if s == 0:
                    continue
                bank.add(self._build_power_token(
                    class_name=str(class_name),
                    stacks=s,
                    is_player=False,
                    owner_id=enemy.entity_id,
                    order=order,
                    owner_snapshot=owner_snap,
                ))
                order += 1
        # player powers
        player_snap = self._owner_snapshot_player(player_rt)
        for class_name, stacks in player_rt.powers.items():
            try:
                s = int(stacks)
            except (TypeError, ValueError):
                s = 0
            if s == 0:
                continue
            bank.add(self._build_power_token(
                class_name=str(class_name),
                stacks=s,
                is_player=True,
                owner_id="player",
                order=order,
                owner_snapshot=player_snap,
            ))
            order += 1

    @staticmethod
    def _owner_snapshot_enemy(e: EnemyRuntime) -> list[float]:
        """4 维 owner identity：[hp_ratio, max_hp/100, block/50, is_hittable]。
        和 _enemy_core_token 的 numeric 前几维对齐，网络可做 owner matching。"""
        return [
            float(e.hp_ratio),
            e.max_hp / _HP,
            e.block / _BLK,
            float(e.is_hittable),
        ]

    @staticmethod
    def _owner_snapshot_player(p: PlayerRuntime) -> list[float]:
        """player 版本。player 永远 hittable=1（设计一致性：attention 可通过 is_player
        和 is_hittable 组合唯一识别 player）。"""
        return [
            float(p.hp_ratio),
            p.max_hp / _HP,
            p.block / _BLK,
            1.0,  # player 固定 hittable
        ]

    def _build_power_token(
        self,
        class_name: str,
        stacks: int,
        is_player: bool,
        owner_id: str,
        order: int,
        owner_snapshot: list[float],
    ) -> Token:
        stack_norm = max(-1.5, min(1.5, stacks / self._POWER_STACK_NORM))
        # 优先用 sim game_catalog 的 base_classes（准确继承链）；fallback 到 heuristic。
        # #1 优化：sim 接通后 semantic group 覆盖从 ~60% → 95%+。
        base_classes = GAME_CATALOG.power_base_classes(class_name)
        semantic = power_semantic_group_onehot(
            class_name,
            base_classes=base_classes if base_classes else None,
        )
        class_norm = power_class_idx_normalized(class_name, is_player=is_player)
        # 优先用 sim 的 is_debuff_hint；None 时 fallback heuristic。
        # #4 优化：sim 的判定更全，不只是 13 个硬编码 stem。
        sim_debuff = GAME_CATALOG.power_is_debuff_hint(class_name)
        if sim_debuff is not None:
            debuff = float(sim_debuff)
        else:
            debuff = float(is_debuff_heuristic(class_name))
        numeric = [
            stack_norm,
            float(is_player),
            debuff,
            *semantic,            # 8 维
            class_norm,
            *owner_snapshot,      # 4 维 owner identity
        ]
        assert len(numeric) == 3 + N_POWER_SEMANTIC_GROUPS + 1 + 4, (
            f"power token numeric dim mismatch: got {len(numeric)}"
        )
        return Token(
            numeric=numeric,
            token_type=TK_POWER_INSTANCE,
            owner_id=owner_id,
            order=order,
            metadata={"class_name": class_name, "stacks": stacks},
        )

    # ------------------------------------------------------------------
    # Action Bank
    # ------------------------------------------------------------------

    def _assemble_actions(self, candidates: list[ActionCandidate]) -> TokenBank:
        bank = TokenBank(bank_name="action")
        for i, act in enumerate(candidates):
            bank.add(self._action_token(act, i))
        return bank

    # ------------------------------------------------------------------
    # Token 构建
    # ------------------------------------------------------------------

    def _player_token(self, p: PlayerRuntime, room_type: str) -> Token:
        """Player token（v2 精简版）：只含实体本身数值。

        所有 power 信息移到 power_bank（每个 active power 独立 token），
        覆盖全部 vocab。原先 6 维核心 stat + 17 维 vocab slot 的硬绑方案导致
        低频 power（~160 个）彻底不可见，已废弃。
        """
        return Token(
            numeric=[
                # HP / block / energy（6 维）
                p.hp / _HP, p.hp_ratio, p.max_hp / _HP,
                p.block / _BLK, p.energy / _NRG, p.max_energy / _NRG,
                # room_type one-hot（3 维）
                float(room_type == "monster"),
                float(room_type == "elite"),
                float(room_type == "boss"),
            ],
            token_type=TK_PLAYER,
        )

    def _hand_card_token(
        self,
        c: HandCardRuntime,
        order: int,
        player: PlayerRuntime | None = None,
        draw_ctx: dict[str, object] | None = None,
    ) -> Token:
        # target_type one-hot（常见值）
        tt = c.target_type
        tt_enemy = float(tt in ("enemy", "single_enemy", "type_1"))
        tt_all = float(tt in ("all_enemies", "type_2"))
        tt_self = float(tt in ("self", "none", "type_0"))
        tt_random = float(tt in ("random_enemy", "type_3"))
        # rarity one-hot
        r = c.rarity
        r_common = float(r in ("basic", "common"))
        r_uncommon = float(r == "uncommon")
        r_rare = float(r == "rare")
        r_special = float(r in ("special", "curse", "status"))

        # ---- Combo / sequencing signals (新增 6 维) ----
        # 修组合/牌序学习的关键派生特征。原 hand_card token 没这些，网络
        # 只能从 attention 间接挖关系，对 buff→damage 链路学得很慢。
        combo = _compute_combo_signals(c, player, draw_ctx)

        return Token(
            numeric=[
                c.current_cost / _COST,
                float(c.card_type == "attack"), float(c.card_type == "skill"),
                float(c.card_type == "power"),
                float(c.can_play), float(c.requires_target), float(c.is_upgraded),
                c.damage_est / _DMG, c.block_est / _BLK, c.draw_est / 3.0,
                float(c.retain), float(c.ethereal), float(c.exhaust),
                float(c.current_cost == 0),
                # target_type(4) + rarity(4) + upgrade_count(1)
                tt_enemy, tt_all, tt_self, tt_random,
                r_common, r_uncommon, r_rare, r_special,
                min(c.upgrade_count / 3.0, 1.0),
                # combo signals (6): buff/strength/draw/cost-budget/is_discard_card/pile-after
                combo["is_buff_provider"],
                combo["benefits_from_strength"],
                combo["is_card_draw"],
                combo["actions_left_at_cost"],
                combo["is_discard_card"],
                combo["pile_after_draw_attack_ratio"],
            ],
            token_type=TK_HAND_CARD, owner_id=c.card_id, order=order,
        )

    @staticmethod
    def _pile_or_empty(
        piles_by_type: dict[str, PileSummary],
        pile_type: str,
    ) -> PileSummary:
        return piles_by_type.get(pile_type, PileSummary(pile_type=pile_type))

    @staticmethod
    def _no_tag_probability(total_cards: int, tagged_cards: int, draws: int) -> float:
        draws = max(int(draws), 0)
        total_cards = max(int(total_cards), 0)
        tagged_cards = max(int(tagged_cards), 0)
        if draws <= 0 or total_cards <= 0:
            return 1.0
        draws = min(draws, total_cards)
        untagged = max(total_cards - tagged_cards, 0)
        if tagged_cards <= 0:
            return 1.0
        if draws > untagged:
            return 0.0
        return math.comb(untagged, draws) / max(math.comb(total_cards, draws), 1)

    def _card_snaps(self, card_ids: list[str]) -> list:
        return [snap for cid in card_ids if (snap := CARD_SEMANTICS.get(cid)).card_id]

    def _mean_cost(self, card_ids: list[str]) -> float:
        snaps = self._card_snaps(card_ids)
        if not snaps:
            return 0.0
        return sum(max(int(s.base_cost), 0) for s in snaps) / len(snaps)

    def _mean_attack_ratio(self, card_ids: list[str]) -> float:
        snaps = self._card_snaps(card_ids)
        if not snaps:
            return 0.0
        return sum(1.0 for s in snaps if s.card_type == "attack") / len(snaps)

    def _mean_zero_cost_density(self, card_ids: list[str]) -> float:
        snaps = self._card_snaps(card_ids)
        if not snaps:
            return 0.0
        return sum(1.0 for s in snaps if int(s.base_cost) == 0) / len(snaps)

    def _tagged_card_count(self, card_ids: list[str], tag: str) -> int:
        return sum(1 for snap in self._card_snaps(card_ids) if snap.has_functional_tag(tag))

    def _weighted_horizon_scalar(
        self,
        draw_ids: list[str],
        discard_ids: list[str],
        horizon: int,
        getter,
    ) -> float:
        draw_take = min(max(horizon, 0), len(draw_ids))
        discard_take = min(max(horizon - draw_take, 0), len(discard_ids))
        total_take = draw_take + discard_take
        if total_take <= 0:
            return 0.0
        acc = 0.0
        if draw_take > 0:
            acc += draw_take * getter(draw_ids)
        if discard_take > 0:
            acc += discard_take * getter(discard_ids)
        return acc / total_take

    def _horizon_tag_probabilities(
        self,
        draw_ids: list[str],
        discard_ids: list[str],
        horizon: int,
    ) -> list[float]:
        draw_take = min(max(horizon, 0), len(draw_ids))
        discard_take = min(max(horizon - draw_take, 0), len(discard_ids))
        probs: list[float] = []
        for tag in FUNCTIONAL_TAGS:
            draw_no = self._no_tag_probability(len(draw_ids), self._tagged_card_count(draw_ids, tag), draw_take)
            discard_no = self._no_tag_probability(
                len(discard_ids),
                self._tagged_card_count(discard_ids, tag),
                discard_take,
            )
            probs.append(1.0 - (draw_no * discard_no))
        return probs

    def _post_shuffle_tag_probabilities(self, discard_ids: list[str], horizon: int = 4) -> list[float]:
        draws = min(max(horizon, 0), len(discard_ids))
        probs: list[float] = []
        for tag in FUNCTIONAL_TAGS:
            no_tag = self._no_tag_probability(len(discard_ids), self._tagged_card_count(discard_ids, tag), draws)
            probs.append(1.0 - no_tag)
        return probs

    def _build_draw_context(self, piles_by_type: dict[str, PileSummary]) -> dict[str, object]:
        draw = self._pile_or_empty(piles_by_type, "draw")
        discard = self._pile_or_empty(piles_by_type, "discard")
        exhaust = self._pile_or_empty(piles_by_type, "exhaust")
        draw_ids = list(draw.card_ids)
        discard_ids = list(discard.card_ids)
        exhaust_ids = list(exhaust.card_ids)

        ctx: dict[str, object] = {
            "draw": {
                "pile": draw,
                "mean_vec": list(CARD_SEMANTICS.mean_functional_tag_vec(draw_ids)),
            },
            "discard": {
                "pile": discard,
                "mean_vec": list(CARD_SEMANTICS.mean_functional_tag_vec(discard_ids)),
            },
            "exhaust": {
                "pile": exhaust,
                "mean_vec": list(CARD_SEMANTICS.mean_functional_tag_vec(exhaust_ids)),
            },
        }
        for horizon_name, horizon in (("next2", 2), ("next4", 4)):
            draw_take = min(horizon, len(draw_ids))
            discard_take = min(max(horizon - draw_take, 0), len(discard_ids))
            ctx[horizon_name] = {
                "tag_probs": self._horizon_tag_probabilities(draw_ids, discard_ids, horizon),
                "reshuffle_prob": 1.0 if discard_take > 0 else 0.0,
                "attack_ratio": self._weighted_horizon_scalar(draw_ids, discard_ids, horizon, self._mean_attack_ratio),
                "zero_cost_density": self._weighted_horizon_scalar(draw_ids, discard_ids, horizon, self._mean_zero_cost_density),
                "mean_cost": self._weighted_horizon_scalar(draw_ids, discard_ids, horizon, self._mean_cost),
                "draw_take": draw_take / max(horizon, 1),
                "discard_take": discard_take / max(horizon, 1),
            }
        ctx["post_shuffle"] = {
            "tag_probs": self._post_shuffle_tag_probabilities(discard_ids, horizon=4),
            "reshuffle_prob": 1.0 if discard.size > 0 else 0.0,
            "attack_ratio": self._mean_attack_ratio(discard_ids),
            "zero_cost_density": self._mean_zero_cost_density(discard_ids),
            "mean_cost": self._mean_cost(discard_ids),
            "draw_take": 0.0,
            "discard_take": 1.0 if discard.size > 0 else 0.0,
        }
        return ctx

    def _draw_dist_token(self, pile_name: str, draw_ctx: dict[str, object], order: int) -> Token:
        pile_info = draw_ctx.get(pile_name) or {}
        pile = pile_info.get("pile") or PileSummary(pile_type=pile_name)
        mean_vec = pile_info.get("mean_vec") or [0.0] * NUM_FUNCTIONAL_TAGS
        return Token(
            numeric=[
                pile.size / 30.0,
                pile.attack_ratio,
                pile.skill_ratio,
                pile.power_count / max(pile.size, 1),
                pile.zero_cost_density,
                pile.reshuffle_proximity,
                float(pile_name == "draw"),
                float(pile_name == "discard"),
                float(pile_name == "exhaust"),
                *mean_vec,
            ],
            token_type=TK_DRAW_DIST,
            owner_id=pile_name,
            order=order,
        )

    def _draw_horizon_token(self, horizon_name: str, draw_ctx: dict[str, object], order: int) -> Token:
        horizon = draw_ctx.get(horizon_name) or {}
        tag_probs = horizon.get("tag_probs") or [0.0] * NUM_FUNCTIONAL_TAGS
        return Token(
            numeric=[
                *tag_probs,
                float(horizon.get("reshuffle_prob", 0.0)),
                float(horizon.get("attack_ratio", 0.0)),
                float(horizon.get("zero_cost_density", 0.0)),
                float(horizon.get("mean_cost", 0.0)) / _COST,
                float(horizon.get("draw_take", 0.0)),
                float(horizon.get("discard_take", 0.0)),
            ],
            token_type=TK_DRAW_HORIZON,
            owner_id=horizon_name,
            order=order,
        )

    def _enemy_core_token(self, e: EnemyRuntime, order: int) -> Token:
        """Enemy core token（v2 精简版）：只含实体本身数值。

        所有 power 信息移到 power_bank（每个 active power 独立 token，带
        semantic group + class_idx 编码），覆盖全部 vocab。原先 13 维 base +
        19 维 vocab slot 的硬绑方案导致低频 monster power（~44 个）彻底不可见，
        已废弃。
        """
        return Token(
            numeric=[
                # HP / block / 存活状态（8 维）
                e.hp / _HP, e.hp_ratio, e.max_hp / _HP, e.block / _BLK,
                float(e.is_hittable), float(e.intends_to_attack),
                e.total_intent_damage / _DMG,
                float(e.max_hp >= 80),
            ],
            token_type=TK_ENEMY_CORE, owner_id=e.entity_id, order=order,
        )

    def _mechanism_token(self, m: ActiveMechanism, order: int) -> Token:
        p = m.primitive
        return Token(
            numeric=[
                float(p.mechanism_type == MechanismType.PHASE_TRANSITION),
                float(p.mechanism_type == MechanismType.WINDOW),
                float(p.mechanism_type == MechanismType.SUMMON_CYCLE),
                float(p.mechanism_type == MechanismType.THRESHOLD_GATE),
                float(p.mechanism_type == MechanismType.SHIELD_PROGRESS),
                float(m.is_active), float(m.window_open),
                float(m.triggered), float(m.broken),
                m.current_layers / 5.0, float(m.summon_active),
                float(bool(m.current_phase_id)),
            ],
            token_type=TK_MECHANISM, owner_id=m.owner_enemy_id, order=order,
            metadata={"description": p.description, "phase_id": m.current_phase_id},
        )

    def _modifier_token(self, mod: ModifierPrimitive, order: int) -> Token:
        return Token(
            numeric=[
                float(mod.modifier_type == ModifierType.DAMAGE_CAP),
                float(mod.modifier_type == ModifierType.TARGET_RESTRICTION),
                float(mod.modifier_type == ModifierType.EFFECT_SCALING),
                float(mod.modifier_type == ModifierType.ON_PLAY_TRIGGER),
                float(mod.modifier_type == ModifierType.ON_HIT_TRIGGER),
                float(mod.modifier_type == ModifierType.DRAW_MODIFIER),
                float(mod.modifier_type == ModifierType.EXHAUST_MODIFIER),
                float(mod.modifier_type == ModifierType.PHASE_TRANSITION_EFFECT),
                float(mod.active), mod.current_value / 10.0,
            ],
            token_type=TK_MODIFIER, owner_id=mod.owner_id, order=order,
            metadata={"description": mod.description},
        )

    def _action_token(self, a: ActionCandidate, order: int) -> Token:
        # ---- 战斗向通道 (15) ----
        combat_axes = [
            float(a.action_type == "play_card"),
            float(a.action_type == "end_turn"),
            float(a.action_type in ("use_potion", "drink_potion")),
            float(a.action_type in ("select_hand_card", "select_card_option")),
            float(a.source_card_type == "attack"),
            float(a.source_card_type == "skill"),
            float(a.source_card_type == "power"),
            a.cost / _COST, a.damage_est / _DMG,
            a.block_est / _BLK, a.draw_est / 3.0,
            float(a.is_zero_cost), float(a.is_x_cost),
            float(a.exhausts), float(a.ends_turn),
        ]
        # ---- 目标/legacy role 通道 (3) ----
        # attack/block 已被 source_card_type 覆盖，不再重复；保留 draw/aoe 作语义区分
        target_axes = [
            float(a.has_target),
            float("draw" in a.roles),
            float("aoe" in a.roles),
        ]
        # ---- non-combat family 通道 (5) ----
        # 让 rest/shop/card_reward/map/event 动作在 token 上直接可分
        family_axes = [
            float(a.family == "rest"),
            float(a.family == "shop"),
            float(a.family == "card_reward"),
            float(a.family == "map"),
            float(a.family == "event_option"),
        ]
        # ---- non-combat role 通道 (6) ----
        # rest=heal/setup/resource/terminal ; shop=build/buff/resource/terminal
        # card_reward=attack/block/buff/terminal ; event=resource/terminal
        role_axes = [
            float("heal" in a.roles),
            float("setup" in a.roles),
            float("resource" in a.roles),
            float("terminal" in a.roles),
            float("build" in a.roles),
            float("buff" in a.roles),
        ]
        # ---- preview 扩展 (1) ----
        HEAL_NORM = 20.0
        extra_axes = [a.heal_est / HEAL_NORM]
        # ---- R2.2: 非战斗 option 新通道 (3 + 9 = 12) ----
        # rarity_weight (1): card_reward/shop 选卡的稀有度连续权重
        # price_ratio (1):  shop 物品价格占当前 gold 的比例（0 = 免费, 1 = 刚好买得起）
        # can_afford (1):   shop 能否购买（0/1）
        # event_kind (9):   EVENT_KINDS one-hot，区分"拿金币"/"拿 relic"/"掉血换收益"等
        # 原先所有 event option 的 roles 都是 "resource"、shop 不编价格、card_reward 不编稀有度
        # → 网络只能靠终局长 horizon 回传区分选项。现在这些通道直接 token 层可分。
        from networkV2.s1_schema.actions import EVENT_KINDS
        noncombat_axes = [
            float(a.rarity_weight),
            float(a.price_ratio),
            float(a.can_afford),
        ]
        event_kind_axes = [float(a.event_kind == k) for k in EVENT_KINDS]
        # ---- U8: route 专属通道 (2) ----
        # route_risk / route_value: map 节点的 [0,1] 威胁 / 价值,不走 _DMG/_BLK 归一化
        route_axes = [float(a.route_risk), float(a.route_value)]
        # ---- Skada priors 通道 (4) ----
        # TODO(cleanup): synergy_prior 现在永远是 0(填值链路已删),重训时砍成 3 维,
        # 同步 max_numeric_dim 58→57。详见 actions.py:145。
        skada_prior_axes = [
            float(a.pick_rate_prior),
            float(a.win_rate_delta_prior),
            float(a.deck_win_rate_prior),
            float(a.synergy_prior),
        ]
        # ---- 路径规划通道 (8) ----
        # 给 route candidate 喂"从此 child 到 boss 的全局路径 stat":
        # 6 维 type rate(rest/shop/elite/treasure/event/monster 在下游路径上的占比)
        # + best_rest_count(下游最优路径的 rest 数归一化)
        # + path_length_norm(到 boss 最短距离归一化)
        # 让网络能做"全局权衡"而非只看下一个节点。非 map action 默认 0。
        route_path_axes = [
            float(a.route_path_rest_rate),
            float(a.route_path_shop_rate),
            float(a.route_path_elite_rate),
            float(a.route_path_treasure_rate),
            float(a.route_path_event_rate),
            float(a.route_path_monster_rate),
            float(a.route_best_rest_count),
            float(a.route_path_length_norm),
        ]
        # Skada victory-runs 挖出的 data-driven 先验(2 维):
        # frequency:该 fingerprint 在高手群体出现频率 → 等价 "受欢迎的赢法"
        # efficiency:1 - duration_normalized → 等价 "更快的赢法"
        route_data_prior_axes = [
            float(a.route_prior_frequency),
            float(a.route_prior_efficiency),
        ]
        # 合计 15+3+5+6+1+3+9+2+4+8+2 = 58 维(= tokenizer max_numeric_dim=58)
        return Token(
            numeric=(
                combat_axes + target_axes + family_axes + role_axes
                + extra_axes + noncombat_axes + event_kind_axes + route_axes
                + skada_prior_axes + route_path_axes + route_data_prior_axes
            ),
            token_type=TK_ACTION_CANDIDATE,
            owner_id=a.source_card_id or a.source_potion_id or a.action_type,
            order=order,
            metadata={"label": a.label, "action_index": a.action_index},
        )
