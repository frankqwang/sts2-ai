"""Bank Assembler：将编译好的 schema 对象组装成 UnifiedTokenBanks。

两步组装：
  1. assemble_shared() → SharedWorldBanks (战斗/非战斗共享)
  2. assemble_combat() → CombatBanks + action_bank (仅战斗)
最终合并为 UnifiedTokenBanks。
"""

from __future__ import annotations

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
    TK_PILE_SUMMARY, TK_MECHANISM, TK_MODIFIER,
    TK_PLAYED_ACTION, TK_TURN_SUMMARY, TK_COMBAT_SUMMARY,
    TK_BUILD_PROFILE, TK_DECK_CARD, TK_RELIC, TK_POTION, TK_OBJECTIVE,
    TK_ECONOMY, TK_COMBAT_FORECAST,
    TK_ACTION_CANDIDATE,
)
from networkV2.s4_compiler.mechanism_compiler import ActiveMechanism
from networkV2.s4_compiler.memory_compiler import MemoryCompiler
from core.relic_rules import relic_feature_vector, potion_feature_vector


# 归一化常量
_HP = 100.0
_BLK = 50.0
_DMG = 30.0
_NRG = 5.0
_COST = 5.0


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
    ) -> SharedWorldBanks:
        shared = SharedWorldBanks()

        # build_bank: 构筑画像 + deck cards
        shared.build_bank.add(Token(
            numeric=self._mem.compile_run_build_memory(rbm),
            token_type=TK_BUILD_PROFILE,
        ))
        for i, card in enumerate(deck_cards[:50]):  # cap at 50
            shared.build_bank.add(Token(
                numeric=[
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
                ],
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

        # route_bank: 路线（非战斗时由 route_compiler 填充，战斗时为空占位）
        # 后续实现

        return shared

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

        # --- board_bank ---
        cb.board_bank.add(self._player_token(player_rt, room_type))
        for i, card in enumerate(hand_cards):
            cb.board_bank.add(self._hand_card_token(card, i))
        for i, enemy in enumerate(enemies):
            cb.board_bank.add(self._enemy_core_token(enemy, i))
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

        # --- mechanism_bank ---
        for i, mech in enumerate(mechanisms):
            cb.mechanism_bank.add(self._mechanism_token(mech, i))

        # --- modifier_bank ---
        for i, mod in enumerate(modifiers):
            cb.modifier_bank.add(self._modifier_token(mod, i))

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
        s = p.powers.get
        return Token(
            numeric=[
                p.hp / _HP, p.hp_ratio, p.max_hp / _HP,
                p.block / _BLK, p.energy / _NRG, p.max_energy / _NRG,
                s("strength", 0) / 10.0, s("dexterity", 0) / 10.0,
                s("weak", 0) / 5.0, s("vulnerable", 0) / 5.0,
                s("frail", 0) / 5.0, s("artifact", 0) / 3.0,
                float(room_type == "monster"),
                float(room_type == "elite"),
                float(room_type == "boss"),
            ],
            token_type=TK_PLAYER,
        )

    def _hand_card_token(self, c: HandCardRuntime, order: int) -> Token:
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
        return Token(
            numeric=[
                c.current_cost / _COST,
                float(c.card_type == "attack"), float(c.card_type == "skill"),
                float(c.card_type == "power"),
                float(c.can_play), float(c.requires_target), float(c.is_upgraded),
                c.damage_est / _DMG, c.block_est / _BLK, c.draw_est / 3.0,
                float(c.retain), float(c.ethereal), float(c.exhaust),
                float(c.current_cost == 0),
                # 新增 9 维：target_type(4) + rarity(4) + upgrade_count(1)
                tt_enemy, tt_all, tt_self, tt_random,
                r_common, r_uncommon, r_rare, r_special,
                min(c.upgrade_count / 3.0, 1.0),
            ],
            token_type=TK_HAND_CARD, owner_id=c.card_id, order=order,
        )

    def _enemy_core_token(self, e: EnemyRuntime, order: int) -> Token:
        g = e.powers.get
        return Token(
            numeric=[
                e.hp / _HP, e.hp_ratio, e.max_hp / _HP, e.block / _BLK,
                float(e.is_hittable), float(e.intends_to_attack),
                e.total_intent_damage / _DMG,
                g("strength", 0) / 10.0, g("vulnerable", 0) / 5.0,
                g("weak", 0) / 5.0, g("poison", 0) / 20.0, g("artifact", 0) / 3.0,
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
        # 合计 15 + 3 + 5 + 6 + 1 = 30 维（<= tokenizer max_numeric_dim=32）
        return Token(
            numeric=combat_axes + target_axes + family_axes + role_axes + extra_axes,
            token_type=TK_ACTION_CANDIDATE,
            owner_id=a.source_card_id or a.source_potion_id or a.action_type,
            order=order,
            metadata={"label": a.label, "action_index": a.action_index},
        )
