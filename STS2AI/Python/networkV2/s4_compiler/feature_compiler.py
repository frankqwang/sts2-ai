"""CombatFeatureCompiler - 特征编译主入口。

纯函数。支持 combat + non-combat domain 路由。

战斗时: RuntimeInstances + MechanismStates + RuleModifiers + ActionCandidates
非战斗时: 对应 domain 的 OptionCompiler 编译选项

Shared banks (build/inventory/economy/objective/forecast) 始终编译。
"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.token_banks import UnifiedTokenBanks
from networkV2.s1_schema.memory import TurnPrefixMemory, CombatMemory, RunBuildMemory
from networkV2.s2_config.mechanism_registry import get_registry

from networkV2.s4_compiler.runtime_compiler import RuntimeCompiler
from networkV2.s4_compiler.mechanism_compiler import MechanismCompiler
from networkV2.s4_compiler.modifier_compiler import ModifierCompiler
from networkV2.s4_compiler.memory_compiler import MemoryCompiler
from networkV2.s4_compiler.action_compiler import ActionCompiler
from networkV2.s4_compiler.bank_assembler import BankAssembler
from networkV2.s4_compiler.noncombat.card_reward_compiler import CardRewardCompiler
from networkV2.s4_compiler.noncombat.shop_compiler import ShopCompiler
from networkV2.s4_compiler.noncombat.route_compiler import RouteCompiler
from networkV2.s4_compiler.noncombat.rest_compiler import RestCompiler
from networkV2.s4_compiler.noncombat.event_compiler import EventCompiler

# domain 判断映射
_COMBAT_DOMAINS = {"monster", "elite", "boss", "combat", "hand_select", "card_select"}
_NONCOMBAT_DOMAIN_MAP = {
    "card_reward": "card_reward",
    "combat_rewards": "card_reward",
    "shop": "shop",
    "map": "route",
    "rest_site": "rest",
    "event": "event",
    # P3 修复：加 treasure / relic_select 映射。原先缺这两个 key → 走 _resolve_domain
    # 的 default fallback 直接进 combat 分支 → 编译空 banks / 错把非战斗 legal 当战斗
    # 动作。选 "event" 作目标 domain 是因为两者都是"选项式非战斗决策"，语义最接近，
    # 暂无专属 compiler。后续可加 treasure_compiler / relic_select_compiler。
    "treasure": "event",
    "relic_select": "event",
    "relic_reward": "event",
}


class CombatFeatureCompiler:
    """统一特征编译器。纯函数设计，支持战斗和非战斗。"""

    def __init__(self) -> None:
        self._runtime = RuntimeCompiler()
        self._mechanism = MechanismCompiler()
        self._modifier = ModifierCompiler()
        self._memory = MemoryCompiler()
        self._action = ActionCompiler()
        self._assembler = BankAssembler()
        # 非战斗 compilers
        self._card_reward = CardRewardCompiler()
        self._shop = ShopCompiler()
        self._route = RouteCompiler()
        self._rest = RestCompiler()
        self._event = EventCompiler()

    def compile(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        combat_memory: CombatMemory | None = None,
        turn_prefix: TurnPrefixMemory | None = None,
        run_build_memory: RunBuildMemory | None = None,
        encounter_id: str = "",
        room_type: str = "monster",
    ) -> UnifiedTokenBanks:
        combat_memory = combat_memory or CombatMemory()
        turn_prefix = turn_prefix or TurnPrefixMemory()
        run_build_memory = run_build_memory or RunBuildMemory()

        # 判断决策域
        state_type = str(obs.get("state_type", "") or "").lower()
        domain = self._resolve_domain(state_type, room_type)

        # Runtime (始终编译 — player, deck, relics, potions 是 shared)
        (player_rt, hand_cards_rt, enemies_rt, piles_rt,
         deck_cards, relics, potions, combat_meta) = self._runtime.compile(obs)

        if not encounter_id and combat_meta.get("encounter_id"):
            encounter_id = combat_meta["encounter_id"]

        if domain == "combat":
            return self._compile_combat(
                obs, legal_actions, player_rt, hand_cards_rt, enemies_rt,
                piles_rt, deck_cards, relics, potions,
                combat_memory, turn_prefix, run_build_memory,
                encounter_id, room_type,
            )
        else:
            return self._compile_noncombat(
                obs, legal_actions, domain,
                player_rt, deck_cards, relics, potions,
                run_build_memory, room_type,
            )

    def _resolve_domain(self, state_type: str, room_type: str) -> str:
        """判断当前决策域。non-combat state_type 优先级高于 room_type。"""
        # 先检查是否是已知的非战斗 state_type
        if state_type in _NONCOMBAT_DOMAIN_MAP:
            return _NONCOMBAT_DOMAIN_MAP[state_type]
        # 否则看是不是战斗
        if state_type in _COMBAT_DOMAINS or room_type in _COMBAT_DOMAINS:
            return "combat"
        return "combat"  # 默认 fallback

    def _compile_combat(self, obs, legal_actions, player_rt, hand_cards_rt,
                        enemies_rt, piles_rt, deck_cards, relics, potions,
                        combat_memory, turn_prefix, run_build_memory,
                        encounter_id, room_type):
        registry = get_registry()
        mech_config = registry.get(encounter_id)
        mechanism_states = self._mechanism.compile(enemies_rt, mech_config)
        modifier_states = self._modifier.compile(enemies_rt, mech_config)
        action_candidates = self._action.compile(legal_actions, hand_cards_rt, enemies_rt)

        return self._assembler.assemble(
            player_rt=player_rt, hand_cards_rt=hand_cards_rt,
            enemies_rt=enemies_rt, piles_rt=piles_rt,
            deck_cards=deck_cards, relics=relics, potions=potions,
            mechanism_states=mechanism_states, modifier_states=modifier_states,
            turn_prefix=turn_prefix, combat_memory=combat_memory,
            run_build_memory=run_build_memory,
            action_candidates=action_candidates, room_type=room_type,
            map_state=obs.get("map"),
        )

    def _compile_noncombat(self, obs, legal_actions, domain,
                           player_rt, deck_cards, relics, potions,
                           run_build_memory, room_type):
        # 选择 domain-specific compiler
        if domain == "card_reward":
            candidates = self._card_reward.compile(obs, legal_actions)
        elif domain == "shop":
            candidates = self._shop.compile(obs, legal_actions)
        elif domain == "route":
            candidates = self._route.compile(obs, legal_actions)
        elif domain == "rest":
            candidates = self._rest.compile(obs, legal_actions)
        elif domain == "event":
            candidates = self._event.compile(obs, legal_actions)
        else:
            candidates = []

        # 非战斗: shared banks + action_bank (options), 无 combat banks
        return self._assembler.assemble(
            player_rt=player_rt,
            hand_cards_rt=[], enemies_rt=[], piles_rt=[],
            deck_cards=deck_cards, relics=relics, potions=potions,
            action_candidates=candidates,
            run_build_memory=run_build_memory,
            room_type=room_type,
            is_combat=False,
            decision_domain=domain,
            map_state=obs.get("map"),
        )
