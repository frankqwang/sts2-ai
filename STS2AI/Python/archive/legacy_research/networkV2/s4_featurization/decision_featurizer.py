"""DecisionFeaturizer - 决策特征化主入口。

纯函数。支持 combat + non-combat domain 路由。

战斗时: RuntimeInstances + MechanismStates + RuleModifiers + ActionCandidates
非战斗时: 对应 domain 的 OptionBuilder 构建选项

Shared banks (build/inventory/economy/objective/forecast) 始终编译。
"""

from __future__ import annotations

from typing import Any

from networkV2.s1_schema.token_banks import UnifiedTokenBanks
from networkV2.s1_schema.memory import TurnPrefixMemory, CombatMemory, RunBuildMemory
from networkV2.s2_rules.encounter_registry import get_encounter_registry

from networkV2.s4_featurization.runtime_extractor import RuntimeExtractor
from networkV2.s4_featurization.mechanism_inferer import MechanismInferer
from networkV2.s4_featurization.modifier_inferer import ModifierInferer
from networkV2.s4_featurization.action_extractor import ActionExtractor
from networkV2.s4_featurization.token_bank_builder import TokenBankBuilder
from networkV2.s4_featurization.noncombat.card_reward_options import CardRewardOptionBuilder
from networkV2.s4_featurization.noncombat.shop_options import ShopOptionBuilder
from networkV2.s4_featurization.noncombat.route_options import RouteOptionBuilder
from networkV2.s4_featurization.noncombat.rest_options import RestOptionBuilder
from networkV2.s4_featurization.noncombat.event_options import EventOptionBuilder
from networkV2.s4_featurization.noncombat.selection_options import SelectionOptionBuilder

# domain 判断映射
_COMBAT_DOMAINS = {"monster", "elite", "boss", "combat", "hand_select", "card_select"}
_NONCOMBAT_DOMAIN_MAP = {
    "card_reward": "card_reward",
    "combat_rewards": "card_reward",
    "shop": "shop",
    "map": "route",
    "rest_site": "rest",
    "event": "event",
    # treasure / relic_select / relic_reward 都是“从若干奖励里选一个”的 selection 域。
    # 之前临时映射到 event 会让 EventOptionBuilder 读不懂 legal_actions，结果 action_bank
    # 为空。这里单独落到 selection builder。
    "treasure": "selection",
    "relic_select": "selection",
    "relic_reward": "selection",
}


class DecisionFeaturizer:
    """统一决策特征化器。纯函数设计，支持战斗和非战斗。"""

    def __init__(self) -> None:
        self._runtime = RuntimeExtractor()
        self._mechanism = MechanismInferer()
        self._modifier = ModifierInferer()
        self._action = ActionExtractor()
        self._bank_builder = TokenBankBuilder()
        self._card_reward = CardRewardOptionBuilder()
        self._shop = ShopOptionBuilder()
        self._route = RouteOptionBuilder()
        self._rest = RestOptionBuilder()
        self._event = EventOptionBuilder()
        self._selection = SelectionOptionBuilder()

    def featurize(
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
         deck_cards, relics, potions, combat_meta) = self._runtime.extract(obs)

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
        ruleset_registry = get_encounter_registry()
        ruleset = ruleset_registry.get(encounter_id)
        mechanism_states = self._mechanism.infer(enemies_rt, ruleset)
        modifier_states = self._modifier.infer(enemies_rt, ruleset)
        action_candidates = self._action.extract(legal_actions, hand_cards_rt, enemies_rt)

        return self._bank_builder.build(
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
        # 选择 domain-specific option builder
        if domain == "card_reward":
            candidates = self._card_reward.build(obs, legal_actions)
        elif domain == "shop":
            candidates = self._shop.build(obs, legal_actions)
        elif domain == "route":
            candidates = self._route.build(obs, legal_actions)
        elif domain == "rest":
            candidates = self._rest.build(obs, legal_actions)
        elif domain == "event":
            candidates = self._event.build(obs, legal_actions)
        elif domain == "selection":
            candidates = self._selection.build(obs, legal_actions)
        else:
            candidates = []

        # 非战斗: shared banks + action_bank (options), 无 combat banks
        return self._bank_builder.build(
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
