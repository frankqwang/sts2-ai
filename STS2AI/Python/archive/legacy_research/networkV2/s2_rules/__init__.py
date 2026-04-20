"""规则层：encounter rules registry + power modifier rules。"""

from networkV2.s2_rules.encounter_registry import (
    EncounterRuleset,
    EncounterRuleRegistry,
    get_encounter_registry,
)
from networkV2.s2_rules.power_modifier_rules import POWER_MODIFIER_RULES, build_power_modifiers
