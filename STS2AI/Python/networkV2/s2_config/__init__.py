"""机制配置层：mechanism_config registry + auto modifier rules。"""

from networkV2.s2_config.mechanism_registry import (
    EncounterMechanismConfig,
    MechanismRegistry,
    get_registry,
)
from networkV2.s2_config.auto_modifier_rules import AUTO_MODIFIER_RULES, compile_auto_modifiers
