"""运行时 sim 常量。"""

from __future__ import annotations

from constants import ARTIFACTS_ROOT, ENV_ROOT, REPO_ROOT, SIM_HOST_EXE, SIM_LEGACY_DLL

PROTO_PROTOCOL_VERSION = 1
PROTO_SCHEMA_ID = "sts2-proto-v1"

OP_HANDSHAKE = 0x00
OP_RESET = 0x01
OP_STATE = 0x02
OP_STEP = 0x03
OP_BATCH_STEP = 0x04
OP_SAVE_STATE = 0x05
OP_LOAD_STATE = 0x06
OP_DELETE_STATE = 0x07
OP_PERF_STATS = 0x08
OP_RESET_PERF_STATS = 0x09
OP_STEP_LOCAL_POLICY = 0x0A
OP_LOAD_ORT_MODEL = 0x0B
OP_RUN_COMBAT_LOCAL = 0x0C
OP_EXPORT_STATE = 0x0D
OP_IMPORT_STATE = 0x0E
OP_SKIP_COMBAT = 0x0F
OP_SEARCH_COMBAT_MCTS = 0x10
OP_COMBAT_RESET = 0x11
OP_COMBAT_STEP = 0x12
OP_COMBAT_STATE = 0x13

STATUS_OK = 0
STATUS_REJECTED_ACTION = 1
STATUS_SIMULATOR_ERROR = 2
STATUS_PROTOCOL_ERROR = 3

COMBAT_STATE_TYPES = frozenset({"monster", "elite", "boss", "hand_select"})

STATE_BEARING_OPCODES = frozenset(
    {
        OP_RESET,
        OP_STATE,
        OP_STEP,
        OP_BATCH_STEP,
        OP_LOAD_STATE,
        OP_IMPORT_STATE,
        OP_LOAD_ORT_MODEL,
        OP_RUN_COMBAT_LOCAL,
        OP_SKIP_COMBAT,
        OP_COMBAT_RESET,
        OP_COMBAT_STEP,
        OP_COMBAT_STATE,
    }
)

ACTION_TYPES: dict[int, str] = {
    1: "wait",
    2: "play_card",
    3: "end_turn",
    4: "choose_map_node",
    5: "claim_reward",
    6: "select_card_reward",
    7: "skip_card_reward",
    8: "choose_rest_option",
    9: "shop_purchase",
    10: "shop_exit",
    11: "choose_event_option",
    12: "proceed",
    13: "advance_dialogue",
    14: "select_card",
    15: "confirm_selection",
    16: "cancel_selection",
    17: "combat_select_card",
    18: "combat_confirm_selection",
    19: "select_card_option",
    20: "use_potion",
    21: "drink_potion",
    22: "claim_treasure_relic",
    23: "select_relic",
    24: "skip_relic_selection",
    25: "skip",
    255: "other",
}
ACTION_CODES: dict[str, int] = {value: key for key, value in ACTION_TYPES.items()}

CANONICAL_POWER_IDS: dict[str, str] = {
    "strength": "STRENGTH_POWER",
    "dexterity": "DEXTERITY_POWER",
    "vulnerable": "VULNERABLE_POWER",
    "weak": "WEAK_POWER",
    "frail": "FRAIL_POWER",
    "metallicize": "METALLICIZE_POWER",
    "regen": "REGEN_POWER",
    "artifact": "ARTIFACT_POWER",
    "poison": "POISON_POWER",
}

SIM_LOG_ROOT = ARTIFACTS_ROOT / "sim_logs"

__all__ = [
    "ARTIFACTS_ROOT",
    "ENV_ROOT",
    "REPO_ROOT",
    "SIM_HOST_EXE",
    "SIM_LEGACY_DLL",
    "SIM_LOG_ROOT",
    "PROTO_PROTOCOL_VERSION",
    "PROTO_SCHEMA_ID",
    "OP_HANDSHAKE",
    "OP_RESET",
    "OP_STATE",
    "OP_STEP",
    "OP_BATCH_STEP",
    "OP_SAVE_STATE",
    "OP_LOAD_STATE",
    "OP_DELETE_STATE",
    "OP_PERF_STATS",
    "OP_RESET_PERF_STATS",
    "OP_STEP_LOCAL_POLICY",
    "OP_LOAD_ORT_MODEL",
    "OP_RUN_COMBAT_LOCAL",
    "OP_EXPORT_STATE",
    "OP_IMPORT_STATE",
    "OP_SKIP_COMBAT",
    "OP_SEARCH_COMBAT_MCTS",
    "OP_COMBAT_RESET",
    "OP_COMBAT_STEP",
    "OP_COMBAT_STATE",
    "STATUS_OK",
    "STATUS_REJECTED_ACTION",
    "STATUS_SIMULATOR_ERROR",
    "STATUS_PROTOCOL_ERROR",
    "COMBAT_STATE_TYPES",
    "STATE_BEARING_OPCODES",
    "ACTION_TYPES",
    "ACTION_CODES",
    "CANONICAL_POWER_IDS",
]
