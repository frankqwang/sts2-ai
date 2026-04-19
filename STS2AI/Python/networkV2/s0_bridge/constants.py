"""协议常量 — 操作码、类型映射、枚举值。

proto / transport 层共用。2026-04-18:手写 binary wire 已废弃,
这些 opcode 值只对应 sim 侧的 BinaryOpcode enum(proto 响应共用同一套编号)。
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 协议版本 & Schema
# ---------------------------------------------------------------------------
PROTO_PROTOCOL_VERSION = 1
PROTO_SCHEMA_ID = "sts2-proto-v1"

# ---------------------------------------------------------------------------
# 操作码(和 sim C# BinaryOpcode enum 一致,proto 侧同一套编码)
# ---------------------------------------------------------------------------
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
OP_COMBAT_RESET = 0x11             # 精确构造战斗 (deck/relics/hp 用 BuildSpec)
OP_COMBAT_STEP = 0x12              # 战斗内 step;response 含权威 legal_actions
OP_COMBAT_STATE = 0x13             # 当前战斗 state(含 legal_actions)

# ---------------------------------------------------------------------------
# 响应状态码
# ---------------------------------------------------------------------------
STATUS_OK = 0
STATUS_REJECTED_ACTION = 1
STATUS_SIMULATOR_ERROR = 2
STATUS_PROTOCOL_ERROR = 3

# ---------------------------------------------------------------------------
# state_type 字符串 → 分类
# ---------------------------------------------------------------------------
COMBAT_STATE_TYPES = frozenset({"monster", "elite", "boss", "hand_select"})

# ---------------------------------------------------------------------------
# 需要读取符号更新的 opcode 集合（proto 版不再需要，仅用于参考）
# ---------------------------------------------------------------------------
STATE_BEARING_OPCODES = frozenset({
    OP_RESET, OP_STATE, OP_STEP, OP_BATCH_STEP,
    OP_LOAD_STATE, OP_IMPORT_STATE, OP_LOAD_ORT_MODEL,
    OP_RUN_COMBAT_LOCAL, OP_SKIP_COMBAT,
    OP_COMBAT_RESET, OP_COMBAT_STEP, OP_COMBAT_STATE,
})

# ---------------------------------------------------------------------------
# Action 类型映射（字符串 ↔ 编码，用于请求编码）
# ---------------------------------------------------------------------------
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
ACTION_CODES: dict[str, int] = {v: k for k, v in ACTION_TYPES.items()}

# ---------------------------------------------------------------------------
# 标准 power ID 别名（二进制协议的 canonical 名到 proto string 映射）
# ---------------------------------------------------------------------------
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
