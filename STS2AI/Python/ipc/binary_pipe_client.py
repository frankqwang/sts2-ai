from __future__ import annotations

import ctypes
import json
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any

from pipe_client import (
    FILE_FLAG_OVERLAPPED,
    GENERIC_READ,
    GENERIC_WRITE,
    INVALID_HANDLE_VALUE,
    OPEN_EXISTING,
    PipeClient,
    _kernel32,
)
from simulator_api_error import SimulatorApiError


STATUS_OK = 0
STATUS_REJECTED_ACTION = 1
STATUS_SIMULATOR_ERROR = 2
STATUS_PROTOCOL_ERROR = 3

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
OP_EXPORT_STATE = 0x0D
OP_IMPORT_STATE = 0x0E
PROTOCOL_VERSION = 9
BINARY_SCHEMA_HASH = "sts2-binary-schema-2026-04-09a"

STATE_TYPES = {
    0: "other",
    1: "menu",
    2: "map",
    3: "event",
    4: "rest_site",
    5: "shop",
    6: "treasure",
    7: "monster",
    8: "elite",
    9: "boss",
    10: "combat_pending",
    11: "combat_rewards",
    12: "card_reward",
    13: "card_select",
    14: "relic_select",
    15: "hand_select",
    16: "game_over",
    17: "rewards",
}

RUN_OUTCOMES = {0: None, 1: "victory", 2: "defeat"}

ACTION_TYPES = {
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
ACTION_CODES = {value: key for key, value in ACTION_TYPES.items()}

NODE_TYPES = {
    0: "unknown",
    1: "monster",
    2: "elite",
    3: "boss",
    4: "rest_site",
    5: "shop",
    6: "event",
    7: "treasure",
}

CARD_TYPES = {0: "unknown", 1: "attack", 2: "skill", 3: "power", 4: "status", 5: "curse", 6: "quest"}
RARITIES = {
    0: "unknown",
    1: "basic",
    2: "common",
    3: "uncommon",
    4: "rare",
    5: "ancient",
    6: "event",
    7: "token",
    8: "status",
    9: "curse",
    10: "quest",
}
INTENT_TYPES = {0: "unknown", 1: "attack", 2: "defend", 3: "buff", 4: "debuff", 5: "escape", 6: "sleep"}
TURN_SIDES = {0: "unknown", 1: "player", 2: "enemy"}
SHOP_CATEGORIES = {0: "unknown", 1: "card", 2: "relic", 3: "potion", 4: "remove_card"}
REWARD_TYPES = {0: "unknown", 1: "gold", 2: "potion", 3: "relic", 4: "card", 5: "remove_card", 6: "special_card"}
REST_OPTIONS = {0: "other", 1: "rest", 2: "smith", 3: "recall", 4: "dig", 5: "lift", 6: "toke"}
TARGET_TYPES = {
    0: None,
    1: "None",
    2: "Self",
    3: "AnyEnemy",
    4: "AnyPlayer",
    5: "AnyAlly",
    6: "TargetedNoCreature",
    7: "AllEnemies",
    8: "RandomEnemy",
    9: "AllAllies",
    10: "Osty",
}
CANONICAL_POWER_IDS = {
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


@dataclass(slots=True)
class _PlayerStatic:
    max_potions: int
    deck: list[dict[str, Any]]
    relics: list[dict[str, Any]]
    potions: list[dict[str, Any]]


class _Reader:
    def __init__(self, data: bytes):
        self._data = data
        self._offset = 0

    def read_u8(self) -> int:
        value = self._data[self._offset]
        self._offset += 1
        return value

    def read_i8(self) -> int:
        value = struct.unpack_from("<b", self._data, self._offset)[0]
        self._offset += 1
        return value

    def read_u16(self) -> int:
        value = struct.unpack_from("<H", self._data, self._offset)[0]
        self._offset += 2
        return value

    def read_i16(self) -> int:
        value = struct.unpack_from("<h", self._data, self._offset)[0]
        self._offset += 2
        return value

    def read_f32(self) -> float:
        value = struct.unpack_from("<f", self._data, self._offset)[0]
        self._offset += 4
        return value

    def read_i32(self) -> int:
        value = struct.unpack_from("<i", self._data, self._offset)[0]
        self._offset += 4
        return value

    def read_string(self) -> str:
        length = self.read_u16()
        value = self._data[self._offset:self._offset + length].decode("utf-8")
        self._offset += length
        return value

    def read_optional_string(self) -> str | None:
        return self.read_string() if self.read_u8() else None


class BinaryPipeClient(PipeClient):
    """Binary named-pipe client that returns training-oriented state dicts."""

    def __init__(
        self,
        port: int = 15527,
        pipe_name: str | None = None,
        default_timeout_s: float = 30.0,
    ):
        super().__init__(port=port, pipe_name=pipe_name or f"sts2_mcts_bin_{port}", default_timeout_s=default_timeout_s)
        self._symbol_table: dict[int, str] = {}
        self._player_static_cache: dict[int, _PlayerStatic] = {}
        self._protocol_version: int | None = None
        self._server_build_git_sha: str | None = None
        self._server_schema_hash: str | None = None

    def connect(self, timeout_s: float = 10.0) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Named pipes are only supported on Windows")

        pipe_path = f"\\\\.\\pipe\\{self.pipe_name}"
        deadline = time.monotonic() + timeout_s
        last_err = None

        while time.monotonic() < deadline:
            try:
                if not _kernel32.WaitNamedPipeW(pipe_path, 200):
                    last_err = f"Pipe {pipe_path} not ready"
                    time.sleep(0.1)
                    continue

                handle = _kernel32.CreateFileW(
                    pipe_path,
                    GENERIC_READ | GENERIC_WRITE,
                    0,
                    None,
                    OPEN_EXISTING,
                    FILE_FLAG_OVERLAPPED,
                    None,
                )
                if handle == INVALID_HANDLE_VALUE:
                    last_err = f"CreateFileW failed: winerror={ctypes.GetLastError()}"
                    time.sleep(0.1)
                    continue

                self._handle = handle
                self._event = _kernel32.CreateEventW(None, True, False, None)
                hello = self._read_response(timeout_s=timeout_s, expect_handshake=True)
                if hello.get("status") != STATUS_OK or hello.get("opcode") != OP_HANDSHAKE:
                    self.close()
                    raise ConnectionError(str(hello.get("error") or f"Unexpected binary handshake: {hello!r}"))
                self._protocol_version = int(hello.get("version") or 0)
                if self._protocol_version != PROTOCOL_VERSION:
                    self.close()
                    raise ConnectionError(
                        f"Binary protocol version mismatch: expected {PROTOCOL_VERSION}, got {self._protocol_version}"
                    )
                self._server_build_git_sha = str(hello.get("build_git_sha") or "").strip() or None
                self._server_schema_hash = str(hello.get("schema_hash") or "").strip() or None
                if self._server_schema_hash != BINARY_SCHEMA_HASH:
                    self.close()
                    raise ConnectionError(
                        "Binary schema mismatch: "
                        f"expected {BINARY_SCHEMA_HASH}, got {self._server_schema_hash or '<missing>'}"
                    )
                return
            except (SimulatorApiError, ConnectionError):
                raise
            except OSError as exc:
                last_err = f"Pipe open failed: {exc}"
                time.sleep(0.1)

        raise ConnectionError(f"Failed to connect after {timeout_s}s: {last_err}")

    def call(self, method: str, params: dict[str, Any] | None = None, timeout_s: float | None = None) -> dict[str, Any]:
        if self._handle is None:
            raise ConnectionError("Not connected. Call connect() first.")
        if timeout_s is None:
            timeout_s = self.default_timeout_s

        request = self._encode_request(method, params or {})
        self._write_bytes(struct.pack("<I", len(request)) + request)
        result = self._read_response(timeout_s=timeout_s)
        status = int(result.get("status", STATUS_SIMULATOR_ERROR))
        if status in {STATUS_PROTOCOL_ERROR, STATUS_SIMULATOR_ERROR}:
            raise SimulatorApiError(
                str(result.get("error") or "Binary pipe error"),
                error_code=result.get("error_code"),
            )
        return result["payload"]

    def _read_response(self, *, timeout_s: float, expect_handshake: bool = False) -> dict[str, Any]:
        timeout_ms = int(timeout_s * 1000)
        length = struct.unpack("<I", self._read_bytes(4, timeout_ms))[0]
        body = self._read_bytes(length, timeout_ms)
        reader = _Reader(body)
        status = reader.read_u8()
        opcode = reader.read_u8()

        if expect_handshake:
            if status == STATUS_OK:
                return {
                    "status": status,
                    "opcode": opcode,
                    "version": reader.read_u16(),
                    "build_git_sha": reader.read_string(),
                    "schema_hash": reader.read_string(),
                }
            return {
                "status": status,
                "opcode": opcode,
                "error_code": reader.read_string(),
                "error": reader.read_string(),
            }

        if status in {STATUS_PROTOCOL_ERROR, STATUS_SIMULATOR_ERROR}:
            return {
                "status": status,
                "opcode": opcode,
                "error_code": reader.read_string(),
                "error": reader.read_string(),
                "payload": {},
            }

        symbol_updates = 0
        if opcode in {OP_RESET, OP_STATE, OP_STEP, OP_BATCH_STEP, OP_LOAD_STATE, OP_IMPORT_STATE, OP_LOAD_ORT_MODEL, 0x0C}:
            symbol_updates = self._read_symbol_updates(reader)
        payload = self._decode_payload(opcode, reader)
        if isinstance(payload, dict):
            payload.setdefault("_binary_meta", {})["symbol_updates"] = symbol_updates
        return {"status": status, "opcode": opcode, "payload": payload}

    def _read_symbol_updates(self, reader: _Reader) -> int:
        count = reader.read_u16()
        for _ in range(count):
            reader.read_u8()
            symbol_id = reader.read_u16()
            self._symbol_table[symbol_id] = reader.read_string()
        return count

    def _decode_payload(self, opcode: int, reader: _Reader) -> dict[str, Any]:
        if opcode in {OP_RESET, OP_STATE, OP_LOAD_STATE, OP_IMPORT_STATE}:
            return self._decode_state(reader)
        if opcode == OP_STEP:
            accepted = bool(reader.read_u8())
            error = reader.read_optional_string()
            state = self._decode_state(reader)
            return {
                "accepted": accepted,
                "error": error,
                "state": state,
                "reward": self._terminal_reward(state),
                "done": bool(state.get("terminal")),
                "info": {
                    "state_type": state.get("state_type"),
                    "run_outcome": state.get("run_outcome"),
                },
            }
        if opcode == OP_BATCH_STEP:
            accepted = bool(reader.read_u8())
            steps_executed = reader.read_u16()
            error = reader.read_optional_string()
            state = self._decode_state(reader)
            return {
                "accepted": accepted,
                "steps_executed": steps_executed,
                "error": error,
                "state": state,
            }
        if opcode == OP_SAVE_STATE:
            return {"state_id": reader.read_string(), "cache_size": reader.read_i32()}
        if opcode == OP_EXPORT_STATE:
            return {"path": reader.read_string(), "cache_size": reader.read_i32()}
        if opcode == OP_DELETE_STATE:
            return {"deleted": bool(reader.read_u8()), "cache_size": reader.read_i32()}
        if opcode == OP_PERF_STATS:
            return json.loads(reader.read_string() or "{}")
        if opcode == OP_RESET_PERF_STATS:
            return {"reset": bool(reader.read_u8())}
        if opcode == OP_LOAD_ORT_MODEL:
            return {"loaded": bool(reader.read_u8())}
        if opcode == 0x0C:  # RunCombatLocal
            combat_steps = reader.read_u16()
            elapsed_ms = reader.read_f32()
            # Timing breakdown (6 floats, may not be present in older builds)
            timing = {}
            try:
                timing["get_snapshot_ms"] = reader.read_f32()
                timing["ort_ms"] = reader.read_f32()
                timing["step_async_ms"] = reader.read_f32()
                timing["wait_async_ms"] = reader.read_f32()
                timing["max_step_ms"] = reader.read_f32()
                timing["max_wait_ms"] = reader.read_f32()
            except Exception:
                pass
            state = self._decode_state(reader)
            return {
                "combat_steps": combat_steps,
                "elapsed_ms": elapsed_ms,
                "timing": timing,
                "state": state,
            }
        raise RuntimeError(f"Unsupported binary response opcode: {opcode}")

    def _encode_request(self, method: str, params: dict[str, Any]) -> bytes:
        method = str(method).strip().lower()
        body = bytearray()
        if method == "reset":
            body.append(OP_RESET)
            self._write_optional_string(body, params.get("character_id") or params.get("character"))
            self._write_optional_string(body, params.get("seed"))
            body.extend(struct.pack("<i", int(params.get("ascension_level", params.get("ascension", 0)) or 0)))
            return bytes(body)
        if method in {"state", "get_state", "legal_actions"}:
            return bytes([OP_STATE])
        if method == "step":
            body.append(OP_STEP)
            self._write_action(body, params)
            return bytes(body)
        if method == "batch_step":
            actions = list(params.get("actions") or [])
            body.append(OP_BATCH_STEP)
            body.extend(struct.pack("<H", len(actions)))
            for action in actions:
                self._write_action(body, action or {})
            return bytes(body)
        if method == "save_state":
            return bytes([OP_SAVE_STATE])
        if method == "export_state":
            body.append(OP_EXPORT_STATE)
            self._write_string(body, str(params["path"]))
            self._write_optional_string(body, params.get("state_id"))
            return bytes(body)
        if method == "import_state":
            body.append(OP_IMPORT_STATE)
            self._write_string(body, str(params["path"]))
            return bytes(body)
        if method == "load_state":
            body.append(OP_LOAD_STATE)
            self._write_string(body, str(params["state_id"]))
            return bytes(body)
        if method in {"delete_state", "clear_state_cache"}:
            clear_all = bool(params.get("clear_all")) or method == "clear_state_cache"
            body.append(OP_DELETE_STATE)
            body.append(1 if clear_all else 0)
            if not clear_all:
                self._write_string(body, str(params["state_id"]))
            return bytes(body)
        if method == "perf_stats":
            return bytes([OP_PERF_STATS])
        if method == "reset_perf_stats":
            return bytes([OP_RESET_PERF_STATS])
        if method == "step_local_policy":
            return bytes([OP_STEP_LOCAL_POLICY])
        if method == "run_combat_local":
            body.append(0x0C)  # OP_RunCombatLocal
            body.extend(struct.pack("<H", int(params.get("max_steps", 600))))
            return bytes(body)
        if method == "load_ort_model":
            body.append(OP_LOAD_ORT_MODEL)
            path_bytes = str(params.get("path", "")).encode("utf-8")
            body.extend(struct.pack("<H", len(path_bytes)))
            body.extend(path_bytes)
            return bytes(body)
        raise ValueError(f"Unsupported binary pipe method: {method}")

    def _write_action(self, body: bytearray, action: dict[str, Any]) -> None:
        action_name = str(action.get("action") or action.get("type") or "other").strip().lower()
        body.append(ACTION_CODES.get(action_name, 255))
        body.extend(struct.pack("<h", self._optional_short(action.get("index"))))
        body.extend(struct.pack("<h", self._optional_short(action.get("card_index"))))
        body.extend(struct.pack("<h", self._optional_short(action.get("target_id"))))
        body.extend(struct.pack("<b", self._optional_sbyte(action.get("col"))))
        body.extend(struct.pack("<b", self._optional_sbyte(action.get("row"))))
        body.extend(struct.pack("<b", self._optional_sbyte(action.get("slot"))))

    @staticmethod
    def _optional_short(value: Any) -> int:
        if value is None:
            return -1
        return max(min(int(value), 32767), -32768)

    @staticmethod
    def _optional_sbyte(value: Any) -> int:
        if value is None:
            return -1
        return max(min(int(value), 127), -128)

    @staticmethod
    def _write_string(body: bytearray, value: str) -> None:
        encoded = value.encode("utf-8")
        body.extend(struct.pack("<H", len(encoded)))
        body.extend(encoded)

    @staticmethod
    def _write_optional_string(body: bytearray, value: Any) -> None:
        if value is None or str(value) == "":
            body.append(0)
            return
        body.append(1)
        BinaryPipeClient._write_string(body, str(value))

    @classmethod
    def _write_optional_string(cls, body: bytearray, value: Any) -> None:
        if value is None or str(value).strip() == "":
            body.append(0)
            return
        body.append(1)
        cls._write_string(body, str(value))

    def _decode_state(self, reader: _Reader) -> dict[str, Any]:
        import os as _os
        _debug = _os.environ.get("STS2_BINARY_DECODE_DEBUG") == "1"
        if _debug:
            _decode_start_offset = reader._offset
            try:
                return self._decode_state_inner(reader)
            except Exception as _exc:
                import os
                buf = reader._data
                off = reader._offset
                # Only dump the first failure per process to avoid log spam.
                if not getattr(BinaryPipeClient, "_debug_dumped", False):
                    BinaryPipeClient._debug_dumped = True
                    state_type_byte = buf[_decode_start_offset] if _decode_start_offset < len(buf) else -1
                    state_type_name = STATE_TYPES.get(state_type_byte, f"<unknown:{state_type_byte}>")
                    print(f"\n[BINARY DEBUG] =============================================", file=sys.stderr)
                    print(f"[BINARY DEBUG] decode failed at offset {off} / {len(buf)}: {_exc}", file=sys.stderr)
                    print(f"[BINARY DEBUG] state_type byte at start_offset={_decode_start_offset}: {state_type_byte} ({state_type_name})", file=sys.stderr)
                    print(f"[BINARY DEBUG] decode region: [{_decode_start_offset}..{len(buf)})  size={len(buf) - _decode_start_offset}", file=sys.stderr)
                    print(f"[BINARY DEBUG] full decode buffer (hex, 16 bytes/row):", file=sys.stderr)
                    for i in range(_decode_start_offset, len(buf), 16):
                        row = buf[i : i + 16]
                        hex_bytes = " ".join(f"{b:02x}" for b in row)
                        marker = " <==" if i <= off < i + 16 else ""
                        print(f"[BINARY DEBUG]   {i:04d}: {hex_bytes}{marker}", file=sys.stderr)
                    print(f"[BINARY DEBUG] =============================================", file=sys.stderr)
                    # Save the raw buffer for offline analysis.
                    try:
                        dump_dir = os.environ.get("STS2_BINARY_DUMP_DIR", "")
                        if dump_dir:
                            dump_path = os.path.join(dump_dir, f"binary_dump_{os.getpid()}_{int(__import__('time').time())}.bin")
                            with open(dump_path, "wb") as f:
                                f.write(bytes(buf))
                            print(f"[BINARY DEBUG] saved buffer to {dump_path}", file=sys.stderr)
                    except Exception as _save_exc:
                        print(f"[BINARY DEBUG] failed to save dump: {_save_exc}", file=sys.stderr)
                raise
        return self._decode_state_inner(reader)

    def _decode_state_inner(self, reader: _Reader) -> dict[str, Any]:
        state_type = STATE_TYPES.get(reader.read_u8(), "other")
        terminal = bool(reader.read_u8())
        run_outcome = RUN_OUTCOMES.get(reader.read_u8())
        act = reader.read_u8()
        floor = reader.read_u8()

        static_version = reader.read_u16()
        if reader.read_u8():
            self._player_static_cache[static_version] = self._decode_player_static(reader)
        player_static = self._player_static_cache.get(static_version, _PlayerStatic(0, [], [], []))
        legal_actions = self._decode_legal_actions(reader)
        player_dynamic = self._decode_player_dynamic(reader)
        player = dict(player_dynamic)
        player["deck"] = player_static.deck
        player["relics"] = player_static.relics
        player["potions"] = player_static.potions
        player["max_potions"] = player_static.max_potions

        state: dict[str, Any] = {
            "state_type": state_type,
            "terminal": terminal,
            "run_outcome": run_outcome,
            "run": {"act": act, "floor": floor},
            "legal_actions": legal_actions,
            "player": player,
        }
        if state_type == "map":
            state["map"] = self._decode_map_state(reader, player)
        elif state_type == "event":
            state["event"] = self._decode_event_state(reader, player)
        elif state_type == "rest_site":
            state["rest_site"] = self._decode_rest_state(reader, player)
        elif state_type == "shop":
            state["shop"] = self._decode_shop_state(reader, player)
        elif state_type == "treasure":
            state["treasure"] = self._decode_treasure_state(reader, player)
        elif state_type == "combat_rewards":
            state["rewards"] = self._decode_rewards_state(reader, player)
        elif state_type == "card_reward":
            state["card_reward"] = self._decode_card_reward_state(reader, player)
        elif state_type == "card_select":
            state["card_select"] = self._decode_card_select_state(reader, player)
        elif state_type == "relic_select":
            state["relic_select"] = self._decode_relic_select_state(reader, player)
        elif state_type in {"monster", "elite", "boss", "hand_select"}:
            battle = self._decode_combat_state(reader, player)
            state["battle"] = battle
            state["enemies"] = battle.get("enemies", [])
            state["round_number_raw"] = battle.get("round_number_raw")
        self._decorate_action_labels(state)
        return state

    def _decode_player_static(self, reader: _Reader) -> _PlayerStatic:
        max_potions = reader.read_u8()
        deck: list[dict[str, Any]] = []
        for i in range(reader.read_u16()):
            symbol_id = reader.read_u16()
            cost = reader.read_i8()
            card_type = CARD_TYPES.get(reader.read_u8(), "unknown")
            rarity = RARITIES.get(reader.read_u8(), "unknown")
            upgraded = bool(reader.read_u8())
            card_id = self._symbol_table.get(symbol_id, "")
            deck.append({
                "index": i,
                "id": card_id,
                "name": card_id,
                "label": card_id,
                "cost": cost,
                "type": card_type,
                "rarity": rarity,
                "is_upgraded": upgraded,
                "upgrades": 1 if upgraded else 0,
            })
        relics = []
        for i in range(reader.read_u8()):
            relic_id = self._symbol_table.get(reader.read_u16(), "")
            relics.append({"index": i, "id": relic_id, "name": relic_id})
        potions = []
        for i in range(reader.read_u8()):
            potion_id = self._symbol_table.get(reader.read_u16(), "")
            potions.append({"index": i, "id": potion_id, "name": potion_id})
        return _PlayerStatic(max_potions=max_potions, deck=deck, relics=relics, potions=potions)

    def _decode_legal_actions(self, reader: _Reader) -> list[dict[str, Any]]:
        actions = []
        for _ in range(reader.read_u16()):
            action_name = ACTION_TYPES.get(reader.read_u8(), "other")
            index = reader.read_i16()
            card_index = reader.read_i16()
            target_id = reader.read_i16()
            col = reader.read_i8()
            row = reader.read_i8()
            slot = reader.read_i8()
            actions.append({
                "action": action_name,
                "type": action_name,
                "index": None if index < 0 else index,
                "card_index": None if card_index < 0 else card_index,
                "target_id": None if target_id < 0 else target_id,
                "col": None if col < 0 else col,
                "row": None if row < 0 else row,
                "slot": None if slot < 0 else slot,
                "is_enabled": True,
            })
        return actions

    def _decode_player_dynamic(self, reader: _Reader) -> dict[str, Any]:
        if not reader.read_u8():
            return {}
        hp = reader.read_i32()
        max_hp = reader.read_i32()
        return {
            "hp": hp,
            "current_hp": hp,
            "max_hp": max_hp,
            "block": reader.read_i32(),
            "gold": reader.read_i32(),
            "energy": reader.read_i32(),
            "max_energy": reader.read_i32(),
            "draw_pile_count": reader.read_i32(),
            "discard_pile_count": reader.read_i32(),
            "exhaust_pile_count": reader.read_i32(),
            "play_pile_count": reader.read_i32(),
            "open_potion_slots": reader.read_i32(),
        }

    def _decode_map_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        options = []
        for _ in range(reader.read_u8()):
            index = reader.read_u8()
            col = reader.read_i8()
            row = reader.read_i8()
            point_type = NODE_TYPES.get(reader.read_u8(), "unknown")
            options.append({
                "index": index,
                "col": col,
                "row": row,
                "point_type": point_type,
                "type": point_type,
                "label": point_type,
            })

        # Full map topology: all nodes with edges for route planning
        nodes = []
        node_count = reader.read_u16()
        for _ in range(node_count):
            n_col = reader.read_i8()
            n_row = reader.read_i8()
            n_type = NODE_TYPES.get(reader.read_u8(), "unknown")
            n_children_count = reader.read_u8()
            children = []
            for _ in range(n_children_count):
                c_col = reader.read_i8()
                c_row = reader.read_i8()
                children.append([c_col, c_row])
            nodes.append({
                "col": n_col,
                "row": n_row,
                "type": n_type,
                "children": children,
            })

        # Boss location
        boss_col = reader.read_i8()
        boss_row = reader.read_i8()

        return {
            "next_options": options,
            "nodes": nodes,
            "boss": {"col": boss_col, "row": boss_row},
            "player": player,
        }

    def _decode_event_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        in_dialogue = bool(reader.read_u8())
        event_id = self._symbol_table.get(reader.read_u16(), "")
        option_count = reader.read_u8()
        options = []
        for i in range(option_count):
            is_locked = bool(reader.read_u8())
            is_chosen = bool(reader.read_u8())
            is_proceed = bool(reader.read_u8())
            label = reader.read_optional_string()
            options.append(
                {
                    "index": i,
                    "text": label,
                    "label": label or f"option_{i}",
                    "is_locked": is_locked,
                    "is_chosen": is_chosen,
                    "is_proceed": is_proceed,
                }
            )
        return {
            "event_id": event_id,
            "in_dialogue": in_dialogue,
            "is_finished": False,
            "options": options,
            "player": player,
        }

    def _decode_rest_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        options = []
        for i in range(reader.read_u8()):
            option_id = REST_OPTIONS.get(reader.read_u8(), "other")
            enabled = bool(reader.read_u8())
            options.append({"index": i, "id": option_id, "name": option_id, "is_enabled": enabled})
        return {"options": options, "can_proceed": True, "player": player}

    def _decode_shop_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        items = []
        for i in range(reader.read_u8()):
            category = SHOP_CATEGORIES.get(reader.read_u8(), "unknown")
            symbol = self._symbol_table.get(reader.read_u16(), "")
            price = reader.read_i32()
            item = {
                "index": i,
                "category": category,
                "cost": price,
                "price": price,
                "can_afford": bool(reader.read_u8()),
                "is_stocked": bool(reader.read_u8()),
                "on_sale": bool(reader.read_u8()),
                "name": symbol or category,
                "id": symbol,
            }
            if category == "card":
                item["card_id"] = symbol
            elif category == "relic":
                item["relic_id"] = symbol
            elif category == "potion":
                item["potion_id"] = symbol
            items.append(item)
        return {"is_open": True, "can_proceed": True, "items": items, "player": player}

    def _decode_treasure_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        can_proceed = bool(reader.read_u8())
        relics = []
        for i in range(reader.read_u8()):
            relic_id = self._symbol_table.get(reader.read_u16(), "")
            relics.append({"index": i, "id": relic_id, "name": relic_id, "rarity": None})
        return {"can_proceed": can_proceed, "relics": relics, "player": player}

    def _decode_rewards_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        can_proceed = bool(reader.read_u8())
        items = []
        for i in range(reader.read_u8()):
            reward_type = REWARD_TYPES.get(reader.read_u8(), "unknown")
            symbol = self._symbol_table.get(reader.read_u16(), "")
            label = reader.read_optional_string()
            reward_key = reader.read_optional_string()
            reward_source = reader.read_optional_string()
            claimable = bool(reader.read_u8())
            claim_block_reason = reader.read_optional_string()
            items.append(
                {
                    "index": i,
                    "type": reward_type,
                    "label": (label or symbol or reward_type),
                    "id": symbol,
                    "reward_key": reward_key,
                    "reward_source": reward_source,
                    "claimable": claimable,
                    "claim_block_reason": claim_block_reason,
                }
            )
        return {"can_proceed": can_proceed, "items": items, "player": player}

    def _decode_card_reward_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        can_skip = bool(reader.read_u8())
        cards = [self._decode_card_like(reader, i) for i in range(reader.read_u8())]
        return {"can_skip": can_skip, "cards": cards, "player": player}

    def _decode_card_select_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        screen_type = reader.read_optional_string() or "card_select"
        selected_count = reader.read_u8()
        can_confirm = bool(reader.read_u8())
        can_cancel = bool(reader.read_u8())
        cards = [self._decode_selectable_card_like(reader) for _ in range(reader.read_u8())]
        selected_cards = [self._decode_selectable_card_like(reader) for _ in range(reader.read_u8())]
        return {
            "screen_type": screen_type,
            "selected_count": selected_count,
            "can_confirm": can_confirm,
            "can_cancel": can_cancel,
            "cards": cards,
            "selected_cards": selected_cards,
            "player": player,
        }

    def _decode_relic_select_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        can_skip = bool(reader.read_u8())
        relics = []
        for i in range(reader.read_u8()):
            relic_id = self._symbol_table.get(reader.read_u16(), "")
            relics.append({"index": i, "id": relic_id, "name": relic_id, "rarity": None})
        return {"can_skip": can_skip, "relics": relics, "player": player}

    def _decode_combat_state(self, reader: _Reader, player: dict[str, Any]) -> dict[str, Any]:
        round_number = reader.read_i16()
        turn_side = TURN_SIDES.get(reader.read_u8(), "unknown")
        is_play_phase = bool(reader.read_u8())
        can_end_turn = bool(reader.read_u8())

        battle_player = dict(player)
        if reader.read_u8():
            hp = reader.read_i32()
            battle_player.update({
                "hp": hp,
                "current_hp": hp,
                "max_hp": reader.read_i32(),
                "block": reader.read_i32(),
                "energy": reader.read_i32(),
                "max_energy": reader.read_i32(),
                "stars": reader.read_i32(),
            })
        battle_player["powers"] = self._decode_power_list(reader, ("strength", "dexterity", "vulnerable", "weak", "frail", "metallicize", "regen", "artifact"))

        hand = []
        for i in range(reader.read_u8()):
            card = self._decode_card_like(reader, i)
            card["can_play"] = bool(reader.read_u8())
            card["requires_target"] = bool(reader.read_u8())
            card["valid_target_ids"] = [reader.read_i16() for _ in range(reader.read_u8())]
            hand.append(card)
        battle_player["hand"] = hand

        enemies = []
        for _ in range(reader.read_u8()):
            enemy_id = self._symbol_table.get(reader.read_u16(), "")
            combat_id = reader.read_i16()
            hp = reader.read_i32()
            max_hp = reader.read_i32()
            block = reader.read_i32()
            is_alive = bool(reader.read_u8())
            intents = []
            for _ in range(reader.read_u8()):
                intent_type = self._symbol_table.get(reader.read_u16(), "") or "unknown"
                intent_damage = reader.read_i32()
                intent_total_damage = reader.read_i32()
                intent_hits = max(1, reader.read_i16())
                intents.append(
                    {
                        "type": intent_type,
                        "label": intent_type,
                        "damage": intent_damage,
                        "total_damage": intent_total_damage,
                        "hits": intent_hits,
                    }
                )
            powers = []
            for _ in range(reader.read_u8()):
                power_id = self._symbol_table.get(reader.read_u16(), "")
                amount = reader.read_i32()
                if amount:
                    powers.append({"id": power_id, "amount": amount})
            # Phase 2.5 boss-state expansion (2026-04-08): is_hittable +
            # intends_to_attack + next_move_id. Wire format:
            #   byte  is_hittable  (1=true 0=false)
            #   byte  intends_to_attack
            #   u16   next_move_id symbol
            is_hittable = bool(reader.read_u8())
            intends_to_attack = bool(reader.read_u8())
            next_move_id = self._symbol_table.get(reader.read_u16(), "") or None
            enemies.append({
                "id": enemy_id,
                "entity_id": enemy_id,
                "monster_id": enemy_id,
                "name": enemy_id,
                "combat_id": combat_id,
                "target_id": combat_id,
                "hp": hp,
                "current_hp": hp,
                "max_hp": max_hp,
                "block": block,
                "is_alive": is_alive,
                "is_hittable": is_hittable,
                "intends_to_attack": intends_to_attack,
                "next_move_id": next_move_id,
                "status": powers,
                "powers": powers,
                "buffs": powers,
                "intents": intents,
                "intent_type": intents[0]["type"] if intents else "unknown",
                "intent_damage": intents[0]["damage"] if intents else 0,
                "intent_hits": intents[0]["hits"] if intents else 1,
            })

        # Pile card ID lists (draw/discard/exhaust)
        draw_pile_cards = [self._symbol_table.get(reader.read_u16(), "") for _ in range(reader.read_u8())]
        discard_pile_cards = [self._symbol_table.get(reader.read_u16(), "") for _ in range(reader.read_u8())]
        exhaust_pile_cards = [self._symbol_table.get(reader.read_u16(), "") for _ in range(reader.read_u8())]

        return {
            "round_number_raw": round_number,
            "turn": turn_side,
            "turn_side": turn_side,
            "is_play_phase": is_play_phase,
            "can_end_turn": can_end_turn,
            "player": battle_player,
            "hand": hand,
            "enemies": enemies,
            "energy": battle_player.get("energy"),
            "max_energy": battle_player.get("max_energy"),
            "draw_pile_cards": draw_pile_cards,
            "discard_pile_cards": discard_pile_cards,
            "exhaust_pile_cards": exhaust_pile_cards,
        }

    def _decode_power_list(self, reader: _Reader, names: tuple[str, ...]) -> list[dict[str, Any]]:
        powers = []
        for name in names:
            amount = reader.read_i32()
            if amount:
                powers.append({"id": CANONICAL_POWER_IDS.get(name, name.upper()), "amount": amount})
        return powers

    def _decode_card_like(self, reader: _Reader, index: int) -> dict[str, Any]:
        card_id = self._symbol_table.get(reader.read_u16(), "")
        cost = reader.read_i8()
        upgraded = bool(reader.read_u8())
        card_type = CARD_TYPES.get(reader.read_u8(), "unknown")
        rarity = RARITIES.get(reader.read_u8(), "unknown")
        target_type = TARGET_TYPES.get(reader.read_u8(), None)
        if card_type == "unknown":
            card_type = ""
        if rarity == "unknown":
            # Let encoder backfill hand/selectable-card rarity from vocab/card id.
            rarity = ""
        return {
            "index": index,
            "id": card_id,
            "name": card_id,
            "label": card_id,
            "cost": cost,
            "type": card_type,
            "rarity": rarity,
            "target_type": target_type,
            "is_upgraded": upgraded,
            "upgrades": 1 if upgraded else 0,
        }

    def _decode_selectable_card_like(self, reader: _Reader) -> dict[str, Any]:
        choice_index = reader.read_i16()
        return self._decode_card_like(reader, choice_index)

    def _decorate_action_labels(self, state: dict[str, Any]) -> None:
        map_options = ((state.get("map") or {}).get("next_options") or [])
        reward_items = ((state.get("rewards") or {}).get("items") or [])
        reward_cards = ((state.get("card_reward") or {}).get("cards") or [])
        card_select_cards = ((state.get("card_select") or {}).get("cards") or [])
        relic_select_items = ((state.get("relic_select") or {}).get("relics") or [])
        treasure_relics = ((state.get("treasure") or {}).get("relics") or [])
        shop_items = ((state.get("shop") or {}).get("items") or [])
        rest_items = ((state.get("rest_site") or {}).get("options") or [])
        event_items = ((state.get("event") or {}).get("options") or [])
        hand_cards = ((state.get("battle") or {}).get("hand") or [])

        for action in state.get("legal_actions") or []:
            action_name = str(action.get("action") or "")
            index = action.get("index")
            card_index = action.get("card_index")
            label = action_name
            if action_name == "choose_map_node":
                option = self._get_indexed_item(map_options, index)
                label = str((option or {}).get("label") or (option or {}).get("point_type") or action_name)
            elif action_name == "claim_reward":
                reward = self._get_indexed_item(reward_items, index)
                label = str((reward or {}).get("label") or (reward or {}).get("type") or action_name)
            elif action_name == "select_card_reward":
                reward_card = self._get_indexed_item(reward_cards, index)
                label = str((reward_card or {}).get("id") or action_name)
            elif action_name in {"select_card", "combat_select_card", "select_card_option"}:
                label = str(
                    (self._get_indexed_item(card_select_cards, index)
                     or self._get_indexed_item(hand_cards, card_index)
                     or self._get_indexed_item(hand_cards, index)
                     or {}).get("id")
                    or action_name
                )
            elif action_name == "play_card":
                label = str((self._get_indexed_item(hand_cards, card_index) or self._get_indexed_item(hand_cards, index) or {}).get("id") or action_name)
            elif action_name == "select_relic":
                label = str((self._get_indexed_item(relic_select_items, index) or self._get_indexed_item(treasure_relics, index) or {}).get("id") or action_name)
            elif action_name == "shop_purchase":
                item = self._get_indexed_item(shop_items, index)
                label = str((item or {}).get("name") or (item or {}).get("id") or (item or {}).get("category") or action_name)
            elif action_name == "choose_rest_option":
                option = self._get_indexed_item(rest_items, index)
                label = str((option or {}).get("id") or (option or {}).get("name") or action_name)
            elif action_name == "choose_event_option":
                label = str((self._get_indexed_item(event_items, index) or {}).get("label") or f"option_{index}")
            elif action_name == "claim_treasure_relic":
                label = str((self._get_indexed_item(treasure_relics, index) or {}).get("id") or action_name)
            action["label"] = label

    @staticmethod
    def _get_indexed_item(items: list[dict[str, Any]], index: Any) -> dict[str, Any] | None:
        if not isinstance(index, int) or index < 0:
            return None
        if index < len(items):
            item = items[index]
            if isinstance(item, dict) and item.get("index", index) == index:
                return item
        for item in items:
            if isinstance(item, dict) and item.get("index") == index:
                return item
        return None

    @staticmethod
    def _terminal_reward(state: dict[str, Any]) -> float:
        if not bool(state.get("terminal")):
            return 0.0
        outcome = str(state.get("run_outcome") or "").strip().lower()
        if outcome in {"victory", "win"}:
            return 1.0
        if outcome in {"defeat", "loss", "death"}:
            return -1.0
        return 0.0
