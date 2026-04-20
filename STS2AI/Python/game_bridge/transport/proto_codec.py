"""ProtoCodec — protobuf pipe protocol codec for PipeConnection.

把 `proto_pipe_client.py` 的 encode_request / decode_payload / handshake
逻辑迁到独立 codec,让 PipeConnection + ProtoCodec 替代 ProtoPipeClient。

Wire 格式 (和 `proto_pipe_client` 一致):
  request:   [u8 opcode][方法特定参数 bytes]
  response:  [u8 status][u8 opcode][payload bytes]
  handshake: [u8 0][u8 OP_HANDSHAKE][u16 version][string sha][string schema_id]

状态 payload (opcode in OP_RESET/STATE/STEP 等) 用 protobuf `GameState` 编码。
新增 combat opcode (OP_COMBAT_RESET/STEP/STATE) 也走 GameState protobuf,
调用方的 combat_reset / combat_step 参数分别为 CombatResetRequest /
CombatStepRequest protobuf 字节。
"""
from __future__ import annotations

import json
import struct
from typing import Any

from game_bridge.transport.codec import ProtocolCodec

from game_bridge.generated import game_state_pb2 as pb
from game_bridge.transport.proto_state_converter import game_state_to_dict
from game_bridge.sim.constants import (
    ACTION_CODES,
    OP_BATCH_STEP,
    OP_COMBAT_RESET,
    OP_COMBAT_STATE,
    OP_COMBAT_STEP,
    OP_DELETE_STATE,
    OP_EXPORT_STATE,
    OP_HANDSHAKE,
    OP_IMPORT_STATE,
    OP_LOAD_ORT_MODEL,
    OP_LOAD_STATE,
    OP_PERF_STATS,
    OP_RESET,
    OP_RESET_PERF_STATS,
    OP_RUN_COMBAT_LOCAL,
    OP_SAVE_STATE,
    OP_SEARCH_COMBAT_MCTS,
    OP_SKIP_COMBAT,
    OP_STATE,
    OP_STEP,
    OP_STEP_LOCAL_POLICY,
    PROTO_PROTOCOL_VERSION,
    PROTO_SCHEMA_ID,
    STATUS_OK,
    STATUS_PROTOCOL_ERROR,
    STATUS_SIMULATOR_ERROR,
)


# ---------------------------------------------------------------------------
# 二进制读/写 (和 proto_pipe_client 保持一致)
# ---------------------------------------------------------------------------

def _read_string(data: bytes, off: int) -> tuple[str, int]:
    length = struct.unpack_from("<H", data, off)[0]
    off += 2
    value = data[off:off + length].decode("utf-8")
    return value, off + length


def _read_optional_string(data: bytes, off: int) -> tuple[str | None, int]:
    if data[off]:
        off += 1
        return _read_string(data, off)
    return None, off + 1


def _write_string(body: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    body.extend(struct.pack("<H", len(encoded)))
    body.extend(encoded)


def _write_optional_string(body: bytearray, value: Any) -> None:
    if value is None or str(value).strip() == "":
        body.append(0)
        return
    body.append(1)
    _write_string(body, str(value))


def _optional_short(value: Any) -> int:
    if value is None:
        return -1
    return max(min(int(value), 32767), -32768)


def _optional_sbyte(value: Any) -> int:
    if value is None:
        return -1
    return max(min(int(value), 127), -128)


def _write_action(body: bytearray, action: dict[str, Any]) -> None:
    action_name = str(action.get("action") or action.get("type") or "other").strip().lower()
    body.append(ACTION_CODES.get(action_name, 255))
    body.extend(struct.pack("<h", _optional_short(action.get("index"))))
    body.extend(struct.pack("<h", _optional_short(action.get("card_index"))))
    body.extend(struct.pack("<h", _optional_short(action.get("target_id"))))
    body.extend(struct.pack("<b", _optional_sbyte(action.get("col"))))
    body.extend(struct.pack("<b", _optional_sbyte(action.get("row"))))
    body.extend(struct.pack("<b", _optional_sbyte(action.get("slot"))))


def _parse_proto_state(data: bytes) -> dict[str, Any]:
    gs = pb.GameState()
    gs.ParseFromString(data)
    result = game_state_to_dict(gs)
    gs.Clear()
    return result


def _terminal_reward(state: dict[str, Any]) -> float:
    if not bool(state.get("terminal")):
        return 0.0
    outcome = str(state.get("run_outcome") or "").strip().lower()
    if outcome in {"victory", "win"}:
        return 1.0
    if outcome in {"defeat", "loss", "death"}:
        return -1.0
    return 0.0


def _decode_mcts_payload(data: bytes) -> dict[str, Any]:
    off = 0
    action_index = struct.unpack_from("<h", data, off)[0]; off += 2
    count = struct.unpack_from("<H", data, off)[0]; off += 2
    visit_counts = []
    for _ in range(count):
        visit_counts.append(struct.unpack_from("<i", data, off)[0]); off += 4
    visit_probs = []
    for _ in range(count):
        visit_probs.append(struct.unpack_from("<f", data, off)[0]); off += 4
    q_values = []
    for _ in range(count):
        q_values.append(struct.unpack_from("<f", data, off)[0]); off += 4
    priors = []
    for _ in range(count):
        priors.append(struct.unpack_from("<f", data, off)[0]); off += 4
    payload: dict[str, Any] = {
        "action_index": action_index,
        "visit_counts": visit_counts,
        "visit_probs": visit_probs,
        "q_values": q_values,
        "priors": priors,
        "root_value": struct.unpack_from("<f", data, off)[0],
        "search_ms": struct.unpack_from("<f", data, off + 4)[0],
        "restored_ok": bool(data[off + 8]),
        "snapshot_count": struct.unpack_from("<i", data, off + 9)[0],
    }
    off += 13
    if off + (11 * 4) + (8 * 4) <= len(data):
        payload["breakdown"] = {
            "simulation_count": struct.unpack_from("<i", data, off)[0],
            "save_state_count": struct.unpack_from("<i", data, off + 4)[0],
            "load_state_count": struct.unpack_from("<i", data, off + 8)[0],
            "delete_state_count": struct.unpack_from("<i", data, off + 12)[0],
            "step_count": struct.unpack_from("<i", data, off + 16)[0],
            "advance_state_count": struct.unpack_from("<i", data, off + 20)[0],
            "eval_call_count": struct.unpack_from("<i", data, off + 24)[0],
            "eval_batch_count": struct.unpack_from("<i", data, off + 28)[0],
            "eval_state_count": struct.unpack_from("<i", data, off + 32)[0],
            "select_child_count": struct.unpack_from("<i", data, off + 36)[0],
            "backprop_count": struct.unpack_from("<i", data, off + 40)[0],
            "save_state_ms": struct.unpack_from("<f", data, off + 44)[0],
            "load_state_ms": struct.unpack_from("<f", data, off + 48)[0],
            "delete_state_ms": struct.unpack_from("<f", data, off + 52)[0],
            "step_ms": struct.unpack_from("<f", data, off + 56)[0],
            "advance_state_ms": struct.unpack_from("<f", data, off + 60)[0],
            "eval_ms": struct.unpack_from("<f", data, off + 64)[0],
            "selection_ms": struct.unpack_from("<f", data, off + 68)[0],
            "backprop_ms": struct.unpack_from("<f", data, off + 72)[0],
        }
        off += 76
    if off < len(data):
        has_trace = data[off]; off += 1
        if has_trace:
            trace_len = struct.unpack_from("<H", data, off)[0]; off += 2
            trace_json = data[off:off + trace_len].decode("utf-8")
            try:
                payload["debug_trace"] = json.loads(trace_json)
            except Exception:
                payload["debug_trace_json"] = trace_json
    return payload


# ---------------------------------------------------------------------------
# ProtoCodec
# ---------------------------------------------------------------------------

class ProtoCodec(ProtocolCodec):
    """Protobuf pipe codec — PipeConnection 用它说 proto wire。

    和 ProtoPipeClient 字节行为一致;PipeConnection(codec=ProtoCodec()) 是
    新代码的推荐入口,ProtoPipeClient 已降级为此 codec 的薄包装。
    """

    name = "proto"

    def encode_request(self, method: str, params: dict[str, Any] | None) -> bytes:
        method = str(method).strip().lower()
        params = params or {}
        body = bytearray()

        if method == "reset":
            body.append(OP_RESET)
            _write_optional_string(body, params.get("character_id") or params.get("character"))
            _write_optional_string(body, params.get("seed"))
            body.extend(struct.pack(
                "<i",
                int(params.get("ascension_level", params.get("ascension", 0)) or 0),
            ))
            build = params.get("build")
            build_json = None if build is None else json.dumps(
                build, ensure_ascii=False, separators=(",", ":"),
            )
            _write_optional_string(body, build_json)
            return bytes(body)

        if method in {"state", "get_state", "legal_actions"}:
            return bytes([OP_STATE])

        if method == "step":
            body.append(OP_STEP)
            _write_action(body, params)
            return bytes(body)

        if method == "batch_step":
            actions = list(params.get("actions") or [])
            body.append(OP_BATCH_STEP)
            body.extend(struct.pack("<H", len(actions)))
            for action in actions:
                _write_action(body, action or {})
            return bytes(body)

        if method == "save_state":
            return bytes([OP_SAVE_STATE])

        if method == "export_state":
            body.append(OP_EXPORT_STATE)
            _write_string(body, str(params["path"]))
            _write_optional_string(body, params.get("state_id"))
            return bytes(body)

        if method == "import_state":
            body.append(OP_IMPORT_STATE)
            _write_string(body, str(params["path"]))
            return bytes(body)

        if method == "load_state":
            body.append(OP_LOAD_STATE)
            _write_string(body, str(params["state_id"]))
            return bytes(body)

        if method in {"delete_state", "clear_state_cache"}:
            clear_all = bool(params.get("clear_all")) or method == "clear_state_cache"
            body.append(OP_DELETE_STATE)
            body.append(1 if clear_all else 0)
            if not clear_all:
                _write_string(body, str(params["state_id"]))
            return bytes(body)

        if method == "perf_stats":
            return bytes([OP_PERF_STATS])

        if method == "reset_perf_stats":
            return bytes([OP_RESET_PERF_STATS])

        if method == "step_local_policy":
            return bytes([OP_STEP_LOCAL_POLICY])

        if method == "skip_combat":
            return bytes([OP_SKIP_COMBAT])

        if method == "run_combat_local":
            body.append(OP_RUN_COMBAT_LOCAL)
            body.extend(struct.pack("<H", int(params.get("max_steps", 600))))
            return bytes(body)

        if method == "load_ort_model":
            body.append(OP_LOAD_ORT_MODEL)
            path_bytes = str(params.get("path", "")).encode("utf-8")
            body.extend(struct.pack("<H", len(path_bytes)))
            body.extend(path_bytes)
            return bytes(body)

        if method == "search_combat_mcts":
            body.append(OP_SEARCH_COMBAT_MCTS)
            body.extend(struct.pack("<H", int(params.get("num_simulations", 0))))
            body.extend(struct.pack("<f", float(params.get("c_puct", 1.5))))
            body.extend(struct.pack("<f", float(params.get("dirichlet_alpha", 0.0))))
            body.extend(struct.pack("<f", float(params.get("dirichlet_fraction", 0.0))))
            body.extend(struct.pack("<H", int(params.get("max_step_budget", 200))))
            mode = str(params.get("final_action_mode", "visit")).strip().lower()
            body.append(1 if mode == "visit_q_blend" else 0)
            body.extend(struct.pack("<H", int(params.get("final_action_top_k", 3))))
            body.extend(struct.pack("<f", float(params.get("final_action_q_weight", 0.35))))
            body.append(1 if bool(params.get("use_continuation_value", False)) else 0)
            body.append(1 if bool(params.get("debug_trace", False)) else 0)
            return bytes(body)

        # --- combat training opcodes (proto schema) ---
        if method == "combat_reset":
            body.append(OP_COMBAT_RESET)
            req = pb.CombatResetRequest(
                character_id=str(params.get("character_id") or params.get("character") or ""),
                encounter_id=str(params.get("encounter_id") or ""),
                ascension_level=int(params.get("ascension_level") or params.get("ascension") or 0),
                seed=str(params.get("seed") or ""),
            )
            build = params.get("build")
            if isinstance(build, dict):
                req.build.current_hp = int(build.get("current_hp") or 0)
                req.build.max_hp = int(build.get("max_hp") or 0)
                req.build.max_energy = int(build.get("max_energy") or 3)
                req.build.gold = int(build.get("gold") or 0)
                for card in build.get("deck") or []:
                    if isinstance(card, dict):
                        req.build.deck.add(
                            id=str(card.get("id") or "").upper(),
                            upgrade_level=int(card.get("upgrade_level") or 0),
                        )
                    elif isinstance(card, str):
                        req.build.deck.add(id=card.upper(), upgrade_level=0)
                for relic in build.get("relics") or []:
                    if isinstance(relic, dict):
                        req.build.relics.add(id=str(relic.get("id") or "").upper())
                    elif isinstance(relic, str):
                        req.build.relics.add(id=relic.upper())
            elif build is not None:
                raise TypeError("combat_reset build must be a dict or None")
            body.extend(req.SerializeToString())
            return bytes(body)

        if method == "combat_step":
            body.append(OP_COMBAT_STEP)
            req = pb.CombatStepRequest()
            _apply_legal_action_to_proto(req.action, params)
            body.extend(req.SerializeToString())
            return bytes(body)

        if method == "combat_state":
            return bytes([OP_COMBAT_STATE])

        raise ValueError(f"Unsupported proto pipe method: {method}")

    def decode_response(self, payload: bytes) -> dict[str, Any]:
        if len(payload) < 2:
            return {"error": f"proto response too short: {len(payload)} bytes"}
        status = payload[0]
        opcode = payload[1]
        data = bytes(payload[2:])
        if status in {STATUS_PROTOCOL_ERROR, STATUS_SIMULATOR_ERROR}:
            off = 0
            error_code, off = _read_string(data, off)
            error_msg, off = _read_string(data, off)
            return {
                "status": status,
                "opcode": opcode,
                "error_code": error_code,
                "error": error_msg,
            }
        inner = _decode_payload_by_opcode(opcode, data)
        if isinstance(inner, dict):
            return inner
        return {"status": status, "opcode": opcode, "payload": inner}

    def read_handshake(self, payload: bytes) -> dict[str, Any]:
        if len(payload) < 2:
            return {"error": f"proto handshake too short: {len(payload)} bytes"}
        status = payload[0]
        opcode = payload[1]
        data = bytes(payload[2:])
        if status != STATUS_OK or opcode != OP_HANDSHAKE:
            off = 0
            try:
                error_code, off = _read_string(data, off)
                error_msg, _ = _read_string(data, off)
            except Exception:
                error_code, error_msg = "", "malformed handshake"
            return {"error": error_msg or "handshake failed", "error_code": error_code}
        # [u16 version][string sha][string schema_id]
        version = struct.unpack_from("<H", data, 0)[0]
        off = 2
        sha, off = _read_string(data, off)
        schema_id, _ = _read_string(data, off)
        if version != PROTO_PROTOCOL_VERSION:
            return {"error": f"proto protocol version mismatch: expected "
                             f"{PROTO_PROTOCOL_VERSION}, got {version}"}
        if schema_id != PROTO_SCHEMA_ID:
            return {"error": f"proto schema mismatch: expected {PROTO_SCHEMA_ID}, "
                             f"got {schema_id or '<missing>'}"}
        return {
            "version": version,
            "build_git_sha": sha,
            "schema_id": schema_id,
        }


# ---------------------------------------------------------------------------
# opcode payload 分派(和 proto_pipe_client._decode_payload 逻辑一致)
# ---------------------------------------------------------------------------

def _decode_payload_by_opcode(opcode: int, data: bytes) -> dict[str, Any]:
    if opcode in {OP_RESET, OP_STATE, OP_LOAD_STATE, OP_IMPORT_STATE,
                  OP_COMBAT_RESET, OP_COMBAT_STATE}:
        return _parse_proto_state(data)

    if opcode in {OP_STEP, OP_COMBAT_STEP}:
        accepted = bool(data[0])
        off = 1
        error, off = _read_optional_string(data, off)
        state = _parse_proto_state(data[off:])
        return {
            "accepted": accepted,
            "error": error,
            "state": state,
            "reward": _terminal_reward(state),
            "done": bool(state.get("terminal")),
            "info": {
                "state_type": state.get("state_type"),
                "run_outcome": state.get("run_outcome"),
            },
        }

    if opcode == OP_BATCH_STEP:
        accepted = bool(data[0])
        steps_executed = struct.unpack_from("<H", data, 1)[0]
        off = 3
        error, off = _read_optional_string(data, off)
        state = _parse_proto_state(data[off:])
        return {
            "accepted": accepted,
            "steps_executed": steps_executed,
            "error": error,
            "state": state,
        }

    if opcode == OP_SAVE_STATE:
        off = 0
        state_id, off = _read_string(data, off)
        cache_size = struct.unpack_from("<i", data, off)[0]
        return {"state_id": state_id, "cache_size": cache_size}

    if opcode == OP_EXPORT_STATE:
        off = 0
        path, off = _read_string(data, off)
        cache_size = struct.unpack_from("<i", data, off)[0]
        return {"path": path, "cache_size": cache_size}

    if opcode == OP_DELETE_STATE:
        return {
            "deleted": bool(data[0]),
            "cache_size": struct.unpack_from("<i", data, 1)[0],
        }

    if opcode == OP_PERF_STATS:
        off = 0
        json_str, _ = _read_string(data, off)
        return json.loads(json_str or "{}")

    if opcode == OP_RESET_PERF_STATS:
        return {"reset": bool(data[0])}

    if opcode == OP_LOAD_ORT_MODEL:
        payload: dict[str, Any] = {"loaded": bool(data[0])}
        off = 1
        if off + 4 <= len(data):
            payload["has_value"] = bool(data[off]); off += 1
            payload["has_deck_inputs"] = bool(data[off]); off += 1
            payload["has_continuation_output"] = bool(data[off]); off += 1
            payload["has_extra_scalars_input"] = bool(data[off]); off += 1
        if off < len(data):
            payload["execution_provider"], off = _read_string(data, off)
        if off < len(data):
            payload["requested_device"], off = _read_string(data, off)
        if off < len(data):
            payload["fell_back_to_cpu"] = bool(data[off])
        return payload

    if opcode == OP_SKIP_COMBAT:
        accepted = bool(data[0])
        off = 1
        error, off = _read_optional_string(data, off)
        state = _parse_proto_state(data[off:])
        return {"accepted": accepted, "error": error, "state": state, "skipped": True}

    if opcode == OP_RUN_COMBAT_LOCAL:
        combat_steps = struct.unpack_from("<H", data, 0)[0]
        elapsed_ms = struct.unpack_from("<f", data, 2)[0]
        off = 6
        timing: dict[str, float] = {}
        try:
            timing["get_snapshot_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
            timing["ort_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
            timing["step_async_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
            timing["wait_async_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
            timing["max_step_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
            timing["max_wait_ms"] = struct.unpack_from("<f", data, off)[0]; off += 4
        except Exception:
            pass
        state = _parse_proto_state(data[off:])
        return {
            "combat_steps": combat_steps,
            "elapsed_ms": elapsed_ms,
            "timing": timing,
            "state": state,
        }

    if opcode == OP_SEARCH_COMBAT_MCTS:
        return _decode_mcts_payload(data)

    raise RuntimeError(f"Unsupported proto response opcode: {opcode}")


def _apply_legal_action_to_proto(target: "pb.LegalAction", action: dict[str, Any]) -> None:
    """把 dict/proto action 填到 pb.LegalAction。"""
    if isinstance(action, pb.LegalAction):
        target.CopyFrom(action)
        return
    target.action = str(action.get("action") or action.get("type") or "other")
    idx = action.get("index", action.get("hand_index"))
    if idx is not None:
        target.index = int(idx)
    ci = action.get("card_index", action.get("hand_index", action.get("index")))
    if ci is not None:
        target.card_index = int(ci)
    if action.get("target_id") is not None:
        target.target_id = int(action["target_id"])
    if action.get("col") is not None:
        target.col = int(action["col"])
    if action.get("row") is not None:
        target.row = int(action["row"])
    if action.get("slot") is not None:
        target.slot = int(action["slot"])
    if action.get("label") is not None:
        target.label = str(action["label"])
    if action.get("card_id") is not None:
        target.card_id = str(action["card_id"])


__all__ = ["ProtoCodec"]
