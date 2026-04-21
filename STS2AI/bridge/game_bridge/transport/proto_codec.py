"""ProtoCodec — 纯 protobuf envelope 的 pipe codec."""
from __future__ import annotations

import json
from typing import Any

from game_bridge.generated import game_state_pb2 as pb
from game_bridge.session.state_semantics import is_failure_outcome, is_victory_outcome
from game_bridge.transport.codec import ProtocolCodec
from game_bridge.transport.proto_state_converter import game_state_to_dict


def _parse_proto_state(state: "pb.GameState") -> dict[str, Any]:
    result = game_state_to_dict(state)
    return result


def _terminal_reward(state: dict[str, Any]) -> float:
    if not bool(state.get("terminal")):
        return 0.0
    outcome = state.get("run_outcome")
    if is_victory_outcome(outcome):
        return 1.0
    if is_failure_outcome(outcome):
        return -1.0
    return 0.0


def _decode_search_mcts_payload(payload: "pb.PipeSearchCombatMctsResult") -> dict[str, Any]:
    result: dict[str, Any] = {
        "action_index": int(payload.action_index),
        "visit_counts": [int(v) for v in payload.visit_counts],
        "visit_probs": [float(v) for v in payload.visit_probs],
        "q_values": [float(v) for v in payload.q_values],
        "priors": [float(v) for v in payload.priors],
        "root_value": float(payload.root_value),
        "search_ms": float(payload.search_ms),
        "restored_ok": bool(payload.restored_ok),
        "snapshot_count": int(payload.snapshot_count),
        "breakdown": {
            "simulation_count": int(payload.simulation_count),
            "save_state_count": int(payload.save_state_count),
            "load_state_count": int(payload.load_state_count),
            "delete_state_count": int(payload.delete_state_count),
            "step_count": int(payload.step_count),
            "advance_state_count": int(payload.advance_state_count),
            "eval_call_count": int(payload.eval_call_count),
            "eval_batch_count": int(payload.eval_batch_count),
            "eval_state_count": int(payload.eval_state_count),
            "select_child_count": int(payload.select_child_count),
            "backprop_count": int(payload.backprop_count),
            "save_state_ms": float(payload.save_state_ms),
            "load_state_ms": float(payload.load_state_ms),
            "delete_state_ms": float(payload.delete_state_ms),
            "step_ms": float(payload.step_ms),
            "advance_state_ms": float(payload.advance_state_ms),
            "eval_ms": float(payload.eval_ms),
            "selection_ms": float(payload.selection_ms),
            "backprop_ms": float(payload.backprop_ms),
        },
    }
    if payload.debug_trace_json:
        try:
            result["debug_trace"] = json.loads(payload.debug_trace_json)
        except Exception:
            result["debug_trace_json"] = payload.debug_trace_json
    return result


def _apply_legal_action_to_proto(target: "pb.LegalAction", action: dict[str, Any]) -> None:
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


class ProtoCodec(ProtocolCodec):
    """Protobuf envelope codec for PipeConnection."""

    name = "proto"

    def encode_request(self, method: str, params: dict[str, Any] | None) -> bytes:
        method = str(method).strip().lower()
        params = params or {}
        req = pb.PipeRequestEnvelope()

        if method == "reset":
            req.method = pb.RESET
            req.reset.character_id = str(params.get("character_id") or params.get("character") or "")
            req.reset.seed = str(params.get("seed") or "")
            req.reset.ascension_level = int(params.get("ascension_level", params.get("ascension", 0)) or 0)
            build = params.get("build")
            if build is not None:
                req.reset.build_json = json.dumps(build, ensure_ascii=False, separators=(",", ":"))
            return req.SerializeToString()

        if method in {"state", "get_state", "legal_actions"}:
            req.method = pb.STATE
            return req.SerializeToString()

        if method == "step":
            req.method = pb.STEP
            _apply_legal_action_to_proto(req.step.action, params)
            return req.SerializeToString()

        if method == "batch_step":
            req.method = pb.BATCH_STEP
            for action in list(params.get("actions") or []):
                _apply_legal_action_to_proto(req.batch_step.actions.add(), action or {})
            return req.SerializeToString()

        if method == "save_state":
            req.method = pb.SAVE_STATE
            return req.SerializeToString()

        if method == "export_state":
            req.method = pb.EXPORT_STATE
            req.export_state.path = str(params["path"])
            if params.get("state_id") is not None:
                req.export_state.state_id = str(params["state_id"])
            return req.SerializeToString()

        if method == "import_state":
            req.method = pb.IMPORT_STATE
            req.import_state.path = str(params["path"])
            return req.SerializeToString()

        if method == "load_state":
            req.method = pb.LOAD_STATE
            req.load_state.state_id = str(params["state_id"])
            return req.SerializeToString()

        if method in {"delete_state", "clear_state_cache"}:
            clear_all = bool(params.get("clear_all")) or method == "clear_state_cache"
            req.method = pb.DELETE_STATE
            req.delete_state.clear_all = clear_all
            if not clear_all:
                req.delete_state.state_id = str(params["state_id"])
            return req.SerializeToString()

        if method == "perf_stats":
            req.method = pb.PERF_STATS
            return req.SerializeToString()

        if method == "reset_perf_stats":
            req.method = pb.RESET_PERF_STATS
            return req.SerializeToString()

        if method == "step_local_policy":
            req.method = pb.STEP_LOCAL_POLICY
            return req.SerializeToString()

        if method == "skip_combat":
            req.method = pb.SKIP_COMBAT
            return req.SerializeToString()

        if method == "run_combat_local":
            req.method = pb.RUN_COMBAT_LOCAL
            req.run_combat_local.max_steps = int(params.get("max_steps", 600))
            return req.SerializeToString()

        if method == "load_ort_model":
            req.method = pb.LOAD_ORT_MODEL
            req.load_ort_model.path = str(params.get("path", ""))
            return req.SerializeToString()

        if method == "search_combat_mcts":
            req.method = pb.SEARCH_COMBAT_MCTS
            req.search_combat_mcts.num_simulations = int(params.get("num_simulations", 0))
            req.search_combat_mcts.c_puct = float(params.get("c_puct", 1.5))
            req.search_combat_mcts.dirichlet_alpha = float(params.get("dirichlet_alpha", 0.0))
            req.search_combat_mcts.dirichlet_fraction = float(params.get("dirichlet_fraction", 0.0))
            req.search_combat_mcts.max_step_budget = int(params.get("max_step_budget", 200))
            req.search_combat_mcts.final_action_mode = str(params.get("final_action_mode", "visit")).strip().lower()
            req.search_combat_mcts.final_action_top_k = int(params.get("final_action_top_k", 3))
            req.search_combat_mcts.final_action_q_weight = float(params.get("final_action_q_weight", 0.35))
            req.search_combat_mcts.use_continuation_value = bool(params.get("use_continuation_value", False))
            req.search_combat_mcts.enable_debug_trace = bool(params.get("debug_trace", False))
            return req.SerializeToString()

        if method == "combat_reset":
            req.method = pb.COMBAT_RESET
            payload = req.combat_reset
            payload.character_id = str(params.get("character_id") or params.get("character") or "")
            payload.encounter_id = str(params.get("encounter_id") or "")
            payload.ascension_level = int(params.get("ascension_level") or params.get("ascension") or 0)
            payload.seed = str(params.get("seed") or "")
            build = params.get("build")
            if isinstance(build, dict):
                payload.build.current_hp = int(build.get("current_hp") or 0)
                payload.build.max_hp = int(build.get("max_hp") or 0)
                payload.build.max_energy = int(build.get("max_energy") or 3)
                payload.build.gold = int(build.get("gold") or 0)
                for card in build.get("deck") or []:
                    if isinstance(card, dict):
                        payload.build.deck.add(
                            id=str(card.get("id") or "").upper(),
                            upgrade_level=int(card.get("upgrade_level") or 0),
                        )
                    elif isinstance(card, str):
                        payload.build.deck.add(id=card.upper(), upgrade_level=0)
                for relic in build.get("relics") or []:
                    if isinstance(relic, dict):
                        payload.build.relics.add(id=str(relic.get("id") or "").upper())
                    elif isinstance(relic, str):
                        payload.build.relics.add(id=relic.upper())
            elif build is not None:
                raise TypeError("combat_reset build must be a dict or None")
            return req.SerializeToString()

        if method == "combat_step":
            req.method = pb.COMBAT_STEP
            _apply_legal_action_to_proto(req.combat_step.action, params)
            return req.SerializeToString()

        if method == "combat_state":
            req.method = pb.COMBAT_STATE
            return req.SerializeToString()

        raise ValueError(f"Unsupported proto pipe method: {method}")

    def decode_response(self, payload: bytes) -> dict[str, Any]:
        resp = pb.PipeResponseEnvelope()
        resp.ParseFromString(payload)

        if resp.status != pb.OK:
            error_code = resp.error.error_code if resp.HasField("error") else None
            error_message = resp.error.error_message if resp.HasField("error") else f"proto request failed: status={resp.status}"
            return {
                "status": int(resp.status),
                "method": int(resp.method),
                "error_code": error_code,
                "error": error_message,
            }

        if resp.method in {pb.RESET, pb.STATE, pb.LOAD_STATE, pb.IMPORT_STATE, pb.COMBAT_RESET, pb.COMBAT_STATE}:
            return _parse_proto_state(resp.state.state)

        if resp.method in {pb.STEP, pb.STEP_LOCAL_POLICY, pb.SKIP_COMBAT, pb.COMBAT_STEP}:
            state = _parse_proto_state(resp.step.state)
            result = {
                "accepted": bool(resp.step.accepted),
                "error": str(resp.step.error or "") or None,
                "state": state,
                "reward": _terminal_reward(state),
                "done": bool(state.get("terminal")),
                "info": {
                    "state_type": state.get("state_type"),
                    "run_outcome": state.get("run_outcome"),
                },
            }
            if resp.method == pb.SKIP_COMBAT:
                result["skipped"] = True
            return result

        if resp.method == pb.BATCH_STEP:
            state = _parse_proto_state(resp.batch_step.state)
            return {
                "accepted": bool(resp.batch_step.accepted),
                "steps_executed": int(resp.batch_step.steps_executed),
                "error": str(resp.batch_step.error or "") or None,
                "state": state,
            }

        if resp.method == pb.SAVE_STATE:
            return {
                "state_id": resp.save_state.state_id,
                "cache_size": int(resp.save_state.cache_size),
            }

        if resp.method == pb.EXPORT_STATE:
            return {
                "path": resp.export_state.path,
                "cache_size": int(resp.export_state.cache_size),
            }

        if resp.method == pb.DELETE_STATE:
            return {
                "deleted": bool(resp.delete_state.deleted),
                "cache_size": int(resp.delete_state.cache_size),
            }

        if resp.method == pb.PERF_STATS:
            return json.loads(resp.perf_stats.json_payload or "{}")

        if resp.method == pb.RESET_PERF_STATS:
            return {"reset": bool(resp.reset_perf_stats.reset)}

        if resp.method == pb.LOAD_ORT_MODEL:
            return {
                "loaded": bool(resp.load_ort_model.loaded),
                "has_value": bool(resp.load_ort_model.has_value_output),
                "has_deck_inputs": bool(resp.load_ort_model.has_deck_inputs),
                "has_continuation_output": bool(resp.load_ort_model.has_continuation_output),
                "has_extra_scalars_input": bool(resp.load_ort_model.has_extra_scalars_input),
                "execution_provider": resp.load_ort_model.execution_provider_name,
                "requested_device": resp.load_ort_model.requested_device,
                "fell_back_to_cpu": bool(resp.load_ort_model.fell_back_to_cpu),
            }

        if resp.method == pb.RUN_COMBAT_LOCAL:
            state = _parse_proto_state(resp.run_combat_local.state)
            return {
                "combat_steps": int(resp.run_combat_local.combat_steps),
                "elapsed_ms": float(resp.run_combat_local.elapsed_ms),
                "timing": {
                    "get_snapshot_ms": float(resp.run_combat_local.get_snapshot_ms),
                    "ort_ms": float(resp.run_combat_local.ort_ms),
                    "step_async_ms": float(resp.run_combat_local.step_async_ms),
                    "wait_async_ms": float(resp.run_combat_local.wait_async_ms),
                    "max_step_ms": float(resp.run_combat_local.max_step_ms),
                    "max_wait_ms": float(resp.run_combat_local.max_wait_ms),
                },
                "state": state,
            }

        if resp.method == pb.SEARCH_COMBAT_MCTS:
            return _decode_search_mcts_payload(resp.search_combat_mcts)

        raise RuntimeError(f"Unsupported proto response method: {resp.method}")

    def read_handshake(self, payload: bytes) -> dict[str, Any]:
        resp = pb.PipeResponseEnvelope()
        resp.ParseFromString(payload)
        if resp.status != pb.OK or resp.method != pb.HANDSHAKE or not resp.HasField("handshake"):
            if resp.HasField("error"):
                return {
                    "error": resp.error.error_message or "handshake failed",
                    "error_code": resp.error.error_code or None,
                }
            return {"error": "malformed handshake"}

        version = int(resp.handshake.protocol_version)
        schema_id = resp.handshake.schema_id
        if version != 1:
            return {"error": f"proto protocol version mismatch: expected 1, got {version}"}
        if schema_id != "sts2-proto-v1":
            return {"error": f"proto schema mismatch: expected sts2-proto-v1, got {schema_id or '<missing>'}"}
        return {
            "version": version,
            "build_git_sha": resp.handshake.build_git_sha,
            "schema_id": schema_id,
        }


__all__ = ["ProtoCodec"]
