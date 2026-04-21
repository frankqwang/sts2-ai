"""sim 与真实观战逻辑的逐步 parity 对拍。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from game_bridge.session import create_full_run_session
from game_bridge.session.state_semantics import normalize_run_outcome

DEFAULT_PARITY_SEED = "123456"


def _normalized_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _action_identity(action: dict[str, Any]) -> tuple[Any, ...]:
    action_name = str(action.get("action") or action.get("type") or "").lower()
    target_id = action.get("target_id")
    if action_name == "use_potion" and target_id == 0:
        target_id = None
    return (
        action_name,
        action.get("index"),
        action.get("card_index"),
        target_id,
        action.get("col"),
        action.get("row"),
        action.get("slot"),
        _normalized_text(action.get("card_id")),
    )


def _canonical_action(action: dict[str, Any]) -> dict[str, Any]:
    action_name = str(action.get("action") or action.get("type") or "").lower()
    target_id = action.get("target_id")
    if target_id == 0 and action_name == "use_potion":
        target_id = None
    return {
        "action": action_name,
        "index": action.get("index"),
        "card_index": action.get("card_index"),
        "target_id": target_id,
        "col": action.get("col"),
        "row": action.get("row"),
        "slot": action.get("slot"),
        "card_id": _normalized_text(action.get("card_id")),
        "is_enabled": bool(action.get("is_enabled", True)),
    }


def _enabled_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(action)
        for action in (state.get("legal_actions") or [])
        if isinstance(action, dict) and action.get("is_enabled") is not False
    ]


def _find_matching_action(
    legal_actions: list[dict[str, Any]],
    reference_action: dict[str, Any],
) -> dict[str, Any] | None:
    reference_key = _action_identity(reference_action)
    for action in legal_actions:
        if _action_identity(action) == reference_key:
            return dict(action)
    return None


def _choose_default_action(legal_actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not legal_actions:
        return None
    preferred_actions = (
        "combat_confirm_selection",
        "confirm_selection",
        "proceed",
        "end_turn",
    )
    for action_name in preferred_actions:
        for action in legal_actions:
            if str(action.get("action") or action.get("type") or "").lower() == action_name:
                return dict(action)
    return dict(legal_actions[0])


def _summarize_player(
    player: dict[str, Any] | None,
    *,
    include_combat_fields: bool = True,
) -> dict[str, Any]:
    current = player or {}
    summary = {
        "hp": current.get("hp"),
        "max_hp": current.get("max_hp"),
        "gold": current.get("gold"),
    }
    if include_combat_fields:
        summary["block"] = current.get("block")
        summary["energy"] = current.get("energy")
        summary["max_energy"] = current.get("max_energy")
    return summary


def _summarize_hand(
    cards: list[dict[str, Any]],
    *,
    target_map: dict[int, list[int]] | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    targets = target_map or {}
    for card in cards:
        card_index = card.get("index")
        mapped_targets = targets.get(card_index) if isinstance(card_index, int) else None
        can_play = bool(card.get("can_play", False))
        valid_target_ids = list(card.get("valid_target_ids") or mapped_targets or [])
        requires_target = bool(card.get("requires_target", False)) or bool(mapped_targets)
        if not can_play:
            # Visible spectator state does not expose stable targetability metadata for
            # currently unplayable cards. Compare only actionable targeting semantics.
            valid_target_ids = []
            requires_target = False
        result.append(
            {
                "index": card_index,
                "id": _normalized_text(card.get("id")),
                "cost": card.get("cost"),
                "can_play": can_play,
                "requires_target": requires_target,
                "valid_target_ids": valid_target_ids,
            }
        )
    return result


def _summarize_enemies(enemies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for enemy in enemies:
        powers = enemy.get("powers") or enemy.get("status") or enemy.get("buffs") or []
        intents = enemy.get("intents") or []
        result.append(
            {
                "combat_id": enemy.get("combat_id"),
                "hp": enemy.get("hp"),
                "max_hp": enemy.get("max_hp"),
                "block": enemy.get("block"),
                "is_alive": bool(enemy.get("is_alive", False)),
                "is_hittable": bool(enemy.get("is_hittable", False)),
                "powers": [
                    {"id": _normalized_text(power.get("id")), "amount": power.get("amount")}
                    for power in powers
                ],
                "intents": [
                    {
                        "type": _normalized_text(str(intent.get("type") or "").lower()),
                        "damage": intent.get("damage", 0),
                        "total_damage": intent.get("total_damage", intent.get("damage", 0)),
                        "hits": intent.get("hits", intent.get("repeats", 1)),
                    }
                    for intent in intents
                ],
            }
        )
    return result


def _combat_target_map(state: dict[str, Any]) -> dict[int, list[int]]:
    mapping: dict[int, set[int]] = {}
    for action in state.get("legal_actions") or []:
        if not isinstance(action, dict):
            continue
        action_name = str(action.get("action") or action.get("type") or "").lower()
        if action_name != "play_card":
            continue
        card_index = action.get("card_index")
        target_id = action.get("target_id")
        if not isinstance(card_index, int) or not isinstance(target_id, int):
            continue
        mapping.setdefault(card_index, set()).add(target_id)
    return {key: sorted(values) for key, values in mapping.items()}


def _normalize_event_id(value: Any) -> str | None:
    text = _normalized_text(value)
    if text is None:
        return None
    if text.startswith("EVENT."):
        return text.removeprefix("EVENT.")
    return text


def _state_player_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    player = state.get("player")
    if isinstance(player, dict) and player:
        return player
    for key in ("event", "map", "shop", "rest_site", "treasure", "rewards", "card_reward"):
        payload = state.get(key)
        if isinstance(payload, dict) and isinstance(payload.get("player"), dict):
            return payload["player"]
    battle = state.get("battle")
    if isinstance(battle, dict) and isinstance(battle.get("player"), dict):
        return battle["player"]
    return None


def _screen_payload(state: dict[str, Any]) -> dict[str, Any]:
    state_type = str(state.get("state_type") or "")
    if state_type == "map":
        data = state.get("map") or {}
        return {
            "next_options": [
                {
                    "index": option.get("index"),
                    "col": option.get("col"),
                    "row": option.get("row"),
                    "point_type": _normalized_text(option.get("point_type") or option.get("type")),
                }
                for option in (data.get("next_options") or [])
            ],
        }
    if state_type == "event":
        data = state.get("event") or {}
        return {
            "event_id": _normalize_event_id(data.get("event_id")),
            "in_dialogue": bool(data.get("in_dialogue", False)),
            "is_finished": bool(data.get("is_finished", False)),
            "options": [
                {
                    "index": option.get("index"),
                    "id": _normalized_text(option.get("id")),
                    "is_locked": bool(option.get("is_locked", False)),
                    "is_proceed": bool(option.get("is_proceed", False)),
                }
                for option in (data.get("options") or [])
            ],
        }
    if state_type == "rest_site":
        data = state.get("rest_site") or {}
        return {
            "can_proceed": bool(data.get("can_proceed", False)),
            "options": [
                {
                    "index": option.get("index"),
                    "id": _normalized_text(option.get("id")),
                    "name": _normalized_text(option.get("name")),
                    "is_enabled": bool(option.get("is_enabled", False)),
                }
                for option in (data.get("options") or [])
            ],
        }
    if state_type == "shop":
        data = state.get("shop") or {}
        return {
            "is_open": bool(data.get("is_open", False)),
            "can_proceed": bool(data.get("can_proceed", False)),
            "items": [
                {
                    "index": item.get("index"),
                    "category": _normalized_text(item.get("category")),
                    "id": _normalized_text(item.get("id")),
                    "name": _normalized_text(item.get("name")),
                    "cost": item.get("cost"),
                    "can_afford": bool(item.get("can_afford", False)),
                    "is_stocked": bool(item.get("is_stocked", False)),
                    "on_sale": bool(item.get("on_sale", False)),
                }
                for item in (data.get("items") or [])
            ],
        }
    if state_type == "card_reward":
        data = state.get("card_reward") or {}
        return {
            "can_skip": bool(data.get("can_skip", False)),
            "cards": _summarize_hand(list(data.get("cards") or [])),
        }
    if state_type == "combat_rewards":
        data = state.get("rewards") or {}
        return {
            "can_proceed": bool(data.get("can_proceed", False)),
            "items": [
                {
                    "index": item.get("index"),
                    "type": _normalized_text(item.get("type")),
                    "claimable": bool(item.get("claimable", False)),
                }
                for item in (data.get("items") or [])
            ],
        }
    if state_type == "card_select":
        data = state.get("card_select") or {}
        return {
            "screen_type": _normalized_text(data.get("screen_type")),
            "selected_count": data.get("selected_count"),
            "can_confirm": bool(data.get("can_confirm", False)),
            "can_cancel": bool(data.get("can_cancel", False)),
            "cards": _summarize_hand(list(data.get("cards") or [])),
            "selected_cards": _summarize_hand(list(data.get("selected_cards") or [])),
        }
    if state_type == "relic_select":
        data = state.get("relic_select") or {}
        return {
            "can_skip": bool(data.get("can_skip", False)),
            "relics": [
                {
                    "index": relic.get("index"),
                    "id": _normalized_text(relic.get("id")),
                    "name": _normalized_text(relic.get("name")),
                }
                for relic in (data.get("relics") or [])
            ],
        }
    if state_type == "treasure":
        data = state.get("treasure") or {}
        return {
            "can_proceed": bool(data.get("can_proceed", False)),
            "relics": [
                {
                    "index": relic.get("index"),
                    "id": _normalized_text(relic.get("id")),
                    "name": _normalized_text(relic.get("name")),
                }
                for relic in (data.get("relics") or [])
            ],
        }
    if state_type in {"monster", "elite", "boss", "hand_select"}:
        battle = state.get("battle") or {}
        hand = battle.get("hand")
        if not isinstance(hand, list):
            hand = (battle.get("player") or {}).get("hand") or []
        target_map = _combat_target_map(state)
        return {
            "round_number_raw": battle.get("round_number_raw", battle.get("round")),
            "turn": _normalized_text(battle.get("turn")),
            "is_play_phase": bool(battle.get("is_play_phase", False)),
            "can_end_turn": any(
                str(action.get("action") or action.get("type") or "").lower() == "end_turn"
                and action.get("is_enabled") is not False
                for action in (state.get("legal_actions") or [])
                if isinstance(action, dict)
            ),
            "player": _summarize_player(battle.get("player")),
            "hand": _summarize_hand(list(hand), target_map=target_map),
            "enemies": _summarize_enemies(list(battle.get("enemies") or [])),
        }
    return {}


def canonicalize_state(state: dict[str, Any]) -> dict[str, Any]:
    current_state_type = str(state.get("state_type") or "")
    include_combat_fields = current_state_type in {"monster", "elite", "boss", "hand_select"}
    return {
        "state_type": current_state_type,
        "terminal": bool(state.get("terminal", False)),
        "run_outcome": normalize_run_outcome(state.get("run_outcome")),
        "run": {
            "act": ((state.get("run") or {}).get("act")),
            "floor": ((state.get("run") or {}).get("floor")),
        },
        "player": _summarize_player(
            _state_player_payload(state),
            include_combat_fields=include_combat_fields,
        ),
        "legal_actions": [_canonical_action(action) for action in _enabled_legal_actions(state)],
        "payload": _screen_payload(state),
    }


def _diff_values(left: Any, right: Any, path: str = "") -> list[str]:
    diffs: list[str] = []
    if type(left) is not type(right):
        diffs.append(f"{path or '<root>'}: type {type(left).__name__} != {type(right).__name__}")
        return diffs
    if isinstance(left, dict):
        keys = sorted(set(left.keys()) | set(right.keys()))
        for key in keys:
            next_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                diffs.append(f"{next_path}: missing on left")
                continue
            if key not in right:
                diffs.append(f"{next_path}: missing on right")
                continue
            diffs.extend(_diff_values(left[key], right[key], next_path))
        return diffs
    if isinstance(left, list):
        if len(left) != len(right):
            diffs.append(f"{path or '<root>'}: len {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            diffs.extend(_diff_values(left_item, right_item, f"{path}[{index}]"))
        return diffs
    if left != right:
        diffs.append(f"{path or '<root>'}: {left!r} != {right!r}")
    return diffs


def compare_states(real_state: dict[str, Any], sim_state: dict[str, Any]) -> list[str]:
    return _diff_values(canonicalize_state(real_state), canonicalize_state(sim_state))


def _wait_until_stable(
    session: Any,
    state: dict[str, Any],
    *,
    max_polls: int,
    poll_interval_s: float,
) -> dict[str, Any]:
    current = state
    for _ in range(max_polls):
        if bool(current.get("terminal")) or _enabled_legal_actions(current):
            return current
        time.sleep(poll_interval_s)
        current = session.get_state()
    return current


@dataclass(slots=True)
class ReplayActionSource:
    actions: list[dict[str, Any]]
    cursor: int = 0

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ReplayActionSource":
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(actions=records)

    def next_action(self) -> dict[str, Any] | None:
        if self.cursor >= len(self.actions):
            return None
        action = dict(self.actions[self.cursor])
        self.cursor += 1
        return action


def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parents[2] / "Artifacts" / "parity" / "game_bridge"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"parity_{stamp}.json"


def _resolve_parity_seed(seed: str | None) -> str:
    text = str(seed or "").strip()
    return text or DEFAULT_PARITY_SEED


def run_full_run_parity(
    *,
    real_base_url: str = "http://127.0.0.1:15526",
    sim_port: int = 15527,
    character_id: str = "IRONCLAD",
    seed: str | None = None,
    max_steps: int = 100,
    stop_on_mismatch: bool = True,
    idle_polls: int = 20,
    idle_poll_interval_s: float = 0.25,
    action_mode: str = "first",
    replay_file: str | None = None,
    auto_launch_sim: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if action_mode not in {"first", "replay"}:
        raise ValueError(f"Unsupported action_mode: {action_mode!r}")
    effective_seed = _resolve_parity_seed(seed)
    replay_source = ReplayActionSource.from_jsonl(replay_file) if action_mode == "replay" else None

    real_session = create_full_run_session(
        base_url=real_base_url,
        use_pipe=False,
        auto_launch=False,
    )
    sim_session = create_full_run_session(
        port=sim_port,
        use_pipe=True,
        transport="proto",
        auto_launch=auto_launch_sim,
    )
    report: dict[str, Any] = {
        "config": {
            "real_base_url": real_base_url,
            "sim_port": sim_port,
            "character_id": character_id,
            "seed": effective_seed,
            "max_steps": max_steps,
            "stop_on_mismatch": stop_on_mismatch,
            "idle_polls": idle_polls,
            "idle_poll_interval_s": idle_poll_interval_s,
            "action_mode": action_mode,
            "replay_file": str(replay_file) if replay_file else None,
        },
        "status": "ok",
        "steps": [],
        "summary": {
            "mismatch_count": 0,
            "steps_executed": 0,
        },
    }

    try:
        real_state = _wait_until_stable(
            real_session,
            real_session.reset(character_id=character_id, seed=effective_seed),
            max_polls=idle_polls,
            poll_interval_s=idle_poll_interval_s,
        )
        sim_state = _wait_until_stable(
            sim_session,
            sim_session.reset(character_id=character_id, seed=effective_seed),
            max_polls=idle_polls,
            poll_interval_s=idle_poll_interval_s,
        )

        for step_index in range(max_steps + 1):
            diffs = compare_states(real_state, sim_state)
            step_record: dict[str, Any] = {
                "step_index": step_index,
                "real_state_type": real_state.get("state_type"),
                "sim_state_type": sim_state.get("state_type"),
                "diffs": diffs,
                "real_legal_actions": canonicalize_state(real_state)["legal_actions"],
                "sim_legal_actions": canonicalize_state(sim_state)["legal_actions"],
            }
            if diffs:
                report["summary"]["mismatch_count"] += 1
                step_record["mismatch"] = True
                report["steps"].append(step_record)
                report["status"] = "mismatch"
                if stop_on_mismatch:
                    break
            else:
                step_record["mismatch"] = False
                report["steps"].append(step_record)

            if bool(real_state.get("terminal")) or bool(sim_state.get("terminal")):
                break

            real_legal = _enabled_legal_actions(real_state)
            sim_legal = _enabled_legal_actions(sim_state)
            if not real_legal or not sim_legal:
                report["status"] = "stalled"
                break

            if action_mode == "replay":
                selected_action = replay_source.next_action() if replay_source is not None else None
            else:
                selected_action = _choose_default_action(real_legal)

            if selected_action is None:
                report["status"] = "stopped"
                break

            real_action = _find_matching_action(real_legal, selected_action)
            sim_action = _find_matching_action(sim_legal, selected_action)
            report["steps"][-1]["selected_action"] = _canonical_action(selected_action)
            if real_action is None or sim_action is None:
                report["status"] = "action_mismatch"
                report["steps"][-1]["action_match"] = {
                    "real_found": real_action is not None,
                    "sim_found": sim_action is not None,
                }
                report["summary"]["mismatch_count"] += 1
                break

            real_state = _wait_until_stable(
                real_session,
                real_session.act(real_action),
                max_polls=idle_polls,
                poll_interval_s=idle_poll_interval_s,
            )
            sim_state = _wait_until_stable(
                sim_session,
                sim_session.act(sim_action),
                max_polls=idle_polls,
                poll_interval_s=idle_poll_interval_s,
            )
            report["summary"]["steps_executed"] = step_index + 1
        else:
            report["status"] = "max_steps_reached"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            real_session.close()
        except Exception:
            pass
        try:
            sim_session.close()
        except Exception:
            pass
        destination = Path(output_path) if output_path else _default_output_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["output_path"] = str(destination)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare full-run sim vs visible/real singleplayer logic step-by-step.")
    parser.add_argument("--real-base-url", type=str, default="http://127.0.0.1:15526")
    parser.add_argument("--sim-port", type=int, default=15527)
    parser.add_argument("--character-id", type=str, default="IRONCLAD")
    parser.add_argument("--seed", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--idle-polls", type=int, default=20)
    parser.add_argument("--idle-poll-interval", type=float, default=0.25)
    parser.add_argument("--action-mode", choices=("first", "replay"), default="first")
    parser.add_argument("--replay-file", type=str, default="")
    parser.add_argument("--stop-on-mismatch", action="store_true")
    parser.add_argument("--no-auto-launch-sim", action="store_true")
    parser.add_argument("--output-path", type=str, default="")
    args = parser.parse_args()

    if args.action_mode == "replay" and not args.replay_file:
        raise SystemExit("--replay-file is required when --action-mode replay")

    report = run_full_run_parity(
        real_base_url=args.real_base_url,
        sim_port=args.sim_port,
        character_id=args.character_id,
        seed=args.seed or None,
        max_steps=args.max_steps,
        stop_on_mismatch=bool(args.stop_on_mismatch),
        idle_polls=args.idle_polls,
        idle_poll_interval_s=args.idle_poll_interval,
        action_mode=args.action_mode,
        replay_file=args.replay_file or None,
        auto_launch_sim=not args.no_auto_launch_sim,
        output_path=args.output_path or None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"ok", "stopped", "max_steps_reached"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
