"""game_bridge.sim CLI。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from game_bridge.session.build_spec import BuildSpecPy, CardSpecPy, PotionSpecPy, RelicSpecPy
from game_bridge.session import create_game_session
from game_bridge.sim import launch_headless_sim, static_consistency_report

_CARD_SPEC_RE = re.compile(r"^(?P<card_id>[^+@]+?)(?:[+@](?P<upgrade>\d+))?$")
_POTION_SLOT_PREFIX_RE = re.compile(r"^(?P<slot>\d+):(?P<potion_id>.+)$")
_POTION_SLOT_SUFFIX_RE = re.compile(r"^(?P<potion_id>.+)@(?P<slot>\d+)$")


def _load_json_value(raw: str, *, source: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{source} 不是合法 JSON: {exc}") from exc


def _parse_card_arg(raw: str) -> CardSpecPy:
    text = raw.strip()
    if not text:
        raise SystemExit("--card 不能为空")
    if text.startswith("{"):
        return CardSpecPy.from_value(_load_json_value(text, source="--card"))

    match = _CARD_SPEC_RE.fullmatch(text)
    if not match:
        raise SystemExit(f"无法解析牌规格: {raw!r}，支持 `card_id` 或 `card_id+2`")

    upgrade = match.group("upgrade")
    return CardSpecPy(
        id=match.group("card_id").strip(),
        upgrade_level=0 if upgrade is None else int(upgrade),
    )


def _parse_relic_arg(raw: str) -> RelicSpecPy:
    text = raw.strip()
    if not text:
        raise SystemExit("--relic 不能为空")
    if text.startswith("{"):
        return RelicSpecPy.from_value(_load_json_value(text, source="--relic"))
    return RelicSpecPy.from_value(text)


def _parse_potion_arg(raw: str) -> PotionSpecPy:
    text = raw.strip()
    if not text:
        raise SystemExit("--potion 不能为空")
    if text.startswith("{"):
        return PotionSpecPy.from_value(_load_json_value(text, source="--potion"))

    prefix_match = _POTION_SLOT_PREFIX_RE.fullmatch(text)
    if prefix_match:
        return PotionSpecPy(id=prefix_match.group("potion_id").strip(), slot=int(prefix_match.group("slot")))

    suffix_match = _POTION_SLOT_SUFFIX_RE.fullmatch(text)
    if suffix_match:
        return PotionSpecPy(id=suffix_match.group("potion_id").strip(), slot=int(suffix_match.group("slot")))

    return PotionSpecPy.from_value(text)


def _load_build_seed(args: argparse.Namespace) -> BuildSpecPy | None:
    seed_payload: dict[str, Any] | None = None

    if args.build_file:
        seed_payload = _load_json_value(Path(args.build_file).read_text(encoding="utf-8"), source="--build-file")
    if args.build_json:
        seed_payload = _load_json_value(args.build_json, source="--build-json")

    if seed_payload is None:
        build = BuildSpecPy()
    elif isinstance(seed_payload, dict):
        build = BuildSpecPy.from_dict(seed_payload)
    else:
        raise SystemExit("build JSON 顶层必须是对象")

    build.deck.extend(_parse_card_arg(raw) for raw in args.card)
    build.relics.extend(_parse_relic_arg(raw) for raw in args.relic)
    build.potions.extend(_parse_potion_arg(raw) for raw in args.potion)

    for attr in ("current_hp", "max_hp", "max_energy", "max_potion_slots", "gold"):
        value = getattr(args, attr)
        if value is not None:
            setattr(build, attr, value)

    return build if build.to_sim_dict() else None


def _player_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = battle.get("player") if isinstance(battle.get("player"), dict) else state.get("player")
    if not isinstance(player, dict):
        return {}

    return {
        "current_hp": player.get("current_hp", player.get("hp")),
        "max_hp": player.get("max_hp"),
        "block": player.get("block"),
        "energy": player.get("energy"),
        "max_energy": player.get("max_energy"),
        "gold": player.get("gold"),
        "deck_count": len(player.get("deck") or []),
        "relic_ids": [relic.get("id") for relic in player.get("relics") or [] if isinstance(relic, dict)],
        "potions": [
            {
                "slot": potion.get("slot", potion.get("index")),
                "id": potion.get("id"),
            }
            for potion in player.get("potions") or []
            if isinstance(potion, dict)
        ],
    }


def _run_combat(args: argparse.Namespace) -> None:
    if not args.encounter:
        raise SystemExit("combat 命令必须提供 --encounter")

    build = _load_build_seed(args)
    build_payload = None if build is None else build.to_sim_dict()

    with create_game_session(
        mode="combat",
        transport="pipe_proto",
        backend="sim",
        port=args.port,
        auto_launch=True,
        connect_timeout_s=args.ready_timeout,
        repo_root=args.repo_root,
        host_path=args.host_path,
    ) as session:
        state = session.reset(
            encounter_id=args.encounter,
            character_id=args.character,
            ascension_level=args.ascension,
            seed=args.seed,
            build=build,
        )
        legal_actions = session.legal_actions

    summary: dict[str, Any] = {
        "character_id": args.character,
        "encounter_id": args.encounter,
        "ascension_level": args.ascension,
        "seed": args.seed,
        "build": build_payload,
        "state_type": state.get("state_type"),
        "terminal": bool(state.get("terminal")),
        "run_outcome": state.get("run_outcome"),
        "player": _player_snapshot(state),
        "legal_action_count": len(legal_actions),
        "legal_actions_preview": legal_actions[: max(0, int(args.show_actions))],
    }
    if args.dump_state:
        summary["state"] = state

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch HeadlessSim via game bridge API.")
    parser.add_argument("command", nargs="?", default="launch", choices=["launch", "report", "combat"])
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--protocol", type=str, default="proto")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--host-path", type=str, default=None)

    parser.add_argument("--encounter", type=str, default=None)
    parser.add_argument("--character", type=str, default="IRONCLAD")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--build-json", type=str, default=None)
    parser.add_argument("--build-file", type=str, default=None)
    parser.add_argument("--card", action="append", default=[])
    parser.add_argument("--relic", action="append", default=[])
    parser.add_argument("--potion", action="append", default=[])
    parser.add_argument("--current-hp", type=int, default=None)
    parser.add_argument("--max-hp", type=int, default=None)
    parser.add_argument("--max-energy", type=int, default=None)
    parser.add_argument("--max-potion-slots", type=int, default=None)
    parser.add_argument("--gold", type=int, default=None)
    parser.add_argument("--show-actions", type=int, default=12)
    parser.add_argument("--dump-state", action="store_true")
    args = parser.parse_args()

    if args.command == "report":
        print(json.dumps(static_consistency_report(), ensure_ascii=False, indent=2))
        return

    if args.command == "combat":
        _run_combat(args)
        return

    handle = launch_headless_sim(
        port=args.port,
        protocol=args.protocol,
        connect_timeout_s=args.ready_timeout,
    )
    print(f"HeadlessSim ready on port {args.port} (pid={handle.pid})")


if __name__ == "__main__":
    main()
