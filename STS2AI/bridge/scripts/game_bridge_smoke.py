from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
PYTHON_ROOT = THIS_FILE.parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from game_bridge import SpectatorController, create_game_session
from game_bridge.sim import DEFAULT_HOST_PATH, launch_headless_sim
from game_bridge.spectate import NullPolicy, OverlayWriter, ReplayPolicy


@dataclass
class FakeSpectateSession:
    terminal_after_act: bool = True

    def __post_init__(self) -> None:
        self._state = {
            "state_type": "map",
            "terminal": False,
            "run_outcome": None,
            "legal_actions": [
                {
                    "action": "proceed",
                    "label": "Proceed",
                    "is_enabled": True,
                }
            ],
        }

    def reset(self, **_kwargs) -> dict[str, Any]:
        self._state = {
            "state_type": "map",
            "terminal": False,
            "run_outcome": None,
            "legal_actions": [
                {
                    "action": "proceed",
                    "label": "Proceed",
                    "is_enabled": True,
                }
            ],
        }
        return dict(self._state)

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def act(self, _action: dict[str, Any]) -> dict[str, Any]:
        self._state = {
            "state_type": "game_over",
            "terminal": self.terminal_after_act,
            "run_outcome": "victory" if self.terminal_after_act else None,
            "legal_actions": [],
        }
        return dict(self._state)

    def close(self) -> None:
        return None


def run_fake_spectate(*, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "replay.jsonl"
    overlay_path = output_dir / "overlay.json"
    replay_path.write_text(json.dumps({"action": "proceed"}) + "\n", encoding="utf-8")

    session = FakeSpectateSession()
    policy = ReplayPolicy.from_jsonl(replay_path)
    controller = SpectatorController(
        session=session,
        policy=policy,
        overlay=OverlayWriter(overlay_path),
    )
    result = controller.play_episode(max_steps=4)
    print(json.dumps({"mode": "fake_spectate", "result": result}, ensure_ascii=False, indent=2))
    print(f"overlay={overlay_path}")
    return 0


def run_sim_launch_check(*, port: int) -> int:
    if not DEFAULT_HOST_PATH.exists():
        print(
            json.dumps(
                {
                    "mode": "sim_launch",
                    "ready": False,
                    "reason": "missing_host_binary",
                    "host_path": str(DEFAULT_HOST_PATH),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    handle = launch_headless_sim(port=port, protocol="proto", connect_timeout_s=10.0)
    try:
        print(
            json.dumps(
                {
                    "mode": "sim_launch",
                    "ready": True,
                    "pid": handle.pid,
                    "port": port,
                    "log_dir": str(handle.log_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        handle.process.terminate()
        handle.process.wait(timeout=5)
    return 0


def run_full_run_state_check(*, port: int) -> int:
    session = create_game_session(
        mode="full_run",
        port=port,
        transport="pipe_proto",
        backend="sim",
        auto_launch=True,
    )
    try:
        state = session.get_state()
        print(json.dumps({"mode": "full_run_state", "state_type": state.get("state_type")}, ensure_ascii=False, indent=2))
    finally:
        session.close()
    return 0


def run_full_run_reset_check(*, port: int) -> int:
    session = create_game_session(
        mode="full_run",
        port=port,
        transport="pipe_proto",
        backend="sim",
        auto_launch=True,
    )
    try:
        state = session.reset(character_id="IRONCLAD", seed=None)
        print(
            json.dumps(
                {
                    "mode": "full_run_reset",
                    "state_type": state.get("state_type"),
                    "legal_actions": len(state.get("legal_actions") or []),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        session.close()
    return 0


def run_combat_build_api_check(*, port: int) -> int:
    session = create_game_session(
        mode="combat",
        transport="pipe_proto",
        backend="sim",
        port=port,
        auto_launch=True,
    )
    build = {
        "deck": [
            {"id": "STRIKE_IRONCLAD"},
            {"id": "STRIKE_IRONCLAD"},
            {"id": "STRIKE_IRONCLAD"},
            {"id": "DEFEND_IRONCLAD"},
            {"id": "DEFEND_IRONCLAD"},
            {"id": "DEFEND_IRONCLAD"},
            {"id": "BASH"},
            {"id": "POMMEL_STRIKE", "upgrade_level": 1},
            {"id": "SETUP_STRIKE", "upgrade_level": 1},
            {"id": "FORGOTTEN_RITUAL"},
            {"id": "BLUDGEON", "upgrade_level": 1},
            {"id": "CINDER", "upgrade_level": 1},
        ],
        "relics": [
            {"id": "BURNING_BLOOD"},
            {"id": "HAND_DRILL"},
            {"id": "MINIATURE_CANNON"},
            {"id": "SILVER_CRUCIBLE"},
        ],
        "current_hp": 80,
        "max_hp": 80,
        "max_energy": 3,
        "gold": 99,
    }
    try:
        state = session.reset(character_id="IRONCLAD", encounter_id="CHOMPERS_NORMAL", build=build)
        top_player = state.get("player") or {}
        battle_player = ((state.get("battle") or {}).get("player")) or {}
        result = {
            "mode": "combat_build_api",
            "state_type": state.get("state_type"),
            "top_deck_count": len(top_player.get("deck") or []),
            "battle_deck_count": len(battle_player.get("deck") or []),
            "top_relic_count": len(top_player.get("relics") or []),
            "battle_relic_count": len(battle_player.get("relics") or []),
            "top_powers_count": len(top_player.get("powers") or []),
            "battle_powers_count": len(battle_player.get("powers") or []),
            "requested_deck_count": len(build["deck"]),
            "requested_relic_count": len(build["relics"]),
        }
        ok = (
            result["top_deck_count"] == result["requested_deck_count"]
            and result["battle_deck_count"] == result["requested_deck_count"]
            and result["top_relic_count"] == result["requested_relic_count"]
            and result["battle_relic_count"] == result["requested_relic_count"]
        )
        result["ok"] = ok
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if ok else 3
    finally:
        session.close()


def run_real_spectate_null(*, output_dir: Path, port: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "real_overlay.json"
    session = create_game_session(
        mode="full_run",
        port=port,
        transport="pipe_proto",
        backend="sim",
        auto_launch=True,
    )
    try:
        controller = SpectatorController(
            session=session,
            policy=NullPolicy(),
            overlay=OverlayWriter(overlay_path),
        )
        result = controller.play_episode(max_steps=5)
        print(
            json.dumps(
                {
                    "mode": "real_spectate_null",
                    "result": result,
                    "overlay_file": str(overlay_path),
                    "overlay_exists": overlay_path.exists(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        session.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke tests for game_bridge.")
    parser.add_argument(
        "--mode",
        choices=("fake_spectate", "sim_launch", "full_run_state", "full_run_reset", "combat_build_api", "real_spectate_null"),
        default="fake_spectate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PYTHON_ROOT.parent / "Artifacts" / "smoke" / "game_bridge",
    )
    parser.add_argument("--port", type=int, default=15527)
    args = parser.parse_args()

    print(
        json.dumps(
            {
                "mode": args.mode,
                "port": args.port,
                "output_dir": str(args.output_dir),
                "default_host_path": str(DEFAULT_HOST_PATH),
                "default_host_exists": DEFAULT_HOST_PATH.exists(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.mode == "fake_spectate":
        return run_fake_spectate(output_dir=args.output_dir)
    if args.mode == "sim_launch":
        return run_sim_launch_check(port=args.port)
    if args.mode == "full_run_state":
        return run_full_run_state_check(port=args.port)
    if args.mode == "full_run_reset":
        return run_full_run_reset_check(port=args.port)
    if args.mode == "combat_build_api":
        return run_combat_build_api_check(port=args.port)
    return run_real_spectate_null(output_dir=args.output_dir, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
