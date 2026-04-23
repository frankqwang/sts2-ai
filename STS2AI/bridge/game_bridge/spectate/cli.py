"""game_bridge.spectate CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from game_bridge.session import create_full_run_session
from game_bridge.spectate.controller import SpectatorController
from game_bridge.spectate.overlay import OverlayWriter
from game_bridge.spectate.policy import ExternalPolicy, ManualPolicy, NullPolicy, ReplayPolicy


def _extract_floor(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("floor", "current_floor", "run_floor"):
        value = payload.get(key)
        if value is not None:
            return int(value)
    nested_build = payload.get("build")
    if isinstance(nested_build, dict):
        return _extract_floor(nested_build)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy-agnostic spectator controller.")
    parser.add_argument("--mode", choices=("manual", "replay", "external", "null"), default="manual")
    parser.add_argument("--replay-file", type=str, default="")
    parser.add_argument("--external-policy", type=str, default="")
    parser.add_argument("--overlay-file", type=str, default="")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:15526")
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--use-pipe", action="store_true")
    parser.add_argument("--transport", type=str, default="proto")
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--ready-timeout-s", type=float, default=60.0)
    parser.add_argument("--repo-root", type=str, default="")
    parser.add_argument("--host-path", type=str, default="")
    parser.add_argument("--dll-path", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--character-id", type=str, default="IRONCLAD")
    parser.add_argument("--encounter-id", type=str, default="")
    parser.add_argument("--seed", type=str, default="")
    parser.add_argument("--build-file", type=str, default="")
    parser.add_argument("--floor", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--step-delay", type=float, default=0.0)
    args = parser.parse_args()

    if args.auto_launch and not args.use_pipe:
        args.use_pipe = True

    resolved_repo_root = args.repo_root or None
    resolved_host_path = args.host_path or args.dll_path or None

    if args.mode == "manual":
        policy = ManualPolicy()
    elif args.mode == "replay":
        policy = ReplayPolicy.from_jsonl(args.replay_file)
    elif args.mode == "external":
        policy = ExternalPolicy.from_import_path(args.external_policy)
    else:
        policy = NullPolicy()

    session = create_full_run_session(
        base_url=args.base_url,
        port=args.port,
        use_pipe=args.use_pipe,
        transport=args.transport,
        request_timeout_s=args.request_timeout_s,
        ready_timeout_s=args.ready_timeout_s,
        auto_launch=args.auto_launch,
        repo_root=resolved_repo_root,
        host_path=resolved_host_path,
    )
    try:
        build_payload = None
        requested_floor = args.floor
        if args.build_file:
            build_payload = json.loads(Path(args.build_file).read_text(encoding="utf-8"))
            if requested_floor is None:
                requested_floor = _extract_floor(build_payload)
        overlay_path = Path(args.overlay_file) if args.overlay_file else None
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "use_pipe": bool(args.use_pipe),
                    "transport": args.transport,
                    "auto_launch": bool(args.auto_launch),
                    "repo_root": str(Path(resolved_repo_root).resolve()) if resolved_repo_root else None,
                    "host_path": str(Path(resolved_host_path).resolve()) if resolved_host_path else None,
                    "overlay_file": str(overlay_path) if overlay_path else None,
                    "build_file": args.build_file or None,
                    "encounter_id": args.encounter_id or None,
                    "floor": requested_floor,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        controller = SpectatorController(
            session=session,
            policy=policy,
            overlay=OverlayWriter(overlay_path) if overlay_path else None,
            step_delay=args.step_delay,
        )
        result = controller.play_episode(
            character_id=args.character_id,
            encounter_id=args.encounter_id or None,
            seed=args.seed or None,
            build=build_payload,
            floor=requested_floor,
            max_steps=args.max_steps,
        )
        print(result)
    finally:
        session.close()


if __name__ == "__main__":
    main()
