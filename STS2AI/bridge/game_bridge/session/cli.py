"""game_bridge.session CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from game_bridge.session import create_combat_session, create_full_run_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect game bridge session state.")
    parser.add_argument("inspect", nargs="?", default="inspect")
    parser.add_argument("--kind", choices=("combat", "full_run"), default="full_run")
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:15526")
    parser.add_argument("--transport", type=str, default="proto")
    parser.add_argument("--use-pipe", action="store_true")
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--repo-root", type=str, default="")
    parser.add_argument("--host-path", type=str, default="")
    parser.add_argument("--dll-path", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--character-id", type=str, default="IRONCLAD")
    parser.add_argument("--encounter-id", type=str, default="")
    parser.add_argument("--seed", type=str, default="")
    args = parser.parse_args()

    if args.auto_launch and not args.use_pipe and args.kind == "full_run":
        args.use_pipe = True

    resolved_repo_root = args.repo_root or None
    resolved_host_path = args.host_path or args.dll_path or None
    print(
        json.dumps(
            {
                "kind": args.kind,
                "use_pipe": bool(args.use_pipe),
                "transport": args.transport,
                "auto_launch": bool(args.auto_launch),
                "repo_root": str(Path(resolved_repo_root).resolve()) if resolved_repo_root else None,
                "host_path": str(Path(resolved_host_path).resolve()) if resolved_host_path else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.kind == "combat":
        with create_combat_session(
            port=args.port,
            auto_launch=args.auto_launch,
            repo_root=resolved_repo_root,
            host_path=resolved_host_path,
        ) as session:
            state = session.reset(
                character_id=args.character_id,
                encounter_id=args.encounter_id,
                seed=args.seed or None,
            )
    else:
        session = create_full_run_session(
            port=args.port,
            base_url=args.base_url,
            use_pipe=args.use_pipe,
            transport=args.transport,
            auto_launch=args.auto_launch,
            repo_root=resolved_repo_root,
            host_path=resolved_host_path,
        )
        try:
            state = session.reset(character_id=args.character_id, seed=args.seed or None)
        finally:
            session.close()
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
