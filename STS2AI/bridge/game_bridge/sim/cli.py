"""game_bridge.sim CLI。"""

from __future__ import annotations

import argparse

from game_bridge.sim import launch_headless_sim


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch HeadlessSim via game bridge API.")
    parser.add_argument("launch", nargs="?", default="launch")
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--protocol", type=str, default="proto")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    args = parser.parse_args()

    handle = launch_headless_sim(
        port=args.port,
        protocol=args.protocol,
        connect_timeout_s=args.ready_timeout,
    )
    print(f"HeadlessSim ready on port {args.port} (pid={handle.pid})")


if __name__ == "__main__":
    main()
