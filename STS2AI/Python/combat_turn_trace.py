#!/usr/bin/env python3
"""Record and compare step-level public combat traces across backends.

This Task-0 harness records a public-state trace entry for every combat step:
state hash, legal mask, player/public pile stats, enemy intents/powers, and
the chosen outgoing action. In compare mode it drives both backends with the
same seed and the same action sequence, then emits state-level mismatches.
"""
from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/{core,ipc,search} to sys.path)

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from full_run_env import FullRunClientLike
from public_state_trace import PublicStateTraceEntry as TraceEntry
from public_state_trace import build_trace_entry
from sim_semantic_audit_common import (
    DEFAULT_GODOT_EXE,
    DEFAULT_HEADLESS_DLL,
    DEFAULT_PORT,
    DEFAULT_REPO_ROOT,
    backend_client,
)
from test_simulator_consistency import COMBAT_TYPES, pick_deterministic_action
from verify_save_load import drive_to_state


@dataclass
class Mismatch:
    step: int
    field: str
    value_a: Any
    value_b: Any

    def __str__(self) -> str:
        return f"step {self.step} [{self.field}]: {self.value_a!r} != {self.value_b!r}"


class CombatTurnTracer:
    """Records public trace entries during combat and compares them step by step."""

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []
        self._step: int = 0

    @property
    def entries(self) -> list[TraceEntry]:
        return list(self._entries)

    def record(self, state: dict[str, Any], action: dict[str, Any] | None = None) -> TraceEntry:
        entry = build_trace_entry(state, step=self._step, action=action)
        self._entries.append(entry)
        self._step += 1
        return entry

    def reset(self) -> None:
        self._entries.clear()
        self._step = 0

    def flush(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for entry in self._entries:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def load_trace(filepath: str | Path) -> list[TraceEntry]:
        entries: list[TraceEntry] = []
        with open(filepath, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data["hand_labels"] = data.get("hand_labels") or []
                data["enemies"] = data.get("enemies") or []
                data["player_statuses"] = [tuple(item) for item in data.get("player_statuses") or []]
                data["legal_mask"] = [tuple(item) for item in data.get("legal_mask") or []]
                entries.append(TraceEntry(**data))
        return entries

    @staticmethod
    def compare_traces(trace_a: list[TraceEntry], trace_b: list[TraceEntry]) -> list[Mismatch]:
        mismatches: list[Mismatch] = []

        if len(trace_a) != len(trace_b):
            mismatches.append(Mismatch(step=-1, field="trace_length", value_a=len(trace_a), value_b=len(trace_b)))

        for step_idx in range(min(len(trace_a), len(trace_b))):
            left = trace_a[step_idx]
            right = trace_b[step_idx]
            for field_name in (
                "public_state_hash",
                "state_type",
                "terminal",
                "floor",
                "act",
                "turn",
                "player_hp",
                "player_block",
                "energy",
                "hand_labels",
                "draw_count",
                "discard_count",
                "exhaust_count",
                "player_statuses",
                "legal_mask",
                "legal_action_count",
            ):
                if getattr(left, field_name) != getattr(right, field_name):
                    mismatches.append(
                        Mismatch(
                            step=step_idx,
                            field=field_name,
                            value_a=getattr(left, field_name),
                            value_b=getattr(right, field_name),
                        )
                    )

            left_enemies = sorted(
                (
                    enemy.get("key"),
                    enemy.get("hp"),
                    enemy.get("max_hp"),
                    enemy.get("block"),
                    bool(enemy.get("is_alive", True)),
                    enemy.get("intent"),
                    tuple(enemy.get("statuses") or []),
                )
                for enemy in left.enemies
            )
            right_enemies = sorted(
                (
                    enemy.get("key"),
                    enemy.get("hp"),
                    enemy.get("max_hp"),
                    enemy.get("block"),
                    bool(enemy.get("is_alive", True)),
                    enemy.get("intent"),
                    tuple(enemy.get("statuses") or []),
                )
                for enemy in right.enemies
            )
            if left_enemies != right_enemies:
                mismatches.append(Mismatch(step=step_idx, field="enemies", value_a=left_enemies, value_b=right_enemies))

        return mismatches


def _pick_trace_action(state: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    legal = [action for action in state.get("legal_actions") or [] if bool(action.get("is_enabled", True))]
    if not legal:
        return None
    return pick_deterministic_action(str(state.get("state_type") or ""), legal, rng)


def _run_combat_trace(
    client: FullRunClientLike,
    seed: str,
    *,
    max_turns: int = 50,
    max_steps: int = 500,
) -> list[TraceEntry]:
    state = drive_to_state(client, seed, COMBAT_TYPES)
    tracer = CombatTurnTracer()
    rng = random.Random(seed)

    for _ in range(max_steps):
        battle = state.get("battle") or {}
        turn = int(battle.get("round_number") or battle.get("round") or 0)
        state_type = str(state.get("state_type") or "")
        if state_type not in COMBAT_TYPES:
            break
        action = None
        if not state.get("terminal") and turn <= max_turns:
            action = _pick_trace_action(state, rng)
        tracer.record(state, action)
        if action is None:
            break
        state = client.act(action)

    return tracer.entries


def _run_shared_action_trace(
    driver_client: FullRunClientLike,
    follower_client: FullRunClientLike,
    seed: str,
    *,
    max_turns: int = 50,
    max_steps: int = 500,
) -> tuple[list[TraceEntry], list[TraceEntry]]:
    driver_state = drive_to_state(driver_client, seed, COMBAT_TYPES)
    follower_state = drive_to_state(follower_client, seed, COMBAT_TYPES)
    driver_tracer = CombatTurnTracer()
    follower_tracer = CombatTurnTracer()
    rng = random.Random(seed)

    for _ in range(max_steps):
        driver_type = str(driver_state.get("state_type") or "")
        follower_type = str(follower_state.get("state_type") or "")
        if driver_type not in COMBAT_TYPES or follower_type not in COMBAT_TYPES:
            break
        driver_turn = int((driver_state.get("battle") or {}).get("round_number") or (driver_state.get("battle") or {}).get("round") or 0)
        action = None
        if (
            not driver_state.get("terminal")
            and not follower_state.get("terminal")
            and driver_turn <= max_turns
        ):
            action = _pick_trace_action(driver_state, rng)

        driver_tracer.record(driver_state, action)
        follower_tracer.record(follower_state, action)
        if action is None:
            break

        driver_state = driver_client.act(action)
        follower_state = follower_client.act(action)

    return driver_tracer.entries, follower_tracer.entries


def _single_direction_comparison(
    driver_client: FullRunClientLike,
    follower_client: FullRunClientLike,
    seed: str,
    *,
    max_turns: int = 50,
    max_steps: int = 500,
) -> dict[str, Any]:
    trace_driver, trace_follower = _run_shared_action_trace(
        driver_client,
        follower_client,
        seed,
        max_turns=max_turns,
        max_steps=max_steps,
    )
    mismatches = CombatTurnTracer.compare_traces(trace_driver, trace_follower)
    first = mismatches[0] if mismatches else None
    step = first.step if first else -1
    return {
        "steps_driver": len(trace_driver),
        "steps_follower": len(trace_follower),
        "mismatch_count": len(mismatches),
        "mismatches": [str(item) for item in mismatches[:20]],
        "first_mismatch": {
            "step": step,
            "field": first.field if first else None,
            "value_driver": first.value_a if first else None,
            "value_follower": first.value_b if first else None,
            "driver_entry": trace_driver[step].to_dict() if first and 0 <= step < len(trace_driver) else None,
            "follower_entry": trace_follower[step].to_dict() if first and 0 <= step < len(trace_follower) else None,
        }
        if first
        else None,
        "passed": len(mismatches) == 0,
    }


def _compare_backends(
    client_a: FullRunClientLike,
    client_b: FullRunClientLike,
    seed: str,
    *,
    max_turns: int = 50,
    max_steps: int = 500,
    driver_mode: str = "bidirectional",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "seed": seed,
        "driver_mode": driver_mode,
    }
    directions = []
    if driver_mode in {"forward", "bidirectional"}:
        directions.append(("forward", client_a, client_b))
    if driver_mode in {"reverse", "bidirectional"}:
        directions.append(("reverse", client_b, client_a))

    total_mismatches = 0
    all_passed = True
    for direction_name, driver_client, follower_client in directions:
        direction_result = _single_direction_comparison(
            driver_client,
            follower_client,
            seed,
            max_turns=max_turns,
            max_steps=max_steps,
        )
        result[direction_name] = direction_result
        total_mismatches += int(direction_result["mismatch_count"])
        all_passed = all_passed and bool(direction_result["passed"])

    result["mismatch_count"] = total_mismatches
    result["passed"] = all_passed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Combat public-state trace parity harness")
    parser.add_argument("--pipe", action="store_true", help="Alias for --backend headless-pipe")
    parser.add_argument(
        "--backend",
        choices=["godot-http", "headless-pipe", "headless-binary"],
        default="headless-pipe",
    )
    parser.add_argument(
        "--backend-a",
        choices=["godot-http", "headless-pipe", "headless-binary"],
        default="headless-pipe",
    )
    parser.add_argument(
        "--backend-b",
        choices=["godot-http", "headless-pipe", "headless-binary"],
        default="headless-pipe",
    )
    parser.add_argument("--auto-launch", action="store_true", default=False)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--godot-exe", type=Path, default=DEFAULT_GODOT_EXE)
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_HEADLESS_DLL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=str, default=None)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=str, default="traces")
    parser.add_argument("--compare", action="store_true", help="Compare two backends")
    parser.add_argument(
        "--driver-mode",
        choices=["forward", "reverse", "bidirectional"],
        default="bidirectional",
        help="Which side drives the shared action sequence during comparison.",
    )
    parser.add_argument("--port-a", type=int, default=DEFAULT_PORT)
    parser.add_argument("--port-b", type=int, default=DEFAULT_PORT + 1)
    parser.add_argument("--report-json", type=str, default=None)
    args = parser.parse_args()

    seed_list = args.seeds or ([args.seed] if args.seed else ["TRACE_001"])
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = "headless-pipe" if args.pipe else args.backend

    if args.compare:
        with backend_client(
            backend=args.backend_a,
            port=args.port_a,
            auto_launch=args.auto_launch,
            repo_root=args.repo_root,
            godot_exe=args.godot_exe,
            headless_dll=args.headless_dll,
        ) as client_a, backend_client(
            backend=args.backend_b,
            port=args.port_b,
            auto_launch=args.auto_launch,
            repo_root=args.repo_root,
            godot_exe=args.godot_exe,
            headless_dll=args.headless_dll,
        ) as client_b:
            results = []
            for seed in seed_list:
                print(f"Comparing seed {seed}...")
                result = _compare_backends(
                    client_a,
                    client_b,
                    seed,
                    max_turns=args.max_turns,
                    max_steps=args.max_steps,
                    driver_mode=args.driver_mode,
                )
                result["backend_a"] = args.backend_a
                result["backend_b"] = args.backend_b
                results.append(result)
                status = "PASS" if result["passed"] else f"FAIL ({result['mismatch_count']} mismatches)"
                print(f"  {seed}: {status}")

        report_path = Path(args.report_json) if args.report_json else out_dir / "combat_trace_comparison.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "backend_a": args.backend_a,
            "backend_b": args.backend_b,
            "driver_mode": args.driver_mode,
            "seed_count": len(results),
            "mismatch_seed_count": sum(1 for item in results if not item["passed"]),
            "passed": all(item["passed"] for item in results),
            "results": results,
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        print(f"\nReport: {report_path}")

        total_pass = sum(1 for item in results if item["passed"])
        print(f"Result: {total_pass}/{len(results)} seeds passed")
        if total_pass != len(results):
            sys.exit(1)
        return

    with backend_client(
        backend=backend,
        port=args.port,
        auto_launch=args.auto_launch,
        repo_root=args.repo_root,
        godot_exe=args.godot_exe,
        headless_dll=args.headless_dll,
    ) as client:
        for seed in seed_list:
            print(f"Recording trace for seed {seed}...")
            entries = _run_combat_trace(client, seed, max_turns=args.max_turns, max_steps=args.max_steps)
            trace_path = out_dir / f"combat_trace_{seed}.jsonl"
            tracer = CombatTurnTracer()
            tracer._entries = entries
            tracer.flush(trace_path)
            print(f"  {len(entries)} steps recorded -> {trace_path}")


if __name__ == "__main__":
    main()
