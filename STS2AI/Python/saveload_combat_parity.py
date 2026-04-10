#!/usr/bin/env python3
"""Save/load combat parity checker — quantifies what's preserved and lost.

Tests save/load fidelity at various combat turn points:
  1. Save at turn start → load → compare (immediate roundtrip)
  2. Save mid-turn (after card plays) → load → compare
  3. Save → play actions → load → compare (rollback verification)
  4. Save → load → save → load → compare (idempotency)

Key output: whether hand order, hand labels, and legal actions are preserved.
This directly answers the MCTS blocker question.

Examples:
    python saveload_combat_parity.py --pipe --port 15527 --seeds 5
    python saveload_combat_parity.py --pipe --port 15527 --seeds 3 --output reports/saveload.json
"""
from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/{core,ipc,search} to sys.path)

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from full_run_env import FullRunClientLike
from sim_semantic_audit_common import (
    DEFAULT_GODOT_EXE,
    DEFAULT_HEADLESS_DLL,
    DEFAULT_PORT,
    DEFAULT_REPO_ROOT,
    backend_client,
)
from test_simulator_consistency import COMBAT_TYPES, normalize_legal_action, state_summary
from verify_save_load import (
    choose_default_action,
    combat_snapshot,
    diff_dict,
    drive_to_state,
)

# Seeds that reliably reach combat quickly
DEFAULT_SEEDS = [f"SL_PARITY_{i:03d}" for i in range(20)]


# ---------------------------------------------------------------------------
# Parity result
# ---------------------------------------------------------------------------

@dataclass
class ParityResult:
    test_case: str
    seed: str
    floor: int
    turn: int
    verdict: str  # "exact" | "resumable" | "diverged"
    matching_fields: list[str]
    mismatched_fields: list[str]
    hand_order_preserved: bool
    hand_labels_preserved: bool
    legal_actions_preserved: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _extract_hand_ordered(state: dict[str, Any]) -> list[tuple[str, int]]:
    """Extract hand as ordered list of (label, cost)."""
    battle = state.get("battle") or {}
    player = battle.get("player") or state.get("player") or {}
    hand = player.get("hand") or []
    return [(c.get("label") or c.get("id") or "?", c.get("cost") or 0) for c in hand]


def _extract_hand_labels_sorted(state: dict[str, Any]) -> list[str]:
    """Extract sorted hand labels (order-independent)."""
    return sorted(label for label, _ in _extract_hand_ordered(state))


def _extract_legal_normalized(state: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Extract normalized legal actions."""
    return [normalize_legal_action(a) for a in state.get("legal_actions") or []]


def _compare_states(
    before: dict[str, Any],
    after: dict[str, Any],
) -> tuple[str, list[str], list[str], dict[str, Any]]:
    """Compare two states, return (verdict, matching, mismatched, details)."""
    snap_before = combat_snapshot(before)
    snap_after = combat_snapshot(after)
    diffs = diff_dict(snap_before, snap_after)

    summary_before = state_summary(before)
    summary_after = state_summary(after)
    summary_diffs = diff_dict(summary_before, summary_after)

    # Classify fields
    all_fields = sorted(set(list(snap_before.keys()) + list(snap_after.keys())))
    matching = [k for k in all_fields if snap_before.get(k) == snap_after.get(k)]
    mismatched = [k for k in all_fields if snap_before.get(k) != snap_after.get(k)]

    details: dict[str, Any] = {}
    if diffs:
        details["snapshot_diffs"] = diffs[:10]
    if summary_diffs:
        details["summary_diffs"] = summary_diffs[:10]

    if not diffs and not summary_diffs:
        verdict = "exact"
    elif not summary_diffs:
        verdict = "exact"
    else:
        # Check if only non-critical fields differ
        critical_diffs = [d for d in summary_diffs if not any(
            skip in d for skip in ["hand_cards"]
        )]
        verdict = "resumable" if not critical_diffs else "diverged"

    return verdict, matching, mismatched, details


def _get_floor(state: dict[str, Any]) -> int:
    return (state.get("run") or {}).get("floor") or 0


def _get_turn(state: dict[str, Any]) -> int:
    battle = state.get("battle") or {}
    return battle.get("round_number") or battle.get("round") or 0


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_immediate_roundtrip(
    client: FullRunClientLike,
    state: dict[str, Any],
    seed: str,
) -> ParityResult:
    """Save at current state → load → compare. No mutations."""
    state_id = client.save_state()
    restored = client.load_state(state_id)
    client.delete_state(state_id)

    hand_before = _extract_hand_ordered(state)
    hand_after = _extract_hand_ordered(restored)
    labels_before = _extract_hand_labels_sorted(state)
    labels_after = _extract_hand_labels_sorted(restored)
    legal_before = _extract_legal_normalized(state)
    legal_after = _extract_legal_normalized(restored)

    verdict, matching, mismatched, details = _compare_states(state, restored)
    details["hand_before"] = hand_before
    details["hand_after"] = hand_after

    return ParityResult(
        test_case="immediate_roundtrip",
        seed=seed,
        floor=_get_floor(state),
        turn=_get_turn(state),
        verdict=verdict,
        matching_fields=matching,
        mismatched_fields=mismatched,
        hand_order_preserved=(hand_before == hand_after),
        hand_labels_preserved=(labels_before == labels_after),
        legal_actions_preserved=(legal_before == legal_after),
        details=details,
    )


def test_mid_turn(
    client: FullRunClientLike,
    state: dict[str, Any],
    seed: str,
    plays: int = 2,
) -> ParityResult:
    """Play N cards, then save → load → compare."""
    # Play some cards first
    current = state
    for _ in range(plays):
        legal = current.get("legal_actions") or []
        play_cards = [a for a in legal if a.get("action") == "play_card"]
        if not play_cards:
            break
        current = client.act(play_cards[0])
        st = str(current.get("state_type") or "")
        if st not in COMBAT_TYPES and st != "hand_select":
            break

    # Handle hand_select states that may appear
    for _ in range(5):
        if str(current.get("state_type") or "") == "hand_select":
            current = client.act(choose_default_action(current))
        else:
            break

    if str(current.get("state_type") or "") not in COMBAT_TYPES:
        return ParityResult(
            test_case="mid_turn",
            seed=seed,
            floor=_get_floor(current),
            turn=_get_turn(current),
            verdict="skipped",
            matching_fields=[],
            mismatched_fields=[],
            hand_order_preserved=False,
            hand_labels_preserved=False,
            legal_actions_preserved=False,
            details={"reason": f"combat ended after {plays} plays"},
        )

    state_id = client.save_state()
    restored = client.load_state(state_id)
    client.delete_state(state_id)

    hand_before = _extract_hand_ordered(current)
    hand_after = _extract_hand_ordered(restored)
    labels_before = _extract_hand_labels_sorted(current)
    labels_after = _extract_hand_labels_sorted(restored)
    legal_before = _extract_legal_normalized(current)
    legal_after = _extract_legal_normalized(restored)

    verdict, matching, mismatched, details = _compare_states(current, restored)
    details["plays_before_save"] = plays
    details["hand_before"] = hand_before
    details["hand_after"] = hand_after

    return ParityResult(
        test_case="mid_turn",
        seed=seed,
        floor=_get_floor(current),
        turn=_get_turn(current),
        verdict=verdict,
        matching_fields=matching,
        mismatched_fields=mismatched,
        hand_order_preserved=(hand_before == hand_after),
        hand_labels_preserved=(labels_before == labels_after),
        legal_actions_preserved=(legal_before == legal_after),
        details=details,
    )


def test_rollback(
    client: FullRunClientLike,
    state: dict[str, Any],
    seed: str,
    actions_between: int = 3,
) -> ParityResult:
    """Save → play N actions → load → compare with pre-save state."""
    state_id = client.save_state()
    snapshot_before = state

    # Mutate with several actions
    current = state
    actions_taken = 0
    for _ in range(actions_between):
        legal = current.get("legal_actions") or []
        if not legal:
            break
        action = choose_default_action(current)
        current = client.act(action)
        actions_taken += 1
        st = str(current.get("state_type") or "")
        if st not in COMBAT_TYPES and st != "hand_select":
            break

    # Rollback
    restored = client.load_state(state_id)
    client.delete_state(state_id)

    hand_before = _extract_hand_ordered(snapshot_before)
    hand_after = _extract_hand_ordered(restored)
    labels_before = _extract_hand_labels_sorted(snapshot_before)
    labels_after = _extract_hand_labels_sorted(restored)
    legal_before = _extract_legal_normalized(snapshot_before)
    legal_after = _extract_legal_normalized(restored)

    verdict, matching, mismatched, details = _compare_states(snapshot_before, restored)
    details["actions_between"] = actions_taken
    details["hand_before"] = hand_before
    details["hand_after"] = hand_after

    return ParityResult(
        test_case="rollback",
        seed=seed,
        floor=_get_floor(snapshot_before),
        turn=_get_turn(snapshot_before),
        verdict=verdict,
        matching_fields=matching,
        mismatched_fields=mismatched,
        hand_order_preserved=(hand_before == hand_after),
        hand_labels_preserved=(labels_before == labels_after),
        legal_actions_preserved=(legal_before == legal_after),
        details=details,
    )


def test_idempotency(
    client: FullRunClientLike,
    state: dict[str, Any],
    seed: str,
) -> ParityResult:
    """Save → load → save → load → compare first and second load."""
    state_id_1 = client.save_state()
    restored_1 = client.load_state(state_id_1)

    state_id_2 = client.save_state()
    restored_2 = client.load_state(state_id_2)

    client.delete_state(state_id_1)
    client.delete_state(state_id_2)

    hand_1 = _extract_hand_ordered(restored_1)
    hand_2 = _extract_hand_ordered(restored_2)
    labels_1 = _extract_hand_labels_sorted(restored_1)
    labels_2 = _extract_hand_labels_sorted(restored_2)
    legal_1 = _extract_legal_normalized(restored_1)
    legal_2 = _extract_legal_normalized(restored_2)

    verdict, matching, mismatched, details = _compare_states(restored_1, restored_2)
    details["hand_load_1"] = hand_1
    details["hand_load_2"] = hand_2

    return ParityResult(
        test_case="idempotency",
        seed=seed,
        floor=_get_floor(restored_1),
        turn=_get_turn(restored_1),
        verdict=verdict,
        matching_fields=matching,
        mismatched_fields=mismatched,
        hand_order_preserved=(hand_1 == hand_2),
        hand_labels_preserved=(labels_1 == labels_2),
        legal_actions_preserved=(legal_1 == legal_2),
        details=details,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all_tests(
    client: FullRunClientLike,
    seed: str,
) -> list[ParityResult]:
    """Run all four test cases on one seed. Returns list of results."""
    results: list[ParityResult] = []

    # Drive to first combat
    try:
        state = drive_to_state(client, seed, COMBAT_TYPES)
    except RuntimeError as exc:
        results.append(ParityResult(
            test_case="all",
            seed=seed,
            floor=0,
            turn=0,
            verdict="error",
            matching_fields=[],
            mismatched_fields=[],
            hand_order_preserved=False,
            hand_labels_preserved=False,
            legal_actions_preserved=False,
            details={"error": str(exc)},
        ))
        return results

    # Test 1: Immediate roundtrip
    results.append(test_immediate_roundtrip(client, state, seed))

    # Re-drive to combat for test 2 (state was consumed by test 1's load)
    try:
        state = drive_to_state(client, seed, COMBAT_TYPES)
    except RuntimeError:
        return results

    # Test 2: Mid-turn save/load
    results.append(test_mid_turn(client, state, seed))

    # Re-drive for test 3
    try:
        state = drive_to_state(client, seed, COMBAT_TYPES)
    except RuntimeError:
        return results

    # Test 3: Rollback
    results.append(test_rollback(client, state, seed))

    # Re-drive for test 4
    try:
        state = drive_to_state(client, seed, COMBAT_TYPES)
    except RuntimeError:
        return results

    # Test 4: Idempotency
    results.append(test_idempotency(client, state, seed))

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(all_results: list[ParityResult]) -> dict[str, Any]:
    """Build summary report from all test results."""
    total = len(all_results)
    exact = sum(1 for r in all_results if r.verdict == "exact")
    resumable = sum(1 for r in all_results if r.verdict == "resumable")
    diverged = sum(1 for r in all_results if r.verdict == "diverged")
    skipped = sum(1 for r in all_results if r.verdict == "skipped")
    errors = sum(1 for r in all_results if r.verdict == "error")

    hand_order_ok = sum(1 for r in all_results if r.hand_order_preserved and r.verdict not in ("skipped", "error"))
    hand_labels_ok = sum(1 for r in all_results if r.hand_labels_preserved and r.verdict not in ("skipped", "error"))
    legal_ok = sum(1 for r in all_results if r.legal_actions_preserved and r.verdict not in ("skipped", "error"))
    testable = total - skipped - errors

    # Group by test case
    by_case: dict[str, list[ParityResult]] = {}
    for r in all_results:
        by_case.setdefault(r.test_case, []).append(r)

    case_summaries = {}
    for case, results in by_case.items():
        case_summaries[case] = {
            "total": len(results),
            "exact": sum(1 for r in results if r.verdict == "exact"),
            "resumable": sum(1 for r in results if r.verdict == "resumable"),
            "diverged": sum(1 for r in results if r.verdict == "diverged"),
            "hand_order_preserved_rate": (
                sum(1 for r in results if r.hand_order_preserved and r.verdict not in ("skipped", "error"))
                / max(1, sum(1 for r in results if r.verdict not in ("skipped", "error")))
            ),
        }

    return {
        "summary": {
            "total_tests": total,
            "exact": exact,
            "resumable": resumable,
            "diverged": diverged,
            "skipped": skipped,
            "errors": errors,
            "hand_order_preserved_rate": hand_order_ok / max(1, testable),
            "hand_labels_preserved_rate": hand_labels_ok / max(1, testable),
            "legal_actions_preserved_rate": legal_ok / max(1, testable),
            "mcts_feasibility": "GOOD" if hand_labels_ok == testable and legal_ok == testable else "NEEDS_WORK",
        },
        "by_test_case": case_summaries,
        "results": [r.to_dict() for r in all_results],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Save/load combat parity checker")
    parser.add_argument(
        "--backend",
        choices=["godot-http", "headless-pipe", "headless-binary"],
        default="headless-pipe",
    )
    parser.add_argument("--pipe", action="store_true", help="Alias for --backend headless-pipe")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to test")
    parser.add_argument("--seed-list", nargs="+", type=str, default=None)
    parser.add_argument("--auto-launch", action="store_true", default=False)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--godot-exe", type=Path, default=DEFAULT_GODOT_EXE)
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_HEADLESS_DLL)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    backend = "headless-pipe" if args.pipe else args.backend
    seeds = args.seed_list or DEFAULT_SEEDS[: args.seeds]

    all_results: list[ParityResult] = []
    with backend_client(
        backend=backend,
        port=args.port,
        auto_launch=args.auto_launch,
        repo_root=args.repo_root,
        godot_exe=args.godot_exe,
        headless_dll=args.headless_dll,
    ) as client:
        for seed in seeds:
            print(f"\n--- Seed: {seed} ---")
            results = run_all_tests(client, seed)
            for r in results:
                status = r.verdict.upper()
                hand_info = f"hand_order={'YES' if r.hand_order_preserved else 'NO'} labels={'YES' if r.hand_labels_preserved else 'NO'}"
                legal_info = f"legal={'YES' if r.legal_actions_preserved else 'NO'}"
                print(f"  {r.test_case:25s} {status:10s} {hand_info}  {legal_info}")
            all_results.extend(results)

    report = build_report(all_results)
    report["backend"] = backend
    report["port"] = args.port
    report["seed_list"] = seeds

    # Print summary
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"SAVE/LOAD COMBAT PARITY REPORT")
    print(f"{'='*60}")
    print(f"Total tests:          {s['total_tests']}")
    print(f"Exact:                {s['exact']}")
    print(f"Resumable:            {s['resumable']}")
    print(f"Diverged:             {s['diverged']}")
    print(f"Hand order preserved: {s['hand_order_preserved_rate']:.0%}")
    print(f"Hand labels preserved:{s['hand_labels_preserved_rate']:.0%}")
    print(f"Legal actions match:  {s['legal_actions_preserved_rate']:.0%}")
    print(f"MCTS feasibility:     {s['mcts_feasibility']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
