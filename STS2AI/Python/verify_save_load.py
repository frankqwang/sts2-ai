#!/usr/bin/env python3
"""Verify full-run save/load behavior across HTTP and named-pipe backends.

Examples:
    python verify_save_load.py --backend godot-http --auto-launch
    python verify_save_load.py --backend headless-pipe --auto-launch
"""
from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/{core,ipc,search} to sys.path)

import argparse
import json
import subprocess
import sys
import time
import random
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from full_run_env import ApiBackedFullRunClient, BinaryBackedFullRunClient, FullRunClientLike, PipeBackedFullRunClient
from headless_sim_runner import DEFAULT_DLL_PATH, start_headless_sim, stop_process
from sts2ai_paths import ARTIFACTS_ROOT, REPO_ROOT
from test_simulator_consistency import pick_deterministic_action, state_summary


COMBAT_TYPES = {"monster", "elite", "boss"}
REWARD_SAVELOAD_SEEDS = ("CARD_SCAN_24",)
CARD_REWARD_SAVELOAD_SEEDS = ("CARD_SCAN_24", "CARD_SCAN_169")
EXACT_STATE_CASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("map", "map", ("SHOP_EXIT_000",)),
    ("event", "event", ("SHOP_EXIT_000", "SHOP_EXIT_001", "SHOP_EXIT_002")),
    ("rest_site", "rest_site", ("SHOP_EXIT_000", "SHOP_EXIT_002", "SHOP_EXIT_003")),
    ("shop", "shop", ("SHOP_EXIT_005", "SHOP_EXIT_010", "SHOP_EXIT_011")),
    ("treasure", "treasure", ("SHOP_EXIT_000", "SHOP_EXIT_009", "SHOP_EXIT_010")),
    ("game_over", "game_over", ("SHOP_EXIT_000", "SHOP_EXIT_001", "SHOP_EXIT_002")),
)
DEFAULT_REPO_ROOT = REPO_ROOT
DEFAULT_GODOT_EXE = Path(r"C:/dev/game/Godot_v4.5.1-stable_mono_win64/Godot_v4.5.1-stable_mono_win64_console.exe")
DEFAULT_HEADLESS_DLL = DEFAULT_DLL_PATH
DEFAULT_REPORT_JSON = ARTIFACTS_ROOT / "verify_save_load" / "latest_report.json"


def create_client(backend: str, port: int) -> FullRunClientLike:
    if backend == "godot-http":
        return ApiBackedFullRunClient(
            base_url=f"http://127.0.0.1:{port}",
            request_timeout_s=30.0,
            ready_timeout_s=30.0,
        )
    if backend == "headless-binary":
        return BinaryBackedFullRunClient(port=port, connect_timeout_s=15.0)
    return PipeBackedFullRunClient(port=port, connect_timeout_s=15.0)


def launch_backend(
    *,
    backend: str,
    port: int,
    repo_root: Path,
    godot_exe: Path,
    headless_dll: Path,
) -> subprocess.Popen | None:
    if backend == "headless-pipe":
        return start_headless_sim(port=port, repo_root=repo_root, dll_path=headless_dll)
    if backend == "headless-binary":
        return start_headless_sim(port=port, repo_root=repo_root, dll_path=headless_dll, protocol="bin")
    if backend in {"godot-http", "godot-pipe"}:
        proc = subprocess.Popen(
            [
                str(godot_exe),
                "--headless",
                "--fixed-fps",
                "1000",
                "--mcp-port",
                str(port),
                "--full-run-sim-server",
                "--path",
                str(repo_root),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        time.sleep(12)
        return proc
    return None


def summarize_state(state: dict[str, Any]) -> str:
    run = state.get("run") or {}
    return f"{state.get('state_type')} floor={run.get('floor')} act={run.get('act')} legal={len(state.get('legal_actions') or [])}"


def combat_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    battle = state.get("battle") or {}
    player = battle.get("player") or {}
    enemies = []
    for enemy in battle.get("enemies") or []:
        enemies.append(
            {
                "name": enemy.get("name"),
                "hp": enemy.get("hp"),
                "max_hp": enemy.get("max_hp"),
                "block": enemy.get("block"),
                "intents": [intent.get("type") for intent in enemy.get("intents") or []],
            }
        )
    return {
        "state_type": state.get("state_type"),
        "run": (state.get("run") or {}).copy(),
        "player": {
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "energy": player.get("energy"),
            "draw": player.get("draw_pile_count"),
            "discard": player.get("discard_pile_count"),
            "exhaust": player.get("exhaust_pile_count"),
        },
        "hand": [(card.get("id"), card.get("label"), card.get("cost")) for card in player.get("hand") or []],
        "enemies": enemies,
    }


def reward_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    reward_data = state.get("rewards") or {}
    card_reward = state.get("card_reward") or {}
    return {
        "state_type": state.get("state_type"),
        "reward_actions": [action.get("action") for action in state.get("legal_actions") or []],
        "reward_items": [(item.get("type"), item.get("label")) for item in reward_data.get("items") or []],
        "card_items": [(item.get("id"), item.get("name")) for item in card_reward.get("cards") or []],
    }


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    return value


def exact_legal_actions(state: dict[str, Any]) -> list[tuple[Any, ...]]:
    actions = []
    for action in state.get("legal_actions") or []:
        actions.append(
            (
                action.get("action"),
                action.get("index"),
                action.get("card_index"),
                action.get("target_id"),
                action.get("col"),
                action.get("row"),
                action.get("slot"),
            )
        )
    return actions


def stable_player_summary(state: dict[str, Any]) -> dict[str, Any]:
    player = {}
    summary_player = state_summary(state).get("player") or {}
    extracted_player = {}
    try:
        from test_simulator_consistency import extract_player

        extracted_player = extract_player(state) or {}
    except Exception:
        extracted_player = {}
    player["hp"] = summary_player.get("hp")
    player["max_hp"] = summary_player.get("max_hp")
    player["block"] = summary_player.get("block")
    player["gold"] = extracted_player.get("gold")
    return player


def exact_state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    state_type = str(state.get("state_type") or "")
    run = state.get("run") or {}
    snapshot: dict[str, Any] = {
        "state_type": state_type,
        "terminal": bool(state.get("terminal", False)),
        "run_outcome": state.get("run_outcome"),
        "floor": run.get("floor"),
        "act": run.get("act"),
        "legal_actions": exact_legal_actions(state),
        "player": stable_player_summary(state),
    }

    if state_type == "map":
        map_state = state.get("map") or {}
        snapshot["map"] = [
            (
                option.get("index"),
                option.get("col"),
                option.get("row"),
                option.get("point_type"),
            )
            for option in map_state.get("next_options") or []
        ]
    elif state_type == "event":
        event_state = state.get("event") or {}
        snapshot["event"] = {
            "event_id": event_state.get("event_id"),
            "in_dialogue": event_state.get("in_dialogue"),
            "is_finished": event_state.get("is_finished"),
            "options": [
                (
                    option.get("index"),
                    option.get("text"),
                    option.get("is_locked"),
                    option.get("is_chosen"),
                    option.get("is_proceed"),
                )
                for option in event_state.get("options") or []
            ],
        }
    elif state_type == "rest_site":
        rest_state = state.get("rest_site") or {}
        snapshot["rest_site"] = {
            "can_proceed": rest_state.get("can_proceed"),
            "options": [
                (
                    option.get("index"),
                    option.get("id"),
                    option.get("name"),
                    option.get("is_enabled"),
                )
                for option in rest_state.get("options") or []
            ],
        }
    elif state_type == "shop":
        shop_state = state.get("shop") or {}
        snapshot["shop"] = {
            "is_open": shop_state.get("is_open"),
            "can_proceed": shop_state.get("can_proceed"),
            "items": [
                (
                    item.get("index"),
                    item.get("category"),
                    item.get("cost"),
                    item.get("can_afford"),
                    item.get("is_stocked"),
                    item.get("on_sale"),
                    item.get("card_id"),
                    item.get("relic_id"),
                    item.get("potion_id"),
                    item.get("name"),
                )
                for item in shop_state.get("items") or []
            ],
        }
    elif state_type == "treasure":
        treasure_state = state.get("treasure") or {}
        snapshot["treasure"] = {
            "can_proceed": treasure_state.get("can_proceed"),
            "relics": [
                (
                    relic.get("index"),
                    relic.get("id"),
                    relic.get("name"),
                    relic.get("rarity"),
                )
                for relic in treasure_state.get("relics") or []
            ],
        }
    elif state_type == "game_over":
        snapshot["game_over"] = {
            "available_actions": exact_legal_actions({"legal_actions": (state.get("game_over") or {}).get("available_actions") or []})
        }
    return snapshot


def diff_dict(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    if before == after:
        return []
    diffs: list[str] = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            diffs.append(f"{key}: {before.get(key)!r} -> {after.get(key)!r}")
    return diffs


def choose_default_action(state: dict[str, Any]) -> dict[str, Any]:
    legal = state.get("legal_actions") or []
    state_type = state.get("state_type")
    if not legal:
        return {"action": "wait"}
    if state_type == "map":
        return next((action for action in legal if action.get("action") == "choose_map_node"), legal[0])
    if state_type in COMBAT_TYPES:
        return next((action for action in legal if action.get("action") == "play_card"), None) or next(
            (action for action in legal if action.get("action") == "end_turn"),
            {"action": "wait"},
        )
    if state_type == "hand_select":
        return next((action for action in legal if action.get("action") == "combat_confirm_selection"), None) or next(
            (action for action in legal if action.get("action") == "combat_select_card"),
            legal[0],
        )
    if state_type == "card_select":
        return next((action for action in legal if action.get("action") == "confirm_selection"), None) or next(
            (action for action in legal if action.get("action") == "select_card"),
            None,
        ) or next(
            (action for action in legal if action.get("action") == "combat_confirm_selection"),
            legal[0],
        )
    if state_type == "combat_pending":
        return {"action": "wait"}
    if state_type == "combat_rewards":
        return next((action for action in legal if action.get("action") == "claim_reward"), None) or next(
            (action for action in legal if action.get("action") == "proceed"),
            {"action": "wait"},
        )
    if state_type == "card_reward":
        return next((action for action in legal if action.get("action") == "select_card_reward"), None) or next(
            (action for action in legal if action.get("action") == "skip_card_reward"),
            {"action": "wait"},
        )
    if state_type == "relic_select":
        return next((action for action in legal if action.get("action") == "select_relic"), None) or next(
            (action for action in legal if action.get("action") == "skip_relic_selection"),
            legal[0],
        )
    if state_type == "treasure":
        return next((action for action in legal if action.get("action") == "claim_treasure_relic"), None) or next(
            (action for action in legal if action.get("action") == "proceed"),
            legal[0],
        )
    if state_type == "shop":
        return next((action for action in legal if action.get("action") == "proceed"), legal[0])
    return legal[0]


def drive_to_state(client: FullRunClientLike, seed: str, targets: set[str], max_steps: int = 400) -> dict[str, Any]:
    rng = random.Random(42)
    state = client.reset(character_id="IRONCLAD", ascension_level=0, seed=seed)
    for _ in range(max_steps):
        if state.get("state_type") in targets:
            return state
        state = client.act(choose_target_action(state, targets, rng))
    raise RuntimeError(f"Failed to reach {sorted(targets)} within {max_steps} steps; last={summarize_state(state)}")


def report(title: str, details: list[str]) -> None:
    print(f"\n=== {title} ===")
    for detail in details:
        print(detail)


def choose_target_action(state: dict[str, Any], targets: set[str], rng: random.Random) -> dict[str, Any]:
    legal = state.get("legal_actions") or []
    state_type = state.get("state_type")
    if not legal:
        return {"action": "wait"}

    if "card_reward" in targets and state_type == "combat_rewards":
        rewards = state.get("rewards") or {}
        for item in rewards.get("items") or []:
            if (item.get("type") or "").lower() == "card":
                target_index = item.get("index")
                action = next(
                    (
                        entry
                        for entry in legal
                        if entry.get("action") == "claim_reward" and entry.get("index") == target_index
                    ),
                    None,
                )
                if action is not None:
                    return action

    return choose_default_action(state) if state_type == "hand_select" else pick_deterministic_action(str(state_type or ""), legal, rng)


def verify_combat_save_load(client: FullRunClientLike) -> tuple[str, list[str]]:
    state = drive_to_state(client, "VERIFY_COMBAT", COMBAT_TYPES)
    before = combat_snapshot(state)
    state_id = client.save_state()
    mutate = choose_default_action(state)
    mutated = client.act(mutate)
    restored = client.load_state(state_id)
    after = combat_snapshot(restored)
    deleted = client.delete_state(state_id)
    diffs = diff_dict(before, after)
    lines = [
        f"saved:   {summarize_state(state)}",
        f"mutated: {summarize_state(mutated)}",
        f"loaded:  {summarize_state(restored)}",
        f"cache delete: {deleted}",
    ]
    if diffs:
        lines.append("classification: resumable")
        lines.append("exact snapshot mismatch:")
        lines.extend(f"  {entry}" for entry in diffs[:10])
        return "resumable", lines
    lines.append("classification: exact")
    lines.append("exact snapshot restored.")
    return "exact", lines


def verify_reward_save_load(client: FullRunClientLike) -> tuple[str, list[str]]:
    lines: list[str] = []
    all_exact = True
    for seed in REWARD_SAVELOAD_SEEDS:
        state = drive_to_state(client, seed, {"combat_rewards"})
        before = reward_snapshot(state)
        state_id = client.save_state()
        mutated = client.act(choose_default_action(state))
        restored = client.load_state(state_id)
        after = reward_snapshot(restored)
        deleted = client.delete_state(state_id)
        diffs = diff_dict(before, after)
        lines.extend(
            [
                f"seed {seed}:",
                f"  saved:   {summarize_state(state)}",
                f"  mutated: {summarize_state(mutated)}",
                f"  loaded:  {summarize_state(restored)}",
                f"  cache delete: {deleted}",
            ]
        )
        if diffs:
            all_exact = False
            lines.append("  classification: unsupported")
            lines.append("  reward snapshot mismatch:")
            lines.extend(f"    {entry}" for entry in diffs[:10])
        else:
            lines.append("  classification: exact")
            lines.append("  reward snapshot restored.")
    return ("exact" if all_exact else "unsupported"), lines


def verify_card_reward_save_load(client: FullRunClientLike) -> tuple[str, list[str]]:
    lines: list[str] = []
    all_exact = True
    for seed in CARD_REWARD_SAVELOAD_SEEDS:
        state = drive_to_state(client, seed, {"card_reward"})
        before = reward_snapshot(state)
        state_id = client.save_state()
        mutated = client.act(choose_default_action(state))
        restored = client.load_state(state_id)
        after = reward_snapshot(restored)
        deleted = client.delete_state(state_id)
        diffs = diff_dict(before, after)
        lines.extend(
            [
                f"seed {seed}:",
                f"  saved:   {summarize_state(state)}",
                f"  mutated: {summarize_state(mutated)}",
                f"  loaded:  {summarize_state(restored)}",
                f"  cache delete: {deleted}",
            ]
        )
        if diffs:
            all_exact = False
            lines.append("  classification: unsupported")
            lines.append("  card reward snapshot mismatch:")
            lines.extend(f"    {entry}" for entry in diffs[:10])
        else:
            lines.append("  classification: exact")
            lines.append("  card reward snapshot restored.")
    return ("exact" if all_exact else "unsupported"), lines


def verify_exact_state_save_load(
    client: FullRunClientLike,
    *,
    title: str,
    target_state: str,
    seeds: tuple[str, ...],
) -> tuple[str, list[str]]:
    lines: list[str] = []
    all_exact = True
    for seed in seeds:
        state = drive_to_state(client, seed, {target_state})
        before = exact_state_snapshot(state)
        state_id = client.save_state()
        legal = state.get("legal_actions") or []
        if state.get("terminal") or not legal:
            mutated = client.get_state()
        else:
            mutated = client.act(choose_default_action(state))
        restored = client.load_state(state_id)
        after = exact_state_snapshot(restored)
        deleted = client.delete_state(state_id)
        diffs = diff_dict(before, after)
        lines.extend(
            [
                f"seed {seed}:",
                f"  saved:   {summarize_state(state)}",
                f"  mutated: {summarize_state(mutated)}",
                f"  loaded:  {summarize_state(restored)}",
                f"  cache delete: {deleted}",
            ]
        )
        if diffs:
            all_exact = False
            lines.append("  classification: unsupported")
            lines.append(f"  {title} snapshot mismatch:")
            lines.extend(f"    {entry}" for entry in diffs[:10])
        else:
            lines.append("  classification: exact")
            lines.append(f"  {title} snapshot restored.")
    return ("exact" if all_exact else "unsupported"), lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify save/load on Godot or standalone full-run simulators.")
    parser.add_argument("--backend", choices=["godot-http", "godot-pipe", "headless-pipe", "headless-binary"], default="headless-pipe")
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--godot-exe", type=Path, default=DEFAULT_GODOT_EXE)
    parser.add_argument("--headless-dll", type=Path, default=DEFAULT_HEADLESS_DLL)
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    args = parser.parse_args()

    proc: subprocess.Popen | None = None
    if args.auto_launch:
        proc = launch_backend(
            backend=args.backend,
            port=args.port,
            repo_root=args.repo_root,
            godot_exe=args.godot_exe,
            headless_dll=args.headless_dll,
        )

    client = create_client(args.backend, args.port)
    hard_failures = 0
    classifications = {"exact": 0, "resumable": 0, "unsupported": 0}
    report_payload: dict[str, Any] = {
        "backend": args.backend,
        "checks": [],
    }
    try:
        checks = [
            ("combat save/load", verify_combat_save_load),
            ("reward save/load", verify_reward_save_load),
            ("card reward save/load", verify_card_reward_save_load),
        ]
        for state_name, target_state, seeds in EXACT_STATE_CASES:
            checks.append(
                (
                    f"{state_name} save/load",
                    lambda client, state_name=state_name, target_state=target_state, seeds=seeds: verify_exact_state_save_load(
                        client,
                        title=state_name,
                        target_state=target_state,
                        seeds=seeds,
                    ),
                )
            )
        for title, fn in checks:
            try:
                classification, lines = fn(client)
            except Exception as exc:
                classification = "unsupported"
                lines = [f"runtime/protocol failure: {exc}"]
                classifications[classification] += 1
                if not isinstance(exc, RuntimeError):
                    hard_failures += 1
            else:
                classifications[classification] += 1
                if title == "combat save/load":
                    if classification == "unsupported":
                        hard_failures += 1
                elif classification != "exact":
                    hard_failures += 1
            report_payload["checks"].append(
                {
                    "title": title,
                    "classification": classification,
                    "lines": lines,
                }
            )
            report(title, lines)

        print("\n=== SUMMARY ===")
        print(f"backend: {args.backend}")
        print(f"hard_failures: {hard_failures}")
        print(f"exact: {classifications['exact']}")
        print(f"resumable: {classifications['resumable']}")
        print(f"unsupported: {classifications['unsupported']}")
        report_payload["summary"] = {
            "hard_failures": hard_failures,
            "exact": classifications["exact"],
            "resumable": classifications["resumable"],
            "unsupported": classifications["unsupported"],
        }
        report_payload["passed"] = hard_failures == 0
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1 if hard_failures else 0
    finally:
        try:
            client.clear_state_cache()
        except Exception:
            pass
        client.close()
        stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
