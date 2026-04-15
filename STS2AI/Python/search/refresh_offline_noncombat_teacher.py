#!/usr/bin/env python3
"""Run a windowed offline non-combat teacher refresh.

Workflow:
1. Read a queue JSON produced from replay summaries.
2. Run route-search labeling on the queued seeds, or reuse an existing route run dir.
3. Re-run the same seeds with the same checkpoint in shadow-baseline mode.
4. Keep only seeds whose route run beats the same-checkpoint shadow baseline.
5. Materialize an accepted offline_noncombat_ranking dataset for the next training window.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    python_root = Path(__file__).resolve().parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

import _path_init  # noqa: F401

import argparse
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from data.derived.build_rl_views import build_ranking_view
from data.raw.raw_dataset_writer import write_raw_branch_exports


def _utc_now() -> str:
    return datetime.now(UTC).replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _load_json(path_like: str | Path) -> Any:
    return json.loads(Path(path_like).read_text(encoding="utf-8-sig"))


def _load_jsonl(path_like: str | Path) -> list[dict[str, Any]]:
    path = Path(path_like)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _replace_config_assignment(config_text: str, key: str, value_literal: str) -> tuple[str, bool]:
    lines = config_text.splitlines()
    replaced = False
    updated_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key} ") and "=" in stripped:
            indent = line[: len(line) - len(line.lstrip())]
            updated_lines.append(f"{indent}{key} = {value_literal}")
            replaced = True
        else:
            updated_lines.append(line)
    suffix = "\n" if config_text.endswith("\n") or not updated_lines else ""
    return "\n".join(updated_lines) + suffix, replaced


def _write_shadow_generator_config(*, source_config: Path, output_path: Path) -> Path:
    config_text = source_config.read_text(encoding="utf-8")
    config_text, replaced_route_search = _replace_config_assignment(
        config_text,
        "tree_route_search",
        "false",
    )
    config_text, replaced_depth = _replace_config_assignment(
        config_text,
        "tree_max_reward_depth",
        "1",
    )
    if not replaced_route_search:
        config_text = config_text.rstrip() + ("\n\n" if config_text.strip() else "") + "tree_route_search = false\n"
    if not replaced_depth:
        config_text = config_text.rstrip() + ("\n" if config_text.strip() else "") + "tree_max_reward_depth = 1\n"
    output_path.write_text(config_text, encoding="utf-8")
    return output_path


def _build_generator_command(
    *,
    queue_seed_file: Path,
    generated_dir: Path,
    generator_config: Path,
    checkpoint: str | None,
    combat_checkpoint: str | None,
    num_envs: int,
    start_port: int,
    transport: str,
    auto_launch: bool,
    headless_dll: str | None,
) -> list[str]:
    script_path = Path(__file__).with_name("generate_offline_noncombat_ranking_data.py")
    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(generator_config),
        "--output",
        str(generated_dir),
        "--seed-file",
        str(queue_seed_file),
        "--num-envs",
        str(num_envs),
        "--start-port",
        str(start_port),
        "--transport",
        str(transport),
    ]
    if checkpoint:
        command.extend(["--checkpoint", str(checkpoint)])
    if combat_checkpoint:
        command.extend(["--combat-checkpoint", str(combat_checkpoint)])
    if auto_launch:
        command.append("--auto-launch")
    if headless_dll:
        command.extend(["--headless-dll", str(headless_dll)])
    return command


def _seed_beats_shadow_baseline(
    *,
    shadow_baseline: dict[str, Any],
    route: dict[str, Any],
    min_floor_gain: int,
) -> tuple[bool, str]:
    baseline_clear = bool(shadow_baseline.get("act1_cleared"))
    route_clear = bool(route.get("act1_cleared"))
    baseline_boss = bool(shadow_baseline.get("boss_reached"))
    route_boss = bool(route.get("boss_reached"))
    baseline_floor = int(shadow_baseline.get("end_floor", 0) or 0)
    route_floor = int(route.get("end_floor", 0) or 0)
    if route_clear and not baseline_clear:
        return True, "act1_clear_upgrade"
    if route_boss and not baseline_boss:
        return True, "boss_reach_upgrade"
    if route_floor >= baseline_floor + int(min_floor_gain):
        return True, f"floor_gain_{route_floor - baseline_floor}"
    return False, "no_improvement"


def _load_episode_logs(run_dir: Path) -> list[dict[str, Any]]:
    episode_logs_path = run_dir / "episode_logs.json"
    if not episode_logs_path.exists():
        raise FileNotFoundError(f"Missing generator episode logs: {episode_logs_path}")
    return _load_json(episode_logs_path)


def _index_logs_by_seed(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("seed") or "").strip(): entry
        for entry in entries
        if str(entry.get("seed") or "").strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh offline non-combat teacher data from a training window queue.")
    parser.add_argument("--queue", required=True, help="Queue JSON produced by build_offline_noncombat_teacher_queue.py")
    parser.add_argument("--output-dir", required=True, help="Teacher refresh output root")
    parser.add_argument("--generated-run-dir", default="", help="Optional existing route generator output dir; if omitted, launch route generation")
    parser.add_argument("--shadow-generated-run-dir", default="", help="Optional existing same-checkpoint shadow-baseline run dir; if omitted, launch shadow generation")
    parser.add_argument("--generator-config", default=str(Path(__file__).parents[1] / "configs" / "offline_noncombat_teacher_route_default.toml"))
    parser.add_argument("--checkpoint", default="", help="Optional hybrid checkpoint passed to generator")
    parser.add_argument("--combat-checkpoint", default="", help="Optional combat checkpoint override")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--start-port", type=int, default=16720)
    parser.add_argument("--transport", choices=["http", "pipe", "pipe-binary"], default="pipe-binary")
    parser.add_argument("--auto-launch", action="store_true", default=False)
    parser.add_argument("--headless-dll", default="")
    parser.add_argument("--min-floor-gain", type=int, default=2)
    parser.add_argument("--boss-damage-margin", type=float, default=0.10, help="Deprecated: retained for CLI compatibility but ignored by shadow-baseline gating")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_payload = _load_json(args.queue)
    queue_entries = list(queue_payload.get("entries") or [])
    if not queue_entries:
        raise ValueError("Teacher queue is empty.")
    queue_seed_file = output_dir / "queued_seeds.txt"
    queue_seed_file.write_text(
        "\n".join(str(entry.get("seed") or "").strip() for entry in queue_entries if str(entry.get("seed") or "").strip()) + "\n",
        encoding="utf-8",
    )

    generator_config_path = Path(args.generator_config)
    generated_dir = Path(args.generated_run_dir) if args.generated_run_dir else (output_dir / "generated")
    shadow_generated_dir = (
        Path(args.shadow_generated_run_dir)
        if args.shadow_generated_run_dir else
        (output_dir / "generated_shadow_baseline")
    )
    if not args.generated_run_dir:
        command = _build_generator_command(
            queue_seed_file=queue_seed_file,
            generated_dir=generated_dir,
            generator_config=generator_config_path,
            checkpoint=(args.checkpoint or None),
            combat_checkpoint=(args.combat_checkpoint or None),
            num_envs=int(args.num_envs),
            start_port=int(args.start_port),
            transport=str(args.transport),
            auto_launch=bool(args.auto_launch),
            headless_dll=(args.headless_dll or None),
        )
        (output_dir / "generator_command.txt").write_text(" ".join(command), encoding="utf-8")
        (output_dir / "route_generator_command.txt").write_text(" ".join(command), encoding="utf-8")
        subprocess.run(command, cwd=Path(__file__).resolve().parents[3], check=True)

    shadow_config_path = output_dir / "shadow_baseline_config.toml"
    if not args.shadow_generated_run_dir:
        _write_shadow_generator_config(
            source_config=generator_config_path,
            output_path=shadow_config_path,
        )
        shadow_command = _build_generator_command(
            queue_seed_file=queue_seed_file,
            generated_dir=shadow_generated_dir,
            generator_config=shadow_config_path,
            checkpoint=(args.checkpoint or None),
            combat_checkpoint=(args.combat_checkpoint or None),
            num_envs=int(args.num_envs),
            start_port=int(args.start_port),
            transport=str(args.transport),
            auto_launch=bool(args.auto_launch),
            headless_dll=(args.headless_dll or None),
        )
        (output_dir / "shadow_baseline_generator_command.txt").write_text(" ".join(shadow_command), encoding="utf-8")
        subprocess.run(shadow_command, cwd=Path(__file__).resolve().parents[3], check=True)

    raw_branch_path = generated_dir / "raw" / "raw_branch_rollout.jsonl"
    if not raw_branch_path.exists():
        raise FileNotFoundError(f"Missing generator raw branch records: {raw_branch_path}")

    route_logs = _load_episode_logs(generated_dir)
    shadow_logs = _load_episode_logs(shadow_generated_dir)
    raw_branch_records = _load_jsonl(raw_branch_path)
    route_by_seed = _index_logs_by_seed(route_logs)
    shadow_by_seed = _index_logs_by_seed(shadow_logs)
    queue_by_seed = _index_logs_by_seed(queue_entries)

    comparisons: list[dict[str, Any]] = []
    accepted_seeds: list[str] = []
    rejection_counts: Counter[str] = Counter()
    for seed, queue_entry in queue_by_seed.items():
        route = route_by_seed.get(seed)
        if route is None:
            rejection_counts["missing_route_result"] += 1
            comparisons.append({
                "seed": seed,
                "accepted": False,
                "reason": "missing_route_result",
                "queue_entry": queue_entry,
                "shadow_baseline": None,
                "route": None,
            })
            continue
        shadow_baseline = shadow_by_seed.get(seed)
        if shadow_baseline is None:
            rejection_counts["missing_shadow_baseline_result"] += 1
            comparisons.append({
                "seed": seed,
                "accepted": False,
                "reason": "missing_shadow_baseline_result",
                "queue_entry": queue_entry,
                "shadow_baseline": None,
                "route": route,
            })
            continue
        accepted, reason = _seed_beats_shadow_baseline(
            shadow_baseline=shadow_baseline,
            route=route,
            min_floor_gain=int(args.min_floor_gain),
        )
        if accepted:
            accepted_seeds.append(seed)
        else:
            rejection_counts[reason] += 1
        comparisons.append(
            {
                "seed": seed,
                "accepted": bool(accepted),
                "reason": reason,
                "queue_entry": queue_entry,
                "shadow_baseline": shadow_baseline,
                "route": route,
            }
        )

    accepted_seed_set = set(accepted_seeds)
    accepted_raw_records = [record for record in raw_branch_records if str(record.get("seed") or "").strip() in accepted_seed_set]

    accepted_dir = output_dir / "accepted_dataset"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "teacher_refresh_version": "offline_noncombat_teacher_refresh.v2",
        "generated_run_dir": str(generated_dir.resolve()),
        "route_generated_run_dir": str(generated_dir.resolve()),
        "shadow_baseline_run_dir": str(shadow_generated_dir.resolve()),
        "queue_path": str(Path(args.queue).resolve()),
        "generator_config_path": str(generator_config_path.resolve()),
        "shadow_baseline_config_path": str(shadow_config_path.resolve()) if shadow_config_path.exists() else None,
        "selected_seed_count": len(queue_entries),
        "accepted_seed_count": len(accepted_seed_set),
        "accepted_sample_count": len(accepted_raw_records),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "comparison_time_utc": _utc_now(),
        "accept_gate": {
            "mode": "same_checkpoint_shadow_baseline",
            "min_floor_gain": int(args.min_floor_gain),
            "boss_damage_margin_configured": float(args.boss_damage_margin),
            "boss_damage_gate_enabled": False,
        },
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps({"metadata": metadata, "comparisons": comparisons}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "accepted_seeds.txt").write_text(
        "\n".join(sorted(accepted_seed_set)) + ("\n" if accepted_seed_set else ""),
        encoding="utf-8",
    )

    if accepted_raw_records:
        write_raw_branch_exports(
            output_dir=accepted_dir,
            branch_records=accepted_raw_records,
            metadata={
                "sample_type_counts": dict(Counter(str(record.get("sample_type") or "unknown") for record in accepted_raw_records)),
                "source": "windowed_offline_noncombat_teacher_refresh",
            },
            partial=False,
        )
        build_ranking_view(
            raw_branch_records=accepted_raw_records,
            output_dir=accepted_dir / "derived" / "rl",
            compatibility_root=accepted_dir / "derived" / "rl",
            partial=False,
        )

    print(
        f"Teacher refresh complete: accepted_seeds={len(accepted_seed_set)} "
        f"accepted_samples={len(accepted_raw_records)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
