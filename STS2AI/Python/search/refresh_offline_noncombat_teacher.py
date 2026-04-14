#!/usr/bin/env python3
"""Run a windowed offline non-combat teacher refresh.

Workflow:
1. Read a queue JSON produced from replay summaries.
2. Run card_reward route-search labeling on the queued seeds, or reuse an existing generated run dir.
3. Compare route-search outcomes against baseline window outcomes.
4. Keep only seeds that beat the baseline gate.
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


def _seed_beats_baseline(
    *,
    baseline: dict[str, Any],
    route: dict[str, Any],
    min_floor_gain: int,
    boss_damage_margin: float,
) -> tuple[bool, str]:
    baseline_clear = bool(baseline.get("baseline_act1_cleared"))
    route_clear = bool(route.get("act1_cleared"))
    baseline_boss = bool(baseline.get("baseline_boss_reached"))
    route_boss = bool(route.get("boss_reached"))
    baseline_floor = int(baseline.get("baseline_end_floor", 0) or 0)
    route_floor = int(route.get("end_floor", 0) or 0)
    baseline_boss_damage = float(baseline.get("baseline_boss_hp_fraction_dealt_mean", 0.0) or 0.0)
    route_boss_damage = float(route.get("boss_hp_fraction_dealt_mean", 0.0) or 0.0)
    if route_clear and not baseline_clear:
        return True, "act1_clear_upgrade"
    if route_boss and not baseline_boss:
        return True, "boss_reach_upgrade"
    if route_floor >= baseline_floor + int(min_floor_gain):
        return True, f"floor_gain_{route_floor - baseline_floor}"
    if route_boss == baseline_boss and route_clear == baseline_clear and route_boss_damage >= baseline_boss_damage + float(boss_damage_margin):
        return True, "boss_damage_margin"
    return False, "no_improvement"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh offline non-combat teacher data from a training window queue.")
    parser.add_argument("--queue", required=True, help="Queue JSON produced by build_offline_noncombat_teacher_queue.py")
    parser.add_argument("--output-dir", required=True, help="Teacher refresh output root")
    parser.add_argument("--generated-run-dir", default="", help="Optional existing generator output dir; if omitted, launch generator")
    parser.add_argument("--generator-config", default=str(Path(__file__).parents[1] / "configs" / "offline_noncombat_teacher_route_default.toml"))
    parser.add_argument("--checkpoint", default="", help="Optional hybrid checkpoint passed to generator")
    parser.add_argument("--combat-checkpoint", default="", help="Optional combat checkpoint override")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--start-port", type=int, default=16720)
    parser.add_argument("--transport", choices=["http", "pipe", "pipe-binary"], default="pipe-binary")
    parser.add_argument("--auto-launch", action="store_true", default=False)
    parser.add_argument("--headless-dll", default="")
    parser.add_argument("--min-floor-gain", type=int, default=2)
    parser.add_argument("--boss-damage-margin", type=float, default=0.10)
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

    generated_dir = Path(args.generated_run_dir) if args.generated_run_dir else (output_dir / "generated")
    if not args.generated_run_dir:
        command = _build_generator_command(
            queue_seed_file=queue_seed_file,
            generated_dir=generated_dir,
            generator_config=Path(args.generator_config),
            checkpoint=(args.checkpoint or None),
            combat_checkpoint=(args.combat_checkpoint or None),
            num_envs=int(args.num_envs),
            start_port=int(args.start_port),
            transport=str(args.transport),
            auto_launch=bool(args.auto_launch),
            headless_dll=(args.headless_dll or None),
        )
        (output_dir / "generator_command.txt").write_text(" ".join(command), encoding="utf-8")
        subprocess.run(command, cwd=Path(__file__).resolve().parents[3], check=True)

    episode_logs_path = generated_dir / "episode_logs.json"
    raw_branch_path = generated_dir / "raw" / "raw_branch_rollout.jsonl"
    if not episode_logs_path.exists():
        raise FileNotFoundError(f"Missing generator episode logs: {episode_logs_path}")
    if not raw_branch_path.exists():
        raise FileNotFoundError(f"Missing generator raw branch records: {raw_branch_path}")

    route_logs = _load_json(episode_logs_path)
    raw_branch_records = _load_jsonl(raw_branch_path)
    route_by_seed = {
        str(entry.get("seed") or "").strip(): entry
        for entry in route_logs
        if str(entry.get("seed") or "").strip()
    }
    queue_by_seed = {
        str(entry.get("seed") or "").strip(): entry
        for entry in queue_entries
        if str(entry.get("seed") or "").strip()
    }

    comparisons: list[dict[str, Any]] = []
    accepted_seeds: list[str] = []
    rejection_counts: Counter[str] = Counter()
    for seed, baseline in queue_by_seed.items():
        route = route_by_seed.get(seed)
        if route is None:
            rejection_counts["missing_route_result"] += 1
            comparisons.append({"seed": seed, "accepted": False, "reason": "missing_route_result", "baseline": baseline, "route": None})
            continue
        accepted, reason = _seed_beats_baseline(
            baseline=baseline,
            route=route,
            min_floor_gain=int(args.min_floor_gain),
            boss_damage_margin=float(args.boss_damage_margin),
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
                "baseline": baseline,
                "route": route,
            }
        )

    accepted_seed_set = set(accepted_seeds)
    accepted_raw_records = [record for record in raw_branch_records if str(record.get("seed") or "").strip() in accepted_seed_set]

    accepted_dir = output_dir / "accepted_dataset"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "teacher_refresh_version": "offline_noncombat_teacher_refresh.v1",
        "generated_run_dir": str(generated_dir.resolve()),
        "queue_path": str(Path(args.queue).resolve()),
        "selected_seed_count": len(queue_entries),
        "accepted_seed_count": len(accepted_seed_set),
        "accepted_sample_count": len(accepted_raw_records),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "comparison_time_utc": _utc_now(),
        "accept_gate": {
            "min_floor_gain": int(args.min_floor_gain),
            "boss_damage_margin": float(args.boss_damage_margin),
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
