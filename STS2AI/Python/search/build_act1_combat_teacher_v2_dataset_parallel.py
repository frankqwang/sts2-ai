from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PYTHON_ROOT = _THIS_FILE.parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))


import argparse
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from search.combat_teacher_dataset import dedupe_samples_by_id, load_combat_teacher_samples, write_combat_teacher_samples
from search.build_act1_combat_teacher_v2_dataset import _load_seeds, _take_balanced_samples


def _chunked(items: list[str], chunks: int) -> list[list[str]]:
    if chunks <= 1:
        return [items]
    buckets: list[list[str]] = [[] for _ in range(chunks)]
    for idx, item in enumerate(items):
        buckets[idx % chunks].append(item)
    return [bucket for bucket in buckets if bucket]


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_metadata(manifests: list[dict]) -> dict:
    sampled_reason_counts: Counter[str] = Counter()
    solver_reason_counts: Counter[str] = Counter()
    per_seed_counts: Counter[str] = Counter()
    considered_states = 0
    supported_states = 0
    candidate_count = 0
    deduped_count = 0
    seeds: list[str] = []

    for manifest in manifests:
        metadata = manifest.get("metadata") or {}
        sampled_reason_counts.update(metadata.get("sampled_reason_counts") or {})
        solver_reason_counts.update(metadata.get("solver_reason_counts") or {})
        per_seed_counts.update(metadata.get("per_seed_counts") or {})
        considered_states += int(metadata.get("considered_states") or 0)
        supported_states += int(metadata.get("supported_states") or 0)
        candidate_count += int(metadata.get("candidate_count") or metadata.get("sample_count") or 0)
        deduped_count += int(metadata.get("deduped_count") or metadata.get("sample_count") or 0)
        for seed in metadata.get("seeds") or []:
            text = str(seed).strip()
            if text and text not in seeds:
                seeds.append(text)

    return {
        "seeds": seeds,
        "considered_states": considered_states,
        "supported_states": supported_states,
        "candidate_count": candidate_count,
        "deduped_count": deduped_count,
        "sampled_reason_counts": dict(sorted(sampled_reason_counts.items())),
        "solver_reason_counts": dict(sorted(solver_reason_counts.items())),
        "per_seed_counts": dict(sorted(per_seed_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel wrapper for build_act1_combat_teacher_v2_dataset.py")
    parser.add_argument("--hybrid-checkpoint", required=True)
    parser.add_argument("--combat-checkpoint", required=True)
    parser.add_argument("--seed-file", default="")
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--num-seeds", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=15527)
    parser.add_argument("--transport", choices=["pipe", "pipe-binary"], default="pipe-binary")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--headless-dll", type=Path, default=None)
    parser.add_argument("--floor-limit", type=int, default=17)
    parser.add_argument("--sample-every-combat-step", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--max-player-actions", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=3200, help="Total candidate pool target across all workers.")
    parser.add_argument("--target-samples", type=int, default=2000)
    parser.add_argument("--balanced-seed-cap", type=int, default=10)
    parser.add_argument("--max-samples-per-seed", type=int, default=12)
    parser.add_argument("--uncertainty-margin-threshold", type=float, default=0.16)
    parser.add_argument("--uncertainty-entropy-threshold", type=float, default=0.78)
    parser.add_argument("--low-hp-attacker-threshold", type=int, default=12)
    parser.add_argument("--danger-net-incoming-threshold", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-partials", action="store_true", default=False)
    args = parser.parse_args()

    seeds = [str(seed).strip() for seed in args.seed if str(seed).strip()]
    if args.seed_file:
        seeds.extend(_load_seeds(args.seed_file, limit=args.num_seeds))
    if not seeds:
        seeds = [f"EVAL_{idx:03d}" for idx in range(1, int(args.num_seeds) + 1)]
    seeds = list(dict.fromkeys(seeds))
    if args.num_seeds > 0:
        seeds = seeds[: int(args.num_seeds)]

    worker_count = max(1, min(int(args.workers), len(seeds)))
    seed_chunks = _chunked(seeds, worker_count)
    candidate_per_worker = int(math.ceil(float(max(1, int(args.max_samples))) / max(1, len(seed_chunks))))

    tmp_root = Path(tempfile.mkdtemp(prefix="act1_teacher_v2_parallel_", dir=str(Path(args.output).resolve().parent)))
    script_path = _THIS_FILE.parent / "build_act1_combat_teacher_v2_dataset.py"
    processes: list[tuple[subprocess.Popen, Path]] = []

    try:
        for worker_idx, seed_chunk in enumerate(seed_chunks):
            partial_output = tmp_root / f"worker_{worker_idx:02d}.jsonl"
            command = [
                sys.executable,
                str(script_path),
                "--auto-launch",
                "--hybrid-checkpoint", str(args.hybrid_checkpoint),
                "--combat-checkpoint", str(args.combat_checkpoint),
                "--transport", str(args.transport),
                "--port", str(int(args.base_port) + worker_idx),
                "--floor-limit", str(int(args.floor_limit)),
                "--sample-every-combat-step", str(int(args.sample_every_combat_step)),
                "--max-episode-steps", str(int(args.max_episode_steps)),
                "--max-player-actions", str(int(args.max_player_actions)),
                "--max-samples", str(candidate_per_worker),
                "--max-samples-per-seed", str(int(args.max_samples_per_seed)),
                "--uncertainty-margin-threshold", str(float(args.uncertainty_margin_threshold)),
                "--uncertainty-entropy-threshold", str(float(args.uncertainty_entropy_threshold)),
                "--low-hp-attacker-threshold", str(int(args.low_hp_attacker_threshold)),
                "--danger-net-incoming-threshold", str(int(args.danger_net_incoming_threshold)),
                "--output", str(partial_output),
            ]
            if args.repo_root is not None:
                command.extend(["--repo-root", str(args.repo_root)])
            if args.headless_dll is not None:
                command.extend(["--headless-dll", str(args.headless_dll)])
            for seed in seed_chunk:
                command.extend(["--seed", seed])
            proc = subprocess.Popen(command, cwd=str(Path.cwd()))
            processes.append((proc, partial_output))

        failures: list[str] = []
        for proc, partial_output in processes:
            code = proc.wait()
            if code != 0:
                failures.append(f"{partial_output.name}: exit={code}")
        if failures:
            raise RuntimeError("Parallel build worker failed: " + "; ".join(failures))

        all_samples = []
        manifests = []
        for _proc, partial_output in processes:
            all_samples.extend(load_combat_teacher_samples(partial_output))
            manifests.append(_read_manifest(partial_output.with_suffix(".manifest.json")))

        all_samples = dedupe_samples_by_id(all_samples)
        if int(args.target_samples) > 0:
            all_samples = _take_balanced_samples(
                all_samples,
                target_samples=int(args.target_samples),
                per_seed_cap=int(args.balanced_seed_cap),
            )

        merged_meta = _merge_metadata(manifests)
        merged_meta.update(
            {
                "workers": len(seed_chunks),
                "base_port": int(args.base_port),
                "transport": str(args.transport),
                "max_samples": int(args.max_samples),
                "target_samples": int(args.target_samples),
                "balanced_seed_cap": int(args.balanced_seed_cap),
                "max_samples_per_seed": int(args.max_samples_per_seed),
                "hybrid_checkpoint": str(args.hybrid_checkpoint),
                "combat_checkpoint": str(args.combat_checkpoint),
                "floor_limit": int(args.floor_limit),
                "sample_every_combat_step": int(args.sample_every_combat_step),
                "max_episode_steps": int(args.max_episode_steps),
                "max_player_actions": int(args.max_player_actions),
                "train_count": sum(1 for sample in all_samples if sample.split != "holdout"),
                "holdout_count": sum(1 for sample in all_samples if sample.split == "holdout"),
            }
        )
        write_combat_teacher_samples(args.output, all_samples, metadata=merged_meta)
        print(json.dumps({"output": str(args.output), "sample_count": len(all_samples), "metadata": merged_meta}, ensure_ascii=False, indent=2))
    finally:
        if args.keep_partials:
            print(f"Kept partial worker outputs in {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
