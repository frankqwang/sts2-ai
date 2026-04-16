"""标准评估运行：用固定配置跑标准化评估。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", default="", help="Tag for the output filename")
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--port", type=int, default=16500)
    p.add_argument("--seed-suite", default="regression", choices=["smoke", "regression", "benchmark"])
    p.add_argument("--no-rerank", action="store_true", help="A/B test: disable combat_safety_rerank")
    p.add_argument("--extra", nargs="*", default=[], help="Any extra flags forwarded to evaluate_ai.py")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.name.strip() or Path(args.checkpoint).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "rerank" if not args.no_rerank else "norerank"
    out_json = out_dir / f"{tag}_{suffix}_{ts}.json"
    stdout_log = out_dir / f"{tag}_{suffix}_{ts}.stdout.log"
    stderr_log = out_dir / f"{tag}_{suffix}_{ts}.stderr.log"

    cmd = [
        sys.executable, "STS2AI/Python/evaluate_ai.py",
        "--checkpoint", args.checkpoint,
        "--transport", "pipe-binary",
        "--port", str(args.port),
        "--auto-launch",
        "--num-games", str(args.num_games),
        "--seed-suite", args.seed_suite,
        "--output", str(out_json),
    ]
    if not args.no_rerank:
        cmd.append("--combat-safety-rerank")
    cmd.extend(args.extra)

    print(f"[canonical-eval] {tag} ({suffix}) → {out_json}")
    print(f"  cmd: {' '.join(cmd)}")

    with open(stdout_log, "w", encoding="utf-8") as so, open(stderr_log, "w", encoding="utf-8") as se:
        rc = subprocess.call(cmd, stdout=so, stderr=se)
    if rc != 0:
        print(f"[canonical-eval] FAILED rc={rc}; see {stderr_log}")
        sys.exit(rc)

    # Print summary
    import json
    d = json.load(open(out_json, encoding="utf-8"))
    games = d["results"]["nn"]
    reach = sum(1 for g in games if g.get("boss_reached"))
    victory = sum(1 for g in games if g.get("outcome") == "victory")
    boss_games = [g for g in games if g.get("boss_reached")]
    avg_hp = sum(g.get("boss_hp_fraction_dealt", 0) for g in boss_games) / len(boss_games) if boss_games else 0
    max_hp = max((g.get("boss_hp_fraction_dealt", 0) for g in boss_games), default=0)
    avg_floor = sum(g.get("max_floor", 0) for g in games) / max(1, len(games))
    avg_combats = sum(g.get("num_combats_won", 0) for g in games) / max(1, len(games))

    print()
    print(f"[canonical-eval] {tag} ({suffix}) — {len(games)} games")
    print(f"  boss_reach : {reach}/{len(games)}")
    print(f"  victory    : {victory}/{len(games)}")
    print(f"  avg boss_hp_dealt : {avg_hp:.3f} (max {max_hp:.3f})")
    print(f"  avg_floor  : {avg_floor:.2f}")
    print(f"  avg combats won : {avg_combats:.2f}")


if __name__ == "__main__":
    main()
