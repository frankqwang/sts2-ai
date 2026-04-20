from __future__ import annotations

"""构建 skada combat replay case 索引。

入口放在 `zero/replay` 下，避免训练相关脚本散落到 `STS2AI/Python`。
这份脚本只负责：
- 过滤 skada runs
- 还原每场战斗开局 build
- 输出长期复用的 replay case 数据集
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ZERO_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(ZERO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ZERO_PACKAGE_ROOT))

from zero.replay import build_cases_from_record, default_starter_build, iter_matching_run_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skada-root", type=Path, default=Path("STS2AI/data/skada/runs_full_detail"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Assets/datasets/zero_skada_replay_cases/v0_103_2_a0_single_combat_v1"),
    )
    parser.add_argument("--game-version", type=str, default="v0.103.2")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--player-count", type=int, default=1)
    parser.add_argument("--character-id", type=str, default="")
    parser.add_argument("--victory-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cases_path = output_root / "cases.jsonl"

    total_runs = 0
    total_cases = 0
    skipped_runs = 0
    skipped_characters: Counter[str] = Counter()
    encounter_counter: Counter[str] = Counter()
    encounter_type_counter: Counter[str] = Counter()
    character_counter: Counter[str] = Counter()

    with cases_path.open("w", encoding="utf-8") as handle:
        for source_path, source_line, record in iter_matching_run_records(
            root=args.skada_root,
            game_version=args.game_version,
            ascension=args.ascension,
            player_count=args.player_count,
            character_id=args.character_id or None,
            victory_only=args.victory_only,
            max_runs=args.max_runs or None,
        ):
            run = record.get("run", {})
            character_id = str(run.get("character") or "")
            try:
                starter_build = default_starter_build(character_id)
            except ValueError:
                skipped_runs += 1
                skipped_characters[character_id or "<empty>"] += 1
                continue
            cases = build_cases_from_record(
                record,
                source_path=source_path,
                source_line=source_line,
                starter_build=starter_build,
            )
            for case in cases:
                if args.max_cases and total_cases >= args.max_cases:
                    break
                handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
                total_cases += 1
                encounter_counter[case.encounter_id] += 1
                encounter_type_counter[case.encounter_type] += 1
                character_counter[case.character_id] += 1
            total_runs += 1
            if args.max_cases and total_cases >= args.max_cases:
                break

    summary = {
        "filters": {
            "game_version": args.game_version,
            "ascension": args.ascension,
            "player_count": args.player_count,
            "character_id": args.character_id or "*",
            "victory_only": bool(args.victory_only),
            "max_runs": args.max_runs,
            "max_cases": args.max_cases,
        },
        "counts": {
            "runs": total_runs,
            "cases": total_cases,
            "skipped_runs": skipped_runs,
        },
        "top_characters": character_counter.most_common(8),
        "skipped_characters": skipped_characters.most_common(12),
        "top_encounter_types": encounter_type_counter.most_common(8),
        "top_encounters": encounter_counter.most_common(20),
        "cases_path": str(cases_path),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases_path": str(cases_path), "summary_path": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
