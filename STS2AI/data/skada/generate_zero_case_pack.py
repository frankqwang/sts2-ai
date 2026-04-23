from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

STS2AI_ROOT = Path(__file__).resolve().parents[2]
if str(STS2AI_ROOT) not in sys.path:
    sys.path.insert(0, str(STS2AI_ROOT))

from zero.paths import STS2AI_ROOT as ZERO_STS2AI_ROOT
from zero.replay import SkadaCombatCase, load_case_index
from zero.replay.naming import dated_artifact_dir_name


BUCKETS = ("submenu_exhaust", "setup_payoff", "resource_dig", "threat_lethal")
SUBMENU_EXHAUST_IDS = {
    "PURITY",
    "TRUE_GRIT",
    "SECOND_WIND",
    "FIEND_FIRE",
    "SEVER_SOUL",
    "DARK_EMBRACE",
    "FEEL_NO_PAIN",
    "BURNING_PACT",
    "CORRUPTION",
    "EXHUME",
    "SENTINEL",
}
SETUP_PAYOFF_IDS = {
    "FEEL_NO_PAIN",
    "DARK_EMBRACE",
    "CORRUPTION",
    "BARRICADE",
    "DEMON_FORM",
    "NOXIOUS_FUMES",
    "ENVENOM",
    "NIGHTMARE",
    "WRAITH_FORM",
    "ELECTRODYNAMICS",
    "BIAS_COGNITION",
    "FOCUS",
    "REAPER",
    "LIMIT_BREAK",
    "POISON_POWER",
}
RESOURCE_DIG_IDS = {
    "BLOODLETTING",
    "BURNING_PACT",
    "INFERNAL_BLADE",
    "OFFERING",
    "POMMEL_STRIKE",
    "SHRUG_IT_OFF",
    "MASTER_OF_STRATEGY",
    "CALCULATED_GAMBLE",
    "TACTICIAN",
    "ACROBATICS",
    "PREPARED",
    "SKIM",
    "COOLHEADED",
    "Hologram".upper(),
}
LETHAL_DAMAGE_THRESHOLD = 30.0


@dataclass(slots=True)
class BucketedCase:
    case_id: str
    bucket: str
    score: float
    encounter_type: str


def normalize_card_id(value: str) -> str:
    return str(value or "").upper().replace("+", "").strip()


def case_card_ids(case: SkadaCombatCase) -> set[str]:
    deck_ids = {
        normalize_card_id(str(card.get("id") or ""))
        for card in case.build.deck
    }
    usage_ids = {normalize_card_id(card_id) for card_id in case.card_usage}
    return {card_id for card_id in [*deck_ids, *usage_ids] if card_id}


def classify_case_bucket(case: SkadaCombatCase) -> str:
    card_ids = case_card_ids(case)
    if card_ids & SUBMENU_EXHAUST_IDS:
        return "submenu_exhaust"
    if card_ids & SETUP_PAYOFF_IDS:
        return "setup_payoff"
    if card_ids & RESOURCE_DIG_IDS:
        return "resource_dig"
    return "threat_lethal"


def score_case(case: SkadaCombatCase) -> float:
    encounter_bonus = 2.0 if str(case.encounter_type).lower() in {"boss", "elite"} else 0.0
    max_damage = max(
        [float((stats or {}).get("damage") or 0.0) for stats in case.card_usage.values()],
        default=0.0,
    )
    hp_delta = max(0.0, float(case.floor_state.get("hp_before") or 0.0) - float(case.floor_state.get("hp_after") or 0.0))
    turns = float(case.metadata.get("combat_turns") or 0.0)
    return encounter_bonus + min(max_damage, 60.0) * 0.05 + hp_delta * 0.05 + min(turns, 12.0) * 0.1


def bucket_cases(cases: Iterable[SkadaCombatCase]) -> dict[str, list[BucketedCase]]:
    buckets: dict[str, list[BucketedCase]] = {bucket: [] for bucket in BUCKETS}
    for case in cases:
        bucket = classify_case_bucket(case)
        buckets[bucket].append(
            BucketedCase(
                case_id=case.case_id,
                bucket=bucket,
                score=score_case(case),
                encounter_type=str(case.encounter_type),
            )
        )
    for bucket in BUCKETS:
        buckets[bucket].sort(key=lambda item: (-item.score, item.case_id))
    return buckets


def build_case_pack(
    cases: list[SkadaCombatCase],
    *,
    train_size: int,
    eval_size: int,
    seed: int,
) -> dict[str, object]:
    if train_size <= 0 or eval_size <= 0:
        raise ValueError("train_size / eval_size 必须为正整数。")
    if len(cases) < train_size + eval_size:
        raise ValueError("case 数量不足，无法生成固定规模 case pack。")

    rng = random.Random(seed)
    bucketed = bucket_cases(cases)
    shuffled_bucketed: dict[str, list[BucketedCase]] = {}
    for bucket, items in bucketed.items():
        copied = list(items)
        rng.shuffle(copied)
        copied.sort(key=lambda item: (-item.score, item.case_id))
        shuffled_bucketed[bucket] = copied

    per_bucket_train = train_size // len(BUCKETS)
    per_bucket_eval = eval_size // len(BUCKETS)
    train_ids: list[str] = []
    eval_ids: list[str] = []
    train_counts = {bucket: 0 for bucket in BUCKETS}
    eval_counts = {bucket: 0 for bucket in BUCKETS}

    leftovers: list[BucketedCase] = []
    for bucket in BUCKETS:
        items = list(shuffled_bucketed[bucket])
        eval_take = min(per_bucket_eval, len(items))
        eval_slice = items[:eval_take]
        eval_ids.extend(item.case_id for item in eval_slice)
        eval_counts[bucket] += len(eval_slice)
        items = items[eval_take:]

        train_take = min(per_bucket_train, len(items))
        train_slice = items[:train_take]
        train_ids.extend(item.case_id for item in train_slice)
        train_counts[bucket] += len(train_slice)
        leftovers.extend(items[train_take:])

    rng.shuffle(leftovers)
    used_ids = set(train_ids) | set(eval_ids)
    for item in leftovers:
        if item.case_id in used_ids:
            continue
        if len(eval_ids) < eval_size:
            eval_ids.append(item.case_id)
            eval_counts[item.bucket] += 1
            used_ids.add(item.case_id)
            continue
        if len(train_ids) < train_size:
            train_ids.append(item.case_id)
            train_counts[item.bucket] += 1
            used_ids.add(item.case_id)
        if len(train_ids) >= train_size and len(eval_ids) >= eval_size:
            break

    if len(train_ids) != train_size or len(eval_ids) != eval_size:
        raise ValueError(
            f"无法填满 case pack：train={len(train_ids)}/{train_size} eval={len(eval_ids)}/{eval_size}"
        )

    manifest = {
        "seed": seed,
        "train_case_ids": train_ids,
        "eval_case_ids": eval_ids,
        "bucket_train_counts": train_counts,
        "bucket_eval_counts": eval_counts,
        "bucket_population": {bucket: len(bucketed[bucket]) for bucket in BUCKETS},
    }
    return manifest


def write_case_pack_manifest(
    *,
    output_root: Path,
    manifest_name: str,
    payload: dict[str, object],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / manifest_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-index",
        type=Path,
        default=ZERO_STS2AI_ROOT / "Assets" / "datasets" / "zero_skada_replay_cases" / "v0_103_2_a0_single_combat_v1" / "cases.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ZERO_STS2AI_ROOT / "Artifacts",
    )
    args = parser.parse_args()

    cases = load_case_index(args.case_index)
    payload = build_case_pack(
        cases,
        train_size=int(args.train_size),
        eval_size=int(args.eval_size),
        seed=int(args.seed),
    )
    payload.update(
        {
            "case_index": str(args.case_index),
            "train_size": int(args.train_size),
            "eval_size": int(args.eval_size),
            "buckets": list(BUCKETS),
        }
    )
    run_dir = args.output_root / dated_artifact_dir_name("zero-case-pack")
    output_path = write_case_pack_manifest(
        output_root=run_dir,
        manifest_name="mechanism_pack_manifest.json",
        payload=payload,
    )
    print(
        json.dumps(
            {
                "output_root": str(run_dir),
                "manifest_path": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
