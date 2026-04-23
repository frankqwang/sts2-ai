from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

STS2AI_ROOT = Path(__file__).resolve().parents[2]
if str(STS2AI_ROOT) not in sys.path:
    sys.path.insert(0, str(STS2AI_ROOT))
BRIDGE_ROOT = STS2AI_ROOT / "bridge"
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from zero.paths import STS2AI_ROOT as ZERO_STS2AI_ROOT
from zero.replay import SkadaCombatCase, load_case_index
from zero.replay.naming import dated_artifact_dir_name
from game_bridge.catalog.sim_catalog import GAME_CATALOG


FLOOR_BANDS = ("low", "mid", "high")
ENCOUNTER_TYPES = ("Normal", "Elite", "Boss")


@dataclass(slots=True)
class CandidateCase:
    case: SkadaCombatCase
    floor_band: str
    encounter_type: str
    encounter_id: str
    run_id: int
    build_signature: str
    richness_score: float
    tie_breaker: float

    @property
    def stratum_key(self) -> tuple[str, str]:
        return self.floor_band, self.encounter_type


def normalize_card_id(value: str) -> str:
    return str(value or "").upper().replace("+", "").strip()


def _unsupported_build_reasons(case: SkadaCombatCase) -> list[str]:
    """返回 build 与当前 sim catalog 不兼容的原因。"""

    reasons: list[str] = []
    for relic in case.build.relics:
        relic_id = normalize_card_id(str(relic.get("id") or ""))
        if relic_id and not GAME_CATALOG.relic_exists(relic_id):
            reasons.append(f"unsupported_relic:{relic_id}")
    for card in case.build.deck:
        card_id = normalize_card_id(str(card.get("id") or ""))
        if card_id and not GAME_CATALOG.card_exists(card_id):
            reasons.append(f"unsupported_card:{card_id}")
    return reasons


def floor_band_for_case(case: SkadaCombatCase) -> str:
    floor = int(case.floor or 0)
    if floor <= 20:
        return "low"
    if floor <= 40:
        return "mid"
    return "high"


def build_signature_for_case(case: SkadaCombatCase) -> str:
    deck_counter: Counter[str] = Counter()
    for card in case.build.deck:
        card_id = normalize_card_id(str(card.get("id") or ""))
        if not card_id:
            continue
        upgrade = int(card.get("upgrade_level") or 0)
        deck_counter[f"{card_id}+{upgrade}"] += 1
    relic_counter: Counter[str] = Counter()
    for relic in case.build.relics:
        relic_id = normalize_card_id(str(relic.get("id") or ""))
        if relic_id:
            relic_counter[relic_id] += 1
    deck_part = "|".join(f"{key}:{deck_counter[key]}" for key in sorted(deck_counter))
    relic_part = "|".join(f"{key}:{relic_counter[key]}" for key in sorted(relic_counter))
    return f"deck={deck_part}||relics={relic_part}"


def richness_score_for_case(case: SkadaCombatCase) -> float:
    floor = float(case.floor or 0)
    hp_before = float(case.floor_state.get("hp_before") or 0.0)
    hp_after = float(case.floor_state.get("hp_after") or 0.0)
    hp_delta = max(0.0, hp_before - hp_after)
    combat_turns = float(case.metadata.get("combat_turns") or 0.0)
    unique_cards = len(
        {
            normalize_card_id(str(card.get("id") or ""))
            for card in case.build.deck
            if normalize_card_id(str(card.get("id") or ""))
        }
    )
    encounter_bonus = 0.0
    enc_type = str(case.encounter_type or "")
    if enc_type == "Boss":
        encounter_bonus = 5.0
    elif enc_type == "Elite":
        encounter_bonus = 2.5
    return (
        encounter_bonus
        + min(floor, 60.0) * 0.05
        + min(hp_delta, 40.0) * 0.08
        + min(combat_turns, 12.0) * 0.12
        + min(unique_cards, 30) * 0.05
    )


def filter_cases(
    cases: Iterable[SkadaCombatCase],
    *,
    character_id: str,
    ascension: int,
) -> list[CandidateCase]:
    filtered: list[CandidateCase] = []
    normalized_character = character_id.strip().upper()
    rng = random.Random(0)
    for case in cases:
        if str(case.character_id or "").upper() != normalized_character:
            continue
        if int(case.ascension or 0) != int(ascension):
            continue
        unsupported_reasons = _unsupported_build_reasons(case)
        if unsupported_reasons:
            continue
        encounter_id = str(case.encounter_id or "").upper()
        # 这类事件战斗 case 在索引里合法，但当前 sim combat_reset 路径并不都支持
        # 直接重开，混进训练池会在 baseline/eval 或 collect 时触发
        # "Setting must be set!" 之类的 reset 失败。
        if "EVENT_ENCOUNTER" in encounter_id:
            continue
        enc_type = str(case.encounter_type or "Normal").title()
        if enc_type not in ENCOUNTER_TYPES:
            enc_type = "Normal"
        filtered.append(
            CandidateCase(
                case=case,
                floor_band=floor_band_for_case(case),
                encounter_type=enc_type,
                encounter_id=str(case.encounter_id or ""),
                run_id=int(case.run_id or 0),
                build_signature=build_signature_for_case(case),
                richness_score=richness_score_for_case(case),
                tie_breaker=rng.random(),
            )
        )
    return filtered


def collect_filter_stats(
    cases: Iterable[SkadaCombatCase],
    *,
    character_id: str,
    ascension: int,
) -> dict[str, object]:
    normalized_character = character_id.strip().upper()
    rejected_reason_counts: Counter[str] = Counter()
    total = 0
    matched_character_ascension = 0
    kept = 0
    for case in cases:
        total += 1
        if str(case.character_id or "").upper() != normalized_character:
            continue
        if int(case.ascension or 0) != int(ascension):
            continue
        matched_character_ascension += 1
        unsupported_reasons = _unsupported_build_reasons(case)
        if unsupported_reasons:
            rejected_reason_counts.update(unsupported_reasons)
            continue
        encounter_id = str(case.encounter_id or "").upper()
        if "EVENT_ENCOUNTER" in encounter_id:
            rejected_reason_counts["unsupported_encounter:event_encounter"] += 1
            continue
        kept += 1
    return {
        "total_case_count": total,
        "matched_character_ascension_count": matched_character_ascension,
        "kept_count": kept,
        "rejected_reason_counts": dict(sorted(rejected_reason_counts.items())),
    }


def allocate_stratum_quotas(
    *,
    counts: dict[tuple[str, str], int],
    total_size: int,
) -> dict[tuple[str, str], int]:
    active_keys = [key for key, count in counts.items() if count > 0]
    if total_size <= 0:
        return {key: 0 for key in counts}
    if not active_keys:
        raise ValueError("没有可用 strata。")

    quotas = {key: 0 for key in counts}
    remaining = int(total_size)

    if total_size >= len(active_keys):
        for key in active_keys:
            quotas[key] = 1
            remaining -= 1

    total_available = sum(counts[key] for key in active_keys)
    raw_targets: dict[tuple[str, str], float] = {}
    for key in active_keys:
        proportion = counts[key] / float(total_available)
        raw_targets[key] = proportion * max(0, remaining)

    remainders: list[tuple[float, tuple[str, str]]] = []
    for key in active_keys:
        extra = int(math.floor(raw_targets[key]))
        quotas[key] += extra
        remainders.append((raw_targets[key] - extra, key))

    assigned = sum(quotas.values())
    leftover = total_size - assigned
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        quotas[key] += 1
        leftover -= 1

    while sum(quotas.values()) > total_size:
        for key in sorted(active_keys, key=lambda item: (quotas[item], item), reverse=True):
            min_allowed = 1 if total_size >= len(active_keys) else 0
            if quotas[key] > min_allowed:
                quotas[key] -= 1
                break

    return quotas


def _diversity_bonus(
    candidate: CandidateCase,
    *,
    seen_encounters_global: set[str],
    seen_builds_global: set[str],
    seen_runs_global: set[int],
    seen_encounters_stratum: set[str],
    seen_builds_stratum: set[str],
    seen_runs_stratum: set[int],
) -> float:
    bonus = 0.0
    if candidate.encounter_id not in seen_encounters_global:
        bonus += 5.0
    if candidate.encounter_id not in seen_encounters_stratum:
        bonus += 2.5
    if candidate.build_signature not in seen_builds_global:
        bonus += 4.0
    if candidate.build_signature not in seen_builds_stratum:
        bonus += 2.0
    if candidate.run_id not in seen_runs_global:
        bonus += 2.5
    if candidate.run_id not in seen_runs_stratum:
        bonus += 1.0
    return bonus


def select_diverse_cases(
    *,
    candidates: list[CandidateCase],
    quotas: dict[tuple[str, str], int],
    seed: int,
) -> tuple[list[CandidateCase], dict[str, object]]:
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[CandidateCase]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[candidate.stratum_key].append(candidate)

    for key, items in by_stratum.items():
        rng.shuffle(items)
        items.sort(key=lambda item: (-item.richness_score, item.case.case_id))

    selected: list[CandidateCase] = []
    seen_case_ids: set[str] = set()
    seen_encounters_global: set[str] = set()
    seen_builds_global: set[str] = set()
    seen_runs_global: set[int] = set()

    selected_by_stratum: dict[tuple[str, str], list[CandidateCase]] = defaultdict(list)

    for key in sorted(quotas):
        need = int(quotas.get(key) or 0)
        if need <= 0:
            continue
        pool = list(by_stratum.get(key, []))
        chosen: list[CandidateCase] = []
        seen_encounters_stratum: set[str] = set()
        seen_builds_stratum: set[str] = set()
        seen_runs_stratum: set[int] = set()
        while pool and len(chosen) < need:
            best_index = 0
            best_score = None
            for index, candidate in enumerate(pool):
                score = candidate.richness_score + _diversity_bonus(
                    candidate,
                    seen_encounters_global=seen_encounters_global,
                    seen_builds_global=seen_builds_global,
                    seen_runs_global=seen_runs_global,
                    seen_encounters_stratum=seen_encounters_stratum,
                    seen_builds_stratum=seen_builds_stratum,
                    seen_runs_stratum=seen_runs_stratum,
                )
                score += candidate.tie_breaker * 0.001
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = index
            candidate = pool.pop(best_index)
            if candidate.case.case_id in seen_case_ids:
                continue
            chosen.append(candidate)
            selected.append(candidate)
            selected_by_stratum[key].append(candidate)
            seen_case_ids.add(candidate.case.case_id)
            seen_encounters_global.add(candidate.encounter_id)
            seen_builds_global.add(candidate.build_signature)
            seen_runs_global.add(candidate.run_id)
            seen_encounters_stratum.add(candidate.encounter_id)
            seen_builds_stratum.add(candidate.build_signature)
            seen_runs_stratum.add(candidate.run_id)
        if len(chosen) < need:
            raise ValueError(
                f"stratum {key} 配额不足：need={need} picked={len(chosen)} available={len(by_stratum.get(key, []))}"
            )

    summary = {
        "stratum_selected_counts": {
            f"{key[0]}::{key[1]}": len(value) for key, value in sorted(selected_by_stratum.items())
        },
        "unique_encounter_count": len({candidate.encounter_id for candidate in selected}),
        "unique_run_count": len({candidate.run_id for candidate in selected}),
        "unique_build_signature_count": len({candidate.build_signature for candidate in selected}),
    }
    return selected, summary


def manifest_summary(candidates: list[CandidateCase]) -> dict[str, object]:
    by_stratum: Counter[tuple[str, str]] = Counter(candidate.stratum_key for candidate in candidates)
    encounter_counts: dict[str, int] = {}
    run_counts: dict[str, int] = {}
    build_counts: dict[str, int] = {}
    for key in sorted(by_stratum):
        key_name = f"{key[0]}::{key[1]}"
        stratum_cases = [candidate for candidate in candidates if candidate.stratum_key == key]
        encounter_counts[key_name] = len({candidate.encounter_id for candidate in stratum_cases})
        run_counts[key_name] = len({candidate.run_id for candidate in stratum_cases})
        build_counts[key_name] = len({candidate.build_signature for candidate in stratum_cases})
    return {
        "stratum_counts": {f"{key[0]}::{key[1]}": by_stratum[key] for key in sorted(by_stratum)},
        "encounter_coverage_by_stratum": encounter_counts,
        "run_coverage_by_stratum": run_counts,
        "build_coverage_by_stratum": build_counts,
        "unique_encounter_count": len({candidate.encounter_id for candidate in candidates}),
        "unique_run_count": len({candidate.run_id for candidate in candidates}),
        "unique_build_signature_count": len({candidate.build_signature for candidate in candidates}),
    }


def write_manifest(*, output_root: Path, filename: str, payload: dict[str, object]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-index",
        type=Path,
        default=ZERO_STS2AI_ROOT / "Assets" / "datasets" / "zero_skada_replay_cases" / "v0_103_2_a0_single_combat_v1" / "cases.jsonl",
    )
    parser.add_argument("--character-id", type=str, default="IRONCLAD")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ZERO_STS2AI_ROOT / "Artifacts",
    )
    args = parser.parse_args()

    cases = load_case_index(args.case_index)
    filter_stats = collect_filter_stats(
        cases,
        character_id=args.character_id,
        ascension=int(args.ascension),
    )
    candidates = filter_cases(
        cases,
        character_id=args.character_id,
        ascension=int(args.ascension),
    )
    if len(candidates) < args.train_size + args.eval_size:
        raise ValueError(
            f"case 不足：filtered={len(candidates)} < train+eval={args.train_size + args.eval_size}"
        )

    counts = Counter(candidate.stratum_key for candidate in candidates)
    train_quotas = allocate_stratum_quotas(counts=dict(counts), total_size=int(args.train_size))
    train_selected, train_selection_summary = select_diverse_cases(
        candidates=candidates,
        quotas=train_quotas,
        seed=int(args.seed),
    )

    selected_train_ids = {candidate.case.case_id for candidate in train_selected}
    remaining_candidates = [candidate for candidate in candidates if candidate.case.case_id not in selected_train_ids]
    remaining_counts = Counter(candidate.stratum_key for candidate in remaining_candidates)
    eval_quotas = allocate_stratum_quotas(counts=dict(remaining_counts), total_size=int(args.eval_size))
    eval_selected, eval_selection_summary = select_diverse_cases(
        candidates=remaining_candidates,
        quotas=eval_quotas,
        seed=int(args.seed) + 1,
    )

    output_dir = args.output_root / dated_artifact_dir_name("ironclad-a0-case-pool")
    payload = {
        "case_index": str(args.case_index),
        "character_id": str(args.character_id).upper(),
        "ascension": int(args.ascension),
        "seed": int(args.seed),
        "train_size": int(args.train_size),
        "eval_size": int(args.eval_size),
        "floor_bands": list(FLOOR_BANDS),
        "encounter_types": list(ENCOUNTER_TYPES),
        "train_case_ids": [candidate.case.case_id for candidate in train_selected],
        "eval_case_ids": [candidate.case.case_id for candidate in eval_selected],
        "filter_summary": filter_stats,
        "population_summary": manifest_summary(candidates),
        "train_summary": {
            **manifest_summary(train_selected),
            **train_selection_summary,
        },
        "eval_summary": {
            **manifest_summary(eval_selected),
            **eval_selection_summary,
        },
    }
    output_path = write_manifest(
        output_root=output_dir,
        filename="ironclad_a0_case_pool_manifest.json",
        payload=payload,
    )
    print(
        json.dumps(
            {
                "output_root": str(output_dir),
                "manifest_path": str(output_path),
                "train_size": len(payload["train_case_ids"]),
                "eval_size": len(payload["eval_case_ids"]),
                "kept_after_filter": filter_stats["kept_count"],
                "train_unique_encounters": payload["train_summary"]["unique_encounter_count"],
                "train_unique_runs": payload["train_summary"]["unique_run_count"],
                "train_unique_builds": payload["train_summary"]["unique_build_signature_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
