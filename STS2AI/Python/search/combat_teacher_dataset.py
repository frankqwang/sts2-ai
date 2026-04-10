from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from combat_teacher_common import COMBAT_TEACHER_SCHEMA_VERSION


STRICT_MOTIF_REGRET_MARGIN = 0.05
LETHAL_SUBMOTIFS = ("direct_lethal_first_action", "turn_lethal_no_end_turn")


@dataclass(slots=True)
class CombatTeacherSample:
    schema_version: str
    sample_id: str
    split: str
    source_bucket: str
    source_seed: str
    source_checkpoint: str
    state_hash: str
    motif_labels: list[str]
    state: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    baseline_logits: list[float]
    baseline_probs: list[float]
    baseline_best_action_index: int
    best_action_index: int
    best_full_turn_line: list[dict[str, Any]]
    per_action_score: list[float]
    per_action_regret: list[float]
    root_value: float
    leaf_breakdown: dict[str, float]
    continuation_targets: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CombatTeacherSample":
        return cls(
            schema_version=str(payload.get("schema_version") or COMBAT_TEACHER_SCHEMA_VERSION),
            sample_id=str(payload.get("sample_id") or ""),
            split=str(payload.get("split") or "train"),
            source_bucket=str(payload.get("source_bucket") or "unknown"),
            source_seed=str(payload.get("source_seed") or ""),
            source_checkpoint=str(payload.get("source_checkpoint") or ""),
            state_hash=str(payload.get("state_hash") or ""),
            motif_labels=[str(item) for item in payload.get("motif_labels") or []],
            state=dict(payload.get("state") or {}),
            legal_actions=[
                dict(item) for item in payload.get("legal_actions") or []
                if isinstance(item, dict)
            ],
            baseline_logits=[float(item) for item in payload.get("baseline_logits") or []],
            baseline_probs=[float(item) for item in payload.get("baseline_probs") or []],
            baseline_best_action_index=int(payload.get("baseline_best_action_index") or 0),
            best_action_index=int(payload.get("best_action_index") or 0),
            best_full_turn_line=[
                dict(item) for item in payload.get("best_full_turn_line") or []
                if isinstance(item, dict)
            ],
            per_action_score=[float(item) for item in payload.get("per_action_score") or []],
            per_action_regret=[float(item) for item in payload.get("per_action_regret") or []],
            root_value=float(payload.get("root_value") or 0.0),
            leaf_breakdown={
                str(key): float(value)
                for key, value in (payload.get("leaf_breakdown") or {}).items()
            },
            continuation_targets={
                str(key): float(value)
                for key, value in (payload.get("continuation_targets") or {}).items()
            },
        )


def stable_split(sample_id: str, *, holdout_fraction: float = 0.2) -> str:
    digest = sample_id[-2:] if sample_id else "00"
    bucket = int(digest, 16) if len(digest) == 2 else 0
    threshold = int(max(0.0, min(1.0, holdout_fraction)) * 255)
    return "holdout" if bucket <= threshold else "train"


def action_card_token(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    for key in ("card_id", "id", "label"):
        value = action.get(key)
        if value is not None:
            text = str(value).strip().upper()
            if text:
                return text
    return str(action.get("action") or "").strip().upper()


def is_end_turn_action(action: dict[str, Any] | None) -> bool:
    return str((action or {}).get("action") or "").strip().lower() == "end_turn"


def is_use_potion_action(action: dict[str, Any] | None) -> bool:
    return str((action or {}).get("action") or "").strip().lower() == "use_potion"


def dedupe_samples_by_id(samples: list[CombatTeacherSample]) -> list[CombatTeacherSample]:
    seen: set[str] = set()
    unique: list[CombatTeacherSample] = []
    for sample in samples:
        sample_id = str(sample.sample_id or "")
        if sample_id in seen:
            continue
        seen.add(sample_id)
        unique.append(sample)
    return unique


def _min_regret_for(sample: CombatTeacherSample, predicate) -> float:
    best = float("inf")
    for idx, action in enumerate(sample.legal_actions):
        if idx >= len(sample.per_action_regret):
            continue
        if not predicate(action):
            continue
        best = min(best, float(sample.per_action_regret[idx]))
    return best


def _min_regret_excluding_best(
    sample: CombatTeacherSample,
    predicate=None,
) -> float:
    best = float("inf")
    for idx, action in enumerate(sample.legal_actions):
        if idx >= len(sample.per_action_regret) or idx == int(sample.best_action_index):
            continue
        if predicate is not None and not predicate(action):
            continue
        best = min(best, float(sample.per_action_regret[idx]))
    return best


def sample_metric_applicable(
    sample: CombatTeacherSample,
    motif: str,
    *,
    min_margin: float = STRICT_MOTIF_REGRET_MARGIN,
) -> bool:
    labels = set(sample.motif_labels or [])
    if motif == "missed_lethal":
        return any(
            sample_metric_applicable(sample, submotif, min_margin=min_margin)
            for submotif in LETHAL_SUBMOTIFS
        )
    if motif not in labels:
        return False

    optimal_action = sample.legal_actions[sample.best_action_index] if 0 <= sample.best_action_index < len(sample.legal_actions) else {}
    optimal_card = action_card_token(optimal_action)
    min_margin = float(max(0.0, min_margin))

    if motif == "direct_lethal_first_action":
        alt_regret = _min_regret_excluding_best(sample)
        return (
            not is_end_turn_action(optimal_action)
            and len(sample.best_full_turn_line or []) <= 1
            and alt_regret >= min_margin
        )

    if motif == "turn_lethal_no_end_turn":
        end_turn_regret = _min_regret_for(sample, is_end_turn_action)
        return (
            not is_end_turn_action(optimal_action)
            and len(sample.best_full_turn_line or []) > 1
            and end_turn_regret >= min_margin
        )

    if motif == "bash_before_strike":
        bash_regret = _min_regret_for(sample, lambda action: action_card_token(action) == "BASH")
        non_bash_regret = _min_regret_for(
            sample,
            lambda action: (not is_end_turn_action(action)) and action_card_token(action) != "BASH",
        )
        return optimal_card == "BASH" and bash_regret <= 1e-6 and non_bash_regret >= min_margin

    if motif == "bodyslam_before_block":
        body_slam_regret = _min_regret_for(sample, lambda action: action_card_token(action) == "BODY_SLAM")
        return optimal_card == "DEFEND_IRONCLAD" and body_slam_regret >= min_margin

    if motif == "bad_end_turn":
        end_turn_regret = _min_regret_for(sample, is_end_turn_action)
        return not is_end_turn_action(optimal_action) and end_turn_regret >= min_margin

    if motif == "potion_misuse":
        potion_regret = _min_regret_for(sample, is_use_potion_action)
        return not is_use_potion_action(optimal_action) and potion_regret >= min_margin

    return False


def load_combat_teacher_samples(path: str | Path) -> list[CombatTeacherSample]:
    samples: list[CombatTeacherSample] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            samples.append(CombatTeacherSample.from_dict(json.loads(line)))
    return samples


def write_combat_teacher_samples(
    path: str | Path,
    samples: list[CombatTeacherSample],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False))
            handle.write("\n")

    if metadata is not None:
        manifest = {
            "schema_version": COMBAT_TEACHER_SCHEMA_VERSION,
            "sample_count": len(samples),
            "metadata": metadata,
        }
        manifest_path = output_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
