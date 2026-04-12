#!/usr/bin/env python3
"""Bridge cleaned Skada card_reward JSONL into MatchupRankingDataset format.

This is the simplest useful bridge for hybrid non-combat ranking experiments:

- Input:
  - cleaned Skada `card_reward` JSONL rows (for example
    `artifacts/skada/ironclad_card_reward.jsonl`)
- Output:
  - `derived/rl/ranking_sample.jsonl`
  - `tensors/sample_XXXXX.npz`
  - `manifest.json`

The generated dataset is intentionally narrow:
- only `card_reward`
- only offered cards (no synthetic skip target yet)
- state/action tensors are built from a synthetic card_reward state so
  `train_hybrid.py` can consume them through `MatchupRankingDataset`

This lets us A/B the non-combat ranking head structure without waiting for a
full regenerated branch-search dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import _path_init  # noqa: F401

from rl_encoder_v2 import build_structured_actions, build_structured_state
from rl_policy_v2 import _structured_actions_to_numpy_dict, _structured_state_to_numpy_dict
from vocab import load_vocab


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_scores(options: list[dict[str, Any]], chosen_index: int) -> list[float]:
    """Build a stable ranking target from cleaned Skada row options.

    We use `context_score` as the base order signal, then ensure the chosen
    option remains best by adding a small margin if needed. This keeps the
    target grounded in the cleaned Skada features without throwing away the
    actual demonstrated choice.
    """
    base_scores = [_safe_float(opt.get("context_score"), 0.0) for opt in options]
    if not base_scores:
        return []
    max_other = max(
        (score for idx, score in enumerate(base_scores) if idx != chosen_index),
        default=base_scores[chosen_index],
    )
    if 0 <= chosen_index < len(base_scores) and base_scores[chosen_index] <= max_other:
        base_scores[chosen_index] = max_other + 0.05
    return [round(float(score), 6) for score in base_scores]


def _synthetic_card_reward_state(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    character = str(row.get("character") or "IRONCLAD")
    floor = _safe_int(row.get("floor"), 1)
    act = _safe_int(row.get("act"), 1)
    ascension = _safe_int(row.get("ascension"), 0)
    hp_before = _safe_int(round(_safe_float(row.get("hp_before"), 60.0)), 60)
    gold_before = _safe_int(round(_safe_float(row.get("gold_before"), 99.0)), 99)
    prior_card_count = max(0, _safe_int(row.get("prior_card_count"), 0))
    prior_relic_count = max(0, _safe_int(row.get("prior_relic_count"), 0))

    reward_cards: list[dict[str, Any]] = []
    legal_actions: list[dict[str, Any]] = []
    for option_index, option in enumerate(row.get("options") or []):
        card_id = str(option.get("card_id") or "")
        reward_cards.append(
            {
                "index": option_index,
                "id": card_id,
                "name": card_id,
                "cost": 1,
                "type": "SKILL",
                "rarity": "COMMON",
                "upgrades": 0,
            }
        )
        legal_actions.append(
            {
                "action": "select_card_reward",
                "card_id": card_id,
                "index": option_index,
            }
        )

    # We intentionally keep the synthetic deck/relic contents very lightweight.
    # For this bridge dataset, the most important learned signal is:
    #   current screen context + offered cards -> ranking target
    synthetic_deck = [
        {
            "id": "STRIKE_IRONCLAD" if character == "IRONCLAD" else "STRIKE_NEUTRAL",
            "cost": 1,
            "type": "ATTACK",
            "upgrades": 0,
            "rarity": "BASIC",
        }
        for _ in range(min(prior_card_count, 30))
    ]
    synthetic_relics = [{"id": "BURNING_BLOOD"}] * min(prior_relic_count, 10)

    state = {
        "state_type": "card_reward",
        "run": {
            "floor": floor,
            "act": act,
            "ascension_level": ascension,
            "character_id": character,
        },
        "player": {
            "hp": hp_before,
            "max_hp": 80,
            "energy": 3,
            "max_energy": 3,
            "gold": gold_before,
            "block": 0,
            "deck": synthetic_deck,
            "relics": synthetic_relics,
            "potions": [],
        },
        "card_reward": {
            "cards": reward_cards,
        },
        "legal_actions": legal_actions,
    }
    return state, legal_actions


def _save_npz(path: Path, state_tensors: dict[str, Any], action_tensors: dict[str, Any]) -> None:
    arrays: dict[str, Any] = {}
    for key, value in state_tensors.items():
        arrays[f"state_{key}"] = value
    for key, value in action_tensors.items():
        arrays[f"action_{key}"] = value
    np.savez_compressed(path, **arrays)


def build_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    character: str | None,
    max_samples: int | None,
) -> dict[str, Any]:
    vocab = load_vocab()
    rows = _load_rows(input_path)
    if character:
        normalized_character = character.strip().upper()
        rows = [
            row for row in rows
            if str(row.get("character") or "").strip().upper() == normalized_character
        ]
    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]

    derived_dir = output_dir / "derived" / "rl"
    tensor_dir = output_dir / "tensors"
    derived_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = derived_dir / "ranking_sample.jsonl"
    sample_count = 0
    score_spreads: list[float] = []
    chosen_is_best = 0

    with ranking_path.open("w", encoding="utf-8") as handle:
        for sample_index, row in enumerate(rows):
            options = row.get("options") or []
            chosen_index = _safe_int(row.get("chosen_index"), 0)
            if len(options) < 2 or not (0 <= chosen_index < len(options)):
                continue

            state, legal_actions = _synthetic_card_reward_state(row)
            structured_state = build_structured_state(state, vocab)
            structured_actions = build_structured_actions(state, legal_actions, vocab)

            scores = _normalize_scores(options, chosen_index)
            if len(scores) < 2:
                continue
            best_idx = int(max(range(len(scores)), key=lambda idx: scores[idx]))
            chosen_is_best += int(best_idx == chosen_index)
            score_spreads.append(float(max(scores) - min(scores)))

            tensor_rel_path = f"tensors/sample_{sample_count:05d}.npz"
            _save_npz(
                output_dir / tensor_rel_path,
                _structured_state_to_numpy_dict(structured_state),
                _structured_actions_to_numpy_dict(structured_actions),
            )

            sample = {
                "dataset_schema_version": "skada_matchup_bridge.v1",
                "sample_type": "card_reward",
                "label_source": "skada_context_plus_choice",
                "offer_id": row.get("offer_id"),
                "run_id": row.get("run_id"),
                "floor": row.get("floor"),
                "act": row.get("act"),
                "character": row.get("character"),
                "options": [
                    {
                        "card_id": str(option.get("card_id") or ""),
                        "card_slug": option.get("card_slug"),
                        "context_score": _safe_float(option.get("context_score"), 0.0),
                    }
                    for option in options
                ],
                "scores": scores,
                "best_idx": best_idx,
                "chosen_index": chosen_index,
                "state_tensors_path": tensor_rel_path,
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            sample_count += 1

    manifest = {
        "schema_version": "skada_matchup_bridge.v1",
        "input_path": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "character": character,
        "num_samples": sample_count,
        "avg_score_spread": round(float(sum(score_spreads) / max(1, len(score_spreads))), 6),
        "chosen_is_best_rate": round(float(chosen_is_best) / max(1, sample_count), 6),
        "ranking_path": str(ranking_path.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge Skada card_reward JSONL into MatchupRankingDataset format")
    parser.add_argument("--input", required=True, help="Skada card_reward JSONL path")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--character", default=None, help="Optional character filter (for example IRONCLAD)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max sample count")
    args = parser.parse_args()

    manifest = build_dataset(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        character=args.character,
        max_samples=args.max_samples,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
