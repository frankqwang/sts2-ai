#!/usr/bin/env python3
"""Bridge cleaned Skada card_reward JSONL into offline non-combat ranking format.

This is the simplest useful bridge for hybrid non-combat ranking experiments.
The historical artifact name used "matchup_bridge"; keep that path readable for
older experiments, but prefer `offline_noncombat_ranking` in new outputs/docs.

- Input:
  - cleaned Skada `card_reward` JSONL rows (for example
    `STS2AI/Artifacts/skada/ironclad_card_reward.jsonl`)
- Output:
  - `derived/rl/ranking_sample.jsonl`
  - `tensors/sample_XXXXX.npz`
  - `manifest.json`

The generated dataset is intentionally narrow:
- only `card_reward`
- only offered cards (no synthetic skip target yet)
- state/action tensors are built from a synthetic card_reward state so
  `train_hybrid.py` can consume them through the offline non-combat ranking loader

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

from network.state_features import build_structured_actions, build_structured_state
from network.fullrun_policy import _structured_actions_to_numpy_dict, _structured_state_to_numpy_dict
from core.vocab import load_vocab


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


def _normalize_scores(
    options: list[dict[str, Any]],
    chosen_index: int,
    is_victory: int = 1,
    soft_mode: bool = False,
) -> list[float]:
    """Build a ranking target from cleaned Skada row options.

    Two modes:

    * Legacy (``soft_mode=False``, default for backward compat): use
      ``context_score`` as the base signal and force the chosen option to be
      best with a ``+0.05`` margin. Produces ``chosen_is_best_rate = 1.0`` but
      relies on the strong "human player was always right" assumption. Audits
      of raw Skada data show this assumption fails ~53% of the time (chosen
      differs from both context-first and win_rate_delta-first). Baking that
      fiction into teacher labels actively teaches the policy to imitate
      sub-optimal human decisions.

    * Softened (``soft_mode=True``): let ``context_score`` rank the options
      honestly. When the player won the run, give ``chosen`` a small nudge so
      its score stays within ``0.05`` of the current max (soft bias toward the
      demonstrated choice without overriding stronger context signal). When
      the player lost, no nudge — the teacher label is whatever ``context_score``
      says, even if that disagrees with the human pick.

    Softened mode produces noisier but more honest labels and lets the
    downstream ranking head recover "this pick was actually bad" signal from
    losing runs (~50% of the dataset).
    """
    base_scores = [_safe_float(opt.get("context_score"), 0.0) for opt in options]
    if not base_scores:
        return []

    if not soft_mode:
        # Legacy behavior — force chosen to be the strict best.
        max_other = max(
            (score for idx, score in enumerate(base_scores) if idx != chosen_index),
            default=base_scores[chosen_index],
        )
        if 0 <= chosen_index < len(base_scores) and base_scores[chosen_index] <= max_other:
            base_scores[chosen_index] = max_other + 0.05
    elif is_victory and 0 <= chosen_index < len(base_scores):
        # Soft nudge when the player won — keep chosen within 0.05 of max_other
        # but never strictly above unless context_score already says so.
        max_other = max(
            (score for idx, score in enumerate(base_scores) if idx != chosen_index),
            default=base_scores[chosen_index],
        )
        base_scores[chosen_index] = max(base_scores[chosen_index], max_other - 0.05)
    # is_victory=0 + soft_mode=True: no modification; context_score ranks raw.

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
    soft_mode: bool = False,
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

            scores = _normalize_scores(
                options,
                chosen_index,
                is_victory=_safe_int(row.get("is_victory"), 0),
                soft_mode=soft_mode,
            )
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
                "label_source": (
                    "skada_context_softened" if soft_mode else "skada_context_plus_choice"
                ),
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
        "soft_mode": bool(soft_mode),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge Skada card_reward JSONL into offline non-combat ranking format")
    parser.add_argument("--input", required=True, help="Skada card_reward JSONL path")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--character", default=None, help="Optional character filter (for example IRONCLAD)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max sample count")
    parser.add_argument(
        "--soft-mode",
        action="store_true",
        default=False,
        help=(
            "Drop the 'chosen = always best' hard assumption. Use context_score "
            "as the honest ranking signal, with a small nudge toward chosen "
            "only when is_victory=1. Produces noisier but more faithful teacher "
            "labels — recovers 'this human pick was bad' signal from losing runs."
        ),
    )
    args = parser.parse_args()

    manifest = build_dataset(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        character=args.character,
        max_samples=args.max_samples,
        soft_mode=bool(args.soft_mode),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
