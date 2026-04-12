#!/usr/bin/env python3
"""Evaluate filtered Ironclad non-combat priors against simple baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from train_campfire_model import CampfireChoiceModel, _load_rows as load_campfire_rows, _prepare_rows as prepare_campfire_rows, _split_rows as split_rows
from train_shop_core_model import (
    ACTION_LABELS,
    ShopActionModel,
    ShopItemChoiceModel,
    _load_rows as load_shop_rows,
    _prepare_item_rows,
    _prepare_visit_rows,
)


def _load_checkpoint_model(path: Path, model_cls, *model_args):
    payload = torch.load(path, map_location="cpu")
    model = model_cls(*model_args)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return payload, model


def _campfire_rule_baseline(row: dict[str, Any]) -> str:
    return "HEAL" if float(row.get("rest_value_proxy") or 0.0) > 0.5 else "SMITH"


def _shop_action_rule_baseline(row: dict[str, Any]) -> str:
    gold_before = float(row.get("gold_before") or 0.0)
    return "remove" if gold_before >= 75.0 else "none"


def evaluate_campfire(
    data_path: Path,
    model_path: Path,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    rows = load_campfire_rows(data_path)
    prepared, vocabs, feature_names, label_names = prepare_campfire_rows(rows)
    _, val_rows = split_rows(prepared, val_ratio, seed)
    payload, model = _load_checkpoint_model(
        model_path,
        CampfireChoiceModel,
        len(vocabs["character"]),
        len(feature_names),
        len(label_names),
    )
    correct = 0
    rule_correct = 0
    majority_idx = label_names.index(str(payload["metrics"]["majority_label"]))
    majority_correct = 0
    examples: list[dict[str, Any]] = []
    for row in val_rows:
        features = torch.tensor([row["feature_vector"]], dtype=torch.float32)
        char_ids = torch.tensor([row["character_idx"]], dtype=torch.long)
        with torch.no_grad():
            pred_idx = int(model(features, char_ids).argmax(dim=-1).item())
        predicted = label_names[pred_idx]
        actual = str(row["campfire_choice"])
        rule = _campfire_rule_baseline(row)
        majority = label_names[majority_idx]
        correct += int(predicted == actual)
        rule_correct += int(rule == actual)
        majority_correct += int(majority == actual)
        if predicted == actual and rule != actual and len(examples) < 8:
            examples.append({
                "sample_id": row["sample_id"],
                "floor": row["floor"],
                "actual": actual,
                "model": predicted,
                "rule_baseline": rule,
                "majority_baseline": majority,
                "hp_before": row.get("hp_before"),
                "rest_value_proxy": row.get("rest_value_proxy"),
            })
    total = max(1, len(val_rows))
    return {
        "num_val_rows": len(val_rows),
        "model_top1_acc": correct / total,
        "rule_baseline_top1_acc": rule_correct / total,
        "majority_baseline_top1_acc": majority_correct / total,
        "lift_vs_rule": (correct - rule_correct) / total,
        "lift_vs_majority": (correct - majority_correct) / total,
        "examples_model_beats_rule": examples,
    }


def evaluate_shop(
    visit_path: Path,
    item_path: Path,
    action_model_path: Path,
    item_model_path: Path,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    visit_rows = load_shop_rows(visit_path)
    raw_item_rows = load_shop_rows(item_path)
    prepared_visits, visit_vocabs, action_feature_names = _prepare_visit_rows(visit_rows)
    prepared_items, item_vocabs, item_feature_names, family_names = _prepare_item_rows(raw_item_rows, visit_vocabs["character"])
    _, val_visits = split_rows(prepared_visits, val_ratio, seed)
    _, val_items = split_rows(prepared_items, val_ratio, seed)

    action_payload, action_model = _load_checkpoint_model(
        action_model_path,
        ShopActionModel,
        len(visit_vocabs["character"]),
        len(action_feature_names),
        len(ACTION_LABELS),
    )
    item_payload, item_model = _load_checkpoint_model(
        item_model_path,
        ShopItemChoiceModel,
        len(item_vocabs["item"]),
        len(item_vocabs["character"]),
        len(item_vocabs["family"]),
        len(item_feature_names),
    )

    action_correct = 0
    action_rule_correct = 0
    action_majority = max(
        ACTION_LABELS,
        key=lambda label: sum(1 for row in prepared_visits if str(row["action_type"]) == label),
    )
    action_majority_correct = 0
    action_examples: list[dict[str, Any]] = []

    for row in val_visits:
        features = torch.tensor([row["feature_vector"]], dtype=torch.float32)
        char_ids = torch.tensor([row["character_idx"]], dtype=torch.long)
        with torch.no_grad():
            pred_idx = int(action_model(features, char_ids).argmax(dim=-1).item())
        predicted = ACTION_LABELS[pred_idx]
        actual = str(row["action_type"])
        rule = _shop_action_rule_baseline(row)
        action_correct += int(predicted == actual)
        action_rule_correct += int(rule == actual)
        action_majority_correct += int(action_majority == actual)
        if predicted == actual and rule != actual and len(action_examples) < 8:
            action_examples.append({
                "visit_id": row["visit_id"],
                "floor": row["floor"],
                "actual": actual,
                "model": predicted,
                "rule_baseline": rule,
                "majority_baseline": action_majority,
                "gold_before": row.get("gold_before"),
                "card_option_count": row.get("card_option_count"),
                "relic_option_count": row.get("relic_option_count"),
            })

    item_correct = 0
    item_baseline_correct = 0
    per_family_total: dict[str, int] = {}
    per_family_correct: dict[str, int] = {}
    per_family_baseline_correct: dict[str, int] = {}
    item_examples: list[dict[str, Any]] = []
    for row in val_items:
        options = row["options"]
        option_features = torch.tensor([[option["feature_vector"] for option in options]], dtype=torch.float32)
        item_ids = torch.tensor([[int(option["item_idx"]) for option in options]], dtype=torch.long)
        option_mask = torch.ones((1, len(options)), dtype=torch.bool)
        char_ids = torch.tensor([row["character_idx"]], dtype=torch.long)
        family_ids = torch.tensor([row["family_idx"]], dtype=torch.long)
        with torch.no_grad():
            pred_idx = int(item_model(item_ids, char_ids, family_ids, option_features, option_mask).argmax(dim=-1).item())
        baseline_idx = max(range(len(options)), key=lambda idx: float(options[idx]["baseline_score"]))
        actual_idx = int(row["chosen_index"])
        family = family_names[int(row["family_idx"])]
        item_correct += int(pred_idx == actual_idx)
        item_baseline_correct += int(baseline_idx == actual_idx)
        per_family_total[family] = per_family_total.get(family, 0) + 1
        per_family_correct[family] = per_family_correct.get(family, 0) + int(pred_idx == actual_idx)
        per_family_baseline_correct[family] = per_family_baseline_correct.get(family, 0) + int(baseline_idx == actual_idx)
        if pred_idx == actual_idx and baseline_idx != actual_idx and len(item_examples) < 12:
            item_examples.append({
                "choice_id": row["choice_id"],
                "family": family,
                "floor": row["floor"],
                "actual": row["options"][actual_idx]["item_id"],
                "model": row["options"][pred_idx]["item_id"],
                "baseline": row["options"][baseline_idx]["item_id"],
            })

    num_val_visits = max(1, len(val_visits))
    num_val_items = max(1, len(val_items))
    return {
        "action": {
            "num_val_rows": len(val_visits),
            "model_top1_acc": action_correct / num_val_visits,
            "rule_baseline_top1_acc": action_rule_correct / num_val_visits,
            "majority_baseline_top1_acc": action_majority_correct / num_val_visits,
            "lift_vs_rule": (action_correct - action_rule_correct) / num_val_visits,
            "lift_vs_majority": (action_correct - action_majority_correct) / num_val_visits,
            "examples_model_beats_rule": action_examples,
            "checkpoint_metrics": action_payload.get("metrics") or {},
        },
        "item": {
            "num_val_rows": len(val_items),
            "model_top1_acc": item_correct / num_val_items,
            "baseline_top1_acc": item_baseline_correct / num_val_items,
            "lift_vs_baseline": (item_correct - item_baseline_correct) / num_val_items,
            "per_family_model_top1_acc": {
                family: per_family_correct.get(family, 0) / total for family, total in per_family_total.items()
            },
            "per_family_baseline_top1_acc": {
                family: per_family_baseline_correct.get(family, 0) / total for family, total in per_family_total.items()
            },
            "examples_model_beats_baseline": item_examples,
            "checkpoint_metrics": item_payload.get("metrics") or {},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate filtered Ironclad non-combat priors")
    parser.add_argument("--campfire-data", required=True)
    parser.add_argument("--campfire-model", required=True)
    parser.add_argument("--shop-visit-data", required=True)
    parser.add_argument("--shop-item-data", required=True)
    parser.add_argument("--shop-action-model", required=True)
    parser.add_argument("--shop-item-model", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    campfire_eval = evaluate_campfire(Path(args.campfire_data), Path(args.campfire_model), args.val_ratio, args.seed)
    shop_eval = evaluate_shop(
        Path(args.shop_visit_data),
        Path(args.shop_item_data),
        Path(args.shop_action_model),
        Path(args.shop_item_model),
        args.val_ratio,
        args.seed,
    )

    total_rows = (
        campfire_eval["num_val_rows"]
        + shop_eval["action"]["num_val_rows"]
        + shop_eval["item"]["num_val_rows"]
    )
    combined_model = (
        campfire_eval["model_top1_acc"] * campfire_eval["num_val_rows"]
        + shop_eval["action"]["model_top1_acc"] * shop_eval["action"]["num_val_rows"]
        + shop_eval["item"]["model_top1_acc"] * shop_eval["item"]["num_val_rows"]
    ) / max(1, total_rows)
    combined_baseline = (
        campfire_eval["rule_baseline_top1_acc"] * campfire_eval["num_val_rows"]
        + shop_eval["action"]["rule_baseline_top1_acc"] * shop_eval["action"]["num_val_rows"]
        + shop_eval["item"]["baseline_top1_acc"] * shop_eval["item"]["num_val_rows"]
    ) / max(1, total_rows)

    payload = {
        "campfire": campfire_eval,
        "shop": shop_eval,
        "combined": {
            "num_val_rows": total_rows,
            "model_top1_acc": combined_model,
            "baseline_top1_acc": combined_baseline,
            "lift_vs_baseline": combined_model - combined_baseline,
        },
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
