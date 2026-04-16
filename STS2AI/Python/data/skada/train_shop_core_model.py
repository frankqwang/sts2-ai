#!/usr/bin/env python3
"""Train action-type and item-choice baselines from the cleaned Skada shop-core dataset."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ACTION_LABELS = ["none", "remove", "buy_card", "buy_relic", "buy_potion", "multi_action"]


def _hash_run(run_id: int, seed: int) -> float:
    value = (int(run_id) * 1103515245 + seed * 12345 + 0x9E3779B1) & 0x7FFFFFFF
    return value / float(0x7FFFFFFF)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row in rows:
        if _hash_run(int(row["run_id"]), seed) < val_ratio:
            val_rows.append(row)
        else:
            train_rows.append(row)
    if not train_rows or not val_rows:
        cutoff = max(1, int(len(rows) * (1.0 - val_ratio)))
        train_rows = rows[:cutoff]
        val_rows = rows[cutoff:]
    return train_rows, val_rows


class ActionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class ItemChoiceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate_action_rows(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.tensor([row["feature_vector"] for row in batch], dtype=torch.float32),
        "char_ids": torch.tensor([row["character_idx"] for row in batch], dtype=torch.long),
        "labels": torch.tensor([row["action_idx"] for row in batch], dtype=torch.long),
    }


def collate_item_rows(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_options = max(len(row["options"]) for row in batch)
    num_features = len(batch[0]["options"][0]["feature_vector"])
    item_ids = torch.zeros((len(batch), max_options), dtype=torch.long)
    char_ids = torch.zeros((len(batch),), dtype=torch.long)
    family_ids = torch.zeros((len(batch),), dtype=torch.long)
    option_features = torch.zeros((len(batch), max_options, num_features), dtype=torch.float32)
    option_mask = torch.zeros((len(batch), max_options), dtype=torch.bool)
    chosen_index = torch.zeros((len(batch),), dtype=torch.long)
    baseline_scores = torch.full((len(batch), max_options), -1e9, dtype=torch.float32)

    for batch_index, row in enumerate(batch):
        char_ids[batch_index] = int(row["character_idx"])
        family_ids[batch_index] = int(row["family_idx"])
        chosen_index[batch_index] = int(row["chosen_index"])
        for option_index, option in enumerate(row["options"]):
            option_mask[batch_index, option_index] = True
            item_ids[batch_index, option_index] = int(option["item_idx"])
            option_features[batch_index, option_index] = torch.tensor(option["feature_vector"], dtype=torch.float32)
            baseline_scores[batch_index, option_index] = float(option["baseline_score"])

    return {
        "item_ids": item_ids,
        "char_ids": char_ids,
        "family_ids": family_ids,
        "option_features": option_features,
        "option_mask": option_mask,
        "chosen_index": chosen_index,
        "baseline_scores": baseline_scores,
    }


class ShopActionModel(nn.Module):
    def __init__(self, num_characters: int, num_features: int, num_labels: int):
        super().__init__()
        self.char_embed = nn.Embedding(max(2, num_characters), 8)
        self.mlp = nn.Sequential(
            nn.Linear(num_features + 8, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels),
        )

    def forward(self, features: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([features, self.char_embed(char_ids)], dim=-1))


class ShopItemChoiceModel(nn.Module):
    def __init__(self, num_items: int, num_characters: int, num_families: int, num_features: int):
        super().__init__()
        self.item_embed = nn.Embedding(max(2, num_items), 32)
        self.char_embed = nn.Embedding(max(2, num_characters), 8)
        self.family_embed = nn.Embedding(max(2, num_families), 8)
        self.mlp = nn.Sequential(
            nn.Linear(32 + 8 + 8 + num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        char_ids: torch.Tensor,
        family_ids: torch.Tensor,
        option_features: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        item_emb = self.item_embed(item_ids)
        char_emb = self.char_embed(char_ids).unsqueeze(1).expand(-1, item_ids.size(1), -1)
        family_emb = self.family_embed(family_ids).unsqueeze(1).expand(-1, item_ids.size(1), -1)
        logits = self.mlp(torch.cat([item_emb, char_emb, family_emb, option_features], dim=-1)).squeeze(-1)
        return logits.masked_fill(~option_mask, -1e9)


def _prepare_visit_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], list[str]]:
    feature_names = [
        "floor",
        "act",
        "ascension",
        "hp_before",
        "gold_before",
        "prior_card_count",
        "prior_relic_count",
        "prior_card_gains",
        "prior_shop_removes",
        "prior_upgrades",
        "deck_size",
        "unique_cards",
        "upgraded_cards",
        "upgradeable_count",
        "basic_card_count",
        "curse_card_count",
        "avg_history_context_score",
        "best_upgrade_context_score",
        "mean_upgrade_context_score",
        "recent_damage_taken",
        "recent_damage_dealt",
        "recent_shop_visits",
        "recent_elites",
        "card_option_count",
        "relic_option_count",
        "action_count",
    ]

    char_vocab = {"<unk>": 0}

    def char_index(value: str) -> int:
        if value not in char_vocab:
            char_vocab[value] = len(char_vocab)
        return char_vocab[value]

    prepared = []
    for row in rows:
        row["character_idx"] = char_index(str(row.get("character") or "UNKNOWN"))
        row["action_idx"] = ACTION_LABELS.index(str(row.get("action_type") or "multi_action"))
        row["feature_vector"] = [
            _safe_float(row.get("floor")) / 55.0,
            _safe_float(row.get("act")) / 3.0,
            _safe_float(row.get("ascension")) / 20.0,
            min(1.0, _safe_float(row.get("hp_before")) / 120.0),
            min(1.0, _safe_float(row.get("gold_before")) / 500.0),
            min(1.0, _safe_float(row.get("prior_card_count")) / 40.0),
            min(1.0, _safe_float(row.get("prior_relic_count")) / 20.0),
            min(1.0, _safe_float(row.get("prior_card_gains")) / 25.0),
            min(1.0, _safe_float(row.get("prior_shop_removes")) / 8.0),
            min(1.0, _safe_float(row.get("prior_upgrades")) / 12.0),
            min(1.0, _safe_float(row.get("deck_size")) / 45.0),
            min(1.0, _safe_float(row.get("unique_cards")) / 30.0),
            min(1.0, _safe_float(row.get("upgraded_cards")) / 12.0),
            min(1.0, _safe_float(row.get("upgradeable_count")) / 20.0),
            min(1.0, _safe_float(row.get("basic_card_count")) / 12.0),
            min(1.0, _safe_float(row.get("curse_card_count")) / 5.0),
            _safe_float(row.get("avg_history_context_score"), 0.5),
            _safe_float(row.get("best_upgrade_context_score"), 0.5),
            _safe_float(row.get("mean_upgrade_context_score"), 0.5),
            min(1.0, _safe_float(row.get("recent_damage_taken")) / 40.0),
            min(1.0, _safe_float(row.get("recent_damage_dealt")) / 120.0),
            min(1.0, _safe_float(row.get("recent_shop_visits")) / 4.0),
            min(1.0, _safe_float(row.get("recent_elites")) / 3.0),
            min(1.0, _safe_float(row.get("card_option_count")) / 10.0),
            min(1.0, _safe_float(row.get("relic_option_count")) / 5.0),
            min(1.0, _safe_float(row.get("action_count")) / 6.0),
        ]
        prepared.append(row)
    return prepared, {"character": char_vocab}, feature_names


def _prepare_item_rows(
    rows: list[dict[str, Any]],
    char_vocab: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], list[str], list[str]]:
    feature_names = [
        "floor",
        "act",
        "ascension",
        "hp_before",
        "gold_before",
        "prior_card_count",
        "prior_relic_count",
        "prior_card_gains",
        "prior_shop_removes",
        "prior_upgrades",
        "deck_size",
        "upgraded_cards",
        "upgradeable_count",
        "basic_card_count",
        "curse_card_count",
        "avg_history_context_score",
        "best_upgrade_context_score",
        "mean_upgrade_context_score",
        "recent_damage_taken",
        "recent_damage_dealt",
        "recent_shop_visits",
        "recent_elites",
        "card_option_count",
        "relic_option_count",
        "action_count",
        "offer_size",
        "count_in_deck",
        "is_basic",
        "is_curse",
        "is_starter_unique",
        "prior_score",
        "pick_rate",
        "win_rate_delta",
        "hold_rate",
        "deck_synergy",
        "context_score",
    ]

    item_vocab = {"<unk>": 0}
    family_vocab: dict[str, int] = {}

    def vocab_index(vocab: dict[str, int], value: str) -> int:
        if value not in vocab:
            vocab[value] = len(vocab)
        return vocab[value]

    prepared = []
    for row in rows:
        character = str(row.get("character") or "UNKNOWN")
        if character not in char_vocab:
            char_vocab[character] = len(char_vocab)
        family = str(row.get("choice_family") or "unknown")
        row["character_idx"] = char_vocab[character]
        row["family_idx"] = vocab_index(family_vocab, family)
        shared = [
            _safe_float(row.get("floor")) / 55.0,
            _safe_float(row.get("act")) / 3.0,
            _safe_float(row.get("ascension")) / 20.0,
            min(1.0, _safe_float(row.get("hp_before")) / 120.0),
            min(1.0, _safe_float(row.get("gold_before")) / 500.0),
            min(1.0, _safe_float(row.get("prior_card_count")) / 40.0),
            min(1.0, _safe_float(row.get("prior_relic_count")) / 20.0),
            min(1.0, _safe_float(row.get("prior_card_gains")) / 25.0),
            min(1.0, _safe_float(row.get("prior_shop_removes")) / 8.0),
            min(1.0, _safe_float(row.get("prior_upgrades")) / 12.0),
            min(1.0, _safe_float(row.get("deck_size")) / 45.0),
            min(1.0, _safe_float(row.get("upgraded_cards")) / 12.0),
            min(1.0, _safe_float(row.get("upgradeable_count")) / 20.0),
            min(1.0, _safe_float(row.get("basic_card_count")) / 12.0),
            min(1.0, _safe_float(row.get("curse_card_count")) / 5.0),
            _safe_float(row.get("avg_history_context_score"), 0.5),
            _safe_float(row.get("best_upgrade_context_score"), 0.5),
            _safe_float(row.get("mean_upgrade_context_score"), 0.5),
            min(1.0, _safe_float(row.get("recent_damage_taken")) / 40.0),
            min(1.0, _safe_float(row.get("recent_damage_dealt")) / 120.0),
            min(1.0, _safe_float(row.get("recent_shop_visits")) / 4.0),
            min(1.0, _safe_float(row.get("recent_elites")) / 3.0),
            min(1.0, _safe_float(row.get("card_option_count")) / 10.0),
            min(1.0, _safe_float(row.get("relic_option_count")) / 5.0),
            min(1.0, _safe_float(row.get("action_count")) / 6.0),
            min(1.0, _safe_float(row.get("offer_size")) / 10.0),
        ]
        options = []
        for option in row.get("options", []):
            item_id = str(option.get("item_id") or "")
            option["item_idx"] = vocab_index(item_vocab, item_id)
            option["feature_vector"] = [
                *shared,
                min(1.0, _safe_float(option.get("count_in_deck")) / 8.0),
                _safe_float(option.get("is_basic")),
                _safe_float(option.get("is_curse")),
                _safe_float(option.get("is_starter_unique")),
                _safe_float(option.get("prior_score"), 0.5),
                _safe_float(option.get("pick_rate")),
                _safe_float(option.get("win_rate_delta")),
                _safe_float(option.get("hold_rate")),
                _safe_float(option.get("deck_synergy")),
                _safe_float(option.get("context_score"), 0.5),
            ]
            options.append(option)
        row["options"] = options
        prepared.append(row)

    family_names = [family for family, _ in sorted(family_vocab.items(), key=lambda item: item[1])]
    return prepared, {"character": char_vocab, "item": item_vocab, "family": family_vocab}, feature_names, family_names


@dataclass
class ActionEvalResult:
    loss: float
    top1_acc: float
    majority_acc: float


@dataclass
class ItemEvalResult:
    loss: float
    top1_acc: float
    baseline_acc: float
    per_family_acc: dict[str, float]
    per_family_baseline_acc: dict[str, float]


def evaluate_actions(model: ShopActionModel, loader: DataLoader, device: torch.device, majority_idx: int) -> ActionEvalResult:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    majority_correct = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            char_ids = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(features, char_ids)
            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            majority_preds = torch.full_like(labels, majority_idx)
            majority_correct += int((majority_preds == labels).sum().item())
            total += int(labels.size(0))
    return ActionEvalResult(
        loss=total_loss / max(1, total),
        top1_acc=correct / max(1, total),
        majority_acc=majority_correct / max(1, total),
    )


def evaluate_items(
    model: ShopItemChoiceModel,
    loader: DataLoader,
    device: torch.device,
    family_names: list[str],
) -> ItemEvalResult:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    baseline_correct = 0
    family_totals: dict[str, int] = {}
    family_correct: dict[str, int] = {}
    family_baseline_correct: dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            item_ids = batch["item_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            family_ids = batch["family_ids"].to(device)
            option_features = batch["option_features"].to(device)
            option_mask = batch["option_mask"].to(device)
            chosen_index = batch["chosen_index"].to(device)

            logits = model(item_ids, char_ids, family_ids, option_features, option_mask)
            loss = F.cross_entropy(logits, chosen_index)
            total_loss += float(loss.item()) * chosen_index.size(0)
            preds = logits.argmax(dim=-1)
            baseline_preds = batch["baseline_scores"].argmax(dim=-1)
            correct += int((preds == chosen_index).sum().item())
            baseline_correct += int((baseline_preds == batch["chosen_index"]).sum().item())
            total += int(chosen_index.size(0))

            for fam_idx, pred_idx, base_idx, label_idx in zip(
                batch["family_ids"].tolist(),
                preds.tolist(),
                baseline_preds.tolist(),
                batch["chosen_index"].tolist(),
            ):
                family = family_names[fam_idx]
                family_totals[family] = family_totals.get(family, 0) + 1
                if pred_idx == label_idx:
                    family_correct[family] = family_correct.get(family, 0) + 1
                if base_idx == label_idx:
                    family_baseline_correct[family] = family_baseline_correct.get(family, 0) + 1

    return ItemEvalResult(
        loss=total_loss / max(1, total),
        top1_acc=correct / max(1, total),
        baseline_acc=baseline_correct / max(1, total),
        per_family_acc={family: family_correct.get(family, 0) / count for family, count in family_totals.items()},
        per_family_baseline_acc={family: family_baseline_correct.get(family, 0) / count for family, count in family_totals.items()},
    )


def _collect_action_examples(
    model: ShopActionModel,
    rows: list[dict[str, Any]],
    device: torch.device,
    majority_idx: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            features = torch.tensor([row["feature_vector"]], dtype=torch.float32, device=device)
            char_ids = torch.tensor([row["character_idx"]], dtype=torch.long, device=device)
            pred_idx = int(model(features, char_ids).argmax(dim=-1).item())
            if pred_idx == int(row["action_idx"]) and pred_idx != majority_idx:
                examples.append({
                    "visit_id": row["visit_id"],
                    "floor": row["floor"],
                    "actual_action": row["action_type"],
                    "predicted_action": ACTION_LABELS[pred_idx],
                    "majority_baseline_action": ACTION_LABELS[majority_idx],
                    "gold_before": row.get("gold_before"),
                    "card_option_count": row.get("card_option_count"),
                    "relic_option_count": row.get("relic_option_count"),
                    "action_count": row.get("action_count"),
                })
                if len(examples) >= limit:
                    break
    return examples


def _collect_item_examples(
    model: ShopItemChoiceModel,
    rows: list[dict[str, Any]],
    device: torch.device,
    family_names: list[str],
    limit: int = 12,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            max_options = len(row["options"])
            item_ids = torch.zeros((1, max_options), dtype=torch.long, device=device)
            option_features = torch.zeros((1, max_options, len(row["options"][0]["feature_vector"])), dtype=torch.float32, device=device)
            option_mask = torch.zeros((1, max_options), dtype=torch.bool, device=device)
            for idx, option in enumerate(row["options"]):
                item_ids[0, idx] = int(option["item_idx"])
                option_features[0, idx] = torch.tensor(option["feature_vector"], dtype=torch.float32, device=device)
                option_mask[0, idx] = True
            char_ids = torch.tensor([row["character_idx"]], dtype=torch.long, device=device)
            family_ids = torch.tensor([row["family_idx"]], dtype=torch.long, device=device)
            logits = model(item_ids, char_ids, family_ids, option_features, option_mask)
            pred_idx = int(logits.argmax(dim=-1).item())
            baseline_idx = max(range(len(row["options"])), key=lambda idx: float(row["options"][idx]["baseline_score"]))
            if pred_idx == int(row["chosen_index"]) and baseline_idx != int(row["chosen_index"]):
                examples.append({
                    "choice_id": row["choice_id"],
                    "family": family_names[int(row["family_idx"])],
                    "floor": row["floor"],
                    "chosen_item_id": row["chosen_item_id"],
                    "predicted_item_id": row["options"][pred_idx]["item_id"],
                    "baseline_item_id": row["options"][baseline_idx]["item_id"],
                })
                if len(examples) >= limit:
                    break
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Skada shop-core models")
    parser.add_argument("--visit-data", required=True, help="JSONL dataset from build_shop_core_dataset.py for visit rows")
    parser.add_argument("--item-data", required=True, help="JSONL dataset from build_shop_core_dataset.py for item rows")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    visit_path = Path(args.visit_data)
    item_path = Path(args.item_data)
    visit_rows = _load_rows(visit_path)
    raw_item_rows = _load_rows(item_path)

    visit_rows, visit_vocabs, action_feature_names = _prepare_visit_rows(visit_rows)
    item_rows, item_vocabs, item_feature_names, family_names = _prepare_item_rows(raw_item_rows, visit_vocabs["character"])

    train_visit_rows, val_visit_rows = _split_rows(visit_rows, args.val_ratio, args.seed)
    train_item_rows, val_item_rows = _split_rows(item_rows, args.val_ratio, args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else visit_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    majority_idx = max(
        range(len(ACTION_LABELS)),
        key=lambda idx: sum(1 for row in train_visit_rows if int(row["action_idx"]) == idx),
    )
    action_counts = {
        idx: max(1, sum(1 for row in train_visit_rows if int(row["action_idx"]) == idx))
        for idx in range(len(ACTION_LABELS))
    }

    visit_train_loader = DataLoader(ActionDataset(train_visit_rows), batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_action_rows)
    visit_val_loader = DataLoader(ActionDataset(val_visit_rows), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_action_rows)
    item_train_loader = DataLoader(ItemChoiceDataset(train_item_rows), batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_item_rows)
    item_val_loader = DataLoader(ItemChoiceDataset(val_item_rows), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_item_rows)

    device = torch.device(args.device)
    action_model = ShopActionModel(len(visit_vocabs["character"]), len(action_feature_names), len(ACTION_LABELS)).to(device)
    item_model = ShopItemChoiceModel(len(item_vocabs["item"]), len(item_vocabs["character"]), len(item_vocabs["family"]), len(item_feature_names)).to(device)
    action_optimizer = torch.optim.AdamW(action_model.parameters(), lr=args.lr, weight_decay=1e-4)
    item_optimizer = torch.optim.AdamW(item_model.parameters(), lr=args.lr, weight_decay=1e-4)
    action_class_weights = torch.tensor(
        [len(train_visit_rows) / (len(ACTION_LABELS) * action_counts[idx]) for idx in range(len(ACTION_LABELS))],
        dtype=torch.float32,
        device=device,
    )

    best_action_acc = -1.0
    best_item_acc = -1.0
    best_action_metrics: dict[str, Any] = {}
    best_item_metrics: dict[str, Any] = {}
    action_path = output_dir / "shop_action_best.pt"
    item_path_out = output_dir / "shop_item_choice_best.pt"

    for epoch in range(1, args.epochs + 1):
        action_model.train()
        running_action_loss = 0.0
        total_action = 0
        correct_action = 0
        for batch in visit_train_loader:
            features = batch["features"].to(device)
            char_ids = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = action_model(features, char_ids)
            loss = F.cross_entropy(logits, labels, weight=action_class_weights)
            action_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=1.0)
            action_optimizer.step()

            running_action_loss += float(loss.item()) * labels.size(0)
            total_action += int(labels.size(0))
            correct_action += int((logits.argmax(dim=-1) == labels).sum().item())

        action_train_loss = running_action_loss / max(1, total_action)
        action_train_acc = correct_action / max(1, total_action)
        action_val = evaluate_actions(action_model, visit_val_loader, device, majority_idx)

        item_train_loss = 0.0
        item_train_acc = 0.0
        if train_item_rows:
            item_model.train()
            running_item_loss = 0.0
            total_item = 0
            correct_item = 0
            for batch in item_train_loader:
                item_ids = batch["item_ids"].to(device)
                char_ids = batch["char_ids"].to(device)
                family_ids = batch["family_ids"].to(device)
                option_features = batch["option_features"].to(device)
                option_mask = batch["option_mask"].to(device)
                chosen_index = batch["chosen_index"].to(device)

                logits = item_model(item_ids, char_ids, family_ids, option_features, option_mask)
                loss = F.cross_entropy(logits, chosen_index)
                item_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(item_model.parameters(), max_norm=1.0)
                item_optimizer.step()

                running_item_loss += float(loss.item()) * chosen_index.size(0)
                total_item += int(chosen_index.size(0))
                correct_item += int((logits.argmax(dim=-1) == chosen_index).sum().item())
            item_train_loss = running_item_loss / max(1, total_item)
            item_train_acc = correct_item / max(1, total_item)
            item_val = evaluate_items(item_model, item_val_loader, device, family_names)
        else:
            item_val = ItemEvalResult(loss=0.0, top1_acc=0.0, baseline_acc=0.0, per_family_acc={}, per_family_baseline_acc={})

        print(
            f"epoch={epoch:02d} "
            f"action_train_loss={action_train_loss:.4f} action_train_top1={action_train_acc:.4f} "
            f"action_val_loss={action_val.loss:.4f} action_val_top1={action_val.top1_acc:.4f} "
            f"action_majority={action_val.majority_acc:.4f} "
            f"item_train_loss={item_train_loss:.4f} item_train_top1={item_train_acc:.4f} "
            f"item_val_loss={item_val.loss:.4f} item_val_top1={item_val.top1_acc:.4f} "
            f"item_baseline={item_val.baseline_acc:.4f}"
        )

        if action_val.top1_acc > best_action_acc:
            best_action_acc = action_val.top1_acc
            best_action_metrics = {
                "epoch": epoch,
                "train_loss": action_train_loss,
                "train_top1_acc": action_train_acc,
                "val_loss": action_val.loss,
                "val_top1_acc": action_val.top1_acc,
                "val_majority_baseline_top1_acc": action_val.majority_acc,
            }
            torch.save(
                {
                    "model_state_dict": action_model.state_dict(),
                    "vocabs": visit_vocabs,
                    "feature_names": action_feature_names,
                    "metrics": best_action_metrics,
                },
                action_path,
            )

        if item_val.top1_acc > best_item_acc:
            best_item_acc = item_val.top1_acc
            best_item_metrics = {
                "epoch": epoch,
                "train_loss": item_train_loss,
                "train_top1_acc": item_train_acc,
                "val_loss": item_val.loss,
                "val_top1_acc": item_val.top1_acc,
                "val_baseline_top1_acc": item_val.baseline_acc,
                "val_per_family_top1_acc": item_val.per_family_acc,
                "val_per_family_baseline_top1_acc": item_val.per_family_baseline_acc,
            }
            torch.save(
                {
                    "model_state_dict": item_model.state_dict(),
                    "vocabs": item_vocabs,
                    "feature_names": item_feature_names,
                    "family_names": family_names,
                    "metrics": best_item_metrics,
                },
                item_path_out,
            )

    action_state = torch.load(action_path, map_location=device)
    action_model.load_state_dict(action_state["model_state_dict"])
    item_state = torch.load(item_path_out, map_location=device)
    item_model.load_state_dict(item_state["model_state_dict"])
    best_action_metrics = dict(action_state["metrics"])
    best_item_metrics = dict(item_state["metrics"])

    examples = {
        "action_examples": _collect_action_examples(action_model, val_visit_rows, device, majority_idx, limit=10),
        "item_examples": _collect_item_examples(item_model, val_item_rows, device, family_names, limit=12),
    }
    examples_path = output_dir / "shop_core_examples.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    config_path = output_dir / "train_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    best_metrics = {
        "action_best": best_action_metrics,
        "item_best": best_item_metrics,
        "num_train_visits": len(train_visit_rows),
        "num_val_visits": len(val_visit_rows),
        "num_train_item_choices": len(train_item_rows),
        "num_val_item_choices": len(val_item_rows),
        "action_feature_names": action_feature_names,
        "item_feature_names": item_feature_names,
        "family_names": family_names,
        "majority_action_label": ACTION_LABELS[majority_idx],
        "visit_data": str(visit_path),
        "item_data": str(item_path),
    }
    best_metrics["examples_path"] = str(examples_path)
    best_metrics["config_path"] = str(config_path)
    metrics_path = output_dir / "shop_core_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))
    print(f"saved_action_model={action_path}")
    print(f"saved_item_model={item_path_out}")
    print(f"saved_metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
