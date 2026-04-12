#!/usr/bin/env python3
"""
Train a lightweight card-reward choice model from the cleaned Skada dataset.

This is a first supervised baseline:
- input: reward-group context + offered cards
- target: chosen card index
- metric: group top-1 accuracy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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


class CardRewardDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], feature_names: list[str]):
        self.rows = rows
        self.feature_names = feature_names

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate_groups(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_options = max(len(item["options"]) for item in batch)
    num_features = len(batch[0]["options"][0]["feature_vector"])

    card_ids = torch.zeros((len(batch), max_options), dtype=torch.long)
    char_ids = torch.zeros((len(batch),), dtype=torch.long)
    room_ids = torch.zeros((len(batch),), dtype=torch.long)
    option_features = torch.zeros((len(batch), max_options, num_features), dtype=torch.float32)
    option_mask = torch.zeros((len(batch), max_options), dtype=torch.bool)
    chosen_index = torch.zeros((len(batch),), dtype=torch.long)
    baseline_scores = torch.full((len(batch), max_options), -1e9, dtype=torch.float32)

    for batch_index, row in enumerate(batch):
        char_ids[batch_index] = int(row["character_idx"])
        room_ids[batch_index] = int(row["room_type_idx"])
        chosen_index[batch_index] = int(row["chosen_index"])
        for option_index, option in enumerate(row["options"]):
            option_mask[batch_index, option_index] = True
            card_ids[batch_index, option_index] = int(option["card_idx"])
            option_features[batch_index, option_index] = torch.tensor(option["feature_vector"], dtype=torch.float32)
            baseline_scores[batch_index, option_index] = float(option["context_score"])

    return {
        "card_ids": card_ids,
        "char_ids": char_ids,
        "room_ids": room_ids,
        "option_features": option_features,
        "option_mask": option_mask,
        "chosen_index": chosen_index,
        "baseline_scores": baseline_scores,
    }


class CardRewardChoiceModel(nn.Module):
    def __init__(self, num_cards: int, num_characters: int, num_rooms: int, num_features: int):
        super().__init__()
        self.card_embed = nn.Embedding(max(2, num_cards), 32)
        self.char_embed = nn.Embedding(max(2, num_characters), 8)
        self.room_embed = nn.Embedding(max(2, num_rooms), 8)
        self.mlp = nn.Sequential(
            nn.Linear(32 + 8 + 8 + num_features, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(
        self,
        card_ids: torch.Tensor,
        char_ids: torch.Tensor,
        room_ids: torch.Tensor,
        option_features: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        card_emb = self.card_embed(card_ids)
        char_emb = self.char_embed(char_ids).unsqueeze(1).expand(-1, card_ids.size(1), -1)
        room_emb = self.room_embed(room_ids).unsqueeze(1).expand(-1, card_ids.size(1), -1)
        x = torch.cat([card_emb, char_emb, room_emb, option_features], dim=-1)
        logits = self.mlp(x).squeeze(-1)
        logits = logits.masked_fill(~option_mask, -1e9)
        return logits


def _prepare_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], list[str]]:
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
        "avg_history_context_score",
        "offer_size",
        "repeat_count",
        "skada_score_norm",
        "pick_rate",
        "win_rate_delta",
        "hold_rate",
        "floor_early",
        "floor_mid",
        "floor_late",
        "context_score",
    ]

    card_vocab = {"<unk>": 0}
    char_vocab = {"<unk>": 0}
    room_vocab = {"<unk>": 0}

    def vocab_index(vocab: dict[str, int], value: str) -> int:
        if value not in vocab:
            vocab[value] = len(vocab)
        return vocab[value]

    prepared = []
    for row in rows:
        character = str(row.get("character") or "UNKNOWN")
        room_type = str(row.get("room_type") or "UNKNOWN")
        row["character_idx"] = vocab_index(char_vocab, character)
        row["room_type_idx"] = vocab_index(room_vocab, room_type)

        shared = {
            "floor": _safe_float(row.get("floor")) / 50.0,
            "act": _safe_float(row.get("act")) / 3.0,
            "ascension": _safe_float(row.get("ascension")) / 20.0,
            "hp_before": min(1.0, _safe_float(row.get("hp_before")) / 120.0),
            "gold_before": min(1.0, _safe_float(row.get("gold_before")) / 500.0),
            "prior_card_count": min(1.0, _safe_float(row.get("prior_card_count")) / 30.0),
            "prior_relic_count": min(1.0, _safe_float(row.get("prior_relic_count")) / 20.0),
            "prior_card_gains": min(1.0, _safe_float(row.get("prior_card_gains")) / 25.0),
            "prior_shop_removes": min(1.0, _safe_float(row.get("prior_shop_removes")) / 8.0),
            "prior_upgrades": min(1.0, _safe_float(row.get("prior_upgrades")) / 10.0),
            "avg_history_context_score": _safe_float(row.get("avg_history_context_score"), 0.5),
            "offer_size": _safe_float(row.get("offer_size")) / 10.0,
        }

        options = []
        for option in row.get("options", []):
            card_id = str(option.get("card_id") or "")
            option["card_idx"] = vocab_index(card_vocab, card_id)
            option["feature_vector"] = [
                shared["floor"],
                shared["act"],
                shared["ascension"],
                shared["hp_before"],
                shared["gold_before"],
                shared["prior_card_count"],
                shared["prior_relic_count"],
                shared["prior_card_gains"],
                shared["prior_shop_removes"],
                shared["prior_upgrades"],
                shared["avg_history_context_score"],
                shared["offer_size"],
                min(1.0, _safe_float(option.get("repeat_count")) / 4.0),
                _safe_float(option.get("skada_score_norm"), 0.5),
                _safe_float(option.get("pick_rate")),
                _safe_float(option.get("win_rate_delta")),
                _safe_float(option.get("hold_rate")),
                _safe_float(option.get("floor_early"), 0.5),
                _safe_float(option.get("floor_mid"), 0.5),
                _safe_float(option.get("floor_late"), 0.5),
                _safe_float(option.get("context_score"), 0.5),
            ]
            options.append(option)
        row["options"] = options
        prepared.append(row)

    vocabs = {
        "card": card_vocab,
        "character": char_vocab,
        "room": room_vocab,
    }
    return prepared, vocabs, feature_names


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
        run_id = int(row["run_id"])
        if _hash_run(run_id, seed) < val_ratio:
            val_rows.append(row)
        else:
            train_rows.append(row)
    if not train_rows or not val_rows:
        cutoff = max(1, int(len(rows) * (1.0 - val_ratio)))
        train_rows = rows[:cutoff]
        val_rows = rows[cutoff:]
    return train_rows, val_rows


@dataclass
class EvalResult:
    loss: float
    top1_acc: float
    baseline_top1_acc: float


def evaluate(model: CardRewardChoiceModel, loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    baseline_correct = 0
    with torch.no_grad():
        for batch in loader:
            card_ids = batch["card_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            room_ids = batch["room_ids"].to(device)
            option_features = batch["option_features"].to(device)
            option_mask = batch["option_mask"].to(device)
            chosen_index = batch["chosen_index"].to(device)
            logits = model(card_ids, char_ids, room_ids, option_features, option_mask)
            loss = F.cross_entropy(logits, chosen_index)
            total_loss += float(loss.item()) * chosen_index.size(0)
            preds = logits.argmax(dim=-1)
            correct += int((preds == chosen_index).sum().item())
            baseline_preds = batch["baseline_scores"].argmax(dim=-1)
            baseline_correct += int((baseline_preds == batch["chosen_index"]).sum().item())
            total += int(chosen_index.size(0))
    return EvalResult(
        loss=total_loss / max(1, total),
        top1_acc=correct / max(1, total),
        baseline_top1_acc=baseline_correct / max(1, total),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Skada card reward choice model")
    parser.add_argument("--data", required=True, help="JSONL dataset from build_card_reward_dataset.py")
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

    data_path = Path(args.data)
    rows = _load_rows(data_path)
    prepared_rows, vocabs, feature_names = _prepare_rows(rows)
    train_rows, val_rows = _split_rows(prepared_rows, args.val_ratio, args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else data_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        CardRewardDataset(train_rows, feature_names),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_groups,
    )
    val_loader = DataLoader(
        CardRewardDataset(val_rows, feature_names),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_groups,
    )

    device = torch.device(args.device)
    model = CardRewardChoiceModel(
        num_cards=len(vocabs["card"]),
        num_characters=len(vocabs["character"]),
        num_rooms=len(vocabs["room"]),
        num_features=len(feature_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = -1.0
    best_metrics: dict[str, Any] = {}
    best_path = output_dir / "card_reward_choice_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            card_ids = batch["card_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            room_ids = batch["room_ids"].to(device)
            option_features = batch["option_features"].to(device)
            option_mask = batch["option_mask"].to(device)
            chosen_index = batch["chosen_index"].to(device)

            logits = model(card_ids, char_ids, room_ids, option_features, option_mask)
            loss = F.cross_entropy(logits, chosen_index)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * chosen_index.size(0)
            total += int(chosen_index.size(0))
            correct += int((logits.argmax(dim=-1) == chosen_index).sum().item())

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_result = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_top1={train_acc:.4f} "
            f"val_loss={val_result.loss:.4f} val_top1={val_result.top1_acc:.4f} "
            f"val_context_baseline={val_result.baseline_top1_acc:.4f}"
        )

        if val_result.top1_acc > best_val_acc:
            best_val_acc = val_result.top1_acc
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1_acc": train_acc,
                "val_loss": val_result.loss,
                "val_top1_acc": val_result.top1_acc,
                "val_context_baseline_top1_acc": val_result.baseline_top1_acc,
                "num_train_groups": len(train_rows),
                "num_val_groups": len(val_rows),
                "num_cards": len(vocabs["card"]),
                "num_characters": len(vocabs["character"]),
                "num_rooms": len(vocabs["room"]),
                "feature_names": feature_names,
                "data": str(data_path),
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocabs": vocabs,
                    "feature_names": feature_names,
                    "metrics": best_metrics,
                },
                best_path,
            )

    metrics_path = output_dir / "card_reward_choice_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))
    print(f"saved_model={best_path}")
    print(f"saved_metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
