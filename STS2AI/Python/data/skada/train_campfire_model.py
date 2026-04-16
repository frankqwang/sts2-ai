#!/usr/bin/env python3
"""Train a lightweight campfire-choice model from the cleaned Skada dataset."""

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


class CampfireDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate_rows(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    features = torch.tensor([row["feature_vector"] for row in batch], dtype=torch.float32)
    char_ids = torch.tensor([row["character_idx"] for row in batch], dtype=torch.long)
    labels = torch.tensor([row["label_idx"] for row in batch], dtype=torch.long)
    return {
        "features": features,
        "char_ids": char_ids,
        "labels": labels,
    }


class CampfireChoiceModel(nn.Module):
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
        x = torch.cat([features, self.char_embed(char_ids)], dim=-1)
        return self.mlp(x)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _prepare_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], list[str], list[str]]:
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
        "rest_value_proxy",
    ]

    char_vocab = {"<unk>": 0}
    label_vocab: dict[str, int] = {}

    def vocab_index(vocab: dict[str, int], value: str) -> int:
        if value not in vocab:
            vocab[value] = len(vocab)
        return vocab[value]

    prepared = []
    for row in rows:
        character = str(row.get("character") or "UNKNOWN")
        label = str(row.get("campfire_choice") or "UNKNOWN")
        row["character_idx"] = vocab_index(char_vocab, character)
        row["label_idx"] = vocab_index(label_vocab, label)
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
            _safe_float(row.get("rest_value_proxy"), 0.5),
        ]
        prepared.append(row)

    label_names = [label for label, _ in sorted(label_vocab.items(), key=lambda item: item[1])]
    vocabs = {"character": char_vocab, "label": label_vocab}
    return prepared, vocabs, feature_names, label_names


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


@dataclass
class EvalResult:
    loss: float
    top1_acc: float
    majority_acc: float
    confusion: dict[str, dict[str, int]]


def evaluate(
    model: CampfireChoiceModel,
    loader: DataLoader,
    device: torch.device,
    majority_idx: int,
    label_names: list[str],
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    majority_correct = 0
    confusion: dict[str, dict[str, int]] = {}
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

            for actual_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
                actual = label_names[actual_idx]
                pred = label_names[pred_idx]
                confusion.setdefault(actual, {})
                confusion[actual][pred] = confusion[actual].get(pred, 0) + 1

    return EvalResult(
        loss=total_loss / max(1, total),
        top1_acc=correct / max(1, total),
        majority_acc=majority_correct / max(1, total),
        confusion=confusion,
    )


def _collect_examples(
    model: CampfireChoiceModel,
    rows: list[dict[str, Any]],
    device: torch.device,
    majority_idx: int,
    label_names: list[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            features = torch.tensor([row["feature_vector"]], dtype=torch.float32, device=device)
            char_ids = torch.tensor([row["character_idx"]], dtype=torch.long, device=device)
            logits = model(features, char_ids)
            pred_idx = int(logits.argmax(dim=-1).item())
            label_idx = int(row["label_idx"])
            if pred_idx == label_idx and majority_idx != label_idx:
                examples.append({
                    "sample_id": row["sample_id"],
                    "floor": row["floor"],
                    "campfire_choice": row["campfire_choice"],
                    "predicted_choice": label_names[pred_idx],
                    "majority_baseline_choice": label_names[majority_idx],
                    "hp_before": row.get("hp_before"),
                    "upgradeable_count": row.get("upgradeable_count"),
                    "rest_value_proxy": row.get("rest_value_proxy"),
                    "smith_targets": row.get("smith_targets", []),
                })
                if len(examples) >= limit:
                    break
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Skada campfire choice model")
    parser.add_argument("--data", required=True, help="JSONL dataset from build_campfire_dataset.py")
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
    prepared_rows, vocabs, feature_names, label_names = _prepare_rows(rows)
    train_rows, val_rows = _split_rows(prepared_rows, args.val_ratio, args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else data_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    majority_idx = max(
        range(len(label_names)),
        key=lambda idx: sum(1 for row in train_rows if int(row["label_idx"]) == idx),
    )
    label_counts = {
        idx: max(1, sum(1 for row in train_rows if int(row["label_idx"]) == idx))
        for idx in range(len(label_names))
    }

    train_loader = DataLoader(CampfireDataset(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_rows)
    val_loader = DataLoader(CampfireDataset(val_rows), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_rows)

    device = torch.device(args.device)
    model = CampfireChoiceModel(
        num_characters=len(vocabs["character"]),
        num_features=len(feature_names),
        num_labels=len(label_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = torch.tensor(
        [len(train_rows) / (len(label_names) * label_counts[idx]) for idx in range(len(label_names))],
        dtype=torch.float32,
        device=device,
    )

    best_val_acc = -1.0
    best_metrics: dict[str, Any] = {}
    best_path = output_dir / "campfire_choice_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            char_ids = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features, char_ids)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            total += int(labels.size(0))
            correct += int((logits.argmax(dim=-1) == labels).sum().item())

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_result = evaluate(model, val_loader, device, majority_idx, label_names)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_top1={train_acc:.4f} "
            f"val_loss={val_result.loss:.4f} val_top1={val_result.top1_acc:.4f} "
            f"val_majority_baseline={val_result.majority_acc:.4f}"
        )

        if val_result.top1_acc > best_val_acc:
            best_val_acc = val_result.top1_acc
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1_acc": train_acc,
                "val_loss": val_result.loss,
                "val_top1_acc": val_result.top1_acc,
                "val_majority_baseline_top1_acc": val_result.majority_acc,
                "num_train_rows": len(train_rows),
                "num_val_rows": len(val_rows),
                "num_characters": len(vocabs["character"]),
                "num_labels": len(label_names),
                "feature_names": feature_names,
                "label_names": label_names,
                "majority_label": label_names[majority_idx],
                "confusion": val_result.confusion,
                "data": str(data_path),
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocabs": vocabs,
                    "feature_names": feature_names,
                    "label_names": label_names,
                    "metrics": best_metrics,
                },
                best_path,
            )

    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    best_metrics = dict(best_state["metrics"])
    examples = _collect_examples(model, val_rows, device, majority_idx, label_names, limit=10)
    examples_path = output_dir / "campfire_choice_examples.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    config_path = output_dir / "train_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    best_metrics["examples_path"] = str(examples_path)
    best_metrics["config_path"] = str(config_path)
    metrics_path = output_dir / "campfire_choice_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))
    print(f"saved_model={best_path}")
    print(f"saved_metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
