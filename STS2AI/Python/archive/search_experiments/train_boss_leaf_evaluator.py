#!/usr/bin/env python3
from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from boss_leaf_evaluator import (
    DEFAULT_LEAF_SCORE_TARGET,
    BossLeafEvaluator,
    LEAF_DATASET_SCHEMA_VERSION,
    LEAF_SCORE_SOFTCLIP_TEMPERATURE,
    LEAF_SCORE_TARGETS,
    LEAF_SCORE_V1_COEFFICIENTS,
    LeafDataset,
    LeafSample,
    MlpLeafEvaluator,
    bucketed_metric_rows,
    collate_leaf_batch,
    collate_signature_batch,
    compute_near_win_recall,
    expected_calibration_error,
    metric_bundle,
    normalize_score_target,
    outputs_to_score,
    pairwise_group_accuracy,
    vocab_snapshot_checksum,
)
from vocab import load_vocab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_boss_leaf_evaluator")
RANKING_PRIMARY_ORDER = ("pair_acc", "near_win_recall", "win_ece", "damage_corr")


@dataclass
class SplitData:
    train: list[LeafSample]
    holdout: list[LeafSample]
    train_parent_ids: list[str]
    holdout_parent_ids: list[str]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _groupwise_split(samples: list[LeafSample], holdout_frac: float, seed: int) -> SplitData:
    grouped: dict[str, list[LeafSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.parent_id, []).append(sample)
    group_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    n_holdout = max(1, int(round(len(group_ids) * holdout_frac)))
    holdout_ids = set(group_ids[:n_holdout])
    train: list[LeafSample] = []
    holdout: list[LeafSample] = []
    for group_id, items in grouped.items():
        if group_id in holdout_ids:
            holdout.extend(items)
        else:
            train.extend(items)
    train_ids = sorted(group_id for group_id in grouped.keys() if group_id not in holdout_ids)
    return SplitData(train=train, holdout=holdout, train_parent_ids=train_ids, holdout_parent_ids=sorted(holdout_ids))


def _split_from_parent_ids(samples: list[LeafSample], holdout_parent_ids: set[str]) -> SplitData:
    grouped: dict[str, list[LeafSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.parent_id, []).append(sample)
    train: list[LeafSample] = []
    holdout: list[LeafSample] = []
    for group_id, items in grouped.items():
        if group_id in holdout_parent_ids:
            holdout.extend(items)
        else:
            train.extend(items)
    train_ids = sorted(group_id for group_id in grouped.keys() if group_id not in holdout_parent_ids)
    return SplitData(train=train, holdout=holdout, train_parent_ids=train_ids, holdout_parent_ids=sorted(holdout_parent_ids))


def _resolve_split(samples: list[LeafSample], *, holdout_frac: float, seed: int, split_file: Path | None) -> tuple[SplitData, dict[str, Any], Path | None]:
    if split_file is not None and split_file.exists():
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        holdout_parent_ids = {str(item) for item in (payload.get("holdout_parent_ids") or [])}
        split = _split_from_parent_ids(samples, holdout_parent_ids)
        metadata = dict(payload)
        metadata["loaded_from"] = str(split_file)
        return split, metadata, split_file

    split = _groupwise_split(samples, holdout_frac, seed)
    metadata = {
        "schema_version": LEAF_DATASET_SCHEMA_VERSION,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "train_parent_ids": split.train_parent_ids,
        "holdout_parent_ids": split.holdout_parent_ids,
        "n_train_groups": len(split.train_parent_ids),
        "n_holdout_groups": len(split.holdout_parent_ids),
    }
    if split_file is not None:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        metadata["saved_to"] = str(split_file)
    return split, metadata, split_file


def _load_hard_case_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "path": None,
            "parent_ids": set(),
            "seed_ids": set(),
            "entries": [],
            "tag_counts": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") or []
    parent_ids: set[str] = set()
    seed_ids: set[str] = set()
    tag_counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        parent_id = str(entry.get("parent_id") or "").strip()
        seed = str(entry.get("seed") or "").strip()
        if parent_id:
            parent_ids.add(parent_id)
        if seed:
            seed_ids.add(seed)
        for tag in entry.get("tags") or []:
            key = str(tag or "").strip()
            if key:
                tag_counts[key] = tag_counts.get(key, 0) + 1
    return {
        "path": str(path),
        "parent_ids": parent_ids,
        "seed_ids": seed_ids,
        "entries": entries,
        "tag_counts": tag_counts,
    }


def _oversample_hard_cases(
    samples: list[LeafSample],
    *,
    manifest: dict[str, Any],
    oversample_factor: float,
) -> tuple[list[LeafSample], dict[str, Any]]:
    if not samples or oversample_factor <= 1.0:
        return list(samples), {"matched_samples": 0, "expanded_count": len(samples)}
    parent_ids = manifest.get("parent_ids") or set()
    seed_ids = manifest.get("seed_ids") or set()
    repeats = max(0, int(round(float(oversample_factor))) - 1)
    if repeats <= 0 or (not parent_ids and not seed_ids):
        return list(samples), {"matched_samples": 0, "expanded_count": len(samples)}
    expanded: list[LeafSample] = []
    matched = 0
    for sample in samples:
        expanded.append(sample)
        is_hard_case = sample.parent_id in parent_ids or str(sample.row.get("seed") or "") in seed_ids
        if not is_hard_case:
            continue
        matched += 1
        for _ in range(repeats):
            expanded.append(sample)
    return expanded, {"matched_samples": matched, "expanded_count": len(expanded)}


def _ranking_primary_view(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "pair_acc": float(metrics.get("pair_acc", 0.0) or 0.0),
        "near_win_recall": float(metrics.get("near_win_recall", 0.0) or 0.0),
        "win_ece": float(metrics.get("win_ece", 0.0) or 0.0),
        "damage_corr": float(metrics.get("damage_corr", 0.0) or 0.0),
    }


def _ranking_sort_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    primary = _ranking_primary_view(metrics)
    return (
        primary["pair_acc"],
        primary["near_win_recall"],
        -primary["win_ece"],
        primary["damage_corr"],
    )


def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _iter_minibatches(samples: list[LeafSample], batch_size: int, rng: random.Random) -> list[list[LeafSample]]:
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    return [[samples[idx] for idx in indices[start : start + batch_size]] for start in range(0, len(indices), batch_size)]


def _pairwise_ranking_loss(parent_ids: list[str], predicted_score: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
    grouped: dict[str, list[int]] = {}
    for index, parent_id in enumerate(parent_ids):
        grouped.setdefault(parent_id, []).append(index)
    terms: list[torch.Tensor] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                gap = target_score[left] - target_score[right]
                if torch.abs(gap) < 1e-6:
                    continue
                sign = torch.sign(gap)
                terms.append(F.softplus(-sign * (predicted_score[left] - predicted_score[right])))
    if not terms:
        return predicted_score.new_zeros(())
    return torch.stack(terms).mean()


def _forward_model(
    model: torch.nn.Module,
    samples: list[LeafSample],
    *,
    model_type: str,
    device: torch.device,
    max_tokens: int,
    score_target: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if model_type == "mlp":
        batch = collate_signature_batch(samples, score_target=score_target)
        outputs = model(batch["features"].to(device))
    else:
        batch = collate_leaf_batch(samples, max_tokens=max_tokens, score_target=score_target)
        outputs = model(
            batch["token_types"].to(device),
            batch["card_ids"].to(device),
            batch["enemy_ids"].to(device),
            batch["aux"].to(device),
            batch["aux_kind"].to(device),
            batch["attention_mask"].to(device),
        )
    return outputs, {key: value.to(device) for key, value in batch.items()}


def _evaluate_samples(
    model: torch.nn.Module,
    samples: list[LeafSample],
    *,
    model_type: str,
    device: torch.device,
    max_tokens: int,
    score_target: str,
) -> dict[str, Any]:
    if not samples:
        return {"score_corr": 0.0, "damage_corr": 0.0, "pair_acc": 0.0, "near_win_recall": 0.0, "win_ece": 0.0, "bucket_metrics": []}
    model.eval()
    with torch.no_grad():
        outputs, batch = _forward_model(
            model,
            samples,
            model_type=model_type,
            device=device,
            max_tokens=max_tokens,
            score_target=score_target,
        )
        pred_score = outputs_to_score(outputs, score_target=score_target).detach().cpu().numpy()
        pred_damage = outputs["boss_damage_ratio"].detach().cpu().numpy()
        pred_win = outputs["win_prob"].detach().cpu().numpy()
        target_score = batch["target_score"].detach().cpu().numpy()
        target_damage = batch["target_damage"].detach().cpu().numpy()
        target_win = batch["target_win"].detach().cpu().numpy()
    metrics = metric_bundle(
        samples,
        predicted_score=pred_score,
        predicted_damage=pred_damage,
        predicted_win=pred_win,
        target_score=target_score,
        target_damage=target_damage,
        target_win=target_win,
    )
    metrics["bucket_metrics"] = bucketed_metric_rows(
        samples,
        pred_score,
        predicted_damage=pred_damage,
        predicted_win=pred_win,
        target_score=target_score,
        target_damage=target_damage,
        target_win=target_win,
    )
    return metrics


def _write_bucket_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bucket", "count", "score_corr", "damage_corr", "pair_acc", "near_win_recall", "win_ece", "sibling_pair_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> int:
    _set_seed(args.seed)
    score_target = normalize_score_target(getattr(args, "score_target", DEFAULT_LEAF_SCORE_TARGET))
    dataset = LeafDataset.from_jsonl(args.dataset)
    if not dataset.samples:
        logger.error("No samples loaded from %s", args.dataset)
        return 1
    split_file = Path(args.split_file) if getattr(args, "split_file", None) else Path(args.output).with_suffix(".split.json")
    split, split_metadata, split_file = _resolve_split(
        dataset.samples,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        split_file=split_file,
    )
    hard_case_path = Path(args.hard_case_manifest) if getattr(args, "hard_case_manifest", None) else None
    hard_case_manifest = _load_hard_case_manifest(hard_case_path)
    train_samples, oversample_stats = _oversample_hard_cases(
        split.train,
        manifest=hard_case_manifest,
        oversample_factor=float(getattr(args, "hard_case_oversample", 3.0)),
    )
    logger.info(
        "Loaded %d samples (%d train / %d holdout, target=%s, hard_case_matches=%d expanded_train=%d)",
        len(dataset.samples),
        len(split.train),
        len(split.holdout),
        score_target,
        int(oversample_stats.get("matched_samples", 0) or 0),
        int(oversample_stats.get("expanded_count", len(train_samples)) or len(train_samples)),
    )

    model_type = str(args.model_type).strip().lower()
    vocab = load_vocab()
    device = torch.device("cpu")
    if model_type == "mlp":
        model: torch.nn.Module = MlpLeafEvaluator(hidden_dim=args.hidden_dim)
    else:
        model = BossLeafEvaluator(
            card_vocab_size=vocab.card_vocab_size,
            monster_vocab_size=vocab.monster_vocab_size,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_tokens=args.max_tokens,
        )
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed + 19)

    best_objective: tuple[float, float, float, float] | None = None
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for batch_samples in _iter_minibatches(train_samples, args.batch_size, rng):
            outputs, batch = _forward_model(
                model,
                batch_samples,
                model_type=model_type,
                device=device,
                max_tokens=args.max_tokens,
                score_target=score_target,
            )
            predicted_score = outputs_to_score(outputs, score_target=score_target)
            loss = (
                args.win_loss_weight * F.binary_cross_entropy(outputs["win_prob"], batch["target_win"])
                + args.damage_loss_weight * F.smooth_l1_loss(outputs["boss_damage_ratio"], batch["target_damage"])
                + args.hp_loss_weight * F.smooth_l1_loss(outputs["hp_loss_ratio"], batch["target_hp_loss"])
                + args.rank_loss_weight * _pairwise_ranking_loss([sample.parent_id for sample in batch_samples], predicted_score, batch["target_score"])
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        metrics = _evaluate_samples(
            model,
            split.holdout,
            model_type=model_type,
            device=device,
            max_tokens=args.max_tokens,
            score_target=score_target,
        )
        mean_train_loss = float(np.mean(losses)) if losses else 0.0
        objective = _ranking_sort_key(metrics)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": mean_train_loss,
                "holdout_pair_acc": metrics["pair_acc"],
                "holdout_near_win_recall": metrics["near_win_recall"],
                "holdout_win_ece": metrics["win_ece"],
                "holdout_damage_corr": metrics["damage_corr"],
                "holdout_score_corr": metrics["score_corr"],
                "selection_key": list(objective),
            }
        )
        if (epoch + 1) % max(1, args.log_every) == 0 or epoch == 0 or epoch + 1 == args.epochs:
            logger.info(
                "epoch %03d train=%.4f pair=%.3f recall=%.3f ece=%.3f dmg_corr=%.3f score_corr=%.3f",
                epoch + 1,
                mean_train_loss,
                metrics["pair_acc"],
                metrics["near_win_recall"],
                metrics["win_ece"],
                metrics["damage_corr"],
                metrics["score_corr"],
            )
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = _evaluate_samples(
        model,
        split.holdout,
        model_type=model_type,
        device=device,
        max_tokens=args.max_tokens,
        score_target=score_target,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_metrics_payload = {
        "score_corr": final_metrics["score_corr"],
        "damage_corr": final_metrics["damage_corr"],
        "pair_acc": final_metrics["pair_acc"],
        "near_win_recall": final_metrics["near_win_recall"],
        "win_ece": final_metrics["win_ece"],
    }
    ranking_primary_metrics = _ranking_primary_view(final_metrics)
    leaf_kinds = sorted({str(sample.row.get("leaf_kind") or "unknown") for sample in dataset.samples})
    input_features = ["state_signature"] if model_type == "mlp" else ["state_features.tokens", "state_features.player_aux", "state_features.boss_token"]
    runtime_leaf_value_target = normalize_score_target(
        getattr(args, "runtime_leaf_value_target", None)
        or ("search_value_softclip" if score_target == "score_v1_raw" else "score_v1_clipped")
    )
    payload = {
        "vocab_snapshot_checksum": vocab_snapshot_checksum(vocab.to_dict()),
        "schema_version": LEAF_DATASET_SCHEMA_VERSION,
        "model_type": model_type,
        "hidden_dim": args.hidden_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_tokens": args.max_tokens,
        "input_dim": len(collate_signature_batch(split.train[:1] or split.holdout[:1])["features"][0]) if model_type == "mlp" else len(collate_signature_batch(dataset.samples[:1])["features"][0]),
        "card_vocab_size": vocab.card_vocab_size,
        "monster_vocab_size": vocab.monster_vocab_size,
        "vocab_snapshot": {
            "format": "vocab.to_dict",
            "checksum": vocab_snapshot_checksum(vocab.to_dict()),
            "data": vocab.to_dict(),
        },
        "card_aux_dim": getattr(model, "card_aux_dim", None),
        "enemy_aux_dim": getattr(model, "enemy_aux_dim", None),
        "player_aux_dim": getattr(model, "player_aux_dim", None),
        "leaf_kind_supported": leaf_kinds,
        "input_features": input_features,
        "label_used_for_training": score_target,
        "score_target": score_target,
        "runtime_leaf_value_target": runtime_leaf_value_target,
        "score_softclip_temperature": LEAF_SCORE_SOFTCLIP_TEMPERATURE,
        "score_v1_coefficients": LEAF_SCORE_V1_COEFFICIENTS,
        "model_state_dict": model.state_dict(),
        "history": history,
        "metrics": final_metrics_payload,
        "ranking_primary_metrics": ranking_primary_metrics,
        "selection_policy": {
            "name": "ranking_first_lexicographic",
            "metric_priority": list(RANKING_PRIMARY_ORDER),
            "score_corr_is_diagnostic_only": True,
        },
        "best_metric": {
            "selection_key": list(best_objective) if best_objective is not None else [],
            "pair_acc": final_metrics["pair_acc"],
            "near_win_recall": final_metrics["near_win_recall"],
            "win_ece": final_metrics["win_ece"],
            "damage_corr": final_metrics["damage_corr"],
        },
        "bucket_metrics": final_metrics["bucket_metrics"],
        "training_dataset": str(args.dataset),
        "n_train": len(split.train),
        "n_train_effective": len(train_samples),
        "n_holdout": len(split.holdout),
        "split_file": str(split_file) if split_file is not None else None,
        "split_metadata": split_metadata,
        "hard_case_manifest": hard_case_manifest.get("path"),
        "hard_case_tag_counts": hard_case_manifest.get("tag_counts") or {},
        "hard_case_oversample": float(getattr(args, "hard_case_oversample", 3.0)),
        "hard_case_matched_samples": int(oversample_stats.get("matched_samples", 0) or 0),
        "hard_case_train_fraction": float(
            (int(oversample_stats.get("matched_samples", 0) or 0) / max(1, len(split.train)))
        ),
        "git_sha": _git_sha(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    torch.save(payload, output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": LEAF_DATASET_SCHEMA_VERSION,
                "model_type": model_type,
                "dataset": str(args.dataset),
                "split_file": str(split_file) if split_file is not None else None,
                "metrics": final_metrics_payload,
                "ranking_primary_metrics": ranking_primary_metrics,
                "bucket_metrics": final_metrics["bucket_metrics"],
                "best_metric": payload["best_metric"],
                "selection_policy": payload["selection_policy"],
                "score_target": score_target,
                "runtime_leaf_value_target": runtime_leaf_value_target,
                "score_v1_coefficients": LEAF_SCORE_V1_COEFFICIENTS,
                "score_softclip_temperature": LEAF_SCORE_SOFTCLIP_TEMPERATURE,
                "hard_case_manifest": hard_case_manifest.get("path"),
                "hard_case_tag_counts": hard_case_manifest.get("tag_counts") or {},
                "hard_case_oversample": float(getattr(args, "hard_case_oversample", 3.0)),
                "hard_case_matched_samples": int(oversample_stats.get("matched_samples", 0) or 0),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_bucket_csv(output_path.with_suffix(".bucket_metrics.csv"), final_metrics["bucket_metrics"])
    logger.info("Saved model to %s", output_path)
    logger.info("Saved metrics to %s", metrics_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train boss leaf evaluator")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-type", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--score-target", choices=list(LEAF_SCORE_TARGETS), default=DEFAULT_LEAF_SCORE_TARGET)
    parser.add_argument("--runtime-leaf-value-target", choices=list(LEAF_SCORE_TARGETS), default=None)
    parser.add_argument("--hard-case-manifest", type=str, default=None)
    parser.add_argument("--hard-case-oversample", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--win-loss-weight", type=float, default=1.0)
    parser.add_argument("--damage-loss-weight", type=float, default=1.0)
    parser.add_argument("--hp-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
