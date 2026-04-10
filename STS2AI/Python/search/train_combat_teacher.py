from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from combat_nn import CombatPolicyValueNetwork, build_combat_action_features, build_combat_features
from combat_teacher_dataset import CombatTeacherSample, load_combat_teacher_samples, sample_metric_applicable
from vocab import load_vocab


def _safe_load_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None)
    }
    model.load_state_dict(filtered, strict=False)


def _load_teacher_init_network(
    checkpoint_path: str | Path,
    *,
    vocab,
    device: torch.device,
) -> CombatPolicyValueNetwork:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("mcts_model") or checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Combat checkpoint has no mcts_model/model_state_dict: {checkpoint_path}")
    card_weight = state_dict.get("entity_emb.card_embed.weight")
    action_proj = state_dict.get("action_proj.weight")
    embed_dim = int(card_weight.shape[1]) if isinstance(card_weight, torch.Tensor) and card_weight.ndim == 2 else 32
    hidden_dim = int(action_proj.shape[0]) if isinstance(action_proj, torch.Tensor) and action_proj.ndim == 2 else 128
    network = CombatPolicyValueNetwork(vocab=vocab, embed_dim=embed_dim, hidden_dim=hidden_dim)
    _safe_load_state_dict(network, state_dict)
    network.to(device)
    return network


class CombatTeacherTorchDataset(Dataset):
    def __init__(
        self,
        samples: list[CombatTeacherSample],
        *,
        vocab,
        sample_weights: list[float] | None = None,
        baseline_anchor_weights: list[float] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.vocab = vocab
        if sample_weights is None:
            sample_weights = [1.0 for _ in self.samples]
        self.sample_weights = [float(weight) for weight in sample_weights]
        if baseline_anchor_weights is None:
            baseline_anchor_weights = [1.0 for _ in self.samples]
        self.baseline_anchor_weights = [float(weight) for weight in baseline_anchor_weights]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        state_features = build_combat_features(sample.state, self.vocab)
        action_features = build_combat_action_features(sample.state, sample.legal_actions, self.vocab)
        action_count = len(sample.legal_actions)
        baseline_probs = np.zeros(action_features["action_mask"].shape[0], dtype=np.float32)
        baseline_probs[: min(action_count, len(sample.baseline_probs))] = np.asarray(sample.baseline_probs[:action_count], dtype=np.float32)
        # Regret padding MUST be finite and bounded — using 1e9 (or in-data 1e9
        # sentinels from the dataset builder) would blow up the ranking loss
        # weights via F.softplus(large_diff). Clamp to [0, 10] to mirror the
        # train_hybrid.py fix at line 3130. See docs/diagnostics/p2p3p4_sweep_20260406.md.
        regrets = np.full(action_features["action_mask"].shape[0], 10.0, dtype=np.float32)
        raw_regrets = np.asarray(sample.per_action_regret[:action_count], dtype=np.float32)
        raw_regrets = np.clip(np.nan_to_num(raw_regrets, nan=10.0, posinf=10.0, neginf=0.0), 0.0, 10.0)
        regrets[: min(action_count, len(sample.per_action_regret))] = raw_regrets
        continuation = np.asarray(
            [
                float(sample.continuation_targets.get("win_prob", 0.0)),
                float(sample.continuation_targets.get("expected_hp_loss", 0.0)),
                float(sample.continuation_targets.get("expected_potion_cost", 0.0)),
            ],
            dtype=np.float32,
        )
        return {
            "state_features": state_features,
            "action_features": action_features,
            "baseline_best_action_index": int(sample.baseline_best_action_index),
            "teacher_best_action_index": int(sample.best_action_index),
            "baseline_probs": baseline_probs,
            "regrets": regrets,
            "continuation_targets": continuation,
            "sample_weight": float(self.sample_weights[idx]) if idx < len(self.sample_weights) else 1.0,
            "baseline_anchor_weight": float(self.baseline_anchor_weights[idx]) if idx < len(self.baseline_anchor_weights) else 1.0,
        }


def _train_sample_weight(
    sample: CombatTeacherSample,
    *,
    missed_lethal_weight: float,
    direct_lethal_weight: float,
    simple_missed_lethal_extra_weight: float,
    regression_weight: float,
    baseline_regret_weight_scale: float,
) -> float:
    weight = 1.0
    if str(sample.source_bucket or "").strip().lower() == "motif_regression":
        weight *= float(max(1.0, regression_weight))
    if sample_metric_applicable(sample, "direct_lethal_first_action"):
        weight *= float(max(1.0, direct_lethal_weight))
    if sample_metric_applicable(sample, "missed_lethal"):
        weight *= float(max(1.0, missed_lethal_weight))
        if len(sample.legal_actions) <= 3:
            weight *= float(max(1.0, simple_missed_lethal_extra_weight))
    baseline_idx = int(sample.baseline_best_action_index)
    raw_baseline_regret = float(sample.per_action_regret[baseline_idx]) if 0 <= baseline_idx < len(sample.per_action_regret) else 0.0
    if not math.isfinite(raw_baseline_regret) or raw_baseline_regret >= 1e6:
        baseline_regret = 0.0
    else:
        baseline_regret = min(max(raw_baseline_regret, 0.0), 1.0)
    if baseline_regret_weight_scale > 0.0 and baseline_regret > 0.0:
        weight *= 1.0 + float(baseline_regret_weight_scale) * baseline_regret
    return float(max(1e-6, weight))


def _baseline_anchor_weight(
    sample: CombatTeacherSample,
    *,
    direct_lethal_baseline_anchor_weight: float,
) -> float:
    weight = 1.0
    if sample_metric_applicable(sample, "direct_lethal_first_action"):
        if int(sample.baseline_best_action_index) == int(sample.best_action_index):
            weight *= float(max(1.0, direct_lethal_baseline_anchor_weight))
    return float(max(1e-6, weight))


def _stack_batch(batch: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    # Use intersection of keys across all items to handle optional fields (e.g., deck_ids)
    state_keys = set(batch[0]["state_features"].keys())
    for item in batch[1:]:
        state_keys &= set(item["state_features"].keys())
    action_keys = set(batch[0]["action_features"].keys())
    for item in batch[1:]:
        action_keys &= set(item["action_features"].keys())
    state_t: dict[str, torch.Tensor] = {}
    for key in state_keys:
        arrays = [item["state_features"][key] for item in batch]
        stacked = np.stack(arrays, axis=0)
        tensor = torch.tensor(stacked)
        if stacked.dtype in (np.int64, np.int32):
            tensor = tensor.long()
        elif stacked.dtype == bool:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        state_t[key] = tensor.to(device)
    action_t: dict[str, torch.Tensor] = {}
    for key in action_keys:
        arrays = [item["action_features"][key] for item in batch]
        stacked = np.stack(arrays, axis=0)
        tensor = torch.tensor(stacked)
        if stacked.dtype in (np.int64, np.int32):
            tensor = tensor.long()
        elif stacked.dtype == bool:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        action_t[key] = tensor.to(device)

    return {
        "state_features": state_t,
        "action_features": action_t,
        "baseline_best_action_index": torch.tensor([item["baseline_best_action_index"] for item in batch], dtype=torch.long, device=device),
        "teacher_best_action_index": torch.tensor([item["teacher_best_action_index"] for item in batch], dtype=torch.long, device=device),
        "baseline_probs": torch.tensor(np.stack([item["baseline_probs"] for item in batch], axis=0), dtype=torch.float32, device=device),
        "regrets": torch.tensor(np.stack([item["regrets"] for item in batch], axis=0), dtype=torch.float32, device=device),
        "continuation_targets": torch.tensor(np.stack([item["continuation_targets"] for item in batch], axis=0), dtype=torch.float32, device=device),
        "sample_weight": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32, device=device),
        "baseline_anchor_weight": torch.tensor([item["baseline_anchor_weight"] for item in batch], dtype=torch.float32, device=device),
    }


def _regret_weighted_pairwise_ranking(
    action_scores: torch.Tensor,
    regrets: torch.Tensor,
    best_action_index: torch.Tensor,
    action_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []
    batch_size = action_scores.shape[0]
    for batch_idx in range(batch_size):
        best_idx = int(best_action_index[batch_idx].item())
        if best_idx < 0 or best_idx >= action_scores.shape[1]:
            continue
        best_score = action_scores[batch_idx, best_idx]
        valid_indices = torch.nonzero(action_mask[batch_idx], as_tuple=False).flatten().tolist()
        for other_idx in valid_indices:
            if other_idx == best_idx:
                continue
            weight = float(max(0.0, regrets[batch_idx, other_idx].item() - regrets[batch_idx, best_idx].item()))
            if weight <= 0.0:
                continue
            sample_weight = float(sample_weights[batch_idx].item()) if sample_weights is not None else 1.0
            losses.append(sample_weight * weight * F.softplus(action_scores[batch_idx, other_idx] - best_score))
    if not losses:
        return action_scores.new_tensor(0.0)
    return torch.stack(losses).mean()


@dataclass(slots=True)
class TrainMetrics:
    loss: float = 0.0
    baseline_policy_ce: float = 0.0
    teacher_best_action_ce: float = 0.0
    regret_weighted_ranking: float = 0.0
    continuation_value_regression: float = 0.0
    kl_to_reference_policy: float = 0.0
    score_kl_to_reference_policy: float = 0.0


def _masked_policy_kl_to_reference(
    masked_scores: torch.Tensor,
    reference_probs: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    ref = reference_probs * action_mask.float()
    ref = ref / ref.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(masked_scores, dim=-1)
    kl_terms = torch.where(
        ref > 0,
        ref * (ref.clamp_min(1e-8).log() - log_probs),
        torch.zeros_like(ref),
    )
    return kl_terms.sum(dim=-1)


def _run_epoch(
    network: CombatPolicyValueNetwork,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    baseline_ce_weight: float = 1.0,
    teacher_ce_weight: float = 1.0,
    ranking_weight: float = 1.0,
    continuation_weight: float = 1.0,
    kl_weight: float = 0.5,
    score_kl_weight: float = 0.0,
) -> TrainMetrics:
    train = optimizer is not None
    network.train(train)
    total = TrainMetrics()
    batches = 0
    for raw_batch in loader:
        batch = _stack_batch(raw_batch, device)
        logits, _value, action_scores, continuation = network.forward_teacher(
            batch["state_features"],
            batch["action_features"],
        )
        action_mask = batch["action_features"]["action_mask"]
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        masked_scores = action_scores.masked_fill(~action_mask, -1e9)
        baseline_ce_per_sample = F.cross_entropy(masked_logits, batch["baseline_best_action_index"], reduction="none")
        baseline_ce = (baseline_ce_per_sample * batch["baseline_anchor_weight"]).sum() / batch["baseline_anchor_weight"].sum().clamp_min(1e-8)
        teacher_ce_per_sample = F.cross_entropy(masked_scores, batch["teacher_best_action_index"], reduction="none")
        teacher_ce = (teacher_ce_per_sample * batch["sample_weight"]).sum() / batch["sample_weight"].sum().clamp_min(1e-8)
        # Defense-in-depth: clamp regrets to avoid numerical blowup in the
        # pairwise ranking loss. Matches train_hybrid.py line 3130.
        clamped_regrets = batch["regrets"].clamp(min=0.0, max=10.0)
        ranking = _regret_weighted_pairwise_ranking(
            masked_scores,
            clamped_regrets,
            batch["teacher_best_action_index"],
            action_mask,
            batch["sample_weight"],
        )
        continuation_reg = F.mse_loss(continuation, batch["continuation_targets"])
        reference_probs = batch["baseline_probs"]
        kl_per_sample = _masked_policy_kl_to_reference(masked_logits, reference_probs, action_mask)
        kl = (kl_per_sample * batch["baseline_anchor_weight"]).sum() / batch["baseline_anchor_weight"].sum().clamp_min(1e-8)
        score_kl_per_sample = _masked_policy_kl_to_reference(masked_scores, reference_probs, action_mask)
        score_kl = (score_kl_per_sample * batch["baseline_anchor_weight"]).sum() / batch["baseline_anchor_weight"].sum().clamp_min(1e-8)
        loss = (
            float(baseline_ce_weight) * baseline_ce
            + float(teacher_ce_weight) * teacher_ce
            + float(ranking_weight) * ranking
            + float(continuation_weight) * continuation_reg
            + float(kl_weight) * kl
            + float(score_kl_weight) * score_kl
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

        total.loss += float(loss.item())
        total.baseline_policy_ce += float(baseline_ce.item())
        total.teacher_best_action_ce += float(teacher_ce.item())
        total.regret_weighted_ranking += float(ranking.item())
        total.continuation_value_regression += float(continuation_reg.item())
        total.kl_to_reference_policy += float(kl.item())
        total.score_kl_to_reference_policy += float(score_kl.item())
        batches += 1

    if batches <= 0:
        return total
    return TrainMetrics(
        loss=total.loss / batches,
        baseline_policy_ce=total.baseline_policy_ce / batches,
        teacher_best_action_ce=total.teacher_best_action_ce / batches,
        regret_weighted_ranking=total.regret_weighted_ranking / batches,
        continuation_value_regression=total.continuation_value_regression / batches,
        kl_to_reference_policy=total.kl_to_reference_policy / batches,
        score_kl_to_reference_policy=total.score_kl_to_reference_policy / batches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train combat teacher stack V1 from combat_teacher_dataset.v1")
    parser.add_argument("--dataset", required=True, help="combat_teacher_dataset.v1 JSONL")
    parser.add_argument("--combat-checkpoint", required=True, help="Combat checkpoint used to initialize the teacher model")
    parser.add_argument("--output-dir", default="artifacts/combat_teacher", help="Output directory root")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--baseline-ce-weight", type=float, default=1.0)
    parser.add_argument("--teacher-ce-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--continuation-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--score-kl-weight", type=float, default=0.25)
    parser.add_argument("--missed-lethal-sample-weight", type=float, default=1.0)
    parser.add_argument("--direct-lethal-sample-weight", type=float, default=1.0)
    parser.add_argument("--direct-lethal-baseline-anchor-weight", type=float, default=1.0)
    parser.add_argument("--simple-missed-lethal-extra-weight", type=float, default=1.0)
    parser.add_argument("--regression-sample-weight", type=float, default=1.0)
    parser.add_argument("--baseline-regret-weight-scale", type=float, default=0.0)
    parser.add_argument("--microbench-output", default=None, help="Optional path for post-train microbench JSON")
    args = parser.parse_args()

    vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_combat_teacher_samples(args.dataset)
    train_samples = [sample for sample in samples if sample.split != "holdout"]
    holdout_samples = [sample for sample in samples if sample.split == "holdout"]
    if not train_samples:
        raise ValueError("No train samples found in combat teacher dataset.")
    if not holdout_samples:
        holdout_samples = list(train_samples)

    network = _load_teacher_init_network(args.combat_checkpoint, vocab=vocab, device=device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_weights = [
        _train_sample_weight(
            sample,
            missed_lethal_weight=args.missed_lethal_sample_weight,
            direct_lethal_weight=args.direct_lethal_sample_weight,
            simple_missed_lethal_extra_weight=args.simple_missed_lethal_extra_weight,
            regression_weight=args.regression_sample_weight,
            baseline_regret_weight_scale=args.baseline_regret_weight_scale,
        )
        for sample in train_samples
    ]
    train_baseline_anchor_weights = [
        _baseline_anchor_weight(
            sample,
            direct_lethal_baseline_anchor_weight=args.direct_lethal_baseline_anchor_weight,
        )
        for sample in train_samples
    ]
    train_dataset = CombatTeacherTorchDataset(
        train_samples,
        vocab=vocab,
        sample_weights=train_weights,
        baseline_anchor_weights=train_baseline_anchor_weights,
    )
    if all(abs(weight - train_weights[0]) < 1e-9 for weight in train_weights):
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=True,
            collate_fn=list,
        )
    else:
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(train_weights, dtype=torch.double),
            num_samples=len(train_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, int(args.batch_size)),
            sampler=train_sampler,
            collate_fn=list,
        )
    holdout_loader = DataLoader(
        CombatTeacherTorchDataset(
            holdout_samples,
            vocab=vocab,
            baseline_anchor_weights=[
                _baseline_anchor_weight(
                    sample,
                    direct_lethal_baseline_anchor_weight=args.direct_lethal_baseline_anchor_weight,
                )
                for sample in holdout_samples
            ],
        ),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        collate_fn=list,
    )

    history: list[dict[str, Any]] = []
    for epoch in range(int(args.epochs)):
        train_metrics = _run_epoch(
            network,
            train_loader,
            optimizer=optimizer,
            device=device,
            baseline_ce_weight=args.baseline_ce_weight,
            teacher_ce_weight=args.teacher_ce_weight,
            ranking_weight=args.ranking_weight,
            continuation_weight=args.continuation_weight,
            kl_weight=args.kl_weight,
            score_kl_weight=args.score_kl_weight,
        )
        eval_metrics = _run_epoch(
            network,
            holdout_loader,
            optimizer=None,
            device=device,
            baseline_ce_weight=args.baseline_ce_weight,
            teacher_ce_weight=args.teacher_ce_weight,
            ranking_weight=args.ranking_weight,
            continuation_weight=args.continuation_weight,
            kl_weight=args.kl_weight,
            score_kl_weight=args.score_kl_weight,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": asdict(train_metrics),
                "holdout": asdict(eval_metrics),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / f"combat_teacher_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "combat_teacher_final.pt"
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "config": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "baseline_ce_weight": float(args.baseline_ce_weight),
                "teacher_ce_weight": float(args.teacher_ce_weight),
                "ranking_weight": float(args.ranking_weight),
                "continuation_weight": float(args.continuation_weight),
                "kl_weight": float(args.kl_weight),
                "score_kl_weight": float(args.score_kl_weight),
                "missed_lethal_sample_weight": float(args.missed_lethal_sample_weight),
                "direct_lethal_sample_weight": float(args.direct_lethal_sample_weight),
                "direct_lethal_baseline_anchor_weight": float(args.direct_lethal_baseline_anchor_weight),
                "simple_missed_lethal_extra_weight": float(args.simple_missed_lethal_extra_weight),
                "regression_sample_weight": float(args.regression_sample_weight),
                "dataset": str(args.dataset),
                "combat_checkpoint": str(args.combat_checkpoint),
            },
            "history": history,
        },
        checkpoint_path,
    )
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.microbench_output:
        from combat_microbench import build_microbench_report

        report = build_microbench_report(
            holdout_samples,
            baseline_checkpoint=args.combat_checkpoint,
            teacher_checkpoint=checkpoint_path,
        )
        Path(args.microbench_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.microbench_output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "checkpoint": str(checkpoint_path),
        "history": history,
        "train_samples": len(train_samples),
        "holdout_samples": len(holdout_samples),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
