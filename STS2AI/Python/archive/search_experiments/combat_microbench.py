from __future__ import annotations

import _path_init  # noqa: F401  (adds tools/python/core to sys.path)

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from combat_teacher_common import (
    COMBAT_MICROBENCH_SCHEMA_VERSION,
    BaselineCombatPolicy,
    load_baseline_combat_policy,
)
from combat_teacher_dataset import (
    CombatTeacherSample,
    STRICT_MOTIF_REGRET_MARGIN,
    dedupe_samples_by_id,
    load_combat_teacher_samples,
    sample_metric_applicable,
)


class SamplePolicy(Protocol):
    def choose_action_index(self, sample: CombatTeacherSample) -> int: ...

    @property
    def name(self) -> str: ...


@dataclass(slots=True)
class BaselineSamplePolicy:
    baseline_policy: BaselineCombatPolicy
    name: str = "baseline"

    def choose_action_index(self, sample: CombatTeacherSample) -> int:
        scored = self.baseline_policy.score(sample.state, sample.legal_actions)
        return int(scored["best_index"])


@dataclass(slots=True)
class TeacherSamplePolicy:
    network: Any
    vocab: Any
    device: Any
    lethal_logit_blend_alpha: float = 0.0
    name: str = "teacher"

    def choose_action_index(self, sample: CombatTeacherSample) -> int:
        from combat_nn import build_combat_action_features, build_combat_features
        import numpy as np
        import torch

        sf = build_combat_features(sample.state, self.vocab)
        af = build_combat_action_features(sample.state, sample.legal_actions, self.vocab)

        state_t: dict[str, torch.Tensor] = {}
        for key, value in sf.items():
            tensor = torch.tensor(value).unsqueeze(0)
            if value.dtype in (np.int64, np.int32):
                tensor = tensor.long()
            elif value.dtype == bool:
                tensor = tensor.bool()
            else:
                tensor = tensor.float()
            state_t[key] = tensor.to(self.device)

        action_t: dict[str, torch.Tensor] = {}
        for key, value in af.items():
            tensor = torch.tensor(value).unsqueeze(0)
            if value.dtype in (np.int64, np.int32):
                tensor = tensor.long()
            elif value.dtype == bool:
                tensor = tensor.bool()
            else:
                tensor = tensor.float()
            action_t[key] = tensor.to(self.device)

        with torch.no_grad():
            logits, _value, action_scores, _continuation = self.network.forward_teacher(state_t, action_t)
        action_count = len(sample.legal_actions)
        mask = action_t["action_mask"][0, :action_count]
        masked_scores = action_scores[0, :action_count].masked_fill(~mask, -1e9)
        alpha = float(max(0.0, self.lethal_logit_blend_alpha))
        lethal_sensitive = any(
            sample_metric_applicable(sample, motif)
            for motif in ("direct_lethal_first_action", "turn_lethal_no_end_turn", "missed_lethal")
        )
        if alpha > 0.0 and lethal_sensitive:
            masked_logits = logits[0, :action_count].masked_fill(~mask, -1e9)
            return int((masked_scores + alpha * masked_logits).argmax().item())
        return int(masked_scores.argmax().item())


def _load_teacher_policy(
    path: str | Path,
    *,
    lethal_logit_blend_alpha: float = 0.0,
) -> TeacherSamplePolicy:
    import torch

    from combat_nn import CombatPolicyValueNetwork
    from vocab import load_vocab

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model_state_dict") or checkpoint.get("mcts_model")
    if not isinstance(model_state, dict):
        raise ValueError(f"Teacher checkpoint missing model_state_dict/mcts_model: {path}")
    card_weight = model_state.get("entity_emb.card_embed.weight")
    action_proj = model_state.get("action_proj.weight")
    embed_dim = int(card_weight.shape[1]) if isinstance(card_weight, torch.Tensor) and card_weight.ndim == 2 else 32
    hidden_dim = int(action_proj.shape[0]) if isinstance(action_proj, torch.Tensor) and action_proj.ndim == 2 else 128
    vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CombatPolicyValueNetwork(vocab=vocab, embed_dim=embed_dim, hidden_dim=hidden_dim)
    current = network.state_dict()
    filtered = {
        key: value
        for key, value in model_state.items()
        if key in current and getattr(current[key], "shape", None) == getattr(value, "shape", None)
    }
    network.load_state_dict(filtered, strict=False)
    network.to(device).eval()
    name = "teacher"
    if lethal_logit_blend_alpha > 0.0:
        name = f"teacher_selective_lethal_blend_{lethal_logit_blend_alpha:g}"
    return TeacherSamplePolicy(
        network=network,
        vocab=vocab,
        device=device,
        lethal_logit_blend_alpha=float(max(0.0, lethal_logit_blend_alpha)),
        name=name,
    )


def _sample_regret(sample: CombatTeacherSample, action_index: int) -> float:
    if 0 <= action_index < len(sample.per_action_regret):
        regret = float(sample.per_action_regret[action_index])
        if math.isfinite(regret) and regret < 1e6:
            return regret
        finite_regrets = [
            float(item)
            for item in sample.per_action_regret[:len(sample.legal_actions)]
            if math.isfinite(float(item)) and float(item) < 1e6
        ]
        if finite_regrets:
            return max(1.0, max(finite_regrets))
        return 1.0
    return float("inf")


def _sample_score(sample: CombatTeacherSample, action_index: int) -> float:
    if 0 <= action_index < len(sample.per_action_score):
        return float(sample.per_action_score[action_index])
    return float("-inf")


def _metric_applicable(sample: CombatTeacherSample, metric: str) -> bool:
    return sample_metric_applicable(sample, metric)


def evaluate_policy_on_samples(
    samples: list[CombatTeacherSample],
    policy: SamplePolicy,
) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    metrics = {
        "sample_count": len(samples),
        "avg_regret": 0.0,
        "missed_lethal_rate": 0.0,
        "direct_lethal_first_action_error_rate": 0.0,
        "turn_lethal_no_end_turn_error_rate": 0.0,
        "bash_before_strike_error_rate": 0.0,
        "bodyslam_before_block_error_rate": 0.0,
        "bad_end_turn_rate": 0.0,
        "potion_misuse_rate": 0.0,
    }
    if not samples:
        return {"policy": policy.name, "metrics": metrics, "samples": per_sample}

    regret_total = 0.0
    motif_errors = {
        "missed_lethal": [0, 0],
        "direct_lethal_first_action": [0, 0],
        "turn_lethal_no_end_turn": [0, 0],
        "bash_before_strike": [0, 0],
        "bodyslam_before_block": [0, 0],
        "bad_end_turn": [0, 0],
        "potion_misuse": [0, 0],
    }
    for sample in samples:
        chosen_index = int(policy.choose_action_index(sample))
        chosen_regret = _sample_regret(sample, chosen_index)
        chosen_score = _sample_score(sample, chosen_index)
        optimal_action = sample.legal_actions[sample.best_action_index] if 0 <= sample.best_action_index < len(sample.legal_actions) else {}
        chosen_action = sample.legal_actions[chosen_index] if 0 <= chosen_index < len(sample.legal_actions) else {}
        regret_total += 0.0 if chosen_regret == float("inf") else chosen_regret
        for motif in motif_errors:
            if _metric_applicable(sample, motif):
                motif_errors[motif][1] += 1
                if chosen_regret >= STRICT_MOTIF_REGRET_MARGIN:
                    motif_errors[motif][0] += 1
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "motif_labels": sample.motif_labels,
                "chosen_action": chosen_action,
                "optimal_action": optimal_action,
                "chosen_regret": chosen_regret,
                "chosen_score": chosen_score,
                "optimal_score": _sample_score(sample, sample.best_action_index),
            }
        )

    metrics["avg_regret"] = regret_total / max(1, len(samples))
    for motif, metric_name in (
        ("missed_lethal", "missed_lethal_rate"),
        ("direct_lethal_first_action", "direct_lethal_first_action_error_rate"),
        ("turn_lethal_no_end_turn", "turn_lethal_no_end_turn_error_rate"),
        ("bash_before_strike", "bash_before_strike_error_rate"),
        ("bodyslam_before_block", "bodyslam_before_block_error_rate"),
        ("bad_end_turn", "bad_end_turn_rate"),
        ("potion_misuse", "potion_misuse_rate"),
    ):
        errors, total = motif_errors[motif]
        metrics[metric_name] = float(errors / total) if total else 0.0
    return {"policy": policy.name, "metrics": metrics, "samples": per_sample}


def build_microbench_report(
    samples: list[CombatTeacherSample],
    *,
    baseline_checkpoint: str | Path | None = None,
    teacher_checkpoint: str | Path | None = None,
    teacher_lethal_logit_blend_alpha: float = 0.0,
) -> dict[str, Any]:
    unique_samples = dedupe_samples_by_id(samples)
    report = {
        "schema_version": COMBAT_MICROBENCH_SCHEMA_VERSION,
        "source_sample_count": len(samples),
        "sample_count": len(unique_samples),
        "comparisons": [],
    }
    report["comparisons"].append(
        {
            "policy": "solver",
            "metrics": {
                "sample_count": len(unique_samples),
                "avg_regret": 0.0,
                "missed_lethal_rate": 0.0,
                "direct_lethal_first_action_error_rate": 0.0,
                "turn_lethal_no_end_turn_error_rate": 0.0,
                "bash_before_strike_error_rate": 0.0,
                "bodyslam_before_block_error_rate": 0.0,
                "bad_end_turn_rate": 0.0,
                "potion_misuse_rate": 0.0,
            },
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "motif_labels": sample.motif_labels,
                    "chosen_action": sample.legal_actions[sample.best_action_index] if 0 <= sample.best_action_index < len(sample.legal_actions) else {},
                    "optimal_action": sample.legal_actions[sample.best_action_index] if 0 <= sample.best_action_index < len(sample.legal_actions) else {},
                    "chosen_regret": 0.0,
                    "chosen_score": _sample_score(sample, sample.best_action_index),
                    "optimal_score": _sample_score(sample, sample.best_action_index),
                }
                for sample in unique_samples
            ],
        }
    )
    if baseline_checkpoint:
        baseline = load_baseline_combat_policy(baseline_checkpoint)
        report["comparisons"].append(
            evaluate_policy_on_samples(unique_samples, BaselineSamplePolicy(baseline))
        )
    if teacher_checkpoint:
        teacher = _load_teacher_policy(
            teacher_checkpoint,
            lethal_logit_blend_alpha=teacher_lethal_logit_blend_alpha,
        )
        report["comparisons"].append(
            evaluate_policy_on_samples(unique_samples, teacher)
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Combat microbench V1")
    parser.add_argument("--dataset", required=True, help="combat_teacher_dataset.v1 JSONL")
    parser.add_argument("--split", choices=["all", "train", "holdout"], default="all")
    parser.add_argument("--baseline-checkpoint", default=None, help="Optional combat baseline checkpoint")
    parser.add_argument("--teacher-checkpoint", default=None, help="Optional teacher checkpoint")
    parser.add_argument("--teacher-lethal-logit-blend-alpha", type=float, default=0.0, help="Optional logits blend used only on lethal-sensitive teacher samples")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    samples = load_combat_teacher_samples(args.dataset)
    if args.split != "all":
        samples = [sample for sample in samples if sample.split == args.split]
    report = build_microbench_report(
        samples,
        baseline_checkpoint=args.baseline_checkpoint,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_lethal_logit_blend_alpha=args.teacher_lethal_logit_blend_alpha,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
