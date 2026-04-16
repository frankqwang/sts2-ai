#!/usr/bin/env python3
"""Behavioral Cloning pre-training for the non-combat brain.

Loads expert trajectory data (rl_transition.jsonl from evaluate_ai.py)
and trains FullRunPolicyNetworkV2 via supervised cross-entropy loss
on the expert's action choices.

Usage:
    python train_bc_noncombat.py \
        --data STS2AI/Artifacts/expert_trajectories/champion_high_floor/derived/rl/rl_transition.jsonl \
        --resume STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt \
        --epochs 30 --lr 3e-4 --batch-size 64
"""

from __future__ import annotations


import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from core.vocab import load_vocab, Vocab
from network.state_features import build_structured_state, build_structured_actions, MAX_ACTIONS
from network.fullrun_policy import (
    FullRunPolicyNetworkV2,
    _structured_state_to_numpy_dict,
    _structured_actions_to_numpy_dict,
)
from constants import MAINLINE_CHECKPOINT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Non-combat screen types that the PPO brain handles
NON_COMBAT_SCREENS = {"map", "card_reward", "shop", "rest_site", "event",
                      "card_select", "hand_select", "treasure", "campfire"}
COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}


class ExpertTrajectoryDataset(Dataset):
    """Dataset of expert non-combat decisions from rl_transition.jsonl."""

    def __init__(self, jsonl_path: str | Path, vocab: Vocab, min_floor: int = 0):
        self.vocab = vocab
        self.samples: list[dict] = []
        skipped_combat = 0
        skipped_floor = 0
        skipped_encode = 0

        logger.info("Loading expert trajectories from %s ...", jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                step = json.loads(line)
                state_type = step.get("state_type", "")

                # Skip combat steps — we only want non-combat decisions
                if state_type in COMBAT_SCREENS:
                    skipped_combat += 1
                    continue

                # Skip low-floor steps if requested
                if step.get("floor", 0) < min_floor:
                    skipped_floor += 1
                    continue

                # We need state, candidate_actions, and the chosen action
                state = step.get("state")
                candidates = step.get("candidate_actions", [])
                action = step.get("action")
                if not state or not candidates or not action:
                    continue

                # Find which candidate index was chosen
                action_idx = self._find_action_index(action, candidates)
                if action_idx < 0:
                    continue

                # Try encoding — skip if encoder fails
                try:
                    ss = build_structured_state(state, vocab)
                    sa = build_structured_actions(state, candidates, vocab)
                    state_np = _structured_state_to_numpy_dict(ss)
                    actions_np = _structured_actions_to_numpy_dict(sa)
                except Exception:
                    skipped_encode += 1
                    continue

                self.samples.append({
                    "state": state_np,
                    "actions": actions_np,
                    "action_idx": action_idx,
                    "screen_type": state_type,
                    "floor": step.get("floor", 0),
                })

        logger.info(
            "Loaded %d non-combat expert samples (skipped: %d combat, %d low-floor, %d encode-fail)",
            len(self.samples), skipped_combat, skipped_floor, skipped_encode,
        )

        # Log screen type distribution
        from collections import Counter
        screen_counts = Counter(s["screen_type"] for s in self.samples)
        for st, count in screen_counts.most_common():
            logger.info("  %s: %d samples", st, count)

    def _find_action_index(self, chosen: dict, candidates: list[dict]) -> int:
        """Find the index of the chosen action in candidates list."""
        chosen_action = chosen.get("action", "")
        chosen_index = chosen.get("index", -1)
        chosen_label = chosen.get("label", "")

        for i, cand in enumerate(candidates):
            if i >= MAX_ACTIONS:
                break
            cand_action = cand.get("action", "")
            cand_index = cand.get("index", -1)
            cand_label = cand.get("label", "")

            # Match by action name + index (most reliable)
            if cand_action == chosen_action and cand_index == chosen_index:
                return i
            # Fallback: match by label
            if chosen_label and cand_label == chosen_label:
                return i

        return -1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        state_t = {}
        for k, v in sample["state"].items():
            t = torch.from_numpy(v)
            if "ids" in k or "idx" in k or "types" in k or "count" in k:
                state_t[k] = t.long()
            elif "mask" in k:
                state_t[k] = t.bool()
            else:
                state_t[k] = t.float()

        actions_t = {}
        for k, v in sample["actions"].items():
            t = torch.from_numpy(v)
            if "mask" in k:
                actions_t[k] = t.bool()
            else:
                actions_t[k] = t.long()

        return {
            **{f"s_{k}": v for k, v in state_t.items()},
            **{f"a_{k}": v for k, v in actions_t.items()},
            "action_idx": torch.tensor(sample["action_idx"], dtype=torch.long),
        }


def collate_bc(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate a list of samples into batched tensors."""
    result = {}
    keys = batch[0].keys()
    for key in keys:
        tensors = [b[key] for b in batch]
        result[key] = torch.stack(tensors, dim=0)
    return result


def train_bc(
    model: FullRunPolicyNetworkV2,
    dataset: ExpertTrajectoryDataset,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    device: torch.device = torch.device("cpu"),
    output_dir: Path = Path("STS2AI/Artifacts/bc_pretrain"),
) -> Path:
    """Train the non-combat brain via behavioral cloning."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Train/val split
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    val_size = max(1, int(n * val_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_bc, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_bc, num_workers=0)

    logger.info("Train: %d samples, Val: %d samples, Batch: %d", len(train_indices), val_size, batch_size)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_acc = 0.0
    best_path = output_dir / "bc_best.pt"

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            state_t = {k[2:]: batch[k].to(device) for k in batch if k.startswith("s_")}
            action_t = {k[2:]: batch[k].to(device) for k in batch if k.startswith("a_")}
            target = batch["action_idx"].to(device)

            # Forward: get action logits (already masked inside model)
            logits, _values, _dq, _br, _adv = model.forward(state_t, action_t)
            # logits shape: (B, MAX_ACTIONS) — already masked to -inf for illegal

            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * target.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)

        scheduler.step()

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                state_t = {k[2:]: batch[k].to(device) for k in batch if k.startswith("s_")}
                action_t = {k[2:]: batch[k].to(device) for k in batch if k.startswith("a_")}
                target = batch["action_idx"].to(device)

                logits, _values, _dq, _br, _adv = model.forward(state_t, action_t)

                loss = F.cross_entropy(logits, target)
                val_loss += loss.item() * target.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.2f%% | val_loss=%.4f val_acc=%.2f%% | lr=%.6f",
            epoch, epochs, avg_train_loss, train_acc * 100, avg_val_loss, val_acc * 100,
            scheduler.get_last_lr()[0],
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "ppo_model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
            }, best_path)
            logger.info("  → New best val_acc=%.2f%%, saved to %s", val_acc * 100, best_path)

    # Save final
    final_path = output_dir / "bc_final.pt"
    torch.save({
        "ppo_model": model.state_dict(),
        "epoch": epochs,
        "val_acc": val_acc,
        "val_loss": avg_val_loss,
        "train_acc": train_acc,
    }, final_path)
    logger.info("Final model saved to %s", final_path)
    logger.info("Best val_acc=%.2f%% at %s", best_val_acc * 100, best_path)

    return best_path


def main() -> int:
    parser = argparse.ArgumentParser(description="BC pre-training for non-combat brain")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to rl_transition.jsonl from evaluate_ai.py trajectory export")
    parser.add_argument("--resume", type=str, default=str(MAINLINE_CHECKPOINT),
                        help="Checkpoint to initialize weights from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-floor", type=int, default=0,
                        help="Only use expert steps from floor >= this value")
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--output-dir", type=str, default="STS2AI/Artifacts/bc_pretrain")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    vocab = load_vocab()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("Device: %s", device)

    # Load dataset
    dataset = ExpertTrajectoryDataset(args.data, vocab, min_floor=args.min_floor)
    if len(dataset) == 0:
        logger.error("No samples loaded. Check data path and filters.")
        return 1

    # Build model
    use_symbolic = False
    symbolic_proj_dim = 16

    # Auto-detect from checkpoint
    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            ppo_state = ckpt.get("ppo_model", {})
            if any("symbolic_head" in k for k in ppo_state):
                use_symbolic = True
                logger.info("Auto-detected SymbolicFeaturesHead in checkpoint")
        except Exception as e:
            logger.warning("Could not inspect checkpoint: %s", e)

    model = FullRunPolicyNetworkV2(
        vocab=vocab,
        embed_dim=args.embed_dim,
        use_symbolic_features=use_symbolic,
        symbolic_proj_dim=symbolic_proj_dim,
    )

    # Load pretrained weights
    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            ppo_state = ckpt.get("ppo_model", {})
            if ppo_state:
                missing, unexpected = model.load_state_dict(ppo_state, strict=False)
                logger.info("Loaded PPO weights from %s (missing=%d, unexpected=%d)",
                           args.resume, len(missing), len(unexpected))
            else:
                logger.warning("No ppo_model in checkpoint, starting from scratch")
        except Exception as e:
            logger.warning("Could not load checkpoint: %s", e)

    # Train
    output_dir = Path(args.output_dir)
    best_path = train_bc(
        model, dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
    )

    logger.info("BC pre-training complete. Best model: %s", best_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
