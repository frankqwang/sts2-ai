"""Skada 离线非战斗训练入口。

纯监督:
  1. 从 skada jsonl(或 sqlite,未来) 读 TrainingSample
  2. 批次 forward UnifiedNet(decision_domain != combat)
  3. OfflineBCLoss(CE + value regression)反传
  4. 按 epoch 保存 checkpoint,输出 metrics

不依赖 sim,不依赖 env,单机可运行。

典型用法:
    python -m networkV2.s6_training.train_noncombat_offline \
        --jsonl-dir data/skada/runs/details \
        --output-dir ../Artifacts/checkpoints/skada_bc_v1 \
        --epochs 10 --batch-size 64 --preset slim

再接 RL finetune:
    python -m networkV2.s6_training.train_full_run_v2 \
        --checkpoint ../Artifacts/checkpoints/skada_bc_v1/epoch_10.pt \
        ...
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.optim as optim

from networkV2.s1_schema.encounter_vocab import encounter_to_index
from networkV2.s5_net.network_config import preset_slim, preset_full, NetworkConfig
from networkV2.s5_net.unified_net import UnifiedNet
from networkV2.s6_training.batch import TrainingSample, collate_training_samples
from networkV2.s6_training.offline_bc_loss import OfflineBCLoss, OfflineBCLossConfig
from networkV2.s6_training.skada_offline_loader import (
    load_samples_from_jsonl_dir, load_samples_from_sqlite,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch iterator(按 decision_domain 分组,每批内 domain 一致,方便 per-domain policy head forward)
# ---------------------------------------------------------------------------

def _iter_batches_by_domain(
    samples: list[TrainingSample],
    batch_size: int,
    shuffle: bool = True,
):
    """按 domain 分桶,每批内 domain 一致。

    不同 domain 的 option 数、语义都不一样,混批会让 padding 浪费且 logits 维度混乱;
    同时 per-domain policy head 需要知道本批属于哪个 domain 才能路由。
    """
    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        d = s.banks.decision_domain or "unknown"
        buckets.setdefault(d, []).append(i)

    # 每 bucket 内 shuffle + 切 batch,最后全局再 shuffle 各 batch 的顺序(防 domain 顺序偏置)
    all_batches: list[tuple[str, list[int]]] = []
    for d, indices in buckets.items():
        if shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            all_batches.append((d, indices[start:start + batch_size]))
    if shuffle:
        random.shuffle(all_batches)
    for d, idx_list in all_batches:
        yield d, [samples[i] for i in idx_list]


def _move_batched_to_device(batched, device):
    """把 BatchedBanks + 各 target tensor 都移到 device。"""
    for bank in batched.banks.values():
        bank.numeric = bank.numeric.to(device)
        bank.type_ids = bank.type_ids.to(device)
        bank.ts_ids = bank.ts_ids.to(device)
        bank.mask = bank.mask.to(device)
    for attr in (
        "action_indices", "old_log_probs", "advantages", "returns",
        "fight_win_targets", "hp_loss_targets", "survival_targets",
        "turn_damage_targets", "leaf_targets", "transition_risk_targets",
        "resource_retention_targets", "boss_readiness_targets",
        "resource_health_targets", "deck_quality_targets", "sample_weights",
    ):
        val = getattr(batched, attr, None)
        if val is not None:
            setattr(batched, attr, val.to(device))
    enc_idx = getattr(batched, "encounter_indices", None)
    if enc_idx is not None:
        batched.encounter_indices = enc_idx.to(device)
    return batched


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_offline(
    samples: list[TrainingSample],
    *,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    preset: str = "slim",
    device: str = "auto",
    save_every: int = 1,
    log_every: int = 20,
    seed: int = 0,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"device: {device}")

    # Config + network
    cfg: NetworkConfig = preset_full() if preset == "full" else preset_slim()
    # 离线训非战斗不需要战斗侧的 encounter conditioning 梯度,
    # 保持默认设置(如 preset 打开则继续开)。
    net = UnifiedNet(config=cfg).to(device)
    loss_fn = OfflineBCLoss(OfflineBCLossConfig()).to(device)

    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"model params: {n_params/1e6:.2f}M (preset={preset})")

    # Filter: 只保留非战斗样本(skada loader 应该全是非战斗,但保险起见)
    samples = [s for s in samples if s.banks.decision_domain != "combat"]
    logger.info(f"training samples (non-combat): {len(samples)}")

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    epoch_metrics_log: list[dict] = []
    tracked_keys = (
        "bc_total_loss", "bc_policy_loss", "bc_vl_run_win",
        "bc_vl_boss_ready", "bc_vl_resource_health", "bc_vl_deck_quality",
        "bc_vl_exp_hp_loss", "bc_vl_exp_dmg_output", "bc_vl_floor_clear",
        "bc_top1_acc", "bc_entropy",
    )
    for epoch in range(1, epochs + 1):
        net.train()
        t0 = time.time()
        epoch_metrics = {k: 0.0 for k in tracked_keys}
        batches_by_domain: dict[str, int] = {}
        n_batches = 0
        n_nan_skips = 0

        for batch_domain, batch_samples in _iter_batches_by_domain(
            samples, batch_size, shuffle=True,
        ):
            if not batch_samples:
                continue
            batched = collate_training_samples(batch_samples)
            batched = _move_batched_to_device(batched, device)
            batched_banks_dict = {name: pb for name, pb in batched.banks.items()}

            # 本批次 domain 一致,直接按该 domain 路由 forward(走 per-domain policy head)
            output = net(
                batched_banks=batched_banks_dict,
                decision_domain=batch_domain,
                encounter_idx=getattr(batched, "encounter_indices", None),
            )

            total, metrics = loss_fn(
                output,
                action_indices=batched.action_indices,
                run_win_targets=batched.fight_win_targets,
                boss_readiness_targets=batched.boss_readiness_targets,
                resource_health_targets=batched.resource_health_targets,
                deck_quality_targets=batched.deck_quality_targets,
                # 字段复用:loader 把 run-level aggregate 填进了这 3 个字段
                expected_hp_loss_targets=batched.hp_loss_targets,
                expected_dmg_output_targets=batched.turn_damage_targets,
                floor_clear_targets=batched.survival_targets,
                sample_weights=batched.sample_weights,
            )

            if torch.isnan(total) or torch.isinf(total):
                logger.warning(f"step {global_step} domain={batch_domain}: NaN/Inf loss, skip")
                n_nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            n_batches += 1
            batches_by_domain[batch_domain] = batches_by_domain.get(batch_domain, 0) + 1
            global_step += 1
            for k in epoch_metrics:
                epoch_metrics[k] += metrics.get(k, 0.0)

            if global_step % log_every == 0:
                logger.info(
                    f"ep{epoch} step{global_step} dom={batch_domain:<11} "
                    f"loss={metrics['bc_total_loss']:.3f} "
                    f"policy={metrics['bc_policy_loss']:.3f} "
                    f"top1={metrics['bc_top1_acc']:.3f} "
                    f"win={metrics['bc_vl_run_win']:.3f} "
                    f"dq={metrics['bc_vl_deck_quality']:.3f} "
                    f"hp={metrics['bc_vl_exp_hp_loss']:.2f} "
                    f"dmg={metrics['bc_vl_exp_dmg_output']:.2f}"
                )

        # epoch 汇总
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)
        epoch_metrics["epoch"] = epoch
        epoch_metrics["n_batches"] = n_batches
        epoch_metrics["nan_skips"] = n_nan_skips
        epoch_metrics["elapsed_sec"] = time.time() - t0
        epoch_metrics["batches_by_domain"] = batches_by_domain
        epoch_metrics_log.append(epoch_metrics)
        logger.info(
            f"[epoch {epoch}/{epochs}] "
            f"loss={epoch_metrics['bc_total_loss']:.3f} "
            f"top1={epoch_metrics['bc_top1_acc']:.3f} "
            f"win={epoch_metrics['bc_vl_run_win']:.4f} "
            f"dq={epoch_metrics['bc_vl_deck_quality']:.4f} "
            f"hp={epoch_metrics['bc_vl_exp_hp_loss']:.3f} "
            f"dmg={epoch_metrics['bc_vl_exp_dmg_output']:.3f} "
            f"fc={epoch_metrics['bc_vl_floor_clear']:.4f} "
            f"dom={batches_by_domain} "
            f"elapsed={epoch_metrics['elapsed_sec']:.1f}s "
            f"nan={n_nan_skips}"
        )

        # 保存 checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = output_dir / f"bc_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": net.state_dict(),
                "network_config": asdict(cfg),
                "metrics": epoch_metrics,
            }, ckpt_path)
            logger.info(f"saved {ckpt_path}")

    # 写 metrics log
    with (output_dir / "bc_metrics.jsonl").open("w", encoding="utf-8") as f:
        for m in epoch_metrics_log:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    logger.info(f"done. metrics saved to {output_dir/'bc_metrics.jsonl'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Skada 离线非战斗训练(BC + value regression)")
    p.add_argument("--jsonl-dir", type=Path, default=None,
                   help="skada run jsonl 目录(优先 --sqlite,都没给默认 jsonl 目录)")
    p.add_argument("--sqlite", type=Path, default=None,
                   help="skada_analytics.sqlite 路径(生产推荐)")
    p.add_argument("--only-victory", action="store_true",
                   help="只用 is_victory=1 的 run")
    p.add_argument("--min-ascension", type=int, default=0,
                   help="最低 ascension 门槛(过滤新手 run)")
    p.add_argument("--characters", type=str, default=None,
                   help="逗号分隔 character 白名单,如 IRONCLAD,REGENT")
    p.add_argument("--max-runs", type=int, default=None,
                   help="最多加载多少 runs(smoke test 用)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="checkpoint 输出目录,放 Artifacts/checkpoints/ 下")
    p.add_argument("--preset", type=str, default="slim", choices=["slim", "full"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-expired", action="store_true", default=True,
                   help="跳过 detail_expired=true 的空骨架 run")
    p.add_argument("--no-value-only", action="store_true",
                   help="关闭 per-floor value-only sample")
    p.add_argument("--no-map-routes", action="store_true",
                   help="关闭 map route sample")
    return p


def main():
    args = _build_parser().parse_args()

    # 数据源选择:sqlite > jsonl > default jsonl
    characters = None
    if args.characters:
        characters = [c.strip().upper() for c in args.characters.split(",") if c.strip()]

    loader_kwargs = dict(
        include_value_only=not args.no_value_only,
        include_map_routes=not args.no_map_routes,
    )

    if args.sqlite is not None:
        logger.info(f"loading from SQLite: {args.sqlite}")
        samples = load_samples_from_sqlite(
            args.sqlite,
            max_runs=args.max_runs,
            only_victory=args.only_victory,
            min_ascension=args.min_ascension,
            characters=characters,
            **loader_kwargs,
        )
    else:
        jsonl_dir = args.jsonl_dir or Path("data/skada/runs/details")
        logger.info(f"loading from JSONL dir: {jsonl_dir}")
        samples = load_samples_from_jsonl_dir(
            jsonl_dir,
            max_runs=args.max_runs,
            skip_expired=args.skip_expired,
            **loader_kwargs,
        )
    logger.info(f"loaded {len(samples)} samples")
    if not samples:
        logger.error("no samples loaded, aborting")
        return

    train_offline(
        samples,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        preset=args.preset,
        device=args.device,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
