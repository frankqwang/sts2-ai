"""Deck Evaluation CLI：独立工具，对指定 checkpoint x deck 跑评估。

用法：
  # 评估 long2 最新 checkpoint，用 starter deck 打 baseline encounters：
  python -m networkV2.s6_training.deck_eval_cli \
    --checkpoint checkpoints/long2/latest.pt \
    --preset full --port 15600 --n-trials 3

输出：
  每个 encounter 的胜率/掉血 + overall deck score
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from networkV2.s5_net.network_config import from_preset
from networkV2.s5_net.unified_net import UnifiedNet
from networkV2.s6_training.deck_eval import (
    evaluate_deck, baseline_act1_set, ironclad_starter_deck,
)
from networkV2.s0_bridge.combat_training_env import PipeBackedCombatTrainingClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="网络 checkpoint 路径；不指定则用随机初始化网络作 baseline")
    ap.add_argument("--preset", type=str, default="slim",
                    choices=["slim", "full", "tiny"])
    ap.add_argument("--port", type=int, default=15600,
                    help="combat sim 端口（不要和长训冲突）")
    ap.add_argument("--n-trials", type=int, default=3,
                    help="每个 encounter 跑几局")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--deck", type=str, default="starter",
                    help="'starter' 或 JSON 文件路径")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    cfg = from_preset(args.preset)
    net = UnifiedNet(config=cfg).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        # checkpoint 可能是 state_dict 或 wrapped dict
        if isinstance(state, dict) and "net" in state:
            state = state["net"]
        try:
            net.load_state_dict(state)
            print(f"[load] {args.checkpoint}")
        except Exception as e:
            print(f"[warn] partial load: {e}")
            net.load_state_dict(state, strict=False)

    if args.deck == "starter":
        deck = ironclad_starter_deck()
    else:
        with open(args.deck, encoding="utf-8") as f:
            deck = json.load(f)

    print(f"[deck] {len(deck['deck'])} cards, relics={deck.get('relics', [])}")
    print(f"[encounters] {len(baseline_act1_set())} baseline (CULTIST..THE_GUARDIAN)")

    client = PipeBackedCombatTrainingClient(port=args.port, auto_launch=True)
    try:
        result = evaluate_deck(
            client, net, deck,
            encounters=baseline_act1_set(),
            n_trials_per_encounter=args.n_trials,
            max_steps_per_combat=args.max_steps,
        )
    finally:
        try:
            client.close()
        except Exception:
            pass

    print()
    print("=== Per-encounter results ===")
    print(f"{'encounter':<22} {'wr':>6} {'avg_steps':>10} {'avg_hp_loss':>12} {'hp_loss%':>10}")
    for e in result["encounter_results"]:
        print(f"  {e['encounter_id']:<20} {e['win_rate']:>5.0%} "
              f"{e['avg_steps']:>10.1f} {e['avg_hp_loss']:>12.1f} "
              f"{e['avg_hp_loss_ratio']*100:>9.1f}%")

    o = result["overall"]
    print()
    print(f"=== Overall ===")
    print(f"  win_rate          : {o['win_rate']*100:.1f}%")
    print(f"  avg_hp_loss_ratio : {o['avg_hp_loss_ratio']*100:.1f}%")
    print(f"  deck_score [-1,1] : {o['deck_score']:+.3f}")


if __name__ == "__main__":
    main()
