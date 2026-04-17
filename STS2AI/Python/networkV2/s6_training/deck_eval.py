"""Deck Evaluation：给定 deck × encounter，用当前 net 跑 K 次战斗评估胜率/掉血。

用途：
1. 客观 benchmark —— 训练每 N iter 跑一次评估，看 agent 是否真在变强
2. 真值 deck_quality_target —— 替代当前粗糙启发式（后续接入）
3. card 评分 —— "选 X 卡 vs skip" 的 expected value 差（重计算量，谨慎使用）
4. 战斗专项 co-trainer —— 给定固定 (deck, encounter) 反复训直到掌握

API:
  evaluate_deck(client_pool, net, deck, encounters, n_trials=3) → dict 评估结果
  baseline_act1_set() → 默认 baseline encounter 列表

不修改主训练循环。集成方式：
  在 train_full_run.py 每 N iter 调一次，写到 metrics 里。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Categorical

from networkV2.s3_state_tracker.combat_env_wrapper import CombatStateTracker
from networkV2.s4_compiler.feature_compiler import CombatFeatureCompiler
from networkV2.s5_net.unified_net import UnifiedNet

logger = logging.getLogger(__name__)


@dataclass
class EncounterResult:
    encounter_id: str
    n_trials: int
    n_wins: int
    avg_steps: float
    avg_hp_loss: float
    avg_hp_loss_ratio: float    # hp_loss / max_hp

    @property
    def win_rate(self) -> float:
        return self.n_wins / max(self.n_trials, 1)


# ---------------------------------------------------------------------------
# Baseline encounter set —— 用作 deck quality 的 reference benchmark
# ---------------------------------------------------------------------------

def baseline_act1_set() -> list[tuple[str, str]]:
    """Act1 baseline encounters：(encounter_id, room_type)。

    使用 STS2 Early Access 的实际 encounter IDs（不是 STS1）。
    覆盖代表性难度（普通/精英/Boss）+ 有典型 mechanism 的 encounter。
    """
    return [
        # 普通 monster
        ("CULTISTS_NORMAL", "monster"),
        ("EXOSKELETONS_NORMAL", "monster"),
        ("FROG_KNIGHT_NORMAL", "monster"),
        # 精英: 测试中期 build 强度
        ("KNIGHTS_ELITE", "elite"),
        ("MECHA_KNIGHT_ELITE", "elite"),
        # Boss: 决定 act 通关
        ("DOORMAKER_BOSS", "boss"),
        ("LAGAVULIN_MATRIARCH_BOSS", "boss"),
    ]


# ---------------------------------------------------------------------------
# 单局 combat rollout（greedy）
# ---------------------------------------------------------------------------

def _greedy_combat_rollout(
    client,                       # PipeBackedCombatTrainingClient
    net: UnifiedNet,
    encounter_id: str,
    room_type: str,
    build: dict[str, Any] | None,
    max_steps: int = 200,
    seed: str | None = None,
) -> dict[str, Any]:
    """跑一场战斗到结束（victory / defeat / max_steps），返回结果字典。"""
    compiler = CombatFeatureCompiler()
    tracker = CombatStateTracker()
    tracker.on_run_start()
    tracker.on_combat_start({"player": {"hp": 80, "max_hp": 80}}, encounter_id, room_type)

    try:
        state = client.reset(encounter_id=encounter_id, build=build, seed=seed)
    except Exception as e:
        return {"error": f"reset failed: {e}", "outcome": "error",
                "steps": 0, "hp_loss": 0, "max_hp": 80}

    hp_at_start = int((state.get("player") or {}).get("hp", 0) or 0)
    max_hp = int((state.get("player") or {}).get("max_hp", 80) or 80)

    steps = 0
    outcome = "incomplete"
    final_hp = hp_at_start
    prev_state = state
    device = next(net.parameters()).device

    for _ in range(max_steps):
        legal = state.get("legal_actions", [])
        if not legal:
            break

        banks = compiler.compile(
            state, legal,
            combat_memory=tracker.combat_memory,
            turn_prefix=tracker.turn_prefix,
            run_build_memory=tracker.run_build_memory,
            encounter_id=encounter_id, room_type=room_type,
        )
        with torch.no_grad():
            out = net(banks=banks)
        logits = out.logits[0, :len(legal)]
        mask = out.action_mask[0, :len(legal)]
        logits = torch.nan_to_num(logits.masked_fill(~mask, float("-inf")), nan=0.0)
        idx = logits.argmax().item()  # greedy
        chosen = legal[idx]

        try:
            next_state, _r, done, _info = client.step(chosen)
        except Exception as e:
            logger.warning(f"deck_eval step failed: {e}")
            break

        tracker.on_step(next_state, chosen, prev_state=state)

        steps += 1
        final_hp = int((next_state.get("player") or {}).get("hp", final_hp) or final_hp)

        if done:
            outc = next_state.get("run_outcome")
            outcome = "victory" if str(outc or "").lower() == "victory" else "defeat"
            break
        prev_state = state
        state = next_state
    else:
        outcome = "timeout"

    return {
        "outcome": outcome,
        "steps": steps,
        "hp_loss": max(hp_at_start - final_hp, 0),
        "final_hp": final_hp,
        "max_hp": max_hp,
    }


# ---------------------------------------------------------------------------
# Evaluation entrypoint
# ---------------------------------------------------------------------------

def evaluate_deck(
    client,                               # PipeBackedCombatTrainingClient
    net: UnifiedNet,
    deck: dict[str, Any],
    encounters: list[tuple[str, str]] | None = None,
    n_trials_per_encounter: int = 3,
    max_steps_per_combat: int = 200,
    seed_prefix: str = "deckeval",
) -> dict[str, Any]:
    """对 (deck, encounter set) 跑 K 次评估。

    返回:
      {
        "encounter_results": [EncounterResult, ...],
        "overall": {
          "win_rate": float,            # 总胜率
          "avg_hp_loss_ratio": float,   # 总平均掉血比
          "deck_score": float,          # 综合分数 ∈ [-1, 1]
        }
      }
    """
    if encounters is None:
        encounters = baseline_act1_set()

    net.eval()
    results: list[EncounterResult] = []
    total_wins = 0
    total_trials = 0
    total_hp_loss_ratio = 0.0

    for enc_id, room_type in encounters:
        wins = 0
        steps_sum = 0
        hp_loss_sum = 0
        max_hp_sum = 0
        for trial in range(n_trials_per_encounter):
            seed = f"{seed_prefix}-{enc_id}-{trial}"
            r = _greedy_combat_rollout(
                client, net, enc_id, room_type, deck,
                max_steps=max_steps_per_combat, seed=seed,
            )
            if r.get("outcome") == "victory":
                wins += 1
            steps_sum += r.get("steps", 0)
            hp_loss_sum += r.get("hp_loss", 0)
            max_hp_sum += r.get("max_hp", 80)

        n = n_trials_per_encounter
        avg_hp_loss = hp_loss_sum / max(n, 1)
        avg_max_hp = max_hp_sum / max(n, 1)
        avg_hp_loss_ratio = avg_hp_loss / max(avg_max_hp, 1)

        er = EncounterResult(
            encounter_id=enc_id, n_trials=n, n_wins=wins,
            avg_steps=steps_sum / max(n, 1),
            avg_hp_loss=avg_hp_loss,
            avg_hp_loss_ratio=avg_hp_loss_ratio,
        )
        results.append(er)
        total_wins += wins
        total_trials += n
        total_hp_loss_ratio += avg_hp_loss_ratio

    overall_win_rate = total_wins / max(total_trials, 1)
    overall_hp_loss_ratio = total_hp_loss_ratio / max(len(encounters), 1)
    # 综合 score: 胜率高 + 掉血少 → score 高
    # win_rate=1 hp_loss=0 → score=1; win_rate=0 hp_loss>1 → score=-1
    deck_score = max(-1.0, min(1.0, overall_win_rate * 2 - 1 - overall_hp_loss_ratio * 0.5))

    return {
        "encounter_results": [
            {
                "encounter_id": r.encounter_id,
                "win_rate": r.win_rate,
                "n_trials": r.n_trials,
                "avg_steps": r.avg_steps,
                "avg_hp_loss": r.avg_hp_loss,
                "avg_hp_loss_ratio": r.avg_hp_loss_ratio,
            } for r in results
        ],
        "overall": {
            "win_rate": overall_win_rate,
            "avg_hp_loss_ratio": overall_hp_loss_ratio,
            "deck_score": deck_score,
        },
    }


# ---------------------------------------------------------------------------
# Default starter decks (Ironclad)
# ---------------------------------------------------------------------------

def ironclad_starter_deck() -> dict[str, Any]:
    """Ironclad（warrior）起手 deck（STS2 Early Access 命名）。"""
    return {
        "deck": (
            ["STRIKE_IRONCLAD"] * 5 +
            ["DEFEND_IRONCLAD"] * 4 +
            ["BASH"]
        ),
        "max_hp": 80,
        "current_hp": 80,
        "gold": 99,
        "max_energy": 3,
    }


def empty_deck() -> None:
    """让 sim 用默认 starter deck（不指定 build）。"""
    return None
