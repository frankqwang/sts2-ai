"""Rollout + Training 诊断 dumper。

每 iteration 把下面数据写到 jsonl / npz，方便事后分析：

  {iter_N}_samples.jsonl
    每行一个 TrainingSample 的元数据（不含 banks 本体，太大）：
      - action_index, reward, advantage, value_target, old_log_prob
      - value_estimate, fight_win_target, hp_loss_target, survival_target
      - sample_weight, encounter_id, room_type, decision_domain

  {iter_N}_metrics.json
    train_step 返回的完整 metrics dict + 超参 snapshot

  {iter_N}_advantages.npz
    advantages / returns / old_log_probs 数组（便于做直方图）

  {iter_N}_episodes.jsonl
    每一局的摘要：outcome, floor, steps, combats, errors

使用：
  from networkV2.s6_training.rollout_dumper import RolloutDumper
  dumper = RolloutDumper("runs/exp1")
  ...
  dumper.dump_iteration(iteration, samples, metrics, episode_infos, extra_config)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from networkV2.s6_training.batch import TrainingSample


class RolloutDumper:
    """把 rollout + training 过程写到磁盘。"""

    def __init__(self, root: str | Path, flush_every: int = 1):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._meta_path = self.root / "run_meta.json"
        self._start_ts = time.time()

    def write_meta(self, meta: dict[str, Any]) -> None:
        """训练开始时调一次，记录超参和环境信息。"""
        full = dict(meta)
        full.setdefault("start_time", self._start_ts)
        full.setdefault("start_time_iso", time.strftime("%Y-%m-%d %H:%M:%S",
                                                        time.localtime(self._start_ts)))
        self._meta_path.write_text(json.dumps(full, indent=2, default=str),
                                   encoding="utf-8")

    def dump_iteration(
        self,
        iteration: int,
        samples: list[TrainingSample],
        metrics: dict[str, float],
        episode_infos: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """保存一个 iteration 的全部诊断数据。"""
        # 1) Samples metadata (不包含 banks)
        samples_path = self.root / f"iter{iteration:04d}_samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as f:
            for s in samples:
                meta = {
                    "action_index": s.action_index,
                    "reward": s.reward,
                    "advantage": s.advantage,
                    "value_target": s.value_target,
                    "old_log_prob": s.old_log_prob,
                    "value_estimate": s.value_estimate,
                    "fight_win_target": s.fight_win_target,
                    "hp_loss_target": s.hp_loss_target,
                    "survival_target": s.survival_target,
                    "leaf_target": s.leaf_target,
                    "sample_weight": s.sample_weight,
                    "encounter_id": s.encounter_id,
                    "room_type": s.room_type,
                    "decision_domain": s.banks.decision_domain,
                    "n_action_tokens": len(s.banks.action_bank),
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # 2) Metrics
        metrics_path = self.root / f"iter{iteration:04d}_metrics.json"
        payload = {
            "iteration": iteration,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "n_samples": len(samples),
            "extra": extra or {},
        }
        metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                encoding="utf-8")

        # 3) Advantages / returns 分布
        adv_path = self.root / f"iter{iteration:04d}_advantages.npz"
        if samples:
            np.savez_compressed(
                adv_path,
                advantages=np.array([s.advantage for s in samples], dtype=np.float32),
                returns=np.array([s.value_target for s in samples], dtype=np.float32),
                rewards=np.array([s.reward for s in samples], dtype=np.float32),
                value_estimates=np.array([s.value_estimate for s in samples], dtype=np.float32),
                old_log_probs=np.array([s.old_log_prob for s in samples], dtype=np.float32),
                fight_win_targets=np.array([s.fight_win_target for s in samples], dtype=np.float32),
                hp_loss_targets=np.array([s.hp_loss_target for s in samples], dtype=np.float32),
                domain_is_combat=np.array(
                    [s.banks.decision_domain == "combat" for s in samples], dtype=bool),
            )

        # 4) Episode summaries（不含 trajectory，那部分单独写以避免单文件爆炸）
        if episode_infos:
            ep_path = self.root / f"iter{iteration:04d}_episodes.jsonl"
            with ep_path.open("w", encoding="utf-8") as f:
                for info in episode_infos:
                    f.write(json.dumps(
                        {k: v for k, v in info.items() if k != "trajectory"},
                        ensure_ascii=False, default=str) + "\n")

        # 5) Trajectories（含 step-by-step 决策序列，仅有 trajectory 字段的 episode）
        traj_episodes = [info for info in (episode_infos or []) if info.get("trajectory")]
        if traj_episodes:
            traj_path = self.root / f"iter{iteration:04d}_trajectories.jsonl"
            with traj_path.open("w", encoding="utf-8") as f:
                for ep_idx, info in enumerate(traj_episodes):
                    record = {
                        "episode_idx": ep_idx,
                        "summary": {k: v for k, v in info.items() if k != "trajectory"},
                        "trajectory": info["trajectory"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
