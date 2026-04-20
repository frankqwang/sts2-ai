"""最近 iter 的 boss 战 per-step 轨迹可视化。

用法：
  python -m networkV2.s7_diagnostics.boss_trajectory_heatmap ../Artifacts/runs/co15

产出：<dump>/analysis/boss_trajectory_heatmap.png

内容：取最新有 trajectory 的 iter，每个 boss 抽一场 defeat + 一场 victory（若有），
画 4 行子图：
  (1) Player HP 曲线
  (2) Player block 曲线
  (3) Enemy 总 HP 曲线
  (4) Chosen card / action per step（文字标注）
按 per-turn 分块背景色区分回合。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_latest_boss_trajectories(dump_dir: Path, max_per_boss_outcome: int = 1):
    """返回 {(encounter_id, outcome): [trajectories...]}，从最新 iter 往回找。"""
    files = sorted(dump_dir.glob("iter*_trajectories.jsonl"), reverse=True)
    collected: dict[tuple[str, str], list] = {}
    for p in files:
        for line in p.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            s = d.get("summary") or {}
            eid = str(s.get("encounter_id", "")).upper()
            outcome = str(s.get("outcome", "")).lower()
            if "BOSS" not in eid:
                continue
            key = (eid, outcome)
            if len(collected.get(key, [])) >= max_per_boss_outcome:
                continue
            collected.setdefault(key, []).append({
                "iter": int(p.stem[4:8]),
                "summary": s,
                "trajectory": d.get("trajectory", []),
            })
        # 如果每个 boss × outcome 都采到了就停
        if len(collected) >= 6 and all(
            len(v) >= max_per_boss_outcome for v in collected.values()):
            break
    return collected


def plot(dump_dir: Path, out: Path) -> None:
    cases = load_latest_boss_trajectories(dump_dir)
    if not cases:
        print(f"no boss trajectories in {dump_dir}")
        return

    rows = sum(len(v) for v in cases.values())
    fig, axes = plt.subplots(rows, 1, figsize=(15, 2.6 * rows), squeeze=False)
    fig.suptitle(f"Boss combat trajectories (last iters from {dump_dir.name})",
                 fontsize=13, fontweight="bold")
    row_idx = 0
    for (eid, outcome), trajs in sorted(cases.items()):
        for case in trajs:
            ax = axes[row_idx, 0]
            traj = case["trajectory"]
            if not traj:
                row_idx += 1
                continue
            steps = list(range(len(traj)))
            hp = [t["hp"] for t in traj]
            block = [t["block"] for t in traj]
            enemy = [t["enemy_hp_total"] for t in traj]
            turns = [t["turn"] for t in traj]

            # 回合背景（交替灰/白）
            for turn_id in set(turns):
                idxs = [i for i, t in enumerate(turns) if t == turn_id]
                if idxs:
                    c = "#f5f5f5" if turn_id % 2 == 0 else "white"
                    ax.axvspan(idxs[0] - 0.5, idxs[-1] + 0.5, color=c, alpha=0.6, zorder=0)

            max_hp = case["summary"].get("max_hp", 80)
            ax.plot(steps, hp, color="#d62728", linewidth=2, label=f"player HP (max {max_hp})")
            ax.plot(steps, block, color="#1f77b4", linewidth=1.5, label="player block")

            # enemy HP on twin axis，方便看（数值量级不同）
            ax2 = ax.twinx()
            ax2.plot(steps, enemy, color="#2ca02c", linewidth=1.8, label="enemy HP total",
                     linestyle="--")
            ax2.set_ylabel("enemy HP", color="#2ca02c")
            ax2.tick_params(axis="y", labelcolor="#2ca02c")

            # 标注 chosen card（简短，只标 play_card 且有 card 名的）
            for i, t in enumerate(traj):
                if t.get("chosen_action") == "play_card" and t.get("chosen_card"):
                    card = t["chosen_card"].replace("_IRONCLAD", "").replace("_", "")[:8]
                    ax.text(i, hp[i] + 2, card, fontsize=6, rotation=45,
                            ha="left", alpha=0.6)
                elif t.get("chosen_action") in ("end_turn", "end"):
                    ax.axvline(i, color="orange", alpha=0.3, linewidth=0.5)

            tag = "WIN" if outcome == "victory" else "LOSE"
            title = (f"iter {case['iter']} | {eid.replace('_BOSS','')} [{tag}] | "
                     f"{len(traj)} steps, {max(turns)+1} turns, final_hp={case['summary'].get('final_hp')}")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("step")
            ax.set_ylabel("player HP / block")
            ax.legend(loc="upper left", fontsize=7)
            ax2.legend(loc="upper right", fontsize=7)
            ax.grid(alpha=0.3)
            row_idx += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[boss_trajectory_heatmap] wrote {out} ({row_idx} trajectories)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = args.dump_dir / "analysis" / "boss_trajectory_heatmap.png"
    plot(args.dump_dir, args.out)


if __name__ == "__main__":
    main()
