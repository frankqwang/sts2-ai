"""Encounter embedding 可视化（Conditional Policy 专属诊断）。

用法：
  python -m networkV2.s7_diagnostics.encounter_embed_viz ../Artifacts/checkpoints/co15 \\
      --out ../Artifacts/runs/co15/analysis/encounter_embed_viz.png

产出：PCA 2D 散点，不同 encounter 的 boss-bias vector 在空间中的位置。

诊断意义：
  - 如果 3 个 boss 的点在图上**显著分开** → conditional policy 确实在学 per-boss bias
  - 如果集中成一团 → gate 没放大 / embedding 没学分化 → conditioning 没生效
  - 如果 norm 很小 → gate 仍在初值 0.1 附近，还没被 PPO 推大

如果 checkpoint 不含 encounter_embed（老架构），输出提示并退出。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """选最新的 cotrainer_iterXX.pt。"""
    files = sorted(checkpoint_dir.glob("cotrainer_iter*.pt"))
    if not files:
        return None
    return files[-1]


def load_embed(checkpoint_path: Path) -> tuple[np.ndarray, float] | None:
    """Load encounter_embed.weight + encounter_gate。"""
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"load fail: {e}")
        return None
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    emb_key = "encounter_embed.weight"
    gate_key = "encounter_gate"
    if emb_key not in state:
        print(f"checkpoint {checkpoint_path} has no encounter_embed (old arch)")
        return None
    emb = state[emb_key].cpu().numpy()
    gate = float(state.get(gate_key, torch.tensor(0.1)).item())
    return emb, gate


def plot(checkpoint_dir: Path, out: Path) -> None:
    ckpt = _find_latest_checkpoint(checkpoint_dir)
    if ckpt is None:
        print(f"no checkpoint found in {checkpoint_dir}")
        return
    res = load_embed(ckpt)
    if res is None:
        return
    emb, gate = res
    # 映射索引 → encounter_id
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from networkV2.s1_schema.encounter_vocab import index_to_encounter, vocab_size
        vs = vocab_size()
    except Exception as e:
        print(f"warning: encounter_vocab unavailable ({e}), using raw indices")
        vs = emb.shape[0]
        index_to_encounter = lambda i: f"idx_{i}"  # type: ignore

    # 只取有效索引（1..vs-1，idx=0 是 UNKNOWN）
    valid_indices = list(range(1, min(vs, emb.shape[0])))
    vectors = emb[valid_indices]  # (N, d)
    names = [index_to_encounter(i) for i in valid_indices]
    norms = np.linalg.norm(vectors, axis=1)

    # PCA 降 2D
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    # SVD 方式，d << N 时用 np.linalg.svd 也行
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc = centered @ Vt[:2].T  # (N, 2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Encounter embedding (ckpt {ckpt.name}, gate={gate:.4f})",
                 fontsize=12, fontweight="bold")

    # 左：PCA 散点
    ax = axes[0]
    # 按 room_type 着色
    rt_color = {"boss": "#d62728", "elite": "#9467bd", "monster": "#1f77b4",
                "normal": "#1f77b4", "weak": "#90d0ff"}
    colors = []
    for n in names:
        nl = (n or "").lower()
        if "boss" in nl: c = rt_color["boss"]
        elif "elite" in nl: c = rt_color["elite"]
        elif "weak" in nl: c = rt_color["weak"]
        else: c = rt_color["monster"]
        colors.append(c)
    ax.scatter(pc[:, 0], pc[:, 1], c=colors, s=40, alpha=0.7, edgecolors="black",
               linewidths=0.3)
    # 标注 boss
    for i, n in enumerate(names):
        if n and "boss" in n.lower():
            ax.annotate(n.replace("_boss", "").upper()[:12], (pc[i, 0], pc[i, 1]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points",
                        fontweight="bold")
    ax.set_title(f"PCA 2D ({len(names)} encounters; explained var: "
                 f"{100*S[0]**2/(S**2).sum():.1f}% + {100*S[1]**2/(S**2).sum():.1f}%)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)

    # 右：embedding norm 柱状图（按 room_type 分组）
    ax = axes[1]
    boss_norms = [n for n, nm in zip(norms, names) if nm and "boss" in nm.lower()]
    elite_norms = [n for n, nm in zip(norms, names) if nm and "elite" in nm.lower()]
    monster_norms = [n for n, nm in zip(norms, names)
                     if nm and "boss" not in nm.lower() and "elite" not in nm.lower()]
    bp = ax.boxplot(
        [boss_norms, elite_norms, monster_norms],
        labels=[f"boss\n(n={len(boss_norms)})", f"elite\n(n={len(elite_norms)})",
                f"monster\n(n={len(monster_norms)})"],
        patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#d62728", "#9467bd", "#1f77b4"]):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_title(f"Embedding norm by room_type  |  gate * mean_norm = "
                 f"{gate * float(np.mean(norms)):.4f}")
    ax.set_ylabel("|| embedding[i] ||")
    ax.grid(alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[encounter_embed_viz] wrote {out}  (gate={gate:.4f}, mean_norm={np.mean(norms):.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir", type=Path,
                    help="checkpoint dir, e.g. Artifacts/checkpoints/co15")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出路径。默认需手动指定（checkpoint_dir 不是 dump_dir）。")
    args = ap.parse_args()
    if args.out is None:
        # 尝试推断 dump_dir：checkpoints/<name> → runs/<name>/analysis
        name = args.checkpoint_dir.name
        dump_guess = args.checkpoint_dir.parent.parent / "runs" / name / "analysis"
        args.out = dump_guess / "encounter_embed_viz.png"
    plot(args.checkpoint_dir, args.out)


if __name__ == "__main__":
    main()
