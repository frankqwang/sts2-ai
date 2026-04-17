"""Policy evolution analyzer：跨 iter 分析 policy 的变化。

利用 samples.jsonl 数据，跨 iter 计算：
1. 各 room_type 的 action distribution 演化（直方图随 iter 变化）
2. policy entropy 趋势（按 room_type 分）→ 早期发现 entropy collapse
3. value_estimate 范围 / std 演化（value head 是否在分化）
4. advantage scale 演化
5. 各 encounter_id 的 sample 占比（哪个 boss 被遇到了多少次）

用法：
  python -m networkV2.s7_diagnostics.policy_evolution runs/long2 [--from 1 --to 50]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _read_samples(root: Path, iteration: int) -> list[dict]:
    p = root / f"iter{iteration:04d}_samples.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def _empirical_action_entropy(action_indices: list[int]) -> float:
    """从 action choice 频率算经验 entropy。collapsed → 接近 0。"""
    if not action_indices:
        return 0.0
    c = Counter(action_indices)
    n = len(action_indices)
    return -sum((v/n) * math.log(v/n) for v in c.values() if v > 0)


def analyze_iter(root: Path, iteration: int) -> dict:
    samples = _read_samples(root, iteration)
    if not samples:
        return {"iter": iteration, "n": 0}

    by_room: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_room[s["room_type"]].append(s)

    room_stats: dict[str, dict] = {}
    for room, ss in by_room.items():
        idxs = [s["action_index"] for s in ss]
        n_opts_avg = sum(s["n_action_tokens"] for s in ss) / len(ss)
        max_entropy = math.log(max(n_opts_avg, 1))  # uniform baseline
        emp_ent = _empirical_action_entropy(idxs)
        # 归一化 entropy 到 [0, 1]：1 = 完全 uniform，0 = 完全 collapsed
        ent_ratio = emp_ent / max_entropy if max_entropy > 0 else 0.0
        # advantage stats
        advs = [s["advantage"] for s in ss]
        adv_mean = sum(advs) / len(advs)
        adv_std = (sum((a - adv_mean) ** 2 for a in advs) / len(advs)) ** 0.5
        # value estimate stats
        vals = [s["value_estimate"] for s in ss]
        v_mean = sum(vals) / len(vals)
        v_std = (sum((v - v_mean) ** 2 for v in vals) / len(vals)) ** 0.5

        # top action choice
        top = Counter(idxs).most_common(1)[0]
        top_ratio = top[1] / len(idxs)

        room_stats[room] = {
            "n": len(ss),
            "avg_options": round(n_opts_avg, 2),
            "entropy": round(emp_ent, 4),
            "max_entropy": round(max_entropy, 4),
            "entropy_ratio": round(ent_ratio, 4),
            "top_action_idx": top[0],
            "top_action_ratio": round(top_ratio, 4),
            "adv_mean": round(adv_mean, 4),
            "adv_std": round(adv_std, 4),
            "value_mean": round(v_mean, 4),
            "value_std": round(v_std, 4),
        }

    # 全局 encounter 分布
    encounters = Counter(s.get("encounter_id", "") for s in samples if s.get("encounter_id"))

    return {
        "iter": iteration,
        "n": len(samples),
        "room_stats": room_stats,
        "encounter_counts": dict(encounters.most_common(20)),
    }


def collapse_alert(stats: dict, room_filter: list[str] | None = None) -> list[str]:
    """检查是否有 entropy collapse 的早期信号。"""
    alerts = []
    rs = stats.get("room_stats", {})
    rooms = room_filter or rs.keys()
    for room in rooms:
        if room not in rs:
            continue
        r = rs[room]
        if r["avg_options"] >= 3 and r["entropy_ratio"] < 0.3:
            alerts.append(
                f"  iter {stats['iter']} room={room}: entropy_ratio={r['entropy_ratio']} "
                f"(top action {r['top_action_idx']} = {100*r['top_action_ratio']:.0f}%) → COLLAPSE"
            )
    return alerts


def report_evolution(root: Path, from_iter: int, to_iter: int) -> str:
    """按 iter 出 evolution table。"""
    iters = []
    for it in range(from_iter, to_iter + 1):
        if (root / f"iter{it:04d}_samples.jsonl").exists():
            iters.append(it)
    if not iters:
        return "no iter samples found"

    rooms_to_track = ["monster", "elite", "boss", "card_reward", "shop", "event"]
    lines = []
    lines.append(f"=== Policy evolution: iter {iters[0]} → {iters[-1]} ({len(iters)} iters) ===")
    lines.append("")
    lines.append("Per-room entropy_ratio over iters (1.0=uniform, 0.0=collapsed):")
    lines.append(f"  {'iter':>5}  " + "  ".join(f"{r[:8]:>10}" for r in rooms_to_track))

    all_alerts = []
    for it in iters:
        st = analyze_iter(root, it)
        rs = st.get("room_stats", {})
        row = [f"  {it:>5}"]
        for r in rooms_to_track:
            if r in rs:
                v = rs[r]["entropy_ratio"]
                row.append(f"{v:>10.3f}")
            else:
                row.append(f"{'-':>10}")
        lines.append("  ".join(row))
        all_alerts.extend(collapse_alert(st, rooms_to_track))

    lines.append("")
    lines.append("Per-room value_std over iters (>0.05 = value head 在分化):")
    lines.append(f"  {'iter':>5}  " + "  ".join(f"{r[:8]:>10}" for r in rooms_to_track))
    for it in iters:
        st = analyze_iter(root, it)
        rs = st.get("room_stats", {})
        row = [f"  {it:>5}"]
        for r in rooms_to_track:
            if r in rs:
                row.append(f"{rs[r]['value_std']:>10.4f}")
            else:
                row.append(f"{'-':>10}")
        lines.append("  ".join(row))

    if all_alerts:
        lines.append("")
        lines.append("ALERTS (entropy collapse detected):")
        for a in all_alerts[-20:]:  # 只展最近的
            lines.append(a)

    # 最新 iter 的 encounter 分布
    latest = analyze_iter(root, iters[-1])
    if latest.get("encounter_counts"):
        lines.append("")
        lines.append(f"iter {iters[-1]} encounter counts (top 10):")
        for eid, n in list(latest["encounter_counts"].items())[:10]:
            lines.append(f"  {eid:<30} {n:>5}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--from", dest="from_iter", type=int, default=1)
    ap.add_argument("--to", dest="to_iter", type=int, default=999)
    ap.add_argument("--save", action="store_true",
                    help="写到 <dump_dir>/analysis/policy_evolution.txt "
                         "（遵循 docs/design/DIAGNOSTICS_CONVENTION.md）")
    args = ap.parse_args()
    rep = report_evolution(args.dump_dir, args.from_iter, args.to_iter)
    print(rep)
    if args.save:
        out_dir = args.dump_dir / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "policy_evolution.txt"
        p.write_text(rep, encoding="utf-8")
        print(f"[saved] {p}")


if __name__ == "__main__":
    main()
