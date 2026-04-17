"""Trajectory 分析：从 iterNNNN_trajectories.jsonl 挖掘决策模式 + stuck loop。

主要功能：
1. find_stuck_loops(): 检测同一房间 / 同一动作连续 N 次的循环
2. find_low_progress_episodes(): 长 episode（>X 步）但低 floor 的局
3. action_repetition_stats(): 每种 chosen_action 的最长连续次数分布
4. floor_progress_curve(): 每个 trajectory 的 floor 随 step 增长曲线
5. failed_action_rate(): act_succeeded=False 的比例（通常是模型选了不可能的动作）

用法：
  python -m networkV2.s7_diagnostics.trajectory_analyzer runs/long2 --iter 5
  python -m networkV2.s7_diagnostics.trajectory_analyzer runs/long2 --all --stuck
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _read_trajectories(root: Path, iteration: int) -> list[dict]:
    p = root / f"iter{iteration:04d}_trajectories.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def find_stuck_loops(traj: list[dict], window: int = 8, min_repeat: int = 6) -> list[dict]:
    """检测 stuck loop：连续 window 步内同一 (room_type, chosen_action) 出现 >= min_repeat 次。"""
    found = []
    for i in range(len(traj) - window + 1):
        win = traj[i:i + window]
        keys = [(s["room_type"], s["chosen_action"]) for s in win]
        c = Counter(keys)
        most_key, most_n = c.most_common(1)[0]
        if most_n >= min_repeat:
            found.append({
                "start_step": traj[i]["step"],
                "end_step": traj[i + window - 1]["step"],
                "room_type": most_key[0],
                "action": most_key[1],
                "repeats": most_n,
                "in_window": window,
                "floor_at_start": traj[i]["floor"],
                "hp_at_start": traj[i]["hp"],
            })
    # 去重相邻的连续 stuck（合并）
    if not found:
        return []
    merged = [found[0]]
    for s in found[1:]:
        last = merged[-1]
        if s["start_step"] <= last["end_step"] + 2 and s["action"] == last["action"]:
            last["end_step"] = s["end_step"]
            last["repeats"] = max(last["repeats"], s["repeats"])
        else:
            merged.append(s)
    return merged


def floor_progress(traj: list[dict]) -> list[tuple[int, int]]:
    """返回 (step, floor) 序列（floor 变化点）。"""
    out = []
    last_floor = -1
    for s in traj:
        if s["floor"] != last_floor:
            out.append((s["step"], s["floor"]))
            last_floor = s["floor"]
    return out


def analyze_one(record: dict) -> dict:
    """分析一个 episode 的 trajectory。"""
    traj = record["trajectory"]
    summary = record["summary"]

    if not traj:
        return {"summary": summary, "stuck": [], "floor_progress": [], "stats": {}}

    stuck = find_stuck_loops(traj)
    fprog = floor_progress(traj)

    actions = Counter(s["chosen_action"] for s in traj)
    rooms = Counter(s["room_type"] for s in traj)
    failed = sum(1 for s in traj if not s["act_succeeded"])

    # 每房间停留时间
    room_steps = defaultdict(int)
    for s in traj:
        room_steps[s["room_type"]] += 1
    longest_room = max(room_steps.items(), key=lambda x: x[1])

    # HP 走势：max-min 区间 + 末尾 hp_ratio
    hps = [s["hp"] for s in traj]
    hp_drop = max(hps) - min(hps) if hps else 0

    return {
        "summary": summary,
        "stuck_loops": stuck,
        "floor_progress": fprog,
        "stats": {
            "n_steps": len(traj),
            "n_unique_actions": len(actions),
            "top_actions": actions.most_common(5),
            "rooms_distribution": dict(rooms),
            "longest_in_room": longest_room,
            "failed_action_count": failed,
            "failed_action_rate": failed / len(traj) if traj else 0,
            "hp_max_drop": hp_drop,
        }
    }


def report_iter(root: Path, iteration: int) -> str:
    """生成一个 iter 的诊断报告。"""
    records = _read_trajectories(root, iteration)
    if not records:
        return f"iter {iteration}: no trajectory data"

    lines = [f"=== iter {iteration}: {len(records)} trajectories ==="]
    total_stuck = 0
    total_failed = 0
    total_steps = 0
    low_progress = []  # episodes that took >X steps but reached <Y floor
    by_outcome: dict[str, int] = Counter()

    for rec in records:
        a = analyze_one(rec)
        s = a["stats"]
        sm = a["summary"]
        total_stuck += len(a["stuck_loops"])
        total_failed += s["failed_action_count"]
        total_steps += s["n_steps"]
        by_outcome[sm.get("outcome", "?")] += 1
        if s["n_steps"] > 200 and sm.get("floor", 0) < 8:
            low_progress.append((sm.get("floor", 0), s["n_steps"], a["stuck_loops"]))

    lines.append(f"  total steps: {total_steps}, avg/ep: {total_steps/len(records):.0f}")
    lines.append(f"  outcomes: {dict(by_outcome)}")
    lines.append(f"  total stuck-loops: {total_stuck}")
    lines.append(f"  total failed actions: {total_failed} ({100*total_failed/max(total_steps,1):.1f}%)")
    lines.append(f"  low-progress eps (>200 steps, <8 floor): {len(low_progress)}")

    # Sample 一个最 stuck 的 episode 详细输出
    if records:
        worst = max(records, key=lambda r: len(find_stuck_loops(r["trajectory"])))
        a = analyze_one(worst)
        if a["stuck_loops"]:
            lines.append("")
            lines.append(f"  worst stuck episode (idx {worst['episode_idx']}, "
                         f"summary: {a['summary'].get('outcome','?')} f{a['summary'].get('floor',0)}):")
            for stuck in a["stuck_loops"][:5]:
                lines.append(f"    step {stuck['start_step']}-{stuck['end_step']} "
                             f"room={stuck['room_type']} action='{stuck['action']}' "
                             f"x{stuck['repeats']} (floor={stuck['floor_at_start']}, hp={stuck['hp_at_start']})")

        # 1 局的 floor progress 示例
        first = records[0]
        a1 = analyze_one(first)
        if a1["floor_progress"]:
            lines.append("")
            lines.append(f"  first episode floor progression:")
            steps = [f"{step}@f{floor}" for step, floor in a1["floor_progress"][:15]]
            lines.append(f"    {' → '.join(steps)}")
            lines.append(f"    final: {a1['summary']}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--iter", type=int, default=None, help="single iter to analyze (default: latest)")
    ap.add_argument("--all", action="store_true", help="analyze all iters")
    ap.add_argument("--save", action="store_true",
                    help="写到 <dump_dir>/analysis/trajectory_report[_iterNN].txt "
                         "（遵循 docs/design/DIAGNOSTICS_CONVENTION.md）")
    args = ap.parse_args()

    if not args.dump_dir.exists():
        print(f"not found: {args.dump_dir}")
        return

    def _maybe_save(iter_id: int | None, content: str):
        if not args.save:
            return
        out_dir = args.dump_dir / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = "trajectory_report_all.txt" if iter_id is None else f"trajectory_report_iter{iter_id:04d}.txt"
        p = out_dir / name
        p.write_text(content, encoding="utf-8")
        print(f"[saved] {p}")

    if args.all:
        traj_files = sorted(args.dump_dir.glob("iter*_trajectories.jsonl"))
        reports = []
        for tf in traj_files:
            it = int(tf.name[4:8])
            rep = report_iter(args.dump_dir, it)
            print(rep)
            print()
            reports.append(rep)
        _maybe_save(None, "\n\n".join(reports))
    else:
        if args.iter is None:
            # find latest
            traj_files = sorted(args.dump_dir.glob("iter*_trajectories.jsonl"))
            if not traj_files:
                print("no trajectory files found")
                return
            args.iter = int(traj_files[-1].name[4:8])
        rep = report_iter(args.dump_dir, args.iter)
        print(rep)
        _maybe_save(args.iter, rep)


if __name__ == "__main__":
    main()
