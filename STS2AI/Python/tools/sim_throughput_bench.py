#!/usr/bin/env python3
"""Sim step 吞吐 benchmark:1/4/8 env 并发。

跑法:
    cd STS2AI/Python && python -m tools.sim_throughput_bench --n 1,4,8 --seconds 20

每 env:reset -> step(end_turn) 循环直到 done -> reset -> ...
统计 step/s 和 reset/s。
"""
from __future__ import annotations
import argparse
import threading
import time
import os
if os.environ.get("SIM_BENCH_CLIENT", "proto") == "json":
    from env.combat_training_env import PipeBackedCombatTrainingClient
else:
    from networkV2.s0_bridge.combat_session import CombatSession as PipeBackedCombatTrainingClient

ENCOUNTER = "bowlbugs_weak"  # act1 monster,小战短平快


def run_worker(port: int, seconds: float, counters: dict, idx: int, ready_barrier: threading.Barrier | None = None):
    client = PipeBackedCombatTrainingClient(port=port, auto_launch=True)
    try:
        steps = 0
        resets = 0
        # auto_launch 每 worker 起新 sim 要 10+s,计时前先完成连接 + reset 一次作为 warmup
        state = client.reset(character_id="IRONCLAD", encounter_id=ENCOUNTER, seed=f"bench-{idx}-warm")
        # 再跑 3 步预热 JIT / cache
        for _ in range(3):
            legal = state.get("legal_actions") or []
            action = {"action": "end_turn"} if any(a.get("action") == "end_turn" for a in legal) else (legal[0] if legal else {"action": "end_turn"})
            state, _r, done, _ = client.step(action)
            if done: state = client.reset(character_id="IRONCLAD", encounter_id=ENCOUNTER, seed=f"bench-{idx}-warm-r")
        # barrier 同步:所有 worker 都 warmup 完才一起开始计时
        if ready_barrier is not None:
            ready_barrier.wait()
        # 正式测量窗口
        state = client.reset(character_id="IRONCLAD", encounter_id=ENCOUNTER, seed=f"bench-{idx}")
        resets += 1
        t_end = time.perf_counter() + seconds
        last_err = None
        while time.perf_counter() < t_end:
            # 优先 end_turn(每步最便宜);如果被拒,取第一个 legal action
            legal = state.get("legal_actions") or []
            action = {"action": "end_turn"} if any(a.get("action") == "end_turn" for a in legal) else (legal[0] if legal else {"action": "end_turn"})
            try:
                state, _r, done, _info = client.step(action)
                steps += 1
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:120]}"
                # 试重置继续
                try:
                    state = client.reset(character_id="IRONCLAD", encounter_id=ENCOUNTER, seed=f"bench-{idx}-{resets}-r")
                    resets += 1
                    continue
                except Exception as e2:
                    counters[idx] = {"steps": steps, "resets": resets, "error": f"{last_err} then {type(e2).__name__}: {str(e2)[:80]}"}
                    return
            if done:
                state = client.reset(character_id="IRONCLAD", encounter_id=ENCOUNTER, seed=f"bench-{idx}-{resets}")
                resets += 1
        counters[idx] = {"steps": steps, "resets": resets, "error": None}
    finally:
        try: client.close()
        except Exception: pass


def bench(n: int, seconds: float, base_port: int) -> dict:
    counters: dict[int, dict] = {}
    threads = []
    barrier = threading.Barrier(n + 1)  # n worker + main
    for i in range(n):
        t = threading.Thread(target=run_worker, args=(base_port + i, seconds, counters, i, barrier), daemon=False)
        t.start()
        threads.append(t)
    # 等所有 worker warmup 完,再开始计时(排除 launcher/reset/JIT 干扰)
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    total_steps = sum(c.get("steps", 0) for c in counters.values())
    total_resets = sum(c.get("resets", 0) for c in counters.values())
    errs = [c["error"] for c in counters.values() if c.get("error")]
    return {
        "n": n,
        "wall_s": wall,
        "total_steps": total_steps,
        "steps_per_s": total_steps / wall if wall > 0 else 0,
        "total_resets": total_resets,
        "errors": errs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=str, default="1,4,8", help="并发 env 数列表,逗号分隔")
    ap.add_argument("--seconds", type=float, default=20.0, help="每个 N 跑多久")
    ap.add_argument("--base-port", type=int, default=19900)
    args = ap.parse_args()
    n_list = [int(x) for x in args.n.split(",") if x.strip()]
    print(f"encounter={ENCOUNTER} seconds={args.seconds} base_port={args.base_port}")
    print(f"{'N':>4} {'wall_s':>8} {'steps':>8} {'resets':>7} {'step/s':>10} {'step/s/env':>12} {'err':>5}")
    for n in n_list:
        # 每个 N 用不同 port 段,避免 sim 复用/残留
        port = args.base_port + 100 * n_list.index(n)
        r = bench(n, args.seconds, port)
        err_n = len(r["errors"])
        print(f"{r['n']:>4} {r['wall_s']:>8.1f} {r['total_steps']:>8d} {r['total_resets']:>7d}"
              f" {r['steps_per_s']:>10.1f} {r['steps_per_s']/r['n']:>12.1f} {err_n:>5d}")
        if r["errors"]:
            for e in r["errors"][:3]:
                print(f"      ERR: {e}")


if __name__ == "__main__":
    main()
