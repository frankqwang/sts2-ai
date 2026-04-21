#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立验证 MCTS 主线 save_state / load_state 接口的正确性与性能。

主线路径：
    GameBridgeCombatRuntime (zero.adapters.game_bridge)
      -> CombatSession (game_bridge.session.combat)
      -> pipe proto (sts2_mcts_proto_{port})
      -> HeadlessSim.exe

本脚本默认端口 16527，使用一份独立编译的 HeadlessSim 二进制（可由
--host-path 指定，默认 STS2AI/Artifacts/verify_mcts_snapshot/sim_build/HeadlessSim.exe）。
不与正在运行的训练 sim 共享端口/可执行文件，杀僵尸进程的筛选器也只会匹配
`--port <本脚本端口>` 且 ExecutablePath 等于我们指定 host_path 的进程，因此
不会误伤训练端的 HeadlessSim。

四类正确性测试：
  T1 save→walk→load 等价性：保存后走若干步，load 回后状态 == 保存瞬间。
  T2 load 幂等性：同一 state_id 多次 load（中间插入扰动），结果完全一致。
  T3 分支可重复性：save → 走动作序列 A → state_A；load → 再走 A → state_A'；两者相等。
  T4 action-level parity：在同一保存点，对每个 legal action 比较
      “原状态直接 step 一次” 与 “load 后 step 同一个 action 一次” 的 next state 是否一致。

Benchmark：save / load / delete / step 各 N 次，给 P50 / P95 / P99 / mean。
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]  # scripts -> bridge -> STS2AI -> <repo>
_STS2AI_ROOT = _REPO_ROOT / "STS2AI"
for _p in (_STS2AI_ROOT, _STS2AI_ROOT / "bridge"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from game_bridge.session.combat import CombatSession  # noqa: E402
from game_bridge.transport.connection import SimulatorApiError  # noqa: E402


DEFAULT_BUILD: dict[str, Any] = {
    "deck": [
        {"id": "STRIKE_IRONCLAD"},
        {"id": "STRIKE_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "DEFEND_IRONCLAD"},
        {"id": "BASH"},
        {"id": "POMMEL_STRIKE", "upgrade_level": 1},
        {"id": "SETUP_STRIKE", "upgrade_level": 1},
        {"id": "FORGOTTEN_RITUAL"},
        {"id": "BLUDGEON", "upgrade_level": 1},
    ],
    "relics": [
        {"id": "BURNING_BLOOD"},
        {"id": "HAND_DRILL"},
    ],
    "current_hp": 80,
    "max_hp": 80,
    "max_energy": 3,
    "gold": 99,
}
CHARACTER_ID = "IRONCLAD"
ENCOUNTER_ID = "CHOMPERS_NORMAL"


def _canonical(raw: Any) -> str:
    return json.dumps(raw, sort_keys=True, ensure_ascii=True)


def _short(v: Any, n: int = 140) -> Any:
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=True)
        return s if len(s) <= n else s[: n - 3] + "..."
    return v


def _diff_raw(a: Any, b: Any, *, path: str = "", limit: int = 60) -> list[dict]:
    diffs: list[dict] = []
    if len(diffs) >= limit:
        return diffs
    if type(a) is not type(b):
        diffs.append({"path": path or "$", "a": _short(a), "b": _short(b), "kind": "type"})
        return diffs
    if isinstance(a, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            if len(diffs) >= limit:
                break
            sub = f"{path}.{k}" if path else k
            if k not in a:
                diffs.append({"path": sub, "a": "<missing>", "b": _short(b[k])})
            elif k not in b:
                diffs.append({"path": sub, "a": _short(a[k]), "b": "<missing>"})
            else:
                diffs.extend(_diff_raw(a[k], b[k], path=sub, limit=limit - len(diffs)))
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append({"path": f"{path}.len", "a": len(a), "b": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            if len(diffs) >= limit:
                break
            diffs.extend(_diff_raw(x, y, path=f"{path}[{i}]", limit=limit - len(diffs)))
        return diffs
    if a != b:
        diffs.append({"path": path or "$", "a": _short(a), "b": _short(b)})
    return diffs


def _reset(session: CombatSession, seed: str) -> dict:
    return session.reset(
        character_id=CHARACTER_ID,
        encounter_id=ENCOUNTER_ID,
        build=DEFAULT_BUILD,
        seed=seed,
    )


def _try_load(session: CombatSession, sid: str) -> tuple[bool, Any]:
    try:
        return True, dict(session.load_state(sid))
    except SimulatorApiError as exc:
        return False, str(exc).splitlines()[0][:200]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:200]


def _try_save(session: CombatSession) -> tuple[bool, Any]:
    try:
        return True, session.save_state()
    except SimulatorApiError as exc:
        return False, str(exc).splitlines()[0][:200]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:200]


def _safe_delete(session: CombatSession, sid: Any) -> None:
    if not isinstance(sid, str):
        return
    try:
        session.delete_state(sid)
    except Exception:
        pass


def _current_legal_actions(state: dict) -> list[dict]:
    la = state.get("legal_actions")
    return la if isinstance(la, list) else []


def _action_summary(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": action.get("action"),
        "type": action.get("type"),
        "index": action.get("index"),
        "card_index": action.get("card_index"),
        "target_id": action.get("target_id"),
        "label": action.get("label"),
        "card_id": action.get("card_id"),
    }


def _step_action(session: CombatSession, action: dict[str, Any]) -> tuple[bool, Any]:
    try:
        state, _reward, _done, _info = session.step(copy.deepcopy(action))
        return True, dict(state)
    except SimulatorApiError as exc:
        return False, str(exc).splitlines()[0][:200]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:200]


def _try_step(session: CombatSession, action_index: int = 0) -> bool:
    state = session.current_state
    if state.get("terminal") or not _current_legal_actions(state):
        return False
    legal = _current_legal_actions(state)
    if action_index < 0 or action_index >= len(legal):
        return False
    try:
        session.step(legal[action_index])
        return True
    except Exception:
        return False


def _walk(session: CombatSession, n: int) -> int:
    walked = 0
    for _ in range(n):
        if not _try_step(session):
            break
        walked += 1
    return walked


def test_equivalence(
    session: CombatSession,
    seed: str,
    steps_after: int,
    diff_dump: Path | None,
) -> dict:
    _reset(session, seed)
    baseline = dict(session.current_state)
    save_ok, sid = _try_save(session)
    result: dict[str, Any] = {
        "name": "T1_save_walk_load",
        "seed": seed,
        "save_ok": save_ok,
        "baseline_bytes": len(_canonical(baseline)),
    }
    if not save_ok:
        result["ok"] = False
        result["save_error"] = sid
        return result
    try:
        walked = _walk(session, steps_after)
        post_walk = dict(session.current_state)
        load_ok, restored = _try_load(session, sid)
    finally:
        _safe_delete(session, sid)

    result["walked_steps"] = walked
    result["walk_produced_change"] = _canonical(post_walk) != _canonical(baseline)
    result["load_ok"] = load_ok
    if not load_ok:
        result["ok"] = False
        result["load_error"] = restored
        # load 失败后 sim 内部状态可能被 RunManager.CleanUp 破坏，重置一次保证后续测试不受影响
        try:
            _reset(session, seed)
        except Exception as exc:
            result["post_fail_reset_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return result
    restored_canon = _canonical(restored)
    baseline_canon = _canonical(baseline)
    result["restored_bytes"] = len(restored_canon)
    ok = baseline_canon == restored_canon
    result["ok"] = ok
    if not ok and diff_dump is not None:
        diffs = _diff_raw(baseline, restored)
        diff_dump.write_text(
            json.dumps({"baseline_vs_restored": diffs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["diff_path"] = str(diff_dump)
        result["diff_count"] = len(diffs)
    if not result["walk_produced_change"]:
        result["note"] = "走了若干步但状态未变 — 弱化测试信号"
    return result


def test_load_idempotent(
    session: CombatSession,
    seed: str,
    load_times: int,
    diff_dump: Path | None,
) -> dict:
    _reset(session, seed)
    save_ok, sid = _try_save(session)
    result: dict[str, Any] = {
        "name": "T2_load_idempotent",
        "seed": seed,
        "save_ok": save_ok,
        "load_times": load_times,
    }
    if not save_ok:
        result["ok"] = False
        result["save_error"] = sid
        return result
    canons: list[str] = []
    raws: list[dict] = []
    first_load_error: str | None = None
    loads_ok = 0
    try:
        for _ in range(load_times):
            load_ok, restored = _try_load(session, sid)
            if not load_ok:
                if first_load_error is None:
                    first_load_error = restored
                break
            loads_ok += 1
            canons.append(_canonical(restored))
            raws.append(restored)
            _walk(session, 2)
    finally:
        _safe_delete(session, sid)
    result["loads_ok"] = loads_ok
    if first_load_error is not None:
        result["load_error"] = first_load_error
        result["ok"] = False
        try:
            _reset(session, seed)
        except Exception:
            pass
        return result
    ok = bool(canons) and all(c == canons[0] for c in canons[1:])
    result["ok"] = ok
    result["canon_bytes"] = len(canons[0]) if canons else 0
    if not ok and diff_dump is not None and canons:
        first_diff_idx = next((i for i, c in enumerate(canons) if c != canons[0]), None)
        if first_diff_idx is not None:
            diffs = _diff_raw(raws[0], raws[first_diff_idx])
            diff_dump.write_text(
                json.dumps(
                    {"first_divergence_index": first_diff_idx, "diff": diffs},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            result["diff_path"] = str(diff_dump)
            result["diff_count"] = len(diffs)
            result["divergent_at"] = first_diff_idx
    return result


def test_branch_repeat(
    session: CombatSession,
    seed: str,
    action_chain_len: int,
    diff_dump: Path | None,
) -> dict:
    _reset(session, seed)
    save_ok, sid = _try_save(session)
    result: dict[str, Any] = {"name": "T3_branch_repeat", "seed": seed, "save_ok": save_ok}
    if not save_ok:
        result["ok"] = False
        result["save_error"] = sid
        return result
    try:
        walked_a = _walk(session, action_chain_len)
        state_a = dict(session.current_state)
        load_ok, restored_or_err = _try_load(session, sid)
        result["load_ok"] = load_ok
        if not load_ok:
            result["ok"] = False
            result["load_error"] = restored_or_err
            try:
                _reset(session, seed)
            except Exception:
                pass
            return result
        walked_b = _walk(session, action_chain_len)
        state_b = dict(session.current_state)
    finally:
        _safe_delete(session, sid)
    ok = _canonical(state_a) == _canonical(state_b)
    result.update({
        "ok": ok,
        "walked_a": walked_a,
        "walked_b": walked_b,
    })
    if walked_a != walked_b:
        result["note"] = "两次走到的步数不同 — 已经是不一致信号"
    if not ok and diff_dump is not None:
        diffs = _diff_raw(state_a, state_b)
        diff_dump.write_text(
            json.dumps({"branch_a_vs_b": diffs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["diff_path"] = str(diff_dump)
        result["diff_count"] = len(diffs)
    return result


def test_action_level_parity(
    session: CombatSession,
    seed: str,
    prewalk_steps: int,
    max_actions: int,
    diff_dump: Path | None,
) -> dict:
    _reset(session, seed)
    walked = _walk(session, prewalk_steps)
    baseline = dict(session.current_state)
    actions = list(_current_legal_actions(baseline))
    if max_actions > 0:
        actions = actions[:max_actions]

    result: dict[str, Any] = {
        "name": "T4_action_level_parity",
        "seed": seed,
        "prewalk_steps": prewalk_steps,
        "walked_steps": walked,
        "baseline_bytes": len(_canonical(baseline)),
        "actions_considered": len(actions),
    }
    if baseline.get("terminal"):
        result["ok"] = False
        result["note"] = "prewalk 后已经 terminal，无法做一步转移等价性验证"
        return result
    if not actions:
        result["ok"] = False
        result["note"] = "baseline 没有 legal actions，无法做一步转移等价性验证"
        return result

    save_ok, sid = _try_save(session)
    result["save_ok"] = save_ok
    if not save_ok:
        result["ok"] = False
        result["save_error"] = sid
        return result

    mismatches: list[dict[str, Any]] = []
    tested = 0
    first_error: str | None = None
    try:
        for action_idx, action in enumerate(actions):
            action_meta = _action_summary(action)

            live_ok, live_next = _step_action(session, action)
            if not live_ok:
                first_error = f"live_step_failed[{action_idx}] {live_next}"
                break

            load_ok, restored = _try_load(session, sid)
            if not load_ok:
                first_error = f"load_after_live_failed[{action_idx}] {restored}"
                break

            restored_ok, restored_next = _step_action(session, action)
            if not restored_ok:
                first_error = f"restored_step_failed[{action_idx}] {restored_next}"
                break

            tested += 1
            if _canonical(live_next) != _canonical(restored_next):
                mismatches.append({
                    "action_index": action_idx,
                    "action": action_meta,
                    "diff": _diff_raw(live_next, restored_next, limit=80),
                })
                if len(mismatches) >= 5:
                    break

            load_ok, restored = _try_load(session, sid)
            if not load_ok:
                first_error = f"load_for_next_action_failed[{action_idx}] {restored}"
                break
    finally:
        _safe_delete(session, sid)

    result["actions_tested"] = tested
    result["ok"] = first_error is None and not mismatches
    if first_error is not None:
        result["error"] = first_error
        try:
            _reset(session, seed)
        except Exception:
            pass
    if mismatches:
        result["mismatch_count"] = len(mismatches)
        if diff_dump is not None:
            diff_dump.write_text(
                json.dumps({"action_level_mismatches": mismatches}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result["diff_path"] = str(diff_dump)
        else:
            result["mismatches"] = mismatches
    return result


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(int(p * len(s)), len(s) - 1)
    return s[k]


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean_ms": round(mean(xs), 3),
        "p50_ms": round(median(xs), 3),
        "p95_ms": round(_percentile(xs, 0.95), 3),
        "p99_ms": round(_percentile(xs, 0.99), 3),
        "min_ms": round(min(xs), 3),
        "max_ms": round(max(xs), 3),
    }


def benchmark(session: CombatSession, seed: str, iters: int, walk_between: int) -> dict:
    _reset(session, seed)
    save_ms: list[float] = []
    load_ms: list[float] = []
    load_fail_ms: list[float] = []
    delete_ms: list[float] = []
    step_ms: list[float] = []
    save_fail = 0
    load_fail = 0
    load_error_samples: list[str] = []

    # warm-up
    warm_ok, warm_sid = _try_save(session)
    if warm_ok:
        _try_load(session, warm_sid)
        _safe_delete(session, warm_sid)
    _walk(session, 2)

    for i in range(iters):
        t0 = time.perf_counter()
        save_ok, sid = _try_save(session)
        dt = (time.perf_counter() - t0) * 1000
        if not save_ok:
            save_fail += 1
            # 尝试把 sim 恢复到可用状态
            try:
                _reset(session, seed)
            except Exception:
                break
            continue
        save_ms.append(dt)

        for _ in range(walk_between):
            state = session.current_state
            if state.get("terminal") or not _current_legal_actions(state):
                break
            legal = _current_legal_actions(state)
            t0 = time.perf_counter()
            try:
                session.step(legal[0])
            except Exception:
                break
            step_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        load_ok, result = _try_load(session, sid)
        dt = (time.perf_counter() - t0) * 1000
        if load_ok:
            load_ms.append(dt)
        else:
            load_fail += 1
            load_fail_ms.append(dt)
            if len(load_error_samples) < 2:
                load_error_samples.append(str(result))
            # load 失败后 sim 可能被破坏，重置
            try:
                _reset(session, seed)
            except Exception:
                break

        t0 = time.perf_counter()
        _safe_delete(session, sid)
        delete_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "seed": seed,
        "iters": iters,
        "walk_between_save_and_load": walk_between,
        "save_state_ok": _stats(save_ms),
        "save_state_fail_count": save_fail,
        "load_state_ok": _stats(load_ms),
        "load_state_fail": _stats(load_fail_ms),
        "load_state_fail_count": load_fail,
        "load_error_samples": load_error_samples,
        "delete_state": _stats(delete_ms),
        "step_baseline": _stats(step_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=16527, help="独立 sim 端口，默认 16527")
    parser.add_argument(
        "--host-path",
        type=Path,
        default=_STS2AI_ROOT / "Artifacts" / "verify_mcts_snapshot" / "sim_build" / "HeadlessSim.exe",
        help="独立 HeadlessSim 可执行文件（与训练端二进制分离）",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["MCTS_VERIFY_A", "MCTS_VERIFY_B", "MCTS_VERIFY_C"],
    )
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--bench-walk", type=int, default=2)
    parser.add_argument("--steps-after-save", type=int, default=3)
    parser.add_argument("--idempotent-loads", type=int, default=5)
    parser.add_argument("--branch-chain-len", type=int, default=3)
    parser.add_argument("--action-parity-prewalk", type=int, default=2)
    parser.add_argument("--action-parity-max-actions", type=int, default=0, help="0 表示测试 baseline 上全部 legal actions")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_STS2AI_ROOT / "Artifacts" / "verify_mcts_snapshot",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.host_path.exists():
        print(f"[verify_mcts_snapshot] ERROR host_path 不存在: {args.host_path}", file=sys.stderr)
        print(
            "提示: dotnet build STS2AI/ENV/Sim/HeadlessSim/HeadlessSim.csproj -c Debug -o "
            f"\"{args.host_path.parent}\"",
            file=sys.stderr,
        )
        return 2

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.out_dir / f"report_{ts}.json"

    print(f"[verify_mcts_snapshot] port={args.port}")
    print(f"[verify_mcts_snapshot] host_path={args.host_path}")
    print(f"[verify_mcts_snapshot] out_dir={args.out_dir}")
    print(f"[verify_mcts_snapshot] 启动独立 HeadlessSim (auto_launch, proto pipe) …")

    t_launch = time.perf_counter()
    session = CombatSession(
        port=args.port,
        auto_launch=True,
        connect_timeout_s=30.0,
        host_path=args.host_path,
    )
    _reset(session, args.seeds[0])
    launch_ms = (time.perf_counter() - t_launch) * 1000
    print(f"[verify_mcts_snapshot] launch+first_reset = {launch_ms:.1f} ms")

    report: dict[str, Any] = {
        "meta": {
            "timestamp": ts,
            "port": args.port,
            "host_path": str(args.host_path),
            "character_id": CHARACTER_ID,
            "encounter_id": ENCOUNTER_ID,
            "seeds": args.seeds,
            "launch_plus_first_reset_ms": round(launch_ms, 2),
        },
        "tests": [],
        "benchmark": None,
    }

    try:
        for seed in args.seeds:
            t1 = test_equivalence(
                session, seed, args.steps_after_save,
                diff_dump=args.out_dir / f"diff_T1_{seed}_{ts}.json",
            )
            report["tests"].append(t1)
            _print_result(t1)

            t2 = test_load_idempotent(
                session, seed, args.idempotent_loads,
                diff_dump=args.out_dir / f"diff_T2_{seed}_{ts}.json",
            )
            report["tests"].append(t2)
            _print_result(t2)

            t3 = test_branch_repeat(
                session, seed, args.branch_chain_len,
                diff_dump=args.out_dir / f"diff_T3_{seed}_{ts}.json",
            )
            report["tests"].append(t3)
            _print_result(t3)

            t4 = test_action_level_parity(
                session, seed,
                prewalk_steps=args.action_parity_prewalk,
                max_actions=args.action_parity_max_actions,
                diff_dump=args.out_dir / f"diff_T4_{seed}_{ts}.json",
            )
            report["tests"].append(t4)
            _print_result(t4)

        print(f"[verify_mcts_snapshot] benchmark iters={args.bench_iters} …")
        report["benchmark"] = benchmark(session, args.seeds[0], args.bench_iters, args.bench_walk)
        _print_benchmark(report["benchmark"])
    finally:
        try:
            session.close()
        except Exception:
            pass

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[verify_mcts_snapshot] 报告已写入 {report_path}")

    all_ok = all(t["ok"] for t in report["tests"])
    print(f"[verify_mcts_snapshot] 全部测试通过: {all_ok}")
    return 0 if all_ok else 1


def _print_result(r: dict) -> None:
    flag = "OK" if r.get("ok") else "FAIL"
    extras = {k: v for k, v in r.items() if k not in {"name", "seed", "ok"}}
    extras_s = " ".join(f"{k}={v}" for k, v in extras.items() if not isinstance(v, (dict, list)))
    print(f"  [{flag}] {r['name']} seed={r['seed']} {extras_s}")


def _print_benchmark(b: dict) -> None:
    print("  benchmark:")
    for key in ("save_state_ok", "load_state_ok", "load_state_fail", "delete_state", "step_baseline"):
        s = b.get(key) or {}
        if s.get("n"):
            print(
                f"    {key:18s} n={s['n']:>3d}  "
                f"p50={s['p50_ms']:>7.2f}ms  p95={s['p95_ms']:>7.2f}ms  "
                f"p99={s['p99_ms']:>7.2f}ms  mean={s['mean_ms']:>7.2f}ms  max={s['max_ms']:>7.2f}ms"
            )
    sf = b.get("save_state_fail_count", 0)
    lf = b.get("load_state_fail_count", 0)
    if sf or lf:
        print(f"    失败次数: save={sf} load={lf}")
    for sample in b.get("load_error_samples") or []:
        print(f"    load 错误样本: {sample}")


if __name__ == "__main__":
    sys.exit(main())
