"""分析 RolloutDumper 导出的 iter 数据。

用法:
  python -m networkV2.s6_training.analyze_rollout runs/exp1                # 分析所有 iter
  python -m networkV2.s6_training.analyze_rollout runs/exp1 --iter 5       # 单 iter 详细
  python -m networkV2.s6_training.analyze_rollout runs/exp1 --diagnose     # 自动诊断异常

诊断项：
  - metrics=0 的真因（adv std 过小 / nan / requires_grad 丢失）
  - advantages 分布异常（全 0 / 极端尾部）
  - sample domain 分布（combat vs non-combat 比例）
  - 样本中是否有 episode 被 act_failed 污染
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_iter(root: Path, iteration: int) -> dict[str, Any]:
    """读取一个 iter 的所有 dump 文件。"""
    data = {"iter": iteration}

    mpath = root / f"iter{iteration:04d}_metrics.json"
    if mpath.exists():
        data["metrics"] = json.loads(mpath.read_text(encoding="utf-8"))

    apath = root / f"iter{iteration:04d}_advantages.npz"
    if apath.exists():
        data["arrays"] = dict(np.load(apath))

    spath = root / f"iter{iteration:04d}_samples.jsonl"
    if spath.exists():
        data["samples"] = [json.loads(l) for l in spath.read_text(encoding="utf-8").splitlines() if l.strip()]

    epath = root / f"iter{iteration:04d}_episodes.jsonl"
    if epath.exists():
        data["episodes"] = [json.loads(l) for l in epath.read_text(encoding="utf-8").splitlines() if l.strip()]

    return data


def _find_iters(root: Path) -> list[int]:
    iters = sorted({int(p.stem.split("_")[0][4:])
                    for p in root.glob("iter*_metrics.json")})
    return iters


def describe_advantages(arr: np.ndarray, label: str) -> str:
    """一行字概括 advantages 分布。"""
    if len(arr) == 0:
        return f"{label}: empty"
    return (f"{label}: n={len(arr)} "
            f"mean={arr.mean():+.4f} std={arr.std():.4f} "
            f"min={arr.min():+.3f} max={arr.max():+.3f} "
            f"p1={np.percentile(arr, 1):+.3f} p99={np.percentile(arr, 99):+.3f} "
            f"zero_ratio={(arr == 0).mean():.2%}")


def diagnose_iter(data: dict, verbose: bool = True) -> list[str]:
    """诊断单 iter 的异常。返回 warning list。"""
    warnings: list[str] = []
    it = data["iter"]
    m = data.get("metrics", {}).get("metrics", {})
    arrays = data.get("arrays", {})

    # 1) metrics 全 0？
    core_keys = ["policy_loss", "value_loss", "vl_hp_loss"]
    zeros = [k for k in core_keys if m.get(k, 0) == 0]
    if zeros and len(zeros) == len(core_keys):
        warnings.append(f"[iter {it}] 所有 combat loss 为 0: {zeros}")

    # 2) advantages 分布异常？
    if "advantages" in arrays:
        adv = arrays["advantages"]
        if adv.std() < 1e-8:
            warnings.append(f"[iter {it}] advantages std={adv.std():.2e} 过小，PPO 会无效")
        if (adv == 0).mean() > 0.5:
            warnings.append(f"[iter {it}] advantages 有 {(adv==0).mean():.1%} 是 0")
        if np.isnan(adv).any():
            warnings.append(f"[iter {it}] advantages 有 NaN: {np.isnan(adv).sum()} / {len(adv)}")

    # 3) rewards 分布
    if "rewards" in arrays:
        r = arrays["rewards"]
        if r.std() < 1e-8:
            warnings.append(f"[iter {it}] rewards std={r.std():.2e} 过小（信号不足）")

    # 4) domain 分布
    if "domain_is_combat" in arrays:
        dc = arrays["domain_is_combat"]
        ratio_c = dc.mean()
        if ratio_c < 0.1 or ratio_c > 0.99:
            warnings.append(f"[iter {it}] domain 极端不均: combat={ratio_c:.1%}")

    # 5) episodes 里的 error
    eps = data.get("episodes", [])
    err_count = sum(1 for e in eps if e.get("outcome") == "error" or e.get("error"))
    if err_count > 0:
        warnings.append(f"[iter {it}] 有 {err_count}/{len(eps)} 个 episode 出错")

    # 6) KL 过大？
    kl = m.get("approx_kl", 0)
    if kl > 0.05:
        warnings.append(f"[iter {it}] approx_kl={kl:.4f} 偏大，策略更新过激")

    # 7) value_estimate 没学到分化？
    if "value_estimates" in arrays:
        ve = arrays["value_estimates"]
        if ve.std() < 0.01:
            warnings.append(f"[iter {it}] value_estimates std={ve.std():.4f} — value 未分化")

    if verbose and warnings:
        for w in warnings:
            print(f"  WARN {w}")

    return warnings


def summary_iter(data: dict) -> None:
    """详细打印单 iter 数据。"""
    it = data["iter"]
    m = data.get("metrics", {}).get("metrics", {})
    arrays = data.get("arrays", {})
    samples = data.get("samples", [])

    print(f"\n=== Iter {it} detailed ===")
    print(f"  Samples: {len(samples)}")

    if arrays:
        print()
        print(f"  advantages       {describe_advantages(arrays.get('advantages', np.array([])), '')}")
        print(f"  rewards          {describe_advantages(arrays.get('rewards', np.array([])), '')}")
        print(f"  value_estimates  {describe_advantages(arrays.get('value_estimates', np.array([])), '')}")
        print(f"  returns          {describe_advantages(arrays.get('returns', np.array([])), '')}")
        print(f"  hp_loss_targets  {describe_advantages(arrays.get('hp_loss_targets', np.array([])), '')}")
        print(f"  old_log_probs    {describe_advantages(arrays.get('old_log_probs', np.array([])), '')}")

    # Domain breakdown
    if "domain_is_combat" in arrays:
        dc = arrays["domain_is_combat"]
        print(f"\n  Domain: combat={dc.sum()}/{len(dc)} ({dc.mean():.1%})")

    # Sample domain unique values
    if samples:
        domains: dict[str, int] = {}
        for s in samples:
            d = s.get("decision_domain", "?")
            domains[d] = domains.get(d, 0) + 1
        print(f"  Domains: {domains}")

    # Metrics
    if m:
        print(f"\n  Metrics:")
        for k in sorted(m.keys()):
            print(f"    {k:20s}: {m[k]:.6f}")

    # Episodes
    eps = data.get("episodes", [])
    if eps:
        outcomes: dict[str, int] = {}
        floors = [e.get("floor", 0) for e in eps]
        for e in eps:
            o = str(e.get("outcome", "unknown"))
            outcomes[o] = outcomes.get(o, 0) + 1
        print(f"\n  Episodes: {len(eps)}")
        print(f"    Outcomes: {outcomes}")
        print(f"    Avg floor: {sum(floors)/len(floors):.2f}")
        print(f"    Max floor: {max(floors)}")

    diagnose_iter(data, verbose=True)


def diagnose_all(root: Path) -> None:
    """扫所有 iter 做自动诊断。"""
    iters = _find_iters(root)
    if not iters:
        print(f"No iter data in {root}")
        return

    print(f"Scanning {len(iters)} iterations in {root}")
    all_warnings: list[str] = []
    for it in iters:
        data = _load_iter(root, it)
        all_warnings.extend(diagnose_iter(data, verbose=False))

    print(f"\n=== Summary of {len(iters)} iterations ===")
    print(f"Total warnings: {len(all_warnings)}")
    for w in all_warnings:
        print(f"  {w}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=str, help="Dump root dir (e.g. runs/exp1)")
    p.add_argument("--iter", type=int, default=-1, help="Specific iter; -1 = all")
    p.add_argument("--diagnose", action="store_true", help="Auto-diagnose all iters")
    args = p.parse_args()

    root = Path(args.root)

    if args.diagnose:
        diagnose_all(root)
    elif args.iter >= 0:
        summary_iter(_load_iter(root, args.iter))
    else:
        iters = _find_iters(root)
        for it in iters:
            summary_iter(_load_iter(root, it))


if __name__ == "__main__":
    main()
