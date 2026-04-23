from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

from zero.paths import STS2AI_ROOT
from zero.replay.naming import dated_artifact_dir_name


DEFAULT_CASE_INDEX = (
    STS2AI_ROOT / "Assets" / "datasets" / "zero_skada_replay_cases" / "v0_103_2_a0_single_combat_v1" / "cases.jsonl"
)
DEFAULT_HOST_PATH = STS2AI_ROOT / "Artifacts" / "tmp" / "headlesssim_build_dynamic_pool" / "HeadlessSim.dll"
DEFAULT_OUTPUT_ROOT = STS2AI_ROOT / "Artifacts" / "zero" / "experiments"
DEFAULT_VARIANTS = ("stateless", "history_transformer", "recurrent_gru")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单 case 过拟合：批量比较不同 zero 模型结构。")
    parser.add_argument("--target-case-id", type=str, required=True)
    parser.add_argument("--case-index", type=Path, default=DEFAULT_CASE_INDEX)
    parser.add_argument("--host-path", type=Path, default=DEFAULT_HOST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260420, 20260421, 20260422])
    parser.add_argument("--port", type=int, default=15527)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--collect-episodes", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--progress-only", action="store_true")
    parser.add_argument("--strict-promotion", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    variants = [_normalize_variant(item) for item in args.variants]
    experiment_root = args.output_root / dated_artifact_dir_name(f"single_case_overfit_{args.target_case_id}")
    runs_root = experiment_root / "runs"
    logs_root = experiment_root / "logs"
    experiment_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "target_case_id": args.target_case_id,
        "case_index": str(args.case_index),
        "variants": variants,
        "seeds": [int(seed) for seed in args.seeds],
        "iterations": int(args.iterations),
        "collect_episodes": int(args.collect_episodes),
        "train_steps": int(args.train_steps),
        "eval_episodes": int(args.eval_episodes),
        "progress_only": bool(args.progress_only),
        "strict_promotion": bool(args.strict_promotion),
    }
    (experiment_root / "experiment_config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records: list[dict[str, object]] = []
    for variant in variants:
        for seed in args.seeds:
            run_info = _run_single(
                variant=variant,
                seed=int(seed),
                port=args.port,
                target_case_id=args.target_case_id,
                case_index=args.case_index,
                host_path=args.host_path,
                runs_root=runs_root,
                logs_root=logs_root,
                iterations=args.iterations,
                collect_episodes=args.collect_episodes,
                train_steps=args.train_steps,
                eval_episodes=args.eval_episodes,
                progress_only=args.progress_only,
                strict_promotion=args.strict_promotion,
            )
            records.append(run_info)

    summary = _build_summary(records)
    (experiment_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_records_csv(experiment_root / "runs.csv", records)
    print(json.dumps({"experiment_root": str(experiment_root), "summary_path": str(experiment_root / "summary.json")}, ensure_ascii=False))


def _normalize_variant(name: str) -> str:
    value = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "stateless": "stateless",
        "no_history": "stateless",
        "history_transformer": "history_transformer",
        "history": "history_transformer",
        "transformer": "history_transformer",
        "recurrent_gru": "recurrent_gru",
        "gru": "recurrent_gru",
        "recurrent": "recurrent_gru",
    }
    if value not in aliases:
        raise ValueError(f"不支持的 variant={name}")
    return aliases[value]


def _run_single(
    *,
    variant: str,
    seed: int,
    port: int,
    target_case_id: str,
    case_index: Path,
    host_path: Path,
    runs_root: Path,
    logs_root: Path,
    iterations: int,
    collect_episodes: int,
    train_steps: int,
    eval_episodes: int,
    progress_only: bool,
    strict_promotion: bool,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "zero.replay.train",
        "--case-index",
        str(case_index),
        "--target-case-id",
        target_case_id,
        "--train-case-limit",
        "1",
        "--eval-case-limit",
        "1",
        "--iterations",
        str(iterations),
        "--collect-episodes",
        str(collect_episodes),
        "--train-steps",
        str(train_steps),
        "--eval-episodes",
        str(eval_episodes),
        "--port",
        str(port),
        "--host-path",
        str(host_path),
        "--output-root",
        str(runs_root),
        "--model-variant",
        variant,
        "--seed",
        str(seed),
        "--from-scratch",
    ]
    if progress_only:
        command.append("--progress-only")
    if strict_promotion:
        command.append("--strict-promotion")
    completed = subprocess.run(
        command,
        cwd=str(STS2AI_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    log_path = logs_root / f"{variant}_seed_{seed}.log"
    log_path.write_text(
        "\n".join(
            [
                "COMMAND:",
                subprocess.list2cmdline(command),
                "",
                "STDOUT:",
                completed.stdout,
                "",
                "STDERR:",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"实验失败: variant={variant}, seed={seed}, 日志={log_path}")

    payload = _extract_json_line(completed.stdout)
    metrics_path = Path(str(payload["metrics_path"]))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return _summarize_run(
        metrics=metrics,
        variant=variant,
        seed=seed,
        log_path=log_path,
        run_output_root=Path(str(payload["output_root"])),
    )


def _extract_json_line(stdout: str) -> dict[str, object]:
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("zero.replay.train 输出中未找到结果 JSON")


def _summarize_run(
    *,
    metrics: dict[str, object],
    variant: str,
    seed: int,
    log_path: Path,
    run_output_root: Path,
) -> dict[str, object]:
    manifests = list(metrics.get("manifests") or [])
    final_manifest = manifests[-1] if manifests else {}
    evaluations = list(final_manifest.get("evaluations") or [])
    train_rows = [dict(item.get("training") or {}) for item in manifests]
    result = {
        "variant": variant,
        "seed": seed,
        "target_case_id": str(metrics.get("target_case_id") or ""),
        "progress_only": bool(metrics.get("progress_only", False)),
        "run_output_root": str(run_output_root),
        "log_path": str(log_path),
        "iterations": len(manifests),
        "final_fight_win_rate": _mean_metric(evaluations, "fight_win_rate"),
        "final_enemy_hp_fraction_dealt": _mean_metric(evaluations, "enemy_hp_fraction_dealt"),
        "final_self_hp_fraction_remaining": _mean_metric(evaluations, "self_hp_fraction_remaining"),
        "final_avg_step_count": _mean_metadata_metric(evaluations, "avg_step_count"),
        "final_fight_quality_score": _mean_metadata_metric(evaluations, "fight_quality_score"),
        "final_total_loss": float((final_manifest.get("training") or {}).get("total_loss") or 0.0),
        "best_fight_win_rate": max([_mean_metric(list(item.get("evaluations") or []), "fight_win_rate") for item in manifests] or [0.0]),
        "best_self_hp_fraction_remaining": max(
            [_mean_metric(list(item.get("evaluations") or []), "self_hp_fraction_remaining") for item in manifests] or [0.0]
        ),
        "mean_total_loss": mean([float(item.get("total_loss") or 0.0) for item in train_rows]) if train_rows else 0.0,
        "promoted_last_iter": bool((final_manifest.get("promotion") or {}).get("promoted", False)),
    }
    return result


def _mean_metric(rows: list[dict[str, object]], key: str) -> float:
    values = [float(item.get(key) or 0.0) for item in rows]
    return mean(values) if values else 0.0


def _mean_metadata_metric(rows: list[dict[str, object]], key: str) -> float:
    values = [float((item.get("metadata") or {}).get(key) or 0.0) for item in rows]
    return mean(values) if values else 0.0


def _build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    by_variant: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_variant.setdefault(str(record["variant"]), []).append(record)
    variants: dict[str, object] = {}
    for variant, rows in by_variant.items():
        variants[variant] = {
            "runs": len(rows),
            "avg_final_fight_win_rate": mean(float(item["final_fight_win_rate"]) for item in rows),
            "avg_final_self_hp_fraction_remaining": mean(float(item["final_self_hp_fraction_remaining"]) for item in rows),
            "avg_best_fight_win_rate": mean(float(item["best_fight_win_rate"]) for item in rows),
            "avg_final_avg_step_count": mean(float(item["final_avg_step_count"]) for item in rows),
            "avg_mean_total_loss": mean(float(item["mean_total_loss"]) for item in rows),
        }
    return {
        "runs": records,
        "by_variant": variants,
    }


def _write_records_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
