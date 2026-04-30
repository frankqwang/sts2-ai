"""Compare two fixed-seed policy eval runs and apply promotion gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.scripts.automation.self_iterate import (  # noqa: E402
    _avg_reward,
    _candidate_passes,
    _invalid_rate,
    _metric_avg,
    _quality_count,
    _strict_json_failure_rate,
    _win_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-metrics", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--candidate-adapter", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-win-rate-delta", type=float, default=0.0)
    parser.add_argument("--max-reward-regression", type=float, default=0.05)
    parser.add_argument("--max-per-encounter-reward-regression", type=float, default=0.15)
    parser.add_argument("--max-per-encounter-win-rate-regression", type=float, default=0.001)
    parser.add_argument("--max-invalid-output-rate", type=float, default=0.02)
    parser.add_argument("--max-mechanism-score-regression", type=float, default=0.03)
    parser.add_argument("--max-missed-visible-lethal-increase", type=int, default=0)
    parser.add_argument("--max-reason-math-contradiction-increase", type=int, default=0)
    parser.add_argument("--max-reason-lethal-claim-error-increase", type=int, default=0)
    parser.add_argument("--max-action-score-lethal-math-contradiction-increase", type=int, default=0)
    parser.add_argument("--max-strict-json-failure-rate", type=float, default=0.05)
    parser.add_argument("--allow-missing-eval-keys", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary(metrics: dict[str, Any], *, adapter: str = "") -> dict[str, Any]:
    return {
        "adapter": adapter or metrics.get("adapter_dir"),
        "win_rate": _win_rate(metrics),
        "reward_avg": _avg_reward(metrics),
        "invalid_output_rate": _invalid_rate(metrics),
        "mechanism_score": _metric_avg(metrics, "mechanism_score", default=1.0),
        "missed_visible_lethal": _quality_count(metrics, "missed_visible_lethal"),
        "reason_math_contradiction": _quality_count(metrics, "reason_math_contradiction"),
        "reason_claims_lethal_but_action_not_lethal": _quality_count(
            metrics,
            "reason_claims_lethal_but_action_not_lethal",
        ),
        "action_score_lethal_math_contradiction": _quality_count(
            metrics,
            "action_score_lethal_math_contradiction",
        ),
        "strict_json_failures": int(((metrics.get("policy_stats") or {}).get("strict_json_failures")) or 0),
        "strict_json_failure_rate": _strict_json_failure_rate(metrics),
        "parse_failures": int(((metrics.get("policy_stats") or {}).get("parse_failures")) or 0),
    }


def main() -> int:
    args = parse_args()
    current_path = Path(args.current_metrics).resolve()
    candidate_path = Path(args.candidate_metrics).resolve()
    current = _read_json(current_path)
    candidate = _read_json(candidate_path)
    passed, reasons = _candidate_passes(
        current=current,
        candidate=candidate,
        min_win_rate_delta=args.min_win_rate_delta,
        max_reward_regression=args.max_reward_regression,
        max_per_encounter_reward_regression=args.max_per_encounter_reward_regression,
        max_per_encounter_win_rate_regression=args.max_per_encounter_win_rate_regression,
        max_invalid_output_rate=args.max_invalid_output_rate,
        max_mechanism_score_regression=args.max_mechanism_score_regression,
        max_missed_visible_lethal_increase=args.max_missed_visible_lethal_increase,
        max_reason_math_contradiction_increase=args.max_reason_math_contradiction_increase,
        max_reason_lethal_claim_error_increase=args.max_reason_lethal_claim_error_increase,
        max_action_score_lethal_math_contradiction_increase=args.max_action_score_lethal_math_contradiction_increase,
        max_strict_json_failure_rate=args.max_strict_json_failure_rate,
        allow_missing_eval_keys=args.allow_missing_eval_keys,
    )
    payload = {
        "passed": passed,
        "reasons": reasons,
        "current_metrics": str(current_path),
        "candidate_metrics": str(candidate_path),
        "current": _summary(current),
        "candidate": _summary(candidate, adapter=args.candidate_adapter),
        "gate_args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"current_metrics", "candidate_metrics", "candidate_adapter", "out"}
        },
    }
    _write_json(Path(args.out).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
