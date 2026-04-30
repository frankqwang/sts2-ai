"""One-command self-iteration loop for the LLM policy.

Pipeline:
  current adapter -> policy rollout -> GRPO-lite candidate -> fixed-seed eval
  current/candidate -> promotion gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.planner_hint import DEFAULT_PLANNER_HINT_REFRESH, PLANNER_HINT_REFRESH_CHOICES
from llm.paths import ARTIFACTS_ROOT, DATASETS_ROOT, GRPO_ROOT, RUNS_ROOT, SFT_ROOT, STS2AI_ROOT, ensure_dirs, resolve_default_python_exe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-adapter", type=str, required=True)
    parser.add_argument("--python-exe", type=str, default="", help="运行 rollout/train/eval 的 Python；默认使用 STS2_LLM_PYTHON_EXE 或 STS2AI/llm/.venv311。")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--rollout-generations", type=int, default=2)
    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--rollout-max-steps", type=int, default=120)
    parser.add_argument("--rollout-port-base", type=int, default=16040)
    parser.add_argument("--encounter-filter", type=str, default="", help="只迭代 encounter_id/tag/key 包含此字符串的 Skada case。")
    parser.add_argument(
        "--tier-filter",
        type=str,
        default="",
        help=(
            "Comma-separated subset of {normal,elite,boss}. 空 = 全部 tier。"
            "示例：--tier-filter elite,boss 跳过 normal（适合 normal 已饱和后专攻 boss）。"
        ),
    )
    parser.add_argument("--case-index", type=str, required=True, help="Skada combat cases.jsonl。")
    parser.add_argument("--case-character", type=str, default="IRONCLAD")
    parser.add_argument("--case-floor-min", type=int, default=1)
    parser.add_argument("--case-floor-max", type=int, default=17)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-sample-seed", type=int, default=0)
    parser.add_argument("--case-sample-mode", choices=["file", "random", "stratified", "diverse"], default="diverse")
    parser.add_argument("--elite-oversample-ratio", type=float, default=0.3, help="强制 elite 占比（rollout/eval 共用）。源数据 elite 仅 6%，不强制小 case-limit 容易没 elite。")
    parser.add_argument("--boss-oversample-ratio", type=float, default=0.0, help="强制 boss 占比（boss 在 floor 17/33/48）。0 表示不训 boss。")
    parser.add_argument("--include-lost-cases", action="store_true")
    parser.add_argument("--eval-episodes-per-encounter", type=int, default=2)
    parser.add_argument("--eval-max-steps", type=int, default=120)
    parser.add_argument("--eval-port-base", type=int, default=16140)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--allow-json-like-rollout", action="store_true")
    parser.add_argument("--planner-hint-adapter-dir", type=str, default="", help="可选 planner-hint LoRA adapter；rollout/eval 时注入 combat prompt。")
    parser.add_argument(
        "--planner-hint-refresh",
        choices=list(PLANNER_HINT_REFRESH_CHOICES),
        default=DEFAULT_PLANNER_HINT_REFRESH,
    )
    parser.add_argument("--planner-hint-max-new-tokens", type=int, default=240)
    parser.add_argument("--co-train-planner", action="store_true", help="同轮训练 planner candidate，并用四格 eval 矩阵评估 C/P 组合。")
    parser.add_argument("--planner-train-dataset-dir", type=str, default="", help="可选 planner-hint SFT 数据集；为空时使用本轮 teacher 产出的 planner_hint 数据。")
    parser.add_argument("--planner-min-train-rows", type=int, default=1)
    parser.add_argument("--planner-num-epochs", type=int, default=1)
    parser.add_argument("--planner-batch-size", type=int, default=1)
    parser.add_argument("--planner-grad-accum", type=int, default=4)
    parser.add_argument("--planner-lr", type=float, default=1e-4)
    parser.add_argument("--planner-max-seq-length", type=int, default=2048)
    parser.add_argument("--planner-load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--max-seq-length", type=int, default=3072,
        help=(
            "Combat LoRA prompt+response 总 token 上限。默认 3072 — boss 战 prompt 经常 "
            "2100-2400 tokens（敌人多 + power 描述 inline + planner_hint），2048 会被 "
            "Unsloth 截断尾部 'Return strict JSON only:' 指令导致 invalid_output。"
        ),
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    # NOTE on naming: every CLI flag below has historically been named
    # ``--kimi-*`` because Kimi was the first teacher provider. The default
    # provider is now DeepSeek-V4-Pro and Kimi is one of several supported
    # backends, so the canonical names are ``--teacher-*``. The ``--kimi-*``
    # spellings remain accepted for backward compatibility and share the
    # same argparse dest, so existing shell scripts keep working.
    parser.add_argument(
        "--teacher-review", "--kimi-teacher",
        dest="kimi_teacher",
        action="store_true",
        help="rollout 后调用 teacher（默认 deepseek-v4-pro）复盘 hard combats 并产出 gold 样本。",
    )
    parser.add_argument(
        "--teacher-provider",
        choices=["deepseek", "kimi", "kimi_code", "claude_cli"],
        default=os.environ.get("TEACHER_PROVIDER", "deepseek"),
        help="teacher 底层 provider（默认 deepseek-v4-pro，也支持 kimi / claude_cli）；--teacher-review/--kimi-teacher 是飞轮开关。",
    )
    parser.add_argument("--teacher-model", type=str, default=os.environ.get("TEACHER_MODEL", ""))
    parser.add_argument(
        "--teacher-max-workers",
        type=int,
        default=4,
        help=(
            "teacher review 并发线程数 (ThreadPoolExecutor)。默认 4，每个线程一个独立 "
            "deepseek/kimi HTTP 调用——典型 8 episodes 并发跑下来 1-2 分钟，串行需 15-30 "
            "分钟。如果撞 provider 限频（429），降到 2 或 1。"
        ),
    )
    parser.add_argument("--teacher-skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-claude-command", type=str, default=os.environ.get("CLAUDE_CLI_COMMAND", "claude"))
    parser.add_argument("--teacher-claude-proxy", type=str, default=os.environ.get("CLAUDE_PROXY", "http://127.0.0.1:7897"))
    parser.add_argument(
        "--teacher-limit-episodes", "--kimi-limit-episodes",
        dest="kimi_limit_episodes",
        type=int, default=20,
        help="本轮最多送 teacher 复盘的 combat 数；0 表示不限。",
    )
    parser.add_argument(
        "--teacher-max-api-calls", "--kimi-max-api-calls",
        dest="kimi_max_api_calls",
        type=int, default=20,
        help="本轮 teacher API 调用硬上限；-1 表示不限。",
    )
    parser.add_argument("--kimi-model", type=str, default=os.environ.get("KIMI_MODEL", "kimi-k2.6"),
                        help="(legacy) Kimi 专用 model 名；当 --teacher-provider=kimi 且 --teacher-model 为空时使用。")
    parser.add_argument(
        "--teacher-base-url", "--kimi-base-url",
        dest="kimi_base_url",
        type=str,
        default=os.environ.get("TEACHER_BASE_URL") or os.environ.get("KIMI_BASE_URL") or "",
        help="OpenAI-compatible base URL；空字符串则按 --teacher-provider 默认（deepseek=https://api.deepseek.com/v1, kimi=https://api.moonshot.cn/v1）。",
    )
    parser.add_argument(
        "--teacher-api-key-env", "--kimi-api-key-env",
        dest="kimi_api_key_env",
        type=str,
        default=os.environ.get("TEACHER_API_KEY_ENV") or "",
        help="读取 API key 的环境变量名；空则按 provider 默认（deepseek=DEEPSEEK_API_KEY，kimi=MOONSHOT_API_KEY）。",
    )
    parser.add_argument("--kimi-max-tokens", type=int, default=4096)
    parser.add_argument("--kimi-thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--kimi-timeout-s", type=float, default=180.0)
    parser.add_argument("--kimi-sleep-s", type=float, default=0.2)
    parser.add_argument("--kimi-max-decision-state-chars", type=int, default=7000)
    parser.add_argument("--kimi-damage-turns", type=int, default=2)
    parser.add_argument("--kimi-min-confidence", type=float, default=0.75)
    parser.add_argument("--kimi-min-review-ok-rate", type=float, default=0.5)
    parser.add_argument("--kimi-min-teacher-rows", type=int, default=0)
    parser.add_argument("--kimi-fail-on-quality-gate", action="store_true")
    parser.add_argument("--kimi-dry-run", action="store_true")
    parser.add_argument("--kimi-append-experience", action="store_true")
    parser.add_argument(
        "--use-teacher-reasons", "--use-kimi-reasons",
        dest="use_teacher_reasons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build_teacher_dataset 用 teacher (deepseek/kimi) corrected_reason 替代 canonical 模板；默认 True。"
             "关闭后 reason 字段回退到 canonical 模板（容易产生 'Deal X damage' 幻觉）。",
    )
    # ``--mask-reason-in-train-data`` 被移除 — combat policy 输出 schema 从
    # ``{action_index, confidence, reason}`` 收紧成 ``{action_index, confidence}``,
    # reason 字段不存在所以无需 mask. 保留 flag 作 no-op 防破老 shell 脚本.
    parser.add_argument(
        "--mask-reason-in-train-data",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sim-rebuild",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "rollout 启动前自动 dotnet build -c Release HeadlessSim, 解决 Release "
            "binary 比 *.cs 旧时 launcher 抛 stale RuntimeError 的问题。--no-sim-rebuild "
            "可跳过（自己保证 binary 是新的）。"
        ),
    )
    parser.add_argument(
        "--candidate-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "训练完成后是否再跑 candidate rollout 做 promotion gate (~80 min)。"
            "默认 True：训完用 candidate 跑一遍 rollout 跟 baseline 对比，promotion "
            "gate 据此决定是否晋级。--no-candidate-rollout 可跳过节省时间，但代价是无 "
            "promotion gate 数据，需要靠下一轮 baseline rollout 才能验证 candidate 是否进步——"
            "适合迭代调试期，不适合正式训练。"
        ),
    )
    parser.add_argument("--train-from-pool-after-teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-pool-ingest", action="store_true", help="跳过长期 dataset_pool ingest/materialize；本轮只用 rollout/teacher 临时产物。")
    parser.add_argument("--pool-train-target-size", type=int, default=5000)
    parser.add_argument("--pool-gold-min-ratio", type=float, default=0.15)
    parser.add_argument(
        "--grpo-loss-scope",
        choices=["full_text", "assistant"],
        default="full_text",
        help="GRPO-lite 训练 loss 范围。full_text 先保证自迭代闭环可运行；assistant 用于 response-only mask 稳定后的训练。",
    )
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
    parser.add_argument(
        "--isolation-evals",
        action="store_true",
        help=(
            "默认只跑 current_eval + joint_eval（gate 唯一依据是 joint）。"
            "打开此开关额外跑 candidate_eval（candidate combat × current planner）和 "
            "planner_eval（current combat × candidate planner）做隔离归因诊断；"
            "代价是每轮多 ~28 分钟。"
        ),
    )
    parser.add_argument("--allow-missing-eval-keys", action="store_true")
    parser.add_argument("--promote", action="store_true", help="达标时写 current_adapter.json 指针。")
    parser.add_argument("--dry-run", action="store_true", help="只写计划，不执行命令。")
    return parser.parse_args()


def _default_python_exe() -> str:
    return str(resolve_default_python_exe())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(
    command: list[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    dry_run: bool,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[self-iterate] cmd: {' '.join(command)}")
    print(f"[self-iterate] stdout -> {stdout_log}")
    print(f"[self-iterate] stderr -> {stderr_log}")
    if dry_run:
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        return 0
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["UNSLOTH_RETURN_LOGITS"] = "1"
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=str(cwd), stdout=out, stderr=err, env=env)
    if proc.returncode != 0:
        print(f"[self-iterate] failed code={proc.returncode}. See {stderr_log}", file=sys.stderr)
    return int(proc.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metrics(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _eval_summary(adapter: Path | None, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter": str(adapter) if adapter is not None else None,
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
        "strict_json_failure_rate": _strict_json_failure_rate(metrics),
    }


def _dataset_rows(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    summary = _read_json(summary_path)
    try:
        return int(summary.get("rows") or 0)
    except (TypeError, ValueError):
        return 0


def _teacher_review_gate(summary: dict[str, Any], *, min_review_ok_rate: float, min_teacher_rows: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    dry_run = bool(summary.get("dry_run"))
    if dry_run:
        return True, reasons
    episode_count = int(summary.get("episode_count") or 0)
    reviews_ok = int(summary.get("reviews_ok") or 0)
    labels = int(summary.get("labels") or 0)
    ok_rate = reviews_ok / max(1, episode_count)
    if episode_count > 0 and ok_rate + 1e-12 < min_review_ok_rate:
        reasons.append(f"Kimi review ok rate {ok_rate:.3f} < {min_review_ok_rate:.3f}")
    if labels < min_teacher_rows:
        reasons.append(f"Kimi labels {labels} < required {min_teacher_rows}")
    return not reasons, reasons


def _avg_reward(metrics: dict[str, Any]) -> float:
    value = ((metrics.get("reward") or {}).get("avg"))
    return float(value or 0.0)


def _win_rate(metrics: dict[str, Any]) -> float:
    value = metrics.get("win_rate")
    return float(value or 0.0)


def _invalid_rate(metrics: dict[str, Any]) -> float:
    value = metrics.get("invalid_output_episode_rate")
    return float(value or 0.0)


def _metric_avg(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    if isinstance(value, dict):
        raw = value.get("avg")
        return float(raw if raw is not None else default)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _quality_count(metrics: dict[str, Any], flag: str) -> int:
    value = metrics.get("action_quality")
    if isinstance(value, dict):
        try:
            return int(value.get(flag) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _strict_json_failure_rate(metrics: dict[str, Any]) -> float:
    """估算 strict-JSON 失败率。

    历史问题：原实现分母为 ``strict_json_failures + strict_json_ok``，但 policy_stats
    现在不再写入 ``strict_json_failures``（命名漂移），导致分子永远 0、gate 永远通过。
    现在做两件事：
      1. 分子优先 ``strict_json_failures``，缺失时退到 ``first_attempt_invalid - retry_recovered``。
      2. 分母优先 ``generated_outputs``（每次 LLM 生成都计数），缺失时退到 ``ok + failures``。
    """
    stats = metrics.get("policy_stats")
    if not isinstance(stats, dict):
        return 0.0
    try:
        ok = int(stats.get("strict_json_ok") or 0)
        raw_failures = stats.get("strict_json_failures")
        if raw_failures is None:
            failures = max(
                0,
                int(stats.get("first_attempt_invalid") or 0)
                - int(stats.get("retry_recovered") or 0),
            )
        else:
            failures = int(raw_failures or 0)
        denom_raw = stats.get("generated_outputs")
        if denom_raw is None:
            total = ok + failures
        else:
            total = max(int(denom_raw or 0), ok + failures)
    except (TypeError, ValueError):
        return 0.0
    return failures / total if total else 0.0


def _by_encounter(metrics: dict[str, Any]) -> dict[str, Any]:
    value = metrics.get("by_encounter")
    return value if isinstance(value, dict) else {}


def _payload_reward(payload: dict[str, Any]) -> float:
    return float(((payload.get("reward") or {}).get("avg")) or 0.0)


def _payload_win(payload: dict[str, Any]) -> float:
    return float(payload.get("win_rate") or 0.0)


def _candidate_passes(
    *,
    current: dict[str, Any],
    candidate: dict[str, Any],
    min_win_rate_delta: float,
    max_reward_regression: float,
    max_per_encounter_reward_regression: float,
    max_per_encounter_win_rate_regression: float,
    max_invalid_output_rate: float,
    max_mechanism_score_regression: float,
    max_missed_visible_lethal_increase: int,
    max_reason_math_contradiction_increase: int,
    max_reason_lethal_claim_error_increase: int,
    max_action_score_lethal_math_contradiction_increase: int,
    max_strict_json_failure_rate: float,
    allow_missing_eval_keys: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    current_win = _win_rate(current)
    candidate_win = _win_rate(candidate)
    current_reward = _avg_reward(current)
    candidate_reward = _avg_reward(candidate)
    candidate_invalid = _invalid_rate(candidate)
    current_mechanism = _metric_avg(current, "mechanism_score", default=1.0)
    candidate_mechanism = _metric_avg(candidate, "mechanism_score", default=1.0)
    current_missed_lethal = _quality_count(current, "missed_visible_lethal")
    candidate_missed_lethal = _quality_count(candidate, "missed_visible_lethal")
    current_reason_math = _quality_count(current, "reason_math_contradiction")
    candidate_reason_math = _quality_count(candidate, "reason_math_contradiction")
    current_lethal_claim_error = _quality_count(current, "reason_claims_lethal_but_action_not_lethal")
    candidate_lethal_claim_error = _quality_count(candidate, "reason_claims_lethal_but_action_not_lethal")
    current_score_lethal_math = _quality_count(current, "action_score_lethal_math_contradiction")
    candidate_score_lethal_math = _quality_count(candidate, "action_score_lethal_math_contradiction")
    candidate_strict_json_failure_rate = _strict_json_failure_rate(candidate)

    if candidate_invalid > max_invalid_output_rate:
        reasons.append(f"candidate invalid rate {candidate_invalid:.4f} > {max_invalid_output_rate:.4f}")
    if candidate_strict_json_failure_rate > max_strict_json_failure_rate:
        reasons.append(
            f"candidate strict JSON failure rate {candidate_strict_json_failure_rate:.4f} "
            f"> {max_strict_json_failure_rate:.4f}"
        )
    if candidate_win + 1e-9 < current_win + min_win_rate_delta:
        reasons.append(
            f"candidate win_rate {candidate_win:.4f} < current {current_win:.4f} + delta {min_win_rate_delta:.4f}"
        )
    if candidate_reward + max_reward_regression + 1e-9 < current_reward:
        reasons.append(
            f"candidate reward {candidate_reward:.4f} regressed below current {current_reward:.4f} "
            f"by more than {max_reward_regression:.4f}"
        )
    if candidate_mechanism + max_mechanism_score_regression + 1e-9 < current_mechanism:
        reasons.append(
            f"candidate mechanism_score {candidate_mechanism:.4f} regressed below current "
            f"{current_mechanism:.4f} by more than {max_mechanism_score_regression:.4f}"
        )
    if candidate_missed_lethal > current_missed_lethal + max_missed_visible_lethal_increase:
        reasons.append(
            f"candidate missed_visible_lethal increased {current_missed_lethal} -> {candidate_missed_lethal}"
        )
    if candidate_reason_math > current_reason_math + max_reason_math_contradiction_increase:
        reasons.append(
            "candidate reason_math_contradiction increased "
            f"{current_reason_math} -> {candidate_reason_math}"
        )
    if candidate_lethal_claim_error > current_lethal_claim_error + max_reason_lethal_claim_error_increase:
        reasons.append(
            "candidate reason_claims_lethal_but_action_not_lethal increased "
            f"{current_lethal_claim_error} -> {candidate_lethal_claim_error}"
        )
    if candidate_score_lethal_math > current_score_lethal_math + max_action_score_lethal_math_contradiction_increase:
        reasons.append(
            "candidate action_score_lethal_math_contradiction increased "
            f"{current_score_lethal_math} -> {candidate_score_lethal_math}"
        )

    current_by = _by_encounter(current)
    candidate_by = _by_encounter(candidate)
    current_keys = set(current_by.keys())
    candidate_keys = set(candidate_by.keys())
    if not allow_missing_eval_keys and current_keys != candidate_keys:
        missing = sorted(current_keys - candidate_keys)
        extra = sorted(candidate_keys - current_keys)
        if missing:
            reasons.append(f"candidate eval missing encounter keys: {missing[:5]}")
        if extra:
            reasons.append(f"candidate eval has unexpected encounter keys: {extra[:5]}")

    for key in sorted(current_keys & candidate_keys):
        cur_payload = current_by[key] if isinstance(current_by.get(key), dict) else {}
        cand_payload = candidate_by[key] if isinstance(candidate_by.get(key), dict) else {}
        cur_win = _payload_win(cur_payload)
        cand_win = _payload_win(cand_payload)
        cur_reward = _payload_reward(cur_payload)
        cand_reward = _payload_reward(cand_payload)
        label = str(cur_payload.get("encounter_label") or key)
        if cand_win + max_per_encounter_win_rate_regression + 1e-9 < cur_win:
            reasons.append(
                f"{label}: win_rate regressed {cur_win:.4f} -> {cand_win:.4f}"
            )
        if cand_reward + max_per_encounter_reward_regression + 1e-9 < cur_reward:
            reasons.append(
                f"{label}: reward regressed {cur_reward:.4f} -> {cand_reward:.4f}"
            )
    return (len(reasons) == 0), reasons


def _ensure_sim_release_binary_fresh() -> None:
    """Pre-flight: rebuild HeadlessSim Release binary so launcher's freshness
    guard (``ensure_host_binary_is_fresh`` in bridge/sim/launcher.py) doesn't
    abort every episode after a sim/proto edit.

    Idempotent. dotnet build is a no-op when nothing changed (~1s). When .cs
    sources or proto files are newer than the binary it does a real rebuild
    (~10–60s). Stale Release binary is the most common reason iter runs
    fail with 100% ``reset_failed:RuntimeError``; centralising the rebuild
    here removes the need for ad-hoc ``launch_*.sh`` wrapper scripts.
    """
    csproj = STS2AI_ROOT / "ENV" / "Sim" / "HeadlessSim" / "HeadlessSim.csproj"
    if not csproj.exists():
        # No sim project to build (e.g. running on a worker without ENV/Sim).
        # Skip silently rather than fail; downstream rollout will give a more
        # informative error if the binary turns out to be missing.
        print(f"[self-iterate] skip sim rebuild — csproj not found at {csproj}")
        return
    print(f"[self-iterate] pre-flight: dotnet build -c Release {csproj.name} ...")
    started = time.time()
    proc = subprocess.run(
        ["dotnet", "build", "-c", "Release", "--nologo", "--verbosity", "quiet", str(csproj)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - started
    if proc.returncode != 0:
        # Show last few lines of stderr/stdout so the user can act without
        # hunting through scrollback. We treat this as fatal because rollout
        # will 100% fail without a fresh binary.
        tail_err = "\n".join((proc.stderr or "").splitlines()[-15:])
        tail_out = "\n".join((proc.stdout or "").splitlines()[-15:])
        raise RuntimeError(
            "dotnet build -c Release HeadlessSim failed; rollout cannot start.\n"
            f"--- stderr tail ---\n{tail_err}\n--- stdout tail ---\n{tail_out}\n"
            "Common causes: (1) HeadlessSim.exe is locked by an earlier sim "
            "process — kill leftover HeadlessSim.exe first; (2) compile error "
            "in a .cs file you just edited — run dotnet build manually for a "
            "full diagnostic."
        )
    print(f"[self-iterate] pre-flight: HeadlessSim Release build OK ({elapsed:.1f}s)")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.sim_rebuild:
        _ensure_sim_release_binary_fresh()
    current_adapter = Path(args.current_adapter).resolve()
    if not current_adapter.exists():
        raise FileNotFoundError(f"current adapter not found: {current_adapter}")

    run_name = args.run_name or f"self_iterate_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root = RUNS_ROOT / run_name
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = f"{run_name}_rollout"
    candidate_name = f"{run_name}_candidate"
    planner_candidate_name = f"{run_name}_planner_candidate"
    current_eval_name = f"{run_name}_current_eval"
    candidate_eval_name = f"{run_name}_candidate_eval"
    planner_eval_name = f"{run_name}_planner_candidate_eval"
    joint_eval_name = f"{run_name}_joint_candidate_eval"
    dataset_dir = DATASETS_ROOT / dataset_name
    candidate_run_dir = GRPO_ROOT / candidate_name
    candidate_adapter = candidate_run_dir / "adapter"
    planner_candidate_run_dir = SFT_ROOT / planner_candidate_name
    planner_candidate_adapter = planner_candidate_run_dir / "adapter"
    audit_dir = ARTIFACTS_ROOT / "reviews" / f"{run_name}_rollout_audit"
    kimi_review_dir = ARTIFACTS_ROOT / "reviews" / f"{run_name}_kimi_teacher"
    teacher_dataset_dir = DATASETS_ROOT / f"{run_name}_kimi_teacher"
    planner_hint_dataset_dir = DATASETS_ROOT / f"{run_name}_planner_hint_teacher"
    pool_training_dataset_dir = DATASETS_ROOT / f"{run_name}_managed_pool_train"

    py = str(Path(args.python_exe or _default_python_exe()).resolve())
    if not Path(py).exists():
        raise FileNotFoundError(f"python exe not found: {py}")
    common_model_flags = ["--parse-retries", str(args.parse_retries)]
    if args.load_in_4bit:
        common_model_flags.append("--load-in-4bit")
    planner_flags: list[str] = []
    planner_adapter = Path(args.planner_hint_adapter_dir).resolve() if args.planner_hint_adapter_dir else None
    if planner_adapter is not None:
        if not planner_adapter.exists():
            raise FileNotFoundError(f"planner-hint adapter not found: {planner_adapter}")
        planner_flags = [
            "--planner-hint-adapter-dir", str(planner_adapter),
            "--planner-hint-refresh", args.planner_hint_refresh,
            "--planner-hint-max-new-tokens", str(args.planner_hint_max_new_tokens),
        ]
    teacher_model = args.teacher_model or (args.kimi_model if args.teacher_provider == "kimi" else "")
    planner_train_dataset_override = (
        Path(args.planner_train_dataset_dir).resolve()
        if args.planner_train_dataset_dir
        else None
    )
    if planner_train_dataset_override is not None and not planner_train_dataset_override.exists():
        raise FileNotFoundError(f"planner train dataset not found: {planner_train_dataset_override}")

    rollout_cmd = [
        py, "-m", "llm.training.grpo_rollout",
        "--adapter-dir", str(current_adapter),
        "--out-subdir", dataset_name,
        "--num-generations", str(args.rollout_generations),
        "--max-steps", str(args.rollout_max_steps),
        "--port-base", str(args.rollout_port_base),
        "--temperature", str(args.rollout_temperature),
        "--seed", str(args.seed),
        # Pass max-seq-length to inference too (not just train) so the model
        # is loaded with the right attention window. Otherwise Unsloth
        # silently truncates anything past the model default (2048) and
        # downstream JSON parsing breaks on cut-off ``Return strict JSON``.
        "--max-seq-length", str(args.max_seq_length),
        *common_model_flags,
        *planner_flags,
    ]
    if args.encounter_filter:
        rollout_cmd += ["--encounter-filter", args.encounter_filter]
    if args.tier_filter:
        rollout_cmd += ["--tier-filter", args.tier_filter]
    rollout_cmd += [
        "--case-index", args.case_index,
        "--case-character", args.case_character,
        "--case-floor-min", str(args.case_floor_min),
        "--case-floor-max", str(args.case_floor_max),
        "--case-limit", str(args.case_limit),
        "--case-sample-seed", str(args.case_sample_seed or args.seed),
        "--case-sample-mode", args.case_sample_mode,
        "--elite-oversample-ratio", str(args.elite_oversample_ratio),
        "--boss-oversample-ratio", str(args.boss_oversample_ratio),
    ]
    if args.include_lost_cases:
        rollout_cmd.append("--include-lost-cases")
    if args.no_thinking:
        rollout_cmd.append("--no-thinking")
    if args.allow_json_like_rollout:
        rollout_cmd.append("--allow-json-like-rollout")
    # mask_reason_in_train_data flag is now a no-op — combat policy
    # output no longer carries a reason field; nothing to mask.

    audit_cmd = [
        py, "-m", "llm.scripts.analysis.audit_rollout_failures",
        "--dataset-dir", str(dataset_dir),
        "--out-dir", str(audit_dir),
        "--log", str(logs_dir / "rollout.stderr.log"),
        "--log", str(logs_dir / "rollout.stdout.log"),
    ]
    pool_ingest_dataset_cmd = [
        py, "-m", "llm.scripts.datasets.manage_dataset_pool",
        "ingest-dataset",
        "--dataset-dir", str(dataset_dir),
        "--source-name", run_name,
    ]
    pool_ingest_audit_cmd = [
        py, "-m", "llm.scripts.datasets.manage_dataset_pool",
        "ingest-audit",
        "--audit-dir", str(audit_dir),
        "--dataset-dir", str(dataset_dir),
        "--source-name", run_name,
    ]

    kimi_review_cmd = [
        py, "-m", "llm.scripts.teacher.run_kimi_combat_review_batch",
        "--trace", str(dataset_dir / "step_trace.jsonl"),
        "--out-dir", str(kimi_review_dir),
        "--limit-episodes", str(args.kimi_limit_episodes),
        "--max-api-calls", str(args.kimi_max_api_calls),
        "--provider", args.teacher_provider,
        "--model", teacher_model,
        "--base-url", args.kimi_base_url,
        "--api-key-env", args.kimi_api_key_env,
        "--claude-command", args.teacher_claude_command,
        "--claude-proxy", args.teacher_claude_proxy,
        "--max-workers", str(args.teacher_max_workers),
        "--max-tokens", str(args.kimi_max_tokens),
        "--thinking", args.kimi_thinking,
        "--timeout-s", str(args.kimi_timeout_s),
        "--sleep-s", str(args.kimi_sleep_s),
        "--max-decision-state-chars", str(args.kimi_max_decision_state_chars),
        "--damage-turns", str(args.kimi_damage_turns),
    ]
    if args.kimi_dry_run:
        kimi_review_cmd.append("--dry-run")
    if args.teacher_skip_existing:
        kimi_review_cmd.append("--skip-existing")

    def _train_cmd(train_dataset_dir: Path) -> list[str]:
        cmd = [
            py, "-m", "llm.training.grpo_lite",
            "--adapter-dir", str(current_adapter),
            "--dataset-dir", str(train_dataset_dir),
            "--run-name", candidate_name,
            "--num-epochs", str(args.num_epochs),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--lr", str(args.lr),
            "--max-seq-length", str(args.max_seq_length),
            "--loss-scope", args.grpo_loss_scope,
        ]
        if args.load_in_4bit:
            cmd.append("--load-in-4bit")
        return cmd

    def _planner_train_cmd(train_dataset_dir: Path) -> list[str]:
        cmd = [
            py, "-m", "llm.training.sft_lora",
            "--run-name", planner_candidate_name,
            "--dataset-dir", str(train_dataset_dir),
            "--num-epochs", str(args.planner_num_epochs),
            "--batch-size", str(args.planner_batch_size),
            "--grad-accum", str(args.planner_grad_accum),
            "--lr", str(args.planner_lr),
            "--max-seq-length", str(args.planner_max_seq_length),
        ]
        if args.planner_load_in_4bit:
            cmd.append("--load-in-4bit")
        return cmd

    def _policy_eval_cmd(
        *,
        adapter_dir: Path,
        eval_name: str,
        port_base: int,
        planner_adapter_dir: Path | None,
    ) -> list[str]:
        cmd = [
            py, "-m", "llm.eval.policy_eval",
            "--adapter-dir", str(adapter_dir),
            "--run-name", eval_name,
            "--episodes-per-encounter", str(args.eval_episodes_per_encounter),
            "--max-steps", str(args.eval_max_steps),
            "--port-base", str(port_base),
            "--seed", str(args.seed),
            "--case-index", args.case_index,
            "--case-character", args.case_character,
            "--case-floor-min", str(args.case_floor_min),
            "--case-floor-max", str(args.case_floor_max),
            "--case-limit", str(args.case_limit),
            "--case-sample-seed", str(args.case_sample_seed or args.seed),
            "--case-sample-mode", args.case_sample_mode,
            "--elite-oversample-ratio", str(args.elite_oversample_ratio),
            "--boss-oversample-ratio", str(args.boss_oversample_ratio),
            *common_model_flags,
        ]
        if args.include_lost_cases:
            cmd.append("--include-lost-cases")
        if planner_adapter_dir is not None:
            cmd += [
                "--planner-hint-adapter-dir", str(planner_adapter_dir),
                "--planner-hint-refresh", args.planner_hint_refresh,
                "--planner-hint-max-new-tokens", str(args.planner_hint_max_new_tokens),
            ]
        if args.encounter_filter:
            cmd += ["--encounter-filter", args.encounter_filter]
        return cmd

    current_eval_cmd = _policy_eval_cmd(
        adapter_dir=current_adapter,
        eval_name=current_eval_name,
        port_base=args.eval_port_base,
        planner_adapter_dir=planner_adapter,
    )
    candidate_eval_cmd = _policy_eval_cmd(
        adapter_dir=candidate_adapter,
        eval_name=candidate_eval_name,
        port_base=args.eval_port_base + 100,
        planner_adapter_dir=planner_adapter,
    )
    planner_eval_cmd = _policy_eval_cmd(
        adapter_dir=current_adapter,
        eval_name=planner_eval_name,
        port_base=args.eval_port_base + 200,
        planner_adapter_dir=planner_candidate_adapter,
    )
    joint_eval_cmd = _policy_eval_cmd(
        adapter_dir=candidate_adapter,
        eval_name=joint_eval_name,
        port_base=args.eval_port_base + 300,
        planner_adapter_dir=planner_candidate_adapter,
    )

    # === 新主线 eval：candidate_rollout（用 candidate combat + candidate planner 跑 rollout-style
    # eval-only 评估）。promotion gate 默认只看 rollout 阶段的 current eval_metrics.json
    # 与本 candidate_rollout 的 eval_metrics.json，免去 4 格 policy_eval 的 ~56min 开销。
    candidate_rollout_subdir = f"{run_name}_candidate_rollout"
    candidate_rollout_dir = DATASETS_ROOT / candidate_rollout_subdir
    candidate_planner_for_rollout = (
        Path(planner_candidate_adapter).resolve() if args.co_train_planner else planner_adapter
    )
    candidate_rollout_cmd = [
        py, "-m", "llm.training.grpo_rollout",
        "--adapter-dir", str(candidate_adapter),
        "--out-subdir", candidate_rollout_subdir,
        "--num-generations", str(args.rollout_generations),
        "--max-steps", str(args.rollout_max_steps),
        "--port-base", str(args.rollout_port_base + 1000),  # 避开 train rollout 端口
        "--temperature", str(args.rollout_temperature),
        "--seed", str(args.seed),
        "--eval-only",
        # Same rationale as baseline rollout — keep model load with the same
        # context window so eval metrics aren't biased by silent truncation.
        "--max-seq-length", str(args.max_seq_length),
        *common_model_flags,
    ]
    if candidate_planner_for_rollout is not None:
        candidate_rollout_cmd += [
            "--planner-hint-adapter-dir", str(candidate_planner_for_rollout),
            "--planner-hint-refresh", args.planner_hint_refresh,
            "--planner-hint-max-new-tokens", str(args.planner_hint_max_new_tokens),
        ]
    if args.encounter_filter:
        candidate_rollout_cmd += ["--encounter-filter", args.encounter_filter]
    if args.tier_filter:
        candidate_rollout_cmd += ["--tier-filter", args.tier_filter]
    candidate_rollout_cmd += [
        "--case-index", args.case_index,
        "--case-character", args.case_character,
        "--case-floor-min", str(args.case_floor_min),
        "--case-floor-max", str(args.case_floor_max),
        "--case-limit", str(args.case_limit),
        "--case-sample-seed", str(args.case_sample_seed or args.seed),
        "--case-sample-mode", args.case_sample_mode,
        "--elite-oversample-ratio", str(args.elite_oversample_ratio),
        "--boss-oversample-ratio", str(args.boss_oversample_ratio),
    ]
    if args.include_lost_cases:
        candidate_rollout_cmd.append("--include-lost-cases")
    if args.no_thinking:
        candidate_rollout_cmd.append("--no-thinking")
    if args.allow_json_like_rollout:
        candidate_rollout_cmd.append("--allow-json-like-rollout")

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_root": str(run_root),
        "current_adapter": str(current_adapter),
        "current_planner_adapter": str(planner_adapter) if planner_adapter is not None else None,
        "dataset_dir": str(dataset_dir),
        "rollout_audit_summary": str(audit_dir / "summary.json"),
        "candidate_adapter": str(candidate_adapter),
        "planner_candidate_adapter": str(planner_candidate_adapter) if args.co_train_planner else None,
        "current_rollout_eval_metrics": str(dataset_dir / "eval_metrics.json"),
        "candidate_rollout_eval_metrics": str(candidate_rollout_dir / "eval_metrics.json"),
        # 旧 4 格 policy_eval；只在 isolation_evals 打开时跑，promotion gate 不再依赖。
        "current_eval_metrics": str(ARTIFACTS_ROOT / "evals" / current_eval_name / "metrics.json"),
        "candidate_eval_metrics": str(ARTIFACTS_ROOT / "evals" / candidate_eval_name / "metrics.json"),
        "planner_eval_metrics": (
            str(ARTIFACTS_ROOT / "evals" / planner_eval_name / "metrics.json")
            if args.co_train_planner
            else None
        ),
        "joint_eval_metrics": (
            str(ARTIFACTS_ROOT / "evals" / joint_eval_name / "metrics.json")
            if args.co_train_planner
            else None
        ),
        "kimi_review_summary": str(kimi_review_dir / "summary.json") if args.kimi_teacher else None,
        "teacher_dataset_summary": str(teacher_dataset_dir / "summary.json") if args.kimi_teacher else None,
        "planner_hint_dataset_summary": str(planner_hint_dataset_dir / "summary.json") if args.kimi_teacher else None,
        "planner_training_dataset_dir": str(planner_train_dataset_override) if planner_train_dataset_override else None,
        "planner_candidate_trained": False if args.co_train_planner else None,
        "pool_training_dataset_summary": (
            str(pool_training_dataset_dir / "summary.json")
            if args.kimi_teacher and args.train_from_pool_after_teacher
            else None
        ),
        "args": vars(args),
        "python_exe": py,
        "commands": {
            "rollout": rollout_cmd,
            "rollout_audit": audit_cmd,
            "pool_ingest_dataset": pool_ingest_dataset_cmd,
            "pool_ingest_audit": pool_ingest_audit_cmd,
            "kimi_review": kimi_review_cmd if args.kimi_teacher else None,
            "planner_hint_dataset": (
                [
                    py, "-m", "llm.scripts.datasets.build_planner_hint_dataset",
                    "--review-root", str(kimi_review_dir),
                    "--out-dir", str(planner_hint_dataset_dir),
                    "--seed", str(args.seed),
                ]
                if args.kimi_teacher
                else None
            ),
            "train": _train_cmd(dataset_dir),
            "planner_train": (
                _planner_train_cmd(planner_train_dataset_override or planner_hint_dataset_dir)
                if args.co_train_planner
                else None
            ),
            "current_eval": current_eval_cmd,
            "candidate_eval": candidate_eval_cmd,
            "planner_eval": planner_eval_cmd if args.co_train_planner else None,
            "joint_eval": joint_eval_cmd if args.co_train_planner else None,
        },
        "status": "planned" if args.dry_run else "running",
    }
    _write_json(run_root / "manifest.json", manifest)

    def _run_step(label: str, cmd: list[str]) -> None:
        manifest.setdefault("commands", {})[label] = cmd
        code = _run(
            cmd,
            cwd=Path(__file__).resolve().parents[3],
            stdout_log=logs_dir / f"{label}.stdout.log",
            stderr_log=logs_dir / f"{label}.stderr.log",
            dry_run=args.dry_run,
        )
        manifest.setdefault("step_results", {})[label] = {
            "returncode": code,
            "stdout": str(logs_dir / f"{label}.stdout.log"),
            "stderr": str(logs_dir / f"{label}.stderr.log"),
        }
        _write_json(run_root / "manifest.json", manifest)
        if code != 0:
            manifest["status"] = "failed"
            manifest["failed_step"] = label
            _write_json(run_root / "manifest.json", manifest)
            raise SystemExit(code)

    initial_steps = [
        ("rollout", rollout_cmd),
        ("rollout_audit", audit_cmd),
    ]
    if not args.skip_pool_ingest:
        initial_steps.extend([
            ("pool_ingest_dataset", pool_ingest_dataset_cmd),
            ("pool_ingest_audit", pool_ingest_audit_cmd),
        ])
    else:
        manifest.setdefault("warnings", []).append("Skipped dataset_pool ingest/materialize for this run.")
        _write_json(run_root / "manifest.json", manifest)

    for label, cmd in initial_steps:
        _run_step(label, cmd)

    train_dataset_dir = dataset_dir
    teacher_rows = 0
    if args.kimi_teacher:
        _run_step("kimi_review", kimi_review_cmd)
        if not args.dry_run:
            kimi_summary_path = kimi_review_dir / "summary.json"
            kimi_summary = _metrics(kimi_summary_path)
            gate_passed, gate_reasons = _teacher_review_gate(
                kimi_summary,
                min_review_ok_rate=args.kimi_min_review_ok_rate,
                min_teacher_rows=args.kimi_min_teacher_rows,
            )
            manifest["kimi_teacher_gate"] = {
                "passed": gate_passed,
                "reasons": gate_reasons,
                "summary": str(kimi_summary_path),
                "reviews_ok": int(kimi_summary.get("reviews_ok") or 0),
                "labels": int(kimi_summary.get("labels") or 0),
                "api_calls_before": kimi_summary.get("api_calls_before"),
                "api_calls_after": kimi_summary.get("api_calls_after"),
                "status_counts": kimi_summary.get("status_counts") or {},
                "parse_counts": kimi_summary.get("parse_counts") or {},
            }
            _write_json(run_root / "manifest.json", manifest)
            if not gate_passed and args.kimi_fail_on_quality_gate:
                manifest["status"] = "failed"
                manifest["failed_step"] = "kimi_teacher_gate"
                _write_json(run_root / "manifest.json", manifest)
                raise SystemExit(2)

            review_paths = [str(path) for path in (kimi_summary.get("review_paths") or [])]
            episode_paths = [str(path) for path in (kimi_summary.get("episode_input_paths") or [])]
            if review_paths and len(review_paths) == len(episode_paths):
                planner_hint_dataset_cmd = [
                    py, "-m", "llm.scripts.datasets.build_planner_hint_dataset",
                    "--review-root", str(kimi_review_dir),
                    "--out-dir", str(planner_hint_dataset_dir),
                    "--seed", str(args.seed),
                ]
                _run_step("planner_hint_dataset", planner_hint_dataset_cmd)
                manifest["planner_hint_dataset_rows"] = _dataset_rows(planner_hint_dataset_dir / "summary.json")
                _write_json(run_root / "manifest.json", manifest)

                teacher_dataset_cmd = [
                    py, "-m", "llm.scripts.datasets.build_teacher_dataset",
                    "--out-dir", str(teacher_dataset_dir),
                    "--min-confidence", str(args.kimi_min_confidence),
                    "--seed", str(args.seed),
                    "--review-root", str(kimi_review_dir),
                ]
                if args.kimi_append_experience:
                    teacher_dataset_cmd.append("--append-experience")
                if not args.use_teacher_reasons:
                    # use_teacher_reasons=False means fall back to canonical
                    # template — pass the legacy build flag that toggles this.
                    teacher_dataset_cmd.append("--no-kimi-reasons-in-review")
                _run_step("kimi_teacher_dataset", teacher_dataset_cmd)
                teacher_rows = _dataset_rows(teacher_dataset_dir / "summary.json")
                manifest["kimi_teacher_dataset_rows"] = teacher_rows
                _write_json(run_root / "manifest.json", manifest)
                if teacher_rows > 0 and not args.skip_pool_ingest:
                    pool_ingest_teacher_cmd = [
                        py, "-m", "llm.scripts.datasets.manage_dataset_pool",
                        "ingest-dataset",
                        "--dataset-dir", str(teacher_dataset_dir),
                        "--source-name", f"{run_name}:kimi_teacher",
                    ]
                    _run_step("pool_ingest_kimi_teacher", pool_ingest_teacher_cmd)
            else:
                manifest["kimi_teacher_dataset_rows"] = 0
                manifest["planner_hint_dataset_rows"] = 0
                manifest.setdefault("warnings", []).append(
                    "Kimi produced no matching review_paths/episode_input_paths; teacher dataset skipped."
                )
                _write_json(run_root / "manifest.json", manifest)

            if teacher_rows > 0 and args.train_from_pool_after_teacher and not args.skip_pool_ingest:
                materialize_cmd = [
                    py, "-m", "llm.scripts.datasets.manage_dataset_pool",
                    "materialize",
                    "--out-dir", str(pool_training_dataset_dir),
                    "--target-size", str(args.pool_train_target_size),
                    "--gold-min-ratio", str(args.pool_gold_min_ratio),
                    "--seed", str(args.seed),
                ]
                _run_step("pool_materialize_train", materialize_cmd)
                train_dataset_dir = pool_training_dataset_dir

    planner_candidate_trained = False
    planner_train_dataset_dir = planner_train_dataset_override
    if args.co_train_planner and planner_train_dataset_dir is None:
        planner_hint_rows = int(manifest.get("planner_hint_dataset_rows") or 0)
        if planner_hint_rows >= args.planner_min_train_rows:
            planner_train_dataset_dir = planner_hint_dataset_dir

    if args.co_train_planner and planner_train_dataset_dir is not None:
        planner_train_cmd = _planner_train_cmd(planner_train_dataset_dir)
        manifest["planner_training_dataset_dir"] = str(planner_train_dataset_dir)
        manifest.setdefault("commands", {})["planner_train"] = planner_train_cmd
        _write_json(run_root / "manifest.json", manifest)
        _run_step("planner_train", planner_train_cmd)
        planner_candidate_trained = True
        manifest["planner_candidate_trained"] = True
        manifest["planner_candidate_adapter"] = str(planner_candidate_adapter)
        _write_json(run_root / "manifest.json", manifest)
    elif args.co_train_planner:
        manifest["planner_candidate_trained"] = False
        manifest.setdefault("warnings", []).append(
            "planner co-train requested but no planner dataset met "
            f"planner_min_train_rows={args.planner_min_train_rows}; planner candidate/eval skipped."
        )
        _write_json(run_root / "manifest.json", manifest)

    train_cmd = _train_cmd(train_dataset_dir)
    manifest["training_dataset_dir"] = str(train_dataset_dir)
    manifest.setdefault("commands", {})["train"] = train_cmd
    _write_json(run_root / "manifest.json", manifest)
    _run_step("train", train_cmd)
    # 新主线评估：candidate_rollout（用 candidate combat + candidate planner 跑 rollout-style 评估）
    # 取代旧 4 格 policy_eval。promotion gate 比较 dataset_dir/eval_metrics.json (current 表现，
    # 来自训练阶段的 rollout) vs candidate_rollout_dir/eval_metrics.json (candidate 表现)。
    if args.candidate_rollout:
        _run_step("candidate_rollout", candidate_rollout_cmd)
    else:
        # User opted out — fabricate a "skipped" placeholder so downstream
        # gate code can detect it and short-circuit promotion checks.
        candidate_rollout_dir.mkdir(parents=True, exist_ok=True)
        skip_payload = {
            "kind": "candidate_rollout_skipped",
            "reason": "--no-candidate-rollout flag",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        (candidate_rollout_dir / "eval_metrics.json").write_text(
            json.dumps(skip_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("[self-iterate] candidate_rollout skipped (--no-candidate-rollout); "
              "promotion gate will short-circuit, candidate auto-promoted as next baseline.")

    # 旧 4 格 policy_eval：仅在 --isolation-evals 时跑（debug / 隔离归因），不影响 gate。
    if args.isolation_evals:
        _run_step("current_eval", current_eval_cmd)
        _run_step("candidate_eval", candidate_eval_cmd)
        if planner_candidate_trained:
            _run_step("planner_eval", planner_eval_cmd)
            _run_step("joint_eval", joint_eval_cmd)

    if args.dry_run:
        manifest["status"] = "dry_run"
        _write_json(run_root / "manifest.json", manifest)
        print(f"[self-iterate] dry-run manifest -> {run_root / 'manifest.json'}")
        return

    # 默认主线：rollout-vs-rollout（current 来自训练阶段 rollout 的 eval_metrics.json，
    # candidate 来自 candidate_rollout 的 eval_metrics.json）；
    # 若 isolation_evals 打开，再额外读 4 格 policy_eval metrics 做诊断。
    current_rollout_metrics_path = Path(manifest["current_rollout_eval_metrics"])
    candidate_rollout_metrics_path = Path(manifest["candidate_rollout_eval_metrics"])
    current_rollout_metrics = _metrics(current_rollout_metrics_path)
    candidate_rollout_metrics = _metrics(candidate_rollout_metrics_path)

    current_metrics_path = Path(manifest["current_eval_metrics"])
    candidate_metrics_path = Path(manifest["candidate_eval_metrics"])
    current_metrics = _metrics(current_metrics_path) if args.isolation_evals else {}
    candidate_metrics = _metrics(candidate_metrics_path) if args.isolation_evals else {}
    planner_metrics = (
        _metrics(Path(manifest["planner_eval_metrics"]))
        if (planner_candidate_trained and args.isolation_evals)
        else {}
    )
    joint_metrics = (
        _metrics(Path(manifest["joint_eval_metrics"]))
        if (planner_candidate_trained and args.isolation_evals)
        else {}
    )

    def _gate_with_baseline(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, list[str]]:
        return _candidate_passes(
            current=baseline,
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
            max_action_score_lethal_math_contradiction_increase=(
                args.max_action_score_lethal_math_contradiction_increase
            ),
            max_strict_json_failure_rate=args.max_strict_json_failure_rate,
            allow_missing_eval_keys=args.allow_missing_eval_keys,
        )

    def _gate(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
        # 兼容旧调用点：基准用 4 格 policy_eval 的 current_metrics
        return _gate_with_baseline(current_metrics, candidate)

    # 主线 gate：rollout-vs-rollout，比较 current 训练阶段 rollout 与 candidate_rollout。
    rollout_gate_passed, rollout_gate_reasons = _gate_with_baseline(
        current_rollout_metrics, candidate_rollout_metrics
    )

    # 旧 4 格 gate（仅在 isolation_evals 时有意义）
    if args.isolation_evals:
        combat_passed, combat_reasons = _gate(candidate_metrics)
    else:
        combat_passed, combat_reasons = (None, [])
    if planner_candidate_trained and args.isolation_evals:
        planner_passed, planner_reasons = _gate(planner_metrics)
    else:
        planner_passed, planner_reasons = (None, [])
    joint_passed, joint_reasons = (
        _gate(joint_metrics) if (planner_candidate_trained and args.isolation_evals) else (None, [])
    )

    # 最终 promotion 由 rollout-vs-rollout 决定（默认主线）
    passed = bool(rollout_gate_passed)
    reasons = rollout_gate_reasons

    # candidate / planner_candidate 字段仅在 isolation_evals 打开时填，否则 None
    # （表示"未跑该格 eval"，区别于"跑了但全 0"）
    rollout_main = {
        "baseline": {
            "adapter": str(current_adapter),
            "planner_adapter": str(planner_adapter) if planner_adapter is not None else None,
            "metrics_path": str(current_rollout_metrics_path),
            "summary": _eval_summary(current_adapter, current_rollout_metrics),
            "by_tier": current_rollout_metrics.get("by_tier") if isinstance(current_rollout_metrics, dict) else None,
        },
        "candidate": {
            "adapter": str(candidate_adapter),
            "planner_adapter": (
                str(planner_candidate_adapter) if planner_candidate_trained else (
                    str(planner_adapter) if planner_adapter is not None else None
                )
            ),
            "metrics_path": str(candidate_rollout_metrics_path),
            "summary": _eval_summary(candidate_adapter, candidate_rollout_metrics),
            "by_tier": candidate_rollout_metrics.get("by_tier") if isinstance(candidate_rollout_metrics, dict) else None,
        },
        "passed": rollout_gate_passed,
        "reasons": rollout_gate_reasons,
    }
    promotion = {
        "passed": passed,
        "reasons": reasons,
        "mode": "joint_combat_planner" if planner_candidate_trained else "combat_only",
        "isolation_evals": bool(args.isolation_evals),
        "rollout_main": rollout_main,
        "current": _eval_summary(current_adapter, current_metrics),
        "candidate": (
            _eval_summary(candidate_adapter, candidate_metrics)
            if (args.isolation_evals or not planner_candidate_trained)
            else None
        ),
        "planner_candidate": (
            _eval_summary(current_adapter, planner_metrics)
            if (planner_candidate_trained and args.isolation_evals)
            else None
        ),
        "joint_candidate": (
            _eval_summary(candidate_adapter, joint_metrics)
            if planner_candidate_trained
            else None
        ),
        "matrix": {
            "current_combat_current_planner": {
                "combat_adapter": str(current_adapter),
                "planner_adapter": str(planner_adapter) if planner_adapter is not None else None,
                "metrics": _eval_summary(current_adapter, current_metrics),
                "metrics_path": str(current_metrics_path),
            },
            "candidate_combat_current_planner": (
                {
                    "combat_adapter": str(candidate_adapter),
                    "planner_adapter": str(planner_adapter) if planner_adapter is not None else None,
                    "metrics": _eval_summary(candidate_adapter, candidate_metrics),
                    "metrics_path": str(candidate_metrics_path),
                    "passed_vs_current": combat_passed,
                    "reasons": combat_reasons,
                }
                if args.isolation_evals or not planner_candidate_trained
                else None
            ),
            "current_combat_candidate_planner": (
                {
                    "combat_adapter": str(current_adapter),
                    "planner_adapter": str(planner_candidate_adapter),
                    "metrics": _eval_summary(current_adapter, planner_metrics),
                    "metrics_path": str(Path(manifest["planner_eval_metrics"])),
                    "passed_vs_current": planner_passed,
                    "reasons": planner_reasons,
                }
                if planner_candidate_trained and args.isolation_evals
                else None
            ),
            "candidate_combat_candidate_planner": (
                {
                    "combat_adapter": str(candidate_adapter),
                    "planner_adapter": str(planner_candidate_adapter),
                    "metrics": _eval_summary(candidate_adapter, joint_metrics),
                    "metrics_path": str(Path(manifest["joint_eval_metrics"])),
                    "passed_vs_current": joint_passed,
                    "reasons": joint_reasons,
                }
                if planner_candidate_trained
                else None
            ),
        },
        "by_encounter": {
            "current": _by_encounter(current_metrics),
            "candidate": _by_encounter(candidate_metrics),
            "planner_candidate": _by_encounter(planner_metrics) if planner_candidate_trained else None,
            "joint_candidate": _by_encounter(joint_metrics) if planner_candidate_trained else None,
        },
    }
    manifest["promotion"] = promotion
    manifest["status"] = "completed"

    if args.promote and passed:
        pointer = ARTIFACTS_ROOT / "current_adapter.json"
        _write_json(pointer, {
            "adapter_dir": str(candidate_adapter),
            "promoted_at": datetime.now().isoformat(timespec="seconds"),
            "source_run": str(run_root),
            "promotion": promotion,
        })
        manifest["promoted_pointer"] = str(pointer)
        if planner_candidate_trained:
            planner_pointer = ARTIFACTS_ROOT / "current_planner_hint_adapter.json"
            _write_json(planner_pointer, {
                "adapter_dir": str(planner_candidate_adapter),
                "promoted_at": datetime.now().isoformat(timespec="seconds"),
                "source_run": str(run_root),
                "promotion": promotion,
            })
            manifest["promoted_planner_pointer"] = str(planner_pointer)
    elif args.promote and not passed:
        manifest["promoted_pointer"] = None
        if args.co_train_planner:
            manifest["promoted_planner_pointer"] = None

    _write_json(run_root / "manifest.json", manifest)
    _write_json(run_root / "promotion.json", promotion)
    print(f"[self-iterate] promotion passed={passed}")
    if reasons:
        for reason in reasons:
            print(f"[self-iterate] gate: {reason}")
    print(f"[self-iterate] manifest -> {run_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
