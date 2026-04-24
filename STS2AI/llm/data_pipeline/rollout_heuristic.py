"""用启发式老师 + GameSession(combat) 跑多场 rollout，产出 LLM SFT 数据。

运行（全局 Python 3.13，带 protobuf；不依赖 unsloth）：

    python STS2AI/llm/data_pipeline/rollout_heuristic.py \\
        --episodes-per-encounter 30 \\
        --max-steps 120 \\
        --port-base 15540 \\
        --out-subdir heuristic_act1_v0

产物：
  STS2AI/Artifacts/llm/datasets/heuristic_act1_v0/
    train.jsonl      # messages 格式，每行一条 (state, legal, chosen)
    eval.jsonl
    meta.json        # 总步数、胜负分布、每个 encounter 的 outcome
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 让脚本能直接 `python rollout_heuristic.py` 执行
_LLM_ROOT = Path(__file__).resolve().parents[1]
_STS2AI_ROOT = _LLM_ROOT.parent
_BRIDGE_ROOT = _STS2AI_ROOT / "bridge"
for p in (_STS2AI_ROOT, _BRIDGE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm.data_pipeline.encounter_pool import ACT1_POOL, EncounterSpec
from llm.data_pipeline.heuristic_teacher import pick_action
from llm.data_pipeline.state_renderer import render_state_text
from llm.paths import DATASETS_ROOT, ensure_dirs
from llm.prompts import load_system_prompt

from game_bridge.session import create_game_session
from game_bridge.session.state_semantics import (
    is_actionable_combat_state,
    is_combat_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-per-encounter", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--port-base", type=int, default=15540)
    p.add_argument("--out-subdir", type=str, default="heuristic_act1_v0")
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--encounter-filter", type=str, default="",
                   help="只跑 encounter_id 含此子串的")
    return p.parse_args()


@dataclass
class EpisodeRecord:
    encounter_id: str
    outcome: str
    steps: int
    duration_s: float
    reason_samples: list[str] = field(default_factory=list)


def _enabled_actions(legal: list[Any]) -> list[dict[str, Any]]:
    return [
        dict(a) for a in (legal or [])
        if isinstance(a, dict) and a.get("is_enabled") is not False
    ]


def _extract_action_dict(action_obj: Any) -> dict[str, Any]:
    """action 可能是 proto 或 dict，这里统一成 dict。"""
    if isinstance(action_obj, dict):
        return action_obj
    try:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(action_obj, preserving_proto_field_name=True)
    except Exception:
        return {}


def _build_sft_sample(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    chosen_index: int,
    reason: str,
    system_prompt: str,
    *,
    encounter_id: str = "",
) -> dict[str, Any]:
    user_msg = render_state_text(state, legal, encounter_id=encounter_id)
    assistant_msg = json.dumps(
        {"action_index": int(chosen_index), "reason": reason[:80]},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "meta": {
            "encounter_id": encounter_id or str((state.get("battle") or {}).get("encounter_id") or ""),
            "state_type": str(state.get("state_type") or ""),
        },
    }


def run_rollout(
    encounters: list[EncounterSpec],
    *,
    episodes_per_encounter: int,
    max_steps: int,
    port_base: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[EpisodeRecord]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    episodes: list[EpisodeRecord] = []
    system_prompt = load_system_prompt()

    for enc_idx, spec in enumerate(encounters):
        port = port_base + enc_idx
        for ep_idx in range(episodes_per_encounter):
            ep_seed = f"{seed}-{enc_idx}-{ep_idx}-{rng.randint(0, 10**9)}"
            session = create_game_session(mode="combat", transport="pipe_proto", backend="sim", port=port, auto_launch=True)
            episode_started = time.monotonic()
            step_count = 0
            outcome = "unknown"
            reason_samples: list[str] = []
            try:
                try:
                    state = session.reset(
                        character_id="IRONCLAD",
                        encounter_id=spec.encounter_id,
                        build=spec.build,
                        seed=ep_seed,
                    )
                except Exception as exc:
                    outcome = f"reset_failed:{type(exc).__name__}"
                    print(f"[rollout][{spec.encounter_id}][ep{ep_idx}] RESET FAIL: {exc}")
                    continue

                for step_idx in range(max_steps):
                    if not is_combat_state(state):
                        outcome = "left_combat"
                        break
                    legal_enabled = _enabled_actions(state.get("legal_actions") or [])
                    if not legal_enabled:
                        outcome = "no_legal_actions"
                        break
                    # 只在可决策的战斗阶段记录样本（过滤结算动画等）
                    if is_actionable_combat_state(state):
                        decision = pick_action(state, legal_enabled)
                        chosen_raw = legal_enabled[decision.action_index]
                        samples.append(
                            _build_sft_sample(
                                state,
                                legal_enabled,
                                decision.action_index,
                                decision.reason,
                                system_prompt,
                                encounter_id=spec.encounter_id,
                            )
                        )
                        if len(reason_samples) < 3:
                            reason_samples.append(decision.reason)
                    else:
                        # 非可决策（结算/等待）：直接选第一个合法动作推进
                        chosen_raw = legal_enabled[0]

                    try:
                        step_result = session.act_gym(chosen_raw)
                    except Exception as exc:
                        outcome = f"step_failed:{type(exc).__name__}"
                        print(f"[rollout][{spec.encounter_id}][ep{ep_idx}] STEP FAIL @{step_idx}: {exc}")
                        break

                    # session.act_gym 返回 (state, reward, done, info)
                    if isinstance(step_result, tuple) and len(step_result) >= 3:
                        state, _reward, done, _info = (
                            step_result[0], step_result[1], step_result[2],
                            step_result[3] if len(step_result) > 3 else {},
                        )
                    else:
                        state = step_result
                        done = False

                    step_count = step_idx + 1
                    if done:
                        outcome = str(state.get("run_outcome") or state.get("combat_outcome") or "terminal") or "terminal"
                        break
                else:
                    outcome = "max_steps"
            finally:
                try:
                    session.close()
                except Exception:
                    pass

            duration = time.monotonic() - episode_started
            episodes.append(
                EpisodeRecord(
                    encounter_id=spec.encounter_id,
                    outcome=outcome,
                    steps=step_count,
                    duration_s=round(duration, 2),
                    reason_samples=reason_samples,
                )
            )
            print(
                f"[rollout][{spec.encounter_id}][ep{ep_idx}] "
                f"outcome={outcome} steps={step_count} duration={duration:.1f}s samples_total={len(samples)}"
            )

    return samples, episodes


def main() -> None:
    args = parse_args()
    ensure_dirs()

    out_dir = DATASETS_ROOT / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = ACT1_POOL
    if args.encounter_filter:
        needle = args.encounter_filter.lower()
        pool = [p for p in ACT1_POOL if needle in p.encounter_id.lower()]
    if not pool:
        raise SystemExit("no encounters matched filter")

    print(f"[rollout] pool size={len(pool)} episodes_per={args.episodes_per_encounter} out={out_dir}")

    samples, episodes = run_rollout(
        pool,
        episodes_per_encounter=args.episodes_per_encounter,
        max_steps=args.max_steps,
        port_base=args.port_base,
        seed=args.seed,
    )

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    eval_n = max(1, int(len(samples) * args.eval_ratio))
    eval_samples = samples[:eval_n]
    train_samples = samples[eval_n:]

    def _dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _dump(out_dir / "train.jsonl", train_samples)
    _dump(out_dir / "eval.jsonl", eval_samples)

    outcome_counts: dict[str, int] = {}
    for ep in episodes:
        outcome_counts[ep.outcome] = outcome_counts.get(ep.outcome, 0) + 1

    meta = {
        "run_id": uuid.uuid4().hex,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pool": [{"encounter_id": p.encounter_id, "tag": p.tag} for p in pool],
        "episodes_per_encounter": args.episodes_per_encounter,
        "total_episodes": len(episodes),
        "total_samples": len(samples),
        "train_size": len(train_samples),
        "eval_size": len(eval_samples),
        "outcomes": outcome_counts,
        "episodes": [
            {
                "encounter_id": ep.encounter_id,
                "outcome": ep.outcome,
                "steps": ep.steps,
                "duration_s": ep.duration_s,
                "reason_samples": ep.reason_samples,
            }
            for ep in episodes
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[rollout] total_samples={len(samples)} train={len(train_samples)} eval={len(eval_samples)}")
    print(f"[rollout] outcome counts: {outcome_counts}")
    print(f"[rollout] meta -> {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
