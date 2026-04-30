"""用 GameSession(full_run) + combat/non-combat 启发式老师，跑完整局 rollout。

和 rollout_heuristic.py 的区别：
- 后者只打单场战斗（GameSession combat mode），数据全是 combat state
- 本脚本走完整 Act1（FullRunSession），收集战斗 + event + map + card_reward
  等所有 state_type 的样本

运行（全局 Python 3.13）：

    python STS2AI/llm/data_pipeline/rollout_full_run.py \\
        --episodes 10 --max-steps 400 --port-base 15660 \\
        --out-subdir heuristic_fullrun_v0

产物：
  STS2AI/Artifacts/llm/datasets/heuristic_fullrun_v0/
    train.jsonl / eval.jsonl / meta.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LLM_ROOT = Path(__file__).resolve().parents[1]
_STS2AI_ROOT = _LLM_ROOT.parent
_BRIDGE_ROOT = _STS2AI_ROOT / "bridge"
for p in (_STS2AI_ROOT, _BRIDGE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm.data_pipeline.heuristic_teacher import pick_action as pick_combat_action
from llm.data_pipeline.non_combat_teacher import pick_non_combat
from llm.data_pipeline.state_renderer import render_state_text
from llm.paths import DATASETS_ROOT, ensure_dirs
from llm.prompts import load_system_prompt

from game_bridge.session import create_game_session
from game_bridge.session.state_semantics import is_actionable_combat_state, is_combat_state


_COMBAT_STATE_TYPES = {"monster", "elite", "boss", "hand_select"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--port-base", type=int, default=15660)
    p.add_argument("--out-subdir", type=str, default="heuristic_fullrun_v0")
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--character-id", type=str, default="IRONCLAD")
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--max-recoverable-errors", type=int, default=10)
    return p.parse_args()


def _enabled(legal: list[Any]) -> list[dict[str, Any]]:
    return [a for a in (legal or []) if isinstance(a, dict) and a.get("is_enabled") is not False]


@dataclass
class EpisodeRecord:
    seed: str
    outcome: str
    total_steps: int
    combat_samples: int
    non_combat_samples: int
    recoverable_errors: int
    duration_s: float
    state_type_counts: dict[str, int] = field(default_factory=dict)


def _build_sample(
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    chosen_index: int,
    reason: str,
    system_prompt: str,
    *,
    source: str,
) -> dict[str, Any]:
    user_msg = render_state_text(state, legal)
    assistant_msg = json.dumps(
        {"action_index": int(chosen_index), "reason": reason[:200]},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "meta": {
            "state_type": str(state.get("state_type") or ""),
            "source": source,
        },
    }


def run_episodes(
    *,
    episodes: int,
    max_steps: int,
    port_base: int,
    character_id: str,
    seed: int,
    max_recoverable_errors: int,
) -> tuple[list[dict], list[EpisodeRecord]]:
    system_prompt = load_system_prompt()
    rng = random.Random(seed)
    samples: list[dict] = []
    episode_records: list[EpisodeRecord] = []

    for ep_idx in range(episodes):
        port = port_base + ep_idx
        ep_seed = f"{seed}-{ep_idx}-{rng.randint(0, 10**9)}"
        session = create_game_session(
            mode="full_run", transport="pipe_proto", backend="sim", port=port, auto_launch=True,
        )
        t0 = time.monotonic()
        combat_n = non_combat_n = 0
        state_type_counts: Counter = Counter()
        recoverable_errors = 0
        outcome = "unknown"
        steps_taken = 0
        try:
            try:
                state = session.reset(character_id=character_id, seed=ep_seed)
            except Exception as exc:
                outcome = f"reset_failed:{type(exc).__name__}"
                print(f"[full-rollout][ep{ep_idx}] RESET FAIL: {exc}")
                continue

            for step in range(max_steps):
                st = str(state.get("state_type") or "").lower()
                state_type_counts[st] += 1
                legal_enabled = _enabled(state.get("legal_actions") or [])
                if not legal_enabled:
                    # 空 legal：游戏可能结算中，等一下再读
                    time.sleep(0.2)
                    try:
                        state = session.get_state()
                    except Exception:
                        outcome = "get_state_failed_on_idle"
                        break
                    legal_enabled = _enabled(state.get("legal_actions") or [])
                    if not legal_enabled:
                        if state.get("terminal"):
                            outcome = str(state.get("run_outcome") or "terminal")
                        else:
                            outcome = "no_legal_after_idle"
                        break

                # 路由老师
                is_combat = st in _COMBAT_STATE_TYPES
                if is_combat and is_actionable_combat_state(state):
                    decision = pick_combat_action(state, legal_enabled)
                    chosen_index = decision.action_index
                    reason = decision.reason
                    samples.append(_build_sample(
                        state, legal_enabled, chosen_index, reason, system_prompt,
                        source="combat_heuristic",
                    ))
                    combat_n += 1
                else:
                    nc = pick_non_combat(state, legal_enabled)
                    if nc is None:
                        chosen_index = 0
                        reason = "default idx=0"
                    else:
                        chosen_index = nc.action_index
                        reason = nc.reason
                    # 战斗结算态（is_combat 但不 actionable）不记录样本，只推进
                    if not (is_combat and not is_actionable_combat_state(state)):
                        samples.append(_build_sample(
                            state, legal_enabled, chosen_index, reason, system_prompt,
                            source="non_combat_heuristic",
                        ))
                        non_combat_n += 1

                chosen = legal_enabled[chosen_index]
                prev_state = state
                try:
                    new_state = session.act(chosen)
                    # 等 state 实际发生变化，避免读到未结算的瞬时态
                    if new_state == prev_state:
                        try:
                            new_state = session.get_state()
                        except Exception:
                            pass
                except Exception as exc:
                    recoverable_errors += 1
                    if recoverable_errors <= 3:
                        # 首次几次错误打印完整细节便于排查
                        print(
                            f"[full-rollout][ep{ep_idx}] step {step} st={st} "
                            f"act_rejected: {type(exc).__name__}: {str(exc)[:200]}"
                        )
                        print(f"  chosen: {json.dumps(chosen, ensure_ascii=False)[:200]}")
                        print(f"  all legal ({len(legal_enabled)}):")
                        for i, a in enumerate(legal_enabled[:8]):
                            print(f"    [{i}] {json.dumps(a, ensure_ascii=False)[:200]}")
                        # 看当前手牌
                        hand = (state.get("battle") or {}).get("hand") or []
                        print(f"  hand ({len(hand)}):")
                        for i, c in enumerate(hand[:8]):
                            print(f"    [{i}] id={c.get('id')} cost={c.get('cost')} req_target={c.get('requires_target')} can_play={c.get('can_play')}")
                    if recoverable_errors > max_recoverable_errors:
                        outcome = f"too_many_rejections:{type(exc).__name__}"
                        break
                    # 给 sim 一点时间喘口气再刷 state
                    time.sleep(0.1)
                    try:
                        new_state = session.get_state()
                    except Exception:
                        outcome = "state_refresh_failed"
                        break
                state = new_state if isinstance(new_state, dict) else state
                steps_taken = step + 1

                if state.get("terminal") or state.get("run_outcome"):
                    outcome = str(state.get("run_outcome") or "terminal")
                    break
            else:
                outcome = "max_steps"
        finally:
            try:
                session.close()
            except Exception:
                pass

        duration = time.monotonic() - t0
        episode_records.append(EpisodeRecord(
            seed=ep_seed,
            outcome=outcome,
            total_steps=steps_taken,
            combat_samples=combat_n,
            non_combat_samples=non_combat_n,
            recoverable_errors=recoverable_errors,
            duration_s=round(duration, 2),
            state_type_counts=dict(state_type_counts),
        ))
        print(
            f"[full-rollout][ep{ep_idx}] outcome={outcome} steps={steps_taken} "
            f"combat={combat_n} non_combat={non_combat_n} errors={recoverable_errors} "
            f"dur={duration:.1f}s total_samples={len(samples)}"
        )

    return samples, episode_records


def main() -> None:
    args = parse_args()
    ensure_dirs()
    out_dir = DATASETS_ROOT / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[full-rollout] episodes={args.episodes} max_steps={args.max_steps} out={out_dir}")

    samples, episodes = run_episodes(
        episodes=args.episodes,
        max_steps=args.max_steps,
        port_base=args.port_base,
        character_id=args.character_id,
        seed=args.seed,
        max_recoverable_errors=args.max_recoverable_errors,
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

    source_counts: Counter = Counter()
    state_type_counts: Counter = Counter()
    for s in samples:
        source_counts[s["meta"].get("source", "?")] += 1
        state_type_counts[s["meta"].get("state_type", "?")] += 1

    meta = {
        "run_id": uuid.uuid4().hex,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": args.episodes,
        "total_samples": len(samples),
        "train_size": len(train_samples),
        "eval_size": len(eval_samples),
        "source_counts": dict(source_counts),
        "state_type_counts": dict(state_type_counts),
        "outcome_counts": dict(Counter(ep.outcome for ep in episodes)),
        "episodes_detail": [
            {
                "seed": ep.seed,
                "outcome": ep.outcome,
                "total_steps": ep.total_steps,
                "combat_samples": ep.combat_samples,
                "non_combat_samples": ep.non_combat_samples,
                "recoverable_errors": ep.recoverable_errors,
                "duration_s": ep.duration_s,
                "state_type_counts": ep.state_type_counts,
            }
            for ep in episodes
        ],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(
        f"\n[full-rollout] total={len(samples)} "
        f"combat={source_counts.get('combat_heuristic', 0)} "
        f"non_combat={source_counts.get('non_combat_heuristic', 0)}"
    )
    print(f"[full-rollout] state_type distribution: {dict(state_type_counts)}")
    print(f"[full-rollout] outcome: {dict(Counter(ep.outcome for ep in episodes))}")


if __name__ == "__main__":
    main()
