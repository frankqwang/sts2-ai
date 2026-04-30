"""Fullrun (act1 完整通关) 评估器：用 combat LoRA + planner LoRA 跑从 floor 1 到
boss 的真实路径，看 act1 通关率，而不是 single-combat reset 的虚假分层指标。

设计原则
========
- single-combat eval（``policy_eval`` / ``grpo_rollout --eval-only``）只能看每场战
  斗的胜率/损血，**无法反映**：玩家是否能从 floor 1 撑到 floor 17 boss、boss 自
  身能否过、map 路径选择和 build 累积是否合理。
- fullrun_eval 让 model 真实跑 act1，统计：
    - act1 clear rate（floor 17 boss 击败率）
    - floor reached: 死在哪一层
    - boss outcome 分布
    - 每场总损血（cumulative）/ 总回合数 / 累积 invalid_output_steps
  这些是 promotion 决策应该看的"上层"指标。

实现注意
========
- 当前没有训过 non-combat LoRA；fullrun 中的 map / event / shop / card_reward / rest
  state 都用 ``pick_non_combat`` 启发式选择。combat 部分才用 LLM policy。
- 这反映"combat 训得好 + non-combat 启发式"的真实 act1 通关上限；non-combat 不再
  是评估变量。
- 工具不参与 self_iterate 主循环，是手动 eval（资源开销大：N episodes × 200+ step）。

用法::

    python -m llm.eval.fullrun_eval \\
        --adapter-dir <combat LoRA path> \\
        --planner-hint-adapter-dir <planner LoRA path> \\
        --episodes 10 \\
        --max-steps 400 \\
        --port-base 25640 \\
        --out-dir <out>

输出 ``fullrun_metrics.json``：act1_clear_rate / floor_reached / outcome 分布等。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LLM_ROOT = Path(__file__).resolve().parents[1]
_STS2AI_ROOT = _LLM_ROOT.parent
_BRIDGE_ROOT = _STS2AI_ROOT / "bridge"
for _p in (_STS2AI_ROOT, _BRIDGE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from llm.data_pipeline.non_combat_teacher import pick_non_combat  # noqa: E402
from llm.data_pipeline.planner_hint import (  # noqa: E402
    DEFAULT_PLANNER_HINT_REFRESH,
    PLANNER_HINT_REFRESH_CHOICES,
)
from llm.training.grpo_rollout import _RolloutPolicy  # noqa: E402
from llm.paths import ARTIFACTS_ROOT, ensure_dirs  # noqa: E402

from game_bridge.session import create_game_session  # noqa: E402
from game_bridge.session.state_semantics import is_actionable_combat_state, is_combat_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM policy fullrun act1 evaluator.")
    p.add_argument("--adapter-dir", type=str, default=None, help="combat LoRA adapter; None=base model")
    p.add_argument("--planner-hint-adapter-dir", type=str, default=None)
    p.add_argument(
        "--planner-hint-refresh",
        choices=list(PLANNER_HINT_REFRESH_CHOICES),
        default=DEFAULT_PLANNER_HINT_REFRESH,
    )
    p.add_argument("--planner-hint-max-new-tokens", type=int, default=240)
    p.add_argument("--episodes", type=int, default=10, help="完整 act1 跑多少次")
    p.add_argument("--max-steps", type=int, default=400, help="单场 fullrun 最长 step (含战斗+非战斗)")
    p.add_argument("--port-base", type=int, default=25640)
    p.add_argument("--character-id", type=str, default="IRONCLAD")
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--parse-retries", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--run-name", type=str, default="")
    return p.parse_args()


@dataclass
class FullrunEpisode:
    seed: str
    final_act: int = 1
    final_floor: int = 0
    max_floor: int = 0
    outcome: str = "unknown"
    duration_s: float = 0.0
    steps: int = 0
    combat_steps: int = 0
    non_combat_steps: int = 0
    invalid_output_steps: int = 0
    state_type_counts: Counter = field(default_factory=Counter)
    floor_reached_combat_clears: int = 0  # 战斗胜场数
    death_floor: int | None = None
    death_state_type: str | None = None
    error: str = ""


def _enabled_actions(legal: list[Any]) -> list[dict[str, Any]]:
    return [dict(a) for a in (legal or []) if isinstance(a, dict) and a.get("is_enabled") is not False]


def _floor_from_state(state: dict[str, Any]) -> int:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    try:
        return int(run.get("floor") or 0)
    except (TypeError, ValueError):
        return 0


def _act_from_state(state: dict[str, Any]) -> int:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    try:
        return int(run.get("act") or 1)
    except (TypeError, ValueError):
        return 1


def _is_terminal_outcome(state: dict[str, Any]) -> tuple[bool, str | None]:
    """检查 state 是否标记 run 终止；返回 (terminal, reason)。"""
    if state.get("terminal"):
        run_outcome = str(state.get("run_outcome") or "").lower()
        return True, run_outcome or "terminal"
    return False, None


def run_one_fullrun(policy: _RolloutPolicy, *, port: int, seed: str, character_id: str, max_steps: int) -> FullrunEpisode:
    """跑一次完整 act1，combat 用 LLM policy，non-combat 用启发式。"""
    ep = FullrunEpisode(seed=seed)
    t0 = time.monotonic()

    session = create_game_session(
        mode="full_run", transport="pipe_proto", backend="sim", port=port, auto_launch=True,
    )

    try:
        try:
            state = session.reset(character_id=character_id, seed=seed)
        except Exception as exc:
            ep.outcome = f"reset_failed:{type(exc).__name__}"
            ep.error = str(exc)
            return ep

        for step_idx in range(max_steps):
            ep.steps = step_idx
            cur_floor = _floor_from_state(state)
            cur_act = _act_from_state(state)
            ep.final_floor = cur_floor
            ep.final_act = cur_act
            ep.max_floor = max(ep.max_floor, cur_floor)
            state_type = str(state.get("state_type") or "").lower()
            ep.state_type_counts[state_type] += 1

            terminal, reason = _is_terminal_outcome(state)
            if terminal:
                ep.outcome = reason or "terminal"
                ep.death_floor = cur_floor
                ep.death_state_type = state_type
                break

            if cur_act >= 2:
                # act1 通关（进入 act2 起点）
                ep.outcome = "act1_cleared"
                break

            legal = _enabled_actions(state.get("legal_actions") or [])
            if not legal:
                time.sleep(0.1)
                try:
                    state = session.get_state()
                except Exception as exc:
                    ep.outcome = f"observe_failed:{type(exc).__name__}"
                    ep.error = str(exc)
                    break
                continue

            try:
                if is_combat_state(state) and is_actionable_combat_state(state):
                    decision = policy.select_action(state, legal)
                    ep.combat_steps += 1
                    if decision.invalid_output:
                        ep.invalid_output_steps += 1
                        action_index = -1
                    else:
                        action_index = decision.action_index
                    if action_index < 0 or action_index >= len(legal):
                        # invalid output → 选第一个 legal action 兜底（不算战术决策，只为推进 episode）
                        action_index = 0
                else:
                    chosen = pick_non_combat(state, legal)
                    ep.non_combat_steps += 1
                    action_index = chosen.action_index if chosen and chosen.action_index >= 0 else 0
            except Exception as exc:
                ep.outcome = f"policy_failed:{type(exc).__name__}"
                ep.error = str(exc)
                break

            try:
                state = session.step(action_index)
            except Exception as exc:
                ep.outcome = f"step_failed:{type(exc).__name__}"
                ep.error = str(exc)
                break
        else:
            ep.outcome = "max_steps"
            ep.death_floor = ep.final_floor
            ep.death_state_type = state_type if state_type else None

    finally:
        try:
            session.close()
        except Exception:
            pass

    ep.duration_s = round(time.monotonic() - t0, 2)
    return ep


def aggregate_metrics(episodes: list[FullrunEpisode]) -> dict[str, Any]:
    if not episodes:
        return {"episodes": 0}
    total = len(episodes)
    cleared = sum(1 for ep in episodes if ep.outcome == "act1_cleared")
    floors = [ep.max_floor for ep in episodes]
    death_floors: Counter = Counter()
    outcome_counts: Counter = Counter(ep.outcome for ep in episodes)
    state_totals: Counter = Counter()
    for ep in episodes:
        if ep.outcome != "act1_cleared" and ep.death_floor is not None:
            death_floors[ep.death_floor] += 1
        for k, v in ep.state_type_counts.items():
            state_totals[k] += v
    durations = [ep.duration_s for ep in episodes]
    invalid_steps = [ep.invalid_output_steps for ep in episodes]
    combat_steps = [ep.combat_steps for ep in episodes]

    return {
        "episodes": total,
        "act1_cleared": cleared,
        "act1_clear_rate": round(cleared / total, 4),
        "floor_reached": {
            "min": min(floors),
            "max": max(floors),
            "avg": round(sum(floors) / total, 2),
            "p50": sorted(floors)[total // 2],
        },
        "outcome_counts": {k: int(v) for k, v in outcome_counts.most_common()},
        "death_floor_counts": {int(k): int(v) for k, v in death_floors.most_common()},
        "state_type_total": {k: int(v) for k, v in state_totals.most_common()},
        "duration_s": {
            "min": min(durations),
            "max": max(durations),
            "avg": round(sum(durations) / total, 2),
        },
        "invalid_output_steps_avg": round(sum(invalid_steps) / total, 2),
        "combat_steps_avg": round(sum(combat_steps) / total, 2),
    }


def main() -> None:
    args = parse_args()
    ensure_dirs()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        ARTIFACTS_ROOT / "evals" / (args.run_name or f"fullrun_eval_{int(time.time())}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    enable_thinking = not args.no_thinking if args.no_thinking else args.enable_thinking
    policy = _RolloutPolicy(
        adapter_dir=args.adapter_dir,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        enable_thinking=enable_thinking,
        parse_retries=args.parse_retries,
        strict_json_required=True,
        planner_hint_adapter_dir=args.planner_hint_adapter_dir,
        planner_hint_refresh=args.planner_hint_refresh,
        planner_hint_max_new_tokens=args.planner_hint_max_new_tokens,
    )

    rng = random.Random(args.seed)
    episodes: list[FullrunEpisode] = []
    episode_traces: list[dict[str, Any]] = []
    for ep_idx in range(args.episodes):
        port = args.port_base + ep_idx
        ep_seed = f"{args.seed}-{ep_idx}-{rng.randint(0, 10**9)}"
        print(f"[fullrun-eval] ep{ep_idx} seed={ep_seed} port={port}")
        ep = run_one_fullrun(
            policy,
            port=port,
            seed=ep_seed,
            character_id=args.character_id,
            max_steps=args.max_steps,
        )
        episodes.append(ep)
        episode_traces.append({
            "seed": ep.seed,
            "outcome": ep.outcome,
            "max_floor": ep.max_floor,
            "final_act": ep.final_act,
            "final_floor": ep.final_floor,
            "death_floor": ep.death_floor,
            "death_state_type": ep.death_state_type,
            "steps": ep.steps,
            "combat_steps": ep.combat_steps,
            "non_combat_steps": ep.non_combat_steps,
            "invalid_output_steps": ep.invalid_output_steps,
            "duration_s": ep.duration_s,
            "state_type_counts": dict(ep.state_type_counts),
            "error": ep.error,
        })
        print(
            f"  -> outcome={ep.outcome} floor={ep.max_floor}/17 "
            f"steps={ep.steps} combat={ep.combat_steps} invalid={ep.invalid_output_steps} dur={ep.duration_s}s"
        )

    metrics = aggregate_metrics(episodes)
    metrics["adapter_dir"] = args.adapter_dir
    metrics["planner_hint_adapter_dir"] = args.planner_hint_adapter_dir
    metrics["character_id"] = args.character_id
    metrics["max_steps"] = args.max_steps
    metrics["seed"] = args.seed
    metrics["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    (out_dir / "fullrun_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "fullrun_episode_trace.jsonl").open("w", encoding="utf-8") as f:
        for tr in episode_traces:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")

    print()
    print(f"[fullrun-eval] done. {metrics['episodes']} episodes")
    print(f"  act1_clear_rate: {metrics['act1_clear_rate']*100:.1f}%  ({metrics['act1_cleared']}/{metrics['episodes']})")
    print(f"  floor_reached avg/max: {metrics['floor_reached']['avg']}/{metrics['floor_reached']['max']}")
    print(f"  outcome distribution: {metrics['outcome_counts']}")
    print(f"  death floor distribution: {metrics['death_floor_counts']}")
    print(f"  metrics -> {out_dir / 'fullrun_metrics.json'}")


if __name__ == "__main__":
    main()
