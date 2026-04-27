"""Collect action-choice SFT rows with engine settlement-event supervision.

This is intentionally small and conservative: it uses the existing heuristic
teacher to pick actions, executes them through GameSession, then turns the
actual CombatHistory delta into a short assistant reason. The prompt still asks
for an action_index, so the dataset can be used by the current SFT trainer
without changing inference.
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

_LLM_ROOT = Path(__file__).resolve().parents[1]
_STS2AI_ROOT = _LLM_ROOT.parent
_BRIDGE_ROOT = _STS2AI_ROOT / "bridge"
for _path in (_STS2AI_ROOT, _BRIDGE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from game_bridge.session import create_game_session  # noqa: E402
from game_bridge.session.state_semantics import is_actionable_combat_state, is_combat_state  # noqa: E402
from llm.data_pipeline.encounter_pool import ACT1_WINNABLE_POOL, EncounterSpec  # noqa: E402
from llm.data_pipeline.heuristic_teacher import pick_action, score_actions  # noqa: E402
from llm.data_pipeline.state_renderer import render_state_text  # noqa: E402
from llm.metrics import summarize_dataset_dir, write_json  # noqa: E402
from llm.paths import DATASETS_ROOT, ensure_dirs  # noqa: E402
from llm.prompts import load_system_prompt  # noqa: E402


@dataclass
class EpisodeSummary:
    encounter_id: str
    seed: str
    outcome: str
    steps: int
    samples: int
    duration_s: float
    event_type_counts: dict[str, int] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-encounter", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--port-base", type=int, default=16720)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--out-subdir", type=str, default="")
    parser.add_argument("--encounter-filter", type=str, default="")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--host-path", type=str, default="")
    parser.add_argument("--reason-max-chars", type=int, default=120)
    return parser.parse_args()


def _enabled_actions(legal: list[Any]) -> list[dict[str, Any]]:
    return [
        dict(action)
        for action in (legal or [])
        if isinstance(action, dict) and action.get("is_enabled") is not False
    ]


def _event_amount(value: Any) -> str:
    try:
        as_float = float(value)
    except Exception:
        return str(value)
    if as_float.is_integer():
        return str(int(as_float))
    return f"{as_float:g}"


def _target_label(event: dict[str, Any]) -> str:
    target = event.get("target_id") or event.get("actor_id") or "target"
    combat_id = event.get("target_combat_id")
    if combat_id:
        return f"{target}#{combat_id}"
    return str(target)


def _compact_reason(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    for separator in (". ", "; ", ", "):
        pos = cut.rfind(separator)
        if pos >= max(40, max_chars // 2):
            cut = cut[: pos + (1 if separator == ". " else 0)]
            break
    return cut.rstrip(" ;,.") + "."


def summarize_settlement_events(
    events: list[dict[str, Any]],
    chosen_action: dict[str, Any],
    *,
    max_chars: int = 120,
) -> str:
    if not events:
        return "The selected action produced no immediate combat-history event."

    fragments: list[str] = []
    total_energy = sum(
        int(event.get("energy_spent") or 0)
        for event in events
        if event.get("type") == "energy_spent"
    )
    if total_energy:
        fragments.append(f"spends {total_energy} energy")

    damage_parts: list[str] = []
    for event in events:
        if event.get("type") != "damage_received":
            continue
        damage = int(event.get("unblocked_damage") or event.get("total_damage") or 0)
        if damage <= 0:
            continue
        suffix = " and kills it" if event.get("target_killed") else ""
        damage_parts.append(f"deals {damage} damage to {_target_label(event)}{suffix}")
    fragments.extend(damage_parts[:3])

    block_total = sum(int(event.get("amount_int") or 0) for event in events if event.get("type") == "block_gained")
    if block_total:
        fragments.append(f"gains {block_total} block")

    power_parts: list[str] = []
    for event in events:
        if event.get("type") != "power_received":
            continue
        power = str(event.get("power_id") or "POWER")
        amount = _event_amount(event.get("amount_value", event.get("amount_int", 0)))
        power_parts.append(f"applies {amount} {power} to {_target_label(event)}")
    fragments.extend(power_parts[:3])

    card_moves = []
    for event_type, verb in (
        ("card_drawn", "draws"),
        ("card_discarded", "discards"),
        ("card_exhausted", "exhausts"),
        ("card_generated", "generates"),
    ):
        cards = [str(event.get("card_id")) for event in events if event.get("type") == event_type and event.get("card_id")]
        if cards:
            card_moves.append(f"{verb} {', '.join(cards[:2])}")
    fragments.extend(card_moves[:3])

    if not fragments:
        event_types = sorted({str(event.get("type") or "event") for event in events})
        fragments.append("produces engine events: " + ", ".join(event_types[:5]))

    card_id = chosen_action.get("card_id")
    action = str(chosen_action.get("action") or chosen_action.get("type") or "action")
    prefix = f"{card_id}: " if card_id else f"{action}: "
    return _compact_reason(prefix + "; ".join(fragments[:4]) + ".", max_chars)


def _confidence_from_action_scores(scores: list[dict[str, Any]]) -> float:
    if len(scores) <= 1:
        return 1.0
    try:
        margin = float(scores[0]["score"]) - float(scores[1]["score"])
    except (KeyError, TypeError, ValueError):
        return 0.55
    return round(max(0.35, min(0.95, 0.55 + margin / 10.0)), 2)


def _teacher_action_scores(state: dict[str, Any], legal: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for scored in score_actions(state, legal)[:4]:
        rows.append({
            "action_index": int(scored.action_index),
            "score": round(float(scored.score), 3),
            "note": scored.reason[:80],
        })
    return rows


def _action_payload(action: dict[str, Any]) -> dict[str, Any]:
    return {
        key: action.get(key)
        for key in ("action", "type", "card_id", "card_index", "target_id", "index", "label")
        if key in action
    }


def _build_sample(
    *,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    chosen_index: int,
    chosen_action: dict[str, Any],
    settlement_events: list[dict[str, Any]],
    system_prompt: str,
    encounter_id: str,
    seed: str,
    step: int,
    reason_max_chars: int,
    action_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    user_msg = render_state_text(state, legal, encounter_id=encounter_id)
    reason = summarize_settlement_events(
        settlement_events,
        chosen_action,
        max_chars=reason_max_chars,
    )
    confidence = _confidence_from_action_scores(action_scores)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "action_index": int(chosen_index),
                        "confidence": confidence,
                        "action_scores": action_scores,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
        "meta": {
            "kind": "settlement_event_supervision",
            "encounter_id": encounter_id,
            "seed": seed,
            "step": step,
            "chosen_action": _action_payload(chosen_action),
            "settlement_events": settlement_events,
            "event_types": [str(event.get("type") or "") for event in settlement_events],
            "reason_source": "engine_combat_history_delta",
            "confidence": confidence,
            "action_scores": action_scores,
        },
    }


def collect_dataset(
    *,
    encounters: list[EncounterSpec],
    episodes_per_encounter: int,
    max_steps: int,
    max_samples: int,
    port_base: int,
    seed: int,
    host_path: str,
    reason_max_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[EpisodeSummary]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    episodes: list[EpisodeSummary] = []
    system_prompt = load_system_prompt("index")

    for enc_idx, spec in enumerate(encounters):
        for ep_idx in range(episodes_per_encounter):
            if len(samples) >= max_samples:
                break
            ep_seed = f"{seed}-{enc_idx}-{ep_idx}-{rng.randint(0, 10**9)}"
            kwargs: dict[str, Any] = {
                "mode": "combat",
                "transport": "pipe_proto",
                "backend": "sim",
                "port": port_base + enc_idx,
                "auto_launch": True,
                "connect_timeout_s": 20.0,
            }
            if host_path:
                kwargs["host_path"] = host_path
            session = create_game_session(**kwargs)
            started = time.monotonic()
            outcome = "unknown"
            steps = 0
            ep_samples = 0
            event_type_counts: dict[str, int] = {}
            try:
                state = session.reset(
                    character_id="IRONCLAD",
                    encounter_id=spec.encounter_id,
                    build=spec.build,
                    seed=ep_seed,
                )
                for step_idx in range(max_steps):
                    if len(samples) >= max_samples:
                        outcome = "max_samples"
                        break
                    if not is_combat_state(state):
                        outcome = "left_combat"
                        break
                    legal = _enabled_actions(state.get("legal_actions") or [])
                    if not legal:
                        outcome = "no_legal_actions"
                        break
                    if is_actionable_combat_state(state):
                        decision = pick_action(state, legal)
                        chosen_index = int(decision.action_index)
                        chosen = legal[chosen_index]
                        state_before = dict(state)
                        action_scores = _teacher_action_scores(state_before, legal)
                    else:
                        chosen_index = 0
                        chosen = legal[0]
                        state_before = dict(state)
                        action_scores = _teacher_action_scores(state_before, legal)

                    next_state, _reward, done, info = session.act_gym(chosen)
                    settlement_events = [
                        dict(event)
                        for event in (info.get("settlement_events") if isinstance(info, dict) else []) or []
                        if isinstance(event, dict)
                    ]
                    for event in settlement_events:
                        event_type = str(event.get("type") or "unknown")
                        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

                    if is_actionable_combat_state(state_before):
                        sample = _build_sample(
                            state=state_before,
                            legal=legal,
                            chosen_index=chosen_index,
                            chosen_action=chosen,
                            settlement_events=settlement_events,
                            system_prompt=system_prompt,
                            encounter_id=spec.encounter_id,
                            seed=ep_seed,
                            step=step_idx,
                            reason_max_chars=reason_max_chars,
                            action_scores=action_scores,
                        )
                        samples.append(sample)
                        ep_samples += 1
                        traces.append({
                            "encounter_id": spec.encounter_id,
                            "seed": ep_seed,
                            "step": step_idx,
                            "state": state_before,
                            "legal_actions": legal,
                            "chosen_action_index": chosen_index,
                            "chosen_action": _action_payload(chosen),
                            "settlement_events": settlement_events,
                            "assistant": sample["messages"][-1]["content"],
                        })

                    state = next_state
                    steps = step_idx + 1
                    if done:
                        outcome = str(state.get("run_outcome") or state.get("combat_outcome") or "terminal")
                        break
                else:
                    outcome = "max_steps"
            except Exception as exc:
                outcome = f"failed:{type(exc).__name__}"
                print(f"[settlement-rollout] {spec.encounter_id} ep={ep_idx} failed: {exc}")
            finally:
                try:
                    session.close()
                except Exception:
                    pass
            episodes.append(EpisodeSummary(
                encounter_id=spec.encounter_id,
                seed=ep_seed,
                outcome=outcome,
                steps=steps,
                samples=ep_samples,
                duration_s=round(time.monotonic() - started, 2),
                event_type_counts=event_type_counts,
            ))
            print(
                f"[settlement-rollout] {spec.encounter_id} ep={ep_idx} "
                f"outcome={outcome} steps={steps} samples={ep_samples} total={len(samples)}"
            )
        if len(samples) >= max_samples:
            break
    return samples, traces, episodes


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    out_subdir = args.out_subdir or f"settlement_supervision_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = DATASETS_ROOT / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = ACT1_WINNABLE_POOL
    if args.encounter_filter:
        needle = args.encounter_filter.lower()
        pool = [spec for spec in pool if needle in spec.encounter_id.lower()]
    if not pool:
        raise SystemExit("no encounters matched filter")

    samples, traces, episodes = collect_dataset(
        encounters=pool,
        episodes_per_encounter=args.episodes_per_encounter,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        port_base=args.port_base,
        seed=args.seed,
        host_path=args.host_path,
        reason_max_chars=args.reason_max_chars,
    )
    if not samples:
        raise SystemExit("no samples collected")

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    eval_n = max(1, int(len(samples) * args.eval_ratio)) if len(samples) > 1 else 0
    eval_rows = samples[:eval_n]
    train_rows = samples[eval_n:]

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    _write_jsonl(out_dir / "step_trace.jsonl", traces)

    event_type_counts: dict[str, int] = {}
    for episode in episodes:
        for key, value in episode.event_type_counts.items():
            event_type_counts[key] = event_type_counts.get(key, 0) + value

    meta = {
        "run_id": uuid.uuid4().hex,
        "kind": "settlement_event_supervision",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pool": [{"encounter_id": spec.encounter_id, "tag": spec.tag} for spec in pool],
        "episodes_per_encounter": args.episodes_per_encounter,
        "total_episodes": len(episodes),
        "total_samples": len(samples),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "max_steps": args.max_steps,
        "max_samples": args.max_samples,
        "host_path": args.host_path,
        "reason_max_chars": args.reason_max_chars,
        "event_type_counts": event_type_counts,
        "episodes": [episode.__dict__ for episode in episodes],
        "train": str(out_dir / "train.jsonl"),
        "eval": str(out_dir / "eval.jsonl"),
        "step_trace": str(out_dir / "step_trace.jsonl"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(out_dir / "metrics.json", {"kind": "settlement_event_supervision", **summarize_dataset_dir(out_dir)})
    print(f"[settlement-rollout] output -> {out_dir}")


if __name__ == "__main__":
    main()
