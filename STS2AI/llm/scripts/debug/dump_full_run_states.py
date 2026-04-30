"""用 full_run session + 总选 index 0 的傻瓜策略走一遍，dump 每种 state_type
的原始 state 结构。用来给启发式老师设计做依据。

运行（全局 Python，不需要 unsloth）：

    python STS2AI/llm/scripts/debug/dump_full_run_states.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_STS2AI_ROOT = Path(__file__).resolve().parents[3]
for p in (_STS2AI_ROOT, _STS2AI_ROOT / "bridge"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from game_bridge.session import create_game_session


def main() -> None:
    out_dir = _STS2AI_ROOT / "Artifacts" / "llm" / "diagnostics" / "state_types"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "samples_by_type.jsonl"
    summary_path = out_dir / "summary.json"

    seen_types: dict[str, list[dict]] = defaultdict(list)
    max_samples_per_type = 3
    step_log = []

    session = create_game_session(
        mode="full_run", transport="pipe_proto", backend="sim", port=15650, auto_launch=True,
    )
    try:
        state = session.reset(character_id="IRONCLAD", seed="dumper-1")
        for step in range(80):
            st = str(state.get("state_type") or "?")
            legal = [
                a for a in (state.get("legal_actions") or [])
                if isinstance(a, dict) and a.get("is_enabled") is not False
            ]
            if len(seen_types[st]) < max_samples_per_type:
                seen_types[st].append({
                    "step": step,
                    "state_type": st,
                    "legal_actions": legal[:10],
                    "battle_hand_len": len((state.get("battle") or {}).get("hand") or []),
                    "enemies_len": len(state.get("enemies") or (state.get("battle") or {}).get("enemies") or []),
                    # 非战斗特有上层字段
                    "map_snippet": str(state.get("map"))[:200],
                    "event_snippet": str(state.get("event"))[:200],
                    "shop_snippet": str(state.get("shop"))[:200],
                    "reward_snippet": str(state.get("reward") or state.get("card_reward") or state.get("rewards"))[:200],
                    "campfire_snippet": str(state.get("campfire") or state.get("rest"))[:200],
                    "card_selection_snippet": str(state.get("card_selection"))[:200],
                })
            step_log.append({"step": step, "state_type": st, "n_legal": len(legal)})
            if not legal:
                break
            # 选 index 0 推进
            try:
                result = session.act(legal[0])
                state = result if isinstance(result, dict) else session.get_state()
            except Exception as exc:
                step_log.append({"step": step, "error": str(exc)[:200]})
                # proceed 失败，刷新继续
                try:
                    state = session.get_state()
                except Exception:
                    break
    finally:
        session.close()

    with sample_path.open("w", encoding="utf-8") as f:
        for st, rows in seen_types.items():
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total_steps": len(step_log),
        "state_types_seen": {k: len(v) for k, v in seen_types.items()},
        "step_log": step_log[-40:],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dumped {len(seen_types)} state_types:", list(seen_types.keys()))
    print(f"samples -> {sample_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
