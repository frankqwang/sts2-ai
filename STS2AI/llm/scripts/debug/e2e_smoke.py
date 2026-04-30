"""端到端冒烟：
1. 用 toy_dataset 里的一个样本当输入
2. 过一遍 LlmExternalPolicyAdapter
3. 打印解析出的动作，校验是否 == 预期 chosen_action

不起游戏进程，纯 Python 链路验证。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm.data_pipeline.toy_dataset import _state_a, _state_b, _state_c, _state_d, _state_e
from llm.inference.llm_policy import LlmExternalPolicyAdapter
from llm.paths import SFT_ROOT, setup_runtime


def main() -> None:
    setup_runtime()
    adapter_dir = os.environ.get("STS2_LLM_ADAPTER_DIR") or str(SFT_ROOT / "toy_mvp" / "adapter")
    print(f"[e2e] adapter_dir = {adapter_dir}")

    policy = LlmExternalPolicyAdapter(adapter_dir=adapter_dir)

    cases = [
        ("state_a (Cultist expect BASH idx=2)", _state_a),
        ("state_b (low HP expect Defend idx=1/2/4)", _state_b),
        ("state_c (small slime expect Strike idx=0)", _state_c),
        ("state_d (no energy expect end_turn idx=0)", _state_d),
        ("state_e (sentries expect BASH_S0 idx=0)", _state_e),
    ]

    ok = 0
    for label, case_fn in cases:
        state, legal, expected, reason = case_fn()
        action = policy.select_action(state, legal, None)
        chosen_idx = policy.last_decode.action_index if policy.last_decode else -1
        hit = chosen_idx == expected
        ok += int(hit)
        print(f"\n[e2e] {label}")
        print(f"    expected_index = {expected}  ({reason})")
        print(f"    got_index      = {chosen_idx}  {'HIT' if hit else 'miss'}")
        print(f"    action         = {json.dumps(action, ensure_ascii=False)}")

    print(f"\n[e2e] top-1 hit: {ok}/{len(cases)}")


if __name__ == "__main__":
    main()
