"""冒烟：用 unsloth 4bit 加载 Qwen3-4B-Instruct-2507 并跑一次 chat。

运行方式（在 unsloth studio 3.13 venv 里）：

    C:\\Users\\Administrator\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe ^
        STS2AI\\llm\\scripts\\smoke_load_qwen.py

用途：
- 确认本地 HF cache 里的 Qwen3-4B snapshot 能被 unsloth 消费
- 确认 CUDA/Blackwell 路径没坑
- 输出一个可读中文回复

通不过这个，后面的训练 / 推理脚本都不用跑。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def _add_llm_root_to_sys_path() -> Path:
    # 让脚本能 `from llm.paths import ...`
    llm_root = Path(__file__).resolve().parents[1]
    sts2ai_root = llm_root.parent
    for candidate in (sts2ai_root, llm_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)
    return llm_root


_add_llm_root_to_sys_path()

from llm.paths import BASE_MODEL_ID, setup_runtime  # noqa: E402
from llm.prompts import load_system_prompt  # noqa: E402


def main() -> None:
    setup_runtime()

    print("[smoke] importing unsloth ...")
    t0 = time.monotonic()
    from unsloth import FastLanguageModel

    print(f"[smoke] unsloth imported in {time.monotonic() - t0:.1f}s")

    print(f"[smoke] loading {BASE_MODEL_ID} in 4bit ...")
    t0 = time.monotonic()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=4096,
        dtype=None,  # 让 unsloth 自选 bf16 / fp16
        load_in_4bit=True,
    )
    print(f"[smoke] model loaded in {time.monotonic() - t0:.1f}s")

    FastLanguageModel.for_inference(model)

    system_prompt = load_system_prompt()
    user_msg = (
        "run: char=IRONCLAD floor=1 encounter=CULTIST gold=99\n"
        "player: hp=80/80 block=0 energy=3/3 buffs=-\n"
        "piles: draw=10 discard=0 exhaust=0\n"
        "enemies:\n"
        "  [0] CULTIST hp=50/50 block=0 intent=ritual buffs=-\n"
        "hand:\n"
        "  [0] STRIKE_RED cost=1 dmg=6 tags=-\n"
        "  [1] DEFEND_RED cost=1 blk=5 tags=-\n"
        "  [2] BASH cost=2 dmg=8 tags=-\n"
        "  [3] STRIKE_RED cost=1 dmg=6 tags=-\n"
        "  [4] STRIKE_RED cost=1 dmg=6 tags=-\n"
        "legal_actions:\n"
        "  [0] play_card card=STRIKE_RED hand_idx=0 target=CULTIST_0\n"
        "  [1] play_card card=DEFEND_RED hand_idx=1\n"
        "  [2] play_card card=BASH hand_idx=2 target=CULTIST_0\n"
        "  [3] play_card card=STRIKE_RED hand_idx=3 target=CULTIST_0\n"
        "  [4] play_card card=STRIKE_RED hand_idx=4 target=CULTIST_0\n"
        "  [5] end_turn\n"
        "请输出一行 JSON，只能从 legal_actions 中选一个 action_index。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print("[smoke] generating ...")
    t0 = time.monotonic()
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print(f"[smoke] generated in {time.monotonic() - t0:.1f}s")
    print("-" * 60)
    print(generated.strip())
    print("-" * 60)


if __name__ == "__main__":
    main()
