"""把 SFT 出来的 LoRA adapter 包成 `spectate.ExternalPolicy` 兼容对象。

对齐接口见 `STS2AI/bridge/game_bridge/spectate/policy.py` 里的 PolicyAdapter：

    def select_action(state, legal_actions, context) -> action_dict | None

环境变量：
- `STS2_LLM_BASE_MODEL`       基础模型 ID，默认 Qwen/Qwen3-4B-Instruct-2507
- `STS2_LLM_ADAPTER_DIR`      SFT 出来的 LoRA 目录；不设就只跑 base
- `STS2_LLM_MAX_NEW_TOKENS`   默认 64
- `STS2_LLM_TEMPERATURE`      默认 0.0（贪心）
- `STS2_LLM_FALLBACK_INDEX`   解析失败时回退的 action index，默认 0
- `STS2_LLM_LOAD_IN_4BIT`     默认 0（走 fp16）；设 1 切 bnb-4bit
- `STS2_LLM_TRACE_PATH`       每步 append 一条 JSONL 到这里，便于实时观察/回放
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.data_pipeline.action_decoder import DecodedAction, decode_action
from llm.data_pipeline.state_renderer import render_state_text
from llm.paths import BASE_MODEL_ID
from llm.prompts import load_system_prompt


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class LlmExternalPolicyAdapter:
    """和 `zero_external_policy.ZeroExternalPolicyAdapter` 同构。

    加载一次模型，`select_action` 里做：
        render state -> chat_template -> generate -> decode -> legal action
    """

    def __init__(
        self,
        *,
        base_model_id: str | None = None,
        adapter_dir: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        fallback_index: int | None = None,
        load_in_4bit: bool | None = None,
        max_seq_length: int = 2048,
    ) -> None:
        self._base_model_id = base_model_id or os.environ.get("STS2_LLM_BASE_MODEL") or BASE_MODEL_ID
        self._adapter_dir = adapter_dir or os.environ.get("STS2_LLM_ADAPTER_DIR") or None
        # JSON 只需 ~30 token，给点余量 64 就够，避免 CoT 失控消耗时间
        self._max_new_tokens = max_new_tokens if max_new_tokens is not None else _env_int("STS2_LLM_MAX_NEW_TOKENS", 64)
        # 默认走 fp16（RTX 5070 Ti 16GB 装 Qwen3-4B fp16 够），速度比 4bit 快约 2x
        # 想省显存可显式设 STS2_LLM_LOAD_IN_4BIT=1
        if load_in_4bit is None:
            load_in_4bit = _env_int("STS2_LLM_LOAD_IN_4BIT", 0) == 1
        self._temperature = temperature if temperature is not None else _env_float("STS2_LLM_TEMPERATURE", 0.0)
        self._fallback_index = fallback_index if fallback_index is not None else _env_int("STS2_LLM_FALLBACK_INDEX", 0)

        # 延迟 import，避免非 LLM 环境 import 本模块就挂
        from unsloth import FastLanguageModel

        print(f"[llm-policy] loading base={self._base_model_id} 4bit={load_in_4bit}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._base_model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        if self._adapter_dir:
            print(f"[llm-policy] attaching LoRA adapter: {self._adapter_dir}")
            from peft import PeftModel  # type: ignore

            model = PeftModel.from_pretrained(model, self._adapter_dir)

        FastLanguageModel.for_inference(model)

        self._model = model
        self._tokenizer = tokenizer
        self._system_prompt = load_system_prompt()
        self._last_decode: DecodedAction | None = None
        self._step_counter = 0
        self._trace_path: Path | None = None
        trace_env = os.environ.get("STS2_LLM_TRACE_PATH", "").strip()
        if trace_env:
            self._trace_path = Path(trace_env)
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            # 清空旧 trace
            self._trace_path.write_text("", encoding="utf-8")

    def reset_episode(self) -> None:
        self._last_decode = None
        self._step_counter = 0

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self._trace_path is None:
            return
        try:
            with self._trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # trace 不是关键路径，写不动别影响决策
            print(f"[llm-policy] trace write failed: {exc}")

    def _end_turn_fallback_index(self, legal_actions: list[dict[str, Any]]) -> int:
        for idx, action in enumerate(legal_actions):
            if str(action.get("action_type", "")).lower() == "end_turn":
                return idx
        return max(0, min(self._fallback_index, len(legal_actions) - 1))

    def _render_prompt(self, state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> str:
        user_msg = render_state_text(state, legal_actions)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, prompt_text: str) -> str:
        import torch  # type: ignore

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._temperature > 0,
                temperature=self._temperature if self._temperature > 0 else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def select_action(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        _context: Any = None,
    ) -> dict[str, Any] | None:
        enabled = [
            action for action in (legal_actions or [])
            if isinstance(action, dict) and action.get("is_enabled") is not False
        ]
        if not enabled:
            self.reset_episode()
            return None

        prompt_text = self._render_prompt(state, enabled)
        gen_t0 = time.monotonic()
        raw_text = self._generate(prompt_text)
        gen_ms = (time.monotonic() - gen_t0) * 1000.0

        decoded = decode_action(
            raw_text,
            enabled,
            fallback_index=self._end_turn_fallback_index(enabled),
        )
        self._last_decode = decoded
        chosen_action = dict(enabled[decoded.action_index])

        step_idx = self._step_counter
        self._step_counter += 1
        print(
            f"[llm-policy] step={step_idx} gen_ms={gen_ms:.0f} "
            f"chosen={decoded.action_index} fallback={decoded.used_fallback} "
            f"reason={decoded.reason!r}"
        )
        # trace 落盘（如果 env 有要求）
        # 只保存 user 消息（prompt_text 末尾那段），不保存整个 system prompt，文件会小很多
        user_msg = prompt_text.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0].strip()
        self._write_trace({
            "step": step_idx,
            "gen_ms": round(gen_ms, 1),
            "user_message": user_msg,
            "raw_generation": raw_text,
            "decoded": {
                "action_index": decoded.action_index,
                "reason": decoded.reason,
                "used_fallback": decoded.used_fallback,
                "fallback_reason": decoded.fallback_reason,
            },
            "chosen_action": {
                k: chosen_action.get(k)
                for k in ("action", "type", "card_id", "card_index", "target_id")
                if k in chosen_action
            },
            "enabled_count": len(enabled),
        })
        return chosen_action

    @property
    def last_decode(self) -> DecodedAction | None:
        return self._last_decode


_SINGLETON: LlmExternalPolicyAdapter | None = None


def _instance() -> LlmExternalPolicyAdapter:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LlmExternalPolicyAdapter()
    return _SINGLETON


def select_action(
    state: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    context: Any = None,
) -> dict[str, Any] | None:
    """SpectatorController 兼容 handler 入口。

    用法（spectate cli）：
        --external-policy llm.inference.llm_policy:select_action
    单例延迟加载，第一次调用时会花 ~2 分钟加载 Qwen 4bit + LoRA。
    """
    return _instance().select_action(state, legal_actions, context)


__all__ = ["LlmExternalPolicyAdapter", "select_action"]
