"""把 SFT 出来的 LoRA adapter 包成 `spectate.ExternalPolicy` 兼容对象。

对齐接口见 `STS2AI/bridge/game_bridge/spectate/policy.py` 里的 PolicyAdapter：

    def select_action(state, legal_actions, context) -> action_dict | None

环境变量：
- `STS2_LLM_BASE_MODEL`       基础模型 ID，默认 Qwen/Qwen3-4B-Instruct-2507
- `STS2_LLM_ADAPTER_DIR`      SFT 出来的 LoRA 目录；不设就只跑 base
- `STS2_LLM_COMBAT_ADAPTER_DIR`     战斗 LoRA；设置后 combat state 使用它
- `STS2_LLM_NON_COMBAT_ADAPTER_DIR` 非战斗 LoRA；设置后 map/event/shop/reward 等使用它
- `STS2_LLM_MAX_NEW_TOKENS`   默认 64
- `STS2_LLM_TEMPERATURE`      默认 0.0（贪心）
- `STS2_LLM_PARSE_RETRIES`    解析失败后的重试次数，默认 1
- `STS2_LLM_LOAD_IN_4BIT`     默认 0（走 fp16）；设 1 切 bnb-4bit
- `STS2_LLM_RESPONSE_PREFIX`  默认 `{`，用 JSON 前缀压住 Qwen 工具调用先验
- `STS2_LLM_SIMPLE_GATE`      默认 1；唯一合法动作等确定场景跳过 LLM
- `STS2_LLM_SURVIVAL_GATE`    默认 1；低血量且可见防御/减伤能降低当前伤害时跳过 LLM
- `STS2_LLM_ALLOW_POTIONS`   默认 0；战斗 use_potion 暂不交给 LLM，避免无策略烧药
- `STS2_LLM_TRACE_PATH`       每步 append 一条 JSONL 到这里，便于实时观察/回放
- `STS2_LLM_ACTION_MODE`      默认 index；设 structured 让模型输出 action/hand_index/target_id
- `STS2_LLM_STRATEGY_CONTEXT` 默认 1；注入 run/combat/turn 级策略上下文
- `STS2_LLM_PLANNER_HINT_ADAPTER_DIR` planner-hint LoRA；输出战斗级提示，不执行动作
- `STS2_LLM_PLANNER_HINT`     默认随 adapter 启用；设 0 禁用
- `STS2_LLM_PLANNER_HINT_REFRESH` 默认 turn（每回合刷新）；可设 combat 退回整场战斗缓存（不推荐）
- `STS2_LLM_GUIDE_RAG`        默认 1；给 planner-hint prompt 注入本地攻略 evidence
- `STS2_LLM_GUIDE_LIMIT`      默认 4；每次最多召回攻略 evidence 条数
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.data_pipeline.action_decoder import (
    DecodedAction,
    action_score_margin,
    decode_action,
    decode_structured_action,
)
from llm.data_pipeline.action_quality import assess_action_quality_report
from llm.data_pipeline.heuristic_teacher import pick_action
from llm.data_pipeline.guide_knowledge import render_retrieved_knowledge_for_state
from llm.data_pipeline.state_renderer import (
    can_render_structured_actions,
    render_state_text,
    render_structured_action_state_text,
)
from llm.data_pipeline.planner_hint import (
    DEFAULT_PLANNER_HINT_REFRESH,
    PLANNER_HINT_REFRESH_CHOICES,
    format_planner_hint,
    parse_planner_hint_json,
    planner_hint_cache_key,
    render_planner_hint_user_message,
)
from llm.data_pipeline.strategy_context import StrategyMemory
from llm.inference.hybrid_gate import choose_simple_action, choose_survival_action
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


def _normalize_action_mode(raw: str | None) -> str:
    value = str(raw or "").strip().lower().replace("-", "_")
    return "structured" if value in {"structured", "compose", "command"} else "index"


def _policy_action_type(action: dict[str, Any]) -> str:
    return str(action.get("action") or action.get("action_type") or action.get("type") or "").strip().lower()


def _is_dead_overlay_state(state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> bool:
    if not legal_actions or any(_policy_action_type(action) != "overlay_press" for action in legal_actions):
        return False
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    battle_player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
    try:
        hp = float(player.get("hp") or player.get("current_hp") or battle_player.get("hp") or 0)
        max_hp = float(player.get("max_hp") or battle_player.get("max_hp") or 0)
    except (TypeError, ValueError):
        return False
    return hp <= 0 and max_hp <= 0


_THINK_END_RE = re.compile(r"</think>\s*", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    if not text:
        return text
    if _THINK_END_RE.search(text):
        return _THINK_END_RE.split(text)[-1].strip()
    return _THINK_BLOCK_RE.sub("", text).strip()


def _strict_json_status(raw_text: str) -> str:
    stripped = _strip_thinking(raw_text).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return "json_parse_failed"
    if not isinstance(payload, dict):
        return "json_not_object"
    if "action_index" not in payload:
        return "action_index_missing"
    if isinstance(payload.get("action_index"), bool) or not isinstance(payload.get("action_index"), int):
        return "action_index_not_int"
    return "ok"


_REASON_CONSISTENCY_FLAGS = {
    "reason_math_contradiction",
    "reason_claims_lethal_but_action_not_lethal",
}
_EXPLANATION_CONSISTENCY_FLAGS = _REASON_CONSISTENCY_FLAGS | {
    "action_score_lethal_math_contradiction",
}
_SAFETY_RETRY_FLAGS = {
    "dangerous_end_turn",
    "dangerous_self_damage",
    "low_hp_self_damage",
}
_URGENT_POTION_IDS = {
    "FORTIFIER",
    "BLOCK_POTION",
    "HEART_OF_IRON",
    "HEALTH_POTION",
    "REGEN_POTION",
}


def _has_any_flag(fallback_reason: str, flags: set[str]) -> bool:
    return any(flag in fallback_reason for flag in flags)

_COMBAT_STATE_TYPES = {
    "battle",
    "boss",
    "combat",
    "elite",
    "hand_select",
    "monster",
}

_COMBAT_DECISION_TYPES = {
    "combat",
    "combat_action",
    "play_card",
}

_NON_COMBAT_DECISION_TYPES = {
    "card_reward",
    "card_select",
    "choose_map_node",
    "combat_rewards",
    "event",
    "event_choice",
    "map",
    "map_choice",
    "relic_select",
    "rest_site",
    "rest_site_choice",
    "shop",
    "shop_choice",
    "treasure",
}


class LlmExternalPolicyAdapter:
    """和 `zero_external_policy.ZeroExternalPolicyAdapter` 同构。

    简单强制动作直接返回；复杂决策首次按需加载模型，然后做：
        render state -> chat_template -> generate -> decode -> legal action
    """

    def __init__(
        self,
        *,
        base_model_id: str | None = None,
        adapter_dir: str | None = None,
        combat_adapter_dir: str | None = None,
        non_combat_adapter_dir: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        load_in_4bit: bool | None = None,
        max_seq_length: int = 2048,
        parse_retries: int | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        self._base_model_id = base_model_id or os.environ.get("STS2_LLM_BASE_MODEL") or BASE_MODEL_ID
        single_adapter_dir = adapter_dir or os.environ.get("STS2_LLM_ADAPTER_DIR") or None
        combat_adapter_dir = combat_adapter_dir or os.environ.get("STS2_LLM_COMBAT_ADAPTER_DIR") or None
        non_combat_adapter_dir = non_combat_adapter_dir or os.environ.get("STS2_LLM_NON_COMBAT_ADAPTER_DIR") or None
        planner_hint_adapter_dir = os.environ.get("STS2_LLM_PLANNER_HINT_ADAPTER_DIR") or None
        self._adapter_dirs = self._build_adapter_dirs(
            single_adapter_dir=single_adapter_dir,
            combat_adapter_dir=combat_adapter_dir,
            non_combat_adapter_dir=non_combat_adapter_dir,
            planner_hint_adapter_dir=planner_hint_adapter_dir,
        )
        self._adapter_key_to_name: dict[str, str] = {}
        self._active_adapter_name: str | None = None
        # 开启 thinking 后模型需要额外 100~300 token 写推理过程
        default_max_new_tokens = 320 if (enable_thinking or _env_int("STS2_LLM_ENABLE_THINKING", 0) == 1) else 160
        self._max_new_tokens = max_new_tokens if max_new_tokens is not None else _env_int("STS2_LLM_MAX_NEW_TOKENS", default_max_new_tokens)
        self._enable_thinking = enable_thinking if enable_thinking is not None else (_env_int("STS2_LLM_ENABLE_THINKING", 0) == 1)
        # 默认走 fp16（RTX 5070 Ti 16GB 装 Qwen3-4B fp16 够），速度比 4bit 快约 2x
        # 想省显存可显式设 STS2_LLM_LOAD_IN_4BIT=1
        if load_in_4bit is None:
            load_in_4bit = _env_int("STS2_LLM_LOAD_IN_4BIT", 0) == 1
        self._load_in_4bit = load_in_4bit
        self._max_seq_length = max_seq_length
        self._temperature = temperature if temperature is not None else _env_float("STS2_LLM_TEMPERATURE", 0.0)
        self._parse_retries = max(0, parse_retries if parse_retries is not None else _env_int("STS2_LLM_PARSE_RETRIES", 1))
        self._response_prefix = os.environ.get("STS2_LLM_RESPONSE_PREFIX", '{"')
        self._simple_gate_enabled = _env_int("STS2_LLM_SIMPLE_GATE", 1) == 1
        self._survival_gate_enabled = _env_int("STS2_LLM_SURVIVAL_GATE", 1) == 1
        self._recover_explanation_invalid = _env_int("STS2_LLM_RECOVER_EXPLANATION_INVALID", 1) == 1
        self._action_mode = _normalize_action_mode(os.environ.get("STS2_LLM_ACTION_MODE", "index"))
        self._strategy_context_enabled = _env_int("STS2_LLM_STRATEGY_CONTEXT", 1) == 1
        self._strategy_memory = StrategyMemory()
        planner_raw = os.environ.get("STS2_LLM_PLANNER_HINT", "").strip().lower()
        self._planner_hint_enabled = (
            planner_raw not in {"0", "false", "off", "no"}
            and "planner_hint" in self._adapter_dirs
            and "combat" in self._adapter_dirs
        )
        self._planner_hint_refresh = os.environ.get(
            "STS2_LLM_PLANNER_HINT_REFRESH", DEFAULT_PLANNER_HINT_REFRESH
        ).strip().lower()
        if planner_raw in {"turn", "per_turn"}:
            self._planner_hint_refresh = "turn"
        if self._planner_hint_refresh not in PLANNER_HINT_REFRESH_CHOICES:
            self._planner_hint_refresh = DEFAULT_PLANNER_HINT_REFRESH
        self._planner_hint_max_new_tokens = _env_int("STS2_LLM_PLANNER_HINT_MAX_NEW_TOKENS", 240)
        self._planner_hint_temperature = _env_float("STS2_LLM_PLANNER_HINT_TEMPERATURE", 0.0)
        self._planner_hint_required = _env_int("STS2_LLM_PLANNER_HINT_REQUIRED", 0) == 1
        self._guide_required = _env_int("STS2_LLM_GUIDE_REQUIRED", 0) == 1
        self._planner_hint_cache_key = ""
        self._planner_hint_cache_text = ""
        self._planner_hint_cache_knowledge: list[dict[str, Any]] = []
        self._last_planner_hint = ""
        self._last_planner_hint_status = "disabled"
        self._last_planner_hint_raw = ""
        self._last_retrieved_knowledge: list[dict[str, Any]] = []
        self._stats: dict[str, int] = {
            "calls": 0,
            "llm_calls": 0,
            "heuristic_calls": 0,
            "model_loads": 0,
            "generated_outputs": 0,
            "strict_json_ok": 0,
            "strict_json_failures": 0,
            "strict_json_rejections": 0,
            "invalid_attempts": 0,
            "first_attempt_invalid": 0,
            "retry_recovered": 0,
            "retry_attempts": 0,
            "invalid_outputs": 0,
            "parse_failures": 0,
            "action_index_not_int": 0,
            "out_of_range": 0,
            "mapping_failures": 0,
            "reason_consistency_failures": 0,
            "safety_rejections": 0,
            "prompt_too_long": 0,
            "structured_calls": 0,
            "structured_fallback_to_index": 0,
            "strategy_context_calls": 0,
            "planner_hint_calls": 0,
            "planner_hint_cache_hits": 0,
            "planner_hint_failures": 0,
            "potion_actions_suppressed": 0,
            "adapter_switches": 0,
            "adapter_switch_failures": 0,
            "explanation_recoveries": 0,
            "survival_gate_calls": 0,
            "terminal_overlay_stops": 0,
        }

        self._system_prompts = {
            "index": load_system_prompt("index"),
            "non_combat": load_system_prompt("non_combat"),
            "planner_hint": load_system_prompt("planner_hint"),
            "structured": load_system_prompt("structured"),
        }
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._last_decode: DecodedAction | None = None
        self._step_counter = 0
        self._trace_path: Path | None = None
        trace_env = os.environ.get("STS2_LLM_TRACE_PATH", "").strip()
        if trace_env:
            self._trace_path = Path(trace_env)
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            # 清空旧 trace
            self._trace_path.write_text("", encoding="utf-8")

    @staticmethod
    def _normalize_adapter_path(path: str | None) -> str | None:
        if not path:
            return None
        return str(Path(path).expanduser().resolve())

    @classmethod
    def _build_adapter_dirs(
        cls,
        *,
        single_adapter_dir: str | None,
        combat_adapter_dir: str | None,
        non_combat_adapter_dir: str | None,
        planner_hint_adapter_dir: str | None,
    ) -> dict[str, str]:
        single = cls._normalize_adapter_path(single_adapter_dir)
        combat = cls._normalize_adapter_path(combat_adapter_dir)
        non_combat = cls._normalize_adapter_path(non_combat_adapter_dir)
        planner_hint = cls._normalize_adapter_path(planner_hint_adapter_dir)
        if combat or non_combat or planner_hint:
            out: dict[str, str] = {}
            has_combat = bool(combat or single)
            if has_combat:
                out["combat"] = combat or single or ""
            if non_combat:
                out["non_combat"] = non_combat
            if planner_hint and has_combat:
                out["planner_hint"] = planner_hint
            return {key: value for key, value in out.items() if value}
        return {"default": single} if single else {}

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        # 延迟 import / load：forced decisions can skip Qwen entirely.
        from unsloth import FastLanguageModel

        print(f"[llm-policy] loading base={self._base_model_id} 4bit={self._load_in_4bit}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._base_model_id,
            max_seq_length=self._max_seq_length,
            dtype=None,
            load_in_4bit=self._load_in_4bit,
        )
        if self._adapter_dirs:
            from peft import PeftModel  # type: ignore

            loaded_by_path: dict[str, str] = {}
            for adapter_key, adapter_path in self._adapter_dirs.items():
                existing_name = loaded_by_path.get(adapter_path)
                if existing_name:
                    self._adapter_key_to_name[adapter_key] = existing_name
                    continue
                adapter_name = adapter_key
                if not loaded_by_path:
                    print(f"[llm-policy] attaching LoRA adapter[{adapter_key}]: {adapter_path}")
                    model = PeftModel.from_pretrained(
                        model,
                        adapter_path,
                        adapter_name=adapter_name,
                    )
                else:
                    print(f"[llm-policy] loading LoRA adapter[{adapter_key}]: {adapter_path}")
                    model.load_adapter(adapter_path, adapter_name=adapter_name)
                loaded_by_path[adapter_path] = adapter_name
                self._adapter_key_to_name[adapter_key] = adapter_name

        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer
        self._stats["model_loads"] += 1

    @staticmethod
    def _context_value(context: Any, key: str) -> Any:
        if isinstance(context, dict):
            return context.get(key)
        return getattr(context, key, None)

    def _adapter_key_for_state(self, state: dict[str, Any], context: Any = None) -> str:
        explicit = (
            self._context_value(context, "decision_type")
            or self._context_value(context, "state_type")
            or state.get("decision_type")
        )
        decision_type = str(explicit or "").strip().lower().replace("-", "_")
        if decision_type in _COMBAT_DECISION_TYPES:
            return "combat"
        if decision_type in _NON_COMBAT_DECISION_TYPES:
            return "non_combat"
        state_type = str(state.get("state_type") or "").strip().lower().replace("-", "_")
        if state_type in _COMBAT_STATE_TYPES:
            return "combat"
        return "non_combat"

    def _adapter_name_for_key(self, adapter_key: str) -> str | None:
        if not self._adapter_dirs:
            return None
        if "default" in self._adapter_dirs:
            return self._adapter_key_to_name.get("default") or "default"
        if adapter_key in self._adapter_dirs:
            return self._adapter_key_to_name.get(adapter_key) or adapter_key
        if adapter_key == "non_combat" and "combat" in self._adapter_dirs:
            return self._adapter_key_to_name.get("combat") or "combat"
        if "non_combat" in self._adapter_dirs:
            return self._adapter_key_to_name.get("non_combat") or "non_combat"
        if "combat" in self._adapter_dirs:
            return self._adapter_key_to_name.get("combat") or "combat"
        return None

    def _activate_adapter(self, adapter_name: str | None) -> float:
        if not adapter_name or adapter_name == self._active_adapter_name:
            return 0.0
        self._ensure_model_loaded()
        assert self._model is not None
        if not hasattr(self._model, "set_adapter"):
            return 0.0
        t0 = time.monotonic()
        try:
            self._model.set_adapter(adapter_name)
        except Exception:
            self._stats["adapter_switch_failures"] += 1
            raise
        switch_ms = (time.monotonic() - t0) * 1000.0
        self._active_adapter_name = adapter_name
        self._stats["adapter_switches"] += 1
        return switch_ms

    def reset_episode(self) -> None:
        self._last_decode = None
        self._step_counter = 0
        self._strategy_memory.reset_combat()
        self._planner_hint_cache_key = ""
        self._planner_hint_cache_text = ""
        self._planner_hint_cache_knowledge = []
        self._last_planner_hint = ""
        self._last_planner_hint_status = "reset"
        self._last_planner_hint_raw = ""
        self._last_retrieved_knowledge = []

    def _write_trace(self, payload: dict[str, Any]) -> None:
        if self._trace_path is None:
            return
        try:
            with self._trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # trace 不是关键路径，写不动别影响决策
            print(f"[llm-policy] trace write failed: {exc}")

    def _resolve_action_mode(self, legal_actions: list[dict[str, Any]]) -> str:
        if self._action_mode != "structured":
            return "index"
        if can_render_structured_actions(legal_actions):
            return "structured"
        self._stats["structured_fallback_to_index"] += 1
        return "index"

    def _render_planner_hint_prompt(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> str:
        self._ensure_model_loaded()
        assert self._tokenizer is not None
        knowledge_text, knowledge_entries = render_retrieved_knowledge_for_state(state)
        self._last_retrieved_knowledge = knowledge_entries
        user_msg = render_planner_hint_user_message(
            state,
            legal_actions,
            memory=self._strategy_memory.planner_memory_text(state),
            previous_hint=self._planner_hint_cache_text,
            knowledge=knowledge_text,
            require_knowledge=self._guide_required,
        )
        messages = [
            {"role": "system", "content": self._system_prompts["planner_hint"]},
            {"role": "user", "content": user_msg},
        ]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking:
            kwargs["enable_thinking"] = True
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _planner_hint_for_state(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        adapter_key: str,
        allow_generate: bool,
    ) -> str:
        self._last_planner_hint = ""
        self._last_planner_hint_raw = ""
        self._last_retrieved_knowledge = []
        if (
            not allow_generate
            or not self._planner_hint_enabled
            or adapter_key != "combat"
        ):
            self._last_planner_hint_status = "disabled"
            return ""

        cache_key = planner_hint_cache_key(state, refresh=self._planner_hint_refresh)
        if cache_key and cache_key == self._planner_hint_cache_key and self._planner_hint_cache_text:
            self._stats["planner_hint_cache_hits"] += 1
            self._last_planner_hint = self._planner_hint_cache_text
            self._last_retrieved_knowledge = list(self._planner_hint_cache_knowledge)
            self._last_planner_hint_status = "cache_hit"
            return self._planner_hint_cache_text

        planner_adapter_name = self._adapter_key_to_name.get("planner_hint") or "planner_hint"
        try:
            prompt_text = self._render_planner_hint_prompt(state, legal_actions)
            self._activate_adapter(planner_adapter_name)
            raw_text = self._generate(
                prompt_text,
                max_new_tokens=self._planner_hint_max_new_tokens,
                temperature=self._planner_hint_temperature,
                response_prefix="{",
            )
            hint_payload, status = parse_planner_hint_json(raw_text)
        except Exception as exc:
            self._stats["planner_hint_failures"] += 1
            self._last_planner_hint_status = f"exception:{type(exc).__name__}"
            if self._planner_hint_required:
                raise
            return ""

        self._stats["planner_hint_calls"] += 1
        self._last_planner_hint_raw = raw_text
        self._last_planner_hint_status = status
        if status != "ok" or hint_payload is None:
            self._stats["planner_hint_failures"] += 1
            if self._planner_hint_required:
                raise RuntimeError(f"planner_hint_invalid:{status}")
            return ""

        hint_text = format_planner_hint(hint_payload)
        if not hint_text:
            self._stats["planner_hint_failures"] += 1
            self._last_planner_hint_status = "empty_hint"
            if self._planner_hint_required:
                raise RuntimeError("planner_hint_empty")
            return ""

        self._planner_hint_cache_key = cache_key
        self._planner_hint_cache_text = hint_text
        self._planner_hint_cache_knowledge = list(self._last_retrieved_knowledge)
        self._last_planner_hint = hint_text
        return hint_text

    def _render_user_message(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        action_mode: str,
        adapter_key: str = "combat",
        include_planner_hint: bool = True,
    ) -> str:
        strategy_context = ""
        if self._strategy_context_enabled:
            planner_hint = self._planner_hint_for_state(
                state,
                legal_actions,
                adapter_key=adapter_key,
                allow_generate=include_planner_hint,
            )
            strategy_context = self._strategy_memory.context_text(
                state,
                legal_actions,
                planner_hint=planner_hint,
            )
            self._stats["strategy_context_calls"] += 1
        if action_mode == "structured":
            return render_structured_action_state_text(
                state,
                legal_actions,
                strategy_context=strategy_context,
            )
        return render_state_text(state, legal_actions, strategy_context=strategy_context)

    def _decode_raw_action(
        self,
        raw_text: str,
        legal_actions: list[dict[str, Any]],
        *,
        action_mode: str,
    ) -> DecodedAction:
        if action_mode == "structured":
            return decode_structured_action(raw_text, legal_actions, fallback_index=0)
        return decode_action(raw_text, legal_actions, fallback_index=0)

    @staticmethod
    def _decoded_payload(decoded: DecodedAction) -> dict[str, Any]:
        return {
            "action_index": decoded.action_index,
            "reason": decoded.reason,
            "confidence": decoded.confidence,
            "action_scores": list(decoded.action_scores),
            "score_margin": action_score_margin(decoded.action_scores),
            "used_fallback": decoded.used_fallback,
            "fallback_reason": decoded.fallback_reason,
        }

    def _reason_quality_flags(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        decoded: DecodedAction,
    ) -> list[str]:
        if decoded.used_fallback or not decoded.reason:
            return []
        report = assess_action_quality_report(
            state,
            legal_actions,
            decoded.action_index,
            reason=decoded.reason,
            action_scores=decoded.action_scores,
        )
        return [
            flag
            for flag in sorted(set(report.flags))
            if flag in _EXPLANATION_CONSISTENCY_FLAGS
        ]

    def _retry_quality_flags(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        decoded: DecodedAction,
    ) -> list[str]:
        if decoded.used_fallback:
            return []
        report = assess_action_quality_report(
            state,
            legal_actions,
            decoded.action_index,
            reason=decoded.reason,
            action_scores=decoded.action_scores,
        )
        return [
            flag
            for flag in sorted(set(report.flags))
            if flag in _SAFETY_RETRY_FLAGS
        ]

    def _render_prompt(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        action_mode: str,
        adapter_key: str = "combat",
    ) -> str:
        self._ensure_model_loaded()
        assert self._tokenizer is not None
        user_msg = self._render_user_message(
            state,
            legal_actions,
            action_mode=action_mode,
            adapter_key=adapter_key,
            include_planner_hint=True,
        )
        system_key = "non_combat" if adapter_key == "non_combat" else action_mode
        messages = [
            {"role": "system", "content": self._system_prompts[system_key]},
            {"role": "user", "content": user_msg},
        ]
        # Qwen3/3.5: enable_thinking via chat_template kwargs
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking:
            kwargs["enable_thinking"] = True
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _prompt_token_count(self, prompt_text: str) -> int:
        self._ensure_model_loaded()
        assert self._tokenizer is not None
        prompt_with_prefix = prompt_text + self._response_prefix
        encoded = self._tokenizer(prompt_with_prefix, add_special_tokens=False)
        return len(encoded.get("input_ids") or [])

    def _render_retry_prompt(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        action_mode: str,
        adapter_key: str = "combat",
        raw_text: str,
        fallback_reason: str,
    ) -> str:
        self._ensure_model_loaded()
        assert self._tokenizer is not None
        user_msg = self._render_user_message(
            state,
            legal_actions,
            action_mode=action_mode,
            adapter_key=adapter_key,
            include_planner_hint=True,
        )
        system_key = "non_combat" if adapter_key == "non_combat" else action_mode
        if action_mode == "structured":
            final_instruction = (
                'Return only one JSON object such as '
                '{"action":"play_card","hand_index":1,"target_id":2} '
                'or {"action":"end_turn"}. '
                "Do not use action_index. Do not output multiple objects, a list, or alternative candidates. "
                "Do not include reason / extra keys — strategy text belongs to the planner model."
            )
        else:
            final_instruction = (
                'Return only one valid JSON object matching this schema: '
                '{"action_index":0,"confidence":0.0}. '
                "Do not output multiple objects, a list, or alternative candidates. "
                "Do not include reason / plan / extra keys — strategy text belongs to the planner model. "
                "No markdown, no comments, no extra text."
            )
        safety_hint = ""
        if "dangerous_self_damage" in fallback_reason or "low_hp_self_damage" in fallback_reason:
            safety_hint = (
                "Safety correction: the previous action loses HP while the current attack/end-turn HP loss can kill you. "
                "Do not choose an action with self_hp_loss. Choose block, a safe potion after block, or lethal that removes incoming damage.\n"
            )
        elif "dangerous_end_turn" in fallback_reason:
            safety_hint = (
                "Safety correction: the previous action ends the turn while current attack/end-turn HP loss is dangerous. "
                "Do not end_turn if block, lethal, or a valid defensive potion can reduce the threat.\n"
            )
        correction = (
            "The previous assistant output was invalid and was not executed.\n"
            f"Invalid reason: {fallback_reason}.\n"
            f"Previous output: {raw_text[:240]}\n\n"
            f"{safety_hint}"
            f"Choose again, then {final_instruction}"
        )
        messages = [
            {"role": "system", "content": self._system_prompts[system_key]},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": raw_text[:240]},
            {"role": "user", "content": correction},
        ]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking:
            kwargs["enable_thinking"] = True
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _generate(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        response_prefix: str | None = None,
    ) -> str:
        import torch  # type: ignore

        self._ensure_model_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        prefix = self._response_prefix if response_prefix is None else response_prefix
        generation_temperature = self._temperature if temperature is None else temperature
        prompt_with_prefix = prompt_text + prefix
        inputs = self._tokenizer(prompt_with_prefix, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self._max_new_tokens,
                do_sample=generation_temperature > 0,
                temperature=generation_temperature if generation_temperature > 0 else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return (prefix + self._tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()

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
        enabled = self._filter_optional_potion_actions(enabled, state)
        if not enabled:
            self.reset_episode()
            return None

        step_idx = self._step_counter
        self._step_counter += 1
        self._stats["calls"] += 1
        action_mode = self._resolve_action_mode(enabled)
        adapter_key = self._adapter_key_for_state(state, _context)
        adapter_name = self._adapter_name_for_key(adapter_key)

        if _is_dead_overlay_state(state, enabled):
            self._stats["terminal_overlay_stops"] += 1
            print(
                f"[llm-policy] step={step_idx} terminal_overlay_stop=true",
                flush=True,
            )
            self._write_trace({
                "step": step_idx,
                "route": "terminal_overlay_stop",
                "action_mode": action_mode,
                "adapter_key": adapter_key,
                "adapter_name": adapter_name,
                "adapter_switch_ms": 0.0,
                "gen_ms": 0.0,
                "user_message": self._render_user_message(
                    state,
                    enabled,
                    action_mode=action_mode,
                    adapter_key=adapter_key,
                    include_planner_hint=False,
                ),
                "raw_generation": "",
                "attempts": [],
                "invalid_output": False,
                "decoded": None,
                "chosen_action": None,
                "enabled_count": len(enabled),
                "stats": dict(self._stats),
            })
            self.reset_episode()
            return None

        if self._simple_gate_enabled:
            simple = choose_simple_action(state, enabled)
            if simple is not None:
                decoded = DecodedAction(
                    action_index=simple.action_index,
                    reason=simple.reason,
                    used_fallback=False,
                )
                self._last_decode = decoded
                self._stats["heuristic_calls"] += 1
                chosen_action = dict(enabled[decoded.action_index])
                quality_report = assess_action_quality_report(
                    state,
                    enabled,
                    decoded.action_index,
                    reason=decoded.reason,
                    action_scores=decoded.action_scores,
                ).as_dict()
                quality_flags = list(quality_report.get("flags") or [])
                chosen_action["_policy_route"] = simple.route
                chosen_action["_policy_action_mode"] = action_mode
                chosen_action["_policy_action_index"] = decoded.action_index
                chosen_action["_policy_reason"] = decoded.reason
                chosen_action["_policy_confidence"] = 1.0
                chosen_action["_policy_action_scores"] = []
                chosen_action["_policy_score_margin"] = None
                chosen_action["_policy_adapter_key"] = adapter_key
                chosen_action["_policy_adapter_name"] = adapter_name
                chosen_action["_policy_adapter_switch_ms"] = 0.0
                chosen_action["_policy_gen_ms"] = 0.0
                chosen_action["_policy_quality_flags"] = quality_flags
                print(
                    f"[llm-policy] step={step_idx} route={simple.route} mode={action_mode} "
                    f"adapter={adapter_name or '-'} chosen={decoded.action_index} reason={decoded.reason!r}"
                )
                self._write_trace({
                    "step": step_idx,
                    "route": simple.route,
                    "action_mode": action_mode,
                    "adapter_key": adapter_key,
                    "adapter_name": adapter_name,
                    "adapter_switch_ms": 0.0,
                    "gen_ms": 0.0,
                    "user_message": self._render_user_message(
                        state,
                        enabled,
                        action_mode=action_mode,
                        adapter_key=adapter_key,
                        include_planner_hint=False,
                    ),
                    "raw_generation": "",
                    "attempts": [],
                    "invalid_output": False,
                    "decoded": self._decoded_payload(decoded),
                    "quality_flags": quality_flags,
                    "quality_report": quality_report,
                    "chosen_action": {
                        k: chosen_action.get(k)
                        for k in ("action", "type", "card_id", "card_index", "target_id")
                        if k in chosen_action
                    },
                    "enabled_count": len(enabled),
                    "stats": dict(self._stats),
                })
                self._strategy_memory.record_action(chosen_action)
                return chosen_action

        if self._survival_gate_enabled and adapter_key == "combat":
            survival = choose_survival_action(state, enabled)
            if survival is not None:
                decoded = DecodedAction(
                    action_index=survival.action_index,
                    reason=survival.reason,
                    used_fallback=False,
                )
                self._last_decode = decoded
                self._stats["heuristic_calls"] += 1
                self._stats["survival_gate_calls"] += 1
                chosen_action = dict(enabled[decoded.action_index])
                quality_report = assess_action_quality_report(
                    state,
                    enabled,
                    decoded.action_index,
                    reason=decoded.reason,
                    action_scores=decoded.action_scores,
                ).as_dict()
                quality_flags = list(quality_report.get("flags") or [])
                chosen_action["_policy_route"] = survival.route
                chosen_action["_policy_action_mode"] = action_mode
                chosen_action["_policy_action_index"] = decoded.action_index
                chosen_action["_policy_reason"] = decoded.reason
                chosen_action["_policy_confidence"] = 1.0
                chosen_action["_policy_action_scores"] = []
                chosen_action["_policy_score_margin"] = None
                chosen_action["_policy_adapter_key"] = adapter_key
                chosen_action["_policy_adapter_name"] = adapter_name
                chosen_action["_policy_adapter_switch_ms"] = 0.0
                chosen_action["_policy_gen_ms"] = 0.0
                chosen_action["_policy_quality_flags"] = quality_flags
                print(
                    f"[llm-policy] step={step_idx} route={survival.route} mode={action_mode} "
                    f"adapter={adapter_name or '-'} chosen={decoded.action_index} reason={decoded.reason!r}"
                )
                self._write_trace({
                    "step": step_idx,
                    "route": survival.route,
                    "action_mode": action_mode,
                    "adapter_key": adapter_key,
                    "adapter_name": adapter_name,
                    "adapter_switch_ms": 0.0,
                    "gen_ms": 0.0,
                    "user_message": self._render_user_message(
                        state,
                        enabled,
                        action_mode=action_mode,
                        adapter_key=adapter_key,
                        include_planner_hint=False,
                    ),
                    "raw_generation": "",
                    "attempts": [],
                    "invalid_output": False,
                    "decoded": self._decoded_payload(decoded),
                    "quality_flags": quality_flags,
                    "quality_report": quality_report,
                    "chosen_action": {
                        k: chosen_action.get(k)
                        for k in ("action", "type", "card_id", "card_index", "target_id")
                        if k in chosen_action
                    },
                    "enabled_count": len(enabled),
                    "stats": dict(self._stats),
                })
                self._strategy_memory.record_action(chosen_action)
                return chosen_action

        self._stats["llm_calls"] += 1
        if action_mode == "structured":
            self._stats["structured_calls"] += 1
        prompt_text = self._render_prompt(state, enabled, action_mode=action_mode, adapter_key=adapter_key)
        attempts: list[dict[str, Any]] = []
        user_msg = prompt_text.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0].strip()
        prompt_tokens = self._prompt_token_count(prompt_text)
        if prompt_tokens > self._max_seq_length:
            teacher = pick_action(state, enabled)
            teacher_idx = max(0, min(int(teacher.action_index), len(enabled) - 1))
            chosen_action = dict(enabled[teacher_idx])
            decoded = DecodedAction(
                action_index=teacher_idx,
                reason=teacher.reason,
                used_fallback=False,
            )
            self._last_decode = decoded
            self._stats["prompt_too_long"] += 1
            self._stats["heuristic_calls"] += 1
            chosen_action["_policy_route"] = "prompt_too_long_heuristic"
            chosen_action["_policy_action_mode"] = action_mode
            chosen_action["_policy_action_index"] = teacher_idx
            chosen_action["_policy_reason"] = teacher.reason
            chosen_action["_policy_adapter_key"] = adapter_key
            chosen_action["_policy_adapter_name"] = adapter_name
            self._write_trace({
                "step": step_idx,
                "route": "prompt_too_long_heuristic",
                "action_mode": action_mode,
                "adapter_key": adapter_key,
                "adapter_name": adapter_name,
                "gen_ms": 0.0,
                "prompt_tokens": prompt_tokens,
                "max_seq_length": self._max_seq_length,
                "user_message": user_msg,
                "planner_hint": self._last_planner_hint,
                "planner_hint_status": self._last_planner_hint_status,
                "retrieved_knowledge": self._last_retrieved_knowledge,
                "raw_generation": "",
                "attempts": [],
                "invalid_output": False,
                "decoded": self._decoded_payload(decoded),
                "chosen_action": {
                    k: chosen_action.get(k)
                    for k in ("action", "type", "card_id", "card_index", "target_id")
                    if k in chosen_action
                },
                "enabled_count": len(enabled),
                "stats": dict(self._stats),
            })
            self._strategy_memory.record_action(chosen_action)
            return chosen_action

        decoded: DecodedAction | None = None
        raw_text = ""
        gen_ms = 0.0
        adapter_switch_ms = 0.0
        for attempt in range(self._parse_retries + 1):
            if attempt > 0:
                self._stats["retry_attempts"] += 1
            gen_t0 = time.monotonic()
            adapter_switch_ms += self._activate_adapter(adapter_name)
            raw_text = self._generate(prompt_text)
            gen_ms = (time.monotonic() - gen_t0) * 1000.0
            decoded = self._decode_raw_action(raw_text, enabled, action_mode=action_mode)
            strict_json_status = _strict_json_status(raw_text) if action_mode == "index" else "not_applicable"
            reason_quality_flags = self._retry_quality_flags(state, enabled, decoded)
            if reason_quality_flags:
                decoded = DecodedAction(
                    action_index=decoded.action_index,
                    reason=decoded.reason,
                    used_fallback=True,
                    fallback_reason=",".join(reason_quality_flags),
                    confidence=decoded.confidence,
                    action_scores=decoded.action_scores,
                )
            if strict_json_status not in {"ok", "not_applicable"} and not decoded.used_fallback:
                decoded = DecodedAction(
                    action_index=decoded.action_index,
                    reason=decoded.reason,
                    used_fallback=True,
                    fallback_reason="strict_json_required",
                    confidence=decoded.confidence,
                    action_scores=decoded.action_scores,
                )
            self._stats["generated_outputs"] += 1
            if strict_json_status in {"ok", "not_applicable"}:
                self._stats["strict_json_ok"] += 1
            else:
                self._stats["strict_json_failures"] += 1
            attempts.append({
                "attempt": attempt,
                "action_mode": action_mode,
                "adapter_key": adapter_key,
                "adapter_name": adapter_name,
                "adapter_switch_ms": round(adapter_switch_ms, 3),
                "gen_ms": round(gen_ms, 1),
                "raw_generation": raw_text,
                "strict_json_status": strict_json_status,
                "strict_json_ok": strict_json_status in {"ok", "not_applicable"},
                "decoded": self._decoded_payload(decoded),
                "reason_quality_flags": reason_quality_flags,
            })
            if not decoded.used_fallback:
                break
            self._stats["invalid_attempts"] += 1
            if attempt == 0:
                self._stats["first_attempt_invalid"] += 1
            if decoded.fallback_reason == "action_index_out_of_range":
                self._stats["out_of_range"] += 1
            elif decoded.fallback_reason == "action_index_not_int":
                self._stats["action_index_not_int"] += 1
            elif decoded.fallback_reason == "parse_failed":
                self._stats["parse_failures"] += 1
            elif decoded.fallback_reason == "strict_json_required":
                self._stats["strict_json_rejections"] += 1
            elif _has_any_flag(decoded.fallback_reason, _EXPLANATION_CONSISTENCY_FLAGS):
                self._stats["reason_consistency_failures"] += 1
            elif _has_any_flag(decoded.fallback_reason, _SAFETY_RETRY_FLAGS):
                self._stats["safety_rejections"] += 1
            else:
                self._stats["mapping_failures"] += 1
            if attempt < self._parse_retries:
                prompt_text = self._render_retry_prompt(
                    state,
                    enabled,
                    action_mode=action_mode,
                    adapter_key=adapter_key,
                    raw_text=raw_text,
                    fallback_reason=decoded.fallback_reason,
                )
                retry_prompt_tokens = self._prompt_token_count(prompt_text)
                if retry_prompt_tokens > self._max_seq_length:
                    self._stats["prompt_too_long"] += 1
                    attempts.append({
                        "attempt": attempt + 1,
                        "action_mode": action_mode,
                        "adapter_key": adapter_key,
                        "adapter_name": adapter_name,
                        "adapter_switch_ms": round(adapter_switch_ms, 3),
                        "gen_ms": 0.0,
                        "raw_generation": "",
                        "strict_json_status": "not_generated",
                        "strict_json_ok": False,
                        "prompt_tokens": retry_prompt_tokens,
                        "max_seq_length": self._max_seq_length,
                        "decoded": {
                            "action_index": -1,
                            "reason": "",
                            "confidence": None,
                            "action_scores": [],
                            "score_margin": None,
                            "used_fallback": True,
                            "fallback_reason": "prompt_too_long",
                        },
                        "reason_quality_flags": [],
                    })
                    decoded = DecodedAction(
                        action_index=-1,
                        reason="",
                        used_fallback=True,
                        fallback_reason="prompt_too_long",
                    )
                    break

        assert decoded is not None
        self._last_decode = decoded
        if not decoded.used_fallback and any(
            bool((attempt_record.get("decoded") or {}).get("used_fallback"))
            for attempt_record in attempts[:-1]
        ):
            self._stats["retry_recovered"] += 1
        if decoded.used_fallback:
            self._stats["invalid_outputs"] += 1
            if (
                self._recover_explanation_invalid
                and _has_any_flag(decoded.fallback_reason, _EXPLANATION_CONSISTENCY_FLAGS)
                and not _has_any_flag(decoded.fallback_reason, _SAFETY_RETRY_FLAGS)
            ):
                teacher = pick_action(state, enabled)
                teacher_idx = max(0, min(int(teacher.action_index), len(enabled) - 1))
                chosen_action = dict(enabled[teacher_idx])
                quality_report = assess_action_quality_report(
                    state,
                    enabled,
                    teacher_idx,
                    reason=teacher.reason,
                    action_scores=[],
                ).as_dict()
                quality_flags = list(quality_report.get("flags") or [])
                self._stats["heuristic_calls"] += 1
                self._stats["explanation_recoveries"] += 1
                chosen_action["_policy_route"] = "heuristic_recovery"
                chosen_action["_policy_action_mode"] = action_mode
                chosen_action["_policy_action_index"] = teacher_idx
                chosen_action["_policy_reason"] = teacher.reason
                chosen_action["_policy_confidence"] = None
                chosen_action["_policy_action_scores"] = []
                chosen_action["_policy_score_margin"] = None
                chosen_action["_policy_adapter_key"] = adapter_key
                chosen_action["_policy_adapter_name"] = adapter_name
                chosen_action["_policy_adapter_switch_ms"] = round(adapter_switch_ms, 3)
                chosen_action["_policy_gen_ms"] = round(gen_ms, 1)
                chosen_action["_policy_raw_generation"] = raw_text
                chosen_action["_policy_quality_flags"] = quality_flags
                chosen_action["_policy_invalid_recovered"] = True
                chosen_action["_policy_invalid_reason"] = decoded.fallback_reason
                self._strategy_memory.record_action(chosen_action)
                print(
                    f"[llm-policy] step={step_idx} recovered_invalid=true "
                    f"reason={decoded.fallback_reason!r} teacher={teacher_idx} "
                    f"teacher_reason={teacher.reason!r}",
                    flush=True,
                )
                self._write_trace({
                    "step": step_idx,
                    "route": "heuristic_recovery",
                    "action_mode": action_mode,
                    "adapter_key": adapter_key,
                    "adapter_name": adapter_name,
                    "adapter_switch_ms": round(adapter_switch_ms, 3),
                    "gen_ms": round(gen_ms, 1),
                    "invalid_output": True,
                    "recovered": True,
                    "recovery_reason": "heuristic_after_explanation_invalid",
                    "attempts": attempts,
                    "user_message": user_msg,
                    "planner_hint": self._last_planner_hint,
                    "planner_hint_status": self._last_planner_hint_status,
                    "retrieved_knowledge": self._last_retrieved_knowledge,
                    "raw_generation": raw_text,
                    "decoded": self._decoded_payload(decoded),
                    "quality_flags": quality_flags,
                    "quality_report": quality_report,
                    "chosen_action": {
                        k: chosen_action.get(k)
                        for k in ("action", "type", "card_id", "card_index", "target_id")
                        if k in chosen_action
                    },
                    "enabled_count": len(enabled),
                    "stats": dict(self._stats),
                })
                return chosen_action
            if _has_any_flag(decoded.fallback_reason, _SAFETY_RETRY_FLAGS):
                rescue_idx, rescue_reason, quality_report = self._last_resort_action(
                    state,
                    enabled,
                    reason=f"last_resort_after_{decoded.fallback_reason}",
                )
                chosen_action = dict(enabled[rescue_idx])
                quality_flags = list(quality_report.get("flags") or [])
                self._stats["heuristic_calls"] += 1
                chosen_action["_policy_route"] = "heuristic_last_resort"
                chosen_action["_policy_action_mode"] = action_mode
                chosen_action["_policy_action_index"] = rescue_idx
                chosen_action["_policy_reason"] = rescue_reason
                chosen_action["_policy_confidence"] = None
                chosen_action["_policy_action_scores"] = []
                chosen_action["_policy_score_margin"] = None
                chosen_action["_policy_adapter_key"] = adapter_key
                chosen_action["_policy_adapter_name"] = adapter_name
                chosen_action["_policy_adapter_switch_ms"] = round(adapter_switch_ms, 3)
                chosen_action["_policy_gen_ms"] = round(gen_ms, 1)
                chosen_action["_policy_raw_generation"] = raw_text
                chosen_action["_policy_quality_flags"] = quality_flags
                chosen_action["_policy_invalid_recovered"] = True
                chosen_action["_policy_invalid_reason"] = decoded.fallback_reason
                self._strategy_memory.record_action(chosen_action)
                print(
                    f"[llm-policy] step={step_idx} safety_last_resort=true "
                    f"reason={decoded.fallback_reason!r} chosen={rescue_idx} "
                    f"rescue_reason={rescue_reason!r}",
                    flush=True,
                )
                self._write_trace({
                    "step": step_idx,
                    "route": "heuristic_last_resort",
                    "action_mode": action_mode,
                    "adapter_key": adapter_key,
                    "adapter_name": adapter_name,
                    "adapter_switch_ms": round(adapter_switch_ms, 3),
                    "gen_ms": round(gen_ms, 1),
                    "invalid_output": True,
                    "recovered": True,
                    "recovery_reason": "last_resort_after_safety_rejection",
                    "attempts": attempts,
                    "user_message": user_msg,
                    "planner_hint": self._last_planner_hint,
                    "planner_hint_status": self._last_planner_hint_status,
                    "retrieved_knowledge": self._last_retrieved_knowledge,
                    "raw_generation": raw_text,
                    "decoded": self._decoded_payload(decoded),
                    "quality_flags": quality_flags,
                    "quality_report": quality_report,
                    "chosen_action": {
                        k: chosen_action.get(k)
                        for k in ("action", "type", "card_id", "card_index", "target_id")
                        if k in chosen_action
                    },
                    "enabled_count": len(enabled),
                    "stats": dict(self._stats),
                })
                return chosen_action
            print(
                f"[llm-policy] step={step_idx} invalid_output=true "
                f"attempts={len(attempts)} reason={decoded.fallback_reason!r}",
                flush=True,
            )
            self._write_trace({
                "step": step_idx,
                "route": "llm",
                "action_mode": action_mode,
                "adapter_key": adapter_key,
                "adapter_name": adapter_name,
                "adapter_switch_ms": round(adapter_switch_ms, 3),
                "invalid_output": True,
                "attempts": attempts,
                "user_message": user_msg,
                "planner_hint": self._last_planner_hint,
                "planner_hint_status": self._last_planner_hint_status,
                "retrieved_knowledge": self._last_retrieved_knowledge,
                "raw_generation": raw_text,
                "decoded": self._decoded_payload(decoded),
                "chosen_action": None,
                "enabled_count": len(enabled),
                "stats": dict(self._stats),
            })
            return None

        chosen_action = dict(enabled[decoded.action_index])
        chosen_action["_policy_route"] = "llm"
        chosen_action["_policy_action_mode"] = action_mode
        chosen_action["_policy_action_index"] = decoded.action_index
        chosen_action["_policy_reason"] = decoded.reason
        chosen_action["_policy_confidence"] = decoded.confidence
        chosen_action["_policy_action_scores"] = list(decoded.action_scores)
        chosen_action["_policy_score_margin"] = action_score_margin(decoded.action_scores)
        chosen_action["_policy_adapter_key"] = adapter_key
        chosen_action["_policy_adapter_name"] = adapter_name
        chosen_action["_policy_adapter_switch_ms"] = round(adapter_switch_ms, 3)
        chosen_action["_policy_gen_ms"] = round(gen_ms, 1)
        chosen_action["_policy_raw_generation"] = raw_text
        quality_report = assess_action_quality_report(
            state,
            enabled,
            decoded.action_index,
            reason=decoded.reason,
            action_scores=decoded.action_scores,
        ).as_dict()
        quality_flags = list(quality_report.get("flags") or [])
        chosen_action["_policy_quality_flags"] = quality_flags
        self._strategy_memory.record_action(chosen_action)
        print(
            f"[llm-policy] step={step_idx} mode={action_mode} gen_ms={gen_ms:.0f} "
            f"adapter={adapter_name or '-'} switch_ms={adapter_switch_ms:.3f} "
            f"chosen={decoded.action_index} retry_count={len(attempts) - 1} "
            f"reason={decoded.reason!r}"
        )
        # trace 落盘（如果 env 有要求）
        # 只保存 user 消息（prompt_text 末尾那段），不保存整个 system prompt，文件会小很多
        self._write_trace({
            "step": step_idx,
            "route": "llm",
            "action_mode": action_mode,
            "adapter_key": adapter_key,
            "adapter_name": adapter_name,
            "adapter_switch_ms": round(adapter_switch_ms, 3),
            "gen_ms": round(gen_ms, 1),
            "user_message": user_msg,
            "planner_hint": self._last_planner_hint,
            "planner_hint_status": self._last_planner_hint_status,
            "retrieved_knowledge": self._last_retrieved_knowledge,
            "raw_generation": raw_text,
            "attempts": attempts,
            "invalid_output": False,
            "decoded": self._decoded_payload(decoded),
            "quality_flags": quality_flags,
            "quality_report": quality_report,
            "chosen_action": {
                k: chosen_action.get(k)
                for k in ("action", "type", "card_id", "card_index", "target_id")
                if k in chosen_action
            },
            "enabled_count": len(enabled),
            "stats": dict(self._stats),
        })
        return chosen_action

    def _is_urgent_potion_action(self, action: dict[str, Any], state: dict[str, Any]) -> bool:
        if _policy_action_type(action) != "use_potion":
            return False
        raw = " ".join(
            str(action.get(key) or "")
            for key in ("label", "potion_id", "id", "name")
        ).upper()
        slot = action.get("slot")
        potions = (state.get("player") if isinstance(state.get("player"), dict) else {}).get("potions") or []
        if isinstance(potions, list):
            try:
                potion = potions[int(slot)] if slot is not None else None
            except (TypeError, ValueError, IndexError):
                potion = None
            if isinstance(potion, dict):
                raw += " " + " ".join(str(potion.get(key) or "") for key in ("id", "potion_id", "name")).upper()
        return any(pid in raw for pid in _URGENT_POTION_IDS)

    def _last_resort_action(
        self,
        state: dict[str, Any],
        enabled: list[dict[str, Any]],
        *,
        reason: str,
    ) -> tuple[int, str, dict[str, Any]]:
        """Pick the least bad legal action when safety retry found no safe move.

        Returning None here aborts the run at the bridge level. If the game
        state is already doomed, executing the least bad action gives us a real
        death trace and usable hardcase instead of a protocol stop.
        """
        weights = {
            "dangerous_self_damage": 140,
            "low_hp_self_damage": 110,
            "dangerous_end_turn": 90,
            "missed_visible_lethal": 80,
            "end_turn_with_playable_cards": 25,
            "floating_energy_end_turn": 15,
            "unnecessary_potion_use": 10,
        }
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, action in enumerate(enabled):
            report = assess_action_quality_report(
                state,
                enabled,
                idx,
                reason=reason,
                action_scores=[],
            ).as_dict()
            flags = [str(flag) for flag in report.get("flags") or []]
            score = float(sum(weights.get(flag, 35) for flag in flags))
            if _policy_action_type(action) == "end_turn":
                score += 5.0
            if _policy_action_type(action) == "use_potion":
                score += 2.0
            scored.append((score, idx, report))
        if not scored:
            return 0, reason, {}
        score, idx, report = min(scored, key=lambda item: (item[0], item[1]))
        action = enabled[idx]
        action_name = _policy_action_type(action) or "action"
        return idx, f"last resort: {action_name} after safety rejection; score={score:g}", report

    def _urgent_potion_allowed(self, enabled: list[dict[str, Any]], state: dict[str, Any]) -> bool:
        if not any(self._is_urgent_potion_action(action, state) for action in enabled):
            return False
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
        battle_player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
        try:
            hp = float(player.get("hp") or player.get("current_hp") or battle_player.get("hp") or 0)
            block = float(battle_player.get("block") if battle_player.get("block") is not None else player.get("block") or 0)
        except (TypeError, ValueError):
            return False

        incoming = 0.0
        enemies = state.get("enemies") or battle.get("enemies") or []
        if isinstance(enemies, list):
            for enemy in enemies:
                if not isinstance(enemy, dict):
                    continue
                intent = str(enemy.get("intent_type") or enemy.get("next_move_id") or enemy.get("intent") or "").upper()
                if "ATTACK" not in intent:
                    continue
                try:
                    damage = float(enemy.get("intent_damage") or enemy.get("move_base_damage") or 0)
                    hits = max(1, int(enemy.get("intent_hits") or enemy.get("move_hits") or 1))
                except (TypeError, ValueError):
                    continue
                incoming += damage * hits

        end_turn_hp_loss = 0.0
        powers = battle_player.get("powers") or player.get("powers") or []
        if isinstance(powers, list):
            for power in powers:
                if not isinstance(power, dict):
                    continue
                power_id = str(power.get("id") or power.get("power_id") or power.get("name") or "").upper()
                if power_id != "CONSTRICT_POWER" and "CONSTRICT" not in power_id:
                    continue
                try:
                    end_turn_hp_loss += max(0.0, float(power.get("amount") or power.get("stacks") or power.get("stack") or 0))
                except (TypeError, ValueError):
                    continue
        return hp > 0 and (max(0.0, incoming - block) + end_turn_hp_loss) >= max(1.0, hp - 2.0)

    def _filter_optional_potion_actions(
        self,
        enabled: list[dict[str, Any]],
        state: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hide combat potion use by default; reward `claim_reward` is separate.

        Potion timing needs a dedicated policy. Until then, exposing
        `use_potion` to the combat LoRA makes it burn potions as generic free
        actions.
        """
        if _env_int("STS2_LLM_ALLOW_POTIONS", 0) == 1:
            return enabled
        if state is not None and self._urgent_potion_allowed(enabled, state):
            filtered = [
                action
                for action in enabled
                if _policy_action_type(action) != "use_potion"
                or self._is_urgent_potion_action(action, state)
            ]
            if filtered and len(filtered) != len(enabled):
                self._stats["potion_actions_suppressed"] += len(enabled) - len(filtered)
                return filtered
            return enabled
        non_potion = [action for action in enabled if _policy_action_type(action) != "use_potion"]
        if non_potion and len(non_potion) != len(enabled):
            self._stats["potion_actions_suppressed"] += len(enabled) - len(non_potion)
            return non_potion
        return enabled

    @property
    def last_decode(self) -> DecodedAction | None:
        return self._last_decode

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)


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
