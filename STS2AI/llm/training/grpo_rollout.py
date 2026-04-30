"""用当前 LLM Policy 跑 rollout，产出带 reward 的 GRPO 训练数据。

与启发式 rollout 的区别：
- 不再用 heuristic_teacher.pick_action，而是调用当前 LoRA adapter 生成动作
- temperature > 0，采样多样化轨迹
- 记录终止状态，但非 boss 战 reward 主要按净掉血和敌方血量推进计算
- 按 (encounter, initial_state) 分组，计算 relative advantage

运行（在 unsloth venv 里，需要加载模型）：

    python STS2AI/llm/training/grpo_rollout.py \\
        --adapter-dir STS2AI/Artifacts/llm/sft/toy_sft/adapter \\
        --num-generations 8 \\
        --episodes-per-encounter 8 \\
        --out-subdir grpo_v0 \\
        --temperature 0.8

产物：
  STS2AI/Artifacts/llm/datasets/grpo_v0/
    rollout.jsonl     # 每条是一个 episode，含 steps + final_reward + advantage
    meta.json         # 统计信息
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_STS2AI_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE_ROOT = _STS2AI_ROOT / "bridge"
for _path in (_STS2AI_ROOT, _BRIDGE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from llm.data_pipeline.action_decoder import DecodedAction, action_score_margin, decode_action
from llm.data_pipeline.encounter_pool import (
    EncounterSpec,
    _parse_archetype_min_count,
    VALID_TIERS,
    encounter_key,
    encounter_label,
    filter_by_tier,
    filter_encounter_pool,
    load_skada_case_pool,
)
from llm.data_pipeline.action_quality import (
    assess_action_quality_report,
    count_quality_flags,
    summarize_quality_reports,
)
from llm.data_pipeline.guide_knowledge import render_retrieved_knowledge_for_state
from llm.data_pipeline.state_renderer import render_state_text
from llm.data_pipeline.planner_hint import (
    DEFAULT_PLANNER_HINT_REFRESH,
    PLANNER_HINT_REFRESH_CHOICES,
    format_planner_hint,
    parse_planner_hint_json,
    planner_hint_cache_key,
    render_planner_hint_user_message,
)
from llm.data_pipeline.strategy_context import StrategyMemory
from llm.inference.hybrid_gate import choose_simple_action
from llm.metrics import summarize_dataset_dir, write_json
from llm.paths import DATASETS_ROOT, ensure_dirs, BASE_MODEL_ID
from llm.prompts import load_system_prompt

from game_bridge.session import create_game_session
from game_bridge.session.state_semantics import is_actionable_combat_state, is_combat_state


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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


def _rollout_action_type(action: dict[str, Any]) -> str:
    return str(action.get("action") or action.get("action_type") or action.get("type") or "").strip().lower()


def _filter_optional_potion_actions(enabled: list[dict[str, Any]], stats: Counter[str] | None = None) -> list[dict[str, Any]]:
    """Match live LLM policy: hide optional combat potion use by default."""
    if os.environ.get("STS2_LLM_ALLOW_POTIONS", "").strip() == "1":
        return enabled
    non_potion = [action for action in enabled if _rollout_action_type(action) != "use_potion"]
    if non_potion and len(non_potion) != len(enabled):
        if stats is not None:
            stats["potion_actions_suppressed"] += len(enabled) - len(non_potion)
        return non_potion
    return enabled


def _is_urgent_potion_action(action: dict[str, Any], state: dict[str, Any]) -> bool:
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
    return any(pid in raw for pid in {"FORTIFIER", "BLOCK_POTION", "HEALTH_POTION", "REGEN_POTION"})


def _urgent_potion_allowed(enabled: list[dict[str, Any]], state: dict[str, Any]) -> bool:
    if not any(_is_urgent_potion_action(action, state) for action in enabled):
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
                incoming += float(enemy.get("intent_damage") or enemy.get("move_base_damage") or 0) * max(
                    1,
                    int(enemy.get("intent_hits") or enemy.get("move_hits") or 1),
                )
            except (TypeError, ValueError):
                continue
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


def _filter_optional_potion_actions_live(
    enabled: list[dict[str, Any]],
    state: dict[str, Any],
    stats: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    """Match live policy: hide potions except whitelisted urgent defensive potion use."""
    if os.environ.get("STS2_LLM_ALLOW_POTIONS", "").strip() == "1":
        return enabled
    if _urgent_potion_allowed(enabled, state):
        filtered = [
            action
            for action in enabled
            if _rollout_action_type(action) != "use_potion"
            or _is_urgent_potion_action(action, state)
        ]
        if filtered and len(filtered) != len(enabled):
            if stats is not None:
                stats["potion_actions_suppressed"] += len(enabled) - len(filtered)
            return filtered
        return enabled
    return _filter_optional_potion_actions(enabled, stats)


# ---------------------------------------------------------------------------
# Reward 函数设计
# ---------------------------------------------------------------------------

def compute_episode_reward(
    outcome: str,
    *,
    player_hp_start: float,
    player_hp_end: float,
    player_max_hp: float,
    num_turns: int,
    damage_dealt: float = 0.0,
    damage_taken: float = 0.0,
    enemy_damage_progress: float | None = None,
) -> dict[str, float]:
    """把一个 episode 的统计数据转换成标量 reward（可拆解查看）。

    非 boss 战的核心质量不是 victory，而是战斗成本：
    1. 已结束战斗主要看净掉血。
    2. 未结束/失败战斗主要看敌方血量推进。
    3. 协议错误和执行错误单独重罚，不把普通 max_steps 直接当成大负样本。
    """
    protocol_failure = (
        outcome.startswith("invalid_output:")
        or outcome.startswith("step_failed:")
        or outcome.startswith("reset_failed:")
        or outcome in {"left_combat", "no_legal_actions"}
    )
    if enemy_damage_progress is None:
        progress = min(1.0, max(0.0, damage_dealt / 100.0))
    else:
        progress = min(1.0, max(0.0, float(enemy_damage_progress)))
    hp_lost = max(0.0, damage_taken, player_hp_start - player_hp_end)
    progress_r = 8.0 * progress
    hp_loss_r = -0.45 * hp_lost
    efficiency_r = -0.05 * max(0, num_turns - 3)
    terminal_r = -6.0 if protocol_failure else 0.0

    total = progress_r + hp_loss_r + efficiency_r + terminal_r
    return {
        "total": total,
        "enemy_damage_progress": progress_r,
        "hp_loss": hp_loss_r,
        "efficiency": efficiency_r,
        "terminal": terminal_r,
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
    }


def _float_field(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _enemy_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    enemies = state.get("enemies")
    if not isinstance(enemies, list):
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        enemies = battle.get("enemies")
    return [enemy for enemy in (enemies or []) if isinstance(enemy, dict)]


def _enemy_key(enemy: dict[str, Any], index: int) -> str:
    for key in ("combat_id", "target_id"):
        value = enemy.get(key)
        if value is not None:
            return f"{key}:{value}"
    monster_id = enemy.get("monster_id") or enemy.get("entity_id") or enemy.get("id") or enemy.get("name")
    return f"enemy:{monster_id}:{index}"


def _enemy_hp_snapshot(state: dict[str, Any]) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for index, enemy in enumerate(_enemy_rows(state)):
        hp = _float_field(enemy, "hp", "current_hp", default=0.0)
        snapshot[_enemy_key(enemy, index)] = max(0.0, hp)
    return snapshot


def compute_enemy_damage_progress(
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
) -> dict[str, float]:
    """Estimate enemy HP progress by stable combat ids.

    Missing final enemies are treated as dead. This matches the sim states where
    killed monsters disappear from the live enemy list.
    """
    start = _enemy_hp_snapshot(initial_state)
    end = _enemy_hp_snapshot(final_state)
    total_start = sum(start.values())
    total_end = 0.0
    damage_dealt = 0.0
    defeated = 0
    for key, start_hp in start.items():
        end_hp = max(0.0, end.get(key, 0.0))
        total_end += end_hp
        if end_hp <= 0.0:
            defeated += 1
        damage_dealt += min(start_hp, max(0.0, start_hp - end_hp))
    progress = damage_dealt / total_start if total_start > 0 else 0.0
    return {
        "enemy_hp_start": round(total_start, 4),
        "enemy_hp_end": round(total_end, 4),
        "enemy_damage_dealt": round(damage_dealt, 4),
        "enemy_damage_progress": round(min(1.0, max(0.0, progress)), 4),
        "enemy_defeated_count": float(defeated),
        "enemy_start_count": float(len(start)),
    }


def _player_hp(state: dict[str, Any]) -> float | None:
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    if not player:
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
    for key in ("hp", "current_hp"):
        value = player.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# 轻量 LLM Policy（用于 rollout，去掉 trace/print 开销）
# ---------------------------------------------------------------------------

class _RolloutPolicy:
    """内部用的轻量 policy，只保留 generate + decode 核心逻辑。"""

    def __init__(
        self,
        adapter_dir: str | None,
        *,
        base_model_id: str = BASE_MODEL_ID,
        temperature: float = 0.8,
        max_new_tokens: int = 320,
        max_seq_length: int = 2048,
        load_in_4bit: bool = False,
        enable_thinking: bool = True,
        parse_retries: int = 1,
        strict_json_required: bool = True,
        planner_hint_adapter_dir: str | None = None,
        planner_hint_refresh: str = DEFAULT_PLANNER_HINT_REFRESH,
        planner_hint_max_new_tokens: int = 240,
    ) -> None:
        from unsloth import FastLanguageModel

        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._max_seq_length = max_seq_length
        self._enable_thinking = enable_thinking
        self._parse_retries = max(0, parse_retries)
        self._strict_json_required = bool(strict_json_required)
        self._response_prefix = '{"'
        self._system_prompt = load_system_prompt()
        self._planner_hint_system_prompt = load_system_prompt("planner_hint")
        self._strategy_memory = StrategyMemory()
        self._planner_hint_enabled = bool(planner_hint_adapter_dir and adapter_dir)
        self._planner_hint_adapter_name = "planner_hint"
        self._combat_adapter_name = "combat" if self._planner_hint_enabled else None
        self._planner_hint_refresh = (
            planner_hint_refresh
            if planner_hint_refresh in PLANNER_HINT_REFRESH_CHOICES
            else DEFAULT_PLANNER_HINT_REFRESH
        )
        self._planner_hint_max_new_tokens = int(planner_hint_max_new_tokens)
        self._planner_hint_required = _env_int("STS2_LLM_PLANNER_HINT_REQUIRED", 0) == 1
        self._guide_required = _env_int("STS2_LLM_GUIDE_REQUIRED", 0) == 1
        self._planner_hint_cache_key = ""
        self._planner_hint_cache_text = ""
        self._planner_hint_cache_knowledge: list[dict[str, Any]] = []
        self._last_planner_hint = ""
        self._last_planner_hint_status = "disabled"
        self._last_planner_hint_raw = ""
        self._last_retrieved_knowledge: list[dict[str, Any]] = []
        self.stats: Counter[str] = Counter()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        if adapter_dir and planner_hint_adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir, adapter_name=self._combat_adapter_name)
            model.load_adapter(planner_hint_adapter_dir, adapter_name=self._planner_hint_adapter_name)
        elif adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)
        FastLanguageModel.for_inference(model)

        self._model = model
        self._tokenizer = tokenizer

    def _prompt_token_count(self, prompt_text: str) -> int:
        prompt_with_prefix = prompt_text + self._response_prefix
        encoded = self._tokenizer(prompt_with_prefix, add_special_tokens=False)
        return len(encoded.get("input_ids") or [])

    def _prompt_too_long_decision(
        self,
        *,
        prompt_text: str,
        user_msg: str,
        strategy_context: str,
        attempts: list[dict[str, Any]] | None = None,
    ) -> "PolicyDecision":
        token_count = self._prompt_token_count(prompt_text)
        self.stats["prompt_too_long"] += 1
        self.stats["invalid_outputs"] += 1
        attempt_rows = list(attempts or [])
        attempt_rows.append({
            "attempt": len(attempt_rows),
            "gen_ms": 0.0,
            "raw_generation": "",
            "strict_json_status": "not_generated",
            "strict_json_ok": False,
            "decoded": {
                "action_index": -1,
                "reason": "",
                "confidence": None,
                "action_scores": [],
                "score_margin": None,
                "used_fallback": True,
                "fallback_reason": "prompt_too_long",
            },
            "prompt_tokens": token_count,
            "max_seq_length": self._max_seq_length,
            "reason_quality_flags": [],
        })
        return PolicyDecision(
            action_index=-1,
            reason="",
            raw_generation="",
            attempts=attempt_rows,
            user_message=user_msg,
            strategy_context=strategy_context,
            invalid_output=True,
            fallback_reason="prompt_too_long",
            gen_ms=0.0,
            confidence=None,
            action_scores=[],
            score_margin=None,
            planner_hint=self._last_planner_hint,
            planner_hint_status=self._last_planner_hint_status,
            retrieved_knowledge=self._last_retrieved_knowledge,
        )

    def reset_episode(self) -> None:
        self._strategy_memory.reset()
        self._planner_hint_cache_key = ""
        self._planner_hint_cache_text = ""
        self._planner_hint_cache_knowledge = []
        self._last_planner_hint = ""
        self._last_planner_hint_status = "reset"
        self._last_planner_hint_raw = ""
        self._last_retrieved_knowledge = []

    def record_action(self, action: dict[str, Any] | None) -> None:
        self._strategy_memory.record_action(action)

    def _set_adapter(self, adapter_name: str | None) -> None:
        if adapter_name and hasattr(self._model, "set_adapter"):
            self._model.set_adapter(adapter_name)

    def _generate(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        adapter_name: str | None = None,
        response_prefix: str | None = None,
    ) -> tuple[str, float]:
        import torch

        self._set_adapter(adapter_name)
        generation_temperature = self._temperature if temperature is None else temperature
        prefix = self._response_prefix if response_prefix is None else response_prefix
        prompt_with_prefix = prompt_text + prefix
        inputs = self._tokenizer(prompt_with_prefix, return_tensors="pt").to("cuda")
        t0 = time.monotonic()
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self._max_new_tokens,
                do_sample=generation_temperature > 0,
                temperature=generation_temperature if generation_temperature > 0 else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = (prefix + self._tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
        return raw_text, (time.monotonic() - t0) * 1000.0

    def _render_planner_hint_prompt(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> str:
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
            {"role": "system", "content": self._planner_hint_system_prompt},
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
    ) -> str:
        self._last_planner_hint = ""
        self._last_planner_hint_raw = ""
        self._last_retrieved_knowledge = []
        if not self._planner_hint_enabled:
            self._last_planner_hint_status = "disabled"
            return ""

        cache_key = planner_hint_cache_key(state, refresh=self._planner_hint_refresh)
        if cache_key and cache_key == self._planner_hint_cache_key and self._planner_hint_cache_text:
            self.stats["planner_hint_cache_hits"] += 1
            self._last_planner_hint = self._planner_hint_cache_text
            self._last_retrieved_knowledge = list(self._planner_hint_cache_knowledge)
            self._last_planner_hint_status = "cache_hit"
            return self._planner_hint_cache_text

        try:
            prompt_text = self._render_planner_hint_prompt(state, legal_actions)
            raw_text, _gen_ms = self._generate(
                prompt_text,
                max_new_tokens=self._planner_hint_max_new_tokens,
                temperature=0.0,
                adapter_name=self._planner_hint_adapter_name,
                response_prefix="{",
            )
            hint_payload, status = parse_planner_hint_json(raw_text)
        except Exception as exc:
            self.stats["planner_hint_failures"] += 1
            self._last_planner_hint_status = f"exception:{type(exc).__name__}"
            if self._planner_hint_required:
                raise
            return ""

        self.stats["planner_hint_calls"] += 1
        self._last_planner_hint_raw = raw_text
        self._last_planner_hint_status = status
        if status != "ok" or hint_payload is None:
            self.stats["planner_hint_failures"] += 1
            if self._planner_hint_required:
                raise RuntimeError(f"planner_hint_invalid:{status}")
            return ""

        hint_text = format_planner_hint(hint_payload)
        if not hint_text:
            self.stats["planner_hint_failures"] += 1
            self._last_planner_hint_status = "empty_hint"
            if self._planner_hint_required:
                raise RuntimeError("planner_hint_empty")
            return ""

        self._planner_hint_cache_key = cache_key
        self._planner_hint_cache_text = hint_text
        self._planner_hint_cache_knowledge = list(self._last_retrieved_knowledge)
        self._last_planner_hint = hint_text
        return hint_text

    def _retry_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        raw_text: str,
        fallback_reason: str,
    ) -> str:
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
            'Return only one valid JSON object matching this schema: '
            '{"action_index":0,"confidence":0.0,"reason":"..."}. '
            "Do not output multiple objects, a list, or alternative candidates. "
            "No markdown, no comments, no extra text."
        )
        retry_messages = [
            *messages,
            {"role": "assistant", "content": raw_text[:240]},
            {"role": "user", "content": correction},
        ]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking:
            kwargs["enable_thinking"] = True
        return self._tokenizer.apply_chat_template(retry_messages, **kwargs)

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

    def select_action(self, state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> "PolicyDecision":
        enabled = [a for a in (legal_actions or []) if isinstance(a, dict) and a.get("is_enabled") is not False]
        enabled = _filter_optional_potion_actions_live(enabled, state, self.stats)
        if not enabled:
            return PolicyDecision(
                action_index=-1,
                reason="",
                raw_generation="",
                attempts=[],
                user_message="",
                strategy_context="",
                invalid_output=True,
                fallback_reason="no_enabled",
                gen_ms=0.0,
            )

        simple = choose_simple_action(state, enabled)
        if simple is not None:
            strategy_context = self._strategy_memory.context_text(state, enabled)
            user_msg = render_state_text(state, enabled, strategy_context=strategy_context)
            self.stats["heuristic_calls"] += 1
            return PolicyDecision(
                action_index=simple.action_index,
                reason=simple.reason,
                raw_generation="",
                attempts=[],
                user_message=user_msg,
                strategy_context=strategy_context,
                invalid_output=False,
                fallback_reason="",
                gen_ms=0.0,
                confidence=1.0,
                action_scores=[],
                score_margin=None,
                route=simple.route,
                trainable=False,
                planner_hint="",
                planner_hint_status="disabled",
                retrieved_knowledge=[],
            )

        planner_hint = self._planner_hint_for_state(state, enabled)
        strategy_context = self._strategy_memory.context_text(state, enabled, planner_hint=planner_hint)
        user_msg = render_state_text(state, enabled, strategy_context=strategy_context)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking:
            kwargs["enable_thinking"] = True
        prompt_text = self._tokenizer.apply_chat_template(messages, **kwargs)
        if self._prompt_token_count(prompt_text) > self._max_seq_length:
            return self._prompt_too_long_decision(
                prompt_text=prompt_text,
                user_msg=user_msg,
                strategy_context=strategy_context,
            )

        attempts: list[dict[str, Any]] = []
        raw_text = ""
        gen_ms = 0.0
        first_invalid = False
        for attempt in range(self._parse_retries + 1):
            if attempt > 0:
                self.stats["retry_attempts"] += 1
            raw_text, gen_ms = self._generate(prompt_text, adapter_name=self._combat_adapter_name)
            self.stats["generated_outputs"] += 1
            strict_status = _strict_json_status(raw_text)
            if strict_status == "ok":
                self.stats["strict_json_ok"] += 1
            else:
                self.stats["strict_json_failures"] += 1
            decoded = decode_action(raw_text, enabled, fallback_index=0)
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
            if self._strict_json_required and strict_status != "ok" and not decoded.used_fallback:
                decoded = DecodedAction(
                    action_index=decoded.action_index,
                    reason=decoded.reason,
                    used_fallback=True,
                    fallback_reason="strict_json_required",
                    confidence=decoded.confidence,
                    action_scores=decoded.action_scores,
                )
            attempts.append({
                "attempt": attempt,
                "gen_ms": round(gen_ms, 1),
                "raw_generation": raw_text,
                "strict_json_status": strict_status,
                "strict_json_ok": strict_status == "ok",
                "decoded": self._decoded_payload(decoded),
                "reason_quality_flags": reason_quality_flags,
            })
            if not decoded.used_fallback:
                if first_invalid:
                    self.stats["retry_recovered"] += 1
                return PolicyDecision(
                    action_index=decoded.action_index,
                    reason=decoded.reason,
                    raw_generation=raw_text,
                    attempts=attempts,
                    user_message=user_msg,
                    strategy_context=strategy_context,
                    invalid_output=False,
                    fallback_reason="",
                    gen_ms=gen_ms,
                    confidence=decoded.confidence,
                    action_scores=list(decoded.action_scores),
                    score_margin=action_score_margin(decoded.action_scores),
                    planner_hint=planner_hint,
                    planner_hint_status=self._last_planner_hint_status,
                    retrieved_knowledge=self._last_retrieved_knowledge,
                )

            self.stats["invalid_attempts"] += 1
            if attempt == 0:
                first_invalid = True
                self.stats["first_attempt_invalid"] += 1
            if decoded.fallback_reason == "parse_failed":
                self.stats["parse_failures"] += 1
            elif decoded.fallback_reason == "action_index_not_int":
                self.stats["action_index_not_int"] += 1
            elif decoded.fallback_reason == "action_index_out_of_range":
                self.stats["out_of_range"] += 1
            elif decoded.fallback_reason == "strict_json_required":
                self.stats["strict_json_rejections"] += 1
            elif any(flag in decoded.fallback_reason for flag in _EXPLANATION_CONSISTENCY_FLAGS):
                self.stats["reason_consistency_failures"] += 1
            elif any(flag in decoded.fallback_reason for flag in _SAFETY_RETRY_FLAGS):
                self.stats["safety_rejections"] += 1
            else:
                self.stats["mapping_failures"] += 1
            if attempt < self._parse_retries:
                prompt_text = self._retry_prompt(messages, raw_text=raw_text, fallback_reason=decoded.fallback_reason)
                if self._prompt_token_count(prompt_text) > self._max_seq_length:
                    return self._prompt_too_long_decision(
                        prompt_text=prompt_text,
                        user_msg=user_msg,
                        strategy_context=strategy_context,
                        attempts=attempts,
                    )

        self.stats["invalid_outputs"] += 1
        return PolicyDecision(
            action_index=-1,
            reason="",
            raw_generation=raw_text,
            attempts=attempts,
            user_message=user_msg,
            strategy_context=strategy_context,
            invalid_output=True,
            fallback_reason=attempts[-1]["decoded"]["fallback_reason"] if attempts else "unknown",
            gen_ms=gen_ms,
            confidence=None,
            action_scores=[],
            score_margin=None,
            planner_hint=planner_hint,
            planner_hint_status=self._last_planner_hint_status,
            retrieved_knowledge=self._last_retrieved_knowledge,
        )


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    action_index: int
    reason: str
    raw_generation: str
    attempts: list[dict[str, Any]]
    user_message: str
    strategy_context: str
    invalid_output: bool
    fallback_reason: str
    gen_ms: float
    confidence: float | None = None
    action_scores: list[dict[str, Any]] = field(default_factory=list)
    score_margin: float | None = None
    route: str = "llm"
    trainable: bool = True
    planner_hint: str = ""
    planner_hint_status: str = "disabled"
    retrieved_knowledge: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepRecord:
    state: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    chosen_index: int
    reason: str
    raw_generation: str
    attempts: list[dict[str, Any]]
    messages: list[dict[str, Any]]  # system + user + assistant (text)
    quality_flags: list[str] = field(default_factory=list)
    quality_report: dict[str, Any] = field(default_factory=dict)
    settlement_events: list[dict[str, Any]] = field(default_factory=list)
    invalid_output: bool = False
    fallback_reason: str = ""
    trainable: bool = True
    confidence: float | None = None
    action_scores: list[dict[str, Any]] = field(default_factory=list)
    score_margin: float | None = None
    route: str = "llm"
    planner_hint: str = ""
    planner_hint_status: str = "disabled"
    retrieved_knowledge: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EpisodeRecord:
    encounter_key: str
    encounter_id: str
    encounter_tag: str
    encounter_label: str
    seed: str
    steps: list[StepRecord]
    outcome: str
    final_state: dict[str, Any]
    reward: dict[str, float]
    duration_s: float
    invalid_output: bool = False
    invalid_reason: str = ""
    quality_flags: dict[str, int] = field(default_factory=dict)
    quality_summary: dict[str, Any] = field(default_factory=dict)
    case_metadata: dict[str, Any] = field(default_factory=dict)
    early_exit_diagnostics: dict[str, Any] = field(default_factory=dict)


def _step_user_message(step: StepRecord) -> str:
    for message in step.messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _chosen_action_payload(step: StepRecord) -> dict[str, Any] | None:
    if step.chosen_index < 0 or step.chosen_index >= len(step.legal_actions):
        return None
    chosen = dict(step.legal_actions[step.chosen_index])
    return {
        key: chosen.get(key)
        for key in ("action", "type", "card_id", "card_index", "target_id", "hand_index")
        if key in chosen
    }


def episode_step_trace_rows(ep: EpisodeRecord) -> list[dict[str, Any]]:
    """Return replay_trace-compatible rows for each decision in one episode."""
    episode_id = f"{ep.encounter_key}::{ep.seed}"
    rows: list[dict[str, Any]] = []
    for step_idx, step in enumerate(ep.steps):
        final_attempt = step.attempts[-1] if step.attempts else {}
        rows.append({
            "episode_id": episode_id,
            "episode_step": step_idx,
            "step": step_idx,
            "encounter_key": ep.encounter_key,
            "encounter_id": ep.encounter_id,
            "encounter_tag": ep.encounter_tag,
            "encounter_label": ep.encounter_label,
            "seed": ep.seed,
            "outcome": ep.outcome,
            "episode_reward": ep.reward,
            "case_metadata": ep.case_metadata,
            "route": step.route,
            "action_mode": "index",
            "gen_ms": float(final_attempt.get("gen_ms") or 0.0),
            "user_message": _step_user_message(step),
            "planner_hint": step.planner_hint,
            "planner_hint_status": step.planner_hint_status,
            "retrieved_knowledge": step.retrieved_knowledge,
            "state": step.state,
            "legal_actions": step.legal_actions,
            "raw_generation": step.raw_generation,
            "attempts": step.attempts,
            "invalid_output": bool(step.invalid_output),
            "decoded": {
                "action_index": step.chosen_index,
                "reason": step.reason,
                "confidence": step.confidence,
                "action_scores": step.action_scores,
                "score_margin": step.score_margin,
                "used_fallback": bool(step.invalid_output),
                "fallback_reason": step.fallback_reason if step.invalid_output else "",
            },
            "chosen_action": _chosen_action_payload(step),
            "settlement_events": step.settlement_events,
            "enabled_count": len(step.legal_actions),
            "quality_flags": step.quality_flags,
            "quality_report": step.quality_report,
        })
    return rows


def episode_trace_summary(ep: EpisodeRecord) -> dict[str, Any]:
    return {
        "episode_id": f"{ep.encounter_key}::{ep.seed}",
        "encounter_key": ep.encounter_key,
        "encounter_id": ep.encounter_id,
        "encounter_tag": ep.encounter_tag,
        "encounter_label": ep.encounter_label,
        "seed": ep.seed,
        "outcome": ep.outcome,
        "steps": len(ep.steps),
        "duration_s": ep.duration_s,
        "reward": ep.reward,
        "case_metadata": ep.case_metadata,
        "invalid_output": ep.invalid_output,
        "invalid_reason": ep.invalid_reason,
        "quality_flags": ep.quality_flags,
        "quality_summary": ep.quality_summary,
        # left_combat / no_legal_actions 这种"非死非赢"退出时的诊断信息
        # （sim 状态、退出 step、敌方残血等），方便事后定位 sim bug 或 model 误用 action
        "early_exit_diagnostics": ep.early_exit_diagnostics or None,
        "action_sequence": [
            {
                "step": idx,
                "action_index": step.chosen_index,
                "reason": step.reason,
                "confidence": step.confidence,
                "action_scores": step.action_scores,
                "score_margin": step.score_margin,
                "chosen_action": _chosen_action_payload(step),
                "planner_hint_status": step.planner_hint_status,
                "settlement_events": step.settlement_events,
                "quality_flags": step.quality_flags,
            }
            for idx, step in enumerate(ep.steps)
        ],
    }


def append_episode_trace_files(*, step_trace_path: Path, episode_trace_path: Path, ep: EpisodeRecord) -> None:
    with step_trace_path.open("a", encoding="utf-8") as handle:
        for row in episode_step_trace_rows(ep):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with episode_trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode_trace_summary(ep), ensure_ascii=False) + "\n")


def _enabled_actions(legal: list[Any]) -> list[dict[str, Any]]:
    return [dict(a) for a in (legal or []) if isinstance(a, dict) and a.get("is_enabled") is not False]


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _spec_act(spec: EncounterSpec) -> int | None:
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    value = _positive_int(metadata.get("act"))
    if value is not None:
        return value
    if isinstance(spec.build, dict):
        value = _positive_int(spec.build.get("act"))
        if value is not None:
            return value
    match = re.search(r"\bact(\d+)\b", spec.tag or "", re.IGNORECASE)
    if match:
        return _positive_int(match.group(1))
    return None


def _spec_floor(spec: EncounterSpec) -> int | None:
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    value = _positive_int(metadata.get("floor"))
    if value is not None:
        return value
    if isinstance(spec.build, dict):
        value = _positive_int(spec.build.get("floor"))
        if value is not None:
            return value
    return None


def _inject_spec_context(state: dict[str, Any], spec: EncounterSpec, *, seed: str = "") -> dict[str, Any]:
    """Attach Skada case context that combat_reset cannot carry through proto."""
    if not isinstance(state, dict):
        return state
    state = dict(state)
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    act = _spec_act(spec)
    floor = _spec_floor(spec)
    stable_encounter_key = encounter_key(spec)
    stable_combat_key = f"{stable_encounter_key}::{seed}" if seed else stable_encounter_key

    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    run = dict(run)
    if act is not None:
        run["act"] = act
    elif _positive_int(run.get("act")) is None:
        run["act"] = "?"
    if floor is not None:
        run["floor"] = floor
        run["floor_reached"] = floor
    elif _positive_int(run.get("floor_reached", run.get("floor"))) is None:
        run["floor"] = "?"
        run["floor_reached"] = "?"
    if seed:
        run["seed"] = seed
    run["encounter_id"] = spec.encounter_id
    run["encounter_key"] = stable_encounter_key
    state["run"] = run

    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    battle = dict(battle)
    battle["encounter_id"] = spec.encounter_id
    battle["encounter_key"] = stable_encounter_key
    battle["encounter_label"] = encounter_label(spec)
    battle["combat_key"] = stable_combat_key
    state["battle"] = battle
    state["encounter_id"] = spec.encounter_id
    state["encounter_key"] = stable_encounter_key
    state["encounter_label"] = encounter_label(spec)
    state["combat_key"] = stable_combat_key
    state["_case_metadata"] = {
        "case_id": metadata.get("case_id"),
        "source": metadata.get("source") or "skada_case",
        "act": act,
        "floor": floor,
        "encounter_id": spec.encounter_id,
        "encounter_key": stable_encounter_key,
        "encounter_label": encounter_label(spec),
        "encounter_tag": spec.tag,
        "encounter_type": metadata.get("encounter_type"),
        "won": metadata.get("won"),
        "floor_state": metadata.get("floor_state"),
        "case_metadata": metadata.get("case_metadata"),
    }
    return state


def rollout_episode(
    policy: _RolloutPolicy,
    spec: EncounterSpec,
    *,
    seed: str,
    max_steps: int,
    port: int,
) -> EpisodeRecord:
    session = create_game_session(mode="combat", transport="pipe_proto", backend="sim", port=port, auto_launch=True)
    t0 = time.monotonic()
    steps: list[StepRecord] = []
    outcome = "unknown"
    state: dict[str, Any] = {}
    initial_state: dict[str, Any] = {}

    player_hp_start = 0.0
    damage_dealt = 0.0
    damage_taken = 0.0
    num_turns = 0

    try:
        policy.reset_episode()
        try:
            state = session.reset(character_id="IRONCLAD", encounter_id=spec.encounter_id, build=spec.build, seed=seed)
            state = _inject_spec_context(state, spec, seed=seed)
            initial_state = dict(state)
        except Exception as exc:
            outcome = f"reset_failed:{type(exc).__name__}"
            return EpisodeRecord(
                encounter_key=encounter_key(spec),
                encounter_id=spec.encounter_id,
                encounter_tag=spec.tag,
                encounter_label=encounter_label(spec),
                seed=seed,
                steps=[],
                outcome=outcome,
                final_state={},
                reward=compute_episode_reward(
                    outcome,
                    player_hp_start=0.0,
                    player_hp_end=0.0,
                    player_max_hp=80.0,
                    num_turns=0,
                    damage_dealt=0.0,
                    damage_taken=0.0,
                ),
                duration_s=round(time.monotonic() - t0, 2),
                invalid_output=False,
                invalid_reason="",
                quality_flags={},
                quality_summary={},
                case_metadata=dict(spec.metadata) if isinstance(spec.metadata, dict) else {},
            )
        player_hp_start = float(_player_hp(state) or 0.0)

        # 把诊断信息写入 episode 元数据，供下游 audit / Kimi review 看清退出原因。
        # left_combat / no_legal_actions 这种"非死非赢"退出之前没法解释；现在 dump
        # state_type / run_outcome / 最后一步的 enemies 状态等，让分析有据可查。
        early_exit_diagnostics: dict[str, Any] = {}

        def _capture_early_exit(reason: str, state_obj: dict[str, Any] | None) -> None:
            try:
                state_obj = state_obj or {}
                player_obj = state_obj.get("player") if isinstance(state_obj.get("player"), dict) else {}
                battle_obj = state_obj.get("battle") if isinstance(state_obj.get("battle"), dict) else {}
                enemies = []
                for en in (state_obj.get("enemies") or []):
                    if not isinstance(en, dict):
                        continue
                    enemies.append({
                        "id": en.get("id"),
                        "monster_id": en.get("monster_id"),
                        "hp": en.get("hp"),
                        "is_alive": en.get("is_alive"),
                    })
                # Also snapshot legal_actions on the early-exit state. Without
                # this, ``left_combat`` traces only tell us "rollout aborted at
                # state_type=X" but not whether sim was waiting for a specific
                # action (select_hand_card, confirm_selection, …) we should
                # have honoured. With the snapshot we can grep the trace and
                # confirm e.g. card_select states actually carried valid
                # legal_actions but the loop bailed — i.e. real bridge bugs
                # vs. genuinely empty action sets.
                legal_actions_snapshot: list[dict[str, Any]] = []
                for la in (state_obj.get("legal_actions") or [])[:20]:
                    if isinstance(la, dict):
                        legal_actions_snapshot.append({
                            "index": la.get("index"),
                            "action": la.get("action") or la.get("type"),
                            "card_id": la.get("card_id"),
                            "target_id": la.get("target_id"),
                            "is_enabled": la.get("is_enabled"),
                        })
                fallback_reason = state_obj.get("legal_actions_fallback_reason") or ""
                early_exit_diagnostics.update({
                    "reason": reason,
                    "state_type": state_obj.get("state_type"),
                    "terminal": state_obj.get("terminal"),
                    "run_outcome": state_obj.get("run_outcome"),
                    "combat_outcome": state_obj.get("combat_outcome"),
                    "battle_state_type": battle_obj.get("state_type") if isinstance(battle_obj, dict) else None,
                    "player_hp": player_obj.get("hp") if isinstance(player_obj, dict) else None,
                    "enemies_snapshot": enemies,
                    "legal_actions_snapshot": legal_actions_snapshot,
                    "legal_actions_fallback_reason": fallback_reason,
                    "step_idx": step_idx,
                })
            except Exception:
                pass

        for step_idx in range(max_steps):
            if not is_combat_state(state):
                outcome = "left_combat"
                _capture_early_exit("left_combat", state)
                break
            legal_enabled = _enabled_actions(state.get("legal_actions") or [])
            if not legal_enabled:
                outcome = "no_legal_actions"
                _capture_early_exit("no_legal_actions", state)
                break

            if is_actionable_combat_state(state):
                # 记录回合数（简单 heuristic：检测到 end_turn 就加一）
                # NOTE: chosen_index=-1 表示 invalid_output；Python 负索引会取末尾元素，
                # 必须先判 >=0 才能用作下标。
                if step_idx > 0 and 0 <= steps[-1].chosen_index < len(steps[-1].legal_actions):
                    last_action_type = str(steps[-1].legal_actions[steps[-1].chosen_index].get("action", "") or steps[-1].legal_actions[steps[-1].chosen_index].get("type", "")).lower()
                    if "end_turn" in last_action_type:
                        num_turns += 1

                decision = policy.select_action(state, legal_enabled)
                if decision.invalid_output:
                    user_msg = decision.user_message or render_state_text(state, legal_enabled)
                    assistant_msg = json.dumps(
                        {
                            "action_index": -1,
                            "confidence": decision.confidence,
                            "reason": decision.fallback_reason or "invalid output",
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    steps.append(StepRecord(
                        # NOTE: dict()/[dict(a)..] 是浅拷贝，sim 后续改 nested 字段会污染历史步状态；
                        # 用 deepcopy 隔离，避免 hp_lost / enemy_progress 计算被改写状态污染。
                        state=copy.deepcopy(state),
                        legal_actions=[copy.deepcopy(a) for a in legal_enabled],
                        chosen_index=-1,
                        reason=decision.reason,
                        raw_generation=decision.raw_generation,
                        attempts=decision.attempts,
                        quality_flags=[f"invalid_output:{decision.fallback_reason or 'unknown'}"],
                        quality_report={},
                        messages=[
                            {"role": "system", "content": policy._system_prompt},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                        invalid_output=True,
                        fallback_reason=decision.fallback_reason or "unknown",
                        trainable=False,
                        confidence=decision.confidence,
                        action_scores=decision.action_scores,
                        score_margin=decision.score_margin,
                        route=decision.route,
                        planner_hint=decision.planner_hint,
                        planner_hint_status=decision.planner_hint_status,
                        retrieved_knowledge=decision.retrieved_knowledge,
                    ))
                    outcome = f"invalid_output:{decision.fallback_reason or 'unknown'}"
                    break
                user_msg = decision.user_message or render_state_text(state, legal_enabled)
                assistant_msg = json.dumps(
                    {
                        "action_index": decision.action_index,
                        "confidence": decision.confidence,
                        "reason": decision.reason[:200],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                quality_report = assess_action_quality_report(
                    state,
                    legal_enabled,
                    decision.action_index,
                    reason=decision.reason,
                    action_scores=decision.action_scores,
                ).as_dict()
                quality_flags = list(quality_report.get("flags") or [])
                trainable = decision.trainable and not any(flag in _EXPLANATION_CONSISTENCY_FLAGS for flag in quality_flags)
                steps.append(StepRecord(
                    # NOTE: 浅拷贝会让 sim 后续 step 改写 hp/enemies 时反过来污染历史步；
                    # 必须 deepcopy 把每一步状态独立化。
                    state=copy.deepcopy(state),
                    legal_actions=[copy.deepcopy(a) for a in legal_enabled],
                    chosen_index=decision.action_index,
                    reason=decision.reason,
                    raw_generation=decision.raw_generation,
                    attempts=decision.attempts,
                    quality_flags=quality_flags,
                    quality_report=quality_report,
                    messages=[
                        {"role": "system", "content": policy._system_prompt},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                    trainable=trainable,
                    confidence=decision.confidence,
                    action_scores=decision.action_scores,
                    score_margin=decision.score_margin,
                    route=decision.route,
                    planner_hint=decision.planner_hint,
                    planner_hint_status=decision.planner_hint_status,
                    retrieved_knowledge=decision.retrieved_knowledge,
                ))
                chosen_raw = legal_enabled[decision.action_index]
                policy.record_action(chosen_raw)
            else:
                chosen_raw = legal_enabled[0]

            try:
                step_result = session.act_gym(chosen_raw)
                if isinstance(step_result, tuple) and len(step_result) >= 3:
                    state, _reward, done, _info = step_result[0], step_result[1], step_result[2], step_result[3] if len(step_result) > 3 else {}
                else:
                    state = step_result
                    done = False
                    _info = {}
                state = _inject_spec_context(state, spec, seed=seed)
                if steps:
                    events = _info.get("settlement_events") if isinstance(_info, dict) else None
                    if events is None:
                        events = getattr(session, "last_settlement_events", [])
                    if isinstance(events, list):
                        steps[-1].settlement_events = [dict(item) for item in events if isinstance(item, dict)]
            except Exception as exc:
                outcome = f"step_failed:{type(exc).__name__}"
                break

            if done:
                outcome = str(state.get("run_outcome") or state.get("combat_outcome") or "terminal")
                break
        else:
            outcome = "max_steps"
    finally:
        try:
            session.close()
        except Exception:
            pass

    # NOTE: sim 在战斗结束时（玩家死、max_steps、left_combat、协议错误等）有可能：
    #   (a) 把 player.hp 反弹到初始值或满血；
    #   (b) 清空 enemies 列表（"战斗已结束"语义而非"敌人都死了"）。
    # 这两种 reset 会让 quality_summary.hp_lost=0、enemy_damage_progress=1.0、enemy_defeated=full
    # 全部错出来。仅依赖 outcome 字符串判断不稳，所以做多重 trust 检查：
    #   1) outcome 在 protocol_failure 集合内 → 不可信。
    #   2) sim final player.hp 反而大于最后决策步的 player.hp（典型 reset 信号）→ 不可信。
    #   3) sim 没敌人但最后决策步还有可观敌方血量 → enemies 被 wipe，不可信。
    last_step_state = steps[-1].state if steps else None
    sim_player_hp_raw = _player_hp(state) if isinstance(state, dict) else None
    last_step_player_hp_raw = _player_hp(last_step_state) if isinstance(last_step_state, dict) else None
    final_state_is_trusted = True
    if (
        outcome.startswith("invalid_output:")
        or outcome.startswith("step_failed:")
        or outcome.startswith("reset_failed:")
        or outcome in {"left_combat", "no_legal_actions", "max_steps"}
    ):
        final_state_is_trusted = False
    if (
        isinstance(sim_player_hp_raw, (int, float))
        and isinstance(last_step_player_hp_raw, (int, float))
        and float(sim_player_hp_raw) > float(last_step_player_hp_raw) + 1e-3
    ):
        final_state_is_trusted = False
    if isinstance(state, dict) and isinstance(last_step_state, dict):
        sim_total = sum(_enemy_hp_snapshot(state).values())
        last_total = sum(_enemy_hp_snapshot(last_step_state).values())
        # 阈值 5：避免把残血敌人合法死亡误判为 wipe；超过 5 hp 的敌方残血 → enemies 被清空。
        if sim_total <= 0.0 and last_total > 5.0:
            final_state_is_trusted = False
    final_state_for_metrics: dict[str, Any]
    if final_state_is_trusted or not isinstance(last_step_state, dict):
        final_state_for_metrics = state
    else:
        final_state_for_metrics = last_step_state

    quality_summary = summarize_quality_reports(steps, final_state=final_state_for_metrics)
    hp_lost_metric = quality_summary.get("hp_lost")
    # player_hp_end 单一来源：从 final_state_for_metrics 读；缺失才回退 hp_lost_metric。
    # 不再让 fallback 与 hp_lost_metric 互相反推（避免 hp_lost 双计入 reward）。
    player_hp_end_raw = _player_hp(final_state_for_metrics)
    if isinstance(player_hp_end_raw, (int, float)):
        player_hp_end = float(player_hp_end_raw)
    elif isinstance(hp_lost_metric, (int, float)):
        player_hp_end = max(0.0, player_hp_start - float(hp_lost_metric))
    else:
        player_hp_end = 0.0
    if isinstance(hp_lost_metric, (int, float)):
        damage_taken = max(0.0, float(hp_lost_metric))
    else:
        damage_taken = max(0.0, player_hp_start - player_hp_end)
    progress_metrics = compute_enemy_damage_progress(
        initial_state or (steps[0].state if steps else {}),
        final_state_for_metrics,
    )
    damage_dealt = float(progress_metrics.get("enemy_damage_dealt") or 0.0)

    reward = compute_episode_reward(
        outcome,
        player_hp_start=player_hp_start,
        player_hp_end=player_hp_end,
        player_max_hp=80.0,
        num_turns=num_turns,
        damage_dealt=damage_dealt,
        damage_taken=damage_taken,
        enemy_damage_progress=float(progress_metrics.get("enemy_damage_progress") or 0.0),
    )
    duration = time.monotonic() - t0
    quality_summary.update(progress_metrics)

    return EpisodeRecord(
        encounter_key=encounter_key(spec),
        encounter_id=spec.encounter_id,
        encounter_tag=spec.tag,
        encounter_label=encounter_label(spec),
        seed=seed,
        steps=steps,
        outcome=outcome,
        final_state=dict(state),
        reward=reward,
        duration_s=round(duration, 2),
        invalid_output=outcome.startswith("invalid_output:"),
        invalid_reason=outcome.split(":", 1)[1] if outcome.startswith("invalid_output:") else "",
        quality_flags=count_quality_flags(steps),
        quality_summary=quality_summary,
        case_metadata=dict(spec.metadata) if isinstance(spec.metadata, dict) else {},
        early_exit_diagnostics=dict(early_exit_diagnostics) if early_exit_diagnostics else {},
    )


# ---------------------------------------------------------------------------
# 主流程：按 encounter 分组 rollout -> 计算 advantage -> 输出训练数据
# ---------------------------------------------------------------------------

def _maybe_mask_reason_in_assistant(messages: list[dict[str, Any]], mask: bool) -> list[dict[str, Any]]:
    """如果 mask 打开，把 assistant 段的 JSON reason 字段强制清空。"""
    if not mask:
        return messages
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        if msg.get("role") != "assistant":
            out.append(dict(msg))
            continue
        content = str(msg.get("content") or "")
        try:
            parsed = json.loads(content)
        except Exception:
            out.append(dict(msg))
            continue
        if isinstance(parsed, dict) and "reason" in parsed:
            parsed["reason"] = ""
            new_content = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            out.append({**msg, "content": new_content})
        else:
            out.append(dict(msg))
    return out


def _build_train_eval_rows(
    grouped: dict[str, list[EpisodeRecord]],
    rng: random.Random,
    *,
    mask_reason: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构建 GRPO 训练 + eval 行（含 relative advantage）。

    抽出来作为函数，让 main() 在 ``--eval-only`` 模式下完全跳过此路径，
    避免污染候选评估的输出文件夹。
    """
    train_rows: list[dict[str, Any]] = []
    for enc_id, eps in grouped.items():
        rewards = [ep.reward["total"] for ep in eps]
        mean_r = sum(rewards) / len(rewards)
        raw_std_r = (sum((r - mean_r) ** 2 for r in rewards) / max(1, len(rewards))) ** 0.5
        std_r = max(raw_std_r, 1e-6)
        use_absolute_advantage = len(eps) < 2 or raw_std_r < 1e-6
        for ep in eps:
            if use_absolute_advantage:
                # With one rollout per exact Skada reset case, relative advantage is
                # always zero. Use the shaped episode reward so victories remain
                # trainable and failed/invalid combats are filtered by GRPO-lite.
                advantage = ep.reward["total"]
                advantage_mode = "absolute_reward"
            else:
                advantage = (ep.reward["total"] - mean_r) / std_r
                advantage_mode = "relative_group_zscore"
            for step in ep.steps:
                if not step.trainable:
                    continue
                masked_messages = _maybe_mask_reason_in_assistant(step.messages, mask_reason)
                train_rows.append({
                    "messages": masked_messages,
                    "meta": {
                        "encounter_id": ep.encounter_id,
                        "encounter_key": ep.encounter_key,
                        "encounter_tag": ep.encounter_tag,
                        "encounter_label": ep.encounter_label,
                        "case_metadata": ep.case_metadata,
                        "case_id": ep.case_metadata.get("case_id") if isinstance(ep.case_metadata, dict) else None,
                        "floor": ep.case_metadata.get("floor") if isinstance(ep.case_metadata, dict) else None,
                        "encounter_type": ep.case_metadata.get("encounter_type") if isinstance(ep.case_metadata, dict) else None,
                        "outcome": ep.outcome,
                        "episode_reward": ep.reward["total"],
                        "advantage": round(advantage, 4),
                        "advantage_mode": advantage_mode,
                        "advantage_group_size": len(eps),
                        "advantage_group_reward_mean": round(mean_r, 4),
                        "advantage_group_reward_std": round(raw_std_r, 4),
                        "episode_quality_summary": ep.quality_summary,
                        "hp_lost": ep.quality_summary.get("hp_lost"),
                        "enemy_damage_dealt": ep.quality_summary.get("enemy_damage_dealt"),
                        "enemy_damage_progress": ep.quality_summary.get("enemy_damage_progress"),
                        "enemy_hp_start": ep.quality_summary.get("enemy_hp_start"),
                        "enemy_hp_end": ep.quality_summary.get("enemy_hp_end"),
                        "policy_generated_attempts": sum(len(step.attempts) for step in ep.steps),
                        "policy_invalid_output": ep.invalid_output,
                        "action_quality_flags": step.quality_flags,
                        "action_quality_report": step.quality_report,
                    },
                })

    rng.shuffle(train_rows)
    eval_n = max(1, int(len(train_rows) * 0.05)) if len(train_rows) > 20 else 0
    eval_rows = train_rows[:eval_n]
    train_rows_remain = train_rows[eval_n:]
    return train_rows_remain, eval_rows


def _classify_tier(ep: EpisodeRecord) -> str:
    """按 encounter_id 粗分 normal/elite/boss，对应 sampling 同一套口径。"""
    eid = (ep.encounter_id or "").upper()
    if "BOSS" in eid:
        return "boss"
    if "ELITE" in eid:
        return "elite"
    return "normal"


def _number_stats(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "avg": round(float(sum(ordered)) / len(ordered), 4),
        "max": ordered[-1],
    }


def _episodes_to_eval_metrics(eps: list[EpisodeRecord]) -> dict[str, Any]:
    """聚合 episode 列表为 policy_eval-compatible metrics。

    用同一份口径让 self_iterate 的 promotion gate 既能读 grpo_rollout（eval-only）
    输出，也能读现有 policy_eval 输出，无需在 gate 端 case-by-case 区分。
    """
    if not eps:
        return {"episodes": 0}
    outcomes = Counter(str(ep.outcome) for ep in eps)
    rewards = [float((ep.reward or {}).get("total") or 0.0) for ep in eps]
    steps = [len(ep.steps) for ep in eps]
    invalid = sum(1 for ep in eps if ep.invalid_output)
    quality: Counter = Counter()
    mech: list[float] = []
    hp_lost: list[float] = []
    enemy_progress: list[float] = []
    seq_score: list[float] = []
    def_score: list[float] = []
    turns: list[float] = []
    spt: list[float] = []
    vds: list[float] = []
    for ep in eps:
        quality.update(ep.quality_flags or {})
        s = ep.quality_summary or {}
        for src, dest in (
            ("mechanism_score", mech),
            ("hp_lost", hp_lost),
            ("enemy_damage_progress", enemy_progress),
            ("sequence_score", seq_score),
            ("defense_score", def_score),
            ("turns", turns),
            ("steps_per_turn", spt),
            ("visible_damage_per_step", vds),
        ):
            if isinstance(s.get(src), (int, float)):
                dest.append(float(s[src]))
    total = len(eps)
    victories = outcomes.get("victory", 0)
    return {
        "episodes": total,
        "victories": victories,
        "win_rate": round(victories / total, 4) if total else None,
        "invalid_output_episodes": invalid,
        "invalid_output_episode_rate": round(invalid / total, 4) if total else None,
        "outcome_counts": {k: int(v) for k, v in outcomes.most_common()},
        "action_quality": {k: int(v) for k, v in quality.most_common()},
        "reward": _number_stats(rewards),
        "steps": _number_stats(steps),
        "mechanism_score": _number_stats(mech),
        "sequence_score": _number_stats(seq_score),
        "defense_score": _number_stats(def_score),
        "hp_lost": _number_stats(hp_lost),
        "enemy_damage_progress": _number_stats(enemy_progress),
        "turns": _number_stats(turns),
        "steps_per_turn": _number_stats(spt),
        "visible_damage_per_step": _number_stats(vds),
    }


def build_eval_metrics(
    all_episodes: list[EpisodeRecord],
    grouped: dict[str, list[EpisodeRecord]],
) -> dict[str, Any]:
    """生成 policy_eval-compatible 的 eval_metrics.json 内容，含 by_encounter / by_tier 分层。"""
    payload = _episodes_to_eval_metrics(all_episodes)
    payload["kind"] = "rollout_eval_metrics"
    by_encounter: dict[str, dict[str, Any]] = {}
    for key, eps in grouped.items():
        sub = _episodes_to_eval_metrics(eps)
        if eps:
            sub["encounter_id"] = eps[0].encounter_id
            sub["encounter_label"] = eps[0].encounter_label
        by_encounter[key] = sub
    payload["by_encounter"] = by_encounter
    by_tier: dict[str, dict[str, Any]] = {}
    tier_groups: dict[str, list[EpisodeRecord]] = {}
    for ep in all_episodes:
        tier_groups.setdefault(_classify_tier(ep), []).append(ep)
    for tier, eps in tier_groups.items():
        by_tier[tier] = _episodes_to_eval_metrics(eps)
    payload["by_tier"] = by_tier
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", type=str, default=None, help="当前要评估的 LoRA adapter 目录；None 则用 base model")
    p.add_argument("--base-model-id", type=str, default=BASE_MODEL_ID)
    p.add_argument("--num-generations", type=int, default=8, help="每个 encounter 采样多少条独立轨迹")
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--port-base", type=int, default=15640)
    p.add_argument("--temperature", type=float, default=0.8, help="采样温度，>0 才有探索")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--planner-hint-adapter-dir", type=str, default=None, help="可选 planner-hint LoRA adapter；生成战斗级策略 hint 后注入 combat prompt")
    p.add_argument(
        "--planner-hint-refresh",
        choices=list(PLANNER_HINT_REFRESH_CHOICES),
        default=DEFAULT_PLANNER_HINT_REFRESH,
        help="planner-hint 缓存粒度：默认每回合刷新一次（turn）；combat 整场战斗只生成一次（不推荐，对 boss buff 反应慢）",
    )
    p.add_argument("--planner-hint-max-new-tokens", type=int, default=240)
    p.add_argument("--out-subdir", type=str, default="grpo_v0")
    p.add_argument("--encounter-filter", type=str, default="")
    p.add_argument(
        "--tier-filter",
        type=str,
        default="",
        help=(
            "Comma-separated subset of {normal,elite,boss}. Empty = keep all "
            "tiers. Use e.g. ``--tier-filter elite,boss`` to skip normal "
            "encounters when normals are already saturated and only hard "
            "fights add training signal."
        ),
    )
    p.add_argument(
        "--case-index",
        type=str,
        required=True,
        help="Skada single-combat cases.jsonl.",
    )
    p.add_argument("--case-character", type=str, default="IRONCLAD")
    p.add_argument("--case-floor-min", type=int, default=1)
    p.add_argument("--case-floor-max", type=int, default=17)
    p.add_argument("--case-limit", type=int, default=0)
    p.add_argument("--case-sample-seed", type=int, default=0)
    p.add_argument("--case-sample-mode", choices=["file", "random", "stratified", "diverse"], default="diverse")
    p.add_argument("--elite-oversample-ratio", type=float, default=0.3, help="强制 elite 在 sample 里至少占该比例（0=不强制，0.3=至少 30%）。源数据 elite 约 6.4%，case-limit 小时 random/stratified 容易选不到 elite。")
    p.add_argument("--boss-oversample-ratio", type=float, default=0.0, help="强制 boss 占比（boss 在 floor 17/33/48），不开就完全不训 boss。typical 0.2 = 32 case 里至少 6-7 个 boss。")
    p.add_argument("--pool-role", choices=["full", "train", "eval"], default="full", help="cases.jsonl 切分角色：train=排除 hold-out 子集，eval=只取 hold-out，full=不切（默认）。")
    p.add_argument("--hold-out-fraction", type=float, default=0.0, help="hold-out 子集占源 pool 比例（0-1）。仅当 pool-role=train/eval 时生效；取 0 等于不切分。")
    p.add_argument("--hold-out-seed", type=int, default=20260101, help="hold-out 切分 hash seed；跨轮固定才能保证 hold-out 集合稳定（不进训练数据）。")
    p.add_argument(
        "--archetype-min-count",
        type=str,
        default="",
        help="strict 强制 sample 含某 archetype 的 case 至少 N 个。格式 'multi_hit=2,aoe=2,power_build=2'。"
             "已知 archetype: multi_hit / aoe / power_build / lethal_burst / exhaust / block_engine。",
    )
    p.add_argument(
        "--mask-reason-in-train-data",
        action="store_true",
        help=(
            "训练数据 (train.jsonl) 的 assistant 段把 reason 字段强制为空字符串。"
            "防止 GRPO advantage 把 model 自己 hallucinated 的 reason（如 'Deal X damage' 套到 skill 卡）"
            "强化进 SFT loss。inference / step_trace 里 model 仍生成 reason 用于审计；"
            "只有 train.jsonl 里的 SFT 标签被清空。"
            "推荐与 build_teacher_dataset 的 use_kimi_reasons 配合使用："
            "Kimi 标签里 reason 是教师写的，进 SFT；rollout train_rows 里 reason 不被强化。"
        ),
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "评估模式：只采样 + 写 trace + 算 metrics，不写 train.jsonl/eval.jsonl，不算 GRPO advantage。"
            "用于 self_iterate 的 candidate_rollout 阶段（拿 candidate 表现指标但不污染训练数据）。"
        ),
    )
    p.add_argument("--include-lost-cases", action="store_true", help="Include Skada cases whose source run combat was not won.")
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--parse-retries", type=int, default=1,
                   help="模型输出 JSON/action_index 无效时重试次数；重试后仍无效则中止该 episode。")
    p.add_argument(
        "--allow-json-like-rollout",
        action="store_true",
        help="Rollout execution may accept decoder-recoverable JSON-like outputs, but saved training rows still use strict JSON.",
    )
    p.add_argument("--enable-thinking", action="store_true", default=True,
                   help="开启 Qwen3/3.5 thinking mode，让模型在 rollout 时做显式推理。默认开启。")
    p.add_argument("--no-thinking", action="store_true", help="禁用 thinking mode（用于对比实验）。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    out_dir = DATASETS_ROOT / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    step_trace_path = out_dir / "step_trace.jsonl"
    episode_trace_path = out_dir / "episode_trace.jsonl"
    step_trace_path.write_text("", encoding="utf-8")
    episode_trace_path.write_text("", encoding="utf-8")

    pool = load_skada_case_pool(
        args.case_index,
        character_id=args.case_character,
        floor_min=args.case_floor_min,
        floor_max=args.case_floor_max,
        won_only=not args.include_lost_cases,
        limit=max(0, int(args.case_limit)),
        sample_seed=int(args.case_sample_seed or args.seed),
        sample_mode=args.case_sample_mode,
        elite_oversample_ratio=float(args.elite_oversample_ratio or 0.0),
        boss_oversample_ratio=float(getattr(args, "boss_oversample_ratio", 0.0) or 0.0),
        pool_role=getattr(args, "pool_role", "full"),
        hold_out_fraction=float(getattr(args, "hold_out_fraction", 0.0) or 0.0),
        hold_out_seed=int(getattr(args, "hold_out_seed", 20260101) or 20260101),
        archetype_min_counts=_parse_archetype_min_count(getattr(args, "archetype_min_count", "") or ""),
    )
    pool = filter_encounter_pool(pool, args.encounter_filter)
    pool = filter_by_tier(pool, getattr(args, "tier_filter", "") or "")
    if not pool:
        raise SystemExit("no encounters matched filter")

    rng = random.Random(args.seed)
    enable_thinking = not args.no_thinking if args.no_thinking else args.enable_thinking
    policy = _RolloutPolicy(
        adapter_dir=args.adapter_dir,
        base_model_id=args.base_model_id,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        enable_thinking=enable_thinking,
        parse_retries=args.parse_retries,
        strict_json_required=not args.allow_json_like_rollout,
        planner_hint_adapter_dir=args.planner_hint_adapter_dir,
        planner_hint_refresh=args.planner_hint_refresh,
        planner_hint_max_new_tokens=args.planner_hint_max_new_tokens,
    )

    all_episodes: list[EpisodeRecord] = []
    for enc_idx, spec in enumerate(pool):
        for gen_idx in range(args.num_generations):
            ep_seed = f"{args.seed}-{enc_idx}-{gen_idx}-{rng.randint(0, 10**9)}"
            port = args.port_base + enc_idx
            print(f"[grpo-rollout] {spec.encounter_id} gen={gen_idx} seed={ep_seed}")
            ep = rollout_episode(policy, spec, seed=ep_seed, max_steps=args.max_steps, port=port)
            all_episodes.append(ep)
            append_episode_trace_files(
                step_trace_path=step_trace_path,
                episode_trace_path=episode_trace_path,
                ep=ep,
            )
            hp_lost = float((ep.quality_summary or {}).get("hp_lost") or 0.0)
            enemy_progress = float((ep.quality_summary or {}).get("enemy_damage_progress") or 0.0)
            print(
                "  -> "
                f"outcome={ep.outcome} hp_lost={hp_lost:.1f} "
                f"enemy_progress={enemy_progress:.2f} reward={ep.reward['total']:.2f} "
                f"steps={len(ep.steps)} dur={ep.duration_s:.1f}s"
            )

    # 按 encounter key 分组：训练阶段用于 advantage，评估阶段用于 by_encounter 报告。
    grouped: dict[str, list[EpisodeRecord]] = {}
    for ep in all_episodes:
        grouped.setdefault(ep.encounter_key, []).append(ep)

    # eval-only 模式：跳过 GRPO advantage 计算，不写 train.jsonl / eval.jsonl
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    if not args.eval_only:
        train_rows, eval_rows = _build_train_eval_rows(
            grouped,
            rng,
            mask_reason=bool(getattr(args, "mask_reason_in_train_data", False)),
        )

    def _dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not args.eval_only:
        # eval-only 模式不写训练数据，避免候选评估的 trace 被下游 GRPO/SFT 当训练样本拉走。
        _dump(out_dir / "train.jsonl", train_rows)
        _dump(out_dir / "eval.jsonl", eval_rows)

    # meta.json
    outcome_counts: dict[str, int] = {}
    quality_counts: Counter[str] = Counter()
    quality_scores: list[float] = []
    hp_lost_values: list[float] = []
    enemy_damage_progress_values: list[float] = []
    turns_values: list[float] = []
    for ep in all_episodes:
        outcome_counts[ep.outcome] = outcome_counts.get(ep.outcome, 0) + 1
        quality_counts.update(ep.quality_flags or {})
        summary = ep.quality_summary or {}
        if isinstance(summary.get("mechanism_score"), (int, float)):
            quality_scores.append(float(summary["mechanism_score"]))
        if isinstance(summary.get("hp_lost"), (int, float)):
            hp_lost_values.append(float(summary["hp_lost"]))
        if isinstance(summary.get("enemy_damage_progress"), (int, float)):
            enemy_damage_progress_values.append(float(summary["enemy_damage_progress"]))
        if isinstance(summary.get("turns"), (int, float)):
            turns_values.append(float(summary["turns"]))

    meta = {
        "run_id": uuid.uuid4().hex,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "adapter_dir": args.adapter_dir,
        "planner_hint_adapter_dir": args.planner_hint_adapter_dir,
        "planner_hint_refresh": args.planner_hint_refresh,
        "planner_hint_max_new_tokens": args.planner_hint_max_new_tokens,
        "base_model": args.base_model_id,
        "num_generations": args.num_generations,
        "total_episodes": len(all_episodes),
        "total_samples": len(train_rows) + len(eval_rows),
        "total_steps": sum(len(ep.steps) for ep in all_episodes),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "action_mode": "index",
        "outcomes": outcome_counts,
        "policy_stats": dict(policy.stats),
        "action_quality": {key: int(value) for key, value in quality_counts.most_common()},
        "mechanism_score_avg": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 1.0,
        "hp_lost_avg": round(sum(hp_lost_values) / len(hp_lost_values), 4) if hp_lost_values else 0.0,
        "enemy_damage_progress_avg": (
            round(sum(enemy_damage_progress_values) / len(enemy_damage_progress_values), 4)
            if enemy_damage_progress_values else 0.0
        ),
        "turns_avg": round(sum(turns_values) / len(turns_values), 4) if turns_values else 0.0,
        "encounter_ids": sorted({ep.encounter_id for ep in all_episodes}),
        "encounter_keys": list(grouped.keys()),
        "case_index": args.case_index,
        "case_character": args.case_character,
        "case_floor_min": args.case_floor_min,
        "case_floor_max": args.case_floor_max,
        "case_limit": args.case_limit,
        "case_sample_seed": int(args.case_sample_seed or args.seed),
        "case_sample_mode": args.case_sample_mode,
        "include_lost_cases": bool(args.include_lost_cases),
        "allow_json_like_rollout": bool(args.allow_json_like_rollout),
        "skada_case_count": len(pool) if args.case_index else 0,
        "step_trace_path": str(step_trace_path),
        "episode_trace_path": str(episode_trace_path),
        "discarded_samples": 0,
        "episodes": [
            {
                "encounter_key": ep.encounter_key,
                "encounter_id": ep.encounter_id,
                "encounter_tag": ep.encounter_tag,
                "encounter_label": ep.encounter_label,
                "outcome": ep.outcome,
                "steps": len(ep.steps),
                "duration_s": ep.duration_s,
                "kept_samples": len(ep.steps),
                "discarded_samples": 0,
                "reward": ep.reward,
                "invalid_output": ep.invalid_output,
                "invalid_reason": ep.invalid_reason,
                "quality_flags": ep.quality_flags,
                "quality_summary": ep.quality_summary,
                "case_metadata": ep.case_metadata,
            }
            for ep in all_episodes
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.eval_only:
        # 训练阶段才写 dataset_pool 兼容的 metrics.json
        write_json(out_dir / "metrics.json", {"kind": "grpo_rollout_dataset", **summarize_dataset_dir(out_dir)})

    # 永远写 eval_metrics.json：policy_eval-compatible schema，含 by_encounter / by_tier，
    # 让 self_iterate promotion gate 在两次 rollout 模式下直接消费。
    eval_metrics = build_eval_metrics(all_episodes, grouped)
    eval_metrics["adapter_dir"] = args.adapter_dir
    eval_metrics["planner_hint_adapter_dir"] = args.planner_hint_adapter_dir
    eval_metrics["base_model"] = args.base_model_id
    eval_metrics["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    eval_metrics["episodes_meta_path"] = str(episode_trace_path)
    write_json(out_dir / "eval_metrics.json", eval_metrics)

    print(f"\n[grpo-rollout] done. episodes={len(all_episodes)} train_steps={len(train_rows)}")
    print(f"[grpo-rollout] outcomes: {outcome_counts}")
    print(f"[grpo-rollout] output -> {out_dir}")
    if not args.eval_only:
        print(f"[grpo-rollout] metrics -> {out_dir / 'metrics.json'}")
    print(f"[grpo-rollout] eval_metrics -> {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
