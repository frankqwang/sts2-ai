"""用当前 LLM Policy 跑 rollout，产出带 reward 的 GRPO 训练数据。

与启发式 rollout 的区别：
- 不再用 heuristic_teacher.pick_action，而是调用当前 LoRA adapter 生成动作
- temperature > 0，采样多样化轨迹
- 记录最终游戏结果，计算 scalar reward
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
    ACT1_WINNABLE_POOL,
    EncounterSpec,
    encounter_key,
    encounter_label,
    filter_encounter_pool,
    load_skada_case_pool,
)
from llm.data_pipeline.action_quality import (
    assess_action_quality_report,
    count_quality_flags,
    summarize_quality_reports,
)
from llm.data_pipeline.state_renderer import render_state_text
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
) -> dict[str, float]:
    """把一个 episode 的统计数据转换成标量 reward（可拆解查看）。

    设计原则：
    1. 胜利是最大正信号（+10），失败是最大负信号（-5）。
    2. 鼓励高效战斗：血量保持高有加分，回合数过多有轻微惩罚（避免拖）。
    3. 造成伤害有微弱正反馈，受到伤害有负反馈（但不如 outcome 重要）。
    """
    bad_terminal = (
        outcome == "defeat"
        or outcome.startswith("invalid_output:")
        or outcome.startswith("step_failed:")
        or outcome in {"max_steps", "no_legal_actions"}
    )
    outcome_r = 10.0 if outcome == "victory" else (-5.0 if bad_terminal else 0.0)
    hp_ratio = max(0.0, player_hp_end) / max(1.0, player_max_hp)
    survival_r = 4.0 * hp_ratio
    efficiency_r = -0.15 * max(0, num_turns - 3)  # 3 回合内解决无惩罚
    dmg_r = 0.02 * damage_dealt
    taken_r = -0.03 * damage_taken

    total = outcome_r + survival_r + efficiency_r + dmg_r + taken_r
    return {
        "total": total,
        "outcome": outcome_r,
        "survival": survival_r,
        "efficiency": efficiency_r,
        "damage_dealt": dmg_r,
        "damage_taken": taken_r,
    }


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
        self._strategy_memory = StrategyMemory()
        self.stats: Counter[str] = Counter()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        if adapter_dir:
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
        )

    def reset_episode(self) -> None:
        self._strategy_memory.reset()

    def record_action(self, action: dict[str, Any] | None) -> None:
        self._strategy_memory.record_action(action)

    def _generate(self, prompt_text: str) -> tuple[str, float]:
        import torch

        prompt_with_prefix = prompt_text + self._response_prefix
        inputs = self._tokenizer(prompt_with_prefix, return_tensors="pt").to("cuda")
        t0 = time.monotonic()
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._temperature > 0,
                temperature=self._temperature if self._temperature > 0 else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = (self._response_prefix + self._tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
        return raw_text, (time.monotonic() - t0) * 1000.0

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

        strategy_context = self._strategy_memory.context_text(state, enabled)
        user_msg = render_state_text(state, enabled, strategy_context=strategy_context)
        simple = choose_simple_action(state, enabled)
        if simple is not None:
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
            )

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
            raw_text, gen_ms = self._generate(prompt_text)
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
        "action_sequence": [
            {
                "step": idx,
                "action_index": step.chosen_index,
                "reason": step.reason,
                "confidence": step.confidence,
                "action_scores": step.action_scores,
                "score_margin": step.score_margin,
                "chosen_action": _chosen_action_payload(step),
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


def _spec_floor(spec: EncounterSpec) -> int | None:
    raw = spec.metadata.get("floor") if isinstance(spec.metadata, dict) else None
    if raw is None and isinstance(spec.build, dict):
        raw = spec.build.get("floor")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _inject_spec_context(state: dict[str, Any], spec: EncounterSpec) -> dict[str, Any]:
    """Attach Skada case context that combat_reset cannot carry through proto."""
    if not isinstance(state, dict):
        return state
    floor = _spec_floor(spec)
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    if floor is not None:
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        run = dict(run)
        run.setdefault("act", 1)
        run["floor"] = floor
        run["floor_reached"] = floor
        state = dict(state)
        state["run"] = run
    if metadata:
        state = dict(state)
        state["_case_metadata"] = {
            "case_id": metadata.get("case_id"),
            "source": metadata.get("source"),
            "floor": metadata.get("floor"),
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

    player_hp_start = 0.0
    damage_dealt = 0.0
    damage_taken = 0.0
    num_turns = 0

    try:
        policy.reset_episode()
        try:
            state = session.reset(character_id="IRONCLAD", encounter_id=spec.encounter_id, build=spec.build, seed=seed)
            state = _inject_spec_context(state, spec)
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
        player_hp_start = float(state.get("player", {}).get("hp", 0) or state.get("battle", {}).get("player", {}).get("hp", 0) or 0)

        for step_idx in range(max_steps):
            if not is_combat_state(state):
                outcome = "left_combat"
                break
            legal_enabled = _enabled_actions(state.get("legal_actions") or [])
            if not legal_enabled:
                outcome = "no_legal_actions"
                break

            if is_actionable_combat_state(state):
                # 记录回合数（简单 heuristic：检测到 end_turn 就加一）
                if step_idx > 0 and steps[-1].chosen_index < len(steps[-1].legal_actions):
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
                        state=dict(state),
                        legal_actions=[dict(a) for a in legal_enabled],
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
                    ))
                    outcome = f"invalid_output:{decision.fallback_reason or 'unknown'}"
                    break
                user_msg = decision.user_message or render_state_text(state, legal_enabled)
                assistant_msg = json.dumps(
                    {
                        "action_index": decision.action_index,
                        "confidence": decision.confidence,
                        "reason": decision.reason[:80],
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
                    state=dict(state),
                    legal_actions=[dict(a) for a in legal_enabled],
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
                state = _inject_spec_context(state, spec)
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

    # 估算伤害统计（简化版：用起始血量 - 结束血量近似）
    player_hp_end = float(state.get("player", {}).get("hp", 0) or state.get("battle", {}).get("player", {}).get("hp", 0) or 0)
    damage_taken = max(0.0, player_hp_start - player_hp_end)
    # damage_dealt 从敌人血量差估算（粗略）
    enemies_start = steps[0].state.get("enemies", []) if steps else []
    enemies_end = state.get("enemies", [])
    for e0, e1 in zip(enemies_start, enemies_end):
        if isinstance(e0, dict) and isinstance(e1, dict):
            hp0 = float(e0.get("hp", e0.get("current_hp", 0)) or 0)
            hp1 = float(e1.get("hp", e1.get("current_hp", 0)) or 0)
            damage_dealt += max(0.0, hp0 - hp1)

    reward = compute_episode_reward(
        outcome,
        player_hp_start=player_hp_start,
        player_hp_end=player_hp_end,
        player_max_hp=80.0,
        num_turns=num_turns,
        damage_dealt=damage_dealt,
        damage_taken=damage_taken,
    )
    duration = time.monotonic() - t0

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
        quality_summary=summarize_quality_reports(steps, final_state=state),
        case_metadata=dict(spec.metadata) if isinstance(spec.metadata, dict) else {},
    )


# ---------------------------------------------------------------------------
# 主流程：按 encounter 分组 rollout -> 计算 advantage -> 输出训练数据
# ---------------------------------------------------------------------------

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
    p.add_argument("--out-subdir", type=str, default="grpo_v0")
    p.add_argument("--encounter-filter", type=str, default="")
    p.add_argument(
        "--case-index",
        type=str,
        default="",
        help="Skada single-combat cases.jsonl. When set, rollout uses real Skada combat reset cases instead of the handcrafted pool.",
    )
    p.add_argument("--case-character", type=str, default="IRONCLAD")
    p.add_argument("--case-floor-min", type=int, default=1)
    p.add_argument("--case-floor-max", type=int, default=17)
    p.add_argument("--case-limit", type=int, default=0)
    p.add_argument("--case-sample-seed", type=int, default=0)
    p.add_argument("--case-sample-mode", choices=["file", "random", "stratified"], default="stratified")
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

    if args.case_index:
        pool = load_skada_case_pool(
            args.case_index,
            character_id=args.case_character,
            floor_min=args.case_floor_min,
            floor_max=args.case_floor_max,
            won_only=not args.include_lost_cases,
            limit=max(0, int(args.case_limit)),
            sample_seed=int(args.case_sample_seed or args.seed),
            sample_mode=args.case_sample_mode,
        )
    else:
        pool = ACT1_WINNABLE_POOL
    pool = filter_encounter_pool(pool, args.encounter_filter)
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
            print(f"  -> outcome={ep.outcome} reward={ep.reward['total']:.2f} steps={len(ep.steps)} dur={ep.duration_s:.1f}s")

    # 按 encounter key 分组，计算 relative advantage。同一敌人不同 build
    # 不能混在一起比较 reward，否则会污染训练信号。
    grouped: dict[str, list[EpisodeRecord]] = {}
    for ep in all_episodes:
        grouped.setdefault(ep.encounter_key, []).append(ep)

    train_rows: list[dict[str, Any]] = []
    for enc_id, eps in grouped.items():
        rewards = [ep.reward["total"] for ep in eps]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / max(1, len(rewards))) ** 0.5
        std_r = max(std_r, 1e-6)
        for ep in eps:
            advantage = (ep.reward["total"] - mean_r) / std_r
            # 把 episode 中每个 step 都展开成一条训练样本
            for step in ep.steps:
                if not step.trainable:
                    continue
                train_rows.append({
                    "messages": step.messages,
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
                        "policy_generated_attempts": sum(len(step.attempts) for step in ep.steps),
                        "policy_invalid_output": ep.invalid_output,
                        "action_quality_flags": step.quality_flags,
                        "action_quality_report": step.quality_report,
                    },
                })

    # shuffle + 输出
    rng.shuffle(train_rows)
    eval_n = max(1, int(len(train_rows) * 0.05)) if len(train_rows) > 20 else 0
    eval_rows = train_rows[:eval_n]
    train_rows = train_rows[eval_n:]

    def _dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _dump(out_dir / "train.jsonl", train_rows)
    _dump(out_dir / "eval.jsonl", eval_rows)

    # meta.json
    outcome_counts: dict[str, int] = {}
    quality_counts: Counter[str] = Counter()
    quality_scores: list[float] = []
    hp_lost_values: list[float] = []
    turns_values: list[float] = []
    for ep in all_episodes:
        outcome_counts[ep.outcome] = outcome_counts.get(ep.outcome, 0) + 1
        quality_counts.update(ep.quality_flags or {})
        summary = ep.quality_summary or {}
        if isinstance(summary.get("mechanism_score"), (int, float)):
            quality_scores.append(float(summary["mechanism_score"]))
        if isinstance(summary.get("hp_lost"), (int, float)):
            hp_lost_values.append(float(summary["hp_lost"]))
        if isinstance(summary.get("turns"), (int, float)):
            turns_values.append(float(summary["turns"]))

    meta = {
        "run_id": uuid.uuid4().hex,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "adapter_dir": args.adapter_dir,
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
    write_json(out_dir / "metrics.json", {"kind": "grpo_rollout_dataset", **summarize_dataset_dir(out_dir)})

    print(f"\n[grpo-rollout] done. episodes={len(all_episodes)} train_steps={len(train_rows)}")
    print(f"[grpo-rollout] outcomes: {outcome_counts}")
    print(f"[grpo-rollout] output -> {out_dir}")
    print(f"[grpo-rollout] metrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
