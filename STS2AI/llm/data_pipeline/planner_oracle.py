"""Real-time planner oracle: invoke a strong external teacher (DeepSeek-V4-Pro
by default) during rollout to (a) provide a high-quality planner_hint at
key moments, and (b) generate paired (LoRA-planner, oracle-planner)
samples that downstream training can use to align the local planner LoRA.

Design notes
============
- This is **not** a replacement for the planner LoRA. The LoRA still runs
  every turn (cheap, on-device). The oracle is a budgeted side-call that
  fires at moments the LoRA is most likely to mis-react: phase changes,
  early-game plan setup, low-HP danger windows, novel power signatures,
  or explicit caller hints. We capture the oracle hint into the trace so
  downstream tools can mine for "LoRA mismatch with oracle" cases.
- The oracle never picks an action. It outputs the same planner_hint
  schema as the LoRA. The combat policy only ever sees one canonical
  planner_hint per turn — `replace_hint=True` swaps the LoRA output for
  the oracle output during rollout, otherwise the LoRA hint is kept and
  the oracle hint lives only in trace metadata.
- API call layer reuses `kimi_review_turn_order.call_openai_chat`
  (OpenAI-compatible POST). DeepSeek / Kimi / any compatible server share
  the same wire protocol; only the base_url + api_key differ.
- Budget enforcement is per-episode + per-rollout (env-level); both layers
  return a `skipped:budget` decision rather than failing rollout.
- Soft-guidance principle: triggers are derived from observable state
  signals (phase tag, hp%, power signature delta, turn index) rather
  than hard-coded card or boss names. The LLM is never told "if X then
  call oracle"; the runtime decides.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Self-contained import — kimi_review_turn_order lives in scripts/, which
# means we depend on PYTHONPATH being set so `llm.scripts...` resolves.
# rollout already sets that.
_LLM_ROOT = Path(__file__).resolve().parents[1]
if str(_LLM_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_LLM_ROOT.parent))


_TRIGGER_REASONS = {
    "first_turn",
    "phase_changed",
    "low_hp",
    "power_signature_changed",
    "uncertainty_low_confidence",
    "lora_failed",
    "explicit_request",
}


@dataclass(slots=True)
class OracleDecision:
    """One trigger evaluation result."""
    fire: bool
    reason: str
    details: str = ""


@dataclass(slots=True)
class OracleInvocation:
    """One actual oracle call result."""
    status: str                      # ok / dry_run / api_error / parse_error / disabled / budget
    reason_for_invocation: str       # which trigger fired
    hint_json: dict[str, Any] | None
    raw_text: str
    latency_ms: float
    provider: str
    model: str
    error: str = ""


@dataclass(slots=True)
class OracleConfig:
    """Static configuration; runtime state lives in PlannerOracle."""
    enabled: bool = False
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"
    budget_per_episode: int = 3
    budget_per_run: int = 200
    timeout_s: float = 30.0
    temperature: float = 0.3
    max_tokens: int = 480
    replace_hint: bool = True        # swap LoRA hint with oracle hint in prompt
    fire_on_first_turn: bool = True
    fire_on_phase_change: bool = True
    low_hp_fraction: float = 0.30    # fire once when player_hp / max_hp drops below
    power_signature_change: bool = True
    dry_run: bool = False            # build prompt + log but don't HTTP


@dataclass(slots=True)
class _EpisodeState:
    """Per-episode tracking."""
    calls: int = 0
    last_phase_signature: str = ""
    last_power_signature: str = ""
    triggered_low_hp: bool = False
    fired_first_turn: bool = False
    first_seen_round: int | None = None


class PlannerOracle:
    """Per-rollout oracle controller.

    Usage
    -----
    >>> oracle = PlannerOracle(OracleConfig(enabled=True))
    >>> oracle.reset_episode("ep_42")
    >>> decision = oracle.should_invoke(state, lora_hint_text, step_index=5)
    >>> if decision.fire:
    ...     result = oracle.invoke(state, legal_actions, lora_hint_text, decision)
    ...     trace_step["oracle_hint"] = result.hint_json
    """

    def __init__(self, config: OracleConfig) -> None:
        self._config = config
        self._calls_total = 0
        self._episode = _EpisodeState()
        self._current_episode_id = ""
        self._api_key: str | None = None
        # Lazy import: only need scripts.teacher when actually configured.
        self._call_fn: Callable[..., tuple[dict[str, Any], float]] | None = None

    # ----- lifecycle -----
    def reset_episode(self, episode_id: str = "") -> None:
        self._episode = _EpisodeState()
        self._current_episode_id = episode_id

    @property
    def config(self) -> OracleConfig:
        return self._config

    @property
    def calls_total(self) -> int:
        return self._calls_total

    @property
    def calls_this_episode(self) -> int:
        return self._episode.calls

    # ----- trigger logic -----
    def should_invoke(
        self,
        state: dict[str, Any],
        lora_hint_text: str,
        *,
        step_index: int,
        explicit_reason: str = "",
        lora_status: str = "",
    ) -> OracleDecision:
        cfg = self._config
        if not cfg.enabled:
            return OracleDecision(False, "disabled")
        if self._calls_total >= cfg.budget_per_run:
            return OracleDecision(False, "budget_run_exhausted")
        if self._episode.calls >= cfg.budget_per_episode:
            return OracleDecision(False, "budget_episode_exhausted")

        # 1. Explicit caller request always fires (e.g. test harness).
        if explicit_reason:
            return OracleDecision(True, "explicit_request", explicit_reason)

        # 2. LoRA failed or returned empty → fall back to oracle.
        if lora_status in {"empty", "parse_error", "failed"} or not (lora_hint_text or "").strip():
            return OracleDecision(True, "lora_failed", lora_status or "empty_hint")

        round_num = self._round_number(state)
        if self._episode.first_seen_round is None:
            self._episode.first_seen_round = round_num

        # 3. First turn of episode — establish a high-quality plan baseline.
        if cfg.fire_on_first_turn and not self._episode.fired_first_turn:
            self._episode.fired_first_turn = True
            return OracleDecision(True, "first_turn", f"round={round_num}")

        # 4. Phase changed — boss flipping ASLEEP→awake or entering invulnerable.
        phase_sig = self._compute_phase_signature(state)
        if cfg.fire_on_phase_change and phase_sig and self._episode.last_phase_signature:
            if phase_sig != self._episode.last_phase_signature:
                self._episode.last_phase_signature = phase_sig
                return OracleDecision(True, "phase_changed", phase_sig[:40])
        self._episode.last_phase_signature = phase_sig or self._episode.last_phase_signature

        # 5. Power signature changed — STEAM_ERUPTION_POWER stack increment, etc.
        if cfg.power_signature_change:
            power_sig = self._compute_power_signature(state)
            if (
                power_sig
                and self._episode.last_power_signature
                and power_sig != self._episode.last_power_signature
            ):
                self._episode.last_power_signature = power_sig
                return OracleDecision(True, "power_signature_changed", power_sig[:40])
            self._episode.last_power_signature = power_sig or self._episode.last_power_signature

        # 6. Low HP danger window — fire once when we cross threshold.
        if not self._episode.triggered_low_hp:
            ratio = self._player_hp_fraction(state)
            if 0 < ratio <= cfg.low_hp_fraction:
                self._episode.triggered_low_hp = True
                return OracleDecision(True, "low_hp", f"hp_frac={ratio:.2f}")

        return OracleDecision(False, "skipped")

    # ----- invocation -----
    def invoke(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        lora_hint_text: str,
        decision: OracleDecision,
    ) -> OracleInvocation:
        cfg = self._config
        if not cfg.enabled:
            return OracleInvocation(
                status="disabled", reason_for_invocation=decision.reason,
                hint_json=None, raw_text="", latency_ms=0.0,
                provider=cfg.provider, model=cfg.model,
            )
        # budget is double-checked because should_invoke might have been bypassed
        if self._calls_total >= cfg.budget_per_run or self._episode.calls >= cfg.budget_per_episode:
            return OracleInvocation(
                status="budget", reason_for_invocation=decision.reason,
                hint_json=None, raw_text="", latency_ms=0.0,
                provider=cfg.provider, model=cfg.model,
            )

        prompt = self._build_oracle_prompt(state, legal_actions, lora_hint_text, decision)
        if cfg.dry_run:
            self._tick_counters()
            return OracleInvocation(
                status="dry_run", reason_for_invocation=decision.reason,
                hint_json=None, raw_text=prompt[:200],
                latency_ms=0.0, provider=cfg.provider, model=cfg.model,
            )

        api_key = self._resolve_api_key()
        if not api_key:
            return OracleInvocation(
                status="api_error", reason_for_invocation=decision.reason,
                hint_json=None, raw_text="", latency_ms=0.0,
                provider=cfg.provider, model=cfg.model,
                error=f"missing api key env {cfg.api_key_env}",
            )

        body = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": _ORACLE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }

        try:
            response, latency_ms = self._do_call(body)
        except Exception as exc:  # noqa: BLE001 — record any HTTP failure
            self._tick_counters()
            return OracleInvocation(
                status="api_error", reason_for_invocation=decision.reason,
                hint_json=None, raw_text="", latency_ms=0.0,
                provider=cfg.provider, model=cfg.model, error=str(exc)[:200],
            )

        self._tick_counters()
        from llm.scripts.teacher.teacher_review_turn_order import parse_review_json, response_content
        text = response_content(response)
        parsed, status = parse_review_json(text)
        return OracleInvocation(
            status="ok" if status == "ok" and parsed else "parse_error",
            reason_for_invocation=decision.reason,
            hint_json=parsed,
            raw_text=text[:1200],
            latency_ms=latency_ms,
            provider=cfg.provider,
            model=cfg.model,
            error="" if status == "ok" else status,
        )

    # ----- helpers -----
    def _tick_counters(self) -> None:
        self._calls_total += 1
        self._episode.calls += 1

    def _do_call(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        if self._call_fn is None:
            from llm.scripts.teacher.teacher_review_turn_order import call_openai_chat
            self._call_fn = call_openai_chat
        api_key = self._resolve_api_key() or ""
        return self._call_fn(  # type: ignore[misc]
            base_url=self._config.base_url,
            api_key=api_key,
            body=body,
            timeout_s=self._config.timeout_s,
        )

    def _resolve_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        self._api_key = os.environ.get(self._config.api_key_env, "") or ""
        return self._api_key

    @staticmethod
    def _round_number(state: dict[str, Any]) -> int:
        battle = state.get("battle") if isinstance(state, dict) else None
        if isinstance(battle, dict):
            for key in ("round_number_raw", "round_number"):
                value = battle.get(key)
                try:
                    if value is not None:
                        return int(value)
                except (TypeError, ValueError):
                    pass
        return 0

    @staticmethod
    def _player_hp_fraction(state: dict[str, Any]) -> float:
        for source_key in ("player",):
            obj = state.get(source_key) if isinstance(state, dict) else None
            if isinstance(obj, dict):
                try:
                    hp = float(obj.get("hp") or obj.get("current_hp") or 0)
                    max_hp = float(obj.get("max_hp") or 0)
                except (TypeError, ValueError):
                    continue
                if max_hp > 0:
                    return hp / max_hp
        return 0.0

    @staticmethod
    def _compute_phase_signature(state: dict[str, Any]) -> str:
        enemies = []
        if isinstance(state, dict):
            enemies = state.get("enemies") or (state.get("battle") or {}).get("enemies") or []
        parts = []
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            eid = str(enemy.get("monster_id") or enemy.get("id") or "")
            phase = str(enemy.get("phase") or enemy.get("phase_id") or "")
            try:
                hp = int(enemy.get("hp") or 0)
            except (TypeError, ValueError):
                hp = 0
            invuln = "invuln" if hp >= 999_000_000 else ""
            parts.append(f"{eid}:{phase}:{invuln}")
        return "|".join(parts)

    @staticmethod
    def _compute_power_signature(state: dict[str, Any]) -> str:
        """Hash of (enemy_id, power_id, amount) tuples — detects new powers
        appearing or stack count changes (STEAM_ERUPTION_POWER + 1, etc.)."""
        enemies = []
        if isinstance(state, dict):
            enemies = state.get("enemies") or (state.get("battle") or {}).get("enemies") or []
        rows: list[str] = []
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            eid = str(enemy.get("monster_id") or enemy.get("id") or "")
            for power in enemy.get("powers") or enemy.get("buffs") or []:
                if not isinstance(power, dict):
                    continue
                pid = str(power.get("id") or power.get("power_id") or power.get("name") or "").upper()
                if not pid:
                    continue
                amt = power.get("amount")
                try:
                    amt_int = int(amt or 0)
                except (TypeError, ValueError):
                    amt_int = 0
                rows.append(f"{eid}:{pid}={amt_int}")
        if not rows:
            return ""
        return hashlib.sha1("|".join(sorted(rows)).encode("utf-8")).hexdigest()[:16]

    def _build_oracle_prompt(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        lora_hint_text: str,
        decision: OracleDecision,
    ) -> str:
        """Self-contained prompt: render visible state + LoRA hint + ask DS for
        a corrected planner_hint. Reuses the same JSON schema the planner LoRA
        emits so downstream tools (build_teacher_dataset, eval_planner_hint)
        treat both sources identically.
        """
        # Localized rendering avoids importing planner_hint here (avoid cycle).
        try:
            from llm.data_pipeline.planner_hint import render_planner_hint_user_message
            user_state = render_planner_hint_user_message(state, legal_actions or [])
        except Exception:
            user_state = json.dumps(
                {"state_keys": sorted((state or {}).keys())[:30]},
                ensure_ascii=False,
            )

        lora_block = (lora_hint_text or "").strip() or "(empty: planner LoRA returned no hint)"
        return (
            "You are a strong Slay the Spire 2 combat oracle. The local "
            "planner LoRA produced the following hint for the current "
            "turn — review it against the ground-truth state below and "
            "emit your own corrected planner_hint JSON.\n\n"
            f"trigger_reason: {decision.reason}\n"
            f"trigger_detail: {decision.details or '-'}\n\n"
            "<lora_planner_hint>\n"
            f"{lora_block}\n"
            "</lora_planner_hint>\n\n"
            "<combat_state>\n"
            f"{user_state}\n"
            "</combat_state>\n\n"
            "Output a single JSON object with exactly these keys: "
            "battle_objective, enemy_focus, deck_usage, risk_tradeoff, "
            "resource_timing, potion_stance, kill_order (list), "
            "danger_notes (list). All string values must be short, "
            "concrete English, and reference card / enemy / power IDs "
            "in original game form. The hint MUST adapt to the current "
            "round_number and any phase change visible in <combat_state>; "
            "do not restate a generic battle objective. Return JSON only "
            "(no markdown fences, no commentary)."
        )


_ORACLE_SYSTEM_PROMPT = (
    "You are a strong external combat oracle for Slay the Spire 2. "
    "You give a battle-level planner_hint, never an action. Use English. "
    "Use original game IDs for cards, enemies, powers, and relics. "
    "When the local planner LoRA's guidance contradicts the visible "
    "state (e.g. it ignores phase changes, scaling powers, lethal "
    "windows, or low-HP danger), output a corrected hint that addresses "
    "the missed signal. Return strict JSON only."
)


__all__ = [
    "OracleConfig",
    "OracleDecision",
    "OracleInvocation",
    "PlannerOracle",
]
