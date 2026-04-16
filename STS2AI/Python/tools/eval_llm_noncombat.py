#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

# Keep inference conservative on Windows workstations.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ in {None, ""}:
    python_root = Path(__file__).resolve().parent
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

import _path_init  # noqa: F401

from backends.full_run_backend import apply_backend_action
from runtime.full_run_action_semantics import (
    choose_auto_progress_action,
    choose_rollout_decision,
    next_reward_claim_signature,
)
from test_simulator_consistency import COMBAT_TYPES
from train_llm_policy import (
    _ensure_train_deps,
    _load_model_and_tokenizer,
    _messages_for_record,
    _render_messages,
    _serialize_action,
)
from generate_card_ranking_data import (
    _build_client,
    _extract_deck_ids,
    _extract_relic_ids,
    _load_combat_evaluator,
    _load_ppo_rollout_policy,
)
from sts2ai_paths import ARTIFACTS_ROOT, MAINLINE_CHECKPOINT
from verify_save_load import choose_default_action


def _state_type(state: dict[str, Any]) -> str:
    return str(state.get("state_type") or "").strip().lower()


def _extract_floor(state: dict[str, Any]) -> int:
    return int((state.get("run") or {}).get("floor") or 0)


def _extract_act(state: dict[str, Any]) -> int:
    return int((state.get("run") or {}).get("act") or 0)


def _extract_player_summary(state: dict[str, Any]) -> dict[str, Any]:
    player = state.get("player") or {}
    return {
        "hp": player.get("hp") or player.get("current_hp"),
        "max_hp": player.get("max_hp"),
        "gold": player.get("gold"),
        "energy": player.get("energy"),
        "potions": [p.get("id") or p.get("name") or "?" for p in (player.get("potions") or [])],
    }


def _extract_screen_context(state: dict[str, Any]) -> dict[str, Any]:
    st = _state_type(state)
    if st == "card_reward":
        reward = state.get("card_reward") or {}
        return {
            "reward_cards": [
                card.get("id") or card.get("name") or "?"
                for card in (reward.get("cards") or [])
            ],
        }
    if st == "rest_site":
        return {"rest_options": state.get("rest_options") or []}
    if st == "shop":
        shop = state.get("shop") or {}
        return {
            "shop_cards": [item.get("id") or item.get("name") or "?" for item in (shop.get("cards") or [])],
            "shop_relics": [item.get("id") or item.get("name") or "?" for item in (shop.get("relics") or [])],
            "shop_potions": [item.get("id") or item.get("name") or "?" for item in (shop.get("potions") or [])],
        }
    if st == "map":
        return {"next_boss": (state.get("run") or {}).get("boss")}
    if st == "event":
        event = state.get("event") or {}
        return {
            "event_id": event.get("id") or event.get("name"),
            "in_dialogue": event.get("in_dialogue"),
            "can_proceed": event.get("can_proceed"),
        }
    return {}


def _build_prompt_state(state: dict[str, Any]) -> str:
    payload = {
        "state_type": _state_type(state),
        "act": _extract_act(state),
        "floor": _extract_floor(state),
        "player": _extract_player_summary(state),
        "deck_ids": _extract_deck_ids(state),
        "relic_ids": _extract_relic_ids(state),
    }
    payload.update(_extract_screen_context(state))
    return json.dumps(payload, ensure_ascii=False)


def _enabled_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        action
        for action in (state.get("legal_actions") or [])
        if isinstance(action, dict) and action.get("is_enabled") is not False
    ]


def _repeat_signature(state: dict[str, Any], legal_actions: list[dict[str, Any]]) -> str:
    payload = {
        "state_type": _state_type(state),
        "floor": _extract_floor(state),
        "hp": (_extract_player_summary(state).get("hp")),
        "energy": (_extract_player_summary(state).get("energy")),
        "legal": [
            {
                "action": action.get("action"),
                "label": action.get("label"),
                "card_index": action.get("card_index"),
                "target_id": action.get("target_id"),
                "slot": action.get("slot"),
            }
            for action in legal_actions
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


class LLMNonCombatChooser:
    def __init__(
        self,
        *,
        model_path: str,
        adapter_dir: str,
        device: str = "auto",
        max_length: int = 1024,
    ) -> None:
        (
            self.torch,
            _LoraConfig,
            _TaskType,
            _get_peft_model,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            _Trainer,
            _TrainingArguments,
        ) = _ensure_train_deps()
        from peft import PeftModel

        if device == "auto":
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = int(max_length)
        base_model, tokenizer, model_meta = _load_model_and_tokenizer(
            model_name_or_path=model_path,
            torch=self.torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoModelForImageTextToText=AutoModelForImageTextToText,
            AutoTokenizer=AutoTokenizer,
            device=device,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
        self.model = self.model.to(device)
        self.model.eval()
        self.model.config.use_cache = True
        self.tokenizer = tokenizer
        self.model_meta = model_meta

    def _score_action(
        self,
        *,
        state: dict[str, Any],
        candidate_actions: list[dict[str, Any]],
        target_action: dict[str, Any],
    ) -> float:
        record = {
            "prompt_state": _build_prompt_state(state),
            "candidate_actions": candidate_actions,
            "target_action": target_action,
        }
        prompt_messages, _full_messages = _messages_for_record(record)
        prompt_text = _render_messages(self.tokenizer, prompt_messages, add_generation_prompt=True)
        completion_text = _serialize_action(target_action)

        prompt_ids = list(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        completion_ids = list(self.tokenizer(completion_text, add_special_tokens=False)["input_ids"])
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            completion_ids = completion_ids + [int(eos_id)]

        if len(completion_ids) >= self.max_length:
            completion_ids = completion_ids[: self.max_length - 1] + ([int(eos_id)] if eos_id is not None else [])
            prompt_ids = []
        else:
            allowed_prompt = self.max_length - len(completion_ids)
            if len(prompt_ids) > allowed_prompt:
                prompt_ids = prompt_ids[-allowed_prompt:]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        attention_mask = [1] * len(input_ids)
        batch = {
            "input_ids": self.torch.tensor([input_ids], dtype=self.torch.long, device=self.device),
            "attention_mask": self.torch.tensor([attention_mask], dtype=self.torch.long, device=self.device),
            "labels": self.torch.tensor([labels], dtype=self.torch.long, device=self.device),
        }
        with self.torch.no_grad():
            outputs = self.model(**batch)
        loss = float(outputs.loss.detach().cpu().item())
        return -loss

    def choose_action(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if not legal_actions:
            return None, []
        scored: list[dict[str, Any]] = []
        for action in legal_actions:
            score = self._score_action(
                state=state,
                candidate_actions=legal_actions,
                target_action=action,
            )
            scored.append(
                {
                    "score": float(score),
                    "action": action,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return (scored[0]["action"] if scored else None), scored


def _start_summary(
    *,
    seed: str,
    model_path: str,
    adapter_dir: str,
    checkpoint_path: str | None,
    combat_checkpoint_path: str | None,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "model_path": model_path,
        "adapter_dir": adapter_dir,
        "checkpoint": checkpoint_path,
        "combat_checkpoint": combat_checkpoint_path,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boss_reached": False,
        "outcome": None,
        "max_floor": 0,
        "total_steps": 0,
    }


def run_llm_mixed_game(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    summary_path = out_dir / "summary.json"

    chooser = LLMNonCombatChooser(
        model_path=args.model,
        adapter_dir=args.adapter_dir,
        device=args.device,
        max_length=args.max_length,
    )
    client = _build_client(args.port, args.transport)
    combat_evaluator = _load_combat_evaluator(args.checkpoint, args.combat_checkpoint)
    ppo_fallback = _load_ppo_rollout_policy(args.checkpoint) if args.use_ppo_fallback else None

    state = client.reset(character_id="IRONCLAD", ascension_level=int(args.ascension), seed=args.seed)
    rng = __import__("random").Random(args.seed)
    last_action_name = ""
    last_reward_claim_sig = ""
    last_reward_claim_count = 0
    last_state_sig = ""
    repeat_state_count = 0
    summary = _start_summary(
        seed=args.seed,
        model_path=args.model,
        adapter_dir=args.adapter_dir,
        checkpoint_path=args.checkpoint,
        combat_checkpoint_path=args.combat_checkpoint,
    )

    with trace_path.open("w", encoding="utf-8") as trace_handle:
        for step_idx in range(int(args.max_steps)):
            st = _state_type(state)
            summary["max_floor"] = max(int(summary["max_floor"]), _extract_floor(state))
            summary["total_steps"] = int(step_idx)
            if st == "boss" or _extract_floor(state) >= 16:
                summary["boss_reached"] = True

            if st == "game_over" or bool(state.get("terminal")):
                outcome = str((state.get("run_outcome") or (state.get("game_over") or {}).get("run_outcome") or "death")).lower()
                summary["outcome"] = outcome
                break

            legal = _enabled_legal_actions(state)
            current_state_sig = _repeat_signature(state, legal)
            if current_state_sig == last_state_sig:
                repeat_state_count += 1
            else:
                repeat_state_count = 0
                last_state_sig = current_state_sig
            if not legal:
                next_state = apply_backend_action(client, state, {"action": "wait"}, wait_timeout_s=1.0)
                trace_handle.write(json.dumps({
                    "step": int(step_idx),
                    "state_type": st,
                    "floor": _extract_floor(state),
                    "source": "wait_no_legal",
                    "action": {"action": "wait"},
                }, ensure_ascii=False) + "\n")
                state = next_state
                last_action_name = "wait"
                last_reward_claim_sig = ""
                last_reward_claim_count = 0
                continue

            auto_action = choose_auto_progress_action(
                state,
                legal,
                last_action_name=last_action_name,
                last_reward_claim_sig=last_reward_claim_sig,
                last_reward_claim_count=last_reward_claim_count,
            )
            scored_actions: list[dict[str, Any]] | None = None
            if auto_action is not None:
                action = auto_action
                source = "auto_progress"
            elif st in COMBAT_TYPES:
                combat_legal = legal
                if repeat_state_count >= 2:
                    combat_legal = [
                        action for action in legal
                        if str(action.get("action") or "").strip().lower() != "use_potion"
                    ] or legal
                decision = choose_rollout_decision(
                    state,
                    combat_legal,
                    rng,
                    combat_evaluator=combat_evaluator,
                    ppo_policy=None,
                )
                action = decision.action
                source = f"{decision.source}_repeat_escape" if repeat_state_count >= 2 else decision.source
            else:
                action, scored_actions = chooser.choose_action(state, legal)
                source = "llm_noncombat"
                if action is None and ppo_fallback is not None:
                    action = ppo_fallback.choose_action(state, legal)
                    source = "ppo_fallback"
                if action is None:
                    action = choose_default_action(state)
                    source = "default_fallback"

            next_state = apply_backend_action(client, state, action, wait_timeout_s=1.0)
            trace_handle.write(json.dumps({
                "step": int(step_idx),
                "state_type": st,
                "floor": _extract_floor(state),
                "source": source,
                "action": action,
                "top_scored_actions": (
                    [
                        {"score": round(float(item["score"]), 4), "action": item["action"]}
                        for item in (scored_actions or [])[: min(3, len(scored_actions or []))]
                    ]
                    if scored_actions is not None else None
                ),
            }, ensure_ascii=False) + "\n")
            last_action_name = str((action or {}).get("action") or "").strip().lower()
            last_reward_claim_sig = next_reward_claim_signature(st, state, action)
            last_reward_claim_count = sum(
                1 for candidate in legal
                if str(candidate.get("action") or "").strip().lower() == "claim_reward"
            )
            state = next_state
        else:
            summary["outcome"] = "timeout"

    if summary["outcome"] is None:
        summary["outcome"] = "unknown"
    summary["ended_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["trace_path"] = str(trace_path)
    summary["model_metadata"] = chooser.model_meta
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        client.close()
    except Exception:
        pass
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one mixed-agent game with LLM non-combat decisions.")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--adapter-dir", default=str(ARTIFACTS_ROOT / "llm_policy" / "adapter"), type=str)
    parser.add_argument("--checkpoint", default=str(MAINLINE_CHECKPOINT), type=str, help="Hybrid/full-run checkpoint for PPO/combat fallback")
    parser.add_argument("--combat-checkpoint", default="", type=str, help="Standalone combat checkpoint override")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--transport", choices=["http", "pipe", "pipe-binary"], default="pipe-binary")
    parser.add_argument("--seed", required=True, type=str)
    parser.add_argument("--output-dir", default=str(ARTIFACTS_ROOT / "llm_noncombat_eval"), type=str)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--use-ppo-fallback", action="store_true")
    args = parser.parse_args()

    summary_path = run_llm_mixed_game(args)
    print(str(summary_path))


if __name__ == "__main__":
    main()
