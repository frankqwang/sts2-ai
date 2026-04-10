#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2ai_paths import ARTIFACTS_ROOT

# Keep the train process conservative on Windows workstations.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _looks_like_gguf(model_path: Path) -> bool:
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return True
    if model_path.is_dir():
        if any(child.suffix.lower() == ".gguf" for child in model_path.glob("*.gguf")):
            return True
    return False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ensure_train_deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    return (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )


def _build_prompt_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Choose the best game action from candidate_actions.",
        "prompt_state": record.get("prompt_state"),
        "candidate_actions": record.get("candidate_actions"),
        "notes": {
            "output_format": "Return exactly one JSON action chosen from candidate_actions.",
            "do_not_explain": True,
        },
    }


def _serialize_action(action: Any) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _messages_for_record(record: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    system_text = (
        "You are a Slay the Spire II non-combat decision model. "
        "Read the structured state and choose exactly one action from candidate_actions. "
        "Return only one JSON action object and no extra text."
    )
    prompt_payload = _build_prompt_payload(record)
    assistant_text = _serialize_action(record.get("target_action"))
    prompt_messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]
    return prompt_messages, full_messages


def _render_messages(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    parts: list[str] = []
    for message in messages:
        parts.append(f"<|{message['role']}|>\n{message['content']}")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _format_sft_example(record: dict[str, Any], tokenizer: Any | None = None) -> dict[str, str]:
    prompt_messages, full_messages = _messages_for_record(record)
    if tokenizer is None:
        prompt = _build_prompt_payload(record)
        target = record.get("target_action")
        return {
            "prompt_text": json.dumps(prompt, ensure_ascii=False),
            "completion_text": _serialize_action(target),
            "text": (
                "<|system|>\nYou are a Slay the Spire II decision model. "
                "Read the structured state and output exactly one action JSON.\n"
                "<|user|>\n"
                f"{json.dumps(prompt, ensure_ascii=False)}\n"
                "<|assistant|>\n"
                f"{json.dumps(target, ensure_ascii=False)}"
            ),
        }
    prompt_text = _render_messages(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = _render_messages(tokenizer, full_messages, add_generation_prompt=False)
    return {
        "prompt_text": prompt_text,
        "completion_text": _serialize_action(record.get("target_action")),
        "text": full_text,
    }


def _write_preview(path: Path, examples: list[dict[str, str]], max_examples: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preview_examples": examples[:max_examples],
        "num_examples": len(examples),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _looks_multimodal_config(config: Any) -> bool:
    if getattr(config, "vision_config", None) is not None:
        return True
    arch = " ".join(getattr(config, "architectures", []) or [])
    return "ConditionalGeneration" in arch


def _pick_target_modules(model: Any) -> list[str]:
    import torch

    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present: set[str] = set()
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            name = module.__class__.__name__
            _ = name
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = module_name.split(".")[-1]
            if leaf in preferred:
                present.add(leaf)
    if present:
        return [name for name in preferred if name in present]
    # Conservative fallback for unusual Qwen3.5 text internals.
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class PackedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class SFTTokenDataset:
    def __init__(
        self,
        *,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples: list[PackedExample] = []
        eos_id = tokenizer.eos_token_id
        for record in records:
            formatted = _format_sft_example(record, tokenizer)
            prompt_text = formatted["prompt_text"]
            completion_text = formatted["completion_text"]
            prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            completion_ids = list(tokenizer(completion_text, add_special_tokens=False)["input_ids"])
            if eos_id is not None:
                completion_ids = completion_ids + [int(eos_id)]

            # Preserve supervision first; trim prompt from the left if needed.
            if len(completion_ids) >= max_length:
                completion_ids = completion_ids[: max_length - 1] + (
                    [int(eos_id)] if eos_id is not None else []
                )
                prompt_ids = []
            else:
                allowed_prompt = max_length - len(completion_ids)
                if len(prompt_ids) > allowed_prompt:
                    prompt_ids = prompt_ids[-allowed_prompt:]

            full_ids = prompt_ids + completion_ids
            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + completion_ids
            attention_mask = [1] * len(full_ids)
            self.examples.append(
                PackedExample(
                    input_ids=[int(x) for x in full_ids],
                    attention_mask=[int(x) for x in attention_mask],
                    labels=[int(x) for x in labels],
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        ex = self.examples[idx]
        return {
            "input_ids": ex.input_ids,
            "attention_mask": ex.attention_mask,
            "labels": ex.labels,
        }


class DataCollatorForMaskedSFT:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        pad_id = int(self.tokenizer.pad_token_id)
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _load_model_and_tokenizer(
    *,
    model_name_or_path: str,
    torch: Any,
    AutoModelForCausalLM: Any,
    AutoModelForImageTextToText: Any,
    AutoTokenizer: Any,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = AutoModelForImageTextToText if _looks_multimodal_config(config) else AutoModelForCausalLM
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model_cls.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.config.use_cache = False
    metadata = {
        "config_class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "model_class": type(model).__name__,
        "multimodal": _looks_multimodal_config(config),
    }
    return model, tokenizer, metadata


def _build_run_plan(
    *,
    args: argparse.Namespace,
    bundle_dir: Path,
    output_dir: Path,
    formatted_train: list[dict[str, str]],
    formatted_val: list[dict[str, str]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = {
        "mode": args.mode,
        "bundle_dir": str(bundle_dir),
        "model": args.model,
        "output_dir": str(output_dir),
        "train_examples": len(formatted_train),
        "val_examples": len(formatted_val),
        "epochs": float(args.epochs),
        "learning_rate": float(args.learning_rate),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "max_length": int(args.max_length),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
    }
    if extra:
        plan.update(extra)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT trainer for STS2 LLM non-combat policy")
    parser.add_argument("--bundle-dir", default=str(ARTIFACTS_ROOT / "llm_bundle"), type=str, help="LLM bundle built by build_llm_bundle.py")
    parser.add_argument("--model", required=True, type=str, help="Trainable base model path (HF directory or model id)")
    parser.add_argument("--output-dir", default=str(ARTIFACTS_ROOT / "llm_policy"), type=str, help="Where to write train artifacts")
    parser.add_argument("--mode", choices=["dry-run", "sft"], default="dry-run")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _looks_like_gguf(model_path):
        raise SystemExit(
            f"Model path '{model_path}' looks like GGUF/LM Studio inference weights. "
            "Use a trainable HF model directory or model id for SFT/LoRA, then export to GGUF after training."
        )

    sft_train_path = bundle_dir / "sft" / "train.sft_dialogue.jsonl"
    sft_val_path = bundle_dir / "sft" / "val.sft_dialogue.jsonl"
    sft_train_records = _load_jsonl(sft_train_path)
    sft_val_records = _load_jsonl(sft_val_path)

    if args.max_train_samples > 0:
        sft_train_records = sft_train_records[: int(args.max_train_samples)]
    if args.max_val_samples > 0:
        sft_val_records = sft_val_records[: int(args.max_val_samples)]

    formatted_train = [_format_sft_example(record) for record in sft_train_records]
    formatted_val = [_format_sft_example(record) for record in sft_val_records]
    train_preview_path = _write_preview(output_dir / "train_preview.json", formatted_train)
    val_preview_path = _write_preview(output_dir / "val_preview.json", formatted_val)

    run_plan = _build_run_plan(
        args=args,
        bundle_dir=bundle_dir,
        output_dir=output_dir,
        formatted_train=formatted_train,
        formatted_val=formatted_val,
        extra={
            "train_preview_path": str(train_preview_path),
            "val_preview_path": str(val_preview_path),
        },
    )
    (output_dir / "run_plan.json").write_text(json.dumps(run_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.mode == "dry-run":
        print(json.dumps(run_plan, ensure_ascii=False, indent=2))
        return

    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    ) = _ensure_train_deps()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, model_meta = _load_model_and_tokenizer(
        model_name_or_path=args.model,
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoModelForImageTextToText=AutoModelForImageTextToText,
        AutoTokenizer=AutoTokenizer,
        device=device,
    )
    target_modules = _pick_target_modules(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SFTTokenDataset(
        records=sft_train_records,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
    )
    val_dataset = SFTTokenDataset(
        records=sft_val_records,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
    )
    collator = DataCollatorForMaskedSFT(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        warmup_ratio=float(args.warmup_ratio),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        eval_steps=int(args.eval_steps),
        save_total_limit=int(args.save_total_limit),
        eval_strategy="steps" if len(val_dataset) > 0 else "no",
        save_strategy="steps",
        logging_strategy="steps",
        do_train=True,
        do_eval=(len(val_dataset) > 0),
        report_to="none",
        bf16=False,
        fp16=(device == "cuda"),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    summary = {
        **run_plan,
        "device": device,
        "model_metadata": model_meta,
        "target_modules": target_modules,
        "trainable_params_note": "Saved adapter contains LoRA weights only.",
        "train_result": {
            "training_loss": float(getattr(train_result, "training_loss", 0.0)),
            "global_step": int(getattr(train_result, "global_step", 0)),
        },
        "adapter_dir": str(output_dir / "adapter"),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
