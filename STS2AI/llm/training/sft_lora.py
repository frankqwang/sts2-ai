"""unsloth + TRL SFT 入口。

默认读 `STS2AI/Artifacts/llm/datasets/toy/{train,eval}.jsonl`，训完 LoRA
adapter 写到 `STS2AI/Artifacts/llm/sft/<run_name>/adapter/`。

运行（在 unsloth studio 3.13 venv 里）：

    C:\\Users\\Administrator\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe ^
        STS2AI\\llm\\training\\sft_lora.py --run-name toy_sft --num-epochs 3
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.metrics import summarize_sft_run, write_json  # noqa: E402
from llm.paths import BASE_MODEL_ID, DATASETS_ROOT, REPO_ROOT, SFT_ROOT, setup_runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None, help="产物子目录名，默认时间戳")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASETS_ROOT / "toy"))
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="走 bnb-4bit QLoRA（省显存但训练慢）。默认 fp16/bf16 全精度权重 + LoRA。")
    parser.add_argument("--save-merged", action="store_true", help="顺便导出 merged fp16 权重")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    setup_runtime()

    run_name = args.run_name or f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = SFT_ROOT / run_name
    adapter_dir = run_root / "adapter"
    logs_dir = run_root / "logs"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.jsonl"
    eval_path = dataset_dir / "eval.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"缺少训练集：{train_path}")
    print(f"[sft] dataset dir = {dataset_dir}")
    print(f"[sft] run dir     = {run_root}")

    print("[sft] importing unsloth / trl ...")
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print(f"[sft] loading {BASE_MODEL_ID} load_in_4bit={args.load_in_4bit} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    train_rows = _load_jsonl(train_path)
    eval_rows = _load_jsonl(eval_path) if eval_path.exists() else []
    print(f"[sft] train={len(train_rows)} eval={len(eval_rows)}")

    def _to_text(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        # Qwen3/3.5 template may insert an empty thinking block in supervised
        # examples when thinking is disabled. Strip it so the SFT target stays
        # clean. NOTE: if your dataset contains REAL thinking content
        # (<think>actual reasoning</think>), do NOT strip it.
        text = text.replace(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<|im_start|>assistant\n",
        )
        return {"text": text}

    train_ds = Dataset.from_list(train_rows).map(_to_text, remove_columns=None)
    eval_ds = Dataset.from_list(eval_rows).map(_to_text, remove_columns=None) if eval_rows else None

    sft_cfg = SFTConfig(
        output_dir=str(run_root / "trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=True,
        fp16=False,
        seed=args.seed,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_cfg,
    )

    # 只在 assistant 回答段算 loss
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("[sft] starting train ...")
    result = trainer.train()
    print(f"[sft] train metrics: {result.metrics}")

    print(f"[sft] saving LoRA adapter -> {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if args.save_merged:
        merged_dir = run_root / "merged_fp16"
        print(f"[sft] saving merged fp16 -> {merged_dir}")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    with (run_root / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": BASE_MODEL_ID,
                "run_name": run_name,
                "dataset_dir": str(dataset_dir),
                "train_size": len(train_rows),
                "eval_size": len(eval_rows),
                "args": vars(args),
                "metrics": result.metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metrics = summarize_sft_run(
        run_root,
        dataset_dir=dataset_dir,
        result_metrics=result.metrics,
        log_history=list(trainer.state.log_history),
    )
    write_json(run_root / "metrics.json", metrics)
    print(f"[sft] metrics -> {run_root / 'metrics.json'}")

    print(f"[sft] done. adapter at: {adapter_dir}")


if __name__ == "__main__":
    main()
