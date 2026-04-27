"""GRPO-Lite: 用 rollout 产出的 advantage 做 Reward-Weighted SFT。

与标准 SFT 的区别：
- 每条训练样本带一个 `advantage`（可以是负的）。
- 当前实现默认做优势过滤 SFT：只保留正 advantage 样本继续模仿。
- 这是保守版 self-improvement，先避免自定义 logits loss 对 Unsloth/Qwen3 的兼容问题。

与标准 GRPO 的区别：
- 不现场采样 completion（不需要 vLLM 或 model.generate 在训练循环里）。
- rollout 和训练解耦：先 `grpo_rollout.py` 跑数据，再 `grpo_lite.py` 训。
- 因此实现简单，稳定性高，适合游戏这种需要与外部环境交互的场景。

运行：

    python STS2AI/llm/training/grpo_lite.py \\
        --dataset-dir STS2AI/Artifacts/llm/datasets/grpo_v0 \\
        --run-name grpo_iter0 \\
        --num-epochs 2

如果已有 SFT adapter，建议先加载它作为 warm-start：

    python STS2AI/llm/training/grpo_lite.py \\
        --adapter-dir STS2AI/Artifacts/llm/sft/toy_sft/adapter \\
        --dataset-dir STS2AI/Artifacts/llm/datasets/grpo_v0 \\
        --run-name grpo_iter0

产物：
  STS2AI/Artifacts/llm/grpo/<run_name>/adapter/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.metrics import summarize_sft_run, write_json
from llm.paths import BASE_MODEL_ID, DATASETS_ROOT, GRPO_ROOT, setup_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--adapter-dir", type=str, default=None, help="warm-start adapter（通常是 SFT 产物）")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help=">0 时覆盖 epoch 步数，用于短跑基准测试")
    parser.add_argument("--save-steps", type=int, default=0, help=">0 时按 steps 保存 checkpoint，否则按 epoch 保存")
    parser.add_argument("--lr", type=float, default=1e-4, help="GRPO 学习率通常比 SFT 低一半")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--save-merged", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=0, help=">0 时只取前 N 条 train rows，用于 smoke/benchmark")
    parser.add_argument("--max-eval-rows", type=int, default=0, help=">0 时只取前 N 条 eval rows，用于 smoke/benchmark")
    parser.add_argument("--min-response-retention", type=float, default=0.95)
    parser.add_argument("--max-p95-tokens", type=int, default=0, help=">0 时启用 p95 token 上限预检")
    parser.add_argument("--max-p95-assistant-start", type=int, default=0, help=">0 时启用 assistant 起点 p95 上限预检")
    parser.add_argument("--max-deleted-frac", type=float, default=0.05, help="response-only masking 后样本删除比例上限")
    parser.add_argument("--advantage-temperature", type=float, default=1.0,
                        help="对 advantage 做温度缩放：weight = advantage / temp。>1 更保守，<1 更激进。")
    parser.add_argument("--advantage-clamp", type=float, default=3.0,
                        help="advantage 裁剪上下界，防止异常样本炸梯度。")
    parser.add_argument("--allow-zero-advantage-fallback", action="store_true",
                        help="允许所有 advantage<=0 时退化为全量 SFT；默认直接失败，避免静默训练坏信号。")
    parser.add_argument("--kl-penalty-coef", type=float, default=0.01,
                        help="KL 惩罚系数（相对 base model）。0 表示不惩罚。")
    parser.add_argument(
        "--loss-scope",
        choices=["assistant", "full_text"],
        default="assistant",
        help="assistant 只训练模型回答段；full_text 训练完整 chat text，用于绕过 response mask 兼容问题。",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def _preflight_texts(
    *,
    tokenizer: Any,
    texts: list[str],
    max_seq_length: int,
    min_response_retention: float,
    max_p95_tokens: int,
    max_p95_assistant_start: int,
) -> dict[str, Any]:
    marker = "<|im_start|>assistant\n"
    token_lengths: list[int] = []
    assistant_starts: list[int] = []
    missing = 0
    retained = 0
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_lengths.append(len(ids))
        pos = text.find(marker)
        if pos < 0:
            missing += 1
            continue
        prefix_ids = tokenizer(text[:pos], add_special_tokens=False)["input_ids"]
        assistant_start = len(prefix_ids)
        assistant_starts.append(assistant_start)
        if assistant_start < max_seq_length:
            retained += 1

    rows = len(texts)
    retention_rate = retained / max(rows, 1)
    report = {
        "rows": rows,
        "max_seq_length": max_seq_length,
        "assistant_marker_missing": missing,
        "assistant_retained_after_truncation": retained,
        "assistant_retention_rate": retention_rate,
        "token_lengths": {
            "p50": _percentile(token_lengths, 0.50),
            "p90": _percentile(token_lengths, 0.90),
            "p95": _percentile(token_lengths, 0.95),
            "p99": _percentile(token_lengths, 0.99),
            "max": max(token_lengths) if token_lengths else 0,
        },
        "assistant_start_tokens": {
            "p50": _percentile(assistant_starts, 0.50),
            "p90": _percentile(assistant_starts, 0.90),
            "p95": _percentile(assistant_starts, 0.95),
            "p99": _percentile(assistant_starts, 0.99),
            "max": max(assistant_starts) if assistant_starts else 0,
        },
    }
    reasons: list[str] = []
    if missing:
        reasons.append(f"assistant marker missing in {missing}/{rows} rows")
    if retention_rate < min_response_retention:
        reasons.append(f"response retention {retention_rate:.3f} < {min_response_retention:.3f}")
    if max_p95_tokens > 0 and report["token_lengths"]["p95"] > max_p95_tokens:
        reasons.append(f"p95 tokens {report['token_lengths']['p95']} > {max_p95_tokens}")
    if max_p95_assistant_start > 0 and report["assistant_start_tokens"]["p95"] > max_p95_assistant_start:
        reasons.append(f"p95 assistant start {report['assistant_start_tokens']['p95']} > {max_p95_assistant_start}")
    report["passed"] = not reasons
    report["reasons"] = reasons
    return report


def main() -> None:
    args = parse_args()
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    setup_runtime()

    run_name = args.run_name or f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = GRPO_ROOT / run_name
    adapter_dir = run_root / "adapter"
    logs_dir = run_root / "logs"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    train_path = dataset_dir / "train.jsonl"
    eval_path = dataset_dir / "eval.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"缺少训练集：{train_path}")
    print(f"[grpo] dataset dir = {dataset_dir}")
    print(f"[grpo] run dir     = {run_root}")

    print("[grpo] importing unsloth / trl ...")
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model_name = args.adapter_dir or BASE_MODEL_ID
    print(f"[grpo] loading {model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    if args.adapter_dir:
        print(f"[grpo] warm-start from {args.adapter_dir}")

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
    if hasattr(model, "for_training"):
        model.for_training(use_gradient_checkpointing=True)

    if args.kl_penalty_coef > 0.0:
        print("[grpo] note: KL penalty is ignored by advantage-filtered SFT mode.")

    train_rows = _load_jsonl(train_path)
    eval_rows = _load_jsonl(eval_path) if eval_path.exists() else []
    if args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_eval_rows > 0:
        eval_rows = eval_rows[: args.max_eval_rows]
    print(f"[grpo] train={len(train_rows)} eval={len(eval_rows)}")

    def _attach_advantage(row: dict[str, Any]) -> float:
        adv = float(row.get("meta", {}).get("advantage", 0.0))
        adv = max(-args.advantage_clamp, min(args.advantage_clamp, adv))
        adv = adv / max(args.advantage_temperature, 1e-6)
        row["advantage"] = adv  # 存回，dataset map 用
        return adv

    # 处理 advantage：裁剪 + 温度缩放。eval rows 也要带 advantage，
    # 否则 Dataset.map(_to_text) 会因为缺字段失败。
    advantages = [_attach_advantage(row) for row in train_rows]
    for row in eval_rows:
        _attach_advantage(row)

    print(f"[grpo] advantage range: [{min(advantages):.2f}, {max(advantages):.2f}]  mean={sum(advantages)/len(advantages):.3f}")
    filtered_train_rows = [row for row in train_rows if float(row.get("advantage", 0.0)) > 0.0]
    if not filtered_train_rows:
        if not args.allow_zero_advantage_fallback:
            raise SystemExit(
                "[grpo] no positive-advantage rows; refusing to fall back to all rows. "
                "Fix rollout advantage generation or pass --allow-zero-advantage-fallback for explicit SFT."
            )
        filtered_train_rows = train_rows
        print("[grpo] no positive-advantage rows; explicit fallback to all rows")
    else:
        print(f"[grpo] keeping positive-advantage rows: {len(filtered_train_rows)}/{len(train_rows)}")

    def _to_text(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        # 同 SFT：去掉 Qwen3 empty thinking block
        text = text.replace(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "<|im_start|>assistant\n",
        )
        return {"text": text, "advantage": example["advantage"]}

    train_ds = Dataset.from_list(filtered_train_rows).map(_to_text, remove_columns=None)
    eval_ds = Dataset.from_list(eval_rows).map(_to_text, remove_columns=None) if eval_rows else None
    preflight = _preflight_texts(
        tokenizer=tokenizer,
        texts=list(train_ds["text"]),
        max_seq_length=args.max_seq_length,
        min_response_retention=args.min_response_retention,
        max_p95_tokens=args.max_p95_tokens,
        max_p95_assistant_start=args.max_p95_assistant_start,
    )
    print(f"[grpo] preflight: {json.dumps(preflight, ensure_ascii=False)}")
    write_json(run_root / "preflight.json", preflight)
    if not preflight["passed"]:
        raise SystemExit(f"[grpo] preflight failed: {preflight['reasons']}")

    save_strategy = "steps" if args.save_steps > 0 else "epoch"

    sft_cfg = SFTConfig(
        output_dir=str(run_root / "trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
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

    if args.loss_scope == "assistant":
        # 只在 assistant 回答段算 loss。某些 Unsloth/Qwen3 版本对模板匹配比较敏感；
        # self-iterate 可临时切到 full_text 保持自动闭环可运行。
        before_mask_rows = len(trainer.train_dataset)
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        after_mask_rows = len(trainer.train_dataset)
        deleted_frac = (before_mask_rows - after_mask_rows) / max(before_mask_rows, 1)
        print(f"[grpo] response-mask kept rows: {after_mask_rows}/{before_mask_rows} deleted_frac={deleted_frac:.3f}")
        if deleted_frac > args.max_deleted_frac:
            raise SystemExit(
                f"[grpo] response mask deleted {deleted_frac:.3f}, "
                f"above limit {args.max_deleted_frac:.3f}"
            )
    else:
        print("[grpo] loss scope = full_text (response-only mask disabled)")

    print("[grpo] starting train ...")
    result = trainer.train()
    print(f"[grpo] train metrics: {result.metrics}")

    print(f"[grpo] saving LoRA adapter -> {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if args.save_merged:
        merged_dir = run_root / "merged_fp16"
        print(f"[grpo] saving merged fp16 -> {merged_dir}")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    with (run_root / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": BASE_MODEL_ID,
                "run_name": run_name,
                "dataset_dir": str(dataset_dir),
                "raw_train_size": len(train_rows),
                "train_size": len(filtered_train_rows),
                "eval_size": len(eval_rows),
                "objective": "advantage_filtered_sft",
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
    metrics["kind"] = "grpo_lite"
    write_json(run_root / "metrics.json", metrics)
    print(f"[grpo] metrics -> {run_root / 'metrics.json'}")

    print(f"[grpo] done. adapter at: {adapter_dir}")


if __name__ == "__main__":
    main()
