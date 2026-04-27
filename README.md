# STS2AI

当前仓库主线是用 LLM policy 打 Slay the Spire 2，并围绕真实 Skada combat reset case 建立长期训练飞轮。

## 当前主线

- 基座模型：`Qwen/Qwen3-4B-Instruct-2507`
- 训练框架：Unsloth + TRL + PEFT LoRA
- 战斗策略：combat adapter，使用 Skada combat reset rollout 迭代
- 非战斗策略：non-combat adapter，使用 Skada victory run SFT 数据
- 推理方式：同一模型进程内按 state type 热切 combat / non-combat adapter
- 游戏接口：`STS2AI/bridge/game_bridge`
- 数据来源：`STS2AI/data/skada` 与 `STS2AI/Assets/datasets/zero_skada_replay_cases`
- 运行产物：`STS2AI/Artifacts/llm`

## 训练飞轮

核心流程：

```text
current adapter
  -> Skada combat reset rollout
  -> rollout failure audit
  -> dataset pool ingest
  -> GRPO-lite / SFT train candidate
  -> fixed policy eval
  -> promotion gate
  -> fullrun spectate validation
```

核心代码：

- `STS2AI/llm/inference/llm_policy.py`：线上推理、JSON 解析、retry、adapter 热切。
- `STS2AI/llm/data_pipeline/state_renderer.py`：LLM 输入渲染、legal actions、strategy context。
- `STS2AI/llm/training/grpo_rollout.py`：用当前模型跑 combat rollout 并生成训练样本。
- `STS2AI/llm/training/grpo_lite.py`：用 positive advantage 样本训练 combat LoRA。
- `STS2AI/llm/training/sft_lora.py`：标准 SFT 训练入口。
- `STS2AI/llm/eval/policy_eval.py`：固定 encounter/seed 的策略评估。
- `STS2AI/llm/scripts/self_iterate.py`：单轮 rollout -> audit -> pool -> train -> eval -> gate。
- `STS2AI/llm/scripts/self_train_loop.py`：多轮 curriculum 自训练。
- `STS2AI/llm/scripts/train_until_act1_clear.py`：fullrun 观战、复盘、训练直到 Act 1 clear。

## Dataset Pool

长期样本池在：

```text
STS2AI/Artifacts/llm/dataset_pool
```

管理脚本：

```text
STS2AI/llm/scripts/manage_dataset_pool.py
```

样本分层：

- `gold`：Kimi / teacher / manual verified 的高质量样本。
- `silver`：干净 rollout 正样本。
- `hardcase`：失败、invalid、掉血回合、left_combat，等待复盘。
- `quarantine`：strict JSON 失败、危险动作、非正 advantage 等不允许直接训练的样本。

常用命令：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"

python -m llm.scripts.manage_dataset_pool report

python -m llm.scripts.manage_dataset_pool materialize `
  --out-dir STS2AI\Artifacts\llm\datasets\managed_combat_pool_latest `
  --target-size 5000 `
  --gold-min-ratio 0.15
```

## 当前主要数据

战斗数据：

- Skada combat reset case index：
  `STS2AI/Assets/datasets/zero_skada_replay_cases/v0_103_2_a0_single_combat_v1/cases.jsonl`
- 当前 rollout / training datasets：
  `STS2AI/Artifacts/llm/datasets`
- 当前 LoRA / checkpoints：
  `STS2AI/Artifacts/llm/grpo`

非战斗数据：

- 数据构建脚本：
  `STS2AI/data/skada/build_non_combat_sft_dataset.py`
- 当前 v2b 数据集：
  `STS2AI/Artifacts/llm/datasets/skada_non_combat_ironclad_v01032_2k_v2b_20260426`
- 当前 v2b adapter：
  `STS2AI/Artifacts/llm/sft/non_combat_skada_ironclad_v01032_2k_v2b_20260426/adapter`

## 观战运行

```powershell
powershell -ExecutionPolicy Bypass -File .\STS2AI\llm\scripts\spectate_llm.ps1 `
  -CombatAdapterDir "C:\Users\Administrator\Desktop\sts2Zero\STS2AI\Artifacts\llm\grpo\<combat_run>\adapter" `
  -NonCombatAdapterDir "C:\Users\Administrator\Desktop\sts2Zero\STS2AI\Artifacts\llm\sft\non_combat_skada_ironclad_v01032_2k_v2b_20260426\adapter" `
  -ActionMode index `
  -MaxSteps 800
```

## 文档

维护中的项目文档统一放在：

```text
STS2AI/Docs
```

当前有效文档：

- `STS2AI/Docs/README.md`
- `STS2AI/Docs/design/game-bridge-current.md`
- `STS2AI/Docs/llm-self-train-loop.md`

旧 zero/replay 训练主线和 2026-04-24 交接文档已删除，不再作为当前实现依据。
