# STS2AI / llm

LLM 训练、推理、评估和数据飞轮代码目录。

## 运行形态

当前使用双 LoRA：

- combat adapter：战斗、精英、Boss、战斗内手牌选择。
- non-combat adapter：地图、选卡、遗物、营火、商店、事件、奖励领取。

`llm.inference.llm_policy` 会根据 `state_type / decision_type` 自动选择 adapter。当前默认 action mode 是 `index`，模型必须输出单个严格 JSON object：

```json
{"action_index":0,"confidence":0.75,"reason":"short reason"}
```

## 核心模块

- `inference/llm_policy.py`：模型推理、adapter 热切、strict JSON、retry、action 映射。
- `data_pipeline/state_renderer.py`：状态渲染、legal actions、strategy context、experience 注入。
- `data_pipeline/action_quality.py`：rollout/eval 的动作质量 flags 和训练 blocklist。
- `training/grpo_rollout.py`：当前模型 rollout，生成带 reward/advantage 的 combat 数据。
- `training/grpo_lite.py`：positive advantage 样本训练 combat LoRA。
- `training/sft_lora.py`：标准 SFT，用于 non-combat、teacher、format repair。
- `eval/policy_eval.py`：固定 encounter/seed 策略评估。
- `scripts/audit_rollout_failures.py`：invalid、defeat、掉血回合、stderr 异常审计。
- `scripts/manage_dataset_pool.py`：长期 gold/silver/hardcase/quarantine 样本池。
- `scripts/self_iterate.py`：单轮训练飞轮。
- `scripts/self_train_loop.py`：多轮 curriculum 自训练。
- `scripts/train_until_act1_clear.py`：fullrun 观战 + 训练直到 Act 1 clear。

## 训练飞轮

```text
current adapter
  -> grpo_rollout
  -> audit_rollout_failures
  -> manage_dataset_pool ingest-dataset
  -> manage_dataset_pool ingest-audit
  -> grpo_lite train candidate
  -> policy_eval current/candidate
  -> promotion gate
```

`self_iterate.py` 已经把这些步骤串起来；`self_train_loop.py` 负责多轮循环；`train_until_act1_clear.py` 负责 fullrun 验证。

## Dataset Pool

长期池位置：

```text
STS2AI/Artifacts/llm/dataset_pool
```

分层：

- `gold`：Kimi / teacher / manual verified 样本。
- `silver`：干净 rollout 正样本。
- `hardcase`：失败、invalid、掉血回合、left_combat，等待复盘。
- `quarantine`：strict JSON 失败、危险动作、非正 advantage，不直接训练。

常用命令：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.manage_dataset_pool report
python -m llm.scripts.manage_dataset_pool materialize `
  --out-dir STS2AI\Artifacts\llm\datasets\managed_combat_pool_latest `
  --target-size 5000 `
  --gold-min-ratio 0.15
```

## 非战斗模型

非战斗 SFT 数据由 `STS2AI/data/skada/build_non_combat_sft_dataset.py` 从 Skada victory run 生成。

当前 v2b 数据覆盖：

- `map_choice`
- `card_reward`
- `relic_select`
- `rest_site_choice`
- `shop_choice`

当前 v2b adapter：

```text
STS2AI/Artifacts/llm/sft/non_combat_skada_ironclad_v01032_2k_v2b_20260426/adapter
```

## 常用入口

Skada combat reset rollout：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.training.grpo_rollout `
  --adapter-dir STS2AI\Artifacts\llm\grpo\<run>\adapter `
  --case-index STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --case-limit 200 `
  --case-sample-mode stratified `
  --out-subdir skada_combat_rollout `
  --load-in-4bit `
  --no-thinking
```

训练 candidate：

```powershell
python -m llm.training.grpo_lite `
  --adapter-dir STS2AI\Artifacts\llm\grpo\<current>\adapter `
  --dataset-dir STS2AI\Artifacts\llm\datasets\<dataset> `
  --run-name combat_candidate `
  --loss-scope assistant `
  --load-in-4bit
```

固定评估：

```powershell
python -m llm.eval.policy_eval `
  --adapter-dir STS2AI\Artifacts\llm\grpo\<candidate>\adapter `
  --run-name policy_eval_candidate `
  --episodes-per-encounter 5 `
  --load-in-4bit
```
