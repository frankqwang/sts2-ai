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
- `training/grpo_rollout.py`：当前模型 rollout，生成带 reward/advantage 的 combat 数据；非 boss 战 reward 主看净掉血和敌方血量推进。
- `training/grpo_lite.py`：positive advantage 样本训练 combat LoRA。
- `training/sft_lora.py`：标准 SFT，用于 non-combat、teacher、format repair。
- `eval/policy_eval.py`：固定 encounter/seed 策略评估。
- `scripts/analysis/audit_rollout_failures.py`：invalid、defeat、掉血回合、stderr 异常审计。
- `scripts/datasets/manage_dataset_pool.py`：长期 gold/silver/hardcase/quarantine 样本池。
- `scripts/automation/self_iterate.py`：单轮训练飞轮。
- `scripts/automation/self_train_loop.py`：多轮 curriculum 自训练。
- `scripts/automation/train_until_act1_clear.py`：fullrun 观战 + 训练直到 Act 1 clear。

## 脚本分组

`scripts/` 下按用途分组，索引见 `scripts/README.md`：

- `analysis/`：trace 审计、失败复盘、动作顺序分析、eval 对比。
- `automation/`：训练飞轮编排。
- `datasets/`：训练集构建、数据池管理、preflight。
- `teacher/`：Kimi/teacher 候选采样和复盘标注。
- `viewers/`：trace replay、metrics summary、HTML 可视化。
- `spectate/`：观战 PowerShell 入口。
- `debug/`：状态 dump 和 smoke。

## 训练飞轮

```text
current adapter
  -> grpo_rollout
  -> planner-hint injection
  -> audit_rollout_failures
  -> manage_dataset_pool ingest-dataset
  -> manage_dataset_pool ingest-audit
  -> teacher review
  -> grpo_lite train combat candidate + sft_lora train planner candidate
  -> policy_eval four-cell matrix
  -> joint quality gate
```

`self_iterate.py` 已经把这些步骤串起来；`self_train_loop.py` 负责多轮循环；`train_until_act1_clear.py` 负责 fullrun 验证。

## Dataset Pool

长期池位置：

```text
STS2AI/Artifacts/llm/dataset_pool
```

分层：

- `gold`：Kimi / teacher verified 样本。
- `silver`：干净 rollout 正样本；非 boss 战必须低净掉血，当前硬门槛 `hp_lost <= 4`。
- `hardcase`：高掉血、未结束但敌方推进高、低敌方推进、invalid、left_combat，等待复盘。
- `quarantine`：strict JSON 失败、危险动作、缺少 `hp_lost / enemy_damage_progress`、非正 advantage，不直接训练。

常用命令：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.datasets.manage_dataset_pool report
python -m llm.scripts.datasets.manage_dataset_pool materialize `
  --out-dir STS2AI\Artifacts\llm\datasets\managed_combat_pool_latest `
  --target-size 5000 `
  --gold-min-ratio 0.15
```

## Trace Web Viewer

`step_trace.jsonl` 可以导出成单文件 HTML，数据会内嵌到 HTML 中，生成后可直接双击本地打开，不需要 HTTP server。

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.viewers.trace_viewer_html `
  --trace STS2AI\Artifacts\llm\spectate_llm\<run>\step_trace.jsonl `
  --out STS2AI\Artifacts\llm\viewers\<run>\index.html `
  --title "<run> prompts"
```

常用参数：

- `--trace`：输入 `step_trace.jsonl`。
- `--out`：输出 HTML；不填时默认写到 trace 同目录。
- `--title`：页面标题。
- `--max-rows`：只导出前 N 行；`0` 表示全部。

页面支持搜索 prompt / action / reason，按 `route` 和 `quality_flags` 过滤，并查看每一步的 `user_message`、`raw_generation`、`decoded`、`chosen_action`。

## Planner-Hint LoRA

planner-hint LoRA 输出战斗级策略提示，替代旧的 rule-based `strategy_context.plan`。它不输出 `action_index`，不输出动作序列，也不直接执行；combat adapter 仍然读取当前 `legal_actions` 单步选动作。

当前只接受 v2 字段：`battle_objective / enemy_focus / deck_usage / risk_tradeoff / resource_timing / potion_stance / kill_order / danger_notes`。旧字段 `combat_plan / encounter_guide / defense_policy / resource_policy / potion_policy` 和未知字段直接 invalid。

当前 combat prompt 的上下文形状是扁平的 `strategy_context`，不再有 `agent_memory` 包裹层；`player:` 和 `legal_actions:` 前都会保留空行方便 trace 阅读：

```text
run: ...
strategy_context:
  short_term:
    recent_actions: none
  long_term: none
  planner_hint:
    battle_objective: ...

player: ...
...
hand:
  ...

legal_actions:
  [0] ...
```

运行时启用：

```powershell
$env:STS2_LLM_PLANNER_HINT_ADAPTER_DIR="C:\Users\Administrator\Desktop\sts2Zero\STS2AI\Artifacts\llm\sft\<planner_hint_run>\adapter"
$env:STS2_LLM_PLANNER_HINT="1"
$env:STS2_LLM_PLANNER_HINT_REFRESH="turn"
$env:STS2_LLM_GUIDE_RAG="1"
```

默认 `STS2_LLM_PLANNER_HINT_REFRESH=turn`（每回合刷新一次），让 planner 能对 boss buff、敌人 intent 变化即时反应。设 `combat` 整场战斗只生成一次缓存（不推荐）。

从 Kimi 整场复盘构建 planner-hint 数据集：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.datasets.build_planner_hint_dataset `
  --review-root STS2AI\Artifacts\llm\reviews\<review_run> `
  --out-dir STS2AI\Artifacts\llm\datasets\planner_hint_<run>
```

本地 guide RAG 语料在：

```text
STS2AI/llm/knowledge/guide_corpus.jsonl
```

硬校验入口：

```powershell
python -m llm.scripts.analysis.check_guide_corpus `
  --corpus STS2AI\llm\knowledge\guide_corpus.jsonl

python -m llm.scripts.analysis.eval_planner_hint_outputs `
  --dataset STS2AI\Artifacts\llm\datasets\planner_hint_<run> `
  --trace STS2AI\Artifacts\llm\datasets\<rollout_dataset>\step_trace.jsonl `
  --require-knowledge
```

训练入口仍用标准 SFT：

```powershell
python STS2AI\llm\training\sft_lora.py `
  --run-name planner_hint_<run> `
  --dataset-dir STS2AI\Artifacts\llm\datasets\planner_hint_<run> `
  --max-seq-length 2048 `
  --batch-size 1 `
  --grad-accum 8 `
  --num-epochs 1 `
  --load-in-4bit
```

rollout / 固定评估可直接传 planner-hint adapter，不依赖环境变量：

```powershell
python -m llm.training.grpo_rollout `
  --adapter-dir STS2AI\Artifacts\llm\grpo\<combat_run>\adapter `
  --case-index STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --planner-hint-adapter-dir STS2AI\Artifacts\llm\sft\<planner_hint_run>\adapter `
  --planner-hint-refresh turn `
  --planner-hint-max-new-tokens 240

python -m llm.eval.policy_eval `
  --adapter-dir STS2AI\Artifacts\llm\grpo\<combat_run>\adapter `
  --case-index STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --planner-hint-adapter-dir STS2AI\Artifacts\llm\sft\<planner_hint_run>\adapter `
  --planner-hint-refresh turn
```

## 非战斗模型

非战斗 SFT 数据由 `STS2AI/data/skada/build_non_combat_sft_dataset.py` 从 Skada victory run 生成。

当前基线数据集：

```text
STS2AI/Artifacts/llm/datasets/skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428
```

覆盖：

- `map_choice`
- `card_reward=10000`
- `relic_select`
- `rest_site_choice`
- `shop_choice`

说明：

- 保留游戏原始卡牌、遗物、power 文案；不要再用手写简写替代。
- 保留 Skada 离线占位符，例如 `{MaxHp}`、`{Energy:energyIcons()}`。
- `current_plan` 只基于当前卡组；`winning_outcome_reference` 单独提供胜利局最终构筑参考，避免把终局方向误当作当前已经成型的体系。
- 旧 v2d/v2e/v2f/v2g 是过渡数据，不作为正式训练输入。

当前线上 non-combat adapter 仍是旧 v2b，尚未用 v2h 重新训练：

```text
STS2AI/Artifacts/llm/sft/non_combat_skada_ironclad_v01032_2k_v2b_20260426/adapter
```

下一步先跑 Kimi 选卡标注，再训练新 non-combat LoRA。详细交接见：

```text
STS2AI/Docs/training-next-steps.md
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

`--case-index` 是 combat reset 入口必填参数；旧手工 Act1 pool 已删除，不再有无参数 fallback。

单轮 combat/planner 飞轮：

```powershell
python -m llm.scripts.automation.self_iterate `
  --current-adapter STS2AI\Artifacts\llm\grpo\<combat_run>\adapter `
  --planner-hint-adapter-dir STS2AI\Artifacts\llm\sft\<planner_hint_run>\adapter `
  --planner-hint-refresh turn `
  --case-index STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --case-limit 4 `
  --case-sample-mode stratified `
  --co-train-planner `
  --kimi-teacher `
  --teacher-provider kimi `
  --kimi-max-api-calls 2 `
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
  --case-index STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --episodes-per-encounter 5 `
  --load-in-4bit
```

非 boss 固定评估不要用胜负当主结论：已结束战斗看 `hp_lost`，未结束/失败战斗看 `enemy_damage_progress`，再结合 `mechanism_score / sequence_score / defense_score`。Boss/fullrun 才把胜负、Act 进度作为主指标。
