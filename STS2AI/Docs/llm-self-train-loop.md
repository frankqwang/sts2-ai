# LLM 自迭代训练循环

目标：用真实 Skada combat reset case、严格审计、长期 dataset pool 和固定评估门槛，持续提升 combat adapter，直到 fullrun 稳定通过 Act 1。

## 循环结构

1. 使用当前 combat adapter 在 Skada combat reset case 上 rollout。
2. 记录 `step_trace.jsonl`、`episode_trace.jsonl`、`train.jsonl`、`eval.jsonl`、`meta.json`。
3. `audit_rollout_failures` 抽出 invalid、defeat、left_combat、掉血回合、stderr 异常。
4. `manage_dataset_pool ingest-dataset` 把训练样本分流到 `gold / silver / quarantine`。
5. `manage_dataset_pool ingest-audit` 把失败和掉血回合写入 `hardcase`。
6. `grpo_lite` 用 positive advantage 样本训练 candidate adapter。
7. `policy_eval` 固定 seed 对比 current 和 candidate。
8. promotion gate 比较胜率、reward、invalid、strict JSON、机制分数和分 encounter 回退。
9. 通过则 candidate 成为下一轮 current；失败则保留原 current，并优先复盘 hardcase。

当前单轮编排入口：

```text
STS2AI/llm/scripts/self_iterate.py
```

多轮 curriculum 入口：

```text
STS2AI/llm/scripts/self_train_loop.py
```

Act 1 闭环入口：

```text
STS2AI/llm/scripts/train_until_act1_clear.py
```

## 长期 Dataset Pool

长期池位置：

```text
STS2AI/Artifacts/llm/dataset_pool
```

管理脚本：

```text
STS2AI/llm/scripts/manage_dataset_pool.py
```

样本分层：

- `gold`：Kimi / teacher / manual verified 样本，优先训练。
- `silver`：干净 rollout 正样本，用于扩量。
- `hardcase`：defeat、invalid、left_combat、掉血回合，等待复盘。
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

## 策略上下文

`strategy_context` 是可审计的明文规划摘要，不依赖隐藏 thinking。它由当前真实 `GameState`、`legal_actions` 和很短的运行记忆确定性生成，并同时用于训练 rollout 和观战推理：

- `memory`：上场战斗摘要、本战斗损血和最近动作、本回合最近动作。
- `plan`：卡组形态、关键牌、遗物倾向、当前威胁、可见斩杀线或优先目标。
- `turn`：本回合能量、主要目标和需要避免的明显错误。

这块保持短文本，目标是补充“容易被单步 prompt 忘掉的上下文”，不是把攻略长文塞进每步输入。prompt 里会明确声明当前 state 和 legal actions 优先级更高，避免策略摘要覆盖实时局面。后续如果要接入额外 LLM planner、战斗复盘、或按 key card 检索攻略，优先复用这块上下文形状，而不是新建另一套 prompt。

## 动作质量指标

rollout/eval 会记录保守的动作质量 flags，先只做统计和样本 meta，不直接影响执行：

- `missed_visible_lethal`：存在可见斩杀动作但没有选择。
- `end_turn_with_playable_cards`：还有可打出的牌却结束回合。
- `floating_energy_end_turn`：还有能量和可用牌却结束回合。
- `dangerous_end_turn`：敌方当前攻击会穿透格挡却结束回合。

每步会记录机会数和失误数，并汇总为：

- `mechanism_score`：机制机会上的综合命中率，用于晋级门槛。
- `sequence_score`：牌序/能量/可见斩杀处理。
- `defense_score`：危险回合是否避免直接结束。
- `hp_lost`：掉血控制。
- `turns` / `steps_per_turn`：回合速度和动作序列长度。
- `visible_damage_per_step`：可见输出节奏。

这些指标不是替代胜率，而是避免模型在简单 eval 胜率打满后仍然靠侥幸或低质量路线晋级。

## 离线样本挖掘

不依赖外部 API 时，先用本地 rollout 产物挖三类资产：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.mine_offline_preferences `
  --dataset-dir STS2AI\Artifacts\llm\datasets\<rollout_dataset> `
  --include-eval
```

输出在 `<rollout_dataset>/offline_mining/`：

- `hard_cases.jsonl`：低 advantage、动作质量 flags、或明显风险样本，用于后续集中评测/人工看/必要时再交给 API teacher。
- `preference_pairs.jsonl`：同 prompt 的高低分动作对，以及规则修复动作对，可用于 DPO/ORPO。
- `repair_sft.jsonl`：保守规则能直接修复的 SFT 样本，目前只自动修 `missed_visible_lethal`。

这个阶段的原则是宁可少修，也不要把不确定的策略判断写成硬标签。比如 `dangerous_end_turn` 只统计，不自动生成修复动作。

## 同局面多次推理

复杂卡牌配合不适合继续堆写死规则。对 hard case 可以让模型在同一个局面上多次独立推理，再让模型只基于当前局面和候选理由做自我复审：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.sample_state_candidates `
  --input-jsonl STS2AI\Artifacts\llm\datasets\<rollout_dataset>\offline_mining\hard_cases.jsonl `
  --adapter-dir STS2AI\Artifacts\llm\grpo\<adapter>\adapter `
  --samples-per-state 8 `
  --limit 50 `
  --self-rerank
```

这个流程不跑真实 rollout，也不需要 save/load。输出：

- `candidate_samples.jsonl`：每个局面的多次候选动作和理由。
- `disagreement_states.jsonl`：模型自己分歧或解析不稳的局面。
- `selected_action`：`self_rerank` 选择的候选动作；如果不开复审，则使用 majority action。

这不是绝对正确答案，但能把“模型自己不确定/推理不一致”的场面集中起来。之后可以选择把这些高价值状态交给大模型 API 排序，而不是把 API 预算花在随机局面上。

## 经验库和复盘

复盘、攻略、常见套路不要直接堆进每一步 prompt，而是沉淀成短经验条目：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.add_experience `
  --tags vulnerable,attack `
  --when "hand can apply Vulnerable and also deal meaningful attack damage" `
  --advice "apply Vulnerable before the largest attack when energy allows" `
  --avoid "do not spend the payoff attack before the debuff" `
  --source manual_seed `
  --confidence 0.7
```

默认经验库：

```text
STS2AI/Artifacts/llm/experience/lessons.jsonl
```

`sample_state_candidates --self-rerank` 会按当前局面检索少量经验，放进自我复审 prompt 的 `experience:` 块。这样可以形成：

```text
战斗/整局复盘 -> 短经验条目 -> hard case 多次推理 -> 自我复审/teacher 选择 -> 训练样本
```

经验条目字段保持小而可控：`tags / applies_when / advice / avoid / source / confidence`。

## 为什么按 encounter key 分组

同一个敌人，不同 deck/relic/build 是不同任务。例如 starter deck 的 `SLIMES_NORMAL` 和 midrun build 的 `SLIMES_NORMAL` 不能混在一起算 advantage，否则 reward 差异会来自 build 强弱，而不是动作质量。

当前 key 形式：

```text
CHOMPERS_NORMAL::act1_midrun::beaab971
```

## 晋级门槛

候选模型必须满足：

- invalid output episode rate 不超过阈值。
- 总体 win rate 不低于 current。
- 总体 reward 不能明显回退。
- `mechanism_score` 不能明显回退。
- `missed_visible_lethal` 不能增加。
- 每个 encounter/build 的 win rate 不能明显回退。
- 每个 encounter/build 的 reward 不能明显回退。

这样避免平均值掩盖局部灾难。

## 推荐起步命令

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.self_train_loop `
  --current-adapter STS2AI\Artifacts\llm\sft\grouped_index_damage_20260424-223329\adapter `
  --run-name self_train_curriculum `
  --iterations 3 `
  --stages "CHOMPERS;SLIMES,CULTISTS;act1_midrun" `
  --focus-hard-cases `
  --rollout-generations 4 `
  --eval-episodes-per-encounter 2 `
  --no-thinking
```

产物集中在：

```text
STS2AI/Artifacts/llm/runs/<run_name>/
STS2AI/Artifacts/llm/datasets/<iter_name>_rollout/
STS2AI/Artifacts/llm/grpo/<iter_name>_candidate/
STS2AI/Artifacts/llm/evals/<iter_name>_{current,candidate}_eval/
```

## 后续改进方向

- 修复 `assistant` loss scope 的 response-only mask 后，把 GRPO-lite 从 `full_text` 切回 `assistant`。
- 扩大固定评估集，增加更多 Act1 encounter/build。
- 把 hard cases 自动混入下一轮 rollout，而不是只在失败后切换 stage。
- 增加跨 stage 的保持性评估，防止新 stage 学会后忘掉旧 stage。
