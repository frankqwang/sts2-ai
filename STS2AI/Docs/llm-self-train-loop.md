# LLM 自迭代训练循环

目标：用真实 Skada combat reset case、严格审计、长期 dataset pool 和固定评估门槛，持续提升 combat/planner adapters，直到 fullrun 稳定通过 Act 1。

所有 combat reset 训练/评估入口都必须显式传 `--case-index` 指向 Skada `cases.jsonl`。旧的手工 Act1 pool 已删除，不再有 fallback。

## 循环结构

有 planner-hint 之后，战斗飞轮按同轮协同训练：

- `combat adapter`：负责每一步根据当前 `GameState` 和 `legal_actions` 输出单个动作。
- `planner-hint adapter`：负责在战斗开始或每回合生成短策略 hint，替代旧 `strategy_context.plan`，但不直接执行动作。
- 两者用同一批 rollout/teacher 反馈一起更新：teacher 一次复盘同时产动作标签和 battle-level planner 标签。

单轮 combat 飞轮：

1. 固定当前 `combat adapter C_t` 和当前 `planner-hint adapter P_t`。
2. rollout 时先由 `P_t` 生成战斗级 hint，注入每一步 combat prompt；`C_t` 仍只输出当前步动作。
3. 记录 `step_trace.jsonl`、`episode_trace.jsonl`、`train.jsonl`、`eval.jsonl`、`meta.json`，其中 step trace 会保留实际 prompt、planner hint 状态、动作质量 flags 和 `legal_actions`。
4. `audit_rollout_failures` 抽出 invalid、left_combat、高掉血回合、低敌方推进、stderr 异常。
5. `manage_dataset_pool ingest-dataset` 按非 boss 战质量分流：已结束战斗看净掉血，未结束/失败战斗看敌方血量推进，`victory` 只记录终止状态。
6. `manage_dataset_pool ingest-audit` 把高掉血、未结束但高推进、低推进和协议错误写入 `hardcase`。
7. Teacher 只复盘高价值 hard combat，产出两条标签流：
   - 动作级 `teacher_turn_labels.jsonl`，进入 combat gold 数据。
   - 战斗级 `planner_hint`，进入 planner-hint SFT 数据。
8. `build_teacher_dataset` 用原始 `legal_actions` 校验动作标签，生成 combat teacher dataset 并写入长期 pool。
9. `build_planner_hint_dataset` 生成 planner-hint SFT dataset。
10. `grpo_lite` 用 materialized pool 训练 `combat candidate C_{t+1}`；训练 prompt 已包含 rollout 时的 planner hint。
11. `sft_lora` 用本轮 planner-hint dataset 训练 `planner candidate P_{t+1}`。
12. `policy_eval` 跑四格矩阵：`C_t/P_t`、`C_{t+1}/P_t`、`C_t/P_{t+1}`、`C_{t+1}/P_{t+1}`。
13. quality gate 默认看 joint 组合 `C_{t+1}/P_{t+1}`；非 boss case 以掉血、敌方推进、动作质量为主，Boss/fullrun 才看胜负，通过则两个 adapter 一起进入下一轮。

当前单轮编排入口：

```text
STS2AI/llm/scripts/automation/self_iterate.py
```

多轮 curriculum 入口：

```text
STS2AI/llm/scripts/automation/self_train_loop.py
```

Act 1 闭环入口：

```text
STS2AI/llm/scripts/automation/train_until_act1_clear.py
```

## 长期 Dataset Pool

长期池位置：

```text
STS2AI/Artifacts/llm/dataset_pool
```

管理脚本：

```text
STS2AI/llm/scripts/datasets/manage_dataset_pool.py
```

样本分层：

- `gold`：Kimi / teacher / manual verified 样本，优先训练。
- `silver`：干净 rollout 正样本；非 boss 战必须低净掉血，当前硬门槛 `hp_lost <= 4`。
- `hardcase`：高掉血、未结束但敌方推进高、低敌方推进、invalid、left_combat，等待复盘。
- `quarantine`：strict JSON 失败、危险动作、缺少 `hp_lost / enemy_damage_progress`、非正 advantage 等不允许直接训练的样本。

常用命令：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.datasets.manage_dataset_pool report
python -m llm.scripts.datasets.manage_dataset_pool materialize `
  --out-dir STS2AI\Artifacts\llm\datasets\managed_combat_pool_latest `
  --target-size 5000 `
  --gold-min-ratio 0.15
```

## 策略上下文和 planner-hint

`strategy_context` 是可审计的明文上下文，不依赖隐藏 thinking。当前主线只允许两类内容进入 combat prompt：

- `short_term / long_term`：本进程产生的动作记忆。它不是规则库，也不是把状态摘要改名成 memory。
- `planner_hint`：由 planner-hint LoRA 生成的战斗级策略提示，使用 v2 schema。

旧的 `memory / threat / target / turn / rule / plan` 规则段已经不再作为默认上下文。旧 planner 字段 `combat_plan / encounter_guide / defense_policy / resource_policy / potion_policy` 直接判 invalid，不做兼容映射。

planner-hint LoRA 的输出不是动作序列，不包含 `action_index`，也不直接执行。当前 v2 schema：

```json
{
  "battle_objective": "Use BASH to create a Vulnerable damage window.",
  "enemy_focus": "Focus one CULTIST at a time.",
  "deck_usage": "Use STRIKE_IRONCLAD follow-up to exploit Vulnerable.",
  "risk_tradeoff": "Accept small HP loss only when it shortens future risk.",
  "resource_timing": "Spend BASH when the debuff has a real attack payoff.",
  "potion_stance": "Save potions unless they prevent a major HP swing.",
  "kill_order": ["enemy1", "enemy2"],
  "danger_notes": ["Do not split damage so both enemies keep scaling."]
}
```

注入到 combat prompt 后渲染为：

```text
run: ...
strategy_context:
  short_term:
    recent_actions: played BASH hand[2] -> enemy1
  long_term: none
  planner_hint:
    battle_objective: Use BASH to create a Vulnerable damage window.
    enemy_focus: Focus one CULTIST at a time.
    kill_order: enemy1 -> enemy2

player: ...
...
hand:
  ...

legal_actions:
  [0] ...
```

combat adapter 仍然逐步读取当前 state 和 `legal_actions`，输出单个动作。当前 state、当前 `legal_actions` 和真实游戏执行结果永远高于 planner hint。`player:` 和 `legal_actions:` 前的空行是 prompt 格式的一部分，用于让 trace 更容易读，不改变字段语义。

## Guide RAG

外部攻略、人类先验和 teacher 复盘沉淀到本地 guide corpus：

```text
STS2AI/llm/knowledge/guide_corpus.jsonl
```

检索层只负责把相关证据块拼到 planner prompt 的 `retrieved_knowledge`，不直接生成规则，不替代 planner LoRA。当前可用环境变量：

- `STS2_LLM_GUIDE_RAG=1`：启用 guide 检索，默认开启。
- `STS2_LLM_GUIDE_LIMIT=4`：每次最多注入几条证据。
- `STS2_LLM_GUIDE_REQUIRED=1`：没有检索证据时直接失败，用于训练/评估硬门槛。
- `STS2_LLM_PLANNER_HINT_REQUIRED=1`：planner 失败、空 hint、invalid hint 时直接失败。

硬校验入口：

```powershell
python -m llm.scripts.analysis.check_guide_corpus `
  --corpus STS2AI\llm\knowledge\guide_corpus.jsonl

python -m llm.scripts.analysis.eval_planner_hint_outputs `
  --dataset STS2AI\Artifacts\llm\datasets\<planner_hint_dataset> `
  --trace STS2AI\Artifacts\llm\datasets\<rollout_dataset>\step_trace.jsonl `
  --require-knowledge
```

## 对齐主流训练和 Agent 框架的缺口

当前已有：SFT/rollout trace、teacher provider 切换、combat/planner 联动训练、固定评估、可视化 trace、guide RAG 雏形。

还缺的关键件：

- 数据注册表：每个 dataset/adapter/eval 要有 manifest、父版本、schema 版本、corpus hash、teacher provider、训练参数和 promotion 结果。
- 硬评估门槛：combat、planner、non-combat 三个 adapter 要有独立 eval 和 joint eval，失败数据进入 hardcase，不让脏样本继续训练。
- Teacher QA：teacher 输出必须经过 schema、英文值、无动作字段、无旧字段、legal action 校验；不满足就重标，不做降级。
- Preference/RM：现在主要是 SFT 和轻量 rollout，还缺系统化 preference pairs、reward model 或 DPO/ORPO/RFT 类流程。
- Agent memory：现在只有短期动作 memory。真正需要的是 episodic memory、semantic guide memory、run-level build memory，并通过检索进入 planner prompt。
- Guardrails 和 observability：需要把 invalid、stale legal action、planner failure、RAG miss、teacher parse fail 都作为可统计 gate，而不是日志里人工找。

## 动作质量指标

rollout/eval 会记录保守的动作质量 flags，先只做统计和样本 meta，不直接影响执行：

- `missed_visible_lethal`：存在可见斩杀动作但没有选择。
- `end_turn_with_playable_cards`：还有可打出的牌却结束回合。
- `floating_energy_end_turn`：还有能量和可用牌却结束回合。
- `dangerous_end_turn`：敌方当前攻击会穿透格挡却结束回合。

每步会记录机会数和失误数，并汇总为：

- `mechanism_score`：机制机会上的综合命中率，用于质量门槛。
- `sequence_score`：牌序/能量/可见斩杀处理。
- `defense_score`：危险回合是否避免直接结束。
- `hp_lost`：掉血控制。
- `turns` / `steps_per_turn`：回合速度和动作序列长度。
- `visible_damage_per_step`：可见输出节奏。

非 boss 评估不把胜负当核心质量。已结束战斗主要看 `hp_lost`，未结束或失败战斗主要看 `enemy_damage_progress`，再结合动作质量和机制 flags。胜负只作为终止状态记录。

## 离线样本挖掘

不依赖外部 API 时，先用本地 rollout 产物挖三类资产：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.datasets.mine_offline_preferences `
  --dataset-dir STS2AI\Artifacts\llm\datasets\<rollout_dataset> `
  --include-eval
```

输出在 `<rollout_dataset>/offline_mining/`：

- `hard_cases.jsonl`：低 advantage、动作质量 flags、或明显风险样本，用于后续集中评测/人工看/必要时再交给 API teacher。
- `preference_pairs.jsonl`：同 prompt 的高低分动作对，以及规则修复动作对，可用于 DPO/ORPO。
- `repair_sft.jsonl`：保守规则能直接修复的 SFT 样本，目前只自动修 `missed_visible_lethal`。

这个阶段的原则是宁可少修，也不要把不确定的策略判断写成硬标签。比如 `dangerous_end_turn` 只统计，不自动生成修复动作。

## Teacher 标注（Kimi / Claude CLI）

Teacher 只用于高价值 hard combat，不随机烧预算。底层 provider 可以切换：

- `kimi`：走 Moonshot/Kimi API，适合实时小批量或后续 Batch。
- `claude_cli`：走本地 `claude -p`，适合消耗 Claude 额度；需要按本机代理设置 `HTTP_PROXY/HTTPS_PROXY`。

Kimi 实时入口：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
$env:MOONSHOT_API_KEY="..."
python -m llm.scripts.teacher.run_kimi_combat_review_batch `
  --provider kimi `
  --trace STS2AI\Artifacts\llm\datasets\<rollout_dataset>\step_trace.jsonl `
  --limit-episodes 20 `
  --max-api-calls 20 `
  --max-tokens 4096 `
  --thinking disabled
```

Claude CLI 入口：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
python -m llm.scripts.teacher.run_kimi_combat_review_batch `
  --provider claude_cli `
  --model claude-sonnet-4-6 `
  --claude-command claude `
  --claude-proxy http://127.0.0.1:7897 `
  --trace STS2AI\Artifacts\llm\datasets\<rollout_dataset>\step_trace.jsonl `
  --out-dir STS2AI\Artifacts\llm\reviews\<review_run> `
  --limit-episodes 20 `
  --max-api-calls 20 `
  --max-workers 2 `
  --max-tokens 4096 `
  --timeout-s 300 `
  --skip-existing
```

输入选择逻辑：按失败、invalid、掉血回合、质量 flags 排序，默认复盘整场战斗，但 prompt 聚焦前 2 回合、中期 2 回合、最后 2 回合和高掉血回合。每个 episode 都会落：

- `episode_input.json`：送给 Kimi 的结构化 combat 摘要。
- `prompt_messages.json`：实际 API messages，不含密钥。
- `teacher_raw_response.json`：原始返回；同时写 provider 专属 raw 文件。
- `turn_order_review.json`：解析成功的 JSON review。
- `teacher_turn_labels.jsonl`：Kimi 给出的候选训练标签。

`turn_order_review.json` 现在也要求包含顶层 `planner_hint`，用于训练战斗级 planner-hint LoRA：

```powershell
python -m llm.scripts.datasets.build_planner_hint_dataset `
  --review-root STS2AI\Artifacts\llm\reviews\<review_run> `
  --out-dir STS2AI\Artifacts\llm\datasets\planner_hint_<run>
```

输出不会直接训练。必须再经过：

```powershell
python -m llm.scripts.datasets.build_teacher_dataset `
  --review <turn_order_review.json> `
  --episode-input <episode_input.json> `
  --min-confidence 0.75 `
  --out-dir STS2AI\Artifacts\llm\datasets\<teacher_dataset>
```

`build_teacher_dataset` 会用原始 prompt 里的 `legal_actions` 做本地验证：非法 action、低置信度、无法匹配 step 的标签全部丢弃；默认还会把 teacher 的长 reason 改写成短的 deterministic reason，避免把错误算术或长篇复盘训练进 4B。

`self_iterate.py --kimi-teacher` 已经把这段接入单轮飞轮。`--kimi-teacher` 现在只是开关名，实际 provider 由 `--teacher-provider` 决定：

```text
rollout with planner -> audit -> pool ingest -> teacher review
        -> combat teacher dataset -> gold ingest
        -> planner-hint dataset
        -> materialize gold/silver pool
        -> train combat candidate + train planner candidate
        -> eval C/P four-cell matrix -> joint quality gate
```

带 planner-hint 和 Claude CLI teacher 的单轮命令形状：

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
  --teacher-provider claude_cli `
  --teacher-model claude-sonnet-4-6 `
  --teacher-max-workers 2 `
  --kimi-limit-episodes 20 `
  --kimi-max-api-calls 20 `
  --teacher-skip-existing
```

API 预算控制：

- `--kimi-limit-episodes` 控制本轮最多复盘多少场。
- `--kimi-max-api-calls` 是本轮新增调用上限，不受历史 usage 影响。
- `--teacher-max-workers` 控制实时并发；Claude CLI 不建议一开始拉太高，先按 2 到 4 验稳定。
- `--teacher-skip-existing` 默认开启，续跑时跳过已有 raw response / parsed review，避免重复计费。
- `--skip-episode-id` 可跳过已标注 combat。
- usage 记录在 `STS2AI/Artifacts/llm/kimi_usage/usage.jsonl`，会记录 provider、状态、耗时和 token usage，不记录密钥。

## 同局面多次推理

复杂卡牌配合不适合继续堆写死规则。对 hard case 可以让模型在同一个局面上多次独立推理，再让模型只基于当前局面和候选理由做自我复审：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.teacher.sample_state_candidates `
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
python -m llm.scripts.teacher.add_experience `
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

同一个敌人，不同 deck/relic/build 是不同任务。例如 Skada 不同楼层、不同构筑的 `SLIMES_NORMAL` 不能混在一起算 advantage，否则 reward 差异会来自 build 强弱，而不是动作质量。

当前 key 形式：

```text
CHOMPERS_NORMAL::skada_floor_07_normal::beaab971
```

## 质量门槛

候选模型必须满足：

- invalid output episode rate 不超过阈值。
- 非 boss 已结束战斗的 `hp_lost` 不能明显回退。
- 非 boss 未结束/失败战斗的 `enemy_damage_progress` 不能明显回退。
- 总体 reward 不能明显回退；reward 只是聚合指标，不单独替代 `hp_lost / enemy_damage_progress`。
- `mechanism_score` 不能明显回退。
- `missed_visible_lethal` 不能增加。
- 每个 encounter/build 的 `hp_lost`、`enemy_damage_progress`、reward 不能明显回退。
- Boss/fullrun eval 才把胜负和 Act 进度作为主指标。

这样避免平均值掩盖局部灾难。

## 推荐起步命令

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python -m llm.scripts.automation.self_train_loop `
  --current-adapter STS2AI\Artifacts\llm\sft\grouped_index_damage_20260424-223329\adapter `
  --case-index STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --run-name self_train_curriculum `
  --iterations 3 `
  --stages "CHOMPERS;SLIMES,CULTISTS;skada_floor_06,skada_floor_07,skada_floor_08" `
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
