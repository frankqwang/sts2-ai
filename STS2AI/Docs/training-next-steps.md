# 训练下一步交接

更新时间：2026-04-29

最新可执行交接入口：

```text
STS2AI/Docs/llm-training-handoff.md
```

本文件保留较长的背景和历史命令；当前正在跑的 run、禁用项、数据池红线、Guide RAG 状态、评估口径和下一轮命令以 `llm-training-handoff.md` 为准。

## 当前判断

当前主线切到带 planner-hint 的战斗数据飞轮。非战斗选卡仍是 P0，但现在已有 planner-hint smoke LoRA，可以先让 combat adapter 在真实 rollout 里吃到战斗级 hint，再继续收 hardcase 和 teacher gold。

核心边界：

1. planner-hint LoRA 不输出动作、不输出回合动作序列，只输出战斗级策略提示。
2. combat adapter 每一步仍必须基于当前 `GameState` 和 `legal_actions` 输出单个动作。
3. 本轮 combat 训练固定 planner-hint adapter，只训练 combat candidate，避免两个 adapter 同时变化导致归因不清。
4. teacher 复盘同时沉淀两类数据：动作级 combat gold、战斗级 planner-hint SFT。
5. planner-hint adapter 低频单独训练和晋级；非 boss 同集评估不看胜负，优先看 `hp_lost / enemy_damage_progress / action_quality`。

## 当前可用数据

非战斗基线数据集：

```text
STS2AI/Artifacts/llm/datasets/skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428
```

分布：

```text
total: 18000
card_reward: 10000
map_choice: 2000
relic_select: 2000
rest_site_choice: 2000
shop_choice: 2000
```

说明：

- 保留游戏原始占位符文案，例如 `{MaxHp}`、`{Energy:energyIcons()}`。
- `current_plan` 只基于当前卡组。
- `winning_outcome_reference` 单独提供胜利局最终构筑方向，作为参考，不混入当前计划。
- `boss`、`route_ahead`、`recent_combats`、`next_risk` 已加入 prompt。

当前线上非战斗 adapter 仍是旧版：

```text
STS2AI/Artifacts/llm/sft/non_combat_skada_ironclad_v01032_2k_v2b_20260426/adapter
```

不要把旧 v2d/v2e/v2f/v2g 当正式训练输入；当前应使用 `v2h_fulltext_placeholders`。

当前可用 planner-hint smoke adapter：

```text
STS2AI/Artifacts/llm/sft/planner_hint_smoke20_kimi_20260428-2154/adapter
```

当前 combat adapter 指针：

```text
STS2AI/Artifacts/llm/current_adapter.json
```

## Kimi 选卡标注

脚本：

```text
STS2AI/data/skada/kimi_label_card_rewards.py
```

用途：

- 从 `v2h_fulltext_placeholders` 抽高价值 `card_reward` 样本。
- 让 Kimi 基于完整上下文标注 `best_action_index`。
- 输出简短 `plan_zh`、`reason_zh`，每个不超过 200 中文字。
- `action_scores` 覆盖全部候选动作，每个负例要写具体原因。

先干跑：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
python STS2AI\data\skada\kimi_label_card_rewards.py `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428 `
  --out-dir STS2AI\Artifacts\llm\datasets\skada_card_reward_kimi_dryrun_v2h_fulltext_placeholders_20260428 `
  --limit 4 `
  --group-size 2 `
  --dry-run
```

真实调用前设置环境变量，不要把 key 写入命令或日志：

```powershell
$env:MOONSHOT_API_KEY="<secret>"
```

建议第一批：

```powershell
python STS2AI\data\skada\kimi_label_card_rewards.py `
  --mode realtime `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428 `
  --out-dir STS2AI\Artifacts\llm\datasets\skada_card_reward_kimi_200_v2h_20260428 `
  --limit 200 `
  --group-size 4 `
  --max-api-calls 60 `
  --min-confidence 0.65 `
  --workers 50
```

说明：

- `realtime` 模式支持 `--workers` 并发调用和默认 `--resume-raw`，会跳过已有 `raw/group_*.json`，避免重复付费。
- 扩大到 1000 条以上时优先走 Batch API，不再用实时并发。

Batch API 准备输入：

```powershell
python STS2AI\data\skada\kimi_label_card_rewards.py `
  --mode batch-prepare `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428 `
  --out-dir STS2AI\Artifacts\llm\datasets\skada_card_reward_kimi_1000_v2h_batch_20260428 `
  --limit 1000 `
  --group-size 4
```

Batch API 提交任务：

```powershell
python STS2AI\data\skada\kimi_label_card_rewards.py `
  --mode batch-submit `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428 `
  --out-dir STS2AI\Artifacts\llm\datasets\skada_card_reward_kimi_1000_v2h_batch_20260428 `
  --limit 1000 `
  --group-size 4 `
  --completion-window 12h
```

Batch API 查询和收集：

```powershell
python STS2AI\data\skada\kimi_label_card_rewards.py `
  --mode batch-status `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428 `
  --out-dir STS2AI\Artifacts\llm\datasets\skada_card_reward_kimi_1000_v2h_batch_20260428

python STS2AI\data\skada\kimi_label_card_rewards.py `
  --mode batch-collect `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428 `
  --out-dir STS2AI\Artifacts\llm\datasets\skada_card_reward_kimi_1000_v2h_batch_20260428 `
  --limit 1000 `
  --group-size 4
```

检查输出：

```text
summary.json
labels.jsonl
invalid.jsonl
train.jsonl
eval.jsonl
```

如果 `invalid` 很多，先抽样看 `invalid.jsonl`，不要直接训练。

## 非战斗训练优先级

优先级 P0：

- 当前训练入口看 `STS2AI/Artifacts/llm/CURRENT.json`。
- 当前可用混合数据集是 `skada_non_combat_ironclad_v01032_card10k_v2h_kimi15_20260428`。
- 如果 Kimi 200 条已完成，则把 Kimi `train/eval.jsonl` 作为 gold 混入训练集，比例先控制在 10% 到 20%。
- 训练后先做离线抽样和 fullrun 小评估，再替换线上 non-combat adapter。

建议训练参数：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
C:\Users\Administrator\Desktop\sts2Zero\STS2AI\llm\.venv311\Scripts\python.exe `
  STS2AI\llm\training\sft_lora.py `
  --run-name non_combat_skada_ironclad_v01032_card10k_v2h_kimi15_20260428 `
  --dataset-dir STS2AI\Artifacts\llm\datasets\skada_non_combat_ironclad_v01032_card10k_v2h_kimi15_20260428 `
  --max-seq-length 2048 `
  --batch-size 1 `
  --grad-accum 8 `
  --num-epochs 1 `
  --load-in-4bit
```

注意：

- Windows 下 Unsloth 输出可能有编码问题，必须设置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。
- `1536` preflight 会截断较多 assistant 监督段；当前推荐 `2048`，如显存允许可用 `3072`。

## 带 planner 的战斗数据飞轮

本阶段的训练结构：

```text
combat C_t + planner P_t
  -> rollout with planner_hint
  -> audit failures / damage turns / invalid
  -> dataset pool: silver/hardcase/quarantine
  -> teacher review on hard combats
  -> combat teacher dataset -> pool gold
  -> planner_hint teacher dataset
  -> materialize gold/silver pool
  -> train combat candidate C_{t+1}
  -> train planner candidate P_{t+1}
  -> eval four cells: C_t/P_t, C_{t+1}/P_t, C_t/P_{t+1}, C_{t+1}/P_{t+1}
  -> joint quality gate
```

当前要先跑小规模一轮，验证：

- planner hint 是否稳定进入 `step_trace.jsonl` 和 prompt。
- teacher provider 是否能续跑跳过已有 episode。
- combat 训练样本是否包含 planner 上下文。
- fixed eval 是否产出 combat-only 增量、planner-only 增量和 joint 增量；非 boss case 最终按 joint 的掉血、敌方推进和动作质量决定是否一起晋级。

推荐先用 Claude CLI teacher 小批量，不碰 Kimi key：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
C:\Users\Administrator\Desktop\sts2Zero\STS2AI\llm\.venv311\Scripts\python.exe -m llm.scripts.automation.self_iterate `
  --current-adapter <combat_adapter> `
  --case-index STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --planner-hint-adapter-dir STS2AI\Artifacts\llm\sft\planner_hint_smoke20_kimi_20260428-2154\adapter `
  --planner-hint-refresh turn `
  --co-train-planner `
  --planner-min-train-rows 1 `
  --kimi-teacher `
  --teacher-provider claude_cli `
  --teacher-model claude-sonnet-4-6 `
  --teacher-max-workers 2 `
  --kimi-limit-episodes 2 `
  --kimi-max-api-calls 2 `
  --kimi-timeout-s 300 `
  --load-in-4bit `
  --no-thinking
```

产物关系：

- `runs/<run_name>/manifest.json`：单轮总控、每步日志、质量门槛结果。
- `datasets/<run_name>_rollout`：带 planner 的 rollout 数据和 trace。
- `reviews/<run_name>_kimi_teacher`：teacher 复盘结果，provider 可能是 Kimi 或 Claude CLI。
- `datasets/<run_name>_kimi_teacher`：动作级 combat gold。
- `datasets/<run_name>_planner_hint_teacher`：战斗级 planner-hint SFT 数据，本轮可直接训练 planner candidate。
- `grpo/<run_name>_candidate/adapter`：combat candidate。
- `sft/<run_name>_planner_candidate/adapter`：planner candidate。
- `evals/<run_name>_{current,candidate,planner_candidate,joint_candidate}_eval`：四格对照评估。

## 战斗侧 planner-hint LoRA

优先级 P0：planner-hint LoRA 和 combat LoRA 同步训练。planner-hint 负责战斗级指导，combat 负责当前 `legal_actions` 单步动作。

边界：

- planner-hint LoRA 输出“战斗级提示”，不是 `action_index`，也不是整回合动作序列。
- combat adapter 仍然负责每一步基于当前 `legal_actions` 选动作。
- `strategy_context` 只放 `short_term / long_term` 和 v2 `planner_hint`，不放 rule-based 策略段。
- 旧字段 `combat_plan / encounter_guide / defense_policy / resource_policy / potion_policy` 直接 invalid，不做兼容映射。
- hint 在战斗开始时生成，必要时每回合刷新；不要每个动作都生成一遍。
- guide RAG 只提供 `retrieved_knowledge` 证据块，不直接生成规则。

目标 prompt 字段形状：

```text
strategy_context:
  short_term:
    recent_actions: ...
  long_term: none
  planner_hint:
    battle_objective
    enemy_focus
    deck_usage
    risk_tradeoff
    resource_timing
    potion_stance
    kill_order
    danger_notes
```

planner-hint v2 输出：

```json
{
  "battle_objective": "Create a Vulnerable damage window before enemy scaling matters.",
  "enemy_focus": "Focus one CULTIST at a time; prefer the target with the best reachable kill.",
  "deck_usage": "Use BASH when STRIKE_IRONCLAD follow-up can exploit Vulnerable.",
  "risk_tradeoff": "Accept small HP loss only when it reduces future risk enough.",
  "resource_timing": "Do not spend 2 energy on BASH if the debuff has no attack payoff.",
  "potion_stance": "Save potions unless they prevent a major HP swing or secure a key kill.",
  "kill_order": ["enemy1", "enemy2"],
  "danger_notes": ["Do not split damage so both enemies keep scaling."]
}
```

注入后渲染成短文本：

```text
planner_hint:
  battle_objective
  enemy_focus
  deck_usage
  risk_tradeoff
  resource_timing
  potion_stance
  kill_order
  danger_notes
```

原则：

- 人类能在游戏内看到或从机制理解的信息必须给模型。
- 怪物 intent、power、relic、potion、card 的可见说明必须完整。
- hint 只指导战斗方向；当前 state、当前 `legal_actions`、实时伤害/格挡计算永远优先。
- 对复杂 boss 和机制怪优先用 guide RAG + teacher 复盘生成战斗级 hint 标签，覆盖当前 fullrun 常见失败点。

优先级 P1：Teacher 复盘整场战斗、掉血回合和未结束战斗的敌方推进。

- 非 boss 战不把 `victory` 当质量指标；已结束战斗看 `hp_lost`，未结束/失败战斗看 `enemy_damage_progress`。
- 按“整场战斗”“整回合掉血”“高推进但未结束”“低推进”抽 hardcase，不只看单动作。
- 输出 battle-level planner hint、错误原因、关键回合提示。
- 进入 planner-hint SFT 数据集；动作级修正仍进入 combat adapter 的 `gold/hardcase`。

执行约束：

- planner-hint 永远不能直接执行。
- hint 不能包含旧 `action_index`。
- hint 不能包含旧 planner 字段或未知字段。
- 若 hint 和当前 `legal_actions` 或实时计算冲突，combat adapter 必须服从当前状态。
- 旧 rule-based `plan` 不再作为默认 fallback。

已落地的代码入口：

- `STS2AI/llm/data_pipeline/planner_hint.py`：planner-hint schema、解析、安全过滤、缓存 key、输入渲染。
- `STS2AI/llm/data_pipeline/guide_knowledge.py`：本地 guide corpus 检索和 `retrieved_knowledge` 渲染。
- `STS2AI/llm/prompts/system_prompt_planner_hint.md`：planner-hint LoRA 的 system prompt。
- `STS2AI/llm/scripts/datasets/build_planner_hint_dataset.py`：从 Kimi 整场复盘构建 planner-hint SFT 数据集。
- `STS2AI/llm/scripts/datasets/build_planner_hint_seed_dataset.py`：从本地 guide knowledge 构建小型 smoke seed。
- `STS2AI/llm/scripts/analysis/check_guide_corpus.py`：guide corpus 硬校验。
- `STS2AI/llm/scripts/analysis/eval_planner_hint_outputs.py`：planner dataset/trace 硬校验。
- `STS2AI/llm/inference/llm_policy.py`：可选加载 `STS2_LLM_PLANNER_HINT_ADAPTER_DIR`，按战斗缓存 hint 并注入 combat prompt。
- `STS2AI/llm/training/grpo_rollout.py` / `STS2AI/llm/eval/policy_eval.py`：支持 `--planner-hint-adapter-dir`，rollout 和固定评估可复用同一套 planner-hint adapter。

运行时启用：

```powershell
$env:STS2_LLM_PLANNER_HINT_ADAPTER_DIR="C:\Users\Administrator\Desktop\sts2Zero\STS2AI\Artifacts\llm\sft\<planner_hint_run>\adapter"
$env:STS2_LLM_PLANNER_HINT="1"
$env:STS2_LLM_PLANNER_HINT_REFRESH="turn"
$env:STS2_LLM_GUIDE_RAG="1"
```

构建 planner-hint SFT 数据：

```powershell
python -m llm.scripts.datasets.build_planner_hint_dataset `
  --review-root STS2AI\Artifacts\llm\reviews\<review_run> `
  --out-dir STS2AI\Artifacts\llm\datasets\planner_hint_<run>
```

硬校验：

```powershell
python -m llm.scripts.analysis.check_guide_corpus `
  --corpus STS2AI\llm\knowledge\guide_corpus.jsonl

python -m llm.scripts.analysis.eval_planner_hint_outputs `
  --dataset STS2AI\Artifacts\llm\datasets\planner_hint_<run> `
  --require-knowledge
```

带 planner-hint 的 combat rollout / eval：

```powershell
python -m llm.training.grpo_rollout `
  --adapter-dir STS2AI\Artifacts\llm\grpo\<combat_run>\adapter `
  --case-index STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --planner-hint-adapter-dir STS2AI\Artifacts\llm\sft\<planner_hint_run>\adapter `
  --planner-hint-refresh turn

python -m llm.eval.policy_eval `
  --adapter-dir STS2AI\Artifacts\llm\grpo\<combat_run>\adapter `
  --case-index STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --planner-hint-adapter-dir STS2AI\Artifacts\llm\sft\<planner_hint_run>\adapter `
  --planner-hint-refresh turn
```

## 推荐顺序

1. 先跑一轮带 planner-hint 的 combat 自迭代小样本，验证 trace、teacher、combat train、fixed eval 都能闭环。
2. 检查 `planner_hint_teacher` 和 `kimi_teacher` 两套数据质量；如果 teacher 输出不稳，先修 prompt/解析，不扩大训练。
3. 用通过检查的 teacher gold 扩大 combat pool，继续跑 CULTISTS/SLIMES/CHOMPERS 小课程。
4. planner-hint teacher 数据累计到 100 条以上后，再训练下一版 planner-hint LoRA，并用固定 combat adapter 做同集评估。
5. 并行补非战斗选卡 Kimi/Batch 数据，训练新 non-combat LoRA，接 fullrun 小评估。
6. 如果 fullrun 的 boss 前卡组、Act 进度、关键战斗损耗和 Boss 战表现都改善，再扩大 card_reward 到 1000 条以上。

## 接手注意

- 不要使用 `v2d/v2e/v2f/v2g` 做正式训练。
- `v2h_fulltext_placeholders` 是当前非战斗基线。
- 当前 Kimi 脚本不会把 API key 写入产物。
- 未完成真实 Kimi 调用，因为当前环境没有 `MOONSHOT_API_KEY`。
- 训练产物和数据都在 `STS2AI/Artifacts/llm`，不要把临时交接文件放仓库根目录。
