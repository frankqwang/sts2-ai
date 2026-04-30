# LLM 训练流程交接

更新时间：2026-04-29 12:20 Asia/Shanghai

## 当前结论

当前主线是 `Skada combat reset -> planner-hint -> combat LoRA -> teacher -> planner/combat 同轮小步训练`。先不要扩大到 fullrun，也不要使用旧的长期 dataset pool 训练。

核心原则：

1. 只用 Skada case / fullrun reset 数据。旧手工 Act1 pool 已删除，不再恢复。
2. 非 boss 战不要用胜负判断质量；看 `hp_lost / enemy_damage_progress / action_quality / mechanism_score`。
3. planner-hint LoRA 输出战斗级指导，不输出动作，不输出 turn 序列。
4. combat LoRA 每一步只根据当前 `GameState` 和 `legal_actions` 输出一个动作。
5. 当前默认 `dataset_pool` 已混入旧 prompt 和旧数据，暂时禁止从它 materialize 正式训练集。
6. 当前 `guide_corpus.jsonl` 只是 RAG smoke seed，质量不够，正式训练先关掉 Guide RAG。

## 当前最新训练

run：

```text
combat_skada_clean_rollout_iter01_20260429-1155
```

后台主进程：

```text
PID 39068（已结束）
```

run 目录：

```text
STS2AI/Artifacts/llm/runs/combat_skada_clean_rollout_iter01_20260429-1155
```

日志目录：

```text
STS2AI/Artifacts/llm/runs/combat_skada_clean_rollout_iter01_20260429-1155/logs
```

当前状态：已完成。`manifest.json` 里 `status=completed`，`promotion.json` 里 `passed=true`，但本轮没有传 `--promote`，所以没有写 current 指针。是否晋级仍要人工看 trace。

本轮重要参数：

- combat base：`STS2AI/Artifacts/llm/grpo/combat_strict_v2_quality_iter02_20260429-104230_candidate/adapter`
- planner base：`STS2AI/Artifacts/llm/sft/combat_planner_cotrain_kimi1_en_20260428-231341_planner_candidate/adapter`
- `case-limit=8`
- `rollout-generations=2`
- Kimi teacher 最多 4 次 API
- `--skip-pool-ingest`
- `--no-train-from-pool-after-teacher`
- `STS2_LLM_GUIDE_RAG=0`
- 没有传 `--promote`，所以不会自动更新 current 指针。

已知结果：

- rollout：`total_episodes=16`、`total_samples=181`、`train_size=172`、`eval_size=9`
- rollout 有 1 个 `invalid_output:dangerous_self_damage`
- `hp_lost_avg=3.0625`
- `enemy_damage_progress_avg=0.9926`
- `mechanism_score_avg=0.9948`
- planner-hint teacher dataset：`rows=4`、`train=3`、`eval=1`
- combat teacher dataset：`rows=2`、`train=2`、`eval=0`
- joint eval：`episodes=8`、`hp_lost.avg=0.0`、`enemy_damage_progress.avg=1.0`、`mechanism_score.avg=0.9917`、`defense_score.avg=1.0`、`invalid_output_episode_rate=0.0`
- joint eval 仍有 `reason_claims_lethal_but_action_not_lethal=1`，需要 trace 抽查。

候选 adapter：

```text
STS2AI/Artifacts/llm/grpo/combat_skada_clean_rollout_iter01_20260429-1155_candidate/adapter
STS2AI/Artifacts/llm/sft/combat_skada_clean_rollout_iter01_20260429-1155_planner_candidate/adapter
```

检查状态：

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*combat_skada_clean_rollout_iter01_20260429-1155*' } |
  Select-Object ProcessId,ParentProcessId,Name,CommandLine

Get-ChildItem STS2AI\Artifacts\llm\runs\combat_skada_clean_rollout_iter01_20260429-1155\logs |
  Sort-Object LastWriteTime -Descending |
  Select-Object Name,Length,LastWriteTime

Get-Content STS2AI\Artifacts\llm\runs\combat_skada_clean_rollout_iter01_20260429-1155\manifest.json -Encoding utf8
```

看最终质量：

```powershell
Get-Content STS2AI\Artifacts\llm\runs\combat_skada_clean_rollout_iter01_20260429-1155\promotion.json -Encoding utf8
Get-Content STS2AI\Artifacts\llm\evals\combat_skada_clean_rollout_iter01_20260429-1155_joint_candidate_eval\metrics.json -Encoding utf8
```

注意：如果 `promotion.json` 不存在，说明 run 还没收尾或失败。不要只看 stdout 最后一行。

## 当前不要使用的产物

上一轮 run：

```text
combat_skada_promptflat_flywheel_20260429-1130
```

这轮虽然写出了 candidate 和 eval，但训练集来自旧 `dataset_pool` materialize。该 materialized 数据里混有旧 prompt、旧手工池残留和 `floor=0 / encounter_id=unknown` 数据，只能当 smoke，不当正式晋级依据。

默认长期池：

```text
STS2AI/Artifacts/llm/dataset_pool
```

当前不要用它训练。问题：

- `selected` 样本混入旧 `bootstrap_combat_pool_strict_2000_20260427-1200`
- 大量旧 `strategy_context: memory:` prompt
- 部分 `floor=0`
- 部分 `encounter_id=unknown`
- 与当前 prompt schema 不一致

下一步应新建干净池，例如：

```text
STS2AI/Artifacts/llm/dataset_pool_skada_v2_20260429
```

新池必须硬要求：

- `source=skada_case`
- `case_id`
- `case_index`
- `run_id`
- `act`
- `floor > 0`
- `encounter_id`
- `encounter_key`
- `prompt_schema_version`
- 当前 `strategy_context` 扁平格式

缺字段直接拒绝，不做兼容。

## 当前 prompt 约定

combat prompt 当前结构：

```text
run: char=IRONCLAD act=1 floor=...
strategy_context:
  short_term:
    recent_actions: ...
  long_term: none
  planner_hint:
    battle_objective: ...
    enemy_focus: ...
    deck_usage: ...
    risk_tradeoff: ...
    resource_timing: ...
    potion_stance: ...
    kill_order: ...
    danger_notes: ...

player: ...
...
piles:
  draw=4 discard=5 exhaust=0
  draw_cards: ...
...
hand:
  ...

legal_actions:
  ...
```

注意：

- 不再有 `agent_memory:` 包装层。
- 不再有旧 `strategy_context: memory / threat / target / turn / rule / plan`。
- `piles` 不能只有数字，必须尽量带 `draw_cards / discard_cards / exhaust_cards`。
- `player:` 和 `legal_actions:` 前保留空行，方便 trace 阅读。
- `legal_actions` 不再重复输出 `hp=... lethal=false`；只有确实斩杀才允许 `lethal=true`。
- 旧 planner 字段 `combat_plan / encounter_guide / defense_policy / resource_policy / potion_policy` 直接 invalid。

planner-hint v2 schema：

```json
{
  "battle_objective": "...",
  "enemy_focus": "...",
  "deck_usage": "...",
  "risk_tradeoff": "...",
  "resource_timing": "...",
  "potion_stance": "...",
  "kill_order": ["enemy1", "enemy2"],
  "danger_notes": ["..."]
}
```

planner-hint 禁止：

- `action_index`
- 具体动作序列
- turn-by-turn 指令
- 中文输出
- 旧字段
- 与当前 `legal_actions` 绑定的动作结论

## 标准小步训练流程

环境固定用 `venv311`：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
$env:UNSLOTH_RETURN_LOGITS="1"
```

Kimi key 只放进当前进程环境变量，不写代码、不写文档、不写命令历史。Claude CLI 需要代理：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
```

当前建议训练命令模板：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
$env:UNSLOTH_RETURN_LOGITS="1"
$env:STS2_LLM_GUIDE_RAG="0"
$env:MOONSHOT_API_KEY="<secret>"

C:\Users\Administrator\Desktop\sts2Zero\STS2AI\llm\.venv311\Scripts\python.exe -m llm.scripts.automation.self_iterate `
  --current-adapter C:\Users\Administrator\Desktop\sts2Zero\STS2AI\Artifacts\llm\grpo\combat_strict_v2_quality_iter02_20260429-104230_candidate\adapter `
  --python-exe C:\Users\Administrator\Desktop\sts2Zero\STS2AI\llm\.venv311\Scripts\python.exe `
  --run-name combat_skada_clean_rollout_iterNN_<date> `
  --case-index C:\Users\Administrator\Desktop\sts2Zero\STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1\cases.jsonl `
  --case-character IRONCLAD `
  --case-floor-min 1 `
  --case-floor-max 12 `
  --case-limit 8 `
  --case-sample-mode stratified `
  --rollout-generations 2 `
  --rollout-max-steps 60 `
  --eval-episodes-per-encounter 1 `
  --eval-max-steps 60 `
  --parse-retries 1 `
  --load-in-4bit `
  --no-thinking `
  --planner-hint-adapter-dir C:\Users\Administrator\Desktop\sts2Zero\STS2AI\Artifacts\llm\sft\combat_planner_cotrain_kimi1_en_20260428-231341_planner_candidate\adapter `
  --planner-hint-refresh turn `
  --planner-hint-max-new-tokens 240 `
  --kimi-teacher `
  --teacher-provider kimi `
  --kimi-limit-episodes 4 `
  --kimi-max-api-calls 4 `
  --kimi-timeout-s 240 `
  --kimi-max-decision-state-chars 3500 `
  --kimi-damage-turns 2 `
  --kimi-min-review-ok-rate 0 `
  --kimi-min-teacher-rows 0 `
  --co-train-planner `
  --planner-min-train-rows 1 `
  --planner-num-epochs 1 `
  --planner-batch-size 1 `
  --planner-grad-accum 4 `
  --planner-lr 0.0001 `
  --planner-max-seq-length 2048 `
  --num-epochs 1 `
  --batch-size 1 `
  --grad-accum 4 `
  --lr 8e-05 `
  --max-seq-length 2048 `
  --grpo-loss-scope assistant `
  --skip-pool-ingest `
  --no-train-from-pool-after-teacher
```

后台跑时用 `Start-Process -WindowStyle Hidden`，日志写到 run 目录。不要在前台长时间阻塞。

## 训练结束后的检查顺序

1. 看 `runs/<run>/manifest.json`：
   - `status` 是否 `completed`
   - `failed_step` 是否为空
   - `training_dataset_dir` 是否是本轮 rollout，而不是旧 `managed_pool_train`
   - `skip_pool_ingest` 是否为 `true`
   - `planner_candidate_trained` 是否符合预期

2. 看 rollout：
   - `datasets/<run>_rollout/meta.json`
   - `total_episodes`
   - `total_samples`
   - `invalid_outputs`
   - `hp_lost_avg`
   - `enemy_damage_progress_avg`
   - `action_quality`
   - `encounter_ids`

3. 看 teacher：
   - `reviews/<run>_kimi_teacher/summary.json`
   - `api_calls_used`
   - `reviews_ok`
   - `labels`
   - `parse_counts`
   - `status_counts`

4. 看 planner dataset：
   - `datasets/<run>_planner_hint_teacher/summary.json`
   - `rows`
   - `invalid`
   - 是否有旧字段

5. 看 combat train：
   - `grpo/<run>_candidate/adapter`
   - `runs/<run>/logs/train.stdout.log`
   - `train_loss`
   - `eval_loss`
   - `train_steps_per_second`

6. 看四格 eval：
   - `evals/<run>_current_eval/metrics.json`
   - `evals/<run>_candidate_eval/metrics.json`
   - `evals/<run>_planner_candidate_eval/metrics.json`
   - `evals/<run>_joint_candidate_eval/metrics.json`

非 boss 战只报告：

- `hp_lost.avg/max`
- `enemy_damage_progress.avg/min`
- `mechanism_score.avg`
- `defense_score.avg`
- `action_quality`
- `invalid_output_episode_rate`
- `duration_s`

不要用“几场全胜”作为主要结论。

## Trace Viewer

生成 trace viewer：

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"
C:\Users\Administrator\Desktop\sts2Zero\STS2AI\llm\.venv311\Scripts\python.exe -m llm.scripts.viewers.trace_viewer_html `
  --trace STS2AI\Artifacts\llm\evals\<eval_run>\step_trace.jsonl `
  --out STS2AI\Artifacts\llm\evals\<eval_run>\trace_viewer.html
```

检查重点：

- `run: act/floor/encounter` 不应是 `0/?`
- `strategy_context` 不应出现 `agent_memory`
- `short_term/long_term` 层级应直接在 `strategy_context` 下
- `piles` 应显示卡名
- `retrieved_knowledge` 当前正式训练应为空，因为 Guide RAG 关闭
- `planner_hint` 不应是中文
- `planner_hint` 不应包含动作或回合序列
- `legal_actions` 不应塞无用 `hp=... lethal=false`

## Guide RAG 状态

文件：

```text
STS2AI/llm/knowledge/guide_corpus.jsonl
```

当前只有 20 条，已经人工 review：它只能验证链路，不适合作为正式攻略知识库。问题：

- 覆盖不够，例如 `SLIMES_WEAK / TOADPOLES_WEAK / FUZZY_WURM_CRAWLER_WEAK` 没有足够实体知识。
- 部分内容太泛，不够可执行。
- `BASH/Vulnerable` 重复条目过多，会挤占 top-k。
- 有 STS1 来源的 `HAND_DRILL`，没有 `source_game` 区分。
- boss 泛 tag 容易误召回其它 boss。

正式训练前保持：

```powershell
$env:STS2_LLM_GUIDE_RAG="0"
```

后续要重建 `guide_corpus_v2`，从 Skada 当前 `encounter_id/card/relic/potion/power` 覆盖出发，补 `aliases / game_version / source_game / conditions / applies_when / avoid_when / priority`。

## Teacher 注意事项

Kimi：

- 小批实时可以用 `--kimi-max-api-calls` 控预算。
- 大批选卡标注走 Batch API，不要用实时并发烧钱。
- 重跑必须 `--skip-existing`，避免重复请求。
- key 只放环境变量，不写入文件。

Claude CLI：

- 本地命令参考 `STS2AI/Docs/claude-cli-use.md`。
- 必须设置代理。
- 整场复盘 prompt 大，300s 可能超时。优先压缩 prompt 或减少 episode，不要盲目扩大并发。

teacher 输出硬要求：

- 动作标签必须能在原始 `legal_actions` 找到。
- planner hint 必须是 v2 schema。
- 不满足字段、语言、schema、legal action 的数据直接 invalid，重新标，不做降级兼容。

## 指针和晋级

`self_iterate --promote` 只会写：

```text
STS2AI/Artifacts/llm/current_adapter.json
STS2AI/Artifacts/llm/current_planner_hint_adapter.json
```

它不会自动更新：

```text
STS2AI/Artifacts/llm/CURRENT.json
```

`CURRENT.json` 目前主要记录 non-combat 训练集和 adapter 指针。后续应改成统一 registry，但在此之前，combat/planner 晋级后必须人工同步文档和指针。

当前这轮没有 `--promote`，所以不会自动晋级。即使 `promotion.passed=true`，也要先人工检查 trace 和非 boss 质量指标。

## 下一步建议

1. 等 `combat_skada_clean_rollout_iter01_20260429-1155` 完成。
2. 看 `promotion.json` 和 joint eval metrics。
3. 生成 joint eval trace viewer，人工抽查 prompt 和 planner hint。
4. 如果质量可以，下一轮保持 `--skip-pool-ingest`，把 `case-limit` 从 8 提到 16。
5. 同时实现新的 `dataset_pool_skada_v2`，硬过滤 Skada/current prompt schema。
6. 新池稳定后，再恢复 pool materialize 训练。
7. Guide RAG 在 `guide_corpus_v2` 完成前继续关闭。
8. 非战斗 LoRA 仍按 `CURRENT.json` 里的 v2h/Kimi gold 线推进，但不要和 combat 当前调试混在一个 run 里。
