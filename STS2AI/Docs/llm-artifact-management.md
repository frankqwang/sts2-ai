# LLM 产物管理

更新时间：2026-04-28

## 只看当前指针

LLM 目录下的时间戳文件夹默认都是历史归档。人工判断“哪个最新可用”时只看：

```text
STS2AI/Artifacts/llm/CURRENT.json
```

当前可训练的非战斗数据集是：

```text
STS2AI/Artifacts/llm/datasets/skada_non_combat_ironclad_v01032_card10k_v2h_kimi15_20260428
```

它已经把 v2h 基线和 Kimi 选卡 gold 混好，`train=20118`、`eval=909`，Kimi gold 在 train 中占 `15%`。

## 当前可用数据

### 训练入口

```text
STS2AI/Artifacts/llm/datasets/skada_non_combat_ironclad_v01032_card10k_v2h_kimi15_20260428
```

用途：下一轮 non-combat LoRA 训练。

推荐训练长度：

- `max_seq_length=2048`：preflight 通过，assistant 保留率 `0.996421`。
- `max_seq_length=3072`：全保留，但更吃显存和时间。
- 不建议继续用 `1536`，因为会截掉较多 assistant 监督段。

### 基线数据

```text
STS2AI/Artifacts/llm/datasets/skada_non_combat_ironclad_v01032_card10k_v2h_fulltext_placeholders_20260428
```

用途：重建混合数据集、对照、回滚。不要再用 v2d/v2e/v2f/v2g 做正式训练。

### Kimi 选卡 gold

```text
STS2AI/Artifacts/llm/datasets/skada_card_reward_kimi_200_v2h_20260428
```

用途：Kimi card_reward gold 源数据。结果为 `labels=199`、`invalid=1`、`train=190`、`eval=9`。

这批是真实普通接口请求，不是 Batch API：

```text
successful_responses=50
prompt_tokens=186873
completion_tokens=63067
total_tokens=249940
web_search_calls=0
```

## 命名规则

以后新产物继续保留时间戳目录，但必须同步更新 `CURRENT.json`：

- 当前训练集：`skada_non_combat_..._<版本>_<信号>_<日期>`
- Kimi 小批实时标注：`skada_card_reward_kimi_<数量>_<版本>_<日期>`
- Kimi 大批 Batch 标注：`skada_card_reward_kimi_<数量>_<版本>_batch_<日期>`
- smoke/dryrun：名字必须包含 `smoke` 或 `dryrun`，默认不可训练。

## 清理规则

默认不删除历史产物。清理时按优先级处理：

1. 可以删：`*smoke*`、`*dryrun*`、明显失败且无后续引用的临时目录。
2. 先别删：`CURRENT.json` 里引用的所有目录。
3. 先别删：已训练 adapter 的 dataset 来源目录，除非已经在 `run_meta.json` 和归档里记录清楚。

## 下一步

1. 用当前训练集启动 non-combat LoRA。
2. fullrun 小评估后，如果选卡有效，扩大 Kimi card_reward 到 1000 条。
3. 1000 条以上走 Batch API，不再用实时并发。
