# AGENTS.md

## Global Repo Rules
- 文档用中文。
- Local clickable file links in Codex responses must use a leading `/` before the absolute Windows path.
- Correct format: `[train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:172)`
- Wrong format: `[train_hybrid.py](C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:172)`
- 引用代码时尽量带行号。
- 除 `README` 外，项目文档统一放在 `STS2AI/docs`。
- 交接文档、协议、工作记录等不要放在仓库根目录。
- 不要修改 `src` 下的任务源码；如果一定要改，先告知用户。仓库里可维护代码在 `STS2AI` 下。
- 为避免中文编码问题：文档内容可用中文，但文件名、脚本名、路径名尽量使用 ASCII；各种工具里尽量使用 UTF-8。
- 比如PowerShell Get-Content等读写操作，都用utf-8
- 耗时任务默认放到后台执行，并在回复里明确 PID、日志和输出目录。

## Training And Offline Data Rules
- 训练主线和离线非战斗数据生成必须共用同一套环境推进语义；不要在不同脚本里各写一套 `screen/action` 状态机判断。
- 任何“从当前 `state` 继续往前走”的逻辑，优先复用 `STS2AI/Python/runtime/full_run_action_semantics.py`；允许单独实现的只有 `save/load snapshot`、分支搜索、分支分数聚合。
- 对外脚本入口统一使用这 3 个：
  - `STS2AI/Python/train_hybrid.py`
  - `STS2AI/Python/search/generate_offline_noncombat_ranking_data.py`
  - `STS2AI/Python/skada/build_offline_noncombat_ranking_from_skada.py`
- 旧命名如 `matchup_*`、`generate_card_ranking_data.py`、`build_matchup_ranking_from_skada.py` 只保留兼容；新文档、新实验记录、新命令默认使用 `offline_noncombat_ranking` 命名。
- 训练期沉淀的 `offline_data/*.pt` 视为优质 episode 资产库，优先用于后续 BC、teacher、数据挖掘；它不是现成的 ranking 数据，不能直接当 `offline_noncombat_ranking` 数据集使用。
- 如果后续要把训练沉淀的优质 episode 转成 ranking 数据，必须补“关键 screen 快照 + 后验重标注”这层派生，不要直接把在线 rollout 当 ranking 样本硬喂。

## Outcome Vocabulary Rules
- `run_outcome` / `outcome` 的业务口径必须走共享常量和归一 helper，统一复用 `STS2AI/Python/runtime/run_outcome_vocab.py`。
- 不允许在训练、离线生成、窗口筛选、分析脚本里各自手写 `"death"`、`"defeat"`、`"loss"`、`"win"` 之类的字符串判断。
- 允许底层协议或 simulator 原始状态继续保留自己的值，例如 wire format 里的 `defeat`；但上层摘要、筛选、统计、门控必须先归一再使用。
- 新增 outcome 相关逻辑时，优先使用：
  - `normalize_run_outcome(...)`
  - `is_victory_outcome(...)`
  - `is_failure_outcome(...)`
- 如果发现某条链路仍然直接比较原始 outcome 字符串，应优先修成共享 helper，而不是在调用点继续追加兼容分支。
