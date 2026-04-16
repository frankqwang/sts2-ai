# AGENTS.md

## Global Repo Rules
- 文档用中文。sts2ai/docs下面放文档，文档上面开头用2026-0416日期开头，好判断时效性
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

## STS2AI Directory Rules
- `STS2AI/Python` 放可执行脚本、训练代码、分析代码、导出代码；不要把运行产物、手工数据和大文件直接堆在这里。
- `STS2AI/docs` 放项目文档、问题记录、流程说明、评审摘要、参数说明；除 `README` 外，不要把说明文档散落到别的目录。
- `STS2AI/Artifacts` 放运行产物：训练输出、日志、分析报告、teacher refresh 输出、临时实验目录等。凡是“跑一次就会再生成”的东西，优先放这里。
- `Assets/datasets` 放可复用的数据资产与知识库成品，例如查询库、桥接后的离线数据集、静态导出目录。凡是“希望跨实验长期复用”的数据，优先放这里。
- `STS2AI/Python/data` 放数据脚本、上游原始结构化底库、轻量元数据，以及导出流程依赖的源码侧输入；不要把面向消费的大型成品数据默认落在这里。
- `STS2AI/Python/data/raw` 放导出流程的中间原始文件；如果中间文件后续会稳定复用、且不只是流水线临时缓存，再考虑提升到 `Assets/datasets`。

## Data Classification Rules
- 区分“原始底库”“中间文件”“运行产物”“长期数据资产”四类数据，不要混放。
- 原始底库：例如 `STS2AI/Python/data/source_knowledge.sqlite`。它是上游生成输入，保留为可再生产底库，不作为默认查询成品目录。
- 中间文件：例如运行时导出的 `STS2AI/Python/data/raw/card_runtime_texts.json`。它可以被下游消费，但默认视为导出中间态。
- 运行产物：例如训练 run、日志、窗口分析、teacher refresh 比较报告，统一放 `STS2AI/Artifacts`。
- 长期数据资产：例如 `Assets/datasets/game_knowledge_catalog`、桥接后的 `offline_noncombat_ranking` 数据集。它们应当可直接查询、复用、被后续训练/分析引用。
- 新数据如果同时满足“可再生成”和“长期复用”，默认保留两层：
  - 上游生成脚本和原始输入留在 `STS2AI/Python/data`
  - 面向使用者的成品目录落到 `Assets/datasets`

## Dataset Output Defaults
- 新增导出脚本时，先明确默认输出属于哪一类数据；不要为了省事把所有输出都放在脚本同目录。
- 查询导向、知识库导向、跨实验复用的数据集，默认输出到 `Assets/datasets/<name>`。
- 单次实验、窗口分析、teacher loop 刷新、A/B 对照等任务产物，默认输出到 `STS2AI/Artifacts/...`。
- 若某份数据已经在 `Assets/datasets` 有正式成品目录，后续文档、查询脚本、分析脚本默认引用成品目录，不再默认引用脚本目录下的旧副本。
- 避免在数据本体里加入只为查询方便的混合语义字段；如果跨表统一展示有需求，优先在 SQL / 查询脚本里用 `AS` 或视图解决，而不是污染原始导出字段。
