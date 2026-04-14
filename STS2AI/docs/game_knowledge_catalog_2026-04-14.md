# 游戏知识导出目录说明

这次补的是一条独立于 `build_source_database.py` 的导出链，目标不是替代 `source_knowledge.sqlite`，而是补一份更适合人和模型直接阅读、查询的目录。

产物默认放在：

- `Assets/datasets/game_knowledge_catalog/`

包含：

- `cards.jsonl`
- `relics.jsonl`
- `potions.jsonl`
- `monsters.jsonl`
- `game_knowledge_catalog.sqlite`
- `manifest.json`

另外，如果跑了运行时导出，还会在：

- `STS2AI/Python/data/raw/card_runtime_texts.json`

生成卡牌运行时文本原始文件，随后被并回 catalog。

## 数据来源

这份目录是把两类数据拼起来导出的：

1. `STS2AI/Python/data/source_knowledge.sqlite`
   这里提供结构化元数据，例如：
   - cost / rarity / target_type
   - tags / keywords / powers / commands
   - monster intents / move labels

2. `localization/{eng,zhs}/*.json`
   这里提供自然语言文本，例如：
   - 卡牌标题和描述
   - 遗物标题、描述、flavor
   - 药水标题和描述
   - 怪物名字与 move title

## 为什么单独做这一份

`source_knowledge.sqlite` 的设计目标是训练特征，不带自然语言文本；这份导出目录的目标是：

- 方便直接按名称、描述、tag 查找
- 方便做卡牌 / 遗物 / 药水 / 怪物知识查询
- 方便后面接分析脚本、RAG、轻量检索工具

所以两者职责不同：

- `source_knowledge.sqlite`：更适合训练特征和符号特征
- `game_knowledge_catalog`：更适合阅读、查询、分析

## 当前字段口径

### cards

每条记录包含：

- `id`
- `class_name`
- `source_path`
- `title_en/title_zhs`
- `description_en/description_zhs`
- `upgrade_preview_static_en/upgrade_preview_static_zhs`
- `description_runtime_en/description_runtime_zhs`
- `upgrade_preview_runtime_en/upgrade_preview_runtime_zhs`
- `cost/card_type/rarity/target_type`
- `tags/keywords/card_tags/powers/dynamic_vars/commands`
- `keyword_details`
- `has_upgrade_tokens`

说明：

- 当前导出的卡牌描述是本地化基础文本。
- 如果文本里带 `IfUpgraded` 这类条件标记，会记录 `has_upgrade_tokens=true`，并额外给出 `upgrade_preview_static_*`。
- `upgrade_preview_static_*` 是静态升级预览：会把 `IfUpgraded:show:升级态|普通态` 这类文本分支直接展开成升级态。
- 这还不是完整运行时升级描述，因为像 `{Damage:diff()}`、`{Block:diff()}` 这类动态变量仍需要运行时 `CardModel.GetDescriptionForUpgradePreview()` 才能拿到精确数值。
- 如果 `STS2AI/Python/data/raw/card_runtime_texts.json` 存在，导出脚本会自动把运行时文本并回同一条卡牌记录：
  - `description_runtime_*`
  - `upgrade_preview_runtime_*`

### relics

每条记录包含：

- `title_en/title_zhs`
- `description_en/description_zhs`
- `flavor_en/flavor_zhs`
- `rarity`
- `powers/dynamic_vars/commands`

### potions

每条记录包含：

- `title_en/title_zhs`
- `description_en/description_zhs`
- `rarity`
- `usage`
- `target_type`
- `powers/commands`

### monsters

每条记录包含：

- `name_en/name_zhs`
- `min_initial_hp_expr/max_initial_hp_expr`
- `intents`
- `move_labels`
- `moves`
- `powers/commands`

其中 `moves` 会把 `move label` 和本地化的 `move title` 对齐起来。

## 查询方式

补了一个轻量查询脚本：

- [query_game_knowledge.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/data/query_game_knowledge.py:1)

如果需要先刷新运行时卡牌文本，再重建 catalog，可以执行：

```powershell
python STS2AI/Python/data/export_runtime_card_texts.py
python STS2AI/Python/data/export_game_knowledge_catalog.py
```

示例：

```powershell
python STS2AI/Python/data/query_game_knowledge.py offering
python STS2AI/Python/data/query_game_knowledge.py vigor --entity relics
python STS2AI/Python/data/query_game_knowledge.py architect --entity monsters
```

## 后续可以再补什么

如果后面需要更完整的卡牌知识库，最值得补的是运行时导出：

- 升级预览描述
- 更准确的动态变量渲染
- 运行时 compendium 实际展示文本

但这一步不需要改 `src` 才能先把静态目录搭起来；当前这版已经足够支持：

- 知识查询
- case mining
- build 解释
- 后续轻量 RAG/检索增强
