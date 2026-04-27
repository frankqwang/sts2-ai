# AGENTS.md

## 当前范围

- 当前项目主题只保留两块：`STS2AI/bridge/game_bridge` 与 `STS2AI/data/skada`。
- 旧的训练主线、networkV2、离线 ranking、teacher/diagnostics 等说明视为过时，不再作为默认上下文。

## 通用协作规范

- 文档和说明默认用中文。
- 除根目录 `README.md` 外，其它项目文档统一放在 `STS2AI/Docs`。
- 在 Codex 回复里引用本地文件时，Windows 绝对路径前必须带 `/`，并尽量附行号。
- 不要把交接记录、临时说明、实验笔记散落到仓库根目录。
- 尽量使用 UTF-8 读写文件；PowerShell 读取文本时优先显式指定 `-Encoding utf8`。
- 耗时任务默认后台执行，并在回复中说明 PID、日志路径和输出目录。

## 代码边界

- `src` 下是上游/反编译源码，默认不修改；如果确实需要改，先告知用户。
- 日常可维护代码默认在 `STS2AI` 下完成。

## 目录约定

- `STS2AI/bridge/game_bridge`：bridge、session、spectate、sim 相关 Python 代码。
- `STS2AI/data/skada`：skada 抓取、明细构建、覆盖率扫描等数据脚本。
- `STS2AI/Artifacts`：运行产物、日志、临时输出。
- `STS2AI/Docs`：项目文档。
