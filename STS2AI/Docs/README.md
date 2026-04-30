# STS2AI Docs

当前有效项目文档集中放在本目录。根目录 `README.md` 只保留项目入口和当前优先级。

## 当前主线

- `llm-training-handoff.md`：当前 LLM 训练流程交接、正在跑的 run、禁用项、评估口径和下一步。
- `llm-self-train-loop.md`：LLM 自迭代、planner-hint、Guide RAG、评估门槛。
- `training-next-steps.md`：训练交接、当前数据、下一步命令。
- `llm-artifact-management.md`：LLM 产物目录、latest/current 管理约定。
- `claude-cli-use.md`：本机 Claude CLI teacher 用法。
- `design/game-bridge-current.md`：当前 game bridge 设计与范围。

## 约定

- 临时实验笔记放到 `STS2AI/Artifacts` 对应 run 目录，不散落到仓库根目录。
- 旧 `zero/replay/networkV2` 训练说明不再作为默认上下文。
- LLM 脚本入口优先看 `STS2AI/llm/scripts/README.md`。
