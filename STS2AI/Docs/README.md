# STS2AI Docs

项目文档只保留当前维护中的实现说明。旧训练主线、旧观战入口、旧 zero/replay 实验交接已经删除，不再作为实现依据。

## 当前有效文档

- `design/game-bridge-current.md`：sim pipe protobuf 与 spectator HTTP protobuf JSON 的当前 bridge 架构。
- `llm-self-train-loop.md`：LLM 战斗训练飞轮、rollout、audit、dataset pool、晋级门槛。
- `training-next-steps.md`：当前训练优先级、非战斗数据、Kimi 标注、planner LoRA 交接计划。

## 当前代码主线

- `STS2AI/llm`：LLM prompt、推理、rollout、训练、评估、dataset pool。
- `STS2AI/bridge/game_bridge`：bridge、session、spectate、sim 相关 Python 代码。
- `STS2AI/data/skada`：Skada 抓取、明细构建、combat/non-combat 数据脚本。
- `STS2AI/Artifacts/llm`：训练数据、评估、trace、adapter、长期 dataset pool 产物。

新增项目文档请放在 `STS2AI/Docs`，不要散落在仓库根目录。
