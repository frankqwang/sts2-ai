# legacy_research 说明

这里存放已经退出主线的研究代码与历史实验脚本。

当前归档原则：

- 不再作为正式入口。
- 不允许 `game_bridge` 主线依赖这里的任何模块。
- 仅用于回溯旧实现、提取历史实验参数、或迁移残留逻辑时参考。

这批代码的共同特征：

- 以 `networkV2` 训练主线为中心组织。
- 强耦合旧的 featurizer / PPO / teacher / rollout runtime。
- 不再符合当前“运行时核心独立、策略层可插拔”的平台边界。

新的主线位置：

- 运行时平台：`STS2AI/Python/game_bridge`
- 主文档：`STS2AI/docs/2026-0419-game-bridge-platform-reorg.md`

使用约束：

- 新代码不要再从这里 import。
- 如果未来要复用其中某段逻辑，优先复制到 `game_bridge` 或新的研究包，再补测试。
- 不要在这里继续叠加新功能。
