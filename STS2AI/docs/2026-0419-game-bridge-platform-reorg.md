# Runtime 平台重组说明

## 背景

当前仓库的长期可复用部分，实际上集中在三类能力：

- HeadlessSim 启动与生命周期管理
- Python 到 sim 的 pipe/proto 连接与状态转换
- 脱离训练框架的会话 API 与观战驱动

原先这部分能力散落在 `networkV2/s0_bridge`、`networkV2/s8_spectate`，并被训练脚本、PPO、teacher、特征编译逻辑反向耦合。  
这次重组的目标，是把运行时核心剥离成一个独立平台包，后续不论接什么策略、搜索器、离线数据生成器，都统一复用这一层。

## 新主线目录

新的主线固定在 `STS2AI/Python/game_bridge`，按职责拆成：

- `game_bridge/sim`
  - sim 路径常量
  - host freshness 检查
  - auto-launch / stop / ready 等进程生命周期逻辑
- `game_bridge/transport`
  - named pipe 连接
  - json/proto codec
  - proto state 转 dict
- `game_bridge/session`
  - `CombatSession`
  - `FullRunSession`
  - `SessionFactory`
  - `SessionPool`
- `game_bridge/spectate`
  - `SpectatorController`
  - `PolicyAdapter`
  - `ManualPolicy`
  - `ReplayPolicy`
  - `ExternalPolicy`
  - `NullPolicy`
- `game_bridge/catalog`
  - 运行时静态 catalog 查询挂接
- `game_bridge/types`
  - `SessionConfig`
  - `StateView`
  - `PolicyContext`

## 对外 API

当前允许外部依赖的公共入口只有：

- `game_bridge.session.create_combat_session(...)`
- `game_bridge.session.create_full_run_session(...)`
- `game_bridge.session.SessionPool(...)`
- `game_bridge.sim.launch_headless_sim(...)`
- `game_bridge.spectate.SpectatorController(...)`
- `game_bridge.spectate.PolicyAdapter`

这层 API 的设计目标是：

- 不关心上层是不是 PPO、搜索、脚本回放或人工操作
- 不让业务脚本自己拼连接、自己管重连、自己管 host 进程
- 会话对象统一具备 `reset / get_state / act / export_state / import_state / close` 这一类最小语义

## CLI 入口

首批保留 3 个正式 CLI：

- `python -m game_bridge.sim.cli launch`
- `python -m game_bridge.session.cli inspect`
- `python -m game_bridge.spectate.cli`

其中观战 CLI 不再绑定任何现有 checkpoint 或 `UnifiedNet`。  
如果后续某个研究策略要接入观战，只需要实现一个 `PolicyAdapter`。

## 归档策略

旧的 `networkV2` 训练主线及其相关测试，已经迁到：

- `STS2AI/Python/archive/legacy_research`

归档后的代码只保留参考价值，不再作为主线依赖。  
`game_bridge` 主线也增加了结构测试，确保不再依赖：

- `networkV2.s5_net`
- `networkV2.s6_training`
- `archive.legacy_research`

## 当前已补的测试

保留并可直接运行的主线测试：

- `STS2AI/Python/tests/test_game_bridge_platform.py`
- `STS2AI/Python/tests/test_proto_state_converter.py`

覆盖范围包括：

- `SessionFactory` 分发
- `SessionPool` 复用与关闭
- `ReplayPolicy` 读取
- `SpectatorController` 推进与 overlay 输出
- `game_bridge` 源码无旧训练依赖
- `proto GameState -> dict` 转换

## 后续建议

下一阶段如果要继续演进，建议严格按“运行时核心”和“研究策略层”分层：

1. `game_bridge` 只维护连接、会话、状态推进、观战和 snapshot 能力。
2. 搜索、离线数据生成、策略网络、teacher、评测脚本另建新包，不回写到 `game_bridge`。
3. 任何“从当前 state 往前推进”的逻辑，都优先复用 `game_bridge` 提供的会话与 action semantics，不再在训练脚本里各写一套。
