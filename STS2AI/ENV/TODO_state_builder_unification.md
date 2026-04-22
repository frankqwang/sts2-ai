# State Builder Unification TODO

## 背景

当前 `visible spectator`、`headless sim full-run`、`headless sim proto/combat` 在“状态构造 + legal action 构造”上仍然存在多套实现。

这次 `PURITY / hand_select` 的 submenu 问题已经暴露出一个结构性风险：

- 一处修了，另一处很容易漏修
- 相同语义可能在不同后端表现不一致
- parity/调试成本会越来越高

目前已经先把“多选/确认动作语义”抽成共享模块：

- [SelectionActionSemantics.cs](/C:/dev/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/Simulation/SelectionActionSemantics.cs:1)

接入点包括：

- visible spectator: [ModFullRunEnv.cs](/C:/dev/sts2-ai/STS2AI/ENV/SpectatorBridgeMod/ModFullRunEnv.cs:905)
- sim full-run: [FullRunSimulationStateBuilder.cs](/C:/dev/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulationStateBuilder.cs:474)
- sim proto: [ProtoStateBuilder.cs](/C:/dev/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/Protocol/ProtoStateBuilder.cs:450)

这只是第一步，还没有把“整套状态构造”统一成单一实现。

## 当前判断

“整套状态构造全并成一份代码”这个方向是合理的，但应当理解为：

- 保留两层很薄的输入采集适配层
  - visible 负责从 Godot/UI 读取原始状态
  - sim 负责从 simulation snapshot 读取原始状态
- 把后续的共享逻辑统一
  - screen payload 装配
  - combat payload 装配
  - selection payload 装配
  - legal action 构造
  - 导出到 `FullRunApiState`
  - 导出到 `Proto GameState`

不建议把“直接读 Godot 节点”和“直接读 sim snapshot”硬塞进一套代码里，那样会让共享层反而更脆弱。

## 建议目标结构

### 1. 输入适配层

- `VisibleExtractor`
  - 输入：Godot live state / UI node / room / screen
  - 输出：共享 typed state source
- `SimExtractor`
  - 输入：`FullRunSimulationStateSnapshot` / combat snapshot / bridge snapshots
  - 输出：同一种 typed state source

### 2. 共享装配层

- `SharedStateBuilder`
  - 统一组装：
  - run/player/battle
  - map/event/shop/rest_site/treasure
  - rewards/card_reward/relic_select/card_select/hand_select
  - legal actions

### 3. 导出层

- `SharedApiExporter`
  - 输出 `FullRunApiState`
- `SharedProtoExporter`
  - 输出 `GameState proto`

## 推荐分阶段改造顺序

### 阶段 1：继续共享 legal action 构造

目标：

- 把 map/event/rewards/shop/rest/combat/select 等 legal action builder 进一步抽成共享实现
- visible 与 sim 只保留极薄的 action source 适配

原因：

- 这部分最容易漂
- 直接影响 agent 行为
- 风险高，收益也最高

### 阶段 2：统一 selection/card/combat payload

目标：

- `hand_select`
- `card_selection`
- `card_select`
- `battle.player/hand/enemies/piles`

统一字段语义、排序、可见性规则、索引规则

### 阶段 3：统一非战斗 screen payload

目标：

- `map`
- `event`
- `shop`
- `rest_site`
- `treasure`
- `rewards`
- `card_reward`
- `relic_select`

### 阶段 4：让 visible 不再以 dict 作为核心中间态

当前 visible 仍然是：

- `McpMod.StateBuilder` 先产 `Dictionary<string, object?>`
- `SpectatorApiStateBuilder` 再转成共享 DTO

后续目标是让 visible 直接走 typed/shared builder，而不是先造 dict 再二次转换。

## 预估成本

粗估：

- 最小可用重构：3-4 天
  - 主要收口 legal action + selection/combat 热点
- 中等完整重构：5-7 天
  - 再覆盖常见 screen payload
- 比较彻底的版本：7-10 天
  - visible 去掉 dict 中间态
  - proto/api 都改成共享 exporter
  - 补齐 parity/regression fixtures

## 建议的第一批回归样例

后续真正开始改造前，建议先固定下面这些回归样例：

- `PURITY`：`4 选 3`，验证选满后只剩 confirm
- `ChooseCard`：单选不暴露多余 confirm/cancel
- `card_select`：预览确认态下 fields 与 legal actions 一致
- `combat card_selection`：已选卡不应继续暴露为 selectable
- `event/shop/rest/map`：visible/sim/proto 三端 legal actions 一致

## 当前状态

这件事先不在本次提交里继续扩展，只记录 TODO，后续单开一轮改造。
