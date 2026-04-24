# Game Bridge 当前架构

更新时间：2026-04-24

## 目标

`GameState` protobuf 是唯一状态 schema。sim 和观战只允许在协议传输、UI 主线程执行、UI settle 等待上不同；状态语义、动作语义、卡牌动态描述构建必须共用同一套代码。

## 协议边界

- sim：Named Pipe + protobuf binary。
- 观战：HTTP `POST /api/game_bridge/rpc` + protobuf JSON mapping。
- HTTP JSON 只能由 `Google.Protobuf.JsonParser/JsonFormatter` 或 Python `google.protobuf.json_format` 处理，不再手写独立 JSON DTO schema。
- 旧 `/api/v1/singleplayer` 与 `/api/v2/full_run_env/*` 不再作为默认协议入口。

## C# 状态链路

共享代码集中在 `STS2AI/ENV/Shared`：

- `Runtime/BridgeRpcDispatcher.cs`：统一 RPC method 分发。
- `Runtime/BridgeGameStateBuilder.cs`：从 snapshot/runtime 对象构建 protobuf `GameState`。
- `Runtime/BridgeCombatSnapshotBuilder.cs`：从当前游戏 runtime 采集 combat snapshot。
- `Simulation/`：full-run snapshot builder、choice bridge、trace/diagnostics。
- `Training/`：combat snapshot DTO、卡牌动态描述、选择 adapter。
- `Selection/`：选择器接口。
- `Legacy/FullRunApiStateDtos.cs`：旧 JSON helper 残留，不参与 `/api/game_bridge/rpc`。

`HeadlessSim` 和 `SpectatorBridgeMod` 都只引用 `ENV/Shared`，观战工程不再直接引用 `Sim/HeadlessSim` 源码。

## 卡牌动态描述

卡牌 `description`、`keywords`、`requires_target`、`valid_target_ids`、`preview_damage_per_target`、`preview_block` 统一在 shared builder 链路里填充。

动态描述来源只允许游戏原逻辑：

- `CardModel.UpdateDynamicVarPreview(...)`
- `CardModel.GetDescriptionForPile(...)`

Python 和 JSON 层不得替换 `{Damage:diff()}` 这类模板。Python 只消费 `GameState` 转换后的 normalized dict。

## Python 入口

默认业务 API 是 `GameSession`：

- `reset(...)`
- `get_state(...)`
- `act(...)`
- `batch_act(...)`

zero 自己的 RL runtime 可以保留 `step(action_index)`，但内部必须调用 `GameSession.act`。

## 已归档文档

以下文档记录的是旧链路或中间方案，已经移入 `archive/2026-04-24/`：

- old visible HTTP v2 观战指引。
- old sim/spectator 统一分析方案。
- old LLM 微调中间计划与交接。
- old zero 主线与单 case 实验说明。

这些文档只作历史参考，不再作为当前实现依据。
