# Combat-Only 指定 Build Reset 说明

## 目标

为 combat-only / full-run reset 增加可选 `build` 参数，让调用方在 reset 时直接指定：

- 卡组 `deck`
- 遗物 `relics`
- 可选标量：`current_hp`、`max_hp`、`max_energy`、`gold`

这条链主要服务两类场景：

- 指定 build 的战斗专训
- 用同一套 build 复现实验、做对照评估

## 当前支持的字段

Python 侧 `reset(..., build=...)` 现在支持这些键：

```python
build = {
    "deck": [
        "STRIKE_IRONCLAD",
        {"id": "BASH", "upgrade_level": 1},
    ],
    "relics": [
        "BURNING_BLOOD",
        {"id": "VAJRA"},
    ],
    "current_hp": 61,
    "max_hp": 77,
    "max_energy": 4,
    "gold": 123,
}
```

同时兼容若干别名：

- `deck` 也可写成 `cards`
- `relics` 也可写成 `relic_ids`
- 卡牌 `id` 也可写成 `card_id` / `name`
- 遗物 `id` 也可写成 `relic_id` / `name`
- `upgrade_level` 也可写成 `upgrades` / `current_upgrade_level`
- `current_hp` 也可写成 `hp`
- `max_energy` 也可写成 `energy`

## 行为语义

- 如果传了 `deck`，会替换角色起始卡组。
- 如果传了 `relics`，会替换角色起始遗物。
- 如果传了 HP / energy / gold，会覆盖默认新局数值。
- `floor_added_to_deck` 会透传到底层 `SerializableCard` / `SerializableRelic`。

底层实现复用了 `Player.SyncWithSerializedPlayer(...)`，所以 deck / relic 的替换不是额外维护一套写法，而是走现有存档同步语义。

## 当前入口

- full-run reset：
  - [FullRunSimulationDtos.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulationDtos.cs:7)
  - [FullRunSimulatorRuntimeFacade.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulatorRuntimeFacade.cs:309)
- combat-only reset：
  - [CombatTrainingResetRequest.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Training/CombatTrainingResetRequest.cs:5)
  - [CombatTrainingSession.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Training/CombatTrainingSession.cs:143)
  - [CombatSimulatorRuntimeFacade.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/CombatSimulatorRuntimeFacade.cs:70)
- 共享 helper：
  - [SimulationBuildSupport.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/SimulationBuildSupport.cs:12)
- Python reset 客户端：
  - [full_run_env.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/ipc/full_run_env.py:102)
  - [binary_pipe_client.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/ipc/binary_pipe_client.py:487)

## 协议变更

binary pipe 协议版本从 `11` 升到 `12`，schema hash 改为：

`sts2-binary-schema-2026-04-14-build-reset`

对应位置：

- [BinaryProtocol.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Runtime/HeadlessSim/BinaryProtocol.cs:180)
- [binary_pipe_client.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/ipc/binary_pipe_client.py:45)

这次改动要求 Python client 和 C# host 一起更新。

## 当前约束

- 卡牌 / 遗物 id 必须能在 `ModelDb` 里解析。
- 升级等级仍由引擎校验；例如超过 `MaxUpgradeLevel` 会直接拒绝 reset。
- 目前只保证“卡组 + 遗物 + 几个基础数值”可指定。
- 药水、抽牌堆/弃牌堆/消耗堆、战斗中状态、地图位置等更细粒度 build 还没有开放成 reset 参数。

## 已做验证

已做一轮最小 smoke：

- 用临时编译的 `headless_sim_host_0991.exe`
- 通过 pipe-binary 调 `reset(build=...)`
- 成功在返回状态里读到指定的 deck / relics / hp / gold
- 非法 `upgrade_level` 也会被引擎正确拒绝
