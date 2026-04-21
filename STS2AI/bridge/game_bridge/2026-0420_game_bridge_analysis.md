# 2026-0420 game_bridge 实现分析与迭代记录

## 1. 目录定位

`STS2AI/Python/game_bridge` 现在承担的是 Python 侧模拟器桥接层，目标是把：

- `sim` 启动/进程管理
- `transport` pipe 协议与连接管理
- `session` 高层 `combat/full_run` 语义
- `catalog` 静态知识查询
- `spectate` 观战控制

解耦成可复用的平台层，避免训练脚本直接绑死旧的 `networkV2`/历史 bridge 代码。

## 2. 当前实现结构结论

### 2.1 分层基本正确

- `transport/pipe_transport.py`
  负责 Win32 Named Pipe 原始字节 IO。
- `transport/connection.py`
  负责连接、重连、锁、握手、错误分类。
- `transport/proto_codec.py`
  负责 proto wire 的 opcode 编解码。
- `session/combat.py`
  负责 combat 专用 reset/state/step。
- `session/full_run.py`
  负责 full-run 的 HTTP v2 / pipe proto 双通道。
- `sim/launcher.py`
  负责 host freshness 校验、拉起 sim、日志落盘。

这条主链已经比旧实现清晰，尤其 combat 已经把 `legal_actions` 权威性收回到 sim 端，不再由 Python 侧自己猜。

### 2.2 主要结构问题

这轮分析前，存在 3 个核心问题：

1. 同一套业务语义在多个文件重复实现。
   - `build` 规格归一同时出现在 `combat.py` 和 `full_run.py`
   - combat settle/actionable 判定只在 `singleplayer_api.py` 私有实现
   - `run_outcome` 归一与胜负判断散落在多个文件里直接比较字符串

2. 桥接层对“一致性问题”的输出能力弱。
   - 有 transport 和 session，但缺少一个统一的“支持矩阵/缺口报告”入口
   - 发现 sim 与原游戏不一致时，更多依赖人工读代码和手工排查

3. proto pipe 还不是完整平台协议。
   - `combat_reset/combat_step/combat_state` 已支持
   - `full_run` 热路径 RPC 已支持
   - 但 `game_catalog/combat_catalog` 仍缺 proto opcode，导致 catalog 层继续 fallback sqlite/json

## 3. 本轮已完成的改造

### 3.1 共享语义层收敛

新增：

- `STS2AI/Python/game_bridge/session/build_spec.py`
- `STS2AI/Python/game_bridge/session/state_semantics.py`
- `STS2AI/Python/game_bridge/sim/consistency.py`

作用：

- `run_outcome` 统一走共享 helper，避免 `"win" / "victory" / "death" / "loss"` 各写各的
- `build` 归一只保留一套实现，combat/full_run 共用
- combat 是否可行动、step 后是否需要等待 settle、menu 是否 ready for reset 改成共享 helper
- 新增静态一致性报告与 state 检查入口

### 3.2 已清理的无用/重复代码

- 删除 `full_run.py` 里重复的 `_normalize_build_spec`
- 删除 `full_run.py` 里未被使用的 `_extract_run_outcome`
- 删除 `singleplayer_api.py` 里重复的 combat state helper
- 删除 `combat.py` 里本地定义的 build spec dataclass，改为共享模块

### 3.3 已补的诊断入口

现在可以直接执行：

```bash
python -m game_bridge.sim.cli report
```

输出静态一致性报告，至少能明确当前 bridge：

- 哪些能力是 supported
- 哪些是 partial
- 哪些还是 missing/unsupported

这不是最终的 live parity 对拍，但已经比“只能靠读代码判断”前进了一步。

## 4. 当前能力矩阵

### 4.1 已到位

- `full_run/http_v2`
  - `state/reset/step/batch_step/save/load/import/export/delete_state`
- `full_run/pipe_proto`
  - `reset/state/step/batch_step/save/load/import/export/delete`
  - `perf_stats/reset_perf_stats`
  - `load_ort_model/run_combat_local/search_combat_mcts`
- `combat/pipe_proto`
  - `combat_reset/combat_state/combat_step`
  - `legal_actions` 由 sim 生成，Python 只消费

### 4.2 部分到位

- 状态语义
  - combat actionable 判定已统一
  - post-action settle 等待已统一
  - menu ready for reset 已统一
  - 但仍缺 live screen-by-screen 对拍脚本

### 4.3 仍缺失

- proto pipe 的 `game_catalog/combat_catalog` opcode
- sim 与原游戏逐 screen 自动回归
  - `map`
  - `event`
  - `shop`
  - `rest_site`
  - `card_select`
  - `combat_rewards`
- 更细粒度的失败诊断
  - 当前很多 rejection 仍以异常文本为主
  - 缺统一的 invariant/failure report 聚合

## 5. 一致性判断

### 5.1 现在相对可靠的部分

- combat 热路径的一致性明显优于旧实现
  - Python 不再自己拼 legal actions
  - proto state 已有较完整的结构化转换
  - reward 的 terminal 胜负口径已统一

### 5.2 现在仍有风险的部分

- `catalog` 不在 proto 主协议里，意味着训练主路径和静态查询仍然不是一条完全统一的桥
- `BinaryBackedFullRunClient` 名称仍保留兼容语义，容易让人误解它还在走旧 binary wire
- `transport/codec.py` 里曾保留过旧 binary 兼容骨架；现主路径只保留 `JsonCodec / ProtoCodec`

## 6. 建议的后续迭代顺序

1. 给 proto pipe 增加 `game_catalog/combat_catalog` opcode
   - 先打通 runtime catalog 主路径
   - 再逐步减少 sqlite fallback 的默认权重

2. 新增 live parity 回归脚本
   - 同 seed、同 character、同 action 序列
   - 逐步对拍 `state_type`、`legal_actions`、关键 screen payload

3. 给 `step/reset` 增加统一 failure code / invariant 报告
   - 不只是在异常对象上挂属性
   - 还要形成可落盘、可比较的结构化报告

4. 再决定是否继续清理兼容外壳
   - `BinaryBackedFullRunClient`
   - 旧 `_pipe.call` 兼容路径

这几项里，真正影响 sim 和原游戏一致性的优先级是 1 和 2，名称兼容和历史骨架清理放后面。

## 7. 本轮验证

已执行：

```bash
python -m pytest STS2AI/Python/tests/test_game_bridge_platform.py STS2AI/Python/tests/test_proto_state_converter.py STS2AI/Python/tests/test_game_bridge_consistency.py -q
```

结果：

- `44 passed`

说明这轮共享语义抽取和一致性报告补充没有破坏现有 `game_bridge` 对外接口。
