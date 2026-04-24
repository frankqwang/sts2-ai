# Sim + 观战协议/代码统一 — 分析与方案

**日期**：2026-04-24
**目标**：识别当前 HeadlessSim 和 SpectatorBridgeMod 两条链路的代码重复 + 协议分裂，给出**可落地的统一方案**。

---

## 1. 现状总览

项目里有**两个独立的进程**对外提供 STS2 状态 + 动作接口：

### 1.1 HeadlessSim（训练主通道）

- 二进制：`/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe`
- 协议：**proto over Named Pipe**（`sts2_mcts_proto_{port}`）
- 调用方：`CombatSession`、`PipeBackedFullRunClient`（训练、rollout）
- 用途：headless，无 UI，高吞吐（一场 2 秒）

### 1.2 Godot + SpectatorBridgeMod（观战通道）

- 挂载：`mods/sts2_mcp_spectator/sts2_mcp_spectator.dll`（加到 Godot 游戏进程）
- 协议：**HTTP/JSON**（端口默认 15526）
- 路径：
  - `/api/v1/singleplayer`（legacy）
  - `/api/v2/full_run_env/{state,reset,step,batch_step,skip_combat,save_state,load_state,export_state,import_state,delete_state}`
- 调用方：`SingleplayerClient`、`ApiBackedFullRunClient`（观战）
- 用途：真游戏 UI，可看可录，低吞吐（几秒每动作，还要渲染）

### 1.3 两条链路并行存在的后果

| 维度 | HeadlessSim | SpectatorBridgeMod |
|---|---|---|
| wire 格式 | **proto 二进制** | **JSON 文本** |
| 传输 | Named Pipe | HTTP |
| 字段定义源 | `/STS2AI/ENV/proto/game_state.proto` | C# 手写 DTO |
| 状态构建 | `ProtoStateBuilder.cs`（1321 行）| `McpMod.StateBuilder.cs`（2131 行）+ `SpectatorApiStateBuilder.cs`（653 行） |
| 动作格式 | `LegalAction` proto | JSON dict |
| 扩字段需要改 | proto + ProtoStateBuilder + Python converter（3 处）| DTO + McpMod builders + ApiBacked 客户端（3 处）|
| **今天加 description 字段** | 只在 HeadlessSim 生效 | **完全没影响观战** |

---

## 2. 具体冗余代码清单

### 2.1 C# 侧（~5000 行重复逻辑）

```
HeadlessSim/
  Protocol/ProtoStateBuilder.cs           (1321 行)  建 GameState proto
  Simulation/FullRunApiStateBuilder.cs    (1178 行)  建 JSON（sim 自己的 v2 HTTP）
  Simulation/FullRunApiStateDtos.cs       (568 行)   对应 DTO
  Simulation/FullRunSimulationStateBuilder.cs (1060 行) 中间层

SpectatorBridgeMod/
  Bridge/McpMod.StateBuilder.cs           (2131 行)  建 JSON（观战 v1 + v2）
  Bridge/McpMod.Formatting.cs             (847 行)   字段格式化
  Bridge/SpectatorApiStateBuilder.cs      (653 行)   观战侧 state builder
  ModFullRunEnv.cs                        (1418 行)  HTTP 路由 + 处理
```

**同一个"把游戏内部 state 导出"的逻辑被写了三次**（Proto 一次、HeadlessSim JSON 一次、Spectator JSON 一次）。

### 2.2 Python 侧（~2000 行）

```
bridge/game_bridge/
  session/
    combat.py           (333 行)  CombatSession  — 只走 pipe proto
    full_run.py         (816 行)  3 个实现类同一抽象 FullRunClientLike:
      ├ ApiBackedFullRunClient       (HTTP v1+v2)
      ├ PipeBackedFullRunClient      (proto pipe)
      └ BinaryBackedFullRunClient    (legacy 废弃)
    singleplayer_api.py (166 行)  HTTP v1 client（被 ApiBacked 用）
  transport/
    connection.py       (312 行)  pipe 协议抽象
    proto_codec.py      (383 行)  proto envelope
    codec.py            (69 行)   codec 接口
    proto_state_converter.py (566 行)  proto → Python dict
    pipe_transport.py   (213 行)
    heartbeat.py        (149 行)
```

### 2.3 schema drift 的实际后果

今天踩过的坑：

1. **加 HandCard.description 字段**：改了 proto + ProtoStateBuilder + Python converter。
   - HeadlessSim 训练路径生效 ✅
   - Godot 观战路径完全不生效 ❌
   - 理论上要再改 SpectatorBridgeMod 的 McpMod.StateBuilder（2000+ 行文件里找对应卡片渲染点）+ ApiBacked 的 JSON 响应解析

2. **字段名不一致**：
   - proto 返 `target_id`（int）
   - JSON 返 `target_id`（有时 string 有时 int，还有 `combat_id`）
   - 下游每个 converter 要做 fallback pick

3. **action 格式**：
   - proto LegalAction：严格类型
   - JSON action：dict，字段多半兼容但不保证

4. **state_type 语义**：
   - proto 枚举
   - JSON 字符串
   - state_semantics.py 里一堆 lowercase + alias 适配

5. **pipe 路径对 combat full_run 有 bug**：之前 rollout_full_run 踩到 sim 拒绝 combat action。HTTP 路径就没事。**说明两条路径逻辑分叉了**。

---

## 3. 为什么会变成这样（简短历史）

- **HeadlessSim 先有**：训练需要高吞吐无 UI，proto+pipe 是合理选择
- **观战后加**：想看模型怎么打，Godot 里加 mod 挂 HTTP 最简单（proto 在 Godot 运行时 P/Invoke 有点折腾）
- **ApiBackedFullRunClient 试图统一 Python 侧**：但只做了客户端层的抽象，wire 格式没统一
- **没人来得及做最后一公里**：两边各自迭代，schema 分叉越来越深

---

## 4. 三个候选方案

### 方案 A：协议统一（激进、彻底）

**把 proto 作为唯一 wire 格式**，SpectatorBridgeMod 也发 proto。

具体动作：
1. SpectatorBridgeMod 废弃 HTTP/JSON
2. SpectatorBridgeMod 改起一个 **Named Pipe 服务**（端口映射 15526），用 ProtoCodec
3. Python 侧丢掉 `ApiBackedFullRunClient`、`singleplayer_api.py`、所有 HTTP 解析
4. 全部走 `PipeConnection + ProtoCodec`

**优点**：
- **一套 wire、一套 Python 客户端、一套 schema**
- 加字段只需要改 1 次 proto + N 个 state builder
- schema drift 物理消失
- spectate 和训练看到**一模一样**的 state

**缺点**：
- Godot 进程起 Named Pipe Server 是**可行但少见**（常见做 HTTP），调试工具链较少
- SpectatorBridgeMod 需要**较大重构**
- 破坏向后兼容（现有 spectate 脚本全要改）
- Windows 上 Godot + NamedPipeServer 的 permission / session 有坑
- 工期：**2-3 周**

**评估**：理论最佳，但是成本高、风险大。除非两边生命周期都要长期维护才值得。

---

### 方案 B：共享 C# 状态构建器（中等）

**保留两个进程 + 两种 wire**，但把 "build GameState from game internals" 这件事**提炼成共享库**。

具体：
1. 新建 `STS2AI/ENV/Shared/StateBuilder/` —— 一个 C# class library
2. 把 `ProtoStateBuilder` 里 "从 CombatTrainingStateSnapshot 构造 GameState" 那部分搬出来做成 public API
3. `ProtoStateBuilder`（HeadlessSim）调用这个共享库生成 proto
4. SpectatorBridgeMod 也调用这个共享库生成 proto，**然后把 proto 转成 JSON 发出去**（或直接发 proto over HTTP binary body）
5. Python 侧 **两条路径共用 proto_state_converter**

**优点**：
- **状态构建逻辑只有一份**，加字段只改一次 C#
- 两端 wire 仍可不同（proto-over-pipe vs proto-over-HTTP），但**数据语义一致**
- Python 侧客户端分歧大幅减小
- 工期：**1-1.5 周**

**缺点**：
- 还是两套 wire codec（pipe proto + HTTP proto）
- SpectatorBridgeMod 输入数据结构（Godot 活动中的 `RunState` + `CombatState`）vs HeadlessSim 的 `FullRunSimulationStateSnapshot`（数据类）不同，共享库要**能同时接受两种输入**
  - 或者：给 SpectatorBridgeMod 加一层 adapter，把活 Godot state 转成和 HeadlessSim 一样的 snapshot
  - 这一层 adapter 本身有工作量

**评估**：性价比最高。schema 统一带来的长期收益明显大于一次性重构成本。

---

### 方案 C：Python 侧收敛（最小变更）

**C# 两边保持现状**，**只优化 Python 侧**。

具体：
1. 删除 `ApiBackedFullRunClient` —— 观战改走 pipe 路径也可以（HeadlessSim 的 SpectatorBridgeMod 替换方案是大工程，先不做）
2. 保留 HTTP 仅用于 Godot 观战，但 Python 内**不再维护 HTTP-specific 逻辑**
3. 把 HTTP JSON 响应 **强制经过一个 adapter 转成 proto-等价字段**，再走同一个 converter
4. 删除 `BinaryBackedFullRunClient`（已废弃）
5. 合并 `combat.py` 和 `full_run.py` 的公共部分到 `session/base.py`

**优点**：
- 工期短：**3-5 天**
- 不碰 C# 代码
- 不破坏 Godot 观战
- Python 侧清爽一些

**缺点**：
- **没根治 C# 侧重复**。每次加字段还是 3 个 builder 各改一遍
- schema drift 仍然可能发生（两个 C# builder 手写代码）
- 观战链路的"信息缺失"问题还是要两边重复修

**评估**：**治标不治本**。但作为临时缓解是 OK 的。

---

### 方案 D：Combat 独立小 API（务实精准）

上面三个都是**大改**。一个更聚焦的做法：

承认我们 80% 的工作集中在 **combat decision**（整个 LLM 路线的核心）。让 combat 这一小块数据通路**彻底统一**，其他的（map/event/shop/treasure/card_select）**允许分叉**，在 90% 的时间里用不到。

具体：
1. 定义 `proto/combat_state.proto`（单独文件，不连整个 GameState 的树）
2. `ProtoStateBuilder.BuildCombatGameStateMessage` 输出这份小 schema
3. `SpectatorBridgeMod` 加一个 **新端点** `/api/combat_state_proto` 输出**同一份小 schema 的 proto 字节**（不管 v1/v2 JSON，额外加）
4. Python 加一个 `CombatStateConverter`（只处理 combat，~200 行）
5. LLM 策略只用这个 combat schema

**优点**：
- 工期**最短**（2-3 天）
- **combat 这个关键路径**两端一致
- 非 combat 的部分继续用老 JSON（反正 LLM 没训练过也不用读那么准）
- 向后兼容 100%

**缺点**：
- 非 combat 部分的 schema 分裂**持续存在**（接受）
- 长期看还是两套维护

**评估**：LLM 项目短期最优。非 combat 的 mess 留着不处理。

---

## 5. 推荐路径

按项目所处阶段 + 资源投入 2 选 1：

### 如果 LLM 路线是 6 个月长期项目 → **方案 B**

1 周重构，换来未来**每次加字段省 60% 工作**。一年下来省几十次小改动。

### 如果 LLM 短期要出 demo → **方案 D**

2-3 天，先把 combat 路径统一，**让今天加的 description/keywords/preview_damage 在观战里也能看到**。非 combat 的 mess 以后再处理。

---

## 6. 方案 D 的详细分解（推荐先做）

### Step 1：定义最小 Combat proto schema（0.5 天）

新文件 `/STS2AI/ENV/proto/combat_state.proto`：

```proto
syntax = "proto3";
package sts2_combat_v1;

message CombatStateResponse {
  // 顶层
  string state_type = 1;     // "monster"/"elite"/"boss"
  int32 round_number = 2;
  bool is_play_phase = 3;
  string turn = 4;
  bool can_end_turn = 5;

  // player
  PlayerState player = 10;

  // enemies
  repeated EnemyState enemies = 20;

  // hand / piles / deck / relics
  repeated HandCard hand = 30;
  repeated CardItem draw_pile = 31;
  repeated CardItem discard_pile = 32;
  repeated CardItem exhaust_pile = 33;
  repeated CardItem deck = 34;
  repeated RelicItem relics = 35;
  repeated PotionItem potions = 36;

  // legal_actions
  repeated LegalAction legal_actions = 50;
}

message PlayerState {
  int32 hp = 1;
  int32 max_hp = 2;
  int32 block = 3;
  int32 energy = 4;
  int32 max_energy = 5;
  int32 gold = 6;
  repeated Power powers = 7;   // 带 description
}

message EnemyState {
  int32 target_id = 1;
  string monster_id = 2;
  int32 hp = 3;
  int32 max_hp = 4;
  int32 block = 5;
  Intent intent = 6;           // 已经 modifier 调整过
  repeated Power powers = 7;   // 带 description
  bool is_alive = 8;
}

message HandCard {
  int32 hand_index = 1;
  string id = 2;
  int32 cost = 3;
  string card_type = 4;
  bool is_upgraded = 5;
  bool can_play = 6;
  bool requires_target = 7;
  repeated int32 valid_target_ids = 8;

  // 动态真实信息（游戏内部已经算好）
  string description = 10;                         // 已解析占位符
  repeated string keywords = 11;
  map<int32, int32> preview_damage_per_target = 12; // 对每目标实际伤害
  int32 preview_block = 13;                         // 实际 block
}

message Intent {
  string type = 1;
  int32 damage_per_hit = 2;  // 已经 modifier 调整过
  int32 hits = 3;
  int32 total_damage = 4;
}

message Power {
  string id = 1;
  int32 amount = 2;
  string description = 3;   // 人类可读文字
}

message CardItem { string id = 1; bool is_upgraded = 2; }
message RelicItem { string id = 1; string description = 2; }
message PotionItem { string id = 1; string description = 2; }

message LegalAction {
  string action = 1;
  int32 card_index = 2;
  int32 target_id = 3;
  string card_id = 4;
  bool is_enabled = 5;
}
```

### Step 2：C# 构造函数（HeadlessSim，0.5 天）

在 `ProtoStateBuilder.cs` 加：

```csharp
public static byte[] BuildCombatStateV1(CombatTrainingStateSnapshot snapshot) {
    var msg = new CombatStateResponse {
        // 从 snapshot 填...
    };
    return msg.ToByteArray();
}
```

加 proto pipe opcode `0x21` 处理这个请求。

### Step 3：SpectatorBridgeMod 加端点（0.5 天）

`ModFullRunEnv.cs` 加：

```csharp
else if (path == "/api/v2/combat_state_proto") {
    var snapshot = GetCurrentCombatSnapshot();
    byte[] proto_bytes = ProtoStateBuilder.BuildCombatStateV1(snapshot);
    response.ContentType = "application/x-protobuf";
    response.OutputStream.Write(proto_bytes, 0, proto_bytes.Length);
}
```

注意：SpectatorBridgeMod 需要引用**同一份 combat_state.proto 生成的 C#**。可以通过共享 DLL 或复制编译。这是**唯一需要两边一致的地方**。

### Step 4：Python 客户端 + converter（0.5-1 天）

新 `bridge/game_bridge/transport/combat_state_converter.py`（~200 行）：

```python
from game_bridge.generated import combat_state_pb2 as pb

def decode_combat_state(payload: bytes) -> dict:
    msg = pb.CombatStateResponse()
    msg.ParseFromString(payload)
    return {
        "state_type": msg.state_type,
        "round_number": msg.round_number,
        ...
        "hand": [_convert_hand_card(hc) for hc in msg.hand],
        ...
    }

def _convert_hand_card(hc):
    return {
        "id": hc.id,
        "cost": hc.cost,
        "description": hc.description,
        "keywords": list(hc.keywords),
        "preview_damage_per_target": dict(hc.preview_damage_per_target),
        "preview_block": hc.preview_block,
        ...
    }
```

### Step 5：把 `llm/inference/llm_policy.py` 切过去（0.5 天）

- 不管底层是 HeadlessSim 还是 SpectatorBridgeMod，都**直接调用 combat_state_proto 端点**
- 得到同样的 combat dict
- LLM 决策逻辑完全一致

### Step 6：验证（0.5 天）

- HeadlessSim rollout 看新 schema
- Godot 观战看新 schema
- **两者应该返回字段完全一致的 dict**
- 老的 full_run 路径保留，不破坏非 combat 观战

---

## 7. 风险 + 注意事项

1. **SpectatorBridgeMod 能不能引用 proto 生成的 C# 类**
   - csproj 里加 `<Protobuf Include="../proto/combat_state.proto">` 就行，和 HeadlessSim 一样
   - 两个 csproj 共享同一份 .proto 文件，Grpc.Tools 自动生成

2. **proto 字段 ID 不能乱改**
   - 一旦定了，后续只能加新字段用新 ID，不能改已有
   - 提前把 combat schema 设计到位（preview_damage、buff description 等必须的字段都预留）

3. **Godot 里运行时依赖 Google.Protobuf**
   - SpectatorBridgeMod 现在已经引了（因为它的 McpMod 里有 proto，虽然没用）
   - 不加依赖也能做，手写 proto 序列化，但不推荐

4. **游戏 state 采样时机**
   - Godot 里的 `SnapshotBuilder` 需要从**活状态**构造 snapshot
   - HeadlessSim 是从**计算出的 snapshot** 构造
   - 两者输入不一样，需要一个共同的 "snapshot" 结构作为 adapter 层

5. **Non-combat state 不碰**
   - map / event / shop / rest / card_select 不在这次统一范围里
   - 继续用旧的 JSON HTTP 路径
   - LLM 暂时不太关心这部分数据精度

---

## 8. 时间表（方案 D）

```
Day 1 上午：  写 combat_state.proto，生成 C# + Python 类
Day 1 下午：  ProtoStateBuilder.BuildCombatStateV1 + 新 pipe opcode
Day 2 上午：  SpectatorBridgeMod 加 /api/v2/combat_state_proto 端点
Day 2 下午：  Python CombatStateConverter + 集成到 CombatSession
Day 3 上午：  把 llm_policy 切到新 schema
Day 3 下午：  验证：HeadlessSim 和 Godot 输出相同 dict
          跑一场 spectate 看新 schema 是否正常
```

### 做完 D 之后的收益

1. **今天加的 description / keywords / preview_damage 在 HeadlessSim 和 Godot 观战都能用**
2. **后续"加字段给 LLM"只改 3 个地方**：proto + HeadlessSim builder + SpectatorBridgeMod builder（不需要改 full_run 老路径）
3. **Python 侧 combat state 语义完全一致**，不用担心 fallback pick 来 pick 去
4. **schema drift 在 combat 这条主线消失**

### 什么时候考虑升级到方案 B

- 非 combat 的 state 也开始需要完整信息（skada 非战斗训练后）
- 或者 D 做完跑几周发现 full_run 老 JSON 路径维护成本还是高

---

## 9. 不推荐的路线

- **方案 A（全部改 proto pipe）**：Godot NamedPipeServer 少见坑多，不值
- **丢掉 HeadlessSim 只保留 Godot**：视觉渲染 + game loop 太慢，跑不了 rollout 量
- **新写一个统一 API gateway**：多一层进程 = 多一层延迟 + 故障点

---

## 10. 一句话总结

**现状**：4-5 个 state builder 做同一件事，两套 wire，Python 侧三路并存，加一个字段要改 5-6 个文件。

**根因**：HeadlessSim 和 SpectatorBridgeMod 都想"把游戏 state 暴露出来"，但各自独立演化。

**最小可行修复（方案 D）**：用 2-3 天统一 **combat 一个路径**的 wire schema，让 LLM 走向的关键数据在两个进程里一致。其余分叉**暂不处理**。

**长期最优修复（方案 B）**：用 1-1.5 周把 C# state builder 代码**共享**，换来一年内加字段工作量降 60%。

推荐：**先做 D，3 个月后根据需要决定要不要升级到 B**。
