# 交接：Transport 层 + Combat Proto 统一迁移（收尾）

**日期**：2026-04-18  
**上一会话 session**：`debd1e8c-f829-465e-b7b3-8458bb24460b`  
**目标**：禁止再造任何 pipe/连接/协议底层轮子，所有业务模块（V2 内部）统一走 `networkV2.s0_bridge.transport` + proto schema。

---

## 0. 给下一个会话的 prompt（直接贴到新会话开头）

```
继续 handoff-2026-04-18-transport-proto-unify.md 里的任务。前置情况:
1. 今天完成了 combat 训练 bug 大清洗 (action schema drift / pipe owner check /
   C# NPE Crusher+SlumberingBeetle / skada v0.103+ 数据过滤 / encounter+card+relic
   白名单 / MP 卡 pattern 过滤 / sim single-session 约束识别)。smoke v17b 5 iter
   ERR=0,Hard% 3.8% 首次 boss 胜。
2. 新建了 networkV2/s0_bridge/transport/ (pipe_transport.py + connection.py +
   heartbeat.py + codec.py) 作为统一 bridge 层骨架。
3. PipeBackedCombatTrainingClient 已迁到 PipeConnection (内部升级, 外部 API 不变)。
4. 扩展了 proto/game_state.proto 加 CombatResetRequest/BuildSpec/CardSpec/
   RelicSpec/CombatStepRequest。Python proto 已 regenerate。
5. BinaryOpcode 枚举加了 CombatReset=0x11 / CombatStep=0x12 / CombatState=0x13。

本 session 未做 (接着干):
- 禁止造轮子: 还有 full_run_env / train_combat_v2 / proto_pipe_client 用老 PipeClient
- C# sim 的 CombatReset/CombatStep opcode 处理 handler (BinaryProtocol + Program.cs)
- CombatSession 实际调 proto combat_reset/combat_step (阻塞于 C# 端)
- combat_cotrainer 迁 CombatSession (Phase 5)
- CUDA graph RNG 冲突 (独立问题, workaround 留着)

按以下 5 阶段做完。每阶段跑对应 smoke 验证。
```

---

## 1. 当前状态快照

### 1.1 今天已做（不动）

| 类别 | 代码 | 状态 |
|-----|------|-----|
| 连接稳定 | `env/combat_training_env.py` PipeBackedCombatTrainingClient 迁 transport | ✓ |
| 连接稳定 | `env/pipe_client.py` owner check 删除 | ✓ |
| C# NPE | `src/Core/Models/Monsters/Crusher.cs` + `SlumberingBeetle.cs` 加 null check | ✓ |
| Action schema | `env/combat_training_env.py::build_combat_legal_actions` 信 sim `RequiresTarget` | ✓ |
| Action schema | `_TARGET_TYPE_NAMES` 对齐 C# enum 顺序（虽然已 sidestep）| ✓ |
| 数据清洗 | `skada_combat_replay.py::iter_combat_chain_from_run` encounter/card/relic 白名单 + MP pattern | ✓ |
| 数据清洗 | `skada_index_dataset.py::sample_clean_runs` 加 `version_prefix="v0.103."` 默认 | ✓ |
| Transport 骨架 | `networkV2/s0_bridge/transport/{pipe_transport,connection,heartbeat,codec}.py` | ✓ |
| Proto schema | `proto/game_state.proto` 加 `CombatResetRequest/BuildSpec/CardSpec/RelicSpec/CombatStepRequest` | ✓ |
| Proto 生成 | `networkV2/s0_bridge/generated/game_state_pb2.py` 重 gen | ✓ |
| C# opcode 枚举 | `BinaryProtocol.cs::BinaryOpcode` 加 `CombatReset=0x11 / CombatStep=0x12 / CombatState=0x13` | ✓ |
| Bridge CombatSession | `networkV2/s0_bridge/combat_session.py` 框架（但未被调用）| ✓（未接入）|

### 1.2 v17b smoke 验证（已通过，可复现）

```
iter 1: 127 combats 21%/ 8.0%/ 3.8%  ERR=0   Hard 3.8%!
iter 2: 106 combats 31%/ 8.3%/ 0.0%  ERR=0
iter 3:  99 combats 35%/ 4.5%/ 0.0%  ERR=0
iter 4: 115 combats 26%/14.3%/ 0.0%  ERR=0   Med 14%!
iter 5:  97 combats 42%/11.1%/ 3.8%  ERR=0   Hard 3.8%
```

运行命令：
```bash
cd STS2AI/Python
python -u -m networkV2.s6_training.combat_cotrainer \
    --preset slim \
    --checkpoint ../Artifacts/checkpoints/skada_bc_v7_full/bc_epoch_5.pt \
    --num-workers 4 --base-port 19600 \
    --max-iterations 5 --episodes-per-iter 10 --max-steps 150 \
    --lr 1e-5 --ppo-epochs 3 --mini-batch-size 64 --target-kl 0.03 \
    --skada-replay-index-db data/skada/derived/skada_runs.sqlite \
    --skada-replay-n-runs 200 \
    --dump-dir ../Artifacts/runs/smoke_verify
```

---

## 2. 下一个 session 要做的（按阶段）

### Phase A — 完成 V2 内部 transport 迁移（**禁止造轮子**）

**目标**：V2 训练里任何地方不直接 import/使用 `env.pipe_client.PipeClient` 或 `env.binary_pipe_client.BinaryPipeClient`。所有都走 `networkV2.s0_bridge.transport.PipeConnection`。

#### A1. `env/full_run_env.py` 迁移

**作用**：V2 `train_full_run_v2` + `demo_play_v2` 用的 full-run 客户端。内部有 `PipeBackedFullRunClient` 维护 `_pipe: PipeClient | BinaryPipeClient | ProtoPipeClient`。

**70+ call sites**：`self._pipe.call(method, params)`，3 种协议（json/bin/proto）混用 `_new_pipe_client()` 决定。

**做法：**
1. 改 `_pipe` 类型为 `PipeConnection`
2. `_new_pipe_client()` 改为根据协议选 codec:
   ```python
   from networkV2.s0_bridge.transport import PipeConnection, PipeConnectionConfig
   from networkV2.s0_bridge.transport.codec import JsonCodec, BinaryOpcodeCodec
   # bin 协议用 BinaryOpcodeCodec(request_encoder=..., response_decoder=...) 
   #   encoder/decoder 从 binary_pipe_client.py 移过来
   # proto 协议用 BinaryOpcodeCodec + proto encode/decode (生成代码)
   # json 用 JsonCodec
   ```
3. `self._pipe.call(method, params)` → `self._conn.safe_call(method, params)`
4. `close()` 用 `self._conn.close()`
5. 跑 `smoke_full_run`（有的话）+ `train_full_run_v2 --max-iterations 1` 验证

**工作量：3-4h**（call sites 多，但模式单一）

#### A2. `networkV2/s6_training/train_combat_v2.py` 迁移

**简单**，直接改 import。2h。

#### A3. `networkV2/s0_bridge/proto_pipe_client.py` **合并进 transport**

当前 `ProtoPipeClient` 是独立 client。其实它就是 **`PipeConnection` + proto codec**。

**做法：**
1. 把 `_encode_request` / `_decode_payload` 逻辑提取成 `ProtoCodec(ProtocolCodec)`（和 `JsonCodec` 并列）
2. `ProtoPipeClient` 要么:
   - (A) 变成薄 wrapper：`PipeConnection(codec=ProtoCodec(), pipe_name_prefix="sts2_mcts_proto")`
   - (B) 废弃，调用方直接用 `PipeConnection`
3. 保留 `ProtoPipeClient` 名字（向后兼容），内部委派到 PipeConnection

**工作量：2-3h**

#### A4. 最终清理

所有 V2 迁完后：
- 确认 `grep -rln "from env.pipe_client\|PipeClient" networkV2/` 零命中
- 确认 `grep -rln "from env.binary_pipe_client\|BinaryPipeClient" networkV2/` 零命中
- **删除 `env/pipe_client.py` 和 `env/binary_pipe_client.py`**（或保留 1 sprint 作 deprecated）

---

### Phase B — C# 端 Combat proto opcode handler（**本 session 只加了 opcode 枚举**）

#### B1. `HeadlessSim/Program.cs` 加 opcode dispatch

找 `BinaryOpcode.Reset / Step` 的 handler，仿写 `CombatReset / CombatStep / CombatState`。

看 Program.cs 里现有的 opcode → handler 对应（应该在 `BinarySimulatorRequest` handler 内）。

**伪代码：**
```csharp
case BinaryOpcode.CombatReset:
    var req = CombatResetRequest.Parser.ParseFrom(requestBytes);
    var state = CombatTrainingEnvService.Instance.ResetAsync(req).Result;
    return BuildCombatStateResponse(BinaryOpcode.CombatReset, state);

case BinaryOpcode.CombatStep:
    var stepReq = CombatStepRequest.Parser.ParseFrom(requestBytes);
    var stepState = CombatTrainingEnvService.Instance.StepAsync(stepReq.Action).Result;
    return BuildCombatStateResponse(BinaryOpcode.CombatStep, stepState);

case BinaryOpcode.CombatState:
    return BuildCombatStateResponse(BinaryOpcode.CombatState, currentState);
```

#### B2. `BinaryProtocol.cs::BuildCombatStateResponse`

仿 `BuildStateResponse` 但 payload 填 `GameState` proto message（**含 legal_actions**，让 Python 不再自己推断）。

关键：sim 侧要 populate `GameState.legal_actions` 字段。可能 sim 内部已有 `CombatTrainingStateSnapshot.LegalActions` 类似字段，不行就现算（从 Hand + 状态派生）。

#### B3. dotnet build + smoke 测 Python 能通过 proto combat

```bash
cd STS2AI/ENV/Sim/HeadlessSim && dotnet build -c Release -v minimal
```

Python 测试（先写 proto_pipe_client 侧配合）：
```python
from networkV2.s0_bridge.proto_pipe_client import ProtoPipeClient
client = ProtoPipeClient(port=20000, protocol="proto")
client.connect()
state = client.combat_reset(character="IRONCLAD", encounter="...", build=BuildSpec(...))
```

**工作量：4-6h**（C# 改完 + 测试）

---

### Phase C — Python proto combat 接入（Phase 3）

#### C1. `networkV2/s0_bridge/proto_pipe_client.py` 加 `combat_reset / combat_step / combat_state`

三个新方法，用 `OP_COMBAT_RESET/STEP/STATE` opcode。

- 请求：encode `CombatResetRequest` / `CombatStepRequest` proto 到 payload
- 响应：decode 成 `GameState` proto → dict（和现 JSON 返回格式对齐，方便下游 normalizer）

#### C2. `combat_session.py` 切到 proto

`CombatSession` 当前用 `PipeBackedCombatTrainingClient`（json wire）。改为用 `ProtoPipeClient` 或直接 `PipeConnection(codec=ProtoCodec())`。

**关键**：sim 的 proto `GameState.legal_actions` 填好后，Python 不再需要 `build_legal_actions_from_state`，直接：
```python
state = self._conn.safe_call("combat_step", step_req_proto)
legal = state["legal_actions"]  # proto 已经给了
```

#### C3. combat_cotrainer 切 CombatSession（Phase 5）

把 `_worker_collect` 里的 `PipeBackedCombatTrainingClient` 替换为 `CombatSession`：
```python
session = CombatSession(port=...)
session.reset(character=..., encounter=..., build=...)
state = session.step(action)
```

**工作量：Phase C 总 3-4h**

---

### Phase D — CUDA graph RNG 冲突（可选）

**PyTorch issue #99820**：capture 期间 dropout kernel access Philox RNG offset → eager training forward 挂。

方案选择（从易到难）：
1. **独立 `torch.Generator()` 传给每个 dropout call** — 改 model forward，~3-4h
2. **训练 forward 也进 graph** — 改 PPO trainer，~1 天
3. **换 WSL2** — Linux triton 可用 torch.compile，免 CUDA graph 手工绑

现 workaround：`patch_dropout_for_graph_safety` + 全局 dropout=0 在 `combat_cotrainer.py`。有些 forward path `_VF.dropout` 绕 monkey-patch 仍挂，**只在 rollout 用 graph + training eager** 就复现。

**先不做**，当前 forward 28ms/step baseline 跑 120 iter ~10h 可接受。

---

### Phase E — 其他遗留清理

| 项 | 位置 | 改动 |
|----|------|-----|
| potion 白名单没用 | `skada_combat_replay.py::iter_combat_chain_from_run` | 加 `supported_potions` 参数 + 过滤 |
| Character suffix 硬编码 | `skada_combat_replay.py::_CHARACTER_SUFFIXES` | 从 sim game_catalog 的 characters 字段派生 |
| 剩 C# NPE | `src/Core/Models/Monsters/` Vantom / Rocket / LagavulinMatriarch | 同 Crusher 加 null check |
| stale check 干扰协作 | `env/headless_sim_runner.py::ensure_host_binary_is_fresh` | 加自动 dotnet build hook（或 CLI `--skip-stale-check`） |

---

## 3. 验证清单（每阶段做完）

### Phase A 验证
```bash
grep -rln "from env.pipe_client\|from env.binary_pipe_client" networkV2/ | grep -v __pycache__ | grep -v s0_bridge/transport/
# 期望:零命中(除了 transport 自己可能对 binary_pipe_client 的兼容 import)
```

### Phase B 验证
```bash
# sim rebuild OK
cd STS2AI/ENV/Sim/HeadlessSim && dotnet build -c Release
# Python 测 proto combat
python -c "from networkV2.s0_bridge.proto_pipe_client import ProtoPipeClient; c=ProtoPipeClient(port=20000, protocol='proto'); c.connect(); ..."
```

### Phase C 验证
```bash
# combat_cotrainer 切 CombatSession 后跑 5 iter smoke
python -u -m networkV2.s6_training.combat_cotrainer \
    --preset slim --checkpoint ... \
    --max-iterations 5 --episodes-per-iter 10 \
    --skada-replay-index-db ... \
    --dump-dir ../Artifacts/runs/smoke_post_migration
# 期望:
# - ERR=0 所有 iter
# - combat 数 + steps/s 不低于 v17b baseline
# - legal_actions 来自 sim 而非 Python 推断
```

---

## 4. 硬规则（后续所有修改都要遵守）

1. **任何 V2 模块不得**：
   - 直接 `from env.pipe_client import PipeClient`
   - 直接 `from env.binary_pipe_client import BinaryPipeClient`
   - 自己写 pipe connect/reconnect/heartbeat 逻辑
   - 自己写 enum int ↔ string 映射（用 proto 生成代码或 sim game_catalog）
2. **V2 模块唯一允许**：
   - `from networkV2.s0_bridge.transport import PipeConnection, PipeConnectionConfig`
   - `from networkV2.s0_bridge import CombatSession / FullRunSession`
3. **新加协议字段**：只改 `proto/game_state.proto` → 两边 regen → 代码自动同步
4. **新加枚举值**：只改 `proto/game_state.proto` → C# + Python 代码同步，不手写 map

---

## 5. 风险 & 回滚

| 风险 | 触发 | 回滚策略 |
|-----|-----|---------|
| full_run_env 70 call sites 迁漏 | train_full_run_v2 启动挂 | git revert + 保留 PipeClient 继续用 |
| C# Combat proto opcode encoder 数据缺字段 | Python 端 decode 拿到空 legal_actions | 对比 `BuildApiState` 的 json 版看遗漏,补字段 |
| Sim 新版本 enum 再错位 | action reject 复现 | proto 强制 schema,不会有错位 |

---

## 6. 文件清单

### 本 session 修改/新建
- `proto/game_state.proto` ✏️ 加 5 个 message
- `networkV2/s0_bridge/constants.py` ✏️ 加 3 个 opcode
- `networkV2/s0_bridge/generated/game_state_pb2.py` ♻️ regen
- `networkV2/s0_bridge/combat_session.py` 🆕 (未接入)
- `networkV2/s0_bridge/transport/pipe_transport.py` 🆕
- `networkV2/s0_bridge/transport/connection.py` 🆕
- `networkV2/s0_bridge/transport/heartbeat.py` 🆕 (但 sim single-session 限制,关闭使用)
- `networkV2/s0_bridge/transport/codec.py` 🆕
- `networkV2/s0_bridge/transport/__init__.py` 🆕
- `networkV2/s5_net/{graph_runner,bank_max_spec,graph_debug}.py` 🆕 (CUDA graph 框架)
- `networkV2/s5_net/tokenizer.py` ✏️ 加 static buffer path
- `networkV2/s5_net/unified_net.py` ✏️ 加 forward_from_static + 修多个 capture-unsafe op
- `networkV2/s5_net/encoders/{build_encoder,common}.py` ✏️ 去 data-dependent branches
- `networkV2/s5_net/action_contextualizer.py` ✏️ `_tag` 用 register_buffer
- `networkV2/s5_net/losses.py` ✏️ `turn_damage_coef=0`
- `networkV2/s6_training/skada_combat_replay.py` ✏️ 大量过滤逻辑 + SimSupported
- `networkV2/s6_training/skada_index_dataset.py` ✏️ `version_prefix` 默认 v0.103+
- `networkV2/s6_training/combat_cotrainer.py` ✏️ action 修 + CUDA graph 整合 + chain replay 修 + sim-health monitor
- `env/combat_training_env.py` ✏️ build_combat_legal_actions + PipeBackedCombatTrainingClient 迁 transport
- `env/pipe_client.py` ✏️ 去 owner check
- `env/headless_sim_runner.py` ✏️ sim stderr 落 artifacts/sim_logs + Linux probe 迁 transport
- `src/Core/Models/Monsters/Crusher.cs` ✏️ Background null check
- `src/Core/Models/Monsters/SlumberingBeetle.cs` ✏️ NCombatRoom.Instance? null check
- `STS2AI/ENV/Sim/HeadlessSim/BinaryProtocol.cs` ✏️ BinaryOpcode 加 CombatReset/Step/State
- `tests/test_graph_runner.py` 🆕 CUDA graph 回归测试

### 未动但要下一 session 改
- `env/full_run_env.py`
- `networkV2/s6_training/train_combat_v2.py`
- `networkV2/s0_bridge/proto_pipe_client.py`
- `STS2AI/ENV/Sim/HeadlessSim/Program.cs` (加 CombatReset/Step opcode handler)
- `STS2AI/ENV/Sim/HeadlessSim/BinaryProtocol.cs` (加 BuildCombatStateResponse)
- `networkV2/s0_bridge/combat_session.py` (切到 proto,接 sim 权威 legal_actions)
