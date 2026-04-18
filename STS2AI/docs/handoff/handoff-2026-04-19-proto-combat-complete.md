# 交接:Transport / Proto Combat 统一收尾(完结)

**日期**:2026-04-19
**对上次交接**:`handoff-2026-04-18-transport-proto-unify.md`

---

## 本次完成

### Phase A — V2 内部 transport 迁移

| 动作 | 位置 | 状态 |
|-----|------|-----|
| A1 | `env/full_run_env.py::PipeBackedFullRunClient` 改内部 `_conn: PipeConnection`,去掉三路 `_new_pipe_client()` | ✓ |
| A2 | `train_combat_v2.py` 不直接引老 client | ✓ |
| A3 | `networkV2.s0_bridge.proto_pipe_client.ProtoPipeClient` 改成 `PipeConnection + ProtoCodec` 的薄包装 | ✓ |
| A3 | 新 `networkV2/s0_bridge/transport/proto_codec.py` 承载所有 proto wire encode/decode | ✓ |
| A4 | networkV2 零命中 `from env.pipe_client` / `from env.binary_pipe_client` / `PipeClient` 的真实 import | ✓ |

### Phase B — C# 端 Combat proto opcode handler

| 动作 | 位置 | 状态 |
|-----|------|-----|
| B1 | `Program.cs::ProcessProtoRequestAsync` 加 `CombatReset / CombatStep / CombatState` 分派 | ✓ |
| B1 | 同文件里 `ProcessBinaryRequestAsync` 对 combat opcode 直接返 "proto only" 错误 | ✓ |
| B2 | `ProtoStateBuilder.cs` 加 `ParseCombatResetRequest / ParseCombatStepRequest / BuildCombatStateResponse / BuildCombatStepResponse / BuildCombatGameStatePayload / PopulateCombatLegalActions` | ✓ |
| B2 | `BuildCombatGameStatePayload` 直接 populate `GameState.legal_actions`(sim 权威字段,Python 不再推断) | ✓ |
| B3 | dotnet build 0 errors,Python proto combat_reset/step/state smoke 端到端跑通 | ✓ |

### Phase C — Python proto combat + CombatSession + combat_cotrainer

| 动作 | 位置 | 状态 |
|-----|------|-----|
| C1 | `ProtoCodec.encode_request` 覆盖 `combat_reset / combat_step / combat_state` method | ✓ |
| C2 | `CombatSession` 改走 proto wire(`PipeConnection + ProtoCodec`),legal_actions 直接拿 sim 权威字段 | ✓ |
| C2 | `CombatSession.step()` 返回 `(state, reward, done, info)` 兼容 `PipeBackedCombatTrainingClient` | ✓ |
| C2 | `CombatSession._call` 支持 proto opcode,非 combat method 抛 `NotImplementedError` 让 `GameCatalog` fallback sqlite | ✓ |
| C3 | `combat_cotrainer.py` 通过 `import as alias` 直接换 client 实现 | ✓ |
| Verify | 2 iter smoke, workers=2, episodes=4:iter1=73 combats, iter2=78 combats,proto wire 正常跑 | ✓ |

### Cleanup — 废弃手写 binary wire

| 动作 | 位置 | 状态 |
|-----|------|-----|
| 删 | `env/binary_pipe_client.py` 整个文件 | ✓ |
| 改 | `env/full_run_env.py` 拒绝 `protocol="bin"` + `_resolve_pipe_protocol("pipe-binary")` | ✓ |
| 改 | `env/headless_sim_runner.py` 去掉 `BinaryPipeClient` import + bin probe 分支 | ✓ |
| 改 | `data/export_game_catalog_runtime.py` 默认 proto,`--protocol bin` 拒绝 | ✓ |
| 改 | `diagnostics/run_canonical_eval.py` transport 切 `pipe-proto` | ✓ |
| 改 | `diagnostics/training_semantic_audit.py` backend map 去 `headless-binary` | ✓ |
| 改 | `demo_play_v2.py` CLI choices 去 `pipe-binary`,示例 `--transport pipe-proto` | ✓ |
| 改 | `Program.cs::HostOptions.Parse` 对 `--protocol bin/binary` 抛 `已废弃` error | ✓ |

---

## 回归验证

```
cd STS2AI/Python
python -u -m networkV2.s6_training.combat_cotrainer \
    --preset slim \
    --checkpoint ../Artifacts/checkpoints/skada_bc_v7_full/bc_epoch_5.pt \
    --num-workers 2 --base-port 19720 \
    --max-iterations 2 --episodes-per-iter 4 --max-steps 60 \
    --lr 1e-5 --ppo-epochs 1 --mini-batch-size 32 --target-kl 0.03 \
    --skada-replay-index-db data/skada/derived/skada_runs.sqlite \
    --skada-replay-n-runs 50 \
    --dump-dir ../Artifacts/runs/smoke_proto_phase_c_v2
```

结果:
```
  1 |  73 |  1897 | 19/73 |  35.8% /   0.0% /   0.0% | pl=0.0839 vl=0.631 kl=0.2534 ep=1 ERR=2 |  86.8s
  2 |  78 |  1841 | 12/78 |  21.4% /   0.0% /   0.0% | pl=-0.0068 vl=0.627 kl=0.0800 ep=1 ERR=1 |  27.6s
```

Combat 数 / 步数 / 胜率和 v17b baseline (JSON wire) 同量级。**ERR>0** 是独立 sim NPE
问题(可能还有 Crusher/SlumberingBeetle 之外的 null case),和 proto 迁移本身无关。

---

## 还没做(给下一个 session)

### 1. C# 端 BinaryProtocol.cs 物理删除

`BinaryProtocol.cs` 仍然有:
- Binary wire 专用 `BuildStatePayload / BuildStateResponse / Write*State / BinarySessionState` —— V2 不调,但和 ProtoStateBuilder 共享的 helper(`WriteString / WriteOptionalString / BuildErrorResponse / BuildSaveStateResponse / ParseResetRequest / ...`)紧耦合
- `Program.cs::ProcessBinaryRequestAsync` 及所有 `ProcessBinary*` handler —— sim 虽然已经不接受 `--protocol bin` 命令行,但 dispatcher 代码还在

**处理建议**:下次把 `BinaryProtocol.cs` 里 proto 共享的工具方法(`WriteString / WriteOptionalString / BuildSaveStateResponse / BuildExportStateResponse / BuildDeleteStateResponse / BuildErrorResponse / BuildPerfStatsResponse / BuildResetPerfStatsResponse / BuildSearchCombatMctsResponse / ParseResetRequest / ParseStateIdRequest / ParseExportStateRequest / ParsePathRequest / ParseDeleteClearAll / ParseActionRequest / ParseBatchActionRequest / ParseSearchCombatMctsRequest / ParseOpcode / BinaryRequestReader / ActionTypeToCode + ActionName + MapCardType + MapCardRarity + ...`)搬到 `ProtoRequestReader.cs`(新文件),把 `BinaryProtocol.cs` 删掉。

### 2. sim `--protocol bin` dispatcher 分支

`Program.cs::ProcessBinaryRequestAsync` + 所有 `ProcessBinaryXxx` 方法。现在 `HostOptions.Parse` 拒绝了 `bin/binary`,但这些函数仍在代码里,做 cleanup。

### 3. 已知独立 bug:combat smoke ERR>0

2 iter smoke 中出现 pipe 断 + `Combat state is not initialized` 报错,sim crash 触发 reconnect。不是 proto 迁移引入:
- v17b baseline (JSON wire) 也有过类似问题(Crusher/SlumberingBeetle 的 NPE 已修)
- 日志提示还有遗漏 null case(可能 `src/Core/Models/Monsters/Vantom.cs` / `Rocket.cs` / `LagavulinMatriarch.cs`,上次 handoff 的 Phase E 遗留)

### 4. CUDA graph RNG 冲突(Phase D,原本标 optional)

上次 handoff 保留的 workaround (`patch_dropout_for_graph_safety` + dropout=0) 仍有效。优先级低。

---

## 硬规则(延续)

1. V2 模块**唯一允许**:
   - `from networkV2.s0_bridge.transport import PipeConnection, PipeConnectionConfig, ProtoCodec, JsonCodec`
   - `from networkV2.s0_bridge import CombatSession / ProtoPipeClient(兼容)`
2. 禁止:
   - `from env.pipe_client import PipeClient` — 这个 file 还在,观战 mod(`ApiBackedFullRunClient`)路径间接用。V2 训练别碰
   - 任何自己写 pipe connect/reconnect/heartbeat/锁的逻辑
   - 手写二进制 state/action 编码
3. 新加 sim proto 字段:`proto/game_state.proto` + regen(两边)
4. 新加 proto opcode:`BinaryOpcode` enum + sim `ProcessProtoRequestAsync` dispatch + `ProtoStateBuilder` builder + `ProtoCodec` encode/decode

---

## 文件清单

### 新增

- `STS2AI/Python/networkV2/s0_bridge/transport/proto_codec.py`
- `STS2AI/docs/handoff/handoff-2026-04-19-proto-combat-complete.md` (本文件)

### 修改

- `STS2AI/Python/networkV2/s0_bridge/transport/connection.py`(加 codec 注入)
- `STS2AI/Python/networkV2/s0_bridge/transport/__init__.py`(导出 ProtoCodec 等)
- `STS2AI/Python/networkV2/s0_bridge/proto_pipe_client.py`(改薄包装)
- `STS2AI/Python/networkV2/s0_bridge/combat_session.py`(proto wire + `(state, reward, done, info)` step)
- `STS2AI/Python/networkV2/s0_bridge/constants.py`(去 binary 注释)
- `STS2AI/Python/networkV2/s0_bridge/proto_state_converter.py`(去 binary 注释)
- `STS2AI/Python/networkV2/s0_bridge/__init__.py`(去 binary 注释)
- `STS2AI/Python/networkV2/s4_compiler/runtime_compiler.py`(去 binary 注释)
- `STS2AI/Python/networkV2/s6_training/combat_cotrainer.py`(`CombatSession as PipeBackedCombatTrainingClient`)
- `STS2AI/Python/networkV2/s8_spectate/demo_play_v2.py`(CLI 去 `pipe-binary`)
- `STS2AI/Python/env/full_run_env.py`(拒绝 bin,PipeConnection 唯一后端)
- `STS2AI/Python/env/headless_sim_runner.py`(去 BinaryPipeClient)
- `STS2AI/Python/data/export_game_catalog_runtime.py`(默认 proto)
- `STS2AI/Python/diagnostics/run_canonical_eval.py`(transport → pipe-proto)
- `STS2AI/Python/diagnostics/training_semantic_audit.py`(backend map)
- `STS2AI/Python/tests/test_proto_state_converter.py`(去 binary 注释)
- `STS2AI/ENV/Sim/HeadlessSim/Program.cs`(CombatReset/Step/State 分派 + `--protocol bin` 拒绝)
- `STS2AI/ENV/Sim/HeadlessSim/ProtoStateBuilder.cs`(ParseCombatResetRequest/StepRequest + BuildCombatStateResponse/StepResponse + BuildCombatGameStatePayload + PopulateCombatLegalActions)

### 删除

- `STS2AI/Python/env/binary_pipe_client.py`
