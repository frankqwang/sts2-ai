# C# MCTS迁移说明

## 目标

本次迁移把 combat MCTS 的以下职责从 Python 挪到 C# 宿主：

- 树搜索与 PUCT 选边
- root Dirichlet noise
- batched leaf evaluation
- virtual loss
- 根节点 visit 统计与最终动作选择
- `SaveState/LoadState/DeleteState` 驱动的分支搜索与恢复

Python 侧保留：

- 训练循环
- replay / optimizer / checkpoint
- episode 编排
- backend 切换

`src/` 下反编译源码未修改，新增实现全部落在 `STS2AI/ENV` 与 `STS2AI/Python`。

## 代码落点

### C# 宿主

- `STS2AI/ENV/Sim/Runtime/HeadlessSim/CombatFeatureEncoder.cs`
  - 负责 combat state / legal actions 到 ONNX 输入张量的编码。
- `STS2AI/ENV/Sim/Runtime/HeadlessSim/OrtCombatEvaluator.cs`
  - 负责 ONNX Runtime 的 `policy_logits + value` 单样本 / 批量评估。
  - 保留了 `OrtActorPolicy` 兼容包装，现有 `run_combat_local` 继续可复用。
- `STS2AI/ENV/Sim/Runtime/HeadlessSim/CombatMcts.cs`
  - 负责 C# 侧 combat MCTS 搜索。
- `STS2AI/ENV/Sim/Runtime/HeadlessSim/BinaryProtocol.cs`
  - 新增 `search_combat_mcts` 二进制协议。
- `STS2AI/ENV/Sim/Runtime/HeadlessSim/Program.cs`
  - 新增 `SearchCombatMcts` 请求分发与 `load_ort_model` 元信息回传。
- `STS2AI/ENV/Sim/Host/headless_sim_host_0991.csproj`
  - 宿主项目现在显式编入新的 MCTS / ORT 文件。

### Python 接线

- `STS2AI/Python/search/combat_mcts_agent.py`
  - `CombatMCTSAgent` 支持 `backend=python|csharp`
  - `backend=csharp` 时通过 pipe 调 `search_combat_mcts`
- `STS2AI/Python/ipc/binary_pipe_client.py`
  - 新增 `OP_SEARCH_COMBAT_MCTS`
  - `load_ort_model` 解码新增模型元信息
- `STS2AI/Python/ipc/full_run_env.py`
  - `PipeBackedFullRunClient` 新增 `search_combat_mcts`
- `STS2AI/Python/train_hybrid.py`
  - 新增训练侧 backend/continuation CLI
- `STS2AI/Python/evaluate_ai.py`
  - 新增评测侧 backend/ORT model CLI
- `STS2AI/Python/diagnostics/mcts_pipe_audit.py`
- `STS2AI/Python/diagnostics/mcts_decision_probe.py`
  - 两个诊断脚本都已支持切到 C# backend

## 二进制协议

### 新请求

`search_combat_mcts`

请求字段：

- `num_simulations`
- `c_puct`
- `dirichlet_alpha`
- `dirichlet_fraction`
- `max_step_budget`
- `final_action_mode`
- `final_action_top_k`
- `final_action_q_weight`
- `use_continuation_value`

返回字段：

- `action_index`
- `visit_counts`
- `visit_probs`
- `q_values`
- `priors`
- `root_value`
- `search_ms`
- `restored_ok`
- `snapshot_count`

数组语义：

- 全部按当前 root `legal_actions` 顺序返回
- 只返回当前 legal action 数长度
- Python 侧如需固定长度策略分布，继续自行 pad

### `load_ort_model` 元信息

返回里额外带出：

- `has_value`
- `has_deck_inputs`
- `has_continuation_output`
- `has_extra_scalars_input`

## CLI 用法

### 训练

```bash
python STS2AI/Python/train_hybrid.py ^
  --pipe ^
  --transport pipe-binary ^
  --combat-mcts-backend csharp ^
  --ort-model-path path\\to\\combat_actor.onnx
```

可选：

- `--combat-mcts-continuation-value`

说明：

- 如果 `--combat-mcts-backend=csharp` 但 transport 不是 `pipe-binary`，会自动回退到 Python backend。
- 如果 backend 是 `csharp`，但没有 `--ort-model-path`，则要求宿主进程已经手动加载过 ORT 模型。

### 评测

```bash
python STS2AI/Python/evaluate_ai.py ^
  --transport pipe-binary ^
  --combat-mcts-sims 16 ^
  --combat-mcts-backend csharp ^
  --ort-model-path path\\to\\combat_actor.onnx
```

可选：

- `--combat-mcts-continuation-value`

### 诊断

```bash
python STS2AI/Python/diagnostics/mcts_pipe_audit.py ^
  --checkpoint path\\to\\ckpt.pt ^
  --seed EVAL_001 ^
  --combat-mcts-backend csharp ^
  --ort-model-path path\\to\\combat_actor.onnx ^
  --output out.json
```

```bash
python STS2AI/Python/diagnostics/mcts_decision_probe.py ^
  --checkpoint path\\to\\ckpt.pt ^
  --trace-json trace.json ^
  --seed EVAL_001 ^
  --combat-mcts-backend csharp ^
  --ort-model-path path\\to\\combat_actor.onnx
```

## 当前验收状态

已完成：

- C# 宿主编译通过：`headless_sim_host_0991.csproj`
- Python 入口脚本通过 `py_compile`
- 新 CLI 参数已暴露到训练、评测、诊断脚本

尚未完成：

- ORT fixture parity
- Python MCTS 与 C# MCTS 的搜索 parity
- 固定 seed 的表现 smoke

## 建议验收顺序

1. 准备可用 combat ONNX，并确认 `load_ort_model` 成功。
2. 先做 evaluator parity。
   - 固定输入 fixture
   - 对比 `policy_logits` / `value`
3. 再做搜索 parity。
   - 固定 snapshot
   - 关闭 Dirichlet
   - 固定 seed
4. 再跑 `mcts_pipe_audit.py`
   - 确认 `restored_ok`
   - 确认 root/child 状态一致性
5. 最后跑小 seed 集 smoke。
   - `train_hybrid.py --combat-mcts-backend=csharp`
   - `evaluate_ai.py --combat-mcts-backend=csharp`

## 已知限制

- 当前 C# backend 只支持 `pipe-binary`
- C# 搜索依赖宿主中已加载的 ORT 模型
- 默认 backend 仍应保持 Python，直到 parity 和 smoke 都通过
