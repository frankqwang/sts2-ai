# ONNX导出修正记录

## 背景

2026-04-13 对 C# combat MCTS 链路继续排查后，确认剩余偏差不在 C# 搜索本体，也不在 C# 特征编码，而在 Python 侧 ONNX 导出器。

问题根因有两类：

1. `export_actor_onnx.py` 在构建 `CombatPolicyValueNetwork` 时，没有把 checkpoint 中的共享 `symbolic_head.*` 真正实例化并加载进去。
2. 默认 `opset 17` 在当前 PyTorch 2.11 的 `dynamo` 导出路径下会先升到 18，再尝试降回 17；这个转换可能失败，严重时会导致导出命令打印成功但目标文件没有落盘。

## 本次修正

文件：

- `STS2AI/Python/export_actor_onnx.py`

修正内容：

- 导出前会合并 `ppo_model` 和 `mcts_model` 中的共享权重。
- 如果 checkpoint 中存在 `symbolic_head.out_proj.weight`，会按实际 `proj_dim` 构造 `SymbolicFeaturesHead`，再把 `symbolic_head.*` 权重完整加载到 combat 网络。
- 新增 `--exporter auto|dynamo|legacy`。
- `auto` 的策略调整为：
  - `opset >= 18` 时优先 `dynamo`
  - `opset < 18` 时优先 `legacy`
- 默认 `--opset` 改为 `18`。
- `export_from_training_snapshot(...)` 的默认 `opset_version` 也同步改为 `18`，避免训练期间自动导出继续走旧默认值。

## 验收结果

固定样本：

- checkpoint：
  - `STS2AI/Artifacts/hybrid_training_main_attention_mcts100_short/hybrid_2env_20260412-234103/hybrid_final.pt`
- root state：
  - `STS2AI/Artifacts/tmp/eval001_root_state.json`

数值对齐结果：

- 导出模型：
  - `STS2AI/Artifacts/tmp/combat_actor_fix_default_auto.onnx`
- 对比结论：
  - `top1_match = true`
  - `max_abs_logit_diff = 1.087784767150879e-06`
  - `mean_abs_logit_diff = 3.725290298461914e-07`
  - `value_abs_diff = 5.960464477539063e-08`

审计结果：

- `STS2AI/Artifacts/tmp/mcts_pipe_audit_python_eval001_fixed_export.json`
- `STS2AI/Artifacts/tmp/mcts_pipe_audit_python_eval002_fixed_export.json`
- `STS2AI/Artifacts/tmp/mcts_pipe_audit_python_eval005_fixed_export.json`
- `STS2AI/Artifacts/tmp/mcts_pipe_audit_csharp_eval001_fixed_export.json`
- `STS2AI/Artifacts/tmp/mcts_pipe_audit_csharp_eval002_fixed_export.json`
- `STS2AI/Artifacts/tmp/mcts_pipe_audit_csharp_eval005_fixed_export.json`

固定 seed `EVAL_001 / EVAL_002 / EVAL_005` 上：

- Python backend 与 C# backend 的根动作标签一致
- `root_save_load.matches = true`
- `search_restore.matches_root = true`
- 子节点状态 hash 一致

Smoke 结果：

- `STS2AI/Artifacts/tmp/eval3_mcts16_csharp_fixed_export.json`
- `STS2AI/Artifacts/tmp/eval3_mcts16_python_fixed_export.json`

这 3 个 smoke seed 上：

- Python backend：`3/3` 有效，`0 error`，`avg_floor = 10.3`，`avg_time_s = 9.4`
- C# backend：`3/3` 有效，`0 error`，`avg_floor = 10.3`，`avg_time_s = 6.3`

## 当前结论

这轮修正后：

- C# encoder 与 Python encoder 已对齐
- C# ORT 与 Python ORT 已对齐
- ONNX 与 PyTorch 在固定 root 上已对齐到 `1e-6` 量级
- C# MCTS 与 Python MCTS 在固定 seed 审计上已达到根动作 parity

后续如果再出现 C# / Python 决策差异，优先检查：

1. 实际加载的 ONNX 是否由新 exporter 导出
2. 是否误用了旧的 `opset 17` 文件
3. 运行时是否加载了和 checkpoint 不匹配的 ONNX
