# 战斗网络 checkpoint 命名约定

## 背景

历史 checkpoint 使用 `mcts_model` 保存战斗策略/价值网络权重。这个名字容易误导，因为这份权重本身不是 MCTS 搜索器，只是 combat policy/value network，可被普通 NN 推理、teacher 训练、MCTS evaluator、ONNX 导出等路径复用。

## 新约定

新生成的 hybrid checkpoint 使用：

- `ppo_model`：局外/全局策略网络权重。
- `combat_model`：战斗策略/价值网络权重。
- `ppo_config`：局外/全局策略网络结构元数据。
- `combat_model_config`：战斗网络结构元数据。

`train_hybrid.py` 新保存的 checkpoint 会写 `checkpoint_schema_version = "hybrid.v2"`，并默认不再写 `mcts_model`。

## 兼容策略

加载器仍然兼容旧 artifact：

- 优先读取 `combat_model`。
- 若不存在，回退读取旧 `mcts_model`。
- standalone combat checkpoint 仍兼容 `model_state_dict`。
- 结构配置优先读取 `combat_model_config`，再回退旧 `mcts_config`。

因此旧 checkpoint 不需要批量重写；新训练产物会自然切到清晰命名。

## CLI 约定

新命令优先使用 `--resume-combat`。旧参数 `--resume-mcts` 暂时保留为兼容 alias。

真正表示搜索开关和搜索预算的参数仍保留 `mcts` 命名，例如 `--mcts`、`--mcts-sims`、`--combat-mcts-backend`，因为这些确实控制 MCTS 搜索。

## 合并脚本

如果局外 AI 和战斗 AI 分开训练，使用：

```powershell
python STS2AI/Python/scripts/merge_hybrid_checkpoints.py `
  --ppo-checkpoint STS2AI/Artifacts/noncombat_run/hybrid_final.pt `
  --combat-checkpoint STS2AI/Artifacts/combat_run/hybrid_final.pt `
  --base-checkpoint STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt `
  --output STS2AI/Artifacts/checkpoint_merges/hybrid_merged.pt
```

脚本默认规则：

- 从 `--ppo-checkpoint` 取 `ppo_model`。
- 从 `--combat-checkpoint` 取 `combat_model`，旧 checkpoint 会 fallback 到 `mcts_model` 或 `model_state_dict`。
- `entity_emb.*`、`symbolic_head.*` 这类运行时共享权重归 `ppo_model` 拥有，并同步写回 `combat_model`，避免加载 combat 权重时覆盖局外侧共享表示。
- 输出 `.merge_report.json` 和 `.merge_report.md`，用于检查来源、参数量、共享 key 对齐情况和 warning。
