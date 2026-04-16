# Network Preset + Simulator Pool

本文档说明两项加速优化：
1. **Simulator 池化 + 预热**（无损加速，直接用）
2. **可配置 attention 层数 slim/full**（有损但可恢复）

---

## 1. Simulator 池化

### 问题

原实现 `BinaryBackedFullRunClient` 每个 worker 独立 `auto_launch`，
第一轮训练会同时冷启动 N 个 .NET 进程：
- 每个 .NET 启动 + JIT 编译 STS2 游戏逻辑：2-5s
- 8 个并发启动时会抢磁盘 + JIT 缓存：总计 60s+
- 更糟：连接失败会触发 `_restart_host_process()` 再冷启动一次

### 方案

`SimClientPool`（见 `networkV2/s6_training/train_full_run_v2.py`）：

```python
pool = SimClientPool(base_port=15527, size=8)
pool.warmup()  # 一次性启动 + dummy reset 触发 JIT

# 训练循环里 worker 直接取，不 close
client = pool.get(worker_id=0)
```

启动路径：
```
不池化：每轮重新起 N 个 sim ≈ 60s/轮
池化 + 预热：首次 ~30s，后续轮 ~0s
```

### 使用

训练脚本 `train_full_run_v2.py` 默认已启用池化，无需额外参数。

---

## 2. 可配置 Attention 层数

### 动机

profile 显示每 step 的 13-20ms 被 attention kernel launch 开销主导，
网络有 22 个 attention ops（详见 `networkV2Final.md §6`）：

```
BoardEncoder        3 层 self-attn
MechanismEncoder    2 层
ModifierEncoder     2 层
TurnPrefixEncoder   2 层
CombatMemoryEncoder 1 层
BuildMemoryEncoder  3 iters × cross
ActionContextualizer 6 cross blocks  ← 大头
DecisionCore        3 层 self-attn
                    合计 22 ops
```

每个 op ~0.5-1ms 固定开销。减层不改变模型容量上限，但**训练速度快 2-3x**。

### 预设

`networkV2/s5_net/network_config.py` 提供 3 档：

| Preset | attention ops | 速度 | 能力 | 用途 |
|--------|--------------|------|------|------|
| `full` | 22 | 1x | 100% | 最终正式训练 |
| `slim` | 7-8 | 2-3x | 80-90% | 快速迭代、早期 PPO |
| `tiny` | 3 | 5x | 60% | 调试、单元测试 |

slim 相对 full 的改动：
- 所有 encoder 层数降到 1
- ActionContextualizer: 6 cross → 2 cross（并行 3→1 + 串行 3→1 合并）
- OptionContextualizer: 同上
- DecisionCore: 3→1 层
- BuildSlots: 3 iter → 1 iter

### 使用

```bash
# slim 训练（推荐早期迭代）
python -m networkV2.s6_training.train_full_run_v2 --preset slim --num-workers 8 ...

# full 训练（最终版本）
python -m networkV2.s6_training.train_full_run_v2 --preset full --num-workers 8 ...

# tiny 调试
python -m networkV2.s6_training.train_full_run_v2 --preset tiny --num-workers 2 ...
```

### 程序化切换

```python
from networkV2.s5_net.network_config import preset_slim, preset_full
from networkV2.s5_net.unified_net import UnifiedNet

# slim 训练
net_slim = UnifiedNet(config=preset_slim())
# ... 训练 ...
torch.save(net_slim.state_dict(), "slim.pt")

# 切到 full 继续 fine-tune
net_full = UnifiedNet(config=preset_full())
report = net_full.load_compatible_params(torch.load("slim.pt"))
print(report)
# {'loaded': 234, 'skipped_shape': 47, 'missing': 12, ...}
```

---

## 3. Checkpoint 跨配置继承

### 基本原则

不同 preset 的层数不同 → 参数 shape 不同 → 不能直接 `load_state_dict(strict=True)`。

但大部分共享组件（tokenizer / policy_head / value_heads / decision_core 第 1 层 / build_encoder）
是 shape 一致的，可以部分加载。

### API

```python
UnifiedNet.load_compatible_params(state_dict, strict_shapes=True)
  # strict_shapes=True  : 只加载完全 shape 匹配的参数（推荐）
  # strict_shapes=False : 允许部分切片（slim→full 的新层会保留随机初始化）
```

### 典型工作流

**场景 A**：slim 预训练 → full 正式训练

```bash
# 第 1 阶段：slim，快速学到基础策略
python ... --preset slim --max-iterations 100

# 第 2 阶段：full，加载 slim 权重继续训练
python ... --preset full --checkpoint checkpoints/slim/unified_v2_final.pt --max-iterations 500
```

训练脚本的 checkpoint 加载逻辑应该自动调用 `load_compatible_params` 而不是
`load_state_dict(strict=True)`——**这个适配 TODO**。

**场景 B**：只用 full

```bash
python ... --preset full --max-iterations 500
```

**场景 C**：只用 slim（最快迭代）

```bash
python ... --preset slim --max-iterations 500
```

---

## 4. 预期效果

假设 d_model=384，8 worker，20ms/step：

| 配置 | 单步 forward | 每 episode | 总吞吐 |
|------|-------------|-----------|--------|
| full + 无池化 | 13-20ms | ~10s | 0.1-0.2 ep/s |
| full + 池化 | 13-20ms | ~4s | 0.5-1.0 ep/s |
| slim + 池化 | 5-7ms | ~2s | 1.5-2.5 ep/s |

slim + 池化能把吞吐推到 **15-25x 原始水平**。

---

## 5. 注意事项

1. **不要在一次训练里混用 preset**——切换 preset 意味着不同网络结构
2. **checkpoint 里记录 preset 名**（TODO：训练脚本应该存 config 到 checkpoint 元数据）
3. **slim 版可能学习更慢**，同样 N 轮后胜率可能低于 full，但总训练时间更短
4. **option_contextualizer_mode 和 contextualizer_mode 独立配置**，可以一边 slim 一边 full
