# 交接:Combat v7 长 run + 速度实验 + 超参调优

**日期**:2026-04-19
**接上次**:`handoff-2026-04-19-proto-combat-complete.md`
**session 产物**:combat_rl_proto_v4/v5/v6/v7 四轮长 run 实验 + 速度优化评估 + PR #4/#5 merge

---

## 完成的工作

### 1. Combat long run 实验(4 轮对比)

| Run | 命令差异 | 结果 | 决策 |
|-----|---------|------|-----|
| **v4** | eps=20, lr=1e-5, pool=200 | 11 iter Easy 平均 ~28%, Hard 4-6% | 用户切配置停(改 eps=50) |
| **v5** | eps=50, lr=1e-5, pool=400 | 13 iter Easy 平均 ~17%(低 v4),Hard 1 次 | 停(大 batch 伤学习速率) |
| **v6** | eps=20, lr=**3e-5**, pool=400 | 5 iter 连跌 28%→8%,KL=0.14 超 target 0.03 | 停(lr 太激进 early stop) |
| **v7** | eps=20, lr=1e-5, pool=400 | **100 iter 跑完,ERR=0**,峰值 Easy 47.9% / Med 23.7% / Hard 5.8% | 完整数据 |

### 2. v7 完整结果

```
room_type    iters  last_wr  best_wr
  boss         100     0.0%     5.8%
  elite        100     2.4%    23.7%
  monster      100    18.5%    47.9%
```

**Hard 命中 iter**(共约 28 次):11 / 12 / 15 / 19-21 / 24-26 / 33 / 36 / 42 / 44-45 / 50 /
62 / 64 / 67 / 70 / 73 / 75 / 78 / 80-81 / 84 / 90 / 92-93

**关键观察**:
- 整体**没看到向上单调趋势**,iter 25-45 是 peak(Easy 40-48% / Med 16-24%),iter 60+ 后平台甚至略降
- Boss 战单个 encounter 最好 100%(KIN 67%, VANTOM 100%),但混合 chain 里 boss 胜率 0(chain 里 act3 boss 样本太少)
- `ERR=0` 全程(proto wire + sqlite + is_known_relic 源头过滤联合效果)

### 3. 速度优化实验

尝试 CUDA graph + 轻量优化 D,**全部失败或无效**:

| 方案 | 结果 | 原因 |
|-----|------|------|
| D: `torch.inference_mode()` + 合并 3 次 `.item()` sync 为 1 次 | 无明显收益(forward 23→29ms 噪声级别) | GPU compute 本身主导,sync 不是大头 |
| A: `--use-cuda-graph` (GraphRunner) | ERR=6 全部失败 | `CUDA error: operation not permitted when stream is capturing`(多 worker capture 冲突 PyTorch 99820 残留) |

**CUDA graph 真正挂的地方**:GraphRunner 2 worker 并发 capture,共享 default CUDA stream 污染。handoff 里提过需要 `torch.Generator()` per dropout 或用 WSL2 + torch.compile 才能根治。

### 4. 其他改动

| 改动 | 位置 | 状态 |
|-----|------|-----|
| `inference_mode` + 合并 sync | `combat_cotrainer.py::combat_rollout` | ✓(无明显收益但保留,不 regression) |
| `--use-cuda-graph` 默认关 | `combat_cotrainer.py` | 保持 opt-in,修好前禁用 |
| v7 dump (402 files, 100 iter 完整) | `Artifacts/runs/combat_rl_proto_v7/` | ✓ 本地,不入 git |
| v7 checkpoint | `Artifacts/checkpoints/combat_rl_proto_v7/cotrainer_final.pt` | ✓ 本地 |

---

## 主要 PR / commit

- **PR #4** (MERGED):proto wire 统一 + ERR 根因修复 + sqlite rebuild + BC/diagnostics 脚本恢复
- **PR #5** (MERGED):删 Assets/datasets/game_knowledge_catalog + README 修正 + skada v0.103.2 精确版本过滤
- 本 session 最后加:`combat_cotrainer.py::combat_rollout` 改 `inference_mode` + 合并 `.item()` sync(待提 PR)

---

## 下次 session 的优先级

### 1. 速度(收益最大)

**修 CUDA graph**(Windows 可做,3-4h):
- 给 GraphRunner 加 per-worker `torch.cuda.Stream()` 隔离
- Capture 用 mutex 串行化(只第一个 worker capture,后面 worker 复用 graph)
- Patch 所有 dropout 用独立 `torch.Generator()` 绕 PyTorch 99820
- 预期 3-5x forward 加速

**或 WSL2 + torch.compile**(一次性切环境,~2h):
- 启用 Windows VM Platform + WSL 功能(重启一次)
- WSL 装 Ubuntu + CUDA toolkit + 匹配 PyTorch + Triton
- `torch.compile(mode="reduce-overhead")` 自动 fuse + graph
- 预期 5-10x forward + fusion 收益

### 2. 训练质量(没速度问题可直接做)

- **Boss curriculum**:skada chain replay 默认每 ep act1→act3,boss 样本 <1/ep。加 `--room-type-weight boss=0.4` 等加权 sampling
- **BC combat head warmstart**:当前 BC 只训 non-combat,combat head 纯随机起步。可以先用 skada action data 做 combat BC 一遍
- **v7 继续接跑**:用 `--checkpoint combat_rl_proto_v7/cotrainer_final.pt` 接力,看 200 iter 能不能突破 Hard 平台期

### 3. 代码清理(低优先级,无运行风险)

- `BinaryProtocol.cs` 里 V2 不用的 `BuildStatePayload / BinarySessionState / Write*State / BinarySymbolKind` 物理删除(共享 helper 搬去 `ProtoStateBuilder`)
- `Program.cs::ProcessBinaryRequestAsync` + `ProcessBinary*` 方法删除(sim 已拒绝 `--protocol bin`)
- 预估 2-3h

---

## 硬规则(延续)

1. V2 模块唯一允许:`PipeConnection + ProtoCodec` / `CombatSession`
2. 禁 `from env.pipe_client import PipeClient`
3. 训练 skada 数据默认 `v0.103.2`
4. `is_known_relic` / `is_known_card` 过滤 skada 垃圾数据保留
5. sqlite(source_knowledge)要和当前 sim rebuild 对齐
6. 文档全中文

---

## 文件清单

### 本 session 修改

- `STS2AI/Python/networkV2/s6_training/combat_cotrainer.py`:`inference_mode` + 合并 `.item()` sync
- `STS2AI/docs/handoff/handoff-2026-04-19-combat-v7-longrun.md`(本文)

### 本 session Artifacts(本地,不入 git)

- `Artifacts/runs/combat_rl_proto_v4/`:11 iter 早期停(eps=20 v1)
- `Artifacts/runs/combat_rl_proto_v5/`:13 iter(eps=50 实验失败)
- `Artifacts/runs/combat_rl_proto_v6/`:5 iter(lr=3e-5 实验失败)
- `Artifacts/runs/combat_rl_proto_v7/`:**100 iter 完整 dump(ERR=0)** ← 主数据
- `Artifacts/checkpoints/combat_rl_proto_v7/cotrainer_final.pt`:最终 checkpoint
- `Artifacts/runs/smoke_cuda_graph*.log`:CUDA graph 失败诊断
- `Artifacts/runs/smoke_inference_mode.log`:D 优化 smoke

---

## 速度数据(参考)

v7 iter 稳态(warm 后):
- forward_ms: 23-29 ms
- step_ms: 0.85-1.5 ms (sim RPC)
- total/step: 25-32 ms
- combats/iter: 210-250
- time/iter: 55-130 s(平均 ~80 s)
- **100 iter 总耗时:~140 min**

要再跑 200 iter 直接约 4.5h,所以**下次速度优化优先级最高**。
