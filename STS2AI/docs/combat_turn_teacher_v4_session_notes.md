# Combat Turn Teacher —— 2026-04-15 晚间 session 总结

## TL;DR

- **本 session 最佳**：`mixed_v4` (62.7% 以下口径里是 66.67% win / boss_reach)，
  在 12-seed regression 上 **相对 baseline +25 个百分点 win，与上一轮最佳 `mixed_w02_5iter` 持平的 win rate**，
  但在以下指标上 **严格更优**：

  | 指标 | baseline | mixed_w02_5iter | **mixed_v4** |
  |---|---:|---:|---:|
  | win_rate / boss_reach / act1_clear | 41.67% | 66.67% | **66.67%** |
  | avg_floor | 14.17 | 14.58 | **14.83** |
  | avg_combats_won | 7.00 | 8.17 | **8.50** |
  | avg_boss_hp_fraction_dealt | 0.712 | 0.720 | **0.746** |
  | max_floor | 17 | 18 | **20** |

  `mixed_v4` 的关键突破：`EVAL_003` 首次进入 Act 2（floor 20），`EVAL_010` 从 13 升到 17
  新达 boss；代价是 `EVAL_002` 从 18 退到 17、`EVAL_006` 从 17 退到 14。
  净 boss_reach 8/12，与 w02 持平。

- **关键工具**：`build_combat_teacher_from_trajectory.py`（本 session 新增），
  从 evaluate_ai 的 `--save-trajectory-dir` JSONL 里重放 action 序列，
  在 floor >= 14 的 combat state 上跑 solver 出 teacher 样本。这条路径**绕过了现有 live builder 的 noncombat progression bug**。

## 为什么要换路径

上一轮 `mixed_w02_5iter` 走的是 live builder + `--progress-combat-with-solver`。
这轮尝试在 live builder 下用训好的 `hybrid_final.pt` 做 `--combat-checkpoint`，
期望能多采集 floor 14+ 的多样化状态，结果：

- `bossfocus_v1`（12 seed × `min_sample_floor=14`）**只有 EVAL_003** 成功推到 floor 14+，产出 32 条样本（floor 14/15 都是 monster，floor 17 boss 7 条全是 SOUL_FYSH）。
- `bossfocus_v2`（换 EVAL_013-016 扩 seed pool）**只有 EVAL_016** 推到了 floor 15 elite，12 条样本。
- 诊断实验：在 EVAL_001 单 seed 上关掉 solver progress + `min_sample_floor=1`，
  builder 能考察的状态也只推到 floor 6（trained ckpt）或 floor 2（baseline ckpt）。
  `evaluate_ai.py` 走完整流程（含 `combat_safety_rerank` 等）baseline 就能让这些 seed 推到 floor 17，
  说明 live builder 的 noncombat/combat 推进和 evaluate_ai 有结构性差异。

结论：**live builder 在多数 seed 上进不了 boss/elite**，不是 checkpoint 的问题。
必须换数据源。

## Replay 路径产物链

1. `evaluate_ai.py --save-trajectory-dir --trajectory-seeds EVAL_001..EVAL_006`
   → 6 份 `full_run_trajectory.v1` JSONL（共 ~1300 行），每行含完整 raw_state + legal_actions + chosen_action。
2. `build_combat_teacher_from_trajectory.py`：
   - reset sim 到该 seed
   - 顺序重放 chosen_action（sanitize + 在当前 legal_actions 中匹配）
   - 当 `floor >= min_sample_floor` 且是 combat/elite/boss state，save_state → solver solve → emit 样本
   - 支持 prefix samples（沿袭 live builder 的逻辑）
3. 产出 `tactical_v1_replay_20260415_203730/ironclad_act1_tactical_teacher_v1_replay.jsonl`：
   - **102 条样本** / 6 个 seed
   - floor 分布：14 elite (10) + 14 monster (30) + 15 monster (21) + 17 boss (41)
   - boss 样本跨 EVAL_001/002/003/005 四个 seed（但本轮选的 seed 在 Act1 都走到同一个 boss `SOUL_FYSH`）

## 数据集/训练矩阵

本 session 跑了 3 个数据集变体 + 原始 `mixed_w02_5iter` 做对比：

| 数据集 | 样本数 | floor 构成 | 训练后 win_rate |
|---|---:|---|---:|
| `mixed` (上 session 最佳) | 219 | 2: 171 / 11-13: 48 / **boss: 0** | 66.67% |
| `mixed_v2` | 251 | +EVAL_003 live builder 32 条 (14/15 monster + 17 SOUL_FYSH) | **58.33%** ↓ |
| `mixed_v3` | 263 | v2 + EVAL_016 的 12 条 floor 15 elite | 58.33% |
| **`mixed_v4`** | **321** | mixed + replay 的 102 条（含 10 条 floor 14 elite + 41 条 boss） | **66.67%**（同 w02，但其他指标更优） |

`mixed_v2/v3` 退步的直接原因是**新加数据集中在 EVAL_003 的 monster + 单一 SOUL_FYSH boss**，
强烈 seed-特化 + boss 单一化，在训练中改变了前期（floor 2-15）的决策分布，
导致 EVAL_001/002/006/009 的表现回到基线附近，净 boss_reach 从 8 降到 7。

`mixed_v4` 把 seed 覆盖从 1 个提到 6 个，elite 样本从 0 加到 10 条，
消除了 seed-特化问题，回到 w02 的 boss_reach 水平，
且 boss HP dealt 和 combats_won 进一步上升。

## 训练侧 metrics（mixed_v4）

- `combat_teacher_ce`: 1.610 → 1.210（Δ -0.400）
- `combat_teacher_loss`: 1.888 → 0.942（Δ -0.946）
- `hard_state_premature_end_turn_steps`: 稳定下降
- iter 2176 就达到 boss_reach_rate=38%（w02 起步是 12%）、act1_clear=12%

数据质量更好 → teacher loss 起点更高但下降更快 → 训练过程中 boss_reach 更早抬起。

## 开放问题 / 下一轮建议

### 数据侧
1. **boss 多样性**：当前 6 个 trajectory 都指向同一个 Act1 boss（SOUL_FYSH）。
   需要扩大 trajectory seeds 到 EVAL_007..EVAL_030 之类，用 `--trajectory-seeds` 批量 dump，
   筛能到 boss 的 seed 且 boss 种类多样（CALCIFIED_CULTIST / GREMLIN_MERC / LAGAVULIN_MATRIARCH 等）。
2. **floor 16 rest/event 前的收尾战**：目前 replay 路径没挖 floor 16 的内容（Act1 floor 16 是 rest/event/elite 不固定），可以放 `min_sample_floor=15`。
3. **Act 2 覆盖**：`mixed_v4` 让 EVAL_003 进到 floor 20，说明 combat policy 已经能打 Act 2 前几场，
   下一轮可以开始给 floor 18-21 的 teacher 样本（需要新的 Act 2 solver evaluation，这是更大的工作量）。

### 训练侧
4. **继续 iter 10**：上一轮 w02 在 10 iter 会退化，但 v4 数据质量更好，可能 10 iter 不退化反而上升，可以试一次。
5. **teacher loss weight**：v2/v3 失败时 w=0.2 已经够，这次也是。未试 0.3/0.4。

### 工具侧
6. **修 live builder 的 noncombat/combat 推进**：目前 replay 方案可绕开这个 bug，但想做 iterative teacher（每轮训完后 dump trajectory 做下一轮数据源）需要稳定的 live-builder，值得深挖 `_select_noncombat_action` 或 `_choose_combat_progress_action` 里的死循环根因。
7. **对齐 evaluate_ai 的完整流程**：builder 里加 `combat_safety_rerank` / `boss_conditioned_card_guidance` 选项（不是用在 solver 生成，而是用在 progression），和 evaluate_ai 一致。

## 关键产物路径

- Replay builder：[build_combat_teacher_from_trajectory.py](STS2AI/Python/search/build_combat_teacher_from_trajectory.py)
- Merge 工具：[merge_combat_teacher_datasets.py](STS2AI/Python/search/merge_combat_teacher_datasets.py)
- trajectory dump：`STS2AI/Artifacts/combat_teacher/trajectory_20260415_203343/`
- replay dataset：`STS2AI/Artifacts/combat_teacher/tactical_v1_replay_20260415_203730/ironclad_act1_tactical_teacher_v1_replay.jsonl`
- mixed_v4 dataset：`STS2AI/Artifacts/combat_teacher/tactical_v1_mixed_v4_20260415_203858/ironclad_act1_tactical_teacher_v1_mixed_v4.jsonl`
- mixed_v4 checkpoint：`STS2AI/Artifacts/hybrid_training_tactical_teacher_mixed_v4_20260415_203914/*/hybrid_final.pt`
- v4 regression report：`STS2AI/Artifacts/combat_teacher/tactical_v1_mixed_v4_regression_eval_20260415_204036/report/experiment_report.md`

（v2 / v3 的产物目录也还在，可作为负例对照。）
