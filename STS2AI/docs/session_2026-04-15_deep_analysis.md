# 2026-04-15 session 深度分析汇总

日期：2026-04-15（接续 `session_2026-04-15_stability_patches.md`）

前一份文档记录了本 session 起点的三个稳定性补丁（gc+empty_cache / force_remove schema / PPO buffer-skip）和 B.1 实验 10-iter 结果。本文档记录之后的深度分析、发现的系统问题、后续补丁草稿。

## 1. 训练系统现有架构图（截至 2026-04-15）

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              训练决策数据流                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Sim (C# HeadlessSim)                                                       │
│    ↓ binary protocol                                                         │
│   Python env.step → state{player, enemies, shop, card_reward, map, ...}      │
│    ↓                                                                         │
│   决策路由 (4 条互斥路径，按顺序尝试)：                                       │
│                                                                              │
│   1. 硬规则 override（确定性，不进 PPO buffer）                                │
│      ├─ act1_route_plan_keep   地图节点选择（99.8% 生效）                     │
│      ├─ shop_force_remove      shop 强制删牌（本 session 修好, 100%）        │
│      ├─ shop_remove_target     删牌界面选哪张（deterministic priority）      │
│      └─ rest/campfire 选项     (不详细，部分 deterministic)                  │
│                                                                              │
│   2. card_reward rerank（soft bias，仍走 PPO）                               │
│      ├─ boss_conditioned_card_bonus  boss.best_cards + _BOSS_CARD_PREFS     │
│      ├─ skada_prior (weight=0.15)    549 cards aggregated 胜率               │
│      ├─ learned_card_evaluator       PPO 学的 alpha=0.30                     │
│      └─ offline_noncombat_ranking    loss_weight 训练时混入 (当前 0.02)      │
│                                                                              │
│   3. combat 决策（纯 PPO + 辅助 rerank）                                      │
│      ├─ PPO actor + combat_safety_rerank  低血/高威胁时压制贪念              │
│      ├─ MCTS（当前关闭，combat_mcts_backend）                                │
│      └─ combat_monster_reward_weight      reward shaping                     │
│                                                                              │
│   4. 非战斗其他                                                              │
│      └─ PPO actor 直接决策（event、potion 使用等）                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 训练 loss 组成

```
total_loss = 
    ppo_policy_loss +                         # 标准 PPO clipped surrogate
    ppo_value_coeff × ppo_value_loss +        # critic regression
    -ppo_entropy_coeff × ppo_entropy +        # 探索保留
    boss_readiness_coeff × boss_readiness_loss  +  # 辅助 head
    offline_combat_teacher_loss × 0.2 +       # combat KD
    offline_noncombat_ranking_loss × 0.02 +   # 卡奖 KD
    combat_ppo_policy_loss +                  # combat 的独立 PPO head
    ...
```

### 1.2 Buffer 与 sample_weight

- `ppo_buffer` (StructuredRolloutBuffer)：
  - 非战斗 PPO 样本，通过 `PPOTrainerV2.update` 训练。
  - **不**支持 `sample_weight`（per-screen 自动加权除外）。
- `combat_buffer` (CombatRolloutBuffer)：
  - 战斗样本，`PPOTrainer.update` 里用 `sample_weights` 做 mini-batch 归一化加权。
  - 当前只受 `_hard_weight`（combat_safety_rerank 系数）影响。
  - **本 session 的补丁**：在 `_combat_ppo_pending` 里额外乘 `_room_type_weight`（monster=1.0 / elite=2.5 / boss=5.0）。

## 2. 本 session 发现的系统问题清单

按**严重性 × 是否可修**排序。

| # | 问题 | 现象 | 根因 | 修法 | 状态 |
|---|---|---|---|---|---|
| 1 | **force_remove schema mismatch** | shop 硬规则生效率 13.7% 而非 100% | 判断 `action.action == "remove_card"` 但实际是 `"shop_purchase"+label` | 改用 `state.shop.items[idx].category` 匹配 index | ✅ 已 commit（前一份 doc 记录）|
| 2 | **force_remove 污染 PPO buffer** | KL 从 0.005 飙到 0.87，policy 学坏 | 硬规则 override 但样本仍写 buffer（log_prob=0 vs 真实 P_new(remove)≈0.01 → ratio 爆）| `_ppo_pending=None` 跳过 buffer | ✅ 已 commit |
| 3 | **bigbatch 下 torch allocator 不释放** | ep/s 从 4.2 掉到 1.2，GPU 94% | iter loop 没调 `empty_cache`/`gc.collect` | iter 末加两行 | ✅ 已 commit |
| 4 | **Combat observation 看不出 Phase 2** | Waterfall 胜率 1.9%（最低），AI 看到 HP=999999993 还继续攻击 | `hp/max_hp=1.0` 跟正常 boss 100%HP 无差异；`ShowsInfiniteHp` 语义没暴露 | 加 `feat[39]=boss_critical_state` sentinel（复用 reserved slot，DIM 不变）| ✅ 已加代码，PoC 验证中 |
| 5 | **max_hp 编码在 Phase 2 下数值爆炸** | `max_hp/200 = 5×10⁶` 溢出 encoder 梯度 | 没 clip | `min(max_hp/200, 10.0)` | ✅ 已加代码 |
| 6 | **combat PPO sample imbalance** | monster ~85%, elite ~3%, boss ~6% transition；gradient 被 monster 主导，boss 打法学不会 | `_combat_ppo_pending` 写 sample_weight 只用 `_hard_weight`（combat_safety 系数），没有 room_type 维度 | 乘 `_room_type_weight = {monster:1.0, elite:2.5, boss:5.0}` | ✅ 已加代码，未 run |
| 7 | **skada 清洗强硬假设"人类永远对"** | `chosen_is_best_rate=1.0`，47% 样本真实是 chosen ≠ context-best，teacher label 教错 | `_normalize_scores` 强拉 chosen > max_other + 0.05 | 加 `soft_mode`：is_victory=1 时轻微 nudge，is_victory=0 不动 | ✅ 已加代码 + 派生 softened bridge，未接入训练 |
| 8 | **路线规划不按 boss 适配** | 三 boss 赢局都需要更多 monster+shop（尤其 Waterfall +1.88 monster/+0.51 shop），但路线规划给所有 boss 一样路径 | `_score_act1_route_plan` 没读 `boss_token` 参数 | 加 boss-specific weights | ❌ 待做 |
| 9 | **Waterfall Phase 2 识别错 38%** | 3/8 抽查失败局是"boss HP=10^9 了还继续打攻击"，40% Waterfall 败局属于这类 | 问题 #4 的直接后果；issue #6 的 sample imbalance 让 agent 没机会学 | 依赖 #4 + #6 | ⏳ 验证中 |
| 10 | **combat_safety_rerank 覆盖不到自损牌** | e026 局 agent 2 HP 时打「放血」(自损 3) 直接死 | 现有 rerank 规则只关心 incoming damage，没考虑 `card.self_damage > hp` | 加 hp-vs-self-damage 检查 | ❌ 待做 |
| 11 | **Soul Fysh 胜率偏低（应该是最好打的）** | 11.1% vs Lagavulin 2.2%、Waterfall 3.8% —— Soul Fysh 最高但人类 tier list 说它最简单，潜力更大 | 选卡 rerank `_BOSS_CARD_PREFS["soul_fysh"]` 可能偏好设置不优 | 审视手写清单，对比 skada boss_best_cards | ❌ 待做 |
| 12 | **SlumberingBeetle 本来直接改 src** | upstream 被手工加 `?.` null-safe | headless sim 下 `NCombatRoom.Instance=null` 导致 NRE | 迁 overlay，src 回滚 | ✅ 已 commit |

## 3. 深度挖掘发现

### 3.1 Build 层面（基于 iter 2293 的 2000 局 + B.1 5000 局）

- **boss 胜率 3.4%**（36/1064，所有 boss）
- **47% 失败局是"满血进 boss 但打不过"** → build 问题不是 HP 问题
- **ANGER 是胜负分水岭**：赢局 0.92/ep vs 输局 0.44/ep
- **赢局常见坏牌被避开**（SHRUG_IT_OFF / PERFECTED_STRIKE / THUNDERCLAP），但强牌（ANGER / HAVOC / CRUELTY）没有相应增多 → agent 学会"拒绝坏牌"但没学会"必选强牌"

### 3.2 Combat 层面（基于 8 局 Waterfall 失败局精读）

- **Phase 2 识别错 38%**：agent 面对 HP=10^9 的 boss 继续打攻击（愤怒 / 痛击）
- **block 估算错 62%**：正常阶段 block 远少于 incoming damage
- **自损牌不被拦**：低 HP 时仍打「放血」自杀

### 3.3 Skada 数据层面（基于 raw 1830 IRONCLAD 样本）

- `chosen_is_best_rate_by_context_score = 46.8%` → 过半人类选项不是 context 最优
- `chosen_is_best_rate_by_win_rate_delta = 46.0%` → 同样不到一半
- 两者不一致比例 21.7%
- `win_rate_delta` 受 Simpson 悖论影响严重（ANGER 的 win_delta = -13.73，跟玩法常识矛盾）
- **原清洗强拉 chosen=best，实际 teacher label 跟真实 context_score 排序分歧过半**

### 3.4 路线层面（基于 B.1 的 5000 局）

- 避 elite 硬规则 100% 生效（elite 触及率 0.02-0.07/局）
- 赢局 vs 输局：赢局多 **+1.54 monster/ep** + **+0.21 shop/ep**
- **Waterfall 对路径最敏感**：赢局多 +1.88 monster + 0.51 shop
- 但现有 `_score_act1_route_plan` 不按 boss 差异化 → 优化空间大

## 4. 本 session 打入代码的补丁与改动

### 4.1 已 commit 的（已 push）

- `46950f8` 修复 train_hybrid 三处训练稳定性问题（gc+empty_cache / force_remove schema / buffer-skip）
- `4a82ebb` 新增 bigbatch 与 offline_noncombat_ranking 训练配置档
- `b3dbef9` 把 SlumberingBeetle 的 headless null-safe 补丁迁到 overlay
- `3bc75a3` 记录 2026-04-15 session 稳定性补丁与 B.1 实验（前一份 doc）
- `fefb7f8` 新增 combat-only 训练链路
- `39405a0` 完善 offline_noncombat_ranking 生成管线（phase1 补尾）

### 4.2 已落代码但未 commit / 未 run（本 session 下半场）

1. **`rl_encoder_v2.py` — boss_critical_state sentinel feature**
   - `feat[1]` 加 clip（`min(max_hp/200, 10.0)`）防 Phase 2 数值爆
   - `feat[39]` 从 reserved → sentinel：`max_hp>10000 ∨ "deathblow" in intent ∨ is_hittable==False`
   - ENEMY_AUX_DIM 保持 40，旧 checkpoint 可直接 load

2. **`train_hybrid.py` — combat PPO room_type sample weighting**
   - `_combat_ppo_pending` 里 `sample_weight = _hard_weight × _room_type_weight`
   - 权重：monster=1.0, elite=2.5, boss=5.0
   - 抵消 monster 85% 样本主导 PPO gradient 的问题

3. **`skada/build_matchup_ranking_from_skada.py` — soft-mode 清洗**
   - `_normalize_scores` 加 `soft_mode` + `is_victory` 参数
   - soft mode：是胜局则 chosen 轻微 nudge（在 max_other-0.05 上方），非胜局不动
   - legacy mode 默认保留
   - 新 CLI `--soft-mode`，manifest 增 `soft_mode: bool`
   - 派生 `STS2AI/Artifacts/skada/ironclad_matchup_bridge_softened/`（1830 samples，chosen_is_best=46.8%）

### 4.3 代码草稿 / 想清楚待写

（未动代码，只记录思路）

- `_score_act1_route_plan` 加 boss-conditioned path weighting（路径层）
- `combat_safety_rerank` 补 hp-vs-self-damage 检查（防放血自杀）
- `_BOSS_CARD_PREFS["soul_fysh"]` 用 skada boss_best_cards 重算

## 5. 未完成的实验

| # | 实验 | 意图 | 状态 |
|---|---|---|---|
| E1 | Generator RANKB2（300 ep） | 扩 teacher 数据到 ~1650 samples | 后台跑着，预计 17:00-17:30 完成 |
| E2 | PoC sentinel 5-iter 小 run | 验证 feat[39] + clip 对 Waterfall 胜率的提升 | 跑着，iter 2294 数据出了 |
| E3 | sentinel + room_type_weight 组合 run | 验证 #4 和 #6 组合效果 | 未起 |
| E4 | softened skada bridge 接入训练 | 看 teacher label 不再硬假设人类对的效果 | 未起 |

## 6. 给接手方的建议

### 6.1 立即能跑的组合（按优先级）

**方案 X（最保守，验证单点）**：PoC sentinel run 结果看完 → 如果 Waterfall 胜率没起色，说明 imbalance 是主矛盾。

**方案 Y（推荐）**：sentinel + room_type_weight 一起起新 run（两条补丁已落代码，只要 resume from `hybrid_02293.pt` 跑 10 iter 就行）。预期：
- Waterfall 胜率 1.9% → 5%+
- 总 boss 胜率 3.4% → 6%+
- act1_clear_rate 2.1% → 4%+

**方案 Z（激进）**：Y + softened skada bridge 加 offline_noncombat_ranking。但三个改动叠加做 A/B 困难，建议先 Y。

### 6.2 中期工作

- 补问题 #8 路线规划 boss 适配
- 补问题 #10 combat_safety_rerank 自损牌拦截
- 补问题 #11 `_BOSS_CARD_PREFS` 从 skada 自动派生
- 写 combat-level teacher（类似 force_remove 但针对 Phase 2）

### 6.3 长期工作

- 暴露 `ShowsInfiniteHp` / `DeathBlowIntent` 到 binary 协议，而不是从 `max_hp>10000` 隐式推断
- 每个 boss 的 structured knowledge base（机制 JSON）
- 关键节点切 MCTS

## 7. 可复现 checkpoint 与数据清单

### Checkpoint
- `hybrid_02293.pt` — 干净起点（三补丁都吃到）
- `hybrid_02303.pt` — B.1 10-iter 末态

### 数据
- `STS2AI/Artifacts/offline_noncombat_ranking/from_hybrid_02293_20260415-031531/` — 274 samples teacher data
- `STS2AI/Artifacts/offline_noncombat_ranking/from_hybrid_02293_rankb2_20260415-110759/` — 300 ep 扩大版，预计 ~1650 samples（进行中）
- `STS2AI/Artifacts/skada/ironclad_matchup_bridge_softened/` — soft-mode 清洗后的 1830 samples

### 新 toml
- `STS2AI/Python/configs/hybrid_train_ironclad_bigbatch_2000ep.toml`
- `STS2AI/Python/configs/hybrid_train_ironclad_bigbatch_500ep_offline_noncombat_002.toml`
