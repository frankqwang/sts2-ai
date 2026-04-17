# STS2 AI 训练交接文档 — 2026-04-17 co12

> 上一交接见 `HANDOFF.md`（long1-long4 阶段）。本文接续讲 co6-co12（combat 专项 co-trainer 阶段）。
> 中文对话，用户英语看不懂。

---

## 启动新会话第一句

```
我是接手的下一个会话。读 STS2AI/docs/design/HANDOFF_co12.md 和
CLAUDE.md 后继续。co12 正在跑（PID tail /tmp/co12.log），
curriculum 只训 STS2 Act 1，目标 boss 稳 ≥20% 胜率。
```

---

## 🔴 硬性规范（先读）

1. **CLAUDE.md**（项目根自动加载）
2. **docs/design/SCHEMA_CONVENTION.md** —— 所有游戏 id/命名 **data-driven**（sim API > sqlite > 禁止手写）
3. **docs/design/DIAGNOSTICS_CONVENTION.md** —— 分析产物落 `runs/<exp>/analysis/`

---

## 1. 当前训练状态（co12）

- **日志**: `/tmp/co12.log`
- **Dump**: `STS2AI/Python/runs/co12/iter*.jsonl+.npz`
- **Checkpoint**: `STS2AI/Python/checkpoints/co12/`（每 20 iter 一个）
- **配置**: slim preset, 8 workers, lr 3e-5, target_kl 0.05, max_steps 200
- **起点**: `co8/cotrainer_iter120.pt`
- **Curriculum**: STS2 Act 1 only（act_index=0）
- **速度**: **~18s/iter**（Release build 后从 37s 减半），200 iter ETA ≈ 1h

### 启动命令

```bash
cd STS2AI/Python
nohup python -u -m networkV2.s6_training.combat_cotrainer \
  --preset slim \
  --checkpoint checkpoints/co8/cotrainer_iter120.pt \
  --num-workers 8 --base-port 15700 \
  --max-iterations 200 --episodes-per-iter 80 \
  --max-steps 200 --min-update-samples 128 \
  --lr 3e-5 --ppo-epochs 3 --mini-batch-size 64 \
  --value-warmup-iters 2 --target-kl 0.05 \
  --dump-dir runs/co12 \
  --output-dir checkpoints/co12 > /tmp/co12.log 2>&1 &
```

---

## 2. co6 → co12 演化

| Run | 关键改动 | 结果 |
|-----|---------|------|
| co6 | 初代 combat 专项 (slim, starter deck, hard curriculum) | stuck easy ~20%, FROG_KNIGHT 永 0% |
| co7 | + 真 starter curriculum, buffed deck, shaping | easy ~45%, 但 iter 30+ CLEAVE 卡不存在致 40% error |
| co8 | + game_catalog API data-driven, 修 card_exists check | easy peak 62.5%, elite peak 70%, boss 0% |
| co9 | + 真实 boss deck (7 from `combat_teacher/tactical_v1_replay_*`) | boss 仍 0%，发现 KAISER_CRAB sim crash |
| co10 | + KAISER_CRAB blacklist | boss iter15 首次 11% (LAGAVULIN=act4) |
| co11 | + act_index 过滤到 act 1 only | 3 boss（CEREMONIAL_BEAST/THE_KIN/VANTOM）+ 15 monster |
| **co12** | + Release build, + EVENT_ENCOUNTER 过滤, max_steps 300→200 | **2x 速度**（37s→18s/iter）|

---

## 3. 已修复的硬编码 bug（7 大类）

全部符合 **"sim API > sqlite > 禁止手写"** 规范：

| 位置 | 原问题 | 修复方式 |
|------|-------|---------|
| `_enemy_core_token` | 手写 5 个 power slots（`metallicize/vulnerable` 等 STS1 名）| 改用 `GAME_CATALOG.monster_power_vocab()` 前 19 个高频 power class |
| `_player_token` | 同上 6 个 power | 改用 `PLAYER_POWER_VOCAB` 前 17 个 |
| `bosses.py / elites.py` | 手写 10 个 encounter_id（STS1 名）| 清空，改 `_auto_derive_configs` 从 GAME_CATALOG 派生 |
| `_DISCARD_TRIGGER_POWERS` | 猜的 Silent 相关 power list | 删除，改用 `card.keywords` 通用字段 |
| `buffed_ironclad_deck` | 含 STS1 `CLEAVE` 致 sim crash | 每张卡 `GAME_CATALOG.card_exists()` 校验 |
| `CURRICULUM` | 手写 encounter 分组 | `_derive_curriculum_pools` 从真实 monster powers 派生（block_heavy vs starter） |
| `route_compiler risk/value` | STS1 名字 + 归一化 bug | 修复归一化，元数据仍需校验 |

---

## 4. Sim API 扩展

### C# 侧: `Program.cs` 的新 endpoint

- **`game_catalog`**: 返回全部静态数据（缓存 9605x 加速）
  - encounters: `{encounter_id, room_type, monster_ids, act_index}`（81 entries）
  - monsters: `{monster_id, class_name, powers}`（102 entries）
  - cards: `{card_id, class_name, card_type, rarity, target_type, base_cost, tags, keywords, gains_block, is_x_cost}`（577）
  - relics: `{relic_id, class_name, rarity, tags}`（289）
  - potions: `{potion_id, class_name, rarity}`（64）
  - powers: `{class_name, base_classes, is_debuff_hint}`（267）

### Python 侧: `networkV2/s1_schema/sim_catalog.py`

- `GAME_CATALOG` 单例，`attach_sim(client)` 启动时预取缓存
- 优先级: game_catalog API > combat_catalog API > sqlite
- 辅助方法: `encounters()`, `monster_powers(id)`, `card_exists(id)`, `find_cards(...)` etc.

---

## 5. 核心发现

### 5.1 Boss 胜率 0% 真因

**不是**：agent 笨 / PPO 参数错  
**是**：
1. 早期用 STS1 卡名（CLEAVE）→ sim reset 失败 → 每场 defeat（co7-co8）
2. KAISER_CRAB_BOSS sim C# 有 NullReferenceException bug → 每场 1 step defeat
3. 跨 act 全选 → 多数 boss 是 act 2/3/4（人类也难赢）
4. ARCHITECT_EVENT_ENCOUNTER 9999 HP 假怪混入 pool → 全 timeout

修复后（co12）目标：Act 1 boss 稳 ≥20%。

### 5.2 AI 牌序学习情况

**没学会**。诊断：
- co8 iter 5/80/120 monster action distribution：top1 永远是 action 0 占 40-43%，entropy_ratio 稳在 1.08（接近 uniform）
- 所谓 "先 Inflame 再 Heavy Strike" 的 combo 决策 = random
- 根本原因：`turn_damage_lookahead` head 是个 weak supervision，没 `play_order` 显式信号
- 解决路径（未实施）：加 `play_order_head` 或 forward simulation 或 MCTS

### 5.3 速度瓶颈

- **85 steps/sec 是 sim pipe IPC 上限**（Debug build），Release 后 ≈ 170 steps/sec
- 单 run 200 iter 从 ~2.1h → **~1h**
- 进一步加速路径：
  - **Rollout + Training pipelining**：current serial 30s rollout + 5s train 串行；拆成两 thread 重叠 → ~10-20% 提升（train:rollout = 1:6 比例限制）
  - 加 workers 8→16（需 RAM 够 + 扩 port 范围）

---

## 6. 重要文件索引

### 训练
- `networkV2/s6_training/train_full_run_v2.py` —— full-run trainer（long1-long5，已弃置）
- `networkV2/s6_training/combat_cotrainer.py` —— combat 专项 (co6-co12)
- `networkV2/s6_training/real_boss_decks.py` —— 真实 deck 加载（从 `Artifacts/combat_teacher/tactical_v1_replay_*`）
- `networkV2/s6_training/deck_eval_cli.py` —— checkpoint 评测

### Schema / 数据驱动
- `networkV2/s1_schema/game_vocab.py` —— power vocab（已被 sim_catalog 替代）
- `networkV2/s1_schema/sim_catalog.py` —— **统一数据访问点**
- `networkV2/s2_config/mechanism_registry.py` —— `_auto_derive_configs` 动态派生
- `networkV2/s2_config/act1/bosses.py/elites.py` —— 已 deprecated

### Sim C#
- `STS2AI/ENV/Sim/Runtime/HeadlessSim/Program.cs` —— `game_catalog` endpoint + `BuildGameCatalog()`
- `STS2AI/Python/constants.py` —— `SIM_HOST_EXE` 优先 Release

### 诊断工具（放 `runs/<exp>/analysis/`）
- `networkV2/s7_diagnostics/live_monitor.py`
- `networkV2/s7_diagnostics/plot_win_rates.py`
- `networkV2/s7_diagnostics/trajectory_analyzer.py`
- `networkV2/s7_diagnostics/policy_evolution.py`

---

## 7. 剩余 TODO（按优先级）

### P0 立即
- [ ] 监控 co12 到 iter 50+ 看 act1 boss 胜率能否 ≥ 20%
- [ ] 如 co12 boss 仍 0% → 检查 CEREMONIAL_BEAST/THE_KIN/VANTOM 是否也有 sim bug

### P1 架构补全
- [ ] **Monster initial powers**（`game_catalog.monsters[].powers` 现在返回空）：需扫源码 `AddPower()` 或从 sqlite fallback
- [ ] **Power trigger 元数据**（onPlay/onHit/onTurn）：需 PowerModel 子类方法反射
- [ ] **Spectator 侧 game_catalog API**（user 明确要求）：共享 lib 架构
- [ ] **card_tags.py / relic_tags.py / action_semantics.py / combat_env_wrapper.py**: 剩余硬编码审计

### P2 性能
- [ ] Rollout + training pipelining（double-buffer samples + thread split）
- [ ] 8 → 16 workers

### P3 战略能力
- [ ] `play_order_head` 让 agent 真正学牌序
- [ ] MCTS in-turn search（boss 战起用）
- [ ] Imitation pre-train（用 combat_teacher 数据监督学习 boss 策略）

---

## 8. cron 监控

Session 内 cron（会话结束即失效，不跟 session 走）：

```
7,22,37,52 * * * *  → 检查 co12 进度
```

用户每隔 15 分钟看到训练状态报告。换新会话需重新 `/cron` 设置。

---

## 9. 快速恢复命令

```bash
cd /c/dev/sts2-ai/STS2AI/Python

# 看 co12 最新
tail -20 /tmp/co12.log

# 看 train processes
tasklist | grep -E "python|headless"

# Plot
python -m networkV2.s7_diagnostics.plot_win_rates runs/co12

# 评测 checkpoint（独立 sim port）
python -m networkV2.s6_training.deck_eval_cli \
  --checkpoint checkpoints/co12/cotrainer_iter60.pt \
  --preset slim --port 15800 --n-trials 3
```

---

## 10. 心态 & 风格注意

- 用户严禁手写硬编码（多次提醒），**任何涉及游戏名字的数据必须从 sim API / sqlite 来**
- 报告格式：表格 > 段落；KL/胜率/时间等指标保留 2-3 位小数
- 说"不知道"比猜更受欢迎（user 的原话："你先别可能，你深挖数据确认下"）
- Cron 报告格式：一两句搞定正常情况，异常立即详细报告
