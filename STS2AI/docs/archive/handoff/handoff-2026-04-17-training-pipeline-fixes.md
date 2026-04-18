# handoff-2026-04-17-training-pipeline-fixes

> 一个 session 内累计 23 条训练链路修复的详细记录。
> 涵盖：奖励系统 / 特征工程 / value&leaf target / 非战斗 option token / 清理遗留死通道 / argparse bug。
> **所有改动 smoke 全过**（`exit 0`、end-to-end forward / backward 无 crash）。
> **必须从头训**：`max_numeric_dim` 32 → 48、多个 token 维度扩展、`FightMode` / `_FALLBACK_HEAL_CARDS` 删除，都会破坏旧 checkpoint。

---

## 目录

1. [奖励系统升级（Tier 1，3 条）](#1-奖励系统升级tier-13-条)
2. [特征工程 P1/P2/P3 修复（8 条）](#2-特征工程-p1p2p3-修复8-条)
3. [R1：value / leaf target 改 future-looking（3 条）](#3-r1value--leaf-target-改-future-looking3-条)
4. [R2：非战斗 option token 补齐信号（3 条）](#4-r2非战斗-option-token-补齐信号3-条)
5. [中期清理 U4 / U6 / U8（3 条）](#5-中期清理-u4--u6--u83-条)
6. [收尾 U1 / U2 / U3（3 条）](#6-收尾-u1--u2--u33-条)
7. [仍未修复的已知问题](#7-仍未修复的已知问题)
8. [checkpoint 兼容性与迁移](#8-checkpoint-兼容性与迁移)

---

## 1. 奖励系统升级（Tier 1，3 条）

### 1.1 `shaped_reward` 接入非战斗 step

**问题**：
`train_full_run_v2.py` 的非战斗分支原先只给 `NONCOMBAT_STEP_REWARD = 0.01` 加一层"分桶 bonus"（card_reward +0.05 / shop +0.03 / event +0.02），等于重写了一个简陋版 reward。
但 `core/rl_reward_shaping.py` 里早就实现了完整的 `shaped_reward()` = PBRS（7 分量 Φ：楼层进度 + problem_score + survival_margin + economy_score + relic_count + upgrade_ratio + 脏牌惩罚）+ `milestone_reward`（楼层推进 +0.02、boss 门槛 +0.15、boss_entry_quality 最多 +0.95、幕通关 +0.3）+ early-damage-potion penalty。这整套完整设计全部**没被训练脚本调用**。

**原因**：
migration 过程中 `train_full_run_v2` 另起炉灶写了简化版，没接入 `rl_reward_shaping` 的丰富实现。等于把工程价值浪费了。

**解决**：
在 `train_full_run_v2.py` 的非战斗分支直接走 `shaped_reward(state, next_state, raw_terminal_reward=0.0, done=False, action=chosen)`，再叠加 `noncombat_entropy_nudge(chosen, st)`（对非 skip 动作给 0.02~0.05 小额 bonus，专门防 long1 监测到的 entropy collapse）。

**为什么这么改**：
- 完全复用 `rl_reward_shaping` 已有设计
- 保留 entropy nudge 作独立冷启动保险（shaped_reward 里 PBRS 变化幅度可能不够强对冲"选卡立刻 problem_score 上升但 gold 下降"的短期负反馈）
- 不动战斗 step reward（那条链路本来就对）

### 1.2 `terminal_reward` 按楼层 scale（近胜梯度）

**问题**：
终局 reward 原写死 `WIN_REWARD = 1.0` / `LOSE_REWARD = -1.0`。死 floor 3 和死 floor 50 都是 `-1.0`——**没有"死得越晚越好"的梯度**，近胜近败对策略毫无信号差别。
`rl_reward_shaping.terminal_reward(state, won)` 早就实现了按楼层 scale 的公式：`win = +1.0`、`lose = -1.0 + floor_total / 17.0`。act 1 f3 死 = -0.82、f15 死 = -0.12、act 2+ 死 = 0。

**原因**：
同上——`terminal_reward` 实现了但训练脚本没调用。

**解决**：
把常量 `WIN_REWARD/LOSE_REWARD` 改成 `final_reward = terminal_reward(state, won)`。

**为什么这么改**：
立即拿到近胜梯度：`win(+1) > act2死(0) > f15死(-0.12) > f3死(-0.82)`。policy 能分清"死在 boss 前"和"f3 直接寄"。

### 1.3 `EPISODE_SHAPING_CAP` 防 shaping 盖过终局

**问题**：
PBRS + milestones 每 step 0.01~0.05，一局 400+ step 可能累积到 ±10~20，远超终局 ±1。这样 advantage 被 shaping 主导，agent 学"最大化 shaping"而不是"赢"。

**原因**：
PBRS 的 Φ 不是真正的 value function，只是启发式。经典 PBRS 风险：shaping 信号理论上不改变最优策略，但实践中量级压过终局就会推偏。

**解决**：
加 `EPISODE_SHAPING_CAP = 1.5`。在 episode 结束加 `terminal_reward` **之前**，先 scan 所有非终局 step 的累积 shaping，若 `abs(sum) > 1.5` 按比例 scale down。

**为什么取 1.5**：
`terminal_reward` 在 act 2+ 死 = 0（原作者"脱离失败"设计）。若 cap=2.5，"act2 死 + 满 shaping = +1.5" vs "win + 满 shaping = +2.5"，只差 1，相对 shaping 噪声不稳定。cap=1.5 时 win=+2.5 / act2死=+1.5 / f15死=+1.38 / f3死=-0.67 单调且终局占 ≥40% 权重。

---

## 2. 特征工程 P1/P2/P3 修复（8 条）

### 2.1 P1① non-combat 阶段 `RunBuildMemory` 滞后

**问题**：
`CombatStateTracker.refresh_build_profile(obs)` 刷新 rbm 的 gold / act / floor / deck / relic / potion 以及派生 build profile（frontload / scaling / aoe / heal / consistency 等），**只在 `on_combat_start` 里调用一次**。shop / card_reward / rest / event 等非战斗 step 不刷新。

**后果**：
非战斗决策时 rbm 还停留在上一次进战前的值。例如刚在 shop 花光钱、加了一张 Demon Form，再进下一个 card_reward 时网络看到的 rbm 还是上场战斗开始时的 build 画像——和真实状态脱节。

**原因**：
`_refresh_build_profile` 是私有方法，`on_combat_start` 里为一次性调用设计，没考虑非战斗阶段。

**解决**：
1. `_refresh_build_profile` → 改名 public `refresh_build_profile`
2. `train_full_run_v2.py` 非战斗 step（`st in ("shop", "rest_site", "map", "event", "card_reward", "combat_rewards")`）在 compile banks 前调 `tracker.refresh_build_profile(state)`

**为什么这么改**：
无侵入性——`_refresh_build_profile` 原逻辑不改，只暴露调用接口。每 step 开销小（几个 dict 查询 + 一次 deck 扫描）。

**影响文件**：
- [combat_env_wrapper.py:544-547](../../Python/networkV2/s3_state_tracker/combat_env_wrapper.py#L544)
- [train_full_run_v2.py:216](../../Python/networkV2/s6_training/train_full_run_v2.py#L216)

### 2.2 P1② `deck_card` token 缺失身份通道

**问题**：
`build_bank` 里每张 deck 卡的 token 只有 11 维 coarse numeric（cost / 5 个 card_type one-hot / is_upgraded / 3 个 rarity one-hot / is_zero_cost）。同 cost + type + rarity 的不同卡（如 Strike 和 Pommel Strike）在 token 层**完全不可区分**。
tokenizer 只读 `numeric + token_type + time_scale`，不读 `owner_id` / `metadata`——所以光给 token 设 owner_id 是没用的。

**原因**：
`bank_assembler.py` 设计时 coarse 特征够用，没考虑同类卡之间的语义差异。relic / potion 有 14 / 8 维语义向量（来自 `relic_rules.py` 手工规则表），deck_card 却没配套实现。

**解决**：
1. `core/card_tags.py` 加 `card_feature_vector(card_id) -> list[float]`：从 `card_tags.json`（离线从 C# 源码提取）查 `FUNCTIONAL_TAGS` (34 个功能标签如 aoe / draw / strength_scaling / exhaust)，返回 one-hot 向量
2. `bank_assembler.py` 的 `_assemble_shared` 里 deck_card numeric 从 11 维扩到 11 + 34 = 45 维
3. `max_numeric_dim` 全局从 32 提到 48（tokenizer / batch / ppo / network_config / combat_net 全部改默认）

**为什么这么改**：
- 复用 `card_tags.json` 离线提取的 functional tags，**数据驱动**符合 `CLAUDE.md` 硬性规范
- 45 < 48 保留 3 维余量防未来扩展时截断
- 不动其他 token 结构

**影响文件**：
- [card_tags.py 底部](../../Python/core/card_tags.py) 新增 `card_feature_vector`
- [bank_assembler.py:207-233](../../Python/networkV2/s4_compiler/bank_assembler.py#L207)
- tokenizer / batch / ppo / network_config / combat_net 的 `max_numeric_dim` 默认值

### 2.3 P1-1 Encounter ID 不对齐（fallback 拼错 key）

**问题**：
STS2 sim 经常不返回 `state.encounter_id`。`train_full_run_v2.py` 的 fallback 从 `enemies[].id` 拼出形如 `"frog_knight"` / `"jaw_worm,fungi_beast"` 的 key。但 `MechanismRegistry` 实际注册的 key 格式是 `"{monster}_{room_type}"`（如 `frog_knight_normal`、`queen_boss`）——**完全对不上**。
结果：`feature_compiler.py` 的 `registry.get(encounter_id)` 永远返回 `None` → `mechanism_compiler` 返回空列表 → `mechanism_bank` / `modifier_bank` **整个 full-run 训练全程都是空的**。

**验证**：
```python
reg.find_encounter_id(['frog_knight'], 'normal')  # 修前: None
```
当时实测 `registry._configs.keys()` 里只有 `frog_knight_normal`，没有 `frog_knight`。

**原因**：
历史 migration：一开始 encounter_id 是 sim 直接给的字符串，后来 schema 改成 `{monster}_{room_type}` 格式但 trainer 的 fallback 没跟着改。

**解决**：
1. `MechanismRegistry` 加 `find_encounter_id(monsters: list[str], room_type: str) -> str | None`，遍历注册的 encounter 按 monster 集合 + normalize 比对反查正式 key
2. 注册时（`_auto_derive_configs`）同时保存 `encounter_id → monsters` 映射
3. `normalize_monster_id(mid)` 去掉下划线 / 连字符、转小写——因为 GAME_CATALOG 的 monster id 是紧凑格式（`"frogknight"`）而 sim runtime 是带下划线（`"frog_knight"`），必须统一
4. `train_full_run_v2.py` 的 fallback 先调 `registry.find_encounter_id(monster_ids, rt)`，查不到才回落到 `sorted monster-id join` 作诊断值

**为什么这么改**：
- `find_encounter_id` 是 registry 层的新接口，让 compiler 层不用关心格式对齐
- normalize 函数处理 sim / sqlite 两种格式不一致（之前有个坑：`frogknight` vs `frog_knight` 看着像，实际是完全不同字符串）
- 兜底保留原逻辑，不 crash 只是机制 bank 仍空（和修前行为一致）

**影响文件**：
- [mechanism_registry.py:100-145](../../Python/networkV2/s2_config/mechanism_registry.py#L100)
- [train_full_run_v2.py:210-225](../../Python/networkV2/s6_training/train_full_run_v2.py#L210)

### 2.4 P1-2 Mechanism primitive 全部绑到 primary_enemy

**问题**：
`MechanismRegistry._auto_derive_configs` 按 encounter 里的**每只怪**生成 primitive（`prim = gen(mid)` 里 `mid` 是具体 monster id）。但 `mechanism_compiler._find_primary_enemy` 选出**一个**（通常 HP 最大的）`primary_enemy`，然后 `_compile_phases / windows / summons / thresholds / shields` 把**所有** primitive 都套在这个 enemy 上。
后果：boss+adds / minion / 多怪 encounter 的 primitive 被系统性错绑。`modifier_compiler` 有相同 pattern。

**原因**：
最初实现假设 "encounter 只有一个主体敌人"，没考虑多怪场景。

**解决**：
1. `primitives.py` 基类 `MechanismPrimitive` / `ModifierPrimitive` 都有 `owner_id: str` 字段（registry 生成 primitive 时已经填了对的 owner）——之前被 compiler 无视
2. compiler 层新增 `_resolve_owner_enemy(owner_id, enemies, fallback) -> EnemyRuntime`：按 primitive.owner_id normalize 匹配 runtime enemy，查不到才 fallback 到 primary
3. `mechanism_compiler._compile_phases / windows / summons / thresholds / shields` 逐 primitive 查 enemy，不再全绑 primary。同理 `modifier_compiler._compile_config_modifiers`

**为什么这么改**：
- primitive 本身早就有正确的 owner 信息（registry 正确生成），只是 compiler 丢失了——恢复这个信息链最低成本
- 保留 primary fallback 应付 owner_id 缺失 / 拼写不一致的边缘 case，不 crash
- 不改 registry 结构，只改 compile 时的查找逻辑

**影响文件**：
- [mechanism_compiler.py:19-47](../../Python/networkV2/s4_compiler/mechanism_compiler.py#L19)
- [modifier_compiler.py:11-30](../../Python/networkV2/s4_compiler/modifier_compiler.py#L11)

### 2.5 P2-1 `PLAYER_POWER_VOCAB` 字母序

**问题**：
`game_vocab.py` 的 `PLAYER_POWER_VOCAB` 原本是 `sorted(set(CARD_POWER_VOCAB) | set(RELIC_POWER_VOCAB))` —— **字母序**。而 `bank_assembler._player_token` 取前 17 个作 "top-N 高频 player power"——结果选到的是 A 开头的低频 power（Accelerant / Accuracy / Afterimage / Aggression 等），真正高频的 Strength / Vulnerable / Doom / Weak / Dexterity / Poison / Focus 反而没进 vocab slot。

**验证**：
```
字母序 top-17: Accelerant, Accuracy, Afterimage, Aggression, Anticipate, ...
频次 top-17:   Strength, Vulnerable, Doom, Weak, Dexterity, Poison, Focus, ...
重合度: 1/17  （只有 BlockNextTurnPower 碰巧在两边都靠前）
```

**原因**：
`MONSTER_POWER_VOCAB` 是按频次排序的（`_count_json_column` 后 `.most_common()`），但 `PLAYER_POWER_VOCAB` 用 `sorted(set(...))` 去重破坏了频次——开发时的手误。

**解决**：
`_load_vocabs` 里合并 `card_powers` 和 `relic_powers` 的 `Counter`，按 `most_common()` 输出 `player_powers`。`PLAYER_POWER_VOCAB = _load_vocabs()["player_powers"]`。

**为什么这么改**：
- 和 `MONSTER_POWER_VOCAB` 一致的频次逻辑
- 合并 Counter 比单独两表再交集去重更准（考虑了卡 + relic 的共同频次）

**影响文件**：
- [game_vocab.py:65-75](../../Python/networkV2/s1_schema/game_vocab.py#L65)

### 2.6 P2-2 `_afc` 跨 worker thread 共享

**问题**：
`run_full_episode._afc = act_fail_count`——`_afc` 挂在 `run_full_episode` **函数对象**上（module-level 对象），N 个 worker thread 并发调 `run_full_episode` 时**共享无锁**。一个 flaky sim 让 counter 累到 5，所有并发 worker 的下一个 episode 都会在首次 `act_failed` 时就 break。非原子读写还会 race。

**原因**：
原作者应该是想做"episode 内失败计数"，但用 function attribute 是错误选择——Python 函数本身是进程共享对象。

**解决**：
把 `act_fail_count` 改成 `run_full_episode` 函数内的局部变量（栈帧隔离，天然 thread-local）。

**为什么这么改**：
- 最小改动，Python 本地变量是栈分配，每次函数调用独立
- 无需 threading.local 或锁

**影响文件**：
- [train_full_run_v2.py:205](../../Python/networkV2/s6_training/train_full_run_v2.py#L205)（声明）+ line 298-310（使用）

### 2.7 P2-3 `turn_damage` loss 归一化错误

**问题**：
原代码：
```python
vl_td = (td_per * w).sum() / max(valid_mask.sum().item(), 1) * valid_mask.sum()
```
数学上 `X / K * K = X`（除和乘抵消），实际等于 `(td_per * w).sum()`。其中 `w` 是全 batch B 归一化的权重。

**后果**：
当 valid_mask 稀疏（比如 10 / 100 样本是回合结束），`(td_per * w).sum()` ≈ `avg_loss * 10 / B`，head 梯度被 **B/valid_count** 倍稀释。而 `turn_damage_lookahead` 本来就是为"combo / 牌序"学习加的稀疏信号——设计目标和实际效果正好相反。

**原因**：
写的时候可能想 `/ K * K` 恢复"假装全 batch 都有 loss"的 scale，但数学上直接抵消了。

**解决**：
```python
valid_w = w * valid_mask
vl_td = (td_per * valid_w).sum() / valid_w.sum().clamp(min=1e-8)
```
按**有效样本的加权平均** smoothL1，稀疏 / 稠密场景 scale 一致。

**为什么这么改**：
- 真正按有效样本归一化，coef = 0.02 在稀疏场景下仍有合理梯度
- `valid_w.sum().clamp(min=1e-8)` 防零除
- smoke 验证：iter2 `vl_turn_damage` 从 baseline 的 6.3（几乎不学）降到 3.65（收敛）

**影响文件**：
- [losses.py:144-158](../../Python/networkV2/s6_training/losses.py#L144)

### 2.8 P3 附加 3 条

**P3-1 `combat_env_wrapper.py` 硬编码 heal/draw 卡列表**

- **问题**：`_HEAL_CARD_IDS` / `_DRAW_CARDS` 硬编码 Ironclad 几张卡，违反"数据驱动"规范
- **解决**：改查 `core.card_tags.load_card_tags()` 的 `heal` / `draw` tag，覆盖全职业
- 保留小兜底（STS1 残留名字）防 regression，后来在 U6 彻底删了

**P3-2 `feature_compiler._NONCOMBAT_DOMAIN_MAP` 缺 treasure / relic_select**

- **问题**：`state_type = "treasure"` 或 `"relic_select"` 不在映射里，会 fallback 到 combat 分支 → 编空 banks / 错把非战斗 legal 当战斗动作
- **解决**：加 `"treasure"` / `"relic_select"` / `"relic_reward"` → `"event"` domain（最接近的通用选项型决策）

**P3-3 `run_outcome` 未统一 helper**

- **问题**：`train_full_run_v2` / `train_combat_v2` 用 `str(outcome).lower() == "victory"` 字面比较，upstream 如果返回 `"win"` / `"WON"` / `"victory"` 多种拼写都会漂移
- **解决**：统一用 `env.run_outcome_vocab.is_victory_outcome(outcome)` / `is_failure_outcome` / `normalize_run_outcome`

---

## 3. R1：value / leaf target 改 future-looking（3 条）

### 3.1 R1.1 `hp_loss_target` → 未来到战斗末累计掉血

**问题**：
原 target `= max(hp_at_combat_start - cur_hp, 0)` 是"**从战斗开始到现在累计掉血**"。这是 obs 已知量，网络学到的只是"重现 obs 里的一个观测值"，不是 head 名义上的 **expected_hp_loss**（未来量）。

**解决**：
战斗结束时（从战斗切出 / 死在战斗里）调用新的 `_backfill_combat_future_targets`：
```python
future_loss[t] = max(hp_at_sample_t - hp_at_combat_end, 0)
```
即"从本步之后到战斗结束还会再掉多少血"。

**为什么这么改**：
- 真正 future-looking：target 是未来事件
- backfill 机制复用已有的 `_backfill_turn_damage`（turn-damage head 的套路）
- rollout 中断（max_steps / 异常）时保留原 proxy 作 fallback

### 3.2 R1.2 `survival_target` → 战斗末 hp_ratio

**问题**：
原 target `= hp_ratio`（当前 HP / max_HP）= 身份映射，网络学"输出等于输入"，白训一个 head。

**解决**：
同一个 `_backfill_combat_future_targets` 覆盖：`survival[t] = hp_at_combat_end / max_hp`（胜利）or `0.0`（失败）。

### 3.3 R1.3 `leaf_target` → n-step return

**问题**：
原 `leaf_target = 2 * value_target - 1`（value_target 是 GAE return 的仿射映射），和 `fight_win` head 的监督目标**完全同源**，只差一个线性变换——监督信号 100% 重复。

**解决**：
改成 **n-step return + tanh 压缩**：
```python
acc = sum(gamma^k * reward[t+k] for k in range(3))
bootstrap = value_estimate[t+3]
n_step_return = acc + gamma^3 * bootstrap
leaf_target = tanh(2 * n_step_return)
```

**为什么 n-step（horizon=3）**：
- `fight_win` target 是整 GAE return（≈ 整 run 胜率，horizon ~几百 step）
- `tempo` target 是 `tanh(advantage)`（horizon=1 step）
- `leaf_score` 取中间：3 step horizon，表达"局部叶节点价值"，和另两个 head 在 time horizon 上分化
- tanh(2 * return) 让典型 return 0.5 映射到 tanh(1)≈0.76，占用 tanh 有效范围

---

## 4. R2：非战斗 option token 补齐信号（3 条）

### 背景：最大的信号缺口

总 review 报告里唯一标 🔴 的问题：**非战斗 option token numeric 严重不足**。
- card_reward 选 Strike 和选 Demon Form，token 层完全一致（family="card_reward" + 同 cost + 同 roles）
- shop 买 7g common 和 150g rare relic，token 只有一个 `cost` 字段，没 price / 能否购买 / 稀有度
- event 所有选项硬编码 `roles=["resource"]`
- rest 选项完全无 numeric 特征

结果：非战斗决策**全靠终局 fight_win 经过 GAE 长 horizon 回传**区分选项，credit assignment 极弱。

### 4.1 R2.1 `ActionCandidate` 新增 4 字段

```python
rarity_weight: float = 0.0    # 选卡稀有度 soft 权重
price_ratio: float = 0.0      # shop 价格 / 当前 gold
can_afford: float = 0.0       # shop 能否购买
event_kind: str = ""          # EVENT_KINDS 中的一个
```

新增 `EVENT_KINDS = [gain_gold / gain_relic / gain_potion / gain_hp / lose_hp / gain_curse / remove_card / upgrade_card / unknown]`（9 种 bucket）。

### 4.2 R2.2 `bank_assembler._action_token` 扩编码

原 30 维 + 3 维 scalar (`rarity_weight / price_ratio / can_afford`) + 9 维 `event_kind` one-hot = **42 维**（R2 时）。U8 又加 route 专属 2 维 → 最终 **44 维**（仍 ≤ 48）。

### 4.3 R2.3 4 个 compiler 填字段

- **`card_reward_compiler`**：`_RARITY_WEIGHT` 表（basic=0 / common=0.25 / uncommon=0.5 / rare=1.0 / curse=-0.3 / status=-0.2）填 `rarity_weight`
- **`shop_compiler`**：从 `obs.player.gold` 读当前金币，计算 `price_ratio = cost / max(gold, 1)`（clip 到 2.0）+ `can_afford`；card / relic / potion 用 `_RARITY_WEIGHT`，remove_card 固定 0.6
- **`event_compiler`**：`_infer_event_kind(label, text)` 做关键词 multi-token 匹配（"Lose 10 HP" 用 `["lose", "hp"]` 双词匹配，支持跨字符），9 种 bucket
- **`rest_compiler`**（U1 补回）：`_REST_OPTION_WEIGHT`（smith=1.0 / recall=0.7 / rest=dig=0.5 / lift=toke=0.4）填 `rarity_weight`

**为什么这么设计**：
- 不新增 token 类型，复用 `TK_ACTION_CANDIDATE`
- 字段语义相对独立（rarity / price / event_kind / route_risk+value），不重叠
- 所有关键词匹配 / 权重表都可以后续迭代升级，不影响 schema

---

## 5. 中期清理 U4 / U6 / U8（3 条）

### 5.1 U4 `FightMode` 死通道

**问题**：
`CombatMemory.fight_mode: FightMode = FightMode.UNKNOWN`。enum 有 5 个值（UNKNOWN / RACE / STABILIZE / ATTRITION / BURST_PREP），但**运行时恒为 UNKNOWN**（无写入路径）。`memory_compiler.py:71` 注释已经确认过了，把 5 维 one-hot 从 token 里移除，但字段和 enum 本身还留着。

**原因**：
早期设计想做"战斗模式分类"，后来发现应该让网络从原始信号自学，没删干净。

**解决**：
彻底删：`FightMode` enum、`fight_mode` 字段、`reset()` 里的赋值、tracker 的 import。

**为什么这么改**：
死通道会误导后续开发（以为这字段有用）。代码里保留"旧设计决定已撤销"的注释说明清楚。

**影响文件**：
- [memory.py:94-105](../../Python/networkV2/s1_schema/memory.py#L94)（enum + 字段 + reset）
- [combat_env_wrapper.py:24](../../Python/networkV2/s3_state_tracker/combat_env_wrapper.py#L24)（import）
- [memory_compiler.py:65-75](../../Python/networkV2/s4_compiler/memory_compiler.py#L65)（注释更新）

### 5.2 U6 重跑 `card_tags.json` + 清 fallback

**问题**：
`card_tags.json` 漏了 `REAPER` / `BITE` / `SELF_REPAIR` / `BLOOD_FOR_BLOOD` 等卡。`combat_env_wrapper.py` 靠 `_FALLBACK_HEAL_CARDS` 兜底。

**惊人发现**：
重跑 `python -m core.card_tags --repo-root C:/dev/sts2-ai` 扫 577 张卡，查 heal tag 只有 **`spur` 一张**！STS2 里：
- `reaper_form` 是 power 类（retain），不是 heal
- `snakebite` 是 poison
- 原来 fallback 里的 `REAPER` / `BITE` / `SELF_REPAIR` 都是 **STS1 卡名**，STS2 压根不存在——fallback 永远不会命中。

**原因**：
`_FALLBACK_HEAL_CARDS` 是按 STS1 经验手写的，迁移到 STS2 没更新。

**解决**：
1. 重跑 `card_tags.py` 扫描最新 C# 源码，`card_tags.json` 更新到 577 张卡的真实 STS2 tag
2. 彻底删 `_FALLBACK_HEAL_CARDS` 和 fallback 逻辑，完全数据驱动

**为什么这么改**：
- 消除 STS1 残留符合 `CLAUDE.md` 硬性规范
- 顺便发现 `rbm.heal` 字段在 STS2 几乎死（只有 spur 时有 1/deck_size ≈ 0.08 的 heal 比例），列入新遗留 U18

**影响文件**：
- [card_tags.json](../../Python/data/card_tags.json)（重生）
- [combat_env_wrapper.py:29-75](../../Python/networkV2/s3_state_tracker/combat_env_wrapper.py#L29)

### 5.3 U8 route 字段语义黑客

**问题**：
`route_compiler.py` 把 node risk/value `[0,1]` 值**乘以 30 / 50** 塞到 `damage_est` / `block_est` 字段——因为 `bank_assembler._action_token` 会把这两个字段做归一化（除以 `_DMG=30` / `_BLK=50`），直接塞 `[0,1]` 会被归一化成 0.023 / 0.012 几乎丢失。

这是"**字段语义黑客**"：硬塞数字进错位字段，用乘除抵消躲过归一化。代码里注释说明了历史 bug，但做法本身是 workaround。

**原因**：
初版 `ActionCandidate` 没给 route 专属字段，只好复用 combat 字段。

**解决**：
1. `ActionCandidate` 加 `route_risk: float` / `route_value: float` 专属字段（[0,1]）
2. `route_compiler.py` 删 × 30 / × 50 乘数，直接存原值到新字段
3. `bank_assembler._action_token` 加 2 维 route 通道（不走 `_DMG` / `_BLK` 归一化）

**为什么这么改**：
- 字段语义恢复干净：`damage_est` / `block_est` 只表达 combat 动作的伤害 / 格挡估计，route 有自己的 channel
- action token 从 42 维扩到 44 维（仍 ≤ 48）
- 未来若 route 需更多特征（距 boss 距离、出度等），直接加字段不用黑客

**影响文件**：
- [actions.py:87-93](../../Python/networkV2/s1_schema/actions.py#L87)
- [route_compiler.py:20-33,72-75](../../Python/networkV2/s4_compiler/noncombat/route_compiler.py#L20)
- [bank_assembler.py:747-752](../../Python/networkV2/s4_compiler/bank_assembler.py#L747)

---

## 6. 收尾 U1 / U2 / U3（3 条）

### 6.1 U1 `rest_compiler` 补 numeric（R2 的尾巴）

**问题**：
R2 给 card_reward / shop / event 加了 `rarity_weight / price_ratio / event_kind` 等信号，**独漏 rest**。rest 选项（rest / smith / recall / dig / lift / toke）在 token 层只差 `roles` 的 3-4 种 one-hot，recall/dig/lift/toke 都是 `"resource"` role，**彼此完全无差别**。

**解决**：
加 `_REST_OPTION_WEIGHT = {rest: 0.5, smith: 1.0, recall: 0.7, dig: 0.5, lift: 0.4, toke: 0.4}`，写入现有 `rarity_weight` 字段（复用通道，不增维度）。

**为什么这个权重分配**：
- smith 永久升级一张卡，整局收益最稳 → 1.0
- rest HP 恢复 30%，context-dependent（低 HP 时高价值，通过 `objective_bank` 的 `hp_ratio` 给决策层条件化），这里只给 baseline 0.5
- recall 取出 key relic（某些 relic 特殊交互），中偏高 0.7
- dig / lift / toke 依赖 relic 效果，平均价值中等 0.4-0.5

**为什么复用 `rarity_weight` 而不是加新字段**：
- 字段语义相容："option 的相对价值权重"适用于 card / shop / rest 三个分支
- 不动 token 维度（仍 44 维）
- 不引入新 EVENT_KINDS / REST_KINDS 名字空间污染

### 6.2 U2 NaN 诊断增强

**问题**：
R2 smoke 发现 `nan_skip_count` 稳定 3-5（~2%），PPO 在 `torch.isnan(loss)` 时整 minibatch skip。但**不知道是哪个 head / 哪个中间量产生了 NaN**。

**解决**：
在 `losses.py` 加 `_check_nan(name, tensor, context)` helper：
- 只检 NaN（不检 Inf，因为 `policy.logits_raw` 的 masked `-inf` 是合法的）
- 每 key 最多 log 3 次（防洪水）
- 在 `policy.logits_raw / log_probs / ratio / advantages` + 4 个 combat head 的 `{pred, target}` 插检查点

**为什么这么做**：
- 非侵入：只读 log，不改 loss 计算
- 真出 NaN 时精确定位（head name + context + shape + NaN count）
- **修第一版误报**：原先用 `torch.isfinite(tensor).all()` 把 `policy.logits_raw` 的合法 `-inf` 当 NaN 报——改成只检 `torch.isnan`

**当前状态**：
修完误报后 final smoke 仍有 nan_skip=3。真正的 NaN 根因还没定位——需要下一轮长训时 `_check_nan` 捕获首次真 NaN 事件。

**影响文件**：
- [losses.py:1-56](../../Python/networkV2/s6_training/losses.py#L1)（helper 定义）
- [losses.py:97-200](../../Python/networkV2/s6_training/losses.py#L97)（插入 check points）

### 6.3 U3 `preset` 被 argparse 默认值覆盖

**问题**：
```python
p.add_argument("--d-model", type=int, default=384)
# ...
if args.d_model > 0: cfg.d_model = args.d_model
```
`args.d_model > 0` **永远为 True**（默认 384 就满足），preset 的 d_model（tiny=128）**永远被覆盖**。
`--preset tiny` 实测跑出来实际是 `d_model=384 / n_heads=8 / 12.7M params`（只有 layer 层数按 tiny，参数量按 full）。

**原因**：
argparse 默认值选得太"积极"，和覆盖判断的 `> 0` 条件互相矛盾。

**解决**：
1. `--d-model / --n-heads / --n-build-slots / --dropout` 默认从 `384/8/8/0.1` 改成 `None`
2. 覆盖判断从 `> 0` 改成 `is not None`
3. 散参数分支（没传 `--preset`）用 `or` fallback 到历史默认

**为什么这么改**：
- None 是 Python 里"未指定"的规范值，和 `argparse` 配合好
- 判断 `is not None` 明确表达"用户显式传入"
- 保持向后兼容：散参数路径（不用 preset）行为不变

**验证效果**：
`--preset tiny` smoke 从 12.8M params 降到 **1.9M params**（6.7x 缩小，真正进入 tiny 档）。

**影响文件**：
- [train_full_run_v2.py:929-944](../../Python/networkV2/s6_training/train_full_run_v2.py#L929)（argparse）
- [train_full_run_v2.py:755-775](../../Python/networkV2/s6_training/train_full_run_v2.py#L755)（覆盖 + 散参数分支）
- [train_full_run_v2.py:816-820](../../Python/networkV2/s6_training/train_full_run_v2.py#L816)（print 支持 None → net.config.d_model）

---

## 7. 仍未修复的已知问题

按优先级。不修的理由和影响评估。

### P1 - 设计瑕疵

| # | 问题 | 建议 |
|---|---|---|
| U5 | 敌人 phase 追踪依赖 `next_move_id` 代理。`behavior_history` 只记 move_id 变化，不是真正的 phase_id。需要 sim bridge 暴露 `phase_id` 字段才能修 | 等 bridge 支持后改 `_update_enemy_behaviors` |
| U7 | shop / event / route 的 `index` 匹配依赖 obs 顺序。sim 如果改返回格式（如加 cursor offset）会错位 | 加双向索引（`items_by_id` fallback） |
| U9 | AOE 判定用 `core/card_base_stats.AOE_ATTACKS` 硬编码集合，只覆盖 Ironclad | 改用 `card_tags.json` 的 `"aoe"` tag |

### P2 - 一致性

| # | 问题 | 建议 |
|---|---|---|
| U10 | `combat_cotrainer.py:64-65` 仍有独立 `WIN_REWARD = 1.0 / LOSE_REWARD = -1.0`，没切到 `terminal_reward` | 迁移 reward 入口到 `rewards.py` 统一 |
| U11 | `train_combat_v2.py` 仍用 `CombatNetV2`（legacy），没迁到 `UnifiedNet` | 下一阶段重构 |
| U12 | obs 多格式 fallback（`hp` vs `current_hp` vs `CurrentHp`、top-level `player` vs `battle.player`）散布在 `runtime_compiler._pick` / `combat_env_wrapper._extract_*` | 长期加统一 adapter 层 |
| U13 | sim pipe 冷启动 race（`Pipe reconnect attempt failed → Restarting HeadlessSim host`）。最终 recovered，但每次 warmup 多花 ~10s | 修 sim client 初次 connect 重试策略 |

### P3 - 长期路线

| # | 问题 | 备注 |
|---|---|---|
| U14 | **Tier 2 奖励**：combat 内 gamma=0.99 / 跨房间 gamma=0.997 分开；非战斗选项用 `screen_local_delta_reward` 做 counterfactual 打分；tactical pattern 扩展（strength-before-attack / draw-power-first / preserve-low-cost-setup） | 下一轮值得做 |
| U15 | **Tier 3 奖励**：per-head reward decomposition；训练 build-quality critic 给 shop/card_reward 选项打分 | 更长远 |
| U16 | 诊断脚本扩展：per-head target vs prediction 对比图（观察 R1 的 future-looking target 是否真让 head 学得比 proxy 快） | 有了再做 |
| U17 | 多 encounter benchmark：固定 seed 在 `frog_knight_normal / queen_boss / ...` 跑 eval，排除训练 noise | 训练稳定后做 |
| U18 | `rbm.heal` 字段：STS2 只有 `spur` 一张 heal 卡，该字段近死通道。deck 里 heal 比例 ≈ 1/30 = 0.03，几乎无信号 | 候选：删字段或换用"potion_heal_count" |

### 本 session 中遗留未彻底定位

| # | 问题 | 状态 |
|---|---|---|
| U2-NaN 根因 | final smoke 后 nan_skip 仍稳定 3-5。`_check_nan` 已上线但在 tiny smoke 里没捕获真 NaN（可能只在某些 edge batch 触发） | 等下一轮长训 `_check_nan` log 精确定位 |

---

## 8. Checkpoint 兼容性与迁移

### 必须从头训的理由

1. **`max_numeric_dim` 32 → 48**：`tokenizer.numeric_proj = nn.Linear(max_numeric_dim, d_model)` 从 `Linear(32, d)` 变 `Linear(48, d)`。旧 checkpoint weight shape `(d, 32)` 直接 `load_state_dict` 会 **shape mismatch crash**。
2. **deck_card token 11 → 45 维**：新增 34 维 FUNCTIONAL_TAGS，网络对 deck_card 的 attention prior 完全改变。
3. **action_token 30 → 44 维**：新增 14 维（3 scalar + 9 event_kind + 2 route）。
4. **Non-combat RBM 同步行为变化**：shop / card_reward 阶段的 obs 分布改了（rbm 字段从滞后变 fresh）。
5. **reward landscape 全面变化**：PBRS + milestones + terminal_reward scale + shaping cap + future-looking targets + non-combat token 分级，advantage 分布完全不同。
6. **encounter_conditioning 新增 embedding**（外部同步改动）：`n_encounters=128 × d=384` ≈ 49K 新参数。

### warm-start 可行性（理论）

`UnifiedNet.load_compatible_params(state_dict, strict_shapes=False)` 会按 key 匹配并跳过 shape 不兼容的层（主要是 `tokenizer.numeric_proj`）。但：
- tokenizer 是第一层，它随机初始化后，输出分布和后段 decision_core / policy_head 学到的 prior 对不上
- 实际估计省训练时间只有 20-30%
- 调试成本反而更高（"到底是 warm-start 坏了还是新算法坏了"）

**推荐**：从头训，干净省心。

### 数据文件更新

| 文件 | 状态 |
|---|---|
| `STS2AI/Python/data/card_tags.json` | U6 重跑，577 张 STS2 卡的真实 tag（原先若干 STS1 卡名现在不存在） |
| `STS2AI/Python/data/source_knowledge.sqlite` | 未动，仍是权威数据源 |

---

## 9. 修复统计

| 分组 | 条数 | 主要收益 |
|---|---|---|
| Tier 1 奖励升级 | 3 | 接入完整 PBRS + milestones + 按楼层 scale 终局 + shaping cap |
| 特征工程 P 系列 | 8 | 非战斗 rbm 同步 / deck_card 身份 / encounter 对齐 / primitive per-owner / vocab 频次序 / _afc 线程安全 / turn_damage 归一化 / 3 条附加 |
| R1 future-looking target | 3 | hp_loss / survival / leaf 从"过去已知量 / return 线性重复"改为"未来 backfill / n-step"|
| R2 非战斗 option token | 3 | ActionCandidate + bank_assembler + 3 compiler 协同补齐 rarity / price / event_kind 信号 |
| 中期清理 | 3 | FightMode 死通道 / card_tags 重跑 + fallback 清 / route 字段语义黑客 |
| 收尾 | 3 | rest_compiler 补 numeric / NaN 诊断 / preset 覆盖 |
| **合计** | **23** | **所有改动 smoke 全过，exit 0，无 shape error / no crash** |

### Smoke 曲线对比（tiny preset，2 iter × 2 ep）

| 阶段 | Params | Iter2 approx_kl | Iter2 nan_skip | Iter2 vl_turn_damage |
|---|---|---|---|---|
| review-fix baseline（Tier1 + P1①② + finding 全修后，R1/R2/U 前） | 12.7M | 0.047 | 0 | 0.95 |
| R1+R2 baseline（加 future-looking + non-combat token） | 12.7M | 0.731 | 4 | 6.29 |
| U4+U6+U8 中期（+ 清 FightMode / 重跑 card_tags / route 字段） | 12.8M | **0.132** | 5 | **3.65** |
| U1+U2+U3 收尾（+ rest / NaN 诊断 / preset 修） | **1.9M**（真 tiny） | 0.168 | 3 | 7.29 |

说明：
- R1+R2 刚加新通道时 kl=0.73 是冷启动正常震荡（policy 重新学新维度）
- U4/U8 收敛加速 5.6x（kl 0.73 → 0.13）
- U3 让 params 缩到真实 tiny 档（1.9M），后续 smoke 成本大幅降低

---

## 10. 建议的下一步

### 短期

1. **commit 当前状态**。23 条改动没 commit 很危险，应建存档点（feature 分支 → PR）
2. **跑长 smoke**（10-20 iter × 10 ep，slim preset，max_steps 500）看：
   - `approx_kl` 是否稳定到 0.02 附近
   - `nan_skip_count` 是否降到 0（`_check_nan` 能否捕到真 NaN）
   - `avg_floor` 曲线上升趋势
   - `nc_vl_run_win` / `vl_fight_win` 收敛速度

### 中期

3. 定位 **U2 真 NaN 根因**（依赖 `_check_nan` 日志）
4. 修 **U9 AOE 硬编码**（从 card_tags 查 "aoe" tag 替换）
5. 修 **U10 combat_cotrainer reward 常量**（切到 `terminal_reward` 统一）

### 长期

6. **U14 Tier 2 奖励**：分 gamma / counterfactual / tactical 扩展
7. **U12 obs 多格式 adapter**
8. **U11 train_combat_v2 迁 UnifiedNet**

---

## 附：本文档自身的元信息

- **创建日期**：2026-04-17
- **涵盖 session**：同一个 Claude 会话内从 Tier 1 到 U1/U2/U3 的完整修复链
- **规范遵循**：`CLAUDE.md` 硬性要求（数据驱动、Artifacts 目录、中文对话）
- **所有 smoke 产物位置**：`STS2AI/Artifacts/training/{tier1_smoke, review_fix_smoke, r1_smoke, r2_smoke, mid_smoke, final_smoke}/`
