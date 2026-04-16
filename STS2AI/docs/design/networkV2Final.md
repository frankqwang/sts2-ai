# STS2 CombatNet V2 - Final Design

> 综合 feature.md / another-feature-engineering.md / networkV2.md 三份设计文档，
> 经讨论后确定的最终实施方案。

---

## 1. 一句话定义

把 STS2 战斗从"状态拼接问题"提升成"规则世界编译问题"：

```
Game State → Feature Compiler → 7 Token Banks → Network → Policy/Value/LeafEval
```

不兼容旧架构，全新实现。代码位于 `STS2AI/Python/networkV2/`。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                  Game Simulator / Bridge                  │
└───────────────────────────┬─────────────────────────────┘
                            │ raw obs + legal_actions
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  CombatEnvWrapper                         │
│  维护跨步状态：combat_memory, run_memory,                 │
│  turn_prefix_history                                      │
│  新战斗/新局时 reset                                       │
└───────────────────────────┬─────────────────────────────┘
                            │ obs + combat_memory + run_memory
                            ▼
┌─────────────────────────────────────────────────────────┐
│            CombatFeatureCompiler (纯函数)                 │
│  输入：当前帧 obs + 跨步状态 + mechanism_config            │
│  输出：9 类 canonical schema 对象 → 7 组 token bank       │
└───────────────────────────┬─────────────────────────────┘
                            │ 7 token banks
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  CombatNetV2 (神经网络)                    │
│                                                           │
│  Layer 1: Tokenizer & Bank Projection                     │
│  Layer 2: Memory Encoders (6 个独立编码器)                 │
│  Layer 3: Action Contextualizer (6 段 cross-attention)    │
│  Layer 4: Decision Core (decision_token + self-attn)      │
│  Layer 5: Heads (Policy + Value×4 + LeafEvaluator)       │
└─────────────────────────────────────────────────────────┘
```

### 角色分工

| 组件 | 职责 | 有无内部状态 |
|------|------|------------|
| CombatEnvWrapper | 管理跨步累积状态（CombatMemory、RunMemory、TurnPrefix） | 有 |
| CombatFeatureCompiler | 将原始数据 + 累积状态编译为结构化 token bank | 无（纯函数） |
| CombatNetV2 | 吃 token bank，输出 policy/value/leaf score | 无（纯前向推理） |

---

## 3. Canonical Schema（9 类对象）

### 3.1 EntitySemantics - 实体本体

表示"这个东西是什么"，不含运行时状态。

| 对象 | 字段 |
|------|------|
| card | entity_id, card_type, rarity, base_cost, tags, keywords |
| relic | entity_id, relic_tags, functional_signals |
| potion | entity_id, potion_type, tags |
| enemy | entity_id, is_boss, is_elite, is_minion |

来源：vocab + source_knowledge.sqlite

### 3.2 RuntimeInstances - 运行时实例状态

表示"当前实例状态是什么"。

| 对象 | 字段 |
|------|------|
| player | hp, max_hp, block, energy, max_energy, powers: dict[str, int] |
| hand_card | card_id, current_cost, damage_est, block_est, retain, ethereal, buffed |
| enemy | hp, max_hp, block, intent(type, damage, hits), powers: dict[str, int] |
| pile | size, attack_ratio, skill_ratio, zero_cost_density, key_card_remaining, reshuffle_proximity |

来源：bridge 运行时状态 + action preview

### 3.3 MechanismStates - 机制状态

表示"战斗进程走到哪了"。

5 种 primitive（第一版）：

| Primitive | 含义 | 示例 |
|-----------|------|------|
| phase_transition | 阶段切换 | boss HP<50% 进入 phase 2 |
| window | 时间窗口 | 护盾破碎后 2 回合易伤 |
| summon_cycle | 召唤循环 | 每 3 回合召唤一个 add |
| threshold_gate | 阈值门控 | 累计 X 伤害触发效果 |
| shield_progress | 护盾进度 | 多层护盾逐层击破 |

来源：mechanism_config（手工配置）+ 运行时状态检测

### 3.4 RuleModifiers - 规则改写

表示"当前哪些规则变了"。

8 种 primitive（第一版）：

| Primitive | 含义 | 示例 |
|-----------|------|------|
| damage_cap | 伤害上限 | 每次只受 1 点伤害 |
| target_restriction | 目标限制 | adds 存活时不可选中本体 |
| effect_scaling | 效果缩放 | 按格挡值造成伤害 |
| on_play_trigger | 出牌触发 | 打 skill 时 boss 加力量 |
| on_hit_trigger | 受击触发 | 荆棘、弹甲 |
| draw_modifier | 抽牌修改 | 抽牌被减少/增加 |
| exhaust_modifier | 消耗修改 | 所有卡打出后被消耗 |
| phase_transition_effect | 阶段切换效果 | 切换时清除所有 debuff |

来源：
- Level 2 power（thorns/split/angry...）→ AUTO_MODIFIER_RULES 自动映射
- Level 3 boss 复杂机制 → mechanism_config 手工配置

### 3.5 TurnPrefixMemory - 本回合历史

| 字段 | 说明 |
|------|------|
| played_actions | 已执行动作序列（card_id, target, effects） |
| energy_spent | 本回合已花费能量 |
| cards_played / attacks / skills / powers | 各类型出牌计数 |
| damage_dealt / block_gained / cards_drawn | 效果累计 |
| active_chains | 当前已建立的连段信息 |

来源：CombatEnvWrapper 每步追踪

### 3.6 CombatMemory - 战斗长程摘要

| 字段 | 说明 |
|------|------|
| turn_index | 当前回合数 |
| cumulative_hp_loss | 总掉血 |
| recent_hp_loss_window | 近几回合掉血趋势 |
| potions_used | 已用药水数 |
| phase_history | 经历的 phase 序列 |
| reshuffle_count | 洗牌次数 |
| exhaust_total | 总消耗卡数 |
| fight_mode | 当前战斗模式（race/stabilize/attrition/burst_prep） |

来源：CombatEnvWrapper 每步累积

### 3.7 RunBuildMemory - 整局慢变量

| 字段 | 说明 |
|------|------|
| build_identity | 构筑风格 |
| deck_stats | frontload/block/draw/scaling/aoe/heal |
| curse_density, high_cost_density, zero_cost_density | 牌库密度 |
| consistency | 构筑一致性 |
| objective_context | survival_priority, resource_priority, preserve_hp_bias... |

来源：复用现有 run_memory + build_profile + objective_context 逻辑

### 3.8 RawActionCandidates - 合法动作集

| 字段 | 说明 |
|------|------|
| action_type | play_card / end_turn / drink_potion / ... |
| source_card | 源卡牌信息 |
| target | 目标 enemy/card |
| preview_values | damage_est, block_est, draw_delta, energy_delta |
| semantic_signature | family, target_scope, roles |

来源：bridge legal_actions + action preview

### 3.9 ActionHypotheses - 动作上下文语义

**不由 Compiler 输出。** 这是网络内部 Action Contextualizer 的产物。

Action Contextualizer 通过 6 段 cross-attention，让每个 action token
依次读取 board/modifier/mechanism/prefix/combat_memory/build，
最终生成的隐层向量即为 ActionHypothesis，隐含：

- 当前规则下的真实效果
- modifier 对动作的影响
- 机制阶段相关的风险/收益
- 与本回合已打序列的配合
- 与构筑长期策略的匹配

---

## 4. Buff 三级分层体系

### Level 1: 通用 Power（数值 buff/debuff）

```
strength, dexterity, weak, vulnerable, frail, poison, regen,
metallicize, artifact, thorns, plated_armor, intangible,
vigor, barricade, rage, ritual, buffer, entangled...
```

- 存储位置：RuntimeInstances.player.powers / RuntimeInstances.enemy.powers
- 处理方式：直接作为数值字段，不产生 mechanism/modifier token
- 适用对象：所有怪物、玩家

### Level 2: 特殊行为 Power（自动映射 RuleModifier）

```
AngryPower → on_play_trigger(打skill加strength)
ThornsPower → on_hit_trigger(反伤)
SplitPower → threshold_gate(分裂)
CurlUpPower → on_hit_trigger(首次受击加block)
SporeCloudPower → on_death_trigger(死亡给vulnerable)
...
```

- 存储位置：AUTO_MODIFIER_RULES 自动映射表
- 处理方式：运行时从 enemy.powers 自动检测，生成 RuleModifier token
- 不需要手工 config
- 适用对象：所有有特殊 power 的怪物

### Level 3: Boss 复杂机制（手工 MechanismConfig）

```
phase_transition, window, summon_cycle, threshold_gate, shield_progress
```

- 存储位置：mechanism_config registry（Python dataclass + 检测函数）
- 处理方式：手工为每个 boss/elite 配置 primitive 组合
- 运行时根据 config + 可观测状态推断当前 mechanism state
- 先覆盖 Act 1 elite + boss

---

## 5. Mechanism Config 设计

### 格式：Python Registry

不用 YAML，因为检测规则需要条件逻辑。

```python
@dataclass
class PhaseTransition:
    phase_id: str
    trigger: Callable[[EnemyRuntime], bool]

@dataclass
class DamageCap:
    cap_value: int
    scope: str  # "per_hit" / "per_turn"
    active_when: Callable[[EnemyRuntime], bool]

@dataclass
class EncounterMechanismConfig:
    encounter_id: str
    phases: list[PhaseTransition]
    damage_caps: list[DamageCap]
    summon_cycles: list[SummonCycle]
    windows: list[Window]
    target_restrictions: list[TargetRestriction]

# 每个 boss 是多个 primitive 的组合
register(EncounterMechanismConfig(
    encounter_id="SomeBoss",
    phases=[
        PhaseTransition("phase_1", lambda e: e.hp_ratio > 0.5),
        PhaseTransition("phase_2", lambda e: e.hp_ratio <= 0.5),
    ],
    damage_caps=[
        DamageCap(1, "per_hit", lambda e: e.has_buff("HardenedShell")),
    ],
))
```

### Primitive 统一字段

每个 primitive 必须包含：
- type: primitive 类型
- scope: 作用范围
- activation: 激活条件
- termination: 终止条件
- params: 参数
- runtime_outputs: 运行时输出字段
- source_kind: 数据来源（config / auto / inferred）

---

## 6. Token Bank 输出规范

Compiler 最终输出 7 组 token bank：

| Bank | 内容来源 | Token 类型 |
|------|---------|-----------|
| build_bank | RunBuildMemory + EntitySemantics(deck/relic/potion) | deck_card, relic, potion, build_profile, objective |
| board_bank | RuntimeInstances(player/hand/enemy/pile) | player, hand_card, enemy_core, enemy_intent, pile_summary |
| mechanism_bank | MechanismStates | phase, window, summon, threshold, shield |
| modifier_bank | RuleModifiers | damage_cap, target_restrict, scaling, trigger... |
| turn_prefix_bank | TurnPrefixMemory | played_action (ordered sequence) |
| combat_memory_bank | CombatMemory | combat_summary, phase_history, fight_mode |
| action_bank | RawActionCandidates | action candidate (one per legal action) |

每个 token 的结构：
```python
@dataclass
class Token:
    numeric: list[float]    # 数值特征
    token_type: str         # 类型标识
    owner_id: str           # 归属实体
    metadata: dict          # 额外元信息
```

---

## 7. 网络架构详细设计

### Layer 1: Tokenizer & Bank Projection

把 Compiler 输出的 Python 对象转成 tensor，每个 token 加上：
- type_embedding
- role_embedding
- owner_embedding
- time_scale_embedding (slow/medium/fast)

统一投影到 d_model 维。

### Layer 2: Memory Encoders

6 个独立编码器，各 bank 先内部整理：

| Encoder | 结构 | 输出 |
|---------|------|------|
| BuildMemoryEncoder | 8-12 learnable latent slots + 2-3 层 slot-to-build cross-attn | build_memory_slots |
| BoardEncoder | 3-4 层 self-attention + relation type bias | board_tokens |
| MechanismEncoder | 2 层 self-attention + owner bias | mechanism_tokens |
| ModifierEncoder | 2 层 self-attention + scope bias | modifier_tokens |
| TurnPrefixEncoder | 2-3 层 causal transformer | prefix_tokens |
| CombatMemoryEncoder | 1-2 层 transformer 或 gated summary | combat_memory_tokens |

### Layer 3: Action Contextualizer

核心模块。6 段 cross-attention：

```
action_tokens →(cross-attn)→ board_tokens        # 知道当前局面
action_tokens →(cross-attn)→ modifier_tokens      # 知道规则怎么改了
action_tokens →(cross-attn)→ mechanism_tokens     # 知道机制阶段
  前 3 段可并行，结果 merge
action_tokens →(cross-attn)→ prefix_tokens        # 知道本回合前序
action_tokens →(cross-attn)→ combat_memory_tokens # 知道长程态势
action_tokens →(cross-attn)→ build_memory_slots   # 知道构筑策略
  后 3 段串行

输出：action_hypothesis_tokens（每个动作一个上下文化向量）
```

### Layer 4: Decision Core

```
[learnable decision_token] + action_hypothesis_tokens
→ 2-4 层 self-attention
→ decision_token_refined + action_hypotheses_refined
```

- decision_token：全局决策摘要，给 Value/LeafEvaluator
- action_hypotheses_refined：每个动作的最终表示，给 Policy

### Layer 5: Heads

**Policy Head**:
- 输入：decision_token_refined + action_hypotheses_refined
- 输出：每个动作的 logit + action_mask
- 结构：bilinear scorer (decision × action → score)

**Value Heads (4 个)**:
1. fight_win_value: 这场战斗的赢面 (sigmoid)
2. expected_hp_loss: 期望掉血 (softplus)
3. survival_2turn: 近 2 回合生存概率 (sigmoid)
4. tempo_value: 节奏/阶段优势 (tanh)

**Leaf Evaluator (独立模块)**:
- 输入：decision_token + combat_memory_tokens + mechanism_summary + resource_summary
- 输出：leaf_score, transition_risk, survival_margin, resource_retention_value
- 用途：搜索时评估中间叶子节点

---

## 8. 参数规模

### 主力版

- d_model = 512
- n_heads = 8
- total attention blocks ≈ 12-16
- FFN dim = 2048
- 参数量：60M - 90M

### 轻量验证版

- d_model = 384
- blocks 略缩
- 参数量：35M - 55M

先用轻量版验证信息流，再上主力版。

---

## 9. 文件结构

目录按数据流向编号 (s1→s6)，方便阅读分层。

```
STS2AI/Python/networkV2/
├── __init__.py
│
├── s1_schema/                      # ① 数据定义层
│   ├── primitives.py               # Mechanism/Modifier primitive 基类和类型
│   ├── entities.py                 # EntitySemantics, RuntimeInstances
│   ├── memory.py                   # TurnPrefixMemory, CombatMemory, RunBuildMemory
│   ├── actions.py                  # ActionCandidate
│   └── token_banks.py              # Token, TokenBank, UnifiedTokenBanks
│
├── s2_config/                      # ② 机制配置层
│   ├── mechanism_registry.py       # 全局注册表
│   ├── auto_modifier_rules.py      # Level 2: power → modifier 自动映射
│   └── act1/                       # Act 1 boss/elite 配置
│       ├── elites.py
│       └── bosses.py
│
├── s3_state_tracker/               # ③ 状态追踪层（跨步累积状态管理）
│   └── combat_env_wrapper.py       # CombatStateTracker
│
├── s4_compiler/                    # ④ 特征编译层（纯函数）
│   ├── feature_compiler.py         # CombatFeatureCompiler 主入口
│   ├── runtime_compiler.py         # RuntimeInstances 编译
│   ├── mechanism_compiler.py       # MechanismStates 编译
│   ├── modifier_compiler.py        # RuleModifiers 编译
│   ├── memory_compiler.py          # Memory → 数值向量
│   ├── action_compiler.py          # ActionCandidates 编译
│   └── bank_assembler.py           # Schema → TokenBanks 组装
│
├── s5_net/                         # ⑤ 网络层
│   ├── combat_net.py               # CombatNetV2 主入口
│   ├── tokenizer.py                # Bank → Tensor (单样本 + batched)
│   ├── action_contextualizer.py    # Layer 3: 6 段 cross-attention
│   ├── decision_core.py            # Layer 4: decision token reasoning
│   ├── encoders/                   # Layer 2: Memory Encoders
│   │   ├── common.py               # TransformerBlock, CrossAttention, SlotAttention
│   │   ├── build_encoder.py        # SlotAttention 慢变量编码
│   │   ├── board_encoder.py        # Self-attention 战场编码
│   │   ├── mechanism_encoder.py
│   │   ├── modifier_encoder.py
│   │   ├── prefix_encoder.py       # Causal transformer
│   │   └── combat_memory_encoder.py
│   └── heads/                      # Layer 5: Output Heads
│       ├── policy_head.py          # Bilinear policy scorer
│       ├── value_heads.py          # 4 value heads
│       └── leaf_evaluator.py       # 搜索用叶子评估器
│
└── s6_training/                    # ⑥ 训练层
    ├── batch.py                    # TrainingSample, collation, batching
    ├── losses.py                   # PPO policy + value + leaf losses
    └── ppo.py                      # CombatPPOTrainerV2
```

---

## 10. 施工顺序

### Phase 1: Schema + Config

- 所有 dataclass 定义
- Primitive 类型系统
- Mechanism registry + auto modifier rules
- Act 1 boss/elite config

### Phase 2: Compiler + Env Wrapper

- CombatFeatureCompiler 主入口
- 各子编译器
- CombatEnvWrapper（跨步状态管理）
- 能从 raw obs 输出 7 组 token bank

### Phase 3: Network (Tokenizer + Encoders)

- Bank → Tensor 转换
- 6 个 Memory Encoder

### Phase 4: Network (Action Contextualizer + Decision Core)

- 6 段 cross-attention
- Decision Core

### Phase 5: Network (Heads)

- Policy Head
- Value Heads × 4
- Leaf Evaluator

### Phase 6: Training

- 接训练流程
- 非战斗决策（后续加入）

---

## 11. 验收标准

### 必须达到

1. 新机制接入时，优先新增 primitive/config，不加 boss 专属分支
2. 每个 feature 能回答：属于哪层、时间尺度是什么、服务什么决策问题
3. TurnPrefixMemory 与 CombatMemory 明确分开
4. MechanismStates 与 RuleModifiers 明确分开
5. ActionHypothesis 由网络 Action Contextualizer 产生，不由 Compiler 手工计算
6. Leaf Evaluator 独立于普通 Value Head
7. Compiler 是纯函数，无内部状态

### 明确不做

- 更大的 state vector 拼接
- 把机制塞进 raw buff list
- 只靠 enemy id 或文本描述表示机制
- ActionHypotheses 的手工规则编译
- 兼容旧架构

---

## 12. 非战斗架构（预留设计）

### 原则

战斗比较 ActionHypothesis，非战斗比较 OptionHypothesis。
两者共享世界观（SharedWorldBanks），但不共享决策语义。

### 统一 Token Bank 结构

```
SharedWorldBanks (6 组，始终编译):
  build_bank, inventory_bank, economy_bank,
  route_bank, objective_bank, forecast_bank

CombatBanks (5 组，仅战斗时):
  board_bank, mechanism_bank, modifier_bank,
  turn_prefix_bank, combat_memory_bank

action_bank (始终有):
  combat 时 = combat action candidates
  non-combat 时 = option candidates (按 domain 分类)
```

### 非战斗 Domain 分支

| Domain | 选项类型 | Option Contextualizer 读取顺序 |
|--------|---------|-------------------------------|
| card_reward | 选牌/三选一/发现 | build → inventory → forecast → objective |
| shop | 购买/remove/skip | build → economy → inventory → forecast → objective |
| route | 路线节点选择 | route → forecast → objective → economy |
| rest | heal/upgrade/remove | objective → build → forecast → economy |
| event | 事件选项 | objective → build → economy → forecast |

### 网络结构

```
SharedWorldBanks → Shared World Encoder (复用 build_encoder 等)
                      ↓
            ┌─── combat domain ──→ Action Contextualizer → Decision Core → Policy/Value/Leaf
            │
domain_router
            │
            └─── non-combat domain ──→ Option Contextualizer → Option Reasoner → Option Policy + Run Evaluator
```

### 后续实现目录

```
compiler/noncombat/
  ├── card_reward_compiler.py
  ├── shop_compiler.py
  ├── route_compiler.py
  ├── rest_compiler.py
  └── event_compiler.py

net/
  ├── option_contextualizer.py
  └── heads/run_evaluator.py
```
