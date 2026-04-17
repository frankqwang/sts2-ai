# 特征工程 / Schema 命名规范

> **硬性规定**：所有涉及游戏资产（卡牌、遗物、药水、敌人 power、state 字段）
> 的 schema、枚举、命名映射，**必须**来自 `data/source_knowledge.sqlite`
> 的真实游戏数据，**不得**由开发者根据记忆或 STS1 经验手写。

---

## 1. 真相之源

## 数据源优先级（高 → 低）

```
1. Sim runtime API (combat_catalog, get_state, ...)    ← 最权威，实时跟随游戏代码
2. data/source_knowledge.sqlite                        ← 从 C# ModelDb 提取的 snapshot
3. 手写列表/枚举                                        ← 禁止（除非文档说明为什么必要）
```

**Python 统一入口**：`networkV2/s1_schema/sim_catalog.py::GAME_CATALOG`
- 启动时可 `attach_sim(client)` → 优先用 sim API
- 否则 fallback sqlite
- 业务代码**只应查这个单例**，不直接接 sqlite 或 hard-list

**Sim API 接入点（当前）**：
- `combat_catalog()` — encounter_id + room_type 列表
- **TODO**（给 sim 侧另一 AI）：加 `game_catalog()` 返回：
  - 所有 cards（id, type, cost, commands, keywords）
  - 所有 relics（id, tags, powers）
  - 所有 monsters（id, powers, hp）
  - 所有 powers（class_name, trigger_events）

---

**唯一权威数据源**：`data/source_knowledge.sqlite`

```
cards      — 577 张卡（id, class_name, powers_json, tags_json, keywords_json, ...）
monsters   — 121 个怪（id, powers_json, intents_json, moves_json, ...）
relics     — 290 个遗物（id, powers_json, commands_json, ...）
potions    — 64 个药水（id, powers_json, commands_json, ...）
encounters — 88 个战斗（id, room_type, monster_ids, ...）
```

**辅助**：
- `core/source_knowledge_features.py::build_knowledge_tables()` —— 构建全局 symbol vocab
- `core/symbolic_features_head.py::SymbolicFeaturesHead` —— 基于此做 cross-attention

数据随 STS2 Early Access 版本更新，**由 `tools/python/data/build_source_database.py` 重建**。

---

## 2. 禁止的做法

❌ **根据 STS1 经验硬编码 power 名**：

```python
# BAD - 完全不符合 STS2 实际命名
_g("metallicize", 0)     # STS1 叫这个, STS2 叫 "plating"
_g("barricade", 0)       # STS2 可能改名
_g("thorns", 0)          # STS2 可能改名
```

❌ **根据经验硬写 encounter 列表**：

```python
# BAD - 人工分级，不可能维护 81 个 encounter
CURRICULUM = {
    "easy": ["CULTIST", "JAW_WORM", ...],  # STS1 名字在 STS2 根本找不到
}
```

❌ **从记忆抄特征清单**：

```python
# BAD - 漏很多，错很多
encoded_powers = ["strength", "vulnerable", "weak", "poison", "artifact"]  # 5 个
# 实际 monster 使用 63 种 power，漏了 58 种
```

---

## 3. 正确的做法

✅ **从 sqlite 提取真实名字 + 频次**：

```python
import sqlite3, json
conn = sqlite3.connect("data/source_knowledge.sqlite")
powers = Counter()
for (pj,) in conn.execute("SELECT powers_json FROM monsters WHERE powers_json IS NOT NULL"):
    for p in json.loads(pj):
        powers[p] += 1
# 得到真实 63 种 power class name + 出现频次
```

✅ **用全局 symbol vocab 做 embedding**：

```python
from core.source_knowledge_features import build_knowledge_tables
tables = build_knowledge_tables()
# tables.meta.global_symbol_vocab - 所有 symbol 的有序 vocab
# tables.monster_symbol_ids - 每个 monster 引用的 symbol ID list
```

✅ **Curriculum 从 catalog 动态派生**：

```python
# 1. 从 sim catalog 取真 encounter_id + room_type
catalog = sim_client.combat_catalog()
# 2. 从 sqlite 取每个 encounter 的 monster 组合、总 hp、power 清单
# 3. 按真实数据算难度
# 4. 按观测胜率动态 curriculum
```

✅ **encounter 难度信号**：

```python
def encounter_difficulty(encounter_id: str, cursor) -> dict:
    """从 sqlite 派生 encounter 难度指纹。"""
    cursor.execute("SELECT monster_ids FROM encounters WHERE id=?", (encounter_id,))
    monster_ids = json.loads(cursor.fetchone()[0])
    total_hp = 0
    has_plating = False
    has_scaling = False
    for mid in monster_ids:
        cursor.execute("SELECT powers_json FROM monsters WHERE id=?", (mid,))
        powers = json.loads(cursor.fetchone()[0] or "[]")
        has_plating |= "PlatingPower" in powers
        has_scaling |= any("Ritual" in p or "Scaling" in p for p in powers)
        # hp 从 min_initial_hp_expr 解析（是表达式）
    return {"total_hp": total_hp, "has_plating": has_plating, ...}
```

---

## 4. 规范要点

### 4.1 新特征 checklist

加新特征前，必须回答以下问题：

- [ ] 涉及的游戏名字是从 sqlite 哪张表哪一列取的？
- [ ] 如果游戏数据更新了，这个特征会自动跟上吗？
- [ ] 涵盖了该字段的所有可能值，还是只挑了几个？
- [ ] 有单测覆盖 "新 encounter/power/card 自动被处理"？

### 4.2 更新时机

- STS2 EA 发版后：运行 `tools/python/data/build_source_database.py` 重建 sqlite
- `core/source_knowledge_features.py` 会检测 SHA1 漂移并告警
- 重跑 training 前须确认 vocab 覆盖了所有将见到的 encounter

### 4.3 自动化检查

每个使用到的 symbol（power name / tag / keyword）**必须**能从 sqlite vocab 查到：

```python
def assert_symbol_exists(name: str, vocab: KnowledgeMeta):
    assert name in vocab.global_symbol_vocab, \
        f"Symbol '{name}' not in sqlite vocab — 可能是 STS1 名字或拼写错误"
```

---

## 4.5 启发式特征不硬编码游戏名

**禁止**：写"基于 power X 存在与否"的启发式，尤其当 X 是 hand-picked 名字：

```python
# BAD
_DISCARD_TRIGGER_POWERS = {"tactician", "reflex", "evolve"}   # 凭记忆挑的
discard_synergy = any(p in _DISCARD_TRIGGER_POWERS for p in player.powers)
```

**正确**：用通用 card/state 字段（game-data-agnostic）：

```python
# GOOD - 基于 card.commands_json 或 keywords，不查 specific power 名
is_discard_card = any("discard" in kw.lower() for kw in card.keywords)
```

**更正确**：把 power 本身当特征暴露，让 attention 自己学 synergy：

```python
# BEST - 通过 player_token 的 PLAYER_POWER_VOCAB 暴露所有 power，
# 网络 attention 自学 "有 Tactician + is_discard_card → discard 价值高"
player_token.numeric.extend([player.powers.get(cls, 0) for cls in PLAYER_POWER_VOCAB])
```

---

## 5. 历史坑（别再犯）

### 2026-04-17: Metallicize vs PlatingPower

- **症状**：FROG_KNIGHT 0% 胜率 13 iter，agent 不学习
- **根因**：`enemy_core_token` 硬编码 `g("metallicize")`，但 STS2 raw 是 `PlatingPower`
- **修复**：改用从 sqlite 提取的真实 power class name
- **教训**：所有 power 名必须来自 sqlite

### 2026-04-17: Hardcoded encounter curriculum

- **症状**：Curriculum 标 FROG_KNIGHT 为 "easy"，但它有 15 层 plating，starter deck 打不穿
- **根因**：开发者按"普通怪"直觉分级，没查真实 power
- **修复**：改用数据派生难度 + 动态胜率 curriculum
- **教训**：不要人工做业务分级，用数据驱动

### 2026-04-17: `_DISCARD_TRIGGER_POWERS` 硬编码 Silent 角色 power

- **症状**：`bank_assembler._compute_combo_signals` 硬编码 `{"tactician","reflex","evolve"}` 检测弃牌 synergy
- **根因**：开发者凭 STS1 Silent 角色印象写死，STS2 里这些 power 罕见（1 张卡）；完全不泛化
- **修复**：删除硬编码列表 → 用 card.keywords 检测 discard 属性；synergy 让 network 从 PLAYER_POWER_VOCAB 学
- **教训**：启发式不要依赖具体游戏名；要么 "data-driven vocab + 网络自学"，要么 "通用字段"

### 2026-04-17: mechanism_registry encounter_id 不匹配 STS2

- **症状**：`act1/bosses.py` 注册 `"haunted_ship"/"doormaker"/...`，真实 DB 是 `doormaker_boss`
- **根因**：参考 STS1 encounter 名字，STS2 加了 `_boss/_elite/_normal` 后缀
- **修复**：encounter_id 必须从 sqlite `encounters.id` 列直接取，不手写
- **教训**：所有 ID 列表（encounter/card/relic/power/event）都查真实 DB，绝不手写

---

## 6. 迁移清单（待做）

当前代码里还有的硬编码（需要逐步清理）：

| 位置 | 硬编码内容 | 清理方式 |
|------|----------|---------|
| `bank_assembler.py::_enemy_core_token` | 5-10 个 power 名查找 | 改用 sqlite vocab 派生 |
| `bank_assembler.py::_player_token` | 6-14 个 power 名查找 | 同上 |
| `combat_cotrainer.py::CURRICULUM` | 硬编码 encounter 分级 | 数据驱动难度 + 动态 curriculum |
| `combat_cotrainer.py::buffed_ironclad_deck` | 硬编码 deck 配置 | 从真实 run 采样 mid-game deck |
| `auto_modifier_rules.py` | STS1 power 名（`metallicize`/`thorns`） | 映射到 STS2 名或直接用 sqlite symbol |
| `action_semantics.py` | `skip_card_reward` 等字符串列表 | 从 sim 真实 action_type 枚举 |
