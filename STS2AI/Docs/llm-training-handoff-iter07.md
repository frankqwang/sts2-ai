# LLM 训练交接 — iter07 系列（v1-v6） / 准备 v7

更新时间：2026-04-30 11:30 Asia/Shanghai
作者：本轮 Claude（用户反复 review 后整理的现状）

## D 路线（v6 设计）核心约束

> **reason 只能由模型输出，禁止任何代码 / canonical 模板 / teacher 改写。**

落地策略 = "**combat 不输出 reason，planner 输出 reason，先用 deepseek 蒸馏 planner，后续切 GRPO**"。

| LoRA | inference 输出 | SFT label 来源 |
|---|---|---|
| **planner** | `{battle_objective, enemy_focus, ..., phase_plan}` | deepseek review 的 corrected planner_hint + phase_plan_zh （蒸馏）|
| **combat** | `{action_index, confidence}` (**无 reason**) | 模型自己 rollout 的 action_index, 不带 reason |

deepseek review 的 reason 字段（reason_en / corrected_reason）被严格限制到 **planner SFT pipeline + 诊断 meta**, 永远不进 combat assistant content.

## 一夜的 commit history

| commit | 内容 |
|---|---|
| 6ebbd36 | iter07 系列改造（teacher 重命名、sim 兜底、prompt 优化、boss idiom）|
| ad2d416 | **Phase A**: combat policy 不再输出 reason |
| 29b2f27 | **Phase B**: 删除所有 reason 重写代码 + mask_reason 机制 |
| 6269301 | **Phase C**: planner SFT 增强（overall_score 过滤 + phase_plan 注入）|
| (next)  | **数据清洗**: sanitize 旧 combat train.jsonl 脚本 |

## TL;DR — 4 轮启动失败的连环 bug 修复链 + 待办

| 版本 | 启动结果 | 根因 |
|---|---|---|
| iter07 v1 (PID 683) | 64/64 episodes 全 reset_failed | sim launcher 检 stale Release binary（dotnet build -c Debug 没碰 Release） |
| iter07 v2 (PID 878) | 跑完 44 ep（normal 26V/0D, elite 7V/9D, boss 0V/2D），价值低 | 旧 prompt（重复 + 截断）+ 串行 teacher + max_seq=2048 + tier 没过滤 normal |
| iter07 v3 (PID 248) | 21min 后发现 max_seq 没透传，kill | self_iterate 默认 3072 但 rollout_cmd 漏传 |
| iter07 v4 (PID 1425) | 跑完 52 ep，**deepseek 0 调用** | run_kimi_combat_review_batch 的 provider gating 把 deepseek 误归 no_claude_cli |
| iter07 v5 (PID 1021) | 11 min 后发现 sim card_select bug，kill | state_semantics COMBAT_STATE_TYPES 缺 card_select |
| iter07 v6 | **未启**（待 reason 改写代码下线 + 用户审核）| - |

> 这是 `llm-training-handoff.md` 的增量交接，**只覆盖 iter05c → iter06 期间发生的变更和待办**。前面的总体原则仍然以 `llm-training-handoff.md` 为准。

---

## 一、最新已落地的代码改动（iter05c → iter06）

### 1. teacher provider 抽象化（kimi → deepseek-v4-pro 默认）
- 文件：`STS2AI/llm/scripts/teacher/kimi_review_turn_order.py`
- 变更：
  - `DEFAULT_TEACHER_PROVIDER = "deepseek"`
  - 新增 `DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"`、`DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"`、`DEFAULT_DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"`
  - 新增 `resolve_provider_base_url(provider, override)`、`resolve_provider_api_key_env(provider, override)`
  - `normalize_provider` 接受 `deepseek` / `DEEPSEEK` / `ds` / `deepseek-v4-pro` / `deepseek_v4` 等别名
- 测试：`STS2AI/llm/tests/test_kimi_review_turn_order.py::test_normalize_provider_accepts_deepseek_aliases` / `test_resolve_provider_defaults_for_deepseek` ✅
- 用户态：必须 `export DEEPSEEK_API_KEY=sk-...`，否则会回退 dry_run。

### 2. self_iterate 默认走 deepseek
- 文件：`STS2AI/llm/scripts/automation/self_iterate.py`
- 变更：默认 `--teacher-provider deepseek`；新增 `--isolation-evals`（旧 4-grid 模式）作为 opt-in；新增 `--candidate-rollout-*` 系列参数；`--elite-oversample-ratio` / `--boss-oversample-ratio` 透传。
- 注意：argparse choices 里 `--diverse-sampling` 之前缺 `diverse` 已修复。

### 3. 软引导：移除硬规则
- 文件：`STS2AI/llm/data_pipeline/state_renderer.py`
- 变更：
  - 删掉 `_POWER_STRATEGIC_HINTS` dict + `counter_strategy[]` 渲染（用户明确说不要硬规则）
  - end_turn 行渲染**事实**而非建议：`predicted_incoming=N your_block=M expected_hp_loss=N-M`
  - AOE 总伤害展示、`reduced_from=N(by=SLIPPERY)`、0 dmg 显式标注
  - 关键词 / power / relic tooltip 全部走 `localization_loader.lookup_*`，没有再走自己写的 fallback dict

### 4. 官方文案集成
- 新文件：`STS2AI/llm/data_pipeline/localization_loader.py`
  - 加载 STS2 官方 `localization/eng/*.json`（`card_keywords.json` / `static_hover_tips.json` / `powers.json` / `relics.json`）
  - `lookup_relic` / `lookup_power` / `lookup_keyword`（多源回退）
  - 去 markup、保留 placeholder
- 新文件：`STS2AI/bridge/game_bridge/transport/localization_resolver.py`
  - bridge 端 LocString → 实际描述解析，已接入 `proto_state_converter`

### 5. hold-out eval pool（确定性 hash 切分）
- 文件：`STS2AI/llm/data_pipeline/encounter_pool.py`
- 新增：`_is_hold_out_case` / `pool_role={'full','train','eval'}` / `hold_out_fraction` / `hold_out_seed`
- 用法：rollout 训练用 `pool_role=train`，eval 用 `pool_role=eval` 同 seed → 永远不交叉。

### 6. archetype-aware sampling（**case 级，不是 deck 级**）
- 文件：`STS2AI/llm/data_pipeline/encounter_pool.py`
- 新增 `_ARCHETYPE_FEATURE_CARDS` / `classify_build_archetypes` / `_enforce_archetype_min_counts` / `_parse_archetype_min_count`
- **重要**：用户明确反对强制 deck 选卡，所以这是「**挑选 build 已含特征卡的 case**」，不是「强迫加卡」。默认不开。

### 7. action_quality 修 hp_lost 假阳性
- 文件：`STS2AI/llm/data_pipeline/action_quality.py`
- bug：sim 在玩家死亡时把 enemies 全部清零、又把 player.hp 重置 → 老逻辑用 final_state 算 hp_lost 全是 0，进度全是 1.0
- 修：final_state 只在 `final_hp ≤ step累加 hp_end` 时才采用，否则回退到 last_step_state
- 文件：`STS2AI/llm/training/grpo_rollout.py` 同步修了 `outcome != "victory"` 的 progress 回退

### 8. early_exit_diagnostics
- 文件：`STS2AI/llm/training/grpo_rollout.py`
- `EpisodeRecord` 新增 `early_exit_diagnostics`，捕获 left_combat 的 sim 状态（state_type / battle.state_type / card_selection / terminal / run_outcome）
- 用途：给 LAGAVULIN / HEADBUTT 类 left_combat 异常做现场快照

### 9. mask reason in train data
- 文件：`STS2AI/llm/training/grpo_rollout.py`
- 新参数：`--mask-reason-in-train-data`
- 实现：`_maybe_mask_reason_in_assistant` 把 assistant JSON 里的 `reason` 字段值替换为空串，**不删字段**（保持结构）
- 目的：reason 幻觉问题（model 把 SKILL 卡也写 "Deal X damage"），mask 后让训练只 supervise `action_index`

### 10. eval-only / by_tier
- 文件：`STS2AI/llm/training/grpo_rollout.py`
- `--eval-only` + `build_eval_metrics` + `_classify_tier`（normal/elite/boss 分桶）

### 11. fullrun eval 工具（独立）
- 新文件：`STS2AI/llm/eval/fullrun_eval.py`
- LLM policy 跑 fullrun，combat 用 LLM、非 combat 用 heuristic
- 指标：act1_clear_rate / floor_reached / death_floor_distribution

### 12. teacher dataset 默认用 Kimi/deepseek 的 corrected_reason
- 文件：`STS2AI/llm/scripts/datasets/build_teacher_dataset.py`
- `_rows_from_review` 新增 `use_kimi_reasons=True`（默认 True）
- legacy canonical template 仍可通过 `use_kimi_reasons=False` 启用（test 里走 False）
- 配套：`_coerce_confidence`（拒 bool）、`_coerce_tag_list`（处理 string）

### 13. planner-hint refresh=turn 模式
- 已经实现，**iter06 还没启用**（默认 episode 级 cache，STEAM 出现到刷新有 3 步延迟）
- iter07 必须改成 `--planner-hint-refresh turn`

---

## 二、iter06 训练结果（boss 为啥还是 0/8）

### 关键发现：**planner_hint 有反应，combat policy 不听**

WATERFALL_GIANT 第二个 episode 演化：
1. early：`Establish block and damage early to survive the boss's buff`
2. mid（STEAM 出）：`Secure block and apply early damage to prevent Steam Eruption`
3. late：`Apply Vulnerable and deal damage before Steam Eruption activates`

**planner LoRA 推理链 OK**。问题：
- combat LoRA 在 planner 说 "Secure block" 时仍然选 STRIKE_IRONCLAD / POMMEL_STRIKE（纯输出），不去 DEFEND/SHRUG_IT_OFF
- planner_hint 默认 episode-level cache，STEAM 出现到刷新有 3 步延迟
- canonical reason 模板让 SKILL 卡也被写成 "Deal X damage"（幻觉）

### iter06 数据点
- run 目录：`STS2AI/Artifacts/llm/datasets/combat_skada_clean_rollout_iter06_20260429-1843_*`
- candidate adapter：`*_candidate_rollout`
- WATERFALL_GIANT 出现 hp=999999999（**这是 ABOUT_TO_BLOW_MOVE 不死阶段，机制不是 bug**，已确认）
- LAGAVULIN ep0 出现 8-step left_combat（HEADBUTT + 空 pile，**待确认**是机制还是 sim bug）

---

## 三、待办（按优先级）

### P0：决策 HEADBUTT + 空 pile 异常根治方案

用户要求"根治"。四个选项还没拍：

- **A 重试**：rollout 时检测到 left_combat + HEADBUTT 序列就重新采样这一步
- **B bridge fix**：bridge 端检测空 pile 时不进 hand_select 状态
- **C sim fix**：改 sim 的 hand_select 触发条件
- **D prompt filter**：state_renderer 在 hand_select empty 时跳过这步（episode 截断）

需要先问用户：sim 是否在用户掌控范围内？看下 `STS2AI/Python/sim_*` 还是 `STS2AI/sim`，A/D 可立即做，B/C 看 sim/bridge 仓库结构决定。

### P0：LAGAVULIN ep0 左战 trace 调查

参考脚本：`STS2AI/Artifacts/llm/temp/check_hand_select.py`（已存在）

需要：
1. dump 完整 step trace（state_type 每步 + settlement_events）
2. 对照 STS2 LAGAVULIN 机制（INFESTED/PLATING/ASLEEP）
3. 像之前 WATERFALL_GIANT 999999999 一样定性：**是机制还是 bug**
4. 不要轻易说 "似乎"、"sim bug"

### P1：iter07 启动

参数：

```bash
python STS2AI/llm/scripts/automation/self_iterate.py \
  --teacher-provider deepseek \
  --planner-hint-refresh turn \
  --mask-reason-in-train-data \
  --boss-oversample-ratio 0.20 \
  --elite-oversample-ratio 0.25 \
  --use-kimi-reasons \
  --case-limit 32 \
  --kimi-limit-episodes 8 \
  ...
```

**待决策**：iter07 的 base adapter 选哪个？
- 选 iter02b（保守，已知稳定）
- 选 iter06 candidate（激进，未验证 boss 改善但 normal 改善）
- 用户倾向："主要训练 boss"，建议先看 iter06 candidate 的 normal 是否 regression，没有 regression 就用它做 base

环境变量：

```bash
export DEEPSEEK_API_KEY=sk-...   # 必须，否则 dry_run
```

### P1：combat policy ↔ planner_hint 对齐

deepseek-v4-pro 的 review prompt 要显式要求 corrected_reason 引用 planner_hint，例如：

```
Following planner's "Secure block" guidance with STEAM imminent, play DEFEND for 5 block
```

而不是：

```
Deal 6 damage with STRIKE
```

文件：`STS2AI/llm/scripts/teacher/kimi_review_turn_order.py`，找 review schema / prompt template，加 instruction：
> When `planner_hint.battle_objective` mentions block/defense, the corrected_reason MUST reference the planner directive verbatim and choose a defensive action.

### P2：planner_hint 在 prompt 里的位置

当前 planner_hint 在 system 之后、user 之前。可以试：在 user message 中**临近 legal_actions** 重复一次 `battle_objective`，加强提示。

文件：`STS2AI/llm/data_pipeline/state_renderer.py` 的 user 渲染部分。

---

## 四、关键路径速查

### 代码
- teacher provider 抽象：`STS2AI/llm/scripts/teacher/kimi_review_turn_order.py`
- self_iterate 入口：`STS2AI/llm/scripts/automation/self_iterate.py`
- rollout：`STS2AI/llm/training/grpo_rollout.py`
- state 渲染：`STS2AI/llm/data_pipeline/state_renderer.py`
- localization：`STS2AI/llm/data_pipeline/localization_loader.py` + `STS2AI/bridge/game_bridge/transport/localization_resolver.py`
- encounter pool：`STS2AI/llm/data_pipeline/encounter_pool.py`
- action quality：`STS2AI/llm/data_pipeline/action_quality.py`
- teacher dataset：`STS2AI/llm/scripts/datasets/build_teacher_dataset.py`
- fullrun eval：`STS2AI/llm/eval/fullrun_eval.py`

### 数据 / artifact
- iter06 dataset 根：`STS2AI/Artifacts/llm/datasets/combat_skada_clean_rollout_iter06_20260429-1843_*`
- iter02b（保守 baseline）：`combat_skada_clean_rollout_iter02b_20260429-1254_*`
- iter05c（最近一次 stable）：`combat_skada_clean_rollout_iter05c_20260429-1621_*`

### 测试
- `STS2AI/llm/tests/test_kimi_review_turn_order.py` ✅ 全过
- 当前已知 256 个测试全过

---

## 五、用户的硬约束（已多次重申，不要违背）

1. **不要硬规则 / 写死战术** — 用软引导，让 AI 自己学
2. **不要做临时方案** — 按最佳实践来，不求快
3. **不要随便说 "似乎" / "sim bug"** — 先查机制描述、看 trace 实证
4. **boss 战是重点** — normal 战占比变高 = 跑偏
5. **deepseek-v4-pro 是默认 teacher**，不再用 kimi
6. **archetype 不强制 deck 选卡**，只是 case 选择倾向
7. 中文沟通，文档放 `STS2AI/Docs`，artifact 放 `STS2AI/Artifacts`
8. 引用本地文件用 `/` 前导绝对路径 + 行号
9. 长任务后台跑，给 PID + 日志路径 + 输出目录

---

## 六、当前最后回答的问题（上下文断点）

用户问："AI 看得到机制描述。问题是它不理解 '机制 → 节奏紧迫性 → 杀那回合需要 block' 这个推理链。 这个难道 planner 不会有所反应吗"

我已回答：
- **planner DOES react**：从 "high damage" 改成 "before STEAM_ERUPTION triggers" / "Secure block"
- **combat policy 不听 planner**：planner 说 block，combat 选 STRIKE
- 解决：(1) `planner-hint-refresh=turn` 减少 cache 延迟；(2) deepseek corrected_reason 显式引用 planner；(3) mask reason 防幻觉但不直接对齐
- 反问用户：(a) 先 dump LAGAVULIN ep0 trace 验证机制 vs bug？还是 (b) 直接进 iter07，HEADBUTT 异常先 prompt 层 filter？

**用户还没回答**。下一轮 Claude 接手时应：
1. 先读这份文档
2. 再读 `llm-training-handoff.md`（总原则）
3. 看用户回复的是 (a) 还是 (b)
4. 不要重新分析一遍 STEAM_ERUPTION 问题，结论已经定了

---

## 七、TodoWrite 当前状态（增量）

用户已确认 / 落地：
- ✅ planner-hint-refresh 默认改 turn（代码 6 处 + 文档 4 处 + 21 测试通过）
- ✅ LAGAVULIN ep0 是机制（用户确认）
- ✅ iter07 base = iter06 candidate（用户确认"用新的"）

新发现（需要修）：

```
1. [pending] 抽 DEFAULT_PLANNER_HINT_REFRESH 常量，5 个 CLI argparse 引用，不再各自写默认值
2. [in_progress] Bridge 补 trigger_card_id + source_pile_type + selection_rules 到 CardSelectState
3. [pending] 修 sim 空 pile + min_select=0 不自动 confirm 的 legal_actions 空僵局 (ProtoStateBuilder.cs:461-475)
4. [pending] state_renderer 改 select 状态渲染: 显示 "because of [trigger_card] from [pile_type]: [filter]"
5. [pending] 验证 state_renderer 是否完整显示 ASLEEP / PLATING / STEAM_ERUPTION_POWER 详情
6. [pending] planner_hint 输入加 enemy_phase + upcoming_intent_chain (sim 需送回预测)
7. [pending] planner_hint system_prompt 加 turn-aware instruction
8. [pending] iter07 启动 (deepseek + mask-reason + boss=0.20 elite=0.25 + base=iter06 candidate)
9. [pending] deepseek review prompt 要求引用 planner_hint
```

## 八、Explorer 排查报告关键结论（2026-04-29 19:30）

### A) Select 类卡牌 sim/bridge 排查

**触发卡（IRONCLAD）**：HEADBUTT、ARMAMENTS、DUAL_WIELD、EXHUME、HAVOC、WARCRY、BURNING_PACT 等

**关键代码路径**：
- Sim 触发：`/STS2AI/ENV/Sim/SrcCompat/Source01032/Core/Commands/CardSelectCmd.cs:516,531`
- Bridge state 构造：`/STS2AI/ENV/Sim/HeadlessSim/Protocol/ProtoStateBuilder.cs:435-441, 456-486, 1124-1176`
- 共享语义：`/STS2AI/ENV/Shared/Simulation/SelectionActionSemantics.cs`
- Python converter：`/STS2AI/bridge/game_bridge/transport/proto_state_converter.py:74-100, 517-531`

**真正的 bug**：
当 SelectableCards 空 + MinSelect=0 时，sim 不自动 confirm，legal_actions 完全为空 → bridge 误判 left_combat。

**Schema 缺**：trigger_card_id、source_pile_type、selection_rules

### B) planner_hint 输入/输出排查

**输入已有**：player/enemy hp/block/intent/powers、deck、relics、potions、hand、pile sizes、glossary

**输入缺**：enemy_phase（ASLEEP/ABOUT_TO_BLOW）、upcoming_intent_chain（未来 2-3 turn）、power_scaling_prediction

**输出缺陷**（最近 boss 战 dump）：
- turn-invariant — T1/T5 几乎一样
- 不预测 phase 转换
- danger_notes 有具体威胁但 battle_objective 空泛（"deal damage"）

最低改进：在 `system_prompt_planner_hint.md` 加 instruction：
> "When round_number > 3, mention how enemy scaling powers (STEAM_ERUPTION_POWER / Strength) change the fight; update battle_objective if phase transitions occur."

## 九、用户最近问的 4 个概念解释（已回答）

1. **combat 不读 planner，方案啥意思** → SFT reason 字段被 canonical 模板写死成"Deal X damage"，跟 planner_hint 没挂钩；让 deepseek corrected_reason verbatim 引用 planner directive，模型才学得到关联
2. **mask reason 啥意思** → canonical 模板让 SKILL 卡也写"Deal X damage"，模型幻觉；mask = reason 字段值置空，loss 只监督 action_index，兜底
3. **改默认值改 6 处是不是没复用** → 不是逻辑重复，是 5 个 CLI 入口各自写 argparse default，建议抽 `DEFAULT_PLANNER_HINT_REFRESH = "turn"` 单一来源
4. **LAGAVULIN AI 处理不好的原因** → 多因素：planner 无 phase awareness、SFT 数据全是失败局、缺破甲一击爆发 pattern、HEADBUTT 异常截断 episode

