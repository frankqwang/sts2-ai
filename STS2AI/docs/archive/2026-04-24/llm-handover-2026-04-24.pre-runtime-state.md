# LLM 微调打 STS2 — 交接文档

**交接时间**：2026-04-24
**目录**：`/C:/Users/Administrator/Desktop/sts2Zero/STS2AI/llm/`
**当前 checkpoint**：`/STS2AI/Artifacts/llm/sft/act1_v1/adapter/`

## 1. 项目定位

用 **Qwen3-4B-Instruct-2507 + LoRA + unsloth** 做 SFT，让 LLM 读 `game_bridge` 的 state JSON，输出合法动作，打杀戮尖塔 2。

核心取舍（已和用户对齐）：
- ❌ **不做 MCTS**：save/load 是作弊，不通用
- ❌ **不做 GRPO**：单卡 5070 Ti 跑不动
- ❌ **不做合成非战斗数据**：假数据训不出真判断
- ❌ **不做海量机械模仿**：学不到"懂"
- ✅ **走 reasoning 路线**：让模型脑内 CoT 推演（R1 风格），不是单纯背答案
- ✅ **信息补全优先于训练**：输入残缺，训出的判断就是错的
- ✅ **接受 4B 上限**：浅推理（2-3 步 × 2-3 分支），想通关 Act3 至少要 9B

## 2. 当前现状

### 2.1 v1 模型（已完成）

- **数据**：2759 样本（2484 train / 275 eval），Ironclad × 7 种 Act1 encounter，启发式老师标注
- **训练**：LoRA r=16，QLoRA 4bit，batch=2 × grad_accum=8，2 epoch，46 分钟
- **结果**：
  - train_loss 0.376，eval_loss 0.183（健康，不过拟合）
  - e2e_smoke 4/5 命中（toy v0 是 1/5）
  - JSON 合规 100%，fallback 率 0%
  - 生成速度 0.9-1.8s/step

### 2.2 端到端链路已通

完整验证过：
- `spectate_llm.ps1` 启动 Godot + LlmExternalPolicyAdapter
- LLM 在 Godot 里实打（单次 run 走完 50 步，完成 2 场战斗）
- `step_trace.jsonl` 记录每步的 thinking + action
- `replay_trace.py` 实时查看模型思考

**已知 sim 边界问题**：`proceed` 动作偶尔被 sim 拒绝（状态切换瞬时竞态）。已在 `SpectatorController` 加 recovery patch，最多容忍 5 次拒绝后继续。

### 2.3 当天做的关键改动

#### Sim 端改动（C# + proto）

| 文件 | 改动 |
|---|---|
| `/STS2AI/ENV/proto/game_state.proto` | HandCard 加 `description` / `keywords` / `preview_damage_per_target` / `preview_block` 字段 |
| `/STS2AI/ENV/Sim/HeadlessSim/Protocol/ProtoStateBuilder.cs` | `BuildBattleState` 把 `card.Description` 和 `Keywords` 填入 proto |
| `/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/` | 重编译通过（8s，309 warnings / 0 errors）|
| `/STS2AI/bridge/game_bridge/generated/game_state_pb2.py` | 重生成 Python proto stubs |
| `/STS2AI/bridge/game_bridge/transport/proto_state_converter.py` | `_convert_hand_card` 读新字段传入 dict |

**发现**：sim 返回的 `description` 是 localization key（`"STRIKE_IRONCLAD.description"`），不是解析后文本。真实文本在 `/STS2AI/data/game_wiki/game_catalog.sqlite` 的 `cards.description_zh` 里（含模板占位符 `{Damage:diff()}`）。

#### Python 端改动（信息补全）

| 文件 | 职责 |
|---|---|
| `/STS2AI/llm/data_pipeline/catalog_loader.py` | 统一加载 card 描述（从 sqlite）+ 手写 50+ 条 relic / 30+ 条 power 描述 |
| `/STS2AI/llm/data_pipeline/state_renderer.py` | 完全重写：加入 relics / potions / 牌堆内容 / 完整牌组 / 敌我 buff 可读描述 / 手牌描述 |
| `/STS2AI/bridge/game_bridge/spectate/controller.py` | 加 `max_recoverable_step_errors` 参数，act 被拒时 refresh state 继续 |

### 2.4 现在 prompt 长什么样

```
run: char=IRONCLAD act=0 floor=0 encounter=CHOMPERS_NORMAL round=1 gold=125
player: hp=70/80 block=0 energy=3/3 buffs=-
relics:
  [0] BURNING_BLOOD — 赤红熔血：战斗结束时回复 6 HP
  [1] HAND_DRILL — 手钻：对破盾的敌人施加 2 易伤
  [2] MINIATURE_CANNON — 迷你加农炮：战斗开始对所有敌人造成 7 伤害
  [3] SILVER_CRUCIBLE — 银坩埚：战斗结束 1/2 概率回 4 HP
deck: STRIKE_IRONCLAD×3, DEFEND_IRONCLAD×3, BASH, POMMEL_STRIKE+, SETUP_STRIKE+, ...
piles:
  draw[7]: STRIKE_IRONCLAD×2, DEFEND_IRONCLAD×2, CINDER, BASH, SETUP_STRIKE
  discard[0]: -
  exhaust[0]: -
enemies:
  id=1 CHOMPER hp=64/64 block=0 intent=Attack(8x2) buffs=神器：免疫下次减益 (×2)
  id=2 CHOMPER hp=62/62 block=0 intent=StatusCard buffs=神器：免疫下次减益 (×2)
hand:
  [0] BLUDGEON cost=3 tags=attack,upg | 造成{Damage:diff()}点伤害。
  [1] STRIKE_IRONCLAD cost=1 tags=attack | 造成{Damage:diff()}点伤害。
  [3] DEFEND_IRONCLAD cost=1 tags=skill | 获得{Block:diff()}点格挡。
  [4] FORGOTTEN_RITUAL cost=1 tags=skill | 如果你在本回合消耗过卡牌，则获得{Energy:energyIcons()}。
legal_actions: ...
```

## 3. 下一步计划

### 阶段 X（短期，1-3 天）—— 把信息彻底补全

前提判断：**不先把 prompt 信息做到完整真实，再训 reasoning 也是训错**。

#### X.1 解析卡牌描述占位符（Python，0.5 天）

`{Damage:diff()}` / `{Block:diff()}` / `{MagicNumber}` 这类模板要替换成实际数值。

来源：
- `game_catalog.sqlite` 的 cards 表有 `payload_json` 包含 `base_damage` 等结构化数据
- sim 也可以出（需要在 HandCardSnapshot 里暴露 `base_damage` 等）

方案：Python 端正则匹配 + catalog 查找，替换成具体数字。`[gold]...[/gold]` markup 已在 `catalog_loader._clean_text` 剥离。

#### X.2 填 sim 的 preview_damage_per_target（C#，1 天）

proto 字段和 Python converter 都已就位，就差 `ProtoStateBuilder.cs` 里真的调 sim 内部的 damage 计算 API 填值。

要找的 API（推测）：
- `CombatDamageCalculator.CalculateIncomingDamage(card, target)`
- 或 `card.GetAttackDamage(target)`（STS1 风格）
- 或 target 的 `ApplyIncomingDamageModifiers(baseDmg)`

找到后在 line 785（foreach ValidTargetIds）那个循环里填入 map。

同理 `preview_block` = `card.GetBlockValue(player)`。

#### X.3 敌人 intent 的 modifier 调整（C#，0.5 天）

现在 `Intent.Damage` 是 base，没考虑玩家易伤 / 敌人力量。`ProtoStateBuilder` 要在构造 Intent 时也走一次 modifier。

#### X.4 system prompt 重写（Python，0.5 天）

当前 system prompt 只有决策原则。要补：
- 核心机制：力量 / 脆弱 / 虚弱 / 敏捷 / 格挡衰减 / 能量 / 回合流程
- 常用 keyword 解释：exhaust / ethereal / retain
- 职业特征（Ironclad）：怒气 / 扫尾打法
- 约 500-800 tokens

### 阶段 Y（中期，3-5 天）—— Reasoning teacher 路线

前提：阶段 X 完成，prompt 信息完整。

#### Y.1 接入 reasoning API

推荐 **DeepSeek-R1**（便宜，中文好，thinking 可见）。
- 每条样本成本 1-2 元
- 500 条 ≈ 500-1000 元

写 `/STS2AI/llm/data_pipeline/reasoning_teacher.py`：
- 拉现有 rollout state
- 构造 prompt（带完整信息）
- 调 R1 API，让它自由思考输出 `<think>...</think> {"action_index": N, "reason": "..."}`
- 保存完整 thinking + action

#### Y.2 训练 v2-reasoning

数据格式变化：assistant 消息不再是简单 JSON，而是 `<think>分支推演</think>JSON`。

训练改动：
- `max_seq_length` 1024 → 4096（长 CoT）
- 每 epoch 时间从 46 min → 可能 2-3 小时（长序列）
- 每条样本 token 量 ~1500-3000

#### Y.3 推理调整

`llm_policy.py`：
- `max_new_tokens` 64 → 2048
- 每步生成 5-15 秒（代价，换取真推理）
- step_delay 可去掉
- trace 里会看到真 thinking 分支

### 阶段 Z（远期，未定时间）—— Outcome RL

在 reasoning SFT 之上做 **DPO 或 GRPO**。
- DPO 更便宜：从 v2 自己 rollout 的胜 / 败轨迹构造 pair
- GRPO 更理想但单卡跑不动

此时不急。先看 reasoning SFT 能到什么水平。

### 非战斗部分（独立推进）

skada 真人数据有丰富的非战斗决策：
- `map_acts.visited_coords`：地图选路
- `floor_timeline.relic_choices`：遗物选择（带 `was_picked`）
- `floor_timeline.ancient_choices`：稀有选择
- 未用：event 选择可能要从 `event_text` parse

做法：
- 写 `/STS2AI/llm/data_pipeline/skada_convert.py`，从 `runs_full_detail/victory/details/*.jsonl` 采 3000-5000 局，转成 messages 样本
- 合进 v2-reasoning 训练数据（weighted sampling 50:50）
- 一个 LoRA 同时处理战斗 + 非战斗

## 4. 关键文件索引

### 代码
```
/STS2AI/llm/
├── paths.py                            # 路径常量 + setup_runtime()
├── prompts/
│   └── system_prompt.md                # 系统提示（阶段 X.4 要重写）
├── data_pipeline/
│   ├── state_renderer.py               # 【重写完】prompt 渲染，信息完整
│   ├── catalog_loader.py               # 【新】card/relic/power 描述
│   ├── heuristic_teacher.py            # 启发式老师（战斗）
│   ├── non_combat_teacher.py           # 启发式老师（非战斗，不会再用）
│   ├── action_decoder.py               # 解析 LLM 输出为合法动作
│   ├── rollout_heuristic.py            # 单战斗 rollout
│   ├── rollout_full_run.py             # 整局 rollout（有 sim bug 未用）
│   └── synthetic_non_combat.py         # 合成非战斗数据（废弃）
├── training/
│   └── sft_lora.py                     # SFT 入口（支持 --load-in-4bit）
├── inference/
│   ├── llm_policy.py                   # SpectatorController 用的 policy
│   └── heuristic_policy.py             # 启发式版 policy
├── scripts/
│   ├── smoke_load_qwen.py              # 冒烟加载模型
│   ├── e2e_smoke.py                    # e2e top-1 测试
│   ├── spectate_llm.ps1                # 观战（LLM）
│   ├── spectate_heuristic.ps1          # 观战（启发式）
│   ├── replay_trace.py                 # 看模型 thinking
│   ├── tail_latest_spectate.py         # 自动找最新 trace
│   ├── audit_state_completeness.py     # 【新】信息完整性审计
│   └── dump_full_run_states.py         # 【新】state_type 采样
└── configs/
    └── build_ironclad_midrun.json      # Ironclad 中期 build
```

### Sim 端
```
/STS2AI/ENV/proto/game_state.proto                          # 已加 description/keywords/preview_*
/STS2AI/ENV/Sim/HeadlessSim/Protocol/ProtoStateBuilder.cs   # 【改】line 772-800 填 Description + Keywords
/STS2AI/ENV/Sim/HeadlessSim/Training/CombatTrainingDtos.cs  # snapshot DTO 定义（已有 Description/Keywords）
/STS2AI/ENV/Sim/HeadlessSim/Training/CombatTrainingEnvService.cs  # line 559 BuildHandCardSnapshot 填 snapshot
```

### 数据
```
/STS2AI/Artifacts/llm/datasets/heuristic_act1_v1/     # v1 训练数据
/STS2AI/Artifacts/llm/sft/act1_v1/adapter/            # v1 LoRA adapter（能用）
/STS2AI/Artifacts/llm/diagnostics/audit/              # state 审计结果
/STS2AI/data/game_wiki/game_catalog.sqlite            # 卡牌描述数据源
/STS2AI/data/skada/runs_full_detail/victory/details/  # 真人 run 数据（非战斗用）
```

### 观战产物
```
/STS2AI/Artifacts/llm/spectate_llm/<timestamp>/
├── manifest.json
├── live_overlay.json
├── step_trace.jsonl       # 每步 thinking + action
└── logs/spectate.stdout.log / spectate.stderr.log
```

## 5. 运行环境

### Python 环境
- **全局 Python 3.13.12**：跑 rollout / game_bridge（不需要 unsloth）
- **Unsloth Studio venv**：`C:\Users\Administrator\.unsloth\studio\unsloth_studio\Scripts\python.exe`
  - 含 unsloth 2026.4.8 + torch 2.10+cu130 + trl 0.23 + peft 0.18

### 硬件
- RTX 5070 Ti 16 GB（Blackwell sm_120）
- CUDA 12.0（驱动）/ toolkit 13.0 / Triton 3.6

### 路径约定（CLAUDE.md）
- 新代码 → `/STS2AI/llm/`
- 新文档 → `/STS2AI/docs/design/`
- 运行产物 → `/STS2AI/Artifacts/llm/`
- 文本 UTF-8
- 中文沟通

## 6. 已知坑

1. **Windows PowerShell 5.1 对 UTF-8 无 BOM 的中文字符串不友好** —— ps1 脚本里 `Write-Host`/`throw` 的字符串用 ASCII，中文放注释
2. **bash shell 会吞 `\l`、`\s` 这类反斜杠** —— ps1 路径用正斜杠
3. **unsloth 4bit 训练比 fp16 慢 2-5 倍** —— 5070 Ti 16GB 装 4B fp16 + LoRA 会 OOM，只能 4bit
4. **sim pipe proto 对 `full_run_env/step` 的 combat play_card 有拒绝 bug** —— `rollout_full_run.py` 无法端到端收非战斗样本，走 skada 替代
5. **ScheduleWakeup prompt 过期** —— 如果你看到很早的 wakeup 还在回放，忽略即可
6. **Qwen/Qwen3-4B-Instruct-2507 vs unsloth/Qwen3-4B-Instruct-2507 是不同的 HF 缓存**，fp16 会触发下载 unsloth 版本（~8GB）

## 7. 立刻能跑的几条命令

### 观战 LLM 打 CHOMPERS
```powershell
cd C:\Users\Administrator\Desktop\sts2Zero\STS2AI
powershell -ExecutionPolicy Bypass -File .\llm\scripts\spectate_llm.ps1 `
  -EncounterId CHOMPERS_NORMAL `
  -BuildFile .\llm\configs\build_ironclad_midrun.json
```

### 跑 e2e 看 v1 命中率
```bash
cd /c/Users/Administrator/Desktop/sts2Zero/STS2AI/Artifacts/llm/runtime
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
  STS2_LLM_ADAPTER_DIR="C:/Users/Administrator/Desktop/sts2Zero/STS2AI/Artifacts/llm/sft/act1_v1/adapter" \
  C:/Users/Administrator/.unsloth/studio/unsloth_studio/Scripts/python.exe \
  ../../../llm/scripts/e2e_smoke.py
```

### 审计 prompt 信息完整度
```bash
PYTHONIOENCODING=utf-8 C:/Users/Administrator/AppData/Local/Programs/Python/Python313/python.exe \
  /c/Users/Administrator/Desktop/sts2Zero/STS2AI/llm/scripts/audit_state_completeness.py
```

### 重新构建 sim（改 C# / proto 后）
```bash
cd /c/Users/Administrator/Desktop/sts2Zero/STS2AI/ENV/Sim/HeadlessSim
dotnet build -c Release --nologo -v minimal
# 如果改了 proto，还要重新生成 Python stubs：
cd ../../proto
C:/Users/Administrator/AppData/Local/Programs/Python/Python313/python.exe \
  -m grpc_tools.protoc --proto_path=. \
  --python_out=../../bridge/game_bridge/generated game_state.proto
```

## 8. 下一个人接手要做的第一件事

按优先级：

1. **读这个文档 + `/STS2AI/docs/design/llm-finetune-plan.md`**
2. **跑一次 `spectate_llm.ps1` 观战**（验证环境通）
3. **跑一次 `audit_state_completeness.py`**（看新 prompt 长啥样）
4. **决定走阶段 X.1（占位符解析）还是 X.2（sim preview damage）**
   - X.1 更简单，半天完成
   - X.2 需要摸 STS2 damage calc API，1 天
5. **阶段 X 全部做完后，评估信息是否够训 reasoning**
6. **确定阶段 X 完成，再开启阶段 Y（R1 teacher）**

**不要做的事**：
- 不要再往 `heuristic_act1_v1` 数据集里加数据（方向已变为 reasoning）
- 不要训合成非战斗数据（已废弃）
- 不要改 4bit 训练到 fp16（VRAM 不够，上次 OOM）
- 不要考虑 MCTS / GRPO 自对弈（短期不做）
