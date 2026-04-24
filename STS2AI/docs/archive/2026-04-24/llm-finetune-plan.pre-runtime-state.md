# 杀戮尖塔 LLM 微调主线设计草案

更新时间：2026-04-23
路线负责人：`STS2AI/llm`

本文是新路线的**最小对齐文档**：确定"要干什么、不干什么、从哪里开始"，不是最终架构。

## 1. 目标

用 `Qwen/Qwen3-4B-Instruct-2507` + unsloth（LoRA 4bit），让模型直接读 `game_bridge` 返回的 JSON 状态，输出一个合法动作。最后替换 `zero_external_policy.ZeroExternalPolicyAdapter`，在 visible 观战或训练链路里跑。

明确**不做**的事：

- 不做预训练，不扩词表
- 不自己实现 tokenizer / sampler / attention kernel
- 不把 skada 的整局统计当 SFT 目标（它没有逐步动作标注）
- 不在这条线上再搭一套新的 HTTP 桥或新的 sim launcher

## 2. 当前可用资产

- `Qwen3-4B-Instruct-2507`：已在 `C:/Users/Administrator/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507`
- GPU：RTX 5070 Ti，16 GB VRAM（4bit + LoRA 足够）
- 游戏桥：`STS2AI/bridge/game_bridge`（HTTP `:15526` singleplayer / `:15527` headless）
- Teacher policy：现有 `zero` checkpoint（PPO/AlphaZero 系），可作为行为克隆教师
- 社区 run 数据：`STS2AI/data/skada/runs_full_detail`（只有聚合统计，留给 retrieval/分析）

## 3. 关键约束

### 3.1 环境

- unsloth 官方支持 Python 3.11 ≤ x < 3.14，**Py 3.13 在列**（核对自官方 requirements 页）。
- RTX 5070 Ti 是 Blackwell，需要 **CUDA 12.8+**，当前 torch 2.11/cu128 OK。
- 直接用 `C:\Users\Administrator\.unsloth\studio\unsloth_studio` 这个 3.13 venv，不另建。
- 全局 Python 不动。

### 3.2 动作空间

- 每个状态 `game_bridge` 都会给 `legal_actions: [...]`，模型输出必须能**一一对应**其中一条。
- 推理阶段用**严格 JSON 解码 + 合法动作过滤**：模型出的动作若不在 legal 集合里，回退到启发式（比如 end_turn 或 index=0）。
- 不让模型自己发明动作字符串。

### 3.3 上下文

- 一次 prompt 控制在 ~3K tokens 内：玩家状态、手牌、敌人、legal_actions、最近 N 步历史。
- 回合 meta（地图、遗物）可以先压成紧凑 KV 文本，后面再考虑 retrieval。

## 4. 分阶段路线

### Phase 0 — 环境与目录

1. 在 `STS2AI/llm/` 建好子目录：`configs/ data_pipeline/ prompts/ training/ inference/ scripts/`
2. 建 `.venv311`，装 unsloth + matching torch。
3. `scripts/smoke_load_qwen.py`：只做 "4bit 加载 + 一次 chat" 冒烟，确认本地能跑。

### Phase 1 — 数据管道（`data_pipeline/`）

1. `rollout_teacher.py`：启 `HeadlessSim` + 现有 zero policy，reset 到指定战斗 / build，逐步记录：
   - 原始 state JSON
   - `legal_actions`
   - teacher 选择的动作（index + 语义字段）
   - 单步 reward proxy（HP 差、结算信号）
   - 战斗结束的 outcome
2. `transcribe.py`：把 rollout 压成 prompt/response 对，写成 `train.jsonl` / `eval.jsonl`。
3. 先只覆盖 **Act 1 单场战斗**（例如 `CHOMPERS_NORMAL`），跑通全链路再扩量。

### Phase 2 — SFT（`training/sft_lora.py`）

- unsloth `FastLanguageModel` + Qwen3-4B，4bit，LoRA r=16 / alpha=32，targets = `q,k,v,o,gate,up,down`。
- `max_seq_length` 先设 4096。
- 训练目标：只在 `response` 段算 loss（用 `train_on_responses_only` 之类 helper）。
- 产物：`STS2AI/Artifacts/llm/sft/<run_name>/adapter/`。

### Phase 3 — 推理接入（`inference/`）

- `llm_policy.py`：复刻 `ZeroExternalPolicyAdapter` 的接口
  - `reset_episode()`
  - `select_action(state, legal_actions, ctx)`
- 内部流程：
  1. 把 state + legal_actions 渲染成 prompt
  2. 用加载好的 LoRA 模型生成
  3. 解析 JSON → 匹配合法动作 → 返回
- `scripts/spectate_llm.ps1`：仿 `spectate_zero_checkpoint.ps1`，把 `STS2_EXTERNAL_POLICY` 指向 `llm_policy:LlmExternalPolicyAdapter`。

### Phase 4 — 可选：GRPO 自对弈

只有 Phase 3 打通并有合理胜率后再考虑：

- reward = 战斗胜利 + 剩余血量 / 回合数奖励项
- unsloth 支持 GRPO；trainer 一次只更新 LoRA；state 由 rollout loop 现场喂入

## 5. 成功判据（粗）

- Phase 0：`smoke_load_qwen.py` 生成一段可读中文回复。
- Phase 1：在一场指定战斗上拿到 ≥ 200 条 teacher 样本，字段完整。
- Phase 2：eval set 上 action-level top-1 命中率 > 随机合法动作基线（手写基线一条）。
- Phase 3：在 `CHOMPERS_NORMAL` 这一场能稳定走完整场不卡住。
- Phase 4：自对弈连跑 50 场，胜率明显高于未微调基线。

## 6. 还没决定、需要再讨论的事

- Teacher 是单 zero checkpoint，还是再加一条启发式 / MCTS baseline 增加多样性
- prompt 语言：英文（对齐 Qwen 训练语料）vs 中文（对齐团队沟通）
- 是否把 skada 汇总信息拼进 system prompt（build 先验）
- 合法动作失败时的兜底策略（end_turn / index=0 / 重采样）

这些先留空，Phase 1 数据出来以后再补。
