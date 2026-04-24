# STS2AI / llm

基于大模型微调直接打杀戮尖塔 2 的主线代码目录。

- 当前 bridge 架构说明：`STS2AI/docs/design/game-bridge-current.md`
- 早期 LLM 微调计划已归档：`STS2AI/docs/archive/2026-04-24/llm-finetune-plan.pre-runtime-state.md`
- 基础模型：`Qwen/Qwen3-4B-Instruct-2507`（HuggingFace 缓存已就位）
- 训练框架：`unsloth` + `trl` + `peft`
- 运行产物：`STS2AI/Artifacts/llm/<stage>/<run_name>/`

## 子目录

- `configs/`：训练 / 推理参数 TOML
- `data_pipeline/`：从 `game_bridge` + 现有 zero policy 产出 SFT 样本
- `prompts/`：系统提示、状态渲染模板（单独文件便于改词）
- `training/`：unsloth SFT / GRPO 入口脚本
- `inference/`：对接 `game_bridge.spectate` 的外部策略 adapter
- `scripts/`：PowerShell 启动脚本（仿 `spectate_zero_checkpoint.ps1`）

## 环境

本目录使用 unsloth studio 自带的 venv，和仓库根的全局 Python 隔离。

- Python 3.13（unsloth 官方支持 `3.11 ≤ Python < 3.14`）
- venv 路径：`C:\Users\Administrator\.unsloth\studio\unsloth_studio`
- CUDA：12.8+（RTX 5070 Ti / Blackwell 必需）
- 激活方式（本仓库运行脚本直接指定这个 Python 可执行文件，不全局激活）

## 约束

- LLM 调用游戏统一走 `GameSession.reset/get_state/act/batch_act`
- 新文档放 `STS2AI/docs/`
- 中间产物、模型权重、训练日志都落 `STS2AI/Artifacts/llm/`
- 文本统一 UTF-8
