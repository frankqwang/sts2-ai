# STS2AI

这是当前仓库里和 AI 训练直接相关的最小说明。  
现阶段默认主线是：

- `STS2AI/zero`：combat-only 的 replay 训练、评估、分析
- `STS2AI/bridge`：Python 侧 bridge / sim 启动 / session 封装
- `STS2AI/data/skada`：skada 原始数据、清洗脚本、权威数据导出

如果你回家继续操作，优先按下面这套流程走。

## 目录

```text
STS2AI/
├── Artifacts/                   单次训练、日志、分析产物
├── Assets/datasets/             长期复用的数据资产
├── data/
│   ├── game_wiki/               权威游戏数据 sqlite
│   └── skada/                   skada 原始数据与清洗脚本
├── bridge/                      Python bridge / sim / session
├── ENV/Sim/HeadlessSim/         C# headless sim
└── zero/
    ├── analysis/                离线分析、轨迹摘要、benchmark
    ├── replay/                  训练 / smoke / case 索引入口
    ├── orchestration/           collect / teacher / loop / trainer
    └── tests/                   单测
```

## 依赖

### Python

先装 bridge 的基础依赖，再补分析依赖：

```powershell
pip install -r STS2AI/bridge/requirements.txt
pip install pandas matplotlib
```

### Sim

训练和 replay 依赖 `HeadlessSim`。  
如果默认 host 过期或 freshness 校验失败，需要先重新 build 一份可用 host。

## 当前默认数据输入

### 训练直接使用

- replay case 索引：
  - [cases.jsonl](/C:/dev/sts2-ai/STS2AI/Assets/datasets/zero_skada_replay_cases/v0_103_2_a0_single_combat_v1/cases.jsonl:1)
- 权威游戏数据：
  - [game_catalog.sqlite](/C:/dev/sts2-ai/STS2AI/data/game_wiki/game_catalog.sqlite:1)

### 不直接作为训练输入

- `STS2AI/data/skada/runs_full_detail`

这是上游原始底库，不进 git，也不应该直接喂给训练。  
训练主线默认只读已经清洗好的 `cases.jsonl`。

## 常用命令

以下命令都建议在 `C:\dev\sts2-ai\STS2AI` 下运行。

### 1. 重建 skada replay case 索引

用途：
- 从 `runs_full_detail` 清洗出长期复用的 combat case 数据集
- 正式训练默认吃这份索引

```powershell
cd C:\dev\sts2-ai\STS2AI
python -m zero.replay.build_case_index `
  --game-version v0.103.2 `
  --ascension 0 `
  --player-count 1 `
  --output-root C:\dev\sts2-ai\STS2AI\Assets\datasets\zero_skada_replay_cases\v0_103_2_a0_single_combat_v1
```

产物：
- `cases.jsonl`
- `summary.json`

### 2. 最小 smoke

用途：
- 快速确认 replay / collect / train / eval 整条链还能跑
- 更适合单 case 检查，不适合作为正式效果结论

```powershell
cd C:\dev\sts2-ai\STS2AI
python -m zero.replay.smoke `
  --port 18131 `
  --episodes 8 `
  --eval-episodes 1 `
  --train-steps 40
```

### 3. 多 case / ordered-run 训练

用途：
- 当前正式训练入口
- 支持随机多 case，也支持同一条 run 的 ordered-run curriculum

ordered-run 示例：

```powershell
cd C:\dev\sts2-ai\STS2AI
python -m zero.replay.train `
  --run-id 1312734 `
  --ordered-run `
  --max-run-combats 10 `
  --parallel-envs 4 `
  --collect-episodes 1000 `
  --iterations 10 `
  --train-steps 512 `
  --eval-episodes 1 `
  --collect-epsilon-greedy 0.05 `
  --collect-temperature 0.20
```

几个关键参数的实际语义：
- `--collect-episodes`：每轮 collect 多少场 combat，不是完整 run 次数
- `--iterations`：多少轮 `collect -> teacher -> train -> eval -> promote`
- `--parallel-envs`：只并发 collect rollout，不并发 trainer / eval
- `--collect-epsilon-greedy`：collect 时小概率随机探索
- `--collect-temperature`：collect 时按 softmax 温度采样；`0` 表示关闭

### 4. 轨迹中文摘要

用途：
- 从已经落盘的 `raw_runs` / `eval` 轨迹里随机抽样
- 生成便于人工排查的中文摘要
- 不绑定训练主循环，独立运行

```powershell
cd C:\dev\sts2-ai\STS2AI
python -m zero.analysis.trace_summary `
  --run-root C:\dev\sts2-ai\STS2AI\Artifacts\zero\某次训练目录 `
  --source both `
  --iters 3 `
  --samples-per-iter 2 `
  --max-steps-per-fight 20
```

### 5. rollout benchmark

用途：
- 测试 1 / 2 / 4 / 8 env 的 rollout 吞吐
- 看的是 collect rollout，不是训练效果

```powershell
cd C:\dev\sts2-ai\STS2AI
python -m zero.analysis.rollout_benchmark `
  --run-id 1312734 `
  --ordered-run `
  --max-run-combats 10 `
  --episodes 200 `
  --env-counts 1 4 8
```

## 当前训练流程

现在的主流程是：

1. 从 `cases.jsonl` 选择训练 / 评估 case
2. collect rollout 采样轨迹
3. `SampleBuilder` 生成训练样本
4. teacher 给关键状态补标签
5. 样本进入 online / teacher / rare 池
6. trainer 从样本池混采训练
7. evaluator 在固定 cohort 上评估
8. promotion judge 决定是否晋级 active model

当前主线不是 RL policy gradient，仍然是：
- teacher 蒸馏
- 行为克隆
- value / ranking / delta / uncertainty 辅助头

## 输出目录怎么看

`STS2AI/Artifacts/zero` 下的直接子目录统一用：

- `MM-DD-HH-MM-name`

例如：
- `04-20-23-10-skada-replay-train`

这样可以直接按名称顺序定位最新训练目录。

一次训练目录里最常看的东西：

- `run_metrics.json`
- `logs/iter_xxxx.status.json`
- `logs/iter_xxxx.events.jsonl`
- `raw_runs/iter_xxxx.jsonl`
- `eval/iter_xxxx_candidate_eval.jsonl`
- `analysis/`

## 注释与维护约定

- 关键文件、关键函数要写“说明意图”的短注释，尤其是：
  - loss
  - 样本评分
  - 样本池保留/淘汰
  - teacher queue
  - promotion gate
  - replay 课程语义
- 代码注释默认用中文。
- 变更逻辑时，注释必须在同一次提交里同步更新。
- PowerShell、bash 等凡是涉及文件输入输出，默认显式使用 UTF-8 读写，避免乱码。

## 当前已知限制

- `runs_full_detail` 很大，不进 git；训练不直接依赖它。
- `game_catalog.sqlite` 已进 git，作为当前权威本地查询库。
- 现在可以从零开始稳定训练；跨机器“正式 resume”底层能力已有，但 CLI 还没单独做一键 resume 参数。
- collect 可以并发；按当前基准，纯 rollout 吞吐的甜点位大概在 `4 env` 左右。
