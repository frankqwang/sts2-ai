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
- 支持 `search_root_sweep` 搜索模式；当前主线默认直接按搜索分布自博弈
- 支持按 `encounter / case` 做 targeted 训练

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
  --teacher-mode search_root_sweep `
  --collect-mode search_only_collect `
  --collect-temperature 0.20
```

更偏 AlphaZero 风格的 root MCTS 自博弈示例：

```powershell
cd C:\dev\sts2-ai\STS2AI
python -m zero.replay.train `
  --run-id 1312734 `
  --ordered-run `
  --max-run-combats 10 `
  --parallel-envs 4 `
  --collect-episodes 400 `
  --iterations 4 `
  --train-steps 256 `
  --eval-episodes 1 `
  --teacher-mode search_root_sweep `
  --teacher-max-root-actions 4 `
  --teacher-rollouts-per-action 8 `
  --teacher-max-branch-steps 16 `
  --collect-mode search_only_collect `
  --collect-temperature 0.20
```

按 bottleneck encounter 做 targeted 训练的示例：

```powershell
cd C:\dev\sts2-ai\STS2AI
python -m zero.replay.train `
  --run-id 1312734 `
  --ordered-run `
  --target-encounter NIBBITS_WEAK `
  --teacher-mode search_root_sweep `
  --collect-mode search_only_collect `
  --collect-episodes 200 `
  --iterations 3 `
  --train-steps 192 `
  --eval-episodes 1
```

按单 case 做 targeted 训练的示例：

```powershell
cd C:\dev\sts2-ai\STS2AI
python -m zero.replay.train `
  --target-case-id run_1312734_floor_2_shrinker_beetle_weak `
  --teacher-mode search_root_sweep `
  --collect-mode search_only_collect `
  --collect-episodes 128 `
  --iterations 2 `
  --train-steps 96 `
  --eval-episodes 1
```

几个关键参数的实际语义：
- `--collect-episodes`：每轮 collect 多少场 combat，不是完整 run 次数
- `--iterations`：多少轮 `collect -> teacher -> train -> eval -> promote`
- `--parallel-envs`：只并发 collect rollout，不并发 trainer / eval
- `--collect-epsilon-greedy`：仅在搜索分布上额外加少量随机探索
- `--collect-temperature`：collect 时按搜索分布温度采样；`0` 表示直接取搜索 top-1
- `--collect-mode`：
  - `search_only_collect`：每步都跑搜索，动作直接来自搜索分布
  - `policy_only_collect`：纯 student rollout
  - `search_guided_collect`：只在高优先级状态上让搜索接管动作
- `--teacher-mode`：
  - `search_root_sweep`：same-seed root MCTS / root sweep 搜索
- `--teacher-max-root-actions`：搜索时最多保留多少个根动作分支
- `--teacher-rollouts-per-action`：每个根动作分支分到多少次搜索模拟
- `--teacher-max-branch-steps`：每条分支最多往前 rollout 多少步
- `--target-encounter`：只训练指定 encounter 的 case
- `--target-case-id`：只训练指定 case
- `--target-source`：预留给后续更细粒度 targeted 来源筛选；当前优先用 `target-encounter / target-case-id`

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
3. 搜索直接产出根动作分布与执行动作
4. `SampleBuilder` 把搜索分布写成训练目标
5. 样本进入 online / teacher / rare 池
6. trainer 以搜索策略分布为主监督训练
7. evaluator 在固定 cohort 上评估
8. promotion judge 决定是否晋级 active model

当前主线更接近 AlphaZero 风格的“搜索改进 + 监督回归”：
- collect 时每步搜索
- 策略头主要学搜索分布
- value / ranking / delta / uncertainty 仍保留辅助监督

## 输出目录怎么看

`STS2AI/Artifacts` 下的直接子目录统一用：

- `MMDD-HHMM-name`

例如：
- `0420-2310-skada-replay-train`

这样可以直接按名称顺序定位最新训练目录。

一次训练目录里最常看的东西：

- `run_metrics.json`
- `logs/iter_xxxx.status.json`
- `logs/iter_xxxx.events.jsonl`
- `raw_runs/iter_xxxx.jsonl`
- `eval/iter_xxxx_candidate_eval.jsonl`
- `analysis/`

`analysis/` 下面当前重点看这些：

- `encounter_coverage.csv`
  - 每轮每个 encounter 的 collect 覆盖：episode 数、transition 数、timeout / no-progress
- `encounter_pool_stats.csv`
  - 每轮每个 encounter 在样本池里的保留情况：平均 `sample_weight / keep_score`
- `encounter_teacher_stats.csv`
  - 每轮每个 encounter 进入 teacher queue 的次数、平均 teacher priority
- `encounter_coverage.png`
  - 覆盖和 timeout/no-progress 的图形透视
- `trace_summaries/`
  - 中文轨迹摘要；当前支持 raw 和 eval 两类摘要
  - 如果是 search teacher，会额外显示：
    - student top-k
    - teacher top-k
    - root action sweep 的搜索分数摘要

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
