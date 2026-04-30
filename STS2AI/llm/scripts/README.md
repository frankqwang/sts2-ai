# LLM Scripts

脚本按用途分组。优先用 `python -m llm.scripts.<group>.<script>` 运行 Python 脚本，PowerShell 入口仍直接用文件路径。

## 目录

- `analysis/`：trace 审计、失败复盘、动作顺序分析、eval 对比。
- `automation/`：单轮/多轮训练飞轮和 Act 1 clear 编排。
- `datasets/`：训练集构建、数据池管理、preflight、离线偏好挖掘。
- `teacher/`：Kimi/teacher 候选采样、复盘标注、经验库追加。
- `viewers/`：trace replay、metrics summary、单文件 HTML 可视化。
- `spectate/`：LLM/heuristic 观战 PowerShell 入口。
- `debug/`：状态 dump、模型加载 smoke、E2E smoke。

## 常用入口

```powershell
$env:PYTHONPATH="C:\Users\Administrator\Desktop\sts2Zero\STS2AI"

python -m llm.scripts.datasets.manage_dataset_pool report
python -m llm.scripts.analysis.review_step_trace --trace STS2AI\Artifacts\llm\spectate_llm\<run>\step_trace.jsonl
python -m llm.scripts.analysis.check_guide_corpus --corpus STS2AI\llm\knowledge\guide_corpus.jsonl
python -m llm.scripts.analysis.eval_planner_hint_outputs --dataset STS2AI\Artifacts\llm\datasets\<planner_hint_run> --require-knowledge
python -m llm.scripts.viewers.trace_viewer_html --trace STS2AI\Artifacts\llm\spectate_llm\<run>\step_trace.jsonl
python -m llm.scripts.datasets.build_planner_hint_dataset --review-root STS2AI\Artifacts\llm\reviews\<run>
python -m llm.scripts.automation.self_train_loop --help
```

Teacher 标注底层模型可切换。默认仍走 Kimi；本机 Claude CLI 可用时，用下面参数切到 Sonnet，并通过本地代理访问：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"

python -m llm.scripts.teacher.run_kimi_combat_review_batch `
  --provider claude_cli `
  --model claude-sonnet-4-6 `
  --claude-proxy "http://127.0.0.1:7897" `
  --trace STS2AI\Artifacts\llm\evals\<run>\step_trace.jsonl `
  --out-dir STS2AI\Artifacts\llm\reviews\<claude_run> `
  --limit-episodes 20 `
  --max-workers 4 `
  --skip-existing
```

观战入口：

```powershell
powershell -ExecutionPolicy Bypass -File .\STS2AI\llm\scripts\spectate\spectate_llm.ps1
```
