# C# ORT CUDA接入说明

## 目的

这次改动把 C# 宿主里的 ONNX Runtime 从纯 CPU 运行时切到了可选 CUDA 运行时，并补上了运行时可观测性，避免出现“代码已经走 C#，但实际还在 CPU 上跑 ORT”这种黑箱。

相关代码：

- [HeadlessSim.csproj](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/HeadlessSim.csproj:142)
- [OrtCombatEvaluator.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/OrtCombatEvaluator.cs:17)
- [Program.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/Program.cs:937)
- [binary_pipe_client.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/ipc/binary_pipe_client.py:388)

## 本次行为变更

### 1. 宿主 ORT 包切到 GPU 版

- `Microsoft.ML.OnnxRuntime` 改成了 `Microsoft.ML.OnnxRuntime.Gpu.Windows`
- 这样 C# 宿主具备 `CUDAExecutionProvider` 能力

### 2. 新增 provider 选择逻辑

`OrtCombatEvaluator` 现在按下面顺序决定推理设备：

1. 读取 `STS2AI_ORT_DEVICE`
2. `cpu` 时强制 `CPUExecutionProvider`
3. `cuda` 或 `gpu` 时强制 `CUDAExecutionProvider`
4. `auto` 时优先尝试 CUDA，失败再回退 CPU

额外环境变量：

- `STS2AI_ORT_CUDA_DEVICE_ID`
  - 选择 CUDA 设备号，默认 `0`
- `STS2AI_ORT_DLL_DIRS`
  - 额外 CUDA/cuDNN DLL 目录，按分号分隔
- `STS2AI_TORCH_LIB_DIR`
  - 手动指定 `torch/lib` 目录
- `STS2AI_PYTHON_EXE`
  - 指定用于自动探测 `torch/lib` 的 Python 可执行文件

### 3. 自动补 torch 的 CUDA DLL 搜索路径

Windows 上 `onnxruntime_providers_cuda.dll` 自己能落地，但它依赖的 `cudnn64_9.dll`、`cublas64_12.dll` 等 DLL 不一定在宿主进程的 `PATH` 里。

当前实现会在首次创建 CUDA session 前：

1. 读取 `STS2AI_ORT_DLL_DIRS`
2. 读取 `STS2AI_TORCH_LIB_DIR`
3. 若还没拿到有效目录，则尝试调用本机 Python：
   - `python -c ...`
   - `py -3 -c ...`
4. 定位到 `torch/lib`
5. 把该目录 prepend 到当前进程的 `PATH`

这样宿主即使不是从 Python 激活环境里直接启动，也能把 PyTorch 自带的 CUDA/cuDNN DLL 借过来给 ORT 使用。

## 协议可观测性

`load_ort_model` 的 binary 响应现在除了原有模型元信息，还会额外返回：

- `execution_provider`
- `requested_device`
- `fell_back_to_cpu`

Python 侧 `BinaryPipeClient` 已同步解码，便于脚本直接断言当前到底在跑哪个 provider。

## 验证结果

### provider 选择 smoke

产物：

- [ort_provider_selection_smoke.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/tmp/ort_provider_selection_smoke.json:1)

结果：

- `STS2AI_ORT_DEVICE=auto` -> `CUDAExecutionProvider`
- `STS2AI_ORT_DEVICE=cpu` -> `CPUExecutionProvider`
- `STS2AI_ORT_DEVICE=cuda` -> `CUDAExecutionProvider`

三种模式都没有发生误判；`auto` 和 `cuda` 都不再回退 CPU。

### CUDA 下的 MCTS 正确性 smoke

产物：

- [mcts_pipe_audit_csharp_eval001_cuda.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/tmp/mcts_pipe_audit_csharp_eval001_cuda.json:1)

结论：

- `root_save_load.matches = true`
- `search_restore.matches_root = true`
- 子节点状态 hash 仍然一致

说明这次 DLL 路径补全只改变了 ORT provider，没有破坏 C# MCTS 的 restore 语义。

### 固定 root 速度

产物：

- 旧 CPU 版：[mcts_speed_eval001_multi_sims.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/tmp/mcts_speed_eval001_multi_sims.json:1)
- 新 CUDA 版：[mcts_speed_eval001_multi_sims_cuda.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/tmp/mcts_speed_eval001_multi_sims_cuda.json:1)

固定 `EVAL_001` 第一个 combat root，重复 5 次后的稳态 C# `search_ms`：

- `16 sims`: `71.67ms -> 62.11ms`
- `64 sims`: `243.84ms -> 183.67ms`
- `128 sims`: `535.51ms -> 336.50ms`

对应对 Python 稳态搜索的速度优势：

- `16 sims`: 约 `1.19x`
- `64 sims`: 约 `1.48x`
- `128 sims`: 约 `1.67x`

## 现阶段结论

- C# 宿主现在已经具备真正可用的 CUDA ORT 推理路径
- `load_ort_model` 可以直接告诉上层到底是 CUDA 还是 CPU
- 固定 root 下，搜索越重，CUDA C# backend 的收益越明显
- 端到端 `evaluate_ai` 仍有一条隔离重启链路需要单独排查，不能把那部分波动算到 ORT CUDA 本身
