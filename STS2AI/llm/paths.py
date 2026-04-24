"""STS2AI/llm 专用路径常量。

只依赖标准库。不 import `bridge.constants`、不 import `zero.paths`，保持
这个目录可以独立在 `.venv311` 里运行。
"""
from __future__ import annotations

from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parent
STS2AI_ROOT = LLM_ROOT.parent
REPO_ROOT = STS2AI_ROOT.parent

ARTIFACTS_ROOT = STS2AI_ROOT / "Artifacts" / "llm"
DATASETS_ROOT = ARTIFACTS_ROOT / "datasets"
SFT_ROOT = ARTIFACTS_ROOT / "sft"
GRPO_ROOT = ARTIFACTS_ROOT / "grpo"
LOGS_ROOT = ARTIFACTS_ROOT / "logs"
# unsloth 运行时会把 patch 后的 TRL trainer 写到 `./unsloth_compiled_cache/`。
# 统一把所有入口 chdir 到 RUNTIME_CWD，避免污染仓库根。
RUNTIME_CWD = ARTIFACTS_ROOT / "runtime"

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# HuggingFace 缓存里已经下好的 Qwen snapshot；本地推理优先用缓存，不走网
HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
QWEN_CACHE_DIR = HF_CACHE_ROOT / "models--Qwen--Qwen3-4B-Instruct-2507"


def ensure_dirs() -> None:
    for path in (ARTIFACTS_ROOT, DATASETS_ROOT, SFT_ROOT, GRPO_ROOT, LOGS_ROOT, RUNTIME_CWD):
        path.mkdir(parents=True, exist_ok=True)


def setup_runtime() -> None:
    """所有 unsloth 入口脚本都调用一次：

    1. 建好子目录
    2. 把 cwd 切到 Artifacts/llm/runtime，让 `unsloth_compiled_cache/` 落那里
    """
    import os

    ensure_dirs()
    os.chdir(RUNTIME_CWD)


__all__ = [
    "ARTIFACTS_ROOT",
    "BASE_MODEL_ID",
    "DATASETS_ROOT",
    "GRPO_ROOT",
    "HF_CACHE_ROOT",
    "LLM_ROOT",
    "LOGS_ROOT",
    "QWEN_CACHE_DIR",
    "REPO_ROOT",
    "RUNTIME_CWD",
    "SFT_ROOT",
    "STS2AI_ROOT",
    "ensure_dirs",
    "setup_runtime",
]
