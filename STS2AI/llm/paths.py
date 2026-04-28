"""STS2AI/llm 专用路径常量。

只依赖标准库。不 import `bridge.constants`、不 import `zero.paths`，保持
这个目录可以独立在 `.venv311` 里运行。
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

LLM_ROOT = Path(__file__).resolve().parent
STS2AI_ROOT = LLM_ROOT.parent
REPO_ROOT = STS2AI_ROOT.parent

ARTIFACTS_ROOT = STS2AI_ROOT / "Artifacts" / "llm"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
DATASETS_ROOT = ARTIFACTS_ROOT / "datasets"
SFT_ROOT = ARTIFACTS_ROOT / "sft"
GRPO_ROOT = ARTIFACTS_ROOT / "grpo"
EVALS_ROOT = ARTIFACTS_ROOT / "evals"
LOGS_ROOT = ARTIFACTS_ROOT / "logs"
# unsloth 运行时会把 patch 后的 TRL trainer 写到 `./unsloth_compiled_cache/`。
# 统一把所有入口 chdir 到 RUNTIME_CWD，避免污染仓库根。
RUNTIME_CWD = ARTIFACTS_ROOT / "runtime"

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# HuggingFace 缓存里已经下好的 Qwen snapshot；本地推理优先用缓存，不走网
HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
QWEN_CACHE_DIR = HF_CACHE_ROOT / "models--Qwen--Qwen3-4B-Instruct-2507"

# Preferred local LLM runtime. This venv is intentionally not committed; it is
# the single default for train/eval/spectate launchers on this workstation.
LLM_VENV_PYTHON = LLM_ROOT / ".venv311" / "Scripts" / "python.exe"
UNSLOTH_STUDIO_PYTHON = Path.home() / ".unsloth" / "studio" / "unsloth_studio" / "Scripts" / "python.exe"


def resolve_default_python_exe(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the Python executable for LLM train/eval/spectate subprocesses.

    Precedence:
    1. Explicit CLI argument
    2. STS2_LLM_PYTHON_EXE
    3. STS2AI/llm/.venv311
    4. legacy Unsloth Studio venv
    5. current interpreter
    """

    candidates: list[str | os.PathLike[str]] = []
    if explicit:
        candidates.append(explicit)
    env_value = os.environ.get("STS2_LLM_PYTHON_EXE", "").strip()
    if env_value:
        candidates.append(env_value)
    candidates.extend([LLM_VENV_PYTHON, UNSLOTH_STUDIO_PYTHON, sys.executable])

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return Path(sys.executable).resolve()


def ensure_dirs() -> None:
    for path in (ARTIFACTS_ROOT, RUNS_ROOT, DATASETS_ROOT, SFT_ROOT, GRPO_ROOT, EVALS_ROOT, LOGS_ROOT, RUNTIME_CWD):
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
    "EVALS_ROOT",
    "GRPO_ROOT",
    "HF_CACHE_ROOT",
    "LLM_ROOT",
    "LOGS_ROOT",
    "LLM_VENV_PYTHON",
    "QWEN_CACHE_DIR",
    "REPO_ROOT",
    "RUNTIME_CWD",
    "RUNS_ROOT",
    "SFT_ROOT",
    "STS2AI_ROOT",
    "UNSLOTH_STUDIO_PYTHON",
    "ensure_dirs",
    "resolve_default_python_exe",
    "setup_runtime",
]
