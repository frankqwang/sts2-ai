"""prompt 渲染模板。模板文本放在同级 .md / .txt，Python 只做装载。"""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_system_prompt(action_mode: str = "index") -> str:
    if action_mode == "structured":
        return (_PROMPTS_DIR / "system_prompt_structured_action.md").read_text(encoding="utf-8")
    if action_mode in {"non_combat", "run_strategy"}:
        return (_PROMPTS_DIR / "system_prompt_non_combat.md").read_text(encoding="utf-8")
    return (_PROMPTS_DIR / "system_prompt.md").read_text(encoding="utf-8")


__all__ = ["load_system_prompt"]
