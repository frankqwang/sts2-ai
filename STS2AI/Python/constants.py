"""全局路径常量：checkpoint、数据集、产物目录等。"""
from __future__ import annotations

from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent
STS2AI_ROOT = PYTHON_ROOT.parent
REPO_ROOT = STS2AI_ROOT.parent

ENV_ROOT = STS2AI_ROOT / "ENV"
ASSETS_ROOT = STS2AI_ROOT / "Assets"
ARTIFACTS_ROOT = STS2AI_ROOT / "Artifacts"

CHECKPOINTS_ROOT = ASSETS_ROOT / "checkpoints"
DATASETS_ROOT = ASSETS_ROOT / "datasets"
SEEDS_ROOT = ASSETS_ROOT / "seeds"

MAINLINE_CHECKPOINT = CHECKPOINTS_ROOT / "act1" / "mainline_iter2270_carddebug.pt"

# 优先新结构 HeadlessSim/bin；旧路径只作兼容 fallback。
_SIM_CANDIDATES = (
    ENV_ROOT / "Sim" / "HeadlessSim" / "bin" / "Release" / "net9.0" / "HeadlessSim.exe",
    ENV_ROOT / "Sim" / "HeadlessSim" / "bin" / "Debug" / "net9.0" / "HeadlessSim.exe",
)
SIM_HOST_EXE = next((path for path in _SIM_CANDIDATES if path.exists()), _SIM_CANDIDATES[0])
SIM_LEGACY_DLL = ENV_ROOT / "Sim" / "HeadlessSim" / "bin" / "Debug" / "net9.0" / "HeadlessSim.dll"
SPECTATOR_MOD_ROOT = ENV_ROOT / "Spectator" / "SpectatorBridgeMod"

SOURCE_KNOWLEDGE_DB = PYTHON_ROOT / "data" / "source_knowledge.sqlite"
SOURCE_KNOWLEDGE_MANIFEST = PYTHON_ROOT / "data" / "source_knowledge.manifest.json"
VOCAB_JSON = PYTHON_ROOT / "data" / "vocab.json"
CARD_TAGS_JSON = PYTHON_ROOT / "data" / "card_tags.json"
RELIC_TAGS_JSON = PYTHON_ROOT / "data" / "relic_tags.json"
