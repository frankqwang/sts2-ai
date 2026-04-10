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

MAINLINE_CHECKPOINT = CHECKPOINTS_ROOT / "act1" / "retrieval_final_iter2175.pt"

SIM_HOST_EXE = ENV_ROOT / "Sim" / "Host" / "bin" / "Debug" / "net9.0" / "headless_sim_host_0991.exe"
SIM_LEGACY_DLL = ENV_ROOT / "Sim" / "Runtime" / "HeadlessSim" / "bin" / "Debug" / "net9.0" / "HeadlessSim.dll"
SPECTATOR_MOD_ROOT = ENV_ROOT / "Spectator" / "SpectatorBridgeMod"

SOURCE_KNOWLEDGE_DB = PYTHON_ROOT / "data" / "source_knowledge.sqlite"
SOURCE_KNOWLEDGE_MANIFEST = PYTHON_ROOT / "data" / "source_knowledge.manifest.json"
VOCAB_JSON = PYTHON_ROOT / "vocab.json"
CARD_TAGS_JSON = PYTHON_ROOT / "card_tags.json"
RELIC_TAGS_JSON = PYTHON_ROOT / "relic_tags.json"
