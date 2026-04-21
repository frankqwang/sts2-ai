"""运行时 sim 常量。"""

from __future__ import annotations

from constants import ARTIFACTS_ROOT, ENV_ROOT, REPO_ROOT, SIM_HOST_EXE, SIM_LEGACY_DLL

CANONICAL_POWER_IDS: dict[str, str] = {
    "strength": "STRENGTH_POWER",
    "dexterity": "DEXTERITY_POWER",
    "vulnerable": "VULNERABLE_POWER",
    "weak": "WEAK_POWER",
    "frail": "FRAIL_POWER",
    "metallicize": "METALLICIZE_POWER",
    "regen": "REGEN_POWER",
    "artifact": "ARTIFACT_POWER",
    "poison": "POISON_POWER",
}

SIM_LOG_ROOT = ARTIFACTS_ROOT / "sim_logs"

__all__ = [
    "ARTIFACTS_ROOT",
    "ENV_ROOT",
    "REPO_ROOT",
    "SIM_HOST_EXE",
    "SIM_LEGACY_DLL",
    "SIM_LOG_ROOT",
    "CANONICAL_POWER_IDS",
]
