"""Teacher 配置：战斗 teacher 的超参数和模式设置。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True, slots=True)
class CombatTurnTeacherConfig:
    search_mode: str = "beam"
    leaf_eval_mode: str = "tactical_mechanism_v1"
    emit_prefix_samples: bool = True
    rerun_solver_per_prefix: bool = True

    hallway_beam_width: int = 4
    elite_beam_width: int = 6
    boss_beam_width: int = 8

    hallway_max_actions: int = 6
    elite_max_actions: int = 8
    boss_max_actions: int = 10

    candidate_cap_per_state: int = 12
    max_nodes_per_solve: int = 2000
    top_n_rerank: int = 8

    damage_progress_weight: float = 0.35
    hp_loss_weight: float = 0.45
    mechanism_weight: float = 0.20

    lethal_bonus: float = 1.00
    bad_end_turn_penalty: float = 0.35
    potion_waste_penalty: float = 0.25
    vulnerable_setup_bonus: float = 0.20
    power_early_bonus_boss: float = 0.40
    power_early_bonus_elite: float = 0.25
    x_cost_first_bonus: float = 0.30
    early_defend_penalty: float = 0.30

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "CombatTurnTeacherConfig":
        if not isinstance(payload, dict):
            return cls()
        known = {item.name: item for item in fields(cls)}
        values: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in known:
                continue
            default = getattr(cls(), key)
            try:
                if isinstance(default, bool):
                    if isinstance(value, str):
                        values[key] = value.strip().lower() in {"1", "true", "yes", "on"}
                    else:
                        values[key] = bool(value)
                elif isinstance(default, int) and not isinstance(default, bool):
                    values[key] = int(value)
                elif isinstance(default, float):
                    values[key] = float(value)
                else:
                    values[key] = str(value)
            except (TypeError, ValueError):
                continue
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> "CombatTurnTeacherConfig":
        clean = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **clean) if clean else self

    def room_kind(self, state: dict[str, Any] | None) -> str:
        state = state if isinstance(state, dict) else {}
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        room = str(run.get("room_type") or state.get("state_type") or "").strip().lower()
        if room == "boss":
            return "boss"
        if room == "elite":
            return "elite"
        return "hallway"

    def beam_width_for_state(self, state: dict[str, Any] | None) -> int:
        room = self.room_kind(state)
        if room == "boss":
            return max(1, int(self.boss_beam_width))
        if room == "elite":
            return max(1, int(self.elite_beam_width))
        return max(1, int(self.hallway_beam_width))

    def max_actions_for_state(self, state: dict[str, Any] | None, fallback: int = 12) -> int:
        room = self.room_kind(state)
        if room == "boss":
            return max(1, int(self.boss_max_actions))
        if room == "elite":
            return max(1, int(self.elite_max_actions))
        return max(1, int(self.hallway_max_actions or fallback))

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def load_combat_turn_teacher_config(path: str | Path | None = None) -> CombatTurnTeacherConfig:
    if not path:
        return CombatTurnTeacherConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"combat turn teacher config not found: {config_path}")
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = payload.get("combat_turn_teacher", payload)
    return CombatTurnTeacherConfig.from_mapping(section if isinstance(section, dict) else {})
