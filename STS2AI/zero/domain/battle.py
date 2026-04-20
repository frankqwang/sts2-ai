from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TargetSummary:
    hp: float = 0.0
    max_hp: float = 0.0
    block: float = 0.0
    intent_id: str = ""
    alive: bool = True
    buffs: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PlayerState:
    hp: float
    max_hp: float
    block: float
    energy: float
    potions: list[str] = field(default_factory=list)
    buffs: dict[str, float] = field(default_factory=dict)
    resources: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EnemyState:
    enemy_id: str
    hp: float
    max_hp: float
    block: float
    intent_id: str
    alive: bool = True
    buffs: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HandCardState:
    card_id: str
    cost_now: float
    damage_now: float = 0.0
    block_now: float = 0.0
    magic_now: float = 0.0
    is_upgraded: bool = False
    retain: bool = False
    exhaust: bool = False
    ethereal: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PileSummary:
    draw_pile_size: int = 0
    discard_pile_size: int = 0
    exhaust_pile_size: int = 0
    attack_count: int = 0
    skill_count: int = 0
    power_count: int = 0
    key_card_counts: dict[str, int] = field(default_factory=dict)
    archetype_stats: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class StaticContext:
    character_id: str = ""
    act: int = 1
    floor: int = 0
    encounter_class: str = "normal"
    encounter_id: str = ""
    relics: list[str] = field(default_factory=list)
    fixed_powers: list[str] = field(default_factory=list)
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)


@dataclass(slots=True)
class LegalAction:
    action_id: str
    action_type: str
    can_execute: bool = True
    card_id: str = ""
    potion_id: str = ""
    special_id: str = ""
    target_id: str = ""
    cost_now: float = 0.0
    damage_now: float = 0.0
    block_now: float = 0.0
    magic_now: float = 0.0
    tags: list[str] = field(default_factory=list)
    target_summary: TargetSummary | None = None
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class BattleState:
    player: PlayerState
    enemies: list[EnemyState]
    hand: list[HandCardState]
    piles: PileSummary
    context: StaticContext
    legal_actions: list[LegalAction]
    terminal: bool = False
    run_outcome: str = ""
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def living_enemies(self) -> list[EnemyState]:
        return [enemy for enemy in self.enemies if enemy.alive]


@dataclass(slots=True)
class TransitionDelta:
    self_hp: float = 0.0
    self_block: float = 0.0
    self_energy: float = 0.0
    enemy_hp: list[float] = field(default_factory=list)
    enemy_block: list[float] = field(default_factory=list)
    self_buffs: dict[str, float] = field(default_factory=dict)
    enemy_buffs: list[dict[str, float]] = field(default_factory=list)
    hand_size: float = 0.0
    draw_pile_size: float = 0.0
    discard_pile_size: float = 0.0


@dataclass(slots=True)
class HistoryStep:
    state: BattleState
    action: LegalAction
    delta: TransitionDelta
