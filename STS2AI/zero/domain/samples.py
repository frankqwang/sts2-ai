from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace

from .battle import (
    BattleState,
    EnemyState,
    HandCardState,
    HistoryStep,
    LegalAction,
    PileSummary,
    PlayerState,
    StaticContext,
    TargetSummary,
    TransitionDelta,
)
from .labels import FightLabel, SearchLabel


@dataclass(slots=True)
class RawTransition:
    """One on-policy combat decision as collected from the runtime.

    Important invariants:
    - `action_index` is only meaningful relative to `state.legal_actions`.
    - `action` is the concrete chosen action snapshot at collection time.
    - We keep both because the trainer needs a categorical behavior label
      (`action_index`) and the sample pipeline also needs the chosen action's
      explicit semantics (`action_id`, card/target payload, etc.).
    """

    run_id: str
    fight_id: str
    step_idx: int
    seed: str
    action_index: int
    state: BattleState
    action: LegalAction
    next_state: BattleState
    done: bool
    fight_outcome: str
    run_outcome: str
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return _serialize_without_raw(self)

    @property
    def action_id(self) -> str:
        return self.action.action_id


@dataclass(slots=True)
class TrainingSample:
    """One immutable-ish training example snapshot.

    Important invariant:
    - A pool entry must own its own `TrainingSample` instance.
    - If the same logical decision point is admitted to multiple pools
      (`recent_online`, `search`, `rare`), callers must clone via
      `clone_for_pool()` instead of mutating and reusing the same object.
    """

    sample_id: str
    run_id: str
    fight_id: str
    step_idx: int
    state: BattleState
    history: list[HistoryStep]
    legal_actions: list[LegalAction]
    # Index into `legal_actions`, used as the categorical behavior label.
    behavior_action_index: int
    delta: TransitionDelta
    fight_label: FightLabel
    search_label: SearchLabel | None = None
    # Stable action identity carried alongside the categorical index so callers
    # do not need to mentally reconstruct "which action was chosen".
    behavior_action_id: str = ""
    bucket_key: str = ""
    pool_name: str = "recent_online"
    main_card_id: str = ""
    risk_band: str = "normal"
    archetype_tags: list[str] = field(default_factory=list)
    rare_cohort_tags: list[str] = field(default_factory=list)
    policy_disagreement: float = 0.0
    search_budget: float = 0.0
    step_progress_score: float = 0.0
    fight_score: float = 0.0
    episode_score_proxy: float = 0.0
    sample_weight: float = 1.0
    keep_score: float = 0.0
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return _serialize_without_raw(self)

    @property
    def behavior_action(self) -> LegalAction:
        if 0 <= self.behavior_action_index < len(self.legal_actions):
            return self.legal_actions[self.behavior_action_index]
        if self.legal_actions:
            return self.legal_actions[0]
        raise IndexError("TrainingSample has no legal_actions to resolve behavior_action.")

    def clone_for_pool(self, *, pool_name: str, keep_score: float | None = None, metadata: dict[str, str | float | int | bool] | None = None, search_label: SearchLabel | None = None) -> "TrainingSample":
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return replace(
            self,
            pool_name=pool_name,
            keep_score=self.keep_score if keep_score is None else keep_score,
            metadata=merged_metadata,
            search_label=self.search_label if search_label is None else search_label,
        )


@dataclass(slots=True)
class SearchRequest:
    request_id: str
    sample: TrainingSample
    priority: float
    reason_tags: list[str] = field(default_factory=list)


def compact_legal_action(action: LegalAction) -> LegalAction:
    """丢弃 raw，保留训练、评估和复盘真正会用到的动作快照。"""
    target_summary = None
    if action.target_summary is not None:
        target_summary = TargetSummary(
            hp=action.target_summary.hp,
            max_hp=action.target_summary.max_hp,
            block=action.target_summary.block,
            intent_id=action.target_summary.intent_id,
            alive=action.target_summary.alive,
            buffs=dict(action.target_summary.buffs),
        )
    return LegalAction(
        action_id=action.action_id,
        action_type=action.action_type,
        can_execute=action.can_execute,
        card_id=action.card_id,
        potion_id=action.potion_id,
        special_id=action.special_id,
        target_id=action.target_id,
        cost_now=action.cost_now,
        damage_now=action.damage_now,
        block_now=action.block_now,
        magic_now=action.magic_now,
        tags=list(action.tags),
        target_summary=target_summary,
        raw={},
    )


def compact_battle_state(state: BattleState) -> BattleState:
    """把运行期 battle state 压成轻量副本，避免完整 bridge JSON 常驻内存。"""
    return BattleState(
        player=PlayerState(
            hp=state.player.hp,
            max_hp=state.player.max_hp,
            block=state.player.block,
            energy=state.player.energy,
            potions=list(state.player.potions),
            buffs=dict(state.player.buffs),
            resources=dict(state.player.resources),
        ),
        enemies=[
            EnemyState(
                enemy_id=enemy.enemy_id,
                hp=enemy.hp,
                max_hp=enemy.max_hp,
                block=enemy.block,
                intent_id=enemy.intent_id,
                alive=enemy.alive,
                buffs=dict(enemy.buffs),
                tags=list(enemy.tags),
            )
            for enemy in state.enemies
        ],
        hand=[
            HandCardState(
                card_id=card.card_id,
                cost_now=card.cost_now,
                damage_now=card.damage_now,
                block_now=card.block_now,
                magic_now=card.magic_now,
                is_upgraded=card.is_upgraded,
                retain=card.retain,
                exhaust=card.exhaust,
                ethereal=card.ethereal,
                tags=list(card.tags),
            )
            for card in state.hand
        ],
        piles=PileSummary(
            draw_pile_size=state.piles.draw_pile_size,
            discard_pile_size=state.piles.discard_pile_size,
            exhaust_pile_size=state.piles.exhaust_pile_size,
            attack_count=state.piles.attack_count,
            skill_count=state.piles.skill_count,
            power_count=state.piles.power_count,
            key_card_counts=dict(state.piles.key_card_counts),
            archetype_stats=dict(state.piles.archetype_stats),
        ),
        context=StaticContext(
            character_id=state.context.character_id,
            act=state.context.act,
            floor=state.context.floor,
            encounter_class=state.context.encounter_class,
            encounter_id=state.context.encounter_id,
            relics=list(state.context.relics),
            fixed_powers=list(state.context.fixed_powers),
            metadata=dict(state.context.metadata),
        ),
        legal_actions=[compact_legal_action(action) for action in state.legal_actions],
        terminal=state.terminal,
        run_outcome=state.run_outcome,
        raw={},
    )


def compact_raw_transition(transition: RawTransition) -> RawTransition:
    """压缩收集后的 transition，减少 transitions 列表与样本池的常驻内存。"""
    return RawTransition(
        run_id=transition.run_id,
        fight_id=transition.fight_id,
        step_idx=transition.step_idx,
        seed=transition.seed,
        action_index=transition.action_index,
        state=compact_battle_state(transition.state),
        action=compact_legal_action(transition.action),
        next_state=compact_battle_state(transition.next_state),
        done=transition.done,
        fight_outcome=transition.fight_outcome,
        run_outcome=transition.run_outcome,
        metadata=dict(transition.metadata),
    )


def _serialize_without_raw(value):
    """导出训练产物时显式跳过 runtime raw 负载，避免深展开冗余。"""
    if is_dataclass(value):
        result = {}
        for item in fields(value):
            if item.name == "raw":
                continue
            result[item.name] = _serialize_without_raw(getattr(value, item.name))
        return result
    if isinstance(value, list):
        return [_serialize_without_raw(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_without_raw(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_without_raw(item) for key, item in value.items()}
    return value
