from __future__ import annotations

"""Build raw online combat decisions into stable training samples.

Design notes:
- `TrainingSample` creation happens exactly once here for the online/base view.
- Pool-specific duplication (`teacher`, `rare`, `reanalyse`) is handled later by
  admission logic so we never mutate one sample instance after insertion.
- Keep-score and uncertainty target are computed from observable signals here,
  not from the model's own uncertainty head output.
"""

from collections import defaultdict, deque

from ..config import EncoderConfig
from ..domain import FightLabel, HistoryStep, RawTransition, TrainingSample
from ..features import compute_transition_delta


class SampleBuilder:
    def __init__(self, config: EncoderConfig):
        self._history_steps = config.history_steps

    def build(self, transitions: list[RawTransition]) -> list[TrainingSample]:
        by_fight: dict[str, list[RawTransition]] = defaultdict(list)
        for transition in transitions:
            if not transition.state.legal_actions:
                continue
            by_fight[transition.fight_id].append(transition)

        samples: list[TrainingSample] = []
        for fight_transitions in by_fight.values():
            samples.extend(self._build_fight_samples(fight_transitions))
        return samples

    def _build_fight_samples(self, transitions: list[RawTransition]) -> list[TrainingSample]:
        if not transitions:
            return []
        transitions = sorted(transitions, key=lambda item: item.step_idx)
        final_state = transitions[-1].next_state
        fight_label = _build_fight_label(final_state)

        samples: list[TrainingSample] = []
        history_window: deque[HistoryStep] = deque(maxlen=self._history_steps)
        for transition in transitions:
            delta = compute_transition_delta(transition.state, transition.next_state)
            behavior_action_index = _resolve_behavior_index(transition)
            sample = TrainingSample(
                sample_id=f"{transition.fight_id}:{transition.step_idx}",
                run_id=transition.run_id,
                fight_id=transition.fight_id,
                step_idx=transition.step_idx,
                state=transition.state,
                history=list(history_window),
                legal_actions=transition.state.legal_actions,
                behavior_action_index=behavior_action_index,
                behavior_action_id=transition.action.action_id,
                delta=delta,
                fight_label=fight_label,
                bucket_key=_build_bucket_key(transition),
                pool_name=_default_pool_name(transition),
                main_card_id=transition.action.card_id,
                risk_band=_risk_band(transition.state),
                archetype_tags=_archetype_tags(transition),
                rare_cohort_tags=_rare_tags(transition),
                student_disagreement=float(transition.metadata.get("top2_gap", 0.0) or 0.0),
                keep_score=_compute_keep_score(transition),
                metadata={
                    **dict(transition.metadata),
                    "behavior_action_id": transition.action.action_id,
                    "uncertainty_target": _compute_uncertainty_target(transition),
                },
            )
            samples.append(sample)
            history_window.append(HistoryStep(state=transition.state, action=transition.action, delta=delta))
        return samples


def _build_fight_label(final_state) -> FightLabel:
    enemy_max_hp = sum(enemy.max_hp for enemy in final_state.enemies) or 1.0
    enemy_remaining = sum(max(0.0, enemy.hp) for enemy in final_state.enemies)
    enemy_hp_fraction_dealt = max(0.0, min(1.0, 1.0 - enemy_remaining / enemy_max_hp))
    self_hp_fraction_remaining = 0.0
    if final_state.player.max_hp > 0:
        self_hp_fraction_remaining = max(0.0, min(1.0, final_state.player.hp / final_state.player.max_hp))
    fight_win = 1.0 if str(final_state.run_outcome).lower() in {"victory", "win"} else 0.0
    return FightLabel(
        fight_win=fight_win,
        enemy_hp_fraction_dealt=enemy_hp_fraction_dealt,
        self_hp_fraction_remaining=self_hp_fraction_remaining,
    )


def _resolve_behavior_index(transition: RawTransition) -> int:
    if 0 <= transition.action_index < len(transition.state.legal_actions):
        return transition.action_index
    for index, action in enumerate(transition.state.legal_actions):
        if action.action_id == transition.action.action_id:
            return index
    return 0


def _build_bucket_key(transition: RawTransition) -> str:
    act = transition.state.context.act
    floor = transition.state.context.floor
    stage = "early" if floor < 8 else "mid" if floor < 17 else "late"
    maturity = "formed" if floor >= 10 else "base"
    return f"combat|A{act}_{stage}|{transition.state.context.encounter_class}|{maturity}"


def _default_pool_name(transition: RawTransition) -> str:
    return "recent_online"


def _risk_band(state) -> str:
    if state.player.max_hp <= 0:
        return "normal"
    ratio = state.player.hp / state.player.max_hp
    if ratio <= 0.2:
        return "near_lethal"
    if ratio <= 0.4:
        return "risky"
    return "normal"


def _rare_tags(transition: RawTransition) -> list[str]:
    tags = []
    if transition.state.context.encounter_class == "elite":
        tags.append("elite")
    if transition.state.context.encounter_class == "boss":
        tags.append("boss")
    if _risk_band(transition.state) == "near_lethal":
        tags.append("near_lethal")
    if bool(transition.metadata.get("rare_build", False)):
        tags.append("rare_build")
    return tags


def _archetype_tags(transition: RawTransition) -> list[str]:
    tags = []
    if transition.state.player.buffs:
        tags.append("buffed")
    attack_count = sum(1 for card in transition.state.hand if "attack" in {tag.lower() for tag in card.tags})
    skill_count = sum(1 for card in transition.state.hand if "skill" in {tag.lower() for tag in card.tags})
    power_count = sum(1 for card in transition.state.hand if "power" in {tag.lower() for tag in card.tags})
    if attack_count >= 3:
        tags.append("attack_dense")
    if skill_count >= 3:
        tags.append("skill_dense")
    if power_count > 0:
        tags.append("power_in_hand")
    return tags


def _compute_keep_score(transition: RawTransition) -> float:
    disagreement = 1.0 - min(1.0, float(transition.metadata.get("top2_gap", 1.0) or 1.0))
    hardness = 1.0 if transition.state.context.encounter_class in {"elite", "boss"} else 0.25
    rarity = 1.0 if _rare_tags(transition) else 0.0
    near_lethal = 1.0 if _risk_band(transition.state) == "near_lethal" else 0.0
    freshness = 1.0
    return (
        0.40 * disagreement
        + 0.20 * rarity
        + 0.20 * max(hardness, near_lethal)
        + 0.10 * freshness
        + 0.10 * float(transition.metadata.get("teacher_budget", 0.0) or 0.0)
    )


def _compute_uncertainty_target(transition: RawTransition) -> float:
    top2_gap = float(transition.metadata.get("top2_gap", 1.0) or 1.0)
    top2_component = 1.0 - max(0.0, min(1.0, top2_gap))
    near_lethal_component = 1.0 if _risk_band(transition.state) == "near_lethal" else 0.0
    elite_component = 1.0 if transition.state.context.encounter_class in {"elite", "boss"} else 0.0
    return min(1.0, 0.5 * top2_component + 0.3 * near_lethal_component + 0.2 * elite_component)
