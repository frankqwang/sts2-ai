from __future__ import annotations

"""Build raw online combat decisions into stable training samples.

Design notes:
- `TrainingSample` creation happens exactly once here for the online/base view.
- Pool-specific duplication (`search`, `rare`, `reanalyse`) is handled later by
  admission logic so we never mutate one sample instance after insertion.
- Keep-score and uncertainty target are computed from observable signals here,
  not from the model's own uncertainty head output.
- 这里还会把 step / fight / episode-proxy 三层后验评分写进样本，
  供 sample_weight、keep_score 和 search queue 共用。
"""

from collections import defaultdict, deque

from ..config import EncoderConfig
from ..domain import (
    FightLabel,
    HistoryStep,
    RawTransition,
    SearchLabel,
    TrainingSample,
    compute_episode_score_proxy,
    compute_fight_score,
    compute_hp_quality_score,
    compute_step_progress_score,
)
from ..features import FeatureExtractor, compute_transition_delta


class SampleBuilder:
    def __init__(self, config: EncoderConfig):
        self._history_steps = config.history_steps
        self._history_extractor = FeatureExtractor(config)

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
        fight_stats = _summarize_fight(transitions, final_state)
        fight_label = _build_fight_label(final_state, truncated=fight_stats["truncated"])
        fight_score = compute_fight_score(
            fight_label,
            encounter_class=final_state.context.encounter_class,
            truncated=bool(fight_stats["truncated"]),
            no_progress_ratio=float(fight_stats["no_progress_ratio"]),
            max_no_progress_streak=int(fight_stats["max_no_progress_streak"]),
            step_count=int(fight_stats["step_count"]),
        )
        hp_quality_score = compute_hp_quality_score(
            fight_label,
            encounter_class=final_state.context.encounter_class,
        )
        floor_value = _resolve_floor_value(transitions, final_state)
        episode_score_proxy = compute_episode_score_proxy(
            fight_score=fight_score,
            floor=floor_value,
            encounter_class=final_state.context.encounter_class,
        )

        samples: list[TrainingSample] = []
        history_window: deque[HistoryStep] = deque(maxlen=self._history_steps)
        prefix_action_indices: list[int] = []
        for transition in transitions:
            delta = compute_transition_delta(transition.state, transition.next_state)
            behavior_action_index = _resolve_behavior_index(transition)
            if behavior_action_index is None:
                # 行为动作一旦对不齐，就不能把“第一个合法动作”当成伪标签继续学。
                # 这里直接丢弃，避免静默污染 imitation 信号。
                continue
            step_progress_score = compute_step_progress_score(
                transition.state,
                transition.next_state,
                chosen_action=transition.action,
            )
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
                search_label=_build_search_label(transition),
                bucket_key=_build_bucket_key(transition),
                pool_name=_default_pool_name(transition),
                main_card_id=transition.action.card_id,
                risk_band=_risk_band(transition.state),
                archetype_tags=_archetype_tags(transition),
                rare_cohort_tags=_rare_tags(transition),
                policy_disagreement=float(transition.metadata.get("top2_gap", 0.0) or 0.0),
                step_progress_score=step_progress_score,
                fight_score=fight_score,
                episode_score_proxy=episode_score_proxy,
                sample_weight=1.0,
                keep_score=_compute_keep_score(
                    transition,
                    step_progress_score=step_progress_score,
                    fight_score=fight_score,
                    hp_quality_score=hp_quality_score,
                    episode_score_proxy=episode_score_proxy,
                    fight_timeout=bool(fight_stats["truncated"]),
                    no_progress_ratio=float(fight_stats["no_progress_ratio"]),
                ),
                metadata={
                    **dict(transition.metadata),
                    "behavior_action_id": transition.action.action_id,
                    "uncertainty_target": _compute_uncertainty_target(transition),
                    "step_progress_score": step_progress_score,
                    "fight_score": fight_score,
                    "hp_quality_score": hp_quality_score,
                    "episode_score_proxy": episode_score_proxy,
                    "fight_timeout": bool(fight_stats["truncated"]),
                    "fight_step_count": int(fight_stats["step_count"]),
                    "fight_no_progress_ratio": float(fight_stats["no_progress_ratio"]),
                    "fight_max_no_progress_streak": int(fight_stats["max_no_progress_streak"]),
                    "fight_progress_steps": int(fight_stats["progress_steps"]),
                    "fight_no_progress_steps": int(fight_stats["no_progress_steps"]),
                    "prefix_action_indices": list(prefix_action_indices),
                },
            )
            samples.append(sample)
            history_window.append(
                HistoryStep(
                    state=None,
                    action=None,
                    delta=delta,
                    history_token=self._history_extractor.encode_history_step_token(
                        transition.state,
                        transition.action,
                        delta,
                    ),
                )
            )
            prefix_action_indices.append(behavior_action_index)
        _assign_sample_weights(
            samples,
            fight_timeout=bool(fight_stats["truncated"]),
            no_progress_ratio=float(fight_stats["no_progress_ratio"]),
        )
        return samples


def _build_fight_label(final_state, *, truncated: bool = False) -> FightLabel:
    enemy_max_hp = sum(enemy.max_hp for enemy in final_state.enemies) or 1.0
    enemy_remaining = sum(max(0.0, enemy.hp) for enemy in final_state.enemies)
    enemy_hp_fraction_dealt = max(0.0, min(1.0, 1.0 - enemy_remaining / enemy_max_hp))
    self_hp_fraction_remaining = 0.0
    if final_state.player.max_hp > 0:
        self_hp_fraction_remaining = max(0.0, min(1.0, final_state.player.hp / final_state.player.max_hp))
    if truncated:
        self_hp_fraction_remaining = 0.0
    fight_win = 1.0 if (not truncated and str(final_state.run_outcome).lower() in {"victory", "win"}) else 0.0
    return FightLabel(
        fight_win=fight_win,
        enemy_hp_fraction_dealt=enemy_hp_fraction_dealt,
        self_hp_fraction_remaining=self_hp_fraction_remaining,
        player_hp=max(0.0, float(final_state.player.hp)),
        player_max_hp=max(0.0, float(final_state.player.max_hp)),
    )


def _resolve_behavior_index(transition: RawTransition) -> int | None:
    if 0 <= transition.action_index < len(transition.state.legal_actions):
        return transition.action_index
    for index, action in enumerate(transition.state.legal_actions):
        if action.action_id == transition.action.action_id:
            return index
    return None


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


def _compute_keep_score(
    transition: RawTransition,
    *,
    step_progress_score: float,
    fight_score: float,
    hp_quality_score: float,
    episode_score_proxy: float,
    fight_timeout: bool,
    no_progress_ratio: float,
) -> float:
    disagreement = 1.0 - min(1.0, float(transition.metadata.get("top2_gap", 1.0) or 1.0))
    hardness = 1.0 if transition.state.context.encounter_class in {"elite", "boss"} else 0.25
    rarity = 1.0 if _rare_tags(transition) else 0.0
    near_lethal = 1.0 if _risk_band(transition.state) == "near_lethal" else 0.0
    progress_attention = 1.0 if step_progress_score < 0.0 else min(1.0, step_progress_score)
    timeout_attention = 1.0 if fight_timeout else 0.0
    freshness = 1.0
    return (
        0.22 * disagreement
        + 0.20 * rarity
        + 0.14 * max(hardness, near_lethal)
        + 0.14 * max(0.0, min(1.0, fight_score / 1.5))
        + 0.08 * hp_quality_score
        + 0.10 * max(0.0, min(1.0, episode_score_proxy / 1.5))
        + 0.10 * progress_attention
        + 0.10 * max(timeout_attention, max(0.0, min(1.0, no_progress_ratio)))
        + 0.10 * freshness
        + 0.05 * float(transition.metadata.get("search_budget", 0.0) or 0.0)
    )


def _compute_uncertainty_target(transition: RawTransition) -> float:
    top2_gap = float(transition.metadata.get("top2_gap", 1.0) or 1.0)
    top2_component = 1.0 - max(0.0, min(1.0, top2_gap))
    near_lethal_component = 1.0 if _risk_band(transition.state) == "near_lethal" else 0.0
    elite_component = 1.0 if transition.state.context.encounter_class in {"elite", "boss"} else 0.0
    return min(1.0, 0.5 * top2_component + 0.3 * near_lethal_component + 0.2 * elite_component)


def _summarize_fight(transitions: list[RawTransition], final_state) -> dict[str, float | int | bool]:
    progress_steps = 0
    no_progress_steps = 0
    max_no_progress_streak = 0
    current_no_progress_streak = 0
    for transition in transitions:
        made_progress = bool(transition.metadata.get("made_progress", False))
        if made_progress:
            progress_steps += 1
            current_no_progress_streak = 0
        else:
            no_progress_steps += 1
            current_no_progress_streak += 1
            max_no_progress_streak = max(max_no_progress_streak, current_no_progress_streak)
    step_count = len(transitions)
    truncated = bool(not final_state.terminal and step_count >= 200)
    return {
        "truncated": truncated,
        "step_count": step_count,
        "progress_steps": progress_steps,
        "no_progress_steps": no_progress_steps,
        "no_progress_ratio": (no_progress_steps / max(step_count, 1)),
        "max_no_progress_streak": max_no_progress_streak,
    }


def _resolve_floor_value(transitions: list[RawTransition], final_state) -> int:
    if transitions:
        metadata_floor = transitions[0].state.context.metadata.get("skada_floor")
        if metadata_floor is not None:
            return int(metadata_floor)
        if transitions[0].state.context.floor > 0:
            return int(transitions[0].state.context.floor)
    return int(final_state.context.floor or 0)


def _assign_sample_weights(
    samples: list[TrainingSample],
    *,
    fight_timeout: bool,
    no_progress_ratio: float,
) -> None:
    if not samples:
        return
    ranked_indices = sorted(
        range(len(samples)),
        key=lambda index: (
            samples[index].fight_score,
            samples[index].episode_score_proxy,
            samples[index].step_progress_score,
            -samples[index].step_idx,
        ),
        reverse=True,
    )
    top_count = max(1, round(len(samples) * 0.2)) if len(samples) >= 5 else 1
    low_count = max(1, round(len(samples) * 0.1)) if len(samples) >= 8 else (1 if len(samples) >= 4 else 0)
    for rank, sample_index in enumerate(ranked_indices):
        sample = samples[sample_index]
        base_weight = _base_sample_weight(
            step_progress_score=sample.step_progress_score,
            fight_score=sample.fight_score,
            hp_quality_score=float(sample.metadata.get("hp_quality_score", 0.0) or 0.0),
            episode_score_proxy=sample.episode_score_proxy,
            fight_timeout=fight_timeout,
            no_progress_ratio=no_progress_ratio,
        )
        score_band = "normal"
        band_multiplier = 1.0
        if rank < top_count:
            score_band = "boost"
            band_multiplier = 1.35
        elif low_count > 0 and rank >= len(samples) - low_count:
            score_band = "downweight"
            band_multiplier = 0.35
        sample.sample_weight = max(0.1, min(2.5, base_weight * band_multiplier))
        behavior_ce_scale = 1.0
        if bool(sample.metadata.get("search_collected", False)):
            behavior_ce_scale = 0.0
        if fight_timeout:
            behavior_ce_scale *= 0.5
        if no_progress_ratio >= 0.70:
            behavior_ce_scale *= 0.7
        if sample.fight_score < 0.55:
            behavior_ce_scale *= 0.7
        if score_band == "downweight":
            behavior_ce_scale *= 0.75
        sample.metadata["behavior_ce_scale"] = max(0.0, min(1.5, behavior_ce_scale))
        sample.metadata["sample_weight"] = sample.sample_weight
        sample.metadata["score_band"] = score_band


def _build_search_label(transition: RawTransition) -> SearchLabel | None:
    policy = transition.metadata.get("search_policy")
    if not isinstance(policy, list) or not policy:
        return None
    search_policy = [float(value) for value in policy]
    return SearchLabel(
        policy=search_policy,
        topk_indices=[
            int(value)
            for value in list(transition.metadata.get("search_topk", []))
            if isinstance(value, (int, float))
        ],
        best_action_index=int(transition.metadata.get("search_best_action_index", -1) or -1),
        ranking_margin=max(0.05, float(transition.metadata.get("search_ranking_margin", 0.05) or 0.05)),
        search_value=float(transition.metadata.get("search_value", 0.0) or 0.0),
        search_trace=list(transition.metadata.get("search_trace", []))
        if isinstance(transition.metadata.get("search_trace"), list)
        else [],
        metadata={
            "search": str(transition.metadata.get("search_source", "search_collect") or "search_collect"),
        },
    )


def _base_sample_weight(
    *,
    step_progress_score: float,
    fight_score: float,
    hp_quality_score: float,
    episode_score_proxy: float,
    fight_timeout: bool,
    no_progress_ratio: float,
) -> float:
    step_component = max(0.0, min(1.0, (step_progress_score + 0.2) / 1.2))
    fight_component = max(0.0, min(1.2, fight_score / 1.5))
    hp_component = max(0.0, min(1.0, hp_quality_score))
    episode_component = max(0.0, min(1.2, episode_score_proxy / 1.5))
    timeout_penalty = 0.55 if fight_timeout else 0.0
    no_progress_penalty = 0.35 * max(0.0, min(1.0, no_progress_ratio))
    return (
        0.30
        + 0.62 * fight_component
        + 0.24 * hp_component
        + 0.26 * episode_component
        + 0.22 * step_component
        - timeout_penalty
        - no_progress_penalty
    )
