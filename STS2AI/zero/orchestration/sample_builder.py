from __future__ import annotations

"""Build raw online combat decisions into stable training samples.

Design notes:
- `TrainingSample` creation happens exactly once here for the online/base view.
- Pool-specific duplication (`search`, `rare`, `reanalyse`) is handled later by
  admission logic so we never mutate one sample instance after insertion.
- Keep-score is computed from observable signals here,
  not from any model-side auxiliary head.
- 这里还会把 step / fight / episode-proxy 三层后验评分写进样本，
  供 sample_weight、keep_score 和 search queue 共用。
"""

from collections import defaultdict, deque

from ..config import EncoderConfig
from ..domain import (
    FightLabel,
    HistoryStep,
    RawTransition,
    TrainingSample,
    compute_episode_score_proxy,
    compute_fight_score,
    compute_hp_quality_score,
    compute_step_progress_score,
)
from ..features import FeatureExtractor, compute_transition_delta

_ENGINE_POWER_IDS = {
    "BARRICADE",
    "CORRUPTION",
    "DARK_EMBRACE",
    "DEMON_FORM",
    "EVOLVE",
    "FEEL_NO_PAIN",
    "INFLAME",
    "METALLICIZE",
    "PYRE",
    "RUPTURE",
}
_EXHAUST_ENABLER_IDS = {
    "BURNING_PACT",
    "FIEND_FIRE",
    "PURITY",
    "SECOND_WIND",
    "SEVER_SOUL",
    "TRUE_GRIT",
}
_EXHAUST_PAYOFF_IDS = {
    "DARK_EMBRACE",
    "FEEL_NO_PAIN",
    "PACTS_END",
    "PYRE",
}
_RESOURCE_CARD_IDS = {
    "BLOODLETTING",
    "BURNING_PACT",
    "INFERNAL_BLADE",
    "OFFERING",
    "POMMEL_STRIKE",
    "SHRUG_IT_OFF",
}
_NON_COMMIT_ACTION_TYPES = {
    "end_turn",
    "confirm_selection",
    "cancel_selection",
    "select_hand_card",
    "select_card",
    "select_card_option",
    "combat_select_card",
}
_SUBMENU_CONFIRM_ACTION_TYPES = {
    "confirm_selection",
    "combat_confirm_selection",
}


class SampleBuilder:
    def __init__(self, config: EncoderConfig, *, ppo_gamma: float = 0.99, ppo_gae_lambda: float = 0.95):
        self._history_steps = config.history_steps
        self._history_extractor = FeatureExtractor(config)
        self._ppo_gamma = float(ppo_gamma)
        self._ppo_gae_lambda = float(ppo_gae_lambda)

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
        ppo_targets = _compute_ppo_targets(transitions, gamma=self._ppo_gamma, gae_lambda=self._ppo_gae_lambda)
        turn_targets = _compute_turn_targets(transitions, gamma=self._ppo_gamma)
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
            future_summary_targets = turn_targets.get(transition.step_idx, {}).get(
                "future_targets",
                _compute_future_summary_targets(transition.state, transition.next_state),
            )
            submenu_confirm_target, submenu_has_confirm = _compute_submenu_confirm_target(
                transition.state,
                transition.action,
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
                old_logprob=float((transition.metadata or {}).get("old_logprob", 0.0) or 0.0),
                old_value=float((transition.metadata or {}).get("value_pred", 0.0) or 0.0),
                old_intent_logprob=float((transition.metadata or {}).get("old_intent_logprob", 0.0) or 0.0),
                old_intent_value=float((transition.metadata or {}).get("old_intent_value", 0.0) or 0.0),
                reward=float(getattr(transition, "reward", 0.0) or 0.0),
                ppo_return=float(ppo_targets.get(transition.step_idx, {}).get("return", 0.0)),
                ppo_advantage=float(ppo_targets.get(transition.step_idx, {}).get("advantage", 0.0)),
                turn_id=int(turn_targets.get(transition.step_idx, {}).get("turn_id", _transition_turn_id(transition))),
                turn_start_mask=float(turn_targets.get(transition.step_idx, {}).get("turn_start_mask", 0.0)),
                active_intent=int((transition.metadata or {}).get("active_intent", 0) or 0),
                turn_return=float(turn_targets.get(transition.step_idx, {}).get("turn_return", 0.0)),
                turn_advantage=float(turn_targets.get(transition.step_idx, {}).get("turn_advantage", 0.0)),
                chosen_action_future_targets=list(future_summary_targets),
                submenu_confirm_target=float(submenu_confirm_target),
                submenu_has_confirm=float(submenu_has_confirm),
                metadata={
                    **dict(transition.metadata),
                    "behavior_action_id": transition.action.action_id,
                    "future_summary_targets": list(future_summary_targets),
                    "future_death_risk_2t": future_summary_targets[0],
                    "future_next_turn_power": future_summary_targets[1],
                    "future_setup_value": future_summary_targets[2],
                    "submenu_confirm_target": float(submenu_confirm_target),
                    "submenu_has_confirm": float(submenu_has_confirm),
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


def _compute_ppo_targets(
    transitions: list[RawTransition],
    *,
    gamma: float,
    gae_lambda: float,
) -> dict[int, dict[str, float]]:
    if not transitions:
        return {}
    ordered = sorted(transitions, key=lambda item: item.step_idx)
    targets: dict[int, dict[str, float]] = {}
    gae = 0.0
    for index in range(len(ordered) - 1, -1, -1):
        transition = ordered[index]
        value = float((transition.metadata or {}).get("value_pred", 0.0) or 0.0)
        reward = float(getattr(transition, "reward", 0.0) or 0.0)
        if transition.done:
            next_value = 0.0
            nonterminal = 0.0
        else:
            next_transition = ordered[index + 1] if index + 1 < len(ordered) else None
            next_value = float((next_transition.metadata or {}).get("value_pred", 0.0) or 0.0) if next_transition else 0.0
            nonterminal = 1.0
        delta = reward + gamma * next_value * nonterminal - value
        gae = delta + gamma * gae_lambda * nonterminal * gae
        targets[transition.step_idx] = {
            "advantage": float(gae),
            "return": float(gae + value),
        }
    return targets


def _compute_turn_targets(
    transitions: list[RawTransition],
    *,
    gamma: float,
) -> dict[int, dict[str, float | int | list[float]]]:
    if not transitions:
        return {}
    ordered = sorted(transitions, key=lambda item: item.step_idx)
    turn_groups: dict[int, list[RawTransition]] = defaultdict(list)
    for transition in ordered:
        turn_groups[_transition_turn_id(transition)].append(transition)
    distinct_turns = sorted(turn_groups)
    first_step_for_turn = {turn_id: min(item.step_idx for item in items) for turn_id, items in turn_groups.items()}
    targets: dict[int, dict[str, float | int | list[float]]] = {}
    for turn_index, turn_id in enumerate(distinct_turns):
        items = sorted(turn_groups[turn_id], key=lambda item: item.step_idx)
        discounted_sum = 0.0
        for reward_index, item in enumerate(items):
            reward = float(getattr(item, "reward", 0.0) or 0.0)
            discounted_sum += (gamma**reward_index) * reward
        first_item = items[0]
        old_intent_value = float((first_item.metadata or {}).get("old_intent_value", 0.0) or 0.0)
        bootstrap = 0.0
        if turn_index + 1 < len(distinct_turns):
            next_turn_id = distinct_turns[turn_index + 1]
            next_turn_items = sorted(turn_groups[next_turn_id], key=lambda item: item.step_idx)
            if next_turn_items:
                next_turn_first = next_turn_items[0]
                bootstrap = float((next_turn_first.metadata or {}).get("old_intent_value", 0.0) or 0.0)
        turn_return = float(discounted_sum + (gamma ** max(len(items), 0)) * bootstrap)
        future_targets = _compute_turn_future_targets(ordered, first_step_for_turn, distinct_turns, turn_index, first_item)
        for item in items:
            targets[item.step_idx] = {
                "turn_id": int(turn_id),
                "turn_start_mask": 1.0 if item.step_idx == first_item.step_idx else 0.0,
                "turn_return": turn_return,
                "turn_advantage": float(turn_return - old_intent_value),
                "future_targets": list(future_targets),
            }
    return targets


def _transition_turn_id(transition: RawTransition) -> int:
    metadata = transition.state.context.metadata or {}
    value = metadata.get("turn_id", metadata.get("round_number_raw", 0)) or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _compute_turn_future_targets(
    ordered: list[RawTransition],
    first_step_for_turn: dict[int, int],
    distinct_turns: list[int],
    turn_index: int,
    first_item: RawTransition,
) -> list[float]:
    current_turn = distinct_turns[turn_index]
    future_turn_ids = distinct_turns[turn_index + 1 : turn_index + 3]
    future_window = [item for item in ordered if _transition_turn_id(item) in future_turn_ids]
    min_future_hp_ratio = 1.0
    defeat_within_window = False
    for item in future_window:
        hp_ratio = _safe_ratio(item.next_state.player.hp, item.next_state.player.max_hp)
        min_future_hp_ratio = min(min_future_hp_ratio, hp_ratio)
        if str(item.next_state.run_outcome or "").lower() in {"defeat", "loss"}:
            defeat_within_window = True
            break
    if defeat_within_window:
        death_risk_2t = 1.0
    else:
        death_risk_2t = max(0.0, min(1.0, 1.0 - min_future_hp_ratio))
    next_turn_power = 0.0
    if future_turn_ids:
        next_turn_id = future_turn_ids[0]
        next_turn_first_step = first_step_for_turn.get(next_turn_id)
        if next_turn_first_step is not None:
            next_turn_item = next(item for item in ordered if item.step_idx == next_turn_first_step)
            next_turn_power = _compute_hand_quality(next_turn_item.state, max(1.0, float(next_turn_item.state.player.max_hp)))
    setup_value = _compute_setup_value(first_item.next_state)
    return [
        float(death_risk_2t),
        float(next_turn_power),
        float(setup_value),
    ]


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


def _compute_submenu_confirm_target(state, chosen_action) -> tuple[float, float]:
    metadata = state.context.metadata or {}
    state_type = str(metadata.get("state_type", "") or "")
    has_submenu_state = state_type in {"hand_select", "card_select"}
    has_confirm = any(
        action.action_type in _SUBMENU_CONFIRM_ACTION_TYPES
        for action in state.legal_actions
    )
    if not has_submenu_state or not has_confirm:
        return 0.0, 0.0
    chosen_action_type = str(chosen_action.action_type or "").strip().lower()
    return (1.0 if chosen_action_type in _SUBMENU_CONFIRM_ACTION_TYPES else 0.0), 1.0
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
    )


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


def _compute_future_summary_targets(state, next_state) -> list[float]:
    combat_scale = max(
        1.0,
        float(state.context.metadata.get("combat_start_hp") or 0.0)
        or float(state.player.max_hp)
        or float(state.player.hp)
        or 1.0,
    )
    death_risk_2t = _compute_death_risk_2t(next_state, combat_scale)
    hand_quality = _compute_hand_quality(next_state, combat_scale)
    setup_value = _compute_setup_value(next_state)
    return [
        float(death_risk_2t),
        float(hand_quality),
        float(setup_value),
    ]


def _compute_death_risk_2t(state, combat_scale: float) -> float:
    hp_ratio = max(0.0, float(state.player.hp)) / combat_scale
    block_ratio = max(0.0, float(state.player.block)) / combat_scale
    attacking_enemies = sum(1 for enemy in state.living_enemies if "attack" in str(enemy.intent_id).lower())
    risk = 1.0 - (hp_ratio + 0.35 * block_ratio - 0.12 * float(attacking_enemies))
    return max(0.0, min(1.5, risk))


def _compute_hand_quality(state, combat_scale: float) -> float:
    scores: list[float] = []
    for action in state.legal_actions:
        if not action.can_execute:
            continue
        if str(action.action_type).lower() in _NON_COMMIT_ACTION_TYPES:
            continue
        score = max(0.0, float(action.damage_now))
        score += 0.60 * max(0.0, float(action.block_now))
        score += 0.20 * max(0.0, float(action.magic_now))
        if _normalize_card_id(action.card_id) in _ENGINE_POWER_IDS:
            score += 4.0
        if _normalize_card_id(action.card_id) in _RESOURCE_CARD_IDS:
            score += 2.0
        scores.append(score)
    if not scores:
        return 0.0
    topk = sum(sorted(scores, reverse=True)[:3])
    return max(0.0, min(1.5, topk / max(1.0, combat_scale * 0.75)))


def _compute_setup_value(state) -> float:
    engine_active = _engine_buff_total(state.player.buffs)
    hand_ids = [_normalize_card_id(card.card_id) for card in state.hand]
    enablers = sum(1 for card_id in hand_ids if card_id in _EXHAUST_ENABLER_IDS)
    payoffs = sum(1 for card_id in hand_ids if card_id in _EXHAUST_PAYOFF_IDS)
    resources = sum(1 for card_id in hand_ids if card_id in _RESOURCE_CARD_IDS)
    setup_cards = sum(1 for card_id in hand_ids if card_id in _ENGINE_POWER_IDS)
    value = (
        0.22 * min(engine_active, 3.0)
        + 0.12 * float(setup_cards)
        + 0.10 * float(enablers)
        + 0.08 * float(payoffs)
        + 0.08 * float(resources)
        + 0.04 * min(float(state.piles.exhaust_pile_size), 5.0)
    )
    return max(0.0, min(1.5, value))

def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _engine_buff_total(buffs: dict[str, float]) -> float:
    return (
        float(buffs.get("FEEL_NO_PAIN_POWER", 0.0) or 0.0)
        + float(buffs.get("DARK_EMBRACE_POWER", 0.0) or 0.0)
        + float(buffs.get("PYRE_POWER", 0.0) or 0.0)
    )


def _normalize_card_id(value: str) -> str:
    return str(value or "").upper().replace("+", "").strip()
