from __future__ import annotations

import unittest

from zero.domain import (
    BattleState,
    EnemyState,
    FightLabel,
    LegalAction,
    PileSummary,
    PlayerState,
    StaticContext,
    compute_fight_score,
    compute_hp_quality_score,
    compute_step_progress_score,
)


def _make_state(*, hp: float, enemy_hp: float, legal_actions: list[LegalAction] | None = None, terminal: bool = False, outcome: str = "") -> BattleState:
    return BattleState(
        player=PlayerState(hp=hp, max_hp=80.0, block=0.0, energy=3.0),
        enemies=[EnemyState(enemy_id="enemy", hp=enemy_hp, max_hp=40.0, block=0.0, intent_id="attack")],
        hand=[],
        piles=PileSummary(),
        context=StaticContext(character_id="IRONCLAD", act=1, floor=2, encounter_class="normal"),
        legal_actions=list(legal_actions or []),
        terminal=terminal,
        run_outcome=outcome,
    )


class ScoringTests(unittest.TestCase):
    def test_hp_quality_is_flat_inside_safe_loss_band(self) -> None:
        safe = FightLabel(
            fight_win=1.0,
            enemy_hp_fraction_dealt=1.0,
            self_hp_fraction_remaining=70.0 / 80.0,
            player_hp=70.0,
            player_max_hp=80.0,
        )
        punished = FightLabel(
            fight_win=1.0,
            enemy_hp_fraction_dealt=1.0,
            self_hp_fraction_remaining=45.0 / 80.0,
            player_hp=45.0,
            player_max_hp=80.0,
        )
        self.assertAlmostEqual(compute_hp_quality_score(safe, encounter_class="normal"), 1.0)
        self.assertLess(compute_hp_quality_score(punished, encounter_class="normal"), 0.8)

    def test_boss_hp_quality_is_more_tolerant_than_normal(self) -> None:
        label = FightLabel(
            fight_win=1.0,
            enemy_hp_fraction_dealt=1.0,
            self_hp_fraction_remaining=8.0 / 80.0,
            player_hp=8.0,
            player_max_hp=80.0,
        )
        self.assertGreater(
            compute_hp_quality_score(label, encounter_class="boss"),
            compute_hp_quality_score(label, encounter_class="normal"),
        )

    def test_fight_score_rewards_faster_victory(self) -> None:
        label = FightLabel(
            fight_win=1.0,
            enemy_hp_fraction_dealt=1.0,
            self_hp_fraction_remaining=68.0 / 80.0,
            player_hp=68.0,
            player_max_hp=80.0,
        )
        fast = compute_fight_score(
            label,
            encounter_class="normal",
            truncated=False,
            no_progress_ratio=0.10,
            max_no_progress_streak=3,
            step_count=10,
        )
        slow = compute_fight_score(
            label,
            encounter_class="normal",
            truncated=False,
            no_progress_ratio=0.10,
            max_no_progress_streak=3,
            step_count=42,
        )
        self.assertGreater(fast, slow)

    def test_step_progress_penalizes_wasted_damage_opportunity(self) -> None:
        strike = LegalAction(
            action_id="play_strike",
            action_type="play_card",
            card_id="STRIKE_IRONCLAD",
            can_execute=True,
            damage_now=6.0,
        )
        defend = LegalAction(
            action_id="play_defend",
            action_type="play_card",
            card_id="DEFEND_IRONCLAD",
            can_execute=True,
            block_now=5.0,
        )
        end_turn = LegalAction(action_id="end_turn", action_type="end_turn", can_execute=True)

        previous = _make_state(hp=80.0, enemy_hp=12.0, legal_actions=[strike, defend, end_turn])
        defend_next = _make_state(hp=80.0, enemy_hp=12.0)
        strike_next = _make_state(hp=80.0, enemy_hp=6.0)

        defend_score = compute_step_progress_score(previous, defend_next, chosen_action=defend)
        strike_score = compute_step_progress_score(previous, strike_next, chosen_action=strike)
        end_turn_score = compute_step_progress_score(previous, defend_next, chosen_action=end_turn)

        self.assertGreater(strike_score, defend_score)
        self.assertLess(end_turn_score, defend_score)


if __name__ == "__main__":
    unittest.main()
