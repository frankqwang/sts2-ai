from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from zero.domain import (
    BattleState,
    EnemyState,
    FightLabel,
    LegalAction,
    PileSummary,
    PlayerState,
    StaticContext,
    TeacherRequest,
    TrainingSample,
    TransitionDelta,
    assess_transition_progress,
)
from zero.replay.skada import (
    MultiCaseAggregateTeacher,
    SkadaBuild,
    SkadaCombatCase,
    _build_eval_label,
    default_starter_build,
    _reconstruct_build_before_floor,
    _split_upgrade_suffix,
)


class SkadaReplayTests(unittest.TestCase):
    def test_assess_transition_progress_marks_enemy_hp_drop(self):
        previous = BattleState(
            player=PlayerState(hp=80.0, max_hp=80.0, block=0.0, energy=3.0),
            enemies=[EnemyState(enemy_id="enemy", hp=20.0, max_hp=20.0, block=0.0, intent_id="attack")],
            hand=[],
            piles=PileSummary(),
            context=StaticContext(character_id="IRONCLAD"),
            legal_actions=[],
        )
        current = BattleState(
            player=PlayerState(hp=78.0, max_hp=80.0, block=0.0, energy=2.0),
            enemies=[EnemyState(enemy_id="enemy", hp=14.0, max_hp=20.0, block=0.0, intent_id="attack")],
            hand=[],
            piles=PileSummary(),
            context=StaticContext(character_id="IRONCLAD"),
            legal_actions=[],
        )
        progress = assess_transition_progress(previous, current)
        self.assertTrue(progress.made_progress)
        self.assertAlmostEqual(progress.enemy_hp_delta, 6.0)

    def test_build_eval_label_timeout_counts_as_failure(self):
        state = BattleState(
            player=PlayerState(hp=60.0, max_hp=80.0, block=0.0, energy=3.0),
            enemies=[EnemyState(enemy_id="enemy", hp=10.0, max_hp=40.0, block=0.0, intent_id="attack")],
            hand=[],
            piles=PileSummary(),
            context=StaticContext(character_id="IRONCLAD"),
            legal_actions=[],
            terminal=False,
            run_outcome="",
        )
        label = _build_eval_label(state, truncated=True)
        self.assertEqual(label.fight_win, 0.0)
        self.assertEqual(label.self_hp_fraction_remaining, 0.0)
        self.assertGreater(label.enemy_hp_fraction_dealt, 0.0)

    def test_split_upgrade_suffix(self):
        self.assertEqual(_split_upgrade_suffix("SHRUG_IT_OFF++"), ("SHRUG_IT_OFF", 2))
        self.assertEqual(_split_upgrade_suffix("armaments"), ("ARMAMENTS", 0))

    def test_reconstruct_build_before_floor_ignores_post_combat_rewards(self):
        starter = SkadaBuild(
            deck=[
                {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
                {"id": "DEFEND_IRONCLAD", "upgrade_level": 0},
            ],
            relics=[{"id": "BURNING_BLOOD"}],
            current_hp=80,
            max_hp=80,
            gold=99,
        )
        timeline = [
            {
                "floor": 1,
                "relic_choices": [{"relic_id": "ARCANE_SCROLL", "was_picked": True}],
                "shop_actions": [{"action_type": "remove", "item_id": "STRIKE_IRONCLAD"}],
            },
            {
                "floor": 2,
                "card_choices": [{"card_id": "ARMAMENTS", "was_picked": True}],
                "hp_before": 73,
                "gold_before": 99,
            },
        ]
        build = _reconstruct_build_before_floor(
            starter_build=starter,
            floor_timeline=timeline,
            combat_floor=2,
            fallback_hp=73,
            fallback_gold=99,
        )
        self.assertEqual([card["id"] for card in build.deck], ["DEFEND_IRONCLAD"])
        self.assertEqual([relic["id"] for relic in build.relics], ["BURNING_BLOOD", "ARCANE_SCROLL"])
        self.assertEqual(build.current_hp, 73)
        self.assertEqual(build.gold, 99)

    def test_default_starter_template_for_ironclad(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "game_catalog.sqlite"
            con = sqlite3.connect(str(db_path))
            try:
                con.execute(
                    "CREATE TABLE characters (id TEXT PRIMARY KEY, starting_deck_json TEXT, starting_relics_json TEXT, starting_potions_json TEXT, starting_hp INTEGER, starting_gold INTEGER, max_energy INTEGER)"
                )
                con.execute(
                    "INSERT INTO characters (id, starting_deck_json, starting_relics_json, starting_potions_json, starting_hp, starting_gold, max_energy) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        "IRONCLAD",
                        '["STRIKE_IRONCLAD","STRIKE_IRONCLAD","STRIKE_IRONCLAD","STRIKE_IRONCLAD","STRIKE_IRONCLAD","DEFEND_IRONCLAD","DEFEND_IRONCLAD","DEFEND_IRONCLAD","DEFEND_IRONCLAD","BASH"]',
                        '["BURNING_BLOOD"]',
                        "[]",
                        80,
                        99,
                        3,
                    ),
                )
                con.commit()
            finally:
                con.close()
            build = default_starter_build("IRONCLAD", db_path=db_path)
        self.assertEqual(len(build.deck), 10)
        self.assertEqual(build.deck[-1]["id"], "BASH")
        self.assertEqual(build.relics, [{"id": "BURNING_BLOOD"}])

    def test_skada_case_round_trip(self):
        case = SkadaCombatCase(
            source_path="abc.jsonl",
            source_line=7,
            run_id=12,
            seed="seed-x",
            game_version="v0.103.2",
            character_id="IRONCLAD",
            ascension=0,
            player_count=1,
            floor=2,
            encounter_id="SHRINKER_BEETLE_WEAK",
            encounter_type="Normal",
            won=True,
            build=SkadaBuild(
                deck=[{"id": "STRIKE_IRONCLAD", "upgrade_level": 0}],
                relics=[{"id": "BURNING_BLOOD"}],
                current_hp=80,
                max_hp=80,
                max_energy=3,
                gold=99,
            ),
        )
        restored = SkadaCombatCase.from_dict(case.to_dict())
        self.assertEqual(restored.case_id, case.case_id)
        self.assertEqual(restored.build.deck[0]["id"], "STRIKE_IRONCLAD")

    def test_multi_case_teacher_routes_by_case_id(self):
        case = SkadaCombatCase(
            source_path="abc.jsonl",
            source_line=7,
            run_id=12,
            seed="seed-x",
            game_version="v0.103.2",
            character_id="IRONCLAD",
            ascension=0,
            player_count=1,
            floor=2,
            encounter_id="SHRINKER_BEETLE_WEAK",
            encounter_type="Normal",
            won=True,
            build=SkadaBuild(
                deck=[{"id": "STRIKE_IRONCLAD", "upgrade_level": 0}],
                relics=[{"id": "BURNING_BLOOD"}],
                current_hp=80,
                max_hp=80,
                max_energy=3,
                gold=99,
            ),
            card_usage={"STRIKE_IRONCLAD": {"plays": 3, "damage": 9.0, "block": 0.0, "energy": 3.0}},
        )
        teacher = MultiCaseAggregateTeacher([case])
        sample = TrainingSample(
            sample_id="sample1",
            run_id="run1",
            fight_id="fight1",
            step_idx=0,
            state=BattleState(
                player=PlayerState(hp=70.0, max_hp=80.0, block=0.0, energy=3.0),
                enemies=[EnemyState(enemy_id="enemy", hp=20.0, max_hp=20.0, block=0.0, intent_id="attack")],
                hand=[],
                piles=PileSummary(),
                context=StaticContext(
                    character_id="IRONCLAD",
                    encounter_id="SHRINKER_BEETLE_WEAK",
                    metadata={"skada_case_id": case.case_id},
                ),
                legal_actions=[
                    LegalAction(action_id="play_strike", action_type="play_card", card_id="STRIKE_IRONCLAD"),
                    LegalAction(action_id="end_turn", action_type="end_turn"),
                ],
            ),
            history=[],
            legal_actions=[
                LegalAction(action_id="play_strike", action_type="play_card", card_id="STRIKE_IRONCLAD"),
                LegalAction(action_id="end_turn", action_type="end_turn"),
            ],
            behavior_action_index=0,
            delta=TransitionDelta(),
            fight_label=FightLabel(fight_win=1.0, enemy_hp_fraction_dealt=1.0, self_hp_fraction_remaining=1.0),
        )
        request = TeacherRequest(request_id="req1", sample=sample, priority=1.0)
        label = teacher.label_request(request)
        self.assertEqual(label.best_action_index, 0)


if __name__ == "__main__":
    unittest.main()
