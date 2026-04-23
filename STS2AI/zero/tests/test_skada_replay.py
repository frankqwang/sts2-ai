from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

_sts2ai_root = Path(__file__).resolve().parents[2]
_bridge_root = _sts2ai_root / "bridge"
if str(_sts2ai_root) not in sys.path:
    sys.path.insert(0, str(_sts2ai_root))
if str(_bridge_root) not in sys.path:
    sys.path.insert(0, str(_bridge_root))

from game_bridge.catalog.sim_catalog import GameCatalog
from data.skada.generate_stratified_case_pool import _unsupported_build_reasons

from zero.domain import (
    BattleState,
    EnemyState,
    FightLabel,
    PileSummary,
    PlayerState,
    StaticContext,
    assess_transition_progress,
)
from zero.replay.skada import (
    OrderedRunCaseEvaluator,
    OrderedRunRuntimeFactory,
    SkadaCaseRuntimeFactory,
    SkadaBuild,
    SkadaCombatCase,
    _build_eval_label,
    default_starter_build,
    _reconstruct_build_before_floor,
    _split_upgrade_suffix,
)


class SkadaReplayTests(unittest.TestCase):
    def _make_case(self, *, floor: int, encounter_id: str, run_id: int = 12) -> SkadaCombatCase:
        return SkadaCombatCase(
            source_path="abc.jsonl",
            source_line=7,
            run_id=run_id,
            seed="seed-x",
            game_version="v0.103.2",
            character_id="IRONCLAD",
            ascension=0,
            player_count=1,
            floor=floor,
            encounter_id=encounter_id,
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

    def test_game_catalog_model_exists_helpers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "game_catalog.sqlite"
            con = sqlite3.connect(str(db_path))
            try:
                con.execute("CREATE TABLE cards (id TEXT PRIMARY KEY)")
                con.execute("CREATE TABLE relics (id TEXT PRIMARY KEY)")
                con.execute("CREATE TABLE potions (id TEXT PRIMARY KEY)")
                con.execute("INSERT INTO cards (id) VALUES ('BASH')")
                con.execute("INSERT INTO relics (id) VALUES ('BURNING_BLOOD')")
                con.execute("INSERT INTO potions (id) VALUES ('FIRE_POTION')")
                con.commit()
            finally:
                con.close()
            catalog = GameCatalog(db_path=db_path)
            self.assertTrue(catalog.card_exists("bash"))
            self.assertTrue(catalog.relic_exists("burning_blood"))
            self.assertTrue(catalog.potion_exists("fire_potion"))
            self.assertFalse(catalog.relic_exists("XUAN_GANG_HUI_GUANG_BI"))

    def test_unsupported_build_reasons_flags_unknown_model_ids(self):
        case = self._make_case(floor=9, encounter_id="CASE_A")
        case.build.relics = [{"id": "BURNING_BLOOD"}, {"id": "EXTRARELICS-ORANGE_PELLETS"}]
        case.build.deck = [{"id": "STRIKE_IRONCLAD"}, {"id": "UNKNOWN_CARD"}]
        with patch("data.skada.generate_stratified_case_pool.GAME_CATALOG") as catalog:
            catalog.relic_exists.side_effect = lambda relic_id: relic_id == "BURNING_BLOOD"
            catalog.card_exists.side_effect = lambda card_id: card_id == "STRIKE_IRONCLAD"
            reasons = _unsupported_build_reasons(case)
        self.assertEqual(
            sorted(reasons),
            [
                "unsupported_card:UNKNOWN_CARD",
                "unsupported_relic:EXTRARELICS-ORANGE_PELLETS",
            ],
        )

    def test_skada_case_round_trip(self):
        case = self._make_case(floor=2, encounter_id="SHRINKER_BEETLE_WEAK")
        restored = SkadaCombatCase.from_dict(case.to_dict())
        self.assertEqual(restored.case_id, case.case_id)
        self.assertEqual(restored.build.deck[0]["id"], "STRIKE_IRONCLAD")

    def test_ordered_run_runtime_factory_resets_after_failure(self):
        cases = [
            self._make_case(floor=2, encounter_id="CASE_A"),
            self._make_case(floor=4, encounter_id="CASE_B"),
        ]
        factory = OrderedRunRuntimeFactory(cases, auto_launch=False)
        self.assertEqual(factory.current_case_id, cases[0].case_id)
        factory.on_episode_end({"outcome": "victory", "truncated": False})
        self.assertEqual(factory.current_case_id, cases[1].case_id)
        factory.on_episode_end({"outcome": "defeat", "truncated": False})
        self.assertEqual(factory.current_case_id, cases[0].case_id)

    def test_skada_case_runtime_factory_ordered_rotates_cases(self):
        cases = [
            self._make_case(floor=2, encounter_id="CASE_A"),
            self._make_case(floor=4, encounter_id="CASE_B"),
        ]
        factory = SkadaCaseRuntimeFactory(cases, mode="ordered", auto_launch=False)
        first = factory()
        second = factory()
        self.assertEqual(first._case.case_id, cases[0].case_id)
        self.assertEqual(second._case.case_id, cases[1].case_id)

    def test_skada_case_runtime_factory_clone_for_port_preserves_mode(self):
        cases = [self._make_case(floor=2, encounter_id="CASE_A")]
        factory = SkadaCaseRuntimeFactory(cases, mode="fixed", port=15527, auto_launch=False)
        clone = factory.clone_for_port(15531)
        runtime = clone()
        self.assertEqual(runtime._case.case_id, cases[0].case_id)
        self.assertEqual(clone._port, 15531)

    def test_ordered_run_evaluator_stops_after_failure(self):
        cases = [
            self._make_case(floor=2, encounter_id="CASE_A"),
            self._make_case(floor=4, encounter_id="CASE_B"),
            self._make_case(floor=6, encounter_id="CASE_C"),
        ]
        evaluator = OrderedRunCaseEvaluator(cases, auto_launch=False, episodes_per_case=2)
        outcomes = {
            cases[0].case_id: [
                {"success": True},
                {"success": False},
            ],
            cases[1].case_id: [
                {"success": False},
            ],
            cases[2].case_id: [],
        }

        def fake_rollout_case_episode(**kwargs):
            case = kwargs["case"]
            queue = outcomes[case.case_id]
            if not queue:
                self.fail(f"unexpected rollout for {case.case_id}")
            result = queue.pop(0)
            return {
                "label": FightLabel(
                    fight_win=1.0 if result["success"] else 0.0,
                    enemy_hp_fraction_dealt=1.0 if result["success"] else 0.2,
                    self_hp_fraction_remaining=0.9 if result["success"] else 0.0,
                ),
                "metrics": {
                    "truncated": False,
                    "progress_steps": 3,
                    "no_progress_steps": 1,
                    "no_progress_ratio": 0.25,
                    "max_no_progress_streak": 1,
                },
                "success": result["success"],
            }

        with patch("zero.replay.skada._rollout_case_episode", side_effect=fake_rollout_case_episode):
            summaries = evaluator.evaluate(policy=object())

        summary_by_floor = {int(item.metadata["floor"]): item for item in summaries}
        self.assertEqual(summary_by_floor[2].metadata["num_episodes"], 2)
        self.assertEqual(summary_by_floor[4].metadata["num_episodes"], 1)
        self.assertEqual(summary_by_floor[6].metadata["num_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
