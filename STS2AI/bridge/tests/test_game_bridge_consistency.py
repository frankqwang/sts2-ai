from __future__ import annotations

import sys
from pathlib import Path

_python_root = Path(__file__).resolve().parents[1]
if str(_python_root) not in sys.path:
    sys.path.insert(0, str(_python_root))

from game_bridge.session.build_spec import BuildSpecPy, normalize_build_spec
from game_bridge.session.state_semantics import (
    is_failure_outcome,
    is_actionable_combat_state,
    is_victory_outcome,
    is_menu_ready_for_v2_reset,
    normalize_run_outcome,
)
from game_bridge.sim.consistency import build_consistency_report


def test_normalize_run_outcome_uses_shared_vocab():
    assert normalize_run_outcome("Win") == "victory"
    assert normalize_run_outcome("death") == "defeat"
    assert normalize_run_outcome("") is None
    assert is_victory_outcome("won") is True
    assert is_failure_outcome("loss") is True


def test_build_spec_normalization_accepts_aliases():
    normalized = normalize_build_spec(
        {
            "cards": [
                "Strike_R",
                {"card_id": "Bash", "upgrades": 1, "props": {"ethereal": False}},
            ],
            "relic_ids": ["BurningBlood"],
            "potion_ids": [
                "FirePotion",
                {"potion_id": "BlockPotion", "slot_index": 2},
            ],
            "hp": 55,
            "max_hp": 80,
            "energy": 4,
            "potion_slot_count": 3,
            "gold": 99,
            "floor": 25,
        }
    )

    assert normalized == {
        "deck": [
            {"id": "Strike_R", "upgrade_level": 0},
            {"id": "Bash", "upgrade_level": 1, "props": {"ethereal": False}},
        ],
        "relics": [{"id": "BurningBlood"}],
        "potions": [
            {"id": "FirePotion", "slot": 0},
            {"id": "BlockPotion", "slot": 2},
        ],
        "current_hp": 55,
        "max_hp": 80,
        "max_energy": 4,
        "max_potion_slots": 3,
        "gold": 99,
        "floor": 25,
    }

    typed = BuildSpecPy.from_dict(
        {
            "deck": ["Strike_R"],
            "relics": ["BurningBlood"],
            "potions": ["FirePotion"],
            "max_potion_slots": 2,
        }
    )
    assert normalize_build_spec(typed) == {
        "deck": [{"id": "Strike_R", "upgrade_level": 0}],
        "relics": [{"id": "BurningBlood"}],
        "potions": [{"id": "FirePotion", "slot": 0}],
        "max_potion_slots": 2,
    }


def test_build_spec_normalization_accepts_nested_case_payload_floor():
    normalized = normalize_build_spec(
        {
            "floor": 25,
            "build": {
                "deck": ["Strike_R"],
                "relics": ["BurningBlood"],
                "current_hp": 55,
            },
        }
    )

    assert normalized == {
        "deck": [{"id": "Strike_R", "upgrade_level": 0}],
        "relics": [{"id": "BurningBlood"}],
        "current_hp": 55,
        "floor": 25,
    }


def test_state_semantics_detects_actionable_combat_and_menu_readiness():
    actionable_state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "player": {
                "hand": [
                    {"id": "Strike_R", "can_play": True},
                ]
            },
        },
    }
    assert is_actionable_combat_state(actionable_state) is True
    assert is_menu_ready_for_v2_reset({"state_type": "menu", "menu": {"is_main_menu_visible": True}}) is True
    assert is_menu_ready_for_v2_reset({"state_type": "menu", "menu": {"is_main_menu_visible": False}}) is False


def test_consistency_report_includes_missing_catalog_opcode_gap():
    report = build_consistency_report(
        state={
            "state_type": "monster",
            "legal_actions": [],
            "battle": {"enemies": [], "hand": []},
        }
    )

    checks = {item["area"]: item for item in report["checks"]}
    assert checks["catalog/proto_pipe"]["status"] == "missing"
    assert report["state_check"]["ok"] is False
    assert any("非可行动" in issue for issue in report["state_check"]["issues"])
