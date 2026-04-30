from __future__ import annotations

import sys
from pathlib import Path

_python_root = Path(__file__).resolve().parents[1]
if str(_python_root) not in sys.path:
    sys.path.insert(0, str(_python_root))

from game_bridge.session.build_spec import BuildSpecPy, normalize_build_spec
from game_bridge.session.state_semantics import (
    is_combat_state,
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


def test_card_select_state_is_treated_as_in_combat():
    """Regression guard for the THE_KIN/LAGAVULIN ``left_combat`` bug.

    HEADBUTT / ARMAMENTS / EXHUME / DUAL_WIELD trigger sim's pile-pick UI
    which sets ``state_type="card_select"`` and *clears* the top-level
    ``enemies`` list. Before the fix, ``is_combat_state`` returned False on
    this state and the rollout main loop bailed out as ``left_combat``,
    silently truncating any episode that played one of these cards.

    Sim is *not* in fact ending combat — the player just owes the sim a
    selection action. So both ``is_combat_state`` and
    ``is_actionable_combat_state`` must report True for these screens.
    """
    headbutt_card_select = {
        "state_type": "card_select",
        # sim 给的 enemies 字段为空（pile-pick UI 不渲染敌人）— 正是这点
        # 之前让 RolloutPolicy 误以为战斗已结束。
        "enemies": [],
        "card_selection": {
            "trigger_card_id": "HEADBUTT",
            "source_pile_type": "discard_pile",
            "selectable_cards": [
                {"id": "Strike_R", "is_enabled": True},
            ],
            "min_select": 1,
            "max_select": 1,
        },
    }
    assert is_combat_state(headbutt_card_select) is True
    assert is_actionable_combat_state(headbutt_card_select) is True

    # An out-of-combat card_select (e.g. card_reward / event card removal)
    # would not carry a card_selection dict referring to an in-fight pile;
    # but bridge already routes those through their own state_type paths
    # (``card_reward``, ``event``), so by the time we see ``card_select`` we
    # are by construction in a combat-derived screen — keep it actionable.
    bare_card_select = {"state_type": "card_select"}
    assert is_combat_state(bare_card_select) is True
    assert is_actionable_combat_state(bare_card_select) is True


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
