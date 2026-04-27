from __future__ import annotations

import json

from llm.data_pipeline.encounter_pool import (
    ACT1_WINNABLE_POOL,
    EncounterSpec,
    encounter_key,
    filter_encounter_pool,
    load_skada_case_pool,
    skada_case_to_spec,
)


def test_encounter_key_distinguishes_same_enemy_different_builds() -> None:
    starter = EncounterSpec("SLIMES_NORMAL", {"deck": [{"id": "STRIKE_IRONCLAD"}]}, "starter")
    midrun = EncounterSpec("SLIMES_NORMAL", {"deck": [{"id": "BLUDGEON", "upgrade_level": 1}]}, "midrun")

    assert encounter_key(starter) != encounter_key(midrun)


def test_filter_encounter_pool_matches_tags_and_comma_tokens() -> None:
    midrun = filter_encounter_pool(ACT1_WINNABLE_POOL, "act1_midrun")
    mixed = filter_encounter_pool(ACT1_WINNABLE_POOL, "CHOMPERS,SLIMES")

    assert midrun
    assert all(spec.tag == "act1_midrun" for spec in midrun)
    assert {spec.encounter_id for spec in mixed} <= {"CHOMPERS_NORMAL", "SLIMES_NORMAL"}


def test_skada_case_to_spec_preserves_floor_and_metadata() -> None:
    spec = skada_case_to_spec({
        "case_id": "run_1_floor_7_slimes",
        "character_id": "IRONCLAD",
        "floor": 7,
        "encounter_id": "SLIMES_NORMAL",
        "encounter_type": "Normal",
        "won": True,
        "build": {
            "deck": [{"id": "STRIKE_IRONCLAD"}],
            "relics": [{"id": "BURNING_BLOOD"}],
            "current_hp": 72,
        },
        "floor_state": {"hp_before": 72, "hp_after": 68},
        "metadata": {"combat_turns": 2},
    })

    assert spec.encounter_id == "SLIMES_NORMAL"
    assert spec.build["floor"] == 7
    assert spec.tag == "skada_floor_07_normal"
    assert spec.metadata["case_id"] == "run_1_floor_7_slimes"
    assert spec.metadata["floor_state"]["hp_after"] == 68


def test_load_skada_case_pool_stratifies_limit(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    rows = []
    for idx in range(12):
        floor = 2 if idx < 6 else 7
        encounter_type = "Normal" if idx % 2 == 0 else "Elite"
        rows.append({
            "case_id": f"case_{idx}",
            "character_id": "IRONCLAD",
            "floor": floor,
            "encounter_id": "SLIMES_NORMAL",
            "encounter_type": encounter_type,
            "won": True,
            "build": {"deck": [{"id": "STRIKE_IRONCLAD"}]},
        })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    specs = load_skada_case_pool(path, limit=4, sample_seed=123, sample_mode="stratified")

    assert len(specs) == 4
    assert {spec.metadata["floor"] for spec in specs} == {2, 7}
    assert {spec.metadata["encounter_type"] for spec in specs} == {"Normal", "Elite"}
