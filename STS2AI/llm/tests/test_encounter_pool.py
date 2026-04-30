from __future__ import annotations

import json

from llm.data_pipeline.encounter_pool import (
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
    specs = [
        skada_case_to_spec({
            "case_id": "case_slimes",
            "character_id": "IRONCLAD",
            "floor": 7,
            "encounter_id": "SLIMES_NORMAL",
            "encounter_type": "Normal",
            "won": True,
            "build": {"deck": [{"id": "STRIKE_IRONCLAD"}]},
        }),
        skada_case_to_spec({
            "case_id": "case_chompers",
            "character_id": "IRONCLAD",
            "floor": 2,
            "encounter_id": "CHOMPERS_NORMAL",
            "encounter_type": "Normal",
            "won": True,
            "build": {"deck": [{"id": "BASH"}]},
        }),
        skada_case_to_spec({
            "case_id": "case_boss",
            "character_id": "IRONCLAD",
            "floor": 17,
            "encounter_id": "THE_GUARDIAN_BOSS",
            "encounter_type": "Boss",
            "won": True,
            "build": {"deck": [{"id": "BASH"}]},
        }),
    ]
    floor_7 = filter_encounter_pool(specs, "skada_floor_07")
    mixed = filter_encounter_pool(specs, "CHOMPERS,SLIMES")

    assert floor_7
    assert all(spec.metadata["floor"] == 7 for spec in floor_7)
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
    assert spec.metadata["act"] == 1
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


def test_load_skada_case_pool_diverse_covers_more_encounter_ids(tmp_path) -> None:
    """``diverse`` 模式按 encounter_id 主分桶，case-limit 小也尽量多 cover 不同怪。"""
    path = tmp_path / "cases.jsonl"
    rows = []
    encounter_ids = [
        "SLIMES_WEAK", "TOADPOLES_WEAK", "SEAPUNK_WEAK",
        "CULTISTS_NORMAL", "VINE_SHAMBLER_NORMAL",
        "BYRDONIS_ELITE", "PHANTASMAL_GARDENERS_ELITE",
    ]
    # 每种怪 5 个 case（同 floor 同 build 不同 case_id），故意制造 normal-skewed 分布
    for eid in encounter_ids:
        encounter_type = "Elite" if "ELITE" in eid else "Normal"
        for k in range(5):
            rows.append({
                "case_id": f"{eid}_{k}",
                "character_id": "IRONCLAD",
                "floor": 5,
                "encounter_id": eid,
                "encounter_type": encounter_type,
                "won": True,
                "build": {
                    "deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}],
                    "relics": [{"id": "BURNING_BLOOD"}, {"id": f"RELIC_{k}"}],
                },
            })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    specs = load_skada_case_pool(path, limit=7, sample_seed=42, sample_mode="diverse")

    seen_encounters = {spec.encounter_id for spec in specs}
    # diverse 模式应覆盖全部 7 种 encounter（case-limit 等于 unique encounter 数时是上限）
    assert seen_encounters == set(encounter_ids), f"expected all 7 encounters covered, got {seen_encounters}"


def test_archetype_min_counts_force_coverage(tmp_path) -> None:
    """archetype_min_counts 强制覆盖含特征卡的 case，即使源 pool 里这种 case 是少数。"""
    path = tmp_path / "cases.jsonl"
    rows = []
    # 20 个 strike-only 普通 case + 3 个含 INFLAME (power_build) + 3 个含 THUNDERCLAP (aoe)
    for k in range(20):
        rows.append({
            "case_id": f"normal_{k}", "character_id": "IRONCLAD", "floor": 5,
            "encounter_id": f"NORMAL_ENC_{k % 5}", "encounter_type": "Normal", "won": True,
            "build": {"deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "DEFEND_IRONCLAD"}], "relics": [{"id": "BURNING_BLOOD"}]},
        })
    for k in range(3):
        rows.append({
            "case_id": f"power_{k}", "character_id": "IRONCLAD", "floor": 8,
            "encounter_id": f"PWR_ENC_{k}", "encounter_type": "Normal", "won": True,
            "build": {"deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "INFLAME"}, {"id": "RUPTURE"}], "relics": [{"id": "BURNING_BLOOD"}]},
        })
    for k in range(3):
        rows.append({
            "case_id": f"aoe_{k}", "character_id": "IRONCLAD", "floor": 10,
            "encounter_id": f"AOE_ENC_{k}", "encounter_type": "Normal", "won": True,
            "build": {"deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "THUNDERCLAP"}, {"id": "CLEAVE"}], "relics": [{"id": "BURNING_BLOOD"}]},
        })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    specs = load_skada_case_pool(
        path, limit=10, sample_seed=42, sample_mode="diverse",
        archetype_min_counts={"power_build": 2, "aoe": 2},
    )
    has_power = sum(1 for s in specs if "INFLAME" in {(c.get("id") or "").upper() for c in (s.build.get("deck") or []) if isinstance(c, dict)})
    has_aoe = sum(1 for s in specs if any((c.get("id") or "").upper() in {"THUNDERCLAP", "CLEAVE"} for c in (s.build.get("deck") or []) if isinstance(c, dict)))
    assert has_power >= 2, f"power_build coverage too low: {has_power}/10"
    assert has_aoe >= 2, f"aoe coverage too low: {has_aoe}/10"


def test_parse_archetype_min_count_string_format() -> None:
    from llm.data_pipeline.encounter_pool import _parse_archetype_min_count
    assert _parse_archetype_min_count("multi_hit=2,aoe=3,power_build=1") == {"multi_hit": 2, "aoe": 3, "power_build": 1}
    assert _parse_archetype_min_count("") == {}
    assert _parse_archetype_min_count("garbage") == {}
    assert _parse_archetype_min_count("aoe=invalid,multi_hit=2") == {"multi_hit": 2}


def test_load_skada_case_pool_hold_out_split_is_deterministic(tmp_path) -> None:
    """pool_role=train/eval 切分要 deterministic，跨调用同 seed 始终给同 train/eval 划分。"""
    path = tmp_path / "cases.jsonl"
    rows = []
    for k in range(40):
        rows.append({
            "case_id": f"case_{k}",
            "character_id": "IRONCLAD",
            "floor": 5,
            "encounter_id": "TEST_ENC",
            "encounter_type": "Normal",
            "won": True,
            "build": {"deck": [{"id": "STRIKE_IRONCLAD"}], "relics": [{"id": "BURNING_BLOOD"}]},
        })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    train1 = load_skada_case_pool(path, pool_role="train", hold_out_fraction=0.2, hold_out_seed=42)
    train2 = load_skada_case_pool(path, pool_role="train", hold_out_fraction=0.2, hold_out_seed=42)
    eval1 = load_skada_case_pool(path, pool_role="eval", hold_out_fraction=0.2, hold_out_seed=42)
    eval2 = load_skada_case_pool(path, pool_role="eval", hold_out_fraction=0.2, hold_out_seed=42)

    train_ids1 = sorted(s.metadata.get("case_id") for s in train1)
    train_ids2 = sorted(s.metadata.get("case_id") for s in train2)
    eval_ids1 = sorted(s.metadata.get("case_id") for s in eval1)
    eval_ids2 = sorted(s.metadata.get("case_id") for s in eval2)

    assert train_ids1 == train_ids2, "train pool not deterministic"
    assert eval_ids1 == eval_ids2, "eval pool not deterministic"
    assert set(train_ids1).isdisjoint(set(eval_ids1)), "train and eval overlap"
    assert len(train_ids1) + len(eval_ids1) == 40
    # ratio 应该在 hold_out_fraction 附近（hash 散列，有方差）
    eval_ratio = len(eval_ids1) / 40
    assert 0.05 < eval_ratio < 0.5, f"hold-out ratio off: {eval_ratio}"

    # 不切分 (full) 时返回全集
    full = load_skada_case_pool(path, pool_role="full")
    assert len(full) == 40

    # 不同 hold_out_seed 切分应该不同
    eval_alt = load_skada_case_pool(path, pool_role="eval", hold_out_fraction=0.2, hold_out_seed=99)
    eval_alt_ids = sorted(s.metadata.get("case_id") for s in eval_alt)
    assert eval_alt_ids != eval_ids1, "different seed should give different hold-out split"


def test_load_skada_case_pool_boss_oversample_forces_minimum(tmp_path) -> None:
    """``boss_oversample_ratio=0.2`` 强制 boss 至少占 20%。
    没这个参数，case-floor-max 即使包含 floor 17 也会因 boss 在源池里稀有而被忽略。"""
    path = tmp_path / "cases.jsonl"
    rows = []
    # 30 个 normal + 1 个 boss —— boss 占 ~3%
    for k in range(30):
        rows.append({
            "case_id": f"normal_{k}",
            "character_id": "IRONCLAD",
            "floor": 5,
            "encounter_id": f"NORMAL_ENC_{k % 4}",
            "encounter_type": "Normal",
            "won": True,
            "build": {"deck": [{"id": "STRIKE_IRONCLAD"}], "relics": [{"id": "BURNING_BLOOD"}]},
        })
    rows.append({
        "case_id": "boss_kin",
        "character_id": "IRONCLAD",
        "floor": 17,
        "encounter_id": "THE_KIN_BOSS",
        "encounter_type": "Boss",
        "won": True,
        "build": {"deck": [{"id": "BASH"}], "relics": [{"id": "BURNING_BLOOD"}]},
    })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    specs = load_skada_case_pool(
        path,
        floor_max=20,  # 让 boss 进 pool
        limit=5,
        sample_seed=11,
        sample_mode="diverse",
        boss_oversample_ratio=0.2,
    )
    bosses = [s for s in specs if str(s.metadata.get("encounter_type", "")).lower() == "boss"]
    assert len(bosses) >= 1, f"expected at least 1 boss in 5-sample with ratio=0.2, got {len(bosses)}"


def test_load_skada_case_pool_elite_oversample_forces_minimum(tmp_path) -> None:
    """``elite_oversample_ratio=0.5`` 强制 elite 占比 ≥50%，即使源数据 elite 极少。"""
    path = tmp_path / "cases.jsonl"
    rows = []
    # 18 个 normal + 2 个 elite —— elite 比例只 10%
    for k in range(18):
        rows.append({
            "case_id": f"normal_{k}",
            "character_id": "IRONCLAD",
            "floor": 5,
            "encounter_id": f"NORMAL_ENC_{k % 3}",
            "encounter_type": "Normal",
            "won": True,
            "build": {"deck": [{"id": "STRIKE_IRONCLAD"}], "relics": [{"id": "BURNING_BLOOD"}]},
        })
    for k in range(2):
        rows.append({
            "case_id": f"elite_{k}",
            "character_id": "IRONCLAD",
            "floor": 7,
            "encounter_id": f"ELITE_ENC_{k}",
            "encounter_type": "Elite",
            "won": True,
            "build": {"deck": [{"id": "BASH"}], "relics": [{"id": "BURNING_BLOOD"}]},
        })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    specs = load_skada_case_pool(
        path,
        limit=4,
        sample_seed=7,
        sample_mode="diverse",
        elite_oversample_ratio=0.5,
    )
    elites = [s for s in specs if str(s.metadata.get("encounter_type", "")).lower() == "elite"]
    assert len(elites) >= 2, f"expected at least 2 elites in 4-sample with ratio=0.5, got {len(elites)}"
