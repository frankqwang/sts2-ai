from __future__ import annotations

import json

from llm.scripts.manage_dataset_pool import main as pool_main


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _row(action_index: int, *, source_kind: str = "self_rollout", advantage: float = 1.0, flags=None):
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    "run: char=IRONCLAD act=1 floor=3 round=1\n"
                    "deck: STRIKE_IRONCLADx4, DEFEND_IRONCLADx4, BASHx1\n"
                    "enemies:\n"
                    "  enemy1: CULTIST hp=30/30 block=0 intent=Attack(6) powers=-\n"
                    "legal_actions:\n"
                    f"  [{action_index}] STRIKE_IRONCLAD hand[0] -> enemy1\n"
                ),
            },
            {"role": "assistant", "content": json.dumps({"action_index": action_index, "confidence": 0.8, "reason": "deal damage"})},
        ],
        "meta": {
            "source_kind": source_kind,
            "outcome": "victory",
            "advantage": advantage,
            "encounter_id": "CULTISTS_NORMAL",
            "encounter_tag": "skada_floor_03_normal",
            "action_quality_flags": flags or [],
        },
    }


def test_dataset_pool_ingest_materialize_and_report(tmp_path, monkeypatch) -> None:
    pool = tmp_path / "pool"
    dataset = tmp_path / "dataset"
    out = tmp_path / "materialized"
    _jsonl(dataset / "train.jsonl", [
        _row(0, advantage=1.2),
        _row(1, source_kind="kimi_teacher_label", advantage=0.0),
        _row(2, advantage=1.0, flags=["dangerous_end_turn"]),
    ])
    _jsonl(dataset / "eval.jsonl", [])

    monkeypatch.setattr("sys.argv", [
        "manage_dataset_pool",
        "--pool-root", str(pool),
        "ingest-dataset",
        "--dataset-dir", str(dataset),
    ])
    assert pool_main() == 0

    registry = (pool / "registry.jsonl").read_text(encoding="utf-8")
    assert '"tier":"silver"' in registry
    assert '"tier":"gold"' in registry
    assert '"tier":"quarantine"' in registry

    monkeypatch.setattr("sys.argv", [
        "manage_dataset_pool",
        "--pool-root", str(pool),
        "materialize",
        "--out-dir", str(out),
        "--target-size", "10",
        "--gold-min-ratio", "0.5",
    ])
    assert pool_main() == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_rows"] == 2
    assert summary["tier_counts"]["gold"] == 1
    assert summary["tier_counts"]["silver"] == 1
    assert (out / "train.jsonl").exists()

    monkeypatch.setattr("sys.argv", [
        "manage_dataset_pool",
        "--pool-root", str(pool),
        "report",
    ])
    assert pool_main() == 0
    report = json.loads((pool / "report.json").read_text(encoding="utf-8"))
    assert report["sample_rows"] == 3


def test_dataset_pool_ingests_audit_hardcases(tmp_path, monkeypatch) -> None:
    pool = tmp_path / "pool"
    audit = tmp_path / "audit"
    _jsonl(audit / "invalid_cases.jsonl", [{
        "episode_id": "ep1",
        "case_id": "case1",
        "outcome": "invalid_output:dangerous_end_turn",
        "cause": {"category": "unsafe_end_turn"},
        "quality_summary": {"hp_lost": 12},
    }])
    _jsonl(audit / "abnormal_cases.jsonl", [{
        "episode_id": "ep2",
        "case_id": "case2",
        "outcome": "defeat",
        "cause": {"category": "combat_loss"},
    }])

    monkeypatch.setattr("sys.argv", [
        "manage_dataset_pool",
        "--pool-root", str(pool),
        "ingest-audit",
        "--audit-dir", str(audit),
    ])
    assert pool_main() == 0

    hardcases = (pool / "hardcases" / "hardcases.jsonl").read_text(encoding="utf-8")
    assert "unsafe_end_turn" in hardcases
    assert "combat_loss" in hardcases
    assert "needs_teacher" in hardcases
    assert "quarantine" in hardcases
