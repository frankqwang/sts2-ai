from __future__ import annotations

import json

from llm.data_pipeline.guide_knowledge import (
    load_guide_entries,
    render_guide_block,
    retrieve_guides_for_state,
)


def _state() -> dict:
    return {
        "run": {"act": 1, "floor": 4},
        "player": {
            "deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}],
            "relics": [{"id": "BURNING_BLOOD"}],
        },
        "battle": {
            "encounter_id": "CULTISTS_NORMAL",
            "hand": [{"id": "BASH", "description": "Deal 8 damage. Apply 2 Vulnerable."}],
        },
        "enemies": [{"monster_id": "CULTIST", "intent_type": "Attack"}],
    }


def test_guide_retrieval_matches_entities(tmp_path) -> None:
    corpus = tmp_path / "guide.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "cultist",
                "scope": "enemy",
                "entity_ids": ["CULTIST", "CULTISTS_NORMAL"],
                "tags": ["scaling"],
                "text": "Focus Cultist pressure before scaling gets out of hand.",
                "source": "test",
                "confidence": 0.8,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    entries = retrieve_guides_for_state(_state(), path=corpus, limit=2)
    block = render_guide_block(entries)

    assert entries
    assert entries[0].entry_id == "cultist"
    assert "retrieved_knowledge:" in block
    assert "source=test" in block


def test_default_guide_corpus_loads() -> None:
    entries = load_guide_entries()

    assert any("BASH" in entry.entity_ids for entry in entries)
    assert any(entry.source.startswith("https://") for entry in entries)
