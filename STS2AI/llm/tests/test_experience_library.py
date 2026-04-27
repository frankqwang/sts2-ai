from __future__ import annotations

from llm.data_pipeline.experience_library import (
    ExperienceEntry,
    append_experience,
    load_experience,
    render_experience_block,
    retrieve_experience,
)


def test_experience_library_roundtrip_and_retrieval(tmp_path) -> None:
    path = tmp_path / "lessons.jsonl"
    entry = ExperienceEntry(
        tags=["vulnerable", "attack"],
        applies_when="hand can apply Vulnerable and also deal high attack damage",
        advice="apply Vulnerable before the biggest attack when energy allows",
        avoid="do not spend the big attack first",
        source="test",
        confidence=0.9,
    )

    append_experience([entry], path)
    loaded = load_experience(path)
    found = retrieve_experience("enemy can be Vulnerable before attack", loaded)
    rendered = render_experience_block(found)

    assert loaded == [entry]
    assert found == [entry]
    assert "experience:" in rendered
    assert "apply Vulnerable before" in rendered
