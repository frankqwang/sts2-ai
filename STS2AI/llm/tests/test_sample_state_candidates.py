from __future__ import annotations

from collections import Counter

from llm.scripts.teacher.sample_state_candidates import (
    CandidateSampler,
    entropy_from_counts,
    listed_action_indices,
    parse_action_index,
    parse_judge_selection,
)


def test_parse_action_index_accepts_json_and_salvages_index() -> None:
    assert parse_action_index('{"action_index": 3, "reason": "x"}') == (3, "ok")
    assert parse_action_index("action_index: 2") == (2, "json_parse_failed_but_index_found")
    assert parse_action_index('{"action_index": "2"}') == (None, "action_index_not_int")


def test_parse_judge_selection_accepts_json_and_salvages_selection() -> None:
    assert parse_judge_selection(
        '{"best_sample_index": 1, "action_index": 4, "reason": "best line"}'
    ) == (1, 4, "best line", "ok")
    assert parse_judge_selection("best_sample_index: 2 action_index: 7") == (
        2,
        7,
        "",
        "json_parse_failed_but_selection_found",
    )


def test_listed_action_indices_and_entropy() -> None:
    user = """legal_actions:
  [0] A
  B hand[1]:
    [1] target=enemy1
    [2] target=enemy2
"""

    assert listed_action_indices(user) == {0, 1, 2}
    assert entropy_from_counts(Counter({0: 4})) == 0.0
    assert entropy_from_counts(Counter({0: 2, 1: 2})) == 1.0


def test_judge_prompt_includes_retrieved_experience(tmp_path) -> None:
    lesson = tmp_path / "lessons.jsonl"
    lesson.write_text(
        '{"tags":["vulnerable","attack"],"applies_when":"Vulnerable and attack are available",'
        '"advice":"apply Vulnerable before attacking","confidence":0.9}\n',
        encoding="utf-8",
    )

    sampler = object.__new__(CandidateSampler)

    class Args:
        experience_path = str(lesson)
        experience_limit = 2

    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return "\n".join(message["content"] for message in messages)

    sampler.args = Args()
    sampler.tokenizer = Tokenizer()
    sampler.experience_entries = []
    from llm.data_pipeline.experience_library import load_experience

    sampler.experience_entries = load_experience(lesson)
    prompt, experience = sampler.judge_prompt(
        [{"role": "user", "content": "legal_actions:\n  [0] apply Vulnerable\n  [1] attack"}],
        [{"sample_index": 0, "action_index": 0, "status": "ok", "raw_generation": '{"reason":"setup"}'}],
    )

    assert "experience:" in prompt
    assert "apply Vulnerable before attacking" in prompt
    assert experience[0]["tags"] == ["vulnerable", "attack"]
