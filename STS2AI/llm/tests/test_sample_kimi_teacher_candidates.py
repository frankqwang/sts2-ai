from __future__ import annotations

from llm.scripts.teacher.sample_kimi_teacher_candidates import (
    _feature_payload,
    _grouped_batch_requests,
    _messages,
    _select_diverse,
)


def _row(episode_id: str = "ep", step: int = 0) -> dict:
    return {
        "episode_id": episode_id,
        "episode_step": step,
        "encounter_id": "CULTISTS_NORMAL",
        "encounter_tag": "act1",
        "user_message": (
            "run: char=IRONCLAD act=0 floor=0 encounter=? round=1 gold=99\n"
            "strategy_context:\n"
            "  memory: repeated\n"
            "  plan: repeated\n"
            "player: hp=80/80 block=0 energy=1/3 powers=-\n"
            "enemies:\n"
            "  enemy1: CULTIST hp=3/30 block=0 intent=Attack(6) powers=-\n"
            "hand:\n"
            "  [0] STRIKE_IRONCLAD cost=1 type=attack | Deal 6 damage.\n"
            "legal_actions:\n"
            "  STRIKE_IRONCLAD hand[0]:\n"
            "    [0] target=enemy1 damage=6\n"
            "  [1] end_turn\n"
            "Return one JSON line: {\"action_index\": N, \"reason\": \"...\"} using a listed action_index."
        ),
        "decoded": {"action_index": 1, "reason": "end turn"},
        "quality_flags": ["missed_visible_lethal"],
    }


def test_feature_payload_marks_visible_lethal() -> None:
    features = _feature_payload(_row())

    assert features is not None
    assert features["visible_lethal_count"] == 1
    assert "visible_lethal" in features["tags"]
    assert features["original_index"] == 1


def test_messages_strip_strategy_context_and_old_return_instruction() -> None:
    features = _feature_payload(_row())
    assert features is not None
    candidate = {
        "candidate_id": "c1",
        "features": {
            key: value
            for key, value in features.items()
            if key not in {"actions", "chosen", "hand", "enemies"}
        },
        "source": {"user_message": _row()["user_message"]},
    }

    messages = _messages(candidate, max_state_chars=2000)
    user = messages[1]["content"]

    assert "strategy_context:" not in user
    assert "Return one JSON line:" not in user
    assert "legal_actions:" in user


def test_select_diverse_avoids_exact_duplicates() -> None:
    base = {
        "score": 10.0,
        "bucket": "CULTISTS_NORMAL|act1|visible_lethal",
        "features": {"exact_hash": "same", "scene_signature": "same_scene"},
        "candidate_id": "a",
    }
    duplicate = {**base, "candidate_id": "b"}
    other = {
        **base,
        "candidate_id": "c",
        "features": {"exact_hash": "other", "scene_signature": "other_scene"},
    }

    selected = _select_diverse([base, duplicate, other], limit=3, seed=1)

    assert [item["candidate_id"] for item in selected] == ["a", "c"]


def test_grouped_batch_requests_pack_multiple_candidates() -> None:
    features = _feature_payload(_row())
    assert features is not None
    candidate = {
        "candidate_id": "c1",
        "features": {
            key: value
            for key, value in features.items()
            if key not in {"actions", "chosen", "hand", "enemies"}
        },
        "source": {"user_message": _row()["user_message"]},
    }
    rows = _grouped_batch_requests(
        [candidate, {**candidate, "candidate_id": "c2"}, {**candidate, "candidate_id": "c3"}],
        model="kimi-k2.6",
        thinking="disabled",
        max_tokens=2048,
        items_per_request=2,
        max_state_chars=2000,
    )

    assert len(rows) == 2
    assert rows[0]["custom_id"] == "kimi-teacher-group-0000"
    assert set(rows[0]) == {"custom_id", "method", "url", "body"}
    assert '"reviews"' in rows[0]["body"]["messages"][1]["content"]
