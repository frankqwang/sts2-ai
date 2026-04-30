from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from llm.scripts.datasets.sanitize_combat_train_jsonl import (
    _normalize_user_return_line,
    _row_quality_flags,
    _strip_reason_from_assistant,
    sanitize_row,
)


def test_strip_reason_from_assistant_drops_reason_field_keeps_others():
    raw = '{"action_index":3,"confidence":0.75,"reason":"Deal 6 damage"}'
    new, dropped = _strip_reason_from_assistant(raw)
    assert dropped is True
    parsed = json.loads(new)
    assert parsed == {"action_index": 3, "confidence": 0.75}


def test_strip_reason_from_assistant_no_reason_returns_unchanged():
    raw = '{"action_index":1,"confidence":0.9}'
    new, dropped = _strip_reason_from_assistant(raw)
    assert dropped is False
    assert json.loads(new) == {"action_index": 1, "confidence": 0.9}


def test_strip_reason_handles_garbage_payload():
    new, dropped = _strip_reason_from_assistant("not json")
    assert dropped is False
    assert new == "not json"


def test_normalize_user_return_line_replaces_legacy_instruction():
    legacy = (
        "run: char=IRONCLAD\n"
        "...\n"
        'Return strict JSON only: {"action_index":N,"confidence":0.0,"reason":"..."} '
        "using one listed action_index. Do not output multiple objects or candidates."
    )
    out = _normalize_user_return_line(legacy)
    assert '"reason"' not in out
    assert '"action_index":N,"confidence":0.0' in out
    assert "strategy text belongs to the planner model" in out


def test_normalize_user_return_line_appends_when_missing():
    bare = "run: char=IRONCLAD\nplayer: hp=80/80\n"
    out = _normalize_user_return_line(bare)
    assert "Return strict JSON only" in out
    assert '"reason"' not in out


def test_row_quality_flags_aggregates_top_level_and_meta():
    row = {
        "quality_flags": ["dangerous_end_turn"],
        "meta": {
            "action_quality_flags": ["missed_visible_lethal"],
            "step_quality_flags": "low_hp_self_damage",
        },
    }
    flags = _row_quality_flags(row)
    assert flags == {"dangerous_end_turn", "missed_visible_lethal", "low_hp_self_damage"}


def test_sanitize_row_strips_reason_and_normalizes_user():
    row = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": (
                "run: char=IRONCLAD\n"
                'Return strict JSON only: {"action_index":N,"confidence":0.0,"reason":"..."} '
                "using one listed action_index. Do not output multiple objects or candidates."
            )},
            {"role": "assistant", "content": '{"action_index":2,"confidence":0.8,"reason":"Gain 5 Block"}'},
        ],
        "meta": {"source": "rollout_legacy"},
    }
    cleaned, status = sanitize_row(row)
    assert status == "kept"
    user = cleaned["messages"][1]["content"]
    assistant = json.loads(cleaned["messages"][2]["content"])
    # v4 instruction 提到 "reason" 用作禁用语的 prose, 但 JSON schema 里
    # 不应再有引号包起来的 ``"reason"`` 字段.
    assert '"reason"' not in user
    assert assistant == {"action_index": 2, "confidence": 0.8}
    assert cleaned["meta"]["sanitized"] is True
    assert cleaned["meta"]["legacy_reason_stripped"] is True


def test_sanitize_row_rejects_no_user_message():
    row = {"messages": [{"role": "system", "content": "S"}]}
    cleaned, status = sanitize_row(row)
    assert cleaned is None
    assert status == "no_user_message"


def test_cli_drops_blocked_flag_rows_and_keeps_clean(tmp_path) -> None:
    in_path = tmp_path / "legacy_train.jsonl"
    bad_row = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "run: char=IRONCLAD\n"},
            {"role": "assistant", "content": '{"action_index":3,"confidence":0.8,"reason":"BLOODLETTING"}'},
        ],
        "meta": {"action_quality_flags": ["dangerous_self_damage"]},
    }
    good_row = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "run: char=IRONCLAD\n"},
            {"role": "assistant", "content": '{"action_index":1,"confidence":0.9,"reason":"deal 6 damage"}'},
        ],
        "meta": {"action_quality_flags": []},
    }
    invalid_row = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "run: char=IRONCLAD\n"},
            {"role": "assistant", "content": '{"action_index":-1,"confidence":0.0,"reason":"fallback"}'},
        ],
        "meta": {},
    }
    in_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in [bad_row, good_row, invalid_row]) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [sys.executable, "-m", "llm.scripts.datasets.sanitize_combat_train_jsonl",
         "--input", str(in_path), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        check=True,
    )

    train_path = out_dir / "train.jsonl"
    dropped_path = out_dir / "dropped.jsonl"
    summary_path = out_dir / "summary.json"
    assert train_path.exists()
    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dropped_rows = [json.loads(line) for line in dropped_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(train_rows) == 1
    asst = json.loads(train_rows[0]["messages"][2]["content"])
    assert "reason" not in asst
    assert asst["action_index"] == 1
    drop_reasons = {row["drop_reason"] for row in dropped_rows}
    assert "blocklist_flag" in drop_reasons
    assert "non_positive_action_index" in drop_reasons
    assert summary["kept_rows"] == 1
    assert summary["legacy_reason_stripped"] == 1
    assert summary["drop_counters"]["dropped_blocked_flag"] == 1
    assert summary["drop_counters"]["dropped_negative_action_index"] == 1
