from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

_python_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _python_root not in sys.path:
    sys.path.insert(0, _python_root)

from networkV2.s1_schema.token_banks import Token, TokenBank, UnifiedTokenBanks
from networkV2.s6_training.batch import TrainingSample
from networkV2.s6_training.combat_teacher_v1 import (
    generate_branch_rollout_dataset,
    load_offline_combat_teacher_entries,
)
from networkV2.s6_training.critical_step_pipeline import annotate_critical_steps, rebalance_training_samples


def _dummy_banks(*, decision_domain: str = "combat", n_actions: int = 2) -> UnifiedTokenBanks:
    return UnifiedTokenBanks(
        action_bank=TokenBank(
            bank_name="action",
            tokens=[Token(numeric=[float(idx)], token_type="action_candidate") for idx in range(n_actions)],
        ),
        decision_domain=decision_domain,
    )


def _sample(
    *,
    room_type: str,
    advantage: float = 0.0,
    turn_damage_target: float = -1.0,
    turn_block_target: float = -1.0,
    fight_win_target: float = -1.0,
    sample_weight: float = 1.0,
    critical_score: float = 0.0,
) -> TrainingSample:
    return TrainingSample(
        banks=_dummy_banks(decision_domain="combat" if room_type in {"monster", "elite", "boss"} else "event"),
        action_index=0,
        old_log_prob=0.0,
        advantage=advantage,
        value_target=0.0,
        value_estimate=0.0,
        turn_damage_target=turn_damage_target,
        turn_block_target=turn_block_target,
        fight_win_target=fight_win_target,
        sample_weight=sample_weight,
        base_sample_weight=sample_weight,
        room_type=room_type,
        critical_score=critical_score,
    )


def test_annotate_critical_steps_marks_expected_rules():
    samples = [
        _sample(room_type="event"),
        _sample(room_type="boss", advantage=5.0, turn_damage_target=14.0, sample_weight=1.5),
        _sample(room_type="elite", advantage=1.0, turn_block_target=13.0, sample_weight=1.2),
        _sample(room_type="monster", advantage=0.2, fight_win_target=1.0),
        _sample(room_type="shop"),
    ]

    metrics = annotate_critical_steps(samples)

    boss_sample = samples[1]
    assert set(boss_sample.critical_tags) == {"boss_room", "high_adv", "turn_swing"}
    assert boss_sample.critical_score == pytest.approx(2.6)
    assert boss_sample.sample_weight == pytest.approx(1.5 * 2.5)

    elite_sample = samples[2]
    assert set(elite_sample.critical_tags) == {"elite_room", "turn_swing"}
    assert elite_sample.critical_score == pytest.approx(1.4)
    assert elite_sample.sample_weight == pytest.approx(1.2 * 1.75)

    terminal_sample = samples[3]
    assert terminal_sample.critical_tags == ("terminal_swing",)
    assert terminal_sample.critical_score == pytest.approx(0.6)
    assert terminal_sample.sample_weight == pytest.approx(1.0)

    assert metrics["critical_combat_count"] == pytest.approx(2.0)
    assert metrics["critical_boss_hits"] == pytest.approx(1.0)
    assert metrics["critical_elite_hits"] == pytest.approx(1.0)
    assert metrics["critical_high_adv_hits"] == pytest.approx(1.0)
    assert metrics["critical_turn_swing_hits"] == pytest.approx(2.0)
    assert metrics["critical_terminal_hits"] == pytest.approx(1.0)


def test_rebalance_training_samples_hits_fixed_bucket_quota():
    samples = []
    samples.extend(_sample(room_type="boss", critical_score=1.6) for _ in range(10))
    samples.extend(_sample(room_type="monster", critical_score=0.0) for _ in range(20))
    samples.extend(_sample(room_type="event", critical_score=0.0) for _ in range(20))

    out, metrics = rebalance_training_samples(samples, rng=__import__("random").Random(0))

    assert len(out) == 50
    critical = sum(1 for sample in out if sample.room_type in {"monster", "elite", "boss"} and sample.critical_score >= 0.8)
    regular = sum(1 for sample in out if sample.room_type in {"monster", "elite", "boss"} and sample.critical_score < 0.8)
    noncombat = sum(1 for sample in out if sample.room_type not in {"monster", "elite", "boss"})
    assert (critical, regular, noncombat) == (18, 22, 10)
    assert metrics["rebalance_target_critical"] == pytest.approx(18.0)
    assert metrics["rebalance_target_regular"] == pytest.approx(22.0)
    assert metrics["rebalance_target_noncombat"] == pytest.approx(10.0)
    assert metrics["rebalance_output_critical"] == pytest.approx(18.0)
    assert metrics["rebalance_output_regular"] == pytest.approx(22.0)
    assert metrics["rebalance_output_noncombat"] == pytest.approx(10.0)


class _DummyCompiler:
    def compile(self, root_state, legal_actions, *, encounter_id: str, room_type: str):
        return _dummy_banks(decision_domain="combat", n_actions=len(legal_actions))


def test_load_offline_combat_teacher_entries_sets_weights_and_invalid_targets(tmp_path: Path):
    path = tmp_path / "critical_step_teacher_v1.jsonl"
    records = [
        {
            "root_state": {"state_type": "monster", "run": {"floor": 9}},
            "legal_actions": [{"action": "strike"}, {"action": "defend"}, {"action": "bash"}],
            "best_idx": 1,
            "scores": [1.0, 5.0, 2.0],
            "encounter_id": "jaw_worm",
            "room_type": "monster",
            "critical_tags": ["high_adv"],
            "critical_score": 0.8,
            "label_source": "branch_search",
        },
        {
            "root_state": {"state_type": "boss", "run": {"floor": 50}},
            "legal_actions": [{"action": "combo"}, {"action": "wait"}],
            "best_idx": 0,
            "scores": [4.5, 1.5],
            "encounter_id": "boss_slug",
            "room_type": "boss",
            "critical_tags": ["boss_room", "turn_swing"],
            "critical_score": 1.8,
            "label_source": "branch_search",
            "terminal_summary": {"run_outcome": "victory", "score": 14.0},
        },
    ]
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")

    entries = load_offline_combat_teacher_entries(path, compiler=_DummyCompiler())

    assert len(entries) == 2
    first = entries[0].sample
    assert first.action_index == 1
    assert first.sample_weight == pytest.approx(3.0)
    assert first.fight_win_target == pytest.approx(-1.0)
    assert first.turn_damage_target == pytest.approx(-1.0)
    assert first.turn_block_target == pytest.approx(-1.0)

    second = entries[1].sample
    assert second.action_index == 0
    assert second.fight_win_target == pytest.approx(1.0)
    assert second.turn_damage_target == pytest.approx(14.0)
    assert second.turn_block_target == pytest.approx(-1.0)
    assert second.room_type == "boss"
    assert second.critical_tags == ("boss_room", "turn_swing")


class _DummyBranchClient:
    def __init__(self, root_state: dict, action_outcomes: dict[str, dict]):
        self._root_state = copy.deepcopy(root_state)
        self._action_outcomes = {key: copy.deepcopy(value) for key, value in action_outcomes.items()}
        self.port = 19999

    def import_state(self, _path: str):
        self._current_state = copy.deepcopy(self._root_state)
        return copy.deepcopy(self._current_state)

    def act(self, action: dict):
        state = copy.deepcopy(self._action_outcomes[str(action.get("action") or "")])
        self._current_state = state
        return state


def test_generate_branch_rollout_dataset_writes_raw_and_teacher_outputs(tmp_path: Path):
    snapshot_path = tmp_path / "snapshots" / "sample.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text("{}", encoding="utf-8")
    root_state = {
        "state_type": "monster",
        "run": {"floor": 7, "act": 1, "room_type": "elite"},
        "battle": {
            "player": {"hp": 50, "max_hp": 80},
            "enemies": [{"hp": 20, "is_alive": True}],
        },
        "player": {"hp": 50, "max_hp": 80},
    }
    action_outcomes = {
        "strike": {
            "state_type": "monster",
            "run": {"floor": 7, "act": 1, "room_type": "elite"},
            "battle": {
                "player": {"hp": 50, "max_hp": 80},
                "enemies": [{"hp": 0, "is_alive": False}],
            },
            "player": {"hp": 50, "max_hp": 80},
            "terminal": True,
            "run_outcome": "victory",
        },
        "end_turn": {
            "state_type": "monster",
            "run": {"floor": 7, "act": 1, "room_type": "elite"},
            "battle": {
                "player": {"hp": 30, "max_hp": 80},
                "enemies": [{"hp": 20, "is_alive": True}],
            },
            "player": {"hp": 30, "max_hp": 80},
        },
    }
    client = _DummyBranchClient(root_state, action_outcomes)
    queue_records = [{
        "seed": "seed-1",
        "episode_id": "ep-1",
        "sample_index": 3,
        "encounter_id": "lagavulin",
        "room_type": "elite",
        "critical_tags": ["elite_room"],
        "critical_score": 1.4,
        "legal_actions": [{"action": "strike"}, {"action": "end_turn"}],
        "snapshot_path": str(snapshot_path),
    }]

    branch_records, teacher_records, raw_path = generate_branch_rollout_dataset(
        queue_records,
        output_dir=tmp_path,
        client=client,
        net=object(),
        compiler=_DummyCompiler(),
        branch_horizon=1,
    )

    assert len(branch_records) == 1
    assert len(teacher_records) == 1
    assert raw_path.exists()
    manifest_path = tmp_path / "raw" / "raw_manifest.json"
    teacher_path = tmp_path / "critical_step_teacher_v1.jsonl"
    assert manifest_path.exists()
    assert teacher_path.exists()

    raw_record = json.loads(raw_path.read_text(encoding="utf-8").strip())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teacher_record = json.loads(teacher_path.read_text(encoding="utf-8").strip())
    assert raw_record["schema_version"] == "raw_branch_rollout.v1"
    assert raw_record["best_idx"] == 0
    assert len(raw_record["legal_actions"]) == 2
    assert manifest["summary"]["sample_type_counts"]["critical_combat"] == 1
    assert teacher_record["label_source"] == "branch_search"
    assert teacher_record["best_idx"] == 0
