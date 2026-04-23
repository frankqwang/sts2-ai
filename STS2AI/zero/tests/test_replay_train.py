from __future__ import annotations

import tempfile
import unittest
import os
import importlib.util
import sys
from pathlib import Path

from zero.config import ZeroConfig
from zero.paths import ZeroPaths
from zero.replay import SkadaBuild, SkadaCombatCase
from zero.replay.train import (
    build_fresh_policy,
    merge_manifest_history,
    resolve_resume_iteration_start,
    resolve_shared_sim_layout,
    resolve_run_output_root,
)
from zero.orchestration import ModelPolicyAdapter
from zero.orchestration.trainer import _sanitize_score_list

_CASE_PACK_MODULE_PATH = Path(__file__).resolve().parents[2] / "data" / "skada" / "generate_zero_case_pack.py"
_CASE_PACK_SPEC = importlib.util.spec_from_file_location("generate_zero_case_pack", _CASE_PACK_MODULE_PATH)
if _CASE_PACK_SPEC is None or _CASE_PACK_SPEC.loader is None:
    raise RuntimeError(f"无法加载 case pack 脚本: {_CASE_PACK_MODULE_PATH}")
_CASE_PACK_MODULE = importlib.util.module_from_spec(_CASE_PACK_SPEC)
sys.modules[_CASE_PACK_SPEC.name] = _CASE_PACK_MODULE
_CASE_PACK_SPEC.loader.exec_module(_CASE_PACK_MODULE)
BUCKETS = _CASE_PACK_MODULE.BUCKETS
build_case_pack = _CASE_PACK_MODULE.build_case_pack
classify_case_bucket = _CASE_PACK_MODULE.classify_case_bucket


def _make_case() -> SkadaCombatCase:
    return SkadaCombatCase(
        source_path="cases.jsonl",
        source_line=1,
        run_id=42,
        seed="seed-42",
        game_version="0.0",
        character_id="IRONCLAD",
        ascension=20,
        player_count=1,
        floor=3,
        encounter_id="JawWorm",
        encounter_type="normal",
        won=True,
        build=SkadaBuild(current_hp=80, max_hp=80),
    )


class ReplayTrainHelpersTests(unittest.TestCase):
    def test_sanitize_score_list_replaces_nonfinite_values(self) -> None:
        sanitized = _sanitize_score_list([1.0, float("nan"), float("inf"), -float("inf"), -3.5])
        self.assertEqual(sanitized[0], 1.0)
        self.assertEqual(sanitized[1], -1.0e9)
        self.assertEqual(sanitized[2], -1.0e9)
        self.assertEqual(sanitized[3], -1.0e9)
        self.assertEqual(sanitized[4], -3.5)

    def test_build_fresh_policy_returns_model_policy_adapter(self) -> None:
        policy = build_fresh_policy(ZeroConfig())
        self.assertIsInstance(policy, ModelPolicyAdapter)

    def test_resolve_run_output_root_prefers_stable_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_root = Path(temp_dir)
            stable_root = base_root / "my_run"
            stable_root.mkdir(parents=True, exist_ok=True)
            resolved = resolve_run_output_root(
                base_output_root=base_root,
                run_name="my_run",
                from_scratch=False,
            )
            self.assertEqual(resolved, stable_root)

    def test_resolve_run_output_root_reuses_latest_legacy_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_root = Path(temp_dir)
            legacy_old = base_root / "0420-1000-skada-replay-train" / "my_run"
            legacy_new = base_root / "0421-1000-skada-replay-train" / "my_run"
            legacy_old.mkdir(parents=True, exist_ok=True)
            legacy_new.mkdir(parents=True, exist_ok=True)
            os.utime(legacy_old, (1, 1))
            os.utime(legacy_new, (2, 2))
            resolved = resolve_run_output_root(
                base_output_root=base_root,
                run_name="my_run",
                from_scratch=False,
            )
            self.assertEqual(resolved, legacy_new)

    def test_resolve_run_output_root_from_scratch_creates_dated_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_root = Path(temp_dir)
            resolved = resolve_run_output_root(
                base_output_root=base_root,
                run_name="my_run",
                from_scratch=True,
            )
            self.assertEqual(resolved.parent, base_root)
            self.assertTrue(resolved.name.endswith("my_run"))

    def test_zero_paths_exposes_expected_subdirs(self) -> None:
        paths = ZeroPaths(root=Path("C:/tmp/zero-run"))
        self.assertEqual(paths.raw_runs, Path("C:/tmp/zero-run/raw_runs"))
        self.assertEqual(paths.dataset_shards, Path("C:/tmp/zero-run/dataset_shards"))
        self.assertEqual(paths.manifests, Path("C:/tmp/zero-run/manifests"))

    def test_resolve_shared_sim_layout_skips_base_sim_for_parallel_progress_only(self) -> None:
        launch_base, collect_ports = resolve_shared_sim_layout(
            base_port=15527,
            parallel_envs=4,
            progress_only=True,
            curriculum_mode="ordered_run",
        )
        self.assertFalse(launch_base)
        self.assertEqual(collect_ports, [15528, 15529, 15530, 15531])

    def test_resolve_shared_sim_layout_keeps_base_sim_when_eval_may_run(self) -> None:
        launch_base, collect_ports = resolve_shared_sim_layout(
            base_port=15527,
            parallel_envs=4,
            progress_only=False,
            curriculum_mode="ordered_run",
        )
        self.assertTrue(launch_base)
        self.assertEqual(collect_ports, [15528, 15529, 15530, 15531])

    def test_resolve_shared_sim_layout_allows_parallel_targeted_cases(self) -> None:
        launch_base, collect_ports = resolve_shared_sim_layout(
            base_port=15527,
            parallel_envs=4,
            progress_only=True,
            curriculum_mode="targeted_cases",
        )
        self.assertFalse(launch_base)
        self.assertEqual(collect_ports, [15528, 15529, 15530, 15531])

    def test_resolve_resume_iteration_start_uses_existing_manifest_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            (run_root / "manifests").mkdir(parents=True, exist_ok=True)
            (run_root / "raw_runs").mkdir(parents=True, exist_ok=True)
            (run_root / "logs").mkdir(parents=True, exist_ok=True)
            (run_root / "manifests" / "iter_0012.json").write_text("{}", encoding="utf-8")
            (run_root / "raw_runs" / "iter_0011.jsonl").write_text("", encoding="utf-8")
            existing_metadata = {
                "manifests": [
                    {"iteration": 9},
                    {"iteration": 10},
                ]
            }
            self.assertEqual(
                resolve_resume_iteration_start(run_root, existing_metadata=existing_metadata),
                13,
            )

    def test_merge_manifest_history_keeps_latest_per_iteration(self) -> None:
        existing = [
            {"iteration": 9, "value": "old9"},
            {"iteration": 10, "value": "old10"},
        ]
        new = [
            {"iteration": 10, "value": "new10"},
            {"iteration": 11, "value": "new11"},
        ]
        merged = merge_manifest_history(existing, new)
        self.assertEqual([row["iteration"] for row in merged], [9, 10, 11])
        self.assertEqual(merged[1]["value"], "new10")

    def test_classify_case_bucket_prefers_submenu_exhaust(self) -> None:
        case = _make_case()
        case.build.deck = [{"id": "PURITY", "upgrade_level": 0}]
        self.assertEqual(classify_case_bucket(case), "submenu_exhaust")

    def test_build_case_pack_balances_fixed_sizes(self) -> None:
        cases = []
        for bucket_index, bucket in enumerate(BUCKETS):
            for item_index in range(12):
                case = _make_case()
                case.run_id = 1000 + bucket_index * 100 + item_index
                case.floor = 10 + item_index
                if bucket == "submenu_exhaust":
                    case.build.deck = [{"id": "PURITY", "upgrade_level": 0}]
                elif bucket == "setup_payoff":
                    case.build.deck = [{"id": "FEEL_NO_PAIN", "upgrade_level": 0}]
                elif bucket == "resource_dig":
                    case.build.deck = [{"id": "BLOODLETTING", "upgrade_level": 0}]
                else:
                    case.build.deck = [{"id": "STRIKE_IRONCLAD", "upgrade_level": 0}]
                    case.encounter_type = "Elite"
                cases.append(case)

        manifest = build_case_pack(cases, train_size=16, eval_size=8, seed=123)
        self.assertEqual(len(manifest["train_case_ids"]), 16)
        self.assertEqual(len(manifest["eval_case_ids"]), 8)
        self.assertTrue(set(manifest["train_case_ids"]).isdisjoint(set(manifest["eval_case_ids"])))
        self.assertEqual(sum(manifest["bucket_train_counts"].values()), 16)
        self.assertEqual(sum(manifest["bucket_eval_counts"].values()), 8)


if __name__ == "__main__":
    unittest.main()
