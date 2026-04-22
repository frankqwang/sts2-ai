from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

from zero.config import ZeroConfig
from zero.paths import ZeroPaths
from zero.replay import SkadaBuild, SkadaCombatCase
from zero.replay.train import (
    build_fresh_policy,
    resolve_run_output_root,
)
from zero.orchestration import ModelPolicyAdapter


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
            legacy_old = base_root / "04-20-10-00-skada-replay-train" / "my_run"
            legacy_new = base_root / "04-21-10-00-skada-replay-train" / "my_run"
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


if __name__ == "__main__":
    unittest.main()
