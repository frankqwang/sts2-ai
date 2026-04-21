from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

from zero.config import SearchConfig, ZeroConfig
from zero.domain import IterationManifest, PromotionDecision, TrainingSummary
from zero.paths import ZeroPaths
from zero.replay import NoopSearchBackend, SkadaBuild, SkadaCombatCase
from zero.replay.train import (
    build_fresh_policy,
    build_search_backend,
    normalize_search_mode,
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

    def test_normalize_search_mode_keeps_only_public_search_modes(self) -> None:
        self.assertEqual(normalize_search_mode("disabled"), "disabled")
        self.assertEqual(normalize_search_mode("search_root_sweep"), "search_root_sweep")
        self.assertEqual(normalize_search_mode("search_branching"), "search_branching")

    def test_build_search_backend_returns_noop_backend_when_search_is_disabled(self) -> None:
        backend = build_search_backend(
            [_make_case()],
            search_mode="disabled",
            config=SearchConfig(),
            port=15527,
            auto_launch=False,
            connect_timeout_s=1.0,
        )
        self.assertIsInstance(backend, NoopSearchBackend)

    def test_build_search_backend_rejects_non_disabled_modes(self) -> None:
        with self.assertRaises(ValueError):
            build_search_backend(
                [_make_case()],
                search_mode="search_root_sweep",
                config=SearchConfig(),
                port=15527,
                auto_launch=False,
                connect_timeout_s=1.0,
            )
        with self.assertRaises(ValueError):
            build_search_backend(
                [_make_case()],
                search_mode="weak",
                config=SearchConfig(),
                port=15527,
                auto_launch=False,
                connect_timeout_s=1.0,
            )

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

    def test_zero_paths_uses_search_labels_as_public_artifact_dir(self) -> None:
        paths = ZeroPaths(root=Path("C:/tmp/zero-run"))
        self.assertEqual(paths.search_labels, Path("C:/tmp/zero-run/search_labels"))
        self.assertEqual(paths.search_labels, paths.search_labels)

    def test_iteration_manifest_to_dict_uses_public_search_policy_names(self) -> None:
        manifest = IterationManifest(
            iteration=1,
            collector_version="policy_v0001",
            search_version="MultiCaseMctsSearcher",
            sample_counts={"search_requests": 3, "search_entries": 2, "search_labeled_samples": 2},
            admission_stats={"search_requests": 3, "search_entries": 2},
            training=TrainingSummary(search_sample_ratio=0.4),
            promotion=PromotionDecision(promoted=False, reason="pending"),
        )
        payload = manifest.to_dict()
        self.assertEqual(payload["search_version"], "MultiCaseMctsSearcher")
        self.assertEqual(payload["sample_counts"]["search_requests"], 3)
        self.assertEqual(payload["sample_counts"]["search_entries"], 2)
        self.assertEqual(payload["sample_counts"]["search_labeled_samples"], 2)
        self.assertEqual(payload["training"]["search_sample_ratio"], 0.4)


if __name__ == "__main__":
    unittest.main()
