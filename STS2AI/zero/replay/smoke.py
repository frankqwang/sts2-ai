from __future__ import annotations

"""最小 skada replay smoke。

默认值统一从 `zero.config.ZERO_RUNTIME_DEFAULTS` 读取，方便入口运行，
但角色/build 等权威数据仍需来自 game_wiki sqlite 与 runtime。
"""

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ZERO_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(ZERO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ZERO_PACKAGE_ROOT))

from zero import ZeroConfig, ZeroLoopRunner
from zero.adapters.game_bridge import GameBridgeCombatRuntime
from zero.buffers import ArtifactStore
from zero.config import EvalConfig, TeacherConfig, TrainConfig, ZERO_RUNTIME_DEFAULTS
from zero.orchestration.trainer import LocalCheckpointStore
from zero.paths import ZeroPaths
from zero.replay.skada import (
    AggregateCardUsageTeacher,
    FixedSkadaCaseEvaluator,
    build_case_from_record,
    find_first_matching_run,
    load_skada_run_record,
    resolve_starting_build_from_runtime,
)


class RandomPolicy:
    def reset_episode(self) -> None:
        return None

    def select_action(self, state) -> int:
        if not state.legal_actions:
            return 0
        return random.randrange(len(state.legal_actions))

    def score_actions(self, state) -> list[float]:
        return [1.0 for _ in state.legal_actions]

    def estimate_uncertainty(self, state) -> float:
        return 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skada-root", type=Path, default=Path("STS2AI/data/skada/runs_full_detail"))
    parser.add_argument("--game-version", type=str, default="v0.103.2")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--player-count", type=int, default=1)
    parser.add_argument("--character-id", type=str, default=ZERO_RUNTIME_DEFAULTS.default_character_id)
    parser.add_argument("--combat-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=ZERO_RUNTIME_DEFAULTS.default_port)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--output-root", type=Path, default=Path("STS2AI/Artifacts/zero/skada_replay_smoke_2026-04-20"))
    args = parser.parse_args()

    source_path, source_line = find_first_matching_run(
        root=args.skada_root,
        game_version=args.game_version,
        ascension=args.ascension,
        player_count=args.player_count,
        character_id=args.character_id,
        victory_only=True,
    )
    record = load_skada_run_record(source_path, source_line)

    sorted_combats = sorted(record.get("combats") or [], key=lambda item: int(item.get("floor") or 0))
    selected_combat = sorted_combats[args.combat_index]
    starter_build = resolve_starting_build_from_runtime(
        character_id=args.character_id,
        encounter_id=str(selected_combat.get("encounter") or ""),
        port=args.port,
        auto_launch=True,
        connect_timeout_s=45.0,
    )
    case = build_case_from_record(
        record,
        source_path=source_path,
        source_line=source_line,
        starter_build=starter_build,
        combat_index=args.combat_index,
    )

    output_root = args.output_root / f"run_{case.run_id}_floor_{case.floor}_{case.encounter_id.lower()}"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selected_case.json").write_text(
        json.dumps(case.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config = ZeroConfig(
        paths=ZeroPaths(root=output_root),
        teacher=TeacherConfig(
            top2_gap_threshold=1.0,
            uncertainty_threshold=0.0,
            near_lethal_hp_ratio=1.0,
            max_requests_per_iteration=2048,
        ),
        train=TrainConfig(
            batch_size=8,
            steps_per_iteration=args.train_steps,
            learning_rate=3e-4,
            weight_decay=1e-4,
            grad_clip_norm=1.0,
        ),
        evaluation=EvalConfig(
            episodes_per_cohort=args.episodes,
            promote_min_win_rate_gain=-1.0,
            allow_hp_remaining_drop=1.0,
        ),
    )

    artifact_store = ArtifactStore(config.paths)
    checkpoint_store = LocalCheckpointStore(config.paths.checkpoints)
    evaluator = FixedSkadaCaseEvaluator([case], port=args.port, auto_launch=True, connect_timeout_s=45.0)
    runner = ZeroLoopRunner(
        config=config,
        artifact_store=artifact_store,
        checkpoint_store=checkpoint_store,
        evaluator=evaluator,
    )
    teacher = AggregateCardUsageTeacher(case)

    def runtime_factory():
        return GameBridgeCombatRuntime(
            port=args.port,
            auto_launch=True,
            connect_timeout_s=45.0,
            character_id=case.character_id,
            encounter_id=case.encounter_id,
            seed=case.seed,
            build=case.build.to_build_dict(),
        )

    baseline = evaluator.evaluate(RandomPolicy())
    manifest = runner.run_iteration(
        iteration=1,
        runtime_factory=runtime_factory,
        student_policy=RandomPolicy(),
        teacher_oracle=teacher,
        baseline_eval=baseline,
    )

    metrics = {
        "selected_case": case.to_dict(),
        "baseline": [asdict(summary) for summary in baseline],
        "manifest": manifest.to_dict(),
    }
    metrics_path = output_root / "smoke_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "metrics_path": str(metrics_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
