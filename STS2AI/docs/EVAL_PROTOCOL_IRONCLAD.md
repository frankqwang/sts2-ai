# IRONCLAD Evaluation Protocol

This protocol keeps teacher and scorer experiments comparable.

## Fixed Seed Ladders

- Quick A/B: `20` seeds
- Stability A/B: `50` seeds

If no seed file is present, `evaluate_ai.py` falls back to deterministic seed
names `EVAL_001`, `EVAL_002`, and so on. That is acceptable for this protocol.

## Required Metrics

- `avg_floor`
- `boss_reach_rate`
- `act1_clear_rate`
- `avg_boss_hp_fraction_dealt`

## Helpful Secondary Metrics

- `combat_teacher_override_counts`
- `avg_combat_teacher_overrides_per_game`
- `games_with_combat_teacher_override`

## Teacher Modes

- `hard_override`
  - Baseline-first mode.
  - Teacher only overrides when the runtime gate says the baseline action is clearly suspect.

- `full_replace`
  - Teacher-first mode.
  - Combat actions come directly from the teacher scorer except for narrow lethal special cases.

## Recommended Current Workflow

1. Compare teacher checkpoints with `hard_override` on `20` seeds.
2. If one candidate is clearly better, rerun it on `50` seeds.
3. Only compare `full_replace` after `hard_override` is stable.
4. Treat `avg_floor` improvements without `act1_clear_rate > 0` as directional, not final.

## Suggested Commands

```powershell
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/run-ironclad-teacher-eval.ps1 `
  -TeacherCheckpoint C:\path\to\combat_teacher_final.pt `
  -Mode hard_override `
  -SeedCount 20
```

```powershell
powershell -ExecutionPolicy Bypass -File STS2AI/Python/scripts/run-ironclad-teacher-eval.ps1 `
  -TeacherCheckpoint C:\path\to\combat_teacher_final.pt `
  -Mode hard_override `
  -SeedCount 50
```
