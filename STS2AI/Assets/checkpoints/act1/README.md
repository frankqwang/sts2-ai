# STS2AI Stable Checkpoints (Act 1)

This directory contains the self-contained mainline checkpoint package used by
the `STS2AI` training and evaluation entrypoints.

## Current Champion

- `retrieval_final_iter2175.pt`
  - role: production champion
  - frozen_at: `2026-04-09`
  - note: contains both the non-combat PPO brain and the combat head used by
    the current mainline.

## Policy

- Mainline only keeps one promoted Act 1 checkpoint in-tree.
- Older milestones, challengers, and deprecated combat-only baselines are not
  part of the active `STS2AI` package.
- Historical recovery should use git history or archived docs instead of
  treating superseded checkpoints as active assets.

## Evaluate

```powershell
python STS2AI/Python/evaluate_ai.py `
  --checkpoint STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt `
  --transport pipe-binary `
  --seeds-file STS2AI/Assets/seeds/full_run_benchmark_seeds_200.json `
  --seed-suite benchmark `
  --num-games 50
```

## Resume Training

```powershell
python STS2AI/Python/train_hybrid.py `
  --resume STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt `
  --retrieval-head `
  --retrieval-proj-dim 16
```

`train_hybrid.py` will auto-enable the checkpoint's retrieval-head settings when
resuming from the mainline champion, so the command above is the intended
copy-and-run entrypoint.

The previously documented default auxiliary datasets
`STS2AI/Assets/datasets/card_ranking_post_wizardly` and
`STS2AI/Assets/datasets/combat_teacher_post_wizardly/teacher.jsonl` were
removed because they contained invalid data. If you need those inputs for a new
training run, provide regenerated and validated replacements explicitly.
