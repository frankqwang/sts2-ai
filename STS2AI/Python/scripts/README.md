# STS2AI Script Wrappers

These wrappers are the self-contained entrypoints intended to travel with the
`STS2AI` folder when it is copied into a fresh decompiled project.

## Mainline Entry Points

- `start-hybrid-training.ps1`
  - launches `STS2AI/Python/train_hybrid.py`
- `start-hybrid-training-mcts.ps1`
  - launches `STS2AI/Python/train_hybrid.py` with the formal MCTS hybrid config
- `canonical-eval.ps1`
  - launches `STS2AI/Python/evaluate_ai.py`
- `run-ironclad-teacher-eval.ps1`
  - fixed-protocol wrapper for `IRONCLAD` teacher A/B checks
  - defaults to `20` or `50` deterministic evaluation seeds
  - exposes `hard_override` / `full_replace` teacher modes
- `run_full_run_recording.ps1`
  - visible demo / recording wrapper
  - defaults to `STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt`
  - writes output to `STS2AI/Artifacts/recording`
- `run_sim_vs_godot_audit.ps1`
  - unified sim-vs-Godot audit wrapper
  - defaults to `STS2AI/Assets/checkpoints/act1/mainline_iter2270_carddebug.pt`
  - writes output to `STS2AI/Artifacts/verification`
- `merge_hybrid_checkpoints.py`
  - merges a separately trained `ppo_model` checkpoint and `combat_model` checkpoint
  - writes canonical hybrid v2 checkpoints using `combat_model` / `combat_model_config`
  - keeps PPO-owned shared weights such as `entity_emb.*` and `symbolic_head.*`

Both PowerShell wrappers still resolve the real game project root one level
above `STS2AI`, so they work after copying `STS2AI` into a new upstream
workspace without needing the legacy `tools/python` or `tools/scripts` paths.
