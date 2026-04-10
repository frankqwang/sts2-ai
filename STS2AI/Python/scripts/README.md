# STS2AI Script Wrappers

These wrappers are the self-contained entrypoints intended to travel with the
`STS2AI` folder when it is copied into a fresh decompiled project.

## Mainline Entry Points

- `start-hybrid-training.ps1`
  - launches `STS2AI/Python/train_hybrid.py`
- `canonical-eval.ps1`
  - launches `STS2AI/Python/evaluate_ai.py`
- `run_full_run_recording.ps1`
  - visible demo / recording wrapper
  - defaults to `STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt`
  - writes output to `STS2AI/Artifacts/recording`
- `run_sim_vs_godot_audit.ps1`
  - unified sim-vs-Godot audit wrapper
  - defaults to `STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt`
  - writes output to `STS2AI/Artifacts/verification`

Both PowerShell wrappers still resolve the real game project root one level
above `STS2AI`, so they work after copying `STS2AI` into a new upstream
workspace without needing the legacy `tools/python` or `tools/scripts` paths.
