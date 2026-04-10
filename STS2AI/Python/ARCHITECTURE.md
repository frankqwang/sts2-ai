# STS2 AI Architecture

## Current Status

This file is the current architecture status page.
Older “PPO + full MCTS mainline” descriptions are no longer the source of truth.

For current project state, also read:
- [docs/README.md](C:/Users/Administrator/Desktop/SlaytheSpire2/docs/README.md)
- [docs/2026-04-03-puresim-cutover-handoff.zh-CN.md](C:/Users/Administrator/Desktop/SlaytheSpire2/docs/2026-04-03-puresim-cutover-handoff.zh-CN.md)

## Mainline Architecture

### Dual PPO

- Non-combat brain:
  - [rl_policy_v2.py](STS2AI/Python/core/rl_policy_v2.py)
  - Handles `map / card_reward / shop / rest / event`
- Combat brain:
  - [combat_nn.py](STS2AI/Python/core/combat_nn.py)
  - Handles combat decisions
- Shared embeddings:
  - card embedding
  - monster embedding

### Mainline Training Entry

- [train_hybrid.py](STS2AI/Python/train_hybrid.py)

### Current Stable Champion

- Mainline checkpoint:
  - [retrieval_final_iter2175.pt](STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt)
- This checkpoint contains both the non-combat PPO brain and the combat head.

## pureSim Status

### What is now true

- pureSim is ready for mainline short training and experiments
- pureSim / Godot baseline parity is close enough for practical use
- pureSim save/load is already server-side in-memory snapshot based

### What is not yet true

- pureSim is not “zero-tail forever”
- Some low-frequency transition/loop issues can still appear
- Godot is still retained as confirmation backend

## MCTS Status

### Not the mainline recipe

MCTS is currently a pre-experiment line, not the default training path.

### What MCTS is used for now

- combat-only pre-experiments
- short lookahead
- tactical ordering investigation
- root diagnostics via [evaluate_ai.py](STS2AI/Python/evaluate_ai.py)

### Current conclusion

- MCTS can help card ordering in principle
- current bottleneck is more in leaf objective than in tree parameters
- do not treat full MCTS as the current default architecture

## Important Supporting Files

| File | Purpose |
|------|---------|
| [rl_encoder_v2.py](STS2AI/Python/core/rl_encoder_v2.py) | structured state encoding |
| [rl_reward_shaping.py](STS2AI/Python/core/rl_reward_shaping.py) | reward shaping and tactical probes |
| [full_run_env.py](STS2AI/Python/ipc/full_run_env.py) | environment client |
| [binary_pipe_client.py](STS2AI/Python/ipc/binary_pipe_client.py) | pureSim binary transport |
| [verify_save_load.py](STS2AI/Python/verify_save_load.py) | save/load audit |
| [test_simulator_consistency.py](STS2AI/Python/test_simulator_consistency.py) | sim-vs-godot audit |
| [combat_mcts_agent.py](STS2AI/Python/search/combat_mcts_agent.py) | combat forward model and MCTS agent |

## Working Assumption Going Forward

The project should currently optimize around:
- pureSim mainline short training
- better combat diagnostics
- tactical short lookahead / MCTS pre-experiments
- stronger build-planning supervision
