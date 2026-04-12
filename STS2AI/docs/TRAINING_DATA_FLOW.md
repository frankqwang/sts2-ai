# Training Data Flow

This note describes which data sources currently influence training, and which
ones are only recorded for future use.

For the companion note that explains the model split, head modes, and which
trainer updates which network, see:

- `STS2AI/docs/NETWORK_AND_TRAINING_OVERVIEW.md`

## Main Sources

- Online non-combat PPO rollouts
  - Collected into `ppo_buffer` from live self-play episodes.
  - Still the primary non-combat training signal.
  - Deeper runs are up-weighted when merged, so later-floor episodes matter more.

- Online combat PPO rollouts
  - Collected only from combat steps where the combat NN itself chose the action.
  - Still the primary online combat learning signal.
  - MCTS-chosen combat actions are not stored in this buffer.

- Online combat search examples
  - Produced only when MCTS is enabled.
  - Higher-quality but more expensive combat supervision.
  - Best treated as a combat-quality booster, not a replacement for all rollouts.

- Offline non-combat ranking data
  - Config alias: `offline_noncombat_ranking_*`
  - Used to shape card reward / shop / campfire style ranking behavior.
  - This is an auxiliary loss on top of online PPO, not the dominant signal.

- Offline non-combat ranking head mode
  - Config: `offline_noncombat_ranking_head_mode`
  - Controls only the scorer used by offline non-combat ranking supervision.
  - `mlp` keeps the legacy option scorer.
  - `light_attention` adds a residual attention block over screen context plus
    candidate options.
  - This is the safest first place to try Transformer-style structure for
    card reward / shop style decisions, because it does not replace the whole
    non-combat PPO trunk.

- Offline combat teacher data
  - Config alias: `offline_combat_teacher_*`
  - Turn-solver or teacher-labeled combat states for reranking / correction.
  - This is also auxiliary inside `train_hybrid.py` unless its update count and loss weight are raised.

- Main combat rollout path mode
  - Config: `combat_main_path_mode`
  - Controls the structure of the real combat `policy/value` path used during rollout.
  - `mlp` keeps the legacy main path.
  - `light_attention` adds a residual state/action attention branch to the main path.
  - This should be evaluated through `train_hybrid.py` A/B, not through teacher-only checkpoint fine-tuning.

- Saved offline episodes
  - Config alias: `saved_offline_episodes_*`
  - High-quality episodes are written to disk as raw material for future dataset building.
  - They do not automatically feed back into the same training run.

## Current Practical Reading

- If you are training the hybrid model end-to-end, live rollout data still dominates.
- If you want offline combat teacher data to matter more, first increase:
  - `offline_combat_teacher_updates_per_iter`
  - and only then consider increasing `offline_combat_teacher_loss_weight`
- If you want offline non-combat ranking data to matter more, first increase:
  - `offline_noncombat_ranking_updates_per_iter`
  - and only then consider increasing `offline_noncombat_ranking_loss_weight`
- If you want to test a new non-combat ranking scorer structure, change:
  - `offline_noncombat_ranking_head_mode`
  - while keeping the rest of the non-combat trunk fixed

## Why This Matters

Fixed loss weights do not automatically stay fair as dataset sizes change.

Examples:

- A combat teacher dataset growing from `200` to `2000` samples does not automatically become more influential.
- With a fixed batch size and one update per iteration, the larger dataset is sampled more sparsely.
- That means “more offline data” and “stronger training signal” are not the same thing.

## Recommended Current Use

- Train combat teacher checkpoints separately first when testing a new teacher dataset or scorer architecture.
- If you want to test a new main combat path structure, do it in `train_hybrid.py`
  so the rollout brain is updated under online PPO/combat training rather than
  only on selected teacher states.
- After that, reintroduce the stronger teacher into `train_hybrid.py` with a higher update count.
- Keep route priors and non-combat ranking experiments separate from combat teacher experiments so regressions are easier to interpret.
