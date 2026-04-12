# Network And Training Overview

This note is the "what is actually training right now" companion to
`TRAINING_DATA_FLOW.md`.

It focuses on three things:

1. Which networks exist in the current hybrid stack.
2. Which parts are MLP, light attention, or transformer-style.
3. Which training loop updates which part of the system.

The goal is to keep future experiments easy to reason about.

## High-Level Split

There are two main brains in the current stack:

- `FullRunPolicyNetworkV2`
  - File: `STS2AI/Python/core/rl_policy_v2.py`
  - Role: non-combat / full-run PPO brain
  - Handles map, card reward, shop, campfire, events, and shared trunk/value logic

- `CombatPolicyValueNetwork`
  - File: `STS2AI/Python/core/combat_nn.py`
  - Role: combat policy/value brain
  - Handles in-combat action scoring and combat value estimation

There are also two training helpers:

- `PPOTrainerV2`
  - File: `STS2AI/Python/core/rl_policy_v2.py`
  - Updates the full-run PPO brain

- `CombatPPOTrainer`
  - File: `STS2AI/Python/train_hybrid.py`
  - Updates the combat policy/value brain from online combat PPO data

And one separate offline fine-tuning route:

- `train_combat_teacher.py`
  - File: `STS2AI/Python/search/train_combat_teacher.py`
  - Best treated as a combat teacher / reranker trainer
  - Not the right tool for directly replacing the main rollout combat brain

## Current Default Sizes

These are the dimensions used by the recent IRONCLAD scratch and A/B configs.

- Shared embedding size: `embed_dim = 48`
- Combat hidden size: `combat_hidden_dim = 192`
- Deck representation size: `deck_repr_dim = 64`
- Retrieval / symbolic projection size: `retrieval_proj_dim = 16`
- PPO trunk hidden size: `trunk_hidden = 256`
- PPO trunk output size: `trunk_output = 128`
- PPO screen head size: `screen_head_dim = 128`
- Attention heads: `num_attn_heads = 4`

Important practical reading:

- This is not a giant LLM-style stack.
- The model is still a structured RL model with small attention blocks.
- Most experiments so far change one head or one path at a time, not the whole system.

### Approximate Parameter Counts

For the recent `embed_dim=48 / combat_hidden_dim=192 / deck_repr_dim=64` family:

- Full-run PPO brain: about `1.36M` parameters
- Combat brain: about `1.34M` parameters
- Combined hybrid stack: about `2.70M` parameters

Useful reference points from the same family:

- `boss_readiness_head`: about `11.2K`
- PPO `matchup_score_head`: about `41.2K`
- Offline non-combat ranking transformer block: about `265K`
- Combat `policy_scorer`: about `36.9K`
- Combat `value_head`: about `18.6K`
- Combat `action_score_head` (teacher scorer): about `74.1K`
- Combat main-path attention block: about `148K` attention + `74K` FFN

This is why the project can still iterate quickly:

- the model is structured and compact
- the "transformer" experiments so far are local head/path upgrades
- we are not training a giant decoder model

## Full-Run PPO Brain

Class:

- `FullRunPolicyNetworkV2`

Main structure:

1. Entity embeddings
2. Set encoders for deck / relics / potions / hand / enemies
3. Shared trunk MLP
4. Screen-specific heads
5. Value heads and auxiliary heads

Key modules:

- `entity_emb`
  - Shared entity embeddings

- `deck_encoder`, `relic_encoder`, `potion_encoder`
  - Set encoders for build state

- `hand_encoder`, `enemy_encoder`
  - Combat-aware context that also feeds the full-run trunk

- `trunk`
  - Shared MLP over scalars + pooled set representations

- `map_head`, `card_reward_head`, `shop_head`, `rest_head`, `event_head`, `combat_head`
  - Screen-specific policy context builders

- `value_heads`
  - Screen-specific value heads

- `deck_quality_head`
  - Auxiliary build-quality predictor

- `boss_readiness_head`
  - Boss-conditioned build-readiness auxiliary head

### Boss Readiness Head

This head predicts a boss-aware readiness score from:

- deck/build representation
- scalar state context
- boss identity embedding

It is an auxiliary head, not the direct action head.

Current practical limitation:

- The raw loss often becomes very small.
- With `boss_readiness_coeff = 0.05`, its final influence can become almost negligible.
- That means it is a weak bias, not a dominant teacher.

## Combat Brain

Class:

- `CombatPolicyValueNetwork`

Main structure:

1. Hand encoder
2. Enemy encoder
3. Scalar state encoder
4. Shared combat state representation
5. Policy scorer
6. Value head
7. Teacher-only auxiliary heads

Key modules:

- `hand_encoder`
- `enemy_encoder`
- `state_encoder`
- `policy_scorer`
  - Main rollout combat action scorer

- `value_head`
  - Main rollout combat value head

- `action_score_head`
  - Teacher / rerank score head

- `continuation_value_head`
  - Teacher continuation targets

### Main Combat Path Mode

Config:

- `combat_main_path_mode`

Supported modes:

- `mlp`
  - Legacy main rollout path

- `light_attention`
  - Residual attention enhancement on the main combat rollout path

Important:

- This changes the real combat `policy/value` path used during rollout.
- It should be evaluated in `train_hybrid.py`, not only in `train_combat_teacher.py`.

### Teacher Scorer Path

The combat network also has a separate teacher/rerank path.

This path is useful for:

- offline teacher scoring
- correction / override experiments
- action reranking

It is not the same thing as the main rollout combat brain.

## MLP vs Light Attention vs Transformer

These names do not all mean the same scope.

### `mlp`

Meaning:

- The legacy scorer/path stays as simple MLP-style projection + scoring

Used in:

- combat main path
- offline non-combat ranking head
- teacher scorer baseline

### `light_attention`

Meaning:

- Add a small residual attention block on top of the legacy representation
- Keep the old representation intact and learn a correction

Why this is used:

- Lower risk than replacing the whole path
- Better checkpoint compatibility
- Easier A/B attribution

Used in:

- combat main rollout path
- offline non-combat ranking head
- teacher scorer experiments

### `transformer`

Meaning in current code:

- A heavier transformer-style residual context scorer for offline non-combat ranking
- More complete attention/FFN style interaction than `light_attention`

Important:

- In current experiments, `transformer` has only been used on the
  `offline_noncombat_ranking_head_mode` path.
- It does not mean the whole game brain is a transformer.

## What Is Actually Transformer Right Now

This is the most common source of confusion.

If a config says:

- `combat_main_path_mode = "mlp"`
- `offline_noncombat_ranking_head_mode = "transformer"`

Then the actual meaning is:

- Combat main rollout brain: still MLP
- Offline non-combat ranking scorer: transformer-style

That does **not** mean "combat and non-combat are both transformer now".

If combat is running with attention, that is controlled separately by:

- `combat_main_path_mode = "light_attention"`

## Training Loops

### 1. `train_hybrid.py`

This is the main end-to-end training loop.

It combines:

- online non-combat PPO rollouts
- online combat PPO rollouts
- optional offline non-combat ranking updates
- optional offline combat teacher updates
- optional saved offline episode recording

This is the correct place to evaluate:

- combat main path structure changes
- end-to-end hybrid behavior changes

### 2. `train_combat_teacher.py`

This is a specialized offline combat teacher trainer.

It is best used for:

- action rerank heads
- teacher checkpoints
- correction modules

It is **not** the right way to judge a new main rollout combat path structure.

Why:

- It trains on selected teacher states
- It does not optimize the full online rollout distribution
- It does not replace PPO-style long-horizon learning

## Offline Data Layers

There are three separate concepts.

### `saved_offline_episodes`

Meaning:

- Raw episode recordings written during hybrid training
- Output by `EpisodeDataSaver`
- Stored under each run's `offline_data` directory

Important:

- These do not automatically feed back into the same training run
- They are raw material for later dataset building

### `offline_noncombat_ranking`

Meaning:

- Cleaned ranking datasets for card reward / shop / campfire style supervision

Current example:

- `artifacts/skada/ironclad_matchup_bridge`

This data is loaded explicitly by `train_hybrid.py` and used every iteration.

### `offline_combat_teacher`

Meaning:

- Cleaned combat teacher datasets

Current examples:

- `artifacts/combat_teacher/ironclad_act1_solver_v2_dataset_320.jsonl`
- `artifacts/combat_teacher/ironclad_act1_solver_v2_dataset_2000_balanced.jsonl`

This data is also loaded explicitly and trained every iteration when enabled.

## Why Offline Recording Is Not The Same As Offline Training

This is another common point of confusion.

When `saved_offline_episodes_enabled = true`:

- hybrid training records useful episodes to disk
- but those files are not automatically re-read into the same run

So:

- recording is on
- storage is happening
- but direct training only happens from datasets explicitly passed into:
  - `offline_noncombat_ranking_data_dir`
  - `offline_combat_teacher_data_dir`

## Current Practical Takeaways

### What has already shown positive signal

- `combat_main_path_mode = light_attention`
  - when trained through `train_hybrid.py`
  - showed positive A/B movement in prior combat-focused experiments

### What has not yet shown reliable gains

- Heavier transformer-style offline non-combat ranking head
  - especially when resumed from a mature base

### What is currently easy to misunderstand

- `combat_ppo` is a training method name, not a network structure name
- `transformer` in current configs usually means only the offline non-combat ranking head
- `boss_readiness_loss` can look present in logs while contributing very little overall gradient

## Monitoring That Actually Helps Tuning

The following signals are the ones most worth watching during experiments.

### Core outcome metrics

- `avg_floor`
- `boss_reach_rate`
- `act1_clear_rate`
- `boss_hp_fraction_dealt_mean`

These tell us whether the system is getting stronger in the only sense that
matters: surviving farther and converting more runs into real boss attempts.

### PPO stability signals

- `ppo_entropy`
  - Lower means more certainty, but not automatically "better"
  - Dangerous when floor is flat and entropy keeps collapsing

- `ppo_clip_fraction`
  - Fraction of PPO samples currently hitting the clip boundary
  - High values suggest updates are pushing too hard relative to the old policy

- `ppo_ratio_mean`
  - Average PPO importance ratio
  - Useful to tell whether the policy is drifting too aggressively from the
    rollout policy

- `ppo_ploss`, `ppo_vloss`
  - Useful, but should be interpreted together with floor, entropy, and clip
    fraction
  - Policy loss sign alone is not a reliable "good/bad" indicator

### Boss-awareness signals

- `boss_readiness_loss`
  - Raw auxiliary MSE

- `boss_readiness_weighted`
  - Actual contribution after multiplying by `boss_readiness_coeff`

The weighted version is the important one for diagnosis. A raw loss that looks
"present" can still be functionally irrelevant after weighting.

### Structure-learning signals

- `offline_ranking_action_context_gate`
- `offline_ranking_state_context_gate`
- `combat_main_action_context_gate`
- `combat_main_state_context_gate`
- `combat_teacher_action_context_gate`

These tell us whether residual attention branches are actually being used.

If a branch is enabled but the gate stays near zero:

- the structure is present
- but training has not learned to rely on it yet

### Offline-data influence signals

- `matchup_rank_loss`
- `combat_teacher_loss`
- dataset sizes in `training_sources.json`
- update counts in `training_flow.md`

These are crucial because "having offline data" is not the same thing as
"offline data is materially influencing this run".

## Current Tuning Judgement

Based on recent experiments:

- Combat main-path `light_attention` is a real positive-signal direction.
- Offline non-combat transformer ranking has not yet shown stronger end-to-end
  returns than simpler MLP baselines.
- `boss_readiness_coeff = 0.05` often makes boss-readiness supervision too weak
  to matter once the raw loss becomes tiny.
- Flat floors with falling entropy are a warning sign, but not enough on their
  own; clip fraction and real outcome metrics should be checked at the same time.

## Recommended Reading Order

If you are orienting yourself in this codebase, read in this order:

1. `TRAINING_DATA_FLOW.md`
2. `NETWORK_AND_TRAINING_OVERVIEW.md` (this file)
3. `core/rl_policy_v2.py`
4. `core/combat_nn.py`
5. `train_hybrid.py`

That sequence is the fastest way to keep "which data trains what" clear in your head.
