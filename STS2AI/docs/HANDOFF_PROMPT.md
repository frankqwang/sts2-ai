# STS2AI Training Handoff Prompt

当前工作目录：

```text
C:\Users\Administrator\Desktop\sts2Raw2
```

当前分支与提交：

```text
branch: codex/training-framework-staging
latest commit: 3d1fafa Relocate training artifacts under STS2AI
```

当前重点约束：

- 所有 AI 相关默认输出、关键数据、关键模型尽量都放在 `STS2AI/` 下面，自包含。
- 其他目录能不改就不改。
- 后面默认以 `STS2AI/Artifacts` 为训练、评测、teacher、Skada 桥接产物根目录。

请先确认这些事实，不要重新分析一遍：

1. `combat_main_path_mode` 已支持 `mlp` / `light_attention`。
2. `offline_noncombat_ranking_head_mode` 已支持 `mlp` / `light_attention` / `transformer`。
3. 主 combat rollout 路径的 `light_attention` 在 `train_hybrid.py` 的正确 PPO 训练里，`50-seed` A/B 是正向的：
   - 平均楼层约 `+1.4`
   - `boss_reach` 约 `16% -> 34%`
   - 但 `timeout` 变多，仍需排查 boss/repeat loop。
4. non-combat ranking 头的 `transformer`：
   - 短跑和 focused 100 都没打过 `mlp`
   - 从 scratch 100 iter 冷启动时比 `mlp` 稍好，但整体都还很弱
   - 当前没有证据支持把 non-combat ranking 头全面切到 transformer。
5. `boss_readiness` 相关信号当前偏弱：
   - `boss_readiness_coeff` 现值仍偏小
   - `boss_entry_quality_weight` 还没真正打开
   - 这条 boss-aware shaping 后续值得继续加强。

当前已完成的整理：

- 根目录被 Git 跟踪的关键 `artifacts/...` 产物已经迁到：

```text
STS2AI/Artifacts/...
```

- 当前已经迁入并提交的关键产物包括：
  - `STS2AI/Artifacts/combat_teacher/ironclad_act1_solver_v2_dataset_320.jsonl`
  - `STS2AI/Artifacts/combat_teacher/ironclad_act1_solver_v2_dataset_2000_balanced.jsonl`
  - `STS2AI/Artifacts/combat_teacher/trace_safety_teacher_dataset.jsonl`
  - `STS2AI/Artifacts/skada/ironclad_matchup_bridge/manifest.json`
  - `STS2AI/Artifacts/hybrid_ab_mainpath_mlp/.../hybrid_final.pt`
  - `STS2AI/Artifacts/hybrid_ab_mainpath_light_attention/.../hybrid_final.pt`
  - `STS2AI/Artifacts/hybrid_ab_noncombat_ranking_transformer_scratch_100/.../hybrid_final.pt`

- `STS2AI/Assets/datasets/skada/skada_analytics.sqlite` 已纳入 Git LFS。
- `.gitattributes` 现在也覆盖了 `STS2AI/Artifacts/**/*.pt`。
- 当前分支已经成功 push 到远端。

最近刚改完的路径/框架相关文件，优先从这些地方继续看：

- `STS2AI/Python/train_hybrid.py`
- `STS2AI/Python/core/combat_nn.py`
- `STS2AI/Python/core/rl_policy_v2.py`
- `STS2AI/docs/NETWORK_AND_TRAINING_OVERVIEW.md`
- `STS2AI/docs/TRAINING_DATA_FLOW.md`
- `STS2AI/docs/configs/hybrid_train_ironclad_teacher.toml`
- `STS2AI/Python/configs/hybrid_train_ironclad_noncombat_ranking*.toml`
- `STS2AI/Python/search/build_act1_combat_teacher_v2_dataset.py`
- `STS2AI/Python/search/train_combat_teacher.py`
- `STS2AI/Python/search/skada_noncombat_priors.py`

本轮已经验证过：

```text
python -m py_compile STS2AI/Python/train_hybrid.py STS2AI/Python/core/rl_policy_v2.py
pytest STS2AI/Python/test_training_smoke.py -q -k "matchup_head_mode_toggles_attention_params or train_hybrid_offline_noncombat_ranking_head_mode_toggles_attention_params"
```

下一步最建议做的事：

1. 先排查 `combat_main_path_mode = light_attention` 下 `timeout` 上升的原因。
   - 抓 timeout 对应 seed 的 trace
   - 看是不是 `repeat_loop / boss loop / bad_end_turn` 被放大了
2. 在 `light_attention` 主 combat 路径上继续做更长一点的 hybrid 训练，看 `act1_clear_rate` 能不能从 `0` 抬起来。
3. 不要继续优先烧 non-combat transformer 长跑。
   - 这条线当前证据不强
   - 除非先把 ranking 数据上下文做厚
4. 如果要继续加强 boss-aware 信号，优先看：
   - `boss_readiness_coeff`
   - `boss_entry_quality_weight`
   - boss/elite 高压样本密度
5. 保持默认输出继续落在 `STS2AI/Artifacts`，不要重新掉回根目录 `artifacts`。

如果要继续工作，可以直接按下面这段 prompt 开新会话：

---

请在 `C:\Users\Administrator\Desktop\sts2Raw2` 继续工作，不要重新从头分析。

当前分支：
- `codex/training-framework-staging`

当前最新提交：
- `3d1fafa Relocate training artifacts under STS2AI`

请先确认：
- AI 相关默认输出现在应该统一放在 `STS2AI/Artifacts`
- `combat_main_path_mode = light_attention` 是当前更值得继续推进的主线
- non-combat ranking transformer 当前没有证明比 mlp 更好

优先任务：
1. 查 `light_attention` 主 combat 路径在 50-seed A/B 里 timeout 变多的具体原因。
2. 抓这些 timeout seed 的 trace，并判断是不是 boss/repeat loop 被放大。
3. 在不破坏当前自包含目录结构的前提下，继续推进主 combat 路径训练和评测。

继续之前先做：
- `git status`
- `git branch -vv`
- 快速检查 `STS2AI/Artifacts` 和当前 configs 的默认路径

不要：
- 不要重新把训练产物默认写回根目录 `artifacts`
- 不要优先继续 non-combat transformer 长跑
- 不要大面积碰 archive/diagnostics 等非当前主线目录

---
