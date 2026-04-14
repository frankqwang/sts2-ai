现在可以把整套流程分成两条线看：`主训练线` 和 `慢 teacher 线`。

**1. 主训练线**
主入口是 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:4902)。它做的事是：

1. 读配置，生成本次 run 目录和 `config.json`，见 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:5260)
2. 启动 `num_envs` 个环境，在线采样 episode
3. 每局通过 [collect_unified_episode(...)](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:2892) 同时收：
   - 非战斗 PPO 数据
   - 战斗 PPO / teacher 数据
   - replay summary
4. 每轮训练后更新模型、写 `metrics.jsonl`，其中离线锚点 loss 现在叫 `offline_noncombat_ranking_loss`，见 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:6804)
5. 按间隔存 checkpoint

你现在主线默认配置还是 [hybrid_train_ironclad_teacher_main_attention.toml](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:1)。它的关键默认值是：
- `transport = "pipe-binary"`，见 [line 11](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:11)
- `auto_launch = true`，见 [line 12](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:12)
- `num_envs = 4`，见 [line 14](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:14)
- `max_iterations = 500`，见 [line 19](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:19)
- `episodes_per_iter = 160`，见 [line 20](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:20)
- `saved_offline_episodes_enabled = true`，见 [line 59](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:59)
- `saved_offline_episodes_min_floor = 14`，见 [line 60](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:60)
- `offline_noncombat_ranking_data_dir = ""`，见 [line 65](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:65)
- `offline_noncombat_ranking_loss_weight = 0.0`，见 [line 67](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:67)
- `boss_entry_quality_weight = 0.0`，见 [line 93](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:93)

也就是说，**主线默认现在并没有开离线非战斗 ranking 回灌**。

**2. 主训练线里哪些参数真影响训练**
最重要的是这几类。

训练规模：
- `num_envs`
- `episodes_per_iter`
- `max_iterations`
- `episode_timeout`
- `max_episode_steps`

它们直接决定吞吐、训练时长、每轮样本量。

PPO / combat PPO：
- `ppo_lr / ppo_epochs / ppo_minibatch / target_kl`
- `combat_ppo_lr / combat_ppo_epochs / combat_ppo_minibatch / combat_target_kl`

它们直接影响优化强度和稳定性。

当前主线策略修正项：
- `act1_no_elite_routes`，进入 episode 时生效，见 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:2769)
- `boss_entry_quality_weight`，见 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:2782)
- `boss_conditioned_card_guidance_weight`，见 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:2784)
- `combat_safety_rerank_weight`，见 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:2785)

这些会直接改非战斗/战斗决策行为。

训练稳定项：
- `screen_value_heads`
- `per_screen_adv_norm`
- `weighted_screen_sampling`
对应 parser 在 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:5074)、[5077](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:5077)、[5080](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:5080)

这些不直接决定策略偏好，但会影响训练稳定性和不同 screen 的学习效率。

离线回灌项：
- `offline_noncombat_ranking_data_dir`
- `offline_noncombat_ranking_loss_weight`
- `offline_noncombat_ranking_updates_per_iter`
- `offline_noncombat_ranking_min_spread`

这组决定是否把离线非战斗 ranking 数据接进正式训练。内部兼容名还是 `matchup_*`，但对外应该用 `offline_noncombat_ranking_*`，见 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:132)

保存与分析项：
- `save_offline_data`
- `offline_min_floor`
- `save_replay_traces`
- `save_replay_structured`
见 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:5095)、[5102](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:5102)、[5108](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:5108)

这组不直接改变训练策略，但决定你后面有没有足够资产做 teacher loop。

**3. `offline_data/*.pt` 是什么**
这个不是模型。

训练里如果 `save_offline_data = true`，会创建 [EpisodeDataSaver](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:5331)，把较优局存到 `run_dir/offline_data`。默认只存 `floor >= 14` 或 victory 的局。它更像“优质 episode 资产库”，适合后验筛 seed、做 BC/teacher/挖 replay，不是直接的 ranking 数据。

**4. 慢 teacher 线**
这就是我们刚补出来的“窗口式异步 teacher loop”，文档在 [offline_noncombat_teacher_loop_2026-04-14.md](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/docs/offline_noncombat_teacher_loop_2026-04-14.md:1)。

流程是：

1. 先跑一段训练窗口
2. 从 `replays/*.summary.json` 里筛 seed
3. 只对这些 seed 跑 `card_reward` 短路线搜索
4. 和 baseline 对比，只收更优样本
5. 物化成一版 `accepted_dataset`
6. 下一训练窗口再把这版数据喂回去

具体脚本是：

- 队列构建：
[build_offline_noncombat_teacher_queue.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/build_offline_noncombat_teacher_queue.py:1)

- refresh worker：
[refresh_offline_noncombat_teacher.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/refresh_offline_noncombat_teacher.py:1)

- 离线生成器：
[generate_card_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_card_ranking_data.py:2698)

这条线当前默认 route 配置是 [offline_noncombat_teacher_route_default.toml](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/offline_noncombat_teacher_route_default.toml:1)，关键默认值是：
- `sample_types = "card_reward"`，见 [line 8](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/offline_noncombat_teacher_route_default.toml:8)
- `tree_route_search = true`，见 [line 10](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/offline_noncombat_teacher_route_default.toml:10)
- `rollout_goal = "terminal"`，见 [line 12](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/offline_noncombat_teacher_route_default.toml:12)
- `num_envs = 2`，见 [line 5](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/offline_noncombat_teacher_route_default.toml:5)
- `auto_launch = true`，见 [line 6](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/offline_noncombat_teacher_route_default.toml:6)

**5. 这条 teacher loop 里哪些参数影响结果**
队列筛选：
- `min_preboss_floor`
- `max_total`
- 每类桶的上限
这些决定“什么 seed 值得送去慢 teacher”。

refresh 门控：
- `min_floor_gain`
- `boss_damage_margin`
在 [offline_noncombat_teacher_loop_2026-04-14.md](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/docs/offline_noncombat_teacher_loop_2026-04-14.md:111) 之后有写。当前默认门槛是：
- `min_floor_gain = 2`
- `boss_damage_margin = 0.10`

route search 本身：
- `sample_types`
- `tree_route_search`
- `rollout_goal`
- `tree_max_reward_depth`
- `tree_beam_width`

这组决定 teacher 标签质量和速度，直接影响离线样本是否值得回灌。

**6. 现在完整训练流程的最好理解**
可以压成一句话：

- 主训练：便宜采样，持续产 checkpoint、metrics、replay、offline_data
- 慢 teacher：按窗口从 replay 挑 seed，route search 提纯 card_reward 标签，做 baseline 门控
- 下一窗口：把 accepted dataset 用小权重 `offline_noncombat_ranking_loss` 回灌进去

也就是：
`在线训练负责吞吐，离线 route search 负责高质量 build 标签，窗口边界做版本化回灌`

**7. 当前后台这条 loop smoke**
我刚挂起来的这条，就是为了验证这套闭环，不是正式主线。配置在 [hybrid_train_ironclad_teacher_main_attention_loop_smoke.toml](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_loop_smoke.toml:1)，关键值是：
- `num_envs = 2`，见 [line 9](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_loop_smoke.toml:9)
- `max_iterations = 2`，见 [line 14](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_loop_smoke.toml:14)
- `episodes_per_iter = 20`，见 [line 15](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_loop_smoke.toml:15)
- `offline_noncombat_ranking_loss_weight = 0.0`，见 [line 57](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_loop_smoke.toml:57)
- `boss_entry_quality_weight = 0.15`，见 [line 83](/C:/Users\Administrator\Desktop\sts2Raw2\STS2AI\Python\configs\hybrid_train_ironclad_teacher_main_attention_loop_smoke.toml:83)
- `boss_conditioned_card_guidance_weight = 0.8`，见 [line 85](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_loop_smoke.toml:85)
- `combat_safety_rerank_weight = 1.0`，见 [line 86](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention_loop_smoke.toml:86)

它现在只是“产第一批带 seed 的训练窗口”，后面我会直接拿它接 queue/refresh。

如果你要，我下一步可以再给你整理一版“参数分层表”，明确哪些参数：
- 改吞吐
- 改策略行为
- 改训练稳定性
- 改离线 teacher 标签质量