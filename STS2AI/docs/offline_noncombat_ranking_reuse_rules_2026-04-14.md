# 离线非战斗数据与复用约束

日期：2026-04-14

## 1. 当前有哪些“离线数据”

当前仓库里和训练相关的数据，按用途分成 4 类：

1. 训练期在线 rollout
   入口是 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:2891) 的 `collect_unified_episode(...)`，以及向量化收集器 [vectorized_collector.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/vectorized_collector.py:158)。
   这类数据直接用于当前 PPO / combat PPO / teacher 训练，不是独立离线数据集。

2. 训练期顺手保存的高质量 episode
   入口是 [EpisodeDataSaver](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/episode_data_saver.py:28)。
   当前主线默认会把 `floor >= 14` 或 victory 的 episode 存到 run 目录下的 `offline_data/`，配置见 [hybrid_train_ironclad_teacher_main_attention.toml](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:58)。
   这类数据适合后续做 BC / teacher / trajectory mining，不是现成的 ranking 数据。

3. 离线非战斗排序数据
   现在对外统一叫 `offline_noncombat_ranking`。
   训练入口在 [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:5284)。
   数据 loader 在 [offline_noncombat_ranking_dataset.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/offline_noncombat_ranking_dataset.py:1) 和 [matchup_dataset.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/matchup_dataset.py:1)。
   这类数据需要“同一 screen 下多个候选动作 + 每个候选的分数/排序”，是当前 `offline_noncombat_ranking_loss` 真正吃的格式。

4. Skada bridge 数据
   原始来源是 Skada 清洗后的 card_reward 数据。
   桥接入口现在对外统一叫 [build_offline_noncombat_ranking_from_skada.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/skada/build_offline_noncombat_ranking_from_skada.py:1)。
   旧名字 `build_matchup_ranking_from_skada.py` 只保留兼容。

## 2. 当前训练到底有没有在用这些离线数据

当前主线默认只稳定使用两类：

1. combat teacher 数据
   配置在 [hybrid_train_ironclad_teacher_main_attention.toml](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:74)。

2. 训练期保存的高质量 episode
   只是保存到磁盘，当前同一次训练不会自动回灌。

当前主线默认没有真正打开离线非战斗 ranking loss：

- `offline_noncombat_ranking_data_dir = ""`
- `offline_noncombat_ranking_loss_weight = 0.0`

见 [hybrid_train_ironclad_teacher_main_attention.toml](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/configs/hybrid_train_ironclad_teacher_main_attention.toml:64)。

## 3. 训练过程里的优质 clear 数据能不能积攒起来再用

可以，但要分用途。

### 3.1 可以直接拿来做什么

训练保存下来的 `offline_data/*.pt` 适合做：

1. 非战斗 BC / imitation
   因为它保存了每一步的编码后 state、候选 actions 张量、被选中的 action index，见 [episode_data_saver.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/episode_data_saver.py:46)。

2. combat teacher / trajectory mining
   因为它也保存了 combat states 与 chosen action index，见 [episode_data_saver.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/episode_data_saver.py:62)。

3. build 分析
   因为它能按 episode 聚合 floor、outcome、extra_stats，用来筛选 clear / 高楼层 / 某类 build。

### 3.2 不能直接当什么

它不能直接当当前 `offline_noncombat_ranking` 数据来用，原因很简单：

1. 它记录的是“实际选了什么”
2. 它没有记录“同一 screen 下其他候选项的后验分数”
3. ranking loss 吃的是 `scores / best_idx / option set`，不是单个 chosen action

所以：

- `offline_data/*.pt` 是很好的“优质行为语料”
- 但它不是现成的 ranking 数据

如果要让这批优质 clear 数据转成 ranking 锚点，需要额外做一层：

1. 先筛选出优质 episode
2. 抽出其中的 map / card_reward / shop / rest_site 决策点
3. 对这些决策点做后验重标注，得到候选排序分数

## 4. 当前离线非战斗 ranking 数据是怎么生成的

当前生成链路入口对外统一用：

- [generate_offline_noncombat_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_offline_noncombat_ranking_data.py:1)

它目前底层仍复用 [generate_card_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_card_ranking_data.py:1) 的实现。

这条链路不是训练在线 rollout 的简单落盘，而是：

1. 跑一条 episode
2. 在 map / card_reward 等关键 screen 截获决策点
3. 保存当前状态
4. 对每个候选动作分支 rollout
5. 计算候选分数
6. 写成 `ranking_sample.jsonl + tensors/*.npz`

也就是说，这条链路本质上是“带分支搜索/保存恢复的离线重标注”，不是普通训练采样。

## 5. 为什么现在会膨化出问题

核心原因不是“脚本太多”本身，而是“分支搜索逻辑和在线训练逻辑没有严格分层”。

训练主线已经有一套统一环境推进语义：

- action/state 兼容约束在 [full_run_action_semantics.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/runtime/full_run_action_semantics.py:343)
- 训练主采样在 [collect_unified_episode](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:2891)
- 向量化采样在 [collect_vectorized_episodes](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/vectorized_collector.py:158)

离线 generator 也已经开始复用其中一部分，比如：

- 复用了 [choose_rollout_decision(...)](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/runtime/full_run_action_semantics.py:343)
- 复用了统一的 deterministic screen 行为

但 generator 额外需要：

1. save/load/import/export snapshot
2. 多分支 rollout
3. reward tree / map tree 递归搜索

这一层目前还没有抽成共享“分支执行器”，于是 generator 自己带了一套“分支恢复后怎么继续推进”的逻辑，问题就容易集中暴露在这里。

## 6. 复用约束

从现在开始，后续相关脚本和实验都要遵守下面这些约束。

### 6.1 单一对外入口

只使用下面这些对外脚本名：

1. 训练： [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:1)
2. 最新模型派生离线非战斗 ranking： [generate_offline_noncombat_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_offline_noncombat_ranking_data.py:1)
3. Skada 桥接： [build_offline_noncombat_ranking_from_skada.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/skada/build_offline_noncombat_ranking_from_skada.py:1)

旧名字只保留兼容，不再作为文档和实验记录里的主名字：

1. `generate_card_ranking_data.py`
2. `build_matchup_ranking_from_skada.py`
3. `matchup_*`

### 6.2 单一状态推进语义

任何需要“从当前 state 继续往前走”的逻辑，都必须优先复用：

- [full_run_action_semantics.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/runtime/full_run_action_semantics.py:1)

禁止新增一份独立的 screen/action 状态机判断。

允许单独实现的只有：

1. save/load snapshot
2. 多分支搜索树
3. 分支分数聚合

但即使在这些逻辑里，screen/action 兼容判断也必须复用统一入口。

### 6.3 在线数据回灌策略

训练保存下来的高质量 episode，以后优先按两条线复用：

1. 先做“优质 episode 资产库”
   目标是 clear、高 floor、好 build、低明显失误。

2. 从这个资产库派生两类离线数据
   一类是 imitation / teacher 数据。
   一类是带后验重标注的 ranking 数据。

不要直接把原始在线 rollout 当成 ranking 数据硬塞给 `offline_noncombat_ranking_loss`。

### 6.4 后续建议的代码演进

后续如果继续做这条线，优先顺序如下：

1. 抽一个共享的 `branch_rollout_executor`
   负责 save/load/import/export、分支恢复、继续推进到下一个采样点。

2. 让 `generate_offline_noncombat_ranking_data.py` 只做 orchestration
   不再自己承载全部状态推进细节。

3. 给 `EpisodeDataSaver` 增加可选“关键 screen 原始快照”
   这样训练沉淀下来的优质 episode 以后就能直接重标注成 ranking 数据。

## 7. 当前结论

当前最值得做的不是再堆一个新脚本，而是把两条线分清楚：

1. 训练主线继续用统一采样器
2. 离线 ranking 生成只负责“分支重标注”
3. 训练产出的优质 clear/high-floor 数据先沉淀成资产库
4. 后续从资产库再派生 BC / teacher / ranking，不混概念

