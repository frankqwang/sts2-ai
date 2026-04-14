# 阶段性 Review 摘要（2026-04-14 / teacher loop phase1）

## 这次改动主要在做什么

这批改动围绕两条主线：

1. 修通 full-run reward flow 的保存/恢复链路，保证 `save_state/load_state` 与 `export_state/import_state` 能在 `combat_rewards/card_reward` 这类尾声态上稳定 roundtrip。
2. 搭建一条可异步运行的 offline non-combat teacher loop：训练窗口产 seed，慢速 route-search 重标注 `card_reward`，门控后产出 `offline_noncombat_ranking` 数据，再回灌到正式训练做小权重实验。

当前状态是：

- 工程闭环已经打通。
- 单 seed 上，`card_reward` 短路线搜索相对单点贪心有正例。
- 端到端训练收益暂时还没有赢 baseline。

## 代码改动分组

### 1. Reward flow / restore 修复

核心文件：

- [FullRunSimulatorRuntimeFacade.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulatorRuntimeFacade.cs:1)
- [FullRunSimulationStateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulationStateBuilder.cs:1)
- [FullRunApiStateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunApiStateBuilder.cs:1)
- [FullRunSimulationSerializerContext.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/Overlay/Simulation/FullRunSimulationSerializerContext.cs:1)
- [verify_save_load.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/verify_save_load.py:1)

重点：

- 把旧 `combat_pending` 拆成 `combat_start_pending` / `combat_post_end_pending`
- restore 等待语义与 exact snapshot 校验统一到 reward flow 口径
- `card_reward` 的 restore 走 overlay 自己的解析链，而不是回错到 `combat_rewards`
- `export/import` 增补专用 serializer context

对应问题文档：

- [reward_flow_restore_and_offline_branching_2026-04-14.md](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/docs/problems/reward_flow_restore_and_offline_branching_2026-04-14.md:1)

### 2. 共享语义层与 host 生命周期收口

核心文件：

- [full_run_action_semantics.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/runtime/full_run_action_semantics.py:1)
- [sim_host_lifecycle.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/ipc/sim_host_lifecycle.py:1)
- [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:1)
- [generate_card_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_card_ranking_data.py:1)

重点：

- 把“当前 state 下哪些动作合法”的保护逻辑集中到共享语义层
- 给训练主线和离线生成统一补 `auto_launch` / host lifecycle 管理
- 避免离线脚本因为没起 host 或状态漂移而各自补一套推进逻辑

### 3. Offline non-combat ranking 命名收口

核心文件：

- [generate_offline_noncombat_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_offline_noncombat_ranking_data.py:1)
- [offline_noncombat_ranking_dataset.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/offline_noncombat_ranking_dataset.py:1)
- [build_offline_noncombat_ranking_from_skada.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/skada/build_offline_noncombat_ranking_from_skada.py:1)
- [matchup_dataset.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/matchup_dataset.py:1)
- [train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:132)

重点：

- 对外入口统一叫 `offline_noncombat_ranking`
- 内部还保留 `matchup_*` 兼容名，但不再作为对外推荐口径
- 训练日志里的 `matchup_rank_loss` 已改成 `offline_noncombat_ranking_loss`

### 4. Card reward route-search teacher

核心文件：

- [card_reward_tree.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/card_reward_tree.py:1)
- [generate_card_ranking_data.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/generate_card_ranking_data.py:1)
- [render_offline_noncombat_branch_report.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/diagnostics/render_offline_noncombat_branch_report.py:1)

重点：

- `card_reward` 分支评估从固定 combat horizon 改到 `terminal`
- 增加终局 tie-break
- 增加短路线搜索（向后看未来若干次 `card_reward`）
- 衍生出可视化 branch report，用来分析单 seed 的 build 分岔

对应分析文档：

- [ONCRVIS_00000_route_search_analysis_2026-04-14.md](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/docs/ONCRVIS_00000_route_search_analysis_2026-04-14.md:1)

### 5. 异步 offline teacher loop

核心文件：

- [build_offline_noncombat_teacher_queue.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/build_offline_noncombat_teacher_queue.py:1)
- [build_offline_noncombat_teacher_queue_multi.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/build_offline_noncombat_teacher_queue_multi.py:1)
- [refresh_offline_noncombat_teacher.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/search/refresh_offline_noncombat_teacher.py:1)
- [run_outcome_vocab.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/runtime/run_outcome_vocab.py:1)

重点：

- 从训练窗口 summary 里抽高价值 seed
- route-search 只做 `card_reward`
- 通过 baseline 对比做门控
- 通过门控的样本落到版本化 `accepted_dataset`
- 再由训练显式指定某一版数据回灌

对应说明文档：

- [offline_noncombat_teacher_loop_2026-04-14.md](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/docs/offline_noncombat_teacher_loop_2026-04-14.md:1)

## 这次 review 最值得看的点

### A. Reward flow restore 是否还存在隐藏状态漂移

重点看：

- `combat_rewards/card_reward` 的 restore 语义
- `combat_post_end_pending` 与 reward screen 的 comparable state 归一
- `verify_save_load.py` 的 exact roundtrip 校验是否覆盖到了关键页面

### B. Card reward route-search 的 score 口径是否足够合理

重点看：

- `terminal` 终局打分和 tie-break 是否仍然过粗
- 短路线搜索的 beam/depth/default 配置是否合适
- 当前 route-search 是否只是在做 seed-specific 改善，还是具备泛化价值

### C. Teacher loop 的门控是否太严或太松

重点看：

- 候选 seed 的筛选桶
- baseline 对比门槛
- accepted sample 当前通过率低，到底是 teacher 质量问题还是门控过严

### D. 小样本回灌为什么没有转正

当前现象：

- 闭环已通
- 但两轮 A/B 都没赢 baseline

重点看：

- `accepted sample` 量级太小
- `offline_noncombat_ranking_loss` 实际梯度影响太弱
- route-teacher 数据是否和当前 PPO 主线分布偏差过大

## 当前阶段性结论

这批改动更接近“把 offline teacher 旁路搭起来，并把 reward flow restore 修到可用”，不是“已经验证 teacher 回灌对训练有稳定收益”。

目前最稳的判断是：

- restore/replay/route-search 相关工程基础已经比昨天完整很多
- `card_reward` 短路线搜索有单 seed 正例
- 端到端训练收益还需要更多高质量 accepted sample 才值得继续判断

## 建议的 review 顺序

1. 先看 restore 与 snapshot 相关改动是否自洽
2. 再看 `card_reward` route-search score 和搜索边界
3. 最后看 teacher loop 的筛选、门控、回灌接口

## 当前不建议 reviewer 花太多时间的点

- 不要把当前 route-teacher A/B 结果当成最终算法结论
- 不要现在就把所有历史 teacher 数据做成累计总库
- 不要现在就把 route-search 直接并进在线训练采样

更准确的定位是：

- 这是一版“工程与实验基础设施阶段性可用”的提交
- 还不是“teacher 数据已经证明有效”的最终方案
