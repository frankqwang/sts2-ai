"""修复 docstring：恢复原始详细英文注释，在第一行替换为中文摘要。"""
import os, re, subprocess, glob

# 中文第一行映射
zh_map = {
    "train_hybrid.py": "统一训练主循环：PPO + MCTS + Teacher 混合训练入口。",
    "evaluate_ai.py": "AI 评估主入口：固定种子跑多局游戏，输出胜率/层数等指标。",
    "demo_play.py": "实时可视化演示：连接游戏客户端，展示 AI 决策过程。",
    "train_combat_only.py": "战斗专项训练：跳过非战斗阶段，只训练战斗策略。",
    "test_training_smoke.py": "训练管线 smoke test：验证 import、网络前向、GAE、梯度流等。",
    "test_simulator_consistency.py": "模拟器一致性测试：对比 Sim 和 Spectator 后端行为。",
    "constants.py": "全局路径常量：checkpoint、数据集、产物目录等。",
    "verify_save_load.py": "存档读档验证：检验 save/load 在不同后端的一致性。",
    "sim_semantic_audit_common.py": "审计脚本共享库：diagnostics/ 下多个审计工具的公共函数。",
    "network/__init__.py": "网络架构包：CombatPolicyValueNetwork、FullRunPolicyNetworkV2、共享编码器。",
    "network/combat_network.py": "战斗策略-价值网络：CombatPolicyValueNetwork + CombatNNEvaluator。",
    "network/fullrun_policy.py": "全局策略网络：FullRunPolicyNetworkV2 + PPOTrainerV2。",
    "network/shared_encoders.py": "共享 NN 模块：EntityEmbeddings、SetEncoder、BilinearActionScorer 等。",
    "network/combat_features.py": "战斗特征工程：状态/动作特征构建 + 房间条件价值聚合。",
    "network/state_features.py": "全局状态特征工程：build_structured_state/actions + 实体特征编码。",
    "training/__init__.py": "训练基础设施包：PPO trainer、诊断、游戏决策、评估辅助。",
    "training/combat_ppo.py": "战斗 PPO 训练器：CombatRolloutBuffer + CombatPPOTrainer + mcts_train_step。",
    "training/combat_diagnostics.py": "战斗诊断日志：手牌/敌人摘要、动作分析、MCTS 可疑原因、中文 trace。",
    "training/game_decisions.py": "游戏决策逻辑：地图路线选择、卡牌奖励评估、商店购买策略。",
    "training/eval_action_selection.py": "推理动作选择：NN/Teacher/MCTS 策略路由 + 致死检测。",
    "training/eval_game_state.py": "游戏状态追踪：进度提取、循环检测、自动推进、奖励领取。",
    "training/combat_safety.py": "战斗安全遮罩：规则化安全检查 + logits 重排序。",
    "training/episode_data_saver.py": "回合数据存储：保存高质量离线 RL 轨迹到磁盘。",
    "training/training_health.py": "训练健康监控：检测异常指标（loss 爆炸、KL 过大等）。",
    "training/heuristic_combat.py": "规则战斗策略：基于手工规则的战斗动作选择（fallback）。",
    "training/rl_segment_buffer.py": "半 MDP 段缓冲区：非战斗 segment 的 GAE 计算和存储。",
    "training/segment_collector.py": "非战斗段收集器：从 episode 中提取非战斗训练段。",
    "training/vectorized_collector.py": "向量化数据收集：多环境并行 episode 收集。",
    "env/__init__.py": "游戏环境接口包：模拟器通信、后端适配、推理服务。",
    "env/full_run_env.py": "游戏环境客户端：HTTP（Godot）和 Pipe（无头模拟器）两种后端统一接口。",
    "env/full_run_backend.py": "后端适配层：屏蔽 HTTP/Pipe 差异，提供统一的状态推进接口。",
    "env/binary_pipe_client.py": "二进制管道通信：与 C# 无头模拟器的高性能二进制协议。",
    "env/pipe_client.py": "JSON 管道通信：与 C# 模拟器的 JSON 文本协议（调试用）。",
    "env/headless_sim_runner.py": "无头模拟器启动器：管理 headless_sim_host 进程的启动和停止。",
    "env/sim_host_lifecycle.py": "模拟器生命周期管理：多进程环境下的进程池和端口分配。",
    "env/combat_training_env.py": "战斗训练环境：封装单场战斗的 step/reset 接口。",
    "env/sts2_singleplayer_env.py": "单人游戏环境：完整一局游戏的 Gym-like 接口。",
    "env/inference_server.py": "GPU 推理服务器：多 worker 共享的批量推理守护线程。",
    "env/simulator_api_error.py": "模拟器错误定义：API 异常类。",
    "env/action_semantics.py": "动作语义：合法动作的类型判断和自动推进规则。",
    "env/run_outcome_vocab.py": "结局词表：胜利/死亡/放弃等结局类型的标准化。",
    "core/__init__.py": "核心工具包：词表、标签、奖励塑形等基础设施。",
    "core/vocab.py": "词表管理：卡牌/遗物/药水/怪物的 ID <-> 索引映射。",
    "core/card_tags.py": "卡牌功能标签：从源码提取的 32 维功能标签（攻击/防御/抽牌等）。",
    "core/relic_tags.py": "遗物功能标签：遗物效果分类标签。",
    "core/card_base_stats.py": "卡牌基础属性：手工维护的伤害/格挡/命中数查找表（Ironclad）。",
    "core/checkpoint_compat.py": "检查点兼容：不同版本 checkpoint 的加载和转换。",
    "core/full_run_agent.py": "全局运行代理：组装 PPO + Combat 网络的推理 agent。",
    "core/rl_reward_shaping.py": "奖励塑形：PBRS 势函数 + 里程碑奖励 + 战斗局部奖励。",
    "core/source_knowledge_features.py": "源码知识特征：从 source_knowledge.sqlite 提取实体符号特征。",
    "core/symbolic_features_head.py": "符号特征头：sqlite-backed cross-attention，给稀有实体提供零样本先验。",
    "search/__init__.py": "搜索和求解包：MCTS、回合求解器、Teacher 数据构建。",
    "search/mcts_core.py": "MCTS 核心：蒙特卡洛树搜索的节点、选择、扩展、回溯。",
    "search/combat_mcts_agent.py": "战斗 MCTS 代理：包装 MCTS 搜索为战斗动作选择器。",
    "search/combat_turn_planner.py": "回合规划器：基于前瞻搜索的单回合最优动作序列。",
    "search/combat_turn_solver.py": "回合求解器：穷举搜索最优回合动作序列。",
    "search/multi_turn_solver_planner.py": "多回合求解器：跨回合的前瞻规划。",
    "search/turn_solver_planner.py": "求解器规划器：集成回合求解和 NN 价值估计。",
    "search/counterfactual_scoring.py": "反事实评分：比较实际动作和替代动作的价值差异。",
    "search/combat_teacher_common.py": "Teacher 公共库：战斗 teacher 数据集的共享工具函数。",
    "search/combat_teacher_dataset.py": "Teacher 数据集：战斗 teacher 训练数据的加载和采样。",
    "search/combat_turn_teacher_config.py": "Teacher 配置：战斗 teacher 的超参数和模式设置。",
    "search/train_combat_teacher.py": "Teacher 训练：离线战斗 teacher 的 loss 计算和训练循环。",
    "search/ranking_loss.py": "排序损失函数：pairwise ranking loss。",
    "search/matchup_dataset.py": "对局数据集：离线非战斗排序训练数据的加载。",
    "search/offline_noncombat_ranking_dataset.py": "离线非战斗排序数据集。",
    "search/boss_leaf_evaluator.py": "Boss 叶节点评估器：MCTS 叶节点的 boss 战专用价值估计。",
    "search/skada_noncombat_priors.py": "Skada 非战斗先验：基于人类数据的卡牌/商店/休息决策先验。",
    "tools/__init__.py": "工具脚本包：导出、审计、演示等非主线工具。",
    "tools/export_actor_onnx.py": "ONNX 导出：将战斗网络导出为 ONNX 供 C# ORT 推理。",
    "tools/combat_turn_trace.py": "战斗回合 trace：记录和对比不同后端的逐步战斗状态。",
    "tools/nn_backend_parity_audit.py": "NN 后端一致性审计：对比不同后端的神经网络输出。",
    "tools/nn_hooks.py": "NN 钩子：PyTorch forward hook，用于可视化网络内部状态。",
    "tools/training_monitor.py": "训练监控：通过 WebSocket 推送训练指标到可视化面板。",
    "tools/demo_action_candidates.py": "演示动作候选：为可视化演示提取和格式化动作候选。",
    "tools/public_state_trace.py": "公共状态 trace：记录游戏状态的公开信息用于调试。",
    "tools/eval_llm_noncombat.py": "LLM 非战斗评估：评估 LLM 策略在非战斗决策上的表现。",
    "tools/saveload_combat_parity.py": "存档一致性：对比存档/读档前后的战斗状态。",
    "tools/train_bc_noncombat.py": "行为克隆训练：用人类数据训练非战斗策略。",
    "tools/train_llm_policy.py": "LLM 策略训练：训练基于 LLM 的游戏策略。",
}

updated = 0
for rel_path, zh_line in zh_map.items():
    path = rel_path.replace("/", os.sep)
    if not os.path.exists(path):
        continue

    content = open(path, encoding="utf-8").read()

    # Case 1: file starts with """xxx""" (single-line docstring we added)
    # Replace ONLY the first line inside the triple quotes
    m = re.match(r'^"""(.+?)"""', content)
    if m:
        old_first = m.group(1)
        # Check if it's already our Chinese one-liner (from previous run)
        # Restore the full English docstring from git, then prepend Chinese
        try:
            git_content = subprocess.run(
                ["git", "show", "HEAD~1:" + "STS2AI/Python/" + rel_path.replace(os.sep, "/")],
                capture_output=True, text=True, encoding="utf-8", errors="replace"
            ).stdout
            if git_content:
                # Extract original docstring
                gm = re.match(r'^(""".*?""")', git_content, re.DOTALL)
                if gm:
                    orig_doc = gm.group(1)
                    # Replace first line of original docstring with Chinese
                    orig_lines = orig_doc.split("\n")
                    # orig_lines[0] is like '"""Combat neural network — ...'
                    orig_first = orig_lines[0]
                    new_first = '"""' + zh_line
                    orig_lines[0] = new_first
                    new_doc = "\n".join(orig_lines)
                    # Replace in content
                    new_content = content.replace(m.group(0), new_doc, 1)
                    open(path, "w", encoding="utf-8").write(new_content)
                    updated += 1
                    continue
        except Exception:
            pass

    # Case 2: file has multi-line """...""" docstring already (wasn't replaced)
    m2 = re.match(r'^("""[^\n]*)', content)
    if m2:
        old_first = m2.group(1)
        new_first = '"""' + zh_line
        if old_first != new_first:
            new_content = content.replace(old_first, new_first, 1)
            open(path, "w", encoding="utf-8").write(new_content)
            updated += 1
            continue

    # Case 3: no docstring at all — insert one
    # (skip, these files already got one-liner from previous run)

print(f"Updated {updated} files: Chinese first line + preserved English detail")
