# 踩坑记录

历史 bug 和修复记录，供排查类似问题时参考。

---

### Worktree .godot 缓存不完整
**症状**：Godot headless 启动后 HTTP/Pipe 不响应，日志有 `GameStartupError`。
**原因**：setup-worktree.ps1 旧版用 `Copy-Item` 漏复制 `.fontdata`/`.mp3str` 文件。
**修复**：改用 `robocopy /MIR` 完整镜像。

### MCTS save/load 后 target_id 不一致
**症状**：`Card 'DEFEND_IRONCLAD' does not accept a target` 等大量 pipe step 错误。
**原因**：MCTS 树中存储的 action 的 target_id 在 load_state 后与 server 端不匹配。
**修复**：`_reconcile_action()` 在 step 前用 server 当前 legal_actions 修正 target。

### policy_loss = NaN
**症状**：训练 loss 出现 NaN。
**原因**：空 action_mask → all-`-inf` logits → `log_softmax` 产生 NaN。
**修复**：`af["action_mask"].any()` 过滤空 mask + `logits.clamp(min=-30)` 防止 `-inf`。

### Pipe 64位 Windows INVALID_HANDLE_VALUE 比较错误
**症状**：pipe read 永久阻塞（~30 步后卡死）。
**原因**：`ctypes.c_void_p(-1).value` = unsigned 64-bit，`CreateFileW` 返回 signed -1，比较永不相等。
**修复**：直接比较 `handle == -1`。改用 overlapped I/O 实现 read timeout。

### Pipe session lock 不释放
**症状**：pipe timeout 后所有后续连接返回 `simulator_busy`。
**原因**：C# 端 step 卡住 → finally 不执行 → SemaphoreSlim 不释放。
**修复**：C# per-read/request timeout + Python auto-reconnect + HTTP fallback。

### 锻造/商店卡牌移除导致 C# 永久阻塞
**症状**：`choose_rest_option`（锻造）后 pipe 永久无响应。
**原因**：card_select UI 等待，TestMode 无 UI → 永远阻塞。
**修复**：C# fire-and-forget + 轮询 IsSelectionActive + 两步操作。

### MCTS reconcile 按 card_index 导致 STRIKE→DEFEND 静默替换
**症状**：MCTS 100+ iter 连第一层都赢不了。
**原因**：save/load 后手牌顺序变了，card_index 指向不同的牌。
**修复**：改为按 label（卡牌名称）匹配。
**教训**：card_index 是位置索引不是身份标识。

### MCTS 鸡生蛋：未训练 NN + 少量 sims = 比随机差
**症状**：MCTS 接管后 floor 掉回 1.0。
**原因**：未训练 NN 给均匀 prior，30 sims 等同随机。
**修复**：Phase A 纯随机预训练 → Phase B 开 MCTS。

### PPO 非战斗奖励错位一步（P0）
**症状**：ppo_pl ≈ 0，非战斗脑完全不学习。
**原因**：reward 在 act() 之前算，绑到了当前 action 上，整条轨迹错位一步。
**修复**：pending-step 模式——act 之后再算 reward。有 `TestRolloutAlignment` 防复发。

### C#/Python Schema 字段名不匹配（P1，多处）
**症状**：地图/buff/意图特征全0，NN 蒙眼打牌。
**原因**：C# 发 `next_options`/`status`/`intents[]`，Python 读 `available_next_nodes`/`powers`/`intent_type`。
**修复**：统一字段名。有 `TestSchemaContract` 防复发。

### Action 表征塌缩（P0）
**症状**：event 选项 A/B/C 在 NN 里完全一样。
**原因**：缺 screen types、action types、target_indices 没喂入网络。
**修复**：补全所有类型 + index_embed。有 `TestActionDistinguishability` 防复发。

### NODE_TYPES[0] = "monster" 被 `> 0` 过滤
**症状**：monster 地图节点没有语义 embedding。
**原因**：`target_node_types > 0` 判断，monster 在 index 0 被过滤。
**修复**：`unknown` 放 index 0，monster 移到 index 1。旧 checkpoint 不兼容。

### 负 combat_potential 违反 PBRS
**原因**：kill progress 用减法，phi 可能为负。
**修复**：改为 `(1 - enemy_hp_ratio) * 0.5`，phi 范围 [0, 1.1]。

### Episode 终止时 pending combat step 丢失
**原因**：game_over 时 `_combat_ppo_pending` 没 flush 到 buffer。
**修复**：break 前检查并 flush pending step。
