# parity 已解决问题记录

更新时间：2026-04-20

本文记录 `STS2AI/bridge/game_bridge/parity.py` 这条 live parity 链路里，已经定位并修复过的问题。只记录“已解决”或“已明确定位”的问题，避免后续重复踩坑。

## 1. parity 产物目录落错到 `Python/Artifacts`

- 现象：
  早期 parity 报告被写到 `STS2AI/Python/Artifacts/parity/game_bridge`，不符合项目目录划分。
- 根因：
  `parity.py` 里默认输出目录从脚本相对路径推导错了。
- 修复：
  默认输出目录改到 `STS2AI/Artifacts/parity/game_bridge`。
- 代码：
  [parity.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/bridge/game_bridge/parity.py:450)

## 2. spectator 实际加载的不是工作区 `bin/Debug` DLL

- 现象：
  明明改了 `SpectatorBridgeMod` 源码并重新 `dotnet build`，但观战端返回的状态仍然像旧逻辑。
- 根因：
  Godot 运行时实际加载的是：
  `D:\dev\Godot_v4.5.1-stable_mono_win64\mods\sts2_mcp_spectator\sts2_mcp_spectator.dll`
  而不是工作区 `STS2AI/ENV/Spectator/SpectatorBridgeMod/bin/Debug/net9.0/` 下的新 DLL。
- 修复：
  每次 spectator 改动后，都要：
  1. 停掉 `Godot*`
  2. 把新 DLL 复制到 Godot 的 mod 目录
  3. 再重启 spectator
- 影响：
  这个问题不修，后面看到的很多“改了没生效”都是假问题。

## 3. Neow 首屏 `advance_dialogue` / `choose_event_option` 不一致

- 现象：
  同 seed reset 后：
  - real：`event`，只有 `advance_dialogue`
  - sim：`event`，直接有 3 个 `choose_event_option`
- 根因：
  有两层问题叠加：
  1. spectator `event` builder 会出现 `options` 已经有值，但 `in_dialogue` 仍然为 `true`
  2. reset / wait 链路会把这种“纯对白过渡态”直接暴露给上游
- 修复：
  1. spectator `event` 的 `in_dialogue` 改成按 `!is_finished && options.Count == 0` 判定
  2. spectator 的 `reset/step` 等待链路统一收口到 DTO wait，并对纯对白 `event` 内部自动执行 `advance_dialogue`
- 代码：
  [McpMod.StateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Spectator/SpectatorBridgeMod/Bridge/McpMod.StateBuilder.cs:748)
  [ModFullRunEnv.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Spectator/SpectatorBridgeMod/ModFullRunEnv.cs:241)
  [ModFullRunEnv.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Spectator/SpectatorBridgeMod/ModFullRunEnv.cs:347)
- 验证：
  `123456` 下 reset 后首屏已经对齐成 3 个 `choose_event_option`，见：
  [parity_20260420_180545.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260420_180545.json:1)

## 4. `game_over` 顶层 `player` 字段不一致

- 现象：
  两边都到 `game_over`，动作和 `run_outcome` 一致，但 real 顶层还挂着 `player`，sim 没有。
- 根因：
  spectator DTO 转换层在 `game_over` 也照常写入了顶层 `player`。
- 修复：
  spectator `game_over` 时不再暴露顶层 `player`。
- 代码：
  [SpectatorApiStateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Spectator/SpectatorBridgeMod/Bridge/SpectatorApiStateBuilder.cs:35)
- 验证：
  `123456` 终局 diff 已清零，见：
  [parity_20260420_180545.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260420_180545.json:1)

## 5. `card_select` 词表不一致：`screen_type` / `card_index`

- 现象：
  在 `card_select` screen：
  - real 侧 `screen_type` 是 `select`
  - sim 侧是 `DeckGeneric`
  - real `select_card` 动作只有 `index`
  - sim 同时有 `index` 和 `card_index`
- 根因：
  spectator 的 `card_select` builder 和 legal action builder 没有和 sim 统一词表。
- 修复：
  1. spectator `screen_type` 统一成 sim 风格：
     - `DeckGeneric`
     - `Transform`
     - `UpgradeSelect`
     - `SimpleSelect`
  2. spectator `select_card` 动作补 `card_index`
- 代码：
  [McpMod.StateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Spectator/SpectatorBridgeMod/Bridge/McpMod.StateBuilder.cs:1568)
  [ModFullRunEnv.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Spectator/SpectatorBridgeMod/ModFullRunEnv.cs:837)
- 验证：
  `223344` 的 `step_index=1` 已对齐，见：
  [parity_20260420_181229.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260420_181229.json:1)

## 6. sim 非战斗 `card_select` 过早 auto-complete

- 现象：
  `223344` 下 Neow 相关 `card_select`：
  - real：选中一张后进入预览态，只剩 `confirm_selection / cancel_selection`
  - sim：会直接继续自动推进，或者继续保留额外 `select_card`
- 根因：
  sim 的 `FullRunSimulationChoiceBridge` 里，非战斗 card selection 存在 `ShouldAutoComplete` 分支，选择后可能提前完成，不符合 visible game 的逐步语义。
- 修复：
  暂时先把 full-run 非战斗 `card_select` 的 `ShouldAutoComplete` 关闭，强制等显式 `confirm_selection`。
- 代码：
  [FullRunSimulationChoiceBridge.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulationChoiceBridge.cs:151)

## 7. sim `card_select` 选满后仍暴露剩余 `select_card`

- 现象：
  `223344` 下选中一张后：
  - real：进入预览确认态，只剩 `confirm_selection / cancel_selection`
  - sim：仍然保留剩余 `select_card`，并且 `cards` 只剩可选卡，不包含已选卡
- 根因：
  还不只是 auto-complete 的问题。sim 的 proto/full-run builder 仍按 `SelectableCards` 直接生成：
  1. legal actions
  2. `card_select.cards`
  3. `can_cancel`

  所以即使选择桥本身没有提前完成，状态表达仍然和真实端不一致。
- 修复：
  1. 当 `selected_count >= max_select` 且可确认时，sim `card_select` 进入预览确认态
  2. 预览态下不再生成剩余 `select_card`
  3. `card_select.cards` 改成 `selectable + selected` 的全量有序列表
  4. 预览态下如果已有已选卡，`can_cancel` 自动放开，与真实端一致
- 代码：
  [FullRunSimulationStateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulationStateBuilder.cs:452)
  [ProtoStateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/ProtoStateBuilder.cs:865)
- 验证：
  `223344` 已打通，见：
  [parity_20260420_224227.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260420_224227.json:1)

## 当前状态

- `123456`：
  已打通，`mismatch_count = 0`
  [parity_20260420_180545.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260420_180545.json:1)
- `223344`：
  已打通，`mismatch_count = 0`
  [parity_20260420_224227.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260420_224227.json:1)

## 8. spectator `event` 状态与 `choose_event_option` 执行口径不一致

- 现象：
  `556677` 下 Neow 选择 `index=0` 后，spectator 一度返回：
  - `state_type = event`
  - `legal_actions = choose_event_option 1/2`

  但同一时刻实际再发：
  - `choose_event_option index=1`

  会直接返回：
  - `accepted = false`
  - `error = No event options available`

  并且这个矛盾状态在 `0s / 0.5s / 2.0s` 复查下都持续存在，不是瞬时抖动。
- 根因：
  这条问题已经通过代码和运行时复现确认，不是猜测：
  1. spectator `event` 状态优先读的是模型里的 `localEvent.CurrentOptions`
  2. 真实执行 `choose_event_option` 走的是 UI 按钮 `NEventOptionButton`
  3. 上游 `NEventRoom.OptionButtonClicked(...)` 在真正执行选择前就会先 `Layout.ClearOptions()`

  所以会出现：
  - 状态端还看到旧 `CurrentOptions`
  - 执行端已经没有按钮可点

  另外，`LostCoffer` 这种 Neow 选项在拿 relic 后会进入奖励流；spectator 原先在 `CurrentRoom is EventRoom` 分支下，没有优先识别 event 期间出现的 reward overlay，导致奖励流也会被错报成 `event`。
- 修复：
  1. spectator `BuildEventState(...)` 改成 UI 按钮优先、模型兜底，避免把已被 UI 清空的旧 `CurrentOptions` 继续暴露出去
  2. spectator `BuildGameState()` 在 `CurrentRoom is EventRoom` 时，也优先识别：
     - `NCardRewardSelectionScreen -> card_reward`
     - `NRewardsScreen -> combat_rewards`
- 代码：
  [NEventRoom.cs](/C:/Users/Administrator/Desktop/sts2Raw2/src/Core/Nodes/Rooms/NEventRoom.cs:191)
  [McpMod.StateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Spectator/SpectatorBridgeMod/Bridge/McpMod.StateBuilder.cs:176)
  [McpMod.StateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Spectator/SpectatorBridgeMod/Bridge/McpMod.StateBuilder.cs:690)
- 验证：
  修复后，同样的 `556677` / `choose_event_option index=0`，真实端不再返回错误的 `event + choose_event_option 1/2`，而是返回：
  - `state_type = combat_rewards`
  - `legal_actions = claim_reward / proceed`

  这说明“观战端自己暴露了不可执行假动作”这条问题已经被修正。

## 9. sim Neow 奖励后 `finished event` 缺少显式 `proceed` option

- 现象：
  `556677` 下按 parity 实际动作序列：
  1. `reset`
  2. `choose_event_option index=0`
  3. `proceed`

  两边都会进入：
  - `state_type = event`
  - `legal_actions = [proceed]`

  但 payload 仍有分叉：
  - real：`event.options = [{ index: 0, text: "继续", is_proceed: true }]`
  - sim：`event.options = []`
- 证据：
  这条已经通过 raw state 复现确认，不是猜测：
  - real raw：观战端 step 2 的 `event.options` 确实保留一条显式 proceed option
  - sim raw：同一步 `event.is_finished = true`，但 `event.options` 为空，仅 `legal_actions` 里有 `proceed`
- 根因：
  sim 的 event state builder 直接透传 `localEvent.CurrentOptions`。
  对这类已完成 event，底层模型已经不再保留 option，但真实游戏暴露给上游的 gamestate 仍保留一条“继续”占位 option。
- 修复：
  对 `event.is_finished == true && options.Count == 0` 的 finished event，sim 各条 event state 输出统一补一条 synthetic proceed option：
  - `index = 0`
  - `is_proceed = true`
  - `is_locked = false`
- 代码：
  [ProtoStateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/ProtoStateBuilder.cs:711)
  [FullRunApiStateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunApiStateBuilder.cs:266)
- 验证：
  `556677` 已打通，见：
  [parity_20260421_556677_after_finished_event_fix.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260421_556677_after_finished_event_fix.json:1)

## 10. spectator `/reset` 不是幂等的，旧 run 会卡住持续对拍

- 现象：
  真实端已经在 run 中时，直接调 `/api/v2/full_run_env/reset` 会返回：
  - `Run already in progress`

  这会让 parity sweep 只能靠手动重启 spectator，无法稳定连续跑多 seed。
- 根因：
  spectator 的 reset 逻辑默认假设调用时已经在 main menu；它只会等待 menu ready，然后直接 `ExecuteStartRun(...)`。当当前 state 不是 menu 时，并不会先清理旧 run。
- 修复：
  1. reset 开始时先读取当前可见 state
  2. 如果当前不是 `menu`，先走 `NGame.Instance.ReturnToMainMenuAfterRun()`
  3. 回到主菜单后再继续原有 reset 流程

  这样 `/reset` 就变成幂等入口，适合持续 parity sweep。
- 代码：
  [ModFullRunEnv.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Spectator/SpectatorBridgeMod/ModFullRunEnv.cs:56)
- 验证：
  修复后，真实端可以在已有 run 的情况下直接再次 `reset` 成功，不再报 `Run already in progress`。

## 11. combat-only proto 把 build 已应用的 `deck/relics` 漏报成空

- 现象：
  combat sim 用 `combat_reset(build=...)` 传入多张 deck / 多个 relic 后，
  step 0 通过 Python bridge 看到的是：
  - `player.deck = []`
  - `player.relics = []`

  但同一个 state 里又能看到：
  - `battle.hand = 5`
  - `battle.draw_pile_cards = 7`

  也就是实际 12 张牌已经进战斗了，只是 API 没把完整 player build 暴露出来。
- 证据：
  这条已经通过现代码直接复现确认，不是猜测：
  1. 用 `combat_reset` 传入 12 张 deck + 4 个 relic
  2. 修复前 raw state 返回：
     - 顶层 `player.deck/relics = []`
     - `battle.player.deck/relics = []`
     - `hand + draw_pile_cards = 12`
  3. 说明不是 build 没应用，而是 combat proto 输出层漏字段
- 根因：
  `ProtoStateBuilder.BuildCombatGameStatePayload(...)` 和
  `BuildBattleState(...)` 只从 `CombatTrainingStateSnapshot.Player`
  拿了 `hp/block/energy/powers` 这类战斗动态字段，没有把 runtime `Player`
  上已经存在的：
  - `Deck`
  - `Relics`
  - `PotionSlots`
  - `Gold`

  一并写入 `GameState.player / battle.player`。

  同时 Python `proto_state_converter` 在 combat state 下也没有把 battle 合并后的
  `player` 回填到顶层 `state["player"]`，进一步放大了这个缺口。
- 修复：
  1. combat-only proto 输出新增 `BuildCombatPlayerState(...)`
  2. `GameState.player` 和 `battle.player` 都改成输出完整 build 视图
  3. Python converter 在 combat state 下把合并后的 `battle.player`
     同步回顶层 `state["player"]`
  4. `_convert_player(...)` 也补上 `powers`
- 代码：
  [ProtoStateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/ProtoStateBuilder.cs:198)
  [ProtoStateBuilder.cs](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/ENV/Sim/HeadlessSim/ProtoStateBuilder.cs:537)
  [proto_state_converter.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/bridge/game_bridge/transport/proto_state_converter.py:65)
- 验证：
  修复后，同样的 build 复现已变成：
  - `top_relic_count = 4`
  - `battle_relic_count = 4`
  - `top_deck_count = 12`
  - `battle_deck_count = 12`
  - `hand_count = 5`
  - `draw_pile_cards_count = 7`

  这说明图里“SIM 丢 3/4 relic / 丢 8/18 deck”的这两类现象，在当前链路上确实是
  API 漏报问题，而且已经修正。

## 当前未解决的主问题

- 仍需要继续扩 seed 和扩 screen 覆盖，确认这些修复不是局部偶合。
- 当前这轮已确认通过：
  - [parity_20260421_556677_after_finished_event_fix.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260421_556677_after_finished_event_fix.json:1)
  - [parity_20260421_seed_667788.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260421_seed_667788.json:1)
  - [parity_20260421_seed_778899.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260421_seed_778899.json:1)
  - [parity_20260421_seed_889900.json](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Artifacts/parity/game_bridge/parity_20260421_seed_889900.json:1)
