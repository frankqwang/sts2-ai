using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Achievements;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Ftue;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Combat;

public class CombatManager
{
    private sealed record PendingLossState(CombatState State, CombatRoom Room);

    public const int baseHandDrawCount = 5;

    private readonly Lock _playerReadyLock = new Lock();

    private readonly HashSet<Player> _playersReadyToEndTurn = new HashSet<Player>();

    private readonly HashSet<Player> _playersReadyToBeginEnemyTurn = new HashSet<Player>();

    private readonly List<Player> _playersTakingExtraTurn = new List<Player>();

    private CombatState? _state;

    private PendingLossState? _pendingLoss;

    private bool _playerActionsDisabled;

    private Task? _pendingTurnTransitionTask;

    public static CombatManager Instance { get; } = new CombatManager();

    public CardModel? DebugForcedTopCardOnNextShuffle { get; private set; }

    public bool IsPaused { get; private set; }

    public bool PlayerActionsDisabled
    {
        get
        {
            return _playerActionsDisabled;
        }
        private set
        {
            if (_playerActionsDisabled != value)
            {
                _playerActionsDisabled = value;
                this.PlayerActionsDisabledChanged?.Invoke(_state);
            }
        }
    }

    public IReadOnlyList<Player> PlayersTakingExtraTurn
    {
        get
        {
            using (_playerReadyLock.EnterScope())
            {
                return _playersTakingExtraTurn.ToList();
            }
        }
    }

    public bool IsPlayPhase { get; private set; }

    public bool IsEnemyTurnStarted { get; private set; }

    public bool EndingPlayerTurnPhaseTwo { get; private set; }

    public bool EndingPlayerTurnPhaseOne { get; private set; }

    public CombatStateTracker StateTracker { get; }

    public CombatHistory History { get; }

    public bool IsInProgress { get; private set; }

    public bool IsAboutToLose => _pendingLoss != null;

    public bool IsEnding
    {
        get
        {
            if (!IsInProgress)
            {
                return false;
            }
            if (_pendingLoss != null)
            {
                return true;
            }
            if (_state != null && _state.Enemies.Any((Creature e) => e != null && e.IsAlive && e.IsPrimaryEnemy))
            {
                return false;
            }
            if (Hook.ShouldStopCombatFromEnding(_state))
            {
                return false;
            }
            return true;
        }
    }

    public bool IsOverOrEnding
    {
        get
        {
            if (!IsEnding)
            {
                return !IsInProgress;
            }
            return true;
        }
    }

    public event Action<CombatState>? CombatSetUp;

    public event Action<CombatRoom>? CombatEnded;

    public event Action<CombatRoom>? CombatWon;

    public event Action<CombatState>? CreaturesChanged;

    public event Action<CombatState>? TurnStarted;

    public event Action<CombatState>? TurnEnded;

    public event Action<Player, bool>? PlayerEndedTurn;

    public event Action<Player>? PlayerUnendedTurn;

    public event Action<CombatState>? AboutToSwitchToEnemyTurn;

    public event Action<CombatState>? PlayerActionsDisabledChanged;

    public CombatState? DebugOnlyGetState()
    {
        return _state;
    }

    public Task? DebugOnlyGetPendingTurnTransitionTask()
    {
        return _pendingTurnTransitionTask;
    }

    private static void TracePureSimCombat(string message)
    {
        if (CombatSimulationRuntime.IsPureCombatSimulator)
        {
            FullRunSimulationTrace.Write($"headless_combat_manager.{message}");
        }
    }

    private static bool SkipCombatPresentation => CombatSimulationRuntime.IsPureCombatSimulator;

    private CombatManager()
    {
        History = new CombatHistory();
        StateTracker = new CombatStateTracker(this);
    }

    public void SetUpCombat(CombatState state)
    {
        if (_state != null)
        {
            throw new InvalidOperationException("Make sure to reset the combat before setting up a new one.");
        }
        _state = state;
        _state.MultiplayerScalingModel?.OnCombatEntered(_state);
        StateTracker.SetState(state);
        using (_playerReadyLock.EnterScope())
        {
            _playersTakingExtraTurn.Clear();
        }
        foreach (Player player in state.Players)
        {
            player.ResetCombatState();
        }
        foreach (Player player2 in state.Players)
        {
            player2.PopulateCombatState(player2.RunState.Rng.Shuffle, state);
        }
        NetCombatCardDb.Instance.StartCombat(state.Players);
        foreach (Creature creature in state.Creatures)
        {
            AddCreature(creature);
        }
        this.CombatSetUp?.Invoke(state);
    }

    public void AfterCombatRoomLoaded()
    {
        TaskHelper.RunSafely(StartCombatInternal());
    }

    public async Task StartCombatInternal()
    {
        if (!SkipCombatPresentation && _state.Encounter.HasBgm)
        {
            NRunMusicController.Instance?.PlayCustomMusic(_state.Encounter.CustomBgm);
        }
        foreach (Creature creature in _state.Creatures)
        {
            await AfterCreatureAdded(creature);
        }
        RunManager.Instance.ActionExecutor.Pause();
        RunManager.Instance.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.NotPlayPhase);
        IsInProgress = true;
        await Hook.BeforeCombatStart(_state.RunState, _state);
        if (!SkipCombatPresentation)
        {
            NRunMusicController.Instance?.UpdateTrack();
            NCombatRulesFtue ftue = null;
            if (SaveManager.Instance.SeenFtue("combat_rules_ftue"))
            {
                NCombatRoom.Instance?.AddChildSafely(NCombatStartBanner.Create());
            }
            else
            {
                ftue = NCombatRulesFtue.Create();
                NModalContainer.Instance?.Add(ftue, showBackstop: false);
            }
            await Cmd.CustomScaledWait(0.5f, 1f);
            await StartTurn();
            ftue?.Start();
            return;
        }
        // If combat piles were restored from a save, skip the initial StartTurn
        // (energy reset + draw) since the piles are already in the correct state.
        if (_state.RunState.CurrentRoom is CombatRoom combatRoom && combatRoom.SkipInitialStartTurn)
        {
            combatRoom.SkipInitialStartTurn = false;
            // Just enter play phase without drawing — piles already correct from save
            RunManager.Instance.ActionExecutor.Unpause();
            RunManager.Instance.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.PlayPhase);
            IsPlayPhase = true;
        }
        else
        {
            await StartTurn();
        }
    }

    private async Task StartTurn(Func<Task>? actionDuringEnemyTurn = null)
    {
        TracePureSimCombat($"start_turn.begin round={_state?.RoundNumber} side={_state?.CurrentSide} in_progress={IsInProgress}");
        if (!IsInProgress)
        {
            return;
        }
        bool isExtraPlayerTurn;
        List<Creature> creaturesStartingTurn;
        List<Player> playersStartingTurn;
        using (_playerReadyLock.EnterScope())
        {
            isExtraPlayerTurn = _playersTakingExtraTurn.Count > 0;
            if (_state.CurrentSide == CombatSide.Player && isExtraPlayerTurn)
            {
                creaturesStartingTurn = _playersTakingExtraTurn.Select((Player p) => p.Creature).ToList();
                playersStartingTurn = _playersTakingExtraTurn.ToList();
            }
            else
            {
                creaturesStartingTurn = _state.CreaturesOnCurrentSide.ToList();
                playersStartingTurn = ((_state.CurrentSide == CombatSide.Player) ? _state.Players.ToList() : new List<Player>());
            }
        }
        foreach (Creature item in creaturesStartingTurn)
        {
            item.BeforeTurnStart(_state.RoundNumber, _state.CurrentSide);
        }
        await Hook.BeforeSideTurnStart(_state, _state.CurrentSide);
        if (_state.CurrentSide == CombatSide.Player)
        {
            PlayerActionsDisabled = false;
            using (_playerReadyLock.EnterScope())
            {
                _playersReadyToEndTurn.Clear();
                _playersReadyToBeginEnemyTurn.Clear();
            }
            if (!SkipCombatPresentation && _state.RoundNumber != 1)
            {
                NCombatRoom.Instance?.AddChildSafely(NPlayerTurnBanner.Create(_state.RoundNumber));
            }
            if (!isExtraPlayerTurn)
            {
                foreach (Creature enemy in _state.Enemies)
                {
                    enemy.PrepareForNextTurn(_state.PlayerCreatures);
                }
            }
        }
        else
        {
            if (!SkipCombatPresentation)
            {
                NCombatRoom.Instance?.AddChildSafely(NEnemyTurnBanner.Create());
            }
        }
        if (!SkipCombatPresentation)
        {
            await Cmd.CustomScaledWait(0.5f, 0.8f);
        }
        foreach (Creature item2 in creaturesStartingTurn)
        {
            await item2.AfterTurnStart(_state.RoundNumber, _state.CurrentSide);
        }
        foreach (Creature item3 in creaturesStartingTurn)
        {
            await Hook.AfterBlockCleared(_state, item3);
        }
        foreach (Player item4 in playersStartingTurn)
        {
            HookPlayerChoiceContext hookPlayerChoiceContext = new HookPlayerChoiceContext(item4, LocalContext.NetId.Value, GameActionType.CombatPlayPhaseOnly);
            Task task = SetupPlayerTurn(item4, hookPlayerChoiceContext);
            await hookPlayerChoiceContext.AssignTaskAndWaitForPauseOrCompletion(task);
        }
        await Hook.AfterSideTurnStart(_state, _state.CurrentSide);
        if (_state.CurrentSide == CombatSide.Player)
        {
            foreach (Player item5 in playersStartingTurn)
            {
                HookPlayerChoiceContext hookPlayerChoiceContext2 = new HookPlayerChoiceContext(item5, LocalContext.NetId.Value, GameActionType.CombatPlayPhaseOnly);
                Task task2 = item5.PlayerCombatState.OrbQueue.AfterTurnStart(hookPlayerChoiceContext2);
                await hookPlayerChoiceContext2.AssignTaskAndWaitForPauseOrCompletion(task2);
            }
            RunManager.Instance.ChecksumTracker.GenerateChecksum("After player turn start", null);
            foreach (Player player in _state.Players)
            {
                if (player.Creature.IsDead || !playersStartingTurn.Contains(player))
                {
                    Log.Info($"Setting player {player.NetId} to ready at start of turn. IsDead: {player.Creature.IsDead}. IsStartingTurn: {playersStartingTurn.Contains(player)}");
                    SetReadyToEndTurn(player, canBackOut: false);
                }
                else
                {
                    await Hook.BeforePlayPhaseStart(_state, player);
                }
            }
            await CheckWinCondition();
            if (IsInProgress)
            {
                RunManager.Instance.ActionExecutor.Unpause();
                RunManager.Instance.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.PlayPhase);
                IsPlayPhase = true;
                IsEnemyTurnStarted = false;
                this.TurnStarted?.Invoke(_state);
            }
        }
        else
        {
            IsEnemyTurnStarted = true;
            this.TurnStarted?.Invoke(_state);
            RunManager.Instance.ChecksumTracker.GenerateChecksum("After enemy turn start", null);
            await WaitForUnpause();
            await CheckWinCondition();
            if (IsInProgress)
            {
                await ExecuteEnemyTurn(actionDuringEnemyTurn);
            }
        }
        TracePureSimCombat($"start_turn.done round={_state?.RoundNumber} side={_state?.CurrentSide} play_phase={IsPlayPhase} enemy_started={IsEnemyTurnStarted} in_progress={IsInProgress}");
    }

    private async Task SetupPlayerTurn(Player player, HookPlayerChoiceContext playerChoiceContext)
    {
        if (player.Creature.IsDead)
        {
            return;
        }
        if (Hook.ShouldPlayerResetEnergy(_state, player))
        {
            SfxCmd.Play("event:/sfx/ui/gain_energy");
            player.PlayerCombatState.ResetEnergy();
        }
        else
        {
            player.PlayerCombatState.AddMaxEnergyToCurrent();
        }
        await Hook.AfterEnergyReset(_state, player);
        await Hook.BeforeHandDraw(_state, player, playerChoiceContext);
        decimal handDraw = Hook.ModifyHandDraw(_state, player, 5m, out IEnumerable<AbstractModel> modifiers);
        await Hook.AfterModifyingHandDraw(_state, modifiers);
        if (_state.RoundNumber == 1)
        {
            CardPile pile = PileType.Draw.GetPile(player);
            List<CardModel> list = pile.Cards.Where((CardModel c) => c.Enchantment?.ShouldStartAtBottomOfDrawPile ?? false).ToList();
            foreach (CardModel item in list)
            {
                pile.MoveToBottomInternal(item);
            }
            List<CardModel> list2 = pile.Cards.Where((CardModel c) => c.Keywords.Contains(CardKeyword.Innate)).Except(list).ToList();
            foreach (CardModel item2 in list2)
            {
                pile.MoveToTopInternal(item2);
            }
            handDraw = Math.Max(handDraw, list2.Count);
            handDraw = Math.Min(handDraw, 10m);
        }
        await CardPileCmd.Draw(playerChoiceContext, handDraw, player, fromHandDraw: true);
        await Hook.AfterPlayerTurnStart(_state, playerChoiceContext, player);
    }

    public void SetReadyToEndTurn(Player player, bool canBackOut, Func<Task>? actionDuringEnemyTurn = null)
    {
        using (_playerReadyLock.EnterScope())
        {
            _playersReadyToEndTurn.Add(player);
        }
        this.PlayerEndedTurn?.Invoke(player, canBackOut);
        if (AllPlayersReadyToEndTurn())
        {
            TracePureSimCombat($"set_ready_to_end_turn.all_ready round={_state?.RoundNumber} side={_state?.CurrentSide} action_running={RunManager.Instance.ActionExecutor.IsRunning}");
            Log.LogMessage(LogLevel.Debug, LogType.GameSync, "All players ready to end turn");
            GameAction currentlyRunningAction = RunManager.Instance.ActionExecutor.CurrentlyRunningAction;
            if (currentlyRunningAction != null && ActionQueueSet.IsGameActionPlayerDriven(currentlyRunningAction))
            {
                TrackPendingTurnTransition(TaskHelper.RunSafely(WaitForActionThenEndTurn(currentlyRunningAction, actionDuringEnemyTurn)));
            }
            else
            {
                TrackPendingTurnTransition(TaskHelper.RunSafely(AfterAllPlayersReadyToEndTurn(actionDuringEnemyTurn)));
            }
        }
    }

    public void UndoReadyToEndTurn(Player player)
    {
        using (_playerReadyLock.EnterScope())
        {
            _playersReadyToEndTurn.Remove(player);
        }
        if (LocalContext.IsMe(player))
        {
            PlayerActionsDisabled = false;
        }
        this.PlayerUnendedTurn?.Invoke(player);
    }

    public void OnEndedTurnLocally()
    {
        PlayerActionsDisabled = true;
    }

    public void SetReadyToBeginEnemyTurn(Player player, Func<Task>? actionDuringEnemyTurn = null)
    {
        if (!IsInProgress)
        {
            Log.Error("Trying to set player ready to begin enemy turn, but combat is over!");
        }
        bool flag;
        using (_playerReadyLock.EnterScope())
        {
            _playersReadyToBeginEnemyTurn.Add(player);
            flag = _playersReadyToBeginEnemyTurn.Count == _state.Players.Count && _state.CurrentSide == CombatSide.Player;
        }
        if (flag || RunManager.Instance.NetService.Type == NetGameType.Singleplayer)
        {
            TracePureSimCombat($"set_ready_to_begin_enemy_turn.begin round={_state?.RoundNumber} side={_state?.CurrentSide} play_phase={IsPlayPhase}");
            TrackPendingTurnTransition(TaskHelper.RunSafely(AfterAllPlayersReadyToBeginEnemyTurn(actionDuringEnemyTurn)));
        }
    }

    public bool IsPlayerReadyToEndTurn(Player player)
    {
        using (_playerReadyLock.EnterScope())
        {
            return _playersReadyToEndTurn.Contains(player);
        }
    }

    public bool AllPlayersReadyToEndTurn()
    {
        bool flag;
        using (_playerReadyLock.EnterScope())
        {
            flag = _playersReadyToEndTurn.Count == _state.Players.Count;
        }
        if (!RunManager.Instance.IsSinglePlayerOrFakeMultiplayer)
        {
            if (flag)
            {
                return _state.CurrentSide == CombatSide.Player;
            }
            return false;
        }
        return true;
    }

    private async Task EndEnemyTurn()
    {
        TracePureSimCombat($"end_enemy_turn.begin round={_state?.RoundNumber} side={_state?.CurrentSide} is_enemy_started={IsEnemyTurnStarted}");
        if (_state.CurrentSide != CombatSide.Enemy)
        {
            throw new InvalidOperationException($"EndPlayerTurn called while the current side is {_state.CurrentSide}!");
        }
        await WaitForUnpause();
        await EndEnemyTurnInternal();
        await CheckWinCondition();
        if (!IsEnding)
        {
            SwitchSides();
            await WaitForUnpause();
            await StartTurn();
        }
        TracePureSimCombat($"end_enemy_turn.done round={_state?.RoundNumber} side={_state?.CurrentSide} in_progress={IsInProgress} play_phase={IsPlayPhase}");
    }

    public void AddCreature(Creature creature)
    {
        if (!_state.ContainsCreature(creature))
        {
            throw new InvalidOperationException("CombatState must already contain creature.");
        }
        creature.Monster?.SetUpForCombat();
        if (creature.SlotName != null)
        {
            _state.SortEnemiesBySlotName();
        }
        StateTracker.Subscribe(creature);
        this.CreaturesChanged?.Invoke(_state);
    }

    public async Task AfterCreatureAdded(Creature creature)
    {
        await creature.AfterAddedToRoom();
        if (creature.IsEnemy && _state.CurrentSide == CombatSide.Player)
        {
            creature.Monster.RollMove(_state.Players.Select((Player p) => p.Creature));
        }
    }

    public async Task CheckForEmptyHand(PlayerChoiceContext choiceContext, Player player)
    {
        if (IsInProgress && !PileType.Hand.GetPile(player).Cards.Any())
        {
            await Hook.AfterHandEmptied(_state, choiceContext, player);
        }
    }

    public void Reset()
    {
        Reset(graceful: true);
    }

    public void Reset(bool graceful)
    {
        if (graceful && _state != null)
        {
            try
            {
                foreach (Creature item in _state.Creatures.ToList())
                {
                    try { item.Reset(); } catch { /* ignore creature cleanup errors during reset */ }
                    try { RemoveCreature(item); } catch { }
                    try { _state.RemoveCreature(item); } catch { }
                }
            }
            catch { /* ignore errors iterating creatures */ }
            _state = null;  // always clear, even if creature cleanup threw
        }
        _pendingLoss = null;
        DebugForcedTopCardOnNextShuffle = null;
        IsInProgress = false;
        IsPlayPhase = false;
        IsEnemyTurnStarted = false;
        _pendingTurnTransitionTask = null;
        try { History.Clear(); } catch { }
        try
        {
            RunManager.Instance.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.NotInCombat);
        }
        catch
        {
            // The headless host may call Reset during partial bootstrap or teardown.
        }
    }

    private void TrackPendingTurnTransition(Task task)
    {
        _pendingTurnTransitionTask = task;
        _ = task.ContinueWith(
            static (completedTask, state) =>
            {
                CombatManager manager = (CombatManager)state!;
                if (ReferenceEquals(manager._pendingTurnTransitionTask, completedTask))
                {
                    manager._pendingTurnTransitionTask = null;
                }
            },
            this,
            TaskScheduler.Default);
    }

    public async Task HandlePlayerDeath(Player player)
    {
        if (IsInProgress)
        {
            CardModel[] cards = new CardPile[5]
            {
                player.PlayerCombatState.Hand,
                player.PlayerCombatState.DrawPile,
                player.PlayerCombatState.DiscardPile,
                player.PlayerCombatState.ExhaustPile,
                player.PlayerCombatState.PlayPile
            }.SelectMany((CardPile p) => p.Cards).ToArray();
            await CardPileCmd.RemoveFromCombat(cards, false);
            await PlayerCmd.SetEnergy(0m, player);
            await PlayerCmd.SetStars(0m, player);
        }
    }

    public void LoseCombat()
    {
        if (!(_pendingLoss != null))
        {
            CombatRoom room = _state.RunState.CurrentRoom as CombatRoom ?? new CombatRoom(_state);
            _pendingLoss = new PendingLossState(_state, room);
        }
    }

    private void ProcessPendingLoss()
    {
        if (!(_pendingLoss == null))
        {
            PendingLossState pendingLoss = _pendingLoss;
            _pendingLoss = null;
            IsInProgress = false;
            this.CombatEnded?.Invoke(pendingLoss.Room);
        }
    }

    public async Task EndCombatInternal()
    {
        CombatState combatState = _state;
        Player localPlayer = LocalContext.GetMe(combatState);
        IRunState runState = combatState.RunState;
        CombatRoom room = runState.CurrentRoom as CombatRoom ?? new CombatRoom(combatState);
        IsInProgress = false;
        IsPlayPhase = false;
        PlayerActionsDisabled = false;
        using (_playerReadyLock.EnterScope())
        {
            _playersTakingExtraTurn.Clear();
        }
        foreach (Player player in combatState.Players)
        {
            await player.ReviveBeforeCombatEnd();
        }
        await Hook.AfterCombatEnd(runState, combatState, room);
        History.Clear();
        room.OnCombatEnded();
        if (!SkipCombatPresentation && !CombatTrainingMode.IsActive && RunManager.Instance.NetService.Type != NetGameType.Replay)
        {
            string profileScopedPath = SaveManager.Instance.GetProfileScopedPath("replays/latest.mcr");
            RunManager.Instance.CombatReplayWriter.WriteReplay(profileScopedPath, stopRecording: true);
        }
        foreach (Player player2 in combatState.Players)
        {
            player2.AfterCombatEnd();
        }
        await Hook.AfterCombatVictory(runState, combatState, room);
        NHoverTipSet.Clear();
        if (!SkipCombatPresentation && runState.CurrentMapPointHistoryEntry != null)
        {
            runState.CurrentMapPointHistoryEntry.Rooms.Last().TurnsTaken = combatState.RoundNumber;
        }
        bool flag = !SkipCombatPresentation && runState.Map.SecondBossMapPoint != null && runState.CurrentMapCoord == runState.Map.SecondBossMapPoint.coord;
        bool flag2 = !SkipCombatPresentation && runState.Map.SecondBossMapPoint == null && runState.CurrentMapCoord == runState.Map.BossMapPoint.coord;
        if (!SkipCombatPresentation && room.RoomType == RoomType.Boss && runState.CurrentActIndex == runState.Acts.Count - 1 && (flag || flag2))
        {
            RunManager.Instance.WinTime = RunManager.Instance.RunTime;
        }
        room.MarkPreFinished();
        if (!SkipCombatPresentation && !CombatTrainingMode.IsActive)
        {
            await SaveManager.Instance.SaveRun(room, saveProgress: false);
        }
        if (!SkipCombatPresentation)
        {
            NMapScreen.Instance?.SetTravelEnabled(enabled: true);
            if (!CombatTrainingMode.IsActive)
            {
                SaveManager.Instance.UpdateProgressAfterCombatWon(localPlayer, room);
                AchievementsHelper.CheckForDefeatedAllEnemiesAchievement(runState.Act, localPlayer);
                SaveManager.Instance.SaveProgressFile();
                if (room.RoomType == RoomType.Boss)
                {
                    AchievementsHelper.AfterBossDefeated(localPlayer);
                }
            }
        }
        combatState.MultiplayerScalingModel?.OnCombatFinished();
        this.CombatWon?.Invoke(room);
        RunManager.Instance.ActionExecutor.Unpause();
        RunManager.Instance.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.NotInCombat);
        if (!SkipCombatPresentation)
        {
            NRunMusicController.Instance?.UpdateTrack();
        }
        this.CombatEnded?.Invoke(room);
    }

    public void RemoveCreature(Creature creature)
    {
        if (creature.IsMonster)
        {
            creature.Monster.BeforeRemovedFromRoom();
            creature.Monster.ResetStateMachine();
        }
        StateTracker.Unsubscribe(creature);
        this.CreaturesChanged?.Invoke(_state);
    }

    public async Task<bool> CheckWinCondition()
    {
        if (_pendingLoss != null)
        {
            ProcessPendingLoss();
            return true;
        }
        if (IsEnding)
        {
            await EndCombatInternal();
            return true;
        }
        return false;
    }

    private async Task ExecuteEnemyTurn(Func<Task>? actionDuringEnemyTurn = null)
    {
        TracePureSimCombat($"execute_enemy_turn.begin round={_state?.RoundNumber} side={_state?.CurrentSide} enemies={_state?.Enemies.Count}");
        if (!IsInProgress)
        {
            return;
        }
        if (actionDuringEnemyTurn != null)
        {
            await actionDuringEnemyTurn();
        }
        foreach (Creature enemy in _state.Enemies.ToList())
        {
            if (_state.ContainsCreature(enemy))
            {
                TracePureSimCombat($"execute_enemy_turn.enemy.begin round={_state.RoundNumber} enemy={enemy.ModelId.Entry} combat_id={enemy.CombatId}");
                NCreature nCreature = SkipCombatPresentation ? null : NCombatRoom.Instance?.GetCreatureNode(enemy);
                if (nCreature != null)
                {
                    await nCreature.PerformIntent();
                }
                await enemy.TakeTurn();
                TracePureSimCombat($"execute_enemy_turn.enemy.after_move round={_state.RoundNumber} enemy={enemy.ModelId.Entry} combat_id={enemy.CombatId} action_running={RunManager.Instance.ActionExecutor.IsRunning}");
                await WaitForUnpause();
                await CheckWinCondition();
                if (!IsInProgress)
                {
                    TracePureSimCombat($"execute_enemy_turn.enemy.ended_combat round={_state.RoundNumber} enemy={enemy.ModelId.Entry}");
                    return;
                }
            }
        }
        RunManager.Instance.ChecksumTracker.GenerateChecksum("After enemy turn end", null);
        TracePureSimCombat($"execute_enemy_turn.before_end_enemy_turn round={_state?.RoundNumber} side={_state?.CurrentSide}");
        await EndEnemyTurn();
        TracePureSimCombat($"execute_enemy_turn.done round={_state?.RoundNumber} side={_state?.CurrentSide}");
    }

    private async Task WaitForActionThenEndTurn(GameAction action, Func<Task>? actionDuringEnemyTurn)
    {
        await action.CompletionTask;
        await AfterAllPlayersReadyToEndTurn(actionDuringEnemyTurn);
    }

    private async Task AfterAllPlayersReadyToEndTurn(Func<Task>? actionDuringEnemyTurn = null)
    {
        TracePureSimCombat($"after_all_players_ready_to_end_turn.begin round={_state?.RoundNumber} side={_state?.CurrentSide}");
        EndingPlayerTurnPhaseOne = true;
        RunManager.Instance.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.EndTurnPhaseOne);
        await WaitUntilQueueIsEmptyOrWaitingOnNonPlayerDrivenAction();
        await EndPlayerTurnPhaseOneInternal();
        if (IsInProgress && RunManager.Instance.NetService.Type != NetGameType.Replay)
        {
            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new ReadyToBeginEnemyTurnAction(LocalContext.GetMe(_state), actionDuringEnemyTurn));
        }
        EndingPlayerTurnPhaseOne = false;
        TracePureSimCombat($"after_all_players_ready_to_end_turn.done round={_state?.RoundNumber} side={_state?.CurrentSide} in_progress={IsInProgress}");
    }

    private async Task WaitUntilQueueIsEmptyOrWaitingOnNonPlayerDrivenAction()
    {
        GameAction currentlyRunningAction = RunManager.Instance.ActionExecutor.CurrentlyRunningAction;
        TaskCompletionSource completionSource;
        if (currentlyRunningAction != null && ActionQueueSet.IsGameActionPlayerDriven(currentlyRunningAction))
        {
            completionSource = new TaskCompletionSource();
            RunManager.Instance.ActionExecutor.AfterActionExecuted += AfterActionExecuted;
            await completionSource.Task;
            RunManager.Instance.ActionExecutor.AfterActionExecuted -= AfterActionExecuted;
        }
        void AfterActionExecuted(GameAction action)
        {
            GameAction readyAction = RunManager.Instance.ActionQueueSet.GetReadyAction();
            if (readyAction == null || !ActionQueueSet.IsGameActionPlayerDriven(readyAction))
            {
                completionSource.SetResult();
            }
        }
    }

    public async Task EndPlayerTurnPhaseOneInternal()
    {
        TracePureSimCombat($"end_player_turn_phase_one.begin round={_state?.RoundNumber} side={_state?.CurrentSide}");
        if (_state.CurrentSide != CombatSide.Player)
        {
            throw new InvalidOperationException($"EndPlayerTurn called while the current side is {_state.CurrentSide}!");
        }
        await WaitForUnpause();
        IsPlayPhase = false;
        await Hook.BeforeTurnEnd(_state, _state.CurrentSide);
        if (await CheckWinCondition())
        {
            return;
        }
        List<Player> playersEndingTurn;
        using (_playerReadyLock.EnterScope())
        {
            playersEndingTurn = ((_playersTakingExtraTurn.Count > 0) ? _playersTakingExtraTurn.ToList() : _state.Players.ToList());
        }
        List<Task> playerEndTasks = new List<Task>();
        foreach (Player item in playersEndingTurn)
        {
            HookPlayerChoiceContext hookPlayerChoiceContext = new HookPlayerChoiceContext(item, LocalContext.NetId.Value, GameActionType.Combat);
            Task task = DoTurnEnd(item, hookPlayerChoiceContext);
            await hookPlayerChoiceContext.AssignTaskAndWaitForPauseOrCompletion(task);
            playerEndTasks.Add(task);
        }
        await Task.WhenAll(playerEndTasks);
        foreach (Player item2 in playersEndingTurn)
        {
            await Hook.BeforeFlush(_state, item2);
        }
        RunManager.Instance.ChecksumTracker.GenerateChecksum("After player turn phase one end", null);
        await CheckWinCondition();
        TracePureSimCombat($"end_player_turn_phase_one.done round={_state?.RoundNumber} side={_state?.CurrentSide} in_progress={IsInProgress}");
    }

    private async Task DoTurnEnd(Player player, PlayerChoiceContext choiceContext)
    {
        await player.PlayerCombatState.OrbQueue.BeforeTurnEnd(choiceContext);
        CardPile pile = PileType.Hand.GetPile(player);
        CardPile discardPile = PileType.Discard.GetPile(player);
        List<CardModel> turnEndCards = new List<CardModel>();
        List<CardModel> list = new List<CardModel>();
        foreach (CardModel card2 in pile.Cards)
        {
            if (card2.HasTurnEndInHandEffect)
            {
                turnEndCards.Add(card2);
            }
            else if (card2.Keywords.Contains(CardKeyword.Ethereal) && Hook.ShouldEtherealTrigger(player.Creature.CombatState, card2))
            {
                list.Add(card2);
            }
        }
        foreach (CardModel item in list)
        {
            await CardCmd.Exhaust(choiceContext, item, causedByEthereal: true);
        }
        foreach (CardModel card in turnEndCards)
        {
            await CardPileCmd.Add(card, PileType.Play);
            if (LocalContext.IsMe(player))
            {
                await Cmd.CustomScaledWait(0.3f, 0.6f);
            }
            await card.OnTurnEndInHand(choiceContext);
            if (card.Keywords.Contains(CardKeyword.Ethereal))
            {
                await CardCmd.Exhaust(choiceContext, card, causedByEthereal: true);
            }
            else
            {
                await CardPileCmd.Add(card, discardPile);
            }
        }
    }

    private async Task EndEnemyTurnInternal()
    {
        await Hook.BeforeTurnEnd(_state, _state.CurrentSide);
        foreach (Player player in _state.Players)
        {
            player.PlayerCombatState.EndOfTurnCleanup();
        }
        await Hook.AfterTurnEnd(_state, _state.CurrentSide);
    }

    private async Task AfterAllPlayersReadyToBeginEnemyTurn(Func<Task>? actionDuringEnemyTurn = null)
    {
        TracePureSimCombat($"after_all_players_ready_to_begin_enemy_turn.begin round={_state?.RoundNumber} side={_state?.CurrentSide}");
        EndingPlayerTurnPhaseTwo = true;
        RunManager.Instance.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.NotPlayPhase);
        this.AboutToSwitchToEnemyTurn?.Invoke(_state);
        await CombatSimulationRuntime.Clock.YieldAsync();
        await EndPlayerTurnPhaseTwoInternal();
        await SwitchFromPlayerToEnemySide(actionDuringEnemyTurn);
        EndingPlayerTurnPhaseTwo = false;
        TracePureSimCombat($"after_all_players_ready_to_begin_enemy_turn.done round={_state?.RoundNumber} side={_state?.CurrentSide} in_progress={IsInProgress}");
    }

    public async Task EndPlayerTurnPhaseTwoInternal()
    {
        TracePureSimCombat($"end_player_turn_phase_two.begin round={_state?.RoundNumber} side={_state?.CurrentSide}");
        if (_state.CurrentSide != CombatSide.Player)
        {
            throw new InvalidOperationException($"EndPlayerTurnPhaseTwo called while the current side is {_state.CurrentSide}!");
        }
        List<Player> list;
        using (_playerReadyLock.EnterScope())
        {
            list = ((_playersTakingExtraTurn.Count > 0) ? _playersTakingExtraTurn.ToList() : _state.Players.ToList());
        }
        foreach (Player player in list)
        {
            CardPile pile = PileType.Hand.GetPile(player);
            List<CardModel> list2 = new List<CardModel>();
            List<CardModel> cardsToRetain = new List<CardModel>();
            foreach (CardModel card in pile.Cards)
            {
                if (card.ShouldRetainThisTurn)
                {
                    cardsToRetain.Add(card);
                }
                else
                {
                    list2.Add(card);
                }
            }
            if (Hook.ShouldFlush(player.Creature.CombatState, player))
            {
                await CardPileCmd.Add(list2, PileType.Discard.GetPile(player));
            }
            foreach (CardModel item in cardsToRetain)
            {
                await Hook.AfterCardRetained(_state, item);
            }
            player.PlayerCombatState.EndOfTurnCleanup();
        }
        await Hook.AfterTurnEnd(_state, _state.CurrentSide);
        RunManager.Instance.ChecksumTracker.GenerateChecksum("after player turn phase two end", null);
        TracePureSimCombat($"end_player_turn_phase_two.done round={_state?.RoundNumber} side={_state?.CurrentSide}");
    }

    public async Task SwitchFromPlayerToEnemySide(Func<Task>? actionDuringEnemyTurn = null)
    {
        TracePureSimCombat($"switch_from_player_to_enemy_side.begin round={_state?.RoundNumber} side={_state?.CurrentSide}");
        List<Player> list;
        using (_playerReadyLock.EnterScope())
        {
            _playersTakingExtraTurn.Clear();
            foreach (Player player in _state.Players)
            {
                if (Hook.ShouldTakeExtraTurn(_state, player))
                {
                    Log.Info($"Player {player.NetId} ({player.Character.Id.Entry}) is taking an extra turn");
                    _playersTakingExtraTurn.Add(player);
                }
            }
            list = _playersTakingExtraTurn.ToList();
        }
        SwitchSides();
        TracePureSimCombat($"switch_from_player_to_enemy_side.after_switch round={_state?.RoundNumber} side={_state?.CurrentSide}");
        foreach (Player item in list)
        {
            await Hook.AfterTakingExtraTurn(_state, item);
        }
        await WaitForUnpause();
        await StartTurn(actionDuringEnemyTurn);
        TracePureSimCombat($"switch_from_player_to_enemy_side.done round={_state?.RoundNumber} side={_state?.CurrentSide}");
    }

    private void SwitchSides()
    {
        bool flag;
        using (_playerReadyLock.EnterScope())
        {
            flag = _playersTakingExtraTurn.Count > 0;
        }
        if (_state.CurrentSide == CombatSide.Player && !flag)
        {
            _state.CurrentSide = CombatSide.Enemy;
        }
        else
        {
            _state.CurrentSide = CombatSide.Player;
            _state.RoundNumber++;
        }
        foreach (Creature creature in _state.Creatures)
        {
            creature.OnSideSwitch();
        }
        this.TurnEnded?.Invoke(_state);
    }

    public void Pause()
    {
        if (!NonInteractiveMode.IsActive && IsInProgress)
        {
            IsPaused = true;
        }
    }

    public void Unpause()
    {
        if (!NonInteractiveMode.IsActive)
        {
            IsPaused = false;
        }
    }

    public async Task WaitForUnpause()
    {
        if (!NonInteractiveMode.IsActive)
        {
            while (IsPaused && IsInProgress)
            {
                await NGame.Instance.ToSignal(NGame.Instance.GetTree(), SceneTree.SignalName.ProcessFrame);
            }
        }
    }

    public void DebugForceTopCardOnNextShuffle(CardModel card)
    {
        card.AssertMutable();
        DebugForcedTopCardOnNextShuffle = card;
    }

    public void DebugClearForcedTopCardOnNextShuffle()
    {
        DebugForcedTopCardOnNextShuffle = null;
    }
}
