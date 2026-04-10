using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.Training;
using MegaCrit.Sts2.Core.Unlocks;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Multiplayer.Game;

namespace MegaCrit.Sts2.Core.Simulation;

public sealed class FullRunSimulatorRuntimeFacade : IFullRunRuntimeFacade, IDisposable
{
	private IDisposable? _runtimeScope;

	private IDisposable? _selectorScope;

	private bool _forceMapView;

	private FullRunSimulationStateSnapshot? _lastObservedState;

	/// <summary>
	/// Prevents calling OfferForRoomEnd more than once per combat.
	/// Reset in ResetAsync (new episode) and StepCombatAsync (new combat action).
	/// </summary>
	private bool _rewardsTriggered;

	private bool _suppressTerminalRewardsTransitionOnce;

	/// <summary>
	/// MCTS state snapshot cache. Maps state_id → serialized run snapshot.
	/// Used for save/load state during tree search.
	/// </summary>
	private readonly Dictionary<string, SavedRunSnapshot> _stateCache = new();
	private int _nextStateId;

	public sealed class SavedRunSnapshot
	{
		public Saves.SerializableRun RunSnapshot { get; set; } = null!;

		public string ExactSignature { get; set; } = string.Empty;

		public string StateType { get; set; } = string.Empty;

		public FullRunPendingSelectionRestoreSnapshot? PendingSelection { get; set; }

		public SavedCombatSnapshot? CombatSnapshot { get; set; }

		public SavedShopSnapshot? ShopSnapshot { get; set; }

		public SavedTreasureSnapshot? TreasureSnapshot { get; set; }
	}

	public sealed class SavedShopSnapshot
	{
		public List<SavedShopCardEntry> CharacterCards { get; init; } = new();

		public List<SavedShopCardEntry> ColorlessCards { get; init; } = new();

		public List<SavedShopRelicEntry> Relics { get; init; } = new();

		public List<SavedShopPotionEntry> Potions { get; init; } = new();

		public SavedShopCardRemovalEntry? CardRemoval { get; init; }
	}

	public sealed class SavedShopCardEntry
	{
		public SerializableCard? Card { get; init; }

		public int Cost { get; init; }

		public bool IsOnSale { get; init; }
	}

	public sealed class SavedShopRelicEntry
	{
		public SerializableRelic? Relic { get; init; }

		public int Cost { get; init; }
	}

	public sealed class SavedShopPotionEntry
	{
		public SerializablePotion? Potion { get; init; }

		public int Cost { get; init; }
	}

	public sealed class SavedShopCardRemovalEntry
	{
		public bool Used { get; init; }

		public int Cost { get; init; }
	}

	public sealed class SavedTreasureSnapshot
	{
		public List<ModelId>? Relics { get; init; }
	}

	public sealed class SavedCombatSnapshot
	{
		public int RoundNumber { get; init; }

		public CombatSide CurrentSide { get; init; }

		public bool IsPlayPhase { get; init; }

		public bool PlayerActionsDisabled { get; init; }

		public bool IsEnemyTurnStarted { get; init; }

		public Dictionary<ulong, SavedCombatPlayerSnapshot> Players { get; init; } = new();

		public List<SerializableCreatureState> Creatures { get; init; } = new();

		public List<SavedCombatMonsterMoveSnapshot> MonsterMoves { get; init; } = new();
	}

	public sealed class SavedCombatPlayerSnapshot
	{
		public int CurrentHp { get; init; }

		public SerializablePlayerCombatState CombatState { get; init; } = new();

		public List<SerializableCard> PlayPile { get; init; } = new();

		public int Stars { get; init; }
	}

	public sealed class SavedCombatMonsterMoveSnapshot
	{
		public uint CombatId { get; init; }

		public string CurrentMoveId { get; init; } = string.Empty;

		public IReadOnlyList<string> StateLogIds { get; init; } = Array.Empty<string>();

		public bool PerformedFirstMove { get; init; }

		public bool CurrentMovePerformedAtLeastOnce { get; init; }

		public bool SpawnedThisTurn { get; init; }
	}

	private static readonly FieldInfo CombatManagerIsPlayPhaseField = typeof(CombatManager).GetField("<IsPlayPhase>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate CombatManager.IsPlayPhase backing field.");

	private static readonly FieldInfo CombatManagerIsEnemyTurnStartedField = typeof(CombatManager).GetField("<IsEnemyTurnStarted>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate CombatManager.IsEnemyTurnStarted backing field.");

	private static readonly FieldInfo CombatManagerEndingPlayerTurnPhaseOneField = typeof(CombatManager).GetField("<EndingPlayerTurnPhaseOne>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate CombatManager.EndingPlayerTurnPhaseOne backing field.");

	private static readonly FieldInfo CombatManagerEndingPlayerTurnPhaseTwoField = typeof(CombatManager).GetField("<EndingPlayerTurnPhaseTwo>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate CombatManager.EndingPlayerTurnPhaseTwo backing field.");

	private static readonly FieldInfo CombatManagerPlayerActionsDisabledField = typeof(CombatManager).GetField("_playerActionsDisabled", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate CombatManager._playerActionsDisabled field.");

	private static readonly FieldInfo MonsterMoveStateMachinePerformedFirstMoveField = typeof(MonsterMoveStateMachine).GetField("_performedFirstMove", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MonsterMoveStateMachine._performedFirstMove field.");

	private static readonly FieldInfo MoveStatePerformedAtLeastOnceField = typeof(MoveState).GetField("_performedAtLeastOnce", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MoveState._performedAtLeastOnce field.");

	private static readonly FieldInfo MonsterSpawnedThisTurnField = typeof(MonsterModel).GetField("_spawnedThisTurn", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MonsterModel._spawnedThisTurn field.");

	private static readonly FieldInfo MerchantEntryCostField = typeof(MerchantEntry).GetField("_cost", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MerchantEntry._cost field.");

	private static readonly FieldInfo MerchantEntryPlayerField = typeof(MerchantEntry).GetField("_player", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MerchantEntry._player field.");

	private static readonly FieldInfo MerchantCardEntryCreationResultField = typeof(MerchantCardEntry).GetField("<CreationResult>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MerchantCardEntry.CreationResult backing field.");

	private static readonly FieldInfo MerchantCardEntryIsOnSaleField = typeof(MerchantCardEntry).GetField("<IsOnSale>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MerchantCardEntry.IsOnSale backing field.");

	private static readonly FieldInfo MerchantRelicEntryModelField = typeof(MerchantRelicEntry).GetField("<Model>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MerchantRelicEntry.Model backing field.");

	private static readonly FieldInfo MerchantPotionEntryModelField = typeof(MerchantPotionEntry).GetField("<Model>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MerchantPotionEntry.Model backing field.");

	private static readonly FieldInfo MerchantCardRemovalUsedField = typeof(MerchantCardRemovalEntry).GetField("<Used>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate MerchantCardRemovalEntry.Used backing field.");

	private static readonly FieldInfo TreasureRoomCurrentRelicsField = typeof(TreasureRoomRelicSynchronizer).GetField("_currentRelics", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate TreasureRoomRelicSynchronizer._currentRelics field.");

	private static readonly FieldInfo TreasureRoomVotesField = typeof(TreasureRoomRelicSynchronizer).GetField("_votes", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate TreasureRoomRelicSynchronizer._votes field.");

	private static readonly FieldInfo TreasureRoomPredictedVoteField = typeof(TreasureRoomRelicSynchronizer).GetField("_predictedVote", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate TreasureRoomRelicSynchronizer._predictedVote field.");

	public bool IsActive => _runtimeScope != null;

	public bool IsPureSimulator => true;

	public int EpisodeNumber { get; private set; }

	public string? CurrentSeed { get; private set; }

	public string? CurrentCharacterId { get; private set; }

	public int CurrentAscensionLevel { get; private set; }

	public async Task<FullRunSimulationStateSnapshot> ResetAsync(FullRunSimulationResetRequest? request = null)
	{
		FullRunSimulationTrace.Write("headless_reset.begin");
		CleanUpPreviousEpisode();
		FullRunSimulationTrace.Write("headless_reset.after_cleanup");
		// TestMode makes the entire engine skip Godot-specific code:
		// PreloadManager skips asset loading, MapRoom.Enter() skips scene creation,
		// Cmd.Wait/VfxCmd/CardCmd skip animations — pure game logic only.
		TestSupport.TestMode.IsOn = true;
		_runtimeScope = new SimulationRuntimeScope();
		_selectorScope = CardSelectCmd.UseSelector(FullRunSimulationChoiceBridge.Instance);
		FullRunSimulationChoiceBridge.Instance.Reset();
		_forceMapView = false;
		_rewardsTriggered = false;
		FullRunSimulationTrace.Write("headless_reset.after_runtime_scope");
		SaveManager.Instance?.SetFtuesEnabled(enabled: false);
		if (SaveManager.Instance != null)
		{
			SaveManager.Instance.PrefsSave.FastMode = FastModeType.Instant;
		}
		LocalContext.NetId = NetSingleplayerGameService.defaultNetId;
		CharacterModel character = CombatTrainingMode.ResolveCharacter(request?.Character ?? request?.CharacterId);
		string seed = CombatTrainingMode.ResolveEpisodeSeed(request?.Seed);
		int ascensionLevel = CombatTrainingMode.ResolveAscensionLevel(request?.Ascension ?? request?.AscensionLevel);
		// Standalone pure-sim runs against mock saves in TestMode. Use a stable
		// fully-unlocked, non-first-run unlock state so tutorial/first-run
		// branches match the mature simulator backend instead of empty test saves.
		UnlockState unlockState = UnlockState.all;
		List<ActModel> acts = ActModel.GetDefaultList().Select(static act => act.ToMutable()).ToList();
		Player player = Player.CreateForNewRun(character, unlockState, NetSingleplayerGameService.defaultNetId);
		RunState runState = RunState.CreateForNewRun(new List<Player> { player }, acts, Array.Empty<ModifierModel>(), ascensionLevel, seed);
		FullRunSimulationTrace.Write($"headless_reset.after_create_run seed={seed} character={character.Id.Entry} ascension={ascensionLevel}");
		try
		{
			RunManager.Instance.SetUpTest(runState, new NetSingleplayerGameService(), shouldSave: false);
		}
		catch (Exception ex)
		{
			FullRunSimulationTrace.Write($"headless_reset.setup_test_exception={ex}");
			throw;
		}
		FullRunSimulationTrace.Write("headless_reset.after_setup_test");
		try
		{
			RunManager.Instance.GenerateRooms();
		}
		catch (Exception ex)
		{
			FullRunSimulationTrace.Write($"headless_reset.generate_rooms_exception={ex}");
			throw;
		}
		FullRunSimulationTrace.Write("headless_reset.after_generate_rooms");
		await PreloadManager.LoadRunAssets(runState.Players.Select(static p => p.Character));
		FullRunSimulationTrace.Write("headless_reset.after_load_run_assets");
		await PreloadManager.LoadActAssets(runState.Acts[0]);
		FullRunSimulationTrace.Write("headless_reset.after_load_act_assets");
		await RunManager.Instance.FinalizeStartingRelics();
		FullRunSimulationTrace.Write("headless_reset.after_finalize_starting_relics");
		RunManager.Instance.Launch();
		FullRunSimulationTrace.Write("headless_reset.after_launch");
		// Pure simulator: bypass EnterAct's Godot scene tree operations.
		// SetActInternal generates the map, PushRoom sets CurrentRoom so
		// the state builder returns "map" instead of "run_bootstrap".
		await RunManager.Instance.SetActInternal(0);
		FullRunSimulationTrace.Write("headless_reset.after_set_act_internal");
		runState.PushRoom(new MapRoom());
		FullRunSimulationTrace.Write("headless_reset.after_push_map_room");
		EpisodeNumber++;
		CurrentSeed = seed;
		CurrentCharacterId = character.Id.Entry;
		CurrentAscensionLevel = ascensionLevel;
		FullRunSimulationStateSnapshot state = await WaitForStateChangeAsync(previousState: null);
		FullRunSimulationTrace.Write($"headless_reset.done state_type={state.StateType} floor={state.TotalFloor} terminal={state.IsTerminal}");
		return state;
	}

	public FullRunSimulationStateSnapshot GetState()
	{
		return BuildStateSnapshot(RunManager.Instance.DebugOnlyGetState(), cachedCombatState: null);
	}

	private FullRunSimulationStateSnapshot BuildStateSnapshot(RunState? runState, CombatTrainingStateSnapshot? cachedCombatState)
	{
		FullRunSimulationStateSnapshot snapshot = FullRunSimulationStateBuilder.Build(
			runState,
			FullRunSimulationChoiceBridge.Instance,
			isPureSimulator: IsPureSimulator,
			backendKind: "full_run_pure_simulator",
			coverageTier: "full_run_v1",
			forceMapView: _forceMapView,
			overrideCharacterId: CurrentCharacterId,
			overrideSeed: CurrentSeed,
			overrideAscensionLevel: CurrentAscensionLevel,
			cachedCombatState: cachedCombatState);
		_lastObservedState = snapshot;
		return snapshot;
	}

	public Task<FullRunSimulationStepResult> StepAsync(FullRunSimulationActionRequest action)
	{
		if (action == null)
		{
			return Task.FromResult(BuildRejected("Action request is required.", "full_run_action_required"));
		}
		return StepInternalAsync(action);
	}

	/// <summary>
	/// Execute a batch of actions sequentially, returning the final result.
	/// All actions run within a single main-thread invocation — no HTTP or
	/// frame-scheduling overhead between steps.  If any action is rejected,
	/// the batch stops and returns that rejection along with how many
	/// actions succeeded.
	/// </summary>
	public async Task<FullRunSimulationBatchStepResult> BatchStepAsync(IReadOnlyList<FullRunSimulationActionRequest> actions)
	{
		if (actions == null || actions.Count == 0)
		{
			return new FullRunSimulationBatchStepResult
			{
				Accepted = false,
				Error = "Empty action batch.",
				StepsExecuted = 0,
				State = GetState()
			};
		}
		int executed = 0;
		FullRunSimulationStateSnapshot? lastState = _lastObservedState;
		for (int i = 0; i < actions.Count; i++)
		{
			var result = await StepInternalAsync(actions[i]);
			executed++;
			if (!result.Accepted)
			{
				return new FullRunSimulationBatchStepResult
				{
					Accepted = false,
					Error = result.Error,
					FailureCode = result.FailureCode,
					StepsExecuted = executed,
					State = result.State ?? GetState()
				};
			}
			// If terminal, stop early
			if (result.State?.IsTerminal == true)
			{
				return new FullRunSimulationBatchStepResult
				{
					Accepted = true,
					StepsExecuted = executed,
					State = result.State
				};
			}
			lastState = result.State;
		}
		return new FullRunSimulationBatchStepResult
		{
			Accepted = true,
			StepsExecuted = executed,
			State = lastState ?? _lastObservedState ?? GetState()
		};
	}

	public void Dispose()
	{
		CleanUpPreviousEpisode();
	}

	private async Task<FullRunSimulationStepResult> StepInternalAsync(FullRunSimulationActionRequest action)
	{
		FullRunSimulationStateSnapshot state = _lastObservedState ?? GetState();
		string actionType = NormalizeActionType(action);
		switch (actionType)
		{
			case "start_run":
				if (state.StateType != "menu")
				{
					return BuildRejected("start_run is only valid from menu.", "full_run_invalid_state");
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await ResetAsync(new FullRunSimulationResetRequest
					{
						CharacterId = action.Value ?? CurrentCharacterId,
						Seed = CurrentSeed,
						AscensionLevel = CurrentAscensionLevel
					})
				};
			case "choose_map_node":
				if (state.StateType != "map")
				{
					return BuildRejected("choose_map_node is only valid from map.", "full_run_invalid_state");
				}
				if (!TryResolveMapCoord(state, action, out MapCoord destination))
				{
					return BuildRejected("Could not resolve requested map node.", "full_run_map_coord_not_found");
				}
				_forceMapView = false;
				await RunManager.Instance.EnterMapCoord(destination);
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "choose_event_option":
				if (state.StateType != "event")
				{
					return BuildRejected("choose_event_option is only valid from event.", "full_run_invalid_state");
				}
				if (!action.Index.HasValue)
				{
					return BuildRejected("choose_event_option requires an index.", "full_run_action_required");
				}
				{
					// Event options are a mix of:
					// 1. Synchronous state transitions (REST -> FIGHT, proceed, etc.)
					// 2. Actions that open a selector and intentionally block until player input
					//    (upgrade/remove/transform events).
					// Start the event task, give it a moment to either finish or register a
					// pending selection, then return whichever state is now actionable.
					Task<bool> eventTask = TryChooseEventOptionAsync(action.Index.Value);
					await CombatSimulationRuntime.Clock.YieldAsync();

					if (eventTask.IsCompleted)
					{
						if (!await eventTask)
						{
							return BuildRejected("Could not resolve requested event option.", "full_run_event_option_not_found");
						}
						return new FullRunSimulationStepResult
						{
							Accepted = true,
							State = await WaitForStateChangeAsync(state)
						};
					}

					// In pure-sim, Clock.YieldAsync() is a no-op — cap to 1 iteration.
					int eventWaitMax = CombatSimulationRuntime.IsPureCombatSimulator ? 1 : 20;
					for (int waitIter = 0; waitIter < eventWaitMax; waitIter++)
					{
						await CombatSimulationRuntime.Clock.YieldAsync();
						if (FullRunSimulationChoiceBridge.Instance.IsSelectionActive || eventTask.IsCompleted)
						{
							break;
						}
					}

					if (eventTask.IsCompleted)
					{
						if (!await eventTask)
						{
							return BuildRejected("Could not resolve requested event option.", "full_run_event_option_not_found");
						}
						return new FullRunSimulationStepResult
						{
							Accepted = true,
							State = await WaitForStateChangeAsync(state)
						};
					}

					return new FullRunSimulationStepResult
					{
						Accepted = true,
						State = GetState()
					};
				}
			case "advance_dialogue":
				if (state.StateType != "event")
				{
					return BuildRejected("advance_dialogue is only valid from event.", "full_run_invalid_state");
				}
				if (!TryAdvanceEventDialogue())
				{
					return BuildRejected("Dialogue advance is not currently available for the active event.", "full_run_event_dialogue_unavailable");
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "proceed":
				return await StepProceedAsync(state);
			case "choose_rest_option":
				if (state.StateType != "rest_site")
				{
					return BuildRejected("choose_rest_option is only valid from rest_site.", "full_run_invalid_state");
				}
				if (!action.Index.HasValue)
				{
					return BuildRejected("choose_rest_option requires an index.", "full_run_action_required");
				}
				{
					// Fire-and-forget: ChooseLocalOption may block waiting for card selection
					// (e.g., smith/forge opens card picker). We start it but don't await completion.
					// Instead, poll for state change or pending card selection.
					Task<bool> restTask = RunManager.Instance.RestSiteSynchronizer.ChooseLocalOption(action.Index.Value);

					// Give it a moment to complete synchronously (rest/heal complete immediately)
					await CombatSimulationRuntime.Clock.YieldAsync();

					if (restTask.IsCompleted)
					{
						if (!restTask.Result)
						{
							return BuildRejected("Could not resolve requested rest site option.", "full_run_rest_site_option_not_found");
						}
						return new FullRunSimulationStepResult
						{
							Accepted = true,
							State = await WaitForStateChangeAsync(state)
						};
					}

					// Task didn't complete → likely waiting for card selection (smith/forge).
					// Check if a card selection prompt is now pending.
					// In pure-sim, Clock.YieldAsync() is a no-op — cap to 1 iteration.
					int restWaitMax = CombatSimulationRuntime.IsPureCombatSimulator ? 1 : 20;
					for (int waitIter = 0; waitIter < restWaitMax; waitIter++)
					{
						await CombatSimulationRuntime.Clock.YieldAsync();
						if (FullRunSimulationChoiceBridge.Instance.IsSelectionActive || restTask.IsCompleted)
							break;
					}

					// Return current state (should be card_select if smith triggered)
					return new FullRunSimulationStepResult
					{
						Accepted = true,
						State = GetState()
					};
				}
			case "shop_purchase":
				if (state.StateType != "shop")
				{
					return BuildRejected("shop_purchase is only valid from shop.", "full_run_invalid_state");
				}
				if (!action.Index.HasValue)
				{
					return BuildRejected("shop_purchase requires an index.", "full_run_action_required");
				}
				{
					// Fire-and-forget: some shop purchases (card removal) open card selection
					Task<bool> shopTask = TryPurchaseShopItemAsync(action.Index.Value);
					await CombatSimulationRuntime.Clock.YieldAsync();

					if (shopTask.IsCompleted)
					{
						if (!shopTask.Result)
						{
							return BuildRejected("Could not resolve requested shop entry.", "full_run_shop_item_not_found");
						}
						return new FullRunSimulationStepResult
						{
							Accepted = true,
							State = await WaitForStateChangeAsync(state)
						};
					}

					// In pure-sim, Clock.YieldAsync() is a no-op — cap to 1 iteration.
					int shopWaitMax = CombatSimulationRuntime.IsPureCombatSimulator ? 1 : 20;
					for (int waitIter = 0; waitIter < shopWaitMax; waitIter++)
					{
						await CombatSimulationRuntime.Clock.YieldAsync();
						if (FullRunSimulationChoiceBridge.Instance.IsSelectionActive || shopTask.IsCompleted)
							break;
					}

					return new FullRunSimulationStepResult
					{
						Accepted = true,
						State = GetState()
					};
				}
			case "claim_treasure_relic":
				if (state.StateType != "treasure")
				{
					return BuildRejected("claim_treasure_relic is only valid from treasure.", "full_run_invalid_state");
				}
				if (!action.Index.HasValue)
				{
					return BuildRejected("claim_treasure_relic requires an index.", "full_run_action_required");
				}
				if (!TryClaimTreasureRelic(action.Index.Value))
				{
					return BuildRejected("Could not resolve requested treasure relic.", "full_run_treasure_relic_not_found");
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "claim_reward":
				if (state.StateType != "combat_rewards")
				{
					return BuildRejected("claim_reward is only valid from combat_rewards.", "full_run_invalid_state");
				}
				if (!action.Index.HasValue)
				{
					return BuildRejected("claim_reward requires an index.", "full_run_action_required");
				}
				if (!FullRunSimulationChoiceBridge.Instance.TrySelectReward(action.Index.Value, out string rewardError))
				{
					return BuildRejected(rewardError, "full_run_reward_not_found");
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "select_card_reward":
				if (state.StateType != "card_reward")
				{
					return BuildRejected("select_card_reward is only valid from card_reward.", "full_run_invalid_state");
				}
				if (!TryResolveSelectionIndex(action, out int cardRewardIndex))
				{
					return BuildRejected("select_card_reward requires an index.", "full_run_action_required");
				}
				if (!FullRunSimulationChoiceBridge.Instance.TrySelectCardReward(cardRewardIndex, out string cardRewardError))
				{
					return BuildRejected(cardRewardError, "full_run_card_reward_not_found");
				}
				if (CombatSimulationRuntime.IsPureCombatSimulator)
				{
					await CombatSimulationRuntime.Clock.YieldAsync();
				}
				else
				{
					await Task.Yield();
				}
				await RunManager.Instance.ActionExecutor.FinishedExecutingActions();
				if (_suppressTerminalRewardsTransitionOnce)
				{
					_suppressTerminalRewardsTransitionOnce = false;
				}
				else
				{
					await TriggerTerminalRewardsTransitionIfNeededAsync();
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "skip_card_reward":
				if (state.StateType != "card_reward")
				{
					return BuildRejected("skip_card_reward is only valid from card_reward.", "full_run_invalid_state");
				}
				if (!FullRunSimulationChoiceBridge.Instance.TrySkipCardReward(out string skipCardRewardError))
				{
					return BuildRejected(skipCardRewardError, "full_run_card_reward_skip_unavailable");
				}
				if (CombatSimulationRuntime.IsPureCombatSimulator)
				{
					await CombatSimulationRuntime.Clock.YieldAsync();
				}
				else
				{
					await Task.Yield();
				}
				await RunManager.Instance.ActionExecutor.FinishedExecutingActions();
				if (_suppressTerminalRewardsTransitionOnce)
				{
					_suppressTerminalRewardsTransitionOnce = false;
				}
				else
				{
					await TriggerTerminalRewardsTransitionIfNeededAsync();
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "select_card":
				if (state.StateType != "card_select")
				{
					return BuildRejected("select_card is only valid from card_select.", "full_run_invalid_state");
				}
				if (!action.Index.HasValue)
				{
					return BuildRejected("select_card requires an index.", "full_run_action_required");
				}
				if (!FullRunSimulationChoiceBridge.Instance.TrySelectCardChoice(action.Index.Value, out string cardSelectError))
				{
					return BuildRejected(cardSelectError, "full_run_card_select_not_found");
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "confirm_selection":
				if (state.StateType != "card_select")
				{
					return BuildRejected("confirm_selection is only valid from card_select.", "full_run_invalid_state");
				}
				if (!FullRunSimulationChoiceBridge.Instance.TryConfirmSelection(out string confirmError))
				{
					return BuildRejected(confirmError, "full_run_card_select_confirm_unavailable");
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "cancel_selection":
				if (state.StateType != "card_select")
				{
					return BuildRejected("cancel_selection is only valid from card_select.", "full_run_invalid_state");
				}
				if (!FullRunSimulationChoiceBridge.Instance.TryCancelSelection(out string cancelError))
				{
					return BuildRejected(cancelError, "full_run_card_select_cancel_unavailable");
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "select_relic":
				if (state.StateType != "relic_select")
				{
					return BuildRejected("select_relic is only valid from relic_select.", "full_run_invalid_state");
				}
				if (!action.Index.HasValue)
				{
					return BuildRejected("select_relic requires an index.", "full_run_action_required");
				}
				if (!FullRunSimulationChoiceBridge.Instance.TrySelectRelic(action.Index.Value, out string relicError))
				{
					return BuildRejected(relicError, "full_run_relic_select_not_found");
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "skip_relic_selection":
				if (state.StateType != "relic_select")
				{
					return BuildRejected("skip_relic_selection is only valid from relic_select.", "full_run_invalid_state");
				}
				if (!FullRunSimulationChoiceBridge.Instance.TrySkipRelicSelection(out string skipRelicError))
				{
					return BuildRejected(skipRelicError, "full_run_relic_select_skip_unavailable");
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitForStateChangeAsync(state)
				};
			case "play_card":
			case "end_turn":
			case "combat_select_card":
			case "combat_confirm_selection":
			case "select_card_option":
			case "use_potion":
				return await StepCombatAsync(state, actionType, action);
			case "wait":
			{
				FullRunSimulationStateSnapshot waitState = state;
				if (waitState.StateType == "combat_pending" && RunManager.Instance.IsInProgress)
				{
					RunState? runState = RunManager.Instance.DebugOnlyGetState();
					if (runState != null && runState.IsGameOver)
					{
						return new FullRunSimulationStepResult
						{
							Accepted = true,
							State = GetState()
						};
					}

					CombatRoom? pendingCombatRoom = runState?.CurrentRoom as CombatRoom;
					bool triggeredRewards = TryTriggerPostCombatRewards(runState, pendingCombatRoom);
					if (triggeredRewards)
					{
						_forceMapView = false;
					}
					return new FullRunSimulationStepResult
					{
						Accepted = true,
						State = await WaitExplicitlyAsync(waitState)
					};
				}
				return new FullRunSimulationStepResult
				{
					Accepted = true,
					State = await WaitExplicitlyAsync(waitState)
				};
			}
			default:
				return BuildRejected($"Full-run pure runtime step is not implemented for action '{actionType}'.", "full_run_step_not_implemented");
		}
	}


	private async Task<FullRunSimulationStepResult> StepProceedAsync(FullRunSimulationStateSnapshot state)
	{
		switch (state.StateType)
		{
			case "event":
				if (!TryProceedEvent())
				{
					return BuildRejected("Proceed is not currently available for the active event.", "full_run_event_proceed_unavailable");
				}
				break;
			case "combat_rewards":
				if (!FullRunSimulationChoiceBridge.Instance.TryProceedRewards(out string rewardsError))
				{
					return BuildRejected(rewardsError, "full_run_rewards_proceed_unavailable");
				}
				if (Engine.GetMainLoop() != null && !CombatSimulationRuntime.IsPureCombatSimulator)
				{
					await Task.Yield();
				}
				await RunManager.Instance.ActionExecutor.FinishedExecutingActions();
				await TriggerTerminalRewardsTransitionIfNeededAsync();
				if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is not EventRoom)
				{
					_forceMapView = true;
				}
				break;
			case "rest_site":
			case "shop":
			case "treasure":
				_forceMapView = true;
				break;
			default:
				return BuildRejected("proceed is not implemented for the current full-run state.", "full_run_step_not_implemented");
		}
		return new FullRunSimulationStepResult
		{
			Accepted = true,
			State = await WaitForStateChangeAsync(state)
		};
	}

	private async Task TriggerTerminalRewardsTransitionIfNeededAsync()
	{
		// Mirrors the logic in NRewardsScreen.OnProceedButtonPressed and also
		// covers boss card_reward closeout, which bypasses the combat_rewards
		// proceed button in headless mode.
		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		if (runState == null || runState.CurrentRoom == null)
		{
			return;
		}
		bool forceMapAfterEventCombat = runState.CurrentRoom is CombatRoom
		{
			ParentEventId: not null,
			ShouldResumeParentEventAfterCombat: false
		};
		if (runState.CurrentRoom.RoomType == RoomType.Boss || runState.CurrentRoom.IsVictoryRoom)
		{
			if (runState.Map.SecondBossMapPoint != null
				&& runState.CurrentMapCoord == runState.Map.BossMapPoint.coord)
			{
				await RunManager.Instance.ProceedFromTerminalRewardsScreen();
			}
			else
			{
				RunManager.Instance.ActChangeSynchronizer.SetLocalPlayerReady();
			}
		}
		else
		{
			await RunManager.Instance.ProceedFromTerminalRewardsScreen();
		}
		if (forceMapAfterEventCombat)
		{
			_forceMapView = true;
		}
	}

	private async Task<FullRunSimulationStepResult> StepCombatAsync(FullRunSimulationStateSnapshot state, string actionType, FullRunSimulationActionRequest action)
	{
		if (!IsCombatState(state.StateType))
		{
			return BuildRejected($"'{actionType}' is only valid during combat.", "full_run_invalid_state");
		}
		CombatTrainingActionRequest combatAction = TranslateCombatAction(state, actionType, action);
		CombatTrainingStepResult result;
		using (FullRunSimulationDiagnostics.Measure("combat_step.game_logic_ms"))
		{
			result = await CombatTrainingEnvService.StepAgainstActiveCombatAsync(combatAction);
		}
		if (!result.Accepted)
		{
			return new FullRunSimulationStepResult
			{
				Accepted = false,
				Error = result.Error,
				FailureCode = "full_run_combat_action_rejected",
				State = _lastObservedState ?? state
			};
		}
		_forceMapView = false;
		_rewardsTriggered = false;
		if (result.State != null && IsActionableCombatState(result.State))
		{
			FullRunSimulationStepResult fastResult;
			using (FullRunSimulationDiagnostics.Measure("combat_step.fast_path_build_ms"))
			{
				fastResult = new FullRunSimulationStepResult
				{
					Accepted = true,
					State = BuildStateSnapshot(RunManager.Instance.DebugOnlyGetState(), result.State)
				};
			}
			FullRunSimulationDiagnostics.Increment("combat_step.fast_path_hits");
			return fastResult;
		}
		FullRunSimulationDiagnostics.Increment("combat_step.slow_path_hits");
		return new FullRunSimulationStepResult
		{
			Accepted = true,
			State = await WaitForCombatFollowupAsync(state)
		};
	}

	private CombatTrainingActionRequest TranslateCombatAction(FullRunSimulationStateSnapshot state, string actionType, FullRunSimulationActionRequest action)
	{
		switch (actionType)
		{
			case "play_card":
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.PlayCard,
					HandIndex = action.CardIndex ?? action.HandIndex ?? action.Index,
					TargetId = ResolveCombatTargetId(action)
				};
			case "end_turn":
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.EndTurn
				};
			case "combat_select_card":
				return new CombatTrainingActionRequest
				{
					Type = state.StateType == "hand_select" ? CombatTrainingActionType.SelectHandCard : CombatTrainingActionType.SelectCardChoice,
					HandIndex = action.CardIndex ?? action.HandIndex ?? action.Index,
					ChoiceIndex = action.CardIndex ?? action.HandIndex ?? action.Index
				};
			case "select_card_option":
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.SelectCardChoice,
					ChoiceIndex = action.CardIndex ?? action.Index
				};
			case "combat_confirm_selection":
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.ConfirmSelection
				};
			case "use_potion":
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.UsePotion,
					Slot = action.Slot ?? action.Index,
					TargetId = ResolveCombatTargetId(action)
				};
			default:
				throw new InvalidOperationException($"Unsupported combat action type '{actionType}'.");
		}
	}

	private async Task<FullRunSimulationStateSnapshot> WaitForStateChangeAsync(FullRunSimulationStateSnapshot? previousState)
	{
		FullRunSimulationDiagnostics.Increment("settle.wait_state_change.calls");
		string? previousSignature = previousState == null ? null : BuildStateChangeSignature(previousState);

		// Phase 1: synchronous settlement — no ProcessFrame wait.
		// In pure-simulator mode most actions (reward claims, event choices, card
		// plays…) execute synchronously, so the state is ready immediately after
		// the ActionExecutor drains. This avoids the 1-frame overhead that the old
		// 240-iteration loop paid on every single action.
		// Use Clock.YieldAsync() instead of Task.Yield() so that ImmediateSimulatorClock
		// returns Task.CompletedTask and continues synchronously (no frame wait).
		await CombatSimulationRuntime.Clock.YieldAsync();
		if (RunManager.Instance.IsInProgress)
		{
			Task execTask = RunManager.Instance.ActionExecutor.FinishedExecutingActions();
			if (!execTask.IsCompleted && CombatSimulationRuntime.IsPureCombatSimulator)
			{
				// In pure-simulator mode, if the executor is blocked (waiting for
				// UI input like card_select/hand_select), awaiting it deadlocks.
				// Return current state so the caller can handle the selection.
			}
			else
			{
				await execTask;
			}
		}
		{
			ObservedState early = ObserveState();
			bool earlyChanged = previousSignature == null
				? IsReadySnapshot(early.Snapshot)
				: early.Signature != previousSignature
					&& IsReadySnapshot(early.Snapshot);
			if (earlyChanged)
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_state_change.phase1_hits");
				return early.Snapshot;
			}
		}

		// Phase 2: fallback for scene transitions (map → room, room → map).
		// In pure simulator mode (ImmediateSimulatorClock), Phase 1 is sufficient —
		// all game logic runs synchronously, so state is always settled after Phase 1.
		// Skip Phase 2 entirely when Godot engine is absent or in pure sim mode.
		ISimulatorClock clock = CombatSimulationRuntime.Clock;
		bool canUseFrameDeadline = Engine.GetMainLoop() != null && !CombatSimulationRuntime.IsPureCombatSimulator;
		ulong stateTimeout = 30UL;
		ulong stateDeadline = canUseFrameDeadline ? Time.GetTicksMsec() + stateTimeout : 0UL;
		// When no frame-based deadline is available (pure-sim or no engine),
		// Clock.YieldAsync() is a no-op and we never Task.Yield(), so state
		// cannot change between iterations — Phase 2 is pure waste. Skip it.
		int maxPhase2Iters = canUseFrameDeadline ? 100 : 0;
		if (maxPhase2Iters == 0)
		{
			FullRunSimulationDiagnostics.Increment("settle.wait_state_change.phase2_skipped_no_frame");
		}
		for (int iter = 0; iter < maxPhase2Iters; iter++)
		{
			FullRunSimulationDiagnostics.Increment("settle.wait_state_change.phase2_iterations");
			if (canUseFrameDeadline && Time.GetTicksMsec() >= stateDeadline)
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_state_change.phase2_deadline_breaks");
				break;
			}
			await clock.YieldAsync();
			if (canUseFrameDeadline)
			{
				await Task.Yield();
			}
			if (RunManager.Instance.IsInProgress)
			{
				Task execTask = RunManager.Instance.ActionExecutor.FinishedExecutingActions();
				if (!execTask.IsCompleted && CombatSimulationRuntime.IsPureCombatSimulator)
				{
					FullRunSimulationDiagnostics.Increment("settle.wait_state_change.phase2_blocked_executor");
					ObservedState blocked = ObserveState();
					if (previousSignature == null)
					{
						if (IsReadySnapshot(blocked.Snapshot))
						{
							FullRunSimulationDiagnostics.Increment("settle.wait_state_change.phase2_hits");
							return blocked.Snapshot;
						}
						continue;
					}
					if (blocked.Signature != previousSignature
						&& IsReadySnapshot(blocked.Snapshot))
					{
						FullRunSimulationDiagnostics.Increment("settle.wait_state_change.phase2_hits");
						return blocked.Snapshot;
					}
					continue;
				}
				await execTask;
			}
			ObservedState current = ObserveState();
			if (previousSignature == null)
			{
				if (IsReadySnapshot(current.Snapshot))
				{
					FullRunSimulationDiagnostics.Increment("settle.wait_state_change.phase2_hits");
					return current.Snapshot;
				}
				continue;
			}
			if (current.Signature != previousSignature
				&& IsReadySnapshot(current.Snapshot))
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_state_change.phase2_hits");
				return current.Snapshot;
			}
		}
		FullRunSimulationDiagnostics.Increment("settle.wait_state_change.fallback_get_state");
		return GetState();
	}

	private async Task<FullRunSimulationStateSnapshot> WaitForCombatFollowupAsync(FullRunSimulationStateSnapshot previousState)
	{
		FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.calls");
		string previousSignature = BuildStateChangeSignature(previousState);

		// Phase 1: synchronous settlement — await the ActionExecutor first,
		// then check if state is already actionable without spending a frame.
		// Use Clock.YieldAsync() so ImmediateSimulatorClock continues synchronously.
		await CombatSimulationRuntime.Clock.YieldAsync();
		if (RunManager.Instance.IsInProgress)
		{
			Task execTask = RunManager.Instance.ActionExecutor.FinishedExecutingActions();
			if (!execTask.IsCompleted)
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase1_blocked_executor");
				// Executor blocked — likely waiting for player input (hand_select).
				// Return early if the state is already actionable to avoid deadlock.
				ObservedState mid = ObserveState();
				if (mid.Signature != previousSignature
					&& (!IsCombatState(mid.Snapshot.StateType) || IsActionableCombatSnapshot(mid.Snapshot)))
				{
					FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase1_hits");
					return mid.Snapshot;
				}
				// In pure-simulator mode, if the executor is blocked (waiting for
				// UI input that doesn't exist), awaiting it will deadlock forever.
				// Return current state so Python can handle card_select/hand_select.
				if (CombatSimulationRuntime.IsPureCombatSimulator)
				{
					FullRunSimulationStateSnapshot? yielded = await TryAdvanceBlockedExecutorAsync(
						"settle.wait_combat_followup.phase1",
						previousSignature,
						static (_, observed, signature) => observed.Signature != signature
							&& (!IsCombatState(observed.Snapshot.StateType) || IsActionableCombatSnapshot(observed.Snapshot)));
					if (yielded != null)
					{
						return yielded;
					}
					FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase1_hits");
					return mid.Snapshot;
				}
			}
			await execTask;
		}
		{
			ObservedState early = ObserveState();
			if (early.Signature != previousSignature
				&& (!IsCombatState(early.Snapshot.StateType) || IsActionableCombatSnapshot(early.Snapshot)))
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase1_hits");
				return early.Snapshot;
			}
			if (CombatSimulationRuntime.IsPureCombatSimulator
				&& IsCombatState(early.Snapshot.StateType)
				&& !IsActionableCombatSnapshot(early.Snapshot)
				&& HasPendingPureCombatContinuation())
			{
				FullRunSimulationStateSnapshot? yielded = await TryAdvancePendingCombatContinuationAsync(
					"settle.wait_combat_followup.phase1_pending",
					previousSignature,
					static (_, observed, signature) => observed.Signature != signature
						&& (!IsCombatState(observed.Snapshot.StateType) || IsActionableCombatSnapshot(observed.Snapshot)));
				if (yielded != null)
				{
					return yielded;
				}
			}
		}

		// Phase 2: fallback — handles enemy-turn resolution and post-combat
		// scene transitions. Skip in pure sim mode (ImmediateSimulatorClock makes
		// all game logic synchronous so Phase 1 is always sufficient) or when
		// no Godot engine is available.
		ISimulatorClock combatClock = CombatSimulationRuntime.Clock;
		bool canUseFrameDeadline = Engine.GetMainLoop() != null && !CombatSimulationRuntime.IsPureCombatSimulator;
		ulong combatTimeout = 30UL;
		ulong combatDeadline = canUseFrameDeadline ? Time.GetTicksMsec() + combatTimeout : 0UL;
		// When no frame-based deadline is available (pure-sim or no engine),
		// Clock.YieldAsync() is a no-op and we never Task.Yield(), so state
		// cannot change between iterations — Phase 2 is pure waste. Skip it.
		int maxCombatIters = canUseFrameDeadline ? 100 : 0;
		if (maxCombatIters == 0)
		{
			FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase2_skipped_no_frame");
		}
		for (int citer = 0; citer < maxCombatIters; citer++)
		{
			FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase2_iterations");
			if (canUseFrameDeadline && Time.GetTicksMsec() >= combatDeadline)
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase2_deadline_breaks");
				break;
			}
			await combatClock.YieldAsync();
			if (canUseFrameDeadline)
			{
				await Task.Yield();
			}
			if (RunManager.Instance.IsInProgress)
			{
				Task execTask = RunManager.Instance.ActionExecutor.FinishedExecutingActions();
				if (!execTask.IsCompleted)
				{
					FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase2_blocked_executor");
					ObservedState mid = ObserveState();
					if (mid.Signature != previousSignature
						&& (!IsCombatState(mid.Snapshot.StateType) || IsActionableCombatSnapshot(mid.Snapshot)))
					{
						FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase2_hits");
						return mid.Snapshot;
					}
					if (CombatSimulationRuntime.IsPureCombatSimulator)
					{
						FullRunSimulationStateSnapshot? yielded = await TryAdvanceBlockedExecutorAsync(
							"settle.wait_combat_followup.phase2",
							previousSignature,
							static (_, observed, signature) => observed.Signature != signature
								&& (!IsCombatState(observed.Snapshot.StateType) || IsActionableCombatSnapshot(observed.Snapshot)));
						if (yielded != null)
						{
							return yielded;
						}
					}
				}
				await execTask;
			}
			ObservedState current = ObserveState();
			if (current.Signature == previousSignature)
			{
				if (CombatSimulationRuntime.IsPureCombatSimulator
					&& IsCombatState(current.Snapshot.StateType)
					&& !IsActionableCombatSnapshot(current.Snapshot)
					&& HasPendingPureCombatContinuation())
				{
					FullRunSimulationStateSnapshot? yielded = await TryAdvancePendingCombatContinuationAsync(
						"settle.wait_combat_followup.phase2_pending",
						previousSignature,
						static (_, observed, signature) => observed.Signature != signature
							&& (!IsCombatState(observed.Snapshot.StateType) || IsActionableCombatSnapshot(observed.Snapshot)));
					if (yielded != null)
					{
						return yielded;
					}
				}
				continue;
			}
			if (!IsCombatState(current.Snapshot.StateType) || IsActionableCombatSnapshot(current.Snapshot))
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.phase2_hits");
				return current.Snapshot;
			}
			if (CombatSimulationRuntime.IsPureCombatSimulator
				&& HasPendingPureCombatContinuation())
			{
				FullRunSimulationStateSnapshot? yielded = await TryAdvancePendingCombatContinuationAsync(
					"settle.wait_combat_followup.phase2_pending_non_actionable",
					previousSignature,
					static (_, observed, signature) => observed.Signature != signature
						&& (!IsCombatState(observed.Snapshot.StateType) || IsActionableCombatSnapshot(observed.Snapshot)));
				if (yielded != null)
				{
					return yielded;
				}
			}
			// State changed but not actionable yet (mid-enemy-turn). Keep waiting.
		}
		FullRunSimulationStateSnapshot fallback = GetState();
		FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.fallback_get_state");
		if (fallback.StateType == "combat_pending")
		{
			FullRunSimulationDiagnostics.Increment("settle.wait_combat_followup.force_map_view");
			_forceMapView = true;
			return GetState();
		}

		return fallback;
	}

	private async Task<FullRunSimulationStateSnapshot> WaitExplicitlyAsync(FullRunSimulationStateSnapshot previousState)
	{
		FullRunSimulationDiagnostics.Increment("settle.wait_explicit.calls");
		string previousSignature = BuildStateChangeSignature(previousState);

		await CombatSimulationRuntime.Clock.YieldAsync();
		if (RunManager.Instance.IsInProgress)
		{
			Task execTask = RunManager.Instance.ActionExecutor.FinishedExecutingActions();
			if (!execTask.IsCompleted && CombatSimulationRuntime.IsPureCombatSimulator)
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase1_blocked_executor");
				ObservedState blocked = ObserveState();
				if (HasExplicitWaitProgress(previousSignature, blocked))
				{
					FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase1_hits");
					return blocked.Snapshot;
				}
				FullRunSimulationStateSnapshot? yielded = await TryAdvanceBlockedExecutorAsync(
					"settle.wait_explicit.phase1",
					previousSignature,
					static (self, observed, signature) => self.HasExplicitWaitProgress(signature, observed));
				if (yielded != null)
				{
					return yielded;
				}
			}
			else
			{
				await execTask;
			}
		}

		ObservedState early = ObserveState();
		if (HasExplicitWaitProgress(previousSignature, early))
		{
			FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase1_hits");
			return early.Snapshot;
		}
		if (CombatSimulationRuntime.IsPureCombatSimulator
			&& IsCombatState(early.Snapshot.StateType)
			&& !IsActionableCombatSnapshot(early.Snapshot)
			&& HasPendingPureCombatContinuation())
		{
			FullRunSimulationStateSnapshot? yielded = await TryAdvancePendingCombatContinuationAsync(
				"settle.wait_explicit.phase1_pending",
				previousSignature,
				static (self, observed, signature) => self.HasExplicitWaitProgress(signature, observed));
			if (yielded != null)
			{
				return yielded;
			}
		}

		ISimulatorClock waitClock = CombatSimulationRuntime.Clock;
		bool canUseFrameDeadline = Engine.GetMainLoop() != null && !CombatSimulationRuntime.IsPureCombatSimulator;
		ulong waitDeadline = canUseFrameDeadline ? Time.GetTicksMsec() + 240UL : 0UL;
		// In pure-sim mode, Clock.YieldAsync() is a no-op, but we can still
		// use Task.Yield() to let pending continuations run. Allow a small
		// number of iterations to avoid requiring multiple Python round-trips.
		int maxWaitIters = canUseFrameDeadline ? 500 : 20;
		if (maxWaitIters == 0)
		{
			FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase2_skipped_no_frame");
		}
		for (int waitIter = 0; waitIter < maxWaitIters; waitIter++)
		{
			FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase2_iterations");
			if (canUseFrameDeadline && Time.GetTicksMsec() >= waitDeadline)
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase2_deadline_breaks");
				break;
			}

			await waitClock.YieldAsync();
			// Always yield to let pending game logic continuations execute.
			// In pure-sim mode, Clock.YieldAsync() is a no-op, so Task.Yield()
			// is essential to advance the game state machine.
			long _yieldT0 = System.Diagnostics.Stopwatch.GetTimestamp();
			await Task.Yield();
			long _yieldT1 = System.Diagnostics.Stopwatch.GetTimestamp();
			double _yieldMs = (_yieldT1 - _yieldT0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
			if (_yieldMs > 10.0)
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase2_slow_yield");
				int _tw, _tio;
				System.Threading.ThreadPool.GetAvailableThreads(out _tw, out _tio);
				System.Console.Error.WriteLine(
					$"[YIELD SLOW] {_yieldMs:F1}ms thread={System.Threading.Thread.CurrentThread.ManagedThreadId} avail_workers={_tw} iter={waitIter}");
			}
			FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase2_yield_total_us", (long)(_yieldMs * 1000));

			if (RunManager.Instance.IsInProgress)
			{
				Task execTask = RunManager.Instance.ActionExecutor.FinishedExecutingActions();
				if (!execTask.IsCompleted && CombatSimulationRuntime.IsPureCombatSimulator)
				{
					FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase2_blocked_executor");
					ObservedState blocked = ObserveState();
					if (HasExplicitWaitProgress(previousSignature, blocked))
					{
						FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase2_hits");
						return blocked.Snapshot;
					}
					FullRunSimulationStateSnapshot? yielded = await TryAdvanceBlockedExecutorAsync(
						"settle.wait_explicit.phase2",
						previousSignature,
						static (self, observed, signature) => self.HasExplicitWaitProgress(signature, observed));
					if (yielded != null)
					{
						return yielded;
					}
					continue;
				}

				await execTask;
			}

			ObservedState current = ObserveState();
			if (HasExplicitWaitProgress(previousSignature, current))
			{
				FullRunSimulationDiagnostics.Increment("settle.wait_explicit.phase2_hits");
				return current.Snapshot;
			}
			if (CombatSimulationRuntime.IsPureCombatSimulator
				&& IsCombatState(current.Snapshot.StateType)
				&& !IsActionableCombatSnapshot(current.Snapshot)
				&& HasPendingPureCombatContinuation())
			{
				FullRunSimulationStateSnapshot? yielded = await TryAdvancePendingCombatContinuationAsync(
					"settle.wait_explicit.phase2_pending",
					previousSignature,
					static (self, observed, signature) => self.HasExplicitWaitProgress(signature, observed));
				if (yielded != null)
				{
					return yielded;
				}
			}
		}

		FullRunSimulationStateSnapshot fallback = GetState();
		FullRunSimulationDiagnostics.Increment("settle.wait_explicit.fallback_get_state");
		if (fallback.StateType == "combat_pending")
		{
			FullRunSimulationDiagnostics.Increment("settle.wait_explicit.force_map_view");
			_forceMapView = true;
			return GetState();
		}

		return fallback;
	}

	// ------------------------------------------------------------------
	// MCTS State Snapshot API
	// ------------------------------------------------------------------

	/// <summary>
	/// Save the current game state as a snapshot for MCTS tree search.
	/// Returns a state_id that can be used with LoadState to restore.
	/// Captures: run progression, player state, RNG counters, map position.
	/// Note: mid-combat creature state (HP, powers) is NOT yet captured —
	/// combat snapshots restore to the beginning of the encounter.
	/// </summary>
	public string SaveState()
	{
		if (!IsActive)
			throw new FullRunSimulationRuntimeException("no_active_episode", "No active episode to save.");

		FullRunSimulationStateSnapshot state = _lastObservedState ?? GetState();
		// NOTE: combat snapshots are allowed for MCTS tree search.
		// Exact signature verification is skipped for combat states
		// since combat save/load is inherently lossy (restores to
		// encounter start, not mid-turn state).

		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		if (runState == null)
		{
			throw new FullRunSimulationRuntimeException("run_state_unavailable", "Could not resolve active run state.");
		}

		AbstractRoom? preFinishedRoom = state.StateType == "map" ? null : runState.CurrentRoom;
		Dictionary<Player, List<Reward>>? originalExtraRewards = null;
		if ((state.StateType == "combat_rewards" || state.StateType == "card_reward") && preFinishedRoom is CombatRoom combatRoom)
		{
			originalExtraRewards = combatRoom.CloneExtraRewardsForSimulationSave();
			combatRoom.ReplaceExtraRewardsForSimulationSave(new Dictionary<Player, List<Reward>>());
		}
		Saves.SerializableRun snapshot;
		try
		{
			snapshot = RunManager.Instance.ToSave(preFinishedRoom: preFinishedRoom);
		}
		finally
		{
			if (originalExtraRewards != null && preFinishedRoom is CombatRoom restoredCombatRoom)
			{
				restoredCombatRoom.ReplaceExtraRewardsForSimulationSave(originalExtraRewards);
			}
		}
		string exactSignature = BuildExactSnapshotSignature(runState, state);

		string stateId = $"s{_nextStateId++}";
		_stateCache[stateId] = new SavedRunSnapshot
		{
			RunSnapshot = snapshot,
			ExactSignature = exactSignature,
			StateType = state.StateType,
			PendingSelection = FullRunSimulationChoiceBridge.Instance.CapturePendingSelection(),
			CombatSnapshot = CaptureCombatSnapshot(runState, state),
			ShopSnapshot = CaptureShopSnapshot(runState, state),
			TreasureSnapshot = CaptureTreasureSnapshot(runState, state)
		};
		return stateId;
	}

	public string ExportStateToFile(string path, string? stateId = null)
	{
		string resolvedStateId = !string.IsNullOrWhiteSpace(stateId) ? stateId : SaveState();
		if (!_stateCache.TryGetValue(resolvedStateId, out SavedRunSnapshot? savedSnapshot))
		{
			throw new FullRunSimulationRuntimeException("state_not_found", $"State '{resolvedStateId}' not found in cache.");
		}

		string fullPath = Path.GetFullPath(path);
		string? dir = Path.GetDirectoryName(fullPath);
		if (!string.IsNullOrWhiteSpace(dir))
		{
			Directory.CreateDirectory(dir);
		}

		var exported = new FullRunExportedRunSnapshot
		{
			ExportedAtUtc = DateTime.UtcNow.ToString("O"),
			SourceStateId = resolvedStateId,
			RunSnapshot = savedSnapshot.RunSnapshot,
			ExactSignature = savedSnapshot.ExactSignature,
			StateType = savedSnapshot.StateType,
			PendingSelection = savedSnapshot.PendingSelection,
			CombatSnapshot = savedSnapshot.CombatSnapshot,
			ShopSnapshot = savedSnapshot.ShopSnapshot,
			TreasureSnapshot = savedSnapshot.TreasureSnapshot
		};
		string json = JsonSerializer.Serialize(exported, JsonSerializationUtility.Options);
		File.WriteAllText(fullPath, json);
		return fullPath;
	}

	/// <summary>
	/// Restore the game to a previously saved state snapshot.
	/// After loading, the game is ready to accept new actions from the restored point.
	/// Uses the game's native save/load path (SetUpSavedSinglePlayer) for correct
	/// map/room/act restoration.
	/// </summary>
	public async Task<FullRunSimulationStateSnapshot> LoadState(string stateId)
	{
		try
		{
		if (!_stateCache.TryGetValue(stateId, out SavedRunSnapshot? savedSnapshot))
			throw new FullRunSimulationRuntimeException("state_not_found", $"State '{stateId}' not found in cache.");

		Saves.SerializableRun snapshot = savedSnapshot.RunSnapshot;

		// Tear down current episode state
		_forceMapView = false;
		_rewardsTriggered = false;
		_suppressTerminalRewardsTransitionOnce = false;
		FullRunSimulationChoiceBridge.Instance.Reset();
		if (RunManager.Instance.IsInProgress)
		{
			RunManager.Instance.CleanUp();
		}
		// Keep runtime scope active (we're still in a simulation session)
		// Don't dispose _runtimeScope or _selectorScope

		// Restore RunState using the game's saved-run path
		RunState restored = RunState.FromSerializable(snapshot);
		RunManager.Instance.SetUpSavedSinglePlayer(restored, snapshot);

		await PreloadManager.LoadRunAssets(restored.Players.Select(static p => p.Character));
		if (restored.CurrentActIndex < restored.Acts.Count)
			await PreloadManager.LoadActAssets(restored.Acts[restored.CurrentActIndex]);

		RunManager.Instance.Launch();
		// Pure simulator: skip NRun.Create (no Godot nodes needed) and EnterAct
		// (which would call SetActInternal → ClearVisitedMapCoordsDebug, destroying
		// the restored visit history, and FadeIn/NRun Godot UI calls).
		// Instead: generate the map from SavedMapsToLoad (set by InitializeSavedRun)
		// then either re-enter the saved room (combat/event) or show the map.
		await RunManager.Instance.GenerateMap();

		CurrentSeed = snapshot.SerializableRng?.Seed;
		CurrentAscensionLevel = snapshot.Ascension;
		CurrentCharacterId = restored.Players
			.FirstOrDefault(static player => player.NetId == NetSingleplayerGameService.defaultNetId)
			?.Character.Id.Entry
			?? restored.Players.FirstOrDefault()?.Character.Id.Entry;

		if (snapshot.PreFinishedRoom != null)
		{
			// Snapshot was taken mid-room (e.g., in combat): restore the full room state.
			// ClearScreens() inside LoadIntoLatestMapCoord is guarded by !TestMode.IsOn.
			// FadeIn() is also guarded. This path is safe in pure-sim mode.
			AbstractRoom preFinishedRoom = AbstractRoom.FromSerializable(snapshot.PreFinishedRoom, restored)!;
			await RunManager.Instance.LoadIntoLatestMapCoord(preFinishedRoom);
			_forceMapView = false;

			switch (savedSnapshot.StateType)
			{
				case "shop":
					RestorePlayerRngState(restored, snapshot);
					if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is MerchantRoom restoredMerchantRoom && savedSnapshot.ShopSnapshot != null)
					{
						ApplyShopSnapshot(restoredMerchantRoom, savedSnapshot.ShopSnapshot);
					}
					break;
				case "treasure":
					RestoreTreasureGeneratorState(restored, snapshot);
					if (savedSnapshot.TreasureSnapshot != null)
					{
						ApplyTreasureSnapshot(savedSnapshot.TreasureSnapshot, restored);
					}
					break;
			}
		}
		else
		{
			// Snapshot was taken at map screen: push a MapRoom so state_type = "map".
			RunManager.Instance.DebugOnlyGetState()?.PushRoom(new MapRoom());
			_forceMapView = true; // After load, show map so player can choose next node
		}

		if (savedSnapshot.PendingSelection != null)
		{
			Player localPlayer = ResolveSignaturePlayer(restored);
			await RestorePendingSelectionFlowAsync(savedSnapshot.PendingSelection, localPlayer, restored.CurrentRoom);
			_forceMapView = false;
		}

		bool hasCombatSnapshot = savedSnapshot.CombatSnapshot != null && RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is CombatRoom;
		if (hasCombatSnapshot)
		{
			await WaitForCombatRoomBootstrapAsync();
			if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is CombatRoom restoredCombatRoom)
			{
				ApplyCombatSnapshot(restoredCombatRoom, savedSnapshot.CombatSnapshot!);
				_forceMapView = false;
			}
		}

		FullRunSimulationStateSnapshot restoredState = hasCombatSnapshot
			? GetState()
			: await WaitForLoadedStateAsync(savedSnapshot.StateType);
		// Skip signature verification for combat states — combat save/load
		// is inherently lossy (restores to encounter start, not exact mid-turn).
		if (SupportsExactSnapshot(savedSnapshot.StateType))
		{
			string restoredSignature = BuildExactSnapshotSignature(restored, restoredState);
			if (!string.Equals(restoredSignature, savedSnapshot.ExactSignature, StringComparison.Ordinal))
			{
				throw new FullRunSimulationRuntimeException(
					"restore_signature_mismatch",
					$"Loaded state '{stateId}' did not match the saved snapshot signature (saved={savedSnapshot.StateType}, restored={restoredState.StateType}).");
			}
		}

		return restoredState;
		}
		catch (FullRunSimulationRuntimeException)
		{
			throw;
		}
		catch (Exception ex)
		{
			throw new FullRunSimulationRuntimeException("load_state_failed", ex.ToString());
		}
	}

	public Task<FullRunSimulationStateSnapshot> LoadStateFromFile(string path)
	{
		string fullPath = Path.GetFullPath(path);
		if (!File.Exists(fullPath))
		{
			throw new FullRunSimulationRuntimeException("snapshot_file_not_found", $"Snapshot file '{fullPath}' not found.");
		}

		string json = File.ReadAllText(fullPath);
		FullRunExportedRunSnapshot? exported = JsonSerializer.Deserialize<FullRunExportedRunSnapshot>(json, JsonSerializationUtility.Options);
		if (exported == null || exported.RunSnapshot == null)
		{
			throw new FullRunSimulationRuntimeException("snapshot_file_invalid", $"Snapshot file '{fullPath}' could not be parsed.");
		}

		string importedStateId = $"s{_nextStateId++}";
		_stateCache[importedStateId] = new SavedRunSnapshot
		{
			RunSnapshot = exported.RunSnapshot,
			ExactSignature = exported.ExactSignature ?? string.Empty,
			StateType = exported.StateType ?? string.Empty,
			PendingSelection = exported.PendingSelection,
			CombatSnapshot = exported.CombatSnapshot,
			ShopSnapshot = exported.ShopSnapshot,
			TreasureSnapshot = exported.TreasureSnapshot
		};
		return LoadState(importedStateId);
	}

	private async Task<FullRunSimulationStateSnapshot> WaitForLoadedStateAsync(string expectedStateType)
	{
		Exception? lastError = null;
		bool canUseFrameYield = !CombatSimulationRuntime.IsPureCombatSimulator && Engine.GetMainLoop() != null;
		for (int attempt = 0; attempt < 40; attempt++)
		{
			await CombatSimulationRuntime.Clock.YieldAsync();
			if (canUseFrameYield)
			{
				await Task.Yield();
			}
			if (RunManager.Instance.IsInProgress)
			{
				await RunManager.Instance.ActionExecutor.FinishedExecutingActions();
			}

			try
			{
				FullRunSimulationStateSnapshot state = GetState();
				if (MatchesLoadedTargetState(state, expectedStateType) || IsReadySnapshot(state) || attempt > 0)
				{
					return state;
				}
			}
			catch (ArgumentOutOfRangeException ex)
			{
				lastError = ex;
			}

			if (canUseFrameYield)
			{
				await Task.Yield();
			}
		}

		if (lastError != null)
		{
			throw new FullRunSimulationRuntimeException(
				"restore_state_unavailable",
				$"Loaded state did not settle cleanly: {lastError.Message}");
		}

		return GetState();
	}

	private async Task RestorePendingSelectionFlowAsync(
		FullRunPendingSelectionRestoreSnapshot snapshot,
		Player localPlayer,
		AbstractRoom? currentRoom)
	{
		if (snapshot.RewardSelection == null && snapshot.CardRewardSelection == null)
		{
			return;
		}

		if (currentRoom == null)
		{
			FullRunSimulationChoiceBridge.Instance.RestorePendingSelection(snapshot, localPlayer);
			return;
		}

		RewardsSet rewardsSet = new RewardsSet(localPlayer).EmptyForRoom(currentRoom);
		if (snapshot.RewardSelection != null)
		{
			rewardsSet.WithCustomRewards(FullRunSimulationChoiceBridge.RestoreRewardsForOffer(snapshot.RewardSelection, localPlayer));
			Task offerTask = rewardsSet.Offer();
			await CombatSimulationRuntime.Clock.YieldAsync();
			if (!offerTask.IsCompleted)
			{
				for (int waitIter = 0; waitIter < 8; waitIter++)
				{
					if (FullRunSimulationChoiceBridge.Instance.IsSelectionActive)
					{
						break;
					}
					await CombatSimulationRuntime.Clock.YieldAsync();
				}
			}
		}
		else if (snapshot.CardRewardSelection != null)
		{
			CardReward reward = FullRunSimulationChoiceBridge.RestoreCardRewardForOffer(snapshot.CardRewardSelection, localPlayer);
			LinkedRewardSet rewardSet = new LinkedRewardSet(new List<Reward> { reward }, localPlayer);
			reward.ParentRewardSet = rewardSet;
			_suppressTerminalRewardsTransitionOnce = true;
			reward.MarkContentAsSeen();
			Task rewardTask = reward.OnSelectWrapper();
			for (int waitIter = 0; waitIter < 8; waitIter++)
			{
				if (FullRunSimulationChoiceBridge.Instance.IsSelectionActive)
				{
					break;
				}
				await CombatSimulationRuntime.Clock.YieldAsync();
			}
			if (!rewardTask.IsCompleted && FullRunSimulationChoiceBridge.Instance.IsSelectionActive)
			{
				return;
			}
		}
	}

	private static async Task WaitForCombatRoomBootstrapAsync()
	{
		bool canUseFrameYield = !CombatSimulationRuntime.IsPureCombatSimulator && Engine.GetMainLoop() != null;
		for (int attempt = 0; attempt < 40; attempt++)
		{
			await CombatSimulationRuntime.Clock.YieldAsync();
			if (canUseFrameYield)
			{
				await Task.Yield();
			}

			if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is not CombatRoom combatRoom)
			{
				continue;
			}

			if (CombatManager.Instance.IsInProgress && !combatRoom.SkipInitialStartTurn)
			{
				return;
			}
		}
	}

	private bool TryTriggerPostCombatRewards(RunState? runState, CombatRoom? pendingCombatRoom)
	{
		if (_rewardsTriggered || runState == null || pendingCombatRoom == null || runState.IsGameOver)
		{
			return false;
		}

		Player? me = LocalContext.GetMe(runState);
		if (me == null)
		{
			return false;
		}

		_rewardsTriggered = true;
		TaskHelper.RunSafely(RewardsCmd.OfferForRoomEnd(me, pendingCombatRoom));
		return true;
	}

	private static bool MatchesLoadedTargetState(FullRunSimulationStateSnapshot state, string expectedStateType)
	{
		if (string.IsNullOrWhiteSpace(expectedStateType))
		{
			return false;
		}

		if (string.Equals(state.StateType, expectedStateType, StringComparison.Ordinal))
		{
			return true;
		}

		return false;
	}

	/// <summary>
	/// Remove a saved state from the cache to free memory.
	/// </summary>
	public bool DeleteState(string stateId)
	{
		return _stateCache.Remove(stateId);
	}

	/// <summary>
	/// Clear all saved state snapshots.
	/// </summary>
	public void ClearStateCache()
	{
		_stateCache.Clear();
		_nextStateId = 0;
	}

	/// <summary>
	/// Get the number of cached state snapshots.
	/// </summary>
	public int StateCacheCount => _stateCache.Count;

	private void CleanUpPreviousEpisode()
	{
		_forceMapView = false;
		_lastObservedState = null;
		_suppressTerminalRewardsTransitionOnce = false;
		FullRunSimulationChoiceBridge.Instance.Reset();
		if (RunManager.Instance.IsInProgress)
		{
			RunManager.Instance.CleanUp();
		}
		_selectorScope?.Dispose();
		_selectorScope = null;
		_runtimeScope?.Dispose();
		_runtimeScope = null;
	}

	private static bool IsCombatState(string stateType)
	{
		return stateType == "monster" || stateType == "elite" || stateType == "boss" || stateType == "hand_select";
	}

	private static bool SupportsExactSnapshot(string stateType)
	{
		return stateType switch
		{
			"map" => true,
			"event" => true,
			"rest_site" => true,
			"shop" => true,
			"treasure" => true,
			// Combat states: skip exact signature verification because
			// NetCombatCardDb rebuild after restore changes card IDs.
			// Combat save/load is inherently lossy (CLAUDE.md §2).
			"monster" => false,
			"elite" => false,
			"boss" => false,
			"hand_select" => true,
			"combat_rewards" => true,
			"card_reward" => true,
			"game_over" => true,
			_ => false
		};
	}

	private static string NormalizeActionType(FullRunSimulationActionRequest action)
	{
		return (action.Action ?? action.Type ?? string.Empty).Trim().ToLowerInvariant();
	}

	private static bool TryResolveSelectionIndex(FullRunSimulationActionRequest action, out int index)
	{
		index = action.Index ?? action.CardIndex ?? action.HandIndex ?? -1;
		return index >= 0;
	}

	private static uint? ResolveCombatTargetId(FullRunSimulationActionRequest action)
	{
		if (action.TargetId.HasValue)
		{
			return action.TargetId.Value;
		}
		string? target = action.Target;
		if (string.IsNullOrWhiteSpace(target))
		{
			return null;
		}
		CombatState? combatState = CombatManager.Instance.DebugOnlyGetState();
		if (combatState == null)
		{
			return null;
		}
		Creature? creature = combatState.Creatures.FirstOrDefault((Creature entry) => string.Equals(entry.Monster?.Id.Entry, target, StringComparison.OrdinalIgnoreCase) || string.Equals(entry.Name, target, StringComparison.OrdinalIgnoreCase));
		return creature?.CombatId;
	}

	private static bool TryResolveMapCoord(FullRunSimulationStateSnapshot state, FullRunSimulationActionRequest action, out MapCoord destination)
	{
		foreach (FullRunSimulationMapOption option in state.MapOptions)
		{
			if (action.Index.HasValue && option.Index == action.Index.Value)
			{
				destination = new MapCoord(option.Col, option.Row);
				return true;
			}
			if (action.Col.HasValue && action.Row.HasValue && option.Col == action.Col.Value && option.Row == action.Row.Value)
			{
				destination = new MapCoord(option.Col, option.Row);
				return true;
			}
		}
		destination = default;
		return false;
	}

	private static async Task<bool> TryChooseEventOptionAsync(int index)
	{
		if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is not EventRoom eventRoom)
		{
			return false;
		}
		EventModel localEvent = eventRoom.LocalMutableEvent;
		if (localEvent.IsFinished || index < 0 || index >= localEvent.CurrentOptions.Count)
		{
			return false;
		}
		EventOption option = localEvent.CurrentOptions[index];
		if (option.IsLocked)
		{
			return false;
		}
		await FullRunUpstreamCompat.ChooseLocalOptionAsync(RunManager.Instance.EventSynchronizer, index);
		return true;
	}

	private bool TryProceedEvent()
	{
		if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is not EventRoom eventRoom)
		{
			return false;
		}
		EventModel localEvent = eventRoom.LocalMutableEvent;
		if (!localEvent.IsFinished && !localEvent.CurrentOptions.Any(static option => option.IsProceed && !option.IsLocked))
		{
			return false;
		}
		_forceMapView = true;
		return true;
	}

	private static bool TryAdvanceEventDialogue()
	{
		return NRun.Instance?.EventRoom?.Layout is NAncientEventLayout layout && layout.TryAdvanceDialogue();
	}

	private static async Task<bool> TryPurchaseShopItemAsync(int index)
	{
		if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is not MerchantRoom merchantRoom)
		{
			return false;
		}
		MerchantInventory? inventory = merchantRoom.Inventory;
		if (inventory == null)
		{
			return false;
		}
		MerchantEntry? entry = GetShopEntryByIndex(inventory, index);
		if (entry == null)
		{
			return false;
		}
		try
		{
			return await entry.OnTryPurchaseWrapper(inventory);
		}
		catch (NullReferenceException)
		{
			// Some shop entries can throw when their backing data is stale
			// (e.g., after a previous purchase modified the inventory).
			return false;
		}
	}

	private static bool TryClaimTreasureRelic(int index)
	{
		IReadOnlyList<RelicModel>? relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		if (relics == null || index < 0 || index >= relics.Count)
		{
			return false;
		}
		RunManager.Instance.TreasureRoomRelicSynchronizer.PickRelicLocally(index);
		return true;
	}

	private static SavedShopSnapshot? CaptureShopSnapshot(RunState runState, FullRunSimulationStateSnapshot state)
	{
		if (state.StateType != "shop" || runState.CurrentRoom is not MerchantRoom merchantRoom || merchantRoom.Inventory == null)
		{
			return null;
		}

		return new SavedShopSnapshot
		{
			CharacterCards = merchantRoom.Inventory.CharacterCardEntries.Select(static entry => new SavedShopCardEntry
			{
				Card = entry.CreationResult?.Card.ToSerializable(),
				Cost = entry.Cost,
				IsOnSale = entry.IsOnSale
			}).ToList(),
			ColorlessCards = merchantRoom.Inventory.ColorlessCardEntries.Select(static entry => new SavedShopCardEntry
			{
				Card = entry.CreationResult?.Card.ToSerializable(),
				Cost = entry.Cost,
				IsOnSale = entry.IsOnSale
			}).ToList(),
			Relics = merchantRoom.Inventory.RelicEntries.Select(static entry => new SavedShopRelicEntry
			{
				Relic = entry.Model?.ToSerializable(),
				Cost = entry.Cost
			}).ToList(),
			Potions = merchantRoom.Inventory.PotionEntries.Select(static entry => new SavedShopPotionEntry
			{
				Potion = entry.Model?.ToSerializable(-1),
				Cost = entry.Cost
			}).ToList(),
			CardRemoval = merchantRoom.Inventory.CardRemovalEntry == null
				? null
				: new SavedShopCardRemovalEntry
				{
					Used = merchantRoom.Inventory.CardRemovalEntry.Used,
					Cost = merchantRoom.Inventory.CardRemovalEntry.Cost
				}
		};
	}

	private static SavedTreasureSnapshot? CaptureTreasureSnapshot(RunState runState, FullRunSimulationStateSnapshot state)
	{
		if (state.StateType != "treasure" || runState.CurrentRoom is not TreasureRoom)
		{
			return null;
		}

		IReadOnlyList<RelicModel>? relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		return new SavedTreasureSnapshot
		{
			Relics = relics?.Select(static relic => relic.Id).ToList()
		};
	}

	private static void RestorePlayerRngState(RunState restored, Saves.SerializableRun snapshot)
	{
		Dictionary<ulong, SerializablePlayer> savedPlayers = snapshot.Players.ToDictionary(static player => player.NetId);
		foreach (Player player in restored.Players)
		{
			if (savedPlayers.TryGetValue(player.NetId, out SerializablePlayer? savedPlayer))
			{
				player.PlayerRng.LoadFromSerializable(savedPlayer.Rng);
			}
		}
	}

	private static void RestoreTreasureGeneratorState(RunState restored, Saves.SerializableRun snapshot)
	{
		restored.Rng.LoadFromSerializable(snapshot.SerializableRng);
		restored.SharedRelicGrabBag.LoadFromSerializable(snapshot.SerializableSharedRelicGrabBag);
	}

	private static void ApplyShopSnapshot(MerchantRoom merchantRoom, SavedShopSnapshot snapshot)
	{
		MerchantInventory? inventory = merchantRoom.Inventory;
		if (inventory == null)
		{
			return;
		}

		ApplyShopCardEntries(inventory.CharacterCardEntries, snapshot.CharacterCards);
		ApplyShopCardEntries(inventory.ColorlessCardEntries, snapshot.ColorlessCards);
		ApplyShopRelicEntries(inventory.RelicEntries, snapshot.Relics);
		ApplyShopPotionEntries(inventory.PotionEntries, snapshot.Potions);

		if (inventory.CardRemovalEntry != null && snapshot.CardRemoval != null)
		{
			MerchantCardRemovalUsedField.SetValue(inventory.CardRemovalEntry, snapshot.CardRemoval.Used);
			MerchantEntryCostField.SetValue(inventory.CardRemovalEntry, snapshot.CardRemoval.Cost);
		}
	}

	private static void ApplyShopCardEntries(IReadOnlyList<MerchantCardEntry> entries, IReadOnlyList<SavedShopCardEntry> savedEntries)
	{
		for (int index = 0; index < Math.Min(entries.Count, savedEntries.Count); index++)
		{
			MerchantCardEntry entry = entries[index];
			SavedShopCardEntry saved = savedEntries[index];
			CardCreationResult? creationResult = null;
			if (saved.Card != null)
			{
				CardModel card = CardModel.FromSerializable(saved.Card);
				Player owner = (Player?)MerchantEntryPlayerField.GetValue(entry)
					?? entry.CreationResult?.Card.Owner
					?? throw new InvalidOperationException("Could not resolve shop card owner during snapshot restore.");
				owner.RunState.AddCard(card, owner);
				creationResult = new CardCreationResult(card);
			}
			MerchantCardEntryCreationResultField.SetValue(entry, creationResult);
			MerchantCardEntryIsOnSaleField.SetValue(entry, saved.IsOnSale);
			MerchantEntryCostField.SetValue(entry, saved.Cost);
		}
	}

	private static void ApplyShopRelicEntries(IReadOnlyList<MerchantRelicEntry> entries, IReadOnlyList<SavedShopRelicEntry> savedEntries)
	{
		for (int index = 0; index < Math.Min(entries.Count, savedEntries.Count); index++)
		{
			MerchantRelicEntry entry = entries[index];
			SavedShopRelicEntry saved = savedEntries[index];
			MerchantRelicEntryModelField.SetValue(entry, saved.Relic == null ? null : RelicModel.FromSerializable(saved.Relic));
			MerchantEntryCostField.SetValue(entry, saved.Cost);
		}
	}

	private static void ApplyShopPotionEntries(IReadOnlyList<MerchantPotionEntry> entries, IReadOnlyList<SavedShopPotionEntry> savedEntries)
	{
		for (int index = 0; index < Math.Min(entries.Count, savedEntries.Count); index++)
		{
			MerchantPotionEntry entry = entries[index];
			SavedShopPotionEntry saved = savedEntries[index];
			MerchantPotionEntryModelField.SetValue(entry, saved.Potion == null ? null : PotionModel.FromSerializable(saved.Potion));
			MerchantEntryCostField.SetValue(entry, saved.Cost);
		}
	}

	private static void ApplyTreasureSnapshot(SavedTreasureSnapshot snapshot, RunState restored)
	{
		TreasureRoomRelicSynchronizer synchronizer = RunManager.Instance.TreasureRoomRelicSynchronizer;
		TreasureRoomPredictedVoteField.SetValue(synchronizer, null);
		if (TreasureRoomVotesField.GetValue(synchronizer) is List<int?> votes)
		{
			votes.Clear();
			foreach (Player _ in restored.Players)
			{
				votes.Add(null);
			}
		}

		List<RelicModel>? relics = snapshot.Relics?
			.Where(static relicId => relicId != null)
			.Select(static relicId => ModelDb.GetById<RelicModel>(relicId!))
			.ToList();
		TreasureRoomCurrentRelicsField.SetValue(synchronizer, relics);
	}

	private SavedCombatSnapshot? CaptureCombatSnapshot(RunState runState, FullRunSimulationStateSnapshot state)
	{
		if (!IsCombatState(state.StateType) || runState.CurrentRoom is not CombatRoom combatRoom)
		{
			return null;
		}

		var players = new Dictionary<ulong, SavedCombatPlayerSnapshot>();
		foreach (Player player in combatRoom.CombatState.Players)
		{
			PlayerCombatState? pcs = player.PlayerCombatState;
			if (pcs == null)
			{
				continue;
			}

			players[player.NetId] = new SavedCombatPlayerSnapshot
			{
				CurrentHp = player.Creature.CurrentHp,
				CombatState = new SerializablePlayerCombatState
				{
					Hand = pcs.Hand.Cards.Select(static card => card.ToSerializable()).ToList(),
					DrawPile = pcs.DrawPile.Cards.Select(static card => card.ToSerializable()).ToList(),
					DiscardPile = pcs.DiscardPile.Cards.Select(static card => card.ToSerializable()).ToList(),
					ExhaustPile = pcs.ExhaustPile.Cards.Select(static card => card.ToSerializable()).ToList(),
					Energy = pcs.Energy,
					Block = player.Creature.Block,
					Powers = player.Creature.Powers.Select(static power => new SerializablePower
					{
						Id = power.Id.Entry,
						Amount = power.Amount
					}).ToList()
				},
				PlayPile = pcs.PlayPile.Cards.Select(static card => card.ToSerializable()).ToList(),
				Stars = pcs.Stars
			};
		}

		List<SerializableCreatureState> creatures = combatRoom.CombatState.Enemies.Select(static creature => new SerializableCreatureState
		{
			Id = creature.ModelId.Entry,
			CombatId = creature.CombatId ?? 0,
			Hp = creature.CurrentHp,
			MaxHp = creature.MaxHp,
			Block = creature.Block,
			Powers = creature.Powers.Select(static power => new SerializablePower
			{
				Id = power.Id.Entry,
				Amount = power.Amount
			}).ToList()
		}).ToList();

		List<SavedCombatMonsterMoveSnapshot> monsterMoves = combatRoom.CombatState.Enemies
			.Where(static creature => creature.Monster?.MoveStateMachine != null)
			.Select(static creature => CaptureMonsterMoveSnapshot(creature))
			.ToList();

		return new SavedCombatSnapshot
		{
			RoundNumber = combatRoom.CombatState.RoundNumber,
			CurrentSide = combatRoom.CombatState.CurrentSide,
			IsPlayPhase = CombatManager.Instance.IsPlayPhase,
			PlayerActionsDisabled = CombatManager.Instance.PlayerActionsDisabled,
			IsEnemyTurnStarted = CombatManager.Instance.IsEnemyTurnStarted,
			Players = players,
			Creatures = creatures,
			MonsterMoves = monsterMoves
		};
	}

	private static SavedCombatMonsterMoveSnapshot CaptureMonsterMoveSnapshot(Creature creature)
	{
		MonsterModel monster = creature.Monster ?? throw new InvalidOperationException("Combat creature has no monster model.");
		MonsterMoveStateMachine machine = monster.MoveStateMachine ?? throw new InvalidOperationException("Monster move state machine is unavailable.");
		return new SavedCombatMonsterMoveSnapshot
		{
			CombatId = creature.CombatId ?? 0,
			CurrentMoveId = monster.NextMove.Id,
			StateLogIds = machine.StateLog.Select(static state => state.Id).ToList(),
			PerformedFirstMove = (bool)MonsterMoveStateMachinePerformedFirstMoveField.GetValue(machine)!,
			CurrentMovePerformedAtLeastOnce = (bool)MoveStatePerformedAtLeastOnceField.GetValue(monster.NextMove)!,
			SpawnedThisTurn = monster.SpawnedThisTurn
		};
	}

	private void ApplyCombatSnapshot(CombatRoom combatRoom, SavedCombatSnapshot snapshot)
	{
		CombatState combatState = combatRoom.CombatState;
		combatState.RoundNumber = snapshot.RoundNumber;
		combatState.CurrentSide = snapshot.CurrentSide;

		foreach (Player player in combatState.Players)
		{
			if (!snapshot.Players.TryGetValue(player.NetId, out SavedCombatPlayerSnapshot? savedPlayer))
			{
				continue;
			}

			PlayerCombatState? pcs = player.PlayerCombatState;
			if (pcs == null)
			{
				continue;
			}

			player.Creature.SetCurrentHpInternal(savedPlayer.CurrentHp);
			RestoreCombatPileState(combatState, player, pcs, savedPlayer.CombatState, savedPlayer.PlayPile);
			pcs.Stars = savedPlayer.Stars;
			RestoreCreatureState(player.Creature, new SerializableCreatureState
			{
				Id = player.Creature.ModelId.Entry,
				CombatId = player.Creature.CombatId ?? 0,
				Hp = savedPlayer.CurrentHp,
				MaxHp = player.Creature.MaxHp,
				Block = savedPlayer.CombatState.Block,
				Powers = savedPlayer.CombatState.Powers.ToList()
			});
		}

		foreach (SerializableCreatureState savedCreature in snapshot.Creatures)
		{
			Creature? creature = combatState.Enemies.FirstOrDefault(enemy => (enemy.CombatId ?? 0) == savedCreature.CombatId || enemy.ModelId.Entry == savedCreature.Id);
			if (creature != null)
			{
				RestoreCreatureState(creature, savedCreature);
			}
		}

		foreach (SavedCombatMonsterMoveSnapshot moveSnapshot in snapshot.MonsterMoves)
		{
			Creature? creature = combatState.Enemies.FirstOrDefault(enemy => (enemy.CombatId ?? 0) == moveSnapshot.CombatId);
			if (creature?.Monster?.MoveStateMachine == null)
			{
				continue;
			}

			MonsterModel monster = creature.Monster;
			MonsterMoveStateMachine machine = monster.MoveStateMachine;
			if (!machine.States.TryGetValue(moveSnapshot.CurrentMoveId, out MonsterState? currentState) || currentState is not MoveState moveState)
			{
				continue;
			}

			monster.SetMoveImmediate(moveState, forceTransition: true);
			MoveStatePerformedAtLeastOnceField.SetValue(moveState, moveSnapshot.CurrentMovePerformedAtLeastOnce);
			machine.StateLog.Clear();
			foreach (string stateId in moveSnapshot.StateLogIds)
			{
				if (machine.States.TryGetValue(stateId, out MonsterState? loggedState))
				{
					machine.StateLog.Add(loggedState);
				}
			}
			MonsterMoveStateMachinePerformedFirstMoveField.SetValue(machine, moveSnapshot.PerformedFirstMove);
			MonsterSpawnedThisTurnField.SetValue(monster, moveSnapshot.SpawnedThisTurn);
		}

		CombatManagerPlayerActionsDisabledField.SetValue(CombatManager.Instance, snapshot.PlayerActionsDisabled);
		CombatManagerIsPlayPhaseField.SetValue(CombatManager.Instance, snapshot.IsPlayPhase);
		CombatManagerIsEnemyTurnStartedField.SetValue(CombatManager.Instance, snapshot.IsEnemyTurnStarted);
		CombatManagerEndingPlayerTurnPhaseOneField.SetValue(CombatManager.Instance, false);
		CombatManagerEndingPlayerTurnPhaseTwoField.SetValue(CombatManager.Instance, false);

		if (snapshot.IsPlayPhase)
		{
			RunManager.Instance.ActionExecutor.Unpause();
			RunManager.Instance.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.PlayPhase);
		}
		else
		{
			RunManager.Instance.ActionExecutor.Pause();
			RunManager.Instance.ActionQueueSynchronizer.SetCombatState(ActionSynchronizerCombatState.NotPlayPhase);
		}

		// Rebuild NetCombatCardDb ID mappings after restoring combat snapshot.
		// Without this, save/load cycles (MCTS probes, lethal probes) leave
		// stale card instances in the singleton DB, causing
		// "Card ... could not be found in combat ID database!" crashes.
		// Use RebuildCardMappings (NOT StartCombat) to avoid re-subscribing
		// to pile events — event handlers are already registered from the
		// original StartCombat call and must not be duplicated.
		try
		{
			var players = combatState.Players;
			if (players != null && players.Count > 0)
			{
				FullRunUpstreamCompat.RebuildCardMappings(GameActions.Multiplayer.NetCombatCardDb.Instance, players);
			}
		}
		catch (Exception)
		{
			// Best-effort — don't let DB rebuild failure block the restore.
		}
	}

	private static void RestoreCombatPileState(CombatState combatState, Player player, PlayerCombatState pcs, SerializablePlayerCombatState saved, IReadOnlyList<SerializableCard> playPile)
	{
		List<CardModel> availableCards = new List<CardModel>(pcs.AllCards);
		pcs.Hand.Clear(silent: true);
		pcs.DrawPile.Clear(silent: true);
		pcs.DiscardPile.Clear(silent: true);
		pcs.ExhaustPile.Clear(silent: true);
		pcs.PlayPile.Clear(silent: true);

		CardModel TakeMatchingCard(SerializableCard serializableCard)
		{
			for (int index = 0; index < availableCards.Count; index++)
			{
				CardModel candidate = availableCards[index];
				if (candidate.Id == serializableCard.Id && candidate.CurrentUpgradeLevel == serializableCard.CurrentUpgradeLevel)
				{
					availableCards.RemoveAt(index);
					return candidate;
				}
			}

			CardModel freshCard = CardModel.FromSerializable(serializableCard);
			combatState.AddCard(freshCard, player);
			return freshCard;
		}

		foreach (SerializableCard serializableCard in saved.Hand)
		{
			pcs.Hand.AddInternal(TakeMatchingCard(serializableCard), silent: true);
		}
		foreach (SerializableCard serializableCard in saved.DrawPile)
		{
			pcs.DrawPile.AddInternal(TakeMatchingCard(serializableCard), silent: true);
		}
		foreach (SerializableCard serializableCard in saved.DiscardPile)
		{
			pcs.DiscardPile.AddInternal(TakeMatchingCard(serializableCard), silent: true);
		}
		foreach (SerializableCard serializableCard in saved.ExhaustPile)
		{
			pcs.ExhaustPile.AddInternal(TakeMatchingCard(serializableCard), silent: true);
		}
		foreach (SerializableCard serializableCard in playPile)
		{
			pcs.PlayPile.AddInternal(TakeMatchingCard(serializableCard), silent: true);
		}

		foreach (CardModel leftover in availableCards)
		{
			combatState.RemoveCard(leftover);
		}

		pcs.Energy = saved.Energy;
	}

	private static void RestoreCreatureState(Creature creature, SerializableCreatureState saved)
	{
		creature.SetMaxHpInternal(saved.MaxHp);
		creature.SetCurrentHpInternal(saved.Hp);
		if (creature.Block > saved.Block)
		{
			creature.LoseBlockInternal(creature.Block - saved.Block);
		}
		else if (creature.Block < saved.Block)
		{
			creature.GainBlockInternal(saved.Block - creature.Block);
		}

		foreach (PowerModel existing in creature.Powers.ToList())
		{
			existing.RemoveInternal();
		}
		foreach (SerializablePower savedPower in saved.Powers)
		{
			try
			{
				PowerModel? powerModel = ModelDb.GetByIdOrNull<PowerModel>(
					new ModelId(ModelDb.GetCategory(typeof(PowerModel)), savedPower.Id));
				if (powerModel != null)
				{
					powerModel.ToMutable().ApplyInternal(creature, savedPower.Amount, silent: true);
				}
				else
				{
					FullRunSimulationTrace.Write(
						$"combat_restore_power.lookup_missing creature={creature.ModelId.Entry} combat_id={creature.CombatId ?? 0} power={savedPower.Id} amount={savedPower.Amount}");
				}
			}
			catch (Exception ex)
			{
				FullRunSimulationTrace.Write(
					$"combat_restore_power.apply_exception creature={creature.ModelId.Entry} combat_id={creature.CombatId ?? 0} power={savedPower.Id} amount={savedPower.Amount} exception={ex}");
			}
		}
	}

	private static MerchantEntry? GetShopEntryByIndex(MerchantInventory inventory, int index)
	{
		if (index < 0)
		{
			return null;
		}
		int currentIndex = 0;
		foreach (MerchantCardEntry entry in inventory.CharacterCardEntries)
		{
			if (currentIndex++ == index)
			{
				return entry;
			}
		}
		foreach (MerchantCardEntry entry2 in inventory.ColorlessCardEntries)
		{
			if (currentIndex++ == index)
			{
				return entry2;
			}
		}
		foreach (MerchantRelicEntry entry3 in inventory.RelicEntries)
		{
			if (currentIndex++ == index)
			{
				return entry3;
			}
		}
		foreach (MerchantPotionEntry entry4 in inventory.PotionEntries)
		{
			if (currentIndex++ == index)
			{
				return entry4;
			}
		}
		if (inventory.CardRemovalEntry != null && currentIndex == index)
		{
			return inventory.CardRemovalEntry;
		}
		return null;
	}

	private string BuildExactSnapshotSignature(RunState runState, FullRunSimulationStateSnapshot snapshot)
	{
		Player localPlayer = ResolveSignaturePlayer(runState);
		FullRunApiState apiState = FullRunApiStateBuilder.Build(runState, snapshot, FullRunSimulationChoiceBridge.Instance, _forceMapView);
		var signature = new System.Text.StringBuilder(256);
		signature.Append("state=").Append(snapshot.StateType);
		signature.Append("|act=").Append(snapshot.CurrentActIndex);
		signature.Append("|floor=").Append(snapshot.TotalFloor);
		signature.Append("|seed=").Append(snapshot.Seed ?? string.Empty);
		signature.Append("|room_type=").Append(snapshot.RoomType ?? string.Empty);
		signature.Append("|room_model=").Append(snapshot.RoomModelId ?? string.Empty);
		signature.Append("|hp=").Append(localPlayer.Creature.CurrentHp);
		signature.Append("|max_hp=").Append(localPlayer.Creature.MaxHp);
		signature.Append("|gold=").Append(localPlayer.Gold);
		signature.Append("|legal=");
		AppendLegalActionSignature(signature, snapshot.LegalActions);
		signature.Append("|payload=");
		AppendStatePayloadSignature(signature, apiState);
		return signature.ToString();
	}

	private static Player ResolveSignaturePlayer(RunState runState)
	{
		return runState.Players.FirstOrDefault(static player => player.NetId == NetSingleplayerGameService.defaultNetId)
			?? runState.Players.First();
	}

	private static void AppendLegalActionSignature(System.Text.StringBuilder signature, IReadOnlyList<FullRunSimulationLegalAction> actions)
	{
		for (int index = 0; index < actions.Count; index++)
		{
			if (index > 0)
			{
				signature.Append(';');
			}
			FullRunSimulationLegalAction action = actions[index];
			signature.Append(action.Action);
			signature.Append(':').Append(action.Index ?? -1);
			signature.Append(':').Append(action.CardIndex ?? -1);
			signature.Append(':').Append(action.Col ?? -1);
			signature.Append(':').Append(action.Row ?? -1);
			signature.Append(':').Append(action.Slot ?? -1);
			signature.Append(':').Append(action.TargetId ?? 0);
			signature.Append(':').Append(action.Target ?? string.Empty);
			signature.Append(':').Append(action.Label ?? string.Empty);
		}
	}

	private static void AppendStatePayloadSignature(System.Text.StringBuilder signature, FullRunApiState apiState)
	{
		switch (apiState.state_type)
		{
			case "map":
				foreach (FullRunApiMapOption option in apiState.map?.next_options ?? Enumerable.Empty<FullRunApiMapOption>())
				{
					signature.Append("map:");
					signature.Append(option.index).Append(',').Append(option.col).Append(',').Append(option.row);
					signature.Append(',').Append(option.point_type ?? string.Empty).Append(';');
				}
				return;
			case "event":
				signature.Append("event:");
				signature.Append(apiState.@event?.in_dialogue == true ? '1' : '0');
				signature.Append(':').Append(apiState.@event?.is_finished == true ? '1' : '0').Append(';');
				foreach (FullRunApiEventOption option in apiState.@event?.options ?? Enumerable.Empty<FullRunApiEventOption>())
				{
					signature.Append(option.index).Append(',').Append(option.text ?? string.Empty);
					signature.Append(',').Append(option.is_locked ? '1' : '0');
					signature.Append(',').Append(option.is_chosen ? '1' : '0');
					signature.Append(',').Append(option.is_proceed ? '1' : '0').Append(';');
				}
				return;
			case "rest_site":
				signature.Append("rest:");
				signature.Append(apiState.rest_site?.can_proceed == true ? '1' : '0').Append(';');
				foreach (FullRunApiRestSiteOption option in apiState.rest_site?.options ?? Enumerable.Empty<FullRunApiRestSiteOption>())
				{
					signature.Append(option.index).Append(',').Append(option.id ?? string.Empty);
					signature.Append(',').Append(option.is_enabled ? '1' : '0').Append(';');
				}
				return;
			case "shop":
				signature.Append("shop:");
				signature.Append(apiState.shop?.can_proceed == true ? '1' : '0').Append(';');
				foreach (FullRunApiShopItem item in apiState.shop?.items ?? Enumerable.Empty<FullRunApiShopItem>())
				{
					signature.Append(item.index).Append(',').Append(item.category ?? string.Empty);
					signature.Append(',').Append(item.cost);
					signature.Append(',').Append(item.can_afford ? '1' : '0');
					signature.Append(',').Append(item.is_stocked ? '1' : '0');
					signature.Append(',').Append(item.on_sale ? '1' : '0');
					signature.Append(',').Append(item.card_id ?? item.relic_id ?? item.potion_id ?? item.name ?? string.Empty).Append(';');
				}
				return;
			case "combat_rewards":
				signature.Append("rewards:");
				signature.Append(apiState.rewards?.can_proceed == true ? '1' : '0').Append(';');
				foreach (FullRunApiRewardItem item in apiState.rewards?.items ?? Enumerable.Empty<FullRunApiRewardItem>())
				{
					signature.Append(item.index).Append(',').Append(item.type ?? string.Empty);
					signature.Append(',').Append(item.label ?? string.Empty).Append(';');
				}
				return;
			case "card_reward":
				signature.Append("card_reward:");
				signature.Append(apiState.card_reward?.can_skip == true ? '1' : '0').Append(';');
				foreach (FullRunApiCardOption card in apiState.card_reward?.cards ?? Enumerable.Empty<FullRunApiCardOption>())
				{
					signature.Append(card.index).Append(',').Append(card.id ?? string.Empty);
					signature.Append(',').Append(card.name ?? string.Empty);
					signature.Append(',').Append(card.type ?? string.Empty);
					signature.Append(',').Append(card.rarity ?? string.Empty);
					signature.Append(',').Append(card.cost ?? -1);
					signature.Append(',').Append(card.is_upgraded ? '1' : '0').Append(';');
				}
				return;
			case "treasure":
				signature.Append("treasure:");
				signature.Append(apiState.treasure?.can_proceed == true ? '1' : '0').Append(';');
				foreach (FullRunApiRelicOption relic in apiState.treasure?.relics ?? Enumerable.Empty<FullRunApiRelicOption>())
				{
					signature.Append(relic.index).Append(',').Append(relic.id ?? string.Empty);
					signature.Append(',').Append(relic.name ?? string.Empty).Append(';');
				}
				return;
			case "relic_select":
				signature.Append("relic_select:");
				signature.Append(apiState.relic_select?.can_skip == true ? '1' : '0').Append(';');
				foreach (FullRunApiRelicOption relic in apiState.relic_select?.relics ?? Enumerable.Empty<FullRunApiRelicOption>())
				{
					signature.Append(relic.index).Append(',').Append(relic.id ?? string.Empty);
					signature.Append(',').Append(relic.name ?? string.Empty).Append(';');
				}
				return;
			case "card_select":
				signature.Append("card_select:");
				signature.Append(apiState.card_select?.screen_type ?? string.Empty);
				signature.Append(':').Append(apiState.card_select?.selected_count ?? 0);
				signature.Append(':').Append(apiState.card_select?.can_confirm == true ? '1' : '0');
				signature.Append(':').Append(apiState.card_select?.can_cancel == true ? '1' : '0').Append(';');
				foreach (FullRunApiCardOption card in apiState.card_select?.cards ?? Enumerable.Empty<FullRunApiCardOption>())
				{
					signature.Append(card.index).Append(',').Append(card.id ?? string.Empty);
					signature.Append(',').Append(card.name ?? string.Empty).Append(';');
				}
				return;
			default:
				signature.Append(apiState.state_type);
				return;
		}
	}

	private static bool IsReadySnapshot(FullRunSimulationStateSnapshot snapshot)
	{
		if (!snapshot.IsRunActive || snapshot.StateType == "menu" || snapshot.StateType == "run_bootstrap")
		{
			return false;
		}
		if (snapshot.StateType == "map")
		{
			return snapshot.MapOptions.Count > 0;
		}
		if (snapshot.StateType == "event")
		{
			return snapshot.LegalActions.Count > 0
				|| RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is EventRoom { LocalMutableEvent.IsFinished: true };
		}
		return true;
	}

	private static bool IsActionableCombatSnapshot(FullRunSimulationStateSnapshot snapshot)
	{
		if (!IsCombatState(snapshot.StateType))
		{
			return true;
		}

		if (snapshot.LegalActions.Count > 0)
		{
			return true;
		}

		CombatTrainingStateSnapshot? combatState = snapshot.CachedCombatState;
		if (combatState == null)
		{
			return false;
		}

		try
		{
			if (!combatState.IsCombatActive)
			{
				return false;
			}
			if (combatState.IsHandSelectionActive || combatState.IsCardSelectionActive)
			{
				return true;
			}
			return combatState.IsPlayPhase && !combatState.PlayerActionsDisabled && !combatState.IsActionQueueRunning;
		}
		catch (Exception ex)
		{
			FullRunSimulationTrace.Write($"headless_combat_actionable_check.exception exception={ex}");
			return false;
		}
	}

	private static bool IsActionableCombatState(CombatTrainingStateSnapshot state)
	{
		if (!state.IsCombatActive)
		{
			return false;
		}

		if (state.IsHandSelectionActive || state.IsCardSelectionActive)
		{
			return true;
		}

		return state.IsPlayPhase && !state.PlayerActionsDisabled && !state.IsActionQueueRunning;
	}

	private readonly record struct ObservedState(FullRunSimulationStateSnapshot Snapshot, string Signature);

	private string BuildStateChangeSignature(FullRunSimulationStateSnapshot snapshot)
	{
		return FullRunApiStateBuilder.Signature(runState: null, snapshot, FullRunSimulationChoiceBridge.Instance, _forceMapView);
	}

	private ObservedState ObserveState()
	{
		FullRunSimulationStateSnapshot snapshot = GetState();
		return new ObservedState(snapshot, BuildStateChangeSignature(snapshot));
	}

	private bool HasExplicitWaitProgress(string previousSignature, ObservedState observedState)
	{
		FullRunSimulationStateSnapshot snapshot = observedState.Snapshot;
		if (snapshot.IsTerminal)
		{
			return true;
		}

		if (!string.Equals(observedState.Signature, previousSignature, StringComparison.Ordinal))
		{
			return snapshot.StateType == "map"
				|| snapshot.StateType == "combat_pending"
				|| snapshot.LegalActions.Count > 0;
		}

		return snapshot.LegalActions.Count > 0 && snapshot.StateType != "combat_pending";
	}

	private bool HasPendingPureCombatContinuation()
	{
		if (!CombatSimulationRuntime.IsPureCombatSimulator || !CombatManager.Instance.IsInProgress)
		{
			return false;
		}

		if (RunManager.Instance.ActionExecutor.IsRunning)
		{
			return true;
		}

		Task? pendingTurnTransition = CombatManager.Instance.DebugOnlyGetPendingTurnTransitionTask();
		if (pendingTurnTransition != null && !pendingTurnTransition.IsCompleted)
		{
			return true;
		}

		CombatState? combatState = CombatManager.Instance.DebugOnlyGetState();
		if (combatState == null)
		{
			return false;
		}

		return CombatManager.Instance.EndingPlayerTurnPhaseOne
			|| CombatManager.Instance.EndingPlayerTurnPhaseTwo
			|| (!CombatManager.Instance.IsPlayPhase
				&& (CombatManager.Instance.IsEnemyTurnStarted || combatState.CurrentSide == CombatSide.Enemy));
	}

	private async Task<FullRunSimulationStateSnapshot?> TryAdvancePendingCombatContinuationAsync(
		string diagnosticsPrefix,
		string previousSignature,
		Func<FullRunSimulatorRuntimeFacade, ObservedState, string, bool> hasProgress)
	{
		// In pure-sim, Clock.YieldAsync() is a no-op — iterating more than once
		// cannot advance state. Keep 1 iteration as a safety check.
		int maxAttempts = CombatSimulationRuntime.IsPureCombatSimulator ? 1 : 64;
		for (int attempt = 0; attempt < maxAttempts; attempt++)
		{
			if (!HasPendingPureCombatContinuation())
			{
				break;
			}

			FullRunSimulationDiagnostics.Increment($"{diagnosticsPrefix}.scheduler_yield_iterations");
			if (CombatSimulationRuntime.IsPureCombatSimulator)
			{
				await CombatSimulationRuntime.Clock.YieldAsync();
			}
			else
			{
				await Task.Yield();
			}

			ObservedState observed = ObserveState();
			if (hasProgress(this, observed, previousSignature))
			{
				FullRunSimulationDiagnostics.Increment($"{diagnosticsPrefix}.scheduler_yield_hits");
				return observed.Snapshot;
			}
		}

		FullRunSimulationDiagnostics.Increment($"{diagnosticsPrefix}.scheduler_yield_exhausted");
		return null;
	}

	private async Task<FullRunSimulationStateSnapshot?> TryAdvanceBlockedExecutorAsync(
		string diagnosticsPrefix,
		string previousSignature,
		Func<FullRunSimulatorRuntimeFacade, ObservedState, string, bool> hasProgress)
	{
		// In pure-sim, Clock.YieldAsync() is a no-op — iterating more than once
		// cannot advance state. Keep 1 iteration as a safety check.
		int maxAttempts = CombatSimulationRuntime.IsPureCombatSimulator ? 1 : 32;
		for (int attempt = 0; attempt < maxAttempts; attempt++)
		{
			FullRunSimulationDiagnostics.Increment($"{diagnosticsPrefix}.yield_iterations");
			if (CombatSimulationRuntime.IsPureCombatSimulator)
			{
				await CombatSimulationRuntime.Clock.YieldAsync();
			}
			else
			{
				await Task.Yield();
			}
			ObservedState observed = ObserveState();
			if (hasProgress(this, observed, previousSignature))
			{
				FullRunSimulationDiagnostics.Increment($"{diagnosticsPrefix}.yield_hits");
				return observed.Snapshot;
			}
			if (!RunManager.Instance.IsInProgress)
			{
				break;
			}
			Task execTask = RunManager.Instance.ActionExecutor.FinishedExecutingActions();
			if (execTask.IsCompleted)
			{
				await execTask;
				break;
			}
		}

		FullRunSimulationDiagnostics.Increment($"{diagnosticsPrefix}.yield_exhausted");
		return null;
	}

	private FullRunSimulationStepResult BuildRejected(string error, string failureCode)
	{
		return new FullRunSimulationStepResult
		{
			Accepted = false,
			Error = error,
			FailureCode = failureCode,
			State = _lastObservedState ?? GetState()
		};
	}
}
