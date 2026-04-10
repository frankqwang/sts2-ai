using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Simulation;

internal static class FullRunSimulationStateBuilder
{
	public static FullRunSimulationStateSnapshot Build(
		RunState? runState,
		FullRunSimulationChoiceBridge bridge,
		bool isPureSimulator,
		string backendKind,
		string coverageTier,
		bool forceMapView,
		string? overrideCharacterId = null,
		string? overrideSeed = null,
		int? overrideAscensionLevel = null,
		CombatTrainingStateSnapshot? cachedCombatState = null)
	{
		FullRunSimulationStateSnapshot snapshot = new FullRunSimulationStateSnapshot
		{
			IsPureSimulator = isPureSimulator,
			BackendKind = backendKind,
			CoverageTier = coverageTier,
			StateType = "menu",
			IsActionable = true
		};
		if (runState == null)
		{
			snapshot.LegalActions.Add(new FullRunSimulationLegalAction
			{
				Action = "start_run",
				IsSupported = true
			});
			return snapshot;
		}

		AbstractRoom? currentRoom = runState.CurrentRoom;
		Player localPlayer = ResolveLocalPlayer(runState);
		snapshot.IsRunActive = true;
		snapshot.CharacterId = overrideCharacterId ?? localPlayer.Character.Id.Entry;
		snapshot.Seed = overrideSeed ?? runState.Rng.StringSeed;
		snapshot.AscensionLevel = overrideAscensionLevel ?? runState.AscensionLevel;
		snapshot.CurrentActIndex = runState.CurrentActIndex;
		snapshot.ActFloor = runState.ActFloor;
		snapshot.TotalFloor = runState.TotalFloor;
		snapshot.RoomType = currentRoom?.RoomType.ToString().ToLowerInvariant();
		snapshot.RoomModelId = currentRoom?.ModelId?.Entry;
		snapshot.StateType = ResolveStateType(runState, currentRoom, bridge, forceMapView);
		if (snapshot.StateType == "map")
		{
			// When we force the map view after a room clear, currentRoom may still
			// refer to the just-finished room (for example a monster encounter).
			// Persisting that stale room metadata makes save/load signatures for
			// map snapshots unstable even when the restored map choices are correct.
			snapshot.RoomType = "map";
			snapshot.RoomModelId = null;
		}
		snapshot.IsTerminal = snapshot.StateType == "game_over";
		snapshot.IsActionable = !snapshot.IsTerminal;
		snapshot.RunOutcome = snapshot.IsTerminal
			? (RunManager.Instance.WinTime > 0 ? "victory" : "defeat")
			: null;
		FullRunChoiceBridgeSnapshotCache bridgeSnapshots = BuildRelevantBridgeSnapshots(bridge, snapshot.StateType);
		snapshot.CachedBridgeSnapshots = bridgeSnapshots;
		snapshot.CachedCombatState = cachedCombatState;
		if (snapshot.StateType == "map")
		{
			snapshot.MapOptions = BuildMapOptions(runState);
			snapshot.MapNodes = BuildFullMapNodes(runState);
			if (runState.Map?.BossMapPoint != null)
			{
				snapshot.BossCol = runState.Map.BossMapPoint.coord.col;
				snapshot.BossRow = runState.Map.BossMapPoint.coord.row;
			}
		}
		snapshot.LegalActions = BuildLegalActions(runState, currentRoom, snapshot, bridgeSnapshots, localPlayer);
		// Filter out unsupported actions (e.g., shop items player can't afford).
		// Only truly executable actions should reach the AI.
		snapshot.LegalActions.RemoveAll(a => !a.IsSupported);
		return snapshot;
	}

	public static string Signature(FullRunSimulationStateSnapshot snapshot)
	{
		string actions = string.Join(";", snapshot.LegalActions.Select(static action => $"{action.Action}:{action.Index}:{action.CardIndex}:{action.Slot}:{action.TargetId}:{action.Target}"));
		string map = string.Join(";", snapshot.MapOptions.Select(static option => $"{option.Index}:{option.Col},{option.Row}:{option.PointType}"));
		return string.Join("|",
			snapshot.StateType,
			snapshot.RoomType ?? string.Empty,
			snapshot.TotalFloor,
			snapshot.RunOutcome ?? string.Empty,
			map,
			actions);
	}

	private static string ResolveStateType(RunState runState, AbstractRoom? currentRoom, FullRunSimulationChoiceBridge bridge, bool forceMapView)
	{
		if (runState.IsGameOver)
		{
			return "game_over";
		}
		if (bridge.IsCardRewardSelectionActive)
		{
			return "card_reward";
		}
		if (bridge.IsRelicSelectionActive)
		{
			return "relic_select";
		}
		if (bridge.IsHandSelectionActive)
		{
			return "hand_select";
		}
		if (bridge.IsRewardSelectionActive)
		{
			return "combat_rewards";
		}
		if (forceMapView)
		{
			return "map";
		}
		if (currentRoom is CombatRoom combatRoom && CombatManager.Instance.IsInProgress)
		{
			return combatRoom.RoomType switch
			{
				RoomType.Monster => "monster",
				RoomType.Elite => "elite",
				RoomType.Boss => "boss",
				_ => "monster"
			};
		}
		if (bridge.IsCardSelectionActive)
		{
			return "card_select";
		}
		if (currentRoom == null)
		{
			return "run_bootstrap";
		}
		// If we're in a combat room but combat is no longer in progress,
		// the fight has ended. Check if the player died (game_over) or
		// if we're in a brief transition before rewards appear.
		if (currentRoom is CombatRoom && !CombatManager.Instance.IsInProgress)
		{
			if (runState.IsGameOver)
				return "game_over";
			// Combat ended but rewards haven't appeared yet — report as
			// "combat_pending" so Python can send a "wait" action to let
			// the game transition to rewards/map/game_over naturally.
			return "combat_pending";
		}
		return currentRoom.RoomType switch
		{
			RoomType.Map => "map",
			RoomType.Shop => "shop",
			RoomType.RestSite => "rest_site",
			RoomType.Event => "event",
			RoomType.Treasure => "treasure",
			RoomType.Monster => "monster",
			RoomType.Elite => "elite",
			RoomType.Boss => "boss",
			_ => "unknown_room"
		};
	}

	private static List<FullRunSimulationLegalAction> BuildLegalActions(
		RunState runState,
		AbstractRoom? currentRoom,
		FullRunSimulationStateSnapshot snapshot,
		FullRunChoiceBridgeSnapshotCache bridgeSnapshots,
		Player localPlayer)
	{
		switch (snapshot.StateType)
		{
			case "menu":
				return new List<FullRunSimulationLegalAction>
				{
					new FullRunSimulationLegalAction
					{
						Action = "start_run",
						IsSupported = true
					}
				};
			case "map":
				return BuildMapLegalActions(snapshot.MapOptions);
			case "event":
				return BuildEventLegalActions(currentRoom as EventRoom);
			case "combat_rewards":
				return BuildRewardsLegalActions(bridgeSnapshots.RewardSelection);
			case "card_reward":
				return BuildCardRewardLegalActions(bridgeSnapshots.CardRewardSelection);
			case "card_select":
				return BuildCardSelectLegalActions(bridgeSnapshots.CardSelection);
			case "relic_select":
				return BuildRelicSelectLegalActions(bridgeSnapshots.RelicSelection);
			case "rest_site":
				return BuildRestSiteLegalActions();
			case "shop":
				return BuildShopLegalActions(currentRoom as MerchantRoom);
			case "treasure":
				return BuildTreasureLegalActions();
			case "hand_select":
			case "monster":
			case "elite":
			case "boss":
				return BuildCombatLegalActions(localPlayer, snapshot);
			default:
				return new List<FullRunSimulationLegalAction>();
		}
	}

	private static FullRunChoiceBridgeSnapshotCache BuildRelevantBridgeSnapshots(FullRunSimulationChoiceBridge bridge, string stateType)
	{
		return stateType switch
		{
			"combat_rewards" => new FullRunChoiceBridgeSnapshotCache
			{
				RewardSelection = bridge.BuildRewardSelectionSnapshot()
			},
			"card_reward" => new FullRunChoiceBridgeSnapshotCache
			{
				CardRewardSelection = bridge.BuildCardRewardSelectionSnapshot()
			},
			"card_select" => new FullRunChoiceBridgeSnapshotCache
			{
				CardSelection = bridge.BuildCardSelectionSnapshot(null)
			},
			"relic_select" => new FullRunChoiceBridgeSnapshotCache
			{
				RelicSelection = bridge.BuildRelicSelectionSnapshot()
			},
			"hand_select" => new FullRunChoiceBridgeSnapshotCache
			{
				HandSelection = bridge.BuildHandSelectionSnapshot(null)
			},
			_ => new FullRunChoiceBridgeSnapshotCache()
		};
	}

	private static List<FullRunSimulationLegalAction> BuildEventLegalActions(EventRoom? eventRoom)
	{
		List<FullRunSimulationLegalAction> actions = new List<FullRunSimulationLegalAction>();
		if (eventRoom == null)
		{
			return actions;
		}
		EventModel localEvent = eventRoom.LocalMutableEvent;
		if (!localEvent.IsFinished && localEvent.CurrentOptions.Count == 0)
		{
			FullRunSimulationTrace.Write($"headless_event_legal_actions.empty id={localEvent.Id.Entry} finished={localEvent.IsFinished}");
		}
		if (localEvent.IsFinished)
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "proceed",
				Label = "Proceed",
				IsSupported = true
			});
			return actions;
		}
		for (int index = 0; index < localEvent.CurrentOptions.Count; index++)
		{
			EventOption option = localEvent.CurrentOptions[index];
			if (option.IsLocked || (option.WasChosen && FullRunUpstreamCompat.IsDisableOnChosen(option)))
			{
				continue;
			}
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = option.IsProceed ? "proceed" : "choose_event_option",
				Index = option.IsProceed ? null : index,
				Label = option.Title.GetRawText(),
				IsSupported = true
			});
		}
		return actions;
	}

	private static List<FullRunSimulationLegalAction> BuildRewardsLegalActions(FullRunPendingRewardSelectionSnapshot? rewards)
	{
		List<FullRunSimulationLegalAction> actions = new List<FullRunSimulationLegalAction>();
		if (rewards == null)
		{
			return actions;
		}
		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		Player? player = LocalContext.GetMe(runState);
		string rewardSource = GetRewardSourceForAction(runState);
		for (int index = 0; index < rewards.Rewards.Count; index++)
		{
			Reward reward = rewards.Rewards[index];
			bool claimable = IsRewardClaimableForAction(reward, player, out string? blockReason);
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "claim_reward",
				Index = index,
				Label = SafeGetText(() => reward.Description),
				RewardType = RewardTypeForAction(reward),
				RewardKey = BuildRewardKeyForAction(reward),
				RewardSource = rewardSource,
				Claimable = claimable,
				ClaimBlockReason = blockReason,
				IsSupported = true
			});
		}
		if (rewards.CanProceed)
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "proceed",
				Label = "Proceed",
				IsSupported = true
			});
		}
		return actions;
	}

	private static string RewardTypeForAction(Reward reward)
	{
		return reward switch
		{
			GoldReward => "gold",
			PotionReward => "potion",
			RelicReward => "relic",
			CardReward => "card",
			CardRemovalReward => "card_remove",
			SpecialCardReward => "special_card",
			_ => reward.GetType().Name.Replace("Reward", string.Empty).ToLowerInvariant()
		};
	}

	private static string BuildRewardKeyForAction(Reward reward)
	{
		string rewardType = RewardTypeForAction(reward);
		string label = SafeGetText(() => reward.Description);
		string specificId = reward switch
		{
			PotionReward potionReward => potionReward.Potion?.Id.Entry ?? string.Empty,
			CardReward => "card_reward",
			CardRemovalReward => "card_remove",
			GoldReward goldReward => goldReward.Amount.ToString(),
			RelicReward => "relic_reward",
			SpecialCardReward => "special_card_reward",
			_ => string.Empty
		};
		return $"{rewardType}|{specificId}|{label}".ToLowerInvariant();
	}

	private static bool IsRewardClaimableForAction(Reward reward, Player? player, out string? claimBlockReason)
	{
		claimBlockReason = null;
		if (reward is PotionReward && player != null && !player.HasOpenPotionSlots)
		{
			claimBlockReason = "potion_slots_full";
			return false;
		}
		return true;
	}

	private static string GetRewardSourceForAction(RunState? runState)
	{
		AbstractRoom? room = runState?.CurrentRoom;
		return room switch
		{
			CombatRoom combatRoom when combatRoom.ParentEventId != null => "event_combat_end",
			CombatRoom => "combat_end",
			EventRoom => "event",
			TreasureRoom => "treasure",
			MerchantRoom => "shop",
			RestSiteRoom => "rest_site",
			_ => "unknown"
		};
	}

	private static List<FullRunSimulationLegalAction> BuildCardRewardLegalActions(FullRunPendingCardRewardSnapshot? reward)
	{
		List<FullRunSimulationLegalAction> actions = new List<FullRunSimulationLegalAction>();
		if (reward == null)
		{
			return actions;
		}
		for (int index = 0; index < reward.Options.Count; index++)
		{
			CardModel card = reward.Options[index].Card;
			FullRunSimulationLegalAction action = new FullRunSimulationLegalAction
			{
				Action = "select_card_reward",
				Index = index,
				CardIndex = index,
				Label = SafeGetText(() => card.Title, card.Id.Entry),
				CardId = card.Id.Entry,
				CardType = card.Type.ToString(),
				CardRarity = card.Rarity.ToString(),
				Cost = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
				IsUpgraded = card.IsUpgraded,
				IsSupported = true
			};
			if (string.IsNullOrEmpty(action.CardId) || string.IsNullOrEmpty(action.CardType) || string.IsNullOrEmpty(action.CardRarity))
			{
				FullRunSimulationTrace.Write(
					$"card_reward_legal_action.source_missing_metadata index={index} " +
					$"card_id={action.CardId ?? "null"} card_type={action.CardType ?? "null"} " +
					$"card_rarity={action.CardRarity ?? "null"} cost={action.Cost ?? "null"} label={action.Label ?? "null"}");
			}
			actions.Add(action);
		}
		if (reward.CanSkip)
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "skip_card_reward",
				Label = "Skip",
				IsSupported = true
			});
		}
		return actions;
	}

	private static List<FullRunSimulationLegalAction> BuildCardSelectLegalActions(CombatTrainingCardSelectionSnapshot? selection)
	{
		List<FullRunSimulationLegalAction> actions = new List<FullRunSimulationLegalAction>();
		if (selection == null)
		{
			return actions;
		}
		foreach (CombatTrainingSelectableCardSnapshot card in selection.SelectableCards)
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "select_card",
				Index = card.ChoiceIndex,
				CardIndex = card.ChoiceIndex,
				Label = card.Title,
				IsSupported = true
			});
		}
		if (selection.CanConfirm)
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "confirm_selection",
				Label = "Confirm",
				IsSupported = true
			});
		}
		if (selection.Cancelable)
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "cancel_selection",
				Label = "Cancel",
				IsSupported = true
			});
		}
		return actions;
	}

	private static List<FullRunSimulationLegalAction> BuildRelicSelectLegalActions(FullRunPendingRelicSelectionSnapshot? relicSelection)
	{
		List<FullRunSimulationLegalAction> actions = new List<FullRunSimulationLegalAction>();
		if (relicSelection == null)
		{
			return actions;
		}
		for (int index = 0; index < relicSelection.Relics.Count; index++)
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "select_relic",
				Index = index,
				Label = SafeGetText(() => relicSelection.Relics[index].Title, relicSelection.Relics[index].Id.Entry),
				IsSupported = true
			});
		}
		if (relicSelection.CanSkip)
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "skip_relic_selection",
				Label = "Skip",
				IsSupported = true
			});
		}
		return actions;
	}

	private static List<FullRunSimulationLegalAction> BuildRestSiteLegalActions()
	{
		List<FullRunSimulationLegalAction> actions = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions().Select((RestSiteOption option, int index) => new FullRunSimulationLegalAction
		{
			Action = "choose_rest_option",
			Index = index,
			Label = SafeGetText(() => option.Title),
			IsSupported = option.IsEnabled
		}).ToList();
		if (actions.Count == 0)
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "proceed",
				Label = "Proceed",
				IsSupported = true
			});
		}
		return actions;
	}

	private static List<FullRunSimulationLegalAction> BuildShopLegalActions(MerchantRoom? merchantRoom)
	{
		List<FullRunSimulationLegalAction> actions = new List<FullRunSimulationLegalAction>();
		if (merchantRoom == null)
		{
			return actions;
		}
		int index = 0;
		foreach (MerchantEntry entry in EnumerateShopEntries(merchantRoom.Inventory))
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "shop_purchase",
				Index = index++,
				Label = GetMerchantEntryLabel(entry),
				IsSupported = entry.IsStocked && entry.EnoughGold
			});
		}
		actions.Add(new FullRunSimulationLegalAction
		{
			Action = "proceed",
			Label = "Proceed",
			IsSupported = true
		});
		return actions;
	}

	private static IEnumerable<MerchantEntry> EnumerateShopEntries(MerchantInventory inventory)
	{
		foreach (MerchantCardEntry entry in inventory.CharacterCardEntries)
		{
			yield return entry;
		}
		foreach (MerchantCardEntry entry2 in inventory.ColorlessCardEntries)
		{
			yield return entry2;
		}
		foreach (MerchantRelicEntry entry3 in inventory.RelicEntries)
		{
			yield return entry3;
		}
		foreach (MerchantPotionEntry entry4 in inventory.PotionEntries)
		{
			yield return entry4;
		}
		if (inventory.CardRemovalEntry != null)
		{
			yield return inventory.CardRemovalEntry;
		}
	}

	private static List<FullRunSimulationLegalAction> BuildTreasureLegalActions()
	{
		List<FullRunSimulationLegalAction> actions = new List<FullRunSimulationLegalAction>();
		IReadOnlyList<RelicModel>? relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		if (relics != null)
		{
			for (int index = 0; index < relics.Count; index++)
			{
				actions.Add(new FullRunSimulationLegalAction
				{
					Action = "claim_treasure_relic",
					Index = index,
					Label = SafeGetText(() => relics[index].Title, relics[index].Id.Entry),
					IsSupported = true
				});
			}
		}
		else
		{
			actions.Add(new FullRunSimulationLegalAction
			{
				Action = "proceed",
				Label = "Proceed",
				IsSupported = true
			});
		}
		return actions;
	}

	private static List<FullRunSimulationLegalAction> BuildCombatLegalActions(Player localPlayer, FullRunSimulationStateSnapshot snapshot)
	{
		List<FullRunSimulationLegalAction> actions = new List<FullRunSimulationLegalAction>();
		using IDisposable _ = FullRunSimulationDiagnostics.Measure("state.build_combat_legal_actions_ms");
		try
		{
			CombatTrainingStateSnapshot? combatState = snapshot.CachedCombatState;
			if (combatState == null)
			{
				ICombatChoiceAdapter choiceAdapter = CombatTrainingEnvService.GetChoiceAdapter();
				bool isPlayPhase = CombatManager.Instance.IsPlayPhase;
				bool playerActionsDisabled = CombatManager.Instance.PlayerActionsDisabled;
				bool isActionQueueRunning = RunManager.Instance.IsInProgress && RunManager.Instance.ActionExecutor.IsRunning;
				if (!CombatManager.Instance.IsInProgress)
				{
					return actions;
				}
				// During enemy turns and queued action execution, many state polls only need to
				// know there are no combat actions available. Avoid building a full combat snapshot
				// unless a hand/card selector is active or the player can actually act.
				if (!choiceAdapter.IsSelectionActive && (!isPlayPhase || playerActionsDisabled || isActionQueueRunning))
				{
					FullRunSimulationDiagnostics.Increment("combat_legal_actions.fast_empty_return");
					return actions;
				}

				combatState = CombatTrainingEnvService.BuildStateSnapshot();
				snapshot.CachedCombatState = combatState;
			}
			if (combatState == null)
			{
				FullRunSimulationDiagnostics.Increment("combat_legal_actions.snapshot_null");
				FullRunSimulationTrace.Write("headless_combat_legal_actions.snapshot_null");
				return actions;
			}
			if (!combatState.IsCombatActive)
			{
				Godot.GD.PrintErr($"[BuildCombatLegalActions] IsCombatActive=false, returning empty. IsPlayPhase={combatState.IsPlayPhase} IsActionQueueRunning={combatState.IsActionQueueRunning} PlayerHP={combatState.Player?.CurrentHp}");
				return actions;
			}
			if (combatState.IsHandSelectionActive && combatState.HandSelection != null)
			{
				foreach (CombatTrainingHandCardSnapshot card in combatState.HandSelection.SelectableCards)
				{
					actions.Add(new FullRunSimulationLegalAction
					{
						Action = "combat_select_card",
						CardIndex = card.HandIndex,
						Label = card.Title,
						IsSupported = true
					});
				}
				if (combatState.HandSelection.CanConfirm)
				{
					actions.Add(new FullRunSimulationLegalAction
					{
						Action = "combat_confirm_selection",
						Label = "Confirm",
						IsSupported = true
					});
				}
				return actions;
			}
			if (combatState.IsCardSelectionActive && combatState.CardSelection != null)
			{
				foreach (CombatTrainingSelectableCardSnapshot card2 in combatState.CardSelection.SelectableCards)
				{
					actions.Add(new FullRunSimulationLegalAction
					{
						Action = "combat_select_card",
						CardIndex = card2.ChoiceIndex,
						Label = card2.Title,
						IsSupported = true
					});
				}
				if (combatState.CardSelection.CanConfirm)
				{
					actions.Add(new FullRunSimulationLegalAction
					{
						Action = "combat_confirm_selection",
						Label = "Confirm",
						IsSupported = true
					});
				}
				return actions;
			}
			if (!combatState.IsPlayPhase || combatState.PlayerActionsDisabled || combatState.IsActionQueueRunning)
			{
				Godot.GD.PrintErr($"[BuildCombatLegalActions] Returning empty: IsPlayPhase={combatState.IsPlayPhase} PlayerActionsDisabled={combatState.PlayerActionsDisabled} IsActionQueueRunning={combatState.IsActionQueueRunning} PlayerHP={combatState.Player?.CurrentHp}");
				return actions;
			}
			foreach (CombatTrainingHandCardSnapshot card3 in combatState.Hand ?? new List<CombatTrainingHandCardSnapshot>())
			{
				if (card3 == null)
				{
					continue;
				}
				if (!card3.CanPlay)
				{
					continue;
				}
				List<uint> validTargetIds = card3.ValidTargetIds ?? new List<uint>();
				if (card3.RequiresTarget && validTargetIds.Count > 0)
				{
					foreach (uint targetId in validTargetIds)
					{
						actions.Add(new FullRunSimulationLegalAction
						{
							Action = "play_card",
							CardIndex = card3.HandIndex,
							TargetId = targetId,
							Label = card3.Title,
							IsSupported = true
						});
					}
					continue;
				}
				actions.Add(new FullRunSimulationLegalAction
				{
					Action = "play_card",
					CardIndex = card3.HandIndex,
					Label = card3.Title,
					IsSupported = true
				});
			}
			foreach (PotionModel potion in localPlayer.Potions ?? Enumerable.Empty<PotionModel>())
			{
				if (potion == null)
				{
					continue;
				}
				int slot = localPlayer.GetPotionSlotIndex(potion);
				List<uint> targetIds = GetValidPotionTargetIds(potion, CombatManager.Instance.DebugOnlyGetState());
				if (potion.TargetType.IsSingleTarget() && targetIds.Count > 0)
				{
					foreach (uint targetId2 in targetIds)
					{
						actions.Add(new FullRunSimulationLegalAction
						{
							Action = "use_potion",
							Slot = slot,
							TargetId = targetId2,
							Label = SafeGetText(() => potion.Title),
							IsSupported = true
						});
					}
					continue;
				}
				actions.Add(new FullRunSimulationLegalAction
				{
					Action = "use_potion",
					Slot = slot,
					Label = SafeGetText(() => potion.Title),
					IsSupported = potion.Usage == PotionUsage.CombatOnly || potion.Usage == PotionUsage.AnyTime
				});
			}
			if (combatState.CanEndTurn)
			{
				actions.Add(new FullRunSimulationLegalAction
				{
					Action = "end_turn",
					Label = "End Turn",
					IsSupported = true
				});
			}
			return actions;
		}
		catch (Exception ex)
		{
			FullRunSimulationDiagnostics.Increment("combat_legal_actions.exception");
			FullRunSimulationTrace.Write($"headless_combat_legal_actions.exception exception={ex}");
			return actions;
		}
	}

	private static readonly List<uint> EmptyUintList = new List<uint>();

	private static List<uint> GetValidPotionTargetIds(PotionModel potion, CombatState? combatState)
	{
		if (combatState == null || !potion.TargetType.IsSingleTarget())
		{
			return EmptyUintList;
		}
		List<uint>? result = null;
		foreach (Creature creature in combatState.Creatures)
		{
			if (creature.CombatId.HasValue && IsPotionTargetValid(potion, creature))
			{
				result ??= new List<uint>();
				result.Add(creature.CombatId.Value);
			}
		}
		return result ?? EmptyUintList;
	}

	private static bool IsPotionTargetValid(PotionModel potion, Creature creature)
	{
		return potion.TargetType switch
		{
			TargetType.Self => creature == potion.Owner.Creature,
			TargetType.AnyEnemy => creature.Side == CombatSide.Enemy && creature.IsHittable,
			TargetType.AnyAlly => creature.Side == CombatSide.Player && creature.IsHittable,
			TargetType.AnyPlayer => creature.IsPlayer && creature.IsHittable,
			_ => true
		};
	}

	private static string GetMerchantEntryLabel(MerchantEntry entry)
	{
		return entry switch
		{
			MerchantCardEntry cardEntry => SafeGetText(() => cardEntry.CreationResult?.Card.Title, "Card"),
			MerchantRelicEntry relicEntry => SafeGetText(() => relicEntry.Model?.Title, "Relic"),
			MerchantPotionEntry potionEntry => SafeGetText(() => potionEntry.Model?.Title, "Potion"),
			MerchantCardRemovalEntry => SafeGetText(() => new LocString("merchant_room", "MERCHANT.cardRemovalService.title"), "Card Removal"),
			_ => entry.GetType().Name
		};
	}

	private static List<FullRunSimulationMapOption> BuildMapOptions(RunState runState)
	{
		IEnumerable<MapPoint> candidatePoints;
		if (runState.CurrentMapPoint != null)
		{
			candidatePoints = runState.CurrentMapPoint.Children;
		}
		else if (runState.Map.startMapPoints.Count > 0)
		{
			candidatePoints = runState.Map.startMapPoints;
		}
		else
		{
			candidatePoints = runState.Map.StartingMapPoint.Children;
		}
		return candidatePoints.OrderBy(static point => point.coord.row).ThenBy(static point => point.coord.col).Select(static (point, index) => new FullRunSimulationMapOption
		{
			Index = index,
			Col = point.coord.col,
			Row = point.coord.row,
			PointType = point.PointType.ToString().ToLowerInvariant()
		}).ToList();
	}

	private static List<FullRunSimulationMapNode> BuildFullMapNodes(RunState runState)
	{
		var nodes = new List<FullRunSimulationMapNode>();
		if (runState.Map == null)
			return nodes;

		// Starting point
		var start = runState.Map.StartingMapPoint;
		if (start != null)
			nodes.Add(MapPointToNode(start));

		// All grid nodes
		foreach (var pt in runState.Map.GetAllMapPoints())
			nodes.Add(MapPointToNode(pt));

		// Boss
		if (runState.Map.BossMapPoint != null)
			nodes.Add(MapPointToNode(runState.Map.BossMapPoint));
		if (runState.Map.SecondBossMapPoint != null)
			nodes.Add(MapPointToNode(runState.Map.SecondBossMapPoint));

		return nodes;
	}

	private static FullRunSimulationMapNode MapPointToNode(MapPoint pt)
	{
		return new FullRunSimulationMapNode
		{
			Col = pt.coord.col,
			Row = pt.coord.row,
			PointType = pt.PointType.ToString().ToLowerInvariant(),
			Children = pt.Children.OrderBy(c => c.coord.col).Select(c => (c.coord.col, c.coord.row)).ToList()
		};
	}

	private static List<FullRunSimulationLegalAction> BuildMapLegalActions(List<FullRunSimulationMapOption> options)
	{
		return options.Select(static option => new FullRunSimulationLegalAction
		{
			Action = "choose_map_node",
			Index = option.Index,
			Col = option.Col,
			Row = option.Row,
			Label = option.PointType,
			IsSupported = true
		}).ToList();
	}

	private static Player ResolveLocalPlayer(RunState runState)
	{
		Player? player = LocalContext.GetMe(runState.Players);
		if (player != null)
		{
			return player;
		}
		player = runState.Players.FirstOrDefault();
		if (player == null)
		{
			throw new InvalidOperationException("No player exists in the current run.");
		}
		LocalContext.NetId = player.NetId;
		return player;
	}

	private static string SafeGetText(Func<object?> getter, string fallback = "")
	{
		try
		{
			return SanitizeText(getter(), fallback);
		}
		catch
		{
			return fallback;
		}
	}

	private static string SanitizeText(object? value, string fallback = "")
	{
		if (value == null)
		{
			return fallback;
		}

		string text;
		try
		{
			text = value switch
			{
				LocString locString => locString.GetFormattedText(),
				_ => value.ToString() ?? fallback
			};
		}
		catch
		{
			return fallback;
		}

		return StripRichTextTags(text).Replace("\r", " ").Replace("\n", " ").Trim();
	}

	private static string StripRichTextTags(string text)
	{
		var sb = new System.Text.StringBuilder(text.Length);
		int index = 0;
		while (index < text.Length)
		{
			if (text[index] == '[')
			{
				int closing = text.IndexOf(']', index + 1);
				if (closing >= 0)
				{
					index = closing + 1;
					continue;
				}
			}

			sb.Append(text[index]);
			index++;
		}

		return sb.ToString();
	}
}
