using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Simulation;

public static class FullRunApiStateBuilder
{
	public static FullRunApiState Build(RunState? runState, FullRunSimulationStateSnapshot snapshot)
	{
		return Build(runState, snapshot, FullRunSimulationChoiceBridge.Instance, forceMapView: false);
	}

	internal static FullRunApiState Build(RunState? runState, FullRunSimulationStateSnapshot snapshot, FullRunSimulationChoiceBridge bridge, bool forceMapView)
	{
		Player? player = runState == null ? null : ResolveLocalPlayer(runState);
		FullRunChoiceBridgeSnapshotCache bridgeSnapshots = snapshot.CachedBridgeSnapshots ?? bridge.CaptureSnapshots();
		FullRunApiPlayerState? cachedPlayerState = null;
		FullRunApiPlayerState GetPlayerState()
		{
			if (cachedPlayerState == null)
			{
				cachedPlayerState = player == null ? new FullRunApiPlayerState() : BuildPlayerState(player);
			}
			return cachedPlayerState;
		}
		FullRunApiState state = new FullRunApiState
		{
			state_type = snapshot.StateType,
			terminal = snapshot.IsTerminal,
			is_pure_simulator = snapshot.IsPureSimulator,
			backend_kind = snapshot.BackendKind,
			coverage_tier = snapshot.CoverageTier,
			run_outcome = snapshot.RunOutcome,
			run = new FullRunApiRun
			{
				character_id = snapshot.CharacterId,
				seed = snapshot.Seed,
				ascension_level = snapshot.AscensionLevel,
				act = snapshot.CurrentActIndex + 1,
				floor = snapshot.TotalFloor,
				room_type = snapshot.RoomType,
				room_model_id = snapshot.RoomModelId
			},
			legal_actions = snapshot.LegalActions.Select(ToApiAction).ToList()
		};

		switch (snapshot.StateType)
		{
			case "menu":
				state.menu = new FullRunApiMenuState
				{
					actions = state.legal_actions.Select(CloneAction).ToList()
				};
				break;
			case "map":
				state.map = new FullRunApiMapState
				{
					player = GetPlayerState(),
					next_options = snapshot.MapOptions.Select(static option => new FullRunApiMapOption
					{
						index = option.Index,
						col = option.Col,
						row = option.Row,
						point_type = option.PointType
					}).ToList()
				};
				break;
			case "event":
				state.@event = BuildEventState(runState?.CurrentRoom as EventRoom, GetPlayerState());
				break;
			case "rest_site":
				state.rest_site = BuildRestSiteState(GetPlayerState());
				break;
			case "shop":
				state.shop = BuildShopState(runState?.CurrentRoom as MerchantRoom, GetPlayerState());
				break;
			case "treasure":
				state.treasure = BuildTreasureState(GetPlayerState());
				break;
			case "combat_rewards":
				state.rewards = BuildRewardsState(bridgeSnapshots.RewardSelection, GetPlayerState());
				break;
			case "card_reward":
				state.card_reward = BuildCardRewardState(bridgeSnapshots.CardRewardSelection, GetPlayerState());
				break;
			case "card_select":
				state.card_select = BuildCardSelectState(bridgeSnapshots.CardSelection, GetPlayerState());
				break;
			case "relic_select":
				state.relic_select = BuildRelicSelectState(bridgeSnapshots.RelicSelection, GetPlayerState());
				break;
			case "hand_select":
			case "monster":
			case "elite":
			case "boss":
				BuildCombatState(state, snapshot, player, cachedPlayerState);
				break;
			case "game_over":
				state.game_over = new FullRunApiGameOverState
				{
					run_outcome = snapshot.RunOutcome,
					available_actions = state.legal_actions.Select(CloneAction).ToList()
				};
				break;
		}

		return state;
	}

	public static string Signature(RunState? runState, FullRunSimulationStateSnapshot snapshot)
	{
		return Signature(runState, snapshot, FullRunSimulationChoiceBridge.Instance, forceMapView: false);
	}

	internal static string Signature(RunState? runState, FullRunSimulationStateSnapshot snapshot, FullRunSimulationChoiceBridge bridge, bool forceMapView)
	{
		// Lightweight signature: avoids full HTTP state build (which takes ~5ms per call).
		// Called up to 240x per WaitForStateChangeAsync loop — must be fast.
		var sb = new System.Text.StringBuilder(64);
		sb.Append(snapshot.StateType);
		sb.Append('|'); sb.Append(snapshot.IsTerminal ? '1' : '0');
		sb.Append('|'); sb.Append(snapshot.IsActionable ? '1' : '0');
		sb.Append('|'); sb.Append(snapshot.TotalFloor);
		sb.Append('|'); sb.Append(snapshot.RunOutcome ?? string.Empty);
		sb.Append('|'); sb.Append(snapshot.LegalActions.Count);
		foreach (var a in snapshot.LegalActions)
		{
			sb.Append('|'); sb.Append(a.Action);
			sb.Append(':'); sb.Append(a.Index ?? -1);
			sb.Append(','); sb.Append(a.Col ?? -1);
			sb.Append(','); sb.Append(a.Row ?? -1);
		}
		return sb.ToString();
	}

	private static void BuildCombatState(FullRunApiState state, FullRunSimulationStateSnapshot snapshot, Player? player, FullRunApiPlayerState? basePlayerState)
	{
		using IDisposable _ = FullRunSimulationDiagnostics.Measure("api.build_combat_state_ms");
		if (player == null)
		{
			FullRunSimulationDiagnostics.Increment("combat_api.player_null");
			return;
		}
		CombatTrainingStateSnapshot? combat = snapshot.CachedCombatState;
		if (combat != null)
		{
			FullRunSimulationDiagnostics.Increment("combat_api.snapshot_reused");
		}
		else
		{
			try
			{
				combat = CombatTrainingEnvService.BuildStateSnapshot();
			}
			catch (Exception ex)
			{
				FullRunSimulationDiagnostics.Increment("combat_api.snapshot_exception");
				FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
				FullRunSimulationTrace.Write($"headless_api_build_combat_state.snapshot_exception exception={ex}");
				combat = new CombatTrainingStateSnapshot
				{
					IsCombatActive = CombatManager.Instance.IsInProgress
				};
			}
			if (combat == null)
			{
				FullRunSimulationDiagnostics.Increment("combat_api.snapshot_null");
				FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
				FullRunSimulationTrace.Write("headless_api_build_combat_state.snapshot_null");
				combat = new CombatTrainingStateSnapshot
				{
					IsCombatActive = CombatManager.Instance.IsInProgress
				};
			}
		}

		List<FullRunApiBattleEnemy> enemies = SafeBuildCombatEnemies(combat);
		FullRunApiPlayerState combatPlayerState = SafeBuildCombatPlayerState(player, combat, basePlayerState);

		try
		{
			FullRunSimulationTrace.Write($"headless_api_build_combat_state.assign.begin player_null={combatPlayerState == null} enemies={enemies.Count} card_select={combat.IsCardSelectionActive} hand_select={(state.state_type == "hand_select")}");
			state.battle = new FullRunApiBattleState
			{
				round = combat.RoundNumber,
				turn = combat.CurrentSide.ToString().ToLowerInvariant(),
				is_play_phase = combat.IsPlayPhase,
				player = combatPlayerState,
				enemies = enemies
			};
			FullRunSimulationTrace.Write("headless_api_build_combat_state.assign.done");
			if (combat.IsCardSelectionActive && combat.CardSelection != null)
			{
				try
				{
					state.card_selection = BuildCombatCardSelectionState(combat.CardSelection);
					state.battle.card_selection = state.card_selection;
				}
				catch (Exception ex)
				{
					FullRunSimulationDiagnostics.Increment("combat_api.card_selection_exception");
					FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
					FullRunSimulationTrace.Write($"headless_api_build_combat_state.card_selection_exception exception={ex}");
				}
			}
			FullRunSimulationTrace.Write("headless_api_build_combat_state.card_selection.done");
			if (state.state_type == "hand_select" && combat.HandSelection != null)
			{
				try
				{
					state.hand_select = BuildHandSelectState(combat.HandSelection);
				}
				catch (Exception ex)
				{
					FullRunSimulationDiagnostics.Increment("combat_api.hand_selection_exception");
					FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
					FullRunSimulationTrace.Write($"headless_api_build_combat_state.hand_selection_exception exception={ex}");
				}
			}
			FullRunSimulationTrace.Write("headless_api_build_combat_state.hand_selection.done");
		}
		catch (Exception ex)
		{
			FullRunSimulationDiagnostics.Increment("combat_api.final_exception");
			FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
			FullRunSimulationTrace.Write($"headless_api_build_combat_state.final_exception exception={ex}");
			FullRunSimulationTrace.Write($"headless_api_build_combat_state.fallback.begin player_null={combatPlayerState == null} enemies={enemies.Count}");
			state.battle = new FullRunApiBattleState
			{
				round = combat.RoundNumber,
				turn = SafeValue(() => combat.CurrentSide.ToString().ToLowerInvariant(), "player"),
				is_play_phase = combat.IsPlayPhase,
				player = combatPlayerState ?? new FullRunApiPlayerState(),
				enemies = enemies ?? new List<FullRunApiBattleEnemy>()
			};
			FullRunSimulationTrace.Write("headless_api_build_combat_state.fallback.done");
		}
	}

	private static FullRunApiEventState BuildEventState(EventRoom? eventRoom, FullRunApiPlayerState playerState)
	{
		FullRunApiEventState state = new FullRunApiEventState
		{
			player = playerState
		};
		if (eventRoom == null)
		{
			return state;
		}
		EventModel localEvent = eventRoom.LocalMutableEvent;
		state.event_id = localEvent.Id.ToString();
		state.is_finished = localEvent.IsFinished;
		state.options = localEvent.CurrentOptions.Select(static (option, index) => new FullRunApiEventOption
		{
			index = index,
			text = option.Title.GetRawText(),
			is_locked = option.IsLocked,
			is_chosen = option.WasChosen,
			is_proceed = option.IsProceed
		}).ToList();
		state.in_dialogue = !state.is_finished && state.options.Count == 0;
		return state;
	}

	private static FullRunApiRestSiteState BuildRestSiteState(FullRunApiPlayerState playerState)
	{
		IReadOnlyList<RestSiteOption> options = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
		return new FullRunApiRestSiteState
		{
			player = playerState,
			can_proceed = options.Count == 0,
			options = options.Select(static (option, index) => new FullRunApiRestSiteOption
			{
				index = index,
				id = option.OptionId.ToString(),
				name = SafeGetText(() => option.Title),
				description = SafeGetText(() => option.Description),
				is_enabled = option.IsEnabled
			}).ToList()
		};
	}

	private static FullRunApiShopState BuildShopState(MerchantRoom? merchantRoom, FullRunApiPlayerState playerState)
	{
		FullRunApiShopState state = new FullRunApiShopState
		{
			player = playerState,
			is_open = merchantRoom != null,
			can_proceed = merchantRoom != null
		};
		if (merchantRoom == null)
		{
			return state;
		}
		int index = 0;
		foreach (MerchantEntry entry in EnumerateShopEntries(merchantRoom.Inventory))
		{
			FullRunApiShopItem item = new FullRunApiShopItem
			{
				index = index++,
				cost = entry.Cost,
				can_afford = entry.EnoughGold,
				is_stocked = entry.IsStocked,
				description = GetMerchantEntryDescription(entry)
			};
			switch (entry)
			{
				case MerchantCardEntry cardEntry:
					item.category = "card";
					item.name = cardEntry.CreationResult?.Card.Title;
					item.card_id = cardEntry.CreationResult?.Card.Id.Entry;
					item.card_name = cardEntry.CreationResult?.Card.Title;
					item.card_type = cardEntry.CreationResult?.Card.Type.ToString();
					item.card_rarity = cardEntry.CreationResult?.Card.Rarity.ToString();
					item.card_description = cardEntry.CreationResult == null ? null : GetCardDescription(cardEntry.CreationResult.Card);
					item.on_sale = cardEntry.IsOnSale;
					item.keywords = cardEntry.CreationResult?.Card.Keywords.Select(static keyword => keyword.ToString()).ToList() ?? new List<string>();
					break;
				case MerchantRelicEntry relicEntry:
					item.category = "relic";
					item.name = SafeGetText(() => relicEntry.Model?.Title);
					item.relic_id = relicEntry.Model?.Id.Entry;
					item.relic_name = SafeGetText(() => relicEntry.Model?.Title);
					item.relic_description = SafeGetText(() => relicEntry.Model?.DynamicDescription);
					break;
				case MerchantPotionEntry potionEntry:
					item.category = "potion";
					item.name = SafeGetText(() => potionEntry.Model?.Title);
					item.potion_id = potionEntry.Model?.Id.Entry;
					item.potion_name = SafeGetText(() => potionEntry.Model?.Title);
					item.potion_description = SafeGetText(() => potionEntry.Model?.DynamicDescription);
					break;
				case MerchantCardRemovalEntry:
					item.category = "card_removal";
					item.name = GetMerchantEntryName(entry);
					break;
				default:
					item.category = "unknown";
					item.name = GetMerchantEntryName(entry);
					break;
			}
			state.items.Add(item);
		}
		return state;
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

	private static FullRunApiTreasureState BuildTreasureState(FullRunApiPlayerState playerState)
	{
		IReadOnlyList<RelicModel>? relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		return new FullRunApiTreasureState
		{
			player = playerState,
			can_proceed = relics == null,
			relics = relics?.Select(static relic => ToApiRelicOption(relic)).ToList() ?? new List<FullRunApiRelicOption>()
		};
	}

	private static FullRunApiRewardsState BuildRewardsState(FullRunPendingRewardSelectionSnapshot? rewards, FullRunApiPlayerState playerState)
	{
		string rewardSource = GetRewardSourceForUi(RunManager.Instance.DebugOnlyGetState());
		return new FullRunApiRewardsState
		{
			player = playerState,
			can_proceed = rewards?.CanProceed ?? false,
			items = rewards?.Rewards.Select((reward, index) => ToApiRewardItem(reward, index, playerState, rewardSource)).ToList()
				?? new List<FullRunApiRewardItem>()
		};
	}

	private static FullRunApiCardRewardState BuildCardRewardState(FullRunPendingCardRewardSnapshot? reward, FullRunApiPlayerState playerState)
	{
		return new FullRunApiCardRewardState
		{
			player = playerState,
			can_skip = reward?.CanSkip ?? false,
			cards = reward?.Options.Select((card, index) => ToApiCardOption(card.Card, index)).ToList() ?? new List<FullRunApiCardOption>()
		};
	}

	private static FullRunApiCardSelectState BuildCardSelectState(CombatTrainingCardSelectionSnapshot? selection, FullRunApiPlayerState playerState)
	{
		return new FullRunApiCardSelectState
		{
			player = playerState,
			screen_type = selection?.Mode,
			prompt = selection?.PromptText,
			min_select = selection?.MinSelect ?? 0,
			max_select = selection?.MaxSelect ?? 0,
			selected_count = selection?.SelectedCards.Count ?? 0,
			remaining_picks = Math.Max(0, (selection?.MaxSelect ?? 0) - (selection?.SelectedCards.Count ?? 0)),
			can_confirm = selection?.CanConfirm ?? false,
			can_cancel = selection?.Cancelable ?? false,
			preview_showing = false,
			requires_manual_confirmation = true,
			cards = selection?.SelectableCards.Select(ToApiSelectableCardOption).ToList() ?? new List<FullRunApiCardOption>(),
			selected_cards = selection?.SelectedCards.Select(ToApiSelectableCardOption).ToList() ?? new List<FullRunApiCardOption>()
		};
	}

	private static FullRunApiRelicSelectState BuildRelicSelectState(FullRunPendingRelicSelectionSnapshot? relics, FullRunApiPlayerState playerState)
	{
		return new FullRunApiRelicSelectState
		{
			player = playerState,
			can_skip = relics?.CanSkip ?? false,
			relics = relics?.Relics.Select((relic, index) => ToApiRelicOption(relic, index)).ToList() ?? new List<FullRunApiRelicOption>()
		};
	}

	private static FullRunApiHandSelectState BuildHandSelectState(CombatTrainingHandSelectionSnapshot selection)
	{
		return new FullRunApiHandSelectState
		{
			prompt = selection.PromptText,
			min_select = selection.MinSelect,
			max_select = selection.MaxSelect,
			can_confirm = selection.CanConfirm,
			cards = selection.SelectableCards.Select(ToApiHandCardOption).ToList(),
			selected_cards = selection.SelectedCards.Select(ToApiHandCardOption).ToList()
		};
	}

	private static FullRunApiCombatCardSelectionState BuildCombatCardSelectionState(CombatTrainingCardSelectionSnapshot selection)
	{
		return new FullRunApiCombatCardSelectionState
		{
			prompt = selection.PromptText,
			mode = selection.Mode,
			min_select = selection.MinSelect,
			max_select = selection.MaxSelect,
			can_confirm = selection.CanConfirm,
			can_cancel = selection.Cancelable,
			selectable_cards = selection.SelectableCards.Select(ToApiSelectableCardOption).ToList(),
			selected_cards = selection.SelectedCards.Select(ToApiSelectableCardOption).ToList()
		};
	}

	private static FullRunApiPlayerState BuildPlayerState(Player player)
	{
		return new FullRunApiPlayerState
		{
			character = SafeValue(() => player.Character.Id.Entry, player.Character?.Id?.Entry),
			hp = SafeValue(() => player.Creature.CurrentHp),
			max_hp = SafeValue(() => player.Creature.MaxHp),
			block = SafeValue(() => player.Creature.Block),
			gold = SafeValue(() => player.Gold),
			energy = SafeValue(() => player.PlayerCombatState?.Energy ?? 0),
			max_energy = SafeValue(() => player.PlayerCombatState?.MaxEnergy ?? player.MaxEnergy, player.MaxEnergy),
			draw_pile_count = SafeValue(() => player.PlayerCombatState?.DrawPile.Cards.Count ?? 0),
			discard_pile_count = SafeValue(() => player.PlayerCombatState?.DiscardPile.Cards.Count ?? 0),
			exhaust_pile_count = SafeValue(() => player.PlayerCombatState?.ExhaustPile.Cards.Count ?? 0),
			open_potion_slots = SafeValue(() => player.PotionSlots.Count(static potion => potion == null)),
			status = SafeBuildPlayerPowers(player),
			deck = SafeBuildDeck(player),
			relics = SafeBuildRelics(player),
			potions = SafeBuildPotions(player)
		};
	}

	private static FullRunApiPlayerState BuildCombatPlayerState(Player player, CombatTrainingStateSnapshot? combat, FullRunApiPlayerState? baseState)
	{
		FullRunApiPlayerState state = baseState ?? BuildPlayerState(player);
		if (combat == null)
		{
			return state;
		}
		state.energy = combat.Player?.Energy ?? state.energy;
		state.max_energy = combat.Player?.MaxEnergy ?? state.max_energy;
		state.draw_pile_count = combat.Piles?.Draw ?? state.draw_pile_count;
		state.discard_pile_count = combat.Piles?.Discard ?? state.discard_pile_count;
		state.exhaust_pile_count = combat.Piles?.Exhaust ?? state.exhaust_pile_count;
		state.hand = SafeBuildCombatHand(combat);
		state.status = SafeBuildCombatPlayerPowers(combat, state.status);
		return state;
	}

	private static FullRunApiBattleEnemy BuildBattleEnemy(CombatTrainingCreatureSnapshot enemy)
	{
		return new FullRunApiBattleEnemy
		{
			entity_id = enemy.Id,
			combat_id = enemy.CombatId,
			name = enemy.Name,
			hp = enemy.CurrentHp,
			max_hp = enemy.MaxHp,
			block = enemy.Block,
			is_alive = enemy.IsAlive,
			// 2026-04-08 PM: forward is_hittable so Python can avoid wasting actions
			// on minion shields / non-targetable boss summons.
			is_hittable = enemy.IsHittable,
			// 2026-04-08: forward boss move/phase id + intent intent flag so Python
			// can distinguish Vantom InkBlot vs Dismember etc.
			next_move_id = enemy.NextMoveId,
			intends_to_attack = enemy.IntendsToAttack,
			status = enemy.Powers?.Where(static power => power != null).Select(static power => new FullRunApiPower
			{
				id = power.Id,
				amount = power.Amount
			}).ToList() ?? new List<FullRunApiPower>(),
			intents = enemy.Intents?.Where(static intent => intent != null).Select(static intent => new FullRunApiIntent
			{
				type = intent.IntentType.ToLowerInvariant(),
				label = intent.TotalDamage.HasValue ? intent.TotalDamage.Value.ToString() : intent.IntentType,
				title = intent.IntentType,
				description = intent.TotalDamage.HasValue ? intent.TotalDamage.Value.ToString() : intent.IntentType,
				total_damage = intent.TotalDamage,
				// 2026-04-08: forward per-hit damage + repeats for accurate multi-hit features.
				// Field renamed `base_damage` -> `damage` 2026-04-08 PM to align with the
				// binary pipe wire format and `rl_encoder_v2._enemy_aux_features`'s read order.
				damage = intent.Damage,
				repeats = intent.Repeats
			}).ToList() ?? new List<FullRunApiIntent>()
		};
	}

	private static FullRunApiAction ToApiAction(FullRunSimulationLegalAction action)
	{
		return new FullRunApiAction
		{
			action = action.Action,
			index = action.Index,
			col = action.Col,
			row = action.Row,
			card_index = action.CardIndex,
			card_id = action.CardId,
			card_type = action.CardType,
			card_rarity = action.CardRarity,
			cost = action.Cost,
			is_upgraded = action.IsUpgraded,
			reward_type = action.RewardType,
			reward_key = action.RewardKey,
			reward_source = action.RewardSource,
			claimable = action.Claimable,
			claim_block_reason = action.ClaimBlockReason,
			slot = action.Slot,
			target_id = action.TargetId,
			target = action.Target,
			label = SanitizeText(action.Label, action.Action ?? string.Empty),
			is_enabled = action.IsSupported,
			note = SanitizeText(action.Note)
		};
	}

	private static FullRunApiAction CloneAction(FullRunApiAction action)
	{
		return new FullRunApiAction
		{
			action = action.action,
			index = action.index,
			col = action.col,
			row = action.row,
			card_index = action.card_index,
			card_id = action.card_id,
			card_type = action.card_type,
			card_rarity = action.card_rarity,
			cost = action.cost,
			is_upgraded = action.is_upgraded,
			slot = action.slot,
			target_id = action.target_id,
			target = action.target,
			label = action.label,
			is_enabled = action.is_enabled,
			note = action.note
		};
	}

	private static FullRunApiPotionState ToApiPotionState(PotionModel potion, int slot, bool canUseInCombat)
	{
		return new FullRunApiPotionState
		{
			slot = slot,
			id = potion.Id.Entry,
			name = SafeGetText(() => potion.Title),
			description = SafeGetText(() => potion.DynamicDescription),
			target_type = potion.TargetType.ToString(),
			can_use_in_combat = canUseInCombat && (potion.Usage == PotionUsage.CombatOnly || potion.Usage == PotionUsage.AnyTime) && !potion.IsQueued
		};
	}

	private static FullRunApiRelicOption ToApiRelicOption(RelicModel relic)
	{
		return ToApiRelicOption(relic, 0);
	}

	private static FullRunApiRelicOption ToApiRelicOption(RelicModel relic, int index)
	{
		return new FullRunApiRelicOption
		{
			index = index,
			id = relic.Id.Entry,
			name = SafeGetText(() => relic.Title),
			rarity = relic.Rarity.ToString(),
			description = SafeGetText(() => relic.Description)
		};
	}

	private static FullRunApiCardOption ToApiCardOption(CardModel card, int index)
	{
		return new FullRunApiCardOption
		{
			index = index,
			id = card.Id.Entry,
			name = SafeGetText(() => card.Title, card.Id.Entry),
			type = card.Type.ToString(),
			rarity = card.Rarity.ToString(),
			cost = card.EnergyCost.GetWithModifiers(CostModifiers.All),
			is_upgraded = card.IsUpgraded,
			target_type = card.TargetType.ToString(),
			description = GetCardDescription(card),
			keywords = card.Keywords.Select(static keyword => keyword.ToString()).ToList()
		};
	}

	private static string GetMerchantEntryName(MerchantEntry entry)
	{
		return entry switch
		{
			MerchantCardEntry cardEntry => cardEntry.CreationResult == null
				? "Card"
				: SafeGetText(() => cardEntry.CreationResult.Card.Title, cardEntry.CreationResult.Card.Id.Entry),
			MerchantRelicEntry relicEntry => SafeGetText(() => relicEntry.Model?.Title, "Relic"),
			MerchantPotionEntry potionEntry => SafeGetText(() => potionEntry.Model?.Title, "Potion"),
			MerchantCardRemovalEntry => SafeGetText(() => new LocString("merchant_room", "MERCHANT.cardRemovalService.title"), "Card Removal"),
			_ => entry.GetType().Name
		};
	}

	private static string GetMerchantEntryDescription(MerchantEntry entry)
	{
		return entry switch
		{
			MerchantCardEntry cardEntry when cardEntry.CreationResult != null => GetCardDescription(cardEntry.CreationResult.Card),
			MerchantRelicEntry relicEntry => SafeGetText(() => relicEntry.Model?.DynamicDescription),
			MerchantPotionEntry potionEntry => SafeGetText(() => potionEntry.Model?.DynamicDescription),
			MerchantCardRemovalEntry cardRemovalEntry => BuildCardRemovalDescription(cardRemovalEntry),
			_ => string.Empty
		};
	}

	private static string BuildCardRemovalDescription(MerchantCardRemovalEntry entry)
	{
		LocString description = new LocString("merchant_room", "MERCHANT.cardRemovalService.description");
		description.Add("Amount", entry.CalcPriceIncrease());
		return SafeGetText(() => description);
	}

	private static string GetCardDescription(CardModel card)
	{
		try
		{
			return SanitizeText(card.GetDescriptionForPile(card.Pile?.Type ?? PileType.Deck), string.Empty);
		}
		catch
		{
			return SafeGetText(() => card.Description);
		}
	}

	private static FullRunApiCardOption ToApiHandCardOption(CombatTrainingHandCardSnapshot card)
	{
		return new FullRunApiCardOption
		{
			index = card.HandIndex,
			id = card.Id,
			name = card.Title,
			type = card.CardType,
			cost = card.EnergyCost,
			is_upgraded = card.IsUpgraded,
			can_play = card.CanPlay,
			target_type = SafeValue(() => card.TargetType.ToString(), string.Empty),
			description = card.Description,
			valid_target_ids = card.ValidTargetIds?.ToList() ?? new List<uint>(),
			keywords = card.Keywords?.ToList() ?? new List<string>()
		};
	}

	private static FullRunApiCardOption ToApiSelectableCardOption(CombatTrainingSelectableCardSnapshot card)
	{
		return new FullRunApiCardOption
		{
			index = card.ChoiceIndex,
			id = card.Id,
			name = card.Title,
			type = SafeValue(() => card.Type.ToString(), string.Empty),
			cost = card.EnergyCost,
			is_upgraded = card.IsUpgraded,
			target_type = SafeValue(() => card.TargetType.ToString(), string.Empty)
		};
	}

	private static string RewardTypeForUi(Reward reward)
	{
		return reward switch
		{
			GoldReward => "gold",
			PotionReward => "potion",
			RelicReward => "relic",
			CardReward => "card",
			CardRemovalReward => "card_remove",
			SpecialCardReward => "special_card",
			_ => reward.GetType().Name.Replace("Reward", "").ToLowerInvariant()
		};
	}

	private static FullRunApiRewardItem ToApiRewardItem(Reward reward, int index, FullRunApiPlayerState playerState, string rewardSource)
	{
		bool claimable = IsRewardClaimableForUi(reward, playerState, out string? claimBlockReason);
		return new FullRunApiRewardItem
		{
			index = index,
			type = RewardTypeForUi(reward),
			label = SafeGetText(() => reward.Description),
			reward_key = BuildRewardKeyForUi(reward),
			reward_source = rewardSource,
			claimable = claimable,
			claim_block_reason = claimBlockReason
		};
	}

	private static string BuildRewardKeyForUi(Reward reward)
	{
		string rewardType = RewardTypeForUi(reward);
		string label = SafeGetText(() => reward.Description);
		string specificId = reward switch
		{
			PotionReward potionReward => potionReward.Potion?.Id.Entry ?? string.Empty,
			CardReward => "card_reward",
			CardRemovalReward => "card_remove",
			GoldReward goldReward => SafeValue(() => goldReward.Amount.ToString(), string.Empty),
			RelicReward => "relic_reward",
			SpecialCardReward => "special_card_reward",
			_ => string.Empty
		};
		return $"{rewardType}|{specificId}|{label}".ToLowerInvariant();
	}

	private static bool IsRewardClaimableForUi(Reward reward, FullRunApiPlayerState playerState, out string? claimBlockReason)
	{
		claimBlockReason = null;
		if (reward is PotionReward && playerState.open_potion_slots <= 0)
		{
			claimBlockReason = "potion_slots_full";
			return false;
		}
		return true;
	}

	private static string GetRewardSourceForUi(RunState? runState)
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
			null => "unknown",
			_ => room.RoomType.ToString().ToLowerInvariant()
		};
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

	private static FullRunApiPlayerState SafeBuildCombatPlayerState(Player player, CombatTrainingStateSnapshot? combat, FullRunApiPlayerState? baseState)
	{
		try
		{
			return BuildCombatPlayerState(player, combat, baseState);
		}
		catch (Exception ex)
		{
			FullRunSimulationDiagnostics.Increment("combat_api.player_exception");
			FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
			FullRunSimulationTrace.Write($"headless_api_build_combat_state.player_exception exception={ex}");
			return baseState ?? SafeBuildPlayerState(player);
		}
	}

	private static FullRunApiPlayerState SafeBuildPlayerState(Player player)
	{
		try
		{
			return BuildPlayerState(player);
		}
		catch (Exception ex)
		{
			FullRunSimulationDiagnostics.Increment("combat_api.player_fallback_exception");
			FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
			FullRunSimulationTrace.Write($"headless_api_build_combat_state.player_fallback_exception exception={ex}");
			return new FullRunApiPlayerState
			{
				character = SafeValue(() => player.Character.Id.Entry, player.Character?.Id?.Entry),
				hp = SafeValue(() => player.Creature.CurrentHp),
				max_hp = SafeValue(() => player.Creature.MaxHp),
				block = SafeValue(() => player.Creature.Block),
				gold = SafeValue(() => player.Gold),
				max_energy = SafeValue(() => player.MaxEnergy, 0)
			};
		}
	}

	private static List<FullRunApiBattleEnemy> SafeBuildCombatEnemies(CombatTrainingStateSnapshot? combat)
	{
		List<FullRunApiBattleEnemy> enemies = new List<FullRunApiBattleEnemy>();
		foreach (CombatTrainingCreatureSnapshot? enemy in combat?.Enemies ?? new List<CombatTrainingCreatureSnapshot>())
		{
			if (enemy == null)
			{
				continue;
			}
			try
			{
				enemies.Add(BuildBattleEnemy(enemy));
			}
			catch (Exception ex)
			{
				FullRunSimulationDiagnostics.Increment("combat_api.enemies_exception");
				FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
				FullRunSimulationTrace.Write($"headless_api_build_combat_state.enemies_exception exception={ex}");
			}
		}
		return enemies;
	}

	private static List<FullRunApiPower> SafeBuildPlayerPowers(Player player)
	{
		try
		{
			return player.Creature?.Powers?
				.Where(static power => power != null)
				.Select(static power => new FullRunApiPower
				{
					id = power.Id?.Entry,
					amount = power.Amount
				})
				.ToList()
				?? new List<FullRunApiPower>();
		}
		catch
		{
			return new List<FullRunApiPower>();
		}
	}

	private static List<FullRunApiCardOption> SafeBuildDeck(Player player)
	{
		try
		{
			return player.Deck?.Cards?
				.Where(static card => card != null)
				.Select((card, index) => ToApiCardOption(card, index))
				.ToList()
				?? new List<FullRunApiCardOption>();
		}
		catch
		{
			return new List<FullRunApiCardOption>();
		}
	}

	private static List<FullRunApiRelicOption> SafeBuildRelics(Player player)
	{
		try
		{
			return player.Relics?
				.Where(static relic => relic != null)
				.Select((relic, index) => ToApiRelicOption(relic, index))
				.ToList()
				?? new List<FullRunApiRelicOption>();
		}
		catch
		{
			return new List<FullRunApiRelicOption>();
		}
	}

	private static List<FullRunApiPotionState> SafeBuildPotions(Player player)
	{
		try
		{
			return player.PotionSlots?
				.Select((potion, index) => potion == null ? null : ToApiPotionState(potion, index, canUseInCombat: player.PlayerCombatState != null))
				.OfType<FullRunApiPotionState>()
				.ToList()
				?? new List<FullRunApiPotionState>();
		}
		catch
		{
			return new List<FullRunApiPotionState>();
		}
	}

	private static List<FullRunApiCardOption> SafeBuildCombatHand(CombatTrainingStateSnapshot? combat)
	{
		List<FullRunApiCardOption> hand = new List<FullRunApiCardOption>();
		foreach (CombatTrainingHandCardSnapshot? card in combat?.Hand ?? new List<CombatTrainingHandCardSnapshot>())
		{
			if (card == null)
			{
				continue;
			}
			try
			{
				hand.Add(ToApiHandCardOption(card));
			}
			catch (Exception ex)
			{
				FullRunSimulationDiagnostics.Increment("combat_api.hand_card_exception");
				FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
				FullRunSimulationTrace.Write($"headless_api_build_combat_state.hand_card_exception exception={ex}");
			}
		}
		return hand;
	}

	private static List<FullRunApiPower> SafeBuildCombatPlayerPowers(CombatTrainingStateSnapshot? combat, List<FullRunApiPower> fallback)
	{
		try
		{
			return combat.Player?.Powers?
				.Where(static power => power != null)
				.Select(static power => new FullRunApiPower
				{
					id = power.Id,
					amount = power.Amount
				})
				.ToList()
				?? fallback;
		}
		catch (Exception ex)
		{
			FullRunSimulationDiagnostics.Increment("combat_api.player_powers_exception");
			FullRunSimulationDiagnostics.Increment("combat_api.fallback_hits");
			FullRunSimulationTrace.Write($"headless_api_build_combat_state.player_powers_exception exception={ex}");
			return fallback;
		}
	}

	private static T SafeValue<T>(Func<T> getter, T fallback = default!)
	{
		try
		{
			return getter();
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
				int closeIndex = text.IndexOf(']', index);
				if (closeIndex >= 0)
				{
					index = closeIndex + 1;
					continue;
				}
			}

			sb.Append(text[index]);
			index++;
		}

		return sb.ToString();
	}
}
