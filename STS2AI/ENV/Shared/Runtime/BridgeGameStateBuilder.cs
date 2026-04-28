using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.Training;

namespace STS2AI.Bridge.Runtime;

public static class BridgeGameStateBuilder
{
	private const int MaxPileCards = 50;

	public static GameState FromFullRunSnapshot(FullRunSimulationStateSnapshot snapshot)
	{
		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		Player? player = TryResolveLocalPlayer(runState);
		GameState gs = new GameState
		{
			StateType = snapshot.StateType ?? "other",
			Terminal = snapshot.IsTerminal,
			RunOutcome = snapshot.RunOutcome ?? "",
			EncounterId = "",
			Run = new RunInfo
			{
				Act = Math.Clamp(snapshot.CurrentActIndex + 1, 0, 255),
				Floor = Math.Clamp(snapshot.TotalFloor, 0, 255)
			}
		};

		if (player != null)
		{
			gs.Player = BuildPlayerState(player, snapshot);
		}
		foreach (FullRunSimulationLegalAction action in snapshot.LegalActions)
		{
			gs.LegalActions.Add(BuildLegalAction(action));
		}

		switch (snapshot.StateType)
		{
			case "map":
				gs.Map = BuildMapState(snapshot);
				break;
			case "event":
				gs.Event = BuildEventState(runState?.CurrentRoom as EventRoom);
				break;
			case "rest_site":
				gs.RestSite = BuildRestSiteState();
				break;
			case "shop":
				gs.Shop = BuildShopState(runState?.CurrentRoom as MerchantRoom);
				break;
			case "treasure":
				gs.Treasure = BuildTreasureState();
				break;
			case "combat_rewards":
			case "combat_post_end_pending":
				gs.CombatRewards = BuildCombatRewardsState(snapshot.CachedBridgeSnapshots?.RewardSelection, player);
				break;
			case "card_reward":
				gs.CardReward = BuildCardRewardState(snapshot.CachedBridgeSnapshots?.CardRewardSelection);
				break;
			case "card_select":
				gs.CardSelect = BuildCardSelectState(snapshot.CachedBridgeSnapshots?.CardSelection);
				break;
			case "relic_select":
				gs.RelicSelect = BuildRelicSelectState(snapshot.CachedBridgeSnapshots?.RelicSelection);
				break;
			case "hand_select":
			case "monster":
			case "elite":
			case "boss":
				gs.Battle = BuildBattleState(snapshot.CachedCombatState, player);
				if (snapshot.CachedCombatState?.IsCardSelectionActive == true
					&& snapshot.CachedCombatState.CardSelection != null)
				{
					gs.CardSelect = BuildCardSelectState(snapshot.CachedCombatState.CardSelection);
				}
				break;
		}
		return gs;
	}

	public static GameState FromCombatSnapshot(CombatTrainingStateSnapshot snapshot)
	{
		Player? runtimePlayer = TryResolveActiveCombatPlayer();
		GameState gs = new GameState
		{
			StateType = DetectCombatStateType(snapshot),
			Terminal = snapshot.IsEpisodeDone,
			RunOutcome = snapshot.IsEpisodeDone
				? (snapshot.Victory == true ? "victory" : (snapshot.Victory == false ? "defeat" : ""))
				: "",
			EncounterId = snapshot.EncounterId ?? "",
			Run = new RunInfo()
		};
		if (snapshot.Player != null || runtimePlayer != null)
		{
			gs.Player = BuildCombatPlayerState(runtimePlayer, snapshot);
		}
		gs.Battle = BuildBattleState(snapshot, runtimePlayer);
		PopulateCombatLegalActions(gs, snapshot);
		return gs;
	}

	private static Player? TryResolveActiveCombatPlayer()
	{
		try
		{
			return TryResolveLocalPlayer(RunManager.Instance.DebugOnlyGetState());
		}
		catch
		{
			return null;
		}
	}

	private static Player? TryResolveLocalPlayer(RunState? runState)
	{
		if (runState == null) return null;
		Player? player = LocalContext.GetMe(runState.Players);
		if (player != null) return player;
		player = runState.Players.FirstOrDefault();
		if (player != null) LocalContext.NetId ??= player.NetId;
		return player;
	}

	private static PlayerState BuildPlayerState(Player player, FullRunSimulationStateSnapshot snapshot)
	{
		bool isCombat = IsCombatLikeStateType(snapshot.StateType);
		CombatTrainingStateSnapshot? combat = snapshot.CachedCombatState;
		bool useCombat = isCombat && combat?.Player != null;
		PlayerState ps = new PlayerState
		{
			Hp = player.Creature.CurrentHp,
			MaxHp = player.Creature.MaxHp,
			Block = player.Creature.Block,
			Gold = player.Gold,
			Energy = useCombat ? combat!.Player!.Energy : 0,
			MaxEnergy = useCombat ? combat!.Player!.MaxEnergy : player.MaxEnergy,
			DrawPileCount = useCombat ? combat!.Piles?.Draw ?? 0 : 0,
			DiscardPileCount = useCombat ? combat!.Piles?.Discard ?? 0 : 0,
			ExhaustPileCount = useCombat ? combat!.Piles?.Exhaust ?? 0 : 0,
			PlayPileCount = useCombat ? combat!.Piles?.Play ?? 0 : 0,
			OpenPotionSlots = player.PotionSlots.Count(static potion => potion == null),
			MaxPotions = player.MaxPotionCount
		};
		foreach (CardModel card in player.Deck.Cards)
		{
			ps.Deck.Add(BuildCardInfo(card, ps.Deck.Count));
		}
		foreach (RelicModel relic in player.Relics)
		{
			ps.Relics.Add(new RelicInfo
			{
				Index = ps.Relics.Count,
				Id = relic.Id.Entry ?? "",
				Name = SafeText(() => relic.Title, relic.Id.Entry ?? "")
			});
		}
		foreach (PotionModel? potion in player.PotionSlots)
		{
			if (potion == null) continue;
			ps.Potions.Add(new PotionInfo
			{
				Index = ps.Potions.Count,
				Id = potion.Id.Entry ?? "",
				Name = SafeText(() => potion.Title, potion.Id.Entry ?? "")
			});
		}
		if (useCombat)
		{
			ps.Stars = combat!.Player!.Stars;
			foreach (CombatTrainingPowerSnapshot power in combat.Player.Powers.Where(static p => !string.IsNullOrEmpty(p.Id) && p.Amount != 0))
			{
				ps.Powers.Add(new Power { Id = power.Id, Amount = power.Amount });
			}
		}
		return ps;
	}

	private static PlayerState BuildCombatPlayerState(Player? runtimePlayer, CombatTrainingStateSnapshot combat)
	{
		PlayerState ps = runtimePlayer == null
			? new PlayerState()
			: BuildPlayerState(runtimePlayer, new FullRunSimulationStateSnapshot
			{
				StateType = DetectCombatStateType(combat),
				CachedCombatState = combat
			});
		if (combat.Player != null)
		{
			ps.Hp = combat.Player.CurrentHp;
			ps.MaxHp = combat.Player.MaxHp;
			ps.Block = combat.Player.Block;
			ps.Energy = combat.Player.Energy;
			ps.MaxEnergy = combat.Player.MaxEnergy;
			ps.Stars = combat.Player.Stars;
			ps.Powers.Clear();
			foreach (CombatTrainingPowerSnapshot power in combat.Player.Powers.Where(static p => !string.IsNullOrEmpty(p.Id) && p.Amount != 0))
			{
				ps.Powers.Add(new Power { Id = power.Id, Amount = power.Amount });
			}
		}
		ps.DrawPileCount = combat.Piles?.Draw ?? ps.DrawPileCount;
		ps.DiscardPileCount = combat.Piles?.Discard ?? ps.DiscardPileCount;
		ps.ExhaustPileCount = combat.Piles?.Exhaust ?? ps.ExhaustPileCount;
		return ps;
	}

	private static CardInfo BuildCardInfo(CardModel card, int index) => new CardInfo
	{
		Index = index,
		Id = card.Id.Entry ?? "",
		Name = SafeText(() => card.Title, card.Id.Entry ?? ""),
		Cost = NormalizeCost(card),
		CardType = card.Type.ToString().ToUpperInvariant(),
		Rarity = card.Rarity.ToString().ToLowerInvariant(),
		IsUpgraded = card.IsUpgraded,
		Upgrades = card.IsUpgraded ? 1 : 0
	};

	private static LegalAction BuildLegalAction(FullRunSimulationLegalAction action) => new LegalAction
	{
		Action = action.Action ?? "",
		Index = action.Index ?? -1,
		CardIndex = action.CardIndex ?? -1,
		TargetId = action.TargetId.HasValue ? (int)action.TargetId.Value : -1,
		Col = action.Col ?? -1,
		Row = action.Row ?? -1,
		Slot = action.Slot ?? -1,
		Label = action.Label ?? "",
		CardId = action.CardId ?? ""
	};

	private static BattleState BuildBattleState(CombatTrainingStateSnapshot? combat, Player? runtimePlayer = null)
	{
		combat ??= SafeBuildCombatSnapshot();
		BattleState bs = new BattleState
		{
			RoundNumber = combat.RoundNumber,
			TurnSide = combat.CurrentSide.ToString().ToLowerInvariant(),
			IsPlayPhase = combat.IsPlayPhase,
			CanEndTurn = combat.CanEndTurn,
			Energy = combat.Player?.Energy ?? 0,
			MaxEnergy = combat.Player?.MaxEnergy ?? 0
		};
		if (combat.Player != null || runtimePlayer != null)
		{
			bs.Player = BuildCombatPlayerState(runtimePlayer, combat);
		}
		foreach (CombatTrainingHandCardSnapshot card in combat.Hand)
		{
			bs.Hand.Add(BuildHandCard(card));
		}
		foreach (CombatTrainingCreatureSnapshot enemy in combat.Enemies)
		{
			bs.Enemies.Add(BuildEnemy(enemy));
		}
		AddPileCards(bs.DrawPileCards, combat.Piles?.DrawCardIds);
		AddPileCards(bs.DiscardPileCards, combat.Piles?.DiscardCardIds);
		AddPileCards(bs.ExhaustPileCards, combat.Piles?.ExhaustCardIds);
		return bs;
	}

	private static CombatTrainingStateSnapshot SafeBuildCombatSnapshot()
	{
		try
		{
			return BridgeCombatSnapshotBuilder.BuildStateSnapshot();
		}
		catch
		{
			return new CombatTrainingStateSnapshot
			{
				IsCombatActive = CombatManager.Instance.IsInProgress
			};
		}
	}

	private static HandCard BuildHandCard(CombatTrainingHandCardSnapshot card)
	{
		HandCard hc = new HandCard
		{
			Index = card.HandIndex,
			Id = card.Id ?? "",
			Name = string.IsNullOrEmpty(card.Title) ? card.Id ?? "" : card.Title,
			Cost = card.EnergyCost,
			CardType = card.CardType ?? "",
			Rarity = "",
			TargetType = MapTargetTypeString(card.TargetType),
			IsUpgraded = card.IsUpgraded,
			CanPlay = card.CanPlay,
			RequiresTarget = card.RequiresTarget,
			Description = card.Description ?? "",
			PreviewBlock = card.PreviewBlock
		};
		foreach (uint targetId in card.ValidTargetIds)
		{
			hc.ValidTargetIds.Add((int)targetId);
		}
		foreach (KeyValuePair<uint, int> item in card.PreviewDamagePerTarget)
		{
			hc.PreviewDamagePerTarget[(int)item.Key] = item.Value;
		}
		foreach (string keyword in card.Keywords.Where(static value => !string.IsNullOrWhiteSpace(value)))
		{
			hc.Keywords.Add(keyword);
		}
		return hc;
	}

	private static HandCard BuildSelectableHandCard(CombatTrainingSelectableCardSnapshot card)
	{
		HandCard hc = new HandCard
		{
			Index = card.ChoiceIndex,
			Id = card.Id ?? "",
			Name = string.IsNullOrEmpty(card.Title) ? card.Id ?? "" : card.Title,
			Cost = card.EnergyCost,
			CardType = card.Type.ToString().ToUpperInvariant(),
			TargetType = MapTargetTypeString(card.TargetType),
			IsUpgraded = card.IsUpgraded,
			Description = card.Description ?? "",
			PreviewBlock = card.PreviewBlock
		};
		foreach (string keyword in card.Keywords.Where(static value => !string.IsNullOrWhiteSpace(value)))
		{
			hc.Keywords.Add(keyword);
		}
		return hc;
	}

	private static HandCard BuildCardModelHandCard(CardModel card, int index, PileType pile)
	{
		HandCard hc = new HandCard
		{
			Index = index,
			Id = card.Id.Entry ?? "",
			Name = SafeText(() => card.Title, card.Id.Entry ?? ""),
			Cost = NormalizeCost(card),
			CardType = card.Type.ToString().ToUpperInvariant(),
			Rarity = card.Rarity.ToString().ToLowerInvariant(),
			TargetType = MapTargetTypeString(card.TargetType),
			IsUpgraded = card.IsUpgraded,
			CanPlay = SafeBool(() => card.CanPlay()),
			RequiresTarget = CardRequiresTarget(card),
			Description = CombatTrainingCardDescription.GetDescription(card, pile),
			PreviewBlock = CombatTrainingCardDescription.GetPreviewBlock(card)
		};
		foreach (string keyword in SafeKeywords(card))
		{
			hc.Keywords.Add(keyword);
		}
		return hc;
	}

	private static Enemy BuildEnemy(CombatTrainingCreatureSnapshot enemy)
	{
		Enemy e = new Enemy
		{
			Id = enemy.Id ?? "",
			CombatId = (int)(enemy.CombatId ?? 0),
			Name = string.IsNullOrEmpty(enemy.Name) ? enemy.Id ?? "" : enemy.Name,
			Hp = enemy.CurrentHp,
			MaxHp = enemy.MaxHp,
			Block = enemy.Block,
			IsAlive = enemy.IsAlive,
			IsHittable = enemy.IsHittable,
			IntendsToAttack = enemy.IntendsToAttack,
			NextMoveId = enemy.NextMoveId ?? ""
		};
		foreach (CombatTrainingIntentSnapshot intent in enemy.Intents)
		{
			int repeats = Math.Max(1, intent.Repeats);
			int totalDamage = intent.TotalDamage ?? intent.Damage ?? 0;
			int perHitDamage = intent.Damage ?? (repeats > 1 && totalDamage > 0 ? totalDamage / repeats : totalDamage);
			e.Intents.Add(new Intent
			{
				Type = intent.IntentType ?? "unknown",
				Label = intent.IntentType ?? "unknown",
				Damage = perHitDamage,
				TotalDamage = totalDamage,
				Hits = repeats
			});
		}
		foreach (CombatTrainingPowerSnapshot power in enemy.Powers.Where(static p => !string.IsNullOrEmpty(p.Id) && p.Amount != 0))
		{
			e.Powers.Add(new Power { Id = power.Id, Amount = power.Amount });
		}
		return e;
	}

	private static void AddPileCards(Google.Protobuf.Collections.RepeatedField<string> target, List<string>? cardIds)
	{
		if (cardIds == null) return;
		int count = Math.Min(cardIds.Count, MaxPileCards);
		for (int i = 0; i < count; i++)
		{
			target.Add(cardIds[i] ?? "");
		}
	}

	private static MapState BuildMapState(FullRunSimulationStateSnapshot snapshot)
	{
		MapState ms = new MapState();
		foreach (FullRunSimulationMapOption option in snapshot.MapOptions)
		{
			ms.NextOptions.Add(new MapOption
			{
				Index = option.Index,
				Col = option.Col,
				Row = option.Row,
				PointType = option.PointType ?? "unknown",
				Label = option.PointType ?? "unknown"
			});
		}
		foreach (FullRunSimulationMapNode node in snapshot.MapNodes)
		{
			MapNode mn = new MapNode
			{
				Col = node.Col,
				Row = node.Row,
				Type = node.PointType ?? "unknown"
			};
			foreach ((int childCol, int childRow) in node.Children)
			{
				mn.Children.Add(new MapEdge { Col = childCol, Row = childRow });
			}
			ms.Nodes.Add(mn);
		}
		ms.Boss = new MapNode
		{
			Col = snapshot.BossCol,
			Row = snapshot.BossRow
		};
		return ms;
	}

	private static EventState BuildEventState(EventRoom? eventRoom)
	{
		EventState es = new EventState();
		EventModel? localEvent = eventRoom?.LocalMutableEvent;
		es.EventId = localEvent?.Id.ToString() ?? "";
		IReadOnlyList<MegaCrit.Sts2.Core.Events.EventOption> options =
			localEvent?.CurrentOptions ?? Array.Empty<MegaCrit.Sts2.Core.Events.EventOption>();
		es.InDialogue = localEvent != null && !localEvent.IsFinished && options.Count == 0;
		es.IsFinished = localEvent?.IsFinished ?? false;
		int index = 0;
		foreach (MegaCrit.Sts2.Core.Events.EventOption option in options)
		{
			string title = SafeFormatLocString(option.Title);
			es.Options.Add(new STS2AI.Bridge.EventOption
			{
				Index = index++,
				Text = title,
				Label = title,
				IsLocked = option.IsLocked,
				IsChosen = option.WasChosen,
				IsProceed = option.IsProceed
			});
		}
		if (es.IsFinished && es.Options.Count == 0)
		{
			es.Options.Add(new STS2AI.Bridge.EventOption
			{
				Index = 0,
				Text = "proceed",
				Label = "proceed",
				IsProceed = true
			});
		}
		return es;
	}

	private static RestSiteState BuildRestSiteState()
	{
		RestSiteState rs = new RestSiteState();
		IReadOnlyList<RestSiteOption> options = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
		rs.CanProceed = options.Count == 0;
		int index = 0;
		foreach (RestSiteOption option in options)
		{
			rs.Options.Add(new RestOption
			{
				Index = index++,
				Id = option.OptionId.ToString().ToLowerInvariant(),
				Name = option.OptionId.ToString().ToLowerInvariant(),
				IsEnabled = option.IsEnabled
			});
		}
		return rs;
	}

	private static ShopState BuildShopState(MerchantRoom? room)
	{
		ShopState ss = new ShopState { IsOpen = room?.Inventory != null, CanProceed = true };
		MerchantInventory? inventory = room?.Inventory;
		if (inventory == null) return ss;
		int index = 0;
		foreach (MerchantEntry entry in EnumerateShopEntries(inventory))
		{
			(string category, string itemId, string itemName) = GetShopItemInfo(entry);
			ss.Items.Add(new ShopItem
			{
				Index = index++,
				Category = category,
				Cost = entry.Cost,
				CanAfford = entry.EnoughGold,
				IsStocked = entry.IsStocked,
				OnSale = IsShopEntryOnSale(entry),
				Id = itemId,
				Name = string.IsNullOrWhiteSpace(itemName) ? itemId : itemName
			});
		}
		return ss;
	}

	private static CombatRewardsState BuildCombatRewardsState(FullRunPendingRewardSelectionSnapshot? rewards, Player? player)
	{
		CombatRewardsState crs = new CombatRewardsState { CanProceed = rewards?.CanProceed ?? false };
		int openPotionSlots = player?.PotionSlots.Count(static p => p == null) ?? 0;
		IReadOnlyList<Reward> items = rewards?.Rewards ?? Array.Empty<Reward>();
		int index = 0;
		foreach (Reward reward in items)
		{
			crs.Items.Add(new RewardItem
			{
				Index = index++,
				Type = RewardTypeName(reward),
				Label = SafeRewardLabel(reward),
				Id = RewardId(reward),
				Claimable = IsRewardClaimable(reward, openPotionSlots)
			});
		}
		return crs;
	}

	private static CardRewardState BuildCardRewardState(FullRunPendingCardRewardSnapshot? reward)
	{
		CardRewardState crs = new CardRewardState { CanSkip = reward?.CanSkip ?? false };
		IReadOnlyList<CardCreationResult> cards = reward?.Options ?? Array.Empty<CardCreationResult>();
		foreach (CardCreationResult option in cards)
		{
			crs.Cards.Add(BuildCardModelHandCard(option.Card, crs.Cards.Count, PileType.None));
		}
		return crs;
	}

	private static CardSelectState BuildCardSelectState(CombatTrainingCardSelectionSnapshot? selection)
	{
		CardSelectState cs = new CardSelectState
		{
			ScreenType = NormalizeCardSelectScreenType(selection?.Mode),
			SelectedCount = selection?.SelectedCards.Count ?? 0,
			CanConfirm = selection?.CanConfirm ?? false,
			CanCancel = selection?.Cancelable ?? false,
			Prompt = selection?.PromptText ?? "",
			MinSelect = selection?.MinSelect ?? 0,
			MaxSelect = selection?.MaxSelect ?? 0
		};
		if (selection == null) return cs;
		bool quotaReached = selection.MaxSelect > 0 && selection.SelectedCards.Count >= selection.MaxSelect;
		bool previewShowing = quotaReached && selection.CanConfirm;
		IEnumerable<CombatTrainingSelectableCardSnapshot> visibleCards = previewShowing
			? selection.SelectableCards.Concat(selection.SelectedCards).OrderBy(static card => card.ChoiceIndex)
			: selection.SelectableCards;
		foreach (CombatTrainingSelectableCardSnapshot card in visibleCards)
		{
			cs.Cards.Add(BuildSelectableHandCard(card));
		}
		foreach (CombatTrainingSelectableCardSnapshot card in selection.SelectedCards)
		{
			cs.SelectedCards.Add(BuildSelectableHandCard(card));
		}
		return cs;
	}

	private static RelicSelectState BuildRelicSelectState(FullRunPendingRelicSelectionSnapshot? selection)
	{
		RelicSelectState rs = new RelicSelectState { CanSkip = selection?.CanSkip ?? false };
		IReadOnlyList<RelicModel> relics = selection?.Relics ?? Array.Empty<RelicModel>();
		int index = 0;
		foreach (RelicModel relic in relics)
		{
			rs.Relics.Add(new RelicInfo
			{
				Index = index++,
				Id = relic.Id.Entry ?? "",
				Name = SafeText(() => relic.Title, relic.Id.Entry ?? "")
			});
		}
		return rs;
	}

	private static TreasureState BuildTreasureState()
	{
		IReadOnlyList<RelicModel>? relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		TreasureState ts = new TreasureState { CanProceed = relics == null };
		if (relics == null) return ts;
		int index = 0;
		foreach (RelicModel relic in relics)
		{
			ts.Relics.Add(new RelicInfo
			{
				Index = index++,
				Id = relic.Id.Entry ?? "",
				Name = SafeText(() => relic.Title, relic.Id.Entry ?? "")
			});
		}
		return ts;
	}

	private static void PopulateCombatLegalActions(GameState gs, CombatTrainingStateSnapshot snapshot)
	{
		if (snapshot.IsEpisodeDone) return;
		if (snapshot.IsHandSelectionActive && snapshot.HandSelection != null)
		{
			CombatTrainingHandSelectionSnapshot hs = snapshot.HandSelection;
			if (SelectionActionSemantics.ShouldExposeSelectionActions(hs.SelectedCards.Count, hs.MaxSelect))
			{
				foreach (CombatTrainingHandCardSnapshot card in hs.SelectableCards)
				{
					gs.LegalActions.Add(new LegalAction
					{
						Action = "select_hand_card",
						Index = card.HandIndex,
						CardIndex = card.HandIndex,
						TargetId = -1,
						Col = -1,
						Row = -1,
						Slot = -1,
						CardId = card.Id ?? "",
						Label = string.IsNullOrEmpty(card.Title) ? card.Id ?? "" : card.Title
					});
				}
			}
			if (hs.CanConfirm) gs.LegalActions.Add(NonIndexedAction("confirm_selection", "Confirm"));
			if (hs.Cancelable) gs.LegalActions.Add(NonIndexedAction("cancel_selection", "Cancel"));
			return;
		}
		if (snapshot.IsCardSelectionActive && snapshot.CardSelection != null)
		{
			CombatTrainingCardSelectionSnapshot cs = snapshot.CardSelection;
			if (SelectionActionSemantics.ShouldExposeSelectionActions(cs.SelectedCards.Count, cs.MaxSelect))
			{
				foreach (CombatTrainingSelectableCardSnapshot card in cs.SelectableCards)
				{
					gs.LegalActions.Add(new LegalAction
					{
						Action = "select_card_option",
						Index = card.ChoiceIndex,
						CardIndex = card.ChoiceIndex,
						TargetId = -1,
						Col = -1,
						Row = -1,
						Slot = -1,
						CardId = card.Id ?? "",
						Label = string.IsNullOrEmpty(card.Title) ? card.Id ?? "" : card.Title
					});
				}
			}
			if (cs.CanConfirm) gs.LegalActions.Add(NonIndexedAction("confirm_selection", "Confirm"));
			if (cs.Cancelable) gs.LegalActions.Add(NonIndexedAction("cancel_selection", "Cancel"));
			return;
		}
		foreach (CombatTrainingHandCardSnapshot card in snapshot.Hand)
		{
			if (!card.CanPlay) continue;
			if (card.RequiresTarget)
			{
				foreach (uint targetId in card.ValidTargetIds)
				{
					gs.LegalActions.Add(new LegalAction
					{
						Action = "play_card",
						Index = card.HandIndex,
						CardIndex = card.HandIndex,
						CardId = card.Id ?? "",
						Label = string.IsNullOrEmpty(card.Title) ? card.Id ?? "" : card.Title,
						TargetId = (int)targetId,
						Col = -1,
						Row = -1,
						Slot = -1
					});
				}
			}
			else
			{
				gs.LegalActions.Add(new LegalAction
				{
					Action = "play_card",
					Index = card.HandIndex,
					CardIndex = card.HandIndex,
					CardId = card.Id ?? "",
					Label = string.IsNullOrEmpty(card.Title) ? card.Id ?? "" : card.Title,
					TargetId = -1,
					Col = -1,
					Row = -1,
					Slot = -1
				});
			}
		}
		if (snapshot.CanEndTurn) gs.LegalActions.Add(NonIndexedAction("end_turn", "End Turn"));
	}

	private static LegalAction NonIndexedAction(string action, string label) => new LegalAction
	{
		Action = action,
		Index = -1,
		CardIndex = -1,
		TargetId = -1,
		Col = -1,
		Row = -1,
		Slot = -1,
		Label = label
	};

	private static string DetectCombatStateType(CombatTrainingStateSnapshot snapshot)
	{
		if (snapshot.IsEpisodeDone) return "game_over";
		if (snapshot.IsHandSelectionActive) return "hand_select";
		if (snapshot.IsCardSelectionActive) return "card_select";
		return "monster";
	}

	private static bool IsCombatLikeStateType(string? stateType)
		=> stateType is "monster" or "elite" or "boss" or "hand_select";

	private static string MapTargetTypeString(TargetType targetType) => targetType switch
	{
		TargetType.None => "None",
		TargetType.Self => "Self",
		TargetType.AnyEnemy => "AnyEnemy",
		TargetType.AnyPlayer => "AnyPlayer",
		TargetType.AnyAlly => "AnyAlly",
		TargetType.TargetedNoCreature => "TargetedNoCreature",
		TargetType.AllEnemies => "AllEnemies",
		TargetType.RandomEnemy => "RandomEnemy",
		TargetType.AllAllies => "AllAllies",
		TargetType.Osty => "Osty",
		_ => ""
	};

	private static bool CardRequiresTarget(CardModel card)
		=> card.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly;

	private static int NormalizeCost(CardModel card)
	{
		int cost = card.EnergyCost.GetWithModifiers(CostModifiers.All);
		return card.EnergyCost.CostsX ? cost : Math.Max(0, cost);
	}

	private static IEnumerable<string> SafeKeywords(CardModel card)
	{
		try
		{
			return card.Keywords.Select(static keyword => keyword.ToString()).Where(static value => !string.IsNullOrWhiteSpace(value)).ToList();
		}
		catch
		{
			return Array.Empty<string>();
		}
	}

	private static string NormalizeCardSelectScreenType(string? mode) => mode switch
	{
		"DeckUpgrade" => "UpgradeSelect",
		"DeckTransform" => "Transform",
		"DeckGeneric" => "DeckGeneric",
		"SimpleGrid" => "SimpleSelect",
		"RewardSimpleGrid" => "SimpleSelect",
		"ChooseCard" => "SimpleSelect",
		null or "" => "card_select",
		_ => mode
	};

	private static (string Category, string Id, string Name) GetShopItemInfo(MerchantEntry entry) => entry switch
	{
		MerchantCardEntry cardEntry when cardEntry.CreationResult != null => (
			"card",
			cardEntry.CreationResult.Card.Id.Entry ?? "",
			SafeText(() => cardEntry.CreationResult.Card.Title, cardEntry.CreationResult.Card.Id.Entry ?? "")),
		MerchantRelicEntry relicEntry => (
			"relic",
			relicEntry.Model?.Id.Entry ?? "",
			SafeText(() => relicEntry.Model?.Title, relicEntry.Model?.Id.Entry ?? "")),
		MerchantPotionEntry potionEntry => (
			"potion",
			potionEntry.Model?.Id.Entry ?? "",
			SafeText(() => potionEntry.Model?.Title, potionEntry.Model?.Id.Entry ?? "")),
		MerchantCardRemovalEntry => ("remove_card", "remove_card", SafeText(() => new LocString("merchant_room", "MERCHANT.cardRemovalService.title"), "Card Removal")),
		_ => ("unknown", "", entry.GetType().Name)
	};

	private static IEnumerable<MerchantEntry> EnumerateShopEntries(MerchantInventory inventory)
	{
		foreach (MerchantCardEntry entry in inventory.CharacterCardEntries) yield return entry;
		foreach (MerchantCardEntry entry in inventory.ColorlessCardEntries) yield return entry;
		foreach (MerchantRelicEntry entry in inventory.RelicEntries) yield return entry;
		foreach (MerchantPotionEntry entry in inventory.PotionEntries) yield return entry;
		if (inventory.CardRemovalEntry != null) yield return inventory.CardRemovalEntry;
	}

	private static bool IsShopEntryOnSale(MerchantEntry entry)
	{
		try
		{
			return entry is MerchantCardEntry cardEntry && cardEntry.IsOnSale;
		}
		catch
		{
			return false;
		}
	}

	private static string RewardTypeName(Reward reward) => reward switch
	{
		GoldReward => "gold",
		PotionReward => "potion",
		RelicReward => "relic",
		CardReward => "card",
		CardRemovalReward => "remove_card",
		SpecialCardReward => "special_card",
		_ => "unknown"
	};

	private static string RewardId(Reward reward) => reward switch
	{
		PotionReward potionReward => potionReward.Potion?.Id.Entry ?? "",
		RelicReward relicReward => FullRunUpstreamCompat.GetRewardRelic(relicReward)?.Id.Entry ?? "",
		CardReward cardReward => FullRunUpstreamCompat.GetCardRewardOptions(cardReward).FirstOrDefault()?.Card.Id.Entry ?? "",
		_ => ""
	};

	private static bool IsRewardClaimable(Reward reward, int openPotionSlots)
		=> reward is not PotionReward || openPotionSlots > 0;

	private static string SafeRewardLabel(Reward reward)
	{
		try
		{
			return reward.Description?.GetFormattedText() ?? "";
		}
		catch
		{
			return "";
		}
	}

	private static string SafeFormatLocString(LocString? locString)
	{
		if (locString == null) return "";
		try
		{
			return locString.GetFormattedText() ?? "";
		}
		catch
		{
			return "";
		}
	}

	private static string SafeText(Func<object?> read, string fallback = "")
	{
		try
		{
			return read()?.ToString() ?? fallback;
		}
		catch
		{
			return fallback;
		}
	}

	private static bool SafeBool(Func<bool> read, bool fallback = false)
	{
		try
		{
			return read();
		}
		catch
		{
			return fallback;
		}
	}
}
