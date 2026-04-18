using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Google.Protobuf;
using MegaCrit.Sts2.Core.CardSelection;
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
using STS2AI.Bridge;

namespace HeadlessSim;

/// <summary>
/// 游戏运行时对象 → Protobuf GameState 消息的映射层。
///
/// 职责：从 <see cref="FullRunSimulationStateSnapshot"/> 等游戏对象中提取数据，
/// 填充 protoc 自动生成的 <see cref="GameState"/> proto message，再序列化为 bytes。
/// 等价于 <see cref="BinaryProtocol"/>.BuildStatePayload() 的 protobuf 版本。
///
/// 帧格式不变: [4字节长度][status][opcode][payload]
/// 差异: state payload 用 protobuf 序列化，不再需要符号表和静态缓存。
/// 请求解析复用 BinaryProtocol（请求格式完全一致）。
/// </summary>
internal static class ProtoStateBuilder
{
	private const ushort ProtocolVersion = 1;
	internal const string ProtoSchemaId = "sts2-proto-v1";
	private const int MaxPileCards = 50;

	private static readonly string BuildGitSha = ResolveBuildGitSha();

	// ================================================================
	// Handshake
	// ================================================================

	public static byte[] BuildHandshakeResponse()
	{
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)BinaryStatus.Ok);
		writer.Write((byte)BinaryOpcode.Handshake);
		writer.Write(ProtocolVersion);
		BinaryProtocol.WriteString(writer, BuildGitSha);
		BinaryProtocol.WriteString(writer, ProtoSchemaId);
		return stream.ToArray();
	}

	// ================================================================
	// State responses — payload = protobuf GameState bytes
	// ================================================================

	public static byte[] BuildStateResponse(BinaryOpcode opcode, FullRunSimulationStateSnapshot snapshot)
	{
		byte[] statePayload = BuildProtoStatePayload(snapshot);
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)BinaryStatus.Ok);
		writer.Write((byte)opcode);
		writer.Write(statePayload);
		FullRunSimulationDiagnostics.Increment("proto.state_bytes", statePayload.Length);
		return stream.ToArray();
	}

	public static byte[] BuildStepResponse(FullRunSimulationStepResult result, FullRunSimulationStateSnapshot snapshot)
	{
		BinaryStatus status = result.Accepted ? BinaryStatus.Ok : BinaryStatus.RejectedAction;
		byte[] statePayload = BuildProtoStatePayload(snapshot);
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)status);
		writer.Write((byte)BinaryOpcode.Step);
		writer.Write((byte)(result.Accepted ? 1 : 0));
		BinaryProtocol.WriteOptionalString(writer, result.Error);
		writer.Write(statePayload);
		FullRunSimulationDiagnostics.Increment("proto.state_bytes", statePayload.Length);
		return stream.ToArray();
	}

	public static byte[] BuildBatchStepResponse(FullRunSimulationBatchStepResult result, FullRunSimulationStateSnapshot snapshot)
	{
		BinaryStatus status = result.Accepted ? BinaryStatus.Ok : BinaryStatus.RejectedAction;
		byte[] statePayload = BuildProtoStatePayload(snapshot);
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)status);
		writer.Write((byte)BinaryOpcode.BatchStep);
		writer.Write((byte)(result.Accepted ? 1 : 0));
		writer.Write((ushort)Math.Max(0, result.StepsExecuted));
		BinaryProtocol.WriteOptionalString(writer, result.Error);
		writer.Write(statePayload);
		FullRunSimulationDiagnostics.Increment("proto.state_bytes", statePayload.Length);
		return stream.ToArray();
	}

	// ================================================================
	// Non-state responses — 格式和 BinaryProtocol 完全一致，直接委托
	// ================================================================

	public static byte[] BuildErrorResponse(BinaryOpcode opcode, BinaryStatus status, string errorCode, string error)
		=> BinaryProtocol.BuildErrorResponse(opcode, status, errorCode, error);

	public static byte[] BuildSaveStateResponse(string stateId, int cacheSize)
		=> BinaryProtocol.BuildSaveStateResponse(stateId, cacheSize);

	public static byte[] BuildExportStateResponse(string path, int cacheSize)
		=> BinaryProtocol.BuildExportStateResponse(path, cacheSize);

	public static byte[] BuildDeleteStateResponse(bool deleted, int cacheSize)
		=> BinaryProtocol.BuildDeleteStateResponse(deleted, cacheSize);

	public static byte[] BuildPerfStatsResponse(Dictionary<string, object?> payload)
		=> BinaryProtocol.BuildPerfStatsResponse(payload);

	public static byte[] BuildResetPerfStatsResponse()
		=> BinaryProtocol.BuildResetPerfStatsResponse();

	public static byte[] BuildSearchCombatMctsResponse(CombatMctsResult result)
		=> BinaryProtocol.BuildSearchCombatMctsResponse(result);

	// ================================================================
	// Core: snapshot → protobuf GameState → byte[]
	// ================================================================

	internal static byte[] BuildProtoStatePayload(FullRunSimulationStateSnapshot snapshot)
	{
		GameState gs = new GameState();
		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		Player? player = TryResolveLocalPlayer(runState);
		FullRunSimulationChoiceBridge bridge = FullRunSimulationChoiceBridge.Instance;

		// Top-level fields
		gs.StateType = snapshot.StateType ?? "other";
		gs.Terminal = snapshot.IsTerminal;
		gs.RunOutcome = snapshot.RunOutcome ?? "";
		gs.EncounterId = "";
		gs.Run = new RunInfo
		{
			Act = Math.Clamp(snapshot.CurrentActIndex + 1, 0, 255),
			Floor = Math.Clamp(snapshot.TotalFloor, 0, 255)
		};

		// Player
		if (player != null)
		{
			gs.Player = BuildPlayerState(player, snapshot);
		}

		// Legal actions
		foreach (FullRunSimulationLegalAction la in snapshot.LegalActions)
		{
			gs.LegalActions.Add(BuildLegalAction(la));
		}

		// State-specific data
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
				gs.CombatRewards = BuildCombatRewardsState(bridge, runState, player);
				break;
			case "card_reward":
				gs.CardReward = BuildCardRewardState(bridge);
				break;
			case "card_select":
				gs.CardSelect = BuildCardSelectState(bridge);
				break;
			case "relic_select":
				gs.RelicSelect = BuildRelicSelectState(bridge);
				break;
			case "hand_select":
			case "monster":
			case "elite":
			case "boss":
				gs.Battle = BuildBattleState(snapshot.CachedCombatState);
				break;
		}

		return gs.ToByteArray();
	}

	// ================================================================
	// Player
	// ================================================================

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
			OpenPotionSlots = player.PotionSlots.Count(static p => p == null),
			MaxPotions = player.MaxPotionCount
		};

		// Deck
		int deckIndex = 0;
		foreach (CardModel card in player.Deck.Cards)
		{
			ps.Deck.Add(new CardInfo
			{
				Index = deckIndex++,
				Id = card.Id.Entry ?? "",
				Name = card.Id.Entry ?? "",
				Cost = card.EnergyCost.GetWithModifiers(CostModifiers.All),
				CardType = card.Type.ToString().ToUpperInvariant(),
				Rarity = card.Rarity.ToString().ToLowerInvariant(),
				IsUpgraded = card.IsUpgraded,
				Upgrades = card.IsUpgraded ? 1 : 0
			});
		}

		// Relics
		int relicIndex = 0;
		foreach (RelicModel relic in player.Relics)
		{
			ps.Relics.Add(new RelicInfo
			{
				Index = relicIndex++,
				Id = relic.Id.Entry ?? "",
				Name = relic.Id.Entry ?? ""
			});
		}

		// Potions
		int potionIndex = 0;
		foreach (PotionModel potion in player.PotionSlots.Where(static p => p != null).OfType<PotionModel>())
		{
			ps.Potions.Add(new PotionInfo
			{
				Index = potionIndex++,
				Id = potion.Id.Entry ?? "",
				Name = potion.Id.Entry ?? ""
			});
		}

		return ps;
	}

	// ================================================================
	// Legal Actions
	// ================================================================

	private static LegalAction BuildLegalAction(FullRunSimulationLegalAction la)
	{
		return new LegalAction
		{
			Action = la.Action ?? "other",
			Index = la.Index ?? -1,
			CardIndex = la.CardIndex ?? -1,
			TargetId = la.TargetId.HasValue ? (int)la.TargetId.Value : -1,
			Col = la.Col ?? -1,
			Row = la.Row ?? -1,
			Slot = la.Slot ?? -1,
			Label = la.Label ?? la.Action ?? "",
			CardId = la.CardId ?? ""
		};
	}

	// ================================================================
	// Battle (combat)
	// ================================================================

	private static BattleState BuildBattleState(CombatTrainingStateSnapshot? combat)
	{
		combat ??= CombatTrainingEnvService.BuildStateSnapshot() ?? new CombatTrainingStateSnapshot();
		BattleState bs = new BattleState
		{
			RoundNumber = combat.RoundNumber,
			TurnSide = combat.CurrentSide.ToString().ToLowerInvariant(),
			IsPlayPhase = combat.IsPlayPhase,
			CanEndTurn = combat.CanEndTurn,
			Energy = combat.Player?.Energy ?? 0,
			MaxEnergy = combat.Player?.MaxEnergy ?? 0
		};

		// Battle player
		if (combat.Player != null)
		{
			CombatTrainingPlayerSnapshot cp = combat.Player;
			PlayerState battlePlayer = new PlayerState
			{
				Hp = cp.CurrentHp,
				MaxHp = cp.MaxHp,
				Block = cp.Block,
				Energy = cp.Energy,
				MaxEnergy = cp.MaxEnergy,
				Stars = cp.Stars
			};
			// Player powers
			if (cp.Powers != null)
			{
				foreach (CombatTrainingPowerSnapshot power in cp.Powers.Where(static p => p?.Id != null && p.Amount != 0))
				{
					battlePlayer.Powers.Add(new Power { Id = power.Id ?? "", Amount = power.Amount });
				}
			}
			bs.Player = battlePlayer;
		}

		// Hand
		foreach (CombatTrainingHandCardSnapshot card in combat.Hand)
		{
			HandCard hc = new HandCard
			{
				Index = card.HandIndex,
				Id = card.Id ?? "",
				Name = card.Id ?? "",
				Cost = card.EnergyCost,
				CardType = card.CardType ?? "",
				Rarity = "",
				TargetType = MapTargetTypeString(card.TargetType),
				IsUpgraded = card.IsUpgraded,
				CanPlay = card.CanPlay,
				RequiresTarget = card.RequiresTarget
			};
			foreach (uint tid in card.ValidTargetIds)
			{
				hc.ValidTargetIds.Add((int)tid);
			}
			bs.Hand.Add(hc);
		}

		// Enemies
		foreach (CombatTrainingCreatureSnapshot enemy in combat.Enemies)
		{
			Enemy e = new Enemy
			{
				Id = enemy.Id ?? "",
				CombatId = (int)(enemy.CombatId ?? 0),
				Name = enemy.Id ?? "",
				Hp = enemy.CurrentHp,
				MaxHp = enemy.MaxHp,
				Block = enemy.Block,
				IsAlive = enemy.IsAlive,
				IsHittable = enemy.IsHittable,
				IntendsToAttack = enemy.IntendsToAttack,
				NextMoveId = enemy.NextMoveId ?? ""
			};

			// Intents
			if (enemy.Intents != null)
			{
				foreach (CombatTrainingIntentSnapshot intent in enemy.Intents)
				{
					int repeats = Math.Max(1, intent?.Repeats ?? 1);
					int totalDamage = intent?.TotalDamage ?? intent?.Damage ?? 0;
					int perHitDamage = intent?.Damage ?? (repeats > 1 && totalDamage > 0 ? totalDamage / repeats : totalDamage);
					e.Intents.Add(new Intent
					{
						Type = intent?.IntentType ?? "unknown",
						Label = intent?.IntentType ?? "unknown",
						Damage = perHitDamage,
						TotalDamage = totalDamage,
						Hits = repeats
					});
				}
			}

			// Powers
			if (enemy.Powers != null)
			{
				foreach (CombatTrainingPowerSnapshot power in enemy.Powers.Where(static p => p?.Id != null && p.Id.Length > 0 && p.Amount != 0))
				{
					e.Powers.Add(new Power { Id = power.Id!, Amount = power.Amount });
				}
			}

			bs.Enemies.Add(e);
		}

		// Pile card ID lists
		AddPileCards(bs.DrawPileCards, combat.Piles?.DrawCardIds);
		AddPileCards(bs.DiscardPileCards, combat.Piles?.DiscardCardIds);
		AddPileCards(bs.ExhaustPileCards, combat.Piles?.ExhaustCardIds);

		return bs;
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

	// ================================================================
	// Map
	// ================================================================

	private static MapState BuildMapState(FullRunSimulationStateSnapshot snapshot)
	{
		MapState ms = new MapState();

		foreach (FullRunSimulationMapOption opt in snapshot.MapOptions)
		{
			ms.NextOptions.Add(new MapOption
			{
				Index = opt.Index,
				Col = opt.Col,
				Row = opt.Row,
				PointType = opt.PointType ?? "unknown",
				Label = opt.PointType ?? "unknown"
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

	// ================================================================
	// Event
	// ================================================================

	private static EventState BuildEventState(EventRoom? eventRoom)
	{
		EventState es = new EventState();
		EventModel? localEvent = eventRoom?.LocalMutableEvent;
		es.EventId = localEvent?.Id.ToString() ?? "";
		IReadOnlyList<MegaCrit.Sts2.Core.Events.EventOption> options = localEvent?.CurrentOptions ?? Array.Empty<MegaCrit.Sts2.Core.Events.EventOption>();
		es.InDialogue = localEvent != null && !localEvent.IsFinished && options.Count == 0;
		es.IsFinished = localEvent?.IsFinished ?? false;

		int i = 0;
		foreach (MegaCrit.Sts2.Core.Events.EventOption opt in options)
		{
			es.Options.Add(new STS2AI.Bridge.EventOption
			{
				Index = i++,
				Text = SafeFormatLocString(opt.Title),
				Label = SafeFormatLocString(opt.Title),
				IsLocked = opt.IsLocked,
				IsChosen = opt.WasChosen,
				IsProceed = opt.IsProceed
			});
		}

		return es;
	}

	// ================================================================
	// Rest Site
	// ================================================================

	private static RestSiteState BuildRestSiteState()
	{
		RestSiteState rs = new RestSiteState { CanProceed = true };
		IReadOnlyList<RestSiteOption> options = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
		int i = 0;
		foreach (RestSiteOption opt in options)
		{
			rs.Options.Add(new RestOption
			{
				Index = i++,
				Id = opt.OptionId.ToString().ToLowerInvariant(),
				Name = opt.OptionId.ToString().ToLowerInvariant(),
				IsEnabled = opt.IsEnabled
			});
		}
		return rs;
	}

	// ================================================================
	// Shop
	// ================================================================

	private static ShopState BuildShopState(MerchantRoom? merchantRoom)
	{
		ShopState ss = new ShopState { IsOpen = true, CanProceed = true };
		if (merchantRoom?.Inventory == null) return ss;

		int i = 0;
		foreach (MerchantEntry entry in EnumerateShopEntries(merchantRoom.Inventory))
		{
			string category = ShopCategory(entry);
			string itemId = "";
			bool onSale = false;
			switch (entry)
			{
				case MerchantCardEntry cardEntry:
					itemId = cardEntry.CreationResult?.Card.Id.Entry ?? "";
					onSale = cardEntry.IsOnSale;
					break;
				case MerchantRelicEntry relicEntry:
					itemId = relicEntry.Model?.Id.Entry ?? "";
					break;
				case MerchantPotionEntry potionEntry:
					itemId = potionEntry.Model?.Id.Entry ?? "";
					break;
			}

			ss.Items.Add(new ShopItem
			{
				Index = i++,
				Category = category,
				Cost = entry.Cost,
				CanAfford = entry.EnoughGold,
				IsStocked = entry.IsStocked,
				OnSale = onSale,
				Id = itemId,
				Name = itemId.Length > 0 ? itemId : category
			});
		}
		return ss;
	}

	// ================================================================
	// Combat Rewards
	// ================================================================

	private static CombatRewardsState BuildCombatRewardsState(
		FullRunSimulationChoiceBridge bridge, RunState? runState, Player? player)
	{
		FullRunPendingRewardSelectionSnapshot? rewards = bridge.BuildRewardSelectionSnapshot();
		CombatRewardsState crs = new CombatRewardsState
		{
			CanProceed = rewards?.CanProceed ?? false
		};

		IReadOnlyList<Reward> items = rewards?.Rewards ?? Array.Empty<Reward>();
		int openPotionSlots = player?.PotionSlots.Count(static p => p == null) ?? 0;
		int i = 0;
		foreach (Reward reward in items)
		{
			bool claimable = IsRewardClaimable(reward, openPotionSlots);
			crs.Items.Add(new RewardItem
			{
				Index = i++,
				Type = RewardTypeName(reward),
				Label = SafeGetRewardLabel(reward),
				Id = RewardId(reward),
				Claimable = claimable
			});
		}
		return crs;
	}

	// ================================================================
	// Card Reward
	// ================================================================

	private static CardRewardState BuildCardRewardState(FullRunSimulationChoiceBridge bridge)
	{
		FullRunPendingCardRewardSnapshot? reward = bridge.BuildCardRewardSelectionSnapshot();
		CardRewardState crs = new CardRewardState { CanSkip = reward?.CanSkip ?? false };
		IReadOnlyList<CardCreationResult> cards = reward?.Options ?? Array.Empty<CardCreationResult>();
		foreach (CardCreationResult option in cards)
		{
			CardModel card = option.Card;
			crs.Cards.Add(new HandCard
			{
				Index = crs.Cards.Count,
				Id = card.Id.Entry ?? "",
				Name = card.Id.Entry ?? "",
				Cost = card.EnergyCost.GetWithModifiers(CostModifiers.All),
				CardType = card.Type.ToString().ToUpperInvariant(),
				Rarity = card.Rarity.ToString().ToLowerInvariant(),
				TargetType = MapTargetTypeString(card.TargetType),
				IsUpgraded = card.IsUpgraded,
			});
		}
		return crs;
	}

	// ================================================================
	// Card Select
	// ================================================================

	private static CardSelectState BuildCardSelectState(FullRunSimulationChoiceBridge bridge)
	{
		CombatTrainingCardSelectionSnapshot? selection = bridge.BuildCardSelectionSnapshot(null);
		List<CombatTrainingSelectableCardSnapshot> selectableCards = selection?.SelectableCards ?? new();
		List<CombatTrainingSelectableCardSnapshot> selectedCards = selection?.SelectedCards ?? new();
		CardSelectState cs = new CardSelectState
		{
			ScreenType = selection?.Mode ?? "card_select",
			SelectedCount = selectedCards.Count,
			CanConfirm = selection?.CanConfirm ?? false,
			CanCancel = selection?.Cancelable ?? false
		};
		foreach (CombatTrainingSelectableCardSnapshot card in selectableCards)
		{
			cs.Cards.Add(new HandCard
			{
				Index = cs.Cards.Count,
				Id = card.Id ?? "",
				Name = card.Id ?? "",
				Cost = card.EnergyCost,
				CardType = card.Type.ToString().ToUpperInvariant(),
				TargetType = MapTargetTypeString(card.TargetType),
				IsUpgraded = card.IsUpgraded,
			});
		}
		foreach (CombatTrainingSelectableCardSnapshot card in selectedCards)
		{
			cs.SelectedCards.Add(new HandCard
			{
				Index = cs.SelectedCards.Count,
				Id = card.Id ?? "",
				Name = card.Id ?? "",
				Cost = card.EnergyCost,
				CardType = card.Type.ToString().ToUpperInvariant(),
				TargetType = MapTargetTypeString(card.TargetType),
				IsUpgraded = card.IsUpgraded,
			});
		}
		return cs;
	}

	// ================================================================
	// Relic Select
	// ================================================================

	private static RelicSelectState BuildRelicSelectState(FullRunSimulationChoiceBridge bridge)
	{
		FullRunPendingRelicSelectionSnapshot? relics = bridge.BuildRelicSelectionSnapshot();
		RelicSelectState rs = new RelicSelectState { CanSkip = relics?.CanSkip ?? false };
		IReadOnlyList<RelicModel> items = relics?.Relics ?? Array.Empty<RelicModel>();
		int i = 0;
		foreach (RelicModel relic in items)
		{
			rs.Relics.Add(new RelicInfo
			{
				Index = i++,
				Id = relic.Id.Entry ?? "",
				Name = relic.Id.Entry ?? ""
			});
		}
		return rs;
	}

	// ================================================================
	// Treasure
	// ================================================================

	private static TreasureState BuildTreasureState()
	{
		IReadOnlyList<RelicModel>? relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		TreasureState ts = new TreasureState { CanProceed = relics == null };
		if (relics == null) return ts;
		int i = 0;
		foreach (RelicModel relic in relics)
		{
			ts.Relics.Add(new RelicInfo
			{
				Index = i++,
				Id = relic.Id.Entry ?? "",
				Name = relic.Id.Entry ?? ""
			});
		}
		return ts;
	}

	// ================================================================
	// Helpers
	// ================================================================

	private static bool IsCombatLikeStateType(string? stateType)
		=> stateType is "monster" or "elite" or "boss" or "hand_select";

	private static string MapTargetTypeString(TargetType targetType)
	{
		return targetType switch
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
	}

	private static string ShopCategory(MerchantEntry entry)
	{
		return entry switch
		{
			MerchantCardEntry => "card",
			MerchantRelicEntry => "relic",
			MerchantPotionEntry => "potion",
			MerchantCardRemovalEntry => "remove_card",
			_ => "unknown"
		};
	}

	private static string RewardTypeName(Reward reward)
	{
		return reward switch
		{
			GoldReward => "gold",
			PotionReward => "potion",
			RelicReward => "relic",
			CardReward => "card",
			CardRemovalReward => "remove_card",
			SpecialCardReward => "special_card",
			_ => "unknown"
		};
	}

	private static string RewardId(Reward reward)
	{
		return reward switch
		{
			PotionReward pr => pr.Potion?.Id.Entry ?? "",
			RelicReward rr => FullRunUpstreamCompat.GetRewardRelic(rr)?.Id.Entry ?? "",
			CardReward cr => FullRunUpstreamCompat.GetCardRewardOptions(cr).FirstOrDefault()?.Card.Id.Entry ?? "",
			_ => ""
		};
	}

	private static bool IsRewardClaimable(Reward reward, int openPotionSlots)
	{
		if (reward is PotionReward && openPotionSlots <= 0) return false;
		return true;
	}

	private static string SafeGetRewardLabel(Reward reward)
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
		try { return locString.GetFormattedText() ?? ""; }
		catch { return ""; }
	}

	private static IEnumerable<MerchantEntry> EnumerateShopEntries(MerchantInventory inventory)
	{
		foreach (MerchantCardEntry e in inventory.CharacterCardEntries) yield return e;
		foreach (MerchantCardEntry e in inventory.ColorlessCardEntries) yield return e;
		foreach (MerchantRelicEntry e in inventory.RelicEntries) yield return e;
		foreach (MerchantPotionEntry e in inventory.PotionEntries) yield return e;
		if (inventory.CardRemovalEntry != null) yield return inventory.CardRemovalEntry;
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

	private static string ResolveBuildGitSha()
	{
		try
		{
			using System.Diagnostics.Process process = new();
			process.StartInfo.FileName = "git";
			process.StartInfo.Arguments = "rev-parse --short=12 HEAD";
			process.StartInfo.WorkingDirectory = AppContext.BaseDirectory;
			process.StartInfo.RedirectStandardOutput = true;
			process.StartInfo.RedirectStandardError = true;
			process.StartInfo.CreateNoWindow = true;
			process.StartInfo.UseShellExecute = false;
			process.Start();
			string output = process.StandardOutput.ReadToEnd().Trim();
			process.WaitForExit(2000);
			if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output)) return output;
		}
		catch { }
		return "UNKNOWN";
	}
}
