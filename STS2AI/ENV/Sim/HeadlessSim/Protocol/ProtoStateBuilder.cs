using System;
using System.Collections.Generic;
using System.Linq;
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
using STS2AI.Bridge.Runtime;

namespace HeadlessSim;

/// <summary>
/// 游戏运行时对象 → Protobuf GameState 消息的映射层。
///
/// 职责：从 <see cref="FullRunSimulationStateSnapshot"/> 等游戏对象中提取数据，
/// 填充 protoc 自动生成的 <see cref="GameState"/> / bridge envelope message。
///
/// 2026-04-21 起，proto pipe 外层也统一使用 protobuf envelope，
/// 不再手写 `[status][opcode][payload]`。
/// </summary>
internal static class ProtoStateBuilder
{
	private const uint ProtocolVersion = 1;
	internal const string ProtoSchemaId = "sts2-proto-v1";
	private const int MaxPileCards = 50;

	private static readonly string BuildGitSha = ResolveBuildGitSha();

	// ================================================================
	// Handshake
	// ================================================================

	public static byte[] BuildHandshakeResponse()
	{
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.Handshake,
			Status = BridgeStatus.Ok,
			Handshake = new BridgeHandshake
			{
				ProtocolVersion = ProtocolVersion,
				BuildGitSha = BuildGitSha,
				SchemaId = ProtoSchemaId,
			},
		}.ToByteArray();
	}

	// ================================================================
	// State / result responses
	// ================================================================

	public static byte[] BuildStateResponse(BridgeMethod method, FullRunSimulationStateSnapshot snapshot)
	{
		GameState state = BuildStateMessage(snapshot);
		FullRunSimulationDiagnostics.Increment("proto.state_bytes", state.CalculateSize());
		return new BridgeResponseEnvelope
		{
			Method = method,
			Status = BridgeStatus.Ok,
			State = new BridgeStatePayload { State = state },
		}.ToByteArray();
	}

	public static byte[] BuildActResponse(BridgeMethod method, FullRunSimulationStepResult result, FullRunSimulationStateSnapshot snapshot)
	{
		GameState state = BuildStateMessage(snapshot);
		FullRunSimulationDiagnostics.Increment("proto.state_bytes", state.CalculateSize());
		return new BridgeResponseEnvelope
		{
			Method = method,
			Status = result.Accepted ? BridgeStatus.Ok : BridgeStatus.RejectedAction,
			Act = new BridgeActPayload
			{
				Accepted = result.Accepted,
				Error = result.Error ?? "",
				State = state,
			},
		}.ToByteArray();
	}

	public static byte[] BuildBatchActResponse(FullRunSimulationBatchStepResult result, FullRunSimulationStateSnapshot snapshot)
	{
		GameState state = BuildStateMessage(snapshot);
		FullRunSimulationDiagnostics.Increment("proto.state_bytes", state.CalculateSize());
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.BatchAct,
			Status = result.Accepted ? BridgeStatus.Ok : BridgeStatus.RejectedAction,
			BatchAct = new BridgeBatchActPayload
			{
				Accepted = result.Accepted,
				StepsExecuted = Math.Max(0, result.StepsExecuted),
				Error = result.Error ?? "",
				State = state,
			},
		}.ToByteArray();
	}

	// ================================================================
	// Auxiliary responses
	// ================================================================

	public static byte[] BuildErrorResponse(BridgeMethod method, BridgeStatus status, string errorCode, string error)
	{
		return new BridgeResponseEnvelope
		{
			Method = method,
			Status = status,
			Error = new BridgeError
			{
				ErrorCode = errorCode ?? "",
				ErrorMessage = error ?? "",
			},
		}.ToByteArray();
	}

	public static byte[] BuildSaveStateResponse(BridgeMethod method, string stateId, int cacheSize)
	{
		return new BridgeResponseEnvelope
		{
			Method = method,
			Status = BridgeStatus.Ok,
			SaveState = new BridgeSaveStateResult
			{
				StateId = stateId ?? "",
				CacheSize = cacheSize,
			},
		}.ToByteArray();
	}

	public static byte[] BuildExportStateResponse(string path, int cacheSize)
	{
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.ExportState,
			Status = BridgeStatus.Ok,
			ExportState = new BridgeExportStateResult
			{
				Path = path ?? "",
				CacheSize = cacheSize,
			},
		}.ToByteArray();
	}

	public static byte[] BuildDeleteStateResponse(bool deleted, int cacheSize)
	{
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.DeleteState,
			Status = BridgeStatus.Ok,
			DeleteState = new BridgeDeleteStateResult
			{
				Deleted = deleted,
				CacheSize = cacheSize,
			},
		}.ToByteArray();
	}

	public static byte[] BuildPerfStatsResponse(Dictionary<string, object?> payload)
	{
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.PerfStats,
			Status = BridgeStatus.Ok,
			PerfStats = new BridgePerfStatsResult
			{
				JsonPayload = JsonSerializer.Serialize(payload),
			},
		}.ToByteArray();
	}

	public static byte[] BuildResetPerfStatsResponse()
	{
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.ResetPerfStats,
			Status = BridgeStatus.Ok,
			ResetPerfStats = new BridgeResetPerfStatsResult { Reset = true },
		}.ToByteArray();
	}

	public static byte[] BuildSearchCombatMctsResponse(CombatMctsResult result)
	{
		BridgeSearchCombatMctsResult payload = new BridgeSearchCombatMctsResult
		{
			ActionIndex = result.ActionIndex,
			RootValue = result.RootValue,
			SearchMs = result.SearchMs,
			RestoredOk = result.RestoredOk,
			SnapshotCount = result.SnapshotCount,
			SimulationCount = result.Breakdown.SimulationCount,
			SaveStateCount = result.Breakdown.SaveStateCount,
			LoadStateCount = result.Breakdown.LoadStateCount,
			DeleteStateCount = result.Breakdown.DeleteStateCount,
			StepCount = result.Breakdown.StepCount,
			AdvanceStateCount = result.Breakdown.AdvanceStateCount,
			EvalCallCount = result.Breakdown.EvalCallCount,
			EvalBatchCount = result.Breakdown.EvalBatchCount,
			EvalStateCount = result.Breakdown.EvalStateCount,
			SelectChildCount = result.Breakdown.SelectChildCount,
			BackpropCount = result.Breakdown.BackpropCount,
			SaveStateMs = result.Breakdown.SaveStateMs,
			LoadStateMs = result.Breakdown.LoadStateMs,
			DeleteStateMs = result.Breakdown.DeleteStateMs,
			StepMs = result.Breakdown.StepMs,
			AdvanceStateMs = result.Breakdown.AdvanceStateMs,
			EvalMs = result.Breakdown.EvalMs,
			SelectionMs = result.Breakdown.SelectionMs,
			BackpropMs = result.Breakdown.BackpropMs,
			DebugTraceJson = result.DebugTraceJson ?? "",
		};
		payload.VisitCounts.Add(result.VisitCounts);
		payload.VisitProbs.Add(result.VisitProbs);
		payload.QValues.Add(result.QValues);
		payload.Priors.Add(result.Priors);
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.SearchCombatMcts,
			Status = BridgeStatus.Ok,
			SearchCombatMcts = payload,
		}.ToByteArray();
	}

	// ================================================================
	// Combat-only responses (2026-04-18)
	//
	// 把 CombatTrainingStateSnapshot 包装成 proto GameState(含 legal_actions),
	// Python 侧不再自己推断 legal actions,直接消费 sim 的权威字段。
	// ================================================================

	public static byte[] BuildCombatStateResponse(BridgeMethod method, CombatTrainingStateSnapshot snapshot)
	{
		GameState state = BuildCombatGameStateMessage(snapshot);
		FullRunSimulationDiagnostics.Increment("proto.combat_state_bytes", state.CalculateSize());
		return new BridgeResponseEnvelope
		{
			Method = method,
			Status = BridgeStatus.Ok,
			State = new BridgeStatePayload { State = state },
		}.ToByteArray();
	}

	public static byte[] BuildCombatActResponse(CombatTrainingStepResult result, CombatTrainingStateSnapshot snapshot)
	{
		GameState state = BuildCombatGameStateMessage(snapshot);
		FullRunSimulationDiagnostics.Increment("proto.combat_state_bytes", state.CalculateSize());
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.CombatAct,
			Status = result.Accepted ? BridgeStatus.Ok : BridgeStatus.RejectedAction,
			Act = new BridgeActPayload
			{
				Accepted = result.Accepted,
				Error = result.Error ?? "",
				State = state,
			},
		}.ToByteArray();
	}

	public static byte[] BuildStatePayload(FullRunSimulationStateSnapshot snapshot)
	{
		return BuildStateMessage(snapshot).ToByteArray();
	}

	public static byte[] BuildLoadOrtModelResponse(
		bool loaded,
		bool hasValueOutput,
		bool hasDeckInputs,
		bool hasContinuationOutput,
		bool hasExtraScalarsInput,
		string executionProviderName,
		string requestedDevice,
		bool fellBackToCpu)
	{
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.LoadOrtModel,
			Status = BridgeStatus.Ok,
			LoadOrtModel = new BridgeLoadOrtModelResult
			{
				Loaded = loaded,
				HasValueOutput = hasValueOutput,
				HasDeckInputs = hasDeckInputs,
				HasContinuationOutput = hasContinuationOutput,
				HasExtraScalarsInput = hasExtraScalarsInput,
				ExecutionProviderName = executionProviderName ?? "",
				RequestedDevice = requestedDevice ?? "",
				FellBackToCpu = fellBackToCpu,
			},
		}.ToByteArray();
	}

	public static byte[] BuildRunCombatLocalResponse(
		int combatSteps,
		float elapsedMs,
		float getSnapshotMs,
		float ortMs,
		float stepAsyncMs,
		float waitAsyncMs,
		float maxStepMs,
		float maxWaitMs,
		FullRunSimulationStateSnapshot finalSnapshot)
	{
		GameState state = BuildStateMessage(finalSnapshot);
		return new BridgeResponseEnvelope
		{
			Method = BridgeMethod.RunCombatLocal,
			Status = BridgeStatus.Ok,
			RunCombatLocal = new BridgeRunCombatLocalResult
			{
				CombatSteps = combatSteps,
				ElapsedMs = elapsedMs,
				GetSnapshotMs = getSnapshotMs,
				OrtMs = ortMs,
				StepAsyncMs = stepAsyncMs,
				WaitAsyncMs = waitAsyncMs,
				MaxStepMs = maxStepMs,
				MaxWaitMs = maxWaitMs,
				State = state,
			},
		}.ToByteArray();
	}

	public static GameState BuildStateMessage(FullRunSimulationStateSnapshot snapshot)
	{
		return BridgeGameStateBuilder.FromFullRunSnapshot(snapshot);
	}

	private static GameState BuildCombatGameStateMessage(CombatTrainingStateSnapshot snapshot)
	{
		return BridgeGameStateBuilder.FromCombatSnapshot(snapshot);
	}

	private static Player? TryResolveActiveCombatPlayer()
	{
		CombatState? combatState = CombatManager.Instance.DebugOnlyGetState();
		if (combatState != null)
		{
			try
			{
				return LocalContext.GetMe(combatState);
			}
			catch
			{
			}
		}

		return TryResolveLocalPlayer(RunManager.Instance.DebugOnlyGetState());
	}

	private static PlayerState BuildCombatPlayerState(Player? player, CombatTrainingStateSnapshot snapshot)
	{
		CombatTrainingPlayerSnapshot? combatPlayer = snapshot.Player;
		PlayerState state = new PlayerState
		{
			Hp = combatPlayer?.CurrentHp ?? player?.Creature?.CurrentHp ?? 0,
			MaxHp = combatPlayer?.MaxHp ?? player?.Creature?.MaxHp ?? 0,
			Block = combatPlayer?.Block ?? player?.Creature?.Block ?? 0,
			Gold = player?.Gold ?? 0,
			Energy = combatPlayer?.Energy ?? 0,
			MaxEnergy = combatPlayer?.MaxEnergy ?? player?.MaxEnergy ?? 0,
			DrawPileCount = snapshot.Piles?.Draw ?? 0,
			DiscardPileCount = snapshot.Piles?.Discard ?? 0,
			ExhaustPileCount = snapshot.Piles?.Exhaust ?? 0,
			PlayPileCount = snapshot.Piles?.Play ?? 0,
			OpenPotionSlots = player?.PotionSlots.Count(static potion => potion == null) ?? 0,
			MaxPotions = player?.MaxPotionCount ?? 0,
			Stars = combatPlayer?.Stars ?? 0,
		};

		if (player != null)
		{
			AppendPlayerDeck(state, player.Deck?.Cards);
			AppendPlayerRelics(state, player.Relics);
			AppendPlayerPotions(state, player.PotionSlots.Where(static potion => potion != null).OfType<PotionModel>());
		}

		if (combatPlayer?.Powers != null)
		{
			foreach (CombatTrainingPowerSnapshot power in combatPlayer.Powers
				.Where(static p => p?.Id != null && p.Amount != 0))
			{
				state.Powers.Add(new Power { Id = power.Id ?? "", Amount = power.Amount });
			}
		}

		return state;
	}

	private static string DetectCombatStateType(CombatTrainingStateSnapshot snapshot)
	{
		if (snapshot.IsHandSelectionActive)
		{
			return "hand_select";
		}
		if (snapshot.IsCardSelectionActive)
		{
			return "card_select";
		}
		// 没有 encounter room_type 信息,用 monster 作默认;训练侧拿 encounter_id 查 catalog
		return "monster";
	}

	private static void PopulateCombatLegalActions(GameState gs, CombatTrainingStateSnapshot snapshot)
	{
		// 规则:优先 hand_selection > card_selection > play-phase (play_card/end_turn)
		// 权威字段(来自 sim):RequiresTarget / ValidTargetIds / CanPlay / CanConfirm / Cancelable
		if (snapshot.IsEpisodeDone)
		{
			return;
		}

		if (snapshot.IsHandSelectionActive && snapshot.HandSelection != null)
		{
			var hs = snapshot.HandSelection;
			if (SelectionActionSemantics.ShouldExposeSelectionActions(hs.SelectedCards.Count, hs.MaxSelect))
			{
				foreach (CombatTrainingHandCardSnapshot card in hs.SelectableCards)
				{
					gs.LegalActions.Add(new LegalAction
					{
						Action = "select_hand_card",
						Index = card.HandIndex,
						CardIndex = card.HandIndex,
						CardId = card.Id ?? "",
						Label = card.Title ?? card.Id ?? "",
					});
				}
			}
			if (hs.CanConfirm)
			{
				gs.LegalActions.Add(new LegalAction { Action = "confirm_selection", Label = "Confirm" });
			}
			if (hs.Cancelable)
			{
				gs.LegalActions.Add(new LegalAction { Action = "cancel_selection", Label = "Cancel" });
			}
			return;
		}

		if (snapshot.IsCardSelectionActive && snapshot.CardSelection != null)
		{
			var cs = snapshot.CardSelection;
			if (SelectionActionSemantics.ShouldExposeSelectionActions(cs.SelectedCards.Count, cs.MaxSelect))
			{
				foreach (CombatTrainingSelectableCardSnapshot opt in cs.SelectableCards)
				{
					gs.LegalActions.Add(new LegalAction
					{
						Action = "select_card_option",
						Index = opt.ChoiceIndex,
						CardIndex = opt.ChoiceIndex,
						CardId = opt.Id ?? "",
						Label = opt.Title ?? opt.Id ?? "",
					});
				}
			}
			if (cs.CanConfirm)
			{
				gs.LegalActions.Add(new LegalAction { Action = "confirm_selection", Label = "Confirm" });
			}
			if (cs.Cancelable)
			{
				gs.LegalActions.Add(new LegalAction { Action = "cancel_selection", Label = "Cancel" });
			}
			return;
		}

		// Play phase
		HashSet<int> validHandIndices = new HashSet<int>();
		foreach (CombatTrainingHandCardSnapshot card in snapshot.Hand)
		{
			validHandIndices.Add(card.HandIndex);
		}
		foreach (CombatTrainingHandCardSnapshot card in snapshot.Hand)
		{
			if (!card.CanPlay) continue;
			if (!validHandIndices.Contains(card.HandIndex)) continue;
			string cardId = card.Id ?? "";
			string label = card.Title ?? cardId;
			if (card.RequiresTarget)
			{
				if (card.ValidTargetIds == null || card.ValidTargetIds.Count == 0) continue;
				foreach (uint tid in card.ValidTargetIds)
				{
					gs.LegalActions.Add(new LegalAction
					{
						Action = "play_card",
						Index = card.HandIndex,
						CardIndex = card.HandIndex,
						CardId = cardId,
						Label = label,
						TargetId = (int)tid,
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
					CardId = cardId,
					Label = label,
				});
			}
		}
		if (snapshot.CanEndTurn)
		{
			gs.LegalActions.Add(new LegalAction { Action = "end_turn", Label = "End Turn" });
		}
	}

	// ================================================================
	// Core: snapshot → protobuf GameState → byte[]
	// ================================================================

	internal static GameState BuildProtoStateMessage(FullRunSimulationStateSnapshot snapshot)
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

		return gs;
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
		AppendPlayerDeck(ps, player.Deck.Cards);

		AppendPlayerRelics(ps, player.Relics);

		AppendPlayerPotions(ps, player.PotionSlots.Where(static p => p != null).OfType<PotionModel>());

		return ps;
	}

	private static void AppendPlayerDeck(PlayerState state, IEnumerable<CardModel>? cards)
	{
		if (cards == null)
		{
			return;
		}

		int deckIndex = 0;
		foreach (CardModel card in cards)
		{
			state.Deck.Add(new CardInfo
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
	}

	private static void AppendPlayerRelics(PlayerState state, IEnumerable<RelicModel>? relics)
	{
		if (relics == null)
		{
			return;
		}

		int relicIndex = 0;
		foreach (RelicModel relic in relics)
		{
			state.Relics.Add(new RelicInfo
			{
				Index = relicIndex++,
				Id = relic.Id.Entry ?? "",
				Name = relic.Id.Entry ?? ""
			});
		}
	}

	private static void AppendPlayerPotions(PlayerState state, IEnumerable<PotionModel>? potions)
	{
		if (potions == null)
		{
			return;
		}

		int potionIndex = 0;
		foreach (PotionModel potion in potions)
		{
			state.Potions.Add(new PotionInfo
			{
				Index = potionIndex++,
				Id = potion.Id.Entry ?? "",
				Name = potion.Id.Entry ?? ""
			});
		}
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

	private static BattleState BuildBattleState(CombatTrainingStateSnapshot? combat, Player? runtimePlayer = null)
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
		if (combat.Player != null || runtimePlayer != null)
		{
			// battle.player 与顶层 player 共享同一份完整 build 视图，避免
			// combat-only API 在不同 payload 分支里出现字段缺口。
			bs.Player = BuildCombatPlayerState(runtimePlayer, combat);
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
				RequiresTarget = card.RequiresTarget,
				// 2026-04-24 动态信息：sim 内部已算好的实时描述，LLM 直接读用
				Description = card.Description ?? ""
			};
			foreach (uint tid in card.ValidTargetIds)
			{
				hc.ValidTargetIds.Add((int)tid);
			}
			if (card.PreviewDamagePerTarget != null)
			{
				foreach (KeyValuePair<uint, int> kv in card.PreviewDamagePerTarget)
				{
					hc.PreviewDamagePerTarget[(int)kv.Key] = kv.Value;
				}
			}
			hc.PreviewBlock = card.PreviewBlock;
			if (card.Keywords != null)
			{
				foreach (string kw in card.Keywords)
				{
					if (!string.IsNullOrEmpty(kw)) hc.Keywords.Add(kw);
				}
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

		if (es.IsFinished && es.Options.Count == 0)
		{
			es.Options.Add(new STS2AI.Bridge.EventOption
			{
				Index = 0,
				Text = "proceed",
				Label = "proceed",
				IsLocked = false,
				IsChosen = false,
				IsProceed = true
			});
		}

		return es;
	}

	// ================================================================
	// Rest Site
	// ================================================================

	private static RestSiteState BuildRestSiteState()
	{
		IReadOnlyList<RestSiteOption> options = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
		RestSiteState rs = new RestSiteState { CanProceed = options.Count == 0 };
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
		int maxSelect = selection?.MaxSelect ?? 0;
		int selectedCount = selectedCards.Count;
		bool canConfirm = selection?.CanConfirm ?? false;
		bool selectionQuotaReached = maxSelect > 0 && selectedCount >= maxSelect;
		bool previewShowing = selectionQuotaReached && canConfirm;
		bool canCancel = selection?.Cancelable ?? false;
		if (!canCancel && previewShowing && selectedCount > 0)
		{
			canCancel = true;
		}
		List<CombatTrainingSelectableCardSnapshot> visibleCards = previewShowing
			? selectableCards.Concat(selectedCards).OrderBy(static card => card.ChoiceIndex).ToList()
			: selectableCards;
		CardSelectState cs = new CardSelectState
		{
			ScreenType = NormalizeCardSelectScreenType(selection?.Mode),
			SelectedCount = selectedCount,
			CanConfirm = canConfirm,
			CanCancel = canCancel
		};
		foreach (CombatTrainingSelectableCardSnapshot card in visibleCards)
		{
			cs.Cards.Add(new HandCard
			{
				Index = card.ChoiceIndex,
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
				Index = card.ChoiceIndex,
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

	private static string NormalizeCardSelectScreenType(string? mode)
	{
		return mode switch
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
