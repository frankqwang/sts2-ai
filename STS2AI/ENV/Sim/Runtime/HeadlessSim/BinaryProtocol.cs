using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
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

namespace HeadlessSim;

internal enum HostProtocol
{
	Json,
	Binary
}

internal enum BinaryOpcode : byte
{
	Handshake = 0x00,
	Reset = 0x01,
	State = 0x02,
	Step = 0x03,
	BatchStep = 0x04,
	SaveState = 0x05,
	LoadState = 0x06,
	DeleteState = 0x07,
	PerfStats = 0x08,
	ResetPerfStats = 0x09,
	StepLocalPolicy = 0x0A,
	LoadOrtModel = 0x0B,
	RunCombatLocal = 0x0C,
	ExportState = 0x0D,
	ImportState = 0x0E
}

internal enum BinaryStatus : byte
{
	Ok = 0,
	RejectedAction = 1,
	SimulatorError = 2,
	ProtocolError = 3
}

internal enum BinarySymbolKind : byte
{
	Card = 1,
	Relic = 2,
	Potion = 3,
	Monster = 4,
	Event = 5,
	Intent = 6
}

internal sealed class BinarySessionState
{
	private sealed class PendingSymbolUpdate
	{
		public byte Kind { get; init; }

		public ushort SymbolId { get; init; }

		public string Value { get; init; } = string.Empty;
	}

	private readonly Dictionary<string, ushort> _symbols = new(StringComparer.Ordinal);
	private readonly Queue<PendingSymbolUpdate> _pendingUpdates = new();
	private ushort _nextSymbolId = 1;

	public ushort PlayerStaticVersion { get; private set; }

	public string? PlayerStaticSignature { get; private set; }

	public ushort GetOrAddSymbol(BinarySymbolKind kind, string? rawValue)
	{
		string value = (rawValue ?? string.Empty).Trim();
		if (value.Length == 0)
		{
			return 0;
		}

		string key = ((byte)kind).ToString() + ":" + value;
		if (_symbols.TryGetValue(key, out ushort existing))
		{
			return existing;
		}

		if (_nextSymbolId == ushort.MaxValue)
		{
			throw new InvalidOperationException("Binary symbol table exceeded ushort capacity.");
		}

		ushort created = _nextSymbolId++;
		_symbols[key] = created;
		_pendingUpdates.Enqueue(new PendingSymbolUpdate
		{
			Kind = (byte)kind,
			SymbolId = created,
			Value = value
		});
		return created;
	}

	public void AdvancePlayerStaticVersion(string signature)
	{
		if (string.Equals(PlayerStaticSignature, signature, StringComparison.Ordinal))
		{
			FullRunSimulationDiagnostics.Increment("binary.player_static_hits");
			return;
		}

		PlayerStaticSignature = signature;
		PlayerStaticVersion++;
		FullRunSimulationDiagnostics.Increment("binary.player_static_misses");
	}

	public void WritePendingSymbolUpdates(BinaryWriter writer)
	{
		writer.Write((ushort)_pendingUpdates.Count);
		int totalBytes = 0;
		while (_pendingUpdates.Count > 0)
		{
			PendingSymbolUpdate update = _pendingUpdates.Dequeue();
			writer.Write(update.Kind);
			writer.Write(update.SymbolId);
			totalBytes += 3;
			totalBytes += BinaryProtocol.WriteString(writer, update.Value);
		}

		if (totalBytes > 0)
		{
			FullRunSimulationDiagnostics.Increment("binary.symbol_update_bytes", totalBytes);
		}
	}
}

internal static class BinaryProtocol
{
	private const ushort ProtocolVersion = 9;
	internal const string BinarySchemaHash = "sts2-binary-schema-2026-04-09a";
	private static readonly string BuildGitSha = ResolveBuildGitSha();

	private static readonly Dictionary<string, byte> ActionTypeToCode = new(StringComparer.OrdinalIgnoreCase)
	{
		["wait"] = 1,
		["play_card"] = 2,
		["end_turn"] = 3,
		["choose_map_node"] = 4,
		["claim_reward"] = 5,
		["select_card_reward"] = 6,
		["skip_card_reward"] = 7,
		["choose_rest_option"] = 8,
		["shop_purchase"] = 9,
		["shop_exit"] = 10,
		["choose_event_option"] = 11,
		["proceed"] = 12,
		["advance_dialogue"] = 13,
		["select_card"] = 14,
		["confirm_selection"] = 15,
		["cancel_selection"] = 16,
		["combat_select_card"] = 17,
		["combat_confirm_selection"] = 18,
		["select_card_option"] = 19,
		["use_potion"] = 20,
		["drink_potion"] = 21,
		["claim_treasure_relic"] = 22,
		["select_relic"] = 23,
		["skip_relic_selection"] = 24,
		["skip"] = 25
	};

	private sealed class PlayerStaticCard
	{
		public ushort SymbolId { get; init; }

		public sbyte Cost { get; init; }

		public byte Type { get; init; }

		public byte Rarity { get; init; }

		public bool IsUpgraded { get; init; }

		public string Signature { get; init; } = string.Empty;
	}

	private sealed class PlayerStaticData
	{
		public byte MaxPotions { get; init; }

		public List<PlayerStaticCard> Deck { get; init; } = new();

		public List<ushort> Relics { get; init; } = new();

		public List<ushort> Potions { get; init; } = new();

		public string Signature { get; init; } = string.Empty;
	}

	public static string PipeName(int port, HostProtocol protocol)
	{
		return protocol == HostProtocol.Binary ? $"sts2_mcts_bin_{port}" : $"sts2_mcts_{port}";
	}

	public static byte[] BuildHandshakeResponse()
	{
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)BinaryStatus.Ok);
		writer.Write((byte)BinaryOpcode.Handshake);
		writer.Write(ProtocolVersion);
		WriteString(writer, BuildGitSha);
		WriteString(writer, BinarySchemaHash);
		return stream.ToArray();
	}

	private static string ResolveBuildGitSha()
	{
		try
		{
			using Process process = new();
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
			if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
			{
				return output;
			}
		}
		catch
		{
		}

		return "UNKNOWN";
	}

	public static byte[] BuildPerfStatsResponse(Dictionary<string, object?> payload)
	{
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)BinaryStatus.Ok);
		writer.Write((byte)BinaryOpcode.PerfStats);
		WriteString(writer, JsonSerializer.Serialize(payload));
		return stream.ToArray();
	}

	public static byte[] BuildResetPerfStatsResponse()
	{
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)BinaryStatus.Ok);
		writer.Write((byte)BinaryOpcode.ResetPerfStats);
		writer.Write((byte)1);
		return stream.ToArray();
	}

	public static byte[] BuildSaveStateResponse(string stateId, int cacheSize)
	{
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)BinaryStatus.Ok);
		writer.Write((byte)BinaryOpcode.SaveState);
		WriteString(writer, stateId);
		writer.Write(cacheSize);
		return stream.ToArray();
	}

	public static byte[] BuildExportStateResponse(string path, int cacheSize)
	{
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)BinaryStatus.Ok);
		writer.Write((byte)BinaryOpcode.ExportState);
		WriteString(writer, path);
		writer.Write(cacheSize);
		return stream.ToArray();
	}

	public static byte[] BuildDeleteStateResponse(bool deleted, int cacheSize)
	{
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)BinaryStatus.Ok);
		writer.Write((byte)BinaryOpcode.DeleteState);
		writer.Write((byte)(deleted ? 1 : 0));
		writer.Write(cacheSize);
		return stream.ToArray();
	}

	public static byte[] BuildErrorResponse(BinaryOpcode opcode, BinaryStatus status, string errorCode, string error)
	{
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)status);
		writer.Write((byte)opcode);
		WriteString(writer, errorCode);
		WriteString(writer, error);
		return stream.ToArray();
	}

	public static BinaryOpcode ParseOpcode(ReadOnlySpan<byte> request)
	{
		if (request.Length == 0)
		{
			throw new InvalidOperationException("Binary request body is empty.");
		}

		byte opcode = request[0];
		if (!Enum.IsDefined(typeof(BinaryOpcode), opcode) || opcode == (byte)BinaryOpcode.Handshake)
		{
			throw new InvalidOperationException($"Unsupported binary opcode: 0x{opcode:X2}");
		}

		return (BinaryOpcode)opcode;
	}

	public static FullRunSimulationResetRequest ParseResetRequest(ReadOnlySpan<byte> request)
	{
		BinaryRequestReader reader = new(request);
		reader.ReadOpcode(BinaryOpcode.Reset);
		FullRunSimulationResetRequest reset = new FullRunSimulationResetRequest
		{
			CharacterId = reader.ReadOptionalString(),
			Seed = reader.ReadOptionalString(),
			AscensionLevel = reader.ReadInt32()
		};
		reader.ThrowIfRemaining();
		return reset;
	}

	public static string ParseStateIdRequest(BinaryOpcode opcode, ReadOnlySpan<byte> request)
	{
		BinaryRequestReader reader = new(request);
		reader.ReadOpcode(opcode);
		string stateId = reader.ReadRequiredString();
		reader.ThrowIfRemaining();
		return stateId;
	}

	public static (string Path, string? StateId) ParseExportStateRequest(ReadOnlySpan<byte> request)
	{
		BinaryRequestReader reader = new(request);
		reader.ReadOpcode(BinaryOpcode.ExportState);
		string path = reader.ReadRequiredString();
		string? stateId = reader.ReadOptionalString();
		reader.ThrowIfRemaining();
		return (path, stateId);
	}

	public static string ParsePathRequest(BinaryOpcode opcode, ReadOnlySpan<byte> request)
	{
		BinaryRequestReader reader = new(request);
		reader.ReadOpcode(opcode);
		string path = reader.ReadRequiredString();
		reader.ThrowIfRemaining();
		return path;
	}

	public static bool ParseDeleteClearAll(ReadOnlySpan<byte> request, out string? stateId)
	{
		BinaryRequestReader reader = new(request);
		reader.ReadOpcode(BinaryOpcode.DeleteState);
		bool clearAll = reader.ReadByte() != 0;
		stateId = clearAll ? null : reader.ReadRequiredString();
		reader.ThrowIfRemaining();
		return clearAll;
	}

	public static FullRunSimulationActionRequest ParseActionRequest(ReadOnlySpan<byte> request)
	{
		BinaryRequestReader reader = new(request);
		reader.ReadOpcode(BinaryOpcode.Step);
		FullRunSimulationActionRequest action = ReadAction(ref reader);
		reader.ThrowIfRemaining();
		return action;
	}

	public static List<FullRunSimulationActionRequest> ParseBatchActionRequest(ReadOnlySpan<byte> request)
	{
		BinaryRequestReader reader = new(request);
		reader.ReadOpcode(BinaryOpcode.BatchStep);
		ushort count = reader.ReadUInt16();
		List<FullRunSimulationActionRequest> actions = new(count);
		for (int i = 0; i < count; i++)
		{
			actions.Add(ReadAction(ref reader));
		}

		reader.ThrowIfRemaining();
		return actions;
	}

	public static byte[] BuildStateResponse(BinaryOpcode opcode, BinarySessionState session, FullRunSimulationStateSnapshot snapshot)
	{
		return BuildStateEnvelope(BinaryStatus.Ok, opcode, session, snapshot);
	}

	public static byte[] BuildStepResponse(BinarySessionState session, FullRunSimulationStepResult result, FullRunSimulationStateSnapshot snapshot)
	{
		BinaryStatus status = result.Accepted ? BinaryStatus.Ok : BinaryStatus.RejectedAction;
		byte[] statePayload = BuildStatePayload(session, snapshot);
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)status);
		writer.Write((byte)BinaryOpcode.Step);
		session.WritePendingSymbolUpdates(writer);
		writer.Write((byte)(result.Accepted ? 1 : 0));
		WriteOptionalString(writer, result.Error);
		writer.Write(statePayload);
		FullRunSimulationDiagnostics.Increment("binary.state_bytes", statePayload.Length);
		return stream.ToArray();
	}

	public static byte[] BuildBatchStepResponse(BinarySessionState session, FullRunSimulationBatchStepResult result, FullRunSimulationStateSnapshot snapshot)
	{
		BinaryStatus status = result.Accepted ? BinaryStatus.Ok : BinaryStatus.RejectedAction;
		byte[] statePayload = BuildStatePayload(session, snapshot);
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)status);
		writer.Write((byte)BinaryOpcode.BatchStep);
		session.WritePendingSymbolUpdates(writer);
		writer.Write((byte)(result.Accepted ? 1 : 0));
		writer.Write((ushort)Math.Max(0, result.StepsExecuted));
		WriteOptionalString(writer, result.Error);
		writer.Write(statePayload);
		FullRunSimulationDiagnostics.Increment("binary.state_bytes", statePayload.Length);
		return stream.ToArray();
	}

	public static int WriteString(BinaryWriter writer, string value)
	{
		byte[] bytes = Encoding.UTF8.GetBytes(value ?? string.Empty);
		writer.Write((ushort)bytes.Length);
		writer.Write(bytes);
		return 2 + bytes.Length;
	}

	public static void WriteOptionalString(BinaryWriter writer, string? value)
	{
		if (string.IsNullOrWhiteSpace(value))
		{
			writer.Write((byte)0);
			return;
		}

		writer.Write((byte)1);
		WriteString(writer, value);
	}

	private static byte[] BuildStateEnvelope(BinaryStatus status, BinaryOpcode opcode, BinarySessionState session, FullRunSimulationStateSnapshot snapshot)
	{
		byte[] statePayload = BuildStatePayload(session, snapshot);
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)status);
		writer.Write((byte)opcode);
		session.WritePendingSymbolUpdates(writer);
		writer.Write(statePayload);
		FullRunSimulationDiagnostics.Increment("binary.state_bytes", statePayload.Length);
		return stream.ToArray();
	}

	internal static byte[] BuildStatePayload(BinarySessionState session, FullRunSimulationStateSnapshot snapshot)
	{
		RunState? runState = RunManager.Instance.DebugOnlyGetState();
		Player? player = TryResolveLocalPlayer(runState);
		FullRunSimulationChoiceBridge bridge = FullRunSimulationChoiceBridge.Instance;
		FullRunPendingRewardSelectionSnapshot? rewardSelection = null;
		FullRunPendingCardRewardSnapshot? cardRewardSelection = null;
		CombatTrainingCardSelectionSnapshot? cardSelection = null;
		FullRunPendingRelicSelectionSnapshot? relicSelection = null;
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

		writer.Write(MapStateType(snapshot.StateType));
		writer.Write((byte)(snapshot.IsTerminal ? 1 : 0));
		writer.Write(MapRunOutcome(snapshot.RunOutcome));
		writer.Write((byte)Math.Clamp(snapshot.CurrentActIndex + 1, 0, byte.MaxValue));
		writer.Write((byte)Math.Clamp(snapshot.TotalFloor, 0, byte.MaxValue));

		PlayerStaticData? staticData = player == null ? null : CapturePlayerStatic(player, session);
		bool hasStaticUpdate = staticData != null && !string.Equals(session.PlayerStaticSignature, staticData.Signature, StringComparison.Ordinal);
		if (staticData != null)
		{
			session.AdvancePlayerStaticVersion(staticData.Signature);
		}

		writer.Write(session.PlayerStaticVersion);
		writer.Write((byte)(hasStaticUpdate ? 1 : 0));
		if (hasStaticUpdate && staticData != null)
		{
			WritePlayerStatic(writer, staticData);
		}

		WriteLegalActions(writer, snapshot.LegalActions);
		WritePlayerDynamic(writer, player, snapshot.StateType, snapshot.CachedCombatState);

		switch (snapshot.StateType)
		{
			case "map":
				WriteMapState(writer, snapshot);
				break;
			case "event":
				WriteEventState(writer, runState?.CurrentRoom as EventRoom, session);
				break;
			case "rest_site":
				WriteRestSiteState(writer);
				break;
			case "shop":
				WriteShopState(writer, runState?.CurrentRoom as MerchantRoom, session);
				break;
			case "treasure":
				WriteTreasureState(writer, session);
				break;
			case "combat_rewards":
				rewardSelection = bridge.BuildRewardSelectionSnapshot();
				WriteRewardsState(writer, rewardSelection, runState, player, session);
				break;
			case "card_reward":
				cardRewardSelection = bridge.BuildCardRewardSelectionSnapshot();
				WriteCardRewardState(writer, cardRewardSelection, session);
				break;
			case "card_select":
				cardSelection = bridge.BuildCardSelectionSnapshot(null);
				WriteCardSelectState(writer, cardSelection, session);
				break;
			case "relic_select":
				relicSelection = bridge.BuildRelicSelectionSnapshot();
				WriteRelicSelectState(writer, relicSelection, session);
				break;
			case "hand_select":
			case "monster":
			case "elite":
			case "boss":
				WriteCombatState(writer, snapshot.CachedCombatState, session);
				break;
		}

		return stream.ToArray();
	}

	private static FullRunSimulationActionRequest ReadAction(ref BinaryRequestReader reader)
	{
		byte actionType = reader.ReadByte();
		short index = reader.ReadInt16();
		short cardIndex = reader.ReadInt16();
		short targetId = reader.ReadInt16();
		sbyte col = reader.ReadInt8();
		sbyte row = reader.ReadInt8();
		sbyte slot = reader.ReadInt8();
		string actionName = ActionName(actionType);
		return new FullRunSimulationActionRequest
		{
			Action = actionName,
			Type = actionName,
			Index = index < 0 ? null : index,
			CardIndex = cardIndex < 0 ? null : cardIndex,
			TargetId = targetId < 0 ? null : (uint)targetId,
			Col = col < 0 ? null : col,
			Row = row < 0 ? null : row,
			Slot = slot < 0 ? null : slot
		};
	}

	private static void WriteLegalActions(BinaryWriter writer, IReadOnlyList<FullRunSimulationLegalAction> actions)
	{
		writer.Write((ushort)actions.Count);
		foreach (FullRunSimulationLegalAction action in actions)
		{
			writer.Write(MapActionType(action.Action));
			writer.Write((short)(action.Index ?? -1));
			writer.Write((short)(action.CardIndex ?? -1));
			writer.Write((short)(action.TargetId.HasValue ? Math.Clamp((int)action.TargetId.Value, short.MinValue, short.MaxValue) : -1));
			writer.Write((sbyte)(action.Col ?? -1));
			writer.Write((sbyte)(action.Row ?? -1));
			writer.Write((sbyte)(action.Slot ?? -1));
		}
	}

	private static PlayerStaticData CapturePlayerStatic(Player player, BinarySessionState session)
	{
		PlayerStaticData data = new PlayerStaticData
		{
			MaxPotions = (byte)Math.Clamp(player.MaxPotionCount, 0, byte.MaxValue)
		};
		StringBuilder signature = new();
		signature.Append(data.MaxPotions).Append('|');

		foreach (CardModel card in player.Deck.Cards)
		{
			PlayerStaticCard entry = new PlayerStaticCard
			{
				SymbolId = session.GetOrAddSymbol(BinarySymbolKind.Card, card.Id.Entry),
				Cost = (sbyte)Math.Clamp(card.EnergyCost.GetWithModifiers(CostModifiers.All), sbyte.MinValue, sbyte.MaxValue),
				Type = MapCardType(card.Type.ToString()),
				Rarity = MapCardRarity(card.Rarity.ToString()),
				IsUpgraded = card.IsUpgraded,
				Signature = $"{card.Id.Entry}:{card.EnergyCost.GetWithModifiers(CostModifiers.All)}:{card.Type}:{card.Rarity}:{(card.IsUpgraded ? 1 : 0)}"
			};
			data.Deck.Add(entry);
			signature.Append(entry.Signature).Append(';');
		}

		signature.Append('|');
		foreach (RelicModel relic in player.Relics)
		{
			data.Relics.Add(session.GetOrAddSymbol(BinarySymbolKind.Relic, relic.Id.Entry));
			signature.Append(relic.Id.Entry).Append(';');
		}

		signature.Append('|');
		foreach (PotionModel potion in player.PotionSlots.Where(static potion => potion != null).OfType<PotionModel>())
		{
			data.Potions.Add(session.GetOrAddSymbol(BinarySymbolKind.Potion, potion.Id.Entry));
			signature.Append(potion.Id.Entry).Append(';');
		}

		return new PlayerStaticData
		{
			MaxPotions = data.MaxPotions,
			Deck = data.Deck,
			Relics = data.Relics,
			Potions = data.Potions,
			Signature = signature.ToString()
		};
	}

	private static void WritePlayerStatic(BinaryWriter writer, PlayerStaticData data)
	{
		writer.Write(data.MaxPotions);
		writer.Write((ushort)data.Deck.Count);
		foreach (PlayerStaticCard card in data.Deck)
		{
			writer.Write(card.SymbolId);
			writer.Write(card.Cost);
			writer.Write(card.Type);
			writer.Write(card.Rarity);
			writer.Write((byte)(card.IsUpgraded ? 1 : 0));
		}

		writer.Write((byte)data.Relics.Count);
		foreach (ushort relic in data.Relics)
		{
			writer.Write(relic);
		}

		writer.Write((byte)data.Potions.Count);
		foreach (ushort potion in data.Potions)
		{
			writer.Write(potion);
		}
	}

	private static void WritePlayerDynamic(
		BinaryWriter writer,
		Player? player,
		string? stateType,
		CombatTrainingStateSnapshot? combat)
	{
		if (player == null)
		{
			writer.Write((byte)0);
			return;
		}

		bool useCombatSnapshot = IsCombatLikeStateType(stateType) && combat?.Player != null;
		if (IsCombatLikeStateType(stateType) && !useCombatSnapshot)
		{
			FullRunSimulationDiagnostics.Increment("binary.player_dynamic.combat_snapshot_missing");
		}

		int energy = useCombatSnapshot ? combat!.Player!.Energy : 0;
		int maxEnergy = useCombatSnapshot ? combat!.Player!.MaxEnergy : player.MaxEnergy;
		int drawPile = useCombatSnapshot ? combat!.Piles?.Draw ?? 0 : 0;
		int discardPile = useCombatSnapshot ? combat!.Piles?.Discard ?? 0 : 0;
		int exhaustPile = useCombatSnapshot ? combat!.Piles?.Exhaust ?? 0 : 0;
		int playPile = useCombatSnapshot ? combat!.Piles?.Play ?? 0 : 0;

		writer.Write((byte)1);
		writer.Write(player.Creature.CurrentHp);
		writer.Write(player.Creature.MaxHp);
		writer.Write(player.Creature.Block);
		writer.Write(player.Gold);
		writer.Write(energy);
		writer.Write(maxEnergy);
		writer.Write(drawPile);
		writer.Write(discardPile);
		writer.Write(exhaustPile);
		writer.Write(playPile);
		writer.Write(player.PotionSlots.Count(static potion => potion == null));
	}

	private static bool IsCombatLikeStateType(string? stateType)
	{
		return stateType switch
		{
			"monster" or "elite" or "boss" or "hand_select" or "combat_pending" => true,
			_ => false
		};
	}

	private static void WriteMapState(BinaryWriter writer, FullRunSimulationStateSnapshot snapshot)
	{
		// Next options (travelable nodes)
		writer.Write((byte)Math.Clamp(snapshot.MapOptions.Count, 0, byte.MaxValue));
		foreach (FullRunSimulationMapOption option in snapshot.MapOptions)
		{
			writer.Write((byte)Math.Clamp(option.Index, 0, byte.MaxValue));
			writer.Write((sbyte)Math.Clamp(option.Col, sbyte.MinValue, sbyte.MaxValue));
			writer.Write((sbyte)Math.Clamp(option.Row, sbyte.MinValue, sbyte.MaxValue));
			writer.Write(MapNodeType(option.PointType));
		}

		// Full map topology: all nodes with edges for route planning
		writer.Write((ushort)Math.Clamp(snapshot.MapNodes.Count, 0, ushort.MaxValue));
		foreach (FullRunSimulationMapNode node in snapshot.MapNodes)
		{
			writer.Write((sbyte)Math.Clamp(node.Col, sbyte.MinValue, sbyte.MaxValue));
			writer.Write((sbyte)Math.Clamp(node.Row, sbyte.MinValue, sbyte.MaxValue));
			writer.Write(MapNodeType(node.PointType));
			writer.Write((byte)Math.Clamp(node.Children.Count, 0, byte.MaxValue));
			foreach ((int childCol, int childRow) in node.Children)
			{
				writer.Write((sbyte)Math.Clamp(childCol, sbyte.MinValue, sbyte.MaxValue));
				writer.Write((sbyte)Math.Clamp(childRow, sbyte.MinValue, sbyte.MaxValue));
			}
		}

		// Boss location
		writer.Write((sbyte)Math.Clamp(snapshot.BossCol, sbyte.MinValue, sbyte.MaxValue));
		writer.Write((sbyte)Math.Clamp(snapshot.BossRow, sbyte.MinValue, sbyte.MaxValue));
	}

	private static void WriteEventState(BinaryWriter writer, EventRoom? eventRoom, BinarySessionState session)
	{
		EventModel? localEvent = eventRoom?.LocalMutableEvent;
		ushort eventSymbol = session.GetOrAddSymbol(BinarySymbolKind.Event, localEvent?.Id.ToString());
		IReadOnlyList<EventOption> options = localEvent?.CurrentOptions ?? Array.Empty<EventOption>();
		int optionCount = options.Count;
		bool inDialogue = localEvent != null && !localEvent.IsFinished && optionCount == 0;
		writer.Write((byte)(inDialogue ? 1 : 0));
		writer.Write(eventSymbol);
		writer.Write((byte)Math.Clamp(optionCount, 0, byte.MaxValue));
		foreach (EventOption option in options)
		{
			writer.Write((byte)(option.IsLocked ? 1 : 0));
			writer.Write((byte)(option.WasChosen ? 1 : 0));
			writer.Write((byte)(option.IsProceed ? 1 : 0));
			WriteOptionalString(writer, SafeFormatLocString(option.Title));
		}
	}

	private static void WriteRestSiteState(BinaryWriter writer)
	{
		IReadOnlyList<RestSiteOption> options = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
		writer.Write((byte)Math.Clamp(options.Count, 0, byte.MaxValue));
		foreach (RestSiteOption option in options)
		{
			writer.Write(MapRestOption(option.OptionId.ToString()));
			writer.Write((byte)(option.IsEnabled ? 1 : 0));
		}
	}

	private static void WriteShopState(BinaryWriter writer, MerchantRoom? merchantRoom, BinarySessionState session)
	{
		if (merchantRoom?.Inventory == null)
		{
			writer.Write((byte)0);
			return;
		}

		List<MerchantEntry> entries = EnumerateShopEntries(merchantRoom.Inventory).ToList();
		writer.Write((byte)Math.Clamp(entries.Count, 0, byte.MaxValue));
		foreach (MerchantEntry entry in entries)
		{
			byte category = MapShopCategory(entry);
			ushort symbolId = 0;
			bool onSale = false;
			switch (entry)
			{
				case MerchantCardEntry cardEntry:
					symbolId = session.GetOrAddSymbol(BinarySymbolKind.Card, cardEntry.CreationResult?.Card.Id.Entry);
					onSale = cardEntry.IsOnSale;
					break;
				case MerchantRelicEntry relicEntry:
					symbolId = session.GetOrAddSymbol(BinarySymbolKind.Relic, relicEntry.Model?.Id.Entry);
					break;
				case MerchantPotionEntry potionEntry:
					symbolId = session.GetOrAddSymbol(BinarySymbolKind.Potion, potionEntry.Model?.Id.Entry);
					break;
			}

			writer.Write(category);
			writer.Write(symbolId);
			writer.Write(entry.Cost);
			writer.Write((byte)(entry.EnoughGold ? 1 : 0));
			writer.Write((byte)(entry.IsStocked ? 1 : 0));
			writer.Write((byte)(onSale ? 1 : 0));
		}
	}

	private static void WriteTreasureState(BinaryWriter writer, BinarySessionState session)
	{
		IReadOnlyList<RelicModel>? relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		bool canProceed = relics == null;
		writer.Write((byte)(canProceed ? 1 : 0));
		int count = relics?.Count ?? 0;
		writer.Write((byte)Math.Clamp(count, 0, byte.MaxValue));
		if (relics == null)
		{
			return;
		}

		foreach (RelicModel relic in relics)
		{
			writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Relic, relic.Id.Entry));
		}
	}

	private static void WriteRewardsState(
		BinaryWriter writer,
		FullRunPendingRewardSelectionSnapshot? rewards,
		RunState? runState,
		Player? player,
		BinarySessionState session)
	{
		writer.Write((byte)((rewards?.CanProceed ?? false) ? 1 : 0));
		IReadOnlyList<Reward> items = rewards?.Rewards ?? Array.Empty<Reward>();
		string rewardSource = GetRewardSource(runState);
		int openPotionSlots = player?.PotionSlots.Count(static potion => potion == null) ?? 0;
		writer.Write((byte)Math.Clamp(items.Count, 0, byte.MaxValue));
		foreach (Reward reward in items)
		{
			writer.Write(MapRewardType(reward));
			writer.Write(ResolveRewardSymbol(reward, session));
			WriteOptionalString(writer, GetRewardLabel(reward));
			WriteOptionalString(writer, BuildRewardKey(reward));
			WriteOptionalString(writer, rewardSource);
			bool claimable = IsRewardClaimable(reward, openPotionSlots, out string? claimBlockReason);
			writer.Write((byte)(claimable ? 1 : 0));
			WriteOptionalString(writer, claimBlockReason);
		}
	}

	private static void WriteCardRewardState(BinaryWriter writer, FullRunPendingCardRewardSnapshot? reward, BinarySessionState session)
	{
		writer.Write((byte)((reward?.CanSkip ?? false) ? 1 : 0));
		IReadOnlyList<CardCreationResult> cards = reward?.Options ?? Array.Empty<CardCreationResult>();
		writer.Write((byte)Math.Clamp(cards.Count, 0, byte.MaxValue));
		foreach (CardCreationResult option in cards)
		{
			WriteCardOption(writer, option.Card, session);
		}
	}

	private static void WriteCardSelectState(BinaryWriter writer, CombatTrainingCardSelectionSnapshot? selection, BinarySessionState session)
	{
		WriteOptionalString(writer, selection?.Mode);
		List<CombatTrainingSelectableCardSnapshot> selectableCards = selection?.SelectableCards ?? new List<CombatTrainingSelectableCardSnapshot>();
		List<CombatTrainingSelectableCardSnapshot> selectedCards = selection?.SelectedCards ?? new List<CombatTrainingSelectableCardSnapshot>();
		writer.Write((byte)Math.Clamp(selectedCards.Count, 0, byte.MaxValue));
		writer.Write((byte)((selection?.CanConfirm ?? false) ? 1 : 0));
		writer.Write((byte)((selection?.Cancelable ?? false) ? 1 : 0));
		writer.Write((byte)Math.Clamp(selectableCards.Count, 0, byte.MaxValue));
		foreach (CombatTrainingSelectableCardSnapshot card in selectableCards)
		{
			WriteSelectableCard(writer, card, session);
		}

		writer.Write((byte)Math.Clamp(selectedCards.Count, 0, byte.MaxValue));
		foreach (CombatTrainingSelectableCardSnapshot card in selectedCards)
		{
			WriteSelectableCard(writer, card, session);
		}
	}

	private static void WriteSelectableCard(BinaryWriter writer, CombatTrainingSelectableCardSnapshot card, BinarySessionState session)
	{
		writer.Write((short)Math.Clamp(card.ChoiceIndex, short.MinValue, short.MaxValue));
		writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Card, card.Id));
		writer.Write((sbyte)Math.Clamp(card.EnergyCost, sbyte.MinValue, sbyte.MaxValue));
		writer.Write((byte)(card.IsUpgraded ? 1 : 0));
		writer.Write(MapCardType(card.Type.ToString()));
		writer.Write(MapCardRarity(string.Empty));
		writer.Write(MapTargetType(card.TargetType));
	}

	private static void WriteRelicSelectState(BinaryWriter writer, FullRunPendingRelicSelectionSnapshot? relics, BinarySessionState session)
	{
		writer.Write((byte)((relics?.CanSkip ?? false) ? 1 : 0));
		IReadOnlyList<RelicModel> items = relics?.Relics ?? Array.Empty<RelicModel>();
		writer.Write((byte)Math.Clamp(items.Count, 0, byte.MaxValue));
		foreach (RelicModel relic in items)
		{
			writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Relic, relic.Id.Entry));
		}
	}

	private static void WriteCombatState(BinaryWriter writer, CombatTrainingStateSnapshot? combat, BinarySessionState session)
	{
		combat ??= CombatTrainingEnvService.BuildStateSnapshot() ?? new CombatTrainingStateSnapshot();
		writer.Write((short)Math.Clamp(combat.RoundNumber, short.MinValue, short.MaxValue));
		writer.Write(MapTurnSide(combat.CurrentSide));
		writer.Write((byte)(combat.IsPlayPhase ? 1 : 0));
		writer.Write((byte)(combat.CanEndTurn ? 1 : 0));
		WriteCombatPlayer(writer, combat.Player);
		WriteCombatPlayerPowers(writer, combat.Player?.Powers ?? (IEnumerable<CombatTrainingPowerSnapshot>)Array.Empty<CombatTrainingPowerSnapshot>());
		WriteCombatHand(writer, combat.Hand, session);
		WriteCombatEnemies(writer, combat.Enemies, session);
		// Pile card lists (draw/discard/exhaust) for pile-specific encoding
		WritePileCardIds(writer, combat.Piles?.DrawCardIds, session);
		WritePileCardIds(writer, combat.Piles?.DiscardCardIds, session);
		WritePileCardIds(writer, combat.Piles?.ExhaustCardIds, session);
	}

	private static void WritePileCardIds(BinaryWriter writer, List<string>? cardIds, BinarySessionState session)
	{
		if (cardIds == null || cardIds.Count == 0)
		{
			writer.Write((byte)0);
			return;
		}
		int count = Math.Clamp(cardIds.Count, 0, 50); // max 50 cards per pile
		writer.Write((byte)count);
		for (int i = 0; i < count; i++)
		{
			writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Card, cardIds[i]));
		}
	}

	private static void WriteCombatPlayer(BinaryWriter writer, CombatTrainingPlayerSnapshot? player)
	{
		if (player == null)
		{
			writer.Write((byte)0);
			return;
		}

		writer.Write((byte)1);
		writer.Write(player.CurrentHp);
		writer.Write(player.MaxHp);
		writer.Write(player.Block);
		writer.Write(player.Energy);
		writer.Write(player.MaxEnergy);
		writer.Write(player.Stars);
	}

	private static void WriteCombatPlayerPowers(BinaryWriter writer, IEnumerable<CombatTrainingPowerSnapshot> powers)
	{
		writer.Write(GetPowerAmount(powers, "strength"));
		writer.Write(GetPowerAmount(powers, "dexterity"));
		writer.Write(GetPowerAmount(powers, "vulnerable"));
		writer.Write(GetPowerAmount(powers, "weak"));
		writer.Write(GetPowerAmount(powers, "frail"));
		writer.Write(GetPowerAmount(powers, "metallicize"));
		writer.Write(GetPowerAmount(powers, "regen"));
		writer.Write(GetPowerAmount(powers, "artifact"));
	}

	private static void WriteCombatHand(BinaryWriter writer, IReadOnlyList<CombatTrainingHandCardSnapshot> hand, BinarySessionState session)
	{
		writer.Write((byte)Math.Clamp(hand.Count, 0, byte.MaxValue));
		foreach (CombatTrainingHandCardSnapshot card in hand)
		{
			writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Card, card.Id));
			writer.Write((sbyte)Math.Clamp(card.EnergyCost, sbyte.MinValue, sbyte.MaxValue));
			writer.Write((byte)(card.IsUpgraded ? 1 : 0));
			writer.Write(MapCardType(card.CardType));
			writer.Write(MapCardRarity(string.Empty));
			writer.Write(MapTargetType(card.TargetType));
			writer.Write((byte)(card.CanPlay ? 1 : 0));
			writer.Write((byte)(card.RequiresTarget ? 1 : 0));
			writer.Write((byte)Math.Clamp(card.ValidTargetIds.Count, 0, byte.MaxValue));
			foreach (uint targetId in card.ValidTargetIds)
			{
				writer.Write((short)Math.Clamp((int)targetId, short.MinValue, short.MaxValue));
			}
		}
	}

	private static void WriteCombatEnemies(BinaryWriter writer, IReadOnlyList<CombatTrainingCreatureSnapshot> enemies, BinarySessionState session)
	{
		writer.Write((byte)Math.Clamp(enemies.Count, 0, byte.MaxValue));
		foreach (CombatTrainingCreatureSnapshot enemy in enemies)
		{
			writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Monster, enemy.Id));
			writer.Write((short)Math.Clamp((int)(enemy.CombatId ?? 0), short.MinValue, short.MaxValue));
			writer.Write(enemy.CurrentHp);
			writer.Write(enemy.MaxHp);
			writer.Write(enemy.Block);
			writer.Write((byte)(enemy.IsAlive ? 1 : 0));
			List<CombatTrainingIntentSnapshot> intents = enemy.Intents ?? new List<CombatTrainingIntentSnapshot>();
			writer.Write((byte)Math.Clamp(intents.Count, 0, byte.MaxValue));
			foreach (CombatTrainingIntentSnapshot intent in intents)
			{
				writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Intent, intent?.IntentType));
				int repeats = Math.Max(1, intent?.Repeats ?? 1);
				int totalDamage = intent?.TotalDamage ?? intent?.Damage ?? 0;
				int perHitDamage = intent?.Damage ?? (repeats > 1 && totalDamage > 0 ? totalDamage / repeats : totalDamage);
				writer.Write(perHitDamage);
				writer.Write(totalDamage);
				writer.Write((short)Math.Clamp(repeats, short.MinValue, short.MaxValue));
			}
			WriteCombatCreaturePowers(writer, enemy.Powers, session);
			// Phase 2.5 boss-state expansion (2026-04-08): is_hittable +
			// intends_to_attack + next_move_id. Python decoder reads these
			// directly from the binary stream and exposes them via
			// ENEMY_AUX_DIM slots 28..30. Without these fields the AI is
			// structurally blind to boss phase / move sequence and to
			// minion shields that prevent targeting.
			writer.Write((byte)(enemy.IsHittable ? 1 : 0));
			writer.Write((byte)(enemy.IntendsToAttack ? 1 : 0));
			writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Intent, enemy.NextMoveId ?? string.Empty));
		}
	}

	private static void WriteCombatCreaturePowers(BinaryWriter writer, IEnumerable<CombatTrainingPowerSnapshot> powers, BinarySessionState session)
	{
		List<CombatTrainingPowerSnapshot> entries = powers?
			.Where(static power => power?.Id != null && power.Id.Length > 0)
			.OrderBy(static power => power.Id, StringComparer.Ordinal)
			.ToList() ?? new List<CombatTrainingPowerSnapshot>();
		writer.Write((byte)Math.Clamp(entries.Count, 0, byte.MaxValue));
		foreach (CombatTrainingPowerSnapshot power in entries)
		{
			writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Intent, power.Id));
			writer.Write(power.Amount);
		}
	}

	private static void WriteCardOption(BinaryWriter writer, CardModel card, BinarySessionState session)
	{
		writer.Write(session.GetOrAddSymbol(BinarySymbolKind.Card, card.Id.Entry));
		writer.Write((sbyte)Math.Clamp(card.EnergyCost.GetWithModifiers(CostModifiers.All), sbyte.MinValue, sbyte.MaxValue));
		writer.Write((byte)(card.IsUpgraded ? 1 : 0));
		writer.Write(MapCardType(card.Type.ToString()));
		writer.Write(MapCardRarity(card.Rarity.ToString()));
		writer.Write(MapTargetType(card.TargetType));
	}

	private static ushort ResolveRewardSymbol(Reward reward, BinarySessionState session)
	{
		return reward switch
		{
			PotionReward potionReward => session.GetOrAddSymbol(BinarySymbolKind.Potion, potionReward.Potion?.Id.Entry),
			RelicReward relicReward => session.GetOrAddSymbol(BinarySymbolKind.Relic, FullRunUpstreamCompat.GetRewardRelic(relicReward)?.Id.Entry),
			CardReward cardReward => session.GetOrAddSymbol(BinarySymbolKind.Card, FullRunUpstreamCompat.GetCardRewardOptions(cardReward).FirstOrDefault()?.Card.Id.Entry),
			_ => 0
		};
	}

	private static IEnumerable<MerchantEntry> EnumerateShopEntries(MerchantInventory inventory)
	{
		foreach (MerchantCardEntry entry in inventory.CharacterCardEntries)
		{
			yield return entry;
		}
		foreach (MerchantCardEntry entry in inventory.ColorlessCardEntries)
		{
			yield return entry;
		}
		foreach (MerchantRelicEntry entry in inventory.RelicEntries)
		{
			yield return entry;
		}
		foreach (MerchantPotionEntry entry in inventory.PotionEntries)
		{
			yield return entry;
		}
		if (inventory.CardRemovalEntry != null)
		{
			yield return inventory.CardRemovalEntry;
		}
	}

	private static Player? TryResolveLocalPlayer(RunState? runState)
	{
		if (runState == null)
		{
			return null;
		}

		Player? player = LocalContext.GetMe(runState.Players);
		if (player != null)
		{
			return player;
		}

		player = runState.Players.FirstOrDefault();
		if (player != null)
		{
			LocalContext.NetId ??= player.NetId;
		}

		return player;
	}

	private static int GetPowerAmount(IEnumerable<CombatTrainingPowerSnapshot> powers, string idSubstring)
	{
		foreach (CombatTrainingPowerSnapshot power in powers)
		{
			if (power?.Id != null && power.Id.IndexOf(idSubstring, StringComparison.OrdinalIgnoreCase) >= 0)
			{
				return power.Amount;
			}
		}

		return 0;
	}

	private static byte MapStateType(string? stateType)
	{
		return NormalizeToken(stateType) switch
		{
			"menu" => 1,
			"map" => 2,
			"event" => 3,
			"restsite" => 4,
			"shop" => 5,
			"treasure" => 6,
			"monster" => 7,
			"elite" => 8,
			"boss" => 9,
			"combatpending" => 10,
			"combatrewards" => 11,
			"cardreward" => 12,
			"cardselect" => 13,
			"relicselect" => 14,
			"handselect" => 15,
			"gameover" => 16,
			"rewards" => 17,
			_ => 0
		};
	}

	private static byte MapRunOutcome(string? runOutcome)
	{
		return NormalizeToken(runOutcome) switch
		{
			"victory" => 1,
			"win" => 1,
			"defeat" => 2,
			"loss" => 2,
			"death" => 2,
			_ => 0
		};
	}

	private static byte MapNodeType(string? pointType)
	{
		return NormalizeToken(pointType) switch
		{
			"monster" => 1,
			"elite" => 2,
			"boss" => 3,
			"restsite" => 4,
			"shop" => 5,
			"event" => 6,
			"treasure" => 7,
			_ => 0
		};
	}

	private static byte MapCardType(string? cardType)
	{
		return NormalizeToken(cardType) switch
		{
			"attack" => 1,
			"skill" => 2,
			"power" => 3,
			"status" => 4,
			"curse" => 5,
			"quest" => 6,
			_ => 0
		};
	}

	private static byte MapCardRarity(string? rarity)
	{
		return NormalizeToken(rarity) switch
		{
			"basic" => 1,
			"common" => 2,
			"uncommon" => 3,
			"rare" => 4,
			"ancient" => 5,
			"event" => 6,
			"token" => 7,
			"status" => 8,
			"curse" => 9,
			"quest" => 10,
			_ => 0
		};
	}

	private static byte MapIntentType(string? intentType)
	{
		return NormalizeToken(intentType) switch
		{
			string text when text.Contains("attack", StringComparison.Ordinal) => 1,
			string text when text.Contains("defend", StringComparison.Ordinal) || text.Contains("block", StringComparison.Ordinal) => 2,
			string text when text.Contains("buff", StringComparison.Ordinal) && !text.Contains("debuff", StringComparison.Ordinal) => 3,
			string text when text.Contains("debuff", StringComparison.Ordinal) => 4,
			string text when text.Contains("escape", StringComparison.Ordinal) => 5,
			string text when text.Contains("sleep", StringComparison.Ordinal) => 6,
			_ => 0
		};
	}

	private static string GetRewardLabel(Reward reward)
	{
		try
		{
			return SafeFormatLocString(reward.Description);
		}
		catch
		{
			return string.Empty;
		}
	}

	private static string SafeFormatLocString(LocString? locString)
	{
		if (locString == null)
		{
			return string.Empty;
		}

		try
		{
			return locString.GetFormattedText() ?? string.Empty;
		}
		catch
		{
			return string.Empty;
		}
	}

	private static string BuildRewardKey(Reward reward)
	{
		string rewardType = RewardTypeForKey(reward);
		string label = GetRewardLabel(reward);
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

	private static string RewardTypeForKey(Reward reward)
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

	private static bool IsRewardClaimable(Reward reward, int openPotionSlots, out string? claimBlockReason)
	{
		claimBlockReason = null;
		if (reward is PotionReward && openPotionSlots <= 0)
		{
			claimBlockReason = "potion_slots_full";
			return false;
		}

		return true;
	}

	private static string GetRewardSource(RunState? runState)
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

	private static byte MapTargetType(TargetType targetType)
	{
		return targetType switch
		{
			TargetType.None => 1,
			TargetType.Self => 2,
			TargetType.AnyEnemy => 3,
			TargetType.AnyPlayer => 4,
			TargetType.AnyAlly => 5,
			TargetType.TargetedNoCreature => 6,
			TargetType.AllEnemies => 7,
			TargetType.RandomEnemy => 8,
			TargetType.AllAllies => 9,
			TargetType.Osty => 10,
			_ => 0
		};
	}

	private static byte MapTurnSide(CombatSide side)
	{
		return NormalizeToken(side.ToString()) switch
		{
			"player" => 1,
			"enemy" => 2,
			_ => 0
		};
	}

	private static byte MapShopCategory(MerchantEntry entry)
	{
		return entry switch
		{
			MerchantCardEntry => 1,
			MerchantRelicEntry => 2,
			MerchantPotionEntry => 3,
			MerchantCardRemovalEntry => 4,
			_ => 0
		};
	}

	private static byte MapRewardType(Reward reward)
	{
		return reward switch
		{
			GoldReward => 1,
			PotionReward => 2,
			RelicReward => 3,
			CardReward => 4,
			CardRemovalReward => 5,
			SpecialCardReward => 6,
			_ => 0
		};
	}

	private static byte MapRestOption(string? optionId)
	{
		return NormalizeToken(optionId) switch
		{
			"rest" => 1,
			"heal" => 1,
			"smith" => 2,
			"upgrade" => 2,
			"recall" => 3,
			"dig" => 4,
			"lift" => 5,
			"toke" => 6,
			_ => 0
		};
	}

	private static byte MapActionType(string? action)
	{
		return ActionTypeToCode.TryGetValue(action ?? string.Empty, out byte value) ? value : (byte)255;
	}

	private static string ActionName(byte actionType)
	{
		foreach (KeyValuePair<string, byte> entry in ActionTypeToCode)
		{
			if (entry.Value == actionType)
			{
				return entry.Key;
			}
		}

		return "other";
	}

	private static string NormalizeToken(string? value)
	{
		return (value ?? string.Empty)
			.Trim()
			.Replace("_", string.Empty, StringComparison.Ordinal)
			.Replace("-", string.Empty, StringComparison.Ordinal)
			.Replace(" ", string.Empty, StringComparison.Ordinal)
			.ToLowerInvariant();
	}

	private ref struct BinaryRequestReader
	{
		private ReadOnlySpan<byte> _buffer;
		private int _offset;

		public BinaryRequestReader(ReadOnlySpan<byte> buffer)
		{
			_buffer = buffer;
			_offset = 0;
		}

		public void ReadOpcode(BinaryOpcode expected)
		{
			byte opcode = ReadByte();
			if (opcode != (byte)expected)
			{
				throw new InvalidOperationException($"Binary request opcode mismatch. Expected {(byte)expected}, got {opcode}.");
			}
		}

		public byte ReadByte()
		{
			EnsureAvailable(1);
			return _buffer[_offset++];
		}

		public sbyte ReadInt8()
		{
			return unchecked((sbyte)ReadByte());
		}

		public short ReadInt16()
		{
			EnsureAvailable(2);
			short value = BitConverter.ToInt16(_buffer.Slice(_offset, 2));
			_offset += 2;
			return value;
		}

		public ushort ReadUInt16()
		{
			EnsureAvailable(2);
			ushort value = BitConverter.ToUInt16(_buffer.Slice(_offset, 2));
			_offset += 2;
			return value;
		}

		public int ReadInt32()
		{
			EnsureAvailable(4);
			int value = BitConverter.ToInt32(_buffer.Slice(_offset, 4));
			_offset += 4;
			return value;
		}

		public string? ReadOptionalString()
		{
			bool hasValue = ReadByte() != 0;
			return hasValue ? ReadRequiredString() : null;
		}

		public string ReadRequiredString()
		{
			ushort length = ReadUInt16();
			EnsureAvailable(length);
			string value = Encoding.UTF8.GetString(_buffer.Slice(_offset, length));
			_offset += length;
			return value;
		}

		public void ThrowIfRemaining()
		{
			if (_offset != _buffer.Length)
			{
				throw new InvalidOperationException($"Binary request had {_buffer.Length - _offset} trailing bytes.");
			}
		}

		private void EnsureAvailable(int count)
		{
			if (_offset + count > _buffer.Length)
			{
				throw new InvalidOperationException("Binary request ended unexpectedly.");
			}
		}
	}
}
