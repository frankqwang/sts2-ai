using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using MegaCrit.Sts2.Core.Simulation;

namespace HeadlessSim;

internal enum HostProtocol
{
	Json,
	Proto
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
	ImportState = 0x0E,
	SkipCombat = 0x0F,
	SearchCombatMcts = 0x10,
	CombatReset = 0x11,
	CombatStep = 0x12,
	CombatState = 0x13
}

internal enum BinaryStatus : byte
{
	Ok = 0,
	RejectedAction = 1,
	SimulatorError = 2,
	ProtocolError = 3
}

internal sealed class BinarySearchCombatMctsRequest
{
	public int NumSimulations { get; init; }

	public float CPuct { get; init; }

	public float DirichletAlpha { get; init; }

	public float DirichletFraction { get; init; }

	public int MaxStepBudget { get; init; }

	public string FinalActionMode { get; init; } = "visit";

	public int FinalActionTopK { get; init; }

	public float FinalActionQWeight { get; init; }

	public bool UseContinuationValue { get; init; }

	public bool EnableDebugTrace { get; init; }
}

/// <summary>
/// Shared pipe request/aux-response codec.
///
/// 2026-04-21 起，手写 binary state transport 已下线；这里只保留：
/// 1. proto pipe 复用的请求解析；
/// 2. 少数非 GameState 的辅助响应编码（perf/mcts/ort 元数据等）。
/// </summary>
internal static class BinaryProtocol
{
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

	public static string PipeName(int port, HostProtocol protocol)
	{
		return protocol switch
		{
			HostProtocol.Proto => $"sts2_mcts_proto_{port}",
			_ => $"sts2_mcts_{port}"
		};
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
			throw new InvalidOperationException("Pipe request body is empty.");
		}

		byte opcode = request[0];
		if (!Enum.IsDefined(typeof(BinaryOpcode), opcode) || opcode == (byte)BinaryOpcode.Handshake)
		{
			throw new InvalidOperationException($"Unsupported opcode: 0x{opcode:X2}");
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
		if (reader.HasRemaining)
		{
			reset.Build = SimulationBuildSupport.ParseJson(reader.ReadOptionalString());
		}
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

	public static BinarySearchCombatMctsRequest ParseSearchCombatMctsRequest(ReadOnlySpan<byte> request)
	{
		BinaryRequestReader reader = new(request);
		reader.ReadOpcode(BinaryOpcode.SearchCombatMcts);
		BinarySearchCombatMctsRequest parsed = new BinarySearchCombatMctsRequest
		{
			NumSimulations = reader.ReadUInt16(),
			CPuct = reader.ReadSingle(),
			DirichletAlpha = reader.ReadSingle(),
			DirichletFraction = reader.ReadSingle(),
			MaxStepBudget = reader.ReadUInt16(),
			FinalActionMode = DecodeFinalActionMode(reader.ReadByte()),
			FinalActionTopK = reader.ReadUInt16(),
			FinalActionQWeight = reader.ReadSingle(),
			UseContinuationValue = reader.ReadByte() != 0,
			EnableDebugTrace = reader.HasRemaining && reader.ReadByte() != 0,
		};
		reader.ThrowIfRemaining();
		return parsed;
	}

	public static byte[] BuildSearchCombatMctsResponse(CombatMctsResult result)
	{
		using MemoryStream stream = new();
		using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write((byte)BinaryStatus.Ok);
		writer.Write((byte)BinaryOpcode.SearchCombatMcts);
		writer.Write((short)result.ActionIndex);
		writer.Write((ushort)result.VisitCounts.Length);
		foreach (int visitCount in result.VisitCounts)
		{
			writer.Write(visitCount);
		}
		foreach (float visitProb in result.VisitProbs)
		{
			writer.Write(visitProb);
		}
		foreach (float qValue in result.QValues)
		{
			writer.Write(qValue);
		}
		foreach (float prior in result.Priors)
		{
			writer.Write(prior);
		}
		writer.Write(result.RootValue);
		writer.Write(result.SearchMs);
		writer.Write((byte)(result.RestoredOk ? 1 : 0));
		writer.Write(result.SnapshotCount);
		writer.Write(result.Breakdown.SimulationCount);
		writer.Write(result.Breakdown.SaveStateCount);
		writer.Write(result.Breakdown.LoadStateCount);
		writer.Write(result.Breakdown.DeleteStateCount);
		writer.Write(result.Breakdown.StepCount);
		writer.Write(result.Breakdown.AdvanceStateCount);
		writer.Write(result.Breakdown.EvalCallCount);
		writer.Write(result.Breakdown.EvalBatchCount);
		writer.Write(result.Breakdown.EvalStateCount);
		writer.Write(result.Breakdown.SelectChildCount);
		writer.Write(result.Breakdown.BackpropCount);
		writer.Write(result.Breakdown.SaveStateMs);
		writer.Write(result.Breakdown.LoadStateMs);
		writer.Write(result.Breakdown.DeleteStateMs);
		writer.Write(result.Breakdown.StepMs);
		writer.Write(result.Breakdown.AdvanceStateMs);
		writer.Write(result.Breakdown.EvalMs);
		writer.Write(result.Breakdown.SelectionMs);
		writer.Write(result.Breakdown.BackpropMs);
		WriteOptionalString(writer, result.DebugTraceJson);
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

	private static string DecodeFinalActionMode(byte code)
	{
		return code == 1 ? "visit_q_blend" : "visit";
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
				throw new InvalidOperationException($"Pipe request opcode mismatch. Expected {(byte)expected}, got {opcode}.");
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

		public float ReadSingle()
		{
			EnsureAvailable(4);
			float value = BitConverter.ToSingle(_buffer.Slice(_offset, 4));
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
				throw new InvalidOperationException($"Pipe request had {_buffer.Length - _offset} trailing bytes.");
			}
		}

		public bool HasRemaining => _offset < _buffer.Length;

		private void EnsureAvailable(int count)
		{
			if (_offset + count > _buffer.Length)
			{
				throw new InvalidOperationException("Pipe request ended unexpectedly.");
			}
		}
	}
}
