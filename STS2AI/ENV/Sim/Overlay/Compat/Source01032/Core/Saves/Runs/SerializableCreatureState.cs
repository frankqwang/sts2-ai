using System.Collections.Generic;
using System.Text.Json.Serialization;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;

namespace MegaCrit.Sts2.Core.Saves.Runs;

/// <summary>
/// Captures a creature's (monster/player) mid-combat state for MCTS save/load.
/// Includes HP, block, and all active powers with their stacks.
/// </summary>
public class SerializableCreatureState : IPacketSerializable
{
	[JsonPropertyName("id")]
	public string Id { get; set; } = "";

	[JsonPropertyName("combat_id")]
	public uint CombatId { get; set; }

	[JsonPropertyName("hp")]
	public int Hp { get; set; }

	[JsonPropertyName("max_hp")]
	public int MaxHp { get; set; }

	[JsonPropertyName("block")]
	public int Block { get; set; }

	[JsonPropertyName("powers")]
	public List<SerializablePower> Powers { get; set; } = new();

	public void Serialize(PacketWriter writer)
	{
		writer.WriteString(Id);
		writer.WriteUInt(CombatId);
		writer.WriteInt(Hp);
		writer.WriteInt(MaxHp);
		writer.WriteInt(Block);
		writer.WriteInt(Powers.Count);
		foreach (var p in Powers)
		{
			writer.WriteString(p.Id);
			writer.WriteInt(p.Amount);
		}
	}

	public void Deserialize(PacketReader reader)
	{
		Id = reader.ReadString();
		CombatId = reader.ReadUInt();
		Hp = reader.ReadInt();
		MaxHp = reader.ReadInt();
		Block = reader.ReadInt();
		int count = reader.ReadInt();
		Powers = new List<SerializablePower>(count);
		for (int i = 0; i < count; i++)
		{
			Powers.Add(new SerializablePower
			{
				Id = reader.ReadString(),
				Amount = reader.ReadInt()
			});
		}
	}
}

/// <summary>
/// A single power/buff/debuff with its ID and stack count.
/// </summary>
public class SerializablePower
{
	[JsonPropertyName("id")]
	public string Id { get; set; } = "";

	[JsonPropertyName("amount")]
	public int Amount { get; set; }
}
