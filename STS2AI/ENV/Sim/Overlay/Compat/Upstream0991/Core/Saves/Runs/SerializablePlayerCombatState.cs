using System.Collections.Generic;
using System.Text.Json.Serialization;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;

namespace MegaCrit.Sts2.Core.Saves.Runs;

/// <summary>
/// Captures the exact card pile ordering and resource state for a player mid-combat.
/// Used by save/load (e.g. MCTS tree search) to restore hand/draw/discard/exhaust
/// piles in the same order, avoiding the reshuffle that would happen with a fresh
/// PopulateCombatState().
/// </summary>
public class SerializablePlayerCombatState : IPacketSerializable
{
	[JsonPropertyName("hand")]
	public List<SerializableCard> Hand { get; set; } = new();

	[JsonPropertyName("draw_pile")]
	public List<SerializableCard> DrawPile { get; set; } = new();

	[JsonPropertyName("discard_pile")]
	public List<SerializableCard> DiscardPile { get; set; } = new();

	[JsonPropertyName("exhaust_pile")]
	public List<SerializableCard> ExhaustPile { get; set; } = new();

	[JsonPropertyName("energy")]
	public int Energy { get; set; }

	[JsonPropertyName("block")]
	public int Block { get; set; }

	/// <summary>Player creature's active powers/buffs/debuffs at save time.</summary>
	[JsonPropertyName("powers")]
	public List<SerializablePower> Powers { get; set; } = new();

	public void Serialize(PacketWriter writer)
	{
		writer.WriteList(Hand);
		writer.WriteList(DrawPile);
		writer.WriteList(DiscardPile);
		writer.WriteList(ExhaustPile);
		writer.WriteInt(Energy);
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
		Hand = reader.ReadList<SerializableCard>();
		DrawPile = reader.ReadList<SerializableCard>();
		DiscardPile = reader.ReadList<SerializableCard>();
		ExhaustPile = reader.ReadList<SerializableCard>();
		Energy = reader.ReadInt();
		Block = reader.ReadInt();
		try
		{
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
		catch
		{
			Powers = new List<SerializablePower>();
		}
	}
}
