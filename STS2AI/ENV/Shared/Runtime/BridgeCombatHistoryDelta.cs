using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;

namespace STS2AI.Bridge.Runtime;

public static class BridgeCombatHistoryDelta
{
	public static int CaptureOffset()
	{
		try
		{
			return CombatManager.Instance.History.Entries.Count();
		}
		catch
		{
			return 0;
		}
	}

	public static List<SettlementEvent> CaptureSince(int startOffset)
	{
		try
		{
			List<CombatHistoryEntry> entries = CombatManager.Instance.History.Entries.ToList();
			int start = Math.Clamp(startOffset, 0, entries.Count);
			List<SettlementEvent> events = new();
			for (int i = start; i < entries.Count; i++)
			{
				events.Add(MapEntry(entries[i], events.Count));
			}
			return events;
		}
		catch
		{
			return new List<SettlementEvent>();
		}
	}

	private static SettlementEvent MapEntry(CombatHistoryEntry entry, int sequence)
	{
		try
		{
			return entry switch
			{
				CardPlayStartedEntry e => MapCardPlayStarted(e, sequence),
				CardPlayFinishedEntry e => MapCardPlayFinished(e, sequence),
				DamageReceivedEntry e => MapDamageReceived(e, sequence),
				CreatureAttackedEntry e => MapCreatureAttacked(e, sequence),
				BlockGainedEntry e => MapBlockGained(e, sequence),
				PowerReceivedEntry e => MapPowerReceived(e, sequence),
				EnergySpentEntry e => MapEnergySpent(e, sequence),
				CardDrawnEntry e => MapCardMoved("card_drawn", e, sequence),
				CardDiscardedEntry e => MapCardMoved("card_discarded", e, sequence),
				CardExhaustedEntry e => MapCardMoved("card_exhausted", e, sequence),
				CardGeneratedEntry e => MapCardGenerated(e, sequence),
				CardAfflictedEntry e => MapCardAfflicted(e, sequence),
				MonsterPerformedMoveEntry e => MapMonsterPerformedMove(e, sequence),
				OrbChanneledEntry e => MapOrbChanneled(e, sequence),
				PotionUsedEntry e => MapPotionUsed(e, sequence),
				StarsModifiedEntry e => MapAmount("stars_modified", e, sequence, e.Amount),
				SummonedEntry e => MapAmount("summoned", e, sequence, e.Amount),
				_ => BaseEvent(entry, sequence, ToSnakeType(entry.GetType().Name), entry.Actor),
			};
		}
		catch (Exception exc)
		{
			return new SettlementEvent
			{
				Type = "unknown",
				Sequence = sequence,
				Description = $"{SafeHistoryString(entry)} [bridge_mapping_error: {exc.GetType().Name}]",
			};
		}
	}

	private static SettlementEvent MapCardPlayStarted(CardPlayStartedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "card_play_started", entry.CardPlay.Card.Owner.Creature);
		ApplyCardPlay(ev, entry.CardPlay);
		return ev;
	}

	private static SettlementEvent MapCardPlayFinished(CardPlayFinishedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "card_play_finished", entry.CardPlay.Card.Owner.Creature);
		ApplyCardPlay(ev, entry.CardPlay);
		return ev;
	}

	private static SettlementEvent MapDamageReceived(DamageReceivedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "damage_received", entry.Dealer ?? entry.Receiver);
		ApplyCreatureTarget(ev, entry.Receiver);
		ev.SourceCardId = SafeCardId(entry.CardSource);
		ev.BlockedDamage = entry.Result.BlockedDamage;
		ev.UnblockedDamage = entry.Result.UnblockedDamage;
		ev.TotalDamage = entry.Result.TotalDamage;
		ev.OverkillDamage = entry.Result.OverkillDamage;
		ev.TargetKilled = entry.Result.WasTargetKilled;
		return ev;
	}

	private static SettlementEvent MapCreatureAttacked(CreatureAttackedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "creature_attacked", entry.Actor);
		foreach (DamageResult result in entry.DamageResults)
		{
			AddRepeatedTarget(ev, result.Receiver);
			ev.TotalDamage += result.TotalDamage;
			ev.BlockedDamage += result.BlockedDamage;
			ev.UnblockedDamage += result.UnblockedDamage;
			ev.OverkillDamage += result.OverkillDamage;
			ev.TargetKilled = ev.TargetKilled || result.WasTargetKilled;
		}
		if (entry.DamageResults.Count > 0)
		{
			ApplyCreatureTarget(ev, entry.DamageResults[0].Receiver);
		}
		return ev;
	}

	private static SettlementEvent MapBlockGained(BlockGainedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(
			entry,
			sequence,
			"block_gained",
			entry.CardPlay?.Card.Owner.Creature ?? entry.Receiver);
		ApplyCreatureTarget(ev, entry.Receiver);
		ev.CardId = SafeCardId(entry.CardPlay?.Card);
		ev.SourceCardId = SafeCardId(entry.CardPlay?.Card);
		ev.AmountInt = entry.Amount;
		ev.AmountValue = entry.Amount;
		return ev;
	}

	private static SettlementEvent MapPowerReceived(PowerReceivedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "power_received", entry.Applier ?? entry.Power.Owner);
		ApplyCreatureTarget(ev, entry.Power.Owner);
		ev.PowerId = SafeModelId(entry.Power);
		ev.AmountInt = DecimalToInt(entry.Amount);
		ev.AmountValue = decimal.ToDouble(entry.Amount);
		return ev;
	}

	private static SettlementEvent MapEnergySpent(EnergySpentEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "energy_spent", entry.Actor);
		ev.AmountInt = entry.Amount;
		ev.AmountValue = entry.Amount;
		ev.EnergySpent = entry.Amount;
		return ev;
	}

	private static SettlementEvent MapCardMoved(string type, CardDrawnEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, type, entry.Actor);
		ev.CardId = SafeCardId(entry.Card);
		ev.AmountInt = 1;
		ev.AmountValue = 1;
		return ev;
	}

	private static SettlementEvent MapCardMoved(string type, CardDiscardedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, type, entry.Actor);
		ev.CardId = SafeCardId(entry.Card);
		ev.AmountInt = 1;
		ev.AmountValue = 1;
		return ev;
	}

	private static SettlementEvent MapCardMoved(string type, CardExhaustedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, type, entry.Actor);
		ev.CardId = SafeCardId(entry.Card);
		ev.AmountInt = 1;
		ev.AmountValue = 1;
		return ev;
	}

	private static SettlementEvent MapCardGenerated(CardGeneratedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "card_generated", entry.Actor);
		ev.CardId = SafeCardId(entry.Card);
		ev.AmountInt = 1;
		ev.AmountValue = 1;
		return ev;
	}

	private static SettlementEvent MapCardAfflicted(CardAfflictedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "card_afflicted", entry.Actor);
		ev.CardId = SafeCardId(entry.Card);
		ev.PowerId = SafeModelId(entry.Affliction);
		ev.AmountInt = 1;
		ev.AmountValue = 1;
		return ev;
	}

	private static SettlementEvent MapMonsterPerformedMove(MonsterPerformedMoveEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "monster_performed_move", entry.Actor);
		ev.MoveId = entry.Move.Id ?? "";
		if (entry.Targets != null)
		{
			foreach (Creature target in entry.Targets)
			{
				AddRepeatedTarget(ev, target);
			}
			Creature? first = entry.Targets.FirstOrDefault();
			if (first != null)
			{
				ApplyCreatureTarget(ev, first);
			}
		}
		return ev;
	}

	private static SettlementEvent MapOrbChanneled(OrbChanneledEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "orb_channeled", entry.Actor);
		ev.OrbId = SafeModelId(entry.Orb);
		ev.AmountInt = 1;
		ev.AmountValue = 1;
		return ev;
	}

	private static SettlementEvent MapPotionUsed(PotionUsedEntry entry, int sequence)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, "potion_used", entry.Actor);
		ev.PotionId = SafeModelId(entry.Potion);
		if (entry.Target != null)
		{
			ApplyCreatureTarget(ev, entry.Target);
		}
		ev.AmountInt = 1;
		ev.AmountValue = 1;
		return ev;
	}

	private static SettlementEvent MapAmount(string type, CombatHistoryEntry entry, int sequence, int amount)
	{
		SettlementEvent ev = BaseEvent(entry, sequence, type, entry.Actor);
		ev.AmountInt = amount;
		ev.AmountValue = amount;
		if (type == "stars_modified")
		{
			ev.StarsSpent = amount < 0 ? -amount : 0;
		}
		return ev;
	}

	private static SettlementEvent BaseEvent(CombatHistoryEntry entry, int sequence, string type, Creature? actor)
	{
		SettlementEvent ev = new()
		{
			Type = type,
			Sequence = sequence,
			RoundNumber = entry.RoundNumber,
			TurnSide = entry.CurrentSide.ToString().ToLowerInvariant(),
			Description = SafeHistoryString(entry),
		};
		ApplyActor(ev, actor ?? entry.Actor);
		return ev;
	}

	private static void ApplyCardPlay(SettlementEvent ev, CardPlay cardPlay)
	{
		ev.CardId = SafeCardId(cardPlay.Card);
		ev.EnergySpent = cardPlay.Resources.EnergySpent;
		ev.StarsSpent = cardPlay.Resources.StarsSpent;
		ev.CardPlayIndex = cardPlay.PlayIndex;
		ev.CardPlayCount = cardPlay.PlayCount;
		ev.IsAutoPlay = cardPlay.IsAutoPlay;
		if (cardPlay.Target != null)
		{
			ApplyCreatureTarget(ev, cardPlay.Target);
		}
	}

	private static void ApplyActor(SettlementEvent ev, Creature? creature)
	{
		if (creature == null)
		{
			return;
		}
		ev.ActorId = SafeCreatureId(creature);
		ev.ActorCombatId = SafeCombatId(creature);
		ev.ActorIsPlayer = creature.IsPlayer;
	}

	private static void ApplyCreatureTarget(SettlementEvent ev, Creature? creature)
	{
		if (creature == null)
		{
			return;
		}
		ev.TargetId = SafeCreatureId(creature);
		ev.TargetCombatId = SafeCombatId(creature);
		ev.TargetIsPlayer = creature.IsPlayer;
	}

	private static void AddRepeatedTarget(SettlementEvent ev, Creature? creature)
	{
		if (creature == null)
		{
			return;
		}
		ev.TargetIds.Add(SafeCreatureId(creature));
		ev.TargetCombatIds.Add(SafeCombatId(creature));
	}

	private static int SafeCombatId(Creature creature)
	{
		try
		{
			return creature.CombatId.HasValue ? (int)creature.CombatId.Value : 0;
		}
		catch
		{
			return 0;
		}
	}

	private static string SafeCreatureId(Creature creature)
	{
		try
		{
			return creature.ModelId.Entry ?? "";
		}
		catch
		{
			return "";
		}
	}

	private static string SafeCardId(CardModel? card)
	{
		if (card == null)
		{
			return "";
		}
		return SafeModelId(card);
	}

	private static string SafeModelId(AbstractModel? model)
	{
		if (model == null)
		{
			return "";
		}
		try
		{
			return model.Id.Entry ?? "";
		}
		catch
		{
			return "";
		}
	}

	private static string SafeHistoryString(CombatHistoryEntry entry)
	{
		try
		{
			return entry.HumanReadableString ?? "";
		}
		catch
		{
			try
			{
				return entry.Description ?? "";
			}
			catch
			{
				return entry.GetType().Name;
			}
		}
	}

	private static int DecimalToInt(decimal value)
	{
		try
		{
			return decimal.ToInt32(value);
		}
		catch
		{
			return 0;
		}
	}

	private static string ToSnakeType(string typeName)
	{
		string name = typeName.EndsWith("Entry", StringComparison.Ordinal)
			? typeName[..^5]
			: typeName;
		List<char> chars = new();
		for (int i = 0; i < name.Length; i++)
		{
			char ch = name[i];
			if (char.IsUpper(ch) && i > 0)
			{
				chars.Add('_');
			}
			chars.Add(char.ToLowerInvariant(ch));
		}
		return new string(chars.ToArray());
	}
}
