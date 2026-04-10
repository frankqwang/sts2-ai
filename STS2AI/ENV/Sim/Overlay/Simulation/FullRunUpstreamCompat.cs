using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Simulation;

internal static class FullRunUpstreamCompat
{
	private static readonly FieldInfo CardRewardCardsField = typeof(CardReward).GetField("_cards", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate CardReward._cards.");

	private static readonly PropertyInfo CardRewardOptionsProperty = typeof(CardReward).GetProperty("Options", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate CardReward.Options.");

	private static readonly FieldInfo RelicRewardRelicField = typeof(RelicReward).GetField("_relic", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate RelicReward._relic.");

	private static readonly PropertyInfo EventOptionDisableOnChosenProperty = typeof(EventOption).GetProperty("DisableOnChosen", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate EventOption.DisableOnChosen.");

	private static readonly FieldInfo NetCombatCardDbNextIdField = typeof(NetCombatCardDb).GetField("_nextId", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate NetCombatCardDb._nextId.");

	private static readonly FieldInfo NetCombatCardDbIdToCardField = typeof(NetCombatCardDb).GetField("_idToCard", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate NetCombatCardDb._idToCard.");

	private static readonly FieldInfo NetCombatCardDbCardToIdField = typeof(NetCombatCardDb).GetField("_cardToId", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate NetCombatCardDb._cardToId.");

	private static readonly MethodInfo NetCombatCardDbIdCardIfNecessaryMethod = typeof(NetCombatCardDb).GetMethod("IdCardIfNecessary", BindingFlags.Instance | BindingFlags.NonPublic)
		?? throw new InvalidOperationException("Could not locate NetCombatCardDb.IdCardIfNecessary.");

	private static readonly MethodInfo? NetCombatCardDbRebuildCardMappingsMethod = typeof(NetCombatCardDb).GetMethod("RebuildCardMappings", BindingFlags.Instance | BindingFlags.Public);

	public static IReadOnlyList<CardCreationResult> GetCardRewardOptions(CardReward reward)
	{
		return (IReadOnlyList<CardCreationResult>)(CardRewardCardsField.GetValue(reward)
			?? throw new InvalidOperationException("CardReward._cards is unavailable."));
	}

	public static CardCreationSource GetCardRewardSource(CardReward reward)
	{
		object? options = CardRewardOptionsProperty.GetValue(reward);
		PropertyInfo sourceProperty = options?.GetType().GetProperty("Source", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
			?? throw new InvalidOperationException("CardReward.Options.Source is unavailable.");
		return (CardCreationSource)(sourceProperty.GetValue(options)
			?? throw new InvalidOperationException("CardReward.Options.Source is unavailable."));
	}

	public static RelicModel? GetRewardRelic(RelicReward reward)
	{
		return (RelicModel?)RelicRewardRelicField.GetValue(reward);
	}

	public static bool IsDisableOnChosen(EventOption option)
	{
		return (bool)(EventOptionDisableOnChosenProperty.GetValue(option)
			?? throw new InvalidOperationException("EventOption.DisableOnChosen is unavailable."));
	}

	public static async Task ChooseLocalOptionAsync(EventSynchronizer synchronizer, int index)
	{
		EventModel beforeEvent = synchronizer.GetLocalEvent();
		string beforeFingerprint = BuildEventFingerprint(beforeEvent);
		RunState? beforeRunState = RunManager.Instance.DebugOnlyGetState();
		object? beforeRoom = beforeRunState?.CurrentRoom;

		synchronizer.ChooseLocalOption(index);

		for (int attempts = 0; attempts < 64; attempts++)
		{
			await CombatSimulationRuntime.Clock.YieldAsync();
			await Task.Yield();

			RunState? runState = RunManager.Instance.DebugOnlyGetState();
			if (runState?.CurrentRoom != beforeRoom)
			{
				return;
			}

			EventModel currentEvent = synchronizer.GetLocalEvent();
			if (currentEvent.IsFinished)
			{
				return;
			}

			string currentFingerprint = BuildEventFingerprint(currentEvent);
			if (!string.Equals(beforeFingerprint, currentFingerprint, StringComparison.Ordinal))
			{
				return;
			}
		}
	}

	public static void RebuildCardMappings(NetCombatCardDb cardDb, IReadOnlyList<Player> players)
	{
		if (NetCombatCardDbRebuildCardMappingsMethod != null)
		{
			NetCombatCardDbRebuildCardMappingsMethod.Invoke(cardDb, new object[] { players });
			return;
		}

		NetCombatCardDbNextIdField.SetValue(cardDb, 0u);

		Dictionary<uint, CardModel> idToCard = (Dictionary<uint, CardModel>)(NetCombatCardDbIdToCardField.GetValue(cardDb)
			?? throw new InvalidOperationException("NetCombatCardDb._idToCard is unavailable."));
		Dictionary<CardModel, uint> cardToId = (Dictionary<CardModel, uint>)(NetCombatCardDbCardToIdField.GetValue(cardDb)
			?? throw new InvalidOperationException("NetCombatCardDb._cardToId is unavailable."));
		idToCard.Clear();
		cardToId.Clear();

		foreach (Player player in players)
		{
			if (player.PlayerCombatState == null)
			{
				continue;
			}

			foreach (CardModel card in player.PlayerCombatState.AllPiles.SelectMany(static pile => pile.Cards))
			{
				NetCombatCardDbIdCardIfNecessaryMethod.Invoke(cardDb, new object[] { card });
			}
		}
	}

	private static string BuildEventFingerprint(EventModel eventModel)
	{
		return string.Join("|", new[]
		{
			eventModel.Id.Entry,
			eventModel.IsFinished ? "finished" : "active",
			string.Join(",", eventModel.CurrentOptions.Select(static option =>
				$"{option.TextKey}:{option.IsLocked}:{option.IsProceed}:{option.WasChosen}"))
		});
	}
}
