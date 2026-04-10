using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Training;

public interface ICombatChoiceAdapter
{
	string BackendKind { get; }

	bool IsSelectionActive { get; }

	bool RequiresFrameSync { get; }

	CombatTrainingHandSelectionSnapshot? BuildHandSelectionSnapshot(CombatState? combatState);

	CombatTrainingCardSelectionSnapshot? BuildCardSelectionSnapshot(CombatState? combatState);

	bool TrySelectHandCard(int handIndex, out string error);

	bool TrySelectCardChoice(int choiceIndex, out string error);

	bool TryConfirmSelection(out string error);

	bool TryCancelSelection(out string error);
}

public static class CombatTrainingChoiceAdapterResolver
{
	private static readonly ICombatChoiceAdapter SimulatorFallback = CombatTrainingSimulatorChoiceBridge.Instance;

	public static ICombatChoiceAdapter Resolve()
	{
		if (CardSelectCmd.Selector is ICombatChoiceAdapter combatChoiceAdapter)
		{
			return combatChoiceAdapter;
		}
		NPlayerHand? hand = NPlayerHand.Instance;
		if (hand != null)
		{
			return new UiCombatChoiceAdapter(hand);
		}
		return SimulatorFallback;
	}
}

internal static class CombatTrainingChoiceSnapshotBuilder
{
	public static CombatTrainingHandCardSnapshot BuildHandCardSnapshot(CardModel card, CombatState? combatState, int? explicitHandIndex = null)
	{
		combatState ??= CombatManager.Instance.DebugOnlyGetState();
		int handIndex = explicitHandIndex ?? GetHandIndex(card);
		return new CombatTrainingHandCardSnapshot
		{
			HandIndex = handIndex,
			CombatCardIndex = NetCombatCard.FromModel(card).CombatCardIndex,
			Id = card.Id.Entry,
			Title = card.Title,
			EnergyCost = NormalizeApiCardCost(card.EnergyCost.GetWithModifiers(CostModifiers.All), card.EnergyCost.CostsX),
			CostsX = card.EnergyCost.CostsX,
			StarCost = card.GetStarCostWithModifiers(),
			TargetType = card.TargetType,
			CanPlay = card.CanPlay(),
			RequiresTarget = CardRequiresTarget(card),
			ValidTargetIds = GetValidTargetIds(card, combatState),
			CardType = card.Type.ToString(),
			Description = SafeCardDescription(card),
			Keywords = SafeCardKeywords(card),
			IsUpgraded = card.IsUpgraded,
			GainsBlock = card.GainsBlock
		};
	}

	private static string SafeCardDescription(CardModel card)
	{
		try
		{
			return card.GetDescriptionForPile(card.Pile?.Type ?? PileType.Hand);
		}
		catch
		{
			try
			{
				return card.Description.GetRawText();
			}
			catch
			{
				return string.Empty;
			}
		}
	}

	private static List<string> SafeCardKeywords(CardModel card)
	{
		try
		{
			return card.Keywords.Select(static keyword => keyword.ToString()).ToList();
		}
		catch
		{
			return new List<string>();
		}
	}

	private static int NormalizeApiCardCost(int energyCost, bool costsX)
	{
		if (costsX)
		{
			return energyCost;
		}
		return energyCost < 0 ? 0 : energyCost;
	}

	public static List<uint> GetValidTargetIds(CardModel card, CombatState? combatState)
	{
		if (combatState == null || !CardRequiresTarget(card))
		{
			return new List<uint>();
		}
		return combatState.Creatures.Where(static creature => creature.CombatId.HasValue).Where(card.IsValidTarget).Select(static creature => creature.CombatId!.Value).ToList();
	}

	public static bool CardRequiresTarget(CardModel card)
	{
		return card.TargetType == TargetType.AnyEnemy || card.TargetType == TargetType.AnyAlly;
	}

	public static int GetHandIndex(CardModel card)
	{
		IReadOnlyList<CardModel> cards = PileType.Hand.GetPile(card.Owner).Cards;
		for (int i = 0; i < cards.Count; i++)
		{
			if (cards[i] == card)
			{
				return i;
			}
		}
		return -1;
	}

	public static CombatTrainingSelectableCardSnapshot BuildSelectableCardSnapshot(CardModel card, int choiceIndex)
	{
		return new CombatTrainingSelectableCardSnapshot
		{
			ChoiceIndex = choiceIndex,
			Id = card.Id.Entry,
			Title = card.Title,
			Type = card.Type,
			EnergyCost = card.EnergyCost.GetWithModifiers(CostModifiers.All),
			CostsX = card.EnergyCost.CostsX,
			StarCost = card.GetStarCostWithModifiers(),
			TargetType = card.TargetType,
			SourcePile = card.Pile?.Type.ToString() ?? PileType.None.ToString(),
			IsUpgraded = card.IsUpgraded,
			IsUpgradable = card.IsUpgradable
		};
	}

	public static string GetPromptText(LocString prompt)
	{
		if (prompt == null || prompt.IsEmpty)
		{
			return "";
		}
		try
		{
			return prompt.GetFormattedText();
		}
		catch
		{
			return "";
		}
	}
}

internal sealed class UiCombatChoiceAdapter : ICombatChoiceAdapter
{
	private readonly NPlayerHand _hand;

	public UiCombatChoiceAdapter(NPlayerHand hand)
	{
		_hand = hand ?? throw new ArgumentNullException(nameof(hand));
	}

	public string BackendKind => "ui";

	public bool IsSelectionActive => _hand.IsInCardSelection;

	public bool RequiresFrameSync => true;

	public CombatTrainingHandSelectionSnapshot? BuildHandSelectionSnapshot(CombatState? combatState)
	{
		if (!_hand.IsInCardSelection)
		{
			return null;
		}
		combatState ??= CombatManager.Instance.DebugOnlyGetState();
		return new CombatTrainingHandSelectionSnapshot
		{
			ChoiceAdapterKind = BackendKind,
			Mode = _hand.CurrentMode.ToString(),
			PromptText = GetHandSelectionPromptText(_hand),
			CanConfirm = CanConfirmHandSelection(_hand),
			Cancelable = false,
			IsBackendAvailable = true,
			SelectableCards = _hand.ActiveHolders.Where(static holder => holder.CardNode != null).Select((NHandCardHolder holder) => CombatTrainingChoiceSnapshotBuilder.BuildHandCardSnapshot(holder.CardNode.Model, combatState)).ToList(),
			SelectedCards = GetSelectedHandCards(_hand).Select((CardModel card) => CombatTrainingChoiceSnapshotBuilder.BuildHandCardSnapshot(card, combatState)).ToList()
		};
	}

	public CombatTrainingCardSelectionSnapshot? BuildCardSelectionSnapshot(CombatState? combatState)
	{
		return null;
	}

	public bool TrySelectHandCard(int handIndex, out string error)
	{
		error = "";
		if (!_hand.IsInCardSelection)
		{
			error = "Hand card selection is not active.";
			return false;
		}
		if (_hand.PeekButton.IsPeeking)
		{
			error = "Cannot select a card while hand peek mode is active.";
			return false;
		}
		NHandCardHolder? holder = _hand.ActiveHolders.FirstOrDefault((NHandCardHolder child) => child.CardNode != null && CombatTrainingChoiceSnapshotBuilder.GetHandIndex(child.CardNode.Model) == handIndex);
		if (holder == null)
		{
			error = $"Hand index {handIndex} is not selectable in the current hand selection prompt.";
			return false;
		}
		holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
		return true;
	}

	public bool TrySelectCardChoice(int choiceIndex, out string error)
	{
		error = "CardSelection is not supported by the UI-backed combat choice adapter yet.";
		return false;
	}

	public bool TryConfirmSelection(out string error)
	{
		error = "";
		if (!_hand.IsInCardSelection)
		{
			error = "Hand card selection is not active.";
			return false;
		}
		NConfirmButton? confirmButton = _hand.GetNodeOrNull<NConfirmButton>("%SelectModeConfirmButton");
		if (confirmButton == null)
		{
			error = "Hand selection confirm button was not found.";
			return false;
		}
		if (!confirmButton.IsEnabled)
		{
			error = "Current hand selection cannot be confirmed yet.";
			return false;
		}
		confirmButton.ForceClick();
		return true;
	}

	public bool TryCancelSelection(out string error)
	{
		error = "CancelSelection is not supported by the UI-backed combat choice adapter yet.";
		return false;
	}

	private static bool CanConfirmHandSelection(NPlayerHand hand)
	{
		return hand.GetNodeOrNull<NConfirmButton>("%SelectModeConfirmButton")?.IsEnabled ?? false;
	}

	private static IReadOnlyList<CardModel> GetSelectedHandCards(NPlayerHand hand)
	{
		NSelectedHandCardContainer? selectedContainer = hand.GetNodeOrNull<NSelectedHandCardContainer>("%SelectedHandCardContainer");
		if (selectedContainer == null)
		{
			return Array.Empty<CardModel>();
		}
		return selectedContainer.Holders.Where(static holder => holder.CardNode != null).Select(static holder => holder.CardNode.Model).ToList();
	}

	private static string GetHandSelectionPromptText(NPlayerHand hand)
	{
		return hand.GetNodeOrNull<MegaRichTextLabel>("%SelectionHeader")?.Text ?? "";
	}
}
