using System;
using System.Linq;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Simulation;

internal static class CombatSimulatorAutoplay
{
	public static CombatTrainingActionRequest BuildAction(CombatTrainingStateSnapshot state)
	{
		CombatTrainingHandSelectionSnapshot? selection = state.HandSelection;
		if (selection != null && state.IsHandSelectionActive)
		{
			if (selection.CanConfirm && (selection.SelectedCards.Count > 0 || selection.SelectableCards.Count == 0))
			{
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.ConfirmSelection
				};
			}
			if (selection.SelectableCards.Count > 0)
			{
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.SelectHandCard,
					HandIndex = selection.SelectableCards[0].HandIndex
				};
			}
			if (selection.Cancelable)
			{
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.CancelSelection
				};
			}
			throw new InvalidOperationException("Hand selection is active but no selectable or confirmable action is available.");
		}

		CombatTrainingCardSelectionSnapshot? cardSelection = state.CardSelection;
		if (cardSelection != null && state.IsCardSelectionActive)
		{
			if (cardSelection.CanConfirm && (cardSelection.SelectedCards.Count > 0 || cardSelection.SelectableCards.Count == 0))
			{
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.ConfirmSelection
				};
			}
			if (cardSelection.SelectableCards.Count > 0)
			{
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.SelectCardChoice,
					ChoiceIndex = cardSelection.SelectableCards[0].ChoiceIndex
				};
			}
			if (cardSelection.Cancelable)
			{
				return new CombatTrainingActionRequest
				{
					Type = CombatTrainingActionType.CancelSelection
				};
			}
			throw new InvalidOperationException("Card selection is active but no selectable or confirmable action is available.");
		}

		CombatTrainingHandCardSnapshot? targetedCard = state.Hand.FirstOrDefault(static card => card.CanPlay && card.RequiresTarget && card.ValidTargetIds.Count > 0);
		if (targetedCard != null)
		{
			return new CombatTrainingActionRequest
			{
				Type = CombatTrainingActionType.PlayCard,
				HandIndex = targetedCard.HandIndex,
				TargetId = targetedCard.ValidTargetIds[0]
			};
		}

		CombatTrainingHandCardSnapshot? untargetedCard = state.Hand.FirstOrDefault(static card => card.CanPlay && !card.RequiresTarget);
		if (untargetedCard != null)
		{
			return new CombatTrainingActionRequest
			{
				Type = CombatTrainingActionType.PlayCard,
				HandIndex = untargetedCard.HandIndex
			};
		}

		if (state.CanEndTurn)
		{
			return new CombatTrainingActionRequest
			{
				Type = CombatTrainingActionType.EndTurn
			};
		}

		throw new InvalidOperationException("No legal simulator autoplay action was available for the current combat snapshot.");
	}
}
