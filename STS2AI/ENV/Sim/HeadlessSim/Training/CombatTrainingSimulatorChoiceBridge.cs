using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Training;

internal sealed class CombatTrainingSimulatorChoiceBridge : ICombatChoiceAdapter, IHandSelectionCardSelector, ICardSelectionPromptSelector
{
	private sealed class PendingCardSelection
	{
		private readonly List<CardModel> _selectedCards = new List<CardModel>();

		public IReadOnlyList<CardModel> Options { get; }

		public CardSelectorPrefs Prefs { get; }

		public string Mode { get; }

		public bool IsHandSelection { get; }

		public TaskCompletionSource<IEnumerable<CardModel>> CompletionSource { get; } = new TaskCompletionSource<IEnumerable<CardModel>>();

		public PendingCardSelection(IReadOnlyList<CardModel> options, CardSelectorPrefs prefs, string mode, bool isHandSelection)
		{
			Options = options;
			Prefs = prefs;
			Mode = mode;
			IsHandSelection = isHandSelection;
		}

		public IReadOnlyList<CardModel> SelectedCards => _selectedCards;

		public IEnumerable<CardModel> SelectableCards => Options.Where((CardModel card) => !_selectedCards.Contains(card));

		public IEnumerable<(int ChoiceIndex, CardModel Card)> SelectableChoices => Options.Select((CardModel card, int index) => (ChoiceIndex: index, Card: card)).Where((valueTuple) => !_selectedCards.Contains(valueTuple.Card));

		public bool CanConfirm => _selectedCards.Count >= Prefs.MinSelect && _selectedCards.Count <= Prefs.MaxSelect;

		public void Select(CardModel card)
		{
			if (_selectedCards.Contains(card))
			{
				return;
			}
			if (Mode == "UpgradeSelect" && _selectedCards.Count > 0)
			{
				_selectedCards.Clear();
			}
			else if (_selectedCards.Count >= Prefs.MaxSelect && _selectedCards.Count > 0)
			{
				_selectedCards.RemoveAt(_selectedCards.Count - 1);
			}
			_selectedCards.Add(card);
		}
	}

	public static CombatTrainingSimulatorChoiceBridge Instance { get; } = new CombatTrainingSimulatorChoiceBridge();

	private PendingCardSelection? _pendingSelection;

	private CombatTrainingSimulatorChoiceBridge()
	{
	}

	public string BackendKind => "simulator";

	public bool IsSelectionActive => _pendingSelection != null;

	public bool RequiresFrameSync => false;

	public void Reset()
	{
		_pendingSelection?.CompletionSource.TrySetResult(Array.Empty<CardModel>());
		_pendingSelection = null;
	}

	public void RegisterHandSelection(IEnumerable<CardModel> options, CardSelectorPrefs prefs, string mode)
	{
		_pendingSelection = new PendingCardSelection(options.ToList(), prefs, mode, isHandSelection: true);
	}

	public void RegisterCardSelection(IEnumerable<CardModel> options, CardSelectorPrefs prefs, string mode)
	{
		_pendingSelection = new PendingCardSelection(options.ToList(), prefs, mode, isHandSelection: false);
	}

	public CombatTrainingHandSelectionSnapshot? BuildHandSelectionSnapshot(CombatState? combatState)
	{
		PendingCardSelection? selection = _pendingSelection;
		if (selection == null || !selection.IsHandSelection)
		{
			return null;
		}
		combatState ??= CombatManager.Instance.DebugOnlyGetState();
		return new CombatTrainingHandSelectionSnapshot
		{
			ChoiceAdapterKind = BackendKind,
			IsBackendAvailable = true,
			Mode = selection.Mode,
			PromptText = CombatTrainingChoiceSnapshotBuilder.GetPromptText(selection.Prefs.Prompt),
			MinSelect = selection.Prefs.MinSelect,
			MaxSelect = selection.Prefs.MaxSelect,
			CanConfirm = selection.CanConfirm,
			Cancelable = selection.Prefs.Cancelable,
			SelectableCards = selection.SelectableCards.Select((CardModel card) => CombatTrainingChoiceSnapshotBuilder.BuildHandCardSnapshot(card, combatState)).ToList(),
			SelectedCards = selection.SelectedCards.Select((CardModel card) => CombatTrainingChoiceSnapshotBuilder.BuildHandCardSnapshot(card, combatState)).ToList()
		};
	}

	public CombatTrainingCardSelectionSnapshot? BuildCardSelectionSnapshot(CombatState? combatState)
	{
		PendingCardSelection? selection = _pendingSelection;
		if (selection == null || selection.IsHandSelection)
		{
			return null;
		}
		return new CombatTrainingCardSelectionSnapshot
		{
			ChoiceAdapterKind = BackendKind,
			IsBackendAvailable = true,
			Mode = selection.Mode,
			PromptText = CombatTrainingChoiceSnapshotBuilder.GetPromptText(selection.Prefs.Prompt),
			MinSelect = selection.Prefs.MinSelect,
			MaxSelect = selection.Prefs.MaxSelect,
			CanConfirm = selection.CanConfirm,
			Cancelable = selection.Prefs.Cancelable,
			SelectableCards = selection.SelectableChoices.Select((valueTuple) => CombatTrainingChoiceSnapshotBuilder.BuildSelectableCardSnapshot(valueTuple.Card, valueTuple.ChoiceIndex)).ToList(),
			SelectedCards = selection.Options.Select((CardModel card, int index) => new { Card = card, ChoiceIndex = index }).Where((entry) => selection.SelectedCards.Contains(entry.Card)).Select((entry) => CombatTrainingChoiceSnapshotBuilder.BuildSelectableCardSnapshot(entry.Card, entry.ChoiceIndex)).ToList()
		};
	}

	public bool TrySelectHandCard(int handIndex, out string error)
	{
		error = "";
		PendingCardSelection? selection = _pendingSelection;
		if (selection == null || !selection.IsHandSelection)
		{
			error = "Hand card selection is not active.";
			return false;
		}
		CardModel? card = selection.SelectableCards.FirstOrDefault((CardModel candidate) => CombatTrainingChoiceSnapshotBuilder.GetHandIndex(candidate) == handIndex);
		if (card == null)
		{
			error = $"Hand index {handIndex} is not selectable in the current hand selection prompt.";
			return false;
		}
		selection.Select(card);
		return true;
	}

	public bool TrySelectCardChoice(int choiceIndex, out string error)
	{
		error = "";
		PendingCardSelection? selection = _pendingSelection;
		if (selection == null || selection.IsHandSelection)
		{
			error = "Card selection is not active.";
			return false;
		}
		(int ChoiceIndex, CardModel Card) choice = selection.SelectableChoices.FirstOrDefault((valueTuple) => valueTuple.ChoiceIndex == choiceIndex);
		if (choice.Card == null)
		{
			error = $"Choice index {choiceIndex} is not selectable in the current card selection prompt.";
			return false;
		}
		selection.Select(choice.Card);
		return true;
	}

	public bool TryConfirmSelection(out string error)
	{
		error = "";
		PendingCardSelection? selection = _pendingSelection;
		if (selection == null)
		{
			error = "Selection is not active.";
			return false;
		}
		if (!selection.CanConfirm)
		{
			error = "Current selection cannot be confirmed yet.";
			return false;
		}
		selection.CompletionSource.TrySetResult(selection.SelectedCards.ToList());
		_pendingSelection = null;
		return true;
	}

	public bool TryCancelSelection(out string error)
	{
		error = "";
		PendingCardSelection? selection = _pendingSelection;
		if (selection == null)
		{
			error = "Selection is not active.";
			return false;
		}
		if (!selection.Prefs.Cancelable)
		{
			error = "Current selection cannot be canceled.";
			return false;
		}
		selection.CompletionSource.TrySetResult(Array.Empty<CardModel>());
		_pendingSelection = null;
		return true;
	}

	public async Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
	{
		PendingCardSelection selection = _pendingSelection ?? new PendingCardSelection(options.ToList(), new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, minSelect, maxSelect), "SimpleSelect", isHandSelection: false);
		_pendingSelection ??= selection;
		try
		{
			return await selection.CompletionSource.Task;
		}
		finally
		{
			if (ReferenceEquals(_pendingSelection, selection))
			{
				_pendingSelection = null;
			}
		}
	}

	public CardModel? GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
	{
		return options.FirstOrDefault()?.Card;
	}
}
