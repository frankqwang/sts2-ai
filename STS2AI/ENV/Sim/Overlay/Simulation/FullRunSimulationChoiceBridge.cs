using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Simulation;

internal sealed class FullRunPendingRewardSelectionSnapshot
{
	public bool CanProceed { get; init; }

	public IReadOnlyList<Reward> Rewards { get; init; } = Array.Empty<Reward>();
}

internal sealed class FullRunPendingCardRewardSnapshot
{
	public bool CanSkip { get; init; }

	public IReadOnlyList<CardCreationResult> Options { get; init; } = Array.Empty<CardCreationResult>();
}

internal sealed class FullRunPendingRelicSelectionSnapshot
{
	public bool CanSkip { get; init; }

	public IReadOnlyList<RelicModel> Relics { get; init; } = Array.Empty<RelicModel>();
}

public sealed class FullRunPendingRewardRestoreEntrySnapshot
{
	public RewardType RewardType { get; init; }

	public SerializableReward SerializableReward { get; init; } = new SerializableReward();

	public ModelId PotionId { get; init; } = ModelId.none;

	public ModelId RelicId { get; init; } = ModelId.none;

	public IReadOnlyList<SerializableCard> ExactCardOptions { get; init; } = Array.Empty<SerializableCard>();

	public bool CardRewardCanSkip { get; init; }

	public bool CardRewardCanReroll { get; init; }
}

public sealed class FullRunPendingRewardSelectionRestoreSnapshot
{
	public bool CanProceed { get; init; }

	public IReadOnlyList<FullRunPendingRewardRestoreEntrySnapshot> Rewards { get; init; } = Array.Empty<FullRunPendingRewardRestoreEntrySnapshot>();
}

public sealed class FullRunPendingCardRewardRestoreSnapshot
{
	public bool CanSkip { get; init; }

	public IReadOnlyList<SerializableCard> Options { get; init; } = Array.Empty<SerializableCard>();
}

public sealed class FullRunPendingSelectionRestoreSnapshot
{
	public FullRunPendingRewardSelectionRestoreSnapshot? RewardSelection { get; init; }

	public FullRunPendingCardRewardRestoreSnapshot? CardRewardSelection { get; init; }
}

internal sealed class FullRunChoiceBridgeSnapshotCache
{
	public CombatTrainingHandSelectionSnapshot? HandSelection { get; init; }

	public CombatTrainingCardSelectionSnapshot? CardSelection { get; init; }

	public FullRunPendingRewardSelectionSnapshot? RewardSelection { get; init; }

	public FullRunPendingCardRewardSnapshot? CardRewardSelection { get; init; }

	public FullRunPendingRelicSelectionSnapshot? RelicSelection { get; init; }
}

internal sealed class FullRunSimulationChoiceBridge : ICombatChoiceAdapter, IHandSelectionCardSelector, ICardSelectionPromptSelector, IRewardSelectionPromptSelector, ICardRewardPromptSelector, IRelicSelectionPromptSelector
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

	private sealed class PendingRewardSelection
	{
		public IReadOnlyList<Reward> Rewards { get; }

		public bool CanProceed { get; }

		public TaskCompletionSource<Reward?> CompletionSource { get; } = new TaskCompletionSource<Reward?>();

		public PendingRewardSelection(IReadOnlyList<Reward> rewards, bool canProceed)
		{
			Rewards = rewards;
			CanProceed = canProceed;
		}
	}

	private sealed class PendingCardRewardSelection
	{
		public IReadOnlyList<CardCreationResult> Options { get; }

		public IReadOnlyList<CardRewardAlternative> Alternatives { get; }

		public bool CanSkip { get; }

		public TaskCompletionSource<CardModel?> CompletionSource { get; } = new TaskCompletionSource<CardModel?>();

		public PendingCardRewardSelection(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives, bool canSkip)
		{
			Options = options;
			Alternatives = alternatives;
			CanSkip = canSkip;
		}
	}

	private sealed class PendingRelicSelection
	{
		public IReadOnlyList<RelicModel> Relics { get; }

		public bool CanSkip { get; }

		public TaskCompletionSource<RelicModel?> CompletionSource { get; } = new TaskCompletionSource<RelicModel?>();

		public PendingRelicSelection(IReadOnlyList<RelicModel> relics, bool canSkip)
		{
			Relics = relics;
			CanSkip = canSkip;
		}
	}

	public static FullRunSimulationChoiceBridge Instance { get; } = new FullRunSimulationChoiceBridge();

	private PendingCardSelection? _pendingCardSelection;

	private PendingRewardSelection? _pendingRewardSelection;

	private PendingCardRewardSelection? _pendingCardRewardSelection;

	private PendingRelicSelection? _pendingRelicSelection;

	private FullRunSimulationChoiceBridge()
	{
	}

	public string BackendKind => "full_run_simulator";

	public bool IsSelectionActive => _pendingCardSelection != null || _pendingRewardSelection != null || _pendingCardRewardSelection != null || _pendingRelicSelection != null;

	public bool RequiresFrameSync => false;

	public bool IsRewardSelectionActive => _pendingRewardSelection != null;

	public bool IsCardRewardSelectionActive => _pendingCardRewardSelection != null;

	public bool IsRelicSelectionActive => _pendingRelicSelection != null;

	public bool IsHandSelectionActive => _pendingCardSelection?.IsHandSelection == true;

	public bool IsCardSelectionActive => _pendingCardSelection != null && !_pendingCardSelection.IsHandSelection;

	public bool IsCardOrHandSelectionActive => _pendingCardSelection != null;

	public void Reset()
	{
		_pendingCardSelection?.CompletionSource.TrySetResult(Array.Empty<CardModel>());
		_pendingCardSelection = null;
		_pendingRewardSelection?.CompletionSource.TrySetResult(null);
		_pendingRewardSelection = null;
		_pendingCardRewardSelection?.CompletionSource.TrySetResult(null);
		_pendingCardRewardSelection = null;
		_pendingRelicSelection?.CompletionSource.TrySetResult(null);
		_pendingRelicSelection = null;
	}

	public void RegisterHandSelection(IEnumerable<CardModel> options, CardSelectorPrefs prefs, string mode)
	{
		_pendingCardSelection = new PendingCardSelection(options.ToList(), prefs, mode, isHandSelection: true);
	}

	public void RegisterCardSelection(IEnumerable<CardModel> options, CardSelectorPrefs prefs, string mode)
	{
		_pendingCardSelection = new PendingCardSelection(options.ToList(), prefs, mode, isHandSelection: false);
	}

	public CombatTrainingHandSelectionSnapshot? BuildHandSelectionSnapshot(CombatState? combatState)
	{
		PendingCardSelection? selection = _pendingCardSelection;
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
		PendingCardSelection? selection = _pendingCardSelection;
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

	public FullRunPendingRewardSelectionSnapshot? BuildRewardSelectionSnapshot()
	{
		PendingRewardSelection? selection = _pendingRewardSelection;
		if (selection == null)
		{
			return null;
		}
		return new FullRunPendingRewardSelectionSnapshot
		{
			CanProceed = selection.CanProceed,
			Rewards = selection.Rewards
		};
	}

	public FullRunPendingCardRewardSnapshot? BuildCardRewardSelectionSnapshot()
	{
		PendingCardRewardSelection? selection = _pendingCardRewardSelection;
		if (selection == null)
		{
			return null;
		}
		return new FullRunPendingCardRewardSnapshot
		{
			CanSkip = selection.CanSkip,
			Options = selection.Options
		};
	}

	public FullRunPendingRelicSelectionSnapshot? BuildRelicSelectionSnapshot()
	{
		PendingRelicSelection? selection = _pendingRelicSelection;
		if (selection == null)
		{
			return null;
		}
		return new FullRunPendingRelicSelectionSnapshot
		{
			CanSkip = selection.CanSkip,
			Relics = selection.Relics
		};
	}

	public FullRunChoiceBridgeSnapshotCache CaptureSnapshots(CombatState? combatState = null)
	{
		CombatTrainingHandSelectionSnapshot? handSelection = null;
		CombatTrainingCardSelectionSnapshot? cardSelection = null;
		if (_pendingCardSelection != null)
		{
			combatState ??= CombatManager.Instance.DebugOnlyGetState();
			handSelection = BuildHandSelectionSnapshot(combatState);
			cardSelection = BuildCardSelectionSnapshot(combatState);
		}

		return new FullRunChoiceBridgeSnapshotCache
		{
			HandSelection = handSelection,
			CardSelection = cardSelection,
			RewardSelection = IsRewardSelectionActive ? BuildRewardSelectionSnapshot() : null,
			CardRewardSelection = IsCardRewardSelectionActive ? BuildCardRewardSelectionSnapshot() : null,
			RelicSelection = IsRelicSelectionActive ? BuildRelicSelectionSnapshot() : null
		};
	}

	public FullRunPendingSelectionRestoreSnapshot? CapturePendingSelection()
	{
		PendingRewardSelection? rewardSelection = _pendingRewardSelection;
		if (rewardSelection != null)
		{
			return new FullRunPendingSelectionRestoreSnapshot
			{
				RewardSelection = new FullRunPendingRewardSelectionRestoreSnapshot
				{
					CanProceed = rewardSelection.CanProceed,
					Rewards = rewardSelection.Rewards.Select(CaptureRewardRestoreEntry).ToList()
				}
			};
		}

		PendingCardRewardSelection? cardRewardSelection = _pendingCardRewardSelection;
		if (cardRewardSelection != null)
		{
			return new FullRunPendingSelectionRestoreSnapshot
			{
				CardRewardSelection = new FullRunPendingCardRewardRestoreSnapshot
				{
					CanSkip = cardRewardSelection.CanSkip,
					Options = cardRewardSelection.Options.Select(static option => option.Card.ToSerializable()).ToList()
				}
			};
		}

		return null;
	}

	public void RestorePendingSelection(FullRunPendingSelectionRestoreSnapshot snapshot, Player player)
	{
		if (snapshot.RewardSelection != null)
		{
			_pendingRewardSelection = new PendingRewardSelection(
				snapshot.RewardSelection.Rewards.Select((entry) => RestoreReward(entry, player)).ToList(),
				snapshot.RewardSelection.CanProceed);
		}

		if (snapshot.CardRewardSelection != null)
		{
			_pendingCardRewardSelection = new PendingCardRewardSelection(
				snapshot.CardRewardSelection.Options.Select((card) => new CardCreationResult(RestoreRewardCard(card, player))).ToList(),
				Array.Empty<CardRewardAlternative>(),
				snapshot.CardRewardSelection.CanSkip);
		}
	}

	public bool TrySelectHandCard(int handIndex, out string error)
	{
		error = "";
		PendingCardSelection? selection = _pendingCardSelection;
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
		PendingCardSelection? selection = _pendingCardSelection;
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
		PendingCardSelection? selection = _pendingCardSelection;
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
		_pendingCardSelection = null;
		return true;
	}

	public bool TryCancelSelection(out string error)
	{
		error = "";
		PendingCardSelection? selection = _pendingCardSelection;
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
		_pendingCardSelection = null;
		return true;
	}

	public bool TrySelectReward(int index, out string error)
	{
		error = "";
		PendingRewardSelection? selection = _pendingRewardSelection;
		if (selection == null)
		{
			error = "Reward selection is not active.";
			return false;
		}
		if (index < 0 || index >= selection.Rewards.Count)
		{
			error = $"Reward index {index} is out of range.";
			return false;
		}
		// Clear _pendingRewardSelection BEFORE TrySetResult to avoid nullifying the next
		// pending selection set up by inline continuations (e.g. RewardsSet.Offer loop
		// calling GetSelectedRewardAsync for the remaining rewards immediately).
		_pendingRewardSelection = null;
		selection.CompletionSource.TrySetResult(selection.Rewards[index]);
		return true;
	}

	public bool TryProceedRewards(out string error)
	{
		error = "";
		PendingRewardSelection? selection = _pendingRewardSelection;
		if (selection == null)
		{
			error = "Reward selection is not active.";
			return false;
		}
		if (!selection.CanProceed)
		{
			error = "Reward selection cannot proceed yet.";
			return false;
		}
		_pendingRewardSelection = null;
		selection.CompletionSource.TrySetResult(null);
		return true;
	}

	public bool TrySelectCardReward(int index, out string error)
	{
		error = "";
		PendingCardRewardSelection? selection = _pendingCardRewardSelection;
		if (selection == null)
		{
			error = "Card reward selection is not active.";
			return false;
		}
		if (index < 0 || index >= selection.Options.Count)
		{
			error = $"Card reward index {index} is out of range.";
			return false;
		}
		_pendingCardRewardSelection = null;
		selection.CompletionSource.TrySetResult(selection.Options[index].Card);
		return true;
	}

	public bool TrySkipCardReward(out string error)
	{
		error = "";
		PendingCardRewardSelection? selection = _pendingCardRewardSelection;
		if (selection == null)
		{
			error = "Card reward selection is not active.";
			return false;
		}
		if (!selection.CanSkip)
		{
			error = "Card reward selection cannot be skipped.";
			return false;
		}
		_pendingCardRewardSelection = null;
		selection.CompletionSource.TrySetResult(null);
		return true;
	}

	public bool TrySelectRelic(int index, out string error)
	{
		error = "";
		PendingRelicSelection? selection = _pendingRelicSelection;
		if (selection == null)
		{
			error = "Relic selection is not active.";
			return false;
		}
		if (index < 0 || index >= selection.Relics.Count)
		{
			error = $"Relic selection index {index} is out of range.";
			return false;
		}
		_pendingRelicSelection = null;
		selection.CompletionSource.TrySetResult(selection.Relics[index]);
		return true;
	}

	public bool TrySkipRelicSelection(out string error)
	{
		error = "";
		PendingRelicSelection? selection = _pendingRelicSelection;
		if (selection == null)
		{
			error = "Relic selection is not active.";
			return false;
		}
		if (!selection.CanSkip)
		{
			error = "Relic selection cannot be skipped.";
			return false;
		}
		_pendingRelicSelection = null;
		selection.CompletionSource.TrySetResult(null);
		return true;
	}

	public async Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
	{
		PendingCardSelection selection = _pendingCardSelection ?? new PendingCardSelection(options.ToList(), new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, minSelect, maxSelect), "SimpleSelect", isHandSelection: false);
		_pendingCardSelection ??= selection;
		try
		{
			return await selection.CompletionSource.Task;
		}
		finally
		{
			if (ReferenceEquals(_pendingCardSelection, selection))
			{
				_pendingCardSelection = null;
			}
		}
	}

	public CardModel? GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
	{
		return options.FirstOrDefault()?.Card;
	}

	public async Task<Reward?> GetSelectedRewardAsync(IReadOnlyList<Reward> rewards, bool canProceed)
	{
		PendingRewardSelection selection = new PendingRewardSelection(rewards.ToList(), canProceed);
		_pendingRewardSelection = selection;
		try
		{
			return await selection.CompletionSource.Task;
		}
		finally
		{
			if (ReferenceEquals(_pendingRewardSelection, selection))
			{
				_pendingRewardSelection = null;
			}
		}
	}

	public async Task<CardModel?> GetSelectedCardRewardAsync(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives, bool canSkip)
	{
		PendingCardRewardSelection selection = new PendingCardRewardSelection(options.ToList(), alternatives.ToList(), canSkip);
		_pendingCardRewardSelection = selection;
		try
		{
			return await selection.CompletionSource.Task;
		}
		finally
		{
			if (ReferenceEquals(_pendingCardRewardSelection, selection))
			{
				_pendingCardRewardSelection = null;
			}
		}
	}

	public async Task<RelicModel?> GetSelectedRelicAsync(IReadOnlyList<RelicModel> relics, bool canSkip)
	{
		PendingRelicSelection selection = new PendingRelicSelection(relics.ToList(), canSkip);
		_pendingRelicSelection = selection;
		try
		{
			return await selection.CompletionSource.Task;
		}
		finally
		{
			if (ReferenceEquals(_pendingRelicSelection, selection))
			{
				_pendingRelicSelection = null;
			}
		}
	}

	private static FullRunPendingRewardRestoreEntrySnapshot CaptureRewardRestoreEntry(Reward reward)
	{
		switch (reward)
		{
			case CardReward cardReward:
				return new FullRunPendingRewardRestoreEntrySnapshot
				{
					RewardType = RewardType.Card,
					SerializableReward = new SerializableReward
					{
						RewardType = RewardType.Card,
						Source = FullRunUpstreamCompat.GetCardRewardSource(cardReward)
					},
					ExactCardOptions = FullRunUpstreamCompat.GetCardRewardOptions(cardReward).Select(static option => option.Card.ToSerializable()).ToList(),
					CardRewardCanSkip = cardReward.CanSkip,
					CardRewardCanReroll = cardReward.CanReroll
				};
			case PotionReward potionReward:
				return new FullRunPendingRewardRestoreEntrySnapshot
				{
					RewardType = RewardType.Potion,
					SerializableReward = potionReward.ToSerializable(),
					PotionId = potionReward.Potion?.Id ?? throw new InvalidOperationException("Potion reward must be populated before capture.")
				};
			case RelicReward relicReward:
				return new FullRunPendingRewardRestoreEntrySnapshot
				{
					RewardType = RewardType.Relic,
					SerializableReward = relicReward.ToSerializable(),
					RelicId = FullRunUpstreamCompat.GetRewardRelic(relicReward)?.Id ?? throw new InvalidOperationException("Relic reward must be populated before capture.")
				};
			default:
				return new FullRunPendingRewardRestoreEntrySnapshot
				{
					RewardType = reward.ToSerializable().RewardType,
					SerializableReward = reward.ToSerializable()
				};
		}
	}

	private static Reward RestoreReward(FullRunPendingRewardRestoreEntrySnapshot snapshot, Player player)
	{
		switch (snapshot.RewardType)
		{
			case RewardType.Potion:
				if (snapshot.PotionId == ModelId.none)
				{
					throw new InvalidOperationException("Potion reward snapshot is missing potion id.");
				}
				return new PotionReward(ModelDb.GetById<PotionModel>(snapshot.PotionId).ToMutable(), player);
			case RewardType.Relic:
				if (snapshot.RelicId == ModelId.none)
				{
					throw new InvalidOperationException("Relic reward snapshot is missing relic id.");
				}
				return new RelicReward(ModelDb.GetById<RelicModel>(snapshot.RelicId).ToMutable(), player);
			case RewardType.Card:
				if (snapshot.ExactCardOptions.Count == 0)
				{
					return Reward.FromSerializable(snapshot.SerializableReward, player);
				}
				List<CardModel> cards = snapshot.ExactCardOptions.Select((card) => RestoreRewardCard(card, player)).ToList();
				return new CardReward(cards, snapshot.SerializableReward.Source, player)
				{
					CanSkip = snapshot.CardRewardCanSkip,
					CanReroll = snapshot.CardRewardCanReroll
				};
			default:
				return Reward.FromSerializable(snapshot.SerializableReward, player);
		}
	}

	private static CardModel RestoreRewardCard(SerializableCard cardSnapshot, Player player)
	{
		CardModel card = CardModel.FromSerializable(cardSnapshot);
		// Reward options must belong to the RunState so the later
		// CardPileCmd.Add(..., PileType.Deck) path can succeed, but they must not be
		// inserted into the actual deck pile until a branch really chooses them.
		player.RunState.AddCard(card, player);
		return card;
	}

	internal static List<Reward> RestoreRewardsForOffer(FullRunPendingRewardSelectionRestoreSnapshot snapshot, Player player)
	{
		return snapshot.Rewards.Select((entry) => RestoreReward(entry, player)).ToList();
	}

	internal static CardReward RestoreCardRewardForOffer(FullRunPendingCardRewardRestoreSnapshot snapshot, Player player, CardCreationSource source = CardCreationSource.Encounter)
	{
		List<CardModel> cards = snapshot.Options.Select((card) => RestoreRewardCard(card, player)).ToList();
		return new CardReward(cards, source, player)
		{
			CanSkip = snapshot.CanSkip
		};
	}
}
