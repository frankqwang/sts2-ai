using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.TestSupport;

public class TestCardSelector : ICardSelector
{
	public delegate CardModel? CardRewardSelectionDelegate(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives);

	private TaskCompletionSource<IEnumerable<CardModel>>? _cardsToSelectTask;

	private TaskCompletionSource<IEnumerable<int>>? _indicesToSelectTask;

	private CardRewardSelectionDelegate? _cardRewardSelectionDelegate;

	private bool _shouldBlock;

	public void Cleanup()
	{
		_cardsToSelectTask = null;
		_indicesToSelectTask = null;
		_shouldBlock = false;
		_cardRewardSelectionDelegate = null;
	}

	public void SetupForAsyncCardSelection()
	{
		if (_indicesToSelectTask != null || _cardsToSelectTask != null)
		{
			throw new InvalidOperationException("Can only set up once!");
		}
		_cardsToSelectTask = new TaskCompletionSource<IEnumerable<CardModel>>();
	}

	public void SetupForAsyncIndexSelection()
	{
		if (_indicesToSelectTask != null || _cardsToSelectTask != null)
		{
			throw new InvalidOperationException("Can only set up once!");
		}
		_indicesToSelectTask = new TaskCompletionSource<IEnumerable<int>>();
	}

	public void PrepareToSelect(IEnumerable<CardModel> cards)
	{
		if (_cardsToSelectTask == null)
		{
			_cardsToSelectTask = new TaskCompletionSource<IEnumerable<CardModel>>();
		}
		_cardsToSelectTask.SetResult(cards);
	}

	public void PrepareToSelect(IEnumerable<int> indices)
	{
		if (_indicesToSelectTask == null)
		{
			_indicesToSelectTask = new TaskCompletionSource<IEnumerable<int>>();
		}
		_indicesToSelectTask.SetResult(indices);
	}

	public void PrepareToSelectCardReward(CardRewardSelectionDelegate del)
	{
		_cardRewardSelectionDelegate = del;
	}

	public CardModel? GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
	{
		if (_cardRewardSelectionDelegate != null)
		{
			return _cardRewardSelectionDelegate?.Invoke(options, alternatives);
		}
		return options.FirstOrDefault()?.Card;
	}

	public void PrepareToBlock()
	{
		_shouldBlock = true;
	}

	public async Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
	{
		if (_shouldBlock)
		{
			await Task.Delay(5000);
			throw new InvalidOperationException("Test told us to block, but it did not finish within 5 seconds!");
		}
		if (_cardsToSelectTask != null)
		{
			IEnumerable<CardModel> enumerable = await _cardsToSelectTask.Task;
			if (enumerable.Any((CardModel c) => !options.Contains(c)))
			{
				throw new InvalidOperationException("Selected card missing from options.");
			}
			return enumerable;
		}
		if (_indicesToSelectTask != null)
		{
			return (await _indicesToSelectTask.Task).Select(options.ElementAt);
		}
		return Array.Empty<CardModel>();
	}
}
