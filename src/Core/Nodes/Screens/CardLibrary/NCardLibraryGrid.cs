using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;

public partial class NCardLibraryGrid : NCardGrid
{
	private struct InitialSorter(List<CardPoolModel> cardPoolModels) : IComparer<CardModel>
	{
		private List<CardPoolModel> _cardPoolModels = cardPoolModels;

		public int Compare(CardModel? x, CardModel? y)
		{
			if (x == null)
			{
				if (y != null)
				{
					return -1;
				}
				return 0;
			}
			if (y == null)
			{
				return 1;
			}
			int num = _cardPoolModels.IndexOf(x.Pool).CompareTo(_cardPoolModels.IndexOf(y.Pool));
			if (num != 0)
			{
				return num;
			}
			int num2 = x.Rarity.CompareTo(y.Rarity);
			if (num2 != 0)
			{
				return num2;
			}
			return x.Id.CompareTo(y.Id);
		}
	}

	private readonly List<CardModel> _allCards = new List<CardModel>();

	private HashSet<ModelId> _seenCards;

	private HashSet<CardModel> _unlockedCards;

	private bool _showStats;

	protected override bool IsCardLibrary => true;

	protected override bool CenterGrid => false;

	public bool ShowStats
	{
		get
		{
			return _showStats;
		}
		set
		{
			_showStats = value;
			foreach (NGridCardHolder item in _cardRows.SelectMany((List<NGridCardHolder> r) => r))
			{
				if (item.CardNode != null)
				{
					NCardLibraryStats node = item.GetNode<NCardLibraryStats>("CardLibraryStats");
					node.Visible = _showStats;
				}
			}
		}
	}

	public IEnumerable<CardModel> VisibleCards => _cards;

	public override void _Ready()
	{
		ConnectSignals();
		List<CardPoolModel> cardPoolModels = ModelDb.AllCardPools.ToList();
		foreach (CardModel allCard in ModelDb.AllCards)
		{
			if (allCard.ShouldShowInCardLibrary)
			{
				_allCards.Add(allCard);
			}
		}
		_allCards.Sort(new InitialSorter(cardPoolModels));
		RefreshVisibility();
	}

	public void RefreshVisibility()
	{
		_seenCards = SaveManager.Instance.Progress.DiscoveredCards.ToHashSet();
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		_unlockedCards = ModelDb.AllCardPools.Select((CardPoolModel p) => p.GetUnlockedCards(unlockState, CardMultiplayerConstraint.None)).SelectMany((IEnumerable<CardModel> c) => c).ToHashSet();
	}

	public void FilterCards(Func<CardModel, bool> filter)
	{
		int num = 1;
		List<SortingOrders> list = new List<SortingOrders>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<SortingOrders> span = CollectionsMarshal.AsSpan(list);
		int index = 0;
		span[index] = SortingOrders.AlphabetAscending;
		FilterCards(filter, list);
	}

	public void FilterCards(Func<CardModel, bool> filter, List<SortingOrders> sortingPriority)
	{
		List<CardModel> cards = _allCards.Where(filter).ToList();
		DisplayCards(cards, sortingPriority);
	}

	private void DisplayCards(List<CardModel> cards, List<SortingOrders> sortingPriority)
	{
		SetCards(cards, PileType.None, sortingPriority, Task.CompletedTask);
	}

	protected override void InitGrid()
	{
		base.InitGrid();
		foreach (NGridCardHolder item in _cardRows.SelectMany((List<NGridCardHolder> r) => r))
		{
			if (item.CardNode != null)
			{
				CardModel model = item.CardNode.Model;
				bool flag = _seenCards.Contains(model.Id);
				item.EnsureCardLibraryStatsExists();
				NCardLibraryStats cardLibraryStats = item.CardLibraryStats;
				cardLibraryStats.UpdateStats(item.CardNode.Model);
				cardLibraryStats.Visible = ShowStats;
				item.Hitbox.MouseDefaultCursorShape = (CursorShape)(flag ? 16 : 0);
			}
		}
	}

	protected override void AssignCardsToRow(List<NGridCardHolder> row, int startIndex)
	{
		base.AssignCardsToRow(row, startIndex);
		foreach (NGridCardHolder item in row)
		{
			if (item.CardNode != null)
			{
				CardModel model = item.CardNode.Model;
				bool flag = _seenCards.Contains(model.Id);
				item.CardLibraryStats.UpdateStats(model);
				item.Hitbox.MouseDefaultCursorShape = (CursorShape)(flag ? 16 : 0);
			}
		}
	}

	protected override ModelVisibility GetCardVisibility(CardModel card)
	{
		if (!_unlockedCards.Contains(card))
		{
			return ModelVisibility.Locked;
		}
		if (!_seenCards.Contains(card.Id))
		{
			return ModelVisibility.NotSeen;
		}
		return ModelVisibility.Visible;
	}

	protected override void UpdateGridNavigation()
	{
		for (int i = 0; i < _cardRows.Count; i++)
		{
			for (int j = 0; j < _cardRows[i].Count; j++)
			{
				NCardHolder nCardHolder = _cardRows[i][j];
				nCardHolder.FocusNeighborLeft = ((j > 0) ? _cardRows[i][j - 1].GetPath() : null);
				nCardHolder.FocusNeighborRight = ((j < _cardRows[i].Count - 1) ? _cardRows[i][j + 1].GetPath() : null);
				nCardHolder.FocusNeighborTop = ((i > 0) ? _cardRows[i - 1][j].GetPath() : null);
				nCardHolder.FocusNeighborBottom = ((i < _cardRows.Count - 1 && j < _cardRows[i + 1].Count) ? _cardRows[i + 1][j].GetPath() : null);
			}
		}
	}
}
