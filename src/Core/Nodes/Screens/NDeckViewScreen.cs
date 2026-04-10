using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NDeckViewScreen : NCardsViewScreen
{
	private Player _player;

	private CardPile _pile;

	private NCardViewSortButton _obtainedSorter;

	private NCardViewSortButton _typeSorter;

	private NCardViewSortButton _costSorter;

	private NCardViewSortButton _alphabetSorter;

	private Control _bg;

	private readonly List<SortingOrders> _sortingPriority = new List<SortingOrders>
	{
		SortingOrders.Ascending,
		SortingOrders.TypeAscending,
		SortingOrders.CostAscending,
		SortingOrders.AlphabetAscending
	};

	private static string ScenePath => SceneHelper.GetScenePath("screens/deck_view_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public override NetScreenType ScreenType => NetScreenType.DeckView;

	public static NDeckViewScreen? ShowScreen(Player player)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NDeckViewScreen nDeckViewScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NDeckViewScreen>(PackedScene.GenEditState.Disabled);
		nDeckViewScreen._player = player;
		NDebugAudioManager.Instance?.Play("map_open.mp3");
		NCapstoneContainer.Instance.Open(nDeckViewScreen);
		return nDeckViewScreen;
	}

	public override void _Ready()
	{
		_cards = _pile.Cards.ToList();
		_infoText = new LocString("gameplay_ui", "DECK_PILE_INFO");
		_bg = GetNode<Control>("%SortingBg");
		_obtainedSorter = GetNode<NCardViewSortButton>("%ObtainedSorter");
		_typeSorter = GetNode<NCardViewSortButton>("%CardTypeSorter");
		_costSorter = GetNode<NCardViewSortButton>("%CostSorter");
		_alphabetSorter = GetNode<NCardViewSortButton>("%AlphabeticalSorter");
		_obtainedSorter.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnObtainedSort));
		_typeSorter.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnCardTypeSort));
		_costSorter.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnCostSort));
		_alphabetSorter.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnAlphabetSort));
		_obtainedSorter.SetLabel(new LocString("gameplay_ui", "SORT_OBTAINED").GetRawText());
		_typeSorter.SetLabel(new LocString("gameplay_ui", "SORT_TYPE").GetRawText());
		_costSorter.SetLabel(new LocString("gameplay_ui", "SORT_COST").GetRawText());
		_alphabetSorter.SetLabel(new LocString("gameplay_ui", "SORT_ALPHABET").GetRawText());
		GetNode<MegaLabel>("%ViewUpgradesLabel").SetTextAutoSize(new LocString("gameplay_ui", "VIEW_UPGRADES").GetFormattedText());
		ShaderMaterial shaderMaterial = (ShaderMaterial)_player.Character.CardPool.FrameMaterial;
		_bg.Material = shaderMaterial;
		_obtainedSorter.SetHue(shaderMaterial);
		_typeSorter.SetHue(shaderMaterial);
		_costSorter.SetHue(shaderMaterial);
		_alphabetSorter.SetHue(shaderMaterial);
		ConnectSignals();
		DisplayCards();
		Control[] array = new Control[4] { _obtainedSorter, _typeSorter, _costSorter, _alphabetSorter };
		for (int i = 0; i < array.Length; i++)
		{
			array[i].FocusNeighborTop = array[i].GetPath();
			array[i].FocusNeighborBottom = ((_grid.DefaultFocusedControl != null) ? _grid.DefaultFocusedControl.GetPath() : array[i].GetPath());
			array[i].FocusNeighborLeft = ((i > 0) ? array[i - 1].GetPath() : array[i].GetPath());
			array[i].FocusNeighborRight = ((i < array.Length - 1) ? array[i + 1].GetPath() : array[i].GetPath());
		}
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		_pile = PileType.Deck.GetPile(_player);
		_pile.ContentsChanged += OnPileContentsChanged;
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_pile.ContentsChanged -= OnPileContentsChanged;
	}

	public override void AfterCapstoneClosed()
	{
		base.AfterCapstoneClosed();
		NRun.Instance?.GlobalUi.TopBar.Deck.ToggleAnimState();
	}

	private void OnPileContentsChanged()
	{
		_cards = _pile.Cards.ToList();
		DisplayCards();
	}

	private void OnObtainedSort(NButton button)
	{
		_sortingPriority.Remove(SortingOrders.Ascending);
		_sortingPriority.Remove(SortingOrders.Descending);
		if (_obtainedSorter.IsDescending)
		{
			_sortingPriority.Insert(0, SortingOrders.Descending);
		}
		else
		{
			_sortingPriority.Insert(0, SortingOrders.Ascending);
		}
		DisplayCards();
	}

	private void OnCardTypeSort(NButton button)
	{
		_sortingPriority.Remove(SortingOrders.TypeAscending);
		_sortingPriority.Remove(SortingOrders.TypeDescending);
		if (_typeSorter.IsDescending)
		{
			_sortingPriority.Insert(0, SortingOrders.TypeDescending);
		}
		else
		{
			_sortingPriority.Insert(0, SortingOrders.TypeAscending);
		}
		DisplayCards();
	}

	private void OnCostSort(NButton button)
	{
		_sortingPriority.Remove(SortingOrders.CostAscending);
		_sortingPriority.Remove(SortingOrders.CostDescending);
		if (_costSorter.IsDescending)
		{
			_sortingPriority.Insert(0, SortingOrders.CostDescending);
		}
		else
		{
			_sortingPriority.Insert(0, SortingOrders.CostAscending);
		}
		DisplayCards();
	}

	private void OnAlphabetSort(NButton button)
	{
		_sortingPriority.Remove(SortingOrders.AlphabetAscending);
		_sortingPriority.Remove(SortingOrders.AlphabetDescending);
		if (_alphabetSorter.IsDescending)
		{
			_sortingPriority.Insert(0, SortingOrders.AlphabetDescending);
		}
		else
		{
			_sortingPriority.Insert(0, SortingOrders.AlphabetAscending);
		}
		DisplayCards();
	}

	private void DisplayCards()
	{
		_grid.YOffset = 100;
		_grid.SetCards(_cards, _pile.Type, _sortingPriority);
		IEnumerable<NGridCardHolder> topRowOfCardNodes = _grid.GetTopRowOfCardNodes();
		if (topRowOfCardNodes == null)
		{
			return;
		}
		foreach (NGridCardHolder item in topRowOfCardNodes)
		{
			item.FocusNeighborTop = _obtainedSorter.GetPath();
		}
	}

	public NDeckViewScreen()
		: base()
	{
	}
}
