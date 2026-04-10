using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.CardPools;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;

public sealed partial class NCardLibrary : NSubmenu
{
	private const int _delayAfterTextFilterChangedMsec = 250;

	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/card_library/card_library");

	private readonly LocString _cardCountLocString = new LocString("card_library", "CARD_COUNT");

	private readonly LocString _noResultsLocString = new LocString("card_library", "NO_RESULTS");

	private IRunState? _runState;

	private NCardLibraryGrid _grid;

	private NSearchBar _searchBar;

	private readonly Dictionary<string, Func<CardModel, bool>> _specialSearchbarKeywords = new Dictionary<string, Func<CardModel, bool>>();

	private readonly Dictionary<CharacterModel, NCardPoolFilter> _cardPoolFilters = new Dictionary<CharacterModel, NCardPoolFilter>();

	private NCardPoolFilter _ironcladFilter;

	private NCardPoolFilter _silentFilter;

	private NCardPoolFilter _defectFilter;

	private NCardPoolFilter _regentFilter;

	private NCardPoolFilter _necrobinderFilter;

	private NCardPoolFilter _colorlessFilter;

	private NCardPoolFilter _ancientsFilter;

	private NCardPoolFilter _miscPoolFilter;

	private readonly Dictionary<NCardPoolFilter, Func<CardModel, bool>> _poolFilters = new Dictionary<NCardPoolFilter, Func<CardModel, bool>>();

	private NCardViewSortButton _typeSorter;

	private NCardTypeTickbox _attackFilter;

	private NCardTypeTickbox _skillFilter;

	private NCardTypeTickbox _powerFilter;

	private NCardTypeTickbox _otherTypeFilter;

	private readonly Dictionary<NCardTypeTickbox, Func<CardModel, bool>> _cardTypeFilters = new Dictionary<NCardTypeTickbox, Func<CardModel, bool>>();

	private NCardViewSortButton _raritySorter;

	private NCardRarityTickbox _commonFilter;

	private NCardRarityTickbox _uncommonFilter;

	private NCardRarityTickbox _rareFilter;

	private NCardRarityTickbox _otherFilter;

	private readonly Dictionary<NCardRarityTickbox, Func<CardModel, bool>> _rarityFilters = new Dictionary<NCardRarityTickbox, Func<CardModel, bool>>();

	private NCardViewSortButton _costSorter;

	private NCardCostTickbox _zeroFilter;

	private NCardCostTickbox _oneFilter;

	private NCardCostTickbox _twoFilter;

	private NCardCostTickbox _threePlusFilter;

	private NCardCostTickbox _xFilter;

	private readonly Dictionary<NCardCostTickbox, Func<CardModel, bool>> _costFilters = new Dictionary<NCardCostTickbox, Func<CardModel, bool>>();

	private NCardViewSortButton _alphabetSorter;

	private NLibraryStatTickbox _viewMultiplayerCards;

	private NLibraryStatTickbox _viewStats;

	private NLibraryStatTickbox _viewUpgrades;

	private MegaRichTextLabel _cardCountLabel;

	private MegaRichTextLabel _noResultsLabel;

	private CancellationTokenSource? _displayCardsShortDelayCancelToken;

	private readonly List<SortingOrders> _sortingPriority = new List<SortingOrders>
	{
		SortingOrders.RarityAscending,
		SortingOrders.TypeAscending,
		SortingOrders.CostAscending,
		SortingOrders.AlphabetAscending
	};

	private Func<CardModel, bool> _filter = (CardModel _) => true;

	private Control? _lastHoveredControl;

	public static string[] AssetPaths => new string[1] { _scenePath };

	protected override Control InitialFocusedControl => _lastHoveredControl ?? _ironcladFilter;

	public static NCardLibrary? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCardLibrary>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		ConnectSignals();
		_grid = GetNode<NCardLibraryGrid>("%CardGrid");
		_grid.Connect(NCardGrid.SignalName.HolderPressed, Callable.From<NCardHolder>(ShowCardDetail));
		_grid.Connect(NCardGrid.SignalName.HolderAltPressed, Callable.From<NCardHolder>(ShowCardDetail));
		_cardCountLabel = GetNode<MegaRichTextLabel>("%CardCountLabel");
		_noResultsLabel = GetNode<MegaRichTextLabel>("%NoResultsLabel");
		_noResultsLabel.Text = _noResultsLocString.GetFormattedText();
		_searchBar = GetNode<NSearchBar>("%SearchBar");
		_searchBar.Connect(NSearchBar.SignalName.QueryChanged, Callable.From<string>(SearchBarQueryChanged));
		_searchBar.Connect(NSearchBar.SignalName.QuerySubmitted, Callable.From<string>(SearchBarQuerySubmitted));
		_ironcladFilter = GetNode<NCardPoolFilter>("%IroncladPool");
		_silentFilter = GetNode<NCardPoolFilter>("%SilentPool");
		_defectFilter = GetNode<NCardPoolFilter>("%DefectPool");
		_regentFilter = GetNode<NCardPoolFilter>("%RegentPool");
		_necrobinderFilter = GetNode<NCardPoolFilter>("%NecrobinderPool");
		_colorlessFilter = GetNode<NCardPoolFilter>("%ColorlessPool");
		_ancientsFilter = GetNode<NCardPoolFilter>("%AncientsPool");
		_miscPoolFilter = GetNode<NCardPoolFilter>("%MiscPool");
		Callable callable = Callable.From<NCardPoolFilter>(UpdateCardPoolFilter);
		_ironcladFilter.Connect(NCardPoolFilter.SignalName.Toggled, callable);
		_silentFilter.Connect(NCardPoolFilter.SignalName.Toggled, callable);
		_defectFilter.Connect(NCardPoolFilter.SignalName.Toggled, callable);
		_regentFilter.Connect(NCardPoolFilter.SignalName.Toggled, callable);
		_necrobinderFilter.Connect(NCardPoolFilter.SignalName.Toggled, callable);
		_colorlessFilter.Connect(NCardPoolFilter.SignalName.Toggled, callable);
		_ancientsFilter.Connect(NCardPoolFilter.SignalName.Toggled, callable);
		_miscPoolFilter.Connect(NCardPoolFilter.SignalName.Toggled, callable);
		_poolFilters.Add(_ironcladFilter, (CardModel c) => c.Pool is IroncladCardPool);
		_poolFilters.Add(_silentFilter, (CardModel c) => c.Pool is SilentCardPool);
		_poolFilters.Add(_defectFilter, (CardModel c) => c.Pool is DefectCardPool);
		_poolFilters.Add(_regentFilter, (CardModel c) => c.Pool is RegentCardPool);
		_poolFilters.Add(_necrobinderFilter, (CardModel c) => c.Pool is NecrobinderCardPool);
		_poolFilters.Add(_colorlessFilter, (CardModel c) => c.Pool is ColorlessCardPool);
		_poolFilters.Add(_ancientsFilter, (CardModel c) => c.Rarity == CardRarity.Ancient);
		_poolFilters.Add(_miscPoolFilter, delegate(CardModel c)
		{
			CardRarity rarity = c.Rarity;
			return (uint)(rarity - 6) <= 4u;
		});
		_typeSorter = GetNode<NCardViewSortButton>("%CardTypeSorter");
		_typeSorter.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnCardTypeSort));
		_attackFilter = GetNode<NCardTypeTickbox>("%AttackType");
		_skillFilter = GetNode<NCardTypeTickbox>("%SkillType");
		_powerFilter = GetNode<NCardTypeTickbox>("%PowerType");
		_otherTypeFilter = GetNode<NCardTypeTickbox>("%OtherType");
		Callable callable2 = Callable.From<NCardTypeTickbox>(UpdateTypeFilter);
		_attackFilter.Connect(NCardTypeTickbox.SignalName.Toggled, callable2);
		_skillFilter.Connect(NCardTypeTickbox.SignalName.Toggled, callable2);
		_powerFilter.Connect(NCardTypeTickbox.SignalName.Toggled, callable2);
		_otherTypeFilter.Connect(NCardTypeTickbox.SignalName.Toggled, callable2);
		_cardTypeFilters.Add(_attackFilter, (CardModel c) => c.Type == CardType.Attack);
		_cardTypeFilters.Add(_skillFilter, (CardModel c) => c.Type == CardType.Skill);
		_cardTypeFilters.Add(_powerFilter, (CardModel c) => c.Type == CardType.Power);
		_cardTypeFilters.Add(_otherTypeFilter, delegate(CardModel c)
		{
			CardType type = c.Type;
			bool flag = (uint)(type - 1) <= 2u;
			return !flag;
		});
		_raritySorter = GetNode<NCardViewSortButton>("%RaritySorter");
		_raritySorter.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnRaritySort));
		_commonFilter = GetNode<NCardRarityTickbox>("%CommonRarity");
		_uncommonFilter = GetNode<NCardRarityTickbox>("%UncommonRarity");
		_rareFilter = GetNode<NCardRarityTickbox>("%RareRarity");
		_otherFilter = GetNode<NCardRarityTickbox>("%OtherRarity");
		Callable callable3 = Callable.From<NTickbox>(UpdateRarityFilter);
		_commonFilter.Connect(NTickbox.SignalName.Toggled, callable3);
		_uncommonFilter.Connect(NTickbox.SignalName.Toggled, callable3);
		_rareFilter.Connect(NTickbox.SignalName.Toggled, callable3);
		_otherFilter.Connect(NTickbox.SignalName.Toggled, callable3);
		_rarityFilters.Add(_commonFilter, (CardModel c) => c.Rarity == CardRarity.Common);
		_rarityFilters.Add(_uncommonFilter, (CardModel c) => c.Rarity == CardRarity.Uncommon);
		_rarityFilters.Add(_rareFilter, (CardModel c) => c.Rarity == CardRarity.Rare);
		_rarityFilters.Add(_otherFilter, delegate(CardModel c)
		{
			CardRarity rarity = c.Rarity;
			bool flag = (uint)(rarity - 2) <= 2u;
			return !flag;
		});
		_costSorter = GetNode<NCardViewSortButton>("%CostSorter");
		_costSorter.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnCostSort));
		_zeroFilter = GetNode<NCardCostTickbox>("%Cost0");
		_oneFilter = GetNode<NCardCostTickbox>("%Cost1");
		_twoFilter = GetNode<NCardCostTickbox>("%Cost2");
		_threePlusFilter = GetNode<NCardCostTickbox>("%Cost3+");
		_xFilter = GetNode<NCardCostTickbox>("%CostX");
		Callable callable4 = Callable.From<NCardCostTickbox>(UpdateCostFilter);
		_zeroFilter.Connect(NClickableControl.SignalName.Released, callable4);
		_oneFilter.Connect(NClickableControl.SignalName.Released, callable4);
		_twoFilter.Connect(NClickableControl.SignalName.Released, callable4);
		_threePlusFilter.Connect(NClickableControl.SignalName.Released, callable4);
		_xFilter.Connect(NClickableControl.SignalName.Released, callable4);
		_costFilters.Add(_zeroFilter, delegate(CardModel c)
		{
			CardEnergyCost energyCost = c.EnergyCost;
			return energyCost != null && energyCost.Canonical <= 0 && !energyCost.CostsX;
		});
		_costFilters.Add(_oneFilter, (CardModel c) => c.EnergyCost.Canonical == 1);
		_costFilters.Add(_twoFilter, (CardModel c) => c.EnergyCost.Canonical == 2);
		_costFilters.Add(_threePlusFilter, (CardModel c) => c.EnergyCost.Canonical >= 3);
		_costFilters.Add(_xFilter, (CardModel c) => c.EnergyCost.CostsX || c.HasStarCostX);
		_alphabetSorter = GetNode<NCardViewSortButton>("%AlphabetSorter");
		_alphabetSorter.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnAlphabetSort));
		_viewStats = GetNode<NLibraryStatTickbox>("%Stats");
		_viewUpgrades = GetNode<NLibraryStatTickbox>("%Upgrades");
		_viewMultiplayerCards = GetNode<NLibraryStatTickbox>("%MultiplayerCards");
		_viewStats.Connect(NTickbox.SignalName.Toggled, Callable.From<NTickbox>(ToggleShowStats));
		_viewUpgrades.Connect(NTickbox.SignalName.Toggled, Callable.From<NTickbox>(ToggleShowUpgrades));
		_viewMultiplayerCards.Connect(NTickbox.SignalName.Toggled, Callable.From<NTickbox>(ToggleFilterMultiplayerCards));
		_typeSorter.SetLabel(new LocString("gameplay_ui", "SORT_TYPE").GetRawText());
		_raritySorter.SetLabel(new LocString("gameplay_ui", "SORT_RARITY").GetRawText());
		_costSorter.SetLabel(new LocString("gameplay_ui", "SORT_COST").GetRawText());
		_alphabetSorter.SetLabel(new LocString("gameplay_ui", "SORT_ALPHABET").GetRawText());
		_commonFilter.SetLabel(new LocString("card_library", "RARITY_COMMON").GetRawText());
		_uncommonFilter.SetLabel(new LocString("card_library", "RARITY_UNCOMMON").GetRawText());
		_rareFilter.SetLabel(new LocString("card_library", "RARITY_RARE").GetRawText());
		_otherFilter.SetLabel(new LocString("card_library", "RARITY_OTHER").GetRawText());
		_viewStats.SetLabel(new LocString("card_library", "VIEW_STATS").GetRawText());
		_viewUpgrades.SetLabel(new LocString("card_library", "VIEW_UPGRADES").GetRawText());
		_viewMultiplayerCards.SetLabel(new LocString("card_library", "VIEW_MULTIPLAYER_CARDS").GetRawText());
		_colorlessFilter.Loc = new LocString("card_library", "POOL_COLORLESS_TIP");
		_ancientsFilter.Loc = new LocString("card_library", "POOL_ANCIENT_TIP");
		_miscPoolFilter.Loc = new LocString("card_library", "POOL_MISC_TIP");
		_attackFilter.Loc = new LocString("card_library", "TYPE_ATTACK_TIP");
		_skillFilter.Loc = new LocString("card_library", "TYPE_SKILL_TIP");
		_powerFilter.Loc = new LocString("card_library", "TYPE_POWER_TIP");
		_otherTypeFilter.Loc = new LocString("card_library", "TYPE_OTHER_TIP");
		_commonFilter.Loc = new LocString("card_library", "RARITY_COMMON_TIP");
		_uncommonFilter.Loc = new LocString("card_library", "RARITY_UNCOMMON_TIP");
		_rareFilter.Loc = new LocString("card_library", "RARITY_RARE_TIP");
		_otherFilter.Loc = new LocString("card_library", "RARITY_OTHER_TIP");
		_zeroFilter.Loc = new LocString("card_library", "COST_ZERO_TIP");
		_oneFilter.Loc = new LocString("card_library", "COST_ONE_TIP");
		_twoFilter.Loc = new LocString("card_library", "COST_TWO_TIP");
		_threePlusFilter.Loc = new LocString("card_library", "COST_THREE_TIP");
		_xFilter.Loc = new LocString("card_library", "COST_X_TIP");
		_cardPoolFilters.Add(ModelDb.Character<Ironclad>(), _ironcladFilter);
		_cardPoolFilters.Add(ModelDb.Character<Silent>(), _silentFilter);
		_cardPoolFilters.Add(ModelDb.Character<Defect>(), _defectFilter);
		_cardPoolFilters.Add(ModelDb.Character<Necrobinder>(), _necrobinderFilter);
		_cardPoolFilters.Add(ModelDb.Character<Regent>(), _regentFilter);
		UnlockState unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
		foreach (KeyValuePair<CharacterModel, NCardPoolFilter> cardPoolFilter in _cardPoolFilters)
		{
			cardPoolFilter.Value.Visible = unlockState.Characters.Contains(cardPoolFilter.Key);
		}
		CardRarity[] values = Enum.GetValues<CardRarity>();
		for (int num = 0; num < values.Length; num++)
		{
			CardRarity keyword = values[num];
			_specialSearchbarKeywords.Add(keyword.ToString().ToLowerInvariant(), (CardModel c) => c.Rarity == keyword);
		}
		_ironcladFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _ironcladFilter;
		}));
		_silentFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _silentFilter;
		}));
		_defectFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _defectFilter;
		}));
		_regentFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _regentFilter;
		}));
		_necrobinderFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _necrobinderFilter;
		}));
		_colorlessFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _colorlessFilter;
		}));
		_ancientsFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _ancientsFilter;
		}));
		_miscPoolFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _miscPoolFilter;
		}));
		_attackFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _attackFilter;
		}));
		_skillFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _skillFilter;
		}));
		_powerFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _powerFilter;
		}));
		_otherTypeFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _otherTypeFilter;
		}));
		_commonFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _commonFilter;
		}));
		_uncommonFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _uncommonFilter;
		}));
		_rareFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _rareFilter;
		}));
		_otherFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _otherFilter;
		}));
		_zeroFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _zeroFilter;
		}));
		_oneFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _oneFilter;
		}));
		_twoFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _twoFilter;
		}));
		_threePlusFilter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _threePlusFilter;
		}));
		_alphabetSorter.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _alphabetSorter;
		}));
		_viewUpgrades.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
		{
			_lastHoveredControl = _viewUpgrades;
		}));
	}

	public void Initialize(IRunState runState)
	{
		_runState = runState;
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
		TaskHelper.RunSafely(DisplayCards());
	}

	private void OnRaritySort(NButton button)
	{
		_sortingPriority.Remove(SortingOrders.RarityAscending);
		_sortingPriority.Remove(SortingOrders.RarityDescending);
		if (_raritySorter.IsDescending)
		{
			_sortingPriority.Insert(0, SortingOrders.RarityAscending);
		}
		else
		{
			_sortingPriority.Insert(0, SortingOrders.RarityDescending);
		}
		TaskHelper.RunSafely(DisplayCards());
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
		TaskHelper.RunSafely(DisplayCards());
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
		TaskHelper.RunSafely(DisplayCards());
	}

	public override void OnSubmenuOpened()
	{
		_grid.RefreshVisibility();
		CharacterModel characterModel = LocalContext.GetMe(_runState)?.Character;
		_searchBar.ClearText();
		if (characterModel != null)
		{
			foreach (NCardPoolFilter key in _poolFilters.Keys)
			{
				key.IsSelected = _cardPoolFilters[characterModel] == key;
			}
		}
		else
		{
			foreach (NCardPoolFilter key2 in _poolFilters.Keys)
			{
				key2.IsSelected = key2 == _ironcladFilter;
			}
		}
		foreach (NCardTypeTickbox key3 in _cardTypeFilters.Keys)
		{
			key3.IsTicked = false;
		}
		foreach (NCardRarityTickbox key4 in _rarityFilters.Keys)
		{
			key4.IsTicked = false;
		}
		foreach (NCardCostTickbox key5 in _costFilters.Keys)
		{
			key5.IsTicked = false;
		}
		_typeSorter.IsDescending = true;
		_raritySorter.IsDescending = true;
		_costSorter.IsDescending = true;
		_alphabetSorter.IsDescending = true;
		_viewUpgrades.IsTicked = false;
		_viewStats.IsTicked = false;
		_viewMultiplayerCards.IsTicked = true;
		ToggleShowStats(_viewStats);
		ToggleShowUpgrades(_viewUpgrades);
		UpdateFilter();
	}

	public override void OnSubmenuClosed()
	{
		_grid.ClearGrid();
	}

	private async Task DisplayCardsAfterShortDelay()
	{
		if (_displayCardsShortDelayCancelToken != null)
		{
			await _displayCardsShortDelayCancelToken.CancelAsync();
		}
		if (!_grid.IsAnimatingOut)
		{
			TaskHelper.RunSafely(_grid.AnimateOut());
		}
		CancellationTokenSource cancelToken = (_displayCardsShortDelayCancelToken = new CancellationTokenSource());
		await Task.Delay(250, cancelToken.Token);
		if (!cancelToken.IsCancellationRequested)
		{
			await DisplayCards();
		}
	}

	private async Task DisplayCards()
	{
		if (_displayCardsShortDelayCancelToken != null)
		{
			await _displayCardsShortDelayCancelToken.CancelAsync();
		}
		await Task.Yield();
		_grid.FilterCards(_filter, _sortingPriority);
		_cardCountLocString.Add("Amount", _grid.VisibleCards.Count());
		_cardCountLabel.Text = "[center]" + _cardCountLocString.GetFormattedText() + "[/center]";
		_noResultsLabel.Visible = !_grid.VisibleCards.Any();
	}

	private void ToggleShowStats(NTickbox tickbox)
	{
		_grid.ShowStats = tickbox.IsTicked;
	}

	private void ToggleShowUpgrades(NTickbox tickbox)
	{
		_grid.IsShowingUpgrades = tickbox.IsTicked;
		if (!string.IsNullOrWhiteSpace(_searchBar.Text))
		{
			UpdateFilter();
		}
	}

	private void ToggleFilterMultiplayerCards(NTickbox tickbox)
	{
		UpdateFilter();
	}

	private void UpdateCardPoolFilter(NCardPoolFilter filter)
	{
		if (filter.IsSelected)
		{
			foreach (NCardPoolFilter key2 in _poolFilters.Keys)
			{
				if (key2 != filter)
				{
					key2.IsSelected = false;
				}
			}
		}
		bool flag = true;
		foreach (KeyValuePair<NCardPoolFilter, Func<CardModel, bool>> poolFilter in _poolFilters)
		{
			NCardPoolFilter key = poolFilter.Key;
			if (key.IsSelected && key != _miscPoolFilter && key != _ancientsFilter)
			{
				flag = false;
				break;
			}
		}
		foreach (NCardRarityTickbox key3 in _rarityFilters.Keys)
		{
			if (flag)
			{
				key3.Disable();
			}
			else
			{
				key3.Enable();
			}
		}
		UpdateFilter();
	}

	private void UpdateTypeFilter(NCardTypeTickbox tickbox)
	{
		UpdateFilter();
	}

	private void UpdateRarityFilter(NTickbox tickbox)
	{
		UpdateFilter();
	}

	private void UpdateCostFilter(NCardCostTickbox tickbox)
	{
		UpdateFilter();
	}

	private void SearchBarQueryChanged(string _ = "")
	{
		UpdateFilter(isTextInput: true);
	}

	private void SearchBarQuerySubmitted(string _ = "")
	{
		UpdateFilter();
	}

	private void UpdateFilter(bool isTextInput = false)
	{
		List<Func<CardModel, bool>> activeRarityFilters = new List<Func<CardModel, bool>>();
		bool flag = true;
		foreach (KeyValuePair<NCardPoolFilter, Func<CardModel, bool>> poolFilter2 in _poolFilters)
		{
			if (poolFilter2.Key.IsSelected && poolFilter2.Key != _miscPoolFilter && poolFilter2.Key != _ancientsFilter)
			{
				flag = false;
				break;
			}
		}
		Func<CardModel, bool> value;
		if (!flag)
		{
			foreach (KeyValuePair<NCardRarityTickbox, Func<CardModel, bool>> rarityFilter in _rarityFilters)
			{
				rarityFilter.Deconstruct(out var key, out value);
				NTickbox nTickbox = key;
				Func<CardModel, bool> item = value;
				if (nTickbox.IsTicked)
				{
					activeRarityFilters.Add(item);
				}
			}
		}
		if (activeRarityFilters.Count == 0)
		{
			activeRarityFilters.Add((CardModel _) => true);
		}
		List<Func<CardModel, bool>> activeCardTypeFilter = new List<Func<CardModel, bool>>();
		foreach (KeyValuePair<NCardTypeTickbox, Func<CardModel, bool>> cardTypeFilter in _cardTypeFilters)
		{
			cardTypeFilter.Deconstruct(out var key2, out value);
			NCardTypeTickbox nCardTypeTickbox = key2;
			Func<CardModel, bool> item2 = value;
			if (nCardTypeTickbox.IsTicked)
			{
				activeCardTypeFilter.Add(item2);
			}
		}
		if (activeCardTypeFilter.Count == 0)
		{
			activeCardTypeFilter.Add((CardModel _) => true);
		}
		List<Func<CardModel, bool>> poolFilter = new List<Func<CardModel, bool>>();
		foreach (KeyValuePair<NCardPoolFilter, Func<CardModel, bool>> poolFilter3 in _poolFilters)
		{
			if (poolFilter3.Key.IsSelected)
			{
				poolFilter.Add(poolFilter3.Value);
			}
		}
		List<Func<CardModel, bool>> activeCostFilter = new List<Func<CardModel, bool>>();
		foreach (KeyValuePair<NCardCostTickbox, Func<CardModel, bool>> costFilter in _costFilters)
		{
			costFilter.Deconstruct(out var key3, out value);
			NCardCostTickbox nCardCostTickbox = key3;
			Func<CardModel, bool> item3 = value;
			if (nCardCostTickbox.IsTicked)
			{
				activeCostFilter.Add(item3);
			}
		}
		if (activeCostFilter.Count == 0)
		{
			activeCostFilter.Add((CardModel _) => true);
		}
		Func<CardModel, bool> multiplayerCardFilter = (CardModel c) => true;
		if (!_viewMultiplayerCards.IsTicked)
		{
			multiplayerCardFilter = (CardModel c) => c.MultiplayerConstraint != CardMultiplayerConstraint.MultiplayerOnly;
		}
		_filter = (CardModel c) => activeCostFilter.Any((Func<CardModel, bool> filter) => filter(c)) && activeRarityFilters.Any((Func<CardModel, bool> filter) => filter(c)) && activeCardTypeFilter.Any((Func<CardModel, bool> filter) => filter(c)) && poolFilter.Any((Func<CardModel, bool> filter) => filter(c)) && TextFilter(c) && multiplayerCardFilter(c);
		Task task = ((!isTextInput) ? DisplayCards() : DisplayCardsAfterShortDelay());
		TaskHelper.RunSafely(task);
		bool TextFilter(CardModel card)
		{
			if (string.IsNullOrWhiteSpace(_searchBar.Text))
			{
				return true;
			}
			if (!SaveManager.Instance.Progress.DiscoveredCards.Contains(card.Id))
			{
				return false;
			}
			string title = card.Title;
			string text;
			if (_viewUpgrades.IsTicked && card.IsUpgradable)
			{
				CardModel cardModel = (CardModel)card.MutableClone();
				cardModel.UpgradeInternal();
				cardModel.UpdateDynamicVarPreview(CardPreviewMode.Upgrade, null, card.DynamicVars);
				text = cardModel.GetDescriptionForUpgradePreview().StripBbCode();
			}
			else
			{
				text = card.GetDescriptionForPile(PileType.None).StripBbCode();
			}
			global::_003C_003Ey__InlineArray2<string> buffer = default(global::_003C_003Ey__InlineArray2<string>);
			buffer[0] = title;
			buffer[1] = NSearchBar.RemoveHtmlTags(text);
			string text2 = string.Join(" ", (ReadOnlySpan<string?>)buffer);
			string text3 = NSearchBar.Normalize(text2);
			string text4 = _searchBar.Text.ToLowerInvariant();
			if (_specialSearchbarKeywords.TryGetValue(text4, out Func<CardModel, bool> value2))
			{
				if (!value2(card))
				{
					return text3.Contains(text4);
				}
				return true;
			}
			return text3.Contains(text4);
		}
	}

	private void ShowCardDetail(NCardHolder holder)
	{
		if (SaveManager.Instance.Progress.DiscoveredCards.Contains(holder.CardModel.Id))
		{
			_lastHoveredControl = holder;
			List<CardModel> list = _grid.VisibleCards.Where((CardModel c) => SaveManager.Instance.Progress.DiscoveredCards.Contains(c.Id)).ToList();
			NGame.Instance.GetInspectCardScreen().Open(list, list.IndexOf(holder.CardModel), _viewUpgrades.IsTicked);
		}
	}

	public NCardLibrary()
		: base()
	{
	}
}
