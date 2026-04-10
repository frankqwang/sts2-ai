using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;

namespace MegaCrit.Sts2.Core.Nodes.Cards;

public partial class NCardGrid : Control
{
	[Signal]
	public delegate void HolderPressedEventHandler(NCardHolder card);

	[Signal]
	public delegate void HolderAltPressedEventHandler(NCardHolder card);

	private Dictionary<SortingOrders, Func<CardModel, CardModel, int>>? _sortingAlgorithms;

	private float _startDrag;

	private float _targetDrag;

	private bool _isDragging;

	private bool _scrollingEnabled = true;

	private const float _topMargin = 80f;

	private const float _bottomMargin = 320f;

	protected Control _scrollContainer;

	private bool _scrollbarPressed;

	private NScrollbar _scrollbar;

	private int _slidingWindowCardIndex;

	private PileType _pileType;

	protected Vector2 _cardSize;

	protected readonly List<CardModel> _cards = new List<CardModel>();

	protected readonly List<List<NGridCardHolder>> _cardRows = new List<List<NGridCardHolder>>();

	private readonly List<CardModel> _highlightedCards = new List<CardModel>();

	private Task? _animatingOutTask;

	private bool _cardsAnimatingOutForSetCards;

	private CancellationTokenSource? _setCardsCancellation;

	private bool _isShowingUpgrades;

	private NCardHolder? _lastFocusedHolder;

	private readonly List<CardModel> _cardsCache = new List<CardModel>();

	private readonly List<CardModel> _sortedCardsCache = new List<CardModel>();

	private bool _needsReinit;

	private Dictionary<SortingOrders, Func<CardModel, CardModel, int>> SortingAlgorithms
	{
		get
		{
			Dictionary<SortingOrders, Func<CardModel, CardModel, int>> dictionary = _sortingAlgorithms;
			if (dictionary == null)
			{
				Dictionary<SortingOrders, Func<CardModel, CardModel, int>> obj = new Dictionary<SortingOrders, Func<CardModel, CardModel, int>>
				{
					{
						SortingOrders.RarityAscending,
						(CardModel a, CardModel b) => GetCardRarityComparisonValue(a).CompareTo(GetCardRarityComparisonValue(b))
					},
					{
						SortingOrders.CostAscending,
						(CardModel a, CardModel b) => a.EnergyCost.Canonical.CompareTo(b.EnergyCost.Canonical)
					},
					{
						SortingOrders.TypeAscending,
						(CardModel a, CardModel b) => a.Type.CompareTo(b.Type)
					},
					{
						SortingOrders.AlphabetAscending,
						(CardModel a, CardModel b) => string.Compare(a.Title, b.Title, LocManager.Instance.CultureInfo, CompareOptions.None)
					},
					{
						SortingOrders.RarityDescending,
						(CardModel a, CardModel b) => -GetCardRarityComparisonValue(a).CompareTo(GetCardRarityComparisonValue(b))
					},
					{
						SortingOrders.CostDescending,
						(CardModel a, CardModel b) => -a.EnergyCost.Canonical.CompareTo(b.EnergyCost.Canonical)
					},
					{
						SortingOrders.TypeDescending,
						(CardModel a, CardModel b) => -a.Type.CompareTo(b.Type)
					},
					{
						SortingOrders.AlphabetDescending,
						(CardModel a, CardModel b) => -string.Compare(a.Title, b.Title, LocManager.Instance.CultureInfo, CompareOptions.None)
					},
					{
						SortingOrders.Ascending,
						(CardModel a, CardModel b) => _cards.IndexOf(a).CompareTo(_cards.IndexOf(b))
					},
					{
						SortingOrders.Descending,
						(CardModel a, CardModel b) => -_cards.IndexOf(a).CompareTo(_cards.IndexOf(b))
					}
				};
				Dictionary<SortingOrders, Func<CardModel, CardModel, int>> dictionary2 = obj;
				_sortingAlgorithms = obj;
				dictionary = dictionary2;
			}
			return dictionary;
		}
	}

	private bool CanScroll
	{
		get
		{
			if (_scrollingEnabled)
			{
				return base.Visible;
			}
			return false;
		}
	}

	private int DisplayedRows { get; set; }

	protected int Columns => (int)((_scrollContainer.Size.X + CardPadding) / (_cardSize.X + CardPadding));

	protected float CardPadding => 40f;

	protected virtual bool IsCardLibrary => false;

	private float ScrollLimitBottom
	{
		get
		{
			if (!(base.Size.Y > _scrollContainer.Size.Y))
			{
				return base.Size.Y - _scrollContainer.Size.Y;
			}
			return (base.Size.Y - _scrollContainer.Size.Y) * 0.5f;
		}
	}

	protected float ScrollLimitTop
	{
		get
		{
			if (!(base.Size.Y > _scrollContainer.Size.Y) || !CenterGrid)
			{
				return 0f;
			}
			return (base.Size.Y - _scrollContainer.Size.Y) * 0.5f;
		}
	}

	public IEnumerable<NGridCardHolder> CurrentlyDisplayedCardHolders => _cardRows.SelectMany((List<NGridCardHolder> r) => r);

	public IEnumerable<CardModel> CurrentlyDisplayedCards => CurrentlyDisplayedCardHolders.Select((NGridCardHolder h) => h.CardModel);

	public bool IsAnimatingOut
	{
		get
		{
			Task animatingOutTask = _animatingOutTask;
			if (animatingOutTask != null)
			{
				return !animatingOutTask.IsCompleted;
			}
			return false;
		}
	}

	public bool IsShowingUpgrades
	{
		get
		{
			return _isShowingUpgrades;
		}
		set
		{
			_isShowingUpgrades = value;
			foreach (List<NGridCardHolder> cardRow in _cardRows)
			{
				foreach (NGridCardHolder item in cardRow)
				{
					if ((_isShowingUpgrades || item.CardModel.CanonicalInstance.IsUpgradable) && (!_isShowingUpgrades || item.CardModel.IsUpgradable))
					{
						item.SetIsPreviewingUpgrade(_isShowingUpgrades);
					}
				}
			}
		}
	}

	public int YOffset { get; set; }

	protected virtual bool CenterGrid => true;

	public Control? DefaultFocusedControl
	{
		get
		{
			if (_lastFocusedHolder != null)
			{
				return _lastFocusedHolder;
			}
			if (_cards.Count == 0)
			{
				return null;
			}
			return _cardRows[0][0];
		}
	}

	public Control? FocusedControlFromTopBar
	{
		get
		{
			if (_cards.Count != 0)
			{
				return _cardRows[0][0];
			}
			return null;
		}
	}

	private int CompareCardVisibility(CardModel a, CardModel b)
	{
		bool flag = GetCardVisibility(a) == ModelVisibility.Locked;
		bool value = GetCardVisibility(b) == ModelVisibility.Locked;
		return flag.CompareTo(value);
	}

	private int GetCardRarityComparisonValue(CardModel a)
	{
		if (a.Rarity <= CardRarity.Ancient)
		{
			return (int)a.Rarity;
		}
		return a.Rarity switch
		{
			CardRarity.Status => 6, 
			CardRarity.Curse => 7, 
			CardRarity.Event => 8, 
			CardRarity.Quest => 9, 
			CardRarity.Token => 10, 
			_ => throw new ArgumentOutOfRangeException("a", a, null), 
		};
	}

	public override void _Ready()
	{
		if (GetType() != typeof(NCardGrid))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected virtual void ConnectSignals()
	{
		_scrollContainer = GetNode<Control>("%ScrollContainer");
		_scrollbar = GetNode<NScrollbar>("Scrollbar");
		_cardSize = NCard.defaultSize * NCardHolder.smallScale;
		_scrollContainer.Connect(CanvasItem.SignalName.ItemRectChanged, Callable.From(UpdateScrollLimitBottom));
		_scrollbar.Visible = false;
		_scrollbar.Connect(NScrollbar.SignalName.MousePressed, Callable.From<InputEvent>(delegate
		{
			_scrollbarPressed = true;
		}));
		_scrollbar.Connect(NScrollbar.SignalName.MouseReleased, Callable.From<InputEvent>(delegate
		{
			_scrollbarPressed = false;
		}));
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		GetViewport().Connect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(ProcessGuiFocus));
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		GetViewport().Disconnect(Viewport.SignalName.GuiFocusChanged, Callable.From<Control>(ProcessGuiFocus));
		foreach (List<NGridCardHolder> cardRow in _cardRows)
		{
			foreach (NGridCardHolder item in cardRow)
			{
				item.QueueFreeSafely();
			}
		}
		_cardRows.Clear();
	}

	private void UpdateScrollLimitBottom()
	{
		float num = base.Size.Y + 320f;
		_scrollbar.Visible = _scrollContainer.Size.Y > num && CanScroll;
		_scrollbar.MouseFilter = (MouseFilterEnum)((_scrollContainer.Size.Y > num && CanScroll) ? 0 : 2);
	}

	public override void _GuiInput(InputEvent inputEvent)
	{
		if (IsVisibleInTree())
		{
			ProcessMouseEvent(inputEvent);
			ProcessScrollEvent(inputEvent);
		}
	}

	public override void _Process(double delta)
	{
		if (IsVisibleInTree() && CanScroll)
		{
			UpdateScrollPosition(delta);
			if (_needsReinit)
			{
				InitGrid();
			}
		}
	}

	public override void _Notification(int what)
	{
		base._Notification(what);
		if ((long)what == 40 && IsNodeReady())
		{
			_needsReinit = true;
		}
	}

	public void SetScrollPosition(float scrollY)
	{
		_targetDrag = scrollY;
		_scrollContainer.Position = new Vector2(_scrollContainer.Position.X, scrollY);
	}

	public void SetCanScroll(bool canScroll)
	{
		_scrollingEnabled = canScroll;
		if (!CanScroll)
		{
			_isDragging = false;
		}
	}

	public void InsetForTopBar()
	{
		SetAnchorAndOffset(Side.Top, 0f, 80f);
	}

	private void ProcessMouseEvent(InputEvent inputEvent)
	{
		if (_isDragging && inputEvent is InputEventMouseMotion inputEventMouseMotion)
		{
			_targetDrag += inputEventMouseMotion.Relative.Y;
		}
		else
		{
			if (!(inputEvent is InputEventMouseButton inputEventMouseButton))
			{
				return;
			}
			if (inputEventMouseButton.ButtonIndex == MouseButton.Left)
			{
				if (inputEventMouseButton.Pressed)
				{
					_isDragging = true;
					_startDrag = _scrollContainer.Position.Y;
					_targetDrag = _startDrag;
				}
				else
				{
					_isDragging = false;
				}
			}
			else if (!inputEventMouseButton.Pressed)
			{
				_isDragging = false;
			}
		}
	}

	private void ProcessScrollEvent(InputEvent inputEvent)
	{
		_targetDrag += ScrollHelper.GetDragForScrollEvent(inputEvent);
	}

	private void ProcessGuiFocus(Control focusedControl)
	{
		if (IsVisibleInTree() && CanScroll && NControllerManager.Instance.IsUsingController && focusedControl.GetParent() == _scrollContainer)
		{
			float value = 0f - focusedControl.Position.Y + base.Size.Y * 0.5f;
			float min = Math.Min(Math.Min(ScrollLimitTop, ScrollLimitBottom), 0f);
			float max = Math.Max(Math.Min(ScrollLimitTop, ScrollLimitBottom), 0f);
			value = Math.Clamp(value, min, max);
			_targetDrag = value;
		}
	}

	private void UpdateScrollPosition(double delta)
	{
		float num = _scrollContainer.Position.Y;
		if (Math.Abs(num - _targetDrag) > 0.1f)
		{
			num = Mathf.Lerp(num, _targetDrag, Mathf.Clamp((float)delta * 15f, 0f, 1f));
			if (Mathf.Abs(num - _targetDrag) < 0.5f)
			{
				num = _targetDrag;
			}
			AllocateCardHolders();
			if (!_scrollbarPressed && CanScroll)
			{
				_scrollbar.SetValueWithoutAnimation(Mathf.Clamp(_scrollContainer.Position.Y / ScrollLimitBottom, 0f, 1f) * 100f);
			}
		}
		if (_scrollbarPressed)
		{
			_targetDrag = Mathf.Lerp(0f, ScrollLimitBottom, (float)_scrollbar.Value / 100f);
			AllocateCardHolders();
		}
		if (!_isDragging)
		{
			if (_targetDrag < Mathf.Min(ScrollLimitBottom, ScrollLimitTop))
			{
				_targetDrag = Mathf.Lerp(_targetDrag, Mathf.Min(ScrollLimitBottom, ScrollLimitTop), (float)delta * 12f);
			}
			else if (_targetDrag > Mathf.Max(ScrollLimitTop, ScrollLimitBottom))
			{
				_targetDrag = Mathf.Lerp(_targetDrag, Mathf.Max(ScrollLimitTop, ScrollLimitBottom), (float)delta * 12f);
			}
		}
		_scrollContainer.Position = new Vector2(_scrollContainer.Position.X, num);
	}

	public void ClearGrid()
	{
		_cardsCache.Clear();
		_cards.Clear();
		TaskHelper.RunSafely(InitGrid(null));
	}

	public void SetCards(IReadOnlyList<CardModel> cardsToDisplay, PileType pileType, List<SortingOrders> sortingPriority, Task? taskToWaitOn = null)
	{
		_cardsCache.Clear();
		_cardsCache.AddRange(cardsToDisplay);
		if (sortingPriority[0] == SortingOrders.Descending)
		{
			_cardsCache.Reverse();
		}
		else if (sortingPriority[0] != SortingOrders.Ascending)
		{
			_cardsCache.Sort(delegate(CardModel x, CardModel y)
			{
				foreach (SortingOrders item in sortingPriority)
				{
					int num = SortingAlgorithms[item](x, y);
					if (num != 0)
					{
						return num;
					}
				}
				return x.Id.CompareTo(y.Id);
			});
		}
		if (IsCardLibrary)
		{
			_sortedCardsCache.Clear();
			_sortedCardsCache.AddRange(_cardsCache.OrderBy((CardModel c) => c, Comparer<CardModel>.Create(CompareCardVisibility)));
			_cardsCache.Clear();
			_cardsCache.AddRange(_sortedCardsCache);
		}
		_cards.Clear();
		_cards.AddRange(_cardsCache);
		_pileType = pileType;
		if (!_cardsAnimatingOutForSetCards)
		{
			TaskHelper.RunSafely(InitGrid(taskToWaitOn));
		}
	}

	public Task AnimateOut()
	{
		_animatingOutTask = AnimateOutInternal();
		return _animatingOutTask;
	}

	private async Task AnimateOutInternal()
	{
		if (!IsCardLibrary)
		{
			return;
		}
		List<NGridCardHolder> list = _cardRows.SelectMany((List<NGridCardHolder> c) => c).ToList();
		if (list.Count <= 0)
		{
			return;
		}
		Tween tween = CreateTween().SetParallel();
		foreach (NGridCardHolder item in list)
		{
			tween.TweenProperty(item, "position:y", item.Position.Y + 40f, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			tween.TweenProperty(item, "modulate:a", 0f, 0.2);
		}
		await ToSignal(tween, Tween.SignalName.Finished);
	}

	private async Task AnimateIn()
	{
		if (!IsCardLibrary)
		{
			return;
		}
		List<NGridCardHolder> list = _cardRows.SelectMany((List<NGridCardHolder> c) => c).ToList();
		if (list.Count <= 0)
		{
			return;
		}
		Tween tween = CreateTween().SetParallel();
		for (int num = 0; num < list.Count; num++)
		{
			NGridCardHolder nGridCardHolder = list[num];
			float num2 = (float)num / (float)list.Count * 0.2f;
			float y = nGridCardHolder.Position.Y;
			Vector2 position = nGridCardHolder.Position;
			position.Y = nGridCardHolder.Position.Y + 40f;
			nGridCardHolder.Position = position;
			Color modulate = nGridCardHolder.Modulate;
			modulate.A = 0f;
			nGridCardHolder.Modulate = modulate;
			tween.TweenProperty(nGridCardHolder, "position:y", y, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
				.SetDelay(num2);
			tween.TweenProperty(nGridCardHolder, "modulate:a", 1f, 0.4).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo)
				.SetDelay(num2);
		}
		_setCardsCancellation = new CancellationTokenSource();
		while (tween.IsRunning())
		{
			if (_setCardsCancellation.IsCancellationRequested)
			{
				tween.Kill();
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
	}

	private async Task InitGrid(Task? taskToWaitOn)
	{
		if (_setCardsCancellation != null)
		{
			await _setCardsCancellation.CancelAsync();
		}
		_cardsAnimatingOutForSetCards = true;
		Task animatingOutTask = _animatingOutTask;
		if (animatingOutTask == null || animatingOutTask.IsCompleted)
		{
			await AnimateOut();
		}
		else
		{
			await _animatingOutTask;
		}
		if (taskToWaitOn != null)
		{
			await taskToWaitOn;
		}
		_cardsAnimatingOutForSetCards = false;
		InitGrid();
		SetScrollPosition(ScrollLimitTop);
		await AnimateIn();
	}

	private int CalculateRowsNeeded()
	{
		int a = Mathf.CeilToInt((base.Size.Y + CardPadding) / (_cardSize.Y + CardPadding)) + 2;
		return Mathf.Min(a, GetTotalRowCount());
	}

	protected virtual void InitGrid()
	{
		_scrollContainer.Position = new Vector2(_scrollContainer.Position.X, ScrollLimitTop);
		_slidingWindowCardIndex = 0;
		_scrollbar.Value = 0.0;
		DisplayedRows = CalculateRowsNeeded();
		foreach (List<NGridCardHolder> cardRow in _cardRows)
		{
			foreach (NGridCardHolder item in cardRow)
			{
				item.Name = string.Concat(item.Name, "-OLD");
				item.QueueFreeSafely();
			}
		}
		_cardRows.Clear();
		if (_cards.Count != 0)
		{
			int num = 0;
			for (int i = 0; i < DisplayedRows; i++)
			{
				List<NGridCardHolder> list = new List<NGridCardHolder>();
				for (int j = 0; j < Columns; j++)
				{
					if (num >= _cards.Count)
					{
						break;
					}
					CardModel card = _cards[num];
					NCard nCard = NCard.Create(card, GetCardVisibility(card));
					NGridCardHolder nGridCardHolder = NGridCardHolder.Create(nCard);
					list.Add(nGridCardHolder);
					nGridCardHolder.Connect(NCardHolder.SignalName.Pressed, Callable.From<NCardHolder>(OnHolderPressed));
					nGridCardHolder.Connect(NCardHolder.SignalName.AltPressed, Callable.From<NCardHolder>(OnHolderAltPressed));
					nGridCardHolder.Visible = true;
					nGridCardHolder.MouseFilter = MouseFilterEnum.Pass;
					nGridCardHolder.Scale = nGridCardHolder.SmallScale;
					_scrollContainer.AddChildSafely(nGridCardHolder);
					nCard.UpdateVisuals(_pileType, CardPreviewMode.Normal);
					if (nGridCardHolder.CardModel.CanonicalInstance.IsUpgradable)
					{
						nGridCardHolder.SetIsPreviewingUpgrade(IsShowingUpgrades);
					}
					num++;
				}
				_cardRows.Add(list);
			}
		}
		_scrollContainer.SetDeferred(Control.PropertyName.Size, new Vector2(_scrollContainer.Size.X, GetContainedCardsSize().Y + 80f + 320f + (float)YOffset));
		UpdateGridPositions(0);
		UpdateGridNavigation();
		_needsReinit = false;
	}

	private Vector2 GetContainedCardsSize()
	{
		int totalRowCount = GetTotalRowCount();
		return new Vector2(Columns, totalRowCount) * _cardSize + new Vector2(Columns - 1, totalRowCount - 1) * CardPadding;
	}

	private void ReflowColumns()
	{
		if (_cards.Count != 0)
		{
			InitGrid();
		}
	}

	private void UpdateGridPositions(int index)
	{
		Vector2 vector = new Vector2((_scrollContainer.Size.X - GetContainedCardsSize().X) * 0.5f, (float)YOffset + 80f) + _cardSize * 0.5f;
		foreach (List<NGridCardHolder> cardRow in _cardRows)
		{
			foreach (NGridCardHolder item in cardRow)
			{
				int num = index / Columns;
				int num2 = index % Columns;
				item.Position = vector + new Vector2((float)num2 * (_cardSize.X + CardPadding), (float)num * (_cardSize.Y + CardPadding));
				index++;
			}
		}
	}

	public NGridCardHolder? GetCardHolder(CardModel model)
	{
		return _cardRows.SelectMany((List<NGridCardHolder> row) => row).FirstOrDefault((NGridCardHolder h) => h.CardModel == model);
	}

	public NCard? GetCardNode(CardModel model)
	{
		return GetCardHolder(model)?.CardNode;
	}

	public IEnumerable<NGridCardHolder>? GetTopRowOfCardNodes()
	{
		return _cardRows.FirstOrDefault();
	}

	private void OnHolderPressed(NCardHolder holder)
	{
		_lastFocusedHolder = holder;
		EmitSignal(SignalName.HolderPressed, holder);
	}

	private void OnHolderAltPressed(NCardHolder holder)
	{
		_lastFocusedHolder = holder;
		EmitSignal(SignalName.HolderAltPressed, holder);
	}

	private int GetTotalRowCount()
	{
		return Mathf.CeilToInt((float)_cards.Count / (float)Columns);
	}

	private void AllocateCardHolders()
	{
		if (_cardRows.Count != 0)
		{
			float y = GetViewportRect().Size.Y;
			float num = y;
			List<NGridCardHolder> list = _cardRows[0];
			float y2 = list[0].GlobalPosition.Y;
			List<List<NGridCardHolder>> cardRows = _cardRows;
			List<NGridCardHolder> list2 = cardRows[cardRows.Count - 1];
			float y3 = list2[0].GlobalPosition.Y;
			if (Mathf.Abs(y2 - 0f) > base.Size.Y * 2f)
			{
				ReallocateAll();
			}
			else if (y2 > 0f)
			{
				List<List<NGridCardHolder>> cardRows2 = _cardRows;
				ReallocateAbove(cardRows2[cardRows2.Count - 1]);
			}
			else if (y3 < num)
			{
				ReallocateBelow(_cardRows[0]);
			}
		}
	}

	private void ReallocateAll()
	{
		List<NGridCardHolder> list = _cardRows[0];
		float y = list[0].GlobalPosition.Y;
		float num = y - 0f;
		int num2 = Mathf.RoundToInt(num / (_cardSize.Y + CardPadding));
		int slidingWindowCardIndex = Mathf.Max(0, _slidingWindowCardIndex - Columns * num2);
		_slidingWindowCardIndex = slidingWindowCardIndex;
		int count = _cardRows.Count;
		for (int i = 0; i < count; i++)
		{
			AssignCardsToRow(_cardRows[i], _slidingWindowCardIndex + i * Columns);
		}
		UpdateGridPositions(_slidingWindowCardIndex);
		UpdateGridNavigation();
	}

	private void ReallocateAbove(List<NGridCardHolder> row)
	{
		int num = _slidingWindowCardIndex - Columns;
		if (num < 0)
		{
			return;
		}
		_slidingWindowCardIndex = num;
		_cardRows.RemoveAt(_cardRows.Count - 1);
		AssignCardsToRow(row, _slidingWindowCardIndex);
		_cardRows.Insert(0, row);
		List<NGridCardHolder> list = _cardRows[1];
		float y = list[0].Position.Y;
		foreach (NGridCardHolder item in row)
		{
			Vector2 position = item.Position;
			position.Y = y - _cardSize.Y - CardPadding;
			item.Position = position;
		}
		UpdateGridNavigation();
	}

	private void ReallocateBelow(List<NGridCardHolder> row)
	{
		int num = Columns * DisplayedRows;
		int num2 = _slidingWindowCardIndex + num;
		if (num2 >= _cards.Count)
		{
			return;
		}
		_slidingWindowCardIndex += Columns;
		_cardRows.RemoveAt(0);
		AssignCardsToRow(row, num2);
		_cardRows.Add(row);
		List<List<NGridCardHolder>> cardRows = _cardRows;
		List<NGridCardHolder> list = cardRows[cardRows.Count - 2];
		float y = list[0].Position.Y;
		foreach (NGridCardHolder item in row)
		{
			Vector2 position = item.Position;
			position.Y = y + _cardSize.Y + CardPadding;
			item.Position = position;
		}
		UpdateGridNavigation();
	}

	public void HighlightCard(CardModel card)
	{
		_highlightedCards.Add(card);
		GetCardNode(card)?.CardHighlight.AnimShow();
	}

	public void UnhighlightCard(CardModel card)
	{
		_highlightedCards.Remove(card);
		GetCardNode(card)?.CardHighlight.AnimHide();
	}

	protected virtual void AssignCardsToRow(List<NGridCardHolder> row, int startIndex)
	{
		for (int i = 0; i < row.Count; i++)
		{
			NGridCardHolder nGridCardHolder = row[i];
			if (startIndex + i >= _cards.Count)
			{
				nGridCardHolder.Visible = false;
				continue;
			}
			CardModel cardModel = _cards[startIndex + i];
			nGridCardHolder.ReassignToCard(cardModel, PileType.None, null, GetCardVisibility(cardModel));
			nGridCardHolder.Visible = true;
			if (_highlightedCards.Contains(cardModel))
			{
				nGridCardHolder.CardNode.CardHighlight.AnimShow();
			}
			else
			{
				nGridCardHolder.CardNode.CardHighlight.AnimHide();
			}
			if (_isShowingUpgrades && cardModel.IsUpgradable)
			{
				nGridCardHolder.SetIsPreviewingUpgrade(showUpgradePreview: true);
			}
		}
	}

	protected virtual ModelVisibility GetCardVisibility(CardModel card)
	{
		return ModelVisibility.Visible;
	}

	protected virtual void UpdateGridNavigation()
	{
		for (int i = 0; i < _cardRows.Count; i++)
		{
			for (int j = 0; j < _cardRows[i].Count; j++)
			{
				NCardHolder nCardHolder = _cardRows[i][j];
				nCardHolder.FocusNeighborLeft = ((j > 0) ? _cardRows[i][j - 1].GetPath() : _cardRows[i][_cardRows[i].Count - 1].GetPath());
				nCardHolder.FocusNeighborRight = ((j < _cardRows[i].Count - 1) ? _cardRows[i][j + 1].GetPath() : _cardRows[i][0].GetPath());
				nCardHolder.FocusNeighborTop = ((i > 0) ? _cardRows[i - 1][j].GetPath() : _cardRows[i][j].GetPath());
				nCardHolder.FocusNeighborBottom = ((i < _cardRows.Count - 1 && j < _cardRows[i + 1].Count) ? _cardRows[i + 1][j].GetPath() : _cardRows[i][j].GetPath());
			}
		}
	}
}
