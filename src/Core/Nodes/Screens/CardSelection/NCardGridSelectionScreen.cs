using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public abstract partial class NCardGridSelectionScreen : Control, IOverlayScreen, IScreenContext, ICardSelector
{
	protected NCardGrid _grid;

	protected NPeekButton _peekButton;

	protected IReadOnlyList<CardModel> _cards;

	protected readonly TaskCompletionSource<IEnumerable<CardModel>> _completionSource = new TaskCompletionSource<IEnumerable<CardModel>>();

	public NetScreenType ScreenType => NetScreenType.CardSelection;

	protected abstract IEnumerable<Control> PeekButtonTargets { get; }

	public bool UseSharedBackstop => true;

	public virtual Control? DefaultFocusedControl
	{
		get
		{
			if (_peekButton.IsPeeking)
			{
				return NCombatRoom.Instance.DefaultFocusedControl;
			}
			return _grid.DefaultFocusedControl;
		}
	}

	public virtual Control? FocusedControlFromTopBar
	{
		get
		{
			if (_peekButton.IsPeeking)
			{
				return NCombatRoom.Instance.FocusedControlFromTopBar;
			}
			return _grid.FocusedControlFromTopBar;
		}
	}

	public override void _Ready()
	{
		if (GetType() != typeof(NCardGridSelectionScreen))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignalsAndInitGrid();
	}

	protected virtual void ConnectSignalsAndInitGrid()
	{
		_grid = GetNode<NCardGrid>("%CardGrid");
		NCardGrid grid = _grid;
		IReadOnlyList<CardModel> cards = _cards;
		int num = 1;
		List<SortingOrders> list = new List<SortingOrders>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<SortingOrders> span = CollectionsMarshal.AsSpan(list);
		int index = 0;
		span[index] = SortingOrders.Ascending;
		grid.SetCards(cards, PileType.None, list);
		_grid.Connect(NCardGrid.SignalName.HolderPressed, Callable.From(delegate(NCardHolder h)
		{
			OnCardClicked(h.CardModel);
		}));
		_grid.Connect(NCardGrid.SignalName.HolderAltPressed, Callable.From(delegate(NCardHolder h)
		{
			ShowCardDetail(h.CardModel);
		}));
		_grid.InsetForTopBar();
		_peekButton = GetNode<NPeekButton>("%PeekButton");
		_peekButton.Connect(NPeekButton.SignalName.Toggled, Callable.From<NPeekButton>(delegate
		{
			if (_peekButton.IsPeeking)
			{
				base.MouseFilter = MouseFilterEnum.Ignore;
			}
			else
			{
				base.MouseFilter = MouseFilterEnum.Stop;
				ActiveScreenContext.Instance.Update();
			}
		}));
		Callable.From(SetPeekButtonTargets).CallDeferred();
	}

	protected abstract void OnCardClicked(CardModel card);

	public async Task<IEnumerable<CardModel>> CardsSelected()
	{
		return await _completionSource.Task;
	}

	public override void _ExitTree()
	{
		if (!_completionSource.Task.IsCompleted)
		{
			_completionSource.SetCanceled();
		}
	}

	private void SetPeekButtonTargets()
	{
		HashSet<Control> hashSet = new HashSet<Control> { _grid };
		hashSet.UnionWith(PeekButtonTargets);
		_peekButton.AddTargets(hashSet.ToArray());
	}

	public virtual void AfterOverlayOpened()
	{
	}

	public virtual void AfterOverlayClosed()
	{
		_peekButton.SetPeeking(isPeeking: false);
		this.QueueFreeSafely();
	}

	public virtual void AfterOverlayShown()
	{
		base.Visible = true;
		if (CombatManager.Instance.IsInProgress)
		{
			_peekButton.Enable();
		}
	}

	public virtual void AfterOverlayHidden()
	{
		base.Visible = false;
		_peekButton.Disable();
	}

	private void ShowCardDetail(CardModel card)
	{
		if (!NControllerManager.Instance.IsUsingController)
		{
			NGame.Instance.GetInspectCardScreen().Open(_cards.ToList(), _cards.IndexOf(card), _grid.IsShowingUpgrades);
		}
	}
}
