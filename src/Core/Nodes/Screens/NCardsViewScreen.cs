using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public abstract partial class NCardsViewScreen : Control, ICapstoneScreen, IScreenContext
{
	private ColorRect _background;

	protected NCardGrid _grid;

	protected NButton _backButton;

	private NTickbox _showUpgrades;

	private RichTextLabel _bottomLabel;

	protected List<CardModel> _cards;

	protected LocString _infoText;

	public abstract NetScreenType ScreenType { get; }

	public Control? DefaultFocusedControl => _grid.DefaultFocusedControl;

	public Control? FocusedControlFromTopBar => _grid.FocusedControlFromTopBar;

	public bool UseSharedBackstop => true;

	public override void _Ready()
	{
		if (GetType() != typeof(NCardsViewScreen))
		{
			Log.Error($"{GetType()}");
			throw new InvalidOperationException("Don't call base._Ready()! Call ConnectSignals() instead.");
		}
		ConnectSignals();
	}

	protected virtual void ConnectSignals()
	{
		_bottomLabel = GetNode<RichTextLabel>("%BottomLabel");
		_backButton = GetNode<NButton>("BackButton");
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(OnReturnButtonPressed));
		_backButton.Enable();
		_grid = GetNode<NCardGrid>("CardGrid");
		_grid.Connect(NCardGrid.SignalName.HolderPressed, Callable.From(delegate(NCardHolder h)
		{
			ShowCardDetail(h.CardModel);
		}));
		_grid.Connect(NCardGrid.SignalName.HolderAltPressed, Callable.From(delegate(NCardHolder h)
		{
			ShowCardDetail(h.CardModel);
		}));
		_grid.InsetForTopBar();
		_bottomLabel.Text = _infoText.GetFormattedText();
		_showUpgrades = GetNode<NTickbox>("%Upgrades");
		_showUpgrades.Connect(NTickbox.SignalName.Toggled, Callable.From<NTickbox>(ToggleShowUpgrades));
		base.ProcessMode = (ProcessModeEnum)(base.Visible ? 0 : 4);
	}

	private void ShowCardDetail(CardModel cardModel)
	{
		_backButton.Disable();
		List<CardModel> list = _grid.CurrentlyDisplayedCards.ToList();
		NInspectCardScreen inspectCardScreen = NGame.Instance.GetInspectCardScreen();
		inspectCardScreen.Open(list, list.IndexOf(cardModel), _grid.IsShowingUpgrades);
		inspectCardScreen.Connect(CanvasItem.SignalName.VisibilityChanged, Callable.From(delegate
		{
			if (!inspectCardScreen.Visible)
			{
				_backButton.Enable();
			}
		}), 4u);
	}

	private void ToggleShowUpgrades(NTickbox tickbox)
	{
		_grid.IsShowingUpgrades = tickbox.IsTicked;
	}

	protected void OnReturnButtonPressed(NButton _)
	{
		NCapstoneContainer.Instance.Close();
	}

	public virtual void AfterCapstoneOpened()
	{
		_showUpgrades.IsTicked = false;
	}

	public virtual void AfterCapstoneClosed()
	{
		base.Visible = false;
		this.QueueFreeSafely();
	}
}
