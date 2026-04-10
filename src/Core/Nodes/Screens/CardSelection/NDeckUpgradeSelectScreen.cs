using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public sealed partial class NDeckUpgradeSelectScreen : NCardGridSelectionScreen
{
	private readonly HashSet<CardModel> _selectedCards = new HashSet<CardModel>();

	private CardSelectorPrefs _prefs;

	private IRunState _runState;

	private Control _upgradeSinglePreviewContainer;

	private NUpgradePreview _singlePreview;

	private NBackButton _singlePreviewCancelButton;

	private NConfirmButton _singlePreviewConfirmButton;

	private NTickbox _viewUpgrades;

	private Control _bottomTextContainer;

	private MegaRichTextLabel _infoLabel;

	private Control _upgradeMultiPreviewContainer;

	private Control _multiPreview;

	private NBackButton _multiPreviewCancelButton;

	private NConfirmButton _multiPreviewConfirmButton;

	private NBackButton _closeButton;

	private static string ScenePath => SceneHelper.GetScenePath("screens/card_selection/deck_upgrade_select_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private bool UseSingleSelection => _prefs.MaxSelect == 1;

	protected override IEnumerable<Control> PeekButtonTargets => new global::_003C_003Ez__ReadOnlyArray<Control>(new Control[3] { _upgradeSinglePreviewContainer, _upgradeMultiPreviewContainer, _closeButton });

	public override Control? DefaultFocusedControl
	{
		get
		{
			if (_upgradeSinglePreviewContainer.Visible || _upgradeMultiPreviewContainer.Visible)
			{
				return null;
			}
			return _grid.DefaultFocusedControl;
		}
	}

	public override Control? FocusedControlFromTopBar
	{
		get
		{
			if (_upgradeSinglePreviewContainer.Visible || _upgradeMultiPreviewContainer.Visible)
			{
				return null;
			}
			return _grid.FocusedControlFromTopBar;
		}
	}

	public override void _Ready()
	{
		ConnectSignalsAndInitGrid();
		_upgradeSinglePreviewContainer = GetNode<Control>("%UpgradeSinglePreviewContainer");
		_singlePreview = _upgradeSinglePreviewContainer.GetNode<NUpgradePreview>("UpgradePreview");
		_singlePreviewCancelButton = _upgradeSinglePreviewContainer.GetNode<NBackButton>("Cancel");
		_singlePreviewConfirmButton = _upgradeSinglePreviewContainer.GetNode<NConfirmButton>("Confirm");
		_upgradeMultiPreviewContainer = GetNode<Control>("%UpgradeMultiPreviewContainer");
		_multiPreview = _upgradeMultiPreviewContainer.GetNode<Control>("Cards");
		_multiPreviewCancelButton = _upgradeMultiPreviewContainer.GetNode<NBackButton>("Cancel");
		_multiPreviewConfirmButton = _upgradeMultiPreviewContainer.GetNode<NConfirmButton>("Confirm");
		_closeButton = GetNode<NBackButton>("%Close");
		_bottomTextContainer = GetNode<Control>("%BottomText");
		_infoLabel = _bottomTextContainer.GetNode<MegaRichTextLabel>("%BottomLabel");
		_infoLabel.Text = _prefs.Prompt.GetFormattedText();
		_singlePreviewCancelButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CancelSelection));
		_singlePreviewConfirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(ConfirmSelection));
		_multiPreviewCancelButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CancelSelection));
		_multiPreviewConfirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(ConfirmSelection));
		_closeButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CloseSelection));
		if (_prefs.Cancelable)
		{
			_closeButton.Enable();
		}
		else
		{
			_closeButton.Disable();
		}
		_upgradeSinglePreviewContainer.Visible = false;
		_upgradeSinglePreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
		_upgradeMultiPreviewContainer.Visible = false;
		_upgradeMultiPreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
		_singlePreviewCancelButton.Disable();
		_singlePreviewConfirmButton.Disable();
		_multiPreviewCancelButton.Disable();
		_multiPreviewConfirmButton.Disable();
		_viewUpgrades = GetNode<NTickbox>("%Upgrades");
		_viewUpgrades.IsTicked = false;
		_viewUpgrades.Connect(NTickbox.SignalName.Toggled, Callable.From<NTickbox>(ToggleShowUpgrades));
		OnControllerStateUpdated();
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(OnControllerStateUpdated));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(OnControllerStateUpdated));
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(OnControllerStateUpdated));
		GetNode<MegaLabel>("%ViewUpgradesLabel").SetTextAutoSize(new LocString("card_selection", "VIEW_UPGRADES").GetFormattedText());
	}

	public static NDeckUpgradeSelectScreen ShowScreen(IReadOnlyList<CardModel> cards, CardSelectorPrefs prefs, IRunState runState)
	{
		NDeckUpgradeSelectScreen nDeckUpgradeSelectScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NDeckUpgradeSelectScreen>(PackedScene.GenEditState.Disabled);
		nDeckUpgradeSelectScreen.Name = "NDeckUpgradeSelectScreen";
		nDeckUpgradeSelectScreen._cards = cards;
		nDeckUpgradeSelectScreen._prefs = prefs;
		nDeckUpgradeSelectScreen._runState = runState;
		NOverlayStack.Instance.Push(nDeckUpgradeSelectScreen);
		return nDeckUpgradeSelectScreen;
	}

	protected override void OnCardClicked(CardModel card)
	{
		if (_selectedCards.Add(card))
		{
			_grid.HighlightCard(card);
			if (UseSingleSelection)
			{
				GetViewport().GuiReleaseFocus();
				_upgradeSinglePreviewContainer.Visible = true;
				_upgradeSinglePreviewContainer.MouseFilter = MouseFilterEnum.Stop;
				_singlePreview.Card = card;
				_singlePreviewCancelButton.Enable();
				_singlePreviewConfirmButton.Enable();
				_grid.SetCanScroll(canScroll: false);
				_closeButton.Disable();
			}
			else
			{
				if (_prefs.MaxSelect != _selectedCards.Count)
				{
					return;
				}
				GetViewport().GuiReleaseFocus();
				_upgradeMultiPreviewContainer.Visible = true;
				_upgradeMultiPreviewContainer.MouseFilter = MouseFilterEnum.Stop;
				_multiPreviewCancelButton.Enable();
				_multiPreviewConfirmButton.Enable();
				foreach (CardModel selectedCard in _selectedCards)
				{
					_grid.UnhighlightCard(selectedCard);
					CardModel cardModel = _runState.CloneCard(selectedCard);
					cardModel.UpgradeInternal();
					cardModel.UpgradePreviewType = CardUpgradePreviewType.Deck;
					NCard nCard = NCard.Create(cardModel);
					_multiPreview.AddChildSafely(NPreviewCardHolder.Create(nCard, showHoverTips: true, scaleOnHover: false));
					nCard.ShowUpgradePreview();
					_grid.SetCanScroll(canScroll: false);
					_closeButton.Disable();
				}
			}
		}
		else
		{
			_selectedCards.Remove(card);
			_grid.UnhighlightCard(card);
		}
	}

	private void CloseSelection(NButton _)
	{
		_completionSource.SetResult(Array.Empty<CardModel>());
		_singlePreviewCancelButton.Disable();
		_singlePreviewConfirmButton.Disable();
		_multiPreviewCancelButton.Disable();
		_multiPreviewConfirmButton.Disable();
		NOverlayStack.Instance.Remove(this);
	}

	private void CancelSelection(NButton _)
	{
		if (UseSingleSelection)
		{
			_upgradeSinglePreviewContainer.Visible = false;
			_upgradeSinglePreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
			_singlePreviewCancelButton.Disable();
			_singlePreviewConfirmButton.Disable();
		}
		else
		{
			_upgradeMultiPreviewContainer.Visible = false;
			_upgradeMultiPreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
			for (int i = 0; i < _multiPreview.GetChildCount(); i++)
			{
				_multiPreview.GetChild(i).QueueFreeSafely();
			}
			_multiPreviewCancelButton.Disable();
			_multiPreviewConfirmButton.Disable();
		}
		_grid.SetCanScroll(canScroll: true);
		if (_prefs.Cancelable)
		{
			_closeButton.Enable();
		}
		foreach (CardModel selectedCard in _selectedCards)
		{
			_grid.UnhighlightCard(selectedCard);
		}
		_grid.GetCardHolder(_selectedCards.Last())?.TryGrabFocus();
		_selectedCards.Clear();
	}

	private void ConfirmSelection(NButton _)
	{
		if (_selectedCards.Count != 0)
		{
			CheckIfSelectionComplete();
		}
	}

	private void CheckIfSelectionComplete()
	{
		_singlePreviewCancelButton.Enable();
		_singlePreviewConfirmButton.Enable();
		if (_selectedCards.Count >= _prefs.MaxSelect)
		{
			_completionSource.SetResult(_selectedCards);
			NOverlayStack.Instance.Remove(this);
		}
	}

	private void ToggleShowUpgrades(NTickbox tickbox)
	{
		_grid.IsShowingUpgrades = tickbox.IsTicked;
	}

	private void OnControllerStateUpdated()
	{
		_viewUpgrades.Visible = !NControllerManager.Instance.IsUsingController;
		if (NControllerManager.Instance.IsUsingController)
		{
			_viewUpgrades.IsTicked = false;
			ToggleShowUpgrades(_viewUpgrades);
		}
	}
}
