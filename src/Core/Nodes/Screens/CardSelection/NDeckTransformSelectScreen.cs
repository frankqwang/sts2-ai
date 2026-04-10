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
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public sealed partial class NDeckTransformSelectScreen : NCardGridSelectionScreen
{
	private readonly HashSet<CardModel> _selectedCards = new HashSet<CardModel>();

	private Func<CardModel, CardTransformation> _cardToTransformation;

	private CardSelectorPrefs _prefs;

	private Control _previewContainer;

	private NTransformPreview _transformPreview;

	private NConfirmButton _confirmButton;

	private NBackButton _previewCancelButton;

	private NConfirmButton _previewConfirmButton;

	private Control _bottomTextContainer;

	private MegaRichTextLabel _infoLabel;

	private NTickbox _viewUpgrades;

	private NBackButton _closeButton;

	private static string ScenePath => SceneHelper.GetScenePath("screens/card_selection/deck_transform_select_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	protected override IEnumerable<Control> PeekButtonTargets => new global::_003C_003Ez__ReadOnlyArray<Control>(new Control[3] { _previewContainer, _closeButton, _bottomTextContainer });

	public override Control? DefaultFocusedControl
	{
		get
		{
			if (_previewContainer.Visible)
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
			if (_previewContainer.Visible)
			{
				return null;
			}
			return _grid.FocusedControlFromTopBar;
		}
	}

	public override void _Ready()
	{
		ConnectSignalsAndInitGrid();
		_confirmButton = GetNode<NConfirmButton>("Confirm");
		_previewContainer = GetNode<Control>("%PreviewContainer");
		_transformPreview = _previewContainer.GetNode<NTransformPreview>("TransformPreview");
		_previewCancelButton = _previewContainer.GetNode<NBackButton>("Cancel");
		_previewConfirmButton = _previewContainer.GetNode<NConfirmButton>("Confirm");
		_closeButton = GetNode<NBackButton>("%Close");
		_previewCancelButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CancelSelection));
		_previewConfirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CompleteSelection));
		_closeButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CloseSelection));
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(ConfirmSelection));
		if (_prefs.Cancelable)
		{
			_closeButton.Enable();
		}
		else
		{
			_closeButton.Disable();
		}
		RefreshConfirmButtonVisibility();
		_previewCancelButton.Disable();
		_previewConfirmButton.Disable();
		_bottomTextContainer = GetNode<Control>("%BottomText");
		_infoLabel = _bottomTextContainer.GetNode<MegaRichTextLabel>("%BottomLabel");
		_infoLabel.Text = _prefs.Prompt.GetFormattedText();
		_viewUpgrades = GetNode<NTickbox>("%Upgrades");
		_viewUpgrades.IsTicked = false;
		_viewUpgrades.Connect(NTickbox.SignalName.Toggled, Callable.From<NTickbox>(ToggleShowUpgrades));
		OnControllerStateUpdated();
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(OnControllerStateUpdated));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(OnControllerStateUpdated));
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(OnControllerStateUpdated));
		GetNode<MegaLabel>("%ViewUpgradesLabel").SetTextAutoSize(new LocString("card_selection", "VIEW_UPGRADES").GetFormattedText());
	}

	public static NDeckTransformSelectScreen ShowScreen(IReadOnlyList<CardModel> cards, Func<CardModel, CardTransformation> cardToTransformation, CardSelectorPrefs prefs)
	{
		NDeckTransformSelectScreen nDeckTransformSelectScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NDeckTransformSelectScreen>(PackedScene.GenEditState.Disabled);
		nDeckTransformSelectScreen.Name = "NDeckTransformSelectScreen";
		nDeckTransformSelectScreen._cards = cards;
		nDeckTransformSelectScreen._cardToTransformation = cardToTransformation;
		nDeckTransformSelectScreen._prefs = prefs;
		NOverlayStack.Instance.Push(nDeckTransformSelectScreen);
		return nDeckTransformSelectScreen;
	}

	private void RefreshConfirmButtonVisibility()
	{
		if (_prefs.MinSelect != _prefs.MaxSelect && _selectedCards.Count >= _prefs.MinSelect)
		{
			_confirmButton.Enable();
		}
		else
		{
			_confirmButton.Disable();
		}
	}

	protected override void OnCardClicked(CardModel card)
	{
		if (_selectedCards.Add(card))
		{
			_grid.HighlightCard(card);
			if (_prefs.MaxSelect == _selectedCards.Count)
			{
				OpenPreviewScreen();
			}
		}
		else
		{
			_selectedCards.Remove(card);
			_grid.UnhighlightCard(card);
		}
		RefreshConfirmButtonVisibility();
	}

	private void CloseSelection(NButton _)
	{
		_completionSource.SetResult(Array.Empty<CardModel>());
		_previewCancelButton.Disable();
		_previewConfirmButton.Disable();
		NOverlayStack.Instance.Remove(this);
	}

	private void CancelSelection(NButton _)
	{
		_previewContainer.Visible = false;
		_previewContainer.MouseFilter = MouseFilterEnum.Ignore;
		_transformPreview.Uninitialize();
		_previewCancelButton.Disable();
		_previewConfirmButton.Disable();
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
		if (_selectedCards.Count >= _prefs.MinSelect)
		{
			if (_prefs.RequireManualConfirmation)
			{
				OpenPreviewScreen();
			}
			else
			{
				CompleteSelection(_);
			}
		}
	}

	private void OpenPreviewScreen()
	{
		GetViewport().GuiReleaseFocus();
		_previewContainer.Visible = true;
		_previewContainer.MouseFilter = MouseFilterEnum.Stop;
		_previewCancelButton.Enable();
		_previewConfirmButton.Enable();
		foreach (CardModel selectedCard in _selectedCards)
		{
			_grid.UnhighlightCard(selectedCard);
		}
		_transformPreview.Initialize(_selectedCards.Select(_cardToTransformation));
		_closeButton.Disable();
	}

	private void CompleteSelection(NButton _)
	{
		_completionSource.SetResult(_selectedCards);
		NOverlayStack.Instance.Remove(this);
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
