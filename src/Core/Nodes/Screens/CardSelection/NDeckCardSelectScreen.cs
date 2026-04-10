using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public sealed partial class NDeckCardSelectScreen : NCardGridSelectionScreen
{
	private readonly HashSet<CardModel> _selectedCards = new HashSet<CardModel>();

	private CardSelectorPrefs _prefs;

	private Control _previewContainer;

	private Control _previewCards;

	private NBackButton _previewCancelButton;

	private NConfirmButton _previewConfirmButton;

	private NBackButton _closeButton;

	private NConfirmButton _confirmButton;

	private MegaRichTextLabel _infoLabel;

	private static string ScenePath => SceneHelper.GetScenePath("screens/card_selection/deck_card_select_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	protected override IEnumerable<Control> PeekButtonTargets => new global::_003C_003Ez__ReadOnlyArray<Control>(new Control[3] { _previewContainer, _closeButton, _confirmButton });

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
		_previewContainer = GetNode<Control>("%PreviewContainer");
		_previewCards = _previewContainer.GetNode<Control>("%Cards");
		_previewCancelButton = _previewContainer.GetNode<NBackButton>("%PreviewCancel");
		_previewConfirmButton = _previewContainer.GetNode<NConfirmButton>("%PreviewConfirm");
		_closeButton = GetNode<NBackButton>("%Close");
		_confirmButton = GetNode<NConfirmButton>("%Confirm");
		_infoLabel = GetNode<MegaRichTextLabel>("%BottomLabel");
		_previewCancelButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CancelSelection));
		_previewConfirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(ConfirmSelection));
		_closeButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CloseSelection));
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From((Action<NButton>)PreviewSelection));
		if (_prefs.Cancelable)
		{
			_closeButton.Enable();
		}
		else
		{
			_closeButton.Disable();
		}
		RefreshConfirmButtonVisibility();
		_previewContainer.Visible = false;
		_previewContainer.MouseFilter = MouseFilterEnum.Ignore;
		_previewCancelButton.Disable();
		_previewConfirmButton.Disable();
		_infoLabel.Text = _prefs.Prompt.GetFormattedText();
	}

	public static NDeckCardSelectScreen Create(IReadOnlyList<CardModel> cards, CardSelectorPrefs prefs)
	{
		NDeckCardSelectScreen nDeckCardSelectScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NDeckCardSelectScreen>(PackedScene.GenEditState.Disabled);
		nDeckCardSelectScreen.Name = "NDeckCardSelectScreen";
		nDeckCardSelectScreen._cards = cards;
		nDeckCardSelectScreen._prefs = prefs;
		return nDeckCardSelectScreen;
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
				PreviewSelection();
			}
		}
		else
		{
			_selectedCards.Remove(card);
			_grid.UnhighlightCard(card);
		}
		RefreshConfirmButtonVisibility();
	}

	private void PreviewSelection(NButton _)
	{
		PreviewSelection();
	}

	private void PreviewSelection()
	{
		GetViewport().GuiReleaseFocus();
		_previewContainer.Visible = true;
		_previewContainer.MouseFilter = MouseFilterEnum.Stop;
		_closeButton.Disable();
		_grid.SetCanScroll(canScroll: false);
		_previewCancelButton.Enable();
		_previewConfirmButton.Enable();
		foreach (CardModel selectedCard in _selectedCards)
		{
			_grid.UnhighlightCard(selectedCard);
			NCard nCard = NCard.Create(selectedCard);
			NPreviewCardHolder child = NPreviewCardHolder.Create(nCard, showHoverTips: true, scaleOnHover: false);
			_previewCards.AddChildSafely(child);
			nCard.UpdateVisuals(selectedCard.Pile.Type, CardPreviewMode.Normal);
		}
		Callable.From(delegate
		{
			_previewCards.PivotOffset = _previewCards.Size / 2f;
			float num = 1f;
			if (_selectedCards.Count > 6)
			{
				num = 0.55f;
			}
			else if (_selectedCards.Count > 3)
			{
				num = 0.8f;
			}
			_previewCards.Scale = Vector2.One * num;
		}).CallDeferred();
	}

	private void CloseSelection(NButton _)
	{
		_completionSource.SetResult(Array.Empty<CardModel>());
		NOverlayStack.Instance.Remove(this);
	}

	private void CancelSelection(NButton _)
	{
		_previewContainer.Visible = false;
		_previewCancelButton.Disable();
		_previewConfirmButton.Disable();
		_grid.SetCanScroll(canScroll: true);
		_previewContainer.MouseFilter = MouseFilterEnum.Ignore;
		for (int i = 0; i < _previewCards.GetChildCount(); i++)
		{
			_previewCards.GetChild(i).QueueFreeSafely();
		}
		_grid.GetCardHolder(_selectedCards.Last())?.TryGrabFocus();
		_selectedCards.Clear();
		if (_prefs.Cancelable)
		{
			_closeButton.Enable();
		}
	}

	private void ConfirmSelection(NButton _)
	{
		CheckIfSelectionComplete();
	}

	private void CheckIfSelectionComplete()
	{
		if (_selectedCards.Count >= _prefs.MinSelect)
		{
			_completionSource.SetResult(_selectedCards);
			NOverlayStack.Instance.Remove(this);
		}
	}

	public override void AfterOverlayShown()
	{
		if (_prefs.Cancelable)
		{
			_closeButton.Enable();
		}
	}

	public override void AfterOverlayHidden()
	{
		_closeButton.Disable();
	}
}
