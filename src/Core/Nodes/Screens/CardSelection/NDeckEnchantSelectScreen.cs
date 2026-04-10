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
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public sealed partial class NDeckEnchantSelectScreen : NCardGridSelectionScreen
{
	private readonly HashSet<CardModel> _selectedCards = new HashSet<CardModel>();

	private CardSelectorPrefs _prefs;

	private EnchantmentModel _enchantment;

	private int _enchantmentAmount;

	private Control _enchantSinglePreviewContainer;

	private NEnchantPreview _singlePreview;

	private NBackButton _singlePreviewCancelButton;

	private NConfirmButton _singlePreviewConfirmButton;

	private NConfirmButton _confirmButton;

	private Control _enchantMultiPreviewContainer;

	private Control _multiPreview;

	private NBackButton _multiPreviewCancelButton;

	private NConfirmButton _multiPreviewConfirmButton;

	private Control _enchantmentDescriptionContainer;

	private MegaLabel _enchantmentTitle;

	private MegaRichTextLabel _enchantmentDescription;

	private TextureRect _enchantmentIcon;

	private Control _bottomTextContainer;

	private MegaRichTextLabel _infoLabel;

	private NBackButton _closeButton;

	private static string ScenePath => SceneHelper.GetScenePath("screens/card_selection/deck_enchant_select_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	private bool UseSingleSelection => _prefs.MaxSelect == 1;

	protected override IEnumerable<Control> PeekButtonTargets => new global::_003C_003Ez__ReadOnlyArray<Control>(new Control[5] { _enchantSinglePreviewContainer, _enchantMultiPreviewContainer, _enchantmentDescriptionContainer, _closeButton, _bottomTextContainer });

	public override Control? DefaultFocusedControl
	{
		get
		{
			if (_enchantSinglePreviewContainer.Visible || _enchantMultiPreviewContainer.Visible)
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
			if (_enchantSinglePreviewContainer.Visible || _enchantMultiPreviewContainer.Visible)
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
		_enchantSinglePreviewContainer = GetNode<Control>("%EnchantSinglePreviewContainer");
		_singlePreview = _enchantSinglePreviewContainer.GetNode<NEnchantPreview>("EnchantPreview");
		_singlePreviewCancelButton = _enchantSinglePreviewContainer.GetNode<NBackButton>("Cancel");
		_singlePreviewConfirmButton = _enchantSinglePreviewContainer.GetNode<NConfirmButton>("Confirm");
		_enchantMultiPreviewContainer = GetNode<Control>("%EnchantMultiPreviewContainer");
		_multiPreview = _enchantMultiPreviewContainer.GetNode<Control>("Cards");
		_multiPreviewCancelButton = _enchantMultiPreviewContainer.GetNode<NBackButton>("Cancel");
		_multiPreviewConfirmButton = _enchantMultiPreviewContainer.GetNode<NConfirmButton>("Confirm");
		_enchantmentDescriptionContainer = GetNode<Control>("%EnchantmentDescriptionContainer");
		_enchantmentIcon = _enchantmentDescriptionContainer.GetNode<TextureRect>("%EnchantmentIcon");
		_enchantmentTitle = _enchantmentDescriptionContainer.GetNode<MegaLabel>("%EnchantmentTitle");
		_enchantmentDescription = _enchantmentDescriptionContainer.GetNode<MegaRichTextLabel>("%EnchantmentDescription");
		_closeButton = GetNode<NBackButton>("%Close");
		_singlePreviewCancelButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CancelSelection));
		_singlePreviewConfirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(ConfirmSelection));
		_multiPreviewCancelButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(CancelSelection));
		_multiPreviewConfirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(ConfirmSelection));
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
		EnchantmentModel enchantmentModel = _enchantment.ToMutable();
		enchantmentModel.Amount = _enchantmentAmount;
		enchantmentModel.RecalculateValues();
		_enchantmentTitle.SetTextAutoSize(enchantmentModel.Title.GetFormattedText());
		_enchantmentDescription.Text = enchantmentModel.DynamicDescription.GetFormattedText();
		_enchantmentIcon.Texture = enchantmentModel.Icon;
		_enchantSinglePreviewContainer.Visible = false;
		_enchantSinglePreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
		_enchantMultiPreviewContainer.Visible = false;
		_enchantMultiPreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
		RefreshConfirmButtonVisibility();
		_bottomTextContainer = GetNode<Control>("%BottomText");
		_infoLabel = _bottomTextContainer.GetNode<MegaRichTextLabel>("%BottomLabel");
		_infoLabel.Text = "[center]" + _prefs.Prompt.GetFormattedText() + "[/center]";
	}

	public static NDeckEnchantSelectScreen ShowScreen(IReadOnlyList<CardModel> cards, EnchantmentModel enchantment, int amount, CardSelectorPrefs prefs)
	{
		NDeckEnchantSelectScreen nDeckEnchantSelectScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NDeckEnchantSelectScreen>(PackedScene.GenEditState.Disabled);
		nDeckEnchantSelectScreen.Name = "NDeckEnchantSelectScreen";
		nDeckEnchantSelectScreen._cards = cards;
		nDeckEnchantSelectScreen._prefs = prefs;
		nDeckEnchantSelectScreen._enchantment = enchantment;
		nDeckEnchantSelectScreen._enchantmentAmount = amount;
		NOverlayStack.Instance.Push(nDeckEnchantSelectScreen);
		return nDeckEnchantSelectScreen;
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

	private void CloseSelection(NButton _)
	{
		_completionSource.SetResult(Array.Empty<CardModel>());
		NOverlayStack.Instance.Remove(this);
	}

	private void CancelSelection(NButton _)
	{
		if (UseSingleSelection)
		{
			_singlePreviewCancelButton.Disable();
			_singlePreviewConfirmButton.Disable();
			_enchantSinglePreviewContainer.Visible = false;
			_enchantSinglePreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
		}
		else
		{
			_multiPreviewCancelButton.Disable();
			_multiPreviewConfirmButton.Disable();
			_enchantMultiPreviewContainer.Visible = false;
			_enchantMultiPreviewContainer.MouseFilter = MouseFilterEnum.Ignore;
			for (int i = 0; i < _multiPreview.GetChildCount(); i++)
			{
				_multiPreview.GetChild(i).QueueFreeSafely();
			}
		}
		_grid.SetCanScroll(canScroll: true);
		ActiveScreenContext.Instance.Update();
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

	private void PreviewSelection(NButton _)
	{
		PreviewSelection();
	}

	private void PreviewSelection()
	{
		if (UseSingleSelection)
		{
			_grid.SetCanScroll(canScroll: false);
			_closeButton.Disable();
			GetViewport().GuiReleaseFocus();
			_enchantSinglePreviewContainer.Visible = true;
			_enchantSinglePreviewContainer.MouseFilter = MouseFilterEnum.Stop;
			_singlePreview.Init(_selectedCards.First(), _enchantment, _enchantmentAmount);
			_singlePreviewCancelButton.Enable();
			_singlePreviewConfirmButton.Enable();
			return;
		}
		_grid.SetCanScroll(canScroll: false);
		_closeButton.Disable();
		GetViewport().GuiReleaseFocus();
		_enchantMultiPreviewContainer.Visible = true;
		_enchantMultiPreviewContainer.MouseFilter = MouseFilterEnum.Stop;
		_multiPreviewCancelButton.Enable();
		_multiPreviewConfirmButton.Enable();
		foreach (CardModel selectedCard in _selectedCards)
		{
			NCard nCard = NCard.Create(selectedCard);
			_multiPreview.AddChildSafely(NPreviewCardHolder.Create(nCard, showHoverTips: true, scaleOnHover: false));
			nCard.UpdateVisuals(selectedCard.Pile.Type, CardPreviewMode.Normal);
		}
	}

	private void ConfirmSelection(NButton inputEvent)
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
		if (_selectedCards.Count >= _prefs.MinSelect && _selectedCards.Count <= _prefs.MaxSelect)
		{
			_completionSource.SetResult(_selectedCards);
			NOverlayStack.Instance.Remove(this);
		}
	}
}
