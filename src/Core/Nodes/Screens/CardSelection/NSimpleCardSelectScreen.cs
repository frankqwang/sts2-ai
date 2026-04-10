using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

public sealed partial class NSimpleCardSelectScreen : NCardGridSelectionScreen
{
	private Control _bottomTextContainer;

	private MegaRichTextLabel _infoLabel;

	private NConfirmButton _confirmButton;

	private NCombatPilesContainer _combatPiles;

	private readonly HashSet<CardModel> _selectedCards = new HashSet<CardModel>();

	private CardSelectorPrefs _prefs;

	private List<CardCreationResult>? _cardResults;

	private static string ScenePath => SceneHelper.GetScenePath("screens/card_selection/simple_card_select_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	protected override IEnumerable<Control> PeekButtonTargets => new global::_003C_003Ez__ReadOnlySingleElementList<Control>(_bottomTextContainer);

	public static NSimpleCardSelectScreen Create(IReadOnlyList<CardModel> cards, CardSelectorPrefs prefs)
	{
		NSimpleCardSelectScreen nSimpleCardSelectScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NSimpleCardSelectScreen>(PackedScene.GenEditState.Disabled);
		nSimpleCardSelectScreen.Name = "NSimpleCardSelectScreen";
		nSimpleCardSelectScreen._cards = cards.ToList();
		nSimpleCardSelectScreen._cardResults = null;
		nSimpleCardSelectScreen._prefs = prefs;
		return nSimpleCardSelectScreen;
	}

	public static NSimpleCardSelectScreen Create(IReadOnlyList<CardCreationResult> cards, CardSelectorPrefs prefs)
	{
		NSimpleCardSelectScreen nSimpleCardSelectScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NSimpleCardSelectScreen>(PackedScene.GenEditState.Disabled);
		nSimpleCardSelectScreen.Name = "NSimpleCardSelectScreen";
		nSimpleCardSelectScreen._cards = cards.Select((CardCreationResult r) => r.Card).ToList();
		nSimpleCardSelectScreen._cardResults = cards.ToList();
		nSimpleCardSelectScreen._prefs = prefs;
		return nSimpleCardSelectScreen;
	}

	public override void _Ready()
	{
		ConnectSignalsAndInitGrid();
		_confirmButton = GetNode<NConfirmButton>("%Confirm");
		_bottomTextContainer = GetNode<Control>("%BottomText");
		_infoLabel = _bottomTextContainer.GetNode<MegaRichTextLabel>("%BottomLabel");
		_infoLabel.Text = _prefs.Prompt.GetFormattedText();
		if (_prefs.MinSelect == 0)
		{
			_confirmButton.Enable();
		}
		else
		{
			_confirmButton.Disable();
		}
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			CompleteSelection();
		}));
	}

	protected override void ConnectSignalsAndInitGrid()
	{
		base.ConnectSignalsAndInitGrid();
		_combatPiles = GetNode<NCombatPilesContainer>("%CombatPiles");
		if (CombatManager.Instance.IsInProgress)
		{
			_combatPiles.Initialize(_cards.First().Owner);
		}
		_combatPiles.Disable();
		_combatPiles.SetVisible(visible: false);
		_peekButton.Connect(NPeekButton.SignalName.Toggled, Callable.From<NPeekButton>(delegate
		{
			if (_peekButton.IsPeeking)
			{
				_combatPiles.Enable();
				_combatPiles.SetVisible(visible: true);
			}
			else
			{
				_combatPiles.Disable();
				_combatPiles.SetVisible(visible: false);
			}
		}));
	}

	public override void AfterOverlayOpened()
	{
		base.AfterOverlayOpened();
		TaskHelper.RunSafely(FlashRelicsOnModifiedCards());
	}

	private async Task FlashRelicsOnModifiedCards()
	{
		if (_cardResults == null)
		{
			return;
		}
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		foreach (CardCreationResult result in _cardResults)
		{
			NGridCardHolder nGridCardHolder = _grid.CurrentlyDisplayedCardHolders.FirstOrDefault((NGridCardHolder h) => h.CardModel == result.Card);
			if (nGridCardHolder == null || !result.HasBeenModified)
			{
				continue;
			}
			foreach (RelicModel modifyingRelic in result.ModifyingRelics)
			{
				modifyingRelic.Flash();
				nGridCardHolder.CardNode?.FlashRelicOnCard(modifyingRelic);
			}
		}
	}

	protected override void OnCardClicked(CardModel card)
	{
		if (_selectedCards.Contains(card))
		{
			_grid.UnhighlightCard(card);
			_selectedCards.Remove(card);
		}
		else
		{
			if (_selectedCards.Count < _prefs.MaxSelect)
			{
				_grid.HighlightCard(card);
				_selectedCards.Add(card);
			}
			if (!_prefs.RequireManualConfirmation)
			{
				CheckIfSelectionComplete();
			}
		}
		if (_selectedCards.Count >= _prefs.MinSelect && _prefs.RequireManualConfirmation)
		{
			_confirmButton.Enable();
		}
		else
		{
			_confirmButton.Disable();
		}
	}

	private void CheckIfSelectionComplete()
	{
		if (_selectedCards.Count >= _prefs.MaxSelect)
		{
			CompleteSelection();
		}
	}

	private void CompleteSelection()
	{
		_completionSource.SetResult(_selectedCards);
		NOverlayStack.Instance.Remove(this);
	}
}
