using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.CardLibrary;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens;

public partial class NSimpleCardsViewScreen : NCardsViewScreen
{
	private NButton _confirmButton;

	private List<CardPileAddResult> _cardResults;

	private static string ScenePath => SceneHelper.GetScenePath("screens/simple_cards_view_screen");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public override NetScreenType ScreenType => NetScreenType.SimpleCardsView;

	public override void _Ready()
	{
		_confirmButton = GetNode<NButton>("ConfirmButton");
		GetNode<MegaLabel>("%ViewUpgradesLabel").SetTextAutoSize(new LocString("gameplay_ui", "VIEW_UPGRADES").GetFormattedText());
		ConnectSignals();
		NCardGrid grid = _grid;
		List<CardModel> cards = _cards;
		int num = 1;
		List<SortingOrders> list = new List<SortingOrders>(num);
		CollectionsMarshal.SetCount(list, num);
		Span<SortingOrders> span = CollectionsMarshal.AsSpan(list);
		int index = 0;
		span[index] = SortingOrders.Ascending;
		grid.SetCards(cards, PileType.Deck, list);
	}

	protected override void ConnectSignals()
	{
		base.ConnectSignals();
		_backButton.Disable();
		_confirmButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(base.OnReturnButtonPressed));
		_confirmButton.Enable();
	}

	public static NCardsViewScreen? ShowScreen(List<CardPileAddResult> cards, LocString infoText)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSimpleCardsViewScreen nSimpleCardsViewScreen = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NSimpleCardsViewScreen>(PackedScene.GenEditState.Disabled);
		nSimpleCardsViewScreen._cards = cards.Select((CardPileAddResult c) => c.cardAdded).ToList();
		nSimpleCardsViewScreen._cardResults = cards;
		nSimpleCardsViewScreen._infoText = infoText;
		NDebugAudioManager.Instance?.Play("map_open.mp3");
		NCapstoneContainer.Instance.Open(nSimpleCardsViewScreen);
		return nSimpleCardsViewScreen;
	}

	public override void AfterCapstoneOpened()
	{
		base.AfterCapstoneOpened();
		TaskHelper.RunSafely(FlashRelicsOnModifiedCards());
	}

	private async Task FlashRelicsOnModifiedCards()
	{
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		foreach (CardPileAddResult result in _cardResults)
		{
			NGridCardHolder nGridCardHolder = _grid.CurrentlyDisplayedCardHolders.FirstOrDefault((NGridCardHolder h) => h.CardModel == result.cardAdded);
			if (nGridCardHolder == null || result.modifyingModels == null || result.modifyingModels.Count == 0)
			{
				continue;
			}
			foreach (RelicModel item in result.modifyingModels.OfType<RelicModel>())
			{
				item.Flash();
				nGridCardHolder.CardNode?.FlashRelicOnCard(item);
			}
		}
	}
}
