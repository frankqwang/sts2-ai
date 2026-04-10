using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;

public partial class NDeckHistory : VBoxContainer
{
	[Signal]
	public delegate void HoveredEventHandler(NDeckHistoryEntry deckHistoryEntry);

	[Signal]
	public delegate void UnhoveredEventHandler(NDeckHistoryEntry deckHistoryEntry);

	private readonly LocString _deckHeader = new LocString("run_history", "DECK_HISTORY.header");

	private readonly LocString _cardCategories = new LocString("run_history", "DECK_HISTORY.categories");

	private MegaRichTextLabel _headerLabel;

	private Control _cardContainer;

	private readonly List<CardModel> _allCards = new List<CardModel>();

	public override void _Ready()
	{
		_headerLabel = GetNode<MegaRichTextLabel>("Header");
		_cardContainer = GetNode<Control>("%CardContainer");
	}

	public void LoadDeck(Player player, IEnumerable<SerializableCard> cards)
	{
		StringBuilder stringBuilder = new StringBuilder();
		Dictionary<CardRarity, int> dictionary = new Dictionary<CardRarity, int>();
		CardRarity[] values = Enum.GetValues<CardRarity>();
		foreach (CardRarity key in values)
		{
			dictionary.Add(key, 0);
		}
		List<SerializableCard> list = cards.ToList();
		CardRarity key2;
		int value;
		foreach (SerializableCard item in list)
		{
			CardModel cardModel = SaveUtil.CardOrDeprecated(item.Id);
			key2 = cardModel.Rarity;
			value = dictionary[key2]++;
		}
		_deckHeader.Add("totalCards", list.Count);
		foreach (KeyValuePair<CardRarity, int> item2 in dictionary)
		{
			item2.Deconstruct(out key2, out value);
			CardRarity cardRarity = key2;
			int num = value;
			_cardCategories.Add(cardRarity.ToString() + "Cards", num);
		}
		StringBuilder stringBuilder2 = stringBuilder;
		StringBuilder.AppendInterpolatedStringHandler handler = new StringBuilder.AppendInterpolatedStringHandler(20, 1, stringBuilder2);
		handler.AppendLiteral("[gold][b]");
		handler.AppendFormatted(_deckHeader.GetFormattedText());
		handler.AppendLiteral("[/b][/gold]");
		stringBuilder2.Append(ref handler);
		stringBuilder.Append(_cardCategories.GetFormattedText().Trim(','));
		_headerLabel.Text = stringBuilder.ToString();
		PopulateCards(player, list);
	}

	private void PopulateCards(Player player, IEnumerable<SerializableCard> cards)
	{
		foreach (Node child in _cardContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
		_allCards.Clear();
		foreach (IGrouping<SerializableCard, SerializableCard> item in from x in cards
			group x by x)
		{
			CardModel cardModel = CardModel.FromSerializable(item.Key);
			cardModel.Owner = player;
			_allCards.Add(cardModel);
			NDeckHistoryEntry entry = NDeckHistoryEntry.Create(cardModel, item.Count(), from c in item
				where c.FloorAddedToDeck.HasValue
				select c.FloorAddedToDeck.Value);
			entry.Connect(NDeckHistoryEntry.SignalName.Clicked, Callable.From<NDeckHistoryEntry>(ShowEntry));
			entry.Connect(NClickableControl.SignalName.Focused, Callable.From<NClickableControl>(delegate
			{
				EmitSignal(SignalName.Hovered, entry);
			}));
			entry.Connect(NClickableControl.SignalName.Unfocused, Callable.From<NClickableControl>(delegate
			{
				EmitSignal(SignalName.Unhovered, entry);
			}));
			_cardContainer.AddChildSafely(entry);
		}
	}

	private void ShowEntry(NDeckHistoryEntry entry)
	{
		NGame.Instance.GetInspectCardScreen().Open(_allCards, _allCards.IndexOf(entry.Card));
	}
}
