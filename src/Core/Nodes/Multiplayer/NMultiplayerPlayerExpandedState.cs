using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Potions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NMultiplayerPlayerExpandedState : Control, ICapstoneScreen, IScreenContext
{
	private class CardGroupKey
	{
		private readonly CardModel _card;

		public CardGroupKey(CardModel card)
		{
			_card = card;
		}

		public override bool Equals(object? obj)
		{
			if (obj == null)
			{
				return false;
			}
			if (obj.GetType() != GetType())
			{
				return false;
			}
			CardGroupKey cardGroupKey = (CardGroupKey)obj;
			if (_card.Id.Equals(cardGroupKey._card.Id) && _card.CurrentUpgradeLevel == cardGroupKey._card.CurrentUpgradeLevel && _card.Enchantment?.Id == cardGroupKey._card.Enchantment?.Id)
			{
				return _card.Enchantment?.Amount == cardGroupKey._card.Enchantment?.Amount;
			}
			return false;
		}

		public override int GetHashCode()
		{
			return HashCode.Combine(_card.Id, _card.CurrentUpgradeLevel, _card.Enchantment?.Id, _card.Enchantment?.Amount);
		}
	}

	private static readonly string _scenePath = SceneHelper.GetScenePath("ui/multiplayer_player_expanded_state");

	private MegaRichTextLabel _playerNameLabel;

	private MegaRichTextLabel _cardsHeader;

	private Control _cardContainer;

	private NBackButton _backButton;

	private MegaRichTextLabel _potionsHeader;

	private Control _potionContainer;

	private MegaRichTextLabel _relicsHeader;

	private Control _relicContainer;

	private Player _player;

	private List<CardModel> _cards = new List<CardModel>();

	public bool UseSharedBackstop => true;

	public NetScreenType ScreenType => NetScreenType.RemotePlayerExpandedState;

	public Control? DefaultFocusedControl => null;

	public static NMultiplayerPlayerExpandedState Create(Player player)
	{
		NMultiplayerPlayerExpandedState nMultiplayerPlayerExpandedState = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NMultiplayerPlayerExpandedState>(PackedScene.GenEditState.Disabled);
		nMultiplayerPlayerExpandedState._player = player;
		return nMultiplayerPlayerExpandedState;
	}

	public void AfterCapstoneOpened()
	{
		_backButton.Enable();
		NGlobalUi globalUi = NRun.Instance.GlobalUi;
		globalUi.TopBar.AnimHide();
		globalUi.RelicInventory.AnimHide();
		globalUi.MultiplayerPlayerContainer.AnimHide();
		globalUi.MoveChild(globalUi.AboveTopBarVfxContainer, globalUi.CapstoneContainer.GetIndex());
		globalUi.MoveChild(globalUi.CardPreviewContainer, globalUi.CapstoneContainer.GetIndex());
		globalUi.MoveChild(globalUi.MessyCardPreviewContainer, globalUi.CapstoneContainer.GetIndex());
	}

	public void AfterCapstoneClosed()
	{
		NGlobalUi globalUi = NRun.Instance.GlobalUi;
		globalUi.TopBar.AnimShow();
		globalUi.RelicInventory.AnimShow();
		globalUi.MultiplayerPlayerContainer.AnimShow();
		globalUi.MoveChild(globalUi.AboveTopBarVfxContainer, globalUi.TopBar.GetIndex() + 1);
		globalUi.MoveChild(globalUi.CardPreviewContainer, globalUi.TopBar.GetIndex() + 1);
		globalUi.MoveChild(globalUi.MessyCardPreviewContainer, globalUi.TopBar.GetIndex() + 1);
		this.QueueFreeSafely();
		_backButton.Disable();
	}

	public override void _Ready()
	{
		_playerNameLabel = GetNode<MegaRichTextLabel>("%PlayerNameLabel");
		_cardsHeader = GetNode<MegaRichTextLabel>("%CardsHeader");
		_cardContainer = GetNode<Control>("%CardContainer");
		_relicsHeader = GetNode<MegaRichTextLabel>("%RelicsHeader");
		_relicContainer = GetNode<Control>("%RelicContainer");
		_potionsHeader = GetNode<MegaRichTextLabel>("%PotionsHeader");
		_potionContainer = GetNode<Control>("%PotionContainer");
		_backButton = GetNode<NBackButton>("%BackButton");
		LocString locString = new LocString("gameplay_ui", "MULTIPLAYER_EXPANDED_STATE.title");
		locString.Add("PlayerName", PlatformUtil.GetPlayerName(RunManager.Instance.NetService.Platform, _player.NetId));
		locString.Add("Character", _player.Character.Title);
		_playerNameLabel.Text = locString.GetFormattedText();
		LocString locString2 = new LocString("gameplay_ui", "MULTIPLAYER_EXPANDED_STATE.relicHeader");
		_relicsHeader.Text = locString2.GetFormattedText();
		LocString locString3 = new LocString("gameplay_ui", "MULTIPLAYER_EXPANDED_STATE.cardHeader");
		_cardsHeader.Text = locString3.GetFormattedText();
		LocString locString4 = new LocString("gameplay_ui", "MULTIPLAYER_EXPANDED_STATE.potionHeader");
		_potionsHeader.Text = locString4.GetFormattedText();
		foreach (RelicModel relic in _player.Relics)
		{
			NRelicBasicHolder holder = NRelicBasicHolder.Create(relic);
			_relicContainer.AddChildSafely(holder);
			holder.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
			{
				OnRelicClicked(holder.Relic);
			}));
			holder.MouseDefaultCursorShape = CursorShape.Help;
		}
		foreach (PotionModel potion in _player.Potions)
		{
			NPotionHolder nPotionHolder = NPotionHolder.Create(isUsable: false);
			NPotion nPotion = NPotion.Create(potion);
			_potionContainer.AddChildSafely(nPotionHolder);
			nPotionHolder.AddPotion(nPotion);
			nPotion.Position = Vector2.Zero;
		}
		_cards.Clear();
		_cards.AddRange(_player.Deck.Cards);
		foreach (IGrouping<CardGroupKey, CardModel> item in from x in _player.Deck.Cards
			group x by new CardGroupKey(x))
		{
			NDeckHistoryEntry nDeckHistoryEntry = NDeckHistoryEntry.Create(item.First(), item.Count());
			nDeckHistoryEntry.Connect(NDeckHistoryEntry.SignalName.Clicked, Callable.From<NDeckHistoryEntry>(ShowEntry));
			_cardContainer.AddChildSafely(nDeckHistoryEntry);
		}
		_backButton.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(BackButtonPressed));
		UpdateNavigation();
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (!IsVisibleInTree() || NDevConsole.Instance.Visible || !NControllerManager.Instance.IsUsingController)
		{
			return;
		}
		Control control = GetViewport().GuiGetFocusOwner();
		bool flag = ((control is TextEdit || control is LineEdit) ? true : false);
		if (!flag && ActiveScreenContext.Instance.IsCurrent(this))
		{
			Control control2 = GetViewport().GuiGetFocusOwner();
			if ((control2 == null || !IsAncestorOf(control2)) && (inputEvent.IsActionPressed(MegaInput.left) || inputEvent.IsActionPressed(MegaInput.right) || inputEvent.IsActionPressed(MegaInput.up) || inputEvent.IsActionPressed(MegaInput.down) || inputEvent.IsActionPressed(MegaInput.select)))
			{
				_relicContainer.GetChild<NRelicBasicHolder>(0).TryGrabFocus();
				GetViewport()?.SetInputAsHandled();
			}
		}
	}

	private void ShowEntry(NDeckHistoryEntry entry)
	{
		NGame.Instance.GetInspectCardScreen().Open(_cards, _cards.IndexOf(entry.Card));
	}

	private void BackButtonPressed(NButton _)
	{
		NCapstoneContainer.Instance.Close();
	}

	private void OnRelicClicked(NRelic node)
	{
		List<RelicModel> list = new List<RelicModel>();
		foreach (NRelicBasicHolder item in _relicContainer.GetChildren().OfType<NRelicBasicHolder>())
		{
			list.Add(item.Relic.Model);
		}
		NGame.Instance.GetInspectRelicScreen().Open(list, node.Model);
	}

	private void UpdateNavigation()
	{
		for (int i = 0; i < _relicContainer.GetChildCount(); i++)
		{
			NRelicBasicHolder child = _relicContainer.GetChild<NRelicBasicHolder>(i);
			child.FocusNeighborLeft = ((i > 0) ? _relicContainer.GetChild<NRelicBasicHolder>(i - 1).GetPath() : _relicContainer.GetChild<NRelicBasicHolder>(i).GetPath());
			child.FocusNeighborRight = ((i < _relicContainer.GetChildCount() - 1) ? _relicContainer.GetChild<NRelicBasicHolder>(i + 1).GetPath() : _relicContainer.GetChild<NRelicBasicHolder>(i).GetPath());
			child.FocusNeighborTop = child.GetPath();
			if (_potionContainer.GetChildCount() > 0)
			{
				child.FocusNeighborBottom = _potionContainer.GetChild<Control>(Mathf.Min(i, _potionContainer.GetChildCount() - 1))?.GetPath();
			}
			else if (_cardContainer.GetChildCount() > 0)
			{
				child.FocusNeighborBottom = _cardContainer.GetChild<Control>(Mathf.Min(i, _cardContainer.GetChildCount() - 1))?.GetPath();
			}
			else
			{
				child.FocusNeighborBottom = child.GetPath();
			}
		}
		for (int j = 0; j < _potionContainer.GetChildCount(); j++)
		{
			NPotionHolder child2 = _potionContainer.GetChild<NPotionHolder>(j);
			child2.FocusNeighborLeft = ((j > 0) ? _potionContainer.GetChild<NPotionHolder>(j - 1).GetPath() : _potionContainer.GetChild<NPotionHolder>(j).GetPath());
			child2.FocusNeighborRight = ((j < _potionContainer.GetChildCount() - 1) ? _potionContainer.GetChild<NPotionHolder>(j + 1).GetPath() : _potionContainer.GetChild<NPotionHolder>(j).GetPath());
			if (_relicContainer.GetChildCount() > 0)
			{
				child2.FocusNeighborTop = _relicContainer.GetChild<Control>(Mathf.Min(j, _relicContainer.GetChildCount() - 1))?.GetPath();
			}
			else
			{
				child2.FocusNeighborTop = child2.GetPath();
			}
			if (_cardContainer.GetChildCount() > 0)
			{
				child2.FocusNeighborBottom = _cardContainer.GetChild<Control>(Mathf.Min(j, _cardContainer.GetChildCount() - 1))?.GetPath();
			}
			else
			{
				child2.FocusNeighborBottom = child2.GetPath();
			}
		}
		for (int k = 0; k < _cardContainer.GetChildCount(); k++)
		{
			NDeckHistoryEntry child3 = _cardContainer.GetChild<NDeckHistoryEntry>(k);
			child3.FocusNeighborLeft = ((k > 0) ? _cardContainer.GetChild<NDeckHistoryEntry>(k - 1).GetPath() : _cardContainer.GetChild<NDeckHistoryEntry>(k).GetPath());
			child3.FocusNeighborRight = ((k < _cardContainer.GetChildCount() - 1) ? _cardContainer.GetChild<NDeckHistoryEntry>(k + 1).GetPath() : _cardContainer.GetChild<NDeckHistoryEntry>(k).GetPath());
			if (_potionContainer.GetChildCount() > 0)
			{
				child3.FocusNeighborTop = _potionContainer.GetChild<Control>(Mathf.Min(k, _potionContainer.GetChildCount() - 1))?.GetPath();
			}
			else if (_relicContainer.GetChildCount() > 0)
			{
				child3.FocusNeighborTop = _relicContainer.GetChild<Control>(Mathf.Min(k, _relicContainer.GetChildCount() - 1))?.GetPath();
			}
			else
			{
				child3.FocusNeighborTop = child3.GetPath();
			}
			child3.FocusNeighborBottom = child3.GetPath();
		}
	}
}
