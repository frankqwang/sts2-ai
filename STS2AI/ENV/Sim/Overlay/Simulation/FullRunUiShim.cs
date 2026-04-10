using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Simulation;

internal sealed class FullRunUiRewardItem
{
	public int Index { get; set; }

	public string Type { get; set; } = "unknown";

	public string? Label { get; set; }
}

internal sealed class FullRunUiCardOption
{
	public int Index { get; set; }

	public string? Id { get; set; }

	public string? Name { get; set; }

	public string? Type { get; set; }

	public string? Rarity { get; set; }

	public int? Cost { get; set; }

	public bool IsUpgraded { get; set; }

	public string? Description { get; set; }

	public List<string> Keywords { get; set; } = new List<string>();

	public bool IsSelected { get; set; }
}

internal sealed class FullRunUiRewardsState
{
	public bool CanProceed { get; set; }

	public List<FullRunUiRewardItem> Items { get; set; } = new List<FullRunUiRewardItem>();
}

internal sealed class FullRunUiCardRewardState
{
	public bool CanSkip { get; set; }

	public List<FullRunUiCardOption> Cards { get; set; } = new List<FullRunUiCardOption>();
}

internal sealed class FullRunUiRelicOption
{
	public int Index { get; set; }

	public string? Id { get; set; }

	public string? Name { get; set; }

	public string? Rarity { get; set; }

	public string? Description { get; set; }
}

internal sealed class FullRunUiRelicSelectState
{
	public bool CanSkip { get; set; }

	public List<FullRunUiRelicOption> Relics { get; set; } = new List<FullRunUiRelicOption>();
}

internal sealed class FullRunUiRestSiteOption
{
	public int Index { get; set; }

	public string Id { get; set; } = string.Empty;

	public string? Name { get; set; }

	public string? Description { get; set; }

	public bool IsEnabled { get; set; }
}

internal sealed class FullRunUiRestSiteState
{
	public bool CanProceed { get; set; }

	public List<FullRunUiRestSiteOption> Options { get; set; } = new List<FullRunUiRestSiteOption>();
}

internal sealed class FullRunUiShopPlayerState
{
	public int Gold { get; set; }

	public int OpenPotionSlots { get; set; }
}

internal sealed class FullRunUiShopItem
{
	public int Index { get; set; }

	public string Category { get; set; } = string.Empty;

	public int Cost { get; set; }

	public bool CanAfford { get; set; }

	public bool IsStocked { get; set; }

	public bool OnSale { get; set; }

	public string? CardId { get; set; }

	public string? CardName { get; set; }

	public string? CardType { get; set; }

	public string? CardRarity { get; set; }

	public string? CardDescription { get; set; }

	public string? RelicId { get; set; }

	public string? RelicName { get; set; }

	public string? RelicDescription { get; set; }

	public string? PotionId { get; set; }

	public string? PotionName { get; set; }

	public string? PotionDescription { get; set; }

	public string? Name { get; set; }

	public string? Description { get; set; }

	public List<string> Keywords { get; set; } = new List<string>();
}

internal sealed class FullRunUiShopState
{
	public bool IsOpen { get; set; }

	public bool CanProceed { get; set; }

	public FullRunUiShopPlayerState Player { get; set; } = new FullRunUiShopPlayerState();

	public List<FullRunUiShopItem> Items { get; set; } = new List<FullRunUiShopItem>();
}

internal sealed class FullRunUiTreasureState
{
	public bool CanProceed { get; set; }

	public List<FullRunUiRelicOption> Relics { get; set; } = new List<FullRunUiRelicOption>();
}

internal sealed class FullRunUiCardSelectState
{
	public string ScreenType { get; set; } = "unknown";

	public string? Prompt { get; set; }

	public int MinSelect { get; set; }

	public int MaxSelect { get; set; }

	public int SelectedCount { get; set; }

	public int RemainingPicks { get; set; }

	public bool CanConfirm { get; set; }

	public bool CanCancel { get; set; }

	public bool PreviewShowing { get; set; }

	public bool RequiresManualConfirmation { get; set; }

	public List<FullRunUiCardOption> Cards { get; set; } = new List<FullRunUiCardOption>();

	public List<FullRunUiCardOption> SelectedCards { get; set; } = new List<FullRunUiCardOption>();
}

internal static class FullRunUiShim
{
	public static FullRunUiRewardsState? InspectRewards()
	{
		if (NOverlayStack.Instance?.Peek() is not NRewardsScreen screen)
		{
			return null;
		}

		var state = new FullRunUiRewardsState
		{
			CanProceed = GetEnabledVisibleProceedButton(screen) != null
		};

		List<NRewardButton> buttons = EnumerateDescendants<NRewardButton>(screen)
			.Where(static button => button.IsVisibleInTree())
			.ToList();

		for (int index = 0; index < buttons.Count; index++)
		{
			Reward? reward = buttons[index].Reward;
			state.Items.Add(new FullRunUiRewardItem
			{
				Index = index,
				Type = RewardTypeForUi(reward),
				Label = reward?.Description?.GetFormattedText()
			});
		}

		return state;
	}

	public static FullRunUiCardRewardState? InspectCardReward()
	{
		if (NOverlayStack.Instance?.Peek() is not NCardRewardSelectionScreen screen)
		{
			return null;
		}

		var state = new FullRunUiCardRewardState();
		IReadOnlyList<CardCreationResult> options = ReadField<IReadOnlyList<CardCreationResult>>(screen, "_options") ?? Array.Empty<CardCreationResult>();
		for (int index = 0; index < options.Count; index++)
		{
			state.Cards.Add(BuildCardOption(options[index].Card, index, isSelected: false));
		}

		IReadOnlyList<CardRewardAlternative> alternatives = ReadField<IReadOnlyList<CardRewardAlternative>>(screen, "_extraOptions") ?? Array.Empty<CardRewardAlternative>();
		state.CanSkip = alternatives.Any(static option => string.Equals(option.OptionId, "skip", StringComparison.OrdinalIgnoreCase));
		return state;
	}

	public static FullRunUiCardSelectState? InspectCardSelect()
	{
		if (NOverlayStack.Instance?.Peek() is NChooseACardSelectionScreen chooseScreen)
		{
			return InspectChooseCardSelection(chooseScreen);
		}

		if (NOverlayStack.Instance?.Peek() is NCardGridSelectionScreen gridScreen)
		{
			return InspectGridCardSelection(gridScreen);
		}

		return null;
	}

	public static FullRunUiRelicSelectState? InspectRelicSelect()
	{
		if (NOverlayStack.Instance?.Peek() is not NChooseARelicSelection screen)
		{
			return null;
		}

		var state = new FullRunUiRelicSelectState();
		IReadOnlyList<RelicModel> relics = ReadField<IReadOnlyList<RelicModel>>(screen, "_relics") ?? Array.Empty<RelicModel>();
		for (int index = 0; index < relics.Count; index++)
		{
			state.Relics.Add(BuildRelicOption(relics[index], index));
		}

		NChoiceSelectionSkipButton? skipButton = screen.GetNodeOrNull<NChoiceSelectionSkipButton>("SkipButton");
		state.CanSkip = skipButton != null && skipButton.IsVisibleInTree() && skipButton.IsEnabled;
		return state;
	}

	public static FullRunUiRestSiteState? InspectRestSite()
	{
		NRestSiteRoom? room = NRestSiteRoom.Instance;
		if (room == null)
		{
			return null;
		}

		var state = new FullRunUiRestSiteState
		{
			CanProceed = room.ProceedButton != null && room.ProceedButton.IsVisibleInTree() && room.ProceedButton.IsEnabled
		};

		for (int index = 0; index < room.Options.Count; index++)
		{
			RestSiteOption option = room.Options[index];
			state.Options.Add(new FullRunUiRestSiteOption
			{
				Index = index,
				Id = option.OptionId,
				Name = option.Title.GetRawText(),
				Description = option.Description.GetRawText(),
				IsEnabled = option.IsEnabled
			});
		}

		return state;
	}

	public static FullRunUiShopState? InspectShop()
	{
		NMerchantRoom? room = NMerchantRoom.Instance;
		NMerchantInventory? inventoryNode = room?.Inventory;
		MerchantInventory? inventory = inventoryNode?.Inventory;
		Player? player = inventory?.Player;
		if (room == null || inventoryNode == null || inventory == null || player == null)
		{
			return null;
		}

		var state = new FullRunUiShopState
		{
			IsOpen = inventoryNode.IsOpen,
			CanProceed = room.ProceedButton.IsVisibleInTree() && room.ProceedButton.IsEnabled,
			Player = new FullRunUiShopPlayerState
			{
				Gold = player.Gold,
				OpenPotionSlots = player.PotionSlots.Count(static potion => potion == null)
			}
		};

		int index = 0;
		foreach (MerchantCardEntry entry in inventory.CharacterCardEntries)
		{
			state.Items.Add(BuildShopCardItem(entry, index++));
		}
		foreach (MerchantCardEntry entry in inventory.ColorlessCardEntries)
		{
			state.Items.Add(BuildShopCardItem(entry, index++));
		}
		foreach (MerchantRelicEntry entry in inventory.RelicEntries)
		{
			state.Items.Add(BuildShopRelicItem(entry, index++));
		}
		foreach (MerchantPotionEntry entry in inventory.PotionEntries)
		{
			state.Items.Add(BuildShopPotionItem(entry, index++));
		}
		if (inventory.CardRemovalEntry != null)
		{
			state.Items.Add(BuildShopCardRemovalItem(inventory.CardRemovalEntry, index));
		}

		return state;
	}

	public static FullRunUiTreasureState? InspectTreasure()
	{
		NTreasureRoom? room = NRun.Instance?.TreasureRoom;
		if (room == null)
		{
			return null;
		}

		var state = new FullRunUiTreasureState
		{
			CanProceed = room.ProceedButton != null && room.ProceedButton.IsVisibleInTree() && room.ProceedButton.IsEnabled
		};

		IReadOnlyList<RelicModel>? relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		if (relics != null)
		{
			for (int index = 0; index < relics.Count; index++)
			{
				state.Relics.Add(BuildRelicOption(relics[index], index));
			}
		}

		return state;
	}

	public static bool TryClaimReward(int index)
	{
		if (NOverlayStack.Instance?.Peek() is not NRewardsScreen screen)
		{
			return false;
		}

		List<NRewardButton> buttons = EnumerateDescendants<NRewardButton>(screen)
			.Where(static button => button.IsVisibleInTree() && button.IsEnabled)
			.ToList();
		NRewardButton? button = index >= 0 && index < buttons.Count ? buttons[index] : null;
		if (button == null)
		{
			return false;
		}

		if (!InvokeProtectedNoArgMethod(button, "OnRelease"))
		{
			button.EmitSignal(NClickableControl.SignalName.Released, button);
		}
		return true;
	}

	public static bool TryProceedRewards()
	{
		if (NOverlayStack.Instance?.Peek() is not NRewardsScreen screen)
		{
			return false;
		}

		NProceedButton? proceedButton = GetEnabledVisibleProceedButton(screen);
		if (proceedButton == null)
		{
			return false;
		}

		proceedButton.EmitSignal(NClickableControl.SignalName.Released, proceedButton);
		return true;
	}

	public static bool TrySelectCardReward(int index)
	{
		if (NOverlayStack.Instance?.Peek() is not NCardRewardSelectionScreen screen)
		{
			return false;
		}

		Control? cardRow = screen.GetNodeOrNull<Control>("UI/CardRow");
		NCardHolder? holder = cardRow?.GetChildren().OfType<NCardHolder>().ElementAtOrDefault(index);
		if (holder == null)
		{
			return false;
		}

		holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
		return true;
	}

	public static bool TrySkipCardReward()
	{
		if (NOverlayStack.Instance?.Peek() is not NCardRewardSelectionScreen screen)
		{
			return false;
		}

		IReadOnlyList<CardRewardAlternative> alternatives = ReadField<IReadOnlyList<CardRewardAlternative>>(screen, "_extraOptions") ?? Array.Empty<CardRewardAlternative>();
		if (!alternatives.Any(static option => string.Equals(option.OptionId, "skip", StringComparison.OrdinalIgnoreCase)))
		{
			return false;
		}

		Control? container = screen.GetNodeOrNull<Control>("UI/RewardAlternatives");
		NButton? button = container?.GetChildren().OfType<NButton>().FirstOrDefault(static child => child.IsVisibleInTree() && child.IsEnabled);
		if (button == null)
		{
			return false;
		}

		button.EmitSignal(NClickableControl.SignalName.Released, button);
		return true;
	}

	public static bool TrySelectCardOption(int index)
	{
		if (NOverlayStack.Instance?.Peek() is NChooseACardSelectionScreen chooseScreen)
		{
			Control? row = chooseScreen.GetNodeOrNull<Control>("CardRow");
			NCardHolder? directHolder = row?.GetChildren().OfType<NCardHolder>().ElementAtOrDefault(index);
			if (directHolder == null)
			{
				return false;
			}

			directHolder.EmitSignal(NCardHolder.SignalName.Pressed, directHolder);
			return true;
		}

		if (NOverlayStack.Instance?.Peek() is not NCardGridSelectionScreen gridScreen)
		{
			return false;
		}

		List<CardModel> cards = ReadField<IReadOnlyList<CardModel>>(gridScreen, "_cards", typeof(NCardGridSelectionScreen))?.ToList() ?? new List<CardModel>();
		CardModel? card = index >= 0 && index < cards.Count ? cards[index] : null;
		if (card == null)
		{
			return false;
		}

		NCardGrid? grid = gridScreen.GetNodeOrNull<NCardGrid>("%CardGrid");
		NGridCardHolder? holder = grid?.GetCardHolder(card);
		if (grid == null || holder == null)
		{
			return false;
		}

		grid.EmitSignal(NCardGrid.SignalName.HolderPressed, holder);
		return true;
	}

	public static bool TryConfirmCardSelection()
	{
		Control? screen = NOverlayStack.Instance?.Peek() as Control;
		NConfirmButton? button = screen == null ? null : GetEnabledVisibleConfirmButton(screen);
		if (button == null)
		{
			return false;
		}

		button.EmitSignal(NClickableControl.SignalName.Released, button);
		return true;
	}

	public static bool TrySelectRelic(int index)
	{
		if (NOverlayStack.Instance?.Peek() is not NChooseARelicSelection screen)
		{
			return false;
		}

		Control? row = screen.GetNodeOrNull<Control>("RelicRow");
		NRelicBasicHolder? holder = row?.GetChildren().OfType<NRelicBasicHolder>().ElementAtOrDefault(index);
		if (holder == null)
		{
			return false;
		}

		if (!InvokeProtectedNoArgMethod(holder, "OnRelease"))
		{
			holder.EmitSignal(NClickableControl.SignalName.Released, holder);
		}
		return true;
	}

	public static bool TrySkipRelicSelection()
	{
		if (NOverlayStack.Instance?.Peek() is not NChooseARelicSelection screen)
		{
			return false;
		}

		NChoiceSelectionSkipButton? skipButton = screen.GetNodeOrNull<NChoiceSelectionSkipButton>("SkipButton");
		if (skipButton == null || !skipButton.IsVisibleInTree() || !skipButton.IsEnabled)
		{
			return false;
		}

		if (!InvokeProtectedNoArgMethod(skipButton, "OnRelease"))
		{
			skipButton.EmitSignal(NClickableControl.SignalName.Released, skipButton);
		}
		return true;
	}

	public static bool TryChooseRestSiteOption(int index)
	{
		NRestSiteRoom? room = NRestSiteRoom.Instance;
		if (room == null || index < 0 || index >= room.Options.Count)
		{
			return false;
		}

		RestSiteOption option = room.Options[index];
		NRestSiteButton? button = room.GetButtonForOption(option);
		if (button == null || !button.IsVisibleInTree() || !button.IsEnabled)
		{
			return false;
		}

		if (!InvokeProtectedNoArgMethod(button, "OnRelease"))
		{
			button.EmitSignal(NClickableControl.SignalName.Released, button);
		}
		return true;
	}

	public static bool TryProceedRestSite()
	{
		NRestSiteRoom? room = NRestSiteRoom.Instance;
		if (room?.ProceedButton == null || !room.ProceedButton.IsVisibleInTree() || !room.ProceedButton.IsEnabled)
		{
			return false;
		}

		if (!InvokeProtectedNoArgMethod(room.ProceedButton, "OnRelease"))
		{
			room.ProceedButton.EmitSignal(NClickableControl.SignalName.Released, room.ProceedButton);
		}
		return true;
	}

	public static async Task<bool> TryPurchaseShopItemAsync(int index)
	{
		NMerchantRoom? room = NMerchantRoom.Instance;
		MerchantInventory? inventory = room?.Inventory?.Inventory;
		if (room == null || inventory == null)
		{
			return false;
		}

		MerchantEntry? entry = GetShopEntryByIndex(inventory, index);
		if (entry == null)
		{
			return false;
		}

		switch (entry)
		{
			case MerchantCardEntry cardEntry:
				await cardEntry.OnTryPurchaseWrapper(inventory);
				return true;
			case MerchantRelicEntry relicEntry:
				await relicEntry.OnTryPurchaseWrapper(inventory);
				return true;
			case MerchantPotionEntry potionEntry:
				await potionEntry.OnTryPurchaseWrapper(inventory);
				return true;
			case MerchantCardRemovalEntry cardRemovalEntry:
				await cardRemovalEntry.OnTryPurchaseWrapper(inventory);
				return true;
			default:
				return false;
		}
	}

	public static bool TryProceedShop()
	{
		NMerchantRoom? room = NMerchantRoom.Instance;
		if (room == null)
		{
			return false;
		}

		if (room.ProceedButton.IsVisibleInTree() && room.ProceedButton.IsEnabled)
		{
			if (!InvokeProtectedNoArgMethod(room.ProceedButton, "OnRelease"))
			{
				room.ProceedButton.EmitSignal(NClickableControl.SignalName.Released, room.ProceedButton);
			}
			return true;
		}

		NMerchantInventory? inventory = room.Inventory;
		if (inventory != null && inventory.IsOpen)
		{
			return InvokeAnyNoArgMethod(inventory, "Close");
		}

		return false;
	}

	public static async Task<bool> TryClaimTreasureRelicAsync(int index)
	{
		NTreasureRoom? room = NRun.Instance?.TreasureRoom;
		if (room == null || index < 0)
		{
			return false;
		}

		IReadOnlyList<RelicModel>? relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
		if (relics == null || index >= relics.Count)
		{
			return false;
		}

		NTreasureRoomRelicCollection? collection = room.GetNodeOrNull<NTreasureRoomRelicCollection>("%RelicCollection");
		if (collection == null)
		{
			return false;
		}

		if (!collection.Visible)
		{
			NButton? chestButton = room.GetNodeOrNull<NButton>("%Chest") ?? room.GetNodeOrNull<NButton>("Chest");
			if (chestButton == null || !chestButton.IsVisibleInTree() || !chestButton.IsEnabled)
			{
				return false;
			}

			if (!InvokeProtectedNoArgMethod(chestButton, "OnRelease"))
			{
				chestButton.EmitSignal(NClickableControl.SignalName.Released, chestButton);
			}
		}

		NTreasureRoomRelicHolder? holder = await WaitForTreasureRelicHolderAsync(room, index);
		if (holder == null || !holder.IsVisibleInTree() || !holder.IsEnabled)
		{
			return false;
		}

		if (!InvokeProtectedNoArgMethod(holder, "OnRelease"))
		{
			holder.EmitSignal(NClickableControl.SignalName.Released, holder);
		}
		return true;
	}

	public static bool TryProceedTreasure()
	{
		NTreasureRoom? room = NRun.Instance?.TreasureRoom;
		if (room?.ProceedButton == null || !room.ProceedButton.IsVisibleInTree() || !room.ProceedButton.IsEnabled)
		{
			return false;
		}

		if (!InvokeProtectedNoArgMethod(room.ProceedButton, "OnRelease"))
		{
			room.ProceedButton.EmitSignal(NClickableControl.SignalName.Released, room.ProceedButton);
		}
		return true;
	}

	public static bool TryCancelCardSelection()
	{
		if (NOverlayStack.Instance?.Peek() is NChooseACardSelectionScreen chooseScreen)
		{
			NButton? skip = chooseScreen.GetNodeOrNull<NButton>("SkipButton");
			if (skip == null || !skip.IsVisibleInTree() || !skip.IsEnabled)
			{
				return false;
			}

			skip.EmitSignal(NClickableControl.SignalName.Released, skip);
			return true;
		}

		Control? screen = NOverlayStack.Instance?.Peek() as Control;
		NButton? button = screen == null ? null : GetEnabledVisibleCancelButton(screen);
		if (button == null)
		{
			return false;
		}

		button.EmitSignal(NClickableControl.SignalName.Released, button);
		return true;
	}

	private static FullRunUiCardSelectState InspectChooseCardSelection(NChooseACardSelectionScreen screen)
	{
		Control? cardRow = screen.GetNodeOrNull<Control>("CardRow");
		NButton? skipButton = screen.GetNodeOrNull<NButton>("SkipButton");
		bool canCancel = skipButton != null && skipButton.IsVisibleInTree() && skipButton.IsEnabled;
		var state = new FullRunUiCardSelectState
		{
			ScreenType = "choose",
			Prompt = "choose_card",
			MinSelect = canCancel ? 0 : 1,
			MaxSelect = 1,
			SelectedCount = 0,
			RemainingPicks = 1,
			CanConfirm = false,
			CanCancel = canCancel,
			PreviewShowing = false,
			RequiresManualConfirmation = false
		};

		foreach (CardModel card in ReadField<IReadOnlyList<CardModel>>(screen, "_cards") ?? Array.Empty<CardModel>())
		{
			state.Cards.Add(BuildCardOption(card, state.Cards.Count, isSelected: false));
		}

		if (state.Cards.Count == 0 && cardRow != null)
		{
			foreach (NCardHolder holder in cardRow.GetChildren().OfType<NCardHolder>())
			{
				state.Cards.Add(BuildCardOption(holder.CardModel, state.Cards.Count, isSelected: false));
			}
		}

		return state;
	}

	private static FullRunUiCardSelectState InspectGridCardSelection(NCardGridSelectionScreen screen)
	{
		CardSelectorPrefs prefs = ReadField<CardSelectorPrefs>(screen, "_prefs");
		HashSet<CardModel> selected = ReadField<HashSet<CardModel>>(screen, "_selectedCards") ?? new HashSet<CardModel>();
		List<CardModel> cards = ReadField<IReadOnlyList<CardModel>>(screen, "_cards", typeof(NCardGridSelectionScreen))?.ToList() ?? new List<CardModel>();
		var state = new FullRunUiCardSelectState
		{
			ScreenType = ScreenTypeForSelection(screen),
			Prompt = prefs.Prompt.GetFormattedText(),
			MinSelect = prefs.MinSelect,
			MaxSelect = prefs.MaxSelect,
			SelectedCount = selected.Count,
			RemainingPicks = Math.Max(0, prefs.MaxSelect - selected.Count),
			CanConfirm = GetEnabledVisibleConfirmButton(screen) != null,
			CanCancel = GetEnabledVisibleCancelButton(screen) != null || prefs.Cancelable,
			PreviewShowing = HasVisiblePreviewContainer(screen),
			RequiresManualConfirmation = prefs.RequireManualConfirmation
		};

		for (int index = 0; index < cards.Count; index++)
		{
			bool isSelected = selected.Contains(cards[index]);
			FullRunUiCardOption option = BuildCardOption(cards[index], index, isSelected);
			state.Cards.Add(option);
			if (isSelected)
			{
				state.SelectedCards.Add(option);
			}
		}

		return state;
	}

	private static FullRunUiCardOption BuildCardOption(CardModel card, int index, bool isSelected)
	{
		return new FullRunUiCardOption
		{
			Index = index,
			Id = card.Id.Entry,
			Name = card.Title,
			Type = card.Type.ToString().ToLowerInvariant(),
			Rarity = card.Rarity.ToString().ToLowerInvariant(),
			Cost = card.EnergyCost.Canonical,
			IsUpgraded = card.IsUpgraded,
			Description = card.Description.GetRawText(),
			Keywords = card.Keywords?.Select(static keyword => keyword.ToString().ToLowerInvariant()).ToList() ?? new List<string>(),
			IsSelected = isSelected
		};
	}

	private static FullRunUiRelicOption BuildRelicOption(RelicModel relic, int index)
	{
		return new FullRunUiRelicOption
		{
			Index = index,
			Id = relic.Id.Entry,
			Name = relic.Title.GetRawText(),
			Rarity = relic.Rarity.ToString().ToLowerInvariant(),
			Description = relic.Description.GetRawText()
		};
	}

	private static FullRunUiShopItem BuildShopCardItem(MerchantCardEntry entry, int index)
	{
		CardModel? card = entry.CreationResult?.Card;
		return new FullRunUiShopItem
		{
			Index = index,
			Category = "card",
			Cost = entry.IsStocked ? entry.Cost : 0,
			CanAfford = entry.EnoughGold,
			IsStocked = entry.IsStocked,
			OnSale = entry.IsOnSale,
			CardId = card?.Id.Entry,
			CardName = card?.Title,
			CardType = card?.Type.ToString().ToLowerInvariant(),
			CardRarity = card?.Rarity.ToString().ToLowerInvariant(),
			CardDescription = card?.Description.GetRawText(),
			Name = card?.Title,
			Description = card?.Description.GetRawText(),
			Keywords = card?.Keywords?.Select(static keyword => keyword.ToString().ToLowerInvariant()).ToList() ?? new List<string>()
		};
	}

	private static FullRunUiShopItem BuildShopRelicItem(MerchantRelicEntry entry, int index)
	{
		RelicModel? relic = entry.Model;
		return new FullRunUiShopItem
		{
			Index = index,
			Category = "relic",
			Cost = entry.IsStocked ? entry.Cost : 0,
			CanAfford = entry.EnoughGold,
			IsStocked = entry.IsStocked,
			RelicId = relic?.Id.Entry,
			RelicName = relic?.Title.GetRawText(),
			RelicDescription = relic?.Description.GetRawText(),
			Name = relic?.Title.GetRawText(),
			Description = relic?.Description.GetRawText()
		};
	}

	private static FullRunUiShopItem BuildShopPotionItem(MerchantPotionEntry entry, int index)
	{
		PotionModel? potion = entry.Model;
		return new FullRunUiShopItem
		{
			Index = index,
			Category = "potion",
			Cost = entry.IsStocked ? entry.Cost : 0,
			CanAfford = entry.EnoughGold,
			IsStocked = entry.IsStocked,
			PotionId = potion?.Id.Entry,
			PotionName = potion?.Title.GetRawText(),
			PotionDescription = potion?.Description.GetRawText(),
			Name = potion?.Title.GetRawText(),
			Description = potion?.Description.GetRawText()
		};
	}

	private static FullRunUiShopItem BuildShopCardRemovalItem(MerchantCardRemovalEntry entry, int index)
	{
		LocString description = new LocString("merchant_room", "MERCHANT.cardRemovalService.description");
		description.Add("Amount", entry.CalcPriceIncrease());
		return new FullRunUiShopItem
		{
			Index = index,
			Category = "card_removal",
			Cost = entry.IsStocked ? entry.Cost : 0,
			CanAfford = entry.EnoughGold,
			IsStocked = entry.IsStocked,
			Name = new LocString("merchant_room", "MERCHANT.cardRemovalService.title").GetRawText(),
			Description = description.GetFormattedText()
		};
	}

	private static async Task<NTreasureRoomRelicHolder?> WaitForTreasureRelicHolderAsync(NTreasureRoom room, int index)
	{
		for (int attempt = 0; attempt < 240; attempt++)
		{
			NTreasureRoomRelicCollection? collection = room.GetNodeOrNull<NTreasureRoomRelicCollection>("%RelicCollection");
			if (collection != null && collection.Visible)
			{
				NTreasureRoomRelicHolder? holder = ResolveTreasureRelicHolder(collection, index);
				if (holder != null)
				{
					return holder;
				}
			}

			if (Engine.GetMainLoop() is SceneTree tree)
			{
				await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
			}
			else
			{
				await Task.Yield();
			}
		}

		return null;
	}

	private static NTreasureRoomRelicHolder? ResolveTreasureRelicHolder(NTreasureRoomRelicCollection collection, int index)
	{
		if (collection.SingleplayerRelicHolder != null &&
			collection.SingleplayerRelicHolder.Visible &&
			collection.SingleplayerRelicHolder.Index == index)
		{
			return collection.SingleplayerRelicHolder;
		}

		return EnumerateDescendants<NTreasureRoomRelicHolder>(collection)
			.FirstOrDefault(holder => holder.Visible && holder.Index == index);
	}

	private static string ScreenTypeForSelection(NCardGridSelectionScreen screen)
	{
		return screen switch
		{
			NSimpleCardSelectScreen => "simple_select",
			NDeckUpgradeSelectScreen => "upgrade",
			NDeckTransformSelectScreen => "transform",
			NDeckEnchantSelectScreen => "enchant",
			NDeckCardSelectScreen => "deck_select",
			_ => screen.GetType().Name.Replace("N", string.Empty).Replace("Screen", string.Empty).ToLowerInvariant()
		};
	}

	private static string RewardTypeForUi(Reward? reward)
	{
		return reward switch
		{
			GoldReward => "gold",
			PotionReward => "potion",
			RelicReward => "relic",
			CardReward => "card",
			LinkedRewardSet => "linked",
			_ => reward?.GetType().Name.Replace("Reward", string.Empty).ToLowerInvariant() ?? "unknown"
		};
	}

	private static bool HasVisiblePreviewContainer(Control screen)
	{
		return EnumerateDescendants<Control>(screen).Any(static control =>
			control.Visible && (control.Name.ToString().Contains("PreviewContainer", StringComparison.OrdinalIgnoreCase) ||
			control.Name.ToString().Contains("UpgradeSinglePreviewContainer", StringComparison.OrdinalIgnoreCase) ||
			control.Name.ToString().Contains("UpgradeMultiPreviewContainer", StringComparison.OrdinalIgnoreCase)));
	}

	private static NProceedButton? GetEnabledVisibleProceedButton(Control root)
	{
		return EnumerateDescendants<NProceedButton>(root).FirstOrDefault(static button => button.IsVisibleInTree() && button.IsEnabled);
	}

	private static NConfirmButton? GetEnabledVisibleConfirmButton(Control root)
	{
		List<NConfirmButton> buttons = EnumerateDescendants<NConfirmButton>(root)
			.Where(static button => button.IsVisibleInTree() && button.IsEnabled)
			.ToList();
		return buttons
			.OrderByDescending(static button => button.Name.ToString().Contains("PreviewConfirm", StringComparison.OrdinalIgnoreCase))
			.ThenByDescending(static button => button.Name.ToString().Equals("Confirm", StringComparison.OrdinalIgnoreCase))
			.FirstOrDefault();
	}

	private static NButton? GetEnabledVisibleCancelButton(Control root)
	{
		List<NButton> buttons = EnumerateDescendants<NButton>(root)
			.Where(static button => button.IsVisibleInTree() && button.IsEnabled)
			.ToList();
		return buttons
			.Where(static button =>
			{
				string name = button.Name.ToString();
				return name.Contains("Cancel", StringComparison.OrdinalIgnoreCase)
					|| name.Contains("Close", StringComparison.OrdinalIgnoreCase)
					|| name.Contains("Skip", StringComparison.OrdinalIgnoreCase);
			})
			.OrderByDescending(static button => button.Name.ToString().Contains("PreviewCancel", StringComparison.OrdinalIgnoreCase))
			.FirstOrDefault();
	}

	private static T? ReadField<T>(object instance, string fieldName, Type? declaringType = null)
	{
		FieldInfo? field = (declaringType ?? instance.GetType()).GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
		if (field == null)
		{
			return default;
		}

		object? value = field.GetValue(instance);
		if (value is T typed)
		{
			return typed;
		}

		if (typeof(T) == typeof(IReadOnlyList<CardModel>) && value is IEnumerable enumerable)
		{
			return (T)(object)enumerable.Cast<CardModel>().ToList();
		}

		return default;
	}

	private static bool InvokeProtectedNoArgMethod(object instance, string methodName)
	{
		for (Type? type = instance.GetType(); type != null; type = type.BaseType)
		{
			MethodInfo? method = type.GetMethod(methodName, BindingFlags.Instance | BindingFlags.NonPublic);
			if (method == null || method.GetParameters().Length != 0)
			{
				continue;
			}

			method.Invoke(instance, Array.Empty<object>());
			return true;
		}

		return false;
	}

	private static bool InvokeAnyNoArgMethod(object instance, string methodName)
	{
		for (Type? type = instance.GetType(); type != null; type = type.BaseType)
		{
			MethodInfo? method = type.GetMethod(methodName, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
			if (method == null || method.GetParameters().Length != 0)
			{
				continue;
			}

			method.Invoke(instance, Array.Empty<object>());
			return true;
		}

		return false;
	}

	private static MerchantEntry? GetShopEntryByIndex(MerchantInventory inventory, int index)
	{
		if (index < 0)
		{
			return null;
		}

		int currentIndex = 0;
		foreach (MerchantCardEntry entry in inventory.CharacterCardEntries)
		{
			if (currentIndex++ == index)
			{
				return entry;
			}
		}
		foreach (MerchantCardEntry entry in inventory.ColorlessCardEntries)
		{
			if (currentIndex++ == index)
			{
				return entry;
			}
		}
		foreach (MerchantRelicEntry entry in inventory.RelicEntries)
		{
			if (currentIndex++ == index)
			{
				return entry;
			}
		}
		foreach (MerchantPotionEntry entry in inventory.PotionEntries)
		{
			if (currentIndex++ == index)
			{
				return entry;
			}
		}
		if (inventory.CardRemovalEntry != null && currentIndex == index)
		{
			return inventory.CardRemovalEntry;
		}

		return null;
	}

	private static IEnumerable<T> EnumerateDescendants<T>(Node root) where T : class
	{
		foreach (Node child in root.GetChildren())
		{
			if (child is T typed)
			{
				yield return typed;
			}

			foreach (T nested in EnumerateDescendants<T>(child))
			{
				yield return nested;
			}
		}
	}
}
