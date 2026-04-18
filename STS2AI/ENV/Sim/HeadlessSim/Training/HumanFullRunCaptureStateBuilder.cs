using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Training;

internal static class HumanFullRunCaptureStateBuilder
{
	private static int FindCardIndex(IReadOnlyList<CardModel> cards, CardModel? target)
	{
		if (target == null)
		{
			return -1;
		}
		for (int i = 0; i < cards.Count; i++)
		{
			if (ReferenceEquals(cards[i], target))
			{
				return i;
			}
		}
		return -1;
	}

	private static List<MerchantEntry> GetShopEntries(MerchantInventory inventory)
	{
		List<MerchantEntry> entries = new List<MerchantEntry>();
		entries.AddRange(inventory.CharacterCardEntries);
		entries.AddRange(inventory.ColorlessCardEntries);
		entries.AddRange(inventory.RelicEntries);
		entries.AddRange(inventory.PotionEntries);
		if (inventory.CardRemovalEntry != null)
		{
			entries.Add(inventory.CardRemovalEntry);
		}
		return entries;
	}

	private static string SafeFormat(LocString? locString)
	{
		if (locString == null || locString.IsEmpty)
		{
			return string.Empty;
		}

		string? rawText = null;
		try
		{
			rawText = locString.GetRawText();
			if (LooksLikeParameterizedLoc(rawText))
			{
				return rawText;
			}
		}
		catch
		{
			rawText = null;
		}

		try
		{
			return locString.GetFormattedText();
		}
		catch (Exception ex)
		{
			if (!string.IsNullOrWhiteSpace(rawText))
			{
				Log.Warn($"[HumanFullRunCapture] Falling back to raw loc text for {locString.LocTable}/{locString.LocEntryKey}: {ex.Message}");
				return rawText;
			}

			Log.Warn($"[HumanFullRunCapture] Falling back to loc key for {locString.LocTable}/{locString.LocEntryKey}: {ex.Message}");
			return $"{locString.LocTable}:{locString.LocEntryKey}";
		}
	}

	private static bool LooksLikeParameterizedLoc(string? text)
	{
		if (string.IsNullOrWhiteSpace(text))
		{
			return false;
		}

		int braceStart = text.IndexOf('{');
		if (braceStart < 0)
		{
			return false;
		}

		int braceEnd = text.IndexOf('}', braceStart + 1);
		if (braceEnd < 0)
		{
			return false;
		}

		int selectorSeparator = text.IndexOf(':', braceStart + 1);
		return selectorSeparator > braceStart && selectorSeparator < braceEnd;
	}

	public static Dictionary<string, object?> BuildRunContext(RunState? runState)
	{
		return new Dictionary<string, object?>
		{
			["act"] = runState?.CurrentActIndex + 1 ?? 0,
			["floor"] = runState?.ActFloor ?? 0,
			["seed"] = runState?.Rng?.StringSeed
		};
	}

	public static Dictionary<string, object?> BuildPlayerSnapshot(Player? player)
	{
		List<Dictionary<string, object?>> deck = new List<Dictionary<string, object?>>();
		List<Dictionary<string, object?>> relics = new List<Dictionary<string, object?>>();
		List<Dictionary<string, object?>> potions = new List<Dictionary<string, object?>>();
		if (player != null)
		{
			deck.AddRange(player.Deck.Cards.Select((CardModel card, int index) => BuildCardSnapshot(card, index)));
			relics.AddRange(player.Relics.Select(BuildRelicSnapshot));
			potions.AddRange(player.PotionSlots.Select((PotionModel? potion, int index) => BuildPotionSnapshot(potion, index)).Where(static p => p != null)!);
		}

		return new Dictionary<string, object?>
		{
			["character_id"] = player?.Character?.Id.Entry,
			["hp"] = player?.Creature?.CurrentHp ?? 0,
			["current_hp"] = player?.Creature?.CurrentHp ?? 0,
			["max_hp"] = player?.Creature?.MaxHp ?? 1,
			["gold"] = player?.Gold ?? 0,
			["deck"] = deck,
			["relics"] = relics,
			["potions"] = potions,
			["open_potion_slots"] = player?.PotionSlots.Count((PotionModel? potion) => potion == null) ?? 0
		};
	}

	public static Dictionary<string, object?> BuildMapState(RunState runState)
	{
		Player? player = LocalContext.GetMe(runState);
		MapPoint? currentMapPoint = runState.CurrentMapPoint;
		List<Dictionary<string, object?>> nextOptions = new List<Dictionary<string, object?>>();
		if (currentMapPoint != null)
		{
			foreach (MapPoint child in currentMapPoint.Children.OrderBy(static point => point.coord))
			{
				nextOptions.Add(new Dictionary<string, object?>
				{
					["index"] = nextOptions.Count,
					["coord"] = $"{child.coord.col},{child.coord.row}",
					["type"] = child.PointType.ToString().ToLowerInvariant()
				});
			}
		}

		return new Dictionary<string, object?>
		{
			["state_type"] = "map",
			["run"] = BuildRunContext(runState),
			["map"] = new Dictionary<string, object?>
			{
				["next_options"] = nextOptions
			},
			["player"] = BuildPlayerSnapshot(player)
		};
	}

	public static Dictionary<string, object?> BuildRestSiteState(RunState runState, IReadOnlyList<RestSiteOption> options)
	{
		Player? player = LocalContext.GetMe(runState);
		return new Dictionary<string, object?>
		{
			["state_type"] = "rest_site",
			["run"] = BuildRunContext(runState),
			["rest_site"] = new Dictionary<string, object?>
			{
				["options"] = options.Select((RestSiteOption option, int index) => new Dictionary<string, object?>
				{
				["index"] = index,
				["id"] = option.OptionId.ToLowerInvariant(),
				["title"] = SafeFormat(option.Title),
				["is_enabled"] = option.IsEnabled
			}).ToList()
			},
			["player"] = BuildPlayerSnapshot(player)
		};
	}

	public static Dictionary<string, object?> BuildEventState(RunState runState, EventModel eventModel)
	{
		Player? player = LocalContext.GetMe(runState);
		List<Dictionary<string, object?>> options = eventModel.CurrentOptions.Select((EventOption option, int index) => new Dictionary<string, object?>
		{
			["index"] = index,
			["id"] = option.TextKey.ToLowerInvariant(),
			["title"] = SafeFormat(option.Title),
			["description"] = SafeFormat(option.Description),
			["is_locked"] = option.IsLocked,
			["is_chosen"] = option.WasChosen,
			["is_proceed"] = option.IsProceed
		}).ToList();

		return new Dictionary<string, object?>
		{
			["state_type"] = "event",
			["run"] = BuildRunContext(runState),
			["event"] = new Dictionary<string, object?>
			{
				["id"] = eventModel.Id.Entry.ToLowerInvariant(),
				["title"] = SafeFormat(eventModel.Title),
				["options"] = options,
				["in_dialogue"] = false
			},
			["player"] = BuildPlayerSnapshot(player)
		};
	}

	public static Dictionary<string, object?> BuildShopState(MerchantInventory inventory)
	{
		Player player = inventory.Player;
		List<Dictionary<string, object?>> items = new List<Dictionary<string, object?>>();
		foreach (MerchantCardEntry entry in inventory.CharacterCardEntries)
		{
			items.Add(BuildShopItemSnapshot(entry, items.Count, "card"));
		}
		foreach (MerchantCardEntry entry2 in inventory.ColorlessCardEntries)
		{
			items.Add(BuildShopItemSnapshot(entry2, items.Count, "card"));
		}
		foreach (MerchantRelicEntry entry3 in inventory.RelicEntries)
		{
			items.Add(BuildShopItemSnapshot(entry3, items.Count, "relic"));
		}
		foreach (MerchantPotionEntry entry4 in inventory.PotionEntries)
		{
			items.Add(BuildShopItemSnapshot(entry4, items.Count, "potion"));
		}
		if (inventory.CardRemovalEntry != null)
		{
			items.Add(BuildShopItemSnapshot(inventory.CardRemovalEntry, items.Count, "card_removal"));
		}

		return new Dictionary<string, object?>
		{
			["state_type"] = "shop",
			["run"] = BuildRunContext(player.RunState as RunState),
			["shop"] = new Dictionary<string, object?>
			{
				["items"] = items,
				["player"] = BuildPlayerSnapshot(player)
			},
			["player"] = BuildPlayerSnapshot(player)
		};
	}

	public static Dictionary<string, object?> BuildCardRewardState(
		Player player,
		IReadOnlyList<CardCreationResult> cards,
		IReadOnlyList<CardRewardAlternative> extraOptions)
	{
		return new Dictionary<string, object?>
		{
			["state_type"] = "card_reward",
			["run"] = BuildRunContext(player.RunState as RunState),
			["card_reward"] = new Dictionary<string, object?>
			{
				["cards"] = cards.Select((CardCreationResult reward, int index) => BuildCardSnapshot(reward.Card, index)).ToList(),
				["can_skip"] = extraOptions.Any(static option => option.OptionId.Equals("Skip", StringComparison.OrdinalIgnoreCase))
			},
			["player"] = BuildPlayerSnapshot(player)
		};
	}

	public static Dictionary<string, object?> BuildTreasureState(RunState runState, IReadOnlyList<RelicModel> relics)
	{
		Player? player = LocalContext.GetMe(runState);
		return new Dictionary<string, object?>
		{
			["state_type"] = "treasure",
			["run"] = BuildRunContext(runState),
			["treasure"] = new Dictionary<string, object?>
			{
				["relics"] = relics.Select((RelicModel relic, int index) =>
				{
					Dictionary<string, object?> snapshot = BuildRelicSnapshot(relic);
					snapshot["index"] = index;
					return snapshot;
				}).ToList()
			},
			["player"] = BuildPlayerSnapshot(player)
		};
	}

	public static Dictionary<string, object?> BuildCardSelectState(
		Player player,
		IReadOnlyList<CardModel> cards,
		CardSelectorPrefs prefs,
		IReadOnlyList<CardModel>? selectedCards = null,
		bool canSkip = false)
	{
		List<CardModel> selected = selectedCards?.ToList() ?? new List<CardModel>();
		return new Dictionary<string, object?>
		{
			["state_type"] = "card_select",
			["run"] = BuildRunContext(player.RunState as RunState),
			["card_select"] = new Dictionary<string, object?>
			{
				["cards"] = cards.Select((CardModel card, int index) => BuildCardSnapshot(card, index)).ToList(),
				["selected_cards"] = selected.Select((CardModel card) =>
				{
					int index = FindCardIndex(cards, card);
					return BuildCardSnapshot(card, index);
				}).ToList(),
				["can_confirm"] = selected.Count >= Math.Max(1, prefs.MinSelect),
				["can_cancel"] = prefs.Cancelable,
				["can_skip"] = canSkip,
				["preview_showing"] = false,
				["requires_manual_confirmation"] = prefs.RequireManualConfirmation,
				["min_select"] = prefs.MinSelect,
				["max_select"] = prefs.MaxSelect,
				["remaining_picks"] = Math.Max(0, prefs.MaxSelect - selected.Count)
			},
			["player"] = BuildPlayerSnapshot(player)
		};
	}

	public static List<Dictionary<string, object?>> BuildMapCandidateActions(RunState runState)
	{
		MapPoint? currentMapPoint = runState.CurrentMapPoint;
		if (currentMapPoint == null)
		{
			return new List<Dictionary<string, object?>>();
		}
		return currentMapPoint.Children
			.OrderBy(static point => point.coord)
			.Select((MapPoint child, int index) => new Dictionary<string, object?>
			{
				["action"] = "choose_map_node",
				["index"] = index,
				["coord"] = $"{child.coord.col},{child.coord.row}"
			})
			.ToList();
	}

	public static Dictionary<string, object?> NormalizeMapAction(RunState runState, MoveToMapCoordAction action)
	{
		List<Dictionary<string, object?>> candidates = BuildMapCandidateActions(runState);
		string targetCoord = $"{action.Destination.col},{action.Destination.row}";
		int index = candidates.FindIndex(candidate => string.Equals(candidate["coord"] as string, targetCoord, StringComparison.Ordinal));
		return new Dictionary<string, object?>
		{
			["action"] = "choose_map_node",
			["index"] = index,
			["coord"] = targetCoord
		};
	}

	public static List<Dictionary<string, object?>> BuildRestSiteCandidateActions(IReadOnlyList<RestSiteOption> options)
	{
		return options.Select((RestSiteOption option, int index) => new Dictionary<string, object?>
		{
			["action"] = "choose_rest_option",
			["index"] = index,
			["option_id"] = option.OptionId.ToLowerInvariant()
		}).ToList();
	}

	public static List<Dictionary<string, object?>> BuildEventCandidateActions(EventModel eventModel)
	{
		return eventModel.CurrentOptions.Select((EventOption option, int index) => new Dictionary<string, object?>
		{
			["action"] = "choose_event_option",
			["index"] = index,
			["option_id"] = option.TextKey.ToLowerInvariant()
		}).ToList();
	}

	public static List<Dictionary<string, object?>> BuildShopCandidateActions(MerchantInventory inventory)
	{
		List<Dictionary<string, object?>> actions = new List<Dictionary<string, object?>>();
		foreach (MerchantEntry entry in GetShopEntries(inventory))
		{
			if (!entry.IsStocked)
			{
				continue;
			}
			actions.Add(new Dictionary<string, object?>
			{
				["action"] = "shop_purchase",
				["index"] = actions.Count
			});
		}
		actions.Add(new Dictionary<string, object?> { ["action"] = "proceed" });
		return actions;
	}

	public static Dictionary<string, object?> NormalizeShopPurchaseAction(MerchantInventory inventory, MerchantEntry entry)
	{
		List<MerchantEntry> stockedEntries = GetShopEntries(inventory).Where(static candidate => candidate.IsStocked).ToList();
		int index = stockedEntries.FindIndex(candidate => ReferenceEquals(candidate, entry));
		return new Dictionary<string, object?>
		{
			["action"] = "shop_purchase",
			["index"] = index
		};
	}

	public static List<Dictionary<string, object?>> BuildCardRewardCandidateActions(
		IReadOnlyList<CardCreationResult> cards,
		IReadOnlyList<CardRewardAlternative> extraOptions)
	{
		List<Dictionary<string, object?>> actions = cards.Select((CardCreationResult reward, int index) => new Dictionary<string, object?>
		{
			["action"] = "select_card_reward",
			["card_index"] = index,
			["card_id"] = reward.Card.Id.Entry.ToLowerInvariant()
		}).ToList();
		if (extraOptions.Any(static option => option.OptionId.Equals("Skip", StringComparison.OrdinalIgnoreCase)))
		{
			actions.Add(new Dictionary<string, object?> { ["action"] = "skip_card_reward" });
		}
		return actions;
	}

	public static List<Dictionary<string, object?>> BuildTreasureCandidateActions(IReadOnlyList<RelicModel> relics)
	{
		return relics.Select((RelicModel relic, int index) => new Dictionary<string, object?>
		{
			["action"] = "claim_treasure_relic",
			["index"] = index,
			["relic_id"] = relic.Id.Entry.ToLowerInvariant()
		}).ToList();
	}

	public static List<Dictionary<string, object?>> BuildCardSelectCandidateActions(
		IReadOnlyList<CardModel> cards,
		CardSelectorPrefs prefs,
		IReadOnlyList<CardModel>? selectedCards = null,
		bool canSkip = false)
	{
		List<CardModel> selected = selectedCards?.ToList() ?? new List<CardModel>();
		List<Dictionary<string, object?>> actions = cards.Select((CardModel card, int index) => new Dictionary<string, object?>
		{
			["action"] = "select_card",
			["index"] = index,
			["card_id"] = card.Id.Entry.ToLowerInvariant(),
			["is_selected"] = selected.Contains(card)
		}).ToList();
		if (selected.Count >= Math.Max(1, prefs.MinSelect))
		{
			actions.Add(new Dictionary<string, object?> { ["action"] = "confirm_selection" });
		}
		if (prefs.Cancelable)
		{
			actions.Add(new Dictionary<string, object?> { ["action"] = "cancel_selection" });
		}
		if (canSkip)
		{
			actions.Add(new Dictionary<string, object?> { ["action"] = "cancel_selection", ["is_skip"] = true });
		}
		return actions;
	}

	private static Dictionary<string, object?> BuildShopItemSnapshot(MerchantEntry entry, int index, string category)
	{
		Dictionary<string, object?> snapshot = new Dictionary<string, object?>
		{
			["index"] = index,
			["category"] = category,
			["cost"] = entry.Cost,
			["can_afford"] = entry.EnoughGold,
			["is_stocked"] = entry.IsStocked
		};

		switch (entry)
		{
		case MerchantCardEntry cardEntry when cardEntry.CreationResult != null:
			snapshot["card_id"] = cardEntry.CreationResult.Card.Id.Entry.ToLowerInvariant();
			snapshot["card_name"] = cardEntry.CreationResult.Card.Title;
			snapshot["card_type"] = cardEntry.CreationResult.Card.Type.ToString().ToLowerInvariant();
			snapshot["card_rarity"] = cardEntry.CreationResult.Card.Rarity.ToString().ToLowerInvariant();
			snapshot["card_description"] = SafeFormat(cardEntry.CreationResult.Card.Description);
			snapshot["keywords"] = cardEntry.CreationResult.Card.Keywords.Select(static keyword => keyword.ToString().ToLowerInvariant()).ToList();
			snapshot["on_sale"] = cardEntry.IsOnSale;
			break;
		case MerchantRelicEntry relicEntry when relicEntry.Model != null:
			snapshot["relic_id"] = relicEntry.Model.Id.Entry.ToLowerInvariant();
			snapshot["relic_name"] = SafeFormat(relicEntry.Model.Title);
			snapshot["relic_description"] = SafeFormat(relicEntry.Model.DynamicDescription);
			break;
		case MerchantPotionEntry potionEntry when potionEntry.Model != null:
			snapshot["potion_id"] = potionEntry.Model.Id.Entry.ToLowerInvariant();
			snapshot["potion_name"] = SafeFormat(potionEntry.Model.Title);
			snapshot["potion_description"] = SafeFormat(potionEntry.Model.DynamicDescription);
			break;
		case MerchantCardRemovalEntry:
			snapshot["removal"] = true;
			break;
		}

		return snapshot;
	}

	private static Dictionary<string, object?> BuildCardSnapshot(CardModel card, int index)
	{
		return new Dictionary<string, object?>
		{
			["index"] = index,
			["id"] = card.Id.Entry.ToLowerInvariant(),
			["name"] = card.Title,
			["type"] = card.Type.ToString().ToLowerInvariant(),
			["rarity"] = card.Rarity.ToString().ToLowerInvariant(),
			["cost"] = card.EnergyCost.GetWithModifiers(CostModifiers.Local),
			["description"] = SafeFormat(card.Description),
			["keywords"] = card.Keywords.Select(static keyword => keyword.ToString().ToLowerInvariant()).ToList()
		};
	}

	private static Dictionary<string, object?> BuildRelicSnapshot(RelicModel relic)
	{
		return new Dictionary<string, object?>
		{
			["id"] = relic.Id.Entry.ToLowerInvariant(),
			["name"] = SafeFormat(relic.Title),
			["description"] = SafeFormat(relic.DynamicDescription)
		};
	}

	private static Dictionary<string, object?>? BuildPotionSnapshot(PotionModel? potion, int index)
	{
		if (potion == null)
		{
			return null;
		}
		return new Dictionary<string, object?>
		{
			["index"] = index,
			["id"] = potion.Id.Entry.ToLowerInvariant(),
			["name"] = SafeFormat(potion.Title),
			["rarity"] = potion.Rarity.ToString().ToLowerInvariant(),
			["description"] = SafeFormat(potion.DynamicDescription)
		};
	}
}
