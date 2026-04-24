using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Cards;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Simulation;

internal static class SimulationBuildSupport
{
	private static readonly JsonSerializerOptions BuildJsonOptions = new()
	{
		PropertyNameCaseInsensitive = true
	};

	public static bool HasOverrides(SimulationBuildSpec? build)
	{
		if (build == null)
		{
			return false;
		}

		return build.Deck != null
			|| build.Relics != null
			|| build.Potions != null
			|| build.CurrentHp.HasValue
			|| build.MaxHp.HasValue
			|| build.MaxEnergy.HasValue
			|| build.MaxPotionSlots.HasValue
			|| build.Gold.HasValue
			|| build.Floor.HasValue;
	}

	public static SimulationBuildSpec? ParseJson(string? buildJson)
	{
		if (string.IsNullOrWhiteSpace(buildJson))
		{
			return null;
		}

		try
		{
			return JsonSerializer.Deserialize<SimulationBuildSpec>(buildJson, BuildJsonOptions);
		}
		catch (JsonException ex)
		{
			throw new InvalidOperationException($"Invalid build spec JSON: {ex.Message}", ex);
		}
	}

	public static SimulationBuildSpec? ParseJsonElement(JsonElement buildElement)
	{
		if (buildElement.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
		{
			return null;
		}

		try
		{
			return JsonSerializer.Deserialize<SimulationBuildSpec>(buildElement.GetRawText(), BuildJsonOptions);
		}
		catch (JsonException ex)
		{
			throw new InvalidOperationException($"Invalid build spec JSON: {ex.Message}", ex);
		}
	}

	public static void ApplyToPlayerIfRequested(Player player, SimulationBuildSpec? build)
	{
		if (!HasOverrides(build))
		{
			return;
		}

		SerializablePlayer save = player.ToSerializable();
		if (build!.Deck != null)
		{
			save.Deck = build.Deck.Select(CreateSerializableCard).ToList();
			AddDiscoveredCards(save, save.Deck);
		}

		if (build.Relics != null)
		{
			save.Relics = build.Relics.Select(CreateSerializableRelic).ToList();
			AddDiscoveredRelics(save, save.Relics);
		}

		if (build.MaxPotionSlots.HasValue)
		{
			save.MaxPotionSlotCount = Math.Max(0, build.MaxPotionSlots.Value);
		}

		if (build.Potions != null)
		{
			save.Potions = build.Potions
				.Select(CreateSerializablePotion)
				.OrderBy(static potion => potion.SlotIndex)
				.ToList();
			if (save.Potions.Count > 0)
			{
				int requiredSlotCount = save.Potions.Max(static potion => potion.SlotIndex) + 1;
				save.MaxPotionSlotCount = Math.Max(save.MaxPotionSlotCount, requiredSlotCount);
			}
			AddDiscoveredPotions(save, save.Potions);
		}

		if (build.MaxHp.HasValue)
		{
			save.MaxHp = Math.Max(1, build.MaxHp.Value);
		}

		if (build.CurrentHp.HasValue)
		{
			save.CurrentHp = Math.Max(0, build.CurrentHp.Value);
		}

		if (build.MaxEnergy.HasValue)
		{
			save.MaxEnergy = Math.Max(0, build.MaxEnergy.Value);
		}

		if (build.Gold.HasValue)
		{
			save.Gold = Math.Max(0, build.Gold.Value);
		}

		save.CurrentHp = Math.Clamp(save.CurrentHp, 0, Math.Max(1, save.MaxHp));
		player.SyncWithSerializedPlayer(save);
	}

	public static void RemoveOwnedRelicsFromGrabBags(RunState runState, Player player)
	{
		foreach (RelicModel ownedRelic in player.Relics.Where(static relic => relic != null && !relic.IsStackable))
		{
			RelicModel canonical = ModelDb.GetById<RelicModel>(ownedRelic.Id);
			if (runState.SharedRelicGrabBag.IsPopulated)
			{
				runState.SharedRelicGrabBag.Remove(canonical);
			}

			if (player.RelicGrabBag.IsPopulated)
			{
				player.RelicGrabBag.Remove(canonical);
			}
		}
	}

	private static SerializableCard CreateSerializableCard(SimulationBuildCardSpec spec)
	{
		CardModel card = ResolveCard(spec.Id).ToMutable();
		return new SerializableCard
		{
			Id = card.Id,
			CurrentUpgradeLevel = Math.Max(0, spec.UpgradeLevel ?? 0),
			FloorAddedToDeck = spec.FloorAddedToDeck,
			Props = ResolveSerializableCardProps(card, spec)
		};
	}

	private static SerializableRelic CreateSerializableRelic(SimulationBuildRelicSpec spec)
	{
		RelicModel relic = ResolveRelic(spec.Id);
		return new SerializableRelic
		{
			Id = relic.Id,
			FloorAddedToDeck = spec.FloorAddedToDeck
		};
	}

	private static SerializablePotion CreateSerializablePotion(SimulationBuildPotionSpec spec)
	{
		PotionModel potion = ResolvePotion(spec.Id);
		int slotIndex = Math.Max(0, spec.Slot ?? spec.SlotIndex ?? 0);
		return new SerializablePotion
		{
			Id = potion.Id,
			SlotIndex = slotIndex
		};
	}

	private static CardModel ResolveCard(string? rawId)
	{
		string id = NormalizeId(rawId, "card");
		ModelId modelId = new(ModelId.SlugifyCategory<CardModel>(), id.ToUpperInvariant());
		return ModelDb.GetById<CardModel>(modelId);
	}

	private static SavedProperties? ResolveSerializableCardProps(CardModel card, SimulationBuildCardSpec spec)
	{
		if (spec.Props != null)
		{
			return spec.Props;
		}

		// Some event cards require saved props to define their playable mode.
		// Human analytics builds only carry card ids, so provide a deterministic
		// default instead of materializing an invalid card that later crashes or
		// advertises impossible legal actions.
		if (card is MadScience madScience)
		{
			madScience.TinkerTimeType = CardType.Attack;
			madScience.TinkerTimeRider = TinkerTime.RiderEffect.None;
			return SavedProperties.From(madScience);
		}

		return null;
	}

	private static RelicModel ResolveRelic(string? rawId)
	{
		string id = NormalizeId(rawId, "relic");
		ModelId modelId = new(ModelId.SlugifyCategory<RelicModel>(), id.ToUpperInvariant());
		return ModelDb.GetById<RelicModel>(modelId);
	}

	private static PotionModel ResolvePotion(string? rawId)
	{
		string id = NormalizeId(rawId, "potion");
		ModelId modelId = new(ModelId.SlugifyCategory<PotionModel>(), id.ToUpperInvariant());
		return ModelDb.GetById<PotionModel>(modelId);
	}

	private static string NormalizeId(string? rawId, string entityType)
	{
		string id = (rawId ?? string.Empty).Trim();
		if (id.Length == 0)
		{
			throw new InvalidOperationException($"Build spec {entityType} entry is missing an id.");
		}

		return id;
	}

	private static void AddDiscoveredCards(SerializablePlayer save, IEnumerable<SerializableCard> cards)
	{
		save.DiscoveredCards ??= new List<ModelId>();
		HashSet<ModelId> discovered = save.DiscoveredCards.ToHashSet();
		foreach (SerializableCard card in cards)
		{
			ModelId? cardId = card.Id;
			if (cardId == null)
			{
				continue;
			}
			if (discovered.Add(cardId))
			{
				save.DiscoveredCards.Add(cardId);
			}
		}
	}

	private static void AddDiscoveredRelics(SerializablePlayer save, IEnumerable<SerializableRelic> relics)
	{
		save.DiscoveredRelics ??= new List<ModelId>();
		HashSet<ModelId> discovered = save.DiscoveredRelics.ToHashSet();
		foreach (SerializableRelic relic in relics)
		{
			ModelId? relicId = relic.Id;
			if (relicId == null)
			{
				continue;
			}
			if (discovered.Add(relicId))
			{
				save.DiscoveredRelics.Add(relicId);
			}
		}
	}

	private static void AddDiscoveredPotions(SerializablePlayer save, IEnumerable<SerializablePotion> potions)
	{
		save.DiscoveredPotions ??= new List<ModelId>();
		HashSet<ModelId> discovered = save.DiscoveredPotions.ToHashSet();
		foreach (SerializablePotion potion in potions)
		{
			ModelId? potionId = potion.Id;
			if (potionId == null)
			{
				continue;
			}
			if (discovered.Add(potionId))
			{
				save.DiscoveredPotions.Add(potionId);
			}
		}
	}
}
