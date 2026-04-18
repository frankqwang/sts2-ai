using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Acts;
using MegaCrit.Sts2.Core.Models.Cards;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Models.Enchantments;
using MegaCrit.Sts2.Core.Models.Encounters;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Modifiers;
using MegaCrit.Sts2.Core.Models.Monsters;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Models.Relics;

namespace MegaCrit.Sts2.Core.Saves;

public static class SaveUtil
{
	public static EventModel EventOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<EventModel>(id) ?? ModelDb.Event<DeprecatedEvent>();
	}

	public static AncientEventModel AncientEventOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<AncientEventModel>(id) ?? ModelDb.Event<DeprecatedAncientEvent>();
	}

	public static EncounterModel EncounterOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<EncounterModel>(id) ?? ModelDb.Encounter<DeprecatedEncounter>();
	}

	public static CardModel CardOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<CardModel>(id) ?? ModelDb.Card<DeprecatedCard>();
	}

	public static RelicModel RelicOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<RelicModel>(id) ?? ModelDb.Relic<DeprecatedRelic>();
	}

	public static PotionModel PotionOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<PotionModel>(id) ?? ModelDb.Potion<DeprecatedPotion>();
	}

	public static ModifierModel ModifierOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<ModifierModel>(id) ?? ModelDb.Modifier<DeprecatedModifier>();
	}

	public static EnchantmentModel EnchantmentOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<EnchantmentModel>(id) ?? ModelDb.Enchantment<DeprecatedEnchantment>();
	}

	public static MonsterModel MonsterOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<MonsterModel>(id) ?? ModelDb.Monster<DeprecatedMonster>();
	}

	public static CharacterModel CharacterOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<CharacterModel>(id) ?? ModelDb.Character<DeprecatedCharacter>();
	}

	public static ActModel ActOrDeprecated(ModelId id)
	{
		return ModelDb.GetByIdOrNull<ActModel>(id) ?? ModelDb.Act<DeprecatedAct>();
	}
}
