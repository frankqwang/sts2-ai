using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;

namespace MegaCrit.Sts2.Core.Training;

internal static class CombatTrainingCardDescription
{
	public static string GetDescription(CardModel card, PileType pileType, Creature? target = null)
	{
		try
		{
			RefreshPreview(card, target);
			return card.GetDescriptionForPile(pileType, target);
		}
		catch
		{
			try
			{
				return card.Description.GetRawText();
			}
			catch
			{
				return string.Empty;
			}
		}
	}

	public static Dictionary<uint, int> BuildPreviewDamagePerTarget(
		CardModel card,
		CombatState combatState,
		IReadOnlyList<uint> validTargetIds)
	{
		if (!HasAnyPreviewVar(card, "Damage", "CalculatedDamage", "OstyDamage"))
		{
			return new Dictionary<uint, int>();
		}

		List<uint> targetIds = ResolvePreviewTargetIds(card, combatState, validTargetIds).Distinct().ToList();
		if (targetIds.Count == 0)
		{
			return new Dictionary<uint, int>();
		}

		Dictionary<uint, int> result = new Dictionary<uint, int>();
		try
		{
			foreach (uint targetId in targetIds)
			{
				Creature? target = combatState.GetCreature(targetId);
				if (target == null)
				{
					continue;
				}

				RefreshPreview(card, target);
				int? damage = GetFirstPreviewValue(card, "Damage", "CalculatedDamage", "OstyDamage");
				if (damage.HasValue)
				{
					result[targetId] = Math.Max(0, damage.Value);
				}
			}
		}
		finally
		{
			RefreshPreview(card, null);
		}

		return result;
	}

	public static int GetPreviewBlock(CardModel card)
	{
		if (!HasAnyPreviewVar(card, "Block", "CalculatedBlock"))
		{
			return 0;
		}

		RefreshPreview(card, null);
		return Math.Max(0, GetFirstPreviewValue(card, "Block", "CalculatedBlock") ?? 0);
	}

	public static void RefreshPreview(CardModel card, Creature? target)
	{
		try
		{
			card.UpdateDynamicVarPreview(CardPreviewMode.Normal, target, card.DynamicVars);
		}
		catch
		{
		}

		try
		{
			EnchantmentModel? enchantment = card.Enchantment;
			if (enchantment != null)
			{
				card.UpdateDynamicVarPreview(CardPreviewMode.Normal, target, enchantment.DynamicVars);
			}
		}
		catch
		{
		}
	}

	private static IEnumerable<uint> ResolvePreviewTargetIds(
		CardModel card,
		CombatState combatState,
		IReadOnlyList<uint> validTargetIds)
	{
		if (validTargetIds.Count > 0)
		{
			return validTargetIds;
		}

		Creature? owner = card.Owner?.Creature;
		if (owner == null)
		{
			return Array.Empty<uint>();
		}

		return card.TargetType switch
		{
			TargetType.AllEnemies or TargetType.RandomEnemy => combatState.Creatures
				.Where(creature => creature.CombatId.HasValue && creature.IsAlive && creature.Side != owner.Side)
				.Select(creature => creature.CombatId!.Value),
			TargetType.AllAllies => combatState.Creatures
				.Where(creature => creature.CombatId.HasValue && creature.IsAlive && creature.Side == owner.Side)
				.Select(creature => creature.CombatId!.Value),
			TargetType.Self or TargetType.Osty => owner.CombatId.HasValue
				? new[] { owner.CombatId.Value }
				: Array.Empty<uint>(),
			_ => Array.Empty<uint>()
		};
	}

	private static bool HasAnyPreviewVar(CardModel card, params string[] names)
	{
		return names.Any(card.DynamicVars.ContainsKey);
	}

	private static int? GetFirstPreviewValue(CardModel card, params string[] names)
	{
		DynamicVarSet vars = card.DynamicVars;
		foreach (string name in names)
		{
			if (vars.TryGetValue(name, out DynamicVar? dynamicVar))
			{
				return (int)dynamicVar.PreviewValue;
			}
		}
		return null;
	}
}
