using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Powers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models.Afflictions;

namespace MegaCrit.Sts2.Core.Models.Powers;

public sealed class HungerPower : PowerModel
{
	public override PowerType Type => PowerType.Buff;

	public override PowerStackType StackType => PowerStackType.Single;

	protected override IEnumerable<IHoverTip> ExtraHoverTips => HoverTipFactory.FromAffliction<Devoured>(base.Amount);

	public override async Task AfterApplied(Creature? applier, CardModel? cardSource)
	{
		foreach (Creature item in base.Owner.CombatState.Allies.ToList())
		{
			if (!item.IsPlayer)
			{
				continue;
			}
			List<CardModel> list = item.Player.PlayerCombatState.AllCards.Where(delegate(CardModel c)
			{
				CardType type = c.Type;
				return (uint)(type - 1) <= 1u;
			}).ToList();
			foreach (CardModel item2 in list)
			{
				await Afflict(item2);
			}
		}
	}

	public override async Task AfterCardEnteredCombat(CardModel card)
	{
		if (card.Affliction == null)
		{
			CardType type = card.Type;
			if ((uint)(type - 1) <= 1u)
			{
				await Afflict(card);
			}
		}
	}

	public override Task AfterRemoved(Creature oldOwner)
	{
		if (oldOwner.CombatState == null)
		{
			return Task.CompletedTask;
		}
		foreach (Creature item in oldOwner.CombatState.Allies.ToList())
		{
			if (!item.IsPlayer)
			{
				continue;
			}
			List<CardModel> list = item.Player.PlayerCombatState.AllCards.Where((CardModel c) => c.Affliction is Devoured).ToList();
			foreach (CardModel item2 in list)
			{
				if (((Devoured)item2.Affliction).AppliedExhaust)
				{
					CardCmd.RemoveKeyword(item2, CardKeyword.Exhaust);
				}
				CardCmd.ClearAffliction(item2);
			}
		}
		return Task.CompletedTask;
	}

	private async Task Afflict(CardModel card)
	{
		if (card.Affliction == null)
		{
			Devoured devoured = await CardCmd.Afflict<Devoured>(card, base.Amount);
			if (devoured != null && !card.Keywords.Contains(CardKeyword.Exhaust))
			{
				CardCmd.ApplyKeyword(card, CardKeyword.Exhaust);
				devoured.AppliedExhaust = true;
			}
		}
	}
}
