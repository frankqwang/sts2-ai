using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Powers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models.Afflictions;

namespace MegaCrit.Sts2.Core.Models.Powers;

public sealed class GraspPower : PowerModel
{
	public override PowerType Type => PowerType.Buff;

	public override PowerStackType StackType => PowerStackType.Single;

	protected override IEnumerable<IHoverTip> ExtraHoverTips => new global::_003C_003Ez__ReadOnlySingleElementList<IHoverTip>(HoverTipFactory.ForEnergy(this));

	public override async Task AfterApplied(Creature? applier, CardModel? cardSource)
	{
		foreach (Creature item in base.Owner.CombatState.Allies.ToList())
		{
			if (!item.IsPlayer)
			{
				continue;
			}
			List<CardModel> list = item.Player.PlayerCombatState.AllCards.ToList();
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
			await Afflict(card);
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
			List<CardModel> list = item.Player.PlayerCombatState.AllCards.Where((CardModel c) => c.Affliction is Weighted).ToList();
			foreach (CardModel item2 in list)
			{
				CardCmd.ClearAffliction(item2);
			}
		}
		return Task.CompletedTask;
	}

	private async Task Afflict(CardModel card)
	{
		if (card.Affliction == null)
		{
			await CardCmd.Afflict<Weighted>(card, base.Amount);
		}
	}
}
