using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;

namespace MegaCrit.Sts2.Core.Models.Orbs.Mock;

public class MockCombatCleanupOrb : OrbModel
{
	public override decimal PassiveVal => 0m;

	public override decimal EvokeVal => 0m;

	public override Color DarkenedColor => new Color("000000");

	public override Task<IEnumerable<Creature>> Evoke(PlayerChoiceContext playerChoiceContext)
	{
		base.Owner.Creature.CombatState.RemoveCreature(base.Owner.Creature);
		return Task.FromResult((IEnumerable<Creature>)Array.Empty<Creature>());
	}
}
