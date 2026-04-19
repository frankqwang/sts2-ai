using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.HoverTips;

namespace MegaCrit.Sts2.Core.Models.Afflictions;

public sealed class Devoured : AfflictionModel
{
	private bool _appliedExhaust;

	public bool AppliedExhaust
	{
		get
		{
			return _appliedExhaust;
		}
		set
		{
			AssertMutable();
			_appliedExhaust = value;
		}
	}

	protected override IEnumerable<IHoverTip> ExtraHoverTips => new global::_003C_003Ez__ReadOnlySingleElementList<IHoverTip>(HoverTipFactory.FromKeyword(CardKeyword.Exhaust));

	public override bool CanAfflictCardType(CardType cardType)
	{
		if ((uint)(cardType - 1) <= 1u)
		{
			return true;
		}
		return false;
	}
}
