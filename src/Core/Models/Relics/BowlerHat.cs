using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Models.Relics;

public sealed class BowlerHat : RelicModel
{
	private const string _goldIncreaseKey = "GoldIncrease";

	private decimal _pendingBonusGold;

	private bool _isApplyingBonus;

	private decimal PendingBonusGold
	{
		get
		{
			return _pendingBonusGold;
		}
		set
		{
			AssertMutable();
			_pendingBonusGold = value;
		}
	}

	private bool IsApplyingBonus
	{
		get
		{
			return _isApplyingBonus;
		}
		set
		{
			AssertMutable();
			_isApplyingBonus = value;
		}
	}

	public override RelicRarity Rarity => RelicRarity.Uncommon;

	public override bool IsAllowedInShops => false;

	protected override IEnumerable<DynamicVar> CanonicalVars => new global::_003C_003Ez__ReadOnlySingleElementList<DynamicVar>(new DynamicVar("GoldIncrease", 1.25m));

	public override bool IsAllowed(IRunState runState)
	{
		return RelicModel.IsBeforeAct3TreasureChest(runState);
	}

	public override bool ShouldGainGold(decimal amount, Player player)
	{
		if (IsApplyingBonus)
		{
			return true;
		}
		if (player != base.Owner)
		{
			return true;
		}
		PendingBonusGold = Math.Floor(amount * (base.DynamicVars["GoldIncrease"].BaseValue - 1m));
		return true;
	}

	public override async Task AfterGoldGained(Player player)
	{
		if (player == base.Owner && !IsApplyingBonus && !(PendingBonusGold <= 0m))
		{
			decimal pendingBonusGold = PendingBonusGold;
			PendingBonusGold = 0m;
			IsApplyingBonus = true;
			Flash();
			await PlayerCmd.GainGold(pendingBonusGold, base.Owner);
			IsApplyingBonus = false;
		}
	}
}
