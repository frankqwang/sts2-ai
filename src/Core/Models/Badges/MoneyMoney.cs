using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class MoneyMoney : Badge
{
	public override string Id => "MONEY_MONEY";

	public override BadgeRarity Rarity
	{
		get
		{
			int gold = _localPlayer.Gold;
			if (gold >= 400)
			{
				if (gold >= 600)
				{
					return BadgeRarity.Gold;
				}
				return BadgeRarity.Silver;
			}
			if (gold >= 200)
			{
				return BadgeRarity.Bronze;
			}
			return BadgeRarity.None;
		}
	}

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public MoneyMoney(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return Rarity != BadgeRarity.None;
	}
}
