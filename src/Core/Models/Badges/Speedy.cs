using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class Speedy : Badge
{
	private const int _winTimeGold = 1800;

	private const int _winTimeSilver = 2400;

	private const int _winTimeBronze = 3000;

	public override string Id => "SPEEDY";

	public override BadgeRarity Rarity
	{
		get
		{
			long winTime = _run.WinTime;
			if (winTime <= 2400)
			{
				if (winTime <= 1800)
				{
					return BadgeRarity.Gold;
				}
				return BadgeRarity.Silver;
			}
			if (winTime <= 3000)
			{
				return BadgeRarity.Bronze;
			}
			return BadgeRarity.None;
		}
	}

	public override bool RequiresWin => true;

	public override bool MultiplayerOnly => false;

	public Speedy(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return Rarity != BadgeRarity.None;
	}
}
