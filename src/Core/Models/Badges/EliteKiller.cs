using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class EliteKiller : Badge
{
	public override string Id => "ELITE";

	public override BadgeRarity Rarity
	{
		get
		{
			int elitesKilledCount = ScoreUtility.GetElitesKilledCount(_run.MapPointHistory);
			if (elitesKilledCount >= 6)
			{
				if (elitesKilledCount >= 9)
				{
					return BadgeRarity.Gold;
				}
				return BadgeRarity.Silver;
			}
			if (elitesKilledCount >= 3)
			{
				return BadgeRarity.Bronze;
			}
			return BadgeRarity.None;
		}
	}

	public override bool RequiresWin => false;

	public override bool MultiplayerOnly => false;

	public EliteKiller(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return Rarity != BadgeRarity.None;
	}
}
