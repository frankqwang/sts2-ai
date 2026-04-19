using System.Linq;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class DoubleSnecko : Badge
{
	public override string Id => "DOUBLE_SNECKO";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => false;

	public override bool MultiplayerOnly => false;

	public DoubleSnecko(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		if (_localPlayer.Relics.Any((SerializableRelic r) => r.Id == ModelDb.Relic<SneckoEye>().Id))
		{
			return _localPlayer.Relics.Any((SerializableRelic r) => r.Id == ModelDb.Relic<FakeSneckoEye>().Id);
		}
		return false;
	}
}
