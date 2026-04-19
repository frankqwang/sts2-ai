using System.Collections.Generic;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class MysteryMachine : Badge
{
	private const int _eventCount = 15;

	public override string Id => "MYSTERY_MACHINE";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => false;

	public override bool MultiplayerOnly => false;

	public MysteryMachine(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		int num = 0;
		foreach (List<MapPointHistoryEntry> item in _run.MapPointHistory)
		{
			foreach (MapPointHistoryEntry item2 in item)
			{
				if (item2.MapPointType == MapPointType.Unknown)
				{
					num++;
				}
			}
		}
		return num >= 15;
	}
}
