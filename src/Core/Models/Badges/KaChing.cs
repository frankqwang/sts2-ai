using System.Collections.Generic;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class KaChing : Badge
{
	private const int _goldRequirement = 1000;

	public override string Id => "KACHING";

	public override BadgeRarity Rarity => BadgeRarity.Bronze;

	public override bool RequiresWin => false;

	public override bool MultiplayerOnly => false;

	public KaChing(SerializableRun run, ulong playerId)
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
				foreach (MapPointRoomHistoryEntry room in item2.Rooms)
				{
					if (room.RoomType != RoomType.Shop)
					{
						continue;
					}
					foreach (PlayerMapPointHistoryEntry playerStat in item2.PlayerStats)
					{
						if (playerStat.PlayerId == _localPlayer.NetId)
						{
							num += playerStat.GoldSpent;
						}
					}
				}
			}
		}
		return num >= 1000;
	}
}
