using System.Collections.Generic;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class Healer : Badge
{
	public override string Id => "HEALER";

	public override BadgeRarity Rarity
	{
		get
		{
			int num = 0;
			foreach (List<MapPointHistoryEntry> item in _run.MapPointHistory)
			{
				foreach (MapPointHistoryEntry item2 in item)
				{
					foreach (MapPointRoomHistoryEntry room in item2.Rooms)
					{
						if (room.RoomType != RoomType.RestSite)
						{
							continue;
						}
						foreach (PlayerMapPointHistoryEntry playerStat in item2.PlayerStats)
						{
							if (playerStat.PlayerId == _localPlayer.NetId && playerStat.RestSiteChoices.Contains("MEND"))
							{
								num++;
							}
						}
					}
				}
			}
			if (num < 3)
			{
				return num switch
				{
					2 => BadgeRarity.Silver, 
					1 => BadgeRarity.Bronze, 
					_ => BadgeRarity.None, 
				};
			}
			return BadgeRarity.Gold;
		}
	}

	public override bool RequiresWin => false;

	public override bool MultiplayerOnly => true;

	public Healer(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return Rarity != BadgeRarity.None;
	}
}
