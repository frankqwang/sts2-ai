using System.Collections.Generic;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Models.Badges;

public class Perfect : Badge
{
	public override string Id => "PERFECT";

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
						if (room.RoomType != RoomType.Boss)
						{
							continue;
						}
						foreach (PlayerMapPointHistoryEntry playerStat in item2.PlayerStats)
						{
							if (playerStat.PlayerId == _localPlayer.NetId && playerStat.DamageTaken <= 0)
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

	public override bool MultiplayerOnly => false;

	public Perfect(SerializableRun run, ulong playerId)
		: base(run, playerId)
	{
	}

	public override bool IsObtained()
	{
		return Rarity != BadgeRarity.None;
	}
}
