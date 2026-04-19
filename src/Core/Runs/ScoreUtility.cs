using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Models.Badges;
using MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Runs;

public static class ScoreUtility
{
	public const int clientScore = -999999999;

	public static int CalculateScore(IRunState runState, bool won)
	{
		return CalculateScore(runState.MapPointHistory, runState.AscensionLevel, won, runState.Players.Count);
	}

	public static int CalculateScore(SerializableRun run, bool won)
	{
		return CalculateScore(run.MapPointHistory, run.Ascension, won, run.Players.Count);
	}

	private static int CalculateScore(IReadOnlyList<IReadOnlyList<MapPointHistoryEntry>> history, int ascension, bool won, int playerCount)
	{
		int num = 0;
		num += GetScoreForFloor(history);
		num += GetScoreForGoldGained(history, playerCount);
		num += GetScoreForElitesKilled(GetElitesKilledCount(history));
		num += GetScoreForBossesSlain(GetBossesSlainCount(history, won));
		return (int)((double)num * (1.0 + (double)ascension * 0.1));
	}

	public static int GetScoreForFloor(IReadOnlyList<IReadOnlyList<MapPointHistoryEntry>> history)
	{
		int num = 0;
		int count = history.Count;
		for (int i = 0; i < count; i++)
		{
			num += history[i].Count * 10 * (i + 1);
		}
		return num;
	}

	public static int GetElitesKilledCount(IReadOnlyList<IReadOnlyList<MapPointHistoryEntry>> history)
	{
		List<MapPointRoomHistoryEntry> list = history.SelectMany((IReadOnlyList<MapPointHistoryEntry> actEntries) => actEntries).SelectMany((MapPointHistoryEntry e) => e.Rooms).ToList();
		int num = list.Count((MapPointRoomHistoryEntry r) => r.RoomType == RoomType.Elite);
		if (list.Count > 0 && list.Last().RoomType == RoomType.Elite)
		{
			num--;
		}
		return num;
	}

	public static int GetScoreForElitesKilled(int elitesKilled)
	{
		return elitesKilled * 50;
	}

	public static int GetBossesSlainCount(IReadOnlyList<IReadOnlyList<MapPointHistoryEntry>> history, bool won)
	{
		int num = 0;
		foreach (IReadOnlyList<MapPointHistoryEntry> item in history)
		{
			foreach (MapPointHistoryEntry item2 in item)
			{
				foreach (MapPointRoomHistoryEntry room in item2.Rooms)
				{
					bool flag = !won && item == history.Last() && item2 == item.Last() && room == item2.Rooms.Last();
					if (room.RoomType == RoomType.Boss && !flag)
					{
						num++;
					}
				}
			}
		}
		return num;
	}

	public static int GetScoreForBossesSlain(int bossCount)
	{
		return bossCount * 100;
	}

	public static int GetScoreForGoldGained(IReadOnlyList<IReadOnlyList<MapPointHistoryEntry>> history, int playerCount)
	{
		return history.SelectMany((IReadOnlyList<MapPointHistoryEntry> actEntries) => actEntries).SelectMany((MapPointHistoryEntry e) => e.PlayerStats).Sum((PlayerMapPointHistoryEntry p) => p.GoldGained) / (100 * playerCount);
	}

	public static List<Badge> GetBadges(SerializableRun run, ulong playerId, bool won)
	{
		List<Badge> list = new List<Badge>();
		foreach (Badge item in BadgePool.CreateAll(run, playerId))
		{
			if ((!item.RequiresWin || won) && (!item.MultiplayerOnly || run.Players.Count != 1) && item.IsObtained())
			{
				list.Add(item);
			}
		}
		return list;
	}

	public static int CalculateDailyScore(SerializableRun run, ulong localPlayerNetId, bool isVictory)
	{
		int num = ((!isVictory) ? 1 : 2);
		int num2 = Math.Clamp(run.VisitedMapCoords.Count, 0, 99);
		int num3 = Math.Clamp(GetBadges(run, localPlayerNetId, isVictory).Count, 0, 99);
		int num4 = (int)Math.Clamp(isVictory ? run.WinTime : run.RunTime, 0L, 9999L);
		return num * 100000000 + num2 * 1000000 + num3 * 10000 + (9999 - num4);
	}

	public static DecodedDailyScore DecodeDailyScore(int encodedScore)
	{
		int num = encodedScore / 100000000;
		int num2 = encodedScore / 1000000 % 100;
		int num3 = encodedScore / 10000 % 100;
		int num4 = 9999 - encodedScore % 10000;
		bool flag = (uint)(num - 1) <= 2u;
		bool isValid = flag && num2 >= 0 && num2 <= 99 && num3 >= 0 && num3 <= 99 && num4 >= 0 && num4 <= 9999;
		return new DecodedDailyScore
		{
			isValid = isValid,
			victory = num,
			floors = num2,
			badges = num3,
			runTime = num4
		};
	}
}
