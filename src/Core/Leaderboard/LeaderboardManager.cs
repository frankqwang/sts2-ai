using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Platform.Null;
using MegaCrit.Sts2.Core.Platform.Steam;

namespace MegaCrit.Sts2.Core.Leaderboard;

public static class LeaderboardManager
{
	private static ILeaderboardStrategy _strategy;

	public static PlatformType CurrentPlatform => _strategy.Platform;

	public static void Initialize()
	{
		if (SteamInitializer.Initialized)
		{
			_strategy = new SteamLeaderboardStrategy();
		}
		else
		{
			_strategy = new NullLeaderboardStrategy();
		}
	}

	public static Task<ILeaderboardHandle> GetOrCreateLeaderboard(string name, CancellationToken cancelToken = default(CancellationToken))
	{
		return _strategy.GetOrCreateLeaderboard(name, cancelToken);
	}

	public static Task<ILeaderboardHandle?> GetLeaderboard(string name, CancellationToken cancelToken = default(CancellationToken))
	{
		return _strategy.GetLeaderboard(name, cancelToken);
	}

	public static Task UploadLocalScore(ILeaderboardHandle handle, int score, IReadOnlyList<ulong> userIds)
	{
		return _strategy.UploadLocalScore(handle, score, userIds);
	}

	public static Task<List<LeaderboardEntry>> QueryLeaderboard(ILeaderboardHandle handle, LeaderboardQueryType type, int startIndex, int resultCount, CancellationToken cancelToken = default(CancellationToken))
	{
		return _strategy.QueryLeaderboard(handle, type, startIndex, resultCount, cancelToken);
	}

	public static Task<List<LeaderboardEntry>> QueryLeaderboardForUsers(ILeaderboardHandle handle, IReadOnlyList<ulong> userIds, CancellationToken cancelToken = default(CancellationToken))
	{
		return _strategy.QueryLeaderboardForUsers(handle, userIds, cancelToken);
	}

	public static int GetLeaderboardEntryCount(ILeaderboardHandle handle)
	{
		return _strategy.GetLeaderboardEntryCount(handle);
	}

	public static void DebugAddEntry(ILeaderboardHandle handle, LeaderboardEntry entry)
	{
		if (!(_strategy is NullLeaderboardStrategy nullLeaderboardStrategy))
		{
			throw new NotImplementedException();
		}
		nullLeaderboardStrategy.DebugAddEntry(handle, entry);
	}
}
