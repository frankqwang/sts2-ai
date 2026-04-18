using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Platform;

namespace MegaCrit.Sts2.Core.Leaderboard;

public interface ILeaderboardStrategy
{
	PlatformType Platform { get; }

	Task<ILeaderboardHandle> GetOrCreateLeaderboard(string name, CancellationToken cancelToken);

	Task<ILeaderboardHandle?> GetLeaderboard(string name, CancellationToken cancelToken);

	Task UploadLocalScore(ILeaderboardHandle handle, int score, IReadOnlyList<ulong> otherIds);

	Task<List<LeaderboardEntry>> QueryLeaderboard(ILeaderboardHandle handle, LeaderboardQueryType type, int startIndex, int count, CancellationToken cancelToken = default(CancellationToken));

	Task<List<LeaderboardEntry>> QueryLeaderboardForUsers(ILeaderboardHandle handle, IReadOnlyList<ulong> userIds, CancellationToken cancelToken = default(CancellationToken));

	int GetLeaderboardEntryCount(ILeaderboardHandle handle);
}
