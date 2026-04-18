using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Leaderboard;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;

public partial class NDailyRunLeaderboardRow : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/daily_run/daily_run_leaderboard_row");

	private MegaLabel _rank;

	private MegaLabel _name;

	private MegaLabel _floor;

	private MegaLabel _badges;

	private MegaLabel _time;

	private LeaderboardEntry _entry;

	private bool _isYou;

	public static NDailyRunLeaderboardRow? Create(LeaderboardEntry entry, bool isYou)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NDailyRunLeaderboardRow nDailyRunLeaderboardRow = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NDailyRunLeaderboardRow>(PackedScene.GenEditState.Disabled);
		nDailyRunLeaderboardRow._entry = entry;
		nDailyRunLeaderboardRow._isYou = isYou;
		return nDailyRunLeaderboardRow;
	}

	public override void _Ready()
	{
		_rank = GetNode<MegaLabel>("%Rank");
		_floor = GetNode<MegaLabel>("%Floor");
		_name = GetNode<MegaLabel>("%Name");
		_badges = GetNode<MegaLabel>("%Badges");
		_time = GetNode<MegaLabel>("%Time");
		IEnumerable<string> values = _entry.userIds.Select((ulong id) => PlatformUtil.GetPlayerName(LeaderboardManager.CurrentPlatform, id));
		DecodedDailyScore decodedDailyScore = ScoreUtility.DecodeDailyScore(_entry.score);
		if (!decodedDailyScore.isValid)
		{
			this.QueueFreeSafely();
			return;
		}
		_rank.SetTextAutoSize($"{_entry.rank + 1}");
		_name.SetTextAutoSize(string.Join(",", values));
		if (_isYou)
		{
			_name.Modulate = StsColors.blue;
		}
		_floor.SetTextAutoSize($"{decodedDailyScore.floors}");
		if (decodedDailyScore.victory == 2)
		{
			GetNode<Control>("%Tick").Visible = true;
		}
		_badges.SetTextAutoSize($"{decodedDailyScore.badges}");
		_time.SetTextAutoSize(FormatHoursAndMinutes(decodedDailyScore.runTime));
	}

	private static string FormatHoursAndMinutes(int value)
	{
		if (value >= 9999)
		{
			return "--:--";
		}
		int value2 = value / 60;
		int value3 = value % 60;
		return $"{value2}:{value3:D2}";
	}
}
