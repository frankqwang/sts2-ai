using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Leaderboard;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.DailyRun;

public partial class NDailyRunLeaderboardRow : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/daily_run/daily_run_leaderboard_row");

	private MegaLabel _rank;

	private MegaRichTextLabel _name;

	private MegaLabel _score;

	private LeaderboardEntry? _entry;

	private bool _isHeader;

	public static NDailyRunLeaderboardRow? Create(LeaderboardEntry entry)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NDailyRunLeaderboardRow nDailyRunLeaderboardRow = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NDailyRunLeaderboardRow>(PackedScene.GenEditState.Disabled);
		nDailyRunLeaderboardRow._entry = entry;
		return nDailyRunLeaderboardRow;
	}

	public static NDailyRunLeaderboardRow? CreateHeader()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NDailyRunLeaderboardRow nDailyRunLeaderboardRow = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NDailyRunLeaderboardRow>(PackedScene.GenEditState.Disabled);
		nDailyRunLeaderboardRow._isHeader = true;
		return nDailyRunLeaderboardRow;
	}

	public override void _Ready()
	{
		_rank = GetNode<MegaLabel>("Rank");
		_name = GetNode<MegaRichTextLabel>("Name");
		_score = GetNode<MegaLabel>("Score");
		if (_isHeader)
		{
			_rank.SetTextAutoSize(" " + new LocString("main_menu_ui", "LEADERBOARDS.rankHeader").GetRawText());
			_name.SetTextAutoSize(new LocString("main_menu_ui", "LEADERBOARDS.nameHeader").GetRawText());
			_score.SetTextAutoSize(new LocString("main_menu_ui", "LEADERBOARDS.scoreHeader").GetRawText() + " ");
		}
		else if (_entry != null)
		{
			IEnumerable<string> values = _entry.userIds.Select((ulong id) => PlatformUtil.GetPlayerName(LeaderboardManager.CurrentPlatform, id));
			_rank.SetTextAutoSize($" {_entry.rank + 1}");
			_name.SetTextAutoSize(string.Join(",", values));
			_score.SetTextAutoSize($"{_entry.score} ");
		}
	}
}
